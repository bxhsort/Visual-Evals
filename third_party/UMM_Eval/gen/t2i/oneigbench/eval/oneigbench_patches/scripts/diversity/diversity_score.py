from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error, ParallelContext, get_output_csv_path, get_cache_dir

import torchvision
torchvision.disable_beta_transforms_warning()
from dreamsim import dreamsim
from scripts.utils.distributed import resolve_device

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def img_similar_score(image_1_path, image_2_path, model, preprocess, device):
    image_1 = preprocess(Image.open(image_1_path)).to(device)
    image_2 = preprocess(Image.open(image_2_path)).to(device)
    distance = model(image_1, image_2)     
    return distance.item()


def score_single_prompt(task, args, cache_dir, model, preprocess, device):
    model_id, model_name, class_item, img_path = task
    img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1]))
    split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
    if len(split_img_list) <= 1:
        return model_name, class_item, img_path.split('/')[-1][:3], None

    score = []
    for i in range(len(split_img_list)):
        for j in range(i + 1, len(split_img_list)):
            score.append(img_similar_score(split_img_list[i], split_img_list[j], model, preprocess, device))

    if not score:
        return model_name, class_item, img_path.split('/')[-1][:3], None
    return model_name, class_item, img_path.split('/')[-1][:3], sum(score) / len(score)

def main():
    args = parse_args()
    parallel_ctx = ParallelContext(args)
    
    cache_dir = get_cache_dir(args, formatted_time, parallel_ctx.rank)
    device = resolve_device()
    model, preprocess = dreamsim(pretrained=True, device=str(device))
    
    diversity_score_csv = get_output_csv_path(args, f"diversity_score_{args.mode}_{formatted_time}.csv")
    diversity_prompt_score_csv = get_output_csv_path(args, f"diversity_prompt_score_{args.mode}_{formatted_time}.csv")

    column_items = args.class_items + ["total average"]
    task_items = []
    for model_id, model_name in enumerate(args.model_names):
        for class_item in args.class_items:
            image_dir = args.image_dirname + '/' + class_item + '/' + model_name
            for img_path in sorted(megfile.smart_glob(image_dir + '/*')):
                task_items.append((model_id, model_name, class_item, img_path))

    worker_items = parallel_ctx.shard_items(task_items) if parallel_ctx.executor_mode == "distributed" else task_items
    partial_results = parallel_ctx.map_items(
        list(tqdm(
            worker_items,
            total=len(worker_items),
            desc=f"Processing diversity rank {parallel_ctx.rank}",
            disable=parallel_ctx.executor_mode == "thread",
        )),
        lambda item: score_single_prompt(item, args, cache_dir, model, preprocess, device),
    )
    gathered_results = parallel_ctx.gather_object(partial_results)

    if parallel_ctx.is_main_process:
        score_csv = pd.DataFrame(index=args.model_names, columns=column_items)
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        class_scores = {
            model_name: {class_item: [] for class_item in args.class_items}
            for model_name in args.model_names
        }

        for partial in gathered_results:
            for model_name, class_item, sample_id, avg_score in partial:
                score_of_prompt_csv.loc[f"{class_item}_{sample_id}", model_name] = avg_score
                if avg_score is not None:
                    class_scores[model_name][class_item].append(avg_score)

        for model_name in args.model_names:
            for class_item in args.class_items:
                if class_scores[model_name][class_item]:
                    score_csv.loc[model_name, class_item] = sum(class_scores[model_name][class_item]) / len(class_scores[model_name][class_item])
                else:
                    score_csv.loc[model_name, class_item] = None

        mean_values = score_of_prompt_csv.mean()
        score_csv["total average"] = mean_values.values
        save2csv(score_csv, diversity_score_csv)
        
        score_of_prompt_csv = score_of_prompt_csv.sort_index()
        save2csv(score_of_prompt_csv, diversity_prompt_score_csv)

    parallel_ctx.barrier()
    parallel_ctx.finalize()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
