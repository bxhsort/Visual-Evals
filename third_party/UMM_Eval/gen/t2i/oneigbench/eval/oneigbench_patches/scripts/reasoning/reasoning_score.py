from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error, ParallelContext, get_output_csv_path, get_cache_dir

import json
from scripts.utils.inference import LLM2CLIP

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")


def score_single_prompt(task, args, cache_dir, answer_gt, llm2clip_model):
    model_id, model_name, img_path = task
    img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1]))
    split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
    img_id = img_path.split('/')[-1][:3]
    answer_text = answer_gt[img_id]
    score = llm2clip_model.text_img_similarity_score(split_img_list, answer_text)
    if not score:
        return model_name, img_id, None
    score = [x for x in score if x is not None]
    return model_name, img_id, None if not score else sum(score) / len(score)

def main():
    args = parse_args()
    parallel_ctx = ParallelContext(args)
    cache_dir = get_cache_dir(args, formatted_time, parallel_ctx.rank)
    
    LLM2CLIP_Model = LLM2CLIP()
    
    if args.mode == "EN":
        answer_json_dir = "scripts/reasoning/gt_answer.json"
    else:
        answer_json_dir = "scripts/reasoning/gt_answer_zh.json"
    with open(answer_json_dir, 'r', encoding='utf-8') as f:
        answer_gt = json.load(f)
        
    reasoning_score_csv = get_output_csv_path(args, f"reasoning_score_{args.mode}_{formatted_time}.csv")
    reasoning_prompt_score_csv = get_output_csv_path(args, f"reasoning_prompt_score_{args.mode}_{formatted_time}.csv")
    
    task_items = []
    for model_id, model_name in enumerate(args.model_names):
        image_dir = args.image_dirname + '/' + model_name
        for img_path in sorted(megfile.smart_glob(image_dir + '/*')):
            task_items.append((model_id, model_name, img_path))

    worker_items = parallel_ctx.shard_items(task_items) if parallel_ctx.executor_mode == "distributed" else task_items
    partial_results = parallel_ctx.map_items(
        list(tqdm(
            worker_items,
            total=len(worker_items),
            desc=f"Processing reasoning rank {parallel_ctx.rank}",
            disable=parallel_ctx.executor_mode == "thread",
        )),
        lambda item: score_single_prompt(item, args, cache_dir, answer_gt, LLM2CLIP_Model),
    )
    gathered_results = parallel_ctx.gather_object(partial_results)

    if parallel_ctx.is_main_process:
        score_csv = pd.DataFrame(index=args.model_names, columns=["reasoning"])
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        for partial in gathered_results:
            for model_name, img_id, score in partial:
                score_of_prompt_csv.loc[img_id, model_name] = score

        mean_values = score_of_prompt_csv.mean()
        score_csv["reasoning"] = mean_values.values
        save2csv(score_csv, reasoning_score_csv)
        
        score_of_prompt_csv = score_of_prompt_csv.sort_index()
        save2csv(score_of_prompt_csv, reasoning_prompt_score_csv)

    parallel_ctx.barrier()
    parallel_ctx.finalize()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
            
