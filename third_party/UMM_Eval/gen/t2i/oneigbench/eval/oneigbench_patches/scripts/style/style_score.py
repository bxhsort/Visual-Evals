from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error, ParallelContext, get_output_csv_path, get_cache_dir

import torch
torch.cuda.empty_cache()
from scripts.utils.inference import CSDStyleEmbedding, SEStyleEmbedding

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

style_list = ['abstract_expressionism', 'art_nouveau', 'baroque', 'chinese_ink_painting', 'cubism', 'fauvism', 'impressionism', 'line_art', 'minimalism', 'pointillism', 'pop_art', 'rococo',  'ukiyo-e', 'clay', 'crayon',  'graffiti','lego', 'comic', 'pencil_sketch', 'stone_sculpture', 'watercolor', 'celluloid', 'chibi',   'cyberpunk',  'ghibli',  'impasto', 'pixar', 'pixel_art',  '3d_rendering']


def move_ref_dict_to_device(ref_dict, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in ref_dict.items()
    }


def score_single_image(task, args, cache_dir, df, csd_encoder, se_encoder, csd_ref, se_ref):
    model_id, model_name, img_path = task
    img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1]))
    sample_id = img_path.split('/')[-1][:3]
    image_style = str(df.loc[df["id"] == sample_id, "class"].values[0])
    if image_style[:3] == "nan":
        return model_name, sample_id, None

    image_style = image_style.lower().replace(' ', '_')
    split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
    if not split_img_list:
        return model_name, sample_id, None

    csd_ref_embeds = csd_ref[image_style]
    se_ref_embeds = se_ref[image_style]
    score = []
    for split_img_path in split_img_list:
        csd_embed = csd_encoder.get_style_embedding(split_img_path)
        se_embed = se_encoder.get_style_embedding(split_img_path)
        csd_max_style_score = max(torch.max(csd_embed @ csd_ref_embeds.T).item(), 0)
        se_max_style_score = max(torch.max(se_embed @ se_ref_embeds.T).item(), 0)
        score.append((csd_max_style_score + se_max_style_score) / 2)

    if not score:
        return model_name, sample_id, None
    return model_name, sample_id, {
        "style": image_style,
        "prompt_score": sum(score) / len(score),
    }

def main():
    args = parse_args()
    parallel_ctx = ParallelContext(args)
    cache_dir = get_cache_dir(args, formatted_time, parallel_ctx.rank)

    style_csv_path = "scripts/style/style.csv"
    df = pd.read_csv(style_csv_path, dtype=str)
    
    CSD_Encoder = CSDStyleEmbedding(model_path="scripts/style/models/checkpoint.pth")
    # SE_Encoder = SEStyleEmbedding(pretrained_path="xingpng/OneIG-StyleEncoder")
    SE_Encoder = SEStyleEmbedding(pretrained_path="./models/xingpng/OneIG-StyleEncoder")

    CSD_embed_pt = "scripts/style/CSD_embed.pt"
    CSD_ref = move_ref_dict_to_device(
        torch.load(CSD_embed_pt, map_location="cpu", weights_only=False),
        CSD_Encoder.device,
    )
    SE_embed_pt = "scripts/style/SE_embed.pt"
    SE_ref = move_ref_dict_to_device(
        torch.load(SE_embed_pt, map_location="cpu"),
        SE_Encoder.device,
    )

    style_score_csv = get_output_csv_path(args, f"style_score_{args.mode}_{formatted_time}.csv")
    style_style_score_csv = get_output_csv_path(args, f"style_style_score_{args.mode}_{formatted_time}.csv")
    style_prompt_score_csv = get_output_csv_path(args, f"style_prompt_score_{args.mode}_{formatted_time}.csv")

    task_items = []
    for model_id, model_name in enumerate(args.model_names):
        image_dir = args.image_dirname + '/' + model_name
        img_list = sorted(megfile.smart_glob(image_dir + '/*'))
        for img_path in img_list:
            task_items.append((model_id, model_name, img_path))

    worker_items = parallel_ctx.shard_items(task_items) if parallel_ctx.executor_mode == "distributed" else task_items
    partial_results = parallel_ctx.map_items(
        list(tqdm(
            worker_items,
            total=len(worker_items),
            desc=f"Processing style rank {parallel_ctx.rank}",
            disable=parallel_ctx.executor_mode == "thread",
        )),
        lambda item: score_single_image(item, args, cache_dir, df, CSD_Encoder, SE_Encoder, CSD_ref, SE_ref),
    )
    gathered_results = parallel_ctx.gather_object(partial_results)

    if parallel_ctx.is_main_process:
        score_csv = pd.DataFrame(index=args.model_names, columns=["style"])
        score_of_style_csv = pd.DataFrame(index=args.model_names, columns=style_list)
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        style_dict_by_model = {
            model_name: {style: [] for style in style_list}
            for model_name in args.model_names
        }

        for partial in gathered_results:
            for model_name, sample_id, result in partial:
                score_of_prompt_csv.loc[sample_id, model_name] = None if result is None else result["prompt_score"]
                if result is not None:
                    style_dict_by_model[model_name][result["style"]].append(result["prompt_score"])

        for model_name in args.model_names:
            for style in style_list:
                if style_dict_by_model[model_name][style]:
                    score_of_style_csv.loc[model_name, style] = sum(style_dict_by_model[model_name][style]) / len(style_dict_by_model[model_name][style])

        mean_values = score_of_prompt_csv.mean()
        score_csv["style"] = mean_values.values
        save2csv(score_csv, style_score_csv)
        save2csv(score_of_style_csv, style_style_score_csv)

        score_of_prompt_csv = score_of_prompt_csv.sort_index()
        save2csv(score_of_prompt_csv, style_prompt_score_csv)

    parallel_ctx.barrier()
    parallel_ctx.finalize()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
