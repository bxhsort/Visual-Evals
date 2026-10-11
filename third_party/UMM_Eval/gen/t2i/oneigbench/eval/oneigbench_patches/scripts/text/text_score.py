from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error, ParallelContext, get_output_csv_path, get_cache_dir

from scripts.text.text_utils import preprocess_string, clean_and_remove_hallucinations, levenshtein_distance, calculate_char_match_ratio
from scripts.utils.inference import Qwen2_5VLBatchInferencer

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")


def score_single_prompt(task, args, cache_dir, inferencer):
    model_id, model_name, sample_id, text_gt = task
    img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1]))
    word_count = len(text_gt.split())
    max_new_tokens = 256 if word_count > 60 else 128
    text_gt_preprocessed = preprocess_string(text_gt)

    img_path = megfile.smart_glob(args.image_dirname + '/' + model_name + '/' + sample_id + '*')
    if len(img_path) != 1:
        return model_name, sample_id, None

    split_img_list = split_mxn_grid(img_path[0], img_grid, cache_dir)
    if len(split_img_list) == 0:
        return model_name, sample_id, None

    ocr_results = inferencer.infer_ocr(split_img_list, max_new_tokens)
    text_ocr_list = clean_and_remove_hallucinations(ocr_results)

    prompt_stats = []
    for text_ocr in text_ocr_list:
        text_ocr_preprocessed = preprocess_string(text_ocr)
        edit_distance = levenshtein_distance(text_ocr_preprocessed, text_gt_preprocessed)
        completion_ratio = 1 if edit_distance == 0 else 0
        match_word_count, text_word_accuracy, gt_word_count = calculate_char_match_ratio(text_gt_preprocessed, text_ocr_preprocessed)
        prompt_stats.append({
            "edit_distance": edit_distance,
            "completion_ratio": completion_ratio,
            "match_word_count": match_word_count,
            "gt_word_count": gt_word_count,
            "text_word_accuracy": text_word_accuracy,
        })

    if not prompt_stats:
        return model_name, sample_id, None

    prompt_score = [
        sum(item["edit_distance"] for item in prompt_stats) / len(prompt_stats),
        sum(item["completion_ratio"] for item in prompt_stats) / len(prompt_stats),
        sum(item["text_word_accuracy"] for item in prompt_stats) / len(prompt_stats),
    ]
    return model_name, sample_id, {
        "prompt_score": prompt_score,
        "stats": prompt_stats,
    }

def main():
    args = parse_args()
    parallel_ctx = ParallelContext(args)
    cache_dir = get_cache_dir(args, formatted_time, parallel_ctx.rank)

    influencer = Qwen2_5VLBatchInferencer()

    if args.mode == "EN":
        text_csv_path = "scripts/text/text_content.csv"
        MAX_EDIT_DISTANCE = 100
    else:
        text_csv_path = "scripts/text/text_content_zh.csv"
        MAX_EDIT_DISTANCE = 50
    text_df = pd.read_csv(text_csv_path, dtype=str)

    text_score_csv = get_output_csv_path(args, f"text_score_{args.mode}_{formatted_time}.csv")
    text_prompt_score_csv = get_output_csv_path(args, f"text_prompt_score_{args.mode}_{formatted_time}.csv")

    task_items = []
    for model_id, model_name in enumerate(args.model_names):
        for sample_id, text_gt in zip(text_df["id"], text_df["text_content"]):
            task_items.append((model_id, model_name, sample_id, text_gt))

    worker_items = parallel_ctx.shard_items(task_items) if parallel_ctx.executor_mode == "distributed" else task_items
    partial_results = []
    iterator = tqdm(
        worker_items,
        total=len(worker_items),
        desc=f"Processing text rank {parallel_ctx.rank}",
        disable=parallel_ctx.executor_mode == "thread",
    )
    for result in parallel_ctx.map_items(list(iterator), lambda item: score_single_prompt(item, args, cache_dir, influencer)):
        partial_results.append(result)

    gathered_results = parallel_ctx.gather_object(partial_results)

    if parallel_ctx.is_main_process:
        score_csv = pd.DataFrame(index=args.model_names, columns=["ED", "CR", "WAC", "text score"])
        score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
        aggregated_stats = {
            model_name: {"edit_distances": [], "completion_ratios": [], "match_word_counts": [], "gt_word_counts": []}
            for model_name in args.model_names
        }

        for partial in gathered_results:
            for model_name, sample_id, result in partial:
                score_of_prompt_csv.loc[sample_id, model_name] = None if result is None else result["prompt_score"]
                if result is None:
                    continue
                for stat in result["stats"]:
                    aggregated_stats[model_name]["edit_distances"].append(stat["edit_distance"])
                    aggregated_stats[model_name]["completion_ratios"].append(stat["completion_ratio"])
                    aggregated_stats[model_name]["match_word_counts"].append(stat["match_word_count"])
                    aggregated_stats[model_name]["gt_word_counts"].append(stat["gt_word_count"])

        for model_name in args.model_names:
            edit_distances = aggregated_stats[model_name]["edit_distances"]
            completion_ratios = aggregated_stats[model_name]["completion_ratios"]
            match_word_counts = aggregated_stats[model_name]["match_word_counts"]
            gt_word_counts = aggregated_stats[model_name]["gt_word_counts"]
            if not edit_distances or not gt_word_counts:
                continue
            ED = sum(edit_distances) / len(edit_distances)
            CR = sum(completion_ratios) / len(completion_ratios)
            WAC = sum(match_word_counts) / sum(gt_word_counts)
            score_csv.loc[model_name, "ED"] = ED
            score_csv.loc[model_name, "CR"] = CR
            score_csv.loc[model_name, "WAC"] = WAC
            score_csv.loc[model_name, "text score"] = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

        save2csv(score_csv, text_score_csv)
        save2csv(score_of_prompt_csv, text_prompt_score_csv)

    parallel_ctx.barrier()
    parallel_ctx.finalize()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
