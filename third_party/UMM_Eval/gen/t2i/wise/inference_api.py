from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing as mp
import os
import random
from pathlib import Path

from inference_api_common import (
    build_output_root,
    build_t2i_image_config,
    build_think_trace,
    chat_t2i,
    decode_image_from_message,
    default_num_workers,
    extract_think_payload,
    get_progress_bar,
    image_size_spec_to_square_pixels,
    make_grid,
    normalize_image_size_spec,
    parse_base_urls,
    parse_csv,
    parse_target_strings,
    resolve_output_dir,
    select_base_url,
    shard_data,
    write_json,
)


DEFAULT_CATEGORIES = [
    "cultural_common_sense",
    "natural_science",
    "spatio-temporal_reasoning",
]
DEFAULT_WISE_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "data")


def load_wise_data(data_root, categories):
    import json

    data = []
    for category in categories:
        category_path = os.path.join(data_root, f"{category}.json")
        if not os.path.isfile(category_path):
            raise FileNotFoundError(f"WISE category file not found: {category_path}")
        with open(category_path, "r", encoding="utf-8") as f:
            category_data = json.load(f)
        for sample in category_data:
            sample["wise_category"] = category
        data.extend(category_data)
    return data


def run_single_sample(sample: dict, worker_config: dict) -> dict:
    prompt_id = str(sample["prompt_id"])
    prompt = sample["Prompt"]
    request_base_url = select_base_url(
        worker_config["base_url"], worker_config.get("base_urls"), sample.get("_dispatch_index", 0)
    )

    output_dir = Path(worker_config["output_dir"])
    output_path = output_dir / f"{prompt_id}.png"
    think_root = output_dir / "think_content"
    think_path = think_root / f"{prompt_id}.json"

    if worker_config["resume"] and output_path.exists() and (
        not worker_config["enable_thinking"] or think_path.exists()
    ):
        return {"status": "skipped", "prompt_id": prompt_id}

    try:
        images = []
        think_payloads = []
        for request_index in range(worker_config["batch_size"]):
            image_config = dict(worker_config["image_config"])
            image_config["seed"] = worker_config["seed"] + request_index
            message = chat_t2i(
                base_url=request_base_url,
                api_key=worker_config["api_key"],
                model_name=worker_config["model_name"],
                user_text=prompt,
                image_config=image_config,
                enable_thinking=worker_config["enable_thinking"],
                temperature=worker_config["temperature"],
                top_p=worker_config["top_p"],
                max_tokens=worker_config["max_tokens"],
                timeout=worker_config["timeout"],
            )
            if worker_config["enable_thinking"]:
                think_payloads.append(extract_think_payload(message))
            image = decode_image_from_message(message)
            if image is not None:
                images.append(image)

        if not images:
            print(message)
            import ipdb;ipdb.set_trace()
            return {"status": "failed", "prompt_id": prompt_id, "error": "No image returned"}

        output_dir.mkdir(parents=True, exist_ok=True)
        if len(images) == 1:
            grid_image = images[0]
        else:
            grid_image = make_grid(images, worker_config["grid_rows"], worker_config["grid_cols"], pad=0)
        grid_image.save(output_path)

        if worker_config["enable_thinking"]:
            think_root.mkdir(parents=True, exist_ok=True)
            think_trace = build_think_trace(think_payloads)
            write_json(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": sample.get("wise_category"),
                    "assistant_content": think_trace["assistant_content"],
                    "think_content": think_trace["think_content"],
                    "micro_batch_think_contents": think_trace["micro_batch_think_contents"],
                    "batch_size": worker_config["batch_size"],
                },
                think_path,
            )

        warnings = []
        if len(images) != worker_config["batch_size"]:
            warnings.append(f"only received {len(images)}/{worker_config['batch_size']} images")
        return {"status": "success", "prompt_id": prompt_id, "warnings": warnings}
    except Exception as exc:
        return {"status": "failed", "prompt_id": prompt_id, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Run WISE via pure API with multiprocessing")
    parser.add_argument("--base_url", type=str, default=os.environ.get("LIGHTLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--base_urls", type=str, default=os.environ.get("API_BASE_URLS"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--wise_data_root", type=str, default=DEFAULT_WISE_DATA_ROOT)
    parser.add_argument("--categories", type=str, default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--target_prompt_ids", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_subdir", type=str, default=None)
    parser.add_argument(
        "--resolution",
        "--image_size",
        dest="image_size",
        type=normalize_image_size_spec,
        default=normalize_image_size_spec(os.environ.get("IMAGE_SIZE", "1K")),
    )
    parser.add_argument("--aspect_ratio", type=str, default="1:1")
    parser.add_argument("--image_type", type=str, default="png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", "--cfg", dest="cfg_scale", type=float, default=7.0)
    parser.add_argument("--cfg_norm", type=str, default="global")
    parser.add_argument("--timestep_shift", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("API_NUM_WORKERS", default_num_workers())))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and "API_SHARD_COUNT" not in os.environ and rank != 0:
        print(
            f"Detected distributed launch (WORLD_SIZE={world_size}, RANK={rank}). "
            "Set API_SHARD_COUNT/API_SHARD_RANK to enable multi-node API sharding. Non-zero ranks exit by default."
        )
        return

    base_urls = parse_base_urls(args.base_urls)
    categories = parse_csv(args.categories) or list(DEFAULT_CATEGORIES)
    data = load_wise_data(args.wise_data_root, categories)
    target_prompt_ids = parse_target_strings(args.target_prompt_ids)
    if target_prompt_ids is not None:
        data = [sample for sample in data if str(sample["prompt_id"]) in target_prompt_ids]

    random.seed(args.shuffle_seed)
    random.shuffle(data)
    data, shard_count, shard_rank = shard_data(data)
    for dispatch_index, sample in enumerate(data):
        sample["_dispatch_index"] = dispatch_index

    output_root = build_output_root(image_size_spec_to_square_pixels(args.image_size), args.cfg_scale, args.timestep_shift)
    output_subdir = args.output_subdir or ("wise_think" if args.enable_thinking else "wise_raw")
    output_dir = resolve_output_dir(args.output_path, output_root, args.model_name, output_subdir)

    grid_cols = min(2, args.batch_size)
    grid_rows = math.ceil(args.batch_size / grid_cols)
    worker_config = {
        "base_url": args.base_url,
        "base_urls": base_urls,
        "api_key": args.api_key,
        "model_name": args.model_name,
        "output_dir": output_dir,
        "image_config": build_t2i_image_config(
            args.image_size,
            args.image_type,
            args.seed,
            aspect_ratio=args.aspect_ratio,
            guidance_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            timestep_shift=args.timestep_shift,
        ),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "resume": args.resume,
    }

    print(f"Using model_name={args.model_name}")
    print(f"Using WISE data root: {args.wise_data_root}")
    print(f"Enable thinking: {args.enable_thinking}")
    if base_urls:
        print(f"Using {len(base_urls)} API base URLs with round-robin dispatch")
    print(f"Processing shard {shard_rank + 1}/{shard_count} with {len(data)} samples using {args.num_workers} workers")
    print(f"Saving outputs to: {output_dir}")

    progress_bar = get_progress_bar()
    success = 0
    skipped = 0
    failed = 0

    def handle_result(result):
        nonlocal success, skipped, failed
        if result["status"] == "success":
            success += 1
            for warning in result.get("warnings", []):
                print(f"[WARN] prompt_id={result['prompt_id']}: {warning}")
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"[ERROR] prompt_id={result['prompt_id']}: {result.get('error', 'unknown error')}")

    if args.num_workers == 1:
        for sample in progress_bar(data, desc="WISE"):
            handle_result(run_single_sample(sample, worker_config))
    else:
        mp_context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
            future_to_id = {
                executor.submit(run_single_sample, sample, worker_config): str(sample["prompt_id"])
                for sample in data
            }
            for future in progress_bar(concurrent.futures.as_completed(future_to_id), total=len(future_to_id), desc="WISE"):
                prompt_id = future_to_id[future]
                try:
                    handle_result(future.result())
                except Exception as exc:
                    failed += 1
                    print(f"[ERROR] prompt_id={prompt_id}: worker crashed: {exc}")

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}, total={len(data)}")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
