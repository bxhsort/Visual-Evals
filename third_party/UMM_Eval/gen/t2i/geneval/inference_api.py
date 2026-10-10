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
    default_num_workers,
    get_progress_bar,
    image_size_spec_to_square_pixels,
    make_grid,
    normalize_image_size_spec,
    parse_target_ints,
    read_jsonl,
    resolve_output_dir,
    shard_data,
    write_jsonl,
)
from providers import get_provider


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval", "prompts", "evaluation_metadata.jsonl"
)


def run_single_sample(sample: dict, worker_config: dict) -> dict:
    index = int(sample["index"])
    prompt = sample["prompt"]
    index_str = str(index).zfill(4)

    output_dir = Path(worker_config["output_dir"])
    sample_root = output_dir / index_str
    grid_path = sample_root / "grid.png"
    metadata_path = sample_root / "metadata.jsonl"
    samples_dir = sample_root / "samples"

    if worker_config["resume"] and grid_path.exists() and metadata_path.exists():
        existing_images = list(samples_dir.glob("*.png")) if samples_dir.exists() else []
        if len(existing_images) >= 1:
            return {"status": "skipped", "index": index}

    try:
        provider = get_provider(worker_config["model_name"], provider_override=worker_config.get("provider_override"))
        images = []
        for request_index in range(worker_config["batch_size"]):
            try:
                image = provider.generate_one(
                    prompt=prompt,
                    image_size=worker_config["image_size"],
                    seed=worker_config["seed"] + request_index,
                    aspect_ratio=worker_config["aspect_ratio"],
                    timeout=worker_config["timeout"],
                    negative_prompt=worker_config["negative_prompt"],
                    guidance_scale=worker_config["cfg_scale"],
                    cfg_norm=worker_config["cfg_norm"],
                    timestep_shift=worker_config["timestep_shift"],
                    image_type=worker_config["image_type"],
                    temperature=worker_config["temperature"],
                    top_p=worker_config["top_p"],
                    max_tokens=worker_config["max_tokens"],
                )
            except Exception as exc:
                print(f"[WARN] index={index} request {request_index} failed: {exc}")
                continue
            if image is not None:
                images.append(image)

        if not images:
            return {"status": "failed", "index": index, "error": "No image returned"}

        sample_root.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

        if len(images) == 1:
            grid_image = images[0]
        else:
            grid_image = make_grid(images, worker_config["grid_rows"], worker_config["grid_cols"], pad=2)

        write_jsonl([sample], metadata_path)
        grid_image.save(grid_path)
        for image_index, image in enumerate(images):
            image.save(samples_dir / f"{image_index:04d}.png")

        warnings = []
        if len(images) != worker_config["batch_size"]:
            warnings.append(f"only received {len(images)}/{worker_config['batch_size']} images")
        return {"status": "success", "index": index, "warnings": warnings}
    except Exception as exc:
        return {"status": "failed", "index": index, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Run GenEval via pure API with multiprocessing")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--provider", type=str, default=os.environ.get("PROVIDER"),
                        help="Override provider routing (gpt_image/nano_banana/seedream/qwen_image/lightllm)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--resolution",
        "--image_size",
        dest="image_size",
        type=normalize_image_size_spec,
        default=normalize_image_size_spec(os.environ.get("IMAGE_SIZE", "512")),
    )
    parser.add_argument("--aspect_ratio", type=str, default="1:1")
    parser.add_argument("--image_type", type=str, default="png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", "--cfg", dest="cfg_scale", type=float, default=7.0)
    parser.add_argument("--cfg_norm", type=str, default="global")
    parser.add_argument("--timestep_shift", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_indices", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
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

    output_root = build_output_root(image_size_spec_to_square_pixels(args.image_size), args.cfg_scale, args.timestep_shift)
    output_dir = resolve_output_dir(args.output_path, output_root, args.model_name, "gen_eval")
    data = read_jsonl(args.data_path)
    for index, sample in enumerate(data):
        sample["index"] = index
    target_indices = parse_target_ints(args.target_indices)
    if target_indices is not None:
        data = [sample for sample in data if int(sample["index"]) in target_indices]

    random.seed(args.shuffle_seed)
    random.shuffle(data)
    data, shard_count, shard_rank = shard_data(data)
    for dispatch_index, sample in enumerate(data):
        sample["_dispatch_index"] = dispatch_index

    grid_cols = min(2, args.batch_size)
    grid_rows = math.ceil(args.batch_size / grid_cols)
    worker_config = {
        "model_name": args.model_name,
        "provider_override": args.provider,
        "output_dir": output_dir,
        "image_size": args.image_size,
        "image_type": args.image_type,
        "aspect_ratio": args.aspect_ratio,
        "cfg_scale": args.cfg_scale,
        "cfg_norm": args.cfg_norm,
        "timestep_shift": args.timestep_shift,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "resume": args.resume,
    }

    print(f"Using model_name={args.model_name}")
    if args.provider:
        print(f"Provider override: {args.provider}")
    print(f"Using prompt file: {args.data_path}")
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
                print(f"[WARN] index={result['index']}: {warning}")
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"[ERROR] index={result['index']}: {result.get('error', 'unknown error')}")

    if args.num_workers == 1:
        for sample in progress_bar(data, desc="GenEval"):
            handle_result(run_single_sample(sample, worker_config))
    else:
        mp_context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
            future_to_index = {executor.submit(run_single_sample, sample, worker_config): int(sample["index"]) for sample in data}
            for future in progress_bar(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="GenEval"):
                index = future_to_index[future]
                try:
                    handle_result(future.result())
                except Exception as exc:
                    failed += 1
                    print(f"[ERROR] index={index}: worker crashed: {exc}")

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}, total={len(data)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
