from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

from PIL import Image
import hashlib
def str_to_seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**16)

from inference_api_common import (
    build_edit_output_folder,
    build_image_config,
    chat_it2i,
    decode_image_from_message,
    default_num_workers,
    get_progress_bar,
    image_size_spec_to_square_pixels,
    normalize_image_size_spec,
    parse_base_urls,
    parse_target_strings,
    resolve_output_dir,
    select_base_url,
    shard_data,
    smart_resize,
)


# Eval scorers + Benchmark data ship co-located under eval/.
DEFAULT_IMGEDIT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
DEFAULT_BENCHMARK_ROOT = os.path.join(DEFAULT_IMGEDIT_ROOT, "Benchmark", "singleturn")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_BENCHMARK_ROOT, "singleturn.json")


def load_edit_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for index, (key, value) in enumerate(data.items()):
        sample = dict(value)
        sample["key"] = str(key)
        sample["index"] = index
        samples.append(sample)
    return samples


def run_single_sample(sample: dict, worker_config: dict) -> dict:
    key = sample["key"]
    prompt = sample["prompt"]
    request_base_url = select_base_url(
        worker_config["base_url"], worker_config.get("base_urls"), sample.get("_dispatch_index", 0)
    )
    # if "<image>" not in prompt:
    #     prompt = "<image>\n" + prompt

    output_dir = Path(worker_config["output_dir"])
    output_path = output_dir / f"{key}.png"
    if worker_config["resume"] and output_path.exists():
        return {"status": "skipped", "key": key}
    # import ipdb;ipdb.set_trace()
    try:
        input_image_path = os.path.join(worker_config["image_root"], sample["id"])
        input_image = Image.open(input_image_path).convert("RGB")
        width, height = input_image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=32,
            min_pixels=worker_config["resize_min_pixels"],
            max_pixels=2048 * 2048,
        )
        input_image = input_image.resize((resized_width, resized_height))
        image_config = build_image_config(
            resized_width,
            resized_height,
            worker_config["image_size"],
            worker_config["image_type"],
            worker_config["seed"],
            guidance_scale=worker_config["txt_scale"],
            cfg_norm=worker_config["cfg_norm"],
            timestep_shift=worker_config["timestep_shift"],
        )
        output_image = None
        image_config['aspect_ratio'] = '1:1'
        image_config['image_size'] = '2K'
        image_config["seed"] = worker_config["seed"] + str_to_seed(key)
        # import ipdb;ipdb.set_trace()
        # if key == '440':
        #     import ipdb;ipdb.set_trace()
        # if image_config['aspect_ratio'] != '1:1':
            # import ipdb;ipdb.set_trace()
        for _ in range(worker_config["batch_size"]):
            message = chat_it2i(
                base_url=request_base_url,
                api_key=worker_config["api_key"],
                model_name=worker_config["model_name"],
                user_text=prompt,
                input_image=input_image,
                image_config=image_config,
                input_image_type=worker_config["input_image_type"],
                temperature=worker_config["temperature"],
                top_p=worker_config["top_p"],
                max_tokens=worker_config["max_tokens"],
                timeout=worker_config["timeout"],
            )
            image = decode_image_from_message(message)
            if image is not None and output_image is None:
                output_image = image

        if output_image is None:
            
            return {"status": "failed", "key": key, "error": "No image returned"}

        output_dir.mkdir(parents=True, exist_ok=True)
        output_image.save(output_path)
        return {"status": "success", "key": key}
    except Exception as exc:
        return {"status": "failed", "key": key, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Run ImgEdit via pure API with multiprocessing")
    parser.add_argument("--base_url", type=str, default=os.environ.get("LIGHTLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--base_urls", type=str, default=os.environ.get("API_BASE_URLS"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--imgedit_root", type=str, default=DEFAULT_IMGEDIT_ROOT)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_root", type=str, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--resolution",
        "--image_size",
        dest="image_size",
        type=normalize_image_size_spec,
        default=normalize_image_size_spec(os.environ.get("IMAGE_SIZE", "512")),
    )
    parser.add_argument("--image_type", type=str, default="png")
    parser.add_argument("--input_image_type", type=str, default="png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--txt_scale", "--txt_cfg", dest="txt_scale", type=float, default=4.0)
    parser.add_argument("--img_cfg", type=float, default=1.5)
    parser.add_argument("--cfg_norm", "--cfg_renorm", dest="cfg_norm", type=str, default="global")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--target_keys", type=str, default=None)
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

    base_urls = parse_base_urls(args.base_urls)
    output_folder = build_edit_output_folder(
        args.txt_scale,
        args.img_cfg,
        args.cfg_norm,
        args.timestep_shift,
        image_size_spec_to_square_pixels(args.image_size),
    )
    output_dir = resolve_output_dir(args.output_path, output_folder, args.model_name, "imgedit")

    data = load_edit_data(args.data_path)
    target_keys = parse_target_strings(args.target_keys)
    if target_keys is not None:
        data = [sample for sample in data if sample["key"] in target_keys]

    random.seed(args.shuffle_seed)
    random.shuffle(data)
    data, shard_count, shard_rank = shard_data(data)
    for dispatch_index, sample in enumerate(data):
        sample["_dispatch_index"] = dispatch_index

    worker_config = {
        "base_url": args.base_url,
        "base_urls": base_urls,
        "api_key": args.api_key,
        "model_name": args.model_name,
        "image_root": args.image_root,
        "output_dir": output_dir,
        "image_size": args.image_size,
        "image_type": args.image_type,
        "input_image_type": args.input_image_type,
        "seed": args.seed,
        "txt_scale": args.txt_scale,
        "cfg_norm": args.cfg_norm,
        "timestep_shift": args.timestep_shift,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "resume": args.resume,
        "resize_min_pixels": image_size_spec_to_square_pixels(args.image_size) ** 2,
    }

    print(f"Using model_name={args.model_name}")
    print(f"Using ImgEdit json: {args.data_path}")
    print(f"Using ImgEdit image root: {args.image_root}")
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
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"[ERROR] key={result['key']}: {result.get('error', 'unknown error')}")

    if args.num_workers == 1:
        for sample in progress_bar(data, desc="ImgEdit"):
            handle_result(run_single_sample(sample, worker_config))
    else:
        mp_context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
            future_to_key = {executor.submit(run_single_sample, sample, worker_config): sample["key"] for sample in data}
            for future in progress_bar(concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc="ImgEdit"):
                key = future_to_key[future]
                try:
                    handle_result(future.result())
                except Exception as exc:
                    failed += 1
                    print(f"[ERROR] key={key}: worker crashed: {exc}")

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}, total={len(data)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
