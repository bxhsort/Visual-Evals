# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.qwen2 import Qwen2Tokenizer


def seed_for_sample(index: int, base_seed: int = 42) -> None:
    seed = int(base_seed) + int(index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def move_generation_input_to_device(generation_input, device):
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            generation_input[key] = value.to(device)
    return generation_input


def atomic_save(image: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    image_format = "WEBP" if Path(path).suffix.lower() == ".webp" else "PNG"
    image.save(tmp_path, format=image_format)
    os.replace(tmp_path, path)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_wise_records(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tiif_records(path: str, specific_file: str = ""):
    root = Path(path)
    if root.is_dir():
        if specific_file:
            files = [root / specific_file]
        else:
            files = sorted(p for p in root.iterdir() if p.suffix == ".jsonl")
    else:
        files = [root]

    records = []
    for file_path in files:
        file_records = read_jsonl(str(file_path))
        for sample_idx, sample in enumerate(file_records):
            record = dict(sample)
            record["_source_file"] = file_path.name
            record["_sample_idx"] = sample_idx
            records.append(record)
    return records


def parse_csv(raw_value: str):
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def read_cvtg_records(path: str, subsets: str, areas: str):
    root = Path(path)
    records = []
    for subset in parse_csv(subsets):
        for area in parse_csv(areas):
            data_path = root / subset / f"{area}_combined.json"
            with data_path.open("r", encoding="utf-8") as handle:
                samples = json.load(handle)
            for key, prompt in samples.items():
                records.append(
                    {
                        "prompt": prompt,
                        "subset": subset,
                        "area": area,
                        "key": str(key),
                    }
                )
    return records


ONEIG_CATEGORY_DIRS = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}


def read_oneig_records(path: str, mode: str):
    root = Path(path)
    filename = "OneIG-Bench-ZH.csv" if mode == "zh" else "OneIG-Bench.csv"
    prompt_key = "prompt_cn" if mode == "zh" else "prompt_en"
    records = []
    with (root / filename).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            category = row["category"]
            records.append(
                {
                    "prompt": row[prompt_key],
                    "id": row["id"],
                    "category": category,
                    "category_dir": ONEIG_CATEGORY_DIRS[category],
                }
            )
    return records


def parse_aspect_ratio(value: str):
    text = str(value or "").strip().lower()
    separator = "x" if "x" in text else ":" if ":" in text else None
    if separator is None:
        return None
    try:
        width, height = (int(part) for part in text.split(separator, 1))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def resolve_bizgeneval_size(sample, long_side: int, default_aspect_ratio: str):
    ratio = parse_aspect_ratio(sample.get("aspect_ratio")) or parse_aspect_ratio(default_aspect_ratio)
    if ratio is None:
        raise ValueError(f"Invalid BizGenEval aspect ratio: {sample.get('aspect_ratio')!r}")
    ratio_width, ratio_height = ratio
    factor = 32
    if ratio_width >= ratio_height:
        width = long_side
        height = round(long_side * ratio_height / ratio_width / factor) * factor
    else:
        height = long_side
        width = round(long_side * ratio_width / ratio_height / factor) * factor
    return max(factor, width), max(factor, height)


def load_bagel_model(
    model_path: str,
    device: str,
    max_latent_size: int,
    accelerate_device_map: bool,
    max_memory: str,
    offload_folder: str,
):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=max_latent_size,
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    checkpoint = os.path.join(model_path, "ema.safetensors")
    if accelerate_device_map:
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_memory for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for key in same_device_modules:
            device_map[key] = device_map.get(key, first_device)
        print(f"Using accelerate device_map with max_memory={max_memory}: {device_map}", flush=True)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint,
            device_map=device_map,
            offload_buffers=True,
            offload_folder=offload_folder,
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    else:
        state_dict = load_file(checkpoint, device="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg, flush=True)
        del state_dict
        model = model.to(device).eval()

    vae_model = vae_model.to(device).eval()
    return model, vae_model, tokenizer, new_token_ids


def generate_images(
    model,
    vae_model,
    tokenizer,
    new_token_ids,
    prompt: str,
    resolution,
    num_images: int,
    batch_size: int,
    num_timesteps: int,
    cfg_scale: float,
    cfg_norm: str,
    timestep_shift: float,
    device: str,
):
    if isinstance(resolution, int):
        height = width = resolution
    else:
        height, width = resolution
    cfg_renorm_type = "global" if cfg_norm == "none" else cfg_norm
    cfg_renorm_min = 1.0 if cfg_norm == "none" else 0.0

    image_list = []
    remaining = num_images
    while remaining > 0:
        cur_batch = min(remaining, batch_size)
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0] * cur_batch
        new_rope = [0] * cur_batch

        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt] * cur_batch,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

        generation_input = model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(height, width)] * cur_batch,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)

        cfg_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        cfg_newlens = [0] * cur_batch
        cfg_new_rope = [0] * cur_batch
        generation_input_cfg = model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_newlens,
            curr_rope=cfg_new_rope,
            image_sizes=[(height, width)] * cur_batch,
        )
        generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=[0, 1.0],
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

        for latent in unpacked_latent:
            latent = latent.reshape(1, height // 16, width // 16, 2, 2, 16)
            latent = torch.einsum("nhwpqc->nchpwq", latent)
            latent = latent.reshape(1, 16, height // 8, width // 8)
            image = vae_model.decode(latent.to(device))
            image = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8)
            image_list.append(Image.fromarray(image.cpu().numpy()))

        del unpacked_latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        remaining -= cur_batch

    return image_list


def make_grid(images, resolution, rows: int, cols: int) -> Image.Image:
    if isinstance(resolution, int):
        height = width = resolution
    else:
        height, width = resolution
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = np.asarray(image)
    return Image.fromarray(grid)


def output_paths_for_sample(args, sample, stable_index):
    benchmark = args.benchmark
    if benchmark == "geneval":
        folder = os.path.join(args.output_dir, "gen_eval", f"{stable_index:05d}")
        sample_dir = os.path.join(folder, "samples")
        paths = [os.path.join(sample_dir, f"{i:05d}.png") for i in range(args.num_images)]
        return folder, sample_dir, paths
    if benchmark == "dpgbench":
        prompt_id = str(sample["id"])
        path = os.path.join(args.output_dir, "dpgbench", f"{prompt_id}.png")
        return os.path.dirname(path), os.path.dirname(path), [path]
    if benchmark == "qwenimagebench":
        sample_id = str(sample["ID"])
        path = os.path.join(args.output_dir, f"qwenimagebench_{args.mode}", "samples", f"{sample_id}.png")
        return os.path.dirname(path), os.path.dirname(path), [path]
    if benchmark == "tiifbench":
        sample_idx = int(sample["_sample_idx"])
        data_type = str(sample["type"])
        root = os.path.join(args.output_dir, f"TIIFBench-{args.tiifbench_split}", data_type, args.model_name)
        paths = [
            os.path.join(root, "short_description", f"{sample_idx}.png"),
            os.path.join(root, "long_description", f"{sample_idx}.png"),
        ]
        return root, root, paths
    if benchmark == "longtextbench":
        prompt_id = int(sample["prompt_id"])
        root = os.path.join(args.output_dir, f"longtextbench_{args.mode}")
        paths = [
            os.path.join(root, "grids", f"{prompt_id}_grid.png"),
        ] + [
            os.path.join(root, "samples", f"{prompt_id}_{repeat_id}.png")
            for repeat_id in range(1, args.num_images + 1)
        ]
        return root, root, paths
    if benchmark == "cvtg_2k":
        path = os.path.join(
            args.output_dir,
            "CVTG",
            sample["subset"],
            str(sample["area"]),
            f"{sample['key']}.png",
        )
        return os.path.dirname(path), os.path.dirname(path), [path]
    if benchmark == "oneigbench":
        bench_dir = "OneIGBench-ZH" if args.mode == "zh" else "OneIGBench-EN"
        path = os.path.join(
            args.output_dir,
            bench_dir,
            sample["category_dir"],
            args.model_name,
            f"{sample['id']}.webp",
        )
        return os.path.dirname(path), os.path.dirname(path), [path]
    if benchmark == "bizgeneval":
        path = os.path.join(args.output_dir, "bizgeneval", f"{stable_index:04d}.png")
        return os.path.dirname(path), os.path.dirname(path), [path]
    if benchmark == "wise":
        prompt_id = str(sample["prompt_id"])
        path = os.path.join(args.output_dir, args.wise_subdir, f"{prompt_id}.png")
        return os.path.dirname(path), os.path.dirname(path), [path]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def all_outputs_exist(paths):
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            return False
    return True


def parse_target_indices(raw):
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def parse_target_prompt_ids(raw):
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def load_records(args):
    if args.benchmark == "tiifbench":
        records = read_tiif_records(args.metadata_file, args.tiifbench_specific_file)
    elif args.benchmark == "cvtg_2k":
        records = read_cvtg_records(args.metadata_file, args.cvtg_subsets, args.cvtg_areas)
    elif args.benchmark == "oneigbench":
        records = read_oneig_records(args.metadata_file, args.mode)
    elif args.benchmark == "wise":
        records = read_wise_records(args.metadata_file)
    else:
        records = read_jsonl(args.metadata_file)
    for idx, sample in enumerate(records):
        sample["_stable_index"] = idx

    target_indices = parse_target_indices(args.target_indices)
    if target_indices is not None:
        records = [sample for sample in records if sample["_stable_index"] in target_indices]

    target_prompt_ids = parse_target_prompt_ids(args.target_prompt_ids)
    if target_prompt_ids is not None:
        if args.benchmark == "dpgbench":
            records = [sample for sample in records if str(sample["id"]) in target_prompt_ids]
        elif args.benchmark == "qwenimagebench":
            records = [sample for sample in records if str(sample["ID"]) in target_prompt_ids]
        elif args.benchmark == "longtextbench":
            records = [sample for sample in records if str(sample["prompt_id"]) in target_prompt_ids]
        elif args.benchmark == "bizgeneval":
            records = [sample for sample in records if str(sample["_stable_index"]) in target_prompt_ids]

    target_keys = set(parse_csv(args.target_keys)) if args.target_keys else None
    if target_keys is not None and args.benchmark == "cvtg_2k":
        records = [sample for sample in records if sample["key"] in target_keys]

    random.seed(42)
    random.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate BAGEL text-to-image benchmark outputs.")
    parser.add_argument(
        "--benchmark",
        choices=[
            "geneval",
            "dpgbench",
            "qwenimagebench",
            "tiifbench",
            "longtextbench",
            "cvtg_2k",
            "oneigbench",
            "bizgeneval",
            "wise",
        ],
        required=True,
    )
    parser.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--model_name", type=str, default="bagel")
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["all", "en", "zh"], default="all")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_timesteps", "--num_steps", dest="num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--cfg_norm", choices=["none", "global", "channel"], default="global")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--accelerate_device_map", action="store_true")
    parser.add_argument("--max_memory", type=str, default="80GiB")
    parser.add_argument("--offload_folder", type=str, default="offload_t2i")
    parser.add_argument("--target_indices", type=str, default="")
    parser.add_argument("--target_prompt_ids", type=str, default="")
    parser.add_argument("--tiifbench_split", type=str, default="testmini")
    parser.add_argument("--tiifbench_specific_file", type=str, default="")
    parser.add_argument("--cvtg_subsets", type=str, default="CVTG,CVTG-Style")
    parser.add_argument("--cvtg_areas", type=str, default="2,3,4,5")
    parser.add_argument("--target_keys", type=str, default="")
    parser.add_argument("--default_aspect_ratio", type=str, default="1:1")
    parser.add_argument("--wise_subdir", type=str, default="wise_raw")
    args = parser.parse_args()

    if args.total_shards <= 0:
        raise ValueError("--total_shards must be positive")
    if not 0 <= args.shard_id < args.total_shards:
        raise ValueError("--shard_id must be in [0, total_shards)")

    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    os.makedirs(args.output_dir, exist_ok=True)

    model, vae_model, tokenizer, new_token_ids = load_bagel_model(
        args.model_path,
        device=device,
        max_latent_size=args.max_latent_size,
        accelerate_device_map=args.accelerate_device_map,
        max_memory=args.max_memory,
        offload_folder=args.offload_folder,
    )

    records = load_records(args)
    rank_records = records[args.shard_id :: args.total_shards]
    print(
        f"benchmark={args.benchmark} shard={args.shard_id}/{args.total_shards} "
        f"records={len(rank_records)} output={args.output_dir}",
        flush=True,
    )

    for sample in tqdm(rank_records, desc=f"{args.benchmark} shard {args.shard_id}"):
        stable_index = int(sample["_stable_index"])
        folder, sample_dir, output_paths = output_paths_for_sample(args, sample, stable_index)
        os.makedirs(sample_dir, exist_ok=True)
        if all_outputs_exist(output_paths):
            print(f"skip existing sample {stable_index}", flush=True)
            continue

        if args.benchmark == "tiifbench":
            prompts = [sample["short_description"], sample["long_description"]]
            for desc_offset, prompt in enumerate(prompts):
                if all_outputs_exist([output_paths[desc_offset]]):
                    continue
                seed_for_sample(stable_index * 2 + desc_offset, args.seed)
                images = generate_images(
                    model,
                    vae_model,
                    tokenizer,
                    new_token_ids,
                    prompt,
                    resolution=args.resolution,
                    num_images=1,
                    batch_size=1,
                    num_timesteps=args.num_timesteps,
                    cfg_scale=args.cfg_scale,
                    cfg_norm=args.cfg_norm,
                    timestep_shift=args.timestep_shift,
                    device=device,
                )
                os.makedirs(os.path.dirname(output_paths[desc_offset]), exist_ok=True)
                atomic_save(images[0], output_paths[desc_offset])
            continue

        if args.benchmark == "wise":
            prompt = sample["Prompt"]
        else:
            prompt = sample["prompt"]
        seed_for_sample(stable_index, args.seed)
        sample_resolution = args.resolution
        if args.benchmark == "bizgeneval":
            width, height = resolve_bizgeneval_size(
                sample,
                long_side=args.resolution,
                default_aspect_ratio=args.default_aspect_ratio,
            )
            sample_resolution = (height, width)
        if args.benchmark == "dpgbench":
            grid_cols = min(2, args.num_images)
            grid_rows = math.ceil(args.num_images / grid_cols)
            images = generate_images(
                model,
                vae_model,
                tokenizer,
                new_token_ids,
                prompt,
                resolution=args.resolution,
                num_images=args.num_images,
                batch_size=args.batch_size,
                num_timesteps=args.num_timesteps,
                cfg_scale=args.cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                device=device,
            )
            atomic_save(make_grid(images, args.resolution, grid_rows, grid_cols), output_paths[0])
            continue

        if args.benchmark == "oneigbench":
            grid_cols = min(2, args.num_images)
            grid_rows = math.ceil(args.num_images / grid_cols)
            images = generate_images(
                model,
                vae_model,
                tokenizer,
                new_token_ids,
                prompt,
                resolution=args.resolution,
                num_images=args.num_images,
                batch_size=args.batch_size,
                num_timesteps=args.num_timesteps,
                cfg_scale=args.cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                device=device,
            )
            atomic_save(make_grid(images, args.resolution, grid_rows, grid_cols), output_paths[0])
            continue

        images = generate_images(
            model,
            vae_model,
            tokenizer,
            new_token_ids,
            prompt,
            resolution=sample_resolution,
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_timesteps=args.num_timesteps,
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            timestep_shift=args.timestep_shift,
            device=device,
        )
        if args.benchmark == "geneval":
            with open(os.path.join(folder, "metadata.jsonl"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps({k: v for k, v in sample.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
            for idx, image in enumerate(images):
                atomic_save(image, output_paths[idx])
        elif args.benchmark == "qwenimagebench":
            atomic_save(images[0], output_paths[0])
        elif args.benchmark == "longtextbench":
            grid_cols = min(2, args.num_images)
            grid_rows = math.ceil(args.num_images / grid_cols)
            atomic_save(make_grid(images, args.resolution, grid_rows, grid_cols), output_paths[0])
            for image, path in zip(images, output_paths[1:]):
                atomic_save(image, path)
        elif args.benchmark in {"cvtg_2k", "bizgeneval", "wise"}:
            atomic_save(images[0], output_paths[0])

    print(f"shard {args.shard_id} finished", flush=True)


if __name__ == "__main__":
    main()
