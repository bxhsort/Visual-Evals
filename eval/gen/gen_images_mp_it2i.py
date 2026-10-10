# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Unified BAGEL image-to-image (editing) benchmark generator.

One entry point for the editing benchmarks (imgedit, rise, geditbench). It mirrors
the text-to-image harness in ``gen_images_mp_t2i.py`` (single process per GPU,
``--shard_id``/``--total_shards`` striding, ``--accelerate_device_map``, skip-existing
and ``--target_indices``), while driving the editing inference core
(``editing_image`` / ``editing_image_with_think``) from ``gen_images_mp_imgedit.py``.

Per-benchmark metadata → source image → output layout:

  imgedit    metadata: JSON dict {key: {id, prompt, ...}}
             source:   {image_root}/{id}
             output:   {output_dir}/{key}.png
  rise       metadata: JSON list [{index, category, instruction, image, ...}]
             source:   {image_root}/{image}
             output:   {output_dir}/{category}/{index}.png
  geditbench metadata: HF dataset (load_from_disk / load_dataset)
             source:   row["input_image"]
             output:   {output_dir}/fullset/{task_type}/{instruction_language}/{key}.png
                       (+ {key}_SRCIMG.png alongside)
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer

from eval.gen import gen_images_mp_imgedit as bagel_edit


# Filled lazily by ``load_records`` for the geditbench HF dataset; rows are resolved
# by index because the source image lives in the dataset, not on disk.
_HF_DATASET = None


BENCHMARK_DEFAULTS = {
    "imgedit": {"cfg_text_scale": 4.0, "cfg_img_scale": 1.5, "cfg_interval": [0.0, 1.0], "crop_bbox": True},
    "rise": {"cfg_text_scale": 4.0, "cfg_img_scale": 2.0, "cfg_interval": [0.0, 1.0], "crop_bbox": True},
    "geditbench": {"cfg_text_scale": 4.0, "cfg_img_scale": 1.5, "cfg_interval": [0.4, 1.0], "crop_bbox": False},
}


def seed_for_sample(index: int, base_seed: int = 42) -> None:
    seed = int(base_seed) + int(index)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_csv(raw_value: str):
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_target_indices(raw):
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def load_gedit_dataset(path: str):
    from datasets import load_dataset, load_from_disk

    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "state.json")):
        return load_from_disk(path)
    return load_dataset(path)["train"]


def load_records(args):
    global _HF_DATASET

    if args.benchmark == "imgedit":
        with open(args.metadata_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        records = []
        for key, value in raw.items():
            record = dict(value)
            record["key"] = str(key)
            records.append(record)
    elif args.benchmark == "rise":
        with open(args.metadata_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        records = [dict(item) for item in raw]
    elif args.benchmark == "geditbench":
        _HF_DATASET = load_gedit_dataset(args.dataset_path)
        records = []
        for idx in range(len(_HF_DATASET)):
            row = _HF_DATASET[idx]
            records.append(
                {
                    "key": str(row["key"]),
                    "task_type": str(row["task_type"]),
                    "instruction_language": str(row["instruction_language"]),
                    "_dataset_idx": idx,
                }
            )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    for idx, sample in enumerate(records):
        sample["_stable_index"] = idx

    target_indices = parse_target_indices(args.target_indices)
    if target_indices is not None:
        records = [sample for sample in records if sample["_stable_index"] in target_indices]

    target_keys = set(parse_csv(args.target_keys)) if args.target_keys else None
    if target_keys is not None:
        records = [sample for sample in records if sample["key"] in target_keys]

    random.seed(42)
    random.shuffle(records)
    return records


def prompt_for_sample(args, sample):
    if args.benchmark == "geditbench":
        return _HF_DATASET[int(sample["_dataset_idx"])]["instruction"]
    if args.benchmark == "imgedit":
        return sample["prompt"]
    return sample["instruction"]


def source_image_for_sample(args, sample):
    if args.benchmark == "geditbench":
        return pil_img2rgb(_HF_DATASET[int(sample["_dataset_idx"])]["input_image"])
    if args.benchmark == "imgedit":
        relative_path = str(sample["id"])
    else:  # rise
        relative_path = str(sample["image"])
    return pil_img2rgb(Image.open(os.path.join(args.image_root, relative_path)))


def output_paths_for_sample(args, sample):
    if args.benchmark == "imgedit":
        return [os.path.join(args.output_dir, f"{sample['key']}.png")]
    if args.benchmark == "rise":
        return [os.path.join(args.output_dir, str(sample["category"]), f"{sample['index']}.png")]
    base = os.path.join(
        args.output_dir,
        "fullset",
        str(sample["task_type"]),
        str(sample["instruction_language"]),
    )
    return [os.path.join(base, f"{sample['key']}.png")]


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


def load_bagel_edit(
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

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

    checkpoint = os.path.join(model_path, "ema.safetensors")
    if accelerate_device_map:
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

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

    # The editing core reads these as module globals; bind them here.
    bagel_edit.gen_model = model
    bagel_edit.vae_model = vae_model
    bagel_edit.tokenizer = tokenizer
    bagel_edit.new_token_ids = new_token_ids
    bagel_edit.vae_transform = vae_transform
    bagel_edit.vit_transform = vit_transform
    return bagel_edit


def main():
    parser = argparse.ArgumentParser(description="Generate BAGEL image-editing benchmark outputs.")
    parser.add_argument(
        "--benchmark",
        choices=["imgedit", "rise", "geditbench"],
        required=True,
    )
    parser.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_file", default="", help="JSON file (imgedit/rise).")
    parser.add_argument("--dataset_path", default="", help="HF dataset id or save_to_disk path (geditbench).")
    parser.add_argument("--image_root", default=None, help="Root dir of the source images (imgedit/rise).")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_img_scale", type=float, default=None)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", "--num_steps", dest="num_timesteps", type=int, default=50)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--min_image_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--accelerate_device_map", action="store_true")
    parser.add_argument("--max_memory", type=str, default="80GiB")
    parser.add_argument("--offload_folder", type=str, default="offload_it2i")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--crop_bbox", action="store_true")
    parser.add_argument("--no_crop_bbox", action="store_true")
    parser.add_argument("--target_indices", type=str, default="")
    parser.add_argument("--target_keys", type=str, default="")
    args = parser.parse_args()

    if args.total_shards <= 0:
        raise ValueError("--total_shards must be positive")
    if not 0 <= args.shard_id < args.total_shards:
        raise ValueError("--shard_id must be in [0, total_shards)")

    defaults = BENCHMARK_DEFAULTS[args.benchmark]
    if args.cfg_text_scale is None:
        args.cfg_text_scale = defaults["cfg_text_scale"]
    if args.cfg_img_scale is None:
        args.cfg_img_scale = defaults["cfg_img_scale"]
    cfg_interval = list(args.cfg_interval) if args.cfg_interval is not None else list(defaults["cfg_interval"])
    if args.no_crop_bbox:
        crop_bbox = False
    elif args.crop_bbox:
        crop_bbox = True
    else:
        crop_bbox = defaults["crop_bbox"]

    if args.benchmark == "geditbench":
        if not args.dataset_path:
            raise ValueError("--dataset_path is required for geditbench")
    else:
        if not args.metadata_file:
            raise ValueError(f"--metadata_file is required for {args.benchmark}")
        if args.image_root is None:
            args.image_root = os.path.dirname(os.path.abspath(args.metadata_file))

    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    os.makedirs(args.output_dir, exist_ok=True)

    edit_api = load_bagel_edit(
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

    edit_kwargs = dict(
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=cfg_interval,
        cfg_renorm_min=0.0,
        cfg_renorm_type=args.cfg_renorm_type,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        max_image_size=args.max_image_size,
        min_image_size=args.min_image_size,
        device=device,
    )

    for sample in tqdm(rank_records, desc=f"{args.benchmark} shard {args.shard_id}"):
        stable_index = int(sample["_stable_index"])
        output_paths = output_paths_for_sample(args, sample)
        outpath = output_paths[0]
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if all_outputs_exist(output_paths):
            print(f"skip existing sample {stable_index}", flush=True)
            continue

        prompt = prompt_for_sample(args, sample)
        image = source_image_for_sample(args, sample)

        seed_for_sample(stable_index, args.seed)
        with torch.inference_mode():
            if args.think:
                edited_image, think_output = edit_api.editing_image_with_think(
                    images=[image],
                    prompt=prompt,
                    max_length=args.max_length,
                    simple_think=False,
                    **edit_kwargs,
                )
            else:
                edited_image = edit_api.editing_image(
                    images=[image],
                    prompt=prompt,
                    **edit_kwargs,
                )

        if crop_bbox and edited_image.getbbox():
            edited_image = edited_image.crop(edited_image.getbbox())

        if args.benchmark == "geditbench":
            image.save(os.path.join(os.path.dirname(outpath), f"{sample['key']}_SRCIMG.png"))

        edited_image.save(outpath)
        if args.think:
            with open(os.path.splitext(outpath)[0] + ".txt", "w", encoding="utf-8") as handle:
                handle.write(think_output)

    print(f"shard {args.shard_id} finished", flush=True)


if __name__ == "__main__":
    main()
