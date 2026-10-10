import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from eval.gen import gen_images_mp_imgedit as bagel_edit
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

from eval.gen.edit_compass.common import iter_task_configs


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def language_result_root(save_root, model_name, language):
    return Path(save_root) / model_name / language


def collect_tasks(data_root, result_root, language, parts):
    tasks = []
    for cfg in iter_task_configs(data_root, result_root, parts):
        if not cfg["json_path"].exists():
            print(f"[edit-compass-gen] missing source json: {cfg['json_path']}")
            continue
        for item in read_json(cfg["json_path"]):
            prompt_key = f"prompt_{language}"
            if prompt_key not in item:
                raise KeyError(f"{cfg['json_path']} item {item.get('save_id')} missing {prompt_key}")
            save_id = str(item["save_id"])
            output_image_path = cfg["save_dir"] / f"{save_id}.png"
            tasks.append(
                {
                    "part": cfg["part"],
                    "task_name": cfg["task_name"],
                    "save_id": item["save_id"],
                    "source_image": str(Path(data_root) / item["source_image"]),
                    "prompt": item[prompt_key],
                    "output_image_path": str(output_image_path),
                    "item": item,
                }
            )
    return tasks


def setup_bagel_model(model_path, device, accelerate_device_map=False, max_memory="80GiB", offload_folder="offload_edit_compass"):
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
        max_latent_size=64,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

    ema_path = os.path.join(model_path, "ema.safetensors")
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
        print(f"[edit-compass-gen] accelerate device_map: {device_map}")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=ema_path,
            device_map=device_map,
            offload_buffers=True,
            offload_folder=offload_folder,
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    else:
        state_dict = load_file(ema_path, device="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        del state_dict
        model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()

    bagel_edit.gen_model = model
    bagel_edit.vae_model = vae_model
    bagel_edit.tokenizer = tokenizer
    bagel_edit.new_token_ids = new_token_ids
    bagel_edit.vae_transform = vae_transform
    bagel_edit.vit_transform = vit_transform
    return bagel_edit


def generate(args):
    result_root = language_result_root(args.save_root, args.model_name, args.language)
    tasks = collect_tasks(args.data_root, result_root, args.language, args.parts)
    shard_tasks = tasks[args.shard_id :: args.total_shards]
    print(
        f"[edit-compass-gen] result_root={result_root} tasks={len(tasks)} "
        f"shard={args.shard_id}/{args.total_shards} shard_tasks={len(shard_tasks)}"
    )

    if args.dry_run:
        return

    set_seeds(args.seed + args.shard_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    model_api = setup_bagel_model(
        args.model_path,
        device=device,
        accelerate_device_map=args.accelerate_device_map,
        max_memory=args.max_memory,
        offload_folder=args.offload_folder,
    )

    shard_rows = []
    for index, task in enumerate(shard_tasks, 1):
        output_path = task["output_image_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        row = {
            "part": task["part"],
            "task_name": task["task_name"],
            "save_id": task["save_id"],
            "output_image_path": output_path,
        }
        try:
            if os.path.exists(output_path) and not args.overwrite:
                row["status"] = "success"
                row["skipped"] = True
                shard_rows.append(row)
                continue

            image = pil_img2rgb(Image.open(task["source_image"]))
            with torch.inference_mode():
                if args.think:
                    out_image, think_output = model_api.editing_image_with_think(
                        images=[image],
                        prompt=task["prompt"],
                        cfg_text_scale=args.cfg_text_scale,
                        cfg_img_scale=args.cfg_img_scale,
                        timestep_shift=args.timestep_shift,
                        num_timesteps=args.num_timesteps,
                        max_image_size=args.max_image_size,
                        min_image_size=args.min_image_size,
                        device=device,
                    )
                    with open(os.path.splitext(output_path)[0] + ".txt", "w", encoding="utf-8") as f:
                        f.write(think_output)
                else:
                    out_image = model_api.editing_image(
                        images=[image],
                        prompt=task["prompt"],
                        cfg_text_scale=args.cfg_text_scale,
                        cfg_img_scale=args.cfg_img_scale,
                        timestep_shift=args.timestep_shift,
                        num_timesteps=args.num_timesteps,
                        max_image_size=args.max_image_size,
                        min_image_size=args.min_image_size,
                        device=device,
                    )
            if args.crop_bbox and out_image.getbbox():
                out_image = out_image.crop(out_image.getbbox())
            out_image.save(output_path)
            row["status"] = "success"
            print(f"[edit-compass-gen] {index}/{len(shard_tasks)} saved {output_path}")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            print(f"[edit-compass-gen] failed {task['task_name']} save_id={task['save_id']}: {exc}")
        shard_rows.append(row)
        write_jsonl(args.shard_result, shard_rows)

    write_jsonl(args.shard_result, shard_rows)
    print(f"[edit-compass-gen] wrote shard result: {args.shard_result}")


def finalize(args):
    result_root = language_result_root(args.save_root, args.model_name, args.language)
    task_rows = {}
    for cfg in iter_task_configs(args.data_root, result_root, args.parts):
        if not cfg["json_path"].exists():
            continue
        rows = []
        for item in read_json(cfg["json_path"]):
            final_item = dict(item)
            output_path = cfg["save_dir"] / f"{item['save_id']}.png"
            final_item["output_image_path"] = str(output_path)
            if output_path.exists():
                final_item["status"] = "success"
            else:
                final_item["status"] = "missing"
            rows.append(final_item)
        task_rows[cfg["task_name"]] = rows
        write_json(str(cfg["save_dir"] / f"{cfg['task_name']}.json"), rows)
        ok = sum(1 for row in rows if row.get("status") == "success")
        print(f"[edit-compass-finalize] {cfg['part']}/{cfg['task_name']}: {ok}/{len(rows)}")

    manifest = {
        "data_root": args.data_root,
        "save_root": str(result_root),
        "model_name": args.model_name,
        "language": args.language,
        "parts": args.parts,
    }
    write_json(str(result_root / "manifest.json"), manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "finalize"], default="generate")
    parser.add_argument("--data_root", default=str(BAGEL_ROOT / "third_party" / "Edit-Compass-data"))
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--model_name", default="bagel")
    parser.add_argument("--language", default="en")
    parser.add_argument("--parts", nargs="+", default=["Part1", "Part2"])
    parser.add_argument("--model_path", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--shard_result", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--crop_bbox", action="store_true")
    parser.add_argument("--accelerate_device_map", action="store_true")
    parser.add_argument("--max_memory", default="80GiB")
    parser.add_argument("--offload_folder", default="offload_edit_compass")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--min_image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "generate":
        if not args.model_path:
            raise ValueError("--model_path is required for generation")
        if not args.shard_result:
            result_root = language_result_root(args.save_root, args.model_name, args.language)
            args.shard_result = str(result_root / "logs" / "generated_shard_00.jsonl")
        generate(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
