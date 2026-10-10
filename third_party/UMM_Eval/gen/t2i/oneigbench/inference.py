import argparse
import os
import random

import megfile
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pe_prompts import load_pe_prompts, resolve_pe_prompt
from sample_seed import seed_for_sample


def create_image_gallery(images, rows=2, cols=2):
    assert len(images) >= rows * cols, "Not enough images provided!"
    img_width, img_height = images[0].size

    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))

    for row in range(rows):
        for col in range(cols):
            img = images[row * cols + col]
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))

    return gallery_image


def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


class NEOT2IInferenceEngine:
    def __init__(self, model_path, device="cuda", device_map=None, max_memory=None):
        self.device = device
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if device_map is not None:
            model_kwargs["device_map"] = device_map
            model_kwargs["low_cpu_mem_usage"] = True
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    def chat(
        self,
        prompt,
        cfg_scale=1.0,
        enable_timestep_shift=True,
        timestep_shift=1,
        cfg_norm="none",
        image_size=(256, 256),
        num_steps=50,
        batch_size=1,
        output_grid_size=None,
        method="euler",
        cfg_interval=(0.1, 1.0),
        t_eps=0.02,
    ):
        output = self.model.t2i_generate(
            self.tokenizer,
            prompt,
            image_size=image_size,
            cfg_scale=cfg_scale,
            timestep_shift=timestep_shift,
            enable_timestep_shift=enable_timestep_shift,
            cfg_norm=cfg_norm,
            num_steps=num_steps,
            batch_size=batch_size,
            method=method,
            cfg_interval=cfg_interval,
            t_eps=t_eps,
        )

        image = self._denorm(output.float())
        image = (image.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if output_grid_size is None:
            output_grid_size = (1, batch_size)

        image_list = [Image.fromarray(image[i]) for i in range(batch_size)]

        pad = 0
        rows, cols = output_grid_size
        width, height = image_size
        if batch_size == 1:
            grid_image = Image.fromarray(image[0])
        else:
            grid = np.zeros(
                (
                    rows * height + (rows - 1) * pad,
                    cols * width + (cols - 1) * pad,
                    3,
                ),
                dtype=np.uint8,
            )

            for idx in range(batch_size):
                r, c = divmod(idx, cols)
                y = r * (height + pad)
                x = c * (width + pad)
                grid[y:y + height, x:x + width] = image[idx]

            grid_image = Image.fromarray(grid)

        return image_list, grid_image


CLASS_ITEM = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneIG-Bench Evaluation for NEO Text2Image Model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Subdirectory name under each category for saved images. "
                             "Must match the --model_names value used by OneIG-Bench eval. "
                             "Falls back to basename(dirname(model_path)) for backward compatibility.")
    parser.add_argument("--output_path", type=str, required=True)
    # Default to the co-located (symlinked) benchmark data; launcher passes --data_root.
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "OneIG-Benchmark"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_norm", type=str, default="global")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--method", type=str, default="euler")
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--t_eps", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_cols", type=int, default=2)
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--use_zh_prompt", action="store_true")
    parser.add_argument("--pe_prompt_path", type=str, default=None,
                        help="Optional structured prompt-enhancement JSONL used for generation only.")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_memory_per_gpu_gb", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip samples whose <id>.webp already exists at the target path.")
    parser.add_argument(
        "--resize_to",
        type=int,
        default=None,
        help="Optional target cell (sub-image) edge length. If set, the saved grid is resized so each cell is resize_to x resize_to.",
    )
    args = parser.parse_args()

    if args.use_zh_prompt:
        args.language = "zh"

    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, num_shards), got shard_rank={args.shard_rank}, num_shards={args.num_shards}"
        )

    enable_timestep_shift = args.timestep_shift >= 1
    assert args.cfg_norm in ["none", "global", "channel", "cfg_zero_star"]

    if args.device_map is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError(
                "device_map mode must be run as a single process. "
                "Use python directly or torchrun --nproc_per_node=1."
            )
        world_size, rank = 1, 0
        device = None
    else:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"
    set_random_seeds(42)

    max_memory = None
    if args.device_map is not None and args.max_memory_per_gpu_gb is not None:
        max_memory = {
            gpu_idx: f"{args.max_memory_per_gpu_gb}GiB"
            for gpu_idx in range(torch.cuda.device_count())
        }

    engine = NEOT2IInferenceEngine(
        args.model_path,
        device=device,
        device_map=args.device_map,
        max_memory=max_memory,
    )
    grid = (args.grid_rows, args.grid_cols)
    total_slots = grid[0] * grid[1]

    use_zh_prompt = args.language == "zh"
    dataset_name = "OneIG-Bench-ZH.csv" if use_zh_prompt else "OneIG-Bench.csv"
    prompt_key = "prompt_cn" if use_zh_prompt else "prompt_en"
    df = pd.read_csv(os.path.join(args.data_root, dataset_name), dtype=str)
    pe_prompts = load_pe_prompts(args.pe_prompt_path) if args.pe_prompt_path else None

    os.makedirs(args.output_path, exist_ok=True)

    df_list = df.to_dict("records")
    # Stable global index (load order, before shard) for reproducible per-sample seeding.
    for global_idx, row in enumerate(df_list):
        row["_global_idx"] = global_idx
    total_shards = world_size * args.num_shards
    global_shard_rank = rank * args.num_shards + args.shard_rank
    rank_samples = df_list[global_shard_rank::total_shards]
    print(
        f"Processing shard {global_shard_rank + 1}/{total_shards} "
        f"(ddp_rank={rank}/{world_size}, local_shard={args.shard_rank}/{args.num_shards}) "
        f"with {len(rank_samples)} samples"
    )

    skipped_count = 0
    if args.model_name:
        model_name_for_path = args.model_name
    else:
        model_name_for_path = os.path.basename(os.path.dirname(args.model_path))
    for row in tqdm(rank_samples, total=len(rank_samples), disable=(rank != 0)):
        category = row["category"]
        subfolder = CLASS_ITEM[category]
        file_path = megfile.smart_path_join(
            args.output_path,
            subfolder,
            model_name_for_path,
            f"{row['id']}.webp",
        )

        if args.resume and megfile.smart_exists(file_path):
            skipped_count += 1
            continue

        original_prompt = row[prompt_key]
        prompt_id = f"{category}/{row['id']}"
        prompt = resolve_pe_prompt(original_prompt, prompt_id, pe_prompts)
        # Reproducible per-sample noise, keyed off the stable (pre-shard) dataset index.
        seed_for_sample(row["_global_idx"])
        images, _ = engine.chat(
            prompt,
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            enable_timestep_shift=enable_timestep_shift,
            timestep_shift=args.timestep_shift,
            batch_size=args.batch_size,
            output_grid_size=grid,
            image_size=(args.resolution, args.resolution),
            num_steps=args.num_steps,
            method=args.method,
            cfg_interval=tuple(args.cfg_interval),
            t_eps=args.t_eps,
        )

        if len(images) == 0:
            black_img = Image.new("RGB", (args.resolution, args.resolution), color=(0, 0, 0))
            images = [black_img] * total_slots
        elif len(images) < total_slots:
            width, height = images[0].size
            black_img = Image.new("RGB", (width, height), color=(0, 0, 0))
            images += [black_img] * (total_slots - len(images))

        image_gallery = create_image_gallery(images, grid[0], grid[1])

        print("*"*20, f"args.resize_to = {args.resize_to}", "*"*20)
        if args.resize_to is not None and args.resize_to > 0:
            target = (args.resize_to * grid[1], args.resize_to * grid[0])
            if image_gallery.size != target:
                image_gallery = image_gallery.resize(target, Image.LANCZOS)

        parent_dir = os.path.dirname(file_path)
        os.makedirs(parent_dir, exist_ok=True)

        with megfile.smart_open(file_path, "wb") as f:
            image_gallery.save(f, format="webp")

    if args.resume:
        print(f"[shard {global_shard_rank}] resume: skipped {skipped_count}/{len(rank_samples)} already-existing samples")

    if args.device_map is None and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"\nOneIG-Bench generation finished. Results saved to: {args.output_path}")
        print(f"Grid: {grid[0]}x{grid[1]}")
        print(f"Resolution: {args.resolution}")
