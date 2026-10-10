import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from PIL import Image
import torch.distributed as dist
from tqdm import tqdm

from sample_seed import seed_for_sample


def set_random_seeds(seed_value, rank=0):
    seed_value += rank
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}", flush=True)
    return local_rank, world_size, rank


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_jsonl_files(input_folder, specific_file=None):
    if specific_file is not None:
        return [os.path.join(input_folder, specific_file)]
    return sorted(
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.endswith(".jsonl")
    )


def load_all_samples(jsonl_files):
    all_samples = []
    file_sample_counts = []

    for jsonl_file in jsonl_files:
        data = read_jsonl(jsonl_file)
        file_sample_counts.append((os.path.basename(jsonl_file), len(data)))
        for sample_idx, item in enumerate(data):
            sample = dict(item)
            sample["_sample_idx"] = sample_idx
            sample["_source_file"] = os.path.basename(jsonl_file)
            all_samples.append(sample)

    # Stable global index (load order, before shard) for reproducible per-sample seeding.
    for global_idx, sample in enumerate(all_samples):
        sample["_global_idx"] = global_idx

    return all_samples, file_sample_counts

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]

class NEOT2IInferenceEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        # total_params = sum(p.numel() for p in self.model.language_model.model.layers.parameters())
        # print(total_params/1e6)

        RL_CKPT = os.getenv('RL_CKPT')
        if RL_CKPT is not None and os.path.exists(RL_CKPT):
            self._load_checkpoint(RL_CKPT)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint with trainable parameters"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load trainable parameters
        missing_keys = []
        unexpected_keys = []
        matched_keys = []
        key_name = 'trainable_state_dict' if 'trainable_state_dict' in checkpoint else 'model'
        for name, param in self.model.named_parameters():
            if name in checkpoint[key_name]:
                param.data.copy_(checkpoint[key_name][name])
                matched_keys.append(name)
            elif param.requires_grad:
                missing_keys.append(name)

        model_keys = set([n for n, p in self.model.named_parameters()])
        unexpected_keys = [name for name in checkpoint[key_name].keys() if name not in model_keys]

        if missing_keys:
            print(f"Warning: Missing keys when loading checkpoint: {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:10]}")
        if matched_keys:
            print(f"Successfully loaded matched keys: {matched_keys[:10]}")

        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        print(f"Loaded checkpoint: epoch={epoch}, global_step={global_step}")

    def chat(self, prompt, cfg_scale=1.0, enable_timestep_shift=True, timestep_shift=1, cfg_norm='none', image_size=(256, 256), num_steps=50, batch_size=1, output_grid_size=None):
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
            cfg_interval=(0, 1)
        )

        image = self._denorm(output.float())
        image = (image.clamp(0,1).permute(0,2,3,1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if output_grid_size is None:
            output_grid_size = (1, batch_size)

        image_list = [Image.fromarray(image[i]) for i in range(batch_size)]

        pad = 2
        rows, cols = output_grid_size
        W, H = image_size
        if batch_size == 1:
            grid_image = Image.fromarray(image[0])
        else:
            grid = np.zeros(
                (rows * H + (rows - 1) * pad,
                cols * W + (cols - 1) * pad,
                3),
                dtype=np.uint8
            )

            for idx in range(batch_size):
                r, c = divmod(idx, cols)
                y = r * (H + pad)
                x = c * (W + pad)
                grid[y:y+H, x:x+W] = image[idx]

            grid_image = Image.fromarray(grid)

        return image_list, grid_image

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        """
        x: [B,3,H,W] normalized ((img-mean)/std). returns [0,1] clamped.
        """
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="TIIF-Bench inference for NEO model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--cfg_norm", type=str, default="global")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--language", type=str, default="en", choices=["en"])
    parser.add_argument("--specific_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, num_shards), got shard_rank={args.shard_rank}, num_shards={args.num_shards}"
        )

    enable_timestep_shift = args.timestep_shift >= 1
    assert args.cfg_norm in ["none", "global", "channel", "cfg_zero_star"]

    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"
    set_random_seeds(args.seed, rank)

    engine = NEOT2IInferenceEngine(args.model_path, device=device)

    jsonl_files = get_jsonl_files(args.input_folder, args.specific_file)
    os.makedirs(args.output_path, exist_ok=True)
    model_name = args.model_name or os.path.basename(os.path.dirname(args.model_path))

    if rank == 0:
        print(f"Total prompt files: {len(jsonl_files)}", flush=True)
        print(f"Output path: {args.output_path}", flush=True)

    all_samples, file_sample_counts = load_all_samples(jsonl_files)
    total_shards = world_size * args.num_shards
    global_shard_rank = rank * args.num_shards + args.shard_rank
    rank_data = all_samples[global_shard_rank::total_shards]

    if rank == 0:
        for file_name, sample_count in file_sample_counts:
            print(f"Loaded file: {file_name} | total samples: {sample_count}", flush=True)
        print(f"Total samples across all files: {len(all_samples)}", flush=True)
        print(
            f"Processing shard {global_shard_rank + 1}/{total_shards} "
            f"(ddp_rank={rank}/{world_size}, local_shard={args.shard_rank}/{args.num_shards}) "
            f"with {len(rank_data)} samples",
            flush=True,
        )

    for item in tqdm(rank_data, disable=rank != 0):
        sample_idx = item["_sample_idx"]
        data_type = item["type"]
        prompts = {
            "short_description": item["short_description"],
            "long_description": item["long_description"],
        }

        save_root = Path(args.output_path) / data_type / model_name
        for desc_offset, (prompt_type, prompt) in enumerate(prompts.items()):
            save_dir = save_root / prompt_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_idx}.png"

            if save_path.exists():
                continue

            # Reproducible per-(sample, description) noise from the stable dataset index.
            seed_for_sample(item["_global_idx"] * 2 + desc_offset)
            _, grid_image = engine.chat(
                    prompt=prompt,
                    cfg_scale=args.cfg_scale,
                    cfg_norm=args.cfg_norm,
                    enable_timestep_shift=enable_timestep_shift,
                    timestep_shift=args.timestep_shift,
                    image_size=(args.resolution, args.resolution),
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    output_grid_size=(args.rows, args.cols),
                )
            grid_image.save(save_path)

    dist.barrier()
    if rank == 0:
        print("TIIF-Bench generation finished.")
        print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
