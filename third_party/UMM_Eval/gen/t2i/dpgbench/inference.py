import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import os
import math
import random
import argparse
import torch.distributed as dist

from expert_usage import ExpertUsageTracker
from pe_prompts import load_pe_prompts, resolve_pe_prompt
from sample_seed import seed_for_sample
import output_paths

# DPG-Bench prompts ship co-located under eval/.
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "prompts.jsonl")


def set_random_seeds(seed_value):
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

    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_numeric(value):
    return f"{value:g}"




NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


class NEOT2IInferenceEngine:
    def __init__(
        self,
        model_path,
        device="cuda",
        device_map=None,
        max_memory=None,
        collect_expert_usage=False,
    ):
        self.device = device
        self.device_map = device_map
        self.expert_usage_tracker = None

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
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(
                self.device
            )

        # total_params = sum(p.numel() for p in self.model.language_model.model.layers.parameters())
        # print(total_params/1e6)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if collect_expert_usage:
            self.expert_usage_tracker = ExpertUsageTracker()
            num_gates = self.expert_usage_tracker.attach(self.model)
            print(f"Attached expert usage hooks to {num_gates} MoE gate modules")
            if num_gates == 0:
                print(
                    "WARNING: collect_expert_usage was enabled, but no MoE modules were matched."
                )

    def chat(
        self,
        prompt,
        cfg_scale=1.0,
        enable_timestep_shift=True,
        timestep_shift=1.0,
        image_size=(256, 256),
        num_steps=50,
        batch_size=1,
        micro_batch_size=1,
        output_grid_size=None,
        return_expert_usage=False,
        resize_to=None,
        cfg_norm='global',
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if micro_batch_size <= 0:
            raise ValueError(
                f"micro_batch_size must be positive, got {micro_batch_size}"
            )

        image_list = []
        remaining = batch_size
        if self.expert_usage_tracker is not None:
            self.expert_usage_tracker.start_capture()
        while remaining > 0:
            cur_batch_size = min(remaining, micro_batch_size)
            with torch.inference_mode():
                output = self.model.t2i_generate(
                    self.tokenizer,
                    prompt,
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    timestep_shift=timestep_shift,
                    enable_timestep_shift=enable_timestep_shift,
                    num_steps=num_steps,
                    batch_size=cur_batch_size,
                    cfg_norm=cfg_norm,
                )

            image = self._denorm(output)
            image = image.detach().to(device="cpu", dtype=torch.float32)
            image = (
                (image.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0)
                .round()
                .astype(np.uint8)
            )
            image_list.extend(Image.fromarray(image[i]) for i in range(cur_batch_size))

            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            remaining -= cur_batch_size

        expert_usage = None
        if self.expert_usage_tracker is not None:
            expert_usage = self.expert_usage_tracker.finish_capture()

        if output_grid_size is None:
            output_grid_size = (1, batch_size)
        pad = 0
        rows, cols = output_grid_size
        W, H = image_size
        if batch_size == 1:
            grid_image = image_list[0]
        else:
            grid = np.zeros(
                (rows * H + (rows - 1) * pad, cols * W + (cols - 1) * pad, 3),
                dtype=np.uint8,
            )

            for idx in range(batch_size):
                r, c = divmod(idx, cols)
                y = r * (H + pad)
                x = c * (W + pad)
                grid[y : y + H, x : x + W] = np.asarray(image_list[idx])

            grid_image = Image.fromarray(grid)

        if resize_to is not None and resize_to > 0:
            target = (resize_to * cols, resize_to * rows)
            if grid_image.size != target:
                grid_image = grid_image.resize(target, Image.LANCZOS)

        if return_expert_usage:
            return grid_image, expert_usage
        return grid_image

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        """
        x: [B,3,H,W] normalized ((img-mean)/std). returns [0,1] clamped.
        """
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--step_name", type=str)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Directory to load model weights from. "
        "model_name/step_name are still used for naming the output dir.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="DPG-Bench prompts JSONL. Defaults to the bundled "
        "benchmarks/dpgbench/prompts.jsonl.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="If set, write generated images directly under this absolute path "
        "(no further subdir composition). Otherwise build "
        "<cwd>/<output_root>/<model_name>/<step_name>/dpgbench under cwd.",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--cfg_norm", type=str, default="global")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_memory_per_gpu_gb", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose output PNG already exists.",
    )
    parser.add_argument(
        "--pe_prompt_path",
        type=str,
        default=None,
        help="Optional structured prompt-enhancement JSONL used for generation only.",
    )
    parser.add_argument("--collect_expert_usage", action="store_true")
    parser.add_argument("--expert_usage_root", type=str, default=None)
    parser.add_argument(
        "--resize_to",
        type=int,
        default=None,
        help="Optional target cell (sub-image) edge length. If set, the saved grid is resized so each cell is resize_to x resize_to.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, num_shards), got shard_rank={args.shard_rank}, num_shards={args.num_shards}"
        )

    if args.device_map is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError(
                "device_map mode must be run as a single process. Use python directly or torchrun --nproc_per_node=1."
            )
        world_size, rank = 1, 0
        device = None
    else:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"

    model_name = args.model_name
    step_name = args.step_name
    if args.model_path is None:
        raise ValueError("--model_path is required.")
    model_path = args.model_path
    data_path = args.data_path
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_folder = output_paths.derive_t2i_root(
            args.image_size, args.cfg_scale, args.timestep_shift, args.cfg_norm
        )
        output_path = os.path.join(output_folder, model_name, step_name, "dpgbench")
    os.makedirs(output_path, exist_ok=True)
    image_size = (args.image_size, args.image_size)
    grid_cols = min(2, args.batch_size)
    grid_rows = math.ceil(args.batch_size / grid_cols)
    enable_timestep_shift = args.timestep_shift >= 1.0

    max_memory = None
    if args.device_map is not None and args.max_memory_per_gpu_gb is not None:
        max_memory = {
            gpu_idx: f"{args.max_memory_per_gpu_gb}GiB"
            for gpu_idx in range(torch.cuda.device_count())
        }

    engine = NEOT2IInferenceEngine(
        model_path,
        device=device,
        device_map=args.device_map,
        max_memory=max_memory,
        collect_expert_usage=args.collect_expert_usage,
    )

    data = read_jsonl(data_path)
    pe_prompts = load_pe_prompts(args.pe_prompt_path) if args.pe_prompt_path else None
    for i, sample in enumerate(data):
        sample["index"] = i
    # set_random_seeds(42)
    random.seed(42)
    random.shuffle(data)
    total_shards = world_size * args.num_shards
    global_shard_rank = rank * args.num_shards + args.shard_rank
    rank_data = data[global_shard_rank::total_shards]
    assigned_count = len(rank_data)
    if args.resume:
        rank_data = [
            sample
            for sample in rank_data
            if not os.path.isfile(os.path.join(output_path, f"{sample['id']}.png"))
        ]
    print(
        f"Processing shard {global_shard_rank + 1}/{total_shards} "
        f"(ddp_rank={rank}/{world_size}, local_shard={args.shard_rank}/{args.num_shards}) "
        f"with {len(rank_data)} pending samples ({assigned_count} assigned)"
    )

    for i in tqdm(range(len(rank_data)), disable=rank != 0):
        cur_sample = rank_data[i]
        prompt_id = cur_sample["id"]
        original_prompt = cur_sample["prompt"]
        prompt = resolve_pe_prompt(original_prompt, prompt_id, pe_prompts)

        # Reproducible per-sample noise, keyed off the stable (pre-shuffle) dataset index.
        seed_for_sample(cur_sample["index"])
        chat_result = engine.chat(
            prompt,
            cfg_scale=args.cfg_scale,
            enable_timestep_shift=enable_timestep_shift,
            timestep_shift=args.timestep_shift,
            image_size=image_size,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            output_grid_size=(grid_rows, grid_cols),
            return_expert_usage=args.collect_expert_usage,
            resize_to=args.resize_to,
            cfg_norm=args.cfg_norm,
        )
        if args.collect_expert_usage:
            grid_image, expert_usage = chat_result
        else:
            grid_image = chat_result
            expert_usage = None

        grid_image.save(os.path.join(output_path, f"{prompt_id}.png"))
        if expert_usage is not None:
            expert_usage_root = (
                args.expert_usage_root
                if args.expert_usage_root is not None
                else os.path.join(output_path, "expert_usage")
            )
            os.makedirs(expert_usage_root, exist_ok=True)
            write_json(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "expert_usage": expert_usage,
                },
                os.path.join(expert_usage_root, f"{prompt_id}.json"),
            )

    print(f"Rank {rank} has finished processing {len(rank_data)} examples")
    if engine.expert_usage_tracker is not None:
        summary_path = os.path.join(
            output_path,
            f"expert_usage_summary_shard_{global_shard_rank}.json",
        )
        write_json(
            {
                "model_name": model_name,
                "step_name": step_name,
                "ddp_rank": rank,
                "world_size": world_size,
                "global_shard_rank": global_shard_rank,
                "num_shards": total_shards,
                "num_processed_samples": len(rank_data),
                "expert_usage": engine.expert_usage_tracker.summarize_aggregate(),
            },
            summary_path,
        )
    if dist.is_initialized():
        dist.barrier()
