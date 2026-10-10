import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from expert_usage import ExpertUsageTracker
from pe_prompts import load_pe_prompts, resolve_pe_prompt
import output_paths


DEFAULT_CATEGORIES = [
    "cultural_common_sense",
    "natural_science",
    "spatio-temporal_reasoning",
]
# Fallback model root only — launchers always pass --model_path. Override via MODEL_ROOT.
DEFAULT_MODEL_ROOT = os.environ.get("MODEL_ROOT", "")
# WISE category json files ship co-located under eval/data/.
DEFAULT_WISE_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "data")
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv(raw_value):
    if raw_value is None:
        return None
    values = [value.strip() for value in raw_value.split(",")]
    values = [value for value in values if value]
    return values if values else None


def parse_target_prompt_ids(raw_value):
    values = parse_csv(raw_value)
    if values is None:
        return None
    return {int(value) for value in values}


def format_numeric(value):
    return f"{value:g}"


def load_wise_data(data_root, categories):
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


def resolve_model_path(model_path, model_root, model_name, step_name):
    if model_path is not None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
        return model_path

    if model_name is None or step_name is None:
        raise ValueError("Either model_path or both model_name and step_name must be provided.")

    candidate_roots = []
    if model_root is not None:
        candidate_roots.append(os.path.join(model_root, model_name))
    if os.path.isdir(model_name):
        candidate_roots.append(model_name)

    checked_paths = []
    for root in candidate_roots:
        candidates = [
            os.path.join(root, f"{step_name}hf"),
            os.path.join(root, f"hf_step{step_name}"),
            os.path.join(root, step_name),
        ]
        for candidate in candidates:
            checked_paths.append(candidate)
            if os.path.isdir(candidate):
                return candidate

    checked_paths_str = "\n".join(f"  - {path}" for path in checked_paths) if checked_paths else "  (no candidates)"
    raise FileNotFoundError(
        f"Unable to resolve model path for model_name={model_name}, step_name={step_name}. "
        f"Checked:\n{checked_paths_str}"
    )


def resolve_output_path(output_path, image_size, cfg_scale, timestep_shift, model_name, step_name, output_subdir, cfg_norm='none'):
    if output_path is not None:
        return output_path
    if model_name is None or step_name is None:
        raise ValueError("Either output_path or both model_name and step_name must be provided.")
    output_root = output_paths.derive_t2i_root(image_size, cfg_scale, timestep_shift, cfg_norm)
    return os.path.join(output_root, model_name, step_name, output_subdir)


def extract_image_and_think(output, model):
    think_text = getattr(model, "last_think_content", "") or ""
    image_output = output

    if isinstance(output, tuple):
        if len(output) == 0:
            raise ValueError("Model returned an empty tuple from t2i_generate.")
        image_output = output[0]
        if len(output) > 1 and output[1] is not None:
            think_text = output[1]

    if think_text is None:
        think_text = ""
    elif not isinstance(think_text, str):
        think_text = str(think_text)

    return image_output, think_text


def build_think_trace(think_texts):
    entries = []
    merged_entries = []

    for micro_batch_index, think_text in enumerate(think_texts):
        normalized_text = think_text or ""
        entries.append(
            {
                "micro_batch_index": micro_batch_index,
                "think_content": normalized_text,
            }
        )
        if normalized_text:
            merged_entries.append(normalized_text)

    merged_think_content = ""
    if merged_entries:
        unique_entries = []
        for text in merged_entries:
            if text not in unique_entries:
                unique_entries.append(text)
        if len(unique_entries) == 1:
            merged_think_content = unique_entries[0]
        else:
            merged_think_content = "\n\n".join(
                f"[micro_batch {idx}]\n{entry['think_content']}"
                for idx, entry in enumerate(entries)
                if entry["think_content"]
            )

    return {
        "think_content": merged_think_content,
        "micro_batch_think_contents": entries,
        "num_micro_batches": len(entries),
        "num_nonempty_micro_batches": sum(bool(entry["think_content"]) for entry in entries),
    }


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
            # debug_key_match=True,
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
        if collect_expert_usage:
            self.expert_usage_tracker = ExpertUsageTracker()
            num_gates = self.expert_usage_tracker.attach(self.model)
            print(f"Attached expert usage hooks to {num_gates} MoE gate modules")
            if num_gates == 0:
                print("WARNING: collect_expert_usage was enabled, but no MoE modules were matched.")

    def chat(
        self,
        prompt,
        cfg_scale=1.0,
        cfg_norm='global',
        enable_timestep_shift=True,
        timestep_shift=1.0,
        image_size=(256, 256),
        num_steps=50,
        batch_size=1,
        micro_batch_size=1,
        output_grid_size=None,
        return_expert_usage=False,
        think_mode=False,
        return_think_trace=False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")

        image_list = []
        think_texts = []
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
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    enable_timestep_shift=enable_timestep_shift,
                    num_steps=num_steps,
                    batch_size=cur_batch_size,
                    think_mode=think_mode,
                )

            if think_mode:
                output, think_text = extract_image_and_think(output, self.model)
                think_texts.append(think_text)

            image = self._denorm(output)
            image = image.detach().to(device="cpu", dtype=torch.float32)
            image = (image.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
            image_list.extend(Image.fromarray(image[i]) for i in range(cur_batch_size))

            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            remaining -= cur_batch_size

        expert_usage = None
        if self.expert_usage_tracker is not None:
            expert_usage = self.expert_usage_tracker.finish_capture()
        think_trace = build_think_trace(think_texts) if think_mode else None

        if output_grid_size is None:
            output_grid_size = (1, batch_size)

        pad = 0
        rows, cols = output_grid_size
        width, height = image_size
        if batch_size == 1:
            grid_image = image_list[0]
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
                row, col = divmod(idx, cols)
                y = row * (height + pad)
                x = col * (width + pad)
                grid[y : y + height, x : x + width] = np.asarray(image_list[idx])

            grid_image = Image.fromarray(grid)

        if return_expert_usage and return_think_trace:
            return grid_image, expert_usage, think_trace
        if return_expert_usage:
            return grid_image, expert_usage
        if return_think_trace:
            return grid_image, think_trace
        return grid_image

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--step_name", type=str, default=None)
    parser.add_argument("--model_root", type=str, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="If set, load weights from this directory directly, "
        "skipping model_root/model_name/step_name resolution.",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wise_data_root", type=str, default=DEFAULT_WISE_DATA_ROOT)
    parser.add_argument("--pe_prompt_path", type=str, default=None,
                        help="Optional structured prompt-enhancement JSONL used for generation only.")
    parser.add_argument("--categories", type=str, default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--target_prompt_ids", type=str, default=None)
    parser.add_argument("--image_size", "--resolution", dest="image_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", "--cfg", dest="cfg_scale", type=float, default=4.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--cfg_norm", type=str, default="global")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--think_mode", type=str_to_bool, default=False)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_memory_per_gpu_gb", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--collect_expert_usage", action="store_true")
    parser.add_argument("--expert_usage_root", type=str, default=None)
    parser.add_argument("--output_subdir", type=str, default=None)
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
                "device_map mode must be run as a single process. "
                "Use python directly or torchrun --nproc_per_node=1."
            )
        world_size, rank = 1, 0
        device = None
    else:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"

    model_path = resolve_model_path(args.model_path, args.model_root, args.model_name, args.step_name)
    print(f"Resolved model path: {model_path}")
    print(f"Think mode: {args.think_mode}")

    categories = parse_csv(args.categories) or list(DEFAULT_CATEGORIES)
    target_prompt_ids = parse_target_prompt_ids(args.target_prompt_ids)

    data = load_wise_data(args.wise_data_root, categories)
    pe_prompts = load_pe_prompts(args.pe_prompt_path) if args.pe_prompt_path else None
    if target_prompt_ids is not None:
        data = [sample for sample in data if int(sample["prompt_id"]) in target_prompt_ids]
        print(f"Filtered to target prompt IDs: {sorted(target_prompt_ids)} => {len(data)} samples")

    if args.output_subdir is None:
        args.output_subdir = "wise_think" if args.think_mode else "wise_raw"

    output_path = resolve_output_path(
        args.output_path,
        args.image_size,
        args.cfg_scale,
        args.timestep_shift,
        args.model_name,
        args.step_name,
        args.output_subdir,
        args.cfg_norm,
    )
    print(f"Saving outputs to: {output_path}")
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

    random.seed(42)
    random.shuffle(data)
    total_shards = world_size * args.num_shards
    global_shard_rank = rank * args.num_shards + args.shard_rank
    rank_data = data[global_shard_rank::total_shards]
    print(
        f"Processing shard {global_shard_rank + 1}/{total_shards} "
        f"(ddp_rank={rank}/{world_size}, local_shard={args.shard_rank}/{args.num_shards}) "
        f"with {len(rank_data)} samples"
    )

    for cur_sample in tqdm(rank_data, disable=rank != 0):
        prompt_id = int(cur_sample["prompt_id"])
        original_prompt = cur_sample["Prompt"]
        prompt = resolve_pe_prompt(original_prompt, prompt_id, pe_prompts)

        chat_result = engine.chat(
            prompt,
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            enable_timestep_shift=enable_timestep_shift,
            timestep_shift=args.timestep_shift,
            image_size=image_size,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            output_grid_size=(grid_rows, grid_cols),
            return_expert_usage=args.collect_expert_usage,
            think_mode=args.think_mode,
            return_think_trace=args.think_mode,
        )
        if args.collect_expert_usage and args.think_mode:
            grid_image, expert_usage, think_trace = chat_result
        elif args.collect_expert_usage:
            grid_image, expert_usage = chat_result
            think_trace = None
        elif args.think_mode:
            grid_image, think_trace = chat_result
            expert_usage = None
        else:
            grid_image = chat_result
            expert_usage = None
            think_trace = None

        grid_image.save(os.path.join(output_path, f"{prompt_id}.png"))
        if think_trace is not None:
            think_content_root = os.path.join(output_path, "think_content")
            os.makedirs(think_content_root, exist_ok=True)
            write_json(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": cur_sample.get("wise_category"),
                    "think_mode": args.think_mode,
                    "think_content": think_trace["think_content"],
                    "micro_batch_think_contents": think_trace["micro_batch_think_contents"],
                    "batch_size": args.batch_size,
                    "micro_batch_size": args.micro_batch_size,
                },
                os.path.join(think_content_root, f"{prompt_id}.json"),
            )
        if expert_usage is not None:
            expert_usage_root = (
                args.expert_usage_root
                if args.expert_usage_root is not None
                else os.path.join(output_path, "expert_usage")
            )
            os.makedirs(expert_usage_root, exist_ok=True)
            expert_usage_payload = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "category": cur_sample.get("wise_category"),
                "expert_usage": expert_usage,
            }
            if think_trace is not None:
                expert_usage_payload["think_mode"] = args.think_mode
                expert_usage_payload["think_content"] = think_trace["think_content"]
                expert_usage_payload["micro_batch_think_contents"] = think_trace["micro_batch_think_contents"]
            write_json(
                expert_usage_payload,
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
                "model_name": args.model_name,
                "step_name": args.step_name,
                "categories": categories,
                "think_mode": args.think_mode,
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
