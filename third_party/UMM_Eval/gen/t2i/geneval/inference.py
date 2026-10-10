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
from sample_seed import seed_for_sample
import output_paths

def set_random_seeds(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
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
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_target_indices(raw_value):
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(',')]
    indices = []
    for part in parts:
        if not part:
            continue
        indices.append(int(part))
    return set(indices) if indices else None


def format_numeric(value):
    return f"{value:g}"

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


NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]

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
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)

        total_params = sum(p.numel() for p in self.model.language_model.model.layers.parameters())
        print(f"total params: {total_params:,}  => {total_params/1e9:.3f}B")

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
        enable_timestep_shift=True,
        timestep_shift=1.0,
        image_size=(256, 256),
        num_steps=50,
        batch_size=1,
        micro_batch_size=1,
        output_grid_size=None,
        return_expert_usage=False,
        think_mode=False,
        cfg_norm='global',
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
                    timestep_shift=timestep_shift,
                    enable_timestep_shift=enable_timestep_shift,
                    num_steps=num_steps,
                    think_mode=think_mode,
                    batch_size=cur_batch_size,
                    cfg_norm=cfg_norm,
                )

            if think_mode:
                output, think_text = extract_image_and_think(output, self.model)
                print(think_text)
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

        if output_grid_size is None:
            output_grid_size = (1, batch_size)
        
        pad = 2
        rows, cols = output_grid_size
        W, H = image_size
        if batch_size == 1:
            grid_image = image_list[0]
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
                grid[y:y+H, x:x+W] = np.asarray(image_list[idx])

            grid_image = Image.fromarray(grid)

        if return_expert_usage:
            return image_list, grid_image, expert_usage
        return image_list, grid_image

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        """
        x: [B,3,H,W] normalized ((img-mean)/std). returns [0,1] clamped.
        """
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--step_name', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--cfg_scale', type=float, default=4.0)
    parser.add_argument('--timestep_shift', type=float, default=3.0)
    parser.add_argument('--cfg_norm', type=str, default='global')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--micro_batch_size', type=int, default=1)
    parser.add_argument('--device_map', type=str, default=None)
    parser.add_argument('--max_memory_per_gpu_gb', type=int, default=None)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_rank', type=int, default=0)
    parser.add_argument('--collect_expert_usage', action='store_true')
    parser.add_argument('--expert_usage_root', type=str, default=None)
    parser.add_argument('--target_indices', type=str, default=None)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, num_shards), got shard_rank={args.shard_rank}, num_shards={args.num_shards}"
        )

    if args.device_map is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError("device_map mode must be run as a single process. Use python directly or torchrun --nproc_per_node=1.")
        world_size, rank = 1, 0
        device = None
    else:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"

    # Fallback model root only — launchers always pass --model_path. Override via MODEL_ROOT.
    model_folder = os.environ.get("MODEL_ROOT", "")
    model_name = args.model_name
    step_name = args.step_name
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(model_folder, model_name, f"{step_name}hf")
    # GenEval prompt metadata ships co-located under eval/prompts/.
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval", "prompts", "evaluation_metadata.jsonl"
    )
    output_folder = output_paths.derive_t2i_root(args.image_size, args.cfg_scale, args.timestep_shift, args.cfg_norm)
    output_path = os.path.join(output_folder, model_name, step_name, 'gen_eval')
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
    for i, sample in enumerate(data):
        sample['index'] = i
    target_indices = parse_target_indices(args.target_indices)
    if target_indices is not None:
        data = [sample for sample in data if sample['index'] in target_indices]
        print(f"Filtered to target indices: {sorted(target_indices)} => {len(data)} samples")
    # set_random_seeds(42)
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

    for i in tqdm(range(len(rank_data)), disable=rank != 0):
        cur_sample = rank_data[i]
        index = cur_sample['index']
        prompt = cur_sample['prompt']
        # prompt = "\"Golden elegance and ancient Egyptian beauty.\""
        # Reproducible per-sample noise, keyed off the stable (pre-shuffle) dataset index.
        seed_for_sample(index)
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
            cfg_norm=args.cfg_norm,
        )
        if args.collect_expert_usage:
            images, grid_image, expert_usage = chat_result
        else:
            images, grid_image = chat_result
            expert_usage = None

        index_str = str(index).zfill(4)

        cur_output_folder = os.path.join(output_path, index_str)
        os.makedirs(cur_output_folder, exist_ok=True)
        os.makedirs(os.path.join(cur_output_folder, 'samples'), exist_ok=True)

        write_jsonl([cur_sample], os.path.join(cur_output_folder, 'metadata.jsonl'))
        grid_image.save(os.path.join(cur_output_folder, 'grid.png'))
        for image_i in range(len(images)):
            images[image_i].save(os.path.join(cur_output_folder, 'samples', str(image_i).zfill(4)+'.png'))
        if expert_usage is not None:
            if args.expert_usage_root is not None:
                expert_usage_dir = os.path.join(args.expert_usage_root, index_str)
                os.makedirs(expert_usage_dir, exist_ok=True)
                expert_usage_path = os.path.join(expert_usage_dir, 'expert_usage.json')
            else:
                expert_usage_path = os.path.join(cur_output_folder, 'expert_usage.json')
            write_json(
                {
                    "index": index,
                    "prompt": prompt,
                    "expert_usage": expert_usage,
                },
                expert_usage_path,
            )

    print(f"Rank {rank} has finished processing {len(rank_data)} examples")
    if engine.expert_usage_tracker is not None:
        summary_path = os.path.join(
            output_path,
            f'expert_usage_summary_shard_{global_shard_rank}.json',
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
