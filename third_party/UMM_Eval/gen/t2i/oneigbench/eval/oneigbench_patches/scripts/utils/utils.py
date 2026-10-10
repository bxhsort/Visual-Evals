import os
import stat
import uuid
import megfile
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description="Run alignment score evaluation.")
    parser.add_argument("--mode", type=str, default="EN", help="Choose language mode.")
    parser.add_argument("--image_dirname", type=str, default="images", help="Directory containing images.")
    parser.add_argument("--model_names", type=str, nargs="+", default=["gpt-4o"], help="List of model names.")
    parser.add_argument("--image_grid", type=str, nargs="+", default=["2,2"], help="List of image grids.")
    parser.add_argument("--class_items", type=str, nargs="+", default=["anime", "human", "object"], help="List of class items.")
    parser.add_argument(
        "--executor_mode",
        type=str,
        choices=["auto", "distributed", "thread", "serial"],
        default="auto",
        help="Execution mode for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Thread worker count for API-based evaluation.",
    )
    parser.add_argument(
        "--api_eval",
        action="store_true",
        help="Use threaded execution instead of distributed GPU workers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to store evaluation CSV outputs.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=None,
        help="Root directory for temporary evaluation caches.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_output_csv_path(args, filename):
    output_dir = ensure_dir(args.output_dir)
    return os.path.join(output_dir, filename)


def get_cache_dir(args, formatted_time, rank):
    cache_root = args.cache_root if args.cache_root is not None else args.output_dir
    return ensure_dir(os.path.join(cache_root, f"tmp_{formatted_time}_rank{rank}"))

def is_black_image(image):
    pixels = image.load()  
    for i in range(image.width):
        for j in range(image.height):
            if pixels[i, j] != (0, 0, 0):
                return False
    return True

def split_mxn_grid(image_path, grid_size, cache_dir):
    with megfile.smart_open(image_path, 'rb') as f:
        grid_image = Image.open(f)

        width, height = grid_image.size

        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        image_list = []

        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (
                    j * individual_width,      
                    i * individual_height,     
                    (j + 1) * individual_width,  
                    (i + 1) * individual_height  
                )

                individual_image = grid_image.crop(box)

                if is_black_image(individual_image):
                    print(f"Detected a black image at position ({i},{j}) in {image_path}")
                else:
                    image_list.append(individual_image)

    image_path_list = []
    sample_cache_dir = os.path.join(cache_dir, uuid.uuid4().hex)
    os.makedirs(sample_cache_dir, exist_ok=True)
    for i, image in enumerate(image_list):
        image_path = os.path.join(sample_cache_dir, f"{i}.jpg")
        image.save(image_path)
        image_path_list.append(image_path)

    return image_path_list

def save2csv(df, csv_path):
    df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


class ParallelContext:
    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.api_eval = args.api_eval or os.environ.get("ONEIGBENCH_API_EVAL", "0") == "1"
        self.executor_mode = self._resolve_executor_mode(args.executor_mode)
        self.thread_workers = self._resolve_thread_workers(args.num_workers)
        self.distributed = self.executor_mode == "distributed" and self.world_size > 1
        if self.distributed and not dist.is_initialized():
            dist.init_process_group(backend=os.environ.get("ONEIGBENCH_DIST_BACKEND", "gloo"))

    def _resolve_executor_mode(self, requested_mode):
        if requested_mode != "auto":
            return requested_mode
        if self.api_eval:
            return "thread"
        if self.world_size > 1:
            return "distributed"
        return "serial"

    def _resolve_thread_workers(self, num_workers):
        if num_workers is not None:
            return max(1, num_workers)
        return max(1, int(os.environ.get("ONEIGBENCH_EVAL_NUM_WORKERS", "8")))

    def shard_items(self, items):
        if self.distributed:
            return items[self.rank::self.world_size]
        return items

    def gather_object(self, obj):
        if not self.distributed:
            return [obj]
        gathered = [None for _ in range(self.world_size)] if self.is_main_process else None
        dist.gather_object(obj, gathered, dst=0)
        return gathered

    def map_items(self, items, fn):
        if self.executor_mode == "thread" and self.thread_workers > 1:
            with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
                return list(executor.map(fn, items))
        return [fn(item) for item in items]

    @property
    def is_main_process(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def finalize(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
