"""
Qwen-Image-Bench Judge Model Inference Tool

Evaluate text-to-image generated images using a fine-tuned Qwen3.6-27B judge model.
Uses ms-swift PtEngine for batch inference.

Per-row output preserves all original input fields plus:
  - judge_model_output: combined raw scores JSON across all L1 dimensions
  - <dim>_judge_output: raw judge model text for each L1 dimension

Bench-level scores (L1 / L2 / Total) are aggregated following the
compute_scores.py methodology and saved alongside the per-row output.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from checklists import (
    DIM_TO_CHECKLIST,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    parse_dims_by_level1,
)
from score_utils import (
    aggregate_total_score,
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)


DIM_OUTPUT_MAP = {
    "Quality": "quality_judge_output",
    "Aesthetics": "aesthetics_judge_output",
    "Alignment": "alignment_judge_output",
    "Real-world Fidelity": "real_world_fidelity_judge_output",
    "Creative Generation": "creative_generation_judge_output",
}


def parse_dimension_batch_sizes(value):
    """Parse a JSON mapping of L1 dimension names to positive batch sizes."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dimension batch-size JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Dimension batch sizes must be a JSON object.")
    unknown = set(parsed) - set(DIM_TO_CHECKLIST)
    if unknown:
        raise ValueError(f"Unknown L1 dimensions: {sorted(unknown)}")
    result = {}
    for dimension, batch_size in parsed.items():
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"Batch size for {dimension!r} must be a positive integer.")
        result[dimension] = batch_size
    return result


def load_and_resize_image(path):
    """Load image and resize to 1024x1024 if any dimension > 1024."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
    img.load()
    return img


def load_input_file(file_path):
    """Load CSV or JSON/JSONL input file."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            return pd.DataFrame(json.loads(content))
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
            return pd.DataFrame(records)
    elif ext == ".jsonl":
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .json, or .jsonl")


def load_bench_metadata(hf_bench_repo=None, local_metadata=None):
    """Load bench metadata containing dims_en per ID."""
    if local_metadata:
        return load_input_file(local_metadata)

    if hf_bench_repo:
        from huggingface_hub import hf_hub_download

        local_file = hf_hub_download(
            repo_id=hf_bench_repo,
            filename="qwen_image_bench_hf_v0518.jsonl",
            repo_type="dataset",
        )
        records = []
        with open(local_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    default_path = Path(__file__).parent / "metadata" / "bench_metadata.json"
    if default_path.exists():
        return load_input_file(str(default_path))

    print("ERROR: No metadata source found. Provide --hf-bench-repo, --local-metadata,")
    print(f"       or place metadata at {default_path}")
    sys.exit(1)


def _parse_output_to_scores(output_text, level1_dim):
    """Parse raw judge model output → fixed score_json. Returns None on failure."""
    score_json = extract_json_from_response(output_text)
    if score_json is None:
        return None
    return fix_score_json(score_json, level1_dim)


def _empty_cuda_cache():
    """Best-effort CUDA cleanup after an OOM without making merge-only imports GPU-heavy."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _is_cuda_oom(exc):
    return "out of memory" in str(exc).lower()


def _run_batch_inference(judge, args, input_df, metadata_df, desc):
    """Batch inference over all rows and dimensions."""
    tasks_by_dimension = defaultdict(list)
    skipped_rows = set()
    image_failures = 0

    for row_idx, (_, row) in enumerate(input_df.iterrows()):
        row_id = row["ID"]
        prompt = row["prompt"]
        image_path = row["image_path"]

        meta_row = metadata_df[metadata_df["ID"] == row_id]
        if meta_row.empty:
            continue

        dims_en = meta_row.iloc[0]["dims_en"]
        dims_by_level1 = parse_dims_by_level1(dims_en)

        try:
            img = load_and_resize_image(image_path)
        except Exception as e:
            image_failures += 1
            skipped_rows.add(row_idx)
            print(f"WARNING: Failed to load image for ID={row_id}, path={image_path}: {e}")
            continue

        for level1_dim in dims_by_level1:
            if level1_dim not in DIM_TO_CHECKLIST:
                continue
            checklist = DIM_TO_CHECKLIST[level1_dim]
            user_text = USER_PROMPT_TEMPLATE.format(
                prompt=prompt,
                level1_dim=level1_dim,
                format_checklist=checklist,
            )
            task = {
                "system_prompt": SYSTEM_PROMPT,
                "user_text": user_text,
                "image": img,
                "level1_dim": level1_dim,
                "row_id": row_id,
            }
            tasks_by_dimension[level1_dim].append((task, (row_idx, level1_dim)))

    total_tasks = sum(len(items) for items in tasks_by_dimension.values())
    print(f"Total inference tasks: {total_tasks}")
    all_outputs = []
    ordered_task_meta = []
    dimension_batch_sizes = getattr(args, "dimension_batch_sizes", {})
    min_batch_size = getattr(args, "min_batch_size", 1)
    bucket_by_dimension = getattr(args, "bucket_by_dimension", True)
    oom_fallbacks = 0
    batch_count = 0
    completion_tokens = 0
    prompt_tokens = 0
    peak_cuda_allocated_bytes = 0
    peak_cuda_reserved_bytes = 0
    cuda_total_bytes = 0
    final_batch_sizes = {}
    started_at_epoch = time.time()
    started_at = time.perf_counter()

    if bucket_by_dimension:
        groups = [
            (dimension, tasks_by_dimension[dimension])
            for dimension in DIM_TO_CHECKLIST
            if tasks_by_dimension[dimension]
        ]
    else:
        combined = []
        for dimension in DIM_TO_CHECKLIST:
            combined.extend(tasks_by_dimension[dimension])
        groups = [("all", combined)]

    progress = tqdm(total=total_tasks, desc=desc)
    try:
        for dimension, task_pairs in groups:
            requested_batch_size = dimension_batch_sizes.get(dimension, args.batch_size)
            batch_size = requested_batch_size
            offset = 0
            while offset < len(task_pairs):
                batch_pairs = task_pairs[offset:offset + batch_size]
                batch = [item for item, _ in batch_pairs]
                judge.set_max_batch_size(batch_size)
                try:
                    outputs = judge.generate_batch(batch)
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc) or batch_size <= min_batch_size:
                        raise
                    next_batch_size = max(min_batch_size, batch_size // 2)
                    if next_batch_size == batch_size:
                        raise
                    oom_fallbacks += 1
                    print(
                        f"WARNING: CUDA OOM for {dimension} batch={batch_size}; "
                        f"retrying with batch={next_batch_size}."
                    )
                    batch_size = next_batch_size
                    _empty_cuda_cache()
                    continue

                if len(outputs) != len(batch_pairs):
                    raise RuntimeError(
                        f"Judge returned {len(outputs)} outputs for a batch of {len(batch_pairs)} tasks."
                    )
                all_outputs.extend(outputs)
                ordered_task_meta.extend(meta for _, meta in batch_pairs)
                offset += len(batch_pairs)
                progress.update(len(batch_pairs))
                batch_count += 1

                batch_metrics = getattr(judge, "last_batch_metrics", {})
                completion_tokens += batch_metrics.get("completion_tokens", 0)
                prompt_tokens += batch_metrics.get("prompt_tokens", 0)
                peak_cuda_allocated_bytes = max(
                    peak_cuda_allocated_bytes,
                    batch_metrics.get("peak_cuda_allocated_bytes", 0),
                )
                peak_cuda_reserved_bytes = max(
                    peak_cuda_reserved_bytes,
                    batch_metrics.get("peak_cuda_reserved_bytes", 0),
                )
                cuda_total_bytes = max(cuda_total_bytes, batch_metrics.get("cuda_total_bytes", 0))

            final_batch_sizes[dimension] = batch_size
    finally:
        progress.close()

    elapsed_seconds = time.perf_counter() - started_at
    finished_at_epoch = time.time()
    perf = {
        "tasks": total_tasks,
        "batches": batch_count,
        "started_at_epoch": started_at_epoch,
        "finished_at_epoch": finished_at_epoch,
        "elapsed_seconds": elapsed_seconds,
        "tasks_per_second": total_tasks / elapsed_seconds if elapsed_seconds else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed_seconds if elapsed_seconds else None,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "cuda_total_bytes": cuda_total_bytes,
        "oom_fallbacks": oom_fallbacks,
        "batch_size_by_dimension": final_batch_sizes,
        "bucket_by_dimension": bucket_by_dimension,
    }
    print("Inference performance: " + json.dumps(perf, ensure_ascii=False))

    row_dim_raw_scores = {}
    row_raw_outputs = {}
    parse_failures = 0

    for (row_idx, level1_dim), output_text in zip(ordered_task_meta, all_outputs):
        if row_idx not in row_dim_raw_scores:
            row_dim_raw_scores[row_idx] = {}
            row_raw_outputs[row_idx] = {}
        row_raw_outputs[row_idx][level1_dim] = output_text
        score_json = _parse_output_to_scores(output_text, level1_dim)
        if score_json is None:
            parse_failures += 1
        row_dim_raw_scores[row_idx][level1_dim] = score_json

    results = []
    all_dim_raw_scores = []
    for row_idx, (_, row) in enumerate(input_df.iterrows()):
        if row_idx in skipped_rows:
            results.append(_empty_result(row))
            all_dim_raw_scores.append({})
            continue
        dim_raw_scores = row_dim_raw_scores.get(row_idx, {})
        dim_raw_outputs = row_raw_outputs.get(row_idx, {})
        results.append(_build_row_result(row, dim_raw_scores, dim_raw_outputs))
        all_dim_raw_scores.append(dim_raw_scores)

    return results, parse_failures, all_dim_raw_scores, image_failures, perf


def run_ms_swift_inference(args, input_df, metadata_df):
    """Run inference using ms-swift PtEngine."""
    from backends.ms_swift_backend import MsSwiftJudge

    print(f"Loading model from: {args.model}")
    engine_batch_size = max([args.max_batch_size, *args.dimension_batch_sizes.values()])
    judge = MsSwiftJudge(
        model_path=args.model,
        max_batch_size=engine_batch_size,
        max_new_tokens=args.max_new_tokens,
        require_fla=args.require_fla,
    )
    print("Model loaded successfully.")
    return _run_batch_inference(judge, args, input_df, metadata_df, desc="Batch inference")


def _empty_result(row):
    """Build an empty result row for skipped entries."""
    result = dict(row)
    result["judge_model_output"] = None
    for col in DIM_OUTPUT_MAP.values():
        result[col] = None
    return result


def _build_row_result(row, dim_raw_scores, dim_raw_outputs):
    """
    Build per-row JSONL record.

    Schema (in order):
      - all original row fields (transparent pass-through)
      - judge_model_output: JSON-serialized {L1_dim: fixed_score_json} for all parsed dims
      - <dim>_judge_output: raw judge text for each L1 dim
    """
    result = dict(row)

    raw_output = {
        dim_name: score_json
        for dim_name, score_json in dim_raw_scores.items()
        if score_json is not None
    }
    result["judge_model_output"] = (
        json.dumps(raw_output, ensure_ascii=False) if raw_output else None
    )

    for dim_name, col_name in DIM_OUTPUT_MAP.items():
        result[col_name] = dim_raw_outputs.get(dim_name)

    return result


def save_output(results, input_path, num_shards=1, shard_rank=0):
    """Save per-row results to file in same directory as input.

    When num_shards>1, each shard writes a distinct ``*_judged.shard{r}-of-{n}`` file;
    merge_shard_scores.py recombines them and computes the bench afterwards.
    """
    input_p = Path(input_path)
    ext = input_p.suffix.lower()
    if num_shards > 1:
        output_name = f"{input_p.stem}_judged.shard{shard_rank}-of-{num_shards}{ext}"
    else:
        output_name = f"{input_p.stem}_judged{ext}"
    output_path = input_p.parent / output_name

    df = pd.DataFrame(results)

    if ext == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    return str(output_path)


def save_perf_metrics(perf, input_path, num_shards=1, shard_rank=0):
    input_p = Path(input_path)
    if num_shards > 1:
        output_name = f"{input_p.stem}_perf.shard{shard_rank}-of-{num_shards}.json"
    else:
        output_name = f"{input_p.stem}_perf.json"
    output_path = input_p.parent / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)
    return str(output_path)


def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else None


def compute_bench_scores(all_dim_raw_scores):
    """
    Bench-level aggregation following compute_scores.py methodology:
      per-row L3→L2→L1→Total nested averaging,
      then arithmetic mean across rows (None values skipped).
    """
    l1_accum = defaultdict(list)
    l2_accum = defaultdict(lambda: defaultdict(list))
    total_accum = []

    for row_scores in all_dim_raw_scores:
        dim_results = {}
        for l1_dim, score_json in row_scores.items():
            if score_json is None:
                continue
            dim_results[l1_dim] = compute_dimension_score(score_json)

        row_total = aggregate_total_score(dim_results)
        if row_total is not None:
            total_accum.append(row_total)

        for l1_dim, dim_data in dim_results.items():
            if dim_data["level1_score"] is not None:
                l1_accum[l1_dim].append(dim_data["level1_score"])
            for l2_name, l2_score in dim_data["level2_scores"].items():
                if l2_score is not None:
                    l2_accum[l1_dim][l2_name].append(l2_score)

    return {
        "level1": {d: _safe_mean(v) for d, v in l1_accum.items()},
        "level2": {
            d: {l2: _safe_mean(v) for l2, v in l2d.items()}
            for d, l2d in l2_accum.items()
        },
        "total": _safe_mean(total_accum),
    }


def save_bench_scores(bench, input_path):
    """Save bench-level scores as JSON + Excel beside the input file."""
    input_p = Path(input_path)
    base = input_p.parent / f"{input_p.stem}_bench_scores"
    json_path = f"{base}.json"
    xlsx_path = f"{base}.xlsx"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        l1_rows = [{"Dimension": d, "Score": s} for d, s in bench["level1"].items()]
        l1_rows.append({"Dimension": "Total", "Score": bench["total"]})
        pd.DataFrame(l1_rows).to_excel(writer, sheet_name="Level-1 Summary", index=False)

        for dim, l2_dict in bench["level2"].items():
            if not l2_dict:
                continue
            df = pd.DataFrame(
                [{"Sub-dimension": l2, "Score": s} for l2, s in l2_dict.items()]
            )
            df.to_excel(writer, sheet_name=dim[:31], index=False)

    return json_path, xlsx_path


def print_bench_scores(bench):
    """Pretty-print bench-level scores to terminal."""
    print("\n" + "=" * 70)
    print("BENCH-LEVEL SCORES")
    print("=" * 70)
    for dim, score in bench["level1"].items():
        s = f"{score:.2f}" if score is not None else "N/A"
        print(f"  L1 {dim:30s}: {s}")
    total = bench["total"]
    total_str = f"{total:.2f}" if total is not None else "N/A"
    print(f"  {'TOTAL':33s}: {total_str}")
    print("-" * 70)
    for dim, l2_dict in bench["level2"].items():
        if not l2_dict:
            continue
        print(f"  [{dim}]")
        for l2, s in l2_dict.items():
            v = f"{s:.2f}" if s is not None else "N/A"
            print(f"    L2 {l2:28s}: {v}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Bench Judge Model Inference Tool"
    )
    parser.add_argument("--input", required=True, help="Input CSV/JSON/JSONL with ID, prompt, image_path")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local model path")
    parser.add_argument("--hf-bench-repo", default=None, help="HF dataset repo for bench metadata")
    parser.add_argument("--local-metadata", default=None, help="Local metadata file path (skip HF download)")
    parser.add_argument("--max-batch-size", type=int, default=12,
                        help="ms-swift PtEngine max_batch_size (default: 12)")
    parser.add_argument(
        "--dimension-batch-sizes",
        default="",
        help='Optional JSON overrides, for example \'{"Quality": 16, "Alignment": 12}\'.',
    )
    parser.add_argument(
        "--no-bucket-by-dimension",
        action="store_false",
        dest="bucket_by_dimension",
        help="Disable L1 dimension bucketing for baseline comparisons.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=8,
        help="Smallest automatic retry batch after CUDA OOM (default: 8).",
    )
    parser.add_argument(
        "--require-fla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast unless Qwen3.5 is using flash-linear-attention kernels.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Data-parallel sharding: this process handles rows[shard_rank::num_shards].")
    parser.add_argument("--shard-rank", type=int, default=0,
                        help="This shard's index in [0, num_shards).")

    args = parser.parse_args()
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be positive")
    if args.min_batch_size <= 0 or args.min_batch_size > args.max_batch_size:
        parser.error("--min-batch-size must be in [1, max-batch-size]")
    try:
        args.dimension_batch_sizes = parse_dimension_batch_sizes(args.dimension_batch_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    args.batch_size = args.max_batch_size

    # Load input
    print(f"Loading input: {args.input}")
    input_df = load_input_file(args.input)
    required_cols = {"ID", "prompt", "image_path"}
    missing = required_cols - set(input_df.columns)
    if missing:
        print(f"ERROR: Input file missing required columns: {missing}")
        sys.exit(1)
    print(f"Input: {len(input_df)} rows")

    # Data-parallel sharding: keep only this shard's rows (stable, contiguous stride).
    if args.num_shards > 1:
        if not 0 <= args.shard_rank < args.num_shards:
            print(f"ERROR: shard_rank must be in [0, num_shards), got {args.shard_rank}/{args.num_shards}")
            sys.exit(1)
        input_df = input_df.iloc[args.shard_rank::args.num_shards].reset_index(drop=True)
        print(f"Shard {args.shard_rank}/{args.num_shards}: {len(input_df)} rows this shard")

    # Load metadata
    print("Loading bench metadata...")
    metadata_df = load_bench_metadata(
        hf_bench_repo=args.hf_bench_repo,
        local_metadata=args.local_metadata,
    )
    print(f"Metadata: {len(metadata_df)} rows")

    # Run inference
    results, parse_failures, all_dim_raw_scores, image_failures, perf = run_ms_swift_inference(
        args, input_df, metadata_df
    )

    # Save per-row JSONL (per-shard file when sharded)
    saved_path = save_output(results, args.input, args.num_shards, args.shard_rank)
    perf_path = save_perf_metrics(perf, args.input, args.num_shards, args.shard_rank)
    print(f"\nPer-row results saved to: {saved_path}")
    print(f"Performance metrics saved to: {perf_path}")
    if image_failures:
        print(f"Skipped (broken images): {image_failures}")
    print(f"Parse failures: {parse_failures}")

    # Bench scores only for unsharded runs (byte-for-byte original behavior). For sharded
    # runs, merge_shard_scores.py recombines the per-shard *_judged files and computes the
    # identical bench afterwards.
    if args.num_shards == 1:
        bench = compute_bench_scores(all_dim_raw_scores)
        json_path, xlsx_path = save_bench_scores(bench, args.input)
        print(f"Bench scores saved to: {json_path}")
        print(f"Bench scores saved to: {xlsx_path}")
        print_bench_scores(bench)
    else:
        print(f"Shard {args.shard_rank}/{args.num_shards} done; "
              f"run merge_shard_scores.py to aggregate bench scores.")


if __name__ == "__main__":
    main()
