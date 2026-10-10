#!/usr/bin/env python3
"""Sweep Q-Judger batch sizes on a small, dimension-complete input subset."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import judge
from checklists import DIM_TO_CHECKLIST, parse_dims_by_level1


def parse_batch_sizes(value):
    try:
        batch_sizes = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Batch sizes must be comma-separated integers.") from exc
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("Batch sizes must be positive.")
    return batch_sizes


def select_dimension_complete_rows(input_df, metadata_df, max_rows):
    if max_rows <= 0:
        raise ValueError("--rows must be positive")
    metadata_by_id = metadata_df.set_index("ID")
    selected_indices = []
    covered = set()

    for index, row in input_df.iterrows():
        if row["ID"] not in metadata_by_id.index:
            continue
        dimensions = set(
            parse_dims_by_level1(metadata_by_id.loc[row["ID"], "dims_en"])
        )
        if dimensions - covered:
            selected_indices.append(index)
            covered.update(dimensions)
        if covered == set(DIM_TO_CHECKLIST):
            break

    for index in input_df.index:
        if len(selected_indices) >= max_rows:
            break
        if index not in selected_indices:
            selected_indices.append(index)

    subset = input_df.loc[selected_indices[:max_rows]].reset_index(drop=True)
    subset_dimensions = set()
    for row_id in subset["ID"]:
        if row_id in metadata_by_id.index:
            subset_dimensions.update(
                parse_dims_by_level1(metadata_by_id.loc[row_id, "dims_en"])
            )
    missing = set(DIM_TO_CHECKLIST) - subset_dimensions
    if missing:
        raise ValueError(f"Input cannot cover all QIB L1 dimensions; missing {sorted(missing)}")
    return subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="QIB judge_input.jsonl")
    parser.add_argument("--model", required=True, help="Local Q-Judger model path")
    parser.add_argument(
        "--local-metadata",
        default=str(Path(__file__).parent / "metadata" / "bench_metadata.json"),
    )
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("8,12,16,20"))
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--memory-limit", type=float, default=0.9)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--require-fla",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not 0 < args.memory_limit <= 1:
        parser.error("--memory-limit must be in (0, 1]")

    from backends.ms_swift_backend import MsSwiftJudge

    input_df = judge.load_input_file(args.input)
    metadata_df = judge.load_bench_metadata(local_metadata=args.local_metadata)
    input_df = select_dimension_complete_rows(input_df, metadata_df, args.rows)
    print(
        f"Benchmark subset: {len(input_df)} rows; "
        f"batch candidates: {args.batch_sizes}"
    )

    engine = MsSwiftJudge(
        model_path=args.model,
        max_batch_size=max(args.batch_sizes),
        max_new_tokens=args.max_new_tokens,
        require_fla=args.require_fla,
    )
    summaries = []
    baseline_scores = None
    for batch_size in args.batch_sizes:
        print(f"\n=== Q-Judger batch size {batch_size} ===")
        run_args = SimpleNamespace(
            batch_size=batch_size,
            dimension_batch_sizes={},
            bucket_by_dimension=True,
            min_batch_size=batch_size,
        )
        engine.set_max_batch_size(batch_size)
        try:
            _, parse_failures, raw_scores, image_failures, perf = judge._run_batch_inference(
                engine,
                run_args,
                input_df,
                metadata_df,
                desc=f"batch={batch_size}",
            )
            scores = judge.compute_bench_scores(raw_scores)
            if baseline_scores is None:
                baseline_scores = scores
            perf.update(
                {
                    "batch_size": batch_size,
                    "parse_failures": parse_failures,
                    "image_failures": image_failures,
                    "scores_match_baseline": scores == baseline_scores,
                    "scores": scores,
                }
            )
            total_bytes = perf.get("cuda_total_bytes", 0)
            reserved_bytes = perf.get("peak_cuda_reserved_bytes", 0)
            perf["peak_reserved_fraction"] = (
                reserved_bytes / total_bytes if total_bytes else None
            )
            summaries.append(perf)
        except RuntimeError as exc:
            if not judge._is_cuda_oom(exc):
                raise
            judge._empty_cuda_cache()
            summaries.append({"batch_size": batch_size, "error": "CUDA out of memory"})

    eligible = [
        summary
        for summary in summaries
        if "error" not in summary
        and summary["scores_match_baseline"]
        and summary["parse_failures"] == 0
        and summary["image_failures"] == 0
        and (
            summary["peak_reserved_fraction"] is None
            or summary["peak_reserved_fraction"] <= args.memory_limit
        )
    ]
    recommended = (
        max(eligible, key=lambda item: item["tasks_per_second"])["batch_size"]
        if eligible
        else None
    )
    report = {
        "input": args.input,
        "rows": len(input_df),
        "memory_limit": args.memory_limit,
        "recommended_batch_size": recommended,
        "results": summaries,
    }
    output = Path(args.output) if args.output else Path(args.input).with_name(
        f"{Path(args.input).stem}_batch_sweep.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRecommended batch size: {recommended}")
    print(f"Benchmark report: {output}")


if __name__ == "__main__":
    main()
