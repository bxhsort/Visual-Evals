#!/usr/bin/env python3
"""Merge data-parallel judge shards -> one bench_scores (+ combined _judged).

Reads ``<stem>_judged.shard{r}-of-{n}{ext}`` for r in range(n), reconstructs each row's
per-dimension scores from its ``judge_model_output`` field, and computes the SAME bench-level
scores judge.py would produce unsharded (reusing judge.compute_bench_scores / save_bench_scores).
The ms-swift backend is imported lazily inside judge.run_ms_swift_inference, so importing judge
here is cheap and GPU-free.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import judge


def _read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _merge_perf_metrics(input_path, num_shards):
    shard_metrics = []
    for rank in range(num_shards):
        path = input_path.parent / (
            f"{input_path.stem}_perf.shard{rank}-of-{num_shards}.json"
        )
        if not path.exists():
            raise SystemExit(f"ERROR: missing judge performance shard {path}")
        with open(path, encoding="utf-8") as f:
            shard_metrics.append(json.load(f))

    if all("started_at_epoch" in item and "finished_at_epoch" in item for item in shard_metrics):
        started_at_epoch = min(item["started_at_epoch"] for item in shard_metrics)
        finished_at_epoch = max(item["finished_at_epoch"] for item in shard_metrics)
        elapsed_seconds = finished_at_epoch - started_at_epoch
    else:
        started_at_epoch = None
        finished_at_epoch = None
        elapsed_seconds = max(item["elapsed_seconds"] for item in shard_metrics)
    tasks = sum(item["tasks"] for item in shard_metrics)
    completion_tokens = sum(item["completion_tokens"] for item in shard_metrics)
    merged = {
        "num_shards": num_shards,
        "tasks": tasks,
        "batches": sum(item["batches"] for item in shard_metrics),
        "started_at_epoch": started_at_epoch,
        "finished_at_epoch": finished_at_epoch,
        "elapsed_seconds": elapsed_seconds,
        "tasks_per_second": tasks / elapsed_seconds if elapsed_seconds else None,
        "prompt_tokens": sum(item["prompt_tokens"] for item in shard_metrics),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": (
            completion_tokens / elapsed_seconds if elapsed_seconds else None
        ),
        "peak_cuda_allocated_bytes": max(
            item["peak_cuda_allocated_bytes"] for item in shard_metrics
        ),
        "peak_cuda_reserved_bytes": max(
            item["peak_cuda_reserved_bytes"] for item in shard_metrics
        ),
        "cuda_total_bytes": max(item["cuda_total_bytes"] for item in shard_metrics),
        "oom_fallbacks": sum(item["oom_fallbacks"] for item in shard_metrics),
        "shards": shard_metrics,
    }
    output = input_path.parent / f"{input_path.stem}_perf.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return output


def _restore_input_order(input_rows, rows):
    if len(rows) != len(input_rows):
        raise ValueError(
            f"merged {len(rows)} rows but input contains {len(input_rows)} rows"
        )
    input_ids = [str(row["ID"]) for row in input_rows]
    row_ids = [str(row["ID"]) for row in rows]
    if Counter(input_ids) != Counter(row_ids):
        raise ValueError("merged shard IDs do not match input IDs")
    input_order = {row_id: index for index, row_id in enumerate(input_ids)}
    try:
        return sorted(rows, key=lambda row: input_order[str(row["ID"])])
    except KeyError as exc:
        raise ValueError(f"shard contains an unknown ID: {exc.args[0]}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="original judge input jsonl (for stem + location)")
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    ext = inp.suffix.lower()

    rows = []
    for r in range(args.num_shards):
        shard = inp.parent / f"{inp.stem}_judged.shard{r}-of-{args.num_shards}{ext}"
        if not shard.exists():
            raise SystemExit(f"ERROR: missing judge shard {shard}")
        rows.extend(_read_jsonl(shard))

    input_rows = _read_jsonl(inp)
    try:
        rows = _restore_input_order(input_rows, rows)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    # judge_model_output is a JSON string of {l1_dim: fixed_score_json} (or None for skipped).
    all_dim_raw_scores = []
    for row in rows:
        jmo = row.get("judge_model_output")
        all_dim_raw_scores.append(json.loads(jmo) if jmo else {})

    bench = judge.compute_bench_scores(all_dim_raw_scores)
    json_path, xlsx_path = judge.save_bench_scores(bench, args.input)

    # Combined per-row file for convenience (same name a single-process run would write).
    combined = inp.parent / f"{inp.stem}_judged{ext}"
    with open(combined, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    perf_path = _merge_perf_metrics(inp, args.num_shards)
    print(f"Merged {len(rows)} rows from {args.num_shards} shards -> {json_path}")
    print(f"Merged performance metrics -> {perf_path}")
    judge.print_bench_scores(bench)


if __name__ == "__main__":
    main()
