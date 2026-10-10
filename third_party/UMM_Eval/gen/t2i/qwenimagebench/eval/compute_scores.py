"""
Compute all models' scores at Level-1, Level-2, Level-3.

Usage:
    python compute_scores.py --input "xxx"
    python compute_scores.py --hf-repo "xxx"
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from score_utils import (
    aggregate_total_score,
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

RESPONSE_PREFIX_TO_DIM = {
    "quality_response_": "Quality",
    "aesthetics_response_": "Aesthetics",
    "alignment_response_": "Alignment",
    "creative_generation_response_": "Creative Generation",
    "real_world_fidelity_response_": "Real-world Fidelity",
}

LEVEL1_DIMS = list(RESPONSE_PREFIX_TO_DIM.values())


def load_data(input_path=None, hf_repo=None):
    if input_path:
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    if hf_repo:
        from huggingface_hub import hf_hub_download
        local_file = hf_hub_download(
            repo_id=hf_repo,
            filename="qwen_image_bench_hf_v0518.jsonl",
            repo_type="dataset",
        )
        return load_data(input_path=local_file)

    print("ERROR: Provide --input or --hf-repo")
    sys.exit(1)


def detect_models(records):
    first = records[0]
    models = []
    prefix = "quality_response_"
    for key in first:
        if key.startswith(prefix):
            model_name = key[len(prefix):]
            models.append(model_name)
    return models


def compute_all_scores(records, models):
    model_row_scores = {m: [] for m in models}

    for row in records:
        row_id = row["ID"]
        for model in models:
            dim_results = {}
            for prefix, dim_name in RESPONSE_PREFIX_TO_DIM.items():
                key = prefix + model
                resp = row.get(key, "")
                if not resp:
                    continue
                score_json = extract_json_from_response(resp)
                if score_json is None:
                    continue
                score_json = fix_score_json(score_json, dim_name)
                dim_results[dim_name] = compute_dimension_score(score_json)

            total = aggregate_total_score(dim_results)
            model_row_scores[model].append({
                "ID": row_id,
                "dims": dim_results,
                "total_score": total,
            })

    return model_row_scores


def _safe_mean(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def aggregate_model_scores(model_row_scores):
    results = {}
    for model, rows in model_row_scores.items():
        l1_accum = defaultdict(list)
        l2_accum = defaultdict(lambda: defaultdict(list))
        total_accum = []

        for row in rows:
            if row["total_score"] is not None:
                total_accum.append(row["total_score"])
            for dim_name, dim_data in row["dims"].items():
                l1_score = dim_data.get("level1_score")
                if l1_score is not None:
                    l1_accum[dim_name].append(l1_score)
                for l2_name, l2_score in dim_data.get("level2_scores", {}).items():
                    if l2_score is not None:
                        l2_accum[dim_name][l2_name].append(l2_score)

        results[model] = {
            "total": _safe_mean(total_accum),
            "level1": {d: _safe_mean(scores) for d, scores in l1_accum.items()},
            "level2": {
                d: {l2: _safe_mean(scores) for l2, scores in l2_dict.items()}
                for d, l2_dict in l2_accum.items()
            },
        }

    return results


def print_results(agg):
    models = sorted(agg.keys(), key=lambda m: agg[m]["total"] or 0, reverse=True)

    print("\n" + "=" * 100)
    print("MODEL SCORES SUMMARY (sorted by Total)")
    print("=" * 100)

    header = f"{'Model':<28}"
    for dim in LEVEL1_DIMS:
        short = dim.replace("Real-world Fidelity", "RWFidelity").replace("Creative Generation", "Creative")
        header += f"{short:>12}"
    header += f"{'Total':>10}"
    print(header)
    print("-" * 100)

    for model in models:
        data = agg[model]
        line = f"{model:<28}"
        for dim in LEVEL1_DIMS:
            val = data["level1"].get(dim)
            line += f"{val:>12.2f}" if val is not None else f"{'N/A':>12}"
        total = data["total"]
        line += f"{total:>10.2f}" if total is not None else f"{'N/A':>10}"
        print(line)

    print("\n" + "=" * 100)
    print("LEVEL-2 DETAIL")
    print("=" * 100)

    for dim in LEVEL1_DIMS:
        all_l2 = set()
        for model in models:
            all_l2.update(agg[model]["level2"].get(dim, {}).keys())
        if not all_l2:
            continue
        all_l2 = sorted(all_l2)

        print(f"\n--- {dim} ---")
        header = f"{'Model':<28}"
        for l2 in all_l2:
            header += f"{l2[:18]:>20}"
        print(header)
        print("-" * (28 + 20 * len(all_l2)))

        for model in models:
            line = f"{model:<28}"
            l2_data = agg[model]["level2"].get(dim, {})
            for l2 in all_l2:
                val = l2_data.get(l2)
                line += f"{val:>20.2f}" if val is not None else f"{'N/A':>20}"
            print(line)

    print()


def save_results(agg, output_dir):
    output_dir = Path(output_dir)
    models = sorted(agg.keys(), key=lambda m: agg[m]["total"] or 0, reverse=True)

    xlsx_path = output_dir / "scores_result.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Sheet 1: Level-1 summary
        summary_data = []
        for model in models:
            data = agg[model]
            row = {"Model": model}
            for dim in LEVEL1_DIMS:
                row[dim] = data["level1"].get(dim)
            row["Total"] = data["total"]
            summary_data.append(row)
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Level-1 Summary", index=False)

        # One sheet per Level-1 dimension showing Level-2 scores
        for dim in LEVEL1_DIMS:
            all_l2 = set()
            for model in models:
                all_l2.update(agg[model]["level2"].get(dim, {}).keys())
            if not all_l2:
                continue
            all_l2 = sorted(all_l2)

            detail_data = []
            for model in models:
                row = {"Model": model}
                l2_data = agg[model]["level2"].get(dim, {})
                for l2 in all_l2:
                    row[l2] = l2_data.get(l2)
                detail_data.append(row)

            df_detail = pd.DataFrame(detail_data)
            sheet_name = dim[:31]  # Excel sheet name max 31 chars
            df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to: {xlsx_path}")

    # Also save detail JSON
    json_path = output_dir / "scores_detail.json"
    serializable = {}
    for model in models:
        serializable[model] = {
            "total": agg[model]["total"],
            "level1": agg[model]["level1"],
            "level2": agg[model]["level2"],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Detail JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute model scores from qwen image bench")
    parser.add_argument("--input", default=None, help="Local JSONL file path")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace dataset repo ID")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    print("Loading data...")
    records = load_data(input_path=args.input, hf_repo=args.hf_repo)
    print(f"Loaded {len(records)} rows")

    models = detect_models(records)
    print(f"Detected {len(models)} models: {', '.join(models)}")

    print("Computing scores...")
    model_row_scores = compute_all_scores(records, models)

    agg = aggregate_model_scores(model_row_scores)

    print_results(agg)

    output_dir = args.output_dir or (str(Path(args.input).parent) if args.input else ".")
    save_results(agg, output_dir)


if __name__ == "__main__":
    main()
