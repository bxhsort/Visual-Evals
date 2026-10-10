import argparse
import json
import os
from pathlib import Path

from eval.gen.edit_compass.common import METRIC_FIELD, iter_task_configs


DG2_ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_score(if_list, vc_list, vq_list, part, task_name):
    if task_name != "Style_Transfer" and not vc_list:
        return None
    scores = if_list + vq_list if task_name == "Style_Transfer" else if_list + vc_list + vq_list
    if not scores:
        return None
    min_score = min(scores)
    if min_score <= 1:
        return min_score
    avg = lambda xs: sum(xs) / len(xs)
    weights = {
        "Style_Transfer": (0.7, 0.0, 0.3),
        "Part1": (0.4, 0.4, 0.2),
        "Part2": (0.4, 0.4, 0.2),
    }
    key = "Style_Transfer" if task_name == "Style_Transfer" else part
    if_w, vc_w, vq_w = weights[key]
    return avg(if_list) ** if_w * (avg(vc_list) ** vc_w if vc_w else 1.0) * avg(vq_list) ** vq_w


def summarize(args, language):
    result_root = Path(args.save_dir) / language
    results = {}
    for cfg in iter_task_configs(args.data_root, result_root, args.parts):
        source_data = read_json(cfg["json_path"])
        per_item = {}
        for item in source_data:
            per_item[item["save_id"]] = {}
            for metric in METRIC_FIELD:
                if item.get(metric) is True:
                    per_item[item["save_id"]][metric] = None

        for metric, metric_field in METRIC_FIELD.items():
            metric_path = cfg["save_dir"] / f"{cfg['task_name']}_{metric}.json"
            if not metric_path.exists():
                continue
            for row in read_json(metric_path):
                score_obj = row.get(metric_field)
                if isinstance(score_obj, dict) and "score" in score_obj:
                    per_item.setdefault(row["save_id"], {})[metric] = score_obj["score"]

        total = 0.0
        count = 0
        for scores in per_item.values():
            if any(value is None for value in scores.values()):
                continue
            if_list, vc_list, vq_list = [], [], []
            for metric, score in scores.items():
                if metric in ("IF", "WA"):
                    if_list.append(score)
                elif metric in ("IC", "URC"):
                    vc_list.append(score)
                elif metric == "VQ":
                    vq_list.append(score)
            overall = get_score(if_list, vc_list, vq_list, cfg["part"], cfg["task_name"])
            if overall is not None:
                total += overall
                count += 1

        results[cfg["task_name"]] = {
            "part": cfg["part"],
            "overall": total / count if count else None,
            "eval_count": count,
            "total_count": len(source_data),
        }
        overall = results[cfg["task_name"]]["overall"]
        text = f"{overall:.4f}" if overall is not None else "N/A"
        print(f"[edit-compass-summary] {language} {cfg['part']}/{cfg['task_name']}: {text} ({count}/{len(source_data)})")

    write_json(str(result_root / "summary.json"), results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(DG2_ROOT / "third_party" / "Edit-Compass-data"))
    parser.add_argument("--save_dir", required=True, help="Root containing language folders, e.g. OUTPUT_ROOT/MODEL_NAME")
    parser.add_argument("--parts", nargs="+", default=["Part1", "Part2"])
    parser.add_argument("--languages", nargs="+", default=["en"])
    args = parser.parse_args()

    for language in args.languages:
        summarize(args, language)


if __name__ == "__main__":
    main()
