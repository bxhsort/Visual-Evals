"""Read-only sample coverage audit; never changes benchmark scores or denominators.

Adapters enumerate the expected units from metadata and inspect raw judge evidence.
An existing image with no verdict is unscored, not automatically a judge failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import tempfile


STATUS_LABELS = {
    "scored": "已评分",
    "missing_image": "图片缺失",
    "corrupt_image": "图片损坏",
    "judge_failed": "judge 失败",
    "unscored": "未评分，原因不明",
}


def _judge_error_type(row):
    kind = row.get("error_type")
    if kind in ("api_error", "parse_error", "image_error"):
        return kind
    message = str(row.get("judge_error") or "").lower()
    if any(token in message for token in ("apierror", "httperror", "ratelimit", "429", "timeout", "timed out", "connection")):
        return "api_error"
    if any(token in message for token in ("parse", "parsing", "jsondecode", "outputformat", "no finite score", "missing score", "malformed", "invalid score", "解析")):
        return "parse_error"
    return "judge_error"


def _image_state(path):
    path = Path(path)
    if not path.is_file():
        return "missing_image", "生成图片不存在", "existence"
    try:
        if path.stat().st_size == 0:
            return "corrupt_image", "图片文件为空", "size"
        try:
            from PIL import Image
        except ImportError:
            return None, None, "existence_only"
        with Image.open(path) as image:
            image.verify()
        return None, None, "decoded"
    except (OSError, ValueError, SyntaxError) as exc:
        return "corrupt_image", str(exc), "decoded"


def audit_rows(bench, root, rows, warnings=(), *, language="en"):
    """Classify each expected unit; successful low/zero scores remain valid verdicts."""
    result_rows = []
    seen = set()
    notices = list(warnings)
    verification = set()
    for raw in rows:
        row = dict(raw)
        row["sample_id"] = str(row["sample_id"])
        row["category"] = str(row.get("category") or "")
        identity = (row["sample_id"], row["category"])
        if identity in seen:
            notices.append(f"重复预期样本：{identity[0]} / {identity[1]}；覆盖率无法确认为完整。")
            continue
        seen.add(identity)
        image_path = Path(row.get("image_path") or "")
        if not image_path.is_absolute():
            image_path = Path(root) / image_path
        row["image_path"] = str(image_path.resolve())
        status, reason, validation = _image_state(image_path)
        verification.add(validation)
        if status is None:
            if row.get("judge_error"):
                status, reason = "judge_failed", str(row["judge_error"])
                row["error_type"] = _judge_error_type(row)
                label = {"api_error": "judge 请求失败", "parse_error": "评分响应解析失败",
                         "image_error": "judge 读取图片失败"}.get(row["error_type"], "judge 失败")
                reason = f"{label}：{reason}"
            elif row.get("judged") is True:
                status, reason = "scored", "原始评分记录有效（包含模型低分或零分）"
            else:
                status, reason = "unscored", "图片存在，但没有可核验的有效评分或明确失败记录"
        # Preserve judge evidence even when a missing image is the primary cause.
        row["status"], row["reason"] = status, reason
        result_rows.append(row)
    if "existence_only" in verification:
        notices.append("当前环境没有 Pillow，非空图片只核验存在性，未核验解码完整性。")
    expected = len(result_rows) if result_rows else None
    if expected is None:
        notices.append("未取得可核验的预期样本清单，应评数量及完整性未知。")
    counts = Counter(row["status"] for row in result_rows)
    count_record = {"expected": expected, "generated": sum(
        row["status"] not in ("missing_image", "corrupt_image") for row in result_rows
    ), "valid_scored": counts["scored"]}
    count_record.update({status: counts[status] for status in STATUS_LABELS if status != "scored"})
    if expected is None:
        # Without a population, no missing/failure count can honestly be zero.
        count_record = {key: None for key in count_record}
    notices = list(dict.fromkeys(notices))
    complete = expected is not None and counts["scored"] == expected and not notices
    return {
        "schema_version": 1, "benchmark": bench, "language": language,
        "results_root": str(Path(root).resolve()), "counts": count_record,
        "coverage_ratio": counts["scored"] / expected if expected else None,
        "coverage_complete": complete,
        "score_policy": "保留原 benchmark 算分方式，不修改分数、权重或分母。",
        "warnings": notices, "samples": result_rows,
    }


def audit(bench, results_root, **options):
    if bench in {"imgedit", "geditbench", "rise", "editcompass"}:
        from coverage_editing import build_samples
    else:
        from coverage_t2i import build_samples
    try:
        rows, warnings = build_samples(bench, Path(results_root), **options)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        rows, warnings = [], [f"无法读取覆盖率证据：{type(exc).__name__}: {exc}"]
    return audit_rows(bench, results_root, rows, warnings, language=options.get("language", "en"))


def _cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _count(value):
    return "未知" if value is None else str(value)


def render_markdown(record):
    counts = record["counts"]
    lines = [f"# {record['benchmark']} 样本覆盖报告（{record['language']}）", "",
             record["score_policy"], ""]
    if record["coverage_complete"]:
        lines.append("已核验预期样本全部取得有效评分。")
    else:
        lines.append("**提示：评测尚未完整完成或完整性未核验；现有分数仍按原评分协议计算。**")
    lines += ["", "| 应评 | 有效评分 | 缺图 | 图片损坏 | judge 失败 | 未评分/原因不明 |",
              "|---:|---:|---:|---:|---:|---:|",
              "| " + " | ".join(_count(counts[k]) for k in (
                  "expected", "valid_scored", "missing_image", "corrupt_image", "judge_failed", "unscored"
              )) + " |", ""]
    for warning in record["warnings"]:
        lines.append(f"- {_cell(warning)}")
    anomalies = [row for row in record["samples"] if row["status"] != "scored"]
    lines += ["", "## 异常样本", ""]
    if not anomalies:
        lines.append("未发现异常样本。" if counts["expected"] is not None else "预期样本清单缺失，无法列出遗漏 ID。")
    else:
        lines += ["| 样本 ID | 类别 | 状态 | 图片路径 | 原因 |", "|---|---|---|---|---|"]
        for row in anomalies:
            lines.append("| " + " | ".join(_cell(v) for v in (
                row["sample_id"], row["category"], STATUS_LABELS[row["status"]], row["image_path"], row["reason"]
            )) + " |")
    return "\n".join(lines) + "\n"


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temp = handle.name
        handle.write(text)
    try:
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_audit(record, prefix):
    prefix = str(prefix)
    _atomic_text(prefix + ".json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(prefix + ".md", render_markdown(record))
    import io
    buffer = io.StringIO()
    fields = ["sample_id", "category", "status", "image_path", "reason", "judge_error", "error_type"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(row for row in record["samples"] if row["status"] != "scored")
    _atomic_text(prefix + ".csv", buffer.getvalue())
    return prefix


def main():
    parser = argparse.ArgumentParser(description="Report missing images and judge failures without changing scores.")
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results-root", required=True)
    for name in ("metadata-file", "image-dir", "judge-root", "out-prefix", "target-indices",
                 "target-prompt-ids", "target-keys", "cvtg-subsets", "cvtg-areas", "tiifbench-specific-file"):
        parser.add_argument("--" + name)
    parser.add_argument("--language", default="en")
    parser.add_argument("--split", default="testmini")
    parser.add_argument("--model-name", default="bagel")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--parts", nargs="+", default=["Part1", "Part2"])
    args = vars(parser.parse_args())
    bench, root, prefix = args.pop("bench"), args.pop("results_root"), args.pop("out_prefix")
    record = audit(bench, root, **args)
    prefix = prefix or str(Path(root) / f"coverage_{bench}_{args['language']}")
    write_audit(record, prefix)
    counts = record["counts"]
    print(f"[coverage] {bench}: 有效评分 {_count(counts['valid_scored'])}/{_count(counts['expected'])}；"
          f"缺图 {_count(counts['missing_image'])}，损坏 {_count(counts['corrupt_image'])}，judge 失败 {_count(counts['judge_failed'])}，"
          f"未评分/原因不明 {_count(counts['unscored'])}。异常清单：{prefix}.csv")
    if not record["coverage_complete"]:
        print("[coverage] WARNING: 现有分数保留；评测尚未完整完成或完整性未核验。")


if __name__ == "__main__":
    main()
