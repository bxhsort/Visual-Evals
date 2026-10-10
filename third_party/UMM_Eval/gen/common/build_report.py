"""Render one aggregated Markdown eval report from collected scores + qualitative review.

Pure rendering: inputs are the JSON records produced by ``collect_results.collect`` and
``qualitative_review.review``. No path/network logic here — the orchestrator passes data in.
"""
from __future__ import annotations

import argparse
import html
import json
import os

import image_thumb


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _bench_label(record):
    name = record["benchmark"]
    if name == "geneval" and record.get("prompt_variant"):
        name += f" ({record['prompt_variant']} prompt)"
    if record.get("thinking"):
        name += " [thinking]"
    return name


def _coverage_status(record):
    if record.get("headline") is None:
        return "无分数"
    coverage = record.get("coverage")
    if coverage is None:
        return "分数已生成，覆盖率未核验"
    if coverage.get("coverage_complete"):
        return "已完成（覆盖率已核验）"
    return "评测未完整完成或完整性未核验"


def _coverage_counts(record):
    coverage = record.get("coverage") or {}
    counts = coverage.get("counts") or {}
    if counts.get("expected") is None:
        return "未知"
    return f"{counts.get('valid_scored', 0)} / {counts['expected']}"


def _md_cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _coverage_section(collected):
    lines = ["## 样本覆盖与异常提示", "",
             "分数保留原 benchmark 算法；以下提示不修改分数、权重或分母。", ""]
    from coverage_audit import STATUS_LABELS
    for record in collected:
        lines += [f"### {_md_cell(_bench_label(record))}", "", _coverage_status(record), ""]
        if record.get("experiment_name"):
            lines += [f"实验：{_md_cell(record['experiment_name'])}；setting：{_md_cell(record.get('setting_name'))}", ""]
        coverage = record.get("coverage")
        if coverage is None:
            lines.append(record.get("coverage_warning") or "没有覆盖率证据，无法确认缺失数量和样本 ID。")
            lines.append("")
            continue
        counts = coverage.get("counts") or {}
        lines.append(f"应评 {_fmt(counts.get('expected'))}，有效评分 {_fmt(counts.get('valid_scored'))}；"
                     f"缺图 {_fmt(counts.get('missing_image'))}，损坏 {_fmt(counts.get('corrupt_image'))}，"
                     f"judge 失败 {_fmt(counts.get('judge_failed'))}，未评分/原因不明 {_fmt(counts.get('unscored'))}。")
        lines += ["", *[f"- {_md_cell(w)}" for w in coverage.get("warnings", [])], ""]
        anomalies = [row for row in coverage.get("samples", []) if row.get("status") != "scored"]
        if anomalies:
            lines += ["| 样本 ID | 类别 | 状态 | 图片路径 | 原因 |", "|---|---|---|---|---|"]
            for row in anomalies:
                lines.append("| " + " | ".join(_md_cell(v) for v in (
                    row.get("sample_id"), row.get("category"), STATUS_LABELS.get(row.get("status"), "原因不明"),
                    row.get("image_path"), row.get("reason")
                )) + " |")
            lines.append("")
    return "\n".join(lines)


def _scores_table(collected):
    lines = ["## 1. Scores", "", "| Benchmark | Headline | Metric | 有效评分 / 应评 | Status |",
             "|---|---|---|---|---|"]
    for r in collected:
        status = _coverage_status(r)
        lines.append(
            f"| {_md_cell(_bench_label(r))} | {_fmt(r.get('headline'))} | "
            f"{r.get('headline_metric') or '—'} | {_coverage_counts(r)} | {status} |"
        )
    lines.append("")
    # Per-benchmark sub-metrics
    for r in collected:
        metrics = r.get("metrics") or {}
        if not metrics:
            continue
        lines.append(f"<details><summary>{_md_cell(_bench_label(r))} sub-metrics</summary>")
        lines.append("")
        for k, v in metrics.items():
            lines.append(f"- `{k}`: {_fmt(v)}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def _qualitative_section(qualitative):
    lines = ["## 2. Qualitative review", ""]
    if not qualitative:
        lines.append("_No qualitative review available._")
        lines.append("")
        return "\n".join(lines)
    for q in qualitative:
        lines.append(f"### {q['benchmark']}")
        lines.append("")
        if q.get("error"):
            lines.append(f"_Skipped: {q['error']}_")
            lines.append("")
            continue
        lines.append(f"_Model: {q.get('model_used','?')}, "
                     f"{q.get('n_sampled','?')} images sampled._")
        lines.append("")
        for dim, text in (q.get("findings") or {}).items():
            lines.append(f"- **{dim}**: {text}")
        lines.append("")
        examples = q.get("failure_examples") or []
        if examples:
            lines.append("Failure samples:")
            lines.append("")
            for ex in examples:
                cat = ex.get("category", "?")
                note = ex.get("note", "")
                lines.append(f"- `{ex.get('path','?')}` — {cat}: {note}")
            lines.append("")
    return "\n".join(lines)


def _run_config_section(run_config):
    lines = ["## 3. Run config", ""]
    for k, v in run_config.items():
        if k == "output_roots":
            continue
        lines.append(f"- **{k}**: {_fmt(v)}")
    roots = run_config.get("output_roots") or {}
    if roots:
        lines.append("- **output_roots**:")
        for b, p in roots.items():
            lines.append(f"    - {b}: `{p}`")
    lines.append("")
    return "\n".join(lines)


def render(collected, qualitative, run_config):
    model = run_config.get("model_path", "?")
    parts = [f"# Eval Report", "",
             f"Model: `{model}`", ""]
    parts.append(_scores_table(collected))
    parts.append(_coverage_section(collected))
    parts.append(_qualitative_section(qualitative))
    parts.append(_run_config_section(run_config))
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML rendering — same content as the Markdown, plus inline (base64) thumbnails
# of the failure-sample images so they are directly viewable in a browser.
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1a1a1a;line-height:1.5}
h1{border-bottom:2px solid #333;padding-bottom:.3rem}
h2{margin-top:2.5rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}
th{background:#f4f4f4}
.bad{color:#b00020;font-weight:bold}
.ok{color:#0a7d2c}
code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
.finding{margin:.3rem 0}
.fails{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0}
.fail{width:260px;border:1px solid #ddd;border-radius:6px;padding:.5rem;background:#fafafa;font-size:.85em}
.fail img{width:100%;height:auto;border-radius:4px;background:#eee}
.cat{font-weight:bold;color:#b00020}
details{margin:.5rem 0}
summary{cursor:pointer;color:#555}
"""


def _h(x):
    return html.escape(str(x))


# English finding keys (kept for parsing) -> Chinese labels for display.
_DIM_LABELS = {
    # t2i
    "defects": "明显缺陷/崩坏",
    "prompt_alignment": "prompt 对齐",
    "aesthetics": "美学/画质",
    "text_rendering": "文字渲染",
    "artifacts": "其他 artifacts",
    # it2i (image editing)
    "instruction_following": "指令遵循",
    "preservation": "无关区域/身份保留",
    "edit_quality": "编辑质量",
}


def _scores_table_html(collected):
    rows = ["<h2>一、点数</h2><table>"
            "<tr><th>榜单</th><th>主指标</th><th>指标名</th><th>生成图数量</th><th>有效评分 / 应评</th><th>状态</th></tr>"]
    for r in collected:
        ok = (r.get("coverage") or {}).get("coverage_complete") and r.get("headline") is not None
        status = f'<span class="{"ok" if ok else "bad"}">{_h(_coverage_status(r))}</span>'
        n_images = r.get("n_images")
        n_disp = _h(n_images) if n_images is not None else "—"
        rows.append(
            f"<tr><td>{_h(_bench_label(r))}</td><td>{_h(_fmt(r.get('headline')))}</td>"
            f"<td>{_h(r.get('headline_metric') or '—')}</td><td>{n_disp}</td>"
            f"<td>{_h(_coverage_counts(r))}</td><td>{status}</td></tr>"
        )
    rows.append("</table>")
    rows.append('<p style="color:#555;font-size:.9em">「生成图数量」= 从输出目录实际统计的图片数，'
                '用于核对评测是否完整覆盖。</p>')
    for r in collected:
        metrics = r.get("metrics") or {}
        if not metrics:
            continue
        rows.append(f"<details><summary>{_h(_bench_label(r))} 子指标</summary><ul>")
        for k, v in metrics.items():
            rows.append(f"<li><code>{_h(k)}</code>: {_h(_fmt(v))}</li>")
        rows.append("</ul></details>")
    return "\n".join(rows)


def _coverage_html(collected):
    from coverage_audit import STATUS_LABELS
    out = ["<h2>样本覆盖与异常提示</h2>",
           "<p>分数保留原 benchmark 算法；以下提示不修改分数、权重或分母。</p>"]
    for record in collected:
        out += [f"<h3>{_h(_bench_label(record))}</h3>", f"<p>{_h(_coverage_status(record))}</p>"]
        if record.get("experiment_name"):
            out.append(f"<p>实验：{_h(record['experiment_name'])}；setting：{_h(record.get('setting_name'))}</p>")
        coverage = record.get("coverage")
        if coverage is None:
            out.append(f"<p>{_h(record.get('coverage_warning') or '没有覆盖率证据，无法确认缺失数量和样本 ID。')}</p>")
            continue
        counts = coverage.get("counts") or {}
        out.append(f"<p>应评 {_h(_fmt(counts.get('expected')))}，有效评分 {_h(_fmt(counts.get('valid_scored')))}；"
                   f"缺图 {_h(_fmt(counts.get('missing_image')))}，损坏 {_h(_fmt(counts.get('corrupt_image')))}，"
                   f"judge 失败 {_h(_fmt(counts.get('judge_failed')))}，"
                   f"未评分/原因不明 {_h(_fmt(counts.get('unscored')))}。</p>")
        for warning in coverage.get("warnings", []):
            out.append(f'<p class="bad">{_h(warning)}</p>')
        anomalies = [row for row in coverage.get("samples", []) if row.get("status") != "scored"]
        if anomalies:
            out.append("<details open><summary>异常样本清单</summary><table>"
                       "<tr><th>样本 ID</th><th>类别</th><th>状态</th><th>图片路径</th><th>原因</th></tr>")
            for row in anomalies:
                out.append("<tr>" + "".join(f"<td>{_h(v)}</td>" for v in (
                    row.get("sample_id", ""), row.get("category", ""),
                    STATUS_LABELS.get(row.get("status"), "原因不明"), row.get("image_path", ""), row.get("reason", "")
                )) + "</tr>")
            out.append("</table></details>")
    return "\n".join(out)


def _qualitative_html(qualitative):
    out = ["<h2>二、定性分析</h2>"]
    if not qualitative:
        out.append("<p><em>无定性分析。</em></p>")
        return "\n".join(out)
    for q in qualitative:
        out.append(f"<h3>{_h(q['benchmark'])}</h3>")
        if q.get("error"):
            out.append(f'<p><em>跳过：{_h(q["error"])}</em></p>')
            continue
        out.append(f'<p><em>判片模型：{_h(q.get("model_used","?"))}，'
                   f'采样 {_h(q.get("n_sampled","?"))} 张。</em></p>')
        for dim, text in (q.get("findings") or {}).items():
            label = _DIM_LABELS.get(dim, dim)
            out.append(f'<p class="finding"><b>{_h(label)}：</b>{_h(text)}</p>')
        examples = [e for e in (q.get("failure_examples") or []) if e.get("path") and e["path"] != "?"]
        if examples:
            out.append('<p><b>失败样本：</b></p><div class="fails">')
            for ex in examples:
                thumb = image_thumb.to_data_url(ex["path"], max_edge=384)
                img = f'<img src="{thumb}" alt="sample"/>' if thumb else "<em>（图片不可用）</em>"
                out.append(
                    f'<div class="fail">{img}'
                    f'<div><span class="cat">{_h(ex.get("category","?"))}</span>：{_h(ex.get("note",""))}</div>'
                    f'<div><code>{_h(ex["path"])}</code></div></div>'
                )
            out.append('</div>')
    return "\n".join(out)


_CFG_LABELS = {
    "model_path": "模型路径", "image_size": "分辨率", "num_nodes": "节点数",
    "partition": "分区", "worker_spec": "worker 规格", "cfg_scale": "CFG scale",
    "timestep_shift": "timestep shift", "cfg_norm": "CFG norm", "log_root": "日志目录",
}


def _run_config_html(run_config):
    out = ["<h2>三、运行配置</h2><ul>"]
    for k, v in run_config.items():
        if k == "output_roots":
            continue
        out.append(f"<li><b>{_h(_CFG_LABELS.get(k, k))}：</b>{_h(_fmt(v))}</li>")
    roots = run_config.get("output_roots") or {}
    if roots:
        out.append("<li><b>输出目录：</b><ul>")
        for b, p in roots.items():
            out.append(f"<li>{_h(b)}: <code>{_h(p)}</code></li>")
        out.append("</ul></li>")
    out.append("</ul>")
    return "\n".join(out)


def render_html(collected, qualitative, run_config):
    model = run_config.get("model_path", "?")
    body = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<title>评测报告</title>", f"<style>{_CSS}</style></head><body>",
        "<h1>评测报告</h1>", f"<p>模型：<code>{_h(model)}</code></p>",
        _scores_table_html(collected),
        _coverage_html(collected),
        _qualitative_html(qualitative),
        _run_config_html(run_config),
        "</body></html>",
    ]
    return "\n".join(body)


def write_report(out_path, collected, qualitative, run_config, fmt=None):
    if fmt is None:
        fmt = "html" if out_path.lower().endswith((".html", ".htm")) else "md"
    text = (render_html if fmt == "html" else render)(collected, qualitative, run_config)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def _main():
    ap = argparse.ArgumentParser(description="Render an aggregated eval-report Markdown.")
    ap.add_argument("--collected", required=True, help="JSON file: list of collect_results records")
    ap.add_argument("--qualitative", default=None, help="JSON file: list of qualitative_review records")
    ap.add_argument("--run-config", required=True, help="JSON file: run_config dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["html", "md"], default=None,
                    help="output format (default: inferred from --out extension)")
    args = ap.parse_args()
    collected = json.load(open(args.collected))
    from collect_results import attach_coverage, attach_experiment
    collected = [attach_coverage(attach_experiment(record), language=record.get("language") or
                                record.get("metrics", {}).get("language") or "en") for record in collected]
    qualitative = json.load(open(args.qualitative)) if args.qualitative else []
    run_config = json.load(open(args.run_config))
    path = write_report(args.out, collected, qualitative, run_config, fmt=args.format)
    print(path)


if __name__ == "__main__":
    _main()
