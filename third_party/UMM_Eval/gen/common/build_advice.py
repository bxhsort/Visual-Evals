"""Read a generated eval-report HTML, ask a strong model for an overall training + data
recommendation based on the scores and qualitative findings, and append it to the report as a
``四、训练与数据建议`` section (idempotent — replaces a prior advice section if present).

Never raises on a model/key failure: writes a visible note into the section instead, so the
report is always left valid.
"""
from __future__ import annotations

import argparse
import html
import os
import re

import eval_api_key

ADVICE_PROMPT = (
    "你是统一多模态生成模型（文生图 + 图像编辑）的训练负责人。下面是该模型在一组生成评测榜单上的"
    "评测报告，包含每个榜单的分数和逐榜的定性分析（缺陷、prompt 对齐、文字渲染、编辑指令遵循等）。"
    "请通读全部内容后，给出一份**整体的训练与数据改进建议**，要求：\n"
    "1. 先用 2-3 句话总结模型当前的整体水平和最突出的强项/短板；\n"
    "2. 按能力维度分点（如：文字渲染、计数/数量、空间关系、属性绑定、长文本、图像编辑的指令遵循与"
    "区域保留、美学等），每点指出证据（引用相关榜单/定性现象），并给出**针对性的数据补充和训练策略"
    "建议**（具体、可执行，例如该补什么类型/规模的数据、调整什么采样配比或训练目标）；\n"
    "3. 最后给出优先级排序（先做什么收益最大）。\n"
    "用简体中文，结构清晰，可用 - 列表。不要复述分数表本身，聚焦诊断与建议。"
)


def _html_to_text(report_html):
    """Strip the report HTML down to the text the model needs (drops embedded base64 images)."""
    s = re.sub(r"<img[^>]*>", "", report_html)
    s = re.sub(r"<style.*?</style>", "", s, flags=re.S)
    s = re.sub(r"<script.*?</script>", "", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def generate_advice(report_text, model="gpt-5.5"):
    """Call the vision/LLM gateway for the recommendation. Returns text, or raises (caller guards)."""
    from openai import OpenAI
    key = eval_api_key.resolve_eval_api_key()
    client = OpenAI(api_key=key, base_url=eval_api_key.TOKENHUB_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ADVICE_PROMPT + "\n\n=== 评测报告 ===\n" + report_text}],
        max_tokens=4000,
    )
    return resp.choices[0].message.content.strip()


def _advice_to_html(advice_text):
    body = html.escape(advice_text)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)  # markdown bold -> <b>
    return ('<h2>四、训练与数据建议</h2>\n'
            '<div style="white-space:pre-wrap;line-height:1.8;background:#f8f9ff;'
            'border-left:4px solid #4a6;padding:1rem 1.2rem;border-radius:6px">'
            + body + '</div>')


_ADVICE_RE = re.compile(r"<h2>四、训练与数据建议</h2>.*?(?=</body>)", re.S)


def append_advice(report_path, model="gpt-5.5", out_path=None):
    """Generate the advice and inject it before </body>. Idempotent. Never raises."""
    with open(report_path, "r", encoding="utf-8") as f:
        report = f.read()
    report = _ADVICE_RE.sub("", report)  # drop any prior advice section
    try:
        advice = generate_advice(_html_to_text(report), model=model)
        section = _advice_to_html(advice)
    except Exception as e:
        section = ('<h2>四、训练与数据建议</h2>\n<div><em>建议生成失败：'
                   + html.escape(f"{type(e).__name__}: {e}") + '</em></div>')
    out = report.replace("</body>", section + "\n</body>", 1) if "</body>" in report else report + section
    out_path = out_path or report_path
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    return out_path


def _main():
    ap = argparse.ArgumentParser(description="Append a training+data recommendation to an eval-report HTML.")
    ap.add_argument("--report", required=True, help="path to the eval-report .html")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--out", default=None, help="write here instead of in place")
    args = ap.parse_args()
    print(append_advice(args.report, model=args.model, out_path=args.out))


if __name__ == "__main__":
    _main()
