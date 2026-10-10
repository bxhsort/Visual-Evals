#!/usr/bin/env python3
"""One-time extractor: Qwen-Image-Bench source jsonl -> committed prompt files.

Reads the 185 MB `qwen_image_bench_hf_v0518.jsonl` (fields: ID, prompt_en, prompt_cn, ...)
and emits the two small committed prompt files this benchmark ships:

    prompts/qwenimagebench_en.jsonl   {"ID": int, "prompt": <prompt_en>}
    prompts/qwenimagebench_zh.jsonl   {"ID": int, "prompt": <prompt_cn>}

Usage:
    python build_prompts.py --src /abs/qwen_image_bench_hf_v0518.jsonl
"""
import argparse
import json
import os

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BENCH_DIR, "prompts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to qwen_image_bench_hf_v0518.jsonl")
    args = ap.parse_args()

    os.makedirs(PROMPTS_DIR, exist_ok=True)
    rows = []
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["ID"]))

    for lang, field in (("en", "prompt_en"), ("zh", "prompt_cn")):
        out = os.path.join(PROMPTS_DIR, f"qwenimagebench_{lang}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"ID": int(r["ID"]), "prompt": r[field]},
                                   ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
