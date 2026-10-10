#!/usr/bin/env python3
"""Build judge.py input ({ID, prompt, image_path}) from generated images.

Judge prompts are always read from the benchmark's bundled raw prompt files.
Generation may use a prompt-enhanced DATA_ROOT, but that must not change the
text Q-Judger evaluates against.
"""
import argparse
import json
from pathlib import Path


RAW_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=("en", "zh"))
    ap.add_argument("--samples-dir", required=True, help="dir holding <ID>.png")
    ap.add_argument("--out", required=True, help="output jsonl path")
    args = ap.parse_args()

    prompts = RAW_PROMPTS_DIR / f"qwenimagebench_{args.lang}.jsonl"
    samples_dir = Path(args.samples_dir)
    n_total = n_written = 0
    with prompts.open(encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            row = json.loads(line)
            img = samples_dir / f"{row['ID']}.png"
            if not img.exists():
                continue
            g.write(json.dumps({"ID": int(row["ID"]), "prompt": row["prompt"],
                                "image_path": str(img)}, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"build_judge_input: {n_written}/{n_total} prompts had images -> {args.out}")
    if n_written == 0:
        raise SystemExit("ERROR: no generated images found; nothing to judge.")


if __name__ == "__main__":
    main()
