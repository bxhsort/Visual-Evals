#!/usr/bin/env python3
"""Symlink Qwen-Image-Bench official reference images into our standard sample layout.

The source dataset jsonl has one image path per model per row (column e.g. `gpt-image-2` =
"images/gpt-image-2/000001_<hash>.png"). This links each row's image to
    <output_base>/<output_root>/<model_name>/<step_name>/qwenimagebench_{en,zh}/samples/<ID>.png
for BOTH languages (same image file), so the normal eval path
(build_judge_input.py -> judge.py) scores them unchanged — judging each language with its own
prompt. Used to reproduce the official leaderboard from the bundled reference images (a
free, exact-parity check of our ported Q-Judger).
"""
import argparse
import json
import os

import output_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-jsonl", required=True, help="qwen_image_bench_hf_v0518.jsonl")
    ap.add_argument("--images-root", required=True,
                    help="dir the image paths are relative to (holds images/<model>/...)")
    ap.add_argument("--model-col", default="gpt-image-2", help="jsonl column with the image path")
    ap.add_argument("--output-base", required=True)
    ap.add_argument("--model-name", default="gpt-image-2-official")
    ap.add_argument("--step-name", default="refs")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--timestep-shift", type=float, default=3.0)
    ap.add_argument("--cfg-norm", default="global")
    ap.add_argument("--langs", default="en zh")
    args = ap.parse_args()

    root = output_paths.derive_t2i_root(args.image_size, args.cfg_scale, args.timestep_shift, args.cfg_norm)
    rows = []
    with open(args.src_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    for lang in args.langs.split():
        samples_dir = os.path.join(args.output_base, root, args.model_name, args.step_name,
                                   f"qwenimagebench_{lang}", "samples")
        os.makedirs(samples_dir, exist_ok=True)
        linked = missing = 0
        for r in rows:
            rel = r.get(args.model_col)
            if not rel:
                missing += 1
                continue
            src = os.path.join(args.images_root, rel)
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(samples_dir, f"{r['ID']}.png")
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            linked += 1
        print(f"[{lang}] linked={linked} missing={missing} -> {samples_dir}")


if __name__ == "__main__":
    main()
