"""Derive a lower-resolution copy of a t2i results tree (e.g. 2048 -> 1024).

Unlike ``resize_outputs.sh`` (which resizes IN-PLACE and overwrites the originals), this
COPIES a model/step's images from one t2i output root to another, resizing on the way, so
both resolutions coexist. The source/dest roots are derived from the single source of truth
``output_paths.derive_t2i_root`` (same scheme the launchers + scorers use), so the result
slots straight into a score-only run: after deriving the 1024 tree, run the bench's scorer
with ``RUN_GENERATION=0 RUN_EVAL=1 IMAGE_SIZE=1024 MODEL_NAME=... STEP_NAME=...``.

Only the per-image ``samples`` payload is copied (PNG/JPG/WEBP under the model/step dir);
score/judge artifacts are NOT copied (the scorer will regenerate them at the new size).

Usage:
    python gen/common/resize_to_root.py \
        --output-base "$PWD/outputs" --model-name M --step-name S \
        --from-size 2048 --to-size 1024 [--cfg-scale 4 --timestep-shift 3 --cfg-norm global] \
        [--workers 16] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from output_paths import derive_t2i_root  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _resize_one(src, dst, size, overwrite):
    try:
        if os.path.exists(dst) and not overwrite:
            return dst, "skip-exists"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(src) as im:
            fmt = im.format
            if im.size == (size, size):
                shutil.copy2(src, dst)
                return dst, "copy-same-size"
            resized = im.resize((size, size), Image.LANCZOS)
        save_kwargs = {"quality": 95} if fmt == "WEBP" else {}
        resized.save(dst, format=fmt, **save_kwargs)
        return dst, "ok"
    except Exception as e:  # noqa: BLE001
        return dst, f"err: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-base", required=True, help="e.g. $PWD/outputs")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--step-name", required=True)
    ap.add_argument("--from-size", type=int, required=True)
    ap.add_argument("--to-size", type=int, required=True)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--timestep-shift", type=float, default=3.0)
    ap.add_argument("--cfg-norm", default="global")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true", help="re-resize even if dest exists")
    args = ap.parse_args()

    src_root = derive_t2i_root(args.from_size, args.cfg_scale, args.timestep_shift, args.cfg_norm)
    dst_root = derive_t2i_root(args.to_size, args.cfg_scale, args.timestep_shift, args.cfg_norm)
    src_dir = os.path.join(args.output_base, src_root, args.model_name, args.step_name)
    dst_dir = os.path.join(args.output_base, dst_root, args.model_name, args.step_name)

    if not os.path.isdir(src_dir):
        print(f"resize_to_root: source not found: {src_dir}", file=sys.stderr)
        sys.exit(1)
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        print("resize_to_root: from-size == to-size; nothing to do", file=sys.stderr)
        sys.exit(1)

    print(f"resize_to_root: {src_dir}\n            -> {dst_dir}  (resize to {args.to_size}x{args.to_size})")

    jobs = []
    for dirpath, _, files in os.walk(src_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in EXTS:
                src = os.path.join(dirpath, name)
                dst = os.path.join(dst_dir, os.path.relpath(src, src_dir))
                jobs.append((src, dst))
    print(f"resize_to_root: {len(jobs)} images")
    if not jobs:
        return

    ok = skip = same = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_resize_one, s, d, args.to_size, args.overwrite) for s, d in jobs]
        for fut in as_completed(futs):
            _, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip-exists":
                skip += 1
            elif status == "copy-same-size":
                same += 1
            else:
                failed += 1
                print(f"  {status}", file=sys.stderr)
    print(f"resize_to_root: done. resized={ok}, copied_same={same}, skipped_existing={skip}, failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
