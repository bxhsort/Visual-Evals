#!/usr/bin/env bash
# Resize every PNG/JPG/WEBP image under <root> to <size>x<size> in-place.
# Usage: bash resize_outputs.sh <root_dir> <size> [num_workers]
set -eo pipefail

ROOT="${1:?root_dir required}"
SIZE="${2:?target size required}"
WORKERS="${3:-16}"

if [[ ! -d "$ROOT" ]]; then
  echo "resize_outputs: root dir does not exist: $ROOT" >&2
  exit 1
fi
if ! [[ "$SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "resize_outputs: target size must be a positive integer, got: $SIZE" >&2
  exit 1
fi

echo "resize_outputs: scanning $ROOT for images to resize to ${SIZE}x${SIZE} (workers=$WORKERS)..."

python - "$ROOT" "$SIZE" "$WORKERS" <<'PY'
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

root, size, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
exts = {".png", ".jpg", ".jpeg", ".webp"}

def collect(root):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                yield os.path.join(dirpath, name)

def resize_one(path, target):
    try:
        with Image.open(path) as im:
            if im.size == (target, target):
                return path, "skip-already"
            fmt = im.format
            mode = im.mode
            resized = im.resize((target, target), Image.LANCZOS)
        save_kwargs = {}
        if fmt == "WEBP":
            save_kwargs["lossless"] = False
            save_kwargs["quality"] = 95
        resized.save(path, format=fmt, **save_kwargs)
        return path, "ok"
    except Exception as e:
        return path, f"err: {e}"

paths = list(collect(root))
print(f"resize_outputs: found {len(paths)} image files", flush=True)
if not paths:
    sys.exit(0)

ok = skipped = failed = 0
with ProcessPoolExecutor(max_workers=workers) as ex:
    futures = [ex.submit(resize_one, p, size) for p in paths]
    for fut in as_completed(futures):
        path, status = fut.result()
        if status == "ok":
            ok += 1
        elif status == "skip-already":
            skipped += 1
        else:
            failed += 1
            print(f"  FAILED {path}: {status}", flush=True)

print(f"resize_outputs: done. resized={ok}, already_target={skipped}, failed={failed}", flush=True)
if failed:
    sys.exit(1)
PY
