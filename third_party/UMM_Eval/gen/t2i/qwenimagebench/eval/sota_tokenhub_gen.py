#!/usr/bin/env python3
"""Qwen-Image-Bench SOTA/baseline generation via TokenHub (e.g. gpt-image-2).

Generates through the SenseTime TokenHub OpenAI-compatible `images.generate` endpoint,
which serves `gpt-image-2` with the same `token_hub.api_key` used elsewhere. Output lands
in the SAME flat layout as the local inference.py:

    <output_base>/<output_root>/<model_name>/<step_name>/qwenimagebench_<mode>/samples/<ID>.png

so the identical Q-Judger eval path (build_judge_input.py -> judge.py) scores it unchanged.
Multi-process (I/O-bound API calls), resumable. Key read at runtime from
~/api_keys.yaml:token_hub.api_key (never printed).

Run with PYTHONPATH including gen/common (for output_paths), e.g.:
    PYTHONPATH=gen/common python gen/t2i/qwenimagebench/eval/sota_tokenhub_gen.py --mode en
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import output_paths


DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")
TOKENHUB_BASE_URL = os.environ.get("QIB_SOTA_BASE_URL", "https://tokenhub.sensetime.com/v1")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_api_key() -> str:
    """token_hub.api_key from env QIB_TOKENHUB_KEY, ~/api_keys.yaml, or the user's absolute
    home (container $HOME may differ). Never printed."""
    import yaml
    env = os.environ.get("QIB_TOKENHUB_KEY")
    if env:
        return env
    for cand in (Path.home() / "api_keys.yaml",):
        if cand.exists():
            return yaml.safe_load(open(cand))["token_hub"]["api_key"]
    raise RuntimeError("token_hub.api_key not found (env QIB_TOKENHUB_KEY or api_keys.yaml)")


def _decode_image(item):
    from PIL import Image

    b64 = item.get("b64_json")
    if b64:
        if isinstance(b64, str):
            payload = b64.strip()
            if payload.lower().startswith("data:"):
                header, separator, payload = payload.partition(",")
                metadata = [part.strip().lower() for part in header[5:].split(";")]
                if (not separator or not metadata[0].startswith("image/")
                        or "base64" not in metadata[1:]):
                    raise ValueError("b64_json has an invalid base64 image data URL")
            # Strict decoding catches malformed API payloads while retaining support for
            # line-wrapped raw base64 returned by some OpenAI-compatible gateways.
            payload = "".join(payload.split()).encode("ascii")
        else:
            # base64.b64decode historically accepted bytes here, so keep doing so.
            payload = b"".join(b64.split())
        image_bytes = base64.b64decode(payload, validate=True)
        if not image_bytes:
            raise ValueError("b64_json decoded to an empty image")
        # Pillow decoding is lazy. Verify the complete payload first, then reopen it to
        # return the same RGB image object callers received before this validation.
        with Image.open(io.BytesIO(image_bytes)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return image.convert("RGB")
    url = item.get("url")
    if url:
        import requests
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise RuntimeError(f"image item missing b64_json/url: {list(item.keys())}")


# Substrings that mark a deterministic content-policy / safety refusal — retrying is pointless.
_REFUSAL_MARKERS = (
    "content_policy", "content policy", "safety", "moderation", "moderation_blocked",
    "rejected", "blocked", "sensitive", "violat", "not allowed", "policy", "risk",
    "审核", "安全", "敏感", "违规", "拒绝",
)


def _is_refusal(err_text: str) -> bool:
    low = err_text.lower()
    return any(m in low or m in err_text for m in _REFUSAL_MARKERS)


def gen_one(task):
    """Worker: generate one image via tokenhub and save it.
    Returns (ID, status, err) where status in {success, skipped, refused, failed}."""
    from openai import OpenAI
    rec = task["rec"]
    out_path = Path(task["out_path"])
    if task["resume"] and out_path.exists():
        return rec["ID"], "skipped", None
    last_err = None
    for attempt in range(task["max_retries"]):
        try:
            client = OpenAI(api_key=task["api_key"], base_url=task["base_url"])
            resp = client.images.generate(
                model=task["model"], prompt=rec["prompt"], size=task["size"], n=1,
            )
            item = resp.data[0]
            d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            img = _decode_image(d)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return rec["ID"], "success", None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if _is_refusal(last_err):
                return rec["ID"], "refused", last_err
            time.sleep(2.0 * (attempt + 1))
    return rec["ID"], "failed", last_err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["en", "zh"], required=True)
    ap.add_argument("--model_name", default="gpt-image-2")
    ap.add_argument("--step_name", default="api")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output_base", default=os.environ.get("OUTPUT_BASE", os.path.join(os.getcwd(), "outputs")))
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--image_size", type=int, default=1024, help="for output-root parity")
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument("--cfg_norm", default="global")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("QIB_SOTA_WORKERS", "48")))
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--target_prompt_ids", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    data_path = os.path.join(args.data_root, f"qwenimagebench_{args.mode}.jsonl")
    output_root = output_paths.derive_t2i_root(args.image_size, args.cfg_scale, args.timestep_shift, args.cfg_norm)
    samples_dir = os.path.join(
        args.output_base, output_root, args.model_name, args.step_name, f"qwenimagebench_{args.mode}", "samples",
    )

    data = read_jsonl(data_path)
    if args.target_prompt_ids:
        want = {x.strip() for x in args.target_prompt_ids.split(",") if x.strip()}
        data = [r for r in data if str(r["ID"]) in want]

    api_key = load_api_key()
    tasks = [{
        "rec": r,
        "out_path": os.path.join(samples_dir, f"{r['ID']}.png"),
        "api_key": api_key, "base_url": TOKENHUB_BASE_URL, "model": args.model_name,
        "size": args.size, "resume": args.resume, "max_retries": args.max_retries,
    } for r in data]

    print(f"[sota] mode={args.mode} model={args.model_name} prompts={len(tasks)} "
          f"workers={args.workers} -> {samples_dir}", flush=True)

    success = skipped = failed = refused = 0
    fails, refusals = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            pid, status, err = fut.result()
            done += 1
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            elif status == "refused":
                refused += 1
                refusals.append({"ID": pid, "error": err})
            else:
                failed += 1
                fails.append(pid)
                print(f"[FAIL] {pid}: {err}", flush=True)
            if done % 100 == 0:
                print(f"[sota] {done}/{len(tasks)} (ok={success} skip={skipped} refused={refused} fail={failed})", flush=True)

    if refusals:
        manifest = os.path.join(os.path.dirname(samples_dir), "refusals.jsonl")
        os.makedirs(os.path.dirname(manifest), exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            for r in refusals:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[sota] {refused} refused -> manifest {manifest}", flush=True)

    print(f"[sota] DONE mode={args.mode}: success={success} skipped={skipped} "
          f"refused={refused} failed={failed}", flush=True)
    if fails:
        print(f"[sota] {len(fails)} transient failures: re-run (resumable) to retry", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
