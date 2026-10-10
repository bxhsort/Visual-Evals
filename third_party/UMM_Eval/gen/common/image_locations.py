"""Benchmark -> generated-image locations + optional failure-sample selection.

Companion to ``output_paths.py``: that module gives the *results root* (where the scorer
writes); this one gives, relative to that same results root, the globs that match the
*generated images*, plus a per-benchmark way to pick failure cases for qualitative review.

Globs verified against real outputs under outputs/eval_output_cfg4_1024_ts3/<model>/<step>/.
"""
from __future__ import annotations

import glob
import json
import os

# bench -> list of globs relative to the results root
BENCH_IMAGE_GLOBS = {
    "geneval":       ["gen_eval/*/samples/*.png"],
    "geneval2":      ["samples/*.png"],
    "dpgbench":      ["dpgbench/*.png"],
    "longtextbench": ["samples/*_sample.png"],          # results root already includes longtextbench_<mode>
    "cvtg_2k":       ["CVTG/*/*/*.png"],
    "tiifbench":     ["TIIFBench-*/*/*/*/*.png"],
    "oneigbench":    ["OneIGBench-*/*/*/*.webp"],
    "imgedit":       ["imgedit/*.png"],
    "geditbench":    ["gedit_bench/*/*/*.png"],
}


# Benchmarks whose images are partitioned by language: when a language is given, narrow the
# glob so e.g. oneigbench-zh samples only the ZH images (not EN+ZH).
LANG_GLOBS = {
    "oneigbench": ["OneIGBench-{lang_upper}/*/*/*.webp"],
    "geditbench": ["gedit_bench/*/{lang}/*.png"],
}


def list_images(bench, results_root, language=None):
    """All generated-image absolute paths for ``bench`` under ``results_root`` (sorted, unique).

    If ``language`` is given and ``bench`` is language-partitioned, only that language's
    images are returned.
    """
    if bench not in BENCH_IMAGE_GLOBS:
        raise ValueError(f"unknown benchmark {bench!r}; known: {sorted(BENCH_IMAGE_GLOBS)}")
    if language and bench in LANG_GLOBS:
        patterns = [p.format(lang=language, lang_upper=language.upper()) for p in LANG_GLOBS[bench]]
    else:
        patterns = BENCH_IMAGE_GLOBS[bench]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(results_root, pattern)))
    # absolute so dedup against the (absolute) failure-selector paths is reliable even
    # when results_root is relative.
    return sorted({os.path.abspath(p) for p in paths})


def _failures_geneval(results_root):
    path = os.path.join(results_root, "results_geneval.jsonl")
    if not os.path.isfile(path):
        return []
    fails = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("correct") is False and rec.get("filename"):
                fails.append(rec["filename"])
    return fails


def _failures_longtextbench(results_root):
    path = os.path.join(results_root, "eval", "results.jsonl")
    if not os.path.isfile(path):
        return []
    scored = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img = rec.get("image")
            acc = rec.get("text_accuray")  # field is spelled this way in the data
            if img is not None and acc is not None:
                scored.append((acc, img))
    scored.sort(key=lambda t: t[0])           # lowest accuracy first
    return [img for _acc, img in scored]


# bench -> selector(results_root) -> ordered list of failing/low-score absolute image paths
FAILURE_SELECTORS = {
    "geneval": _failures_geneval,
    "longtextbench": _failures_longtextbench,
}


def sample_images(bench, results_root, k=12, prefer_failures=True, language=None):
    """Pick up to ``k`` generated images, preferring failure cases when available.

    Returns absolute paths, deterministic. Failures (if any and ``prefer_failures``) come
    first, then evenly-spaced fill from the remaining images for coverage. ``language``
    narrows the image set for language-partitioned benchmarks (e.g. oneigbench EN vs ZH).
    """
    all_imgs = list_images(bench, results_root, language=language)
    if not all_imgs:
        return []
    chosen = []
    if prefer_failures and bench in FAILURE_SELECTORS:
        for p in FAILURE_SELECTORS[bench](results_root):
            ap = p if os.path.isabs(p) else os.path.join(results_root, p)
            ap = os.path.abspath(ap)  # match list_images' absolute paths for dedup
            if os.path.isfile(ap) and ap not in chosen:
                chosen.append(ap)
            if len(chosen) >= k:
                return chosen[:k]
    remaining = [p for p in all_imgs if p not in chosen]
    need = k - len(chosen)
    if need > 0 and remaining:
        if len(remaining) <= need:
            chosen.extend(remaining)
        else:
            step = len(remaining) / need
            for i in range(need):
                chosen.append(remaining[int(i * step)])
    return chosen[:k]


# ---------------------------------------------------------------------------
# it2i (image-editing) context: an edited result image alone can't be judged —
# the reviewer also needs the ORIGINAL input image and the edit INSTRUCTION.
# Both imgedit and geditbench key their metadata by the result filename stem.
# ---------------------------------------------------------------------------

IT2I_BENCHES = {"imgedit", "geditbench"}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_it2i_meta_cache = {}


def _imgedit_meta():
    if "imgedit" not in _it2i_meta_cache:
        path = os.path.join(_REPO_ROOT, "gen/it2i/imgedit/eval/Benchmark/singleturn/singleturn.json")
        with open(path, "r", encoding="utf-8") as f:
            _it2i_meta_cache["imgedit"] = json.load(f)  # key -> {id, prompt, edit_type}
    return _it2i_meta_cache["imgedit"]


def _geditbench_meta():
    if "geditbench" not in _it2i_meta_cache:
        path = os.path.join(_REPO_ROOT, "gen/it2i/geditbench/eval/prompts.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # list of {task_type, key, instruction}
        _it2i_meta_cache["geditbench"] = {str(e["key"]): e for e in data}
    return _it2i_meta_cache["geditbench"]


def it2i_context(bench, result_path):
    """For an it2i result image, return ``{"original": abspath|None, "instruction": str|None}``.

    Looks the original input image + edit instruction up by the result filename stem (the
    sample key). Never raises — returns Nones if metadata/files are missing.
    """
    key = os.path.splitext(os.path.basename(result_path))[0]
    try:
        if bench == "imgedit":
            entry = _imgedit_meta().get(key)
            if not entry:
                return {"original": None, "instruction": None}
            orig = os.path.join(_REPO_ROOT, "gen/it2i/imgedit/eval/Benchmark/singleturn", entry["id"])
            return {"original": orig if os.path.isfile(orig) else None,
                    "instruction": entry.get("prompt")}
        if bench == "geditbench":
            entry = _geditbench_meta().get(key)
            orig = os.path.join(_REPO_ROOT, "gen/it2i/geditbench/eval/input_images", key + ".png")
            return {"original": orig if os.path.isfile(orig) else None,
                    "instruction": entry.get("instruction") if entry else None}
    except Exception:
        pass
    return {"original": None, "instruction": None}
