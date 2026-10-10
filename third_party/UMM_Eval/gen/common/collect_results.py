"""Unified results extraction across v1 benchmarks.

Each benchmark's scorer writes its headline metric in its own format and location.
This module gives one contract: given a benchmark + the directory its scorer wrote
into (the *results root*, see ``output_paths.derive_output_root``), return a normalized
record::

    {"benchmark", "model_name", "step_name", "output_root",
     "headline", "headline_metric", "metrics", "raw_files", "complete"}

``complete`` is False (and ``headline`` None) when the score file is absent or unparsable
— this lets an orchestrator poll until the scores have landed.

Score sources (verified against each scorer):
    geneval        <root>/results_geneval.jsonl_scores.txt   "Overall score (avg. over tasks): X"  (0-1)
    dpgbench       <root>/dpgbench_results.txt                last "DPG-Bench score: X"             (0-100)
    longtextbench  <root>/eval/scores.txt                     "Text Score: X"                       (0-1)
    imgedit        <root>/imgedit_typescore.json              JSON dict, key "overall"
    geditbench     <root>/gedit_*_scores.json                 JSON, languages[<lang>].overall.overall

CLI::

    python collect_results.py --bench geneval --results-root /abs/.../model/step
    # or let it derive the root from the sampling knobs:
    python collect_results.py --bench geneval --output-base /abs/outputs \
        --model-name m --step-name s   # (knobs default to the tuned defaults)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import output_paths


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _result(bench, output_root, model_name, step_name, *, headline=None,
            headline_metric=None, metrics=None, raw_files=None):
    return {
        "benchmark": bench,
        "model_name": model_name,
        "step_name": step_name,
        "output_root": output_root,
        "headline": headline,
        "headline_metric": headline_metric,
        "metrics": metrics or {},
        "raw_files": raw_files or [],
        "complete": headline is not None,
    }


# ---------------------------------------------------------------------------
# Per-benchmark adapters: each takes the results root and returns (headline,
# headline_metric, metrics, raw_files). headline is None if scores not present.
# ---------------------------------------------------------------------------

def _collect_geneval(root):
    # summary_scores.py writes "results_geneval.jsonl".replace(".jsonl", "_scores.txt")
    # => results_geneval_scores.txt. Glob as a fallback for alternate names.
    path = os.path.join(root, "results_geneval_scores.txt")
    if not os.path.isfile(path):
        cands = sorted(glob.glob(os.path.join(root, "results_geneval*scores.txt")))
        if cands:
            path = cands[-1]
    if not os.path.isfile(path):
        return None, None, {}, []
    text = _read_text(path)
    metrics = {}
    m = re.search(r"Overall score \(avg\. over tasks\):\s*([\d.]+)", text)
    headline = float(m.group(1)) if m else None
    for label, key in (("% correct images", "correct_images"),
                        ("% correct prompts", "correct_prompts")):
        mm = re.search(rf"{re.escape(label)}:\s*([\d.]+)%", text)
        if mm:
            metrics[key] = float(mm.group(1)) / 100.0
    for tag, val in re.findall(r"^(\w[\w ]*?)\s*=\s*([\d.]+)%", text, re.MULTILINE):
        metrics[f"task/{tag.strip()}"] = float(val) / 100.0
    if headline is not None:
        metrics["overall"] = headline
    return headline, "overall_score", metrics, [path]


def _collect_geneval2(root):
    path = os.path.join(root, "eval", "results.json")
    if not os.path.isfile(path):
        return None, None, {}, []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    headline = data.get("score")
    return headline, data.get("method", "soft_tifa_gm"), data, [path]


def _collect_dpgbench(root):
    path = os.path.join(root, "dpgbench_results.txt")
    if not os.path.isfile(path):
        return None, None, {}, []
    text = _read_text(path)
    matches = re.findall(r"DPG-Bench score:\s*([\d.]+)", text)
    if not matches:
        return None, None, {}, [path]
    headline = float(matches[-1])  # file is appended; take the final summary
    metrics = {"dpg_bench_score": headline}
    for cat, val in re.findall(r"^\t([^:\n]+):\s*([\d.]+)\s*$", text, re.MULTILINE):
        metrics[f"category/{cat.strip()}"] = float(val)
    return headline, "dpg_bench_score", metrics, [path]


def _collect_longtextbench(root):
    path = os.path.join(root, "eval", "scores.txt")
    if not os.path.isfile(path):
        return None, None, {}, []
    text = _read_text(path)
    m = re.search(r"Text Score:\s*([\d.]+)", text)
    headline = float(m.group(1)) if m else None
    metrics = {"text_score": headline} if headline is not None else {}
    return headline, "text_score", metrics, [path]


def _collect_imgedit(root):
    path = os.path.join(root, "imgedit_typescore.json")
    if not os.path.isfile(path):
        return None, None, {}, []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    headline = data.get("overall")
    metrics = {k: v for k, v in data.items()}
    return headline, "overall", metrics, [path]


def _collect_geditbench(root, language="en"):
    # filename is gedit_<languages-list-repr>_<backbone>_scores.json — glob it.
    candidates = sorted(glob.glob(os.path.join(root, "gedit_*_scores.json")))
    if not candidates:
        return None, None, {}, []
    path = candidates[-1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    langs = data.get("languages", {})
    lang = language if language in langs else (next(iter(langs), None))
    headline = None
    metrics = {}
    if lang is not None:
        overall = langs[lang].get("overall", {})
        # Two known layouts:
        #  - repo calculate_statistics.py: overall = {<group>:..., "semantics":avg, "quality":avg, "overall":avg}
        #  - reference outputs:            overall = {<group>:..., "average": {"semantics","quality","overall"}}
        avg = overall.get("average") if isinstance(overall, dict) else None
        src = avg if isinstance(avg, dict) else overall
        headline = src.get("overall")
        metrics = {
            "language": lang,
            "semantics": src.get("semantics"),
            "quality": src.get("quality"),
            "overall": src.get("overall"),
        }
    return headline, "overall", metrics, [path]


def _collect_cvtg_2k(root):
    path = os.path.join(root, "CVTG_results.json")
    if not os.path.isfile(path):
        return None, None, {}, []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ov = data.get("overall_results", {})
    headline = ov.get("word_accuracy")
    metrics = {
        "word_accuracy": ov.get("word_accuracy"),
        "ned": ov.get("ned"),
        "clipscore": ov.get("clipscore"),
        "vqascore": ov.get("vqascore"),
        "aesthetic_score": ov.get("aesthetic_score"),
    }
    return headline, "word_accuracy", metrics, [path]


def _collect_tiifbench(root, split="testmini"):
    path = os.path.join(root, f"tiifbench-{split}_results", "result_summary_dimension.txt")
    if not os.path.isfile(path):
        cands = sorted(glob.glob(os.path.join(root, "tiifbench-*_results", "result_summary_dimension.txt")))
        if cands:
            path = cands[-1]
    if not os.path.isfile(path):
        return None, None, {}, []
    lines = _read_text(path).splitlines()
    short = long = None
    for i, line in enumerate(lines):
        if "Overall average score of 9 sub-attributes" in line:
            for data_line in lines[i + 1:]:
                nums = re.findall(r"[0-9]*\.[0-9]+", data_line)
                if len(nums) >= 2:
                    short, long = float(nums[0]), float(nums[1])
                    break
            break
    metrics = {"overall_short": short, "overall_long": long}
    return short, "overall_short", metrics, [path]


def _collect_oneigbench(root, language="en"):
    # Read ONLY the requested language's results dir. NEVER fall back to another language's
    # file — that would silently report the wrong language's scores as this one's (and mark
    # an unfinished language "complete").
    results_dir = os.path.join(root, f"oneigbench-{language.upper()}_results")
    cands = sorted(glob.glob(os.path.join(results_dir, "result_*.md")))
    if not cands:
        return None, None, {}, []
    path = cands[-1]
    lines = _read_text(path).splitlines()
    vals = None
    for i, line in enumerate(lines):
        if "Overall Scores" in line:
            for data_line in lines[i + 1:]:
                stripped = data_line.strip()
                if stripped.startswith("|") and "---" not in stripped and "Alignment" not in stripped:
                    nums = re.findall(r"[0-9]*\.[0-9]+", data_line)
                    if len(nums) >= 5:
                        vals = [float(x) for x in nums[:5]]
                        break
            break
    if not vals:
        return None, None, {}, [path]
    a, d, r, s, t = vals
    metrics = {"alignment": a, "diversity": d, "reasoning": r, "style": s, "text": t}
    return a, "alignment", metrics, [path]


_ADAPTERS = {
    "geneval": _collect_geneval,
    "geneval2": _collect_geneval2,
    "dpgbench": _collect_dpgbench,
    "longtextbench": _collect_longtextbench,
    "imgedit": _collect_imgedit,
    "geditbench": _collect_geditbench,
    "cvtg_2k": _collect_cvtg_2k,
    "tiifbench": _collect_tiifbench,
    "oneigbench": _collect_oneigbench,
}


def collect(bench, results_root, *, model_name="model", step_name="step", language="en", split="testmini"):
    if bench not in _ADAPTERS:
        raise ValueError(f"unknown benchmark {bench!r}; known: {sorted(_ADAPTERS)}")
    adapter = _ADAPTERS[bench]
    # Native language folders live inside the managed benchmark leaf.
    if bench == "longtextbench" and os.path.isfile(os.path.join(results_root, "experiment_manifest.json")):
        results_root = os.path.join(results_root, f"longtextbench_{language}")
    if bench in ("geditbench", "oneigbench"):
        headline, hmetric, metrics, raw_files = adapter(results_root, language=language)
    elif bench == "tiifbench":
        headline, hmetric, metrics, raw_files = adapter(results_root, split=split)
    else:
        headline, hmetric, metrics, raw_files = adapter(results_root)
    record = _result(bench, results_root, model_name, step_name,
                     headline=headline, headline_metric=hmetric,
                     metrics=metrics, raw_files=raw_files)
    record["language"] = language
    record = attach_experiment(record)
    return attach_coverage(record, language=language)


def attach_experiment(record):
    """Read exact local experiment identity; never discover another setting's scores."""
    from pathlib import Path
    record = dict(record)
    root = Path(record.get("output_root", "")).resolve()
    for candidate in (root, *list(root.parents)[:3]):
        path = candidate / "experiment_manifest.json"
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            identity = manifest["identity"]
            if not isinstance(identity, dict) or not isinstance(identity.get("config"), dict):
                raise ValueError("invalid experiment identity structure")
            if identity["benchmark"] != record["benchmark"]:
                raise ValueError("experiment manifest benchmark does not match requested result")
            record["experiment_name"] = identity["experiment_name"]
            record["setting_name"] = manifest["setting_name"]
            record["thinking"] = identity["config"].get("THINK_MODE", False)
            if record["benchmark"] == "geneval":
                record["prompt_variant"] = identity["config"].get("GENEVAL_PROMPT_MODE", "long")
            record["experiment_manifest"] = str(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["experiment_warning"] = f"实验身份未核验：{exc}"
        break
    return record


def attach_coverage(record, language="en"):
    """Attach the exact language's diagnostic; preserve score and legacy complete flag.

    The report distinguishes score availability from verified sample coverage.
    No coverage file means unknown completeness, not zero missing samples.
    """
    record = dict(record)
    if "coverage" in record:
        return record
    path = os.path.join(record.get("output_root", ""),
                        f"coverage_{record['benchmark']}_{language}.json")
    record["coverage"] = None
    record["score_available"] = record.get("headline") is not None
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                coverage = json.load(handle)
            if not isinstance(coverage, dict) or not isinstance(coverage.get("counts"), dict) or not isinstance(coverage.get("samples"), list):
                raise ValueError("invalid coverage report structure")
            if coverage.get("benchmark") != record["benchmark"] or coverage.get("language") != language:
                raise ValueError("coverage benchmark/language does not match requested result")
            record["coverage"] = coverage
            record["raw_files"] = list(record.get("raw_files", [])) + [path]
        except (OSError, ValueError, TypeError) as exc:
            record["coverage_warning"] = f"覆盖率文件不可用：{exc}"
    return record


def _main():
    ap = argparse.ArgumentParser(description="Extract a benchmark's headline score.")
    ap.add_argument("--bench", required=True, choices=sorted(_ADAPTERS))
    ap.add_argument("--results-root", default=None,
                    help="Directory the scorer wrote into. If omitted, derived from --output-base + knobs.")
    ap.add_argument("--output-base", default=None)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--step-name", default="step")
    ap.add_argument("--language", default="en", help="geditbench/oneigbench: which language to report")
    ap.add_argument("--split", default="testmini", help="tiifbench: which split's results dir")
    # sampling knobs (only used when deriving the root)
    ap.add_argument("--image-size", default="512")
    ap.add_argument("--cfg-scale", default="4.0")
    ap.add_argument("--cfg-norm", default="global")
    ap.add_argument("--txt-cfg", default="4.0")
    ap.add_argument("--img-cfg", default="1.0")
    ap.add_argument("--cfg-renorm", default="global")
    ap.add_argument("--timestep-shift", default="3.0")
    ap.add_argument("--mode", default="en")
    ap.add_argument("--output-subdir", default=None)
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--setting-name", default=None, help="Exact full setting name from the launcher/manifest")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--legacy-layout", action="store_true", help="Read an explicitly requested old directory layout")
    ap.add_argument("--out", default=None, help="write the JSON record here too")
    args = ap.parse_args()

    results_root = args.results_root
    if results_root is None:
        if args.output_base is None:
            ap.error("either --results-root or --output-base is required")
        if not args.legacy_layout and not (args.experiment_name and args.setting_name):
            ap.error("new layout requires --experiment-name and --setting-name (or explicit --results-root)")
        results_root = output_paths.derive_output_root(
            bench=args.bench, output_base=args.output_base,
            model_name=args.model_name, step_name=args.step_name,
            image_size=args.image_size, cfg_scale=args.cfg_scale, cfg_norm=args.cfg_norm,
            txt_cfg=args.txt_cfg, img_cfg=args.img_cfg, cfg_renorm=args.cfg_renorm,
            timestep_shift=args.timestep_shift, mode=args.mode, output_subdir=args.output_subdir,
            experiment_name=args.experiment_name, setting_name=args.setting_name, thinking=args.thinking,
        )

    record = collect(args.bench, results_root, model_name=args.model_name,
                     step_name=args.step_name, language=args.language, split=args.split)
    payload = json.dumps(record, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)


if __name__ == "__main__":
    _main()
