"""Read-only coverage evidence for the nine active DG2 T2I benchmarks.

Rows enumerate metadata, never the set of surviving images. Packed DPG/OneIG
grids are one row per prompt; GenEval is one row per requested image. Missing
judgments are unknown, not API errors. No evaluator or scoring code is invoked.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
T2I = REPO / "third_party/UMM_Eval/gen/t2i"
ONEIG = {"Anime_Stylization": "anime", "Portrait": "human", "General_Object": "object",
         "Text_Rendering": "text", "Knowledge_Reasoning": "reasoning", "Multilingualism": "multilingualism"}
QCOLS = {"Quality": "quality_judge_output", "Aesthetics": "aesthetics_judge_output",
         "Alignment": "alignment_judge_output", "Creative Generation": "creative_generation_judge_output",
         "Real-world Fidelity": "real_world_fidelity_judge_output"}


def _read(path, warnings):
    path = Path(path)
    if not path.is_file():
        warnings.append(f"Evidence unavailable: {path}")
        return []
    try:
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() != ".jsonl":
            return json.loads(text)
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except (ValueError, TypeError) as exc:
                warnings.append(f"Malformed JSONL evidence {path}:{lineno}: {exc}")
        return records
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        warnings.append(f"Unreadable evidence {path}: {type(exc).__name__}: {exc}")
        return []


def _records(path, warnings):
    data = _read(path, warnings)
    if not isinstance(data, list):
        warnings.append(f"Expected a record list in {path}; coverage cannot be certified")
        return []
    records = []
    for i, rec in enumerate(data):
        if isinstance(rec, dict):
            records.append(rec)
        else:
            warnings.append(f"Malformed record {path}:{i + 1}; coverage cannot be certified")
    return records


def _number(value):
    try:
        return value is not None and str(value).strip() != "" and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _error_type(message):
    text = message.lower()
    if any(x in text for x in ("apierror", "httperror", "ratelimit", "429", "timeout", "timed out", "connection")):
        return "api_error"
    if any(x in text for x in ("parse", "json", "outputformat", "preds count", "missing_from_output")):
        return "parse_error"
    if any(x in text for x in ("load image", "cannot identify image", "image file", "no image found")):
        return "image_error"
    return "judge_error"


def _fail(row, message):
    row.update(judged=False, judge_error=str(message), error_type=_error_type(str(message)))


def _mark(row, rec, valid):
    error = rec.get("error") or rec.get("judge_error") or rec.get("exception")
    if error:
        _fail(row, error)
    elif valid:
        row.update(judged=True, judge_error=None, error_type=None)


def _qscore(data):
    if not isinstance(data, dict) or not data:
        return False
    if "score" in data:
        return data["score"] in (0, 1, 2, "0", "1", "2", "N/A")
    return all(_qscore(v) for v in data.values())


def _logs(root, judges, rows, warnings):
    # Only explicit evaluator error messages identify a sample failure.
    by_id = {r["sample_id"]: r for r in rows}
    by_name = {Path(r["image_path"]).name: r for r in rows}
    by_path = {r["image_path"]: r for r in rows}
    patterns = [r"\[ERR\]\s+(\d+):\s*(.*)", r"prompt_id=(\d+) failed:\s*(.*)",
                r"Failed to load image for ID=(\d+), path=(.*?):\s*(.*)",
                r"\[Failed\]\s+(.*?)\s*\|\s*(.*)", r"Failed filename:\s+(\S+)\s+(.*)"]
    candidates = set((root / "logs").rglob("*.log")) if (root / "logs").is_dir() else set()
    if judges.is_dir() and judges != root:
        candidates.update(judges.rglob("*.log"))
    for path in sorted(candidates):
        # Generation failures are not evaluator failures. Logs without a clear
        # evaluator filename/location cannot safely identify judge errors.
        label = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        if "gen_" in path.name.lower() or not re.search(r"judge|eval", label, re.I):
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                for pattern in patterns:
                    m = re.search(pattern, line)
                    if m:
                        key, message = m.group(1), m.group(m.lastindex)
                        row = by_id.get(key) or by_path.get(key) or by_name.get(Path(key).name)
                        if row and not row["judged"]:
                            _fail(row, message)
                        break
        except OSError as exc:
            warnings.append(f"Unreadable evaluator log {path}: {exc}")


def build_samples(bench, root: Path, **options):
    """Return (expected sample rows, honest evidence warnings), using stdlib only.

    metadata_file overrides the active wrapper default. image_dir is the direct
    benchmark image directory; judge_root is the direct per-benchmark judge dir.
    Scalar zero / False is valid scoring evidence. Missing/null evidence is not.
    """
    root = Path(root).resolve()
    warnings, rows = [], []
    lang = options.get("language", "en")
    split, model = options.get("split", "testmini"), options.get("model_name", "bagel")
    n = options.get("num_images", 4)
    try:
        n = int(n)
        if n < 1:
            raise ValueError("must be positive")
    except (TypeError, ValueError):
        return [], ["num_images must be a positive integer; coverage cannot be certified"]
    defaults = {
        "geneval": REPO / "eval/gen/geneval/prompts/evaluation_metadata_long.jsonl",
        "dpgbench": T2I / "dpgbench/eval/prompts.jsonl",
        "longtextbench": T2I / "longtextbench/eval" / ("text_prompts_zh.jsonl" if lang == "zh" else "text_prompts.jsonl"),
        "cvtg_2k": T2I / "cvtg_2k/eval/CVTG-2K",
        "tiifbench": T2I / "tiifbench/eval/TIIF-Bench/data" / f"{split}_prompts",
        "oneigbench": T2I / "oneigbench/eval/OneIG-Benchmark",
        "wise": REPO / "eval/gen/wise/final_data.json",
        "bizgeneval": T2I / "bizgeneval/eval/test.jsonl",
        "qwenimagebench": T2I / "qwenimagebench/prompts" / f"qwenimagebench_{lang}.jsonl",
    }
    if bench not in defaults:
        return [], [f"Unsupported T2I benchmark: {bench}"]
    meta = Path(options.get("metadata_file") or defaults[bench])
    direct_long = root.name == f"longtextbench_{lang}" or (root / "samples").is_dir()
    langroot = root if direct_long else root / f"longtextbench_{lang}"
    image_defaults = {"geneval": root / "gen_eval", "dpgbench": root / "dpgbench",
                      "longtextbench": langroot / "samples", "cvtg_2k": root / "CVTG",
                      "tiifbench": root / f"TIIFBench-{split}", "oneigbench": root / f"OneIGBench-{lang.upper()}",
                      "wise": root / "wise_raw", "bizgeneval": root / "bizgeneval",
                      "qwenimagebench": root / f"qwenimagebench_{lang}/samples"}
    images = Path(options.get("image_dir") or image_defaults[bench])
    judges = Path(options.get("judge_root") or root)
    index = {}

    def values(name, default=""):
        raw = options.get(name) or default
        return [str(x).strip() for x in (raw.split(",") if isinstance(raw, str) else raw) if str(x).strip()]

    try:
        target_indices = {int(x) for x in values("target_indices")}
    except ValueError:
        return [], ["Invalid target_indices; expected comma-separated integers"]
    target_ids, target_keys = set(values("target_prompt_ids")), set(values("target_keys"))
    if target_indices or target_ids or target_keys or options.get("tiifbench_specific_file"):
        warnings.append("Subset scope requested; this audit is not evidence of full benchmark coverage")
    id_benches = {"dpgbench", "longtextbench", "qwenimagebench", "bizgeneval"}
    if target_ids and bench not in id_benches:
        warnings.append(f"Active {bench} generator does not filter target_prompt_ids; preserving its expected scope")
    if target_keys and bench != "cvtg_2k":
        warnings.append(f"Active {bench} generator does not filter target_keys; preserving its expected scope")

    def selected(i, prompt_id=None, key=None):
        return not ((target_indices and i not in target_indices)
                    or (target_ids and bench in id_benches and str(prompt_id) not in target_ids)
                    or (target_keys and bench == "cvtg_2k" and str(key) not in target_keys))

    def add(sid, category, path, **extra):
        row = dict(sample_id=str(sid), category=str(category or "unknown"), image_path=str(path),
                   judged=False, judge_error=None, error_type=None, **extra)
        if row["sample_id"] in index:
            warnings.append(f"Duplicate expected sample ID: {row['sample_id']}")
        rows.append(row)
        index[row["sample_id"]] = row
        return row

    try:
        if bench == "cvtg_2k":
            subsets, areas = values("cvtg_subsets", "CVTG,CVTG-Style"), values("cvtg_areas", "2,3,4,5")
            if set(subsets) != {"CVTG", "CVTG-Style"} or set(areas) != {"2", "3", "4", "5"}:
                warnings.append("CVTG subset/area scope requested; this is not full benchmark coverage")
            files = [meta / subset / f"{area}_combined.json" for subset in subsets for area in areas] if meta.is_dir() else [meta]
            if not files:
                warnings.append(f"Expected CVTG metadata unavailable: {meta}")
            stable_index = 0
            for file in files:
                data = _read(file, warnings)
                if not isinstance(data, dict):
                    continue
                subset, area = file.parent.name, file.stem.split("_")[0]
                for key in data:
                    if selected(stable_index, key=key):
                        add(f"{subset}/{area}/{key}", f"{subset}/{area}", images / subset / area / f"{key}.png")
                    stable_index += 1
            warnings.append("CVTG: aggregate CVTG_results.json alone cannot establish per-sample judgment coverage; raw evaluator schema unavailable")
        elif bench == "tiifbench":
            files = sorted(meta.glob("*.jsonl")) if meta.is_dir() else [meta]
            if options.get("tiifbench_specific_file") and meta.is_dir():
                files = [meta / str(options["tiifbench_specific_file"])]
            stable_index = 0
            for file in files:
                for i, rec in enumerate(_records(file, warnings)):
                    keep = selected(stable_index)
                    stable_index += 1
                    if not keep:
                        continue
                    if "type" not in rec:
                        warnings.append(f"Missing TIIF type in {file}:{i + 1}")
                        continue
                    kind = str(rec["type"])
                    for desc in ("short_description", "long_description"):
                        sid = f"{kind}/{i}/{desc}"
                        row = add(sid, kind, images / kind / model / desc / f"{i}.png")
                        jbase = judges if options.get("judge_root") else root / f"tiifbench-{split}_results/eval_json"
                        path = jbase / model / kind / ("short" if desc.startswith("short") else "long") / f"{i}.json"
                        if path.is_file():
                            result = _read(path, warnings)
                            if isinstance(result, dict):
                                preds = result.get("model_pred")
                                _mark(row, result, isinstance(preds, list) and bool(preds) and len(preds) == len(result.get("questions", [])) and all(x in ("yes", "no") for x in preds))
                            else:
                                _fail(row, f"Malformed TIIF judgment JSON: {path}")
        else:
            if bench == "oneigbench" and meta.is_dir():
                meta /= "OneIG-Bench-ZH.csv" if lang == "zh" else "OneIG-Bench.csv"
            for i, rec in enumerate(_records(meta, warnings)):
                if not isinstance(rec, dict):
                    warnings.append(f"Malformed metadata record {meta}:{i + 1}")
                    continue
                keys = {"dpgbench": ("id",), "longtextbench": ("prompt_id",), "oneigbench": ("id", "category"),
                        "wise": ("prompt_id",), "qwenimagebench": ("ID",)}.get(bench, ())
                if any(key not in rec for key in keys) or (bench == "oneigbench" and rec.get("category") not in ONEIG):
                    warnings.append(f"Missing/unknown metadata key in {meta}:{i + 1}; coverage cannot be certified")
                    continue
                prompt_id = rec.get("id", rec.get("ID", rec.get("prompt_id", i)))
                if not selected(i, i if bench == "bizgeneval" else prompt_id):
                    continue
                if bench == "geneval":
                    for j in range(n):
                        add(f"{i:05d}/{j:05d}", rec.get("tag"), images / f"{i:05d}/samples/{j:05d}.png")
                elif bench == "dpgbench":
                    add(rec["id"], "all", images / f"{rec['id']}.png")
                elif bench == "longtextbench":
                    for repeat_id in range(1, n + 1):
                        sid = f"{int(rec['prompt_id'])}_{repeat_id}"
                        path = images / f"{sid}.png"
                        legacy = images / f"{int(rec['prompt_id'])}_sample.png"
                        if n == 1 and not path.is_file() and legacy.is_file():
                            path = legacy
                        add(sid, rec.get("category"), path)
                elif bench == "oneigbench":
                    cat = ONEIG[rec["category"]]
                    add(f"{cat}_{rec['id']}", cat, images / cat / model / f"{rec['id']}.webp", dataset_id=str(rec["id"]))
                elif bench == "wise":
                    add(rec["prompt_id"], rec.get("Category", rec.get("Subcategory")), images / f"{rec['prompt_id']}.png")
                elif bench == "bizgeneval":
                    add(i, rec.get("dimension", rec.get("eval_tag")), images / f"{i:04d}.png")
                elif bench == "qwenimagebench":
                    add(rec["ID"], rec.get("category"), images / f"{rec['ID']}.png")
        _judgments(bench, root, judges, images, langroot, lang, model, index, warnings)
        _logs(root, judges, rows, warnings)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        warnings.append(f"Coverage evidence/schema incomplete for {bench}: {type(exc).__name__}: {exc}")
    if not rows:
        warnings.append("No expected samples could be enumerated; full dataset coverage is unknown")
    return rows, list(dict.fromkeys(warnings))


def _judgments(bench, root, judges, images, langroot, lang, model, index, warnings):
    if bench == "geneval":
        for rec in _records(judges / "results_geneval.jsonl", warnings):
            path = Path(rec.get("filename", ""))
            sid = f"{path.parent.parent.name}/{path.stem}"
            if sid in index:
                _mark(index[sid], rec, isinstance(rec.get("correct"), bool))
    elif bench == "dpgbench":
        path = judges / "dpgbench_results.txt"
        if not path.is_file():
            warnings.append(f"Per-prompt DPG judgments unavailable: {path}")
            return
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^(.*?\.png),\s*(.*)$", line)
            if match:
                sid = Path(match.group(1)).stem
                values = match.group(2).split(",")
                if sid in index and values and all(_number(x) for x in values):
                    index[sid]["judged"] = True
    elif bench == "longtextbench":
        folder = judges if judges != root else langroot / "eval"
        for rec in _records(folder / "results.jsonl", warnings):
            match = re.fullmatch(r"(\d+)_(\d+)\.png", Path(str(rec.get("image", ""))).name)
            if match:
                sid = f"{int(match[1])}_{int(match[2])}"
                if sid in index and str(rec.get("prompt_id", "")) in (match[1], str(int(match[1]))):
                    _mark(index[sid], rec, _number(rec.get("text_accuray")))
            else:
                # Legacy single-image results must never certify four repeats.
                sid = f"{rec.get('prompt_id', '')}_1"
                row = index.get(sid)
                if row and Path(row["image_path"]).name.endswith("_sample.png"):
                    _mark(row, rec, _number(rec.get("text_accuray")))
    elif bench == "wise":
        paths = sorted(judges.glob(f"*_scores_results_{images.name}.jsonl"))
        if not paths:
            warnings.append(f"WISE per-prompt judgments unavailable: {judges}")
        for path in paths:
            for rec in _records(path, warnings):
                sid = str(rec.get("prompt_id", ""))
                if sid in index:
                    _mark(index[sid], rec, all(_number(rec.get(x)) for x in ("consistency", "realism", "aesthetic_quality")))
    elif bench == "bizgeneval":
        folder = judges if judges != root else root / "bizgeneval_eval"
        paths = sorted((folder / "items").glob("*.json"))
        if not paths:
            warnings.append(f"BizGenEval per-item judgments unavailable: {folder / 'items'}")
        for path in paths:
            rec = _read(path, warnings)
            file_id = path.stem.lstrip("0") or "0"
            sid = str(rec.get("prompt_id", file_id)) if isinstance(rec, dict) else file_id
            if sid in index and isinstance(rec, dict):
                _mark(index[sid], rec, _number(rec.get("accuracy")))
                meta = rec.get("meta_info", {})
                if isinstance(meta, dict) and any(isinstance(v, dict) and v.get("reason") == "missing_from_output" for v in meta.values()):
                    _fail(index[sid], "Judge response missing checklist answers (missing_from_output)")
            elif sid in index:
                _fail(index[sid], f"Malformed BizGenEval judgment JSON: {path}")
    elif bench == "qwenimagebench":
        folder = judges if judges != root else root / f"qwenimagebench_{lang}/eval"
        if folder.name == "eval_api":
            warnings.append(f"Qwen API per-row schema unavailable in bundled source; judgment coverage is unknown: {folder}")
            return
        expected = {}
        for meta in _records(T2I / "qwenimagebench/eval/metadata/bench_metadata.json", warnings):
            if isinstance(meta, dict):
                expected[str(meta.get("ID"))] = {x.split("/")[0].strip() for x in str(meta.get("dims_en", "")).split(";") if x.strip()}
        files = [folder / "judge_input_judged.jsonl"]
        if not files[0].is_file():
            files = sorted(folder.glob("judge_input_judged.shard*-of-*.jsonl"))
        if not files:
            warnings.append(f"Qwen per-row judgments unavailable: {folder}")
        for path in files:
            for rec in _records(path, warnings):
                sid = str(rec.get("ID", ""))
                if sid not in index:
                    continue
                raw = rec.get("judge_model_output")
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    _fail(index[sid], "Malformed judge_model_output JSON")
                    continue
                dims = expected.get(sid, set())
                valid = isinstance(parsed, dict) and bool(parsed) and all(_qscore(v) for v in parsed.values()) and (not dims or dims <= set(parsed))
                _mark(index[sid], rec, valid)
                failed = [dim for dim, col in QCOLS.items() if rec.get(col) and (not isinstance(parsed, dict) or dim not in parsed)]
                if failed:
                    _fail(index[sid], "Unparsed judge response dimensions: " + ", ".join(failed))
    elif bench == "oneigbench":
        folder = judges if judges != root else root / f"oneigbench-{lang.upper()}_results"
        evidence = {}
        for metric in ("alignment", "diversity", "reasoning", "style", "text"):
            paths = sorted(folder.glob(f"{metric}_prompt_score_{lang.upper()}*.csv"))
            if not paths:
                warnings.append(f"OneIG {metric} per-prompt CSV unavailable: {folder}")
                continue
            evidence[metric] = {str(next(iter(r.values()), "")): r.get(model) for r in _records(paths[-1], warnings)}
        required = {"anime": ("alignment", "diversity", "style"), "human": ("alignment", "diversity"),
                    "object": ("alignment", "diversity"), "multilingualism": ("alignment", "diversity"),
                    "text": ("diversity", "text"), "reasoning": ("diversity", "reasoning")}
        for sid, row in index.items():
            valid = []
            for metric in required[row["category"]]:
                key = row["dataset_id"] if metric in ("text", "style", "reasoning") else sid
                value = evidence.get(metric, {}).get(key)
                if metric == "text" and value:
                    try:
                        value = ast.literal_eval(re.sub(r"np\.float64\(([^)]+)\)", r"\1", value))
                    except (ValueError, SyntaxError):
                        _fail(row, "Malformed OneIG per-prompt text metric")
                        valid.append(False)
                        continue
                    valid.append(isinstance(value, (list, tuple)) and len(value) == 3 and all(_number(x) for x in value))
                else:
                    valid.append(_number(value))
            row["judged"] = bool(valid) and all(valid) and row["judge_error"] is None
