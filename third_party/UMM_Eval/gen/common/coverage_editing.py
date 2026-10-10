"""Read editing manifests and scorer evidence without changing official scores.

Rows describe expected samples. Filesystem validation and status precedence belong
to the coverage reporter. Missing score evidence is unscored, not an API failure.
Only JSON/JSONL/CSV and XLSX data are read; pickle caches are never deserialized.
"""
from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


_UMM = Path(__file__).resolve().parents[2]
_DG2 = Path(__file__).resolve().parents[4]
_METRICS = {"IC": "Identity_Consistency", "IF": "Instrcution_Following",
            "VQ": "Visual_Quality", "URC": "Unedited_Region_Consistency",
            "WA": "World_Awareness"}


def _read(path, warnings):
    path = Path(path)
    if not path.is_file():
        warnings.append(f"Expected metadata/evidence file not found: {path}")
        return None
    try:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                return list(csv.DictReader(stream))
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        warnings.append(f"Cannot read {path}: {exc}")
        return None


def _optional(path, warnings):
    return _read(path, warnings) if Path(path).is_file() else None


def _records(data):
    if isinstance(data, list):
        return [(str(i), item) for i, item in enumerate(data) if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("samples", "expected_samples", "records"):
            if isinstance(data.get(key), list):
                return _records(data[key])
        return [(str(key), item) for key, item in data.items() if isinstance(item, dict)]
    return []


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _error(value):
    if not isinstance(value, dict):
        return None
    for field in ("judge_error", "error", "error_message"):
        error = value.get(field)
        if error:
            return str(error)
    for field, status in value.items():
        if (field == "judge_status" or field.startswith("eval_") and field.endswith("_status")) and str(status).lower() in {
                "error", "failed", "failure", "parse_failed"}:
            return f"{field}={status}"
    return None


def _row(sample_id, category, image_path, *, judged=False, error=None,
         error_type=None, evidence=None, score=None):
    result = {"sample_id": str(sample_id), "category": str(category),
              "image_path": str(image_path), "judged": judged,
              "judge_error": error, "error_type": error_type}
    if evidence:
        result["evidence"] = str(evidence)
    if score is not None:
        result["score"] = score
    return result


def _metadata(options, candidates, warnings):
    explicit = options.get("metadata_file")
    if explicit is not None:
        return Path(explicit)
    for path in candidates:
        if Path(path).is_file():
            return Path(path)
    warnings.append("Expected sample metadata is unavailable; completeness cannot be determined.")
    return None


def _imgedit(root, options, warnings):
    metadata = _metadata(options, [root / "expected_samples.json",
        _UMM / "gen/it2i/imgedit/eval/Benchmark/singleturn/singleturn.json"], warnings)
    if metadata is None:
        return []
    expected = _read(metadata, warnings)
    images = Path(options.get("image_dir") or root / "bagel")
    judge = Path(options.get("judge_root") or root)
    raw_path, avg_path = judge / "imgedit_result.json", judge / "imgedit_average_score.json"
    raw = _optional(raw_path, warnings) or {}
    average = _optional(avg_path, warnings) or {}
    rows = []
    for key, item in _records(expected):
        sample_id = str(item.get("sample_id", key))
        value = raw.get(sample_id) if isinstance(raw, dict) else None
        score = _number(average.get(sample_id)) if isinstance(average, dict) else None
        error = _error(value)
        judged = score is not None
        # Scored zero/refusal remains a judged sample under the existing protocol.
        if not judged and not error and isinstance(value, str):
            judged = bool(re.findall(r":\s*\**\s*(\d+)", value))
            if not judged:
                error = "Captured judge response has no score accepted by the ImgEdit parser."
        if not judged and not error and isinstance(value, dict):
            judged = _number(value.get("score")) is not None
        rows.append(_row(sample_id, item.get("edit_type", item.get("category", "unknown")),
            images / f"{sample_id}.png", judged=judged, error=error,
            error_type="judge_error" if error else None,
            evidence=avg_path if score is not None else raw_path if value is not None else None,
            score=score))
    return rows


def _geditbench(root, options, warnings):
    metadata = _metadata(options, [root / "expected_samples.json", root / "expected_samples.jsonl"], warnings)
    if metadata is None:
        return []
    if metadata.is_dir():
        candidates = [metadata / "expected_samples.json", metadata / "expected_samples.jsonl",
                      root / "expected_samples.json", root / "expected_samples.jsonl"]
        metadata = next((p for p in candidates if p.is_file()), None)
        if metadata is None:
            warnings.append("GEdit expected metadata is Arrow-only or absent. Export key, task_type and "
                            "instruction_language to expected_samples.json; completeness is unknown.")
            return []
    expected = _read(metadata, warnings)
    language = options.get("language", "en")
    images = Path(options.get("image_dir") or root / "fullset")
    judge = Path(options.get("judge_root") or root)
    model_name = options.get("model_name", "bagel")
    evidence = {}
    candidates = sorted(judge.rglob(f"{model_name}_*_vie_score*.csv"))
    if len({path.parent for path in candidates}) > 1:
        warnings.append("Multiple GEdit scorer directories were found. Set judge_root to the selected "
                        "model/backbone directory; judge coverage is not combined across scorers.")
        candidates = []
    # Prefer merged files to stale shard records; never blend different model names.
    candidates.sort(key=lambda p: "_part" not in p.stem)
    for path in candidates:
        prefix = path.stem.split("_vie_score", 1)[0][len(model_name) + 1:]
        group = prefix.rsplit("_", 1)[0]
        for _, item in _records(_read(path, warnings)):
            item_lang = item.get("instruction_language", language)
            key = (str(item.get("key", "")), item_lang, item.get("task_type", group))
            evidence[key] = (item, path)
    rows = []
    for key, item in _records(expected):
        item_lang = item.get("instruction_language")
        if item_lang is None:
            warnings.append("GEdit metadata lacks instruction_language; expected denominator is ambiguous.")
            continue
        if language != "all" and item_lang != language:
            continue
        sample_id = str(item.get("key", item.get("sample_id", key)))
        category = item.get("task_type", item.get("category", "unknown"))
        result, path = evidence.get((sample_id, item_lang, category), ({}, None))
        semantics = _number(result.get("sementics_score", result.get("semantics_score")))
        quality = _number(result.get("quality_score"))
        error = _error(result)
        judged = semantics is not None and quality is not None
        if result and not judged and not error:
            error = "Recorded GEdit judge row is missing finite semantics/quality scores."
        rows.append(_row(f"{item_lang}/{category}/{sample_id}", category,
            images / category / item_lang / f"{sample_id}.png", judged=judged,
            error=error, error_type="judge_error" if error else None, evidence=path,
            score={"semantics": semantics, "quality": quality} if judged else None))
    return rows


def _xlsx(path, warnings):
    """Read the first tabular worksheet produced by RISE's pandas.to_excel."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with zipfile.ZipFile(path) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                strings = ["".join(node.itertext()) for node in shared.findall("s:si", ns)]
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            table = []
            for row in sheet.findall(".//s:sheetData/s:row", ns):
                cells = {}
                for cell in row.findall("s:c", ns):
                    column = re.match(r"[A-Z]+", cell.attrib.get("r", "A1")).group()
                    value = cell.findtext("s:v", default="", namespaces=ns)
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(cell.find("s:is", ns).itertext())
                    cells[column] = value
                table.append(cells)
            if not table:
                return []
            headers = table[0]
            return [{name: row.get(column, "") for column, name in headers.items()} for row in table[1:]]
    except (OSError, ValueError, KeyError, IndexError, AttributeError, zipfile.BadZipFile, ET.ParseError) as exc:
        warnings.append(f"Cannot read RISE judge workbook {path}: {exc}")
        return []


def _rise(root, options, warnings):
    metadata = _metadata(options, [root / "expected_samples.json",
        _UMM / "gen/it2i/rise/eval/data/datav2_total_w_subtask.json"], warnings)
    if metadata is None:
        return []
    expected = _read(metadata, warnings)
    images = Path(options.get("image_dir") or root)
    judge = Path(options.get("judge_root") or root)
    evidence = {}
    for path in sorted(judge.glob("*_judge.xlsx")) + sorted(judge.glob("*_judge.csv")):
        data = _xlsx(path, warnings) if path.suffix == ".xlsx" else _read(path, warnings)
        for _, item in _records(data):
            if "index" in item:  # *_judge.csv normally contains aggregates, not samples.
                evidence[str(item["index"])] = (item, path)
    if not evidence and any(judge.glob("*.pkl")):
        warnings.append("RISE has only a pickle judge cache; it is not deserialized. "
                        "Sample judge coverage remains unscored until the judge workbook is written.")
    rows = []
    for key, item in _records(expected):
        sample_id = str(item.get("index", item.get("sample_id", key)))
        category = item.get("category", "unknown")
        result, path = evidence.get(sample_id, ({}, None))
        score = _number(result.get("score"))
        error = _error(result)
        if result.get("match_log") == "failed":
            error = error or "RISE recorded match_log=failed (judge score parsing failed)."
        elif result and score is None and not error:
            error = "Recorded RISE judge row has no finite aggregate sample score."
        rows.append(_row(sample_id, category, images / category / f"{sample_id}.png",
            judged=score is not None, error=error, error_type="judge_error" if error else None,
            evidence=path, score=score))
    return rows


def _language_root(root, language, model_name):
    if root.name == language:
        return root
    if (root / model_name / language).is_dir():
        return root / model_name / language
    return root / language


def _editcompass(root, options, warnings):
    source = Path(options.get("metadata_file") or _DG2 / "third_party/Edit-Compass-data")
    language = options.get("language", "en")
    images = Path(options.get("image_dir") or _language_root(root, language, options.get("model_name", "bagel")))
    judge = _language_root(Path(options.get("judge_root") or root), language, options.get("model_name", "bagel"))
    parts = options.get("parts") or ["Part1", "Part2"]
    if isinstance(parts, str):
        parts = parts.replace(",", " ").split()
    configs = []
    if source.is_file():
        configs = [(str(item.get("part", "Part1")), str(item.get("task_name", item.get("category", "unknown"))), item)
                   for _, item in _records(_read(source, warnings))]
    else:
        for part in parts:
            found = False
            for path in sorted((source / part).glob("*/*.json")):
                if path.stem != path.parent.name:
                    continue
                found = True
                configs.extend((part, path.parent.name, item) for _, item in _records(_read(path, warnings)))
            if not found:
                warnings.append(f"EditCompass expected metadata is absent for {source / part}.")
    evidence = {}
    for part, task, _ in configs:
        for metric, field in _METRICS.items():
            key = (part, task, metric)
            if key in evidence:
                continue
            path = judge / part / task / f"{task}_{metric}.json"
            evidence[key] = {str(item.get("save_id")): (item, path)
                             for _, item in _records(_optional(path, warnings))}
    rows = []
    for part, task, item in configs:
        if part not in parts:
            continue
        sample_id = str(item.get("save_id", item.get("sample_id", "")))
        required = [metric for metric in _METRICS if item.get(metric) is True]
        complete, failures, paths, scores = bool(required), [], [], {}
        for metric in required:
            result, path = evidence[(part, task, metric)].get(sample_id, ({}, None))
            score_obj = result.get(_METRICS[metric], {})
            score = _number(score_obj.get("score")) if isinstance(score_obj, dict) else None
            if path:
                paths.append(str(path))
            error = _error(result)
            if result and score is None and not error:
                error = f"Recorded {metric} judge result has no finite score."
            if error:
                failures.append(f"{metric}: {error}")
            complete = complete and score is not None
            if score is not None:
                scores[metric] = score
        if not required:
            warnings.append(f"EditCompass {part}/{task}/{sample_id} has no required metric flags; judge coverage is unknown.")
        rows.append(_row(f"{part}/{task}/{sample_id}", f"{part}/{task}",
            images / part / task / f"{sample_id}.png", judged=complete,
            error="; ".join(failures) or None, error_type="judge_error" if failures else None,
            evidence="; ".join(paths) or None, score=scores if scores else None))
    return rows


def build_samples(bench, root: Path, **options):
    """Return (expected_sample_rows, warnings) for one editing benchmark."""
    warnings = []
    adapters = {"imgedit": _imgedit, "geditbench": _geditbench, "gedit": _geditbench,
                "rise": _rise, "editcompass": _editcompass, "edit_compass": _editcompass}
    if bench not in adapters:
        return [], [f"Unsupported editing benchmark: {bench}"]
    rows = adapters[bench](Path(root), options, warnings)
    if not rows:
        warnings.append("No expected samples were loaded; completeness is unknown, not 100%.")
    return rows, list(dict.fromkeys(warnings))
