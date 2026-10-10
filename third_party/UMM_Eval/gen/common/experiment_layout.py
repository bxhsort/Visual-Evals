"""Three-level experiment layout and local cache identity; no model or API calls."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata

MANIFEST = "experiment_manifest.json"
EXECUTION_KEYS = {"RUN_GENERATION", "RUN_EVAL", "RUN_SUMMARY", "FINALIZE_EVAL", "OVERWRITE",
                  "MAX_MEMORY", "OFFLOAD_FOLDER", "MASTER_PORT", "EVAL_NUM_WORKERS", "EVAL_NUM_PROCESSES"}
SECRET_KEYS = {"API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "PASSWORD", "ACCESS_TOKEN", "SECRET"}
INPUT_KEYS = {"METADATA_FILE", "GEDIT_DATASET_PATH", "DATA_ROOT", "RISE_DATA_JSON", "PROMPTS_JSON",
              "IMAGE_ROOT", "RISE_INPUT_ROOT", "TIIFBENCH_PROMPT_DIR", "TIIFBENCH_EVAL_PROMPT_DIR",
              "QIB_METADATA_FILE_EN", "QIB_METADATA_FILE_ZH", "LONGTEXTBENCH_METADATA_FILE_EN",
              "LONGTEXTBENCH_METADATA_FILE_ZH", "GENERATION_IMAGE_ROOT", "BENCHMARK_ROOT",
              "WISE_DATA_ROOT", "INPUT_JSON_EN", "INPUT_JSON_CN"}
METADATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".arrow", ".parquet"}
MODEL_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".json"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
GENERATION_KEYS = {"MODEL_PATH", "MODEL_NAME", "STEP_NAME", "IMAGE_SIZE", "CFG_SCALE", "CFG_NORM",
                   "NUM_STEPS", "NUM_TIMESTEPS", "NUM_IMAGES", "BATCH_SIZE", "MAX_LATENT_SIZE",
                   "TIMESTEP_SHIFT", "TXT_CFG", "IMG_CFG", "CFG_RENORM", "CFG_RENORM_TYPE", "CFG_TEXT_SCALE", "CFG_IMG_SCALE",
                   "CFG_TYPE", "CFG_INTERVAL", "MAX_IMAGE_SIZE", "MIN_IMAGE_SIZE", "SEED", "TEMPERATURE",
                   "GENERATION_SHARDS", "CROP_BBOX", "CROP_BBOX_WH", "CROP_BBOX_SIZE", "CROP_BBOX_MIN_SIDE",
                   "BIZGENEVAL_DEFAULT_ASPECT_RATIO", "GENERATOR", "SEED_STRATEGY", "MAX_LENGTH"}
GENERATION_ALIASES = {"NUM_TIMESTEPS": "NUM_STEPS", "TXT_CFG": "CFG_TEXT_SCALE",
                      "IMG_CFG": "CFG_IMG_SCALE", "MAX_IMAGE_SIZE": "IMAGE_SIZE",
                      "CFG_RENORM_TYPE": "CFG_RENORM"}
NUMERIC_GENERATION_KEYS = {"IMAGE_SIZE", "CFG_SCALE", "NUM_STEPS", "NUM_IMAGES",
                           "BATCH_SIZE", "MAX_LATENT_SIZE", "TIMESTEP_SHIFT", "CFG_TEXT_SCALE",
                           "CFG_IMG_SCALE", "MIN_IMAGE_SIZE", "SEED", "TEMPERATURE",
                           "GENERATION_SHARDS", "CROP_BBOX_SIZE", "CROP_BBOX_MIN_SIDE", "MAX_LENGTH"}


def _component(value, name):
    value = str(value or "").strip()
    if not value or value in {".", ".."} or any(c in value for c in ("/", "\\")) or any(
        unicodedata.category(c) == "Cc" for c in value
    ):
        raise ValueError(f"{name} must be a nonempty single folder name (no slash or control characters)")
    if len(value.encode("utf-8")) > 240:
        raise ValueError(f"{name} is too long for a folder name")
    return value


def _thinking(value):
    if isinstance(value, bool):
        return value
    value = str(value or "0").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("THINK_MODE must be 0/1, false/true, no/yes, or off/on")


def derive_experiment_root(output_base, experiment_name, setting, benchmark, thinking=False):
    """Return OUTPUT_BASE / user experiment / setting / benchmark[_thinking]."""
    experiment_name = _component(experiment_name, "EXPERIMENT_NAME")
    setting = _component(setting, "setting")
    benchmark = _component(benchmark, "benchmark")
    if _thinking(thinking):
        benchmark += "_thinking"
    return str(Path(output_base).expanduser().resolve() / experiment_name / setting / benchmark)


def _clean_config(config):
    clean = {}
    for key, value in config.items():
        key = str(key).upper()
        if key in EXECUTION_KEYS or key in SECRET_KEYS or key.endswith("_API_KEY"):
            continue
        if value is None or value == "":
            continue
        clean[key] = value
    clean["THINK_MODE"] = _thinking(clean.get("THINK_MODE", False))
    return clean


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Generation parameters must be finite numbers")
    return int(number) if number.is_integer() else number


def _generation_config(config):
    """Match actual settings across wrappers with different parameter spellings."""
    canonical = {}
    for key, value in config.items():
        if key not in GENERATION_KEYS:
            continue
        key = GENERATION_ALIASES.get(key, key)
        if key in NUMERIC_GENERATION_KEYS:
            value = _number(value)
        elif key == "CFG_INTERVAL":
            if isinstance(value, str):
                value = json.loads(value)
            value = [_number(item) for item in value]
        elif key == "MODEL_PATH":
            value = str(Path(str(value)).expanduser().resolve())
        if key in canonical and canonical[key] != value:
            raise ValueError(f"Conflicting generation parameter aliases for {key}")
        canonical[key] = value
    return canonical


def _file_identity(path):
    stat = path.stat()
    result = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # Hash small prompt/config files; large weights and Arrow files use local file identity.
    if path.suffix.lower() not in ({".arrow", ".parquet", ".safetensors", ".bin", ".pt", ".pth"} | IMAGE_EXTENSIONS):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
        result.pop("mtime_ns")  # copying unchanged metadata does not change its identity
    return result


def _path_identity(value, model=False):
    path = Path(str(value)).expanduser().resolve()
    result = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        result["file"] = _file_identity(path)
    elif path.is_dir():
        extensions = MODEL_EXTENSIONS if model else METADATA_EXTENSIONS | IMAGE_EXTENSIONS
        records = {}
        for directory, subdirs, files in os.walk(path, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if d not in {".git", "__pycache__", "outputs", ".venv"})
            for name in sorted(files):
                item = Path(directory) / name
                if item.suffix.lower() in extensions and item.is_file():
                    records[str(item.relative_to(path))] = _file_identity(item)
        result["files"] = records
    return result


def ensure_manifest(results_root, manifest):
    """Allow matching resumes; refuse unidentified or incompatible existing artifacts."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / MANIFEST
    with (root / ".experiment_manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            try:
                previous = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable cache manifest: {target}") from exc
            if not isinstance(previous, dict) or previous.get("identity") != manifest.get("identity"):
                raise ValueError(f"Existing results have different data, scope, or judge configuration: {root}. Use a new EXPERIMENT_NAME; existing images/scores were not reused.")
            return
        if any(item.name != ".experiment_manifest.lock" for item in root.iterdir()):
            raise ValueError(f"Existing results have no experiment manifest: {root}. Use a new EXPERIMENT_NAME; old files were not reused.")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False,
                                         prefix=".experiment_manifest.") as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_experiment(benchmark, output_base, experiment_name, setting_label, config):
    config = _clean_config(config)
    generation_config = _generation_config(config)
    benchmark = _component(benchmark, "benchmark")
    experiment_name = _component(experiment_name, "EXPERIMENT_NAME")
    label = _component(setting_label, "setting label")
    if benchmark == "geneval":
        variant = str(config.get("GENEVAL_PROMPT_MODE", "long"))
        if variant not in {"short", "long"}:
            raise ValueError("A single GenEval run requires GENEVAL_PROMPT_MODE=short or long")
        config["GENEVAL_PROMPT_MODE"] = variant
        if f"prompt_{variant}" not in label:
            label += f"_prompt_{variant}"
    steps = generation_config.get("NUM_STEPS")
    if steps is not None:
        label += f"_steps{steps}"
    if generation_config.get("NUM_IMAGES") is not None:
        label += f"_n{generation_config['NUM_IMAGES']}"
    identities = {key: _path_identity(value) for key, value in config.items() if key in INPUT_KEYS}
    if config.get("MODEL_PATH"):
        identities["MODEL_PATH"] = _path_identity(config["MODEL_PATH"], model=True)
    for key in ("QJUDGER_MODEL_PATH", "CVTG_CLIP_MODEL_PATH", "CVTG_AESTHETIC_MODEL_PATH", "CVTG_VQA_MODEL_PATH", "CVTG_VQA_VISION_MODEL_PATH"):
        if config.get(key):
            identities[key] = _path_identity(config[key], model=True)
    identity = {"schema_version": 1, "benchmark": benchmark, "experiment_name": experiment_name,
                "config": config, "inputs": identities}
    # Different benchmarks share a setting folder when generation settings match.
    # Their metadata, subset and judge identities remain strict in each leaf's manifest.
    digest_identity = {"schema_version": 1,
                       "model": identities.get("MODEL_PATH"),
                       "config": generation_config}
    fingerprint = hashlib.sha256(json.dumps(digest_identity, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    setting_name = _component(f"{label}__{fingerprint[:12]}", "setting")
    root = derive_experiment_root(output_base, experiment_name, setting_name, benchmark, config["THINK_MODE"])
    manifest = {"schema_version": 1, "results_root": root, "setting_name": setting_name,
                "fingerprint": fingerprint, "config": config, "identity": identity}
    ensure_manifest(root, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--setting-label", required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    try:
        config = json.loads(args.config_json)
        if not isinstance(config, dict):
            raise ValueError("DG2_EFFECTIVE_CONFIG_JSON must contain a JSON object")
        for entry in args.config:
            key, value = entry.split("=", 1)
            config[key] = value
        manifest = prepare_experiment(args.bench, args.output_base, args.experiment_name, args.setting_label, config)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"[experiment] {exc}\n")
    print(manifest["results_root"])
    print(manifest["setting_name"])


if __name__ == "__main__":
    main()
