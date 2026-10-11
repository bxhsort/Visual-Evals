"""Resolve OneIG judges from the downloaded checkpoint directories."""

import os
from pathlib import Path


DEFAULT_CKPT_ROOT = (
    "/inspire/hdd3/project/agent-driven-world-model/"
    "wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt"
)
DEFAULT_QWEN_MODEL_PATH = (
    "/inspire/hdd3/project/agent-driven-world-model/"
    "public/models/public_models/Qwen/Qwen2.5-VL-7B-Instruct"
)


def checkpoint_root():
    return Path(os.environ.get("ONEIG_CKPT_ROOT") or DEFAULT_CKPT_ROOT).expanduser()


def local_model_path(relative_path, explicit_path=None):
    path = Path(explicit_path or checkpoint_root() / relative_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            "OneIG local model not found: {}. Check the downloaded files or "
            "set ONEIG_CKPT_ROOT.".format(path)
        )
    return str(path)


def qwen_model_path(explicit_path=None):
    path = (
        explicit_path
        or os.environ.get("QWEN_VL_MODEL_PATH")
        or os.environ.get("QWEN_OCR_MODEL_PATH")
        or DEFAULT_QWEN_MODEL_PATH
    )
    return local_model_path("", path)


def csd_model_path(explicit_path=None):
    if explicit_path:
        return local_model_path("", explicit_path)
    folder = checkpoint_root() / "CSD"
    for filename in ("checkpoint.pth", "pytorch_model.bin", "csd-vit-l.pth"):
        path = folder / filename
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "OneIG CSD checkpoint not found under {}. Expected checkpoint.pth, "
        "pytorch_model.bin or csd-vit-l.pth.".format(folder)
    )


def dreamsim_model_path():
    folder = Path(local_model_path("dreamsim"))
    required_files = (
        "clip_vitb16_pretrain.pth.tar",
        "dino_vitb16_pretrain.pth",
        "open_clip_vitb16_pretrain.pth.tar",
    )
    missing = [name for name in required_files if not (folder / name).is_file()]
    if not (folder / "ensemble_lora").is_dir():
        missing.append("ensemble_lora/")
    if missing:
        raise FileNotFoundError(
            "OneIG DreamSim weights missing under {}: {}".format(
                folder, ", ".join(missing)
            )
        )
    return str(folder)
