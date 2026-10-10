"""Run the official TextCrafter scorer with explicit local CLIP/head weights.

The external scorer remains responsible for OCR, VQA, preprocessing and scoring.
Only the two CLIP weight lookups and the aesthetic head lookup are redirected.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def run_local_scorer(scorer, clip_path: Path, head_path: Path, scorer_args: list[str]):
    import clip
    import open_clip.factory as factory
    import torch

    original_clip_load = clip.load
    original_download = factory.download_pretrained
    clip_config = factory.get_pretrained_cfg("ViT-L-14", "openai")
    if not clip_config:
        raise RuntimeError("OpenCLIP has no ViT-L-14/openai pretrained configuration")

    def local_clip_load(name, *args, **kwargs):
        if name == "ViT-L/14":
            name = str(clip_path)
        return original_clip_load(name, *args, **kwargs)

    def local_openclip_download(config, *args, **kwargs):
        # Keep the official 'openai' model/preprocessing configuration, including
        # its handling of OpenAI TorchScript .pt files. Other models use upstream.
        if config == clip_config:
            return str(clip_path)
        return original_download(config, *args, **kwargs)

    def load_local_aesthetic_head(self):
        model = torch.nn.Linear(768, 1)
        model.load_state_dict(torch.load(str(head_path), map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    print(f"CVTG CLIP weights: {clip_path}", flush=True)
    print(f"CVTG aesthetic head: {head_path}", flush=True)
    with patch.object(clip, "load", local_clip_load), \
            patch.object(factory, "download_pretrained", local_openclip_download), \
            patch.object(scorer.UnifiedMetricsEvaluator, "_load_aesthetic_model", load_local_aesthetic_head), \
            patch.object(sys, "argv", [scorer.__file__, *scorer_args]):
        scorer.main()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--textcrafter_root", required=True)
    parser.add_argument("--clip_model_path", required=True)
    parser.add_argument("--aesthetic_model_path", required=True)
    args, scorer_args = parser.parse_known_args(argv)
    # Validate before importing model libraries so missing local files cannot
    # silently fall back to online downloads or a zero aesthetic score.
    clip_path = require_file(args.clip_model_path, "CVTG CLIP weights")
    head_path = require_file(args.aesthetic_model_path, "CVTG aesthetic head")
    source = require_file(str(Path(args.textcrafter_root) / "unified_metrics_eval.py"), "TextCrafter scorer")
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("cvtg_official_metrics", source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    run_local_scorer(scorer, clip_path, head_path, scorer_args)


if __name__ == "__main__":
    main()
