"""Run the official TextCrafter scorer with explicit local model weights.

The external scorer remains responsible for OCR, VQA, preprocessing and scoring.
Local weights are routed without changing metrics; PaddleOCR runs on CPU.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_vqa_model(value: str, label: str = "CVTG VQA") -> Path:
    root = Path(value).expanduser().resolve()
    require_file(str(root / "config.json"), f"{label} config")
    # Check every indexed shard before importing model libraries. A partial
    # download must not trigger a fallback to the remote model repository.
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / filename
        if index.is_file():
            weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"{label} index has no weight_map: {index}")
            for shard in set(weights.values()):
                require_file(str(root / shard), f"{label} weight shard")
            return root
    if not any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"{label} weights/index do not exist: {root}")
    return root


def require_ocr_models(value: str) -> dict:
    root = Path(value).expanduser().resolve()
    directories = {
        "det_model_dir": root / "whl/det/en/en_PP-OCRv3_det_infer",
        "rec_model_dir": root / "whl/rec/en/en_PP-OCRv4_rec_infer",
        "cls_model_dir": root / "whl/cls/ch/ch_ppocr_mobile_v2.0_cls_infer",
    }
    for name, directory in directories.items():
        for filename in ("inference.pdmodel", "inference.pdiparams"):
            require_file(str(directory / filename), f"CVTG OCR {name}")
    return {name: str(directory) for name, directory in directories.items()}


def require_tokenizer(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    for name in ("tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        require_file(str(root / name), "CVTG VQA tokenizer")
    return root


def run_local_scorer(scorer, clip_path: Path, head_path: Path, scorer_args: list[str],
                     vqa_path: Path = None, vision_path: Path = None,
                     ocr_model_dirs: dict = None, tokenizer_path: Path = None):
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
    with ExitStack() as patches:
        if ocr_model_dirs is not None:
            import paddleocr
            original_paddle_ocr = paddleocr.PaddleOCR

            def local_paddle_ocr(*args, **kwargs):
                # Explicit model dirs bypass PaddleOCR's cache lookup. Keep
                # official language, angle classifier and OCR preprocessing.
                # CPU avoids silent empty detections observed on the GPU backend.
                kwargs.update(ocr_model_dirs, use_gpu=False, enable_mkldnn=False)
                return original_paddle_ocr(*args, **kwargs)

            patches.enter_context(patch.object(paddleocr, "PaddleOCR", local_paddle_ocr))
            print("CVTG OCR device: cpu (use_gpu=False, enable_mkldnn=False)", flush=True)
            for name, directory in ocr_model_dirs.items():
                print(f"CVTG OCR {name}: {directory}", flush=True)
        if vqa_path is not None or tokenizer_path is not None:
            # t2v_metrics reads this registry during VQAScore construction.
            # Redirect file sources, preserving conversation and tokenizer options.
            from t2v_metrics.models.vqascore_models.clip_t5_model import CLIP_T5_MODELS
            config = CLIP_T5_MODELS["clip-flant5-xxl"]
            if vqa_path is not None:
                patches.enter_context(patch.dict(config["model"], {"path": str(vqa_path)}))
                print(f"CVTG VQA main weights: {vqa_path}", flush=True)
            if tokenizer_path is not None:
                patches.enter_context(patch.dict(config["tokenizer"], {"path": str(tokenizer_path)}))
                print(f"CVTG VQA tokenizer: {tokenizer_path}", flush=True)
        if vision_path is not None:
            from t2v_metrics.models.vqascore_models.clip_t5.model.multimodal_encoder.clip_encoder import CLIPVisionTower
            original_vision_init = CLIPVisionTower.__init__

            def local_vision_init(self, vision_tower, *args, **kwargs):
                # Route before __init__ loads config in delay_load mode, then
                # keep the same path for the later processor/weight loads.
                if vision_tower == "openai/clip-vit-large-patch14-336":
                    vision_tower = str(vision_path)
                original_vision_init(self, vision_tower, *args, **kwargs)

            patches.enter_context(patch.object(CLIPVisionTower, "__init__", local_vision_init))
            print(f"CVTG VQA vision weights: {vision_path}", flush=True)
        patches.enter_context(patch.object(clip, "load", local_clip_load))
        patches.enter_context(patch.object(factory, "download_pretrained", local_openclip_download))
        patches.enter_context(patch.object(scorer.UnifiedMetricsEvaluator, "_load_aesthetic_model", load_local_aesthetic_head))
        patches.enter_context(patch.object(sys, "argv", [scorer.__file__, *scorer_args]))
        scorer.main()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--textcrafter_root", required=True)
    parser.add_argument("--clip_model_path", required=True)
    parser.add_argument("--aesthetic_model_path", required=True)
    parser.add_argument("--vqa_model_path")
    parser.add_argument("--vqa_vision_model_path")
    parser.add_argument("--ocr_model_root")
    parser.add_argument("--vqa_tokenizer_path")
    args, scorer_args = parser.parse_known_args(argv)
    # Validate before importing model libraries so missing local files cannot
    # silently fall back to online downloads or a zero aesthetic score.
    clip_path = require_file(args.clip_model_path, "CVTG CLIP weights")
    head_path = require_file(args.aesthetic_model_path, "CVTG aesthetic head")
    vqa_path = require_vqa_model(args.vqa_model_path) if args.vqa_model_path else None
    vision_path = None
    if args.vqa_vision_model_path:
        vision_path = require_vqa_model(args.vqa_vision_model_path, "CVTG VQA vision")
        require_file(str(vision_path / "preprocessor_config.json"), "CVTG VQA vision preprocessor")
    ocr_model_dirs = require_ocr_models(args.ocr_model_root) if args.ocr_model_root else None
    tokenizer_path = require_tokenizer(args.vqa_tokenizer_path) if args.vqa_tokenizer_path else None
    source = require_file(str(Path(args.textcrafter_root) / "unified_metrics_eval.py"), "TextCrafter scorer")
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("cvtg_official_metrics", source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    run_local_scorer(scorer, clip_path, head_path, scorer_args,
                     vqa_path=vqa_path, vision_path=vision_path,
                     ocr_model_dirs=ocr_model_dirs, tokenizer_path=tokenizer_path)


if __name__ == "__main__":
    main()
