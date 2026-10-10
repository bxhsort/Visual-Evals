"""Offline checks: local weight routing must not download or alter other models."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from local_metrics_eval import main, require_ocr_models, require_tokenizer, require_vqa_model, run_local_scorer


class LocalWeightTests(unittest.TestCase):
    def test_local_weights_and_tokenizer_route_correctly_and_restore_loaders(self):
        clip = types.ModuleType("clip")
        clip.load = Mock(return_value="clip-result")
        factory = types.ModuleType("open_clip.factory")
        config = {"url": "official-vit-l-14", "mean": "official-preprocessing"}
        factory.get_pretrained_cfg = Mock(return_value=config)
        factory.download_pretrained = Mock(return_value="other-model-cache")
        open_clip = types.ModuleType("open_clip")
        open_clip.__path__ = []
        open_clip.factory = factory
        torch = types.ModuleType("torch")
        torch.load = Mock(return_value="local-head-state")
        linear = Mock()
        torch.nn = types.SimpleNamespace(Linear=Mock(return_value=linear))
        vqa_module = types.ModuleType("t2v_metrics.models.vqascore_models.clip_t5_model")
        vqa_config = {"tokenizer": {"path": "google/flan-t5-xxl", "model_max_length": 2048},
                      "model": {"path": "zhiqiulin/clip-flant5-xxl", "conversation": "t5_chat"}}
        vqa_module.CLIP_T5_MODELS = {"clip-flant5-xxl": vqa_config}
        vqa_path = Path("/local weights/clip-flant5-xxl")
        vision_path = Path("/local weights/clip-vit-large-patch14-336")
        vision_module = types.ModuleType("t2v_metrics.models.vqascore_models.clip_t5.model.multimodal_encoder.clip_encoder")

        class VisionTower:
            def __init__(self, vision_tower, args, delay_load=False):
                self.name = vision_tower
                self.delay_load = delay_load

        original_vision_init = VisionTower.__init__
        vision_module.CLIPVisionTower = VisionTower
        paddleocr = types.ModuleType("paddleocr")
        original_paddle_ocr = Mock(return_value="local-ocr")
        paddleocr.PaddleOCR = original_paddle_ocr
        ocr_dirs = {"det_model_dir": "/downloaded/ocr/det",
                    "rec_model_dir": "/downloaded/ocr/rec",
                    "cls_model_dir": "/downloaded/ocr/cls"}

        class Evaluator:
            device = "cuda"

            def _load_aesthetic_model(self):
                raise AssertionError("default head lookup must not run")

        original_head_loader = Evaluator._load_aesthetic_model
        original_clip_load = clip.load
        original_download = factory.download_pretrained
        original_argv = sys.argv
        clip_path = Path("/local weights/ViT-L-14.pt")
        head_path = Path("/local weights/sa_0_4_vit_l_14_linear.pth")
        scorer_args = ["--benchmark_dir", "/metadata", "--result_dir", "/old-images/CVTG",
                       "--output_file", "/scores.json", "--cache_dir", "/vqa-cache"]

        def official_main():
            self.assertEqual(paddleocr.PaddleOCR(use_angle_cls=True, lang="en", show_log=False,
                                               det_model_dir="/wrong/cache"), "local-ocr")
            self.assertEqual(vqa_config["model"]["path"], str(vqa_path))
            self.assertEqual(vqa_config["model"]["conversation"], "t5_chat")
            self.assertEqual(vqa_config["tokenizer"]["path"], str(vqa_path))
            self.assertEqual(vqa_config["tokenizer"]["model_max_length"], 2048)
            for delay in (True, False):
                tower = VisionTower("openai/clip-vit-large-patch14-336", args=None, delay_load=delay)
                self.assertEqual(tower.name, str(vision_path))
                self.assertEqual(tower.delay_load, delay)
            self.assertEqual(VisionTower("openai/other-model", None).name, "openai/other-model")
            self.assertEqual(sys.argv, ["/official/unified_metrics_eval.py", *scorer_args])
            self.assertEqual(clip.load("ViT-L/14", device="cuda", jit=False), "clip-result")
            self.assertEqual(factory.download_pretrained(config, cache_dir="/vqa-cache"), str(clip_path))
            # VQA's 336px vision encoder and any unrelated model retain upstream loading.
            clip.load("ViT-L/14@336px", device="cuda")
            self.assertEqual(factory.download_pretrained({"url": "vqa-vision-336"}), "other-model-cache")
            self.assertIs(Evaluator()._load_aesthetic_model(), linear)

        scorer = types.SimpleNamespace(__file__="/official/unified_metrics_eval.py",
                                       UnifiedMetricsEvaluator=Evaluator, main=official_main)
        with patch.dict(sys.modules, {"clip": clip, "open_clip": open_clip,
                                      "open_clip.factory": factory, "torch": torch,
                                      vqa_module.__name__: vqa_module,
                                      vision_module.__name__: vision_module,
                                      "paddleocr": paddleocr}):
            run_local_scorer(scorer, clip_path, head_path, scorer_args,
                             vqa_path=vqa_path, vision_path=vision_path, ocr_model_dirs=ocr_dirs,
                             tokenizer_path=vqa_path)
            self.assertIs(paddleocr.PaddleOCR, original_paddle_ocr)
            self.assertEqual(vqa_config["model"]["path"], "zhiqiulin/clip-flant5-xxl")
            self.assertEqual(vqa_config["tokenizer"]["path"], "google/flan-t5-xxl")
            self.assertIs(VisionTower.__init__, original_vision_init)
            scorer.main = Mock(side_effect=RuntimeError("scorer failed"))
            with self.assertRaisesRegex(RuntimeError, "scorer failed"):
                run_local_scorer(scorer, clip_path, head_path, scorer_args,
                                 vqa_path=vqa_path, vision_path=vision_path, ocr_model_dirs=ocr_dirs,
                                 tokenizer_path=vqa_path)
            self.assertIs(paddleocr.PaddleOCR, original_paddle_ocr)
            self.assertEqual(vqa_config["model"]["path"], "zhiqiulin/clip-flant5-xxl")
            self.assertEqual(vqa_config["tokenizer"]["path"], "google/flan-t5-xxl")
            self.assertIs(VisionTower.__init__, original_vision_init)
        self.assertIs(clip.load, original_clip_load)
        self.assertIs(factory.download_pretrained, original_download)
        self.assertIs(Evaluator._load_aesthetic_model, original_head_loader)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(original_clip_load.call_args_list[0].args, (str(clip_path),))
        self.assertEqual(original_clip_load.call_args_list[0].kwargs, {"device": "cuda", "jit": False})
        original_download.assert_called_once_with({"url": "vqa-vision-336"})
        torch.nn.Linear.assert_called_once_with(768, 1)
        torch.load.assert_called_once_with(str(head_path), map_location="cuda")
        linear.load_state_dict.assert_called_once_with("local-head-state")
        linear.eval.assert_called_once_with()
        linear.to.assert_called_once_with("cuda")
        original_paddle_ocr.assert_called_once_with(use_angle_cls=True, lang="en", show_log=False,
                                                   **ocr_dirs)

    def test_all_ocr_files_required_before_importing_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / relative / name for relative in (
                "whl/det/en/en_PP-OCRv3_det_infer", "whl/rec/en/en_PP-OCRv4_rec_infer",
                "whl/cls/ch/ch_ppocr_mobile_v2.0_cls_infer")
                for name in ("inference.pdmodel", "inference.pdiparams")]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            dirs = require_ocr_models(str(root))
            self.assertEqual(dirs["det_model_dir"], str(paths[0].parent.resolve()))
            self.assertEqual(dirs["rec_model_dir"], str(paths[2].parent.resolve()))
            self.assertEqual(dirs["cls_model_dir"], str(paths[4].parent.resolve()))
            clip_path = root / "clip.pt"
            head_path = root / "head.pth"
            clip_path.touch()
            head_path.touch()
            for path in paths:
                with self.subTest(missing=path.name, model=path.parent.name):
                    path.unlink()
                    with patch.object(importlib.util, "spec_from_file_location") as load_source:
                        with self.assertRaisesRegex(FileNotFoundError, "CVTG OCR"):
                            main(["--textcrafter_root", str(root), "--clip_model_path", str(clip_path),
                                  "--aesthetic_model_path", str(head_path), "--ocr_model_root", str(root)])
                        load_source.assert_not_called()
                    path.touch()

    def test_missing_vqa_shard_fails_before_importing_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{}')
            (root / "pytorch_model.bin.index.json").write_text(json.dumps(
                {"weight_map": {"layer1": "part1.bin", "layer2": "part2.bin"}}))
            (root / "part1.bin").touch()
            clip_path = root / "ViT-L-14.pt"
            head_path = root / "head.pth"
            clip_path.touch()
            head_path.touch()
            with patch.object(importlib.util, "spec_from_file_location") as load_source:
                with self.assertRaisesRegex(FileNotFoundError, "CVTG VQA weight shard"):
                    main(["--textcrafter_root", str(root), "--clip_model_path", str(clip_path),
                          "--aesthetic_model_path", str(head_path), "--vqa_model_path", str(root)])
                load_source.assert_not_called()
            (root / "part2.bin").touch()
            self.assertEqual(require_vqa_model(str(root)), root.resolve())

    def test_tokenizer_files_required_before_importing_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("tokenizer_config.json", "special_tokens_map.json", "spiece.model")]
            for path in paths:
                path.touch()
            self.assertEqual(require_tokenizer(str(root)), root.resolve())
            clip_path = root / "clip.pt"
            head_path = root / "head.pth"
            clip_path.touch()
            head_path.touch()
            for path in paths:
                with self.subTest(missing=path.name):
                    path.unlink()
                    with patch.object(importlib.util, "spec_from_file_location") as load_source:
                        with self.assertRaisesRegex(FileNotFoundError, "CVTG VQA tokenizer"):
                            main(["--textcrafter_root", str(root), "--clip_model_path", str(clip_path),
                                  "--aesthetic_model_path", str(head_path), "--vqa_tokenizer_path", str(root)])
                        load_source.assert_not_called()
                    path.touch()

    def test_single_file_vqa_weights_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "CVTG VQA config"):
                require_vqa_model(str(root))
            (root / "config.json").write_text('{}')
            with self.assertRaisesRegex(FileNotFoundError, "CVTG VQA weights/index"):
                require_vqa_model(str(root))
            (root / "model.safetensors").touch()
            self.assertEqual(require_vqa_model(str(root)), root.resolve())

    def test_missing_local_weight_fails_before_importing_or_downloading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip_path = root / "ViT-L-14.pt"
            head_path = root / "head.pth"
            args = ["--textcrafter_root", str(root), "--clip_model_path", str(clip_path),
                    "--aesthetic_model_path", str(head_path)]
            with patch.object(importlib.util, "spec_from_file_location") as load_source:
                with self.assertRaisesRegex(FileNotFoundError, "CVTG CLIP weights"):
                    main(args)
                clip_path.touch()
                with self.assertRaisesRegex(FileNotFoundError, "CVTG aesthetic head"):
                    main(args)
                load_source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
