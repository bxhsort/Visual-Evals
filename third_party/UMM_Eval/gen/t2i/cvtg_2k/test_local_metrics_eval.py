"""Offline checks: local weight routing must not download or alter other models."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from local_metrics_eval import main, run_local_scorer


class LocalWeightTests(unittest.TestCase):
    def test_both_clip_loaders_and_head_use_local_files_without_changing_vqa(self):
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
                                      "open_clip.factory": factory, "torch": torch}):
            run_local_scorer(scorer, clip_path, head_path, scorer_args)
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
