"""Exercise real LongText save/resume code without importing GPU/model packages."""
import ast
import glob
import math
import os
import re
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[4]


def source_tree(path):
    return ast.parse((REPO / path).read_text())


class FakeImage:
    def __init__(self, payload):
        self.payload = payload

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def verify(self):
        if not self.payload:
            raise ValueError("empty test image")


class LongTextSamplesTest(unittest.TestCase):
    def test_wrapper_default_prepares_four_repeat_experiment_without_running_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "bin"
            binary.mkdir()
            (binary / "python").symlink_to(sys.executable)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       BENCHMARK="longtextbench", EXPERIMENT_NAME="offline-default",
                       OUTPUT_BASE=str(Path(tmp) / "outputs"), MODEL_PATH="", GPU_LIST="0",
                       RUN_GENERATION="0", RUN_EVAL="0", COVERAGE_REPORT="0", THINK_MODE="0")
            for key in ("NUM_IMAGES", "BATCH_SIZE", "TARGET_INDICES", "TARGET_PROMPT_IDS",
                        "EXPERIMENT_SETTING", "METADATA_FILE", "LONGTEXTBENCH_METADATA_FILE_EN",
                        "LONGTEXTBENCH_METADATA_FILE_ZH", "LONGTEXTBENCH_LANGS"):
                env.pop(key, None)
            result = subprocess.run(["bash", "scripts/eval/run_t2i_accelerate.sh"], cwd=REPO,
                                    env=env, text=True, capture_output=True, timeout=30, check=True)
            self.assertIn("num_images=4 batch_size=4", result.stdout)
            self.assertEqual(len(list((Path(tmp) / "outputs").rglob("experiment_manifest.json"))), 1)

    def test_unified_saves_all_four_repeats_and_resumes_only_when_complete(self):
        tree = source_tree("eval/gen/gen_images_mp_t2i.py")
        namespace = dict(os=os, math=math,
                         Image=SimpleNamespace(open=lambda path: FakeImage(Path(path).read_bytes())))
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"output_paths_for_sample", "all_outputs_exist"}]
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "generator helpers", "exec"), namespace)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        branch = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and ast.dump(node.test) == ast.dump(ast.parse("args.benchmark == 'longtextbench'", mode="eval").body)
                      and any(isinstance(stmt, ast.For) for stmt in node.body))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(benchmark="longtextbench", mode="en", output_dir=tmp,
                                   num_images=4, resolution=1)
            _, _, paths = namespace["output_paths_for_sample"](args, {"prompt_id": 7}, 0)
            legacy = Path(tmp) / "longtextbench_en/samples/7_sample.png"
            FakeImage(b"legacy").save(legacy)
            self.assertFalse(namespace["all_outputs_exist"](paths))
            images = [FakeImage(bytes([repeat])) for repeat in range(1, 5)]
            namespace.update(args=args, images=images, output_paths=paths,
                             make_grid=lambda *args: FakeImage(b"grid"),
                             atomic_save=lambda image, path: image.save(path))
            exec(compile(ast.Module(body=branch.body, type_ignores=[]), "generator save branch", "exec"), namespace)
            for repeat in range(1, 5):
                self.assertEqual(Path(paths[repeat]).name, f"7_{repeat}.png")
                self.assertEqual(Path(paths[repeat]).read_bytes(), bytes([repeat]))
            self.assertTrue(namespace["all_outputs_exist"](paths))
            Path(paths[4]).unlink()
            self.assertFalse(namespace["all_outputs_exist"](paths))

    def test_legacy_generator_also_saves_all_repeats_separately(self):
        tree = source_tree("third_party/UMM_Eval/gen/t2i/longtextbench/inference.py")
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.dump(node.iter) == ast.dump(ast.parse("enumerate(images, start=1)", mode="eval").body))
        with tempfile.TemporaryDirectory() as tmp:
            namespace = dict(os=os, samples_output_dir=tmp, prompt_id=3,
                             images=[FakeImage(bytes([i])) for i in range(1, 5)])
            exec(compile(ast.Module(body=[loop], type_ignores=[]), "legacy save loop", "exec"), namespace)
            self.assertEqual(sorted(path.name for path in Path(tmp).glob("*.png")),
                             [f"3_{i}.png" for i in range(1, 5)])
            self.assertEqual(len({path.read_bytes() for path in Path(tmp).glob("*.png")}), 4)

    def test_evaluator_reads_four_repeats_without_counting_legacy_image_twice(self):
        tree = source_tree("third_party/UMM_Eval/gen/t2i/longtextbench/eval/evaluate_text_reward.py")
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "sample_image_files")
        namespace = dict(os=os, glob=glob, re=re)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "scorer image selection", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["0_sample.png", "1_sample.png"] + [f"0_{i}.png" for i in range(1, 5)]:
                FakeImage(b"image").save(Path(tmp) / name)
            actual = [Path(path).name for path in namespace["sample_image_files"](tmp)]
            self.assertEqual(actual, ["0_1.png", "0_2.png", "0_3.png", "0_4.png", "1_sample.png"])


if __name__ == "__main__":
    unittest.main()
