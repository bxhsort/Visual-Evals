"""Offline regressions for experiment isolation and safe evaluation-cache reuse.

Run with: python -m unittest discover -s third_party/UMM_Eval/gen/common
No model, GPU, network, evaluator, or API is invoked.
"""
from __future__ import annotations

import copy
import os
import tempfile
import unittest
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiment_layout import derive_experiment_root, ensure_manifest, prepare_experiment
import build_report
import collect_results
import image_locations
import coverage_t2i


class ExperimentLayoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dg2-layout-test-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.outputs = self.base / "outputs"
        self.model = self.base / "model"
        self.model.mkdir()
        (self.model / "ema.safetensors").write_bytes(b"offline fake weights")
        self.metadata = self.base / "prompts.jsonl"
        self.metadata.write_text('{"prompt": "a red cube", "tag": "single_object"}\n', encoding="utf-8")
        self.config = {
            "MODEL_PATH": str(self.model),
            "MODEL_NAME": "bagel",
            "METADATA_FILE": str(self.metadata),
            "NUM_STEPS": "50",
            "NUM_IMAGES": "4",
            "BATCH_SIZE": "4",
            "SEED": "42",
            "IMAGE_SIZE": "1024",
            "CFG_SCALE": "4.0",
            "THINK_MODE": "0",
            "RUN_GENERATION": "1",
            "RUN_EVAL": "1",
        }

    def prepare(self, config=None, *, benchmark="geneval_short", experiment="regression"):
        return prepare_experiment(
            benchmark, str(self.outputs), experiment, "baseline", self.config if config is None else config
        )

    def test_exact_three_level_layout_and_required_experiment(self):
        root = derive_experiment_root(str(self.outputs), "experiment-1", "setting-1", "geneval_short")
        self.assertEqual(Path(root), self.outputs.resolve() / "experiment-1" / "setting-1" / "geneval_short")
        with self.assertRaises((ValueError, TypeError)):
            derive_experiment_root(str(self.outputs), "", "setting-1", "geneval_short")

    def test_names_cannot_escape_the_selected_output_directory(self):
        for invalid in ("..", "../elsewhere", "/absolute", "nested/name"):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                derive_experiment_root(str(self.outputs), invalid, "setting-1", "geneval_short")

    def test_benchmarks_variants_thinking_and_experiments_are_isolated(self):
        benches = (
            "geneval_short", "geneval_long", "dpgbench", "longtextbench", "cvtg_2k", "tiifbench",
            "oneigbench", "wise", "bizgeneval", "qwenimagebench", "imgedit", "geditbench", "rise", "editcompass",
        )
        roots = [derive_experiment_root(str(self.outputs), "exp", "setting", bench, thinking=thinking)
                 for bench in benches for thinking in (False, True)]
        self.assertEqual(len(set(roots)), len(roots))
        for bench in benches:
            self.assertEqual(Path(derive_experiment_root(str(self.outputs), "exp", "setting", bench, True)).name,
                             bench + "_thinking")
        self.assertNotEqual(derive_experiment_root(str(self.outputs), "exp-a", "setting", "wise"),
                            derive_experiment_root(str(self.outputs), "exp-b", "setting", "wise"))

    def test_identical_configuration_can_resume_without_replacing_manifest(self):
        first = self.prepare()
        manifest = Path(first["results_root"]) / "experiment_manifest.json"
        self.assertTrue(manifest.is_file())
        contents = manifest.read_bytes()
        stamp = manifest.stat().st_mtime_ns
        second = self.prepare(copy.deepcopy(self.config))
        self.assertEqual(first["results_root"], second["results_root"])
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(manifest.read_bytes(), contents)
        self.assertEqual(manifest.stat().st_mtime_ns, stamp)

    def test_active_geneval_short_and_long_prompt_protocols_are_separate(self):
        short = self.prepare(dict(self.config, GENEVAL_PROMPT_MODE="short"), benchmark="geneval")
        long = self.prepare(dict(self.config, GENEVAL_PROMPT_MODE="long"), benchmark="geneval")
        self.assertNotEqual(short["results_root"], long["results_root"])
        self.assertEqual(short["fingerprint"], long["fingerprint"])
        self.assertNotEqual(short["setting_name"], long["setting_name"])
        self.assertIn("prompt_short", short["setting_name"])
        self.assertIn("prompt_long", long["setting_name"])
        self.assertEqual(Path(short["results_root"]).name, "geneval")
        self.assertEqual(Path(long["results_root"]).name, "geneval")
        with self.assertRaises(ValueError):
            self.prepare(dict(self.config, GENEVAL_PROMPT_MODE="both"), benchmark="geneval")

    def test_sampling_configuration_creates_new_setting_but_scope_changes_reject_cache(self):
        baseline = self.prepare()
        changes = {"NUM_STEPS": "100", "NUM_IMAGES": "8", "SEED": "43", "BATCH_SIZE": "1",
                   "IMAGE_SIZE": "512", "CFG_SCALE": "7", "BIZGENEVAL_DEFAULT_ASPECT_RATIO": "16:9"}
        for key, value in changes.items():
            config = dict(self.config, **{key: value})
            with self.subTest(key=key):
                updated = self.prepare(config)
                self.assertNotEqual(updated["fingerprint"], baseline["fingerprint"])
                self.assertNotEqual(updated["results_root"], baseline["results_root"])
        manifest = Path(baseline["results_root"]) / "experiment_manifest.json"
        original = manifest.read_bytes()
        for key, value in {"TIIFBENCH_SPLIT": "test", "QIB_EVAL_BACKEND": "api",
                           "EVAL_MODEL": "different-judge", "TARGET_INDICES": "1,2"}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(dict(self.config, **{key: value}))
            self.assertEqual(manifest.read_bytes(), original)

    def test_metadata_content_and_weight_identity_invalidate_same_path_cache(self):
        baseline = self.prepare()
        manifest = Path(baseline["results_root"]) / "experiment_manifest.json"
        original_manifest = manifest.read_bytes()
        # Deliberately retain file length and timestamp: metadata identity must use content.
        old_stat = self.metadata.stat()
        old_contents = self.metadata.read_bytes()
        self.metadata.write_text('{"prompt": "a tan cube", "tag": "single_object"}\n', encoding="utf-8")
        os.utime(self.metadata, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(manifest.read_bytes(), original_manifest)
        self.metadata.write_bytes(old_contents)
        os.utime(self.metadata, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        weights = self.model / "ema.safetensors"
        weights.write_bytes(b"new larger offline fake weights")
        changed_weights = self.prepare()
        self.assertNotEqual(changed_weights["fingerprint"], baseline["fingerprint"])
        self.assertNotEqual(changed_weights["results_root"], baseline["results_root"])

    def test_cvtg_scorer_weight_changes_reject_old_scores(self):
        for key in ("CVTG_CLIP_MODEL_PATH", "CVTG_AESTHETIC_MODEL_PATH", "CVTG_VQA_MODEL_PATH", "CVTG_VQA_VISION_MODEL_PATH"):
            with self.subTest(key=key):
                weights = self.base / (key + ".pt")
                weights.write_bytes(b"original scorer weights")
                config = dict(self.config, **{key: str(weights)})
                first = self.prepare(config, benchmark="cvtg_2k", experiment=key)
                self.assertIn(key, first["identity"]["inputs"])
                manifest = Path(first["results_root"]) / "experiment_manifest.json"
                original = manifest.read_bytes()
                weights.write_bytes(b"changed and larger scorer weights")
                with self.assertRaises(ValueError):
                    self.prepare(config, benchmark="cvtg_2k", experiment=key)
                self.assertEqual(manifest.read_bytes(), original)

    def test_edit_source_image_identity_invalidates_unchanged_metadata_cache(self):
        sources = self.base / "source_images"
        sources.mkdir()
        image = sources / "source.png"
        image.write_bytes(b"offline source image placeholder")
        config = dict(self.config, IMAGE_ROOT=str(sources))
        first = self.prepare(config, benchmark="imgedit")
        manifest = Path(first["results_root"]) / "experiment_manifest.json"
        original = manifest.read_bytes()
        image.write_bytes(b"modified offline source image placeholder with different dimensions")
        with self.assertRaises(ValueError):
            self.prepare(config, benchmark="imgedit")
        self.assertEqual(manifest.read_bytes(), original)

    def test_judge_model_weight_identity_invalidates_unchanged_model_path_cache(self):
        judge = self.base / "judge_model"
        judge.mkdir()
        weights = judge / "model.safetensors"
        weights.write_bytes(b"offline judge weights")
        config = dict(self.config, QJUDGER_MODEL_PATH=str(judge))
        first = self.prepare(config, benchmark="qwenimagebench")
        manifest = Path(first["results_root"]) / "experiment_manifest.json"
        original = manifest.read_bytes()
        weights.write_bytes(b"different offline judge weights")
        with self.assertRaises(ValueError):
            self.prepare(config, benchmark="qwenimagebench")
        self.assertEqual(manifest.read_bytes(), original)

    def test_shared_generation_setting_groups_benchmarks_and_thinking_variants(self):
        dpg = self.prepare(benchmark="dpgbench")
        other_metadata = self.base / "other_benchmark.jsonl"
        other_metadata.write_text('{"id": "other", "prompt": "a blue sphere"}\n', encoding="utf-8")
        oneig = self.prepare(dict(self.config, METADATA_FILE=str(other_metadata)), benchmark="oneigbench")
        self.assertEqual(dpg["setting_name"], oneig["setting_name"])
        self.assertEqual(Path(dpg["results_root"]).parent, Path(oneig["results_root"]).parent)
        self.assertNotEqual(dpg["results_root"], oneig["results_root"])
        thinking = self.prepare(dict(self.config, THINK_MODE="1"), benchmark="dpgbench")
        self.assertEqual(thinking["setting_name"], dpg["setting_name"])
        self.assertEqual(Path(thinking["results_root"]).name, "dpgbench_thinking")
        alternate_experiment = self.prepare(benchmark="dpgbench", experiment="another-experiment")
        self.assertEqual(alternate_experiment["setting_name"], dpg["setting_name"])
        self.assertEqual(alternate_experiment["fingerprint"], dpg["fingerprint"])
        self.assertNotEqual(alternate_experiment["results_root"], dpg["results_root"])

    def test_equivalent_generation_parameter_aliases_share_setting(self):
        first_config = dict(self.config, CFG_TEXT_SCALE=4.0, CFG_IMG_SCALE=1.5,
                            CFG_INTERVAL="[0,1.0]", CFG_RENORM="text_channel", CFG_NORM="global")
        first = self.prepare(first_config, benchmark="imgedit")
        second_config = dict(first_config)
        second_config["NUM_TIMESTEPS"] = int(second_config.pop("NUM_STEPS"))
        second_config["TXT_CFG"] = str(int(second_config.pop("CFG_TEXT_SCALE")))
        second_config["IMG_CFG"] = str(second_config.pop("CFG_IMG_SCALE"))
        second_config["MAX_IMAGE_SIZE"] = int(second_config.pop("IMAGE_SIZE"))
        second_config["CFG_RENORM_TYPE"] = second_config.pop("CFG_RENORM")
        second_config["CFG_INTERVAL"] = [0.0, 1]
        second = self.prepare(second_config, benchmark="editcompass")
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["setting_name"], second["setting_name"])
        self.assertEqual(Path(first["results_root"]).parent, Path(second["results_root"]).parent)
        manifest = Path(first["results_root"]) / "experiment_manifest.json"
        before = manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "Conflicting generation parameter aliases"):
            self.prepare(dict(first_config, NUM_TIMESTEPS=100), benchmark="imgedit")
        self.assertEqual(manifest.read_bytes(), before)

    def test_real_editing_crop_and_cfg_protocols_create_separate_settings(self):
        unified = dict(self.config, CFG_TEXT_SCALE=4.0, CFG_IMG_SCALE=1.5,
                       CFG_INTERVAL=[0, 1], CROP_BBOX=True)
        imgedit = self.prepare(unified, benchmark="imgedit")
        gedit = self.prepare(dict(unified, CFG_INTERVAL=[0.4, 1], CROP_BBOX=False),
                             benchmark="geditbench")
        self.assertNotEqual(imgedit["fingerprint"], gedit["fingerprint"])
        self.assertNotEqual(Path(imgedit["results_root"]).parent,
                            Path(gedit["results_root"]).parent)

    def test_generator_seed_strategy_and_length_change_actual_generation_setting(self):
        config = dict(self.config, GENERATOR="gen_images_mp_t2i.py",
                      SEED_STRATEGY="base_plus_stable_index", MAX_LENGTH=2048)
        first = self.prepare(config, benchmark="dpgbench")
        # These can change generated pixels with unchanged sampling values and
        # metadata. They therefore need fresh generation roots, not cache reuse.
        for key, value in {"GENERATOR": "benchmark_specific_generator.py",
                           "SEED_STRATEGY": "worker_rng", "MAX_LENGTH": 4096}.items():
            with self.subTest(key=key):
                changed = self.prepare(dict(config, **{key: value}), benchmark="dpgbench")
                self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
                self.assertNotEqual(first["results_root"], changed["results_root"])
        equivalent = self.prepare(dict(config, MAX_LENGTH="2048"), benchmark="oneigbench")
        self.assertEqual(first["fingerprint"], equivalent["fingerprint"])

    def test_wise_selected_stable_index_coverage_matches_generated_image_and_judgment(self):
        metadata = self.base / "wise.json"
        metadata.write_text(json.dumps([
            {"prompt_id": 104, "Prompt": "one", "Category": "science"},
            {"prompt_id": 8, "Prompt": "two", "Category": "culture"},
            {"prompt_id": 300, "Prompt": "three", "Category": "science"},
        ]), encoding="utf-8")
        root = self.base / "wise-results"
        images = root / "wise_custom"
        images.mkdir(parents=True)
        (images / "8.png").write_bytes(b"offline image placeholder")
        (root / "culture_scores_results_wise_custom.jsonl").write_text(
            json.dumps({"prompt_id": 8, "consistency": 0, "realism": 0,
                        "aesthetic_quality": 0}) + "\n", encoding="utf-8")
        rows, warnings = coverage_t2i.build_samples(
            "wise", root, metadata_file=metadata, image_dir=images,
            judge_root=root, target_indices="1", num_images=1)
        self.assertEqual([row["sample_id"] for row in rows], ["8"])
        self.assertEqual(Path(rows[0]["image_path"]), images / "8.png")
        self.assertTrue(rows[0]["judged"], "valid zero scores must count as judged")
        self.assertTrue(any("Subset scope requested" in warning for warning in warnings))
        self.assertFalse(any("does not filter target_indices" in warning for warning in warnings))
        # The same generator does not accept prompt-ID filtering for WISE.
        # Reporting must retain the full scope and explicitly explain that fact.
        rows, warnings = coverage_t2i.build_samples(
            "wise", root, metadata_file=metadata, image_dir=images,
            target_prompt_ids="8", num_images=1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(any("does not filter target_prompt_ids" in warning for warning in warnings))

    def test_execution_switches_and_secrets_do_not_change_identity_or_leak(self):
        baseline = self.prepare()
        secret = "synthetic-api-key-for-offline-test"
        config = dict(self.config, RUN_GENERATION="0", RUN_EVAL="1", API_KEY=secret,
                      OPENAI_API_KEY=secret, JUDGE_API_KEY=secret)
        resumed = self.prepare(config)
        self.assertEqual(baseline["fingerprint"], resumed["fingerprint"])
        self.assertEqual(baseline["results_root"], resumed["results_root"])
        self.assertNotIn(secret, (Path(resumed["results_root"]) / "experiment_manifest.json").read_text())

    def test_unmanaged_artifacts_and_conflicting_manifests_are_rejected(self):
        prepared = self.prepare()
        root = Path(prepared["results_root"])
        before = (root / "experiment_manifest.json").read_bytes()
        other = copy.deepcopy(prepared)
        other["identity"] = {"other_run": True}
        with self.assertRaises((ValueError, RuntimeError)):
            ensure_manifest(str(root), other)
        self.assertEqual((root / "experiment_manifest.json").read_bytes(), before)
        (root / "experiment_manifest.json").unlink()
        artifact = root / "already-generated.png"
        artifact.write_bytes(b"existing output that must not be silently adopted")
        with self.assertRaises((ValueError, RuntimeError)):
            ensure_manifest(str(root), prepared)
        self.assertEqual(artifact.read_bytes(), b"existing output that must not be silently adopted")
        self.assertFalse((root / "experiment_manifest.json").exists())

    def test_concurrent_same_identity_preparation_is_safe(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.prepare(copy.deepcopy(self.config)), range(4)))
        self.assertEqual(len({entry["results_root"] for entry in results}), 1)
        self.assertEqual(len({entry["fingerprint"] for entry in results}), 1)
        self.assertTrue((Path(results[0]["results_root"]) / "experiment_manifest.json").is_file())

    def test_collector_and_report_keep_geneval_prompt_protocol_identity(self):
        records = []
        for variant, score in (("short", 0.6), ("long", 0.8)):
            manifest = self.prepare(dict(self.config, GENEVAL_PROMPT_MODE=variant), benchmark="geneval")
            root = Path(manifest["results_root"])
            (root / "results_geneval_scores.txt").write_text(
                f"Overall score (avg. over tasks): {score}\n", encoding="utf-8")
            record = collect_results.collect("geneval", str(root))
            self.assertEqual(record["prompt_variant"], variant)
            self.assertEqual(record["experiment_name"], "regression")
            self.assertEqual(record["setting_name"], manifest["setting_name"])
            self.assertEqual(record["headline"], score)
            records.append(record)
        for rendered in (build_report.render(records, [], {}), build_report.render_html(records, [], {})):
            self.assertIn("geneval (short prompt)", rendered)
            self.assertIn("geneval (long prompt)", rendered)

    def test_collector_cli_uses_exact_setting_and_requires_explicit_old_layout(self):
        manifest = self.prepare(dict(self.config, GENEVAL_PROMPT_MODE="short"), benchmark="geneval")
        root = Path(manifest["results_root"])
        (root / "results_geneval_scores.txt").write_text(
            "Overall score (avg. over tasks): 0.61\n", encoding="utf-8")
        script = Path(collect_results.__file__)
        arguments = [sys.executable, str(script), "--bench", "geneval", "--output-base", str(self.outputs)]
        failed = subprocess.run(arguments, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("--experiment-name", failed.stderr)
        result = subprocess.run(arguments + ["--experiment-name", "regression", "--setting-name", manifest["setting_name"]],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(Path(record["output_root"]), root)
        self.assertEqual(record["headline"], 0.61)
        self.assertEqual(record["prompt_variant"], "short")

    def test_active_editing_image_globs_respect_language_and_exclude_source_images(self):
        imgedit = Path(self.prepare(benchmark="imgedit")["results_root"])
        result = imgedit / "bagel" / "sample.png"
        result.parent.mkdir()
        result.write_bytes(b"offline image path fixture")
        self.assertEqual(image_locations.list_images("imgedit", str(imgedit)), [str(result.resolve())])
        gedit = Path(self.prepare(benchmark="geditbench")["results_root"])
        english = gedit / "gen_image/fullset/add/en/sample.png"
        chinese = gedit / "gen_image/fullset/add/cn/sample.png"
        source = english.with_name("sample_SRCIMG.png")
        for image in (english, chinese, source):
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"offline image path fixture")
        self.assertEqual(image_locations.list_images("geditbench", str(gedit), language="en"), [str(english.resolve())])
        self.assertEqual(image_locations.list_images("geditbench", str(gedit), language="cn"), [str(chinese.resolve())])
        self.assertEqual(set(image_locations.list_images("geditbench", str(gedit))),
                         {str(english.resolve()), str(chinese.resolve())})


if __name__ == "__main__":
    unittest.main()
