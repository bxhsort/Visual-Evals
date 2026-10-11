"""Offline regression tests for report completeness; no model/API dependencies."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

import coverage_audit
import collect_results
import build_report

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a3xkAAAAASUVORK5CYII=")


class CoverageReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
        return str(path)

    def dump(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_missing_image_judge_error_and_unknown_are_separate(self):
        rows = [
            dict(sample_id="zero", image_path=self.image("zero.png"), judged=True, score=0),
            dict(sample_id="missing", image_path=str(self.root / "missing.png"), judged=False),
            dict(sample_id="api", image_path=self.image("api.png"), judged=False,
                 judge_error="HTTP 429", error_type="api_error"),
            dict(sample_id="unknown", image_path=self.image("unknown.png"), judged=False),
            dict(sample_id="empty", image_path=str(self.root / "empty.png"), judged=False),
        ]
        (self.root / "empty.png").write_bytes(b"")
        record = coverage_audit.audit_rows("imgedit", self.root, rows)
        self.assertEqual(record["counts"], dict(expected=5, generated=3, valid_scored=1,
                         missing_image=1, corrupt_image=1, judge_failed=1, unscored=1))
        self.assertFalse(record["coverage_complete"])
        self.assertEqual([r["status"] for r in record["samples"]],
                         ["scored", "missing_image", "judge_failed", "unscored", "corrupt_image"])

    def test_missing_metadata_is_unknown_not_complete(self):
        record = coverage_audit.audit_rows("geneval", self.root, [])
        self.assertIsNone(record["counts"]["expected"])
        self.assertIsNone(record["counts"]["missing_image"])
        self.assertIsNone(record["coverage_ratio"])
        self.assertFalse(record["coverage_complete"])
        self.assertIn("无法列出遗漏 ID", coverage_audit.render_markdown(record))

    def test_explicit_request_and_parse_failures_have_separate_reasons(self):
        rows = [dict(sample_id="api", image_path=self.image("api.png"), judge_error="APIError: timeout"),
                dict(sample_id="parse", image_path=self.image("parse.png"), judge_error="Failed parsing score response")]
        record = coverage_audit.audit_rows("imgedit", self.root, rows)
        self.assertEqual(record["counts"]["judge_failed"], 2)
        self.assertIn("judge 请求失败", record["samples"][0]["reason"])
        self.assertIn("评分响应解析失败", record["samples"][1]["reason"])

    def test_malformed_coverage_is_unknown_without_losing_score(self):
        (self.root / "imgedit_typescore.json").write_text('{"overall": 4.5}')
        self.dump("coverage_imgedit_en.json", [])
        record = collect_results.collect("imgedit", str(self.root))
        self.assertEqual(record["headline"], 4.5)
        self.assertIsNone(record["coverage"])
        self.assertIn("覆盖率文件不可用", record["coverage_warning"])

    def test_duplicate_expected_ids_cannot_certify_completeness(self):
        row = dict(sample_id="one", image_path=self.image("one.png"), judged=True)
        record = coverage_audit.audit_rows("imgedit", self.root, [row, row])
        self.assertFalse(record["coverage_complete"])
        self.assertTrue(any("重复" in warning for warning in record["warnings"]))

    def test_score_summary_alone_does_not_certify_coverage(self):
        (self.root / "results_geneval_scores.txt").write_text("Overall score (avg. over tasks): 0.77\n")
        record = collect_results.collect("geneval", str(self.root))
        self.assertEqual(record["headline"], 0.77)
        self.assertTrue(record["complete"])  # original score-availability contract retained
        self.assertIsNone(record["coverage"])
        self.assertIn("覆盖率未核验", build_report.render([record], [], {}))

    def test_report_keeps_score_and_lists_each_anomaly_with_escaped_html(self):
        rows = [dict(sample_id="bad|<script>", category="add", image_path=str(self.root / "missing.png"), judged=False),
                dict(sample_id="judge", category="remove", image_path=self.image("judge.png"), judged=False,
                     judge_error="timeout <script>")]
        audit = coverage_audit.audit_rows("imgedit", self.root, rows)
        record = dict(benchmark="imgedit", headline=5.0, headline_metric="overall", complete=True, coverage=audit)
        md = build_report.render([record], [], {})
        html = build_report.render_html([record], [], {})
        self.assertEqual(record["headline"], 5.0)
        self.assertIn("bad\\|<script>", md)
        self.assertIn("judge", md)
        self.assertIn("评测未完整完成", md)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        prefix = coverage_audit.write_audit(audit, self.root / "coverage_imgedit_en")
        self.assertTrue(Path(prefix + ".md").exists())
        csv = Path(prefix + ".csv").read_text()
        self.assertIn("missing_image", csv)
        self.assertIn("judge_failed", csv)

    def test_collect_attaches_only_matching_language(self):
        (self.root / "imgedit_typescore.json").write_text('{"overall": 4.5}')
        coverage = coverage_audit.audit_rows("imgedit", self.root, [])
        self.dump("coverage_imgedit_en.json", coverage)
        record = collect_results.collect("imgedit", str(self.root), language="en")
        self.assertEqual(record["headline"], 4.5)
        self.assertIsNotNone(record["coverage"])
        record = collect_results.collect("imgedit", str(self.root), language="cn")
        self.assertIsNone(record["coverage"])

    def test_geneval_includes_entire_expected_population_and_false_verdict(self):
        meta = self.root / "metadata.jsonl"
        meta.write_text('{"tag":"single_object","prompt":"bench"}\n{"tag":"counting","prompt":"two cats"}\n')
        image = self.image("gen_eval/00000/samples/00000.png")
        (self.root / "results_geneval.jsonl").write_text(json.dumps({"filename": image, "correct": False}) + "\n")
        record = coverage_audit.audit("geneval", self.root, metadata_file=str(meta), num_images=4)
        self.assertEqual(record["counts"]["expected"], 8)
        self.assertEqual(record["counts"]["valid_scored"], 1)
        self.assertEqual(record["counts"]["missing_image"], 7)

    def test_imgedit_preserves_valid_zero_and_identifies_explicit_api_failure(self):
        meta = self.dump("metadata.json", {"one":{"id":"one.png","edit_type":"add"},
                                            "two":{"id":"two.png","edit_type":"remove"}})
        self.image("bagel/one.png")
        self.image("bagel/two.png")
        self.dump("imgedit_average_score.json", {"one":0})
        self.dump("imgedit_result.json", {"one":"Instruction: 0\nQuality: 0\nPreservation: 0",
                                            "two":{"error":"APIError: timeout"}})
        record = coverage_audit.audit("imgedit", self.root, metadata_file=meta)
        self.assertEqual(record["counts"]["expected"], 2)
        # Adapter fixtures use actual active filenames; absent evidence must never be guessed.
        statuses = {row["sample_id"]:row["status"] for row in record["samples"]}
        self.assertEqual(statuses["one"], "scored")
        self.assertEqual(statuses["two"], "judge_failed")

    def test_longtext_counts_each_repeat_and_does_not_expand_one_judgment(self):
        meta = self.root / "prompts.jsonl"
        meta.write_text('{"prompt_id":0,"category":"sign"}\n{"prompt_id":1,"category":"poster"}\n')
        for repeat in range(1, 5):
            self.image(f"samples/0_{repeat}.png")
        scored = dict(prompt_id=0, image=str(self.root / "samples/0_1.png"), text_accuray=0)
        result = self.root / "eval/results.jsonl"
        result.parent.mkdir()
        result.write_text(json.dumps(scored) + "\n" + json.dumps(scored) + "\n")
        record = coverage_audit.audit("longtextbench", self.root, metadata_file=str(meta), num_images=4)
        self.assertEqual(record["counts"]["expected"], 8)
        self.assertEqual(record["counts"]["valid_scored"], 1)
        self.assertEqual(record["counts"]["missing_image"], 4)
        self.assertFalse(record["coverage_complete"])
        statuses = {row["sample_id"]: row["status"] for row in record["samples"]}
        self.assertEqual(statuses["0_1"], "scored")
        self.assertEqual(statuses["0_2"], "unscored")

    def test_longtext_legacy_single_image_cannot_cover_four_repeats(self):
        meta = self.root / "prompts.jsonl"
        meta.write_text('{"prompt_id":0,"category":"sign"}\n')
        image = self.image("samples/0_sample.png")
        result = self.root / "eval/results.jsonl"
        result.parent.mkdir()
        result.write_text(json.dumps(dict(prompt_id=0, image=image, text_accuray=1)) + "\n")
        record = coverage_audit.audit("longtextbench", self.root, metadata_file=str(meta), num_images=4)
        self.assertEqual(record["counts"]["expected"], 4)
        self.assertEqual(record["counts"]["valid_scored"], 0)
        self.assertEqual(record["counts"]["missing_image"], 4)
        legacy = coverage_audit.audit("longtextbench", self.root, metadata_file=str(meta), num_images=1)
        self.assertEqual(legacy["counts"]["expected"], 1)
        self.assertEqual(legacy["counts"]["valid_scored"], 1)


if __name__ == "__main__":
    unittest.main()
