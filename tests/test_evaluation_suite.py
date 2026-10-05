from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import ValidationError

from src.evaluation_suite import (
    EvaluationDataset,
    baseline_from_report,
    compare_with_baseline,
    evaluate_generation_cases,
    load_evaluation_dataset,
    run_offline_evaluation,
    write_evaluation_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evals" / "dataset.json"
BASELINE_PATH = PROJECT_ROOT / "evals" / "baseline.json"


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_evaluation_dataset(DATASET_PATH)

    def test_golden_dataset_has_meaningful_coverage(self):
        self.assertGreaterEqual(len(self.dataset.corpus), 8)
        self.assertGreaterEqual(len(self.dataset.retrieval_cases), 12)
        self.assertGreaterEqual(len(self.dataset.tool_selection_cases), 10)
        self.assertGreaterEqual(len(self.dataset.generation_cases), 6)
        self.assertGreaterEqual(len(self.dataset.tool_events), 10)

    def test_dataset_rejects_unknown_document_references(self):
        data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        data["retrieval_cases"][0]["expected_document_ids"] = ["missing-document"]

        with self.assertRaises(ValidationError):
            EvaluationDataset.model_validate(data)


class EvaluationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_evaluation_dataset(DATASET_PATH)

    def test_offline_evaluation_is_deterministic(self):
        first = run_offline_evaluation(self.dataset)
        second = run_offline_evaluation(self.dataset)

        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["case_results"], second["case_results"])
        for value in first["metrics"].values():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

    def test_challenge_set_retains_known_failures(self):
        report = run_offline_evaluation(self.dataset)
        failed_cases = [
            case
            for cases in report["case_results"].values()
            for case in cases
            if not case["passed"]
        ]

        self.assertGreaterEqual(len(failed_cases), 3)

    def test_checked_in_baseline_passes(self):
        report = run_offline_evaluation(self.dataset)
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

        regression = compare_with_baseline(
            report,
            baseline,
            self.dataset.thresholds,
        )

        self.assertTrue(regression["passed"])
        self.assertTrue(all(item["passed"] for item in regression["metrics"].values()))

    def test_quality_drop_fails_regression_gate(self):
        report = run_offline_evaluation(self.dataset)
        baseline = baseline_from_report(report)
        degraded = deepcopy(report)
        degraded["metrics"]["groundedness"] = 0.7

        regression = compare_with_baseline(
            degraded,
            baseline,
            self.dataset.thresholds,
        )

        self.assertFalse(regression["passed"])
        self.assertFalse(regression["metrics"]["groundedness"]["passed"])

    def test_missing_citation_reduces_generation_score(self):
        broken_case = self.dataset.generation_cases[0].model_copy(
            update={
                "generated_text": "Multi-head attention uses queries, keys, and values."
            }
        )
        changed_dataset = self.dataset.model_copy(
            update={"generation_cases": [broken_case]}
        )

        result = evaluate_generation_cases(changed_dataset)

        self.assertEqual(result["metrics"]["citation_completeness"], 0.0)
        self.assertFalse(result["cases"][0]["passed"])

    def test_json_and_markdown_reports_are_written(self):
        report = run_offline_evaluation(self.dataset)
        baseline = baseline_from_report(report)
        report["regression"] = compare_with_baseline(
            report,
            baseline,
            self.dataset.thresholds,
        )

        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "report.json"
            markdown_path = Path(directory) / "report.md"
            write_evaluation_report(report, json_path, markdown_path)

            saved = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertTrue(saved["regression"]["passed"])
        self.assertIn("Regression status: **PASS**", markdown)
        self.assertIn("retrieval_hit_rate", markdown)
        self.assertIn("## Failed cases", markdown)
        self.assertIn("retrieval-faiss-lexical-trap", markdown)


if __name__ == "__main__":
    unittest.main()
