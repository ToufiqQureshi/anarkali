import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def load_suite():
    name = "evaluate_release_suite"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    name = "fake"

    def __init__(self, answers):
        self.answers = answers

    def predict(self, state, questions):
        choice = self.answers[state["case_id"]]
        return {"answers": {"route": {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}}}}


class ReleaseBenchmarkTests(unittest.TestCase):
    def test_suite_file_is_valid(self):
        suite = load_suite()
        rows = suite.load_cases(REPO / "benchmarks" / "anarkali-routing-v1" / "cases.jsonl")
        self.assertGreaterEqual(len(rows), 30)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertTrue(all("route" in row["expected"] for row in rows))

    def test_evaluator_reports_failures_and_tags(self):
        suite = load_suite()
        rows = [
            {"id": "a", "state": {"case_id": "a"}, "questions": {"route": {}},
             "expected": {"route": "product"}, "tags": ["scope"]},
            {"id": "b", "state": {"case_id": "b"}, "questions": {"route": {}},
             "expected": {"route": "incident"}, "tags": ["scope"]},
        ]
        report = suite.evaluate(FakeEngine({"a": "product", "b": "product"}), rows)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["failures"][0]["id"], "b")
        self.assertEqual(report["by_tag"]["scope"]["accuracy"], 0.5)

    def test_main_enforces_accuracy_threshold(self):
        suite = load_suite()
        tmp = REPO / "benchmarks" / "anarkali-routing-v1" / "_tmp_test_cases.jsonl"
        try:
            tmp.write_text(json.dumps({"id": "a", "state": {"case_id": "a"}, "questions": {"route": {}},
                                       "expected": {"route": "product"}}) + "\n", encoding="utf-8")
            with patch("anarkali.engine.Engine.load", return_value=FakeEngine({"a": "incident"})):
                with self.assertRaises(SystemExit):
                    suite.main(["--model", "fake", "--cases", str(tmp), "--min-accuracy", "0.85"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_local_release_model_passes_when_runtime_is_available(self):
        model = REPO / "release-150m"
        if not model.exists():
            self.skipTest("release-150m is not present")
        with (model / "model.onnx").open("rb") as stream:
            if stream.read(40).startswith(b"version https://git-lfs"):
                self.skipTest("release-150m/model.onnx is a Git LFS pointer; run git lfs pull")
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"ONNX runtime dependency is unavailable: {exc}")
        suite = load_suite()
        report = suite.main(["--model", str(model),
                             "--cases", str(REPO / "benchmarks" / "anarkali-routing-v1" / "cases.jsonl"),
                             "--min-accuracy", "0.60"])
        self.assertGreaterEqual(report["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()
