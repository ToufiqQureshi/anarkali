import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from anarkali.engine import Engine
from test_engine_orders import PositionBiasedBackend

REPO = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("realworld_benchmark", REPO / "scripts" / "realworld_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["realworld_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def case(i, cause, blocks):
    return {"case_id": f"org/repo#job{i}", "workflow": "coding_ci_failure",
            "state": {"repository": "org/repo", "error_lines": ["Process completed with exit code 1.",
                                                                f"FAILED tests/test_{i}.py::test_x"]},
            "evidence": {"url": f"https://github.com/org/repo/actions/runs/1/job/{i}"},
            "gold": {"cause": cause, "blocks_release": blocks}}


class RealWorldBenchmarkTests(unittest.TestCase):
    def test_report_scores_each_labelled_question_with_evidence(self):
        bench = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            # the fake model always prefers the first option: flaky_test, and "false" for noul
            rows = [case(0, "flaky_test", "false"), case(1, "real_regression", "false"), case(2, "flaky_test", "true")]
            cases.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            def fake_load(path, **kwargs):
                backend = PositionBiasedBackend()
                backend.tokenizer.model_max_length = 4096  # one token per character; CI options are long
                return Engine(backend, name="fake", max_tokens=4096)

            with patch.object(Engine, "load", side_effect=fake_load):
                report = bench.main(["--cases", str(cases), "--model", "a=x", "--model", "b=y",
                                     "--output", str(Path(tmp) / "out")])
            summary = report["models"]["a"]["summary"]
            self.assertEqual(summary["cause"]["n"], 3)
            self.assertAlmostEqual(summary["cause"]["accuracy"], 2 / 3)
            self.assertAlmostEqual(summary["blocks_release"]["accuracy"], 2 / 3)
            self.assertAlmostEqual(summary["all"]["accuracy"], 4 / 6)
            self.assertAlmostEqual(summary["cause=real_regression"]["accuracy"], 0.0)
            self.assertAlmostEqual(report["majority_baseline"], 4 / 6)  # flaky_test 2/3, false 2/3
            table = (Path(tmp) / "out" / "report.md").read_text(encoding="utf-8")
            self.assertIn("(https://github.com/org/repo/actions/runs/1/job/1)", table)
            self.assertIn("FAILED tests/test_1.py::test_x", table)  # the exit-code line is skipped
            self.assertIn("| real_regression | ❌ flaky_test", table)

    def test_bad_labels_are_refused(self):
        bench = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            bad = case(0, "cosmic_rays", "false")
            cases.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                bench.load_cases(cases)


if __name__ == "__main__":
    unittest.main()
