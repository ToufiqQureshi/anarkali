import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anarkali import Candidate, ChoiceRequest, DecisionPolicy
from anarkali.cli import main
from anarkali.evaluation import compare_predictions, summarize, validate_prediction


def request():
    return ChoiceRequest("A payment was duplicated", "Which team?", (
        Candidate("billing", "Payments"), Candidate("technical", "Software bugs"),
        Candidate("sales", "New contracts")))


def row(identity, correct=True, accepted=True, group=None):
    chosen = "a" if correct else "b"
    return {"case_id": identity, "source_group": group or identity,
            "request_sha256": "0" * 64, "expected": "a", "raw_choice": chosen,
            "choice": chosen if accepted else None,
            "probabilities": {"a": .8 if correct else .2, "b": .2 if correct else .8},
            "latency_ms": 10}


class CoreTests(unittest.TestCase):
    def test_candidate_ids_are_unique_and_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            ChoiceRequest("x", "y", (Candidate("a", "one"), Candidate("a", "two")))
        with self.assertRaises(ValueError):
            Candidate(" a", "one")
        with self.assertRaises(ValueError):
            ChoiceRequest.from_dict({"state": "x", "question": "y", "candidates": [], "typo": 1})

    def test_random_model_cannot_issue_an_accepted_decision(self):
        result = DecisionPolicy().decide(request(), [10, 0, 0])
        self.assertTrue(result["abstain"])
        self.assertEqual(result["reason"], "untrained_model")
        self.assertIsNone(result["choice"])
        self.assertFalse(result["calibration_fitted"])

    def test_candidate_order_does_not_change_policy_or_probabilities(self):
        original = request()
        scores = {"billing": 3., "technical": 0., "sales": -1.}
        baseline = DecisionPolicy().decide(original, [scores[c.id] for c in original.candidates], trained=True)
        for candidates in itertools.permutations(original.candidates):
            result = DecisionPolicy().decide(ChoiceRequest(original.state, original.question, candidates),
                [scores[c.id] for c in candidates], trained=True)
            self.assertEqual(result["choice"], "billing")
            for key in scores:
                self.assertAlmostEqual(result["probabilities"][key], baseline["probabilities"][key], places=14)

    def test_ties_abstain_instead_of_selecting_first_candidate(self):
        policy = DecisionPolicy(min_probability=0, min_margin=0)
        for candidates in itertools.permutations(request().candidates):
            result = policy.decide(ChoiceRequest("x", "y", candidates), [0, 0, 0], trained=True)
            self.assertEqual(result["reason"], "ambiguous_tie")
            self.assertIsNone(result["choice"])

    def test_no_supported_candidate_overrides_high_probability(self):
        result = DecisionPolicy().decide(request(), [10, 0, 0], trained=True, answerability=.01)
        self.assertEqual(result["reason"], "no_supported_candidate")

    def test_extreme_logits_remain_normalized(self):
        result = DecisionPolicy().decide(request(), [1e300, 0, -1e300], temperature=1e-300)
        self.assertEqual(sum(result["probabilities"].values()), 1)
        self.assertEqual(result["probabilities"]["billing"], 1)
        self.assertTrue(math.isfinite(result["normalized_entropy"]))
        for invalid in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                DecisionPolicy().decide(request(), [invalid, 0, 0])

    def test_package_import_does_not_load_torch(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; sys.path.insert(0,'src'); import anarkali; print('torch' in sys.modules)"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_cli_validates_example_without_claiming_prediction(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = main(["validate-request", str(Path(__file__).resolve().parents[1] / "examples/choice.json")])
        self.assertEqual(status, 0)
        self.assertFalse(json.loads(stream.getvalue())["prediction_produced"])


class EvaluationTests(unittest.TestCase):
    def test_coverage_prevents_rejected_correct_cases_being_hidden(self):
        report = summarize([row("1"), row("2", correct=False, accepted=False), row("3", accepted=False)])
        self.assertAlmostEqual(report["accuracy"], 2/3)
        self.assertAlmostEqual(report["coverage"], 1/3)
        self.assertEqual(report["accepted_accuracy"], 1)
        empty_accepts = summarize([row("1", accepted=False)])
        self.assertIsNone(empty_accepts["accepted_accuracy"])

    def test_identical_models_have_zero_paired_difference(self):
        records = [row("1"), row("2", correct=False), row("3")]
        report = compare_predictions(records, records[::-1], bootstrap_samples=100)
        self.assertEqual(report["accuracy_delta"], 0)
        self.assertEqual(report["accuracy_delta_cluster_bootstrap_95_interval"], [0, 0])
        self.assertFalse(report["superiority_established"])

    def test_incompatible_inputs_cannot_be_compared(self):
        baseline = [row("1"), row("2")]
        changed = [row("1"), row("2")]
        changed[1]["request_sha256"] = "1" * 64
        with self.assertRaises(ValueError):
            compare_predictions(baseline, changed, bootstrap_samples=100)
        with self.assertRaises(ValueError):
            compare_predictions(baseline, changed[:1], bootstrap_samples=100)
        with self.assertRaises(ValueError):
            compare_predictions(baseline + [baseline[0]], baseline, bootstrap_samples=100)

    def test_one_source_group_does_not_get_a_fake_interval(self):
        rows = [row("1", group="same"), row("2", group="same")]
        report = compare_predictions(rows, rows, bootstrap_samples=100)
        self.assertIsNone(report["accuracy_delta_cluster_bootstrap_95_interval"])

    def test_invalid_probabilities_and_choices_are_rejected(self):
        invalid = row("1")
        invalid["probabilities"]["a"] = float("nan")
        with self.assertRaises(ValueError):
            validate_prediction(invalid)
        invalid = row("1")
        invalid["choice"] = "b"
        with self.assertRaises(ValueError):
            validate_prediction(invalid)


if __name__ == "__main__":
    unittest.main()
