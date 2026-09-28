import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anarkali.typed import question_candidates  # noqa: E402
from anarkali.workflows.coding import QUESTIONS, sample_case  # noqa: E402

EXPECTED = {
    "coding_ci_failure": {
        "cause": ("choice", ["flaky_test", "real_regression", "infra_outage", "dependency_change", "config_error"]),
        "action": ("choice", ["rerun", "fix_code", "quarantine_test", "escalate_infra", "revert_commit"]),
        "blocks_release": ("noul", ["false", "true"]),
        "urgency": ("score", ["0", "1", "2", "3"]),
    },
    "coding_pr_triage": {
        "review_decision": ("choice", ["approve", "request_changes", "needs_senior_review", "split_pr"]),
        "category": ("choice", ["feature", "bugfix", "refactor", "docs", "dependency", "test"]),
        "needs_security_review": ("noul", ["false", "true"]),
        "risk": ("score", ["0", "1", "2", "3"]),
    },
    "coding_agent_step": {
        "next_action": ("choice", ["continue", "retry_with_fix", "revert_last_change", "ask_human", "stop_done"]),
        "constraint_violation": ("noul", ["false", "true"]),
        "progress": ("score", ["0", "1", "2", "3"]),
    },
}


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_coding_decisions",
                                                  REPO / "scripts" / "generate_coding_decisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CodingSchemaTests(unittest.TestCase):
    def test_ids_match_shared_schema(self):
        self.assertEqual(set(QUESTIONS), set(EXPECTED))
        for workflow, questions in EXPECTED.items():
            self.assertEqual(list(QUESTIONS[workflow]), list(questions))
            for qid, (kind, labels) in questions.items():
                parsed_kind, text, candidates = question_candidates(QUESTIONS[workflow][qid])
                self.assertEqual(parsed_kind, kind)
                self.assertEqual([c["id"] for c in candidates], labels)
                self.assertEqual(text.count("True or false:"), 1 if kind == "noul" else 0)

    def test_sampling_is_deterministic_and_hides_latent_factors(self):
        for workflow in QUESTIONS:
            first, second = sample_case(workflow, "s:1"), sample_case(workflow, "s:1")
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
            _, state, _, gold = first
            self.assertFalse(any(key.startswith("latent") for key in state))
            for qid, distribution in gold.items():
                self.assertAlmostEqual(sum(distribution.values()), 1.0, places=6)


class GeneratorTests(unittest.TestCase):
    def test_rows_split_by_source_group(self):
        generator = load_generator()
        rows = generator.build_rows(12, 7)
        splits = generator.split_by_source_group(rows, 7)
        groups = {name: {r["source_group"] for r in part} for name, part in splits.items()}
        names = list(groups)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertFalse(groups[left] & groups[right])
        self.assertEqual(sum(len(p) for p in splits.values()), len(rows))
        for row in rows:
            self.assertEqual(len(row["target"]), len(row["candidates"]))
            self.assertAlmostEqual(sum(row["target"]), 1.0, places=4)

    def test_cyclic_orders_map_back_to_original_options(self):
        generator = load_generator()
        row = {"candidates": [{"id": x, "text": x} for x in "abcd"]}
        for offset in range(3):
            texts = generator.cyclic_texts(row, offset)
            for position, text in enumerate(texts):
                self.assertEqual(text, "abcd"[(position - offset) % 4])

    def test_rows_pack_with_cached_minilm(self):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/MiniLM-L12-H384-uncased", revision="44acabbec0ef496f6dbc93adadea57f376b7c0ec",
                cache_dir=str(REPO / ".cache" / "huggingface"), local_files_only=True)
        except Exception as exc:
            self.skipTest(f"MiniLM tokenizer not cached: {exc}")
        from anarkali.packing import pack_row
        generator = load_generator()
        for row in generator.build_rows(5, 3):
            ids, spans, stats = pack_row(row, tokenizer, 512)
            self.assertEqual(len(spans), len(row["candidates"]))
            self.assertEqual(stats["state_tokens_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
