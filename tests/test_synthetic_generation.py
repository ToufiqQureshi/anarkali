"""Generator v1: attributes, answer steering, generator labels, parallel calls and imported chat replies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "calibration", "test")


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_all(out):
    return [json.loads(line) for split in SPLITS for line in (out / f"{split}.jsonl").read_text().splitlines()]


class SyntheticGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load("generate_domain_decisions")
        cls.catalog = cls.gen.load_catalogs([cls.gen.BENCHMARK_CATALOG, cls.gen.CATALOG])

    def test_benchmark_catalog_matches_the_benchmark_questions(self):
        domains = self.gen.load_catalog(self.gen.BENCHMARK_CATALOG)["domains"]
        self.assertEqual(set(domains), {"customer_service", "invoice_processing", "security_incidents",
                                        "agent_trace_observability"})
        for domain in domains.values():
            self.assertEqual(len(domain["questions"]), 5)
            self.assertIsInstance(domain["example_state"], dict)
        self.assertEqual(len(self.catalog["domains"]), 24)

    def test_prompts_vary_and_steer_through_every_option(self):
        domain = self.catalog["domains"]["ecommerce_return"]
        plans = [self.gen.call_plan("ecommerce_return", domain, i, 7, 8) for i in range(40)]
        self.assertGreater(len({p["attributes"]["industry"] for p in plans}), 5)
        self.assertGreater(len({p["attributes"]["region"] for p in plans}), 5)
        self.assertEqual({p["difficulty"].split(",")[0] for p in plans},
                         {"clear-cut", "borderline", "misleading", "incomplete", "conflicting"})
        steered = {p["steer"].split('"')[1] for p in plans}
        self.assertEqual(steered, set(domain["questions"]["resolution"]["criteria"]))
        self.assertEqual(plans[3]["prompt"], self.gen.call_plan("ecommerce_return", domain, 3, 7, 8)["prompt"])
        example = self.gen.call_plan("invoice_processing", self.catalog["domains"]["invoice_processing"], 0, 7, 8)
        self.assertIn('"purchase_order"', example["prompt"])  # the benchmark structure is shown
        self.assertNotIn("Write them as", example["prompt"])

    def test_generator_labels_become_a_teacher_and_bad_ones_are_ignored(self):
        domain = self.catalog["domains"]["it_helpdesk"]
        good = {qid: {c["id"]: (1.0 if i == 0 else 0.0) for i, c in enumerate(self.gen.question_candidates(q)[2])}
                for qid, q in domain["questions"].items()}
        bad = dict(good, route={"nonsense": 1.0})
        reply = json.dumps([{"case": {"ticket": "VPN drops every hour " * 3}, "intended": good},
                            {"case": {"ticket": "Printer on floor 3 jams " * 3, "expected_answer": "x"},
                             "intended": bad},
                            {"ticket": "legacy bare case without labels " * 2}])
        rows, kept, dropped = self.gen.collect("it_helpdesk", domain, [(reply, {"style": "t"})], set(), 10)
        self.assertEqual((kept, dropped), (3, 0))
        by_case = {}
        for row in rows:
            by_case.setdefault(row["source_group"], []).append(row)
        labelled = [sum("teacher_targets" in r for r in group) for group in by_case.values()]
        self.assertEqual(sorted(labelled), [0, 2, 3])  # full, one bad question dropped, none
        for row in rows:
            self.assertEqual(row["label_source"], "none")
            self.assertNotIn("expected_answer", row["state"])
            if "teacher_targets" in row:
                self.assertAlmostEqual(sum(row["teacher_targets"]["generator"]), 1.0)

    def test_imported_chat_replies_build_a_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "replies"
            folder.mkdir()
            cases = [{"case": {"email": f"Hello, I want pricing for {i} seats of your product."}} for i in range(6)]
            (folder / "email_intent__run1.txt").write_text("Sure!\n```json\n" + json.dumps(cases) + "\n```")
            (folder / "email_intent__run2.txt").write_text(json.dumps(cases[:2]))  # duplicates are kept once
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = self.gen.main(["--catalog", str(self.gen.CATALOG), "--import-replies", str(folder),
                                          "--output", str(Path(tmp) / "out")])
            self.assertEqual(manifest["domains"]["email_intent"]["cases"], 6)
            self.assertEqual(manifest["domains"]["email_intent"]["dropped"], 2)
            self.assertEqual(len(read_all(Path(tmp) / "out")), 6 * 3)
            (folder / "unknown_domain__x.txt").write_text("[]")
            with self.assertRaises(SystemExit):
                self.gen.main(["--import-replies", str(folder), "--output", str(Path(tmp) / "o2")])

    def test_print_prompt_needs_no_teacher(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.gen.main(["--catalog", str(self.gen.BENCHMARK_CATALOG), "--print-prompt", "security_incidents"])
        self.assertIn('id "disposition"', out.getvalue())
        self.assertIn('"intended"', out.getvalue())


if __name__ == "__main__":
    unittest.main()
