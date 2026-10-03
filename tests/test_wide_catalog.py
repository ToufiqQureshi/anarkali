"""The 220-domain wide catalog: valid, current, disjoint from the other catalogs, and usable end to end."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WideCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load("generate_domain_decisions", REPO / "scripts" / "generate_domain_decisions.py")
        cls.builder = load("build_wide_catalog", REPO / "scripts" / "domains" / "build_wide_catalog.py")
        cls.wide = cls.gen.load_catalog(cls.gen.WIDE_CATALOG)

    def test_committed_catalog_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh = self.builder.build(Path(tmp) / "wide.json")
            self.assertEqual(self.gen.WIDE_CATALOG.read_text(encoding="utf-8"), fresh.read_text(encoding="utf-8"),
                             "run python scripts/domains/build_wide_catalog.py")

    def test_broad_and_disjoint(self):
        names = set(self.wide["domains"])
        self.assertGreaterEqual(len(names), 200)
        self.assertEqual(len(self.builder.DOMAINS), len(names))  # no name repeats
        others = self.gen.load_catalogs([self.gen.BENCHMARK_CATALOG, self.gen.CATALOG])["domains"]
        self.assertFalse(names & set(others))
        merged = self.gen.load_catalogs([self.gen.BENCHMARK_CATALOG, self.gen.CATALOG, self.gen.WIDE_CATALOG])
        self.assertEqual(len(merged["domains"]), len(names) + len(others))

    def test_every_domain_has_one_question_of_each_type(self):
        for name, domain in self.wide["domains"].items():
            kinds = {}
            for qid, question in domain["questions"].items():
                kind, text, candidates = self.gen.question_candidates(question)
                kinds[kind] = len(candidates)
                self.assertTrue(text.strip(), name)
                self.assertTrue(all(c["text"].strip() for c in candidates), name)
            self.assertEqual(set(kinds), {"choice", "noul", "score"}, name)
            self.assertTrue(3 <= kinds["choice"] <= 5, name)
            self.assertEqual(kinds["noul"], 2, name)
            self.assertEqual(kinds["score"], 4, name)

    def test_pinned_attributes_reach_every_prompt(self):
        domain = self.wide["domains"]["gst_invoice_check"]
        for index in range(12):
            plan = self.gen.call_plan("gst_invoice_check", domain, index, 20260929, 8)
            self.assertEqual(plan["attributes"]["region"], "India")
            self.assertIn("India", plan["prompt"])
        unpinned = self.gen.call_plan("expense_report_review", self.wide["domains"]["expense_report_review"], 0, 1, 8)
        self.assertIn(unpinned["attributes"]["region"], self.gen.REGIONS)

    def test_imported_replies_build_rows_for_a_wide_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "replies"
            folder.mkdir()
            cases = [{"case": {"invoice": {"supplier_gstin": f"27ABCDE{i:04d}F1Z5", "amount_inr": 1000 + i,
                                           "note": "Place of supply Maharashtra, IGST charged."}},
                      "intended": {"action": {"accept": 0.1, "request_correction": 0.7, "hold_itc": 0.15,
                                            "reject": 0.05}}} for i in range(4)]
            (folder / "gst_invoice_check__1.txt").write_text(json.dumps(cases))
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = self.gen.main(["--catalog", str(self.gen.WIDE_CATALOG), "--import-replies", str(folder),
                                          "--output", str(Path(tmp) / "out")])
            self.assertEqual(manifest["domains"]["gst_invoice_check"]["cases"], 4)
            rows = [json.loads(line) for split in ("train", "development", "calibration", "test")
                    for line in (Path(tmp) / "out" / f"{split}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4 * 3)
            self.assertTrue(all(r["label_source"] == "none" for r in rows))
            labelled = [r for r in rows if r["case_id"].endswith("::action")]
            self.assertTrue(all("generator" in r["teacher_targets"] for r in labelled))


if __name__ == "__main__":
    unittest.main()
