from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[1]


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeGenerator(BaseHTTPRequestHandler):
    """Answers every generation prompt with a fenced JSON array; one case leaks an answer, one repeats."""
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        FakeGenerator.calls.append(prompt)
        seed = re.search(r"variety seed (\d+)", prompt).group(1)
        batch = int(re.search(r"Write (\d+) different cases", prompt).group(1))
        cases = [{"record": {"id": f"{seed}-{i}", "amount": 100 + i, "note": "customer writes about the order " * 2},
                  "recommended_action": "approve"} for i in range(batch)]
        cases.append(cases[0])  # a duplicate
        text = "Here you go:\n```json\n" + json.dumps(cases) + "\n```"
        payload = json.dumps({"choices": [{"message": {"content": text}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class DomainGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load("generate_domain_decisions")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGenerator)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_catalog_is_valid_and_broad(self):
        catalog = self.gen.load_catalog(self.gen.CATALOG)
        self.assertGreaterEqual(len(catalog["domains"]), 20)
        kinds = {q["type"] for d in catalog["domains"].values() for q in d["questions"].values()}
        self.assertEqual(kinds, {"choice", "noul", "score"})

    def test_parse_and_scrub(self):
        self.assertEqual(self.gen.parse_cases("no json here"), [])
        self.assertEqual(self.gen.parse_cases('```json\n[{"a": 1}, 2]\n```'), [{"a": 1}])
        case = self.gen.scrub({"record": {"x": "y" * 50, "correct_answer": "b"}, "Recommendation": "approve"})
        self.assertEqual(case, {"record": {"x": "y" * 50}})
        self.assertIsNone(self.gen.scrub({"x": "y" * 5000}))

    def test_generation_end_to_end_and_resume(self):
        FakeGenerator.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            argv = ["--teacher", f"gen=fake@{self.url}", "--domains", "ecommerce_return,it_helpdesk",
                    "--cases-per-domain", "7", "--batch", "3", "--output", str(out)]
            manifest = self.gen.main(argv)
            first_calls = len(FakeGenerator.calls)
            self.assertEqual({d: s["cases"] for d, s in manifest["domains"].items()},
                             {"ecommerce_return": 7, "it_helpdesk": 7})
            rows = [json.loads(line) for split in ("train", "development", "calibration", "test")
                    for line in (out / f"{split}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 14 * 3)  # every case once per question
            for row in rows:
                self.assertEqual(row["label_source"], "none")
                self.assertNotIn("recommended_action", row["state"])
                self.assertEqual(len(set(row["target"])), 1)
            groups = {}
            for split in ("train", "development", "calibration", "test"):
                groups[split] = {json.loads(line)["source_group"] for line in (out / f"{split}.jsonl").read_text().splitlines()}
            self.assertFalse(groups["train"] & groups["test"])
            self.assertTrue(any("borderline" in p for p in FakeGenerator.calls))
            self.gen.main(argv)  # served from the cache
            self.assertEqual(len(FakeGenerator.calls), first_calls)

    def test_generated_test_split_can_be_labelled_but_real_ones_cannot(self):
        relabel = load("relabel_with_teachers")
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            row = {"case_id": "x::q", "source_group": "d::x", "workflow": "d", "state": {"a": 1},
                   "question": "Pick.", "candidates": [{"id": "a", "text": "right"}, {"id": "b", "text": "wrong"}],
                   "target": [0.5, 0.5], "label_source": "none"}
            for split in ("train", "development", "calibration", "test"):
                (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (data / "manifest.json").write_text("{}", encoding="utf-8")
            base = ["--input", str(data), "--output", str(Path(tmp) / "o"), "--splits", "train,test",
                    "--teacher", "t=m@http://offline.invalid/v1", "--offline"]
            with self.assertRaises(SystemExit):
                relabel.main(base)  # the original target would vote
            with self.assertRaises(relabel.CacheMiss):
                relabel.main(base + ["--no-original"])  # allowed; only the missing scores stop it
            (data / "test.jsonl").write_text(json.dumps(dict(row, label_source="teacher")) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                relabel.main(base + ["--no-original"])


if __name__ == "__main__":
    unittest.main()
