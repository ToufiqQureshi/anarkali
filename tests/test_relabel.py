from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[1]


def load_relabel():
    spec = importlib.util.spec_from_file_location("relabel_with_teachers", REPO / "scripts" / "relabel_with_teachers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


class FakeTeachers(BaseHTTPRequestHandler):
    """'good' answers with logprobs for the option whose text is 'right'; 'plain' samples
    without logprobs and prefers 'wrong' whenever the state says contested."""
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeTeachers.requests.append(body)
        prompt = body["messages"][0]["content"]
        options = dict(re.findall(r"^([A-H])\. (.+)$", prompt, flags=re.M))
        wanted = "wrong" if body["model"] == "plain" and "contested" in prompt else "right"
        letter = next(key for key, text in options.items() if text == wanted)
        if body["model"] == "good":
            top = [{"token": letter, "logprob": math.log(0.8)}]
            top += [{"token": " " + other, "logprob": math.log(0.2 / (len(options) - 1))}
                    for other in options if other != letter]
            top.append({"token": "The", "logprob": math.log(0.01)})
            choice = {"message": {"content": letter}, "logprobs": {"content": [{"top_logprobs": top}]}}
        else:
            choice = {"message": {"content": f" {letter}."}}
        payload = json.dumps({"choices": [choice]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def decision(case, texts, target, state="calm"):
    return {"case_id": f"{case}::q", "source_group": f"w::{case}", "workflow": "w",
            "state": {"note": state}, "question": "Pick one.",
            "candidates": [{"id": f"o{i}", "text": t} for i, t in enumerate(texts)], "target": target}


class RelabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.relabel = load_relabel()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTeachers)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeTeachers.requests = []
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / "in"
        self.input.mkdir()
        splits = {
            "train": [decision("a", ["wrong", "right", "meh"], [0.2, 0.7, 0.1]),
                      decision("b", ["right", "wrong"], [0.9, 0.1]),
                      decision("c", ["meh", "wrong", "right"], [0.1, 0.1, 0.8], state="contested")],
            "development": [decision("d", ["right", "wrong"], [0.5, 0.5])],
            "calibration": [decision("e", ["wrong", "right"], [0.5, 0.5])],
            "test": [decision("f", ["wrong", "right"], [0.3, 0.7])],
        }
        for name, rows in splits.items():
            (self.input / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (self.input / "manifest.json").write_text(json.dumps({"dataset": "toy", "revision": "r1"}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_relabel(self, *extra):
        return self.relabel.main(["--input", str(self.input), "--output", str(self.root / "out"),
                                  "--teacher", f"good=good@{self.url}", "--teacher", f"plain=plain@{self.url}",
                                  "--samples", "3", "--workers", "2", *extra])

    def rows(self, name):
        path = self.root / "out" / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_targets_follow_the_right_option_through_every_cyclic_order(self):
        self.run_relabel()
        by_case = {row["case_id"]: row for row in self.rows("train.jsonl")}
        for case, right in (("a::q", 1), ("b::q", 0)):
            row = by_case[case]
            winner = max(range(len(row["target"])), key=row["target"].__getitem__)
            self.assertEqual(winner, right)
            self.assertAlmostEqual(sum(row["target"]), 1.0, places=9)
            self.assertAlmostEqual(row["teacher_targets"]["good"][right], 0.8, places=6)
            self.assertEqual(row["teacher_targets"]["plain"][right], 1.0)
            self.assertEqual(row["teacher_sources"], {"good": ["logprobs"], "plain": ["sample"]})
            self.assertEqual(row["teacher_position_disagreement"]["good"], 0.0)

    def test_majority_keeps_a_row_one_teacher_disputes(self):
        self.run_relabel()
        row = next(r for r in self.rows("train.jsonl") if r["case_id"] == "c::q")
        self.assertAlmostEqual(row["teacher_agreement"], 2 / 3, places=5)
        self.assertEqual(row["original_target"], [0.1, 0.1, 0.8])

    def test_disagreement_is_dropped_and_other_splits_are_untouched(self):
        manifest = self.run_relabel("--min-agreement", "1")
        self.assertEqual([r["case_id"] for r in self.rows("dropped-train.jsonl")], ["c::q"])
        self.assertEqual(sorted(r["case_id"] for r in self.rows("train.jsonl")), ["a::q", "b::q"])
        for name in ("development", "calibration", "test"):
            self.assertEqual((self.root / "out" / f"{name}.jsonl").read_bytes(),
                             (self.input / f"{name}.jsonl").read_bytes())
        self.assertEqual(manifest["relabel_stats"]["train"]["dropped"], 1)
        self.assertEqual(manifest["split_counts"]["train"]["decision_cases"], 2)

    def test_second_run_is_served_from_the_cache(self):
        self.run_relabel()
        first = len(FakeTeachers.requests)
        self.assertGreater(first, 0)
        self.run_relabel()
        self.assertEqual(len(FakeTeachers.requests), first)

    def test_extra_body_reaches_every_request(self):
        self.run_relabel("--limit", "1", "--extra-body", '{"chat_template_kwargs": {"enable_thinking": false}}')
        self.assertTrue(FakeTeachers.requests)
        for body in FakeTeachers.requests:
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertIn(body["model"], {"good", "plain"})

    def test_test_split_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_relabel("--splits", "train,test")


if __name__ == "__main__":
    unittest.main()
