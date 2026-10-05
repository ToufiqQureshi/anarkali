"""Data tooling and training objectives kept for the web decisions pipeline: merging decision sets
and the distillation losses in anarkali.objectives."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "calibration", "test")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def decision(case, texts, target, state="calm", question="Pick one.", **extra):
    return {"case_id": f"{case}::q", "source_group": f"w::{case}", "workflow": "w", "state": {"note": state},
            "question": question, "candidates": [{"id": f"o{i}", "text": t} for i, t in enumerate(texts)],
            "target": target, **extra}


class MergeTests(unittest.TestCase):
    def test_gold_and_pool_may_share_groups_within_a_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind, split_of_g in (("gold", "train"), ("pool", "train"), ("bad", "development")):
                (root / kind).mkdir()
                counts = {}
                for name in SPLITS:
                    rows = [decision("g", ["x", "y"], [1.0, 0.0])] if name == split_of_g else []
                    path = root / kind / f"{name}.jsonl"
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                    counts[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                (root / kind / "manifest.json").write_text(json.dumps({"revision": kind, "split_counts": counts}))
            script = str(REPO / "scripts" / "merge_decision_sets.py")
            import subprocess
            ok = subprocess.run([sys.executable, script, "--inputs", str(root / "gold"), str(root / "pool"),
                                 "--output", str(root / "merged")], capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual(len(read_rows(root / "merged" / "train.jsonl")), 2)
            bad = subprocess.run([sys.executable, script, "--inputs", str(root / "gold"), str(root / "bad"),
                                  "--output", str(root / "m2")], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("appears in", bad.stderr)

    def test_unicode_line_separator_inside_json_string_is_not_a_record_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            counts = {}
            for name in SPLITS:
                rows = [decision("unicode", ["x", "y"], [1.0, 0.0])] if name == "train" else []
                if rows:
                    rows[0]["state"] = {"text": "before\u2028after\u2029still one JSON record"}
                path = source / f"{name}.jsonl"
                path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                counts[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (source / "manifest.json").write_text(
                json.dumps({"revision": "unicode", "split_counts": counts}), encoding="utf-8")
            script = str(REPO / "scripts" / "merge_decision_sets.py")
            import subprocess
            result = subprocess.run([sys.executable, script, "--inputs", str(source),
                                     "--output", str(root / "merged")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            merged = read_rows(root / "merged" / "train.jsonl")
            self.assertEqual(merged[0]["state"]["text"], "before\u2028after\u2029still one JSON record")


class DistillationTests(unittest.TestCase):
    def test_losses(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from anarkali.objectives import distillation_kl, option_vector_loss
        logits = torch.tensor([[2.0, 0.0, float("-inf")], [0.5, 0.1, 0.3]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        self.assertLess(float(distillation_kl(logits, logits, mask)), 1e-6)
        self.assertGreater(float(distillation_kl(logits, logits.flip(-1).nan_to_num(neginf=0.0), mask)), 0.01)
        vectors = torch.randn(2, 3, 4)
        self.assertLess(float(option_vector_loss(vectors, 3 * vectors, mask)), 1e-5)
        self.assertGreater(float(option_vector_loss(vectors, -vectors, mask)), 1.9)


if __name__ == "__main__":
    unittest.main()
