import json
from pathlib import Path
import unittest

from anarkali.typed import question_candidates


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "release" / "anarkali-lite-v4"
CHECKPOINT = ROOT / "artifacts" / "lite-20260928-063656" / "best.pt"
DEV_ROWS = ROOT / "artifacts" / "typed-decisions-v1" / "development.jsonl"


def load_release_engine():
    if not RELEASE.exists():
        raise unittest.SkipTest("release directory is missing")
    try:
        from anarkali.engine import Engine
        return Engine.load(RELEASE)
    except Exception as exc:
        raise unittest.SkipTest(f"release engine unavailable: {exc}") from exc


class EngineTests(unittest.TestCase):
    def test_predict_all_question_types(self):
        engine = load_release_engine()
        result = engine.predict({"summary": "PR changes auth checks and adds tests."}, {
            "decision": {
                "type": "choice",
                "instructions": "Choose a review decision.",
                "criteria": {"approve": "Approve.", "request_changes": "Request changes."},
            },
            "security": {
                "type": "noul",
                "instructions": "This needs a security review.",
            },
            "risk": {
                "type": "score",
                "instructions": "Estimate risk.",
                "criteria": ["trivial", "low", "moderate", "high"],
            },
        })
        self.assertEqual(set(result["answers"]), {"decision", "security", "risk"})
        choice = result["answers"]["decision"]
        self.assertEqual(choice["type"], "choice")
        self.assertAlmostEqual(sum(choice["probabilities"].values()), 1.0, places=3)
        self.assertIn(choice["choice"], choice["probabilities"])
        self.assertEqual(result["answers"]["security"]["type"], "noul")
        self.assertEqual(result["answers"]["risk"]["type"], "score")
        self.assertAlmostEqual(sum(result["answers"]["risk"]["probabilities"].values()), 1.0, places=3)

    def test_torch_vs_onnx_argmax_on_dev_rows(self):
        if not CHECKPOINT.exists() or not DEV_ROWS.exists() or not RELEASE.exists():
            self.skipTest("checkpoint, dev rows or release directory is missing")
        try:
            import numpy as np
            from anarkali.engine import Engine
            from anarkali.packing import pack_row
        except Exception as exc:
            self.skipTest(f"engine parity dependencies unavailable: {exc}")
        try:
            torch_engine = Engine.load(CHECKPOINT)
            onnx_engine = Engine.load(RELEASE, graph="model.onnx")
        except Exception as exc:
            self.skipTest(f"torch or onnx engine unavailable: {exc}")
        rows = [json.loads(line) for line in DEV_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()][:20]
        for row in rows:
            packed_torch = [pack_row(row, torch_engine.backend.tokenizer, torch_engine.max_tokens)]
            packed_onnx = [pack_row(row, onnx_engine.backend.tokenizer, onnx_engine.max_tokens)]
            torch_logits = torch_engine.backend.logits(packed_torch)[0]
            onnx_logits = onnx_engine.backend.logits(packed_onnx)[0]
            self.assertEqual(int(np.argmax(torch_logits)), int(np.argmax(onnx_logits)))


class EngineShapeHelperTests(unittest.TestCase):
    def test_example_questions_are_valid(self):
        request_dir = ROOT / "examples" / "requests"
        for path in sorted(request_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.name):
                self.assertIsInstance(payload["questions"], dict)
                for question in payload["questions"].values():
                    kind, _text, candidates = question_candidates(question)
                    self.assertIn(kind, {"choice", "noul", "score"})
                    self.assertGreaterEqual(len(candidates), 2)


class ResolveTests(unittest.TestCase):
    def test_local_paths_stay_local_and_repo_ids_download(self):
        from unittest.mock import patch
        from anarkali.engine import _resolve
        self.assertEqual(_resolve(ROOT, None), ROOT)
        self.assertEqual(_resolve("model-dir", None), Path("model-dir"))
        with patch("huggingface_hub.snapshot_download", return_value="/cache/snap") as download:
            self.assertEqual(_resolve("owner/anarkali-v3", None), Path("/cache/snap"))
        self.assertEqual(download.call_args.args[0], "owner/anarkali-v3")


if __name__ == "__main__":
    unittest.main()
