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


def load_benchmark():
    name = "benchmark_release"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def row(i, right, kind="choice"):
    texts = ["wrong", "meh", "okay"]
    texts.insert(right, "right")
    target = [0.1] * 4
    target[right] = 0.7
    out = {"case_id": f"c{i}::q", "source_group": f"w::c{i}", "workflow": "w", "state": {"n": "x" * 60},
           "question": "Choose from the options.\nPick one.",
           "candidates": [{"id": f"o{j}", "text": t} for j, t in enumerate(texts)], "target": target}
    if kind != "choice":
        out["question_type"] = kind
    return out


class BenchmarkTests(unittest.TestCase):
    def test_variants_calibration_and_report(self):
        bench = load_benchmark()
        rows = [row(i, i % 4, "score" if i % 3 == 0 else "choice") for i in range(24)]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for split in ("calibration", "test"):
                (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            def fake_load(path, **kwargs):
                return Engine(PositionBiasedBackend(), name="fake", max_tokens=kwargs.get("max_tokens") or 256)

            with patch.object(Engine, "load", side_effect=fake_load):
                report = bench.main(["--model", "fake", "--data", str(data), "--output", str(Path(tmp) / "out"),
                                     "--variants", "orders=1", "orders=4", "orders=1,max_tokens=300"])
            one, four = report["variants"]["orders=1"], report["variants"]["orders=4"]
            # the biased first slot wins once in four without averaging; averaging finds the right option
            self.assertAlmostEqual(one["test_raw"]["all"]["accuracy"], 0.25)
            self.assertAlmostEqual(four["test_raw"]["all"]["accuracy"], 1.0)
            self.assertEqual(set(four["temperatures"]), {"choice", "score"})
            for variant in (one, four):
                kl = variant["test_calibrated"]["all"]["kl_from_gold"]
                self.assertGreaterEqual(kl, -1e-9)
                self.assertLess(kl, variant["test_calibrated"]["all"]["soft_ce"])
            self.assertIn("type:score", four["test_calibrated"])
            self.assertEqual(report["variants"]["orders=1,max_tokens=300"]["max_tokens"], 300)
            table = (Path(tmp) / "out" / "benchmark.md").read_text(encoding="utf-8")
            self.assertIn("`orders=4`", table)

    def test_rescale_and_fit(self):
        bench = load_benchmark()
        self.assertEqual([round(x, 6) for x in bench.rescale([0.25, 0.75], 1.0)], [0.25, 0.75])
        sharp = bench.rescale([0.25, 0.75], 0.5)
        self.assertGreater(sharp[1], 0.75)
        rows = [{"target": [0.5, 0.5]}] * 4
        # over-confident predictions against flat targets need a high temperature
        self.assertGreater(bench.fit_temperatures(rows, [[0.05, 0.95]] * 4)["choice"], 2.0)

    def test_bad_variant(self):
        bench = load_benchmark()
        with self.assertRaises(SystemExit):
            bench.parse_variant("beams=2")


if __name__ == "__main__":
    unittest.main()
