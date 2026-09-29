import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_distill", REPO / "scripts" / "build_colab_notebook_distill.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_flags(script: str) -> set[str]:
    out = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True,
                         check=True).stdout
    return set(re.findall(r"--[a-z0-9][a-z0-9-]*", out))


class NotebookDistillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            path = cls.builder.build(Path(tmp) / "distill.ipynb")
            cls.cells = [("".join(c["source"]), c["cell_type"]) for c in json.loads(path.read_text())["cells"]]
        cls.code = [source for source, kind in cls.cells if kind == "code"]

    def test_cells_parse(self):
        for index, source in enumerate(self.code):
            ast.parse(source, filename=f"distill cell {index}")

    def test_test_splits_are_read_only_after_the_winner_is_fixed(self):
        setup, train, final = self.code
        self.assertIn("WINNER =", train)
        self.assertIn('"--dev-data", BASE', train)  # epochs are selected on benchmark development cases
        for early in (setup, train):
            self.assertNotIn("evaluate_checkpoint.py", early)
            self.assertNotIn("realworld_benchmark.py", early)
            self.assertNotIn("test.jsonl", early)
        self.assertIn("evaluate_checkpoint.py", final)

    def test_every_flag_exists(self):
        calls = re.findall(r'run\("([a-z_]+\.py)"(.*?)\)\n', "\n".join(self.code), flags=re.S)
        self.assertGreaterEqual(len(calls), 8)
        cache = {}
        for script, args in calls:
            flags = cache.setdefault(script, script_flags(script))
            for flag in re.findall(r'"(--[a-z0-9-]+)"', args):
                self.assertIn(flag, flags, f"{script}: {flag}")
        train_flags = script_flags("train_anarkali.py")
        for flag in [a for a in self.builder.STUDENT_ARGS + self.builder.KD_ARGS if a.startswith("--")]:
            self.assertIn(flag, train_flags)
        for flag in ("--teacher-checkpoint", "--dev-data", "--max-train-rows"):
            self.assertIn(flag, train_flags)

    def test_the_test_split_is_never_teacher_labelled(self):
        setup = self.code[0]
        self.assertIn('"--splits", "train,development,calibration"', setup)
        self.assertNotIn('"--splits", "train,development,calibration,test"', setup)

    def test_committed_notebook_is_current(self):
        committed = REPO / "notebooks" / "Anarkali_Distill.ipynb"
        with tempfile.TemporaryDirectory() as tmp:
            fresh = self.builder.build(Path(tmp) / "distill.ipynb")
            self.assertEqual(committed.read_text(encoding="utf-8"), fresh.read_text(encoding="utf-8"),
                             "run python scripts/build_colab_notebook_distill.py")


if __name__ == "__main__":
    unittest.main()
