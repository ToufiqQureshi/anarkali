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
    spec = importlib.util.spec_from_file_location("build_v4", REPO / "scripts" / "build_colab_notebook_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_flags(script: str) -> set[str]:
    out = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True,
                         check=True).stdout
    return set(re.findall(r"--[a-z0-9][a-z0-9-]*", out))


class NotebookV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            path = cls.builder.build(Path(tmp) / "v4.ipynb")
            cls.cells = [("".join(c["source"]), c["cell_type"]) for c in json.loads(path.read_text())["cells"]]
        cls.code = [source for source, kind in cls.cells if kind == "code"]

    def test_cells_parse(self):
        for index, source in enumerate(self.code):
            ast.parse(source, filename=f"v4 cell {index}")

    def test_final_splits_are_read_only_after_the_winner_is_fixed(self):
        setup, train, final = self.code
        self.assertIn("WINNER =", train)
        for early in (setup, train):
            self.assertNotIn("evaluate_checkpoint.py", early)
            self.assertNotIn("test.jsonl", early)
        self.assertIn("evaluate_checkpoint.py", final)

    def test_every_flag_exists(self):
        train_flags = script_flags("train_anarkali.py")
        for recipe in self.builder.RECIPES:
            for flag in [a for a in self.builder.BASE + recipe["args"] if a.startswith("--")]:
                self.assertIn(flag, train_flags, f"{recipe['name']}: {flag}")
        final = self.code[2]
        for script in ("export_onnx.py", "evaluate_checkpoint.py"):
            flags = script_flags(script)
            call = final[final.index(f'run("{script}"'):]
            call = call[:call.index(")\n")]
            for flag in re.findall(r'"(--[a-z-]+)"', call):
                self.assertIn(flag, flags, f"{script}: {flag}")
        for script, flag in (("merge_decision_sets.py", "--inputs"), ("import_agent_traces.py", "--per-source"),
                             ("prepare_typed_decisions.py", "--question-types")):
            self.assertIn(flag, script_flags(script))

    def test_baseline_recipe_is_the_released_one(self):
        baseline = self.builder.RECIPES[0]
        self.assertEqual(baseline["name"], "v3-baseline")
        self.assertFalse(any(a in baseline["args"] for a in ("--brier-weight", "--consistency-weight", "--llrd")))

    def test_backbone_bakeoff_is_opt_in(self):
        optional = [r for r in self.builder.RECIPES if r.get("optional")]
        self.assertGreaterEqual(len({r["model"] for r in optional}), 4)
        self.assertIn("if recipe.get(\"optional\") and not BACKBONE_BAKEOFF", self.code[1])
        self.assertIn("BACKBONE_BAKEOFF = False", self.code[0])

    def test_committed_notebook_is_current(self):
        committed = REPO / "notebooks" / "Anarkali_V4.ipynb"
        with tempfile.TemporaryDirectory() as tmp:
            fresh = self.builder.build(Path(tmp) / "v4.ipynb")
            self.assertEqual(committed.read_text(encoding="utf-8"), fresh.read_text(encoding="utf-8"),
                             "run scripts/build_colab_notebook_v4.py")


if __name__ == "__main__":
    unittest.main()
