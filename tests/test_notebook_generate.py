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
    spec = importlib.util.spec_from_file_location("build_generate", REPO / "scripts" / "build_colab_notebook_generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NotebookGenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            path = cls.builder.build(Path(tmp) / "gen.ipynb")
            cls.cells = [("".join(c["source"]), c["cell_type"]) for c in json.loads(path.read_text())["cells"]]
        cls.code = [source for source, kind in cls.cells if kind == "code"]

    def test_cells_parse(self):
        for index, source in enumerate(self.code):
            ast.parse(source, filename=f"generate cell {index}")

    def test_kaggle_outputs_use_the_persistent_working_directory(self):
        setup = self.code[0]
        self.assertLess(setup.index('Path("/kaggle/working/anarkali")'),
                        setup.index('Path("/content/anarkali")'))

    def test_every_generator_flag_exists_and_domains_are_known(self):
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "generate_domain_decisions.py"), "--help"],
                             capture_output=True, text=True, check=True).stdout
        flags = set(re.findall(r"--[a-z0-9][a-z0-9-]*", out))
        call = self.code[2][self.code[2].index('run("generate_domain_decisions.py"'):]
        for flag in re.findall(r'"(--[a-z0-9-]+)"', call):
            self.assertIn(flag, flags)
        gen = importlib.util.spec_from_file_location("gen_for_nb", REPO / "scripts" / "generate_domain_decisions.py")
        module = importlib.util.module_from_spec(gen)
        sys.modules[gen.name] = module
        gen.loader.exec_module(module)
        catalog = module.load_catalogs([module.BENCHMARK_CATALOG, module.CATALOG])
        self.assertTrue(set(self.builder.DOMAINS) <= set(catalog["domains"]))
        self.assertEqual(len(self.builder.DOMAINS), 10)

    def test_wide_catalog_run_is_optional_and_uses_real_flags(self):
        self.assertIn("USE_WIDE_CATALOG = False", self.code[0])
        generate = self.code[2]
        wide = generate[generate.index("if USE_WIDE_CATALOG:"):]
        self.assertIn('"catalog_wide.json"', wide)
        self.assertNotIn('"--domains"', wide)  # every wide domain
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "generate_domain_decisions.py"), "--help"],
                             capture_output=True, text=True, check=True).stdout
        flags = set(re.findall(r"--[a-z0-9][a-z0-9-]*", out))
        for flag in re.findall(r'"(--[a-z0-9-]+)"', wide):
            self.assertIn(flag, flags)
        self.assertIn('(["wide"] if USE_WIDE_CATALOG else [])', self.code[3])

    def test_outputs_survive_the_session_and_can_resume(self):
        setup = self.code[0]
        self.assertIn('Path("/kaggle/working")', setup)
        self.assertIn("RESUME_FROM", setup)
        self.assertIn('"--dtype", "half"', self.code[1])  # T4s have no bfloat16
        self.assertIn("enable_thinking", self.code[2])

    def test_committed_notebook_is_current(self):
        committed = REPO / "notebooks" / "Anarkali_Generate.ipynb"
        with tempfile.TemporaryDirectory() as tmp:
            fresh = self.builder.build(Path(tmp) / "gen.ipynb")
            self.assertEqual(committed.read_text(encoding="utf-8"), fresh.read_text(encoding="utf-8"),
                             "run python scripts/build_colab_notebook_generate.py")


if __name__ == "__main__":
    unittest.main()
