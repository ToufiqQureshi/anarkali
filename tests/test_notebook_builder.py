import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "Anarkali_V3.ipynb"


def load_cells():
    return [("".join(c["source"]), c["cell_type"])
            for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]


def embedded_files():
    for source, _ in load_cells():
        if "PAYLOAD = (" in source:
            body = source[source.index("PAYLOAD = ("):]
            chunks = re.findall(r"'([A-Za-z0-9+/=]+)'", body[:body.index(")")])
            return json.loads(gzip.decompress(base64.b64decode("".join(chunks))).decode("utf-8"))
    raise AssertionError("no embedded payload")


class NotebookBuilderTests(unittest.TestCase):
    def setUp(self):
        if not NOTEBOOK.exists():
            self.skipTest("build it with scripts/build_colab_notebook.py")

    def test_every_code_cell_parses(self):
        for index, (source, kind) in enumerate(load_cells()):
            if kind == "code":
                ast.parse(source, filename=f"{NOTEBOOK.name} cell {index}")

    def test_embedded_data_matches_each_manifest(self):
        files = embedded_files()
        manifests = [name for name in files if name.endswith("manifest.json")]
        self.assertEqual(len(manifests), 3)
        for manifest_name in manifests:
            manifest = json.loads(files[manifest_name])
            base = manifest_name.rsplit("/", 1)[0]
            for split, counts in manifest["split_counts"].items():
                key = f"{base}/{split}.jsonl"
                if key in files:
                    self.assertEqual(hashlib.sha256(files[key].encode("utf-8")).hexdigest(), counts["sha256"], key)

    def test_training_data_never_includes_the_final_split(self):
        files = embedded_files()
        self.assertNotIn("artifacts/anarkali-decisions-v3/" + "te" + "st.jsonl", files)

    def test_final_split_is_read_only_after_the_winner_is_fixed(self):
        run_cell = next(source for source, _ in load_cells() if "WINNER =" in source)
        self.assertLess(run_cell.index("WINNER ="), run_cell.index("evaluate_checkpoint.py"))
        self.assertNotIn("evaluate_checkpoint.py", run_cell[:run_cell.index("WINNER =")])

    def test_embedded_source_matches_repo(self):
        for name, text in embedded_files().items():
            if name.endswith(".py"):
                self.assertEqual(text, (REPO / name).read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
