"""The data-and-distillation pipeline: harvest -> teachers (checkpoint, Brio) -> combine -> distil.

Everything runs offline: the Hub, the colibri server and the encoders are replaced by fakes.
"""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "calibration", "test")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


CLINC_NAMES = ["transfer", "balance", "freeze_account", "report_fraud", "pin_change", "oos", "weather", "timer"]


def fake_loader(source, split, seed):
    if source.name == "civil_comments":
        records = [{"text": f"comment number {i} " + ("you idiot" if i % 3 == 0 else "nice point"),
                    "toxicity": 0.8 if i % 3 == 0 else 0.3, "insult": 0.7, "threat": 0.0,
                    "identity_attack": 0.0, "obscene": 0.1} for i in range(40)]
        records.append(dict(records[1]))  # a duplicate text is kept once
        return records, None
    if source.name == "clinc_oos":
        return [{"text": f"please do thing {i}", "intent": i % len(CLINC_NAMES)} for i in range(30)], CLINC_NAMES
    if source.name == "commonsense_qa":
        return [{"question": f"Where would you keep item {i}?", "question_concept": "item",
                 "choices": {"label": ["A", "B", "C"], "text": ["drawer", "ocean", "sky"]}, "answerKey": "A"}
                for i in range(12)], None
    raise AssertionError(f"unexpected source {source.name}")


class HarvestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harvest = load_script("harvest_public_decisions")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "public"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = cls.harvest.main(["--output", str(cls.out), "--sources",
                                           "civil_comments,clinc_oos,commonsense_qa", "--pool-rate", "1"],
                                          loader=fake_loader)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_are_valid_typed_decisions(self):
        from anarkali.typed import QUESTION_TYPES  # noqa: F401  (import check only)
        for kind in ("gold", "pool"):
            for name in SPLITS:
                for row in read_rows(self.out / kind / f"{name}.jsonl"):
                    self.assertEqual(len(row["target"]), len(row["candidates"]))
                    self.assertAlmostEqual(sum(row["target"]), 1.0, places=6)
                    self.assertGreaterEqual(len(row["candidates"]), 2)
                    self.assertIn(row["label_source"], ("gold", "gold-soft") if kind == "gold" else ("none",))

    def test_manifests_match_files_and_splits_share_no_group(self):
        where = {}
        for kind in ("gold", "pool"):
            manifest = json.loads((self.out / kind / "manifest.json").read_text())
            self.assertEqual({s["license"] for s in manifest["sources"]}, {"CC0-1.0", "CC-BY-3.0", "MIT"})
            for name in SPLITS:
                path = self.out / kind / f"{name}.jsonl"
                self.assertEqual(manifest["split_counts"][name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                for row in read_rows(path):
                    self.assertEqual(where.setdefault(row["source_group"], name), name)
        self.assertIn("CC0-1.0", (self.out / "CREDITS.md").read_text())

    def test_soft_labels_duplicates_and_pool(self):
        gold = [r for name in SPLITS for r in read_rows(self.out / "gold" / f"{name}.jsonl")]
        civil = [r for r in gold if r["workflow"] == "public:civil_comments"]
        toxic = [r for r in civil if r["question"].startswith("True or false: This comment is toxic")]
        self.assertTrue(any(abs(r["target"][1] - 0.8) < 1e-9 for r in toxic))  # rater share, not 0/1
        texts = [r["state"]["comment"] for r in toxic]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertGreaterEqual(self.result["per_source"]["civil_comments"]["duplicates"], 1)
        clinc = [r for r in gold if r["workflow"] == "public:clinc_oos"]
        for row in clinc:  # the gold intent is always among the offered options
            gold_id = row["candidates"][row["target"].index(1.0)]["id"]
            self.assertIn(gold_id, CLINC_NAMES)
        pool = [r for name in SPLITS for r in read_rows(self.out / "pool" / f"{name}.jsonl")]
        self.assertTrue(pool)
        self.assertEqual({r["workflow"] for r in pool}, {"public:civil_comments"})  # only sources with a domain

    def test_share_alike_needs_opt_in(self):
        with self.assertRaises(SystemExit):
            self.harvest.main(["--output", str(Path(self.tmp.name) / "x"), "--sources", "snli"], loader=fake_loader)


class FakeBrio(BaseHTTPRequestHandler):
    """Prefers the option whose text contains 'right'; answers each question of a request."""
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeBrio.requests.append((self.path, body))
        answers = []
        for q in body["questions"]:
            weights = [8.0 if "right" in o else 1.0 for o in q["options"]]
            total = sum(weights)
            answers.append({"question": q["question"], "answer": q["options"][weights.index(max(weights))],
                            "choices": [{"option": o, "p": w / total} for o, w in zip(q["options"], weights)]})
        payload = json.dumps({"object": "brio.answers", "answers": answers}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def decision(case, texts, target, state="calm", question="Pick one.", **extra):
    return {"case_id": f"{case}::q", "source_group": f"w::{case}", "workflow": "w", "state": {"note": state},
            "question": question, "candidates": [{"id": f"o{i}", "text": t} for i, t in enumerate(texts)],
            "target": target, **extra}


class BrioTeacherTests(unittest.TestCase):
    def test_brio_scores_every_question_on_a_state_in_one_request(self):
        relabel = load_script("relabel_with_teachers")
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBrio)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        FakeBrio.requests = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "in").mkdir()
                rows = {"train": [decision("a", ["wrong", "right"], [0.5, 0.5], question="Q1?"),
                                  decision("a2", ["right", "wrong", "meh"], [1 / 3] * 3, question="Q2?"),
                                  decision("b", ["same", "same"], [0.5, 0.5], state="other")]}
                for name in SPLITS:
                    (root / "in" / f"{name}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in rows.get(name, [])), encoding="utf-8")
                (root / "in" / "manifest.json").write_text(json.dumps({"dataset": "toy", "revision": "r"}))
                argv = ["--input", str(root / "in"), "--output", str(root / "out"), "--no-original",
                        "--brio-teacher", f"colibri=qwen36@{url}", "--min-agreement", "0.5"]
                with contextlib.redirect_stdout(io.StringIO()):
                    relabel.main(argv)
                out = {r["case_id"]: r for r in read_rows(root / "out" / "train.jsonl")}
                self.assertAlmostEqual(out["a::q"]["target"][1], 8 / 9)
                self.assertAlmostEqual(out["a2::q"]["target"][0], 0.8)
                self.assertEqual(out["b::q"]["target"], [0.5, 0.5])  # repeated texts sent as "id: text"
                paths = {p for p, _ in FakeBrio.requests}
                self.assertEqual(paths, {"/v1/brio"})
                self.assertEqual(len(FakeBrio.requests), 2)  # one request per distinct state
                states = [b for _, b in FakeBrio.requests if len(b["questions"]) == 2]
                self.assertEqual(len(states), 1)
                self.assertEqual(json.loads(states[0]["state"]), {"note": "calm"})
                self.assertNotIn("right", states[0]["state"])  # options never go into the prompt
                FakeBrio.requests = []
                with contextlib.redirect_stdout(io.StringIO()):
                    relabel.main(argv + ["--offline"])  # everything served from the cache
                self.assertEqual(FakeBrio.requests, [])
        finally:
            server.shutdown()
            server.server_close()


class CombineTests(unittest.TestCase):
    def test_teachers_are_calibrated_mixed_and_gold_is_kept(self):
        combine = load_script("combine_teachers")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in").mkdir()
            over = {"t": [0.99, 0.01]}  # an overconfident teacher, right 3 times in 4
            cal = [decision(f"c{i}", ["x", "y"], [1.0, 0.0] if i % 4 else [0.0, 1.0], label_source="gold",
                            teacher_targets=over) for i in range(8)]
            train = [decision("g", ["x", "y"], [0.0, 1.0], label_source="gold",
                              teacher_targets={"t": [0.99, 0.01], "u": [0.5, 0.5]}),
                     decision("p", ["x", "y"], [0.5, 0.5], label_source="none",
                              teacher_targets={"t": [0.9, 0.1], "u": [0.8, 0.2]}),
                     decision("n", ["x", "y"], [0.5, 0.5], label_source="none")]
            data = {"train": train, "calibration": cal, "development": [], "test": [dict(train[1])]}
            for name in SPLITS:
                (root / "in" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data[name]))
            (root / "in" / "manifest.json").write_text(json.dumps({"dataset": "toy", "revision": "r"}))
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = combine.main(["--input", str(root / "in"), "--output", str(root / "out"),
                                         "--gold-weight", "0.5"])
            self.assertGreater(manifest["combine"]["temperatures"]["t"], 1.5)  # softened toward 75%
            out = {r["case_id"]: r for r in read_rows(root / "out" / "train.jsonl")}
            self.assertNotIn("n::q", out)  # unlabelled and no teacher: dropped
            self.assertGreater(out["g::q"]["target"][1], 0.5)  # the human label still wins
            self.assertEqual(out["g::q"]["original_target"], [0.0, 1.0])
            self.assertEqual(out["p::q"]["teacher_agreement"], 1.0)
            self.assertAlmostEqual(sum(out["p::q"]["target"]), 1.0)
            test = read_rows(root / "out" / "test.jsonl")
            self.assertEqual(test[0]["target"], [0.5, 0.5])  # the test split is never relabelled


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


def tiny_setup(test):
    try:
        import torch
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        test.skipTest("optional encoder dependencies unavailable")
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "approve": 4, "reject": 5, "invoice": 6, "note": 7}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]",
                                        cls_token="[CLS]", sep_token="[SEP]")

    def encoder(hidden):
        return BertModel(BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=2 * hidden, max_position_embeddings=128))
    return torch, tokenizer, encoder


def distil_rows(split, n, unlabelled_every=0):
    rows = []
    for i in range(n):
        row = {"case_id": f"{split}{i}::q", "source_group": f"{split}{i}", "workflow": "invoice",
               "state": {"invoice": i}, "question": "invoice note",
               "candidates": [{"id": "a", "text": "approve"}, {"id": "b", "text": "reject"}],
               "target": [0.8, 0.2] if i % 2 else [0.3, 0.7], "label_source": "gold"}
        if unlabelled_every and i % unlabelled_every == 0:
            row.update(target=[0.5, 0.5], label_source="none")
        rows.append(row)
    return rows


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

    def test_label_with_checkpoint_then_train_a_student_with_a_teacher(self):
        torch, tokenizer, encoder = tiny_setup(self)
        from anarkali.checkpoint import PackedCheckpoint
        from anarkali.packed import PackedChoiceModel
        torch.manual_seed(0)
        teacher_model = PackedChoiceModel(encoder(32)).eval()
        teacher = PackedCheckpoint(teacher_model, tokenizer, 64, False, {"model_id": "tiny-teacher"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            counts = {}
            for split, n in (("train", 8), ("development", 4), ("calibration", 4), ("test", 2)):
                path = data / f"{split}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in distil_rows(split, n, unlabelled_every=2)))
                counts[split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (data / "manifest.json").write_text(json.dumps({"dataset": "toy", "revision": "r", "split_counts": counts}))
            (root / "teacher.pt").write_bytes(b"fixture")

            label = load_script("label_with_checkpoint")
            with contextlib.redirect_stdout(io.StringIO()):
                label.main(["--checkpoint", str(root / "teacher.pt"), "--input", str(data), "--output",
                            str(root / "labelled"), "--name", "t400", "--fill-unlabelled", "--device", "cpu",
                            "--batch-size", "3", "--chunk-rows", "5"], loader=lambda *a: teacher)
            labelled = read_rows(root / "labelled" / "train.jsonl")
            self.assertEqual(len(labelled), 8)
            self.assertTrue(all("t400" in r["teacher_targets"] for r in labelled))
            self.assertTrue(all(r["label_source"] != "none" for r in labelled))
            filled = [r for r in labelled if r["label_source"] == "teacher:t400"]
            self.assertEqual(filled[0]["target"], filled[0]["teacher_targets"]["t400"])
            self.assertEqual(read_rows(root / "labelled" / "test.jsonl")[0]["label_source"], "none")  # untouched

            trainer = load_script("train_anarkali")
            argv = ["train", "--data", str(data), "--output", str(root / "student"), "--architecture", "packed",
                    "--epochs", "2", "--batch-size", "3", "--device", "cpu", "--packed-max-tokens", "64",
                    "--brier-weight", "0.25", "--teacher-checkpoint", str(root / "teacher.pt"),
                    "--kd-weight", "1.0", "--kd-temperature", "2.0", "--hidden-weight", "0.5"]
            with (patch.object(sys, "argv", argv),
                  patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="fixture")),
                  patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                  patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: encoder(16)),
                  patch("anarkali.checkpoint.load_packed_checkpoint", return_value=teacher),
                  contextlib.redirect_stdout(io.StringIO())):
                trainer.main()
            history = json.loads((root / "student" / "training.json").read_text())
            config = history["run_config"]
            self.assertEqual(config["unlabelled_train_rows_teacher_filled"], 4)
            self.assertEqual(config["unlabelled_development_rows_skipped"], 2)
            self.assertEqual(config["teacher_model_id"], "tiny-teacher")
            self.assertTrue(all(e["train_loss"] < 1e6 for e in history["history"]))
            state = torch.load(root / "student" / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
            self.assertFalse(any(k.startswith("projector") for k in state))  # the projector is not shipped

            argv_no_teacher = [a for a in argv[:-8]]
            with (patch.object(sys, "argv", argv_no_teacher),
                  patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="fixture")),
                  self.assertRaisesRegex(ValueError, "label_source 'none'")):
                trainer.main()


if __name__ == "__main__":
    unittest.main()
