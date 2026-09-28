"""Run the trainer end to end with every V4 objective switched on, with tiny random weights."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class TrainingV4Tests(unittest.TestCase):
    def test_packed_trains_with_all_objectives(self):
        try:
            import torch
            from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError:
            self.skipTest("optional encoder dependencies unavailable")
        spec = importlib.util.spec_from_file_location("training_v4_fixture", ROOT / "scripts" / "train_anarkali.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "approve": 4, "reject": 5, "low": 6,
                 "mid": 7, "high": 8, "invoice": 9}
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]",
                                            cls_token="[CLS]", sep_token="[SEP]")

        def tiny_encoder(*args, **kwargs):
            return BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                                        num_attention_heads=4, intermediate_size=32, max_position_embeddings=128))

        def row(group, index):
            if index % 2:
                return {"case_id": group + "::risk", "source_group": group, "workflow": "invoice",
                        "state": {"invoice": index}, "question": "invoice", "question_type": "score",
                        "candidates": [{"id": "0", "text": "low"}, {"id": "1", "text": "mid"},
                                       {"id": "2", "text": "high"}],
                        "target": [0.1, 0.2, 0.7], "teacher_agreement": 0.5}
            return {"case_id": group + "::q", "source_group": group, "workflow": "invoice",
                    "state": {"invoice": index}, "question": "invoice",
                    "candidates": [{"id": "a", "text": "approve"}, {"id": "b", "text": "reject"}],
                    "target": [0.8, 0.2], "teacher_agreement": 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            manifest = {"split_counts": {}}
            for split in ("train", "development"):
                path = data / f"{split}.jsonl"
                path.write_text("".join(json.dumps(row(split + str(i), i)) + "\n" for i in range(6)), encoding="utf-8")
                manifest["split_counts"][split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(tmp) / "v4"
            argv = ["train", "--data", str(data), "--output", str(output), "--architecture", "packed",
                    "--epochs", "2", "--batch-size", "3", "--device", "cpu", "--packed-max-tokens", "64",
                    "--brier-weight", "0.5", "--rps-weight", "0.5", "--consistency-weight", "1.0",
                    "--weight-field", "teacher_agreement", "--llrd", "0.8", "--warmup-ratio", "0.25",
                    "--schedule", "cosine", "--ema-decay", "0.9"]
            with (patch.object(sys, "argv", argv),
                  patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="fixture")),
                  patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                  patch("transformers.AutoModel.from_pretrained", side_effect=tiny_encoder),
                  contextlib.redirect_stdout(io.StringIO())):
                trainer.main()
            history = json.loads((output / "training.json").read_text())
            config = history["run_config"]
            self.assertEqual((config["rps_weight"], config["consistency_weight"], config["schedule"]),
                             (0.5, 1.0, "cosine"))
            self.assertEqual(len(history["history"]), 2)
            for epoch in history["history"]:
                self.assertTrue(all(isinstance(v, (int, float)) for k, v in epoch.items()))
                self.assertLess(epoch["train_loss"], 1e6)
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint["state_dict"].values()
                                if v.dtype.is_floating_point))


if __name__ == "__main__":
    unittest.main()
