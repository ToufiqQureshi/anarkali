"""Run a released Anarkali model: `Engine.load(path).predict(state, questions)`.

A released model directory holds `anarkali.json`, `tokenizer.json` and an ONNX graph
(`model.int8.onnx` preferred, then `model.onnx`). That path needs only numpy,
onnxruntime and tokenizers, no torch. A training checkpoint (`best.pt`) also loads, via
torch and transformers, for parity checks and development.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .packing import pack_row
from .typed import MAX_OPTIONS, format_answer, question_candidates, softmax

MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50_000


class _FastTokenizer:
    """The slice of a Hugging Face tokenizer that `pack_row` uses, from tokenizer.json alone."""

    def __init__(self, path: Path, cls_id: int, sep_id: int, pad_id: int, max_length: int):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(path))
        self.cls_token_id, self.sep_token_id, self.pad_token_id = cls_id, sep_id, pad_id
        self.model_max_length = max_length

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids


def _batch_arrays(packed, pad_id):
    import numpy as np
    width = max(len(ids) for ids, _, _ in packed)
    k = max(len(spans) for _, spans, _ in packed)
    input_ids = np.full((len(packed), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(packed), width), dtype=np.int64)
    spans = np.zeros((len(packed), k, width), dtype=np.float32)
    for i, (ids, ranges, _) in enumerate(packed):
        input_ids[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1
        for j, (start, end) in enumerate(ranges):
            spans[i, j, start:end] = 1.0
    return input_ids, mask, spans


class _OnnxBackend:
    def __init__(self, directory: Path, config: dict, graph: str | None, threads: int | None):
        import onnxruntime as ort
        name = graph or next((g for g in ("model.int8.onnx", "model.onnx") if (directory/g).exists()), None)
        if name is None:
            raise FileNotFoundError(f"no ONNX graph in {directory}")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(directory/name), options, providers=["CPUExecutionProvider"])
        self.graph = name
        tok = config["tokenizer"]
        self.tokenizer = _FastTokenizer(directory/"tokenizer.json", tok["cls_token_id"], tok["sep_token_id"],
                                        tok["pad_token_id"], tok["model_max_length"])

    def logits(self, packed):
        input_ids, mask, spans = _batch_arrays(packed, self.tokenizer.pad_token_id)
        out = self.session.run(["logits"], {"input_ids": input_ids, "attention_mask": mask,
                                             "candidate_spans": spans})[0]
        return [row[:len(p[1])].tolist() for row, p in zip(out, packed)]


class _TorchBackend:
    def __init__(self, checkpoint: Path, cache_dir: str | None, device: str):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from .packed import PackedChoiceModel
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("run_config", {}).get("architecture") != "packed":
            raise ValueError("only packed checkpoints can be served")
        options = dict(revision=state["model_revision"], cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(state["model_id"], **options)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(state["model_id"], **options))
        head = state["head_config"]
        self.model = PackedChoiceModel(encoder, head["hidden_dim"], head["dropout"])
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.to(device).eval()
        self.device, self.torch = torch.device(device), torch
        self.meta = {k: state[k] for k in ("model_id", "model_revision")}
        self.meta["max_tokens"] = state["run_config"].get("packed_max_tokens", 512)

    def logits(self, packed):
        torch = self.torch
        input_ids, mask, spans = (torch.from_numpy(a).to(self.device)
                                  for a in _batch_arrays(packed, self.tokenizer.pad_token_id))
        with torch.inference_mode():
            tokens = self.model.encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
            out = self.model.head(tokens, spans.bool()).float().cpu().tolist()
        return [row[:len(p[1])] for row, p in zip(out, packed)]


def _resolve(path: str | Path, cache_dir: str | None) -> Path:
    """Return a local path; an 'owner/name' that is not on disk is a Hugging Face model repo."""
    local = Path(path)
    if local.exists() or isinstance(path, Path) or str(path).count("/") != 1:
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(path), cache_dir=cache_dir,
                                  allow_patterns=["anarkali.json", "tokenizer.json", "model*.onnx"]))


class Engine:
    """Typed decisions (choice / noul / score) over one state, Jev `/v1/systemone` shaped."""

    def __init__(self, backend, *, name: str, max_tokens: int, temperature: float = 1.0,
                 abstain_below: float | None = None):
        self.backend, self.name = backend, name
        self.max_tokens, self.temperature, self.abstain_below = max_tokens, temperature, abstain_below

    @classmethod
    def load(cls, path: str | Path, *, graph: str | None = None, threads: int | None = None,
             cache_dir: str | None = None, device: str = "cpu") -> "Engine":
        path = _resolve(path, cache_dir)
        if path.is_file() and path.suffix == ".pt":
            backend = _TorchBackend(path, cache_dir, device)
            return cls(backend, name=f"anarkali:{path.parent.name}", max_tokens=backend.meta["max_tokens"])
        config = json.loads((path/"anarkali.json").read_text(encoding="utf-8"))
        backend = _OnnxBackend(path, config, graph, threads)
        return cls(backend, name=config.get("name", "anarkali"), max_tokens=config["max_tokens"],
                   temperature=config.get("temperature", 1.0), abstain_below=config.get("abstain_below"))

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("'questions' must be a non-empty object")
        if len(questions) > MAX_QUESTIONS:
            raise ValueError(f"too many questions ({len(questions)} > {MAX_QUESTIONS})")
        if len(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)) > MAX_STATE_CHARS:
            raise ValueError(f"state longer than {MAX_STATE_CHARS} characters")
        started = time.perf_counter()
        parsed, packed = [], []
        for qid, question in questions.items():
            try:
                kind, text, candidates = question_candidates(question)
                packed.append(pack_row({"state": state, "question": text, "candidates": candidates},
                                       self.backend.tokenizer, self.max_tokens))
            except ValueError as exc:
                raise ValueError(f"question {qid!r}: {exc}") from None
            parsed.append((qid, kind, candidates))
        answers = {}
        for (qid, kind, candidates), logits in zip(parsed, self.backend.logits(packed)):
            probabilities = softmax(logits, self.temperature)
            answers[qid] = format_answer(kind, candidates, probabilities, self.abstain_below)
        tokens = sum(p[2]["input_tokens"] for p in packed)
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


__all__ = ["Engine", "MAX_OPTIONS"]
