"""Export a packed checkpoint to a torch-free release directory and gate parity.

The release always includes fp32 ONNX when parity passes. Int8 is included only
when one of the tested recipes has zero argmax mismatches and stays within the
configured probability drift gate. If no int8 recipe passes, the release keeps
fp32 plus an ORT-optimized fp32 graph for measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows if limit is None else rows[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO / "artifacts" / "typed-decisions-v1")
    parser.add_argument("--cache", type=Path, default=REPO / ".cache" / "huggingface")
    parser.add_argument("--name", default="anarkali-lite")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--abstain-below", type=float)
    parser.add_argument("--temperature-by-type", type=json.loads, default=None,
                        help='per-type temperatures fitted on held-out data, e.g. \'{"choice": 1.2, "noul": 0.9}\'')
    parser.add_argument("--orders", type=int, default=1, help="default option orders the engine averages over")
    parser.add_argument("--max-int8-drift", type=float, default=0.02)
    parser.add_argument("--no-int8", action="store_true",
                        help="ship fp32 only; skips the int8 recipes, which take long on large encoders")
    parser.add_argument("--parity-rows", type=int, default=200)
    parser.add_argument("--latency-samples", type=int, default=100)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from anarkali.engine import Engine, _OnnxBackend, _batch_arrays
    from anarkali.packing import pack_row
    from anarkali.typed import softmax
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)

    reference = Engine.load(args.checkpoint, cache_dir=str(args.cache))
    backend = reference.backend
    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = checkpoint_meta["run_config"]
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in ("model*.onnx", "model*.rejected.onnx"):
        for stale in args.output.glob(pattern):
            stale.unlink()

    tokenizer = backend.tokenizer
    tokenizer.save_pretrained(args.output / "hf-tokenizer")
    shutil.copy(args.output / "hf-tokenizer" / "tokenizer.json", args.output / "tokenizer.json")
    shutil.rmtree(args.output / "hf-tokenizer")

    shared = backend.model.shared_option_positions

    class Graph(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, candidate_spans, position_ids=None):
            tokens = self.model.encode(input_ids, attention_mask, position_ids)
            return self.model.head(tokens, candidate_spans)

    rows = load_rows(args.data / "development.jsonl", args.parity_rows)
    example = _batch_arrays([pack_row(r, tokenizer, reference.max_tokens) for r in rows[:2]],
                            tokenizer.pad_token_id, shared)
    names = ["input_ids", "attention_mask", "candidate_spans"] + (["position_ids"] if shared else [])
    axes = {
        "input_ids": {0: "batch", 1: "tokens"},
        "attention_mask": {0: "batch", 1: "tokens"},
        "candidate_spans": {0: "batch", 1: "options", 2: "tokens"},
        "logits": {0: "batch", 1: "options"},
    }
    if shared:
        axes["position_ids"] = {0: "batch", 1: "tokens"}
    fp32 = args.output / "model.onnx"
    torch.onnx.export(
        Graph(backend.model).eval(),
        tuple(torch.from_numpy(a) for a in example),
        str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )

    config = {
        "name": args.name,
        "format": "anarkali-onnx-v1",
        "max_tokens": reference.max_tokens,
        "temperature": args.temperature,
        "abstain_below": args.abstain_below,
        **({"temperature_by_type": args.temperature_by_type} if args.temperature_by_type else {}),
        **({"orders": args.orders} if args.orders != 1 else {}),
        **({"shared_option_positions": True} if shared else {}),
        "question_types": ["choice", "noul", "score"],
        "tokenizer": {
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "model_max_length": min(tokenizer.model_max_length, 10**6),
        },
        "source": {
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "base_model": checkpoint_meta["model_id"],
            "base_revision": checkpoint_meta["model_revision"],
            "dataset_revision": checkpoint_meta["manifest"].get("revision"),
            "parameters": run_config.get("parameter_count"),
        },
    }
    (args.output / "anarkali.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def packed_for(row, engine_backend):
        return [pack_row(row, engine_backend.tokenizer, reference.max_tokens)]

    def probabilities(engine_backend):
        return [softmax(engine_backend.logits(packed_for(row, engine_backend))[0]) for row in rows]

    torch_probs = probabilities(backend)

    def measure_latency(graph: str, threads: int | None) -> dict[str, float | int | str]:
        onnx_backend = _OnnxBackend(args.output, config, graph, threads)
        sample = rows[:args.latency_samples]
        for row in sample[:10]:
            onnx_backend.logits(packed_for(row, onnx_backend))
        latencies = []
        for row in sample:
            started = time.perf_counter()
            onnx_backend.logits(packed_for(row, onnx_backend))
            latencies.append((time.perf_counter() - started) * 1000)
        return {
            "threads": threads or "default",
            "samples": len(sample),
            "warmup": min(10, len(sample)),
            "cpu_latency_p50_ms": statistics.median(latencies),
            "cpu_latency_p95_ms": percentile(latencies, 0.95),
            "latency_includes": "tokenization + packing + onnxruntime, batch 1",
        }

    def evaluate_graph(graph: str) -> dict[str, Any]:
        onnx_backend = _OnnxBackend(args.output, config, graph, None)
        probs = probabilities(onnx_backend)
        drift = max(abs(a - b) for expected, actual in zip(torch_probs, probs)
                    for a, b in zip(expected, actual))
        flips = sum(int(np.argmax(expected) != np.argmax(actual)) for expected, actual in zip(torch_probs, probs))
        latency = measure_latency(graph, None)
        return {
            "rows": len(rows),
            "max_probability_drift": drift,
            "argmax_mismatches": flips,
            "bytes": (args.output / graph).stat().st_size,
            **latency,
        }

    def exclude_head_nodes(model_path: Path) -> list[str]:
        model = onnx.load(str(model_path))
        excluded = []
        for node in model.graph.node:
            label = " ".join([node.name, *node.input, *node.output]).lower()
            if any(part in label for part in ("head", "scorer", "score")):
                excluded.append(node.name)
        return excluded

    class PackedCalibrationReader(CalibrationDataReader):
        def __init__(self, calibration_rows: list[dict[str, Any]], limit: int):
            self._items = []
            for row in calibration_rows[:limit]:
                input_ids, attention_mask, spans = _batch_arrays(
                    [pack_row(row, tokenizer, reference.max_tokens)],
                    tokenizer.pad_token_id,
                )
                self._items.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "candidate_spans": spans,
                })
            self._index = 0

        def get_next(self):
            if self._index >= len(self._items):
                return None
            item = self._items[self._index]
            self._index += 1
            return item

    def save_optimized_graph() -> None:
        options = ort.SessionOptions()
        # EXTENDED, not ALL: ALL adds hardware-specific NCHWc layouts that fail to serialize
        # and would tie the saved graph to this CPU.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.optimized_model_filepath = str(args.output / "model.optimized.onnx")
        ort.InferenceSession(str(fp32), options, providers=["CPUExecutionProvider"])

    recipes = [
        {
            "name": "int8_per_channel",
            "graph": "model.int8.per_channel.onnx",
            "fn": lambda out: quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8,
                                               per_channel=True),
        },
        {
            "name": "int8_per_channel_reduce_range",
            "graph": "model.int8.per_channel_reduce_range.onnx",
            "fn": lambda out: quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8,
                                               per_channel=True, reduce_range=True),
        },
        {
            "name": "int8_matmul_only_no_head",
            "graph": "model.int8.matmul_no_head.onnx",
            "fn": lambda out: quantize_dynamic(
                str(fp32),
                str(out),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
                nodes_to_exclude=exclude_head_nodes(fp32),
            ),
        },
        {
            "name": "int8_static_qdq",
            "graph": "model.int8.static_qdq.onnx",
            "fn": lambda out: quantize_static(
                str(fp32),
                str(out),
                PackedCalibrationReader(load_rows(args.data / "calibration.jsonl"), 100),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            ),
        },
    ]

    parity: dict[str, Any] = {}

    def write_parity() -> None:
        """Persist after every step so a later failure keeps the measured results."""
        (args.output / "parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")

    fp32_metrics = evaluate_graph("model.onnx")
    fp32_metrics["passed"] = fp32_metrics["max_probability_drift"] <= 1e-3 and fp32_metrics["argmax_mismatches"] == 0
    parity["model.onnx"] = fp32_metrics
    if not fp32_metrics["passed"]:
        raise SystemExit(f"fp32 ONNX parity failed: {fp32_metrics}")

    passing_int8 = []
    for recipe in [] if args.no_int8 else recipes:
        graph_path = args.output / recipe["graph"]
        try:
            recipe["fn"](graph_path)
            metrics = evaluate_graph(recipe["graph"])
            metrics["passed"] = (
                metrics["max_probability_drift"] <= args.max_int8_drift
                and metrics["argmax_mismatches"] == 0
            )
        except Exception as exc:
            metrics = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        parity[recipe["name"]] = metrics
        write_parity()
        if metrics.get("passed"):
            passing_int8.append({"recipe": recipe, "metrics": metrics})

    try:
        save_optimized_graph()
        optimized_metrics = evaluate_graph("model.optimized.onnx")
        optimized_metrics["passed"] = (
            optimized_metrics["max_probability_drift"] <= 1e-3
            and optimized_metrics["argmax_mismatches"] == 0
        )
    except Exception as exc:
        optimized_metrics = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "model.optimized.onnx").unlink(missing_ok=True)
    parity["model.optimized.onnx"] = optimized_metrics
    write_parity()

    if passing_int8:
        winner = min(passing_int8, key=lambda item: item["metrics"]["cpu_latency_p50_ms"])
        shutil.copy(args.output / winner["recipe"]["graph"], args.output / "model.int8.onnx")
        parity["selected_int8"] = {
            "recipe": winner["recipe"]["name"],
            "graph": "model.int8.onnx",
            **evaluate_graph("model.int8.onnx"),
            "passed": True,
        }
    else:
        parity["selected_int8"] = None
        plain_int8 = args.output / "model.int8.onnx"
        if plain_int8.exists():
            plain_int8.rename(args.output / "model.int8.rejected.onnx")

    thread_graphs = ["model.onnx"] + (["model.optimized.onnx"] if optimized_metrics.get("passed") else [])
    if (args.output / "model.int8.onnx").exists():
        thread_graphs.append("model.int8.onnx")
    parity["thread_latency"] = {
        graph: {str(threads): measure_latency(graph, threads) for threads in (1, 4, 6)}
        for graph in thread_graphs
    }

    (args.output / "parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parity, indent=2))


if __name__ == "__main__":
    main()
