---
license: apache-2.0
library_name: onnxruntime
tags:
  - anarkali
  - classification
  - decision-support
  - onnx
---

# Anarkali 150M

Anarkali 150M is an ONNX release of a packed encoder choice model for structured decision-support questions. It ranks the candidate options supplied in an input request; it does not establish real-world correctness and must not be used as an autonomous decision-maker.

## Files

- `model.onnx`: validated FP32 inference graph.
- `model.optimized.onnx`: ORT-optimized FP32 graph.
- `tokenizer.json`: tokenizer required by the Anarkali runtime.
- `anarkali.json`: runtime configuration and provenance.

## Validation

The release passed PyTorch-to-ONNX parity on 200 held-out-format development rows:

- Argmax mismatches: **0**
- Maximum probability drift: **1.96e-6**
- FP32 CPU latency (batch 1, tokenization and packing included): p50 446 ms, p95 704 ms.

These metrics measure agreement with evaluation labels/teachers, not proof of real-world correctness or safety. The model must be reviewed against the deployment's baseline, acceptance criteria, and human oversight requirements.

## Usage

Install the project runtime and load the Hub repository:

```python
from anarkali.engine import Engine

engine = Engine.load("toufiqqureshi651/anarkali")
```

See the project repository for request schemas, evaluation scripts, and service integration guidance.

## License and provenance

Apache-2.0. The model uses the Ettin 150M encoder and was trained on the `anarkali-decisions-v3` data revision recorded in `anarkali.json`. Refer to the project `LICENSE`, `NOTICE`, and documentation for third-party licenses, data lineage, and limitations.
