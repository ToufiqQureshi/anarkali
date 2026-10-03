# Session notes — 2026-10-03

## Summary

Anarkali 150M ONNX release was verified locally, then restored from Hugging Face after the local release folder was found incomplete. A benchmark/evaluator workflow was added so future model changes can be measured against targeted hard routing cases.

## Model location

```text
release-150m/
```

Current files:

```text
release-150m/anarkali.json
release-150m/model.onnx
release-150m/model.optimized.onnx
release-150m/parity.json
release-150m/README.md
release-150m/tokenizer.json
```

The model was tested with:

```powershell
python -m anarkali decide --model release-150m --request examples\requests\support_routing.json
```

## Cleanup note

Earlier generated/local artifacts were removed:

```text
.cache/
.kaggle-run/
.freebuff/
results.zip
anarkali-150m-best.pt
temporary upload/export logs
```

The `.pt` checkpoint and `results.zip` are gone from the local repo, but the ONNX release model is restored and working.

## Added workflow

Added:

```text
docs/COMPETITIVE_STRATEGY.md
docs/SESSION_NOTES_2026-10-03.md
benchmarks/anarkali-routing-v1/cases.jsonl
scripts/build_routing_benchmark.py
scripts/evaluate_release_suite.py
tests/test_release_benchmark.py
```

## Benchmark direction

The hard benchmark focuses on:

- Scope: one customer vs many customers.
- Product vs incident.
- Product vs identity.
- Security vs identity.
- Billing vs product.
- Temporal traps.
- Priority/multi-issue routing.
- Long context / signal burial.

## Next step

Generate counterfactual training data from hard-suite failures, retrain, export ONNX, and compare against the benchmark.
