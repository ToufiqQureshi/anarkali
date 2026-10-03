# Anarkali competitive strategy

Anarkali should beat Jev/Laya-style typed-decision systems through a production scorecard, not a vanity metric.

## Positioning

- Jev: hosted typed-decision API with structured answers and probabilities.
- Laya: open/self-hostable typed-decision model direction.
- Anarkali: compact typed-decision engine with local ONNX deployment, transparent evaluation, and Jev-compatible product shape.

## What winning means

- Better calibration.
- Better accuracy per millisecond and per dollar.
- Strong ONNX/local deployment.
- Transparent benchmarks and release gates.
- Safe abstention/review behavior for low-confidence calls.
- Broader domain coverage across support, incident, security, billing, product, and agent workflows.

## Current verified release

The local `release-150m/` ONNX model was restored and smoke-tested successfully.

Release files:

```text
release-150m/anarkali.json
release-150m/model.onnx
release-150m/model.optimized.onnx
release-150m/parity.json
release-150m/README.md
release-150m/tokenizer.json
```

## Improvement loop

```text
evaluate -> find failures -> generate counterfactuals -> train -> export ONNX -> evaluate again
```

Priority failure families:

- One-customer product issue vs widespread incident.
- Product vs identity near-ties.
- Billing/product entitlement confusion.
- Temporal traps where old resolved issues distract from the current unresolved issue.
- Security vs identity edge cases.

## Next target

Use the hard routing benchmark to generate counterfactual training data from failures. First target is to raise hard-suite accuracy above 0.80, then 0.85, then 0.90.
