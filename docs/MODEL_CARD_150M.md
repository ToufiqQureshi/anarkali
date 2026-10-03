# Anarkali 150M model card

This document explains what the current 150M ONNX release is good at, where it is weak, and where it should or should not be used.

## Summary

Anarkali 150M is a compact typed-decision model. It does not write long answers like a chat model. It takes:

```text
state + typed question + allowed answers
```

and returns:

```text
answer + probabilities + confidence
```

Supported question types:

- `choice`: choose one option.
- `noul`: true/false probability.
- `score`: expected score over ordered levels.

The request contract follows the Jev/Laya shape: each question is type-checked before inference, `state` must be JSON-safe, and each `criteria` object or list must match the required schema for the selected question type. This keeps the API compact, structured, and type-safe for routing, triage, classification, and lightweight guardrail checks.

## Release files

The checked-in release lives at:

```text
release-150m/
```

Files:

```text
release-150m/anarkali.json
release-150m/model.onnx
release-150m/model.optimized.onnx
release-150m/parity.json
release-150m/README.md
release-150m/tokenizer.json
```

The ONNX files are stored with Git LFS.

## Current quality

### Smoke / notebook evaluation

The Hugging Face/local release smoke test produced:

```text
24 cases
21 correct
87.5% accuracy
```

### Current hard routing suite

The repository hard routing benchmark currently produces:

```text
32 cases
24 correct
75.0% accuracy
```

Command:

```bash
python scripts/evaluate_release_suite.py --model release-150m \
  --cases benchmarks/anarkali-routing-v1/cases.jsonl --min-accuracy 0.60
```

Interpretation:

- The model works.
- The ONNX package is valid.
- It is useful as a compact typed-decision engine.
- It is not yet strong enough to be treated as a high-stakes autonomous decision-maker.

## Strengths

Anarkali 150M is strongest when:

- The task has explicit allowed answers.
- The input is short or medium length.
- The categories are clearly defined.
- The decision is similar to support routing, incident triage, billing/product routing, identity/security triage, or coding-agent step checks.
- The system can use confidence thresholds and fallback to human review.

Good fit examples:

- Support ticket routing.
- Incident vs product triage.
- Billing vs product entitlement routing.
- Identity vs security routing.
- CI failure triage.
- PR review category selection.
- Agent guardrail checks before tool use.
- Lightweight internal workflow decisions.
- Batch classification where low-confidence rows can be reviewed.

## Weaknesses

Known weak areas:

- One-customer product issues can be confused with widespread incidents.
- Product vs identity near-ties can be unstable.
- Billing/product entitlement cases need more examples.
- Security vs identity edge cases can flip when both contain authentication language.
- Temporal cases can confuse old/resolved issues with the current unresolved issue.
- Confidence is useful, but not perfect; some wrong answers have moderate confidence.
- The current hard benchmark is still small and should grow to hundreds or thousands of cases.

### Hand-labelled real CI failures

On 2026-10-03, the released Hub model was evaluated on all 65 cases in `benchmarks/realworld-ci-v0/cases.jsonl`. It predicted the `cause` label correctly on **5/65 cases (7.7%)**; an always-most-common-label baseline scores **75.4%**. Mean probability on the gold label was 0.205. Per-case latency on the evaluation machine was p50 1.58 s and p95 2.31 s. This is evidence that the synthetic coding workflow results do not transfer to these real CI logs. The model is not suitable for automated CI triage or release gating without retraining and a passing, independently labelled evaluation.

Three-order averaging scored **8/65 (12.3%)** on those CI cases and **23/32 (71.9%)** on the hard routing suite, so it did not close the CI generalization gap. The shipped optimized ONNX graph matched the primary graph's top answers on all 97 cases in these two suites.

Reproduce with:

```bash
python scripts/realworld_benchmark.py --cases benchmarks/realworld-ci-v0/cases.jsonl \
  --model anarkali=toufiqqureshi651/anarkali --output artifacts/realworld-ci-v0
```

Current failure themes:

```text
single customer affected -> sometimes predicted as incident
product issue with report/export language -> sometimes predicted as incident
credential stuffing / MFA rollout -> identity/security confusion
near-tie weak evidence -> unstable route
```

## Where it should be used

Use it when the decision is:

- Reversible.
- Low or medium risk.
- Auditable.
- Constrained to known options.
- Backed by fallback rules or human review.

Recommended deployment pattern:

```text
if confidence is high:
    use the model answer
else:
    route to review / ask for more information
```

Suggested starting thresholds:

```text
confidence < 0.15 -> review
top-1 probability minus top-2 probability < 0.08 -> review
high-risk category involved -> review unless confidence is very high
```

## Where it should not be used yet

Do not use the current model as the only decision-maker for:

- Medical decisions.
- Legal decisions.
- Financial approval or denial.
- Security incident closure.
- Account suspension.
- Production incident declaration without human or telemetry confirmation.
- Irreversible automated actions.
- Any workflow where a wrong route causes serious harm.

For those workflows, use Anarkali only as an assistant signal, not as authority.

## Production readiness

Current assessment:

```text
Prototype / research release: strong
Local ONNX package: good
Production classifier: needs more validation
High-stakes autonomous use: not ready
Potential after targeted data: high
```

Practical score:

```text
Current model quality: 7/10
Engineering foundation: 8/10
Data/eval maturity: 5/10
Production readiness: 6/10
```

## How to improve it

The next improvements should come from data and evaluation, not just a bigger model.

Priority loop:

```text
evaluate -> inspect failures -> generate counterfactuals -> train -> export ONNX -> evaluate again
```

Highest-value training data:

- Counterfactual pairs.
- Hard negatives.
- Real tickets/traces.
- Teacher-ensemble labels.
- Human-reviewed cases where teachers disagree.

Examples:

```text
one workspace affected + monitoring normal -> product
many unrelated customers affected + monitoring degraded -> incident
old billing issue resolved + current product crash -> product
normal MFA rollout failure -> identity
MFA bypass with suspicious token creation -> security
```

Targets:

```text
current hard suite: 75%
first target: 80%
next target: 85%
strong target: 90%
```

## Bottom line

Anarkali 150M is a useful compact typed-decision model with a working ONNX release and clear product shape. It is good enough for experimentation, internal workflow routing, and low-risk assisted decisions. It is not yet a fully production-safe autonomous router for high-stakes workflows. The fastest path forward is targeted counterfactual training data from its known failures.
