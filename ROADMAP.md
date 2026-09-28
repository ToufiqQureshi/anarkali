# Anarkali roadmap: how a 68M model becomes the default typed-decision engine

This is the plan to make Anarkali the model people reach for when software has to make a small decision many times: fast, cheap, open, and honest about how sure it is. It covers where we stand, what "number one" should mean, the work already in the repo, and the order in which to do the rest.

## 1. Where we stand

Public typed-decisions benchmark, test split, 2,000 decisions:

| Model | Kind | Size | Accuracy | Calibration (ECE) | Open |
|---|---|---:|---:|---:|:---:|
| meraGPT Decider 1 | general, zero-shot | undisclosed | 76.8% | not published | no |
| Laya typed-decisions | fine-tuned on this benchmark | 421M | 76.6% | 0.213 | yes |
| **Anarkali 0.3.0** | fine-tuned on this benchmark | **68M** | 74.0% | **0.135** | **yes** |
| TypeSafe Jev 1.13.0 | general, zero-shot | undisclosed | 72.7% | 0.144 | no |

Laya, Jev and Decider figures are the ones their authors published. The dataset's own labelling teacher agrees with itself 73.5% of the time.

What these numbers mean:

- **We win on calibration and cost, and lose narrowly on accuracy.** Laya is six times larger and over-confident. Its own card says its temperatures were fitted on training data.
- **The accuracy race on this benchmark has a ceiling.** Above the teacher's 73.5% self-agreement, extra points increasingly measure how well a model copies one teacher's quirks. Chasing 80% here is not the goal.
- **Decider and Jev are zero-shot generalists.** The honest weakness of Anarkali, and of Laya, is breadth: both were fine-tuned on four synthetic workflows, and Anarkali adds three more synthetic coding ones.
- **Data is the bottleneck, not the architecture.** Training has 7,880 decisions across 7 workflows. Benchmark states are short (all under about 1,800 characters, roughly 500 tokens), so a longer context changes nothing on this benchmark. It matters for real CI logs and agent traces.

## 2. What "number one" means

Four things, all measured. We claim number one only where a number supports it.

1. **Best calibrated at any size.** Lowest ECE and KL from gold on typed-decisions, and the best selective accuracy: when Anarkali says 0.7 or more, it is right most often.
2. **Best accuracy per millisecond and per dollar.** Within a few points of the best accuracy, at a sixth of the size, on a laptop CPU.
3. **General.** Good on domains it was not trained on, measured on a held-out-domain set we publish.
4. **The guardrail for AI coding agents.** A step checker fast enough to run on every tool call of Claude Code, Codex or OpenHands.

## 3. Shipped in this PR

| Change | Where | What it buys | How to run |
|---|---|---|---|
| **Option-order averaging** at inference | `Engine(orders=N)`, `anarkali decide/serve --orders` | Removes position bias, the packed encoder's known weakness. 2.6% of answers flip when options are reversed. No retraining. | `Engine.load(path, orders=3)` |
| **Per-type temperatures** | `anarkali.json: temperature_by_type`, `export_onnx.py --temperature-by-type` | The released model ships with temperature 1.0, uncalibrated. Fitting one temperature per question type on held-out data is the standard fix. | fitted by `benchmark_release.py` and `evaluate_checkpoint.py` |
| **Release benchmark in CI** | `scripts/benchmark_release.py`, `.github/workflows/benchmark.yml` | Real numbers on the real model for every inference change: accuracy, KL from gold, ECE, Brier, selective accuracy, truncation, latency. | Actions → Benchmark → Run |
| **Training objectives** | `src/anarkali/objectives.py`, `train_anarkali.py` flags | Brier beside soft CE; ranked probability score for ordinal questions (our weakest type, 71.6%); permutation-consistency R-Drop; layer-wise LR decay; warmup and decay schedule; EMA weights; per-row weights from teacher agreement. All off by default. | `--brier-weight --rps-weight --consistency-weight --llrd --schedule --ema-decay --weight-field` |
| **Shared option positions** (architecture) | `--shared-option-positions`, `PackedChoiceModel.encode` | Every option starts at the same position ID, and ModernBERT's local window is measured in positions. The encoder cannot see option order at all, at 1× compute. On an Ettin-shaped model, the score change under permutation drops from 9e-4 to 6e-8, and ONNX export keeps it. It must be trained, and it is a V4 recipe. | V4 notebook `v4-shared` |
| **Claude Code guardrail** | `examples/claude-code/` | A `PreToolUse` hook that denies or asks when `constraint_violation` is high. It is the product demo for step 4. | see its README |
| **V4 notebook** | `notebooks/Anarkali_V4.ipynb` | One Run All on a free T4. It rebuilds the data, trains the 0.3.0 recipe as the control against two V4 recipes (and optionally Ettin-150M), picks on dev, tests, and exports ONNX with fitted temperatures. | Colab or Kaggle, T4 |
| **Multi-teacher relabelling** | `scripts/relabel_with_teachers.py` | Labels from several open teachers (Qwen, Mistral), averaged over option orders, with disagreement filtered out. It attacks the 73.5% teacher-noise ceiling directly. | see README |
| **Real agent traces** | `scripts/import_agent_traces.py` | about 530k public coding-agent runs (80k + 67k + 318k + 66k) from 4 CC-BY/MIT datasets become `coding_agent_step` data, with injected rule-breaking steps. | `--per-source 2000` |
| **20-domain generator** | `scripts/generate_domain_decisions.py`, `scripts/domains/catalog.json` | Breadth: HR, moderation, insurance, loans, fraud, contracts, IT, KYC, cloud cost, DB migrations and more, at five difficulty styles. | see the script's docstring |

Every piece has tests that run without a network or GPU. None of it has trained a model yet: that needs a GPU, and it is step 2 below.

## 4. The plan, in order

### Step 1: today, no GPU (about 1 hour)

- Read the Benchmark workflow's summary on this PR. It scores the released model with `orders=1/2/3` and with per-type temperatures fitted on the calibration split.
- If `orders=2` or `orders=3` improves accuracy or ECE, set that as the release default: add `"orders": N` and the fitted `"temperature_by_type"` to `anarkali.json` on the Hugging Face repo. That is a config-only release, 0.3.1.
- Update the README table with the new numbers. Report latency with `orders` included, honestly.

### Step 2: this week, one free T4 (about 2 hours)

- Run `notebooks/Anarkali_V4.ipynb` with `AGENT_TRACES_PER_SOURCE = 500`.
- Ship the winner only if it beats `v3-baseline` on development and the typed test. The bake-off also shows which change helped.
- Export with its temperatures and release it as 0.4.0.

### Step 3: weeks 2 and 3, the data flywheel (free Kaggle GPU)

- Serve `Qwen/Qwen3-14B-AWQ` (thinking off) or `Qwen3-30B-A3B-Instruct-2507` with vLLM on Kaggle's T4 ×2.
- Generate 20 domains × 200 cases with `generate_domain_decisions.py`.
- Label them with two Apache-2.0 teachers through `relabel_with_teachers.py --no-original --splits train,development,calibration,test`, and drop rows the teachers disagree on.
- Relabel the typed-decisions training split with the same teachers. The test split stays untouched.
- Add agent traces at 2,000 per source.
- Retrain with V4, passing the labelled sets as `EXTRA_SETS`.
- Publish **Anarkali-Domains**: a held-out-domain benchmark. Train on 16 domains, test on 4 unseen ones, so generality becomes a number.

### Step 4: week 3, the coding-agent guardrail (the wedge)

- Hand-label 300 to 500 real agent steps as the evaluation set. Never train on it.
- Ship a Claude Code `PreToolUse` hook and a GitHub Action that call `anarkali serve`. The hook blocks a step when `constraint_violation` is at least 0.8.
- Write it up with latency and catch rate on the hand-labelled set.

### Step 5: month 2, the model family

- `anarkali` (68M, default, CPU), `anarkali-large` (Ettin-150M or 400M, when accuracy matters), and optionally `anarkali-mini` (Ettin-32M, edge).
- Raise `packed-max-tokens` to 1024 for log-heavy workflows, trained at that length. Ettin supports about 8k positions.
- Revisit int8 quantisation with static calibration. Every dynamic int8 variant failed the parity gate.

### Step 6: visibility

- **Submit to the typed-decisions leaderboard.** Its card invites it: score the test split with full distributions and open a discussion.
- Publish `anarkali` on PyPI, a Hugging Face Space demo, and a short technical report with every ablation from the V4 bake-off.

### Step 7: funding

- Apply to Google for Startups Cloud (Start tier, up to $2,000 without funding) and NVIDIA Inception.
- The pitch: *the calibrated decision layer for AI agents, a sixth of the size of the nearest open model, on a CPU.*
- Bring the benchmark table, the guardrail demo and two design partners.

## 5. Risks, stated plainly

- **Distillation limits.** A student rarely beats its teachers. Better teachers and disagreement filtering raise the ceiling, but they do not remove it.
- **Benchmark overfitting.** Every recipe is chosen on development data, and the test split is read once, after the choice. Keep it that way.
- **Licences.** Only Apache-2.0 or MIT teachers and datasets. Credit CC-BY sources. Several agent-trace datasets were produced with commercial models whose terms also apply.
- **Latency claims.** `orders=3` costs about three times the compute. Quote the setting next to every latency number.

## 6. References

- Guo et al. 2017, *On Calibration of Modern Neural Networks*, arXiv:1706.04599: temperature scaling.
- Tang et al. 2023, *Found in the Middle: Permutation Self-Consistency*, arXiv:2310.07712: averaging over option orders.
- Zheng et al. 2023, *On Large Language Models' Selection Bias in Multi-Choice Questions*, arXiv:2309.03882: option-position bias.
- Liang et al. 2021, *R-Drop: Regularized Dropout for Neural Networks*, arXiv:2106.14448: consistency regularisation.
- Sun et al. 2019, *How to Fine-Tune BERT for Text Classification?*, arXiv:1905.05583: layer-wise learning-rate decay, warmup.
- Izmailov et al. 2018, *Averaging Weights Leads to Wider Optima and Better Generalization*, arXiv:1803.05407: weight averaging.
- Epstein 1969, *A Scoring System for Probability Forecasts of Ranked Categories*, J. Appl. Meteorology: ranked probability score.
- Weller et al. 2025, *Seq vs Seq: An Open Suite of Paired Encoders and Decoders* (Ettin), arXiv:2507.11412.
