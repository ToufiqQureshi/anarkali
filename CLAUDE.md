# Anarkali

## What we are building

Anarkali is a small, fast **decision model for web data work**. It sits in front of a team's code
or LLM and answers small typed questions about a web page: pick one option, true/false, or a score.
It runs on a CPU and gives a calibrated confidence. When it is unsure it abstains, and the page goes
to the LLM or a person.

**Business goal: revenue.** We sell cost and noise reduction:
- **On top of an LLM:** it handles most pages itself, so far fewer pages need an LLM call. The headline
  number is "% of LLM calls saved at the same accuracy".
- **On top of code:** it stops false alerts and wrong extractions. The headline number is "% of bad
  alerts or records blocked".

Never publish a number that an eval in this repo has not produced.

## Where it should be an expert

All five workflows share one input type, a web page or the diff of a page, so one model serves them all.

| Workflow | Decisions | Status |
|---|---|---|
| Site scrape | `page_type`, `price_field`, `in_stock`, `record_valid`, `needs_llm` | first 3 in training (v0) |
| Price monitoring | `price_field`, `in_stock`, `same_product`, `price_change_real` | first 2 in training (v0) |
| Competitor site monitoring | `change_type` (pricing / launch / content / cosmetic), `alert_worthy` | planned |
| SEO monitoring | `keyword_intent`, `seo_impact` of a page change | planned |
| Site crawl | `follow_link`, `recrawl_priority` | planned |

Add one decision at a time. A decision ships only when it passes the quality bar below.

## Production-grade bar (every decision, every release)

1. **Real labels only.** Synthetic or LLM-generated data alone is not enough. (The old CI-triage
   model scored 7.7% on real logs against a 75% majority baseline because it was trained on synthetic
   data.) Free real label sources include schema.org JSON-LD, HTTP status codes and Common Crawl
   snapshot diffs. A label source must never appear in the model's input.
2. **Unseen sites in test.** Splits are by host. The test split is read once, after the model is chosen.
3. **Beat the baselines.** A release must clearly beat majority-label, simple rules and, where it
   applies, an LLM zero-shot run.
4. **Calibration.** When confidence is 0.9 or higher, accuracy must be 95% or higher. Report the
   coverage at that threshold.
5. **Latency.** p50 under 500 ms on CPU, batch 1. int8 ships only if the parity check passes.
6. **Never lose weights.** Keep `best.pt` with every run (Drive or a private HF repo). ONNX alone
   cannot be fine-tuned.

## How to add a decision without forgetting the old ones

One model serves every decision, so every new version is trained on **all** decisions at once
(multi-task). Training only on the new decision makes the model forget the old ones.

1. Add the label function for the new decision in `web.py`, plus tests. `page_decisions` must emit the
   old decisions **and** the new one, so a single dataset build covers everything.
2. Rebuild the dataset with the same `--seed`. Keep mixing the old typed-decisions data (`--mix`).
3. Start from the previous version's `best.pt` instead of the base encoder. This needs an
   `--init-checkpoint` flag in `train_anarkali.py`, which is **not built yet** (TODO). Until it exists,
   train from the base encoder on the full mix.
4. Regression gate: the new version must match or beat the previous version on every old decision's
   test score, as well as pass the bar above on the new decision. A drop on an old decision blocks the
   release.
5. Upload `best.pt`, the dataset and `test-report.json` to `toufiqqureshi651/anarkali-web` (private) under
   `runN/`, and add a row to the run log below.

## Run log

| Run | Decisions | Data | Training | Result | Where |
|---|---|---|---|---|---|
| run1 (2026-10-05) | page_type, price_field, in_stock | CC-MAIN-2026-39, 4 WARCs, 85k HTML pages, 43k labelled; 44k train rows (4.8k old mixed) | Ettin-150M packed, 512 tokens, bs 16, stopped after ~82 min on a T4 (at least 1 of 3 epochs) | see below | HF `toufiqqureshi651/anarkali-web` → `run1/` |

run1 test (unseen sites, uncalibrated argmax):

| Decision | n | Model | Baseline | Confidence ≥ 0.9 |
|---|---|---|---|---|
| page_type | 2617 | 76.9% | 39.2% (always "other") | 24% of pages, 94.9% correct |
| price_field | 165 | 46.7% | random pick (not computed) | 8% of pages, 100% correct |
| in_stock | 305 | 88.2% | 86.9% (always "true") | 2% of pages |

Overall dev 77.3%, calibrated ECE 0.020, 3.7% of answers change when options are reversed, GPU p50
47 ms (CPU not measured). Verdict: page_type works; price_field is learning but has too little data
(~2.1k train rows); in_stock has not been learned (only 578 out-of-stock train rows).
Plan for run2: product-focused data (more WARCs, keep product pages, downsample the rest), finish all
epochs with live progress, `MAX_TOKENS=384`, and score Julia-1 on the same test split as a baseline.

Lessons from run1:
- A T4 trains about one epoch of 44k rows in roughly 30–40 min. Plan for that, or use `EPOCHS=2`,
  `MAX_TOKENS=384` or a bigger GPU.
- The notebook must show live training progress. Piping through `| tail` hid it for over an hour.
- The VS Code/Antigravity Colab extension cannot download files over ~512 MB. Save checkpoints to HF
  (or Drive) from inside the runtime instead.
- Notebooks opened from `main` with `--depth 1` cannot `git checkout <branch>`. Use
  `git fetch origin <branch> && git checkout FETCH_HEAD`.

## Repo map

- `Anarkali_Web_Train.ipynb`: the one training notebook (Colab/Kaggle GPU). It builds data, trains,
  evaluates, exports and saves.
- `src/anarkali/web.py`: `page_state(html, url)` (the model input) and the schema.org labels. The same
  code runs at training and inference time.
- `src/anarkali/engine.py`: inference (`Engine.load(path).predict(state, questions)`). `server.py` is
  the HTTP API, `typed.py` is the question schema.
- `scripts/build_web_decisions.py`: builds the Common Crawl dataset.
- `scripts/train_anarkali.py`: training (use `--architecture packed`).
- `scripts/evaluate_checkpoint.py`: calibrated test report.
- `scripts/export_onnx.py`: ONNX release with a parity check.
- `release-150m/`: the old general 150M ONNX model (Git LFS). It is not trained for web decisions.

## Commands

```bash
pip install -e ".[train,onnx]" warcio
python -m unittest discover -s tests          # all tests; run them before every push
python scripts/build_web_decisions.py --warcs 4 --output artifacts/web-decisions-v0
```

## Rules for working here

- Keep `page_state` identical at training and inference. Changing it means retraining.
- Every new decision needs: a label function in `web.py`, tests with real-looking HTML, a row in the
  table above, and a result in the notebook report.
- No bot-block bypassing features. Detect blocks and stop. Respect robots.txt and site terms.
- Talk to the owner in short Hinglish, like a senior dev mentor.
