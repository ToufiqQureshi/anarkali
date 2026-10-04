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
