# Anarkali

## Anarkali is a web expert

Anarkali is a small, fast model that **understands any web page on any site**: news, blogs, docs,
forums, government, SaaS, job boards, directories, shops, everything. Give it a page and a small typed
question (pick one option, true/false, or a score) and it answers in milliseconds on a CPU, with a
calibrated confidence. When it is unsure it says so, and the page goes to an LLM or a person.

It sits in front of whatever a team already runs on web pages (scrapers, crawlers, monitors, LLM
pipelines) and makes the easy decisions itself, cheaply and the same way every time.

**Business goal: revenue.** We sell cost and noise reduction:
- **On top of an LLM:** it answers most pages itself, so far fewer pages need an LLM call. The headline
  number is "% of LLM calls saved at the same accuracy".
- **On top of code:** it stops bad pages and false alerts from entering a pipeline. The headline number
  is "% of bad records or alerts blocked".

Never publish a number that an eval in this repo has not produced.

## What it does

Questions that come up on every site, not on one kind of site:

| Decision | The question | Why every pipeline needs it |
|---|---|---|
| `page_type` | What kind of page is this? (~25 types: home, article/news, blog post, docs, forum thread, Q&A, listing/category, search results, profile, about, contact, login/signup, legal/policy, job post, event, recipe, video, product, directory index, …) | the first thing any scraper or crawler must know |
| `page_status` | Is this a real page, or an error, soft-404, login wall, paywall, bot-block/captcha, parked domain, placeholder or empty JS shell? | keeps junk out of the pipeline |
| `main_content` | Which block is the real content, and which is navigation, ads or footer? | the base of every extraction |
| field location | Which text is the title / publish date / author / main heading? | needed on almost every site |
| `change_real` | This page changed: is it a real change, or only dates, ads or layout? | every monitoring job |
| `follow_link` | Is this link worth crawling (detail, pagination, navigation, junk)? | every crawler |
| `answerable` | Does this page answer question X? | decides whether to send the page to an LLM |
| site category, language | What is the site about, and what language is the page in? | filtering and routing |

Add decisions one at a time. A decision ships only when it passes the quality bar below.

## What the model sees

`page_state` must carry the page's **structure**, not only its text: which text is a heading, inside
`<nav>`, `<main>`, `<article>`, `<footer>`, a `<form>` (and which fields, such as password), a table,
a `<time>`, and a short summary of links. Drop scripts, styles, classes and ids. A model that only
sees flattened text reads navigation menus first and cannot tell a login wall from an article
(see `RUNS.md`).

## Where the labels come from

Every label must be real, and checked against more than one source where it can be:
- the page itself: schema.org types (all of them, not one), OpenGraph, `<html lang>`, canonical,
  `datePublished`/`author`, robots meta;
- the crawl: HTTP status, redirects and non-200 responses (Common Crawl crawldiagnostics), the same URL
  across crawls for change decisions;
- human-labelled public datasets with a license that allows commercial use (check each one);
- a gold test set checked by people, across all page types and many languages. This is the score we
  publish.

LLM teacher labels may help training only if the teacher's terms allow training a model we sell
(OpenAI and Anthropic terms do not), and never as the test.

## Production-grade bar (every decision, every release)

1. **Real labels only.** Synthetic or LLM-generated data alone is not enough. (The old CI-triage
   model scored 7.7% on real logs against a 75% majority baseline because it was trained on synthetic
   data.) A label source must never appear in the model's input.
2. **Unseen sites in test.** Splits are by host. The test split is read once, after the model is chosen.
3. **Beat the baselines.** A release must clearly beat majority-label, simple rules and, where it
   applies, an LLM zero-shot run.
4. **Calibration.** When confidence is 0.9 or higher, accuracy must be 95% or higher. Report the
   coverage at that threshold.
5. **Latency.** p50 under 500 ms on CPU, batch 1. int8 ships only if the parity check passes.
6. **Real-world check.** Before a release, run `scripts/realworld_test.py` (CPU) on fresh pages from
   unseen sites and read the review sample.
7. **Never lose weights.** Keep `best.pt` with every run in the private HF repo. ONNX alone cannot be
   fine-tuned.

## How to add a decision without forgetting the old ones

One model serves every decision, so every new version is trained on **all** decisions at once
(multi-task). Training only on the new decision makes the model forget the old ones.

1. Add the label function for the new decision in `web.py`, plus tests with real-looking HTML.
   `page_decisions` must emit the old decisions **and** the new one, so one dataset build covers all.
2. Rebuild the dataset with the same `--seed`.
3. Start from the previous version's `best.pt` with `train_anarkali.py --init-checkpoint` (the notebook's
   `INIT_FROM`), unless `page_state` changed; then train from the base model.
4. Regression gate: the new version must match or beat the previous one on every old decision's test
   score and pass the bar on the new one. A drop on an old decision blocks the release.
5. Upload `best.pt`, the dataset and `test-report.json` to `toufiqqureshi651/anarkali-web` (private) under
   `runN/`, and add the run to `RUNS.md`.

## Repo map

- `Anarkali_Web_Train.ipynb`: the training notebook (Colab/Kaggle GPU). It builds data, trains,
  evaluates, exports and saves. Follow `RUN_CHECKLIST.md` for every run.
- `scripts/realworld_test.py`: scores a finished run on fresh pages from unseen sites and saves a review
  sample (CPU, local).
- `RUNS.md`: past runs, results and lessons.
- `src/anarkali/web.py`: `page_state(html, url)` (the model input) and the labels. The same code runs
  at training and inference time.
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

- Anarkali is for the whole web. Do not steer data, decisions or examples toward one kind of site
  (such as shops); product pages are one page type among many.
- Keep `page_state` identical at training and inference. Changing it means retraining.
- Every new decision needs: a label function in `web.py`, tests with real-looking HTML, a row in the
  table above, and a result in the notebook report.
- No bot-block bypassing features. Detect blocks and stop. Respect robots.txt and site terms.
- Talk to the owner in short Hinglish, like a senior dev mentor.
