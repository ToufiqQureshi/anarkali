# Run history

Past training runs, kept as a record. run1 and run2 were narrow v0 experiments with only 4 page types
and two shop-page decisions. They are not the direction of the model (see `CLAUDE.md`); their main value
is the lessons at the bottom.

| Run | Decisions | Data | Training | Where |
|---|---|---|---|---|
| run1 (2026-10-05) | page_type, price_field, in_stock | CC-MAIN-2026-39, 4 WARCs, 85k HTML pages, 43k labelled; 44k train rows (4.8k old mixed) | Ettin-150M packed, 512 tokens, bs 16, stopped after ~82 min on a T4 (at least 1 of 3 epochs) | HF `toufiqqureshi651/anarkali-web` → `run1/` |
| run2 (2026-10-06) | page_type, price_field, in_stock | CC-MAIN-2026-39, 12 WARCs, 256k HTML pages (78k non-product skipped), 52k labelled; 59k train rows (4.8k old mixed) | from run1 `best.pt`, 512 tokens, bs 16, 3 epochs in ~2.6 h on a T4; best = epoch 1 (dev 79.6%, then 79.2%, 77.4%) | HF → `run2/` |
| run3 (2026-10-06) | page_type (22), page_status, language, published_date | web-expert-v1: CC-MAIN-2026-39, 12 + 3 error WARCs, 268k HTML pages; page_type/page_status rows re-read by Qwen3-4B (8.3k of 27.5k dropped); 61.9k train decisions | base Ettin-150M (new `page_state`), 512 tokens, bs 16, 2 epochs in ~1.9 h on a T4; dev 96.5% → 96.8% | HF → `run3/` |

## run3 test (unseen sites, calibrated by type, temperature 1.74)

| Decision | n | Model | Baseline | Confidence ≥ 0.9 |
|---|---|---|---|---|
| language | 8011 | 99.7% | 20.0% (random) | 100% of answers, 99.8% correct |
| page_status | 521 | 98.7% | 73.3% (majority ok) | 97%, 99.6% correct |
| page_type | 851 | 81.3% | 14.3% (majority) | 57%, 97.1% correct |
| published_date | 718 | 83.4% | 36.3% (random) | 60%, 98.6% correct |

Overall 97.0%, ECE 0.005. 76 of the 159 page_type errors are article ↔ blog_post ↔ news_article: sites pick
these schema.org types freely, so the label carries little page truth. page_status errors are all missed
soft-404s (not_found → ok). ONNX parity passed (0 of 200 answers changed); CPU p50 1.13 s on Kaggle
(optimized fp32), so the 500 ms bar fails.

int8 (local i7-9750H, `export_onnx.py`): per_channel flipped 29 of 200 answers, reduce_range 4,
matmul-only 5 (drift 0.72–0.97); best p50 1.20 s vs fp32 1.62 s. Static QDQ ran out of RAM. As in run2,
int8 does not pass parity and is not the way to 500 ms.

## run3 real-world test (`scripts/realworld_test.py`, local CPU, HF `run3/realworld/`)

1500 pages from 1435 sites outside the run's dataset (CC-MAIN-2026-39 is still the newest crawl, so new
sites, not newer pages): 1000 normal pages and 500 non-200 pages. 1.85 s per decision on the laptop CPU.

| Against automatic labels | n | Model | Baseline | Confidence ≥ 0.9 |
|---|---|---|---|---|
| language | 578 | 99.5% | 28.9% | 99%, 99.5% correct |
| page_status | 537 | 94.0% | 83.4% | 93%, 97.6% correct |
| page_type | 104 | 59.6% | 30.8% | 43%, 100% correct |
| published_date | 19 | 84.2% | 47.4% | too few |

On all pages: page_status is confident on 87%; page_type on only 33% of the 1000 normal pages.
Reading the 150-page review sample:
- Hard 404s are caught on every page, in every language. A few soft errors too (a 200 forum "Error"
  page → server_error, a 301 "page not found" → not_found).
- page_type is also answered on 404 pages, sometimes confidently (a 404 under `/product/` → product 0.99):
  the model leans on the URL. page_type must only be used when page_status is ok.
- Confident page_type on normal pages is mostly right (tag/category archives, products, home pages,
  forum threads, videos, news), but by eye about 1 in 10 is wrong (a forum index or settings page →
  forum_thread, a file catalog → forum_thread), so it is below the 95% bar off the test set.
- Common pages have no class: business service pages ("our services", "garage doors", "competences"),
  directory indexes ("Index of /"), parked, placeholder and maintenance pages. The model spreads them
  over article/blog_post with low confidence.
- Some pages are mojibake (GBK, TIS-620, ISO-8859-2): the builder decodes by the HTTP charset only, not
  `<meta charset>`. The same bug is in the training data.

## run1 test (unseen sites, uncalibrated argmax)

| Decision | n | Model | Baseline | Confidence ≥ 0.9 |
|---|---|---|---|---|
| page_type | 2617 | 76.9% | 39.2% (always "other") | 24% of pages, 94.9% correct |
| price_field | 165 | 46.7% | random pick (not computed) | 8% of pages, 100% correct |
| in_stock | 305 | 88.2% | 86.9% (always "true") | 2% of pages |

Overall dev 77.3%, calibrated ECE 0.020, 3.7% of answers change when options are reversed, GPU p50 47 ms.

## run2 test (unseen sites; a different split than run1, so not directly comparable)

| Decision | n | Model | Baseline | Confidence ≥ 0.9 |
|---|---|---|---|---|
| page_type | 3165 | 81.6% | 36.9% (majority) | 45% of pages, 94.0% correct |
| price_field | 509 | 55.6% | 25.8% (random pick) | 20% of pages, 99.0% correct |
| in_stock | 961 | 86.7% | 85.5% (majority) | 8% of pages, 98.8% correct |

Overall test 79.8%, calibrated-by-type ECE 0.013, 5.0% of answers change when options are reversed,
GPU p50 49 ms. ONNX parity passed, but CPU p50 is ~780 ms on Kaggle's CPU (fp32, 512 tokens).
Epochs 2 and 3 made dev worse: 1–2 epochs are enough when starting from a checkpoint.

int8 (`Anarkali_Int8_Export.ipynb`, Kaggle CPU, 4 vCPUs): all 4 recipes failed parity (max probability
drift 0.76–0.84, 16–145 of 200 answers flipped) and were only ~20% faster (int8 p50 ~1.16 s vs fp32
~1.45 s on that machine). int8 is not the way to the 500 ms bar; fewer tokens or a smaller encoder is.

## run2 real-world test (`Anarkali_RealWorld_Test.ipynb`, HF `run2/realworld/`)

6000 random pages from 5552 sites outside the run's dataset (same crawl), plus the Zyte product
benchmark (human labels).
- page_type vs schema.org (n=2833): 77.3% (majority 38.7%); at ≥ 0.9: 30% of pages, 91.8% correct.
  Only 24% of all pages get a confident page_type.
- in_stock on Zyte (n=140): 92.1%, below the 93.6% "always in stock" baseline; 0 of 9 out-of-stock found.
- price_field on Zyte (n=85): 62.4% (random 23.7%), confident on only 3.5%; the price candidates missed
  the true price on 49 of 140 pages.
- Reading the review sample: the state's first ~400 characters are mostly navigation menus; login,
  directory-index, forum, docs, soft-404 and placeholder pages had no class of their own; some
  schema.org labels are wrong (a "Not Found" page marked as a listing).

## Lessons

- The model needs page structure (main content vs navigation), many more page types, and labels checked
  against more than one source. Plain text with 4 page types is not enough for the real web.
- A T4 trains about one epoch of 44k rows at 512 tokens in 30–40 min.
- The notebook must show live training progress. Piping through `| tail` hid it for over an hour.
- The VS Code/Antigravity Colab extension cannot download files over ~512 MB, and has no Colab secrets
  panel. Save checkpoints to HF from inside the runtime; ask for the token with `getpass`.
- run2's first attempt crashed at `MAX_TOKENS=384`: an 8-option question needs up to 463 tokens of
  schema, and packing requires 32 tokens of state. The notebook preflight now checks every row.
- A Jupyter kernel started before `pip install -e` cannot import the package: add `src` to `sys.path`.
- Notebooks opened from `main` with `--depth 1` cannot `git checkout <branch>`. Use
  `git fetch origin <branch> && git checkout FETCH_HEAD`.
