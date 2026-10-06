# Run history

Past training runs, kept as a record. run1 and run2 were narrow v0 experiments with only 4 page types
and two shop-page decisions. They are not the direction of the model (see `CLAUDE.md`); their main value
is the lessons at the bottom.

| Run | Decisions | Data | Training | Where |
|---|---|---|---|---|
| run1 (2026-10-05) | page_type, price_field, in_stock | CC-MAIN-2026-39, 4 WARCs, 85k HTML pages, 43k labelled; 44k train rows (4.8k old mixed) | Ettin-150M packed, 512 tokens, bs 16, stopped after ~82 min on a T4 (at least 1 of 3 epochs) | HF `toufiqqureshi651/anarkali-web` → `run1/` |
| run2 (2026-10-06) | page_type, price_field, in_stock | CC-MAIN-2026-39, 12 WARCs, 256k HTML pages (78k non-product skipped), 52k labelled; 59k train rows (4.8k old mixed) | from run1 `best.pt`, 512 tokens, bs 16, 3 epochs in ~2.6 h on a T4; best = epoch 1 (dev 79.6%, then 79.2%, 77.4%) | HF → `run2/` |

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
