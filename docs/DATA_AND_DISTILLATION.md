# Data and distillation: how to use this pipeline

This guide is written for people and for AI agents picking up the work. It explains what Anarkali is
for, what the data looks like, and the exact order of steps from raw data to a released 68M model.

## What Anarkali is

Anarkali is a **typed + general decision model**. Software gives it a state (any JSON: a ticket, an
invoice, an alert, an agent trace), a question and a fixed list of options. It returns a calibrated
probability for every option in about 10 ms on a CPU, and never generates text. The three
question types are `choice` (pick one), `noul` (true/false) and `score` (an ordinal level), the
same interface as Jev's `/v1/systemone`.

- **Typed decisions** means the public [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
  benchmark. It has four workflows (customer service, invoice processing, security incidents, agent
  traces), and it is the one Jev and Laya report on.
- **General decisions** means everything else: HR, returns, moderation, fraud, loans, email triage,
  and more. The 20 domains in `scripts/domains/catalog.json` plus the public datasets make the model
  general instead of a benchmark specialist.
- **Breadth:** `scripts/domains/catalog_wide.json` adds 220 more domains, 10 in each of 22 areas (finance,
  payments, investing, AI agents, security, support, sales, HR, legal, healthcare admin, e-commerce,
  logistics, IT, education, public sector, real estate, insurance, telecom, travel, manufacturing,
  energy, media). Each has one choice, one yes/no and one score question. It is built from
  `scripts/domains/build_wide_catalog.py`, one line per domain; rebuild after editing it.
- **Coding and CI data are parked for now.** The scripts stay in the repo (`generate_coding_decisions.py`,
  `import_agent_traces.py`, `realworld_benchmark.py`), but the distillation notebook does not use them.

## What a training row looks like

Every set in this repo is a directory with `train/development/calibration/test.jsonl` and a
`manifest.json` holding each file's sha256. One line is one decision:

```json
{"case_id": "customer_service_ab12::needs_human", "source_group": "customer_service::ab12",
 "workflow": "customer_service",
 "state": {"account": {"tier": "standard", "tenure_months": 14},
           "orders": [{"amount_usd": 299, "status": "settled"}],
           "thread": [{"role": "customer", "text": "I need an invoice for the $299 charge..."}]},
 "question": "True or false: This conversation requires a human agent rather than automated handling.",
 "question_type": "noul",
 "candidates": [{"id": "false", "text": "Automation can carry this to resolution."},
                {"id": "true", "text": "A human must take over."}],
 "target": [0.91, 0.09],
 "label_source": "gold | gold-soft | none | teacher:<name>",
 "teacher_targets": {"generator": [0.9, 0.1], "anarkali400m": [0.93, 0.07]}}
```

The rules every script keeps:

- `source_group` is the unit of splitting. One group never appears in two splits.
- `label_source: "none"` means the target is a uniform placeholder. `train_anarkali.py` refuses
  such rows unless a teacher fills them.
- **The test split is never teacher-labelled** and is read only after the model is chosen on
  development data.

## Where the data comes from

| Source | Script | Labels | Size |
|---|---|---|---|
| typed-decisions benchmark (Apache-2.0) | `prepare_typed_decisions.py` | three sampled LLM teachers (dataset's own) | 7.9k train rows |
| Public datasets: Civil Comments (CC0), Amazon polarity (Apache-2.0), CLINC150 (CC-BY-3.0), GoEmotions (Apache-2.0), CommonsenseQA (MIT), prompt-injections (Apache-2.0) | `harvest_public_decisions.py` | human (`gold`), soft where raters disagreed. The same texts with catalog questions go to `pool` for teachers | up to about 1.5M texts |
| Generated cases, from DeepSeek or another open model | `generate_domain_decisions.py` | the generator's own probabilities, then the 400M teacher, kept only where they agree | 20k rows per domain per run |
| Brio (colibri + Qwen3.6, optional) | `relabel_with_teachers.py --brio-teacher` | option probabilities read from a large open model | as many rows as time allows |

Do not use Claude (or any model whose terms forbid training competing models) to generate or label
training data. Use open-weight models: DeepSeek (MIT), Qwen (Apache-2.0), and similar.

## Why the generated data trains well

`generate_domain_decisions.py` builds every prompt from ideas in the synthetic-data literature:

- **Variety** (attributed prompts, Yu et al. 2023; Li et al. 2023). Each call draws an industry,
  region, organisation size, writer's tone and case length.
- **Hard cases** (dataset cartography, Swayamdipta et al. 2020). Each call is clear-cut, borderline,
  misleading, incomplete or conflicting.
- **Balance.** Each call steers the first choice question toward one option, cycling through all of
  them.
- **Soft labels** (Peterson et al. 2019). The generator gives each option the probability an expert
  panel would give it, and it is told not to be overconfident on ambiguous cases.
- **Agreement filter.** The generator's labels are one teacher and the 400M checkpoint is another.
  `combine_teachers.py --min-agreement 0.99` keeps a row only when both pick the same answer.
- **Benchmark structure.** The four benchmark workflows (`scripts/domains/benchmark.json`) include one
  example state from the typed-decisions train split, so generated cases match its field names.

See one prompt without calling anything:

```bash
python scripts/generate_domain_decisions.py --catalog scripts/domains/benchmark.json \
  --catalog scripts/domains/catalog.json --print-prompt customer_service --batch 8
```

## Step by step

### 0. The teacher (already done once)

Run `notebooks/Anarkali_V4.ipynb` with `ONLY_RECIPES = ['bb-ettin-400m']`. Keep
`/kaggle/working/bb-ettin-400m-best.pt`: that is the 400M teacher (78.7% on typed-decisions test).

### 1a. Free: generate on Kaggle's GPUs with Qwen (no API key)

`notebooks/Anarkali_Generate.ipynb` (built by `scripts/build_colab_notebook_generate.py`) does this for free:

1. Choose Kaggle **GPU T4 x2** and turn **Internet on**, then Run All.
2. The notebook installs vLLM and serves `Qwen/Qwen3-8B` (Apache-2.0) in fp16 across both T4s, as an
   OpenAI-compatible server on `127.0.0.1:8000`.
3. It runs `generate_domain_decisions.py` against that server for the ten domains below, with thinking
   switched off.
4. Every reply is cached in `/kaggle/working/synth/`, which survives as the notebook's output. When
   the 12-hour session ends, attach that output as a dataset in the next session and set `RESUME_FROM`.
   Nothing is generated twice.
5. The last cell prints rows per domain, the share with generator labels and the answer balance,
   then zips each domain.

**Breadth run:** set `USE_WIDE_CATALOG = True` to also generate the 220 wide domains into
`synth/synth-wide`, `WIDE_ROWS_PER_DOMAIN` rows each (600 by default: 200 cases per domain, about 5.5k
requests). Many tasks with fewer rows each generalise better than a few large tasks (Hsieh et al. 2023;
the zeroshot-v2.0 classifiers). It takes several Kaggle sessions; `RESUME_FROM` continues it. Attach
`synth-wide` to the distillation notebook like any other synthetic set.

Qwen3-8B writes somewhat weaker cases than DeepSeek. The 400M agreement filter in step 2 is what
keeps the bad ones out. Plan on one or two hours per domain on two T4s. That is an estimate, and
the report cell shows the real rate.

### 1b. Or: generate data with the DeepSeek API: 10 runs, 10 domains, about 20k rows each

DeepSeek's API is OpenAI-compatible. Set `DEEPSEEK_API_KEY`, and check the current model name and
price on DeepSeek's site; `deepseek-chat` is the non-thinking model, which is the right one for JSON.

| Run | Domain | Questions per case | Cases for 20k rows |
|---|---|---:|---:|
| 1 | customer_service (benchmark) | 5 | 4,000 |
| 2 | invoice_processing (benchmark) | 5 | 4,000 |
| 3 | security_incidents (benchmark) | 5 | 4,000 |
| 4 | agent_trace_observability (benchmark) | 5 | 4,000 |
| 5 | content_moderation | 3 | 6,667 |
| 6 | ecommerce_return | 3 | 6,667 |
| 7 | payment_fraud | 3 | 6,667 |
| 8 | it_helpdesk | 3 | 6,667 |
| 9 | email_intent | 3 | 6,667 |
| 10 | loan_application | 3 | 6,667 |

```bash
export DEEPSEEK_API_KEY=...
for d in customer_service invoice_processing security_incidents agent_trace_observability \
         content_moderation ecommerce_return payment_fraud it_helpdesk email_intent loan_application; do
  python scripts/generate_domain_decisions.py \
    --teacher deepseek=deepseek-chat@https://api.deepseek.com/v1 \
    --catalog scripts/domains/benchmark.json --catalog scripts/domains/catalog.json \
    --domains "$d" --rows-per-domain 20000 --batch 8 --workers 8 --max-tokens 7000 \
    --output "artifacts/synth-$d"
done
```

- A run stopped by a rate limit resumes from `generation-cache.jsonl`: run the same command again.
- Each run prints `rows_with_generator_labels`. If that count is far below the row count, the
  model is not following the reply format: look at a few replies in the cache.
- **No API key?** Paste the `--print-prompt` output into the DeepSeek chat, save each reply as
  `artifacts/replies/<domain>__<n>.txt`, then run
  `python scripts/generate_domain_decisions.py --catalog scripts/domains/benchmark.json --catalog scripts/domains/catalog.json --import-replies artifacts/replies --output artifacts/synth-chat`.
  One chat reply holds about 8 cases, so this path suits a few hundred cases, not 20k.

Upload the ten `artifacts/synth-*` folders to Kaggle as one dataset.

### 2. Run the distillation notebook

In `notebooks/Anarkali_Distill.ipynb`, cell 1:

- `TEACHER_CKPT`: the 400M `best.pt` from step 0.
- `SYNTHETIC_SETS`: the ten folders from step 1, e.g. `["/kaggle/input/anarkali-synth/synth-customer_service", ...]`.
- `PUBLIC_MAX_PER_SOURCE = 50000`: public data size. 0 means the full caps (Colab Pro A100).
- `BRIO_URL`: only if a colibri server is reachable.

Then Run All. The notebook:

1. prepares typed-decisions;
2. harvests public data;
3. labels the gold, pool and synthetic sets with the 400M teacher;
4. mixes the teachers (`combine_teachers.py`) and drops the synthetic rows the teachers disagree on;
5. merges everything;
6. trains three 68M models, each selected epoch by epoch on typed-decisions development cases:

| Run | What it is | What it measures |
|---|---|---|
| `distilled` | all data + the 400M teacher online | the full recipe |
| `distilled-base` | typed-decisions only + the teacher | whether the extra data helps or dilutes |
| `control` | all data, no teacher | what distillation adds |
| `distilled-150m` | all data + the teacher, on Ettin-150M (`EXTRA_STUDENTS`) | what a bigger student buys |

The winner on development data is evaluated on the typed-decisions test split, exported to ONNX,
and zipped to `/kaggle/working/anarkali-distill-backup.zip`. A bigger student wins only if it beats
the best 68M run by `BIGGER_MIN_GAIN` (1.5 points) on development data, because it is about twice as
slow on CPU. The test table prints every run's accuracy, ECE and p50 latency side by side.

Why Ettin-150M and not another backbone: at 150M to 230M no public encoder is clearly ahead of the
ModernBERT/Ettin family on English tasks. LiquidAI's LFM2.5-Encoder-230M leads its own 17-task table
because of multilingual tasks, but on English MNLI, SST-2 and QNLI it is level with or behind
ModernBERT-base, its licence stops being free above $10M annual revenue, and its convolution layers
may not work with the packed shared-position input. Ettin-150M is MIT and drops into the existing code.

### 3. Decide what to ship

Compare against the published numbers: Laya 76.6%, Anarkali 0.3.0 74.0%, Jev 72.7%, 400M teacher 78.7%.

- Ship only if the winner beats 0.3.0 on accuracy with ECE no worse than 0.135 + 0.01.
- If `distilled-base` beats `distilled`, the extra data diluted the benchmark. Keep the data for
  generality, but report both numbers.

## For AI agents changing this code

- After changing anything in `src/` or `scripts/`, rebuild every notebook before pushing:
  `python scripts/build_colab_notebook.py && python scripts/build_colab_notebook_v4.py && python scripts/build_colab_notebook_distill.py && python scripts/build_colab_notebook_generate.py`.
  After changing `scripts/domains/build_wide_catalog.py`, run it too: a test checks that `catalog_wide.json` is current.
  The V3 notebook embeds the source, and CI fails when it is stale.
- Tests: `python -m unittest discover -s tests`. The distillation pipeline is covered by
  `tests/test_distill_pipeline.py`, `tests/test_synthetic_generation.py`, `tests/test_domain_generation.py`,
  `tests/test_wide_catalog.py`, `tests/test_notebook_generate.py` and `tests/test_notebook_distill.py`. Everything runs offline with fakes.
- Invariants that tests enforce:
  - no test split is ever labelled or read before the winner is fixed;
  - source groups never cross splits;
  - share-alike data is opt-in;
  - generated rows keep `label_source: "none"` until a teacher fills them.
