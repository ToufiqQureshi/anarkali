# freegpu: one long GPU job across free providers

Free GPUs come in short sessions: Colab cuts you off after a few hours, and Kaggle gives about 30 GPU hours a week. `freegpu` splits a long job across all of them. Each session does as much as it can and saves its progress, and the next session, on any provider, carries on from there.

It runs [`scripts/relabel_with_teachers.py`](../scripts/relabel_with_teachers.py): open LLM teachers served with vLLM label Anarkali's training data.

## The pieces

| File | What it is |
|---|---|
| `hub.py` | **Shared storage.** A private Hugging Face dataset repo (or any shared folder) holding the job, the progress and the result. Every worker reads and writes here, so no machine needs to stay alive. |
| `worker.py` | **Does the work** on whatever GPU it runs on. Detects Colab, Kaggle or other, starts the teacher model, saves progress to the hub every 10 minutes, and stops cleanly before the session ends. |
| `worker.ipynb` | **The one notebook for every provider.** Open it on Colab, Kaggle or any Jupyter GPU box and press Run all. |
| `orchestrator.py` | **The scheduler.** Uploads a job (`submit`), shows progress (`status`), and decides where the next worker runs (`tick`). |
| `job.example.json` | Example job: two Apache-2.0 teachers that fit a single 16 GB T4. |
| [`../.github/workflows/freegpu.yml`](../.github/workflows/freegpu.yml) | Runs `tick` every hour on GitHub Actions for free. |

## How switching works

```
every hour, tick:
  job finished?                     → nothing to do
  a worker saved in the last 25 min → it is still running, wait
  otherwise, try providers in order:
    Kaggle → weekly quota left? start the worker through the Kaggle API (automatic)
    Colab  → open a GitHub issue with a one-click notebook link (you press Run all)
  a provider that failed or never answered is skipped for 24 h, then tried again
```

Colab has no API to start a notebook, and driving it with a browser bot breaks its terms, so Colab is the one step that needs a click. Everything else is automatic.

## Setup (once)

1. Create a Hugging Face token with write access.
2. In the GitHub repo settings:
   - **Variables:** `ANARKALI_HUB` = `hf:<your-hf-name>/anarkali-work`, and `ANARKALI_JOB` = any job name.
   - **Secrets:** `HF_TOKEN`, plus `KAGGLE_USERNAME` and `KAGGLE_KEY` (from kaggle.com → Settings → API).
3. Upload the job:
   ```bash
   export HF_TOKEN=...
   python freegpu/orchestrator.py --hub hf:<you>/anarkali-work --job <job> submit \
     --config freegpu/job.example.json --input artifacts/typed-decisions-v2
   ```
4. After the first Kaggle run, open the `anarkali-gpu-worker` notebook on Kaggle once and attach a secret named `HF_TOKEN` (Add-ons → Secrets).

To check progress:

```bash
python freegpu/orchestrator.py --hub hf:<you>/anarkali-work --job <job> status
```

When every teacher is done, the relabelled data is in `jobs/<job>/output/` in the hub repo.

## Run a worker by hand

Open [`worker.ipynb`](worker.ipynb) on any provider, set `HUB` and `JOB`, and press Run all. Or from a shell on any GPU machine:

```bash
export HF_TOKEN=...
python freegpu/worker.py --hub hf:<you>/anarkali-work --job <job> --max-hours 10
```

Two workers at once are fine: each writes its own progress file, so they never overwrite each other.

## Rules

- **One account per provider.** This spreads one job across the free quota you have. It does not get around the limits.
- **GPU:** vLLM needs an NVIDIA T4 or newer. Kaggle's P100 will not work, so the orchestrator asks Kaggle for a T4.
- **Licences:** check the licence of every teacher model before training on its outputs. The example teachers are Apache-2.0.
