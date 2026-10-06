# Training run checklist: never lose a run

Use this for every Kaggle or Colab training run. Everything a run produces must end up in the private
HF repo `toufiqqureshi651/anarkali-web` under `runN/`. Kaggle's `/kaggle/working` is deleted when the
session ends.

## Before the run

1. **Code on GitHub.** The notebook clones `main`. Any change to `src/` or `scripts/` must be pushed
   first, or Kaggle runs the old code. A notebook-only change does not need a push.
2. **Run the tests locally:** `python -m unittest discover -s tests`.
3. **Check the settings cell:**
   - `RUN_NAME` is new (for example `run3`), so an old run is never overwritten.
   - `INIT_FROM` points to the previous run's `best.pt`.
   - `DATA_FROM` reuses a dataset already on HF. Set `DATA_FROM = ""` only when the data must change.
   - `MAX_TOKENS = 512`. Never go below the "largest question+options size" that the preflight prints
     (463 in run2), plus 32.
   - `EPOCHS` fits the time limit (see below).
4. **Kaggle setup:**
   - Secret `HF_TOKEN` is attached (Add-ons → Secrets), and Internet is on.
   - GPU is a T4 (x1 or x2). The trainer uses one GPU, so x2 is not faster.
5. **Use the right notebook version.** After File → Import Notebook, search for a line you just
   changed. If it is not there, the old version was imported.
6. **Start with Save Version → Save & Run All (Commit).** Do not train in the interactive draft: it
   stops when the browser is closed or idle.

## Time limits (Kaggle)

- A commit run stops after **12 hours**. GPU quota is about **30 hours a week**.
- Speed is about 1 epoch of 44k rows in 30–40 min (512 tokens, T4, batch 16). Estimate
  `rows / 44k × 40 min × EPOCHS`, and keep it under ~10 hours.
- Do not start a second GPU notebook while a run is going. It uses the same quota.

## While it runs

- **Preflight (first ~8 min).** It must print `PREFLIGHT OK: the long training can start`. If it
  fails, only minutes were lost: fix the bug, re-import and re-run.
- **Do not cancel** after the preflight unless something is clearly wrong.
- **What is saved automatically:**
  - after every cell: `runN/notebook-log.txt`;
  - every time a better epoch is saved: `runN/web-run/best.pt` and `training.json`, within ~2 min;
  - optional steps (Julia baseline, demo, ONNX) cannot stop the final save if they fail.
- **Check progress** on Kaggle (Logs tab), or on HF (`runN/notebook-log.txt`, `runN/web-run/best.pt`
  date).

## If it crashes or times out

1. **Do not panic.** The data and the last best epoch are on HF.
2. Read the error at the end of `runN/notebook-log.txt`.
3. Resume instead of starting over:
   - `INIT_FROM = "runN/web-run/best.pt"`;
   - `DATA_FROM = "runN/web-decisions-v0"`;
   - set `EPOCHS` to the epochs still left.
4. Add the lesson to `CLAUDE.md` (Lessons), so the same failure does not happen twice.

## After the run

1. **Check that HF `runN/` has everything:**
   - `web-decisions-v0/` (dataset; for a reused dataset it stays under the run that built it);
   - `web-run/best.pt`, `training.json`, `test-report.json`;
   - `notebook-log.txt`;
   - `release-web/` (ONNX).
2. **Compare with the previous run** on every decision's test score. A drop on any decision blocks
   the release.
3. Add a row to the run log in `CLAUDE.md`. Only use numbers from `test-report.json`.
4. Rotate any token that was shared in a chat or a log.

## Rules for changing the notebook

- Every new code path must also run in the preflight on the tiny data. The long training is not the
  place to find bugs.
- Never pipe training output through `| tail` or `| head`. Progress must be live.
- Optional cells go inside `try/except`, so they cannot block the save.
- Anything new that the run produces must be uploaded to HF by the notebook itself.
- Never put a token in the notebook. Use the Kaggle secret.

## Failures so far (do not repeat)

| Run | What failed | Fix |
|---|---|---|
| run1 | `| tail` hid progress for over an hour | live output |
| run1 | files over ~512 MB could not be downloaded from Colab | upload to HF from inside the runtime |
| run2 try 1 | `MAX_TOKENS=384`: price_field rows need up to 463 tokens of schema | keep 512; the preflight checks every row |
| run2 try 2 | `No module named anarkali.packing`: the kernel started before `pip install -e` | `sys.path.insert(0, REPO/src)` |
