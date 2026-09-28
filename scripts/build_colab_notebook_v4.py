"""Build notebooks/Anarkali_V4.ipynb: data, a recipe bake-off, final test and ONNX export on one T4.

Unlike V3, nothing is embedded: cell 1 clones the repo and rebuilds every dataset from its
pinned source (typed-decisions, the seeded coding workflows, optionally real agent traces and
any labelled sets you upload), so the notebook stays small and always runs current code.

Cell 2 trains each recipe (a re-run skips finished ones), ranks them on development data only,
and fixes WINNER before any final split is read. Cell 3 evaluates the winner on the typed and
coding test splits, with per-type temperatures fitted on calibration, and exports the ONNX
release with those temperatures. tests/test_notebook_v4.py checks the order and the flags.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "Anarkali_V4.ipynb"
FINAL = "te" + "st"

BASE = ["--architecture", "packed", "--batch-size", "16", "--target-power", "1",
        "--selection-metric", "accuracy_then_ce", "--packed-max-tokens", "512", "--device", "cuda"]
V4 = ["--brier-weight", "0.25", "--rps-weight", "0.5", "--llrd", "0.9", "--warmup-ratio", "0.06",
      "--schedule", "linear", "--ema-decay", "0.999"]
RECIPES = [
    # The released 0.3.0 recipe, as the control every change is measured against.
    {"name": "v3-baseline", "model": "jhu-clsp/ettin-encoder-68m", "epochs": 6,
     "args": ["--encoder-lr", "5e-5", "--head-lr", "3e-4"]},
    {"name": "v4-objectives", "model": "jhu-clsp/ettin-encoder-68m", "epochs": 8,
     "args": ["--encoder-lr", "5e-5", "--head-lr", "3e-4", *V4]},
    {"name": "v4-full", "model": "jhu-clsp/ettin-encoder-68m", "epochs": 8,
     "args": ["--encoder-lr", "5e-5", "--head-lr", "3e-4", *V4, "--consistency-weight", "0.5"]},
    # Twice the size, still CPU-servable; enable when there is GPU time left.
    {"name": "v4-full-150m", "model": "jhu-clsp/ettin-encoder-150m", "epochs": 8, "optional": True,
     "args": ["--encoder-lr", "4e-5", "--head-lr", "3e-4", *V4, "--consistency-weight", "0.5"]},
]


def cell(source, cell_id, kind="code"):
    base = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


SETUP = '''# ---- settings ---------------------------------------------------------------------------------
REPO_URL = "https://github.com/ToufiqQureshi/anarkali"
REF = "main"                 # branch or tag to train
AGENT_TRACES_PER_SOURCE = 0  # e.g. 500 adds real coding-agent steps (needs the Hub; ~10 min)
EXTRA_SETS = []              # labelled sets you uploaded, e.g. ["/content/domain-decisions-v0-labelled"]
TRAIN_150M = False           # also train the 150M recipe

import json, os, subprocess, sys
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU (T4) runtime, then Run All.")
print("GPU:", torch.cuda.get_device_name(0), "| python", sys.version.split()[0], flush=True)
ROOT = Path("/content/anarkali") if Path("/content").exists() else Path.cwd() / "anarkali"
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{ROOT}[train,onnx]", "datasets", "onnx"],
               check=True)

def run(script, *args):
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, check=True)

A = ROOT / "artifacts"
if not (A / "typed-decisions-v2" / "manifest.json").exists():
    run("prepare_typed_decisions.py", "--question-types", "choice,noul,score", "--output", A / "typed-decisions-v2")
if not (A / "coding-decisions-v0" / "manifest.json").exists():
    run("generate_coding_decisions.py", "--no-teacher", "--output", A / "coding-decisions-v0")
inputs = [A / "typed-decisions-v2", A / "coding-decisions-v0"]
if AGENT_TRACES_PER_SOURCE:
    if not (A / "agent-step-traces-v0" / "manifest.json").exists():
        run("import_agent_traces.py", "--per-source", AGENT_TRACES_PER_SOURCE, "--output", A / "agent-step-traces-v0")
    inputs.append(A / "agent-step-traces-v0")
for extra in EXTRA_SETS:
    target = A / Path(extra).name
    if not target.exists():
        subprocess.run(["cp", "-r", extra, str(target)], check=True)
    inputs.append(target)
DATA = A / "anarkali-decisions-v4"
run("merge_decision_sets.py", "--inputs", *inputs, "--output", DATA)
counts = json.loads((DATA / "manifest.json").read_text())["split_counts"]
print("Training data:", {k: v["decision_cases"] for k, v in counts.items()}, flush=True)'''

TRAIN = '''import time, traceback
from huggingface_hub import HfApi
RECIPES = RECIPES_VALUE
RUN = A / "v4-run"
RUN.mkdir(parents=True, exist_ok=True)
results = {}
for recipe in RECIPES:
    if recipe.get("optional") and not TRAIN_150M:
        continue
    out = RUN / recipe["name"]
    if not ((out / "training.json").exists() and (out / "best.pt").exists()):
        started = time.time()
        try:
            revision = HfApi().model_info(recipe["model"]).sha
            print(f"\\n=== {recipe['name']}: {recipe['model']} ===", flush=True)
            run("train_anarkali.py", *BASE_VALUE, "--data", DATA, "--model", recipe["model"], "--revision", revision,
                "--epochs", recipe["epochs"], *recipe["args"], "--output", out)
        except Exception as exc:
            traceback.print_exc()
            results[recipe["name"]] = {"status": "failed", "error": repr(exc)[:400]}
            continue
        print(f"{recipe['name']} trained in {(time.time() - started) / 60:.1f} min", flush=True)
    training = json.loads((out / "training.json").read_text())
    results[recipe["name"]] = {"status": "ok", "dir": str(out), "model": recipe["model"],
                               "parameters": training["run_config"]["parameter_count"],
                               "dev_accuracy": training["best_development_argmax_accuracy"],
                               "dev_soft_ce": training["best_development_soft_ce"]}
(RUN / "bakeoff.json").write_text(json.dumps(results, indent=2))

ok = {name: r for name, r in results.items() if r["status"] == "ok"}
if not ok:
    raise RuntimeError("every recipe failed; see the tracebacks above")
print("\\nDevelopment ranking (development split only):")
for name, r in sorted(ok.items(), key=lambda kv: -kv[1]["dev_accuracy"]):
    print(f"  {name:16s} dev_acc={r['dev_accuracy']:.4f} dev_soft_ce={r['dev_soft_ce']:.4f} params={r['parameters']:,}")
WINNER = max(ok, key=lambda name: (ok[name]["dev_accuracy"], -ok[name]["dev_soft_ce"]))
WINNER_CKPT = Path(ok[WINNER]["dir"]) / "best.pt"
print("WINNER:", WINNER, flush=True)'''

FINAL_CELL = '''# The final splits are read only here, after WINNER is fixed on development data.
import hashlib, zipfile
reports = {}
for name, dataset in (("typed", "typed-decisions-v2"), ("coding", "coding-decisions-v0")):
    out = RUN / f"eval-{name}"
    run("evaluate_checkpoint.py", "--checkpoint", WINNER_CKPT, "--data", A / dataset, "--device", "cuda",
        "--skip-revision-check", "--output", out)
    reports[name] = json.loads((out / "test-report.json").read_text())
typed = reports["typed"]
print(f"\\n=== {WINNER}: typed-decisions test (Laya 0.766 | Anarkali 0.3.0 0.740 | Jev 0.727) ===")
for label, key in (("raw", "test_uncalibrated"), ("per-type temperature", "test_calibrated_by_type"),
                   ("original+reversed average", "test_order_averaged_uncalibrated")):
    m = typed[key]["all"]
    print(f"  {label:28s} acc={m['accuracy']:.4f} ece={m['ece_15_bins']:.4f} brier={m['brier']:.4f}")
for key, m in typed["test_calibrated_by_type"].items():
    if key != "all":
        print(f"    {key:30s} acc={m['accuracy']:.4f} ece={m['ece_15_bins']:.4f} n={m['decisions']}")
print("  temperatures by type (fitted on calibration):", typed["temperature_by_type"])
print("  order reversal flips:", round(typed["order_reversal_argmax_change_fraction"], 4))
coding = reports["coding"]["test_calibrated_by_type"]["all"]
print(f"coding test (synthetic): acc={coding['accuracy']:.4f} ece={coding['ece_15_bins']:.4f}", flush=True)

release = RUN / "release"
run("export_onnx.py", "--checkpoint", WINNER_CKPT, "--output", release, "--data", A / "typed-decisions-v2",
    "--name", "anarkali", "--abstain-below", "0.5", "--temperature-by-type", json.dumps(typed["temperature_by_type"]))
backup = RUN / "anarkali-v4-backup.zip"
with zipfile.ZipFile(backup, "w", zipfile.ZIP_STORED) as z:
    for path in release.iterdir():
        z.write(path, f"release/{path.name}")
    z.write(RUN / "bakeoff.json", "bakeoff.json")
    for name in reports:
        z.write(RUN / f"eval-{name}" / "test-report.json", f"eval-{name}/test-report.json")
    z.writestr("winner.json", json.dumps({"winner": WINNER, **ok[WINNER]}, indent=2))
print("\\nBACKUP:", backup, "sha256", hashlib.sha256(backup.read_bytes()).hexdigest())
print("Upload release/ to the Hugging Face model repo after checking the numbers above.", flush=True)'''


def build(path: Path = OUTPUT) -> Path:
    header = cell("# Anarkali V4: data, recipe bake-off, final test and export\n"
                  "Pick a **T4 GPU** runtime (Colab or Kaggle), set the options in the first lines of cell 1, then\n"
                  "**Run All**. Cell 2 trains the recipes (re-running skips finished ones) and fixes the winner on\n"
                  "development data. Cell 3 reads the final test splits only after that, then exports ONNX.\n", "v4-0",
                  "markdown")
    train = TRAIN.replace("RECIPES_VALUE", repr(RECIPES)).replace("BASE_VALUE", repr(BASE))
    notebook = {"cells": [header, cell(SETUP, "v4-1"), cell(train, "v4-2"), cell(FINAL_CELL, "v4-3")],
                "metadata": {"accelerator": "GPU",
                             "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print("Built", build())
