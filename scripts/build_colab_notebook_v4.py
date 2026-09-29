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
    # Options share position IDs, so the model cannot see their order at all (exact invariance,
    # tested); the consistency term would be zero, so it is left out.
    {"name": "v4-shared", "model": "jhu-clsp/ettin-encoder-68m", "epochs": 8,
     "args": ["--encoder-lr", "5e-5", "--head-lr", "3e-4", *V4, "--shared-option-positions"]},
    # Backbone bake-off (BACKBONE_BAKEOFF = True): the same V4 objectives as v4-objectives, so only
    # the encoder changes. All are ModernBERT-architecture models that load without remote code.
    # NeoBERT is left out because it needs trust_remote_code.
    {"name": "bb-ettin-150m", "model": "jhu-clsp/ettin-encoder-150m", "epochs": 8, "optional": True,
     "args": ["--encoder-lr", "4e-5", "--head-lr", "3e-4", *V4]},
    {"name": "bb-ettin-400m", "model": "jhu-clsp/ettin-encoder-400m", "epochs": 6, "optional": True,
     "args": ["--encoder-lr", "3e-5", "--head-lr", "3e-4", "--batch-size", "8", *V4]},
    {"name": "bb-modernbert-base", "model": "answerdotai/ModernBERT-base", "epochs": 8, "optional": True,
     "args": ["--encoder-lr", "4e-5", "--head-lr", "3e-4", *V4]},
    # Multilingual, for Hindi and Hinglish users later.
    {"name": "bb-mmbert-small", "model": "jhu-clsp/mmBERT-small", "epochs": 8, "optional": True,
     "args": ["--encoder-lr", "5e-5", "--head-lr", "3e-4", *V4]},
]


def cell(source, cell_id, kind="code"):
    base = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


SETUP = '''# ---- settings ---------------------------------------------------------------------------------
REPO_URL = "https://github.com/ToufiqQureshi/anarkali"
REF = "main"                 # branch or tag to train
AGENT_TRACES_PER_SOURCE = 0  # e.g. 500 adds real coding-agent steps (needs the Hub; ~10 min)
EXTRA_SETS = []              # labelled sets you uploaded, e.g. ["/content/domain-decisions-v0-labelled"]
BACKBONE_BAKEOFF = False     # also train Ettin-150M/400M, ModernBERT-base, mmBERT-small (hours on a T4)
ONLY_RECIPES = []            # train just these recipe names, in this order; [] trains every enabled recipe
SILENCE_LIMIT_MIN = 30       # kill a script that prints nothing for this long, instead of burning GPU hours
EXPORT_INT8 = True           # False skips the int8 recipes (slow on 400M encoders; 0.3.0's all failed parity)

import collections, json, os, queue, subprocess, sys, threading, time
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU (T4) runtime, then Run All.")
GPUS = torch.cuda.device_count()
print("GPU:", GPUS, "x", torch.cuda.get_device_name(0), "| python", sys.version.split()[0], flush=True)
ROOT = Path("/content/anarkali") if Path("/content").exists() else Path.cwd() / "anarkali"
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{ROOT}[train,onnx]", "datasets", "onnx"],
               check=True)
LOGS = ROOT / "artifacts" / "logs"

def run(script, *args, gpu=None, tag=""):
    """Run a script with its stdout and stderr in the cell and in a log file.

    A failure raises with the script's last lines, so the real error is never hidden, and a script
    that stays silent for SILENCE_LIMIT_MIN is killed so a hang cannot use up the GPU quota.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{time.strftime('%H%M%S')}-{tag or Path(script).stem}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1",
               TRANSFORMERS_VERBOSITY="error")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines, tail = queue.Queue(), collections.deque(maxlen=60)

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    prefix = f"[{tag}] " if tag else ""
    with log.open("w", encoding="utf-8") as stream:
        while True:
            try:
                line = lines.get(timeout=SILENCE_LIMIT_MIN * 60)
            except queue.Empty:
                proc.kill()
                raise RuntimeError(f"{script} printed nothing for {SILENCE_LIMIT_MIN} min and was killed; log {log}")
            if line is None:
                break
            print(prefix + line, end="", flush=True)
            stream.write(line)
            stream.flush()
            tail.append(line)
    code = proc.wait()
    if code:
        raise RuntimeError(f"{script} exited with {code}; last lines:\\n" + "".join(tail))

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

TRAIN = '''import time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
RECIPES = RECIPES_VALUE
RUN = A / "v4-run"
RUN.mkdir(parents=True, exist_ok=True)
by_name = {recipe["name"]: recipe for recipe in RECIPES}
if ONLY_RECIPES:
    todo = [by_name[name] for name in ONLY_RECIPES]
else:
    todo = []
    for recipe in RECIPES:
        if recipe.get("optional") and not BACKBONE_BAKEOFF:
            continue
        todo.append(recipe)
free_gpus = queue.Queue()
for gpu in range(GPUS):
    free_gpus.put(gpu)

def train(recipe):
    """Train one recipe on a free GPU; with two GPUs (Kaggle T4 x2) two recipes run at once."""
    out = RUN / recipe["name"]
    if not ((out / "training.json").exists() and (out / "best.pt").exists()):
        gpu = free_gpus.get()
        started = time.time()
        try:
            revision = HfApi().model_info(recipe["model"]).sha
            print(f"\\n=== {recipe['name']}: {recipe['model']} on GPU {gpu} ===", flush=True)
            run("train_anarkali.py", *BASE_VALUE, "--data", DATA, "--model", recipe["model"], "--revision", revision,
                "--epochs", recipe["epochs"], *recipe["args"], "--output", out, gpu=gpu, tag=recipe["name"])
        except Exception as exc:
            print(f"\\n!!! {recipe['name']} FAILED:\\n{exc}", flush=True)
            return {"status": "failed", "error": str(exc)[-3000:]}
        finally:
            free_gpus.put(gpu)
        print(f"{recipe['name']} trained in {(time.time() - started) / 60:.1f} min", flush=True)
    training = json.loads((out / "training.json").read_text())
    return {"status": "ok", "dir": str(out), "model": recipe["model"],
            "parameters": training["run_config"]["parameter_count"],
            "dev_accuracy": training["best_development_argmax_accuracy"],
            "dev_soft_ce": training["best_development_soft_ce"]}

with ThreadPoolExecutor(max_workers=GPUS) as pool:
    results = dict(zip([r["name"] for r in todo], pool.map(train, todo)))
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
raw, cal = typed["development_uncalibrated"], typed["development_calibrated_by_type"]
# Ship the fitted temperatures only if they improve calibration on development data, which the fit did
# not see; the test split only reports, it never decides. 0.3.0 was already calibrated.
use_temperatures = cal["ece_15_bins"] < raw["ece_15_bins"] and cal["soft_ce"] < raw["soft_ce"]
print("per-type temperatures", "shipped" if use_temperatures else "not shipped (no calibration gain)", flush=True)
int8_flags = [] if EXPORT_INT8 else ["--no-int8"]
run("export_onnx.py", "--checkpoint", WINNER_CKPT, "--output", release, "--data", A / "typed-decisions-v2",
    "--name", "anarkali", "--abstain-below", "0.5", *int8_flags,
    *(["--temperature-by-type", json.dumps(typed["temperature_by_type"])] if use_temperatures else []))
backup = RUN / "anarkali-v4-backup.zip"
with zipfile.ZipFile(backup, "w", zipfile.ZIP_STORED) as z:
    for path in release.iterdir():
        z.write(path, f"release/{path.name}")
    z.write(RUN / "bakeoff.json", "bakeoff.json")
    for name in reports:
        z.write(RUN / f"eval-{name}" / "test-report.json", f"eval-{name}/test-report.json")
    z.writestr("winner.json", json.dumps({"winner": WINNER, **ok[WINNER]}, indent=2))
    for path in LOGS.glob("*.log"):
        z.write(path, f"logs/{path.name}")
print("\\nBACKUP:", backup, "sha256", hashlib.sha256(backup.read_bytes()).hexdigest())
# Kaggle keeps only /kaggle/working after a run; Colab users download from the file browser.
import shutil
if Path("/kaggle/working").exists():
    shutil.copy(backup, "/kaggle/working/" + backup.name)
    shutil.copy(WINNER_CKPT, "/kaggle/working/" + WINNER + "-best.pt")
    print("Copied the backup and the winner checkpoint to /kaggle/working", flush=True)
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
