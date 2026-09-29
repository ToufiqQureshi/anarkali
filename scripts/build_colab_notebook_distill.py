"""Build notebooks/Anarkali_Distill.ipynb: public data + teachers (400M, optional Brio) -> distilled 68M.

Cell 1 rebuilds the benchmark data, streams the public sets (harvest_public_decisions.py), labels
them with the 400M checkpoint from the V4 run (label_with_checkpoint.py) and, when a colibri server
is reachable, with Brio (relabel_with_teachers.py --brio-teacher), then mixes every teacher with the
human labels (combine_teachers.py). Cell 2 trains the 68M student with the 400M teacher online,
next to the same student distilled on the benchmark data alone and a no-teacher control (two
T4s run two at a time), and
selects each epoch on the benchmark's own development cases. Cell 3 opens the test splits only
after that and exports ONNX. Coding and CI data are left out for now: the goal is the typed +
general decision model. tests/test_notebook_distill.py checks it.
"""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "Anarkali_Distill.ipynb"


def v4_builder():
    spec = importlib.util.spec_from_file_location("build_v4", REPO / "scripts" / "build_colab_notebook_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V4 = v4_builder()
# The V4 run() helper: streams each script's output, keeps a log, kills a silent script.
RUN_HELPER = V4.SETUP[V4.SETUP.index("import collections"):V4.SETUP.index('A = ROOT / "artifacts"')]
STUDENT_ARGS = ["--architecture", "packed", "--batch-size", "32", "--target-power", "1",
                "--selection-metric", "accuracy_then_ce", "--packed-max-tokens", "512", "--device", "cuda",
                "--encoder-lr", "5e-5", "--head-lr", "3e-4", *V4.V4]
KD_ARGS = ["--kd-weight", "1.0", "--kd-temperature", "2.0", "--hidden-weight", "0.5"]

SETUP = '''# ---- settings ---------------------------------------------------------------------------------
REPO_URL = "https://github.com/ToufiqQureshi/anarkali"
REF = "claude/distill-brio-data"  # "main" once this is merged
# The 400M teacher: V4 copies it to /kaggle/working as bb-ettin-400m-best.pt. Add that file to this
# notebook as a Kaggle dataset (or upload it on Colab) and point here.
TEACHER_CKPT = "/kaggle/input/anarkali-400m/bb-ettin-400m-best.pt"
PUBLIC_MAX_PER_SOURCE = 50000  # texts per public source; about 300k rows, fits one Kaggle session.
                               # 0 = each source's own cap (about 1.5M texts): Colab Pro A100 or several sessions
ALLOW_SHARE_ALIKE = False    # also SNLI, MultiNLI, BoolQ, DBpedia (CC-BY-SA)
BRIO_URL = ""                # e.g. "http://127.0.0.1:8000/v1" when colibri `coli serve` runs; "" skips Brio
BRIO_MODEL = "qwen36"
BRIO_MAX_ROWS = 20000        # Brio reads a whole LLM per state: label a sample of the pool with it
# Folders from generate_domain_decisions.py (e.g. DeepSeek runs), attached as Kaggle datasets
SYNTHETIC_SETS = []          # e.g. ["/kaggle/input/anarkali-synth/synth-customer_service", ...]
SYNTHETIC_MIN_AGREEMENT = 0.99  # with two teachers (generator + 400M): keep a row only when both agree
STUDENT = "jhu-clsp/ettin-encoder-68m"
EPOCHS = 3
MAX_TRAIN_ROWS = 0           # 0 = all; set it when the teacher makes an epoch too slow for the session
TRAIN_CONTROL = True         # the same student without the teacher, to measure what distillation adds
SILENCE_LIMIT_MIN = 30
EXPORT_INT8 = True

import collections, json, os, queue, subprocess, sys, threading, time
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime, then Run All.")
GPUS = torch.cuda.device_count()
print("GPU:", GPUS, "x", torch.cuda.get_device_name(0), "| python", sys.version.split()[0], flush=True)
if not Path(TEACHER_CKPT).exists():
    raise FileNotFoundError(f"TEACHER_CKPT {TEACHER_CKPT} not found: attach the V4 400M checkpoint first")
ROOT = Path("/content/anarkali") if Path("/content").exists() else Path.cwd() / "anarkali"
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{ROOT}[train,onnx]", "datasets", "onnx"],
               check=True)
LOGS = ROOT / "artifacts" / "logs"

RUN_HELPER_VALUE
A = ROOT / "artifacts"
if not (A / "typed-decisions-v2" / "manifest.json").exists():
    run("prepare_typed_decisions.py", "--question-types", "choice,noul,score", "--output", A / "typed-decisions-v2")
BASE = A / "typed-decisions-v2"  # the benchmark: its development split selects every epoch

PUBLIC = A / "public-decisions-v0"
if not (PUBLIC / "pool" / "manifest.json").exists():
    run("harvest_public_decisions.py", "--output", PUBLIC,
        *(["--max-per-source", PUBLIC_MAX_PER_SOURCE] if PUBLIC_MAX_PER_SOURCE else []),
        *(["--allow-share-alike"] if ALLOW_SHARE_ALIKE else []))

pool = PUBLIC / "pool"
if BRIO_URL:
    # One Brio request per text, every catalog question on it at once; the rest of the pool keeps
    # only the 400M teacher. The test split is never labelled.
    run("relabel_with_teachers.py", "--input", pool, "--output", A / "public-pool-brio", "--no-original",
        "--splits", "train,development,calibration", "--limit", BRIO_MAX_ROWS,
        "--brio-teacher", f"colibri={BRIO_MODEL}@{BRIO_URL}", "--workers", 2)
    pool = A / "public-pool-brio"
labelled = {}
for name, source in (("gold", PUBLIC / "gold"), ("pool", pool)):
    out = A / f"public-{name}-t400"
    if not (out / "manifest.json").exists():
        run("label_with_checkpoint.py", "--checkpoint", TEACHER_CKPT, "--input", source, "--output", out,
            "--name", "anarkali400m", "--fill-unlabelled", "--device", "cuda", "--batch-size", 64)
    combined = A / f"public-{name}-combined"
    run("combine_teachers.py", "--input", out, "--output", combined, "--gold-weight", "0.5")
    labelled[name] = combined
# Generated sets (generate_domain_decisions.py with DeepSeek or another open model): the generator's
# own labels are one teacher, the 400M is the second, and rows they disagree on are dropped.
synthetic = []
for index, source in enumerate(SYNTHETIC_SETS):
    out = A / f"synthetic-{index}-t400"
    if not (out / "manifest.json").exists():
        run("label_with_checkpoint.py", "--checkpoint", TEACHER_CKPT, "--input", source, "--output", out,
            "--name", "anarkali400m", "--fill-unlabelled", "--device", "cuda", "--batch-size", 64)
    combined = A / f"synthetic-{index}-combined"
    run("combine_teachers.py", "--input", out, "--output", combined, "--min-agreement", SYNTHETIC_MIN_AGREEMENT)
    synthetic.append(combined)
DATA = A / "distill-decisions-v0"
run("merge_decision_sets.py", "--inputs", BASE, labelled["gold"], labelled["pool"], *synthetic, "--output", DATA)
counts = json.loads((DATA / "manifest.json").read_text())["split_counts"]
print("Training data:", {k: v["decision_cases"] for k, v in counts.items()}, flush=True)'''

TRAIN = '''from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
RUN = A / "distill-run"
RUN.mkdir(parents=True, exist_ok=True)
STUDENT_REVISION = HfApi().model_info(STUDENT).sha
common = [STUDENT_ARGS_VALUE, "--data", DATA, "--dev-data", BASE, "--model", STUDENT,
          "--revision", STUDENT_REVISION, "--epochs", EPOCHS,
          *(["--max-train-rows", MAX_TRAIN_ROWS] if MAX_TRAIN_ROWS else [])]
jobs = {"distilled": [*common, "--teacher-checkpoint", TEACHER_CKPT, KD_ARGS_VALUE]}
# The same teacher on the benchmark data alone: shows whether the public data helps or dilutes.
base_only = [a for a in common]
base_only[base_only.index("--data") + 1] = BASE
jobs["distilled-base"] = [*base_only, "--teacher-checkpoint", TEACHER_CKPT, KD_ARGS_VALUE]
if TRAIN_CONTROL:
    jobs["control"] = list(common)
free_gpus = queue.Queue()
for gpu in range(GPUS):
    free_gpus.put(gpu)

def train(item):
    name, args = item
    out = RUN / name
    if not ((out / "training.json").exists() and (out / "best.pt").exists()):
        gpu = free_gpus.get()
        try:
            run("train_anarkali.py", *args, "--output", out, gpu=gpu, tag=name)
        except Exception as exc:
            print(f"\\n!!! {name} FAILED:\\n{exc}", flush=True)
            return name, {"status": "failed", "error": str(exc)[-3000:]}
        finally:
            free_gpus.put(gpu)
    training = json.loads((out / "training.json").read_text())
    return name, {"status": "ok", "dir": str(out), "dev_accuracy": training["best_development_argmax_accuracy"],
                  "dev_soft_ce": training["best_development_soft_ce"]}

with ThreadPoolExecutor(max_workers=GPUS) as pool_:
    results = dict(pool_.map(train, jobs.items()))
(RUN / "runs.json").write_text(json.dumps(results, indent=2))
ok = {name: r for name, r in results.items() if r["status"] == "ok"}
if not ok:
    raise RuntimeError("every run failed; see the tracebacks above")
for name, r in ok.items():
    print(f"  {name:10s} dev_acc={r['dev_accuracy']:.4f} dev_soft_ce={r['dev_soft_ce']:.4f} (benchmark development)")
# The shipped model is chosen on development data only, before any test split is read.
WINNER = max(ok, key=lambda name: (ok[name]["dev_accuracy"], -ok[name]["dev_soft_ce"]))
WINNER_CKPT = Path(ok[WINNER]["dir"]) / "best.pt"
print("WINNER:", WINNER, flush=True)'''

FINAL_CELL = '''# The test splits are read only here, after WINNER is fixed on development data.
import hashlib, shutil, zipfile
reports = {}
for name, dataset in (("typed", "typed-decisions-v2"),):
    for model in ok:
        out = RUN / f"eval-{name}-{model}"
        run("evaluate_checkpoint.py", "--checkpoint", Path(ok[model]["dir"]) / "best.pt", "--data", A / dataset,
            "--device", "cuda", "--skip-revision-check", "--output", out)
        reports[(name, model)] = json.loads((out / "test-report.json").read_text())
print("\\n=== typed-decisions test (Laya 0.766 | 400M V4 0.787 | Anarkali 0.3.0 0.740 | Jev 0.727) ===")
for (name, model), report in sorted(reports.items()):
    m = report["test_uncalibrated"]["all"]
    print(f"  {name:6s} {model:10s} acc={m['accuracy']:.4f} ece={m['ece_15_bins']:.4f} brier={m['brier']:.4f} "
          f"p50={report['latency']['p50_ms']:.1f}ms")
typed = reports[("typed", WINNER)]
raw, cal = typed["development_uncalibrated"], typed["development_calibrated_by_type"]
use_temperatures = cal["ece_15_bins"] < raw["ece_15_bins"] and cal["soft_ce"] < raw["soft_ce"]
print("per-type temperatures", "shipped" if use_temperatures else "not shipped (no calibration gain)", flush=True)
release = RUN / "release"
run("export_onnx.py", "--checkpoint", WINNER_CKPT, "--output", release, "--data", A / "typed-decisions-v2",
    "--name", "anarkali", "--abstain-below", "0.5", *([] if EXPORT_INT8 else ["--no-int8"]),
    *(["--temperature-by-type", json.dumps(typed["temperature_by_type"])] if use_temperatures else []))
backup = RUN / "anarkali-distill-backup.zip"
with zipfile.ZipFile(backup, "w", zipfile.ZIP_STORED) as z:
    for path in release.iterdir():
        z.write(path, f"release/{path.name}")
    z.write(RUN / "runs.json", "runs.json")
    for (name, model) in reports:
        z.write(RUN / f"eval-{name}-{model}" / "test-report.json", f"eval-{name}-{model}/test-report.json")
    for path in LOGS.glob("*.log"):
        z.write(path, f"logs/{path.name}")
print("\\nBACKUP:", backup, "sha256", hashlib.sha256(backup.read_bytes()).hexdigest())
on_kaggle = Path("/kaggle").exists() or any(k.startswith("KAGGLE_") for k in os.environ)
if on_kaggle:
    keep = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    shutil.copy(backup, keep / backup.name)
    shutil.copy(WINNER_CKPT, keep / f"distill-{WINNER}-best.pt")
    print("Copied the backup and the winner checkpoint to", keep, flush=True)
else:
    print("Not on Kaggle: download", backup, "from the file browser", flush=True)'''


def cell(source, cell_id, kind="code"):
    return V4.cell(source, cell_id, kind)


def build(path: Path = OUTPUT) -> Path:
    header = cell("# Anarkali distillation: public data + teachers -> a 68M student\n"
                  "1. Run the V4 notebook with `ONLY_RECIPES = ['bb-ettin-400m']` and keep `bb-ettin-400m-best.pt`.\n"
                  "2. Attach it here and set `TEACHER_CKPT` in cell 1. Optional: start a colibri server "
                  "(`coli serve --model-id qwen36`, see its docs/brio.md) and set `BRIO_URL`.\n"
                  "3. Pick a GPU runtime (Kaggle T4 x2 trains the student and the control at once) and **Run All**.\n\n"
                  "Cell 2 selects on the benchmark's development cases only; cell 3 opens the test splits after "
                  "that.\n\nGenerated sets (DeepSeek or another open model, see docs/DATA_AND_DISTILLATION.md) go in "
                  "`SYNTHETIC_SETS`.\n", "distill-0", "markdown")
    setup = SETUP.replace("RUN_HELPER_VALUE", RUN_HELPER.rstrip() + "\n")
    train = TRAIN.replace("STUDENT_ARGS_VALUE", repr(STUDENT_ARGS)[1:-1]).replace("KD_ARGS_VALUE", repr(KD_ARGS)[1:-1])
    notebook = {"cells": [header, cell(setup, "distill-1"), cell(train, "distill-2"), cell(FINAL_CELL, "distill-3")],
                "metadata": {"accelerator": "GPU",
                             "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print("Built", build())
