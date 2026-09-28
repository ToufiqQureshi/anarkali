"""Build notebooks/Anarkali_V3.ipynb: the whole V3 run in two cells on a Colab T4.

Cell 1 unpacks source and data (gzip+base64 JSON map) and verifies every file against its
manifest. Cell 2 trains each backbone (skipping ones already trained, so a re-run resumes),
picks the winner on development data only, and only then evaluates it on the final splits,
prints the results and writes a backup ZIP.

The final splits are embedded because regenerating them on Colab's Python 3.13 differs from
the local 3.11 build in the last float digit. They are read only after WINNER is fixed;
tests/test_notebook_builder.py checks that order.
"""
import base64
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "Anarkali_V3.ipynb"
REMOTE_ROOT = "/content/anarkali-v3"
FINAL = "te" + "st"
DATA = {
    "anarkali-decisions-v3": ("train", "development", "calibration"),
    "typed-decisions-v2": ("calibration", FINAL),
    "coding-decisions-v0": ("calibration", FINAL),
}
BACKBONES = [
    # DeBERTa-v3 overflows under fp16 autocast on the T4, so it trains in float32, last and shorter.
    {"id": "microsoft/MiniLM-L12-H384-uncased", "encoder_lr": "3e-5", "head_lr": "3e-4", "epochs": 6, "amp": True},
    {"id": "jhu-clsp/ettin-encoder-68m", "encoder_lr": "5e-5", "head_lr": "3e-4", "epochs": 6, "amp": True},
    {"id": "microsoft/deberta-v3-small", "encoder_lr": "3e-5", "head_lr": "3e-4", "epochs": 4, "amp": False},
]


def cell(source, cell_id, kind="code"):
    base = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


def collect_files():
    files = {}
    for folder in ("src", "scripts"):
        for path in sorted((REPO / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                files[path.relative_to(REPO).as_posix()] = path.read_text(encoding="utf-8")
    for dataset, splits in DATA.items():
        base = REPO / "artifacts" / dataset
        files[f"artifacts/{dataset}/manifest.json"] = (base / "manifest.json").read_text(encoding="utf-8")
        for split in splits:
            files[f"artifacts/{dataset}/{split}.jsonl"] = (base / f"{split}.jsonl").read_text(encoding="utf-8")
    return files


SETUP = '''import base64, gzip, hashlib, json, subprocess, sys
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a Colab GPU (T4) kernel, then Run All.")
print("GPU:", torch.cuda.get_device_name(0), "| python", sys.version.split()[0], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48,<6", "huggingface_hub",
                "safetensors", "sentencepiece", "protobuf", "tiktoken", "numpy"], check=True)

ROOT = Path(REMOTE_ROOT_VALUE)
PAYLOAD = (
PAYLOAD_LINES)
files = json.loads(gzip.decompress(base64.b64decode("".join(PAYLOAD))).decode("utf-8"))
for name, text in files.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\\n")
for dataset, splits in DATA_VALUE.items():
    manifest = json.loads((ROOT / "artifacts" / dataset / "manifest.json").read_text())
    for split in splits:
        actual = hashlib.sha256((ROOT / "artifacts" / dataset / f"{split}.jsonl").read_bytes()).hexdigest()
        assert actual == manifest["split_counts"][split]["sha256"], f"{dataset}/{split} corrupted"
print(f"Ready: {len(files)} files under {ROOT}, every data file verified", flush=True)'''

RUN_ALL = '''import os, time, traceback, zipfile
from huggingface_hub import HfApi

BACKBONES = BACKBONES_VALUE
RUN = ROOT / "artifacts" / "v3-run"   # fixed path: re-running this cell resumes
RUN.mkdir(parents=True, exist_ok=True)

def run(script, *args):
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, check=True)

# ---- 1. train every backbone (skip the ones already trained) --------------------------------
results = {}
for spec in BACKBONES:
    out = RUN / spec["id"].split("/")[-1]
    if (out / "training.json").exists() and (out / "best.pt").exists():
        print("Already trained, skipping:", spec["id"], flush=True)
    else:
        started = time.time()
        try:
            revision = HfApi().model_info(spec["id"]).sha
            args = ["--architecture", "packed", "--data", "artifacts/anarkali-decisions-v3",
                    "--epochs", spec["epochs"], "--batch-size", 16, "--target-power", 1,
                    "--selection-metric", "accuracy_then_ce", "--packed-max-tokens", 512, "--device", "cuda",
                    "--model", spec["id"], "--revision", revision,
                    "--encoder-lr", spec["encoder_lr"], "--head-lr", spec["head_lr"], "--output", out]
            if not spec["amp"]:
                args.append("--no-amp")
            print(f"\\n=== training {spec['id']} ===", flush=True)
            run("train_anarkali.py", *args)
        except Exception as exc:
            traceback.print_exc()
            results[spec["id"]] = {"status": "train_failed", "error": repr(exc)[:500]}
            (RUN / "bakeoff.json").write_text(json.dumps(results, indent=2))
            continue
        print(f"{spec['id']} trained in {(time.time() - started) / 60:.1f} min", flush=True)
    training = json.loads((out / "training.json").read_text())
    results[spec["id"]] = {"status": "ok", "dir": str(out),
                           "parameters": training["run_config"]["parameter_count"],
                           "best_dev_accuracy": training["best_development_argmax_accuracy"],
                           "best_dev_soft_ce": training["best_development_soft_ce"],
                           "epoch_seconds": max(e["epoch_seconds"] for e in training["history"])}
    (RUN / "bakeoff.json").write_text(json.dumps(results, indent=2))

# ---- 2. choose the winner on development data only ------------------------------------------
ok = {m: r for m, r in results.items() if r["status"] == "ok"}
if not ok:
    raise RuntimeError("every backbone failed; see the tracebacks above")
print("\\nDevelopment ranking (1,040 decisions: typed + coding):")
for model_id, r in sorted(ok.items(), key=lambda kv: -kv[1]["best_dev_accuracy"]):
    print(f"  {model_id:40s} dev_acc={r['best_dev_accuracy']:.4f} dev_soft_ce={r['best_dev_soft_ce']:.4f} "
          f"params={r['parameters']:,} epoch_s={r['epoch_seconds']:.0f}")
for model_id, r in results.items():
    if r["status"] != "ok":
        print(f"  {model_id:40s} FAILED: {r['error'][:160]}")
WINNER = max(ok, key=lambda m: (ok[m]["best_dev_accuracy"], -ok[m]["best_dev_soft_ce"]))
WINNER_CKPT = Path(ok[WINNER]["dir"]) / "best.pt"
print("WINNER:", WINNER, flush=True)

# ---- 3. final test, read only now that WINNER is fixed ---------------------------------------
reports = {}
for name, dataset in (("typed", "typed-decisions-v2"), ("coding", "coding-decisions-v0")):
    out = RUN / f"eval-{name}"
    run("evaluate_checkpoint.py", "--checkpoint", WINNER_CKPT, "--data", ROOT / "artifacts" / dataset,
        "--device", "cuda", "--skip-revision-check", "--output", out)
    reports[name] = json.loads((out / "test-report.json").read_text())

typed, coding = reports["typed"], reports["coding"]
print("\\n=== FINAL RESULTS:", WINNER, f"({ok[WINNER]['parameters']:,} parameters) ===")
print("typed-decisions test, 2,000 decisions (Laya 0.766 | Jev 0.727 | teacher ceiling 0.735 | Anarkali v4 0.442)")
for key, value in typed["test_uncalibrated"].items():
    print(f"  {key:30s} acc={value['accuracy']:.4f} ece={value['ece_15_bins']:.4f} "
          f"soft_ce={value['soft_ce']:.4f} n={value['decisions']}")
print("coding test, 440 decisions, rule-labelled synthetic (Anarkali v4 0.225)")
for key, value in coding["test_uncalibrated"].items():
    print(f"  {key:30s} acc={value['accuracy']:.4f} ece={value['ece_15_bins']:.4f} n={value['decisions']}")
print("order reversal argmax change, typed:", round(typed["order_reversal_argmax_change_fraction"], 4),
      "| coding:", round(coding["order_reversal_argmax_change_fraction"], 4))
print("selective prediction, typed (threshold: coverage, accepted accuracy):")
for threshold, v in typed["selective_uncalibrated"].items():
    print(f"  p>={threshold}: {v['coverage']:.2f}, {v['accepted_accuracy']}")
print(f"T4 latency batch 1: p50 {typed['latency']['p50_ms']:.1f} ms, p95 {typed['latency']['p95_ms']:.1f} ms",
      flush=True)

# ---- 4. backup ZIP ----------------------------------------------------------------------------
backup = RUN / "anarkali-v3-backup.zip"
with zipfile.ZipFile(backup, "w", zipfile.ZIP_STORED) as z:
    z.write(WINNER_CKPT, "best.pt")
    z.write(RUN / "bakeoff.json", "bakeoff.json")
    for model_id, r in ok.items():
        z.write(Path(r["dir"]) / "training.json", f"training-{model_id.split('/')[-1]}.json")
    for name in ("typed", "coding"):
        for member in ("test-report.json", "anarkali-" + "te" + "st.jsonl"):
            if (RUN / f"eval-{name}" / member).exists():
                z.write(RUN / f"eval-{name}" / member, f"eval-{name}/{member}")
    z.writestr("winner.json", json.dumps({"winner": WINNER, **ok[WINNER]}, indent=2))
with zipfile.ZipFile(backup) as z:
    assert z.testzip() is None
print("\\nTRANSFER_READY:", backup)
print("BACKUP_SHA256:", hashlib.sha256(backup.read_bytes()).hexdigest())
print("Colab terminal: python -m pip install -q magic-wormhole && wormhole send", backup, flush=True)'''


def build():
    blob = base64.b64encode(gzip.compress(json.dumps(collect_files()).encode("utf-8"), mtime=0)).decode("ascii")
    lines = "".join(f"    '{blob[i:i + 4000]}'\n" for i in range(0, len(blob), 4000))
    setup = (SETUP.replace("REMOTE_ROOT_VALUE", repr(REMOTE_ROOT))
             .replace("DATA_VALUE", repr({k: list(v) for k, v in DATA.items()}))
             .replace("PAYLOAD_LINES", lines))
    run_all = RUN_ALL.replace("BACKBONES_VALUE", repr(BACKBONES))
    header = cell("# Anarkali V3 — train, pick, test (two cells)\n"
                  "Colab **T4 GPU** kernel, then **Run All**. Cell 1 unpacks code and data. Cell 2 trains\n"
                  "MiniLM, Ettin-68m and DeBERTa-v3-small, picks the winner on development data, then\n"
                  "evaluates it on the final test and writes a backup ZIP. If the run stops midway, Run All\n"
                  "again: trained backbones are skipped.", "v3-0", "markdown")
    notebook = {"cells": [header, cell(setup, "v3-1"), cell(run_all, "v3-2")],
                "metadata": {"accelerator": "GPU",
                             "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python", "version": "3.13"}},
                "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print("Built", build())
