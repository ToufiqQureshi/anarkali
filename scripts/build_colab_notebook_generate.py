"""Build notebooks/Anarkali_Generate.ipynb: generate decision data for free on Kaggle's GPUs with Qwen.

No API key and no cost. Cell 1 clones the repo and installs vLLM. Cell 2 starts an OpenAI-compatible
vLLM server on the GPUs with an open model (Qwen3, Apache-2.0) and waits until it answers. Cell 3
runs generate_domain_decisions.py against it for every chosen domain; every reply is cached, so a
cell or session that stops can be re-run and continues where it left off. Cell 4 checks the result
(rows, share with generator labels, answer balance, one example) and zips each domain for download.
The zips go into notebooks/Anarkali_Distill.ipynb as SYNTHETIC_SETS. tests/test_notebook_generate.py
checks it.
"""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "Anarkali_Generate.ipynb"


def v4_builder():
    spec = importlib.util.spec_from_file_location("build_v4", REPO / "scripts" / "build_colab_notebook_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V4 = v4_builder()
RUN_HELPER = V4.SETUP[V4.SETUP.index("import collections"):V4.SETUP.index('A = ROOT / "artifacts"')]
DOMAINS = ["customer_service", "invoice_processing", "security_incidents", "agent_trace_observability",
           "content_moderation", "ecommerce_return", "payment_fraud", "it_helpdesk", "email_intent",
           "loan_application"]

SETUP = '''# ---- settings ---------------------------------------------------------------------------------
REPO_URL = "https://github.com/ToufiqQureshi/anarkali"
REF = "claude/distill-brio-data"   # "main" once merged
# Qwen3-8B fits two T4s in fp16 (Kaggle "GPU T4 x2"). Faster, a bit weaker: "Qwen/Qwen3-4B-Instruct-2507".
MODEL = "Qwen/Qwen3-8B"
DOMAINS = DOMAINS_VALUE
ROWS_PER_DOMAIN = 20000        # about 4k cases for the benchmark domains (5 questions), 6.7k for the rest (3)
BATCH = 8                      # cases per request
WORKERS = 16                   # parallel requests; vLLM batches them on the GPU
MAX_TOKENS = 7000
# To continue a run from an earlier session, attach its output as a Kaggle dataset and point here:
# Breadth: the 220 domains of scripts/domains/catalog_wide.json, fewer rows each, into synth/synth-wide.
# Many tasks with fewer rows generalise better than a few big ones. 600 rows = 200 cases per domain,
# about 5.5k requests in all: several Kaggle sessions (RESUME_FROM continues where one stopped).
USE_WIDE_CATALOG = False
WIDE_ROWS_PER_DOMAIN = 600
RESUME_FROM = ""               # e.g. "/kaggle/input/anarkali-synth-part1/synth"
SILENCE_LIMIT_MIN = 45

import collections, json, os, queue, shutil, subprocess, sys, threading, time
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime (Kaggle: GPU T4 x2), then Run All.")
GPUS = torch.cuda.device_count()
print("GPU:", GPUS, "x", torch.cuda.get_device_name(0), flush=True)
# Kaggle images can also contain /content. Generated shards must live in /kaggle/working so they
# remain downloadable if a later domain fails.
ROOT = (Path("/kaggle/working/anarkali") if Path("/kaggle/working").is_dir()
        else Path("/content/anarkali") if Path("/content").is_dir()
        else Path.cwd() / "anarkali")
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
LOGS = ROOT / "artifacts" / "logs"
KEEP = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else ROOT / "artifacts"
SYNTH = KEEP / "synth"          # /kaggle/working survives the session as its output
SYNTH.mkdir(parents=True, exist_ok=True)
if RESUME_FROM:
    for folder in Path(RESUME_FROM).iterdir():
        if folder.is_dir() and not (SYNTH / folder.name).exists():
            shutil.copytree(folder, SYNTH / folder.name)
            print("resuming from", folder, flush=True)

RUN_HELPER_VALUE'''

SERVER = '''# Start an OpenAI-compatible server on the GPUs. T4s have no bfloat16, hence fp16.
import urllib.request
LOGS.mkdir(parents=True, exist_ok=True)
server_log = open(LOGS / "vllm.log", "w")
server = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--dtype", "half",
     "--tensor-parallel-size", str(GPUS), "--max-model-len", "8192", "--gpu-memory-utilization", "0.92",
     "--port", "8000"], stdout=server_log, stderr=subprocess.STDOUT)
started = time.time()
while True:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
        break
    except Exception:
        if server.poll() is not None or time.time() - started > 30 * 60:
            print(open(LOGS / "vllm.log").read()[-4000:])
            raise RuntimeError("the vLLM server did not start; its log is above")
        time.sleep(10)
print(f"vLLM is serving {MODEL} after {(time.time() - started) / 60:.1f} min", flush=True)'''

GENERATE = '''# One domain at a time. Re-running this cell continues from the cache: nothing is asked twice.
# Qwen3 is a thinking model; enable_thinking false makes it answer with the JSON directly.
os.environ.setdefault("QWEN_API_KEY", "local")
extra = json.dumps({"chat_template_kwargs": {"enable_thinking": False}}) if "Qwen3-" in MODEL and "Instruct" not in MODEL else "{}"
for domain in DOMAINS:
    started = time.time()
    run("generate_domain_decisions.py", "--teacher", f"qwen={MODEL}@http://127.0.0.1:8000/v1",
        "--catalog", ROOT / "scripts" / "domains" / "benchmark.json",
        "--catalog", ROOT / "scripts" / "domains" / "catalog.json",
        "--domains", domain, "--rows-per-domain", ROWS_PER_DOMAIN, "--batch", BATCH, "--workers", WORKERS,
        "--max-tokens", MAX_TOKENS, "--extra-body", extra, "--output", SYNTH / f"synth-{domain}", tag=domain)
    print(f"{domain}: {(time.time() - started) / 60:.1f} min", flush=True)
if USE_WIDE_CATALOG:
    started = time.time()
    run("generate_domain_decisions.py", "--teacher", f"qwen={MODEL}@http://127.0.0.1:8000/v1",
        "--catalog", ROOT / "scripts" / "domains" / "catalog_wide.json",
        "--rows-per-domain", WIDE_ROWS_PER_DOMAIN, "--batch", BATCH, "--workers", WORKERS,
        "--max-tokens", MAX_TOKENS, "--extra-body", extra, "--output", SYNTH / "synth-wide", tag="wide")
    print(f"wide catalog: {(time.time() - started) / 60:.1f} min", flush=True)'''

REPORT = '''# Check before using: row counts, how many rows carry the generator's labels, answer balance.
import zipfile
for domain in DOMAINS + (["wide"] if USE_WIDE_CATALOG else []):
    folder = SYNTH / f"synth-{domain}"
    if not (folder / "manifest.json").exists():
        print(f"{domain}: not generated yet")
        continue
    rows = [json.loads(line) for name in ("train", "development", "calibration", "test")
            for line in (folder / f"{name}.jsonl").read_text().splitlines()]
    labelled = [r for r in rows if "teacher_targets" in r]
    first_q = rows[0]["case_id"].split("::")[1]
    winners = collections.Counter(
        r["candidates"][max(range(len(r["candidates"])), key=r["teacher_targets"]["generator"].__getitem__)]["id"]
        for r in labelled if r["case_id"].endswith("::" + first_q))
    print(f"{domain}: {len(rows)} rows, {len(labelled) / max(1, len(rows)):.0%} with generator labels; "
          f"'{first_q}' answers: {dict(winners)}")
    archive = KEEP / f"synth-{domain}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob("*"):
            z.write(path, path.relative_to(SYNTH))
example = None
for domain in DOMAINS:
    train = SYNTH / f"synth-{domain}" / "train.jsonl"
    if train.exists() and train.stat().st_size:
        example = json.loads(train.read_text().splitlines()[0])
        break
if example:
    print("\\nOne generated case:\\n", json.dumps(example["state"], indent=1, ensure_ascii=False)[:1500])
print("\\nDownload the synth-*.zip files (or the synth/ folder) from", KEEP,
      "and attach them to the distillation notebook as SYNTHETIC_SETS.")
server.terminate()'''


def cell(source, cell_id, kind="code"):
    return V4.cell(source, cell_id, kind)


def build(path: Path = OUTPUT) -> Path:
    header = cell("# Anarkali data generation on Kaggle (free, no API key)\n"
                  "1. Kaggle: **Settings → Accelerator → GPU T4 x2**, and **Internet on**.\n"
                  "2. Set `DOMAINS` and `ROWS_PER_DOMAIN` in cell 1, then **Run All**.\n"
                  "3. A session ends after about 12 hours. Save the version (its output keeps `synth/`), attach "
                  "that output as a dataset in the next session and set `RESUME_FROM`: nothing is generated twice.\n\n"
                  "The model is Qwen3 (Apache-2.0), served on the GPUs with vLLM. Each domain folder is ready for "
                  "`SYNTHETIC_SETS` in `Anarkali_Distill.ipynb`, where the 400M teacher checks every row and rows "
                  "it disagrees with are dropped. Details: docs/DATA_AND_DISTILLATION.md.\n", "gen-0", "markdown")
    setup = SETUP.replace("DOMAINS_VALUE", repr(DOMAINS)).replace("RUN_HELPER_VALUE", RUN_HELPER.rstrip() + "\n")
    notebook = {"cells": [header, cell(setup, "gen-1"), cell(SERVER, "gen-2"), cell(GENERATE, "gen-3"),
                          cell(REPORT, "gen-4")],
                "metadata": {"accelerator": "GPU",
                             "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print("Built", build())
