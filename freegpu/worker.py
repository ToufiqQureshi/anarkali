"""Advance a relabelling job on whatever GPU this machine has, then hand it back to the hub.

Run the same command on Kaggle, Colab or any other box; workers pick up where the last one
stopped because teacher scores are cached in the hub (see hub.py):

    python freegpu/worker.py --hub hf:you/anarkali-work --job typed-v2 --max-hours 8

The provider (colab, kaggle or other) is detected, and HF_TOKEN is read from the
environment, Colab secrets or Kaggle secrets, so the same command runs everywhere.

For each teacher in job.json not yet marked done, the worker starts `vllm serve` (or uses
the teacher's own base_url, e.g. Groq), fills the cache with relabel_with_teachers.py
--only-score, and pushes its cache shard and a heartbeat every --sync-minutes. A killed
session loses at most that much work. When every teacher is done it builds the relabelled
set offline, from the cache alone, and uploads it to jobs/<job>/output/.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hub import open_hub  # noqa: E402

RELABEL = HERE.parent / "scripts" / "relabel_with_teachers.py"
OFFLINE_URL = "http://offline.invalid/v1"


def detect_provider() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "other"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def serve_teacher(teacher: dict, log_dir: Path, ready_timeout: float):
    """Yield a base URL for the teacher: its own base_url, or a local vLLM server we start."""
    if teacher.get("base_url"):
        yield teacher["base_url"]
        return
    port = free_port()
    log_path = log_dir / f"vllm-{teacher['name']}.log"
    command = ["vllm", "serve", teacher["model"], "--host", "127.0.0.1", "--port", str(port),
               *teacher.get("vllm_args", [])]
    print("starting:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{port}/v1"
            deadline = time.monotonic() + ready_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {process.returncode}; see {log_path}:\n"
                                       + log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
                try:
                    with urllib.request.urlopen(f"{url}/models", timeout=5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"vLLM not ready after {ready_timeout:.0f}s; see {log_path}")
                    time.sleep(10)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()


class Syncer:
    """Push this worker's cache shard and a heartbeat now and then, from a background thread."""

    def __init__(self, hub, job: str, worker: str, provider: str, shard: Path, minutes: float):
        self.hub, self.job, self.worker, self.provider, self.shard = hub, job, worker, provider, shard
        self.interval, self.state = minutes * 60, "starting"
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.sync()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join()
        if exc is not None:
            self.state = f"failed: {exc}"[:500]
        self.sync(final=True)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sync()
            except Exception as exc:  # a failed upload must not kill the scoring run
                print(f"sync failed, will retry: {exc}", flush=True)

    def sync(self, final: bool = False):
        with self.lock:
            if self.shard.exists() and self.shard.stat().st_size:
                self.hub.push_file(self.shard, f"jobs/{self.job}/cache/{self.shard.name}")
            self.hub.write_json(f"jobs/{self.job}/heartbeat/{self.worker}.json", {
                "worker": self.worker, "provider": self.provider, "state": self.state,
                "time": time.time(), "final": final})


def run_relabel(args: list[str], deadline: float | None) -> bool:
    """Run relabel_with_teachers.py; False if the time budget ran out first."""
    process = subprocess.Popen([sys.executable, str(RELABEL), *args])
    while True:
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if deadline is not None and time.time() > deadline:
                process.terminate()
                process.wait()
                return False
            continue
        if code != 0:
            raise RuntimeError(f"relabel_with_teachers.py exited with {code}")
        return True


def teacher_args(teacher: dict, base_url: str) -> list[str]:
    """--teacher (and its own request fields) exactly as scored, so offline cache keys match."""
    args = ["--teacher", f"{teacher['name']}={teacher['model']}@{base_url}"]
    if teacher.get("extra_body"):
        args += ["--teacher-extra", f"{teacher['name']}={json.dumps(teacher['extra_body'])}"]
    return args


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub", required=True, help="hf:owner/repo or a shared directory")
    parser.add_argument("--job", required=True)
    parser.add_argument("--provider", default=os.environ.get("ANARKALI_PROVIDER") or detect_provider(),
                        help="label for heartbeats and worker ids (default: detected)")
    parser.add_argument("--worker", default=None, help="unique worker id (default provider-timestamp)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly before the provider's session limit")
    parser.add_argument("--sync-minutes", type=float, default=10.0)
    parser.add_argument("--ready-timeout", type=float, default=1800.0, help="seconds to wait for vLLM")
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    deadline = started + args.max_hours * 3600 if args.max_hours else None
    worker = args.worker or f"{args.provider}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(started))}"
    hub = open_hub(args.hub)
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="anarkali-worker-"))
    job_dir = hub.pull(f"jobs/{args.job}", workdir)
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if (job_dir / "output" / "manifest.json").exists():
        print("job already finished", flush=True)
        return "finished"

    cache_dir = job_dir / "cache"
    shard = cache_dir / f"{worker}.jsonl"
    common = ["--input", str(job_dir / "input"), "--cache-dir", str(cache_dir), "--cache-shard", worker,
              "--splits", job.get("splits", "train"), *job.get("relabel_args", [])]
    done_dir = job_dir / "done"
    with Syncer(hub, args.job, worker, args.provider, shard, args.sync_minutes) as syncer:
        for teacher in job["teachers"]:
            if (done_dir / f"{teacher['name']}.json").exists():
                continue
            syncer.state = f"scoring {teacher['name']}"
            with serve_teacher(teacher, workdir, args.ready_timeout) as base_url:
                finished = run_relabel([*common, "--only-score", "--output", str(workdir / "scratch"),
                                        *teacher_args(teacher, base_url)], deadline)
            if not finished:
                syncer.state = "out of time"
                print("time budget used up; the next worker continues from the cache", flush=True)
                return "paused"
            syncer.sync()
            hub.write_json(f"jobs/{args.job}/done/{teacher['name']}.json",
                           {"teacher": teacher["name"], "worker": worker, "time": time.time()})
            (done_dir / f"{teacher['name']}.json").parent.mkdir(parents=True, exist_ok=True)
            (done_dir / f"{teacher['name']}.json").write_text("{}", encoding="utf-8")

        syncer.state = "building output"
        output = workdir / "output"
        run_relabel([*common, "--offline", "--output", str(output),
                     *[arg for t in job["teachers"] for arg in teacher_args(t, OFFLINE_URL)]], None)
        hub.push_folder(output, f"jobs/{args.job}/output")
        syncer.state = "finished"
    print(f"job finished; output in jobs/{args.job}/output", flush=True)
    return "finished"


if __name__ == "__main__":
    main()
