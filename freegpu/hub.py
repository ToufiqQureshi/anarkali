"""Shared job storage for GPU workers on different free providers.

A job lives under jobs/<name>/ in a private Hugging Face dataset repo (HfHub) or in any
directory every worker can reach, such as a mounted drive (LocalHub):

    job.json                  teachers and relabel settings
    input/                    the prepared decision set to relabel
    cache/<worker>.jsonl      teacher scores, one shard per worker so writes never collide
    done/<teacher>.json       a teacher finished scoring every row
    heartbeat/<worker>.json   last sign of life from a worker
    launches/<provider>.json  when the orchestrator last started or asked for that provider
    output/                   the relabelled decision set, once every teacher is done
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile


class LocalHub:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def pull(self, prefix: str, local_root: Path) -> Path:
        source, target = self.root / prefix, Path(local_root) / prefix
        if source.exists():
            shutil.copytree(source, target, dirs_exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def push_file(self, local_path: Path, remote_path: str):
        target = self.root / remote_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)).replace("\\", "/") for p in base.rglob("*") if p.is_file())

    def read_json(self, remote_path: str):
        path = self.root / remote_path
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_json(self, remote_path: str, payload: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self.push_file(path, remote_path)

    def push_folder(self, local_dir: Path, remote_prefix: str):
        for path in sorted(Path(local_dir).rglob("*")):
            if path.is_file():
                self.push_file(path, f"{remote_prefix}/{path.relative_to(local_dir).as_posix()}")


class HfHub(LocalHub):
    """A private dataset repo on the Hugging Face Hub; needs HF_TOKEN with write access."""

    def __init__(self, repo_id: str, token: str | None = None):
        from huggingface_hub import HfApi
        self.repo_id, self.api = repo_id, HfApi(token=token)
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def pull(self, prefix: str, local_root: Path) -> Path:
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo_id, repo_type="dataset", allow_patterns=[f"{prefix}/*"],
                          local_dir=str(local_root), token=self.api.token)
        target = Path(local_root) / prefix
        target.mkdir(parents=True, exist_ok=True)
        return target

    def push_file(self, local_path: Path, remote_path: str):
        self.api.upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_path,
                             repo_id=self.repo_id, repo_type="dataset",
                             commit_message=f"update {remote_path}")

    def push_folder(self, local_dir: Path, remote_prefix: str):
        self.api.upload_folder(folder_path=str(local_dir), path_in_repo=remote_prefix,
                               repo_id=self.repo_id, repo_type="dataset",
                               commit_message=f"update {remote_prefix}")

    def list(self, prefix: str) -> list[str]:
        return sorted(f for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
                      if f.startswith(prefix.rstrip("/") + "/"))

    def read_json(self, remote_path: str):
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.errors import EntryNotFoundError
        except ImportError:  # huggingface_hub < 0.23
            from huggingface_hub.utils import EntryNotFoundError
        try:
            path = hf_hub_download(self.repo_id, remote_path, repo_type="dataset", token=self.api.token,
                                   force_download=True)
        except EntryNotFoundError:
            return None
        return json.loads(Path(path).read_text(encoding="utf-8"))


def find_secret(name: str) -> str | None:
    """An environment variable, else a Colab secret, else a Kaggle secret of that name."""
    import os
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def open_hub(spec: str):
    """'hf:owner/repo' for the Hugging Face Hub, anything else is a local or mounted directory."""
    if spec.startswith("hf:"):
        token = find_secret("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN not found in the environment, Colab secrets or Kaggle secrets")
        return HfHub(spec[3:], token=token)
    return LocalHub(spec)
