#!/usr/bin/env python3
"""Colab-ready Anarkali training runner.

Use this in a Colab runtime with a GPU:

!git clone https://github.com/ToufiqQureshi/anarkali.git
%cd anarkali
!python scripts/colab_train.py --epochs 4 --batch-size 16 --device cuda --max-train-rows 2000 --max-dev-rows 300

It will install the repo extras, prepare typed decision data, and run the strongest
repo-native training pass available for this project.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(command: str, *, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {command}\n")
    subprocess.run(command, shell=True, check=True, cwd=str(REPO), env=env)


def install_deps() -> None:
    run("python -m pip install --upgrade pip")
    run("python -m pip install -e '.[train,onnx,serve]'")


def prepare_data() -> None:
    run(
        "python scripts/prepare_typed_decisions.py "
        "--question-types choice,noul,score "
        "--output artifacts/focused-retrain-v1"
    )


def train(
    *,
    epochs: int,
    batch_size: int,
    device: str,
    max_train_rows: int,
    max_dev_rows: int,
) -> None:
    run(
        "python scripts/train_anarkali.py "
        "--data artifacts/focused-retrain-v1 "
        "--output artifacts/colab-heavy-run "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        "--evaluate-test "
        f"--device {device} "
        f"--max-train-rows {max_train_rows} "
        f"--max-dev-rows {max_dev_rows}"
    )


def evaluate() -> None:
    run(
        "python scripts/evaluate_release_suite.py "
        "--model release-150m "
        "--cases benchmarks/anarkali-routing-v1/cases.jsonl "
        "--min-accuracy 0.75"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-rows", type=int, default=2000)
    parser.add_argument("--max-dev-rows", type=int, default=300)
    args = parser.parse_args()

    if not args.skip_install:
        install_deps()
    if not args.skip_prepare:
        prepare_data()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        max_train_rows=args.max_train_rows,
        max_dev_rows=args.max_dev_rows,
    )
    if not args.skip_evaluate:
        evaluate()


if __name__ == "__main__":
    main()
