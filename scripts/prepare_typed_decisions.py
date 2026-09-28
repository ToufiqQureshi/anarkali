"""Fetch the versioned Typed Decisions benchmark and isolate complete source cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO / ".cache" / "huggingface" / "hub"))
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
from anarkali.typed import DEFAULT_NOUL_CRITERIA, QUESTION_PREFIX  # noqa: E402
DATASET = "LocalLLaMA/typed-decisions"
PUBLISHED_COMMIT = "468b146"


def split_train_rows(rows, seed):
    by_workflow = {}
    for row in rows:
        workflow = row.get("workflow")
        case_id = row.get("id")
        if not isinstance(workflow, str) or not isinstance(case_id, str) or not case_id:
            raise ValueError("training row is missing workflow or case id")
        by_workflow.setdefault(workflow, []).append(row)
    parts = {name: [] for name in ("train", "development", "calibration")}
    for workflow in sorted(by_workflow):
        cases = by_workflow[workflow]
        random.Random(f"{seed}:{workflow}").shuffle(cases)
        n = len(cases)
        n_dev = max(1, round(n * .10))
        n_cal = max(1, round(n * .10))
        if n - n_dev - n_cal < 1:
            raise ValueError(f"not enough cases in workflow {workflow!r} for three splits")
        parts["development"].extend(cases[:n_dev])
        parts["calibration"].extend(cases[n_dev:n_dev + n_cal])
        parts["train"].extend(cases[n_dev + n_cal:])
    return parts


def choice_records(rows, *, max_cases=None):
    return decision_records(rows, types=("choice",), max_cases=max_cases)


def decision_records(rows, *, types=("choice",), max_cases=None):
    """Expand source cases into candidate-choice records; noul and score become choices over their labels.

    Choice rows are byte-identical to the v1 choice-only export; other types carry question_type.
    """
    records = []
    for row in rows:
        case_id, workflow = row["id"], row["workflow"]
        state = json.loads(row["state"])
        questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
        for qid, question in questions.items():
            kind = question.get("type")
            if kind not in types or qid not in gold:
                continue
            criteria = question.get("criteria")
            if kind == "score" and isinstance(criteria, list):
                criteria = {str(level): text for level, text in enumerate(criteria)}
            if kind == "noul" and criteria is None:
                criteria = DEFAULT_NOUL_CRITERIA
            gold_probabilities = gold[qid].get("probabilities")
            if not isinstance(criteria, dict) or len(criteria) < 2 or not isinstance(gold_probabilities, dict):
                raise ValueError(f"invalid {kind} question in {case_id}:{qid}")
            labels = list(criteria)
            if not set(labels) <= gold_probabilities.keys():
                raise ValueError(f"incomplete gold distribution in {case_id}:{qid}")
            target = [float(gold_probabilities[key]) for key in labels]
            if any(value < 0 for value in target) or sum(target) <= 0:
                raise ValueError(f"invalid gold distribution in {case_id}:{qid}")
            total = sum(target)
            target = [value / total for value in target]
            record = {
                "case_id": f"{case_id}::{qid}", "source_group": f"{workflow}::{case_id}",
                "workflow": workflow, "state": state,
                "question": QUESTION_PREFIX[kind] + str(question["instructions"]),
                "candidates": [{"id": label, "text": str(criteria[label])} for label in labels],
                "target": target,
            }
            if kind != "choice":
                record["question_type"] = kind
            records.append(record)
            if max_cases and len(records) >= max_cases:
                return records
    return records


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts" / "typed-decisions-v1")
    parser.add_argument("--cache", type=Path, default=REPO / ".cache" / "datasets")
    parser.add_argument("--revision", default=PUBLISHED_COMMIT,
                        help="verified dataset commit prefix; default is the published 1,200/400 split")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--question-types", default="choice",
                        help="comma-separated subset of choice,noul,score; the split is identical for any subset")
    args = parser.parse_args()
    types = tuple(t.strip() for t in args.question_types.split(",") if t.strip())
    if not types or not set(types) <= set(QUESTION_PREFIX):
        raise SystemExit("--question-types must name choice, noul and/or score")

    try:
        from huggingface_hub import HfApi
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install Anarkali's optional [data] extra first") from exc
    info = HfApi().dataset_info(DATASET, revision=args.revision)
    revision = info.sha
    loaded = load_dataset(DATASET, "all", revision=revision, cache_dir=str(args.cache))
    if not {"train", "test"} <= set(loaded):
        raise ValueError("pinned benchmark must contain train and test splits")
    partitions = split_train_rows(list(loaded["train"]), args.seed)
    rows_by_split = {**partitions, "test": list(loaded["test"])}
    cases_by_split = {name: decision_records(rows, types=types) for name, rows in rows_by_split.items()}
    groups = {name: {row["source_group"] for row in records} for name, records in cases_by_split.items()}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if overlap := groups[left] & groups[right]:
                raise ValueError(f"source leakage between {left} and {right}: {next(iter(overlap))}")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, records in cases_by_split.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[name] = {"decision_cases": len(records), "source_groups": len(groups[name]),
                        "sha256": digest_file(path)}
    manifest = {
        "dataset": DATASET, "revision": revision, "config": "all",
        "published_revision_prefix": args.revision, "seed": args.seed,
        **({"question_types": list(types)} if types != ("choice",) else {}),
        "label_source": "three sampled teacher distributions; agreement is not ground-truth correctness",
        "split_unit": "whole benchmark source cases, stratified by workflow before question expansion",
        "split_counts": counts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
