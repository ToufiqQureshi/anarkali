"""Generate seeded coding-workflow decisions and optionally label them with a teacher model.

Rows use the typed-decisions-v2 format exactly: case_id, source_group, workflow, state,
question, candidates, target, and question_type for noul/score. Splits are disjoint by
source_group and stratified by workflow (train 70 / development 10 / calibration 10 / test 10).

Teacher mode reads next-token probabilities of the option letters, renormalized over the
valid letters, averaged over 3 cyclic option orders to cancel position bias. Per row it
also records teacher_disagreement: max minus min probability of the winning option
across the three orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anarkali.typed import question_candidates  # noqa: E402
from anarkali.workflows.coding import QUESTIONS, sample_case  # noqa: E402

DEFAULT_TEACHER = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT_VERSION = "coding-teacher-v0"
SPLIT_NAMES = ("train", "development", "calibration", "test")
SPLIT_WEIGHTS = {"train": 70, "development": 10, "calibration": 10, "test": 10}
LETTERS = "ABCDEFGH"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(values: list[float]) -> list[float]:
    """Full precision, summing to one within float error; rounding broke the 1e-6 training check."""
    total = sum(values)
    return [v / total for v in values]


def build_rows(cases_per_workflow: int, seed: int) -> list[dict]:
    """Expand latent cases into typed-decision rows; one source_group per generated case."""
    rows: list[dict] = []
    for workflow, questions in sorted(QUESTIONS.items()):
        for index in range(cases_per_workflow):
            case_id, state, workflow_questions, gold = sample_case(workflow, f"{seed}:{workflow}:{index}")
            source_group = f"{workflow}::case{index:05d}"
            for qid, question in workflow_questions.items():
                kind, text, candidates = question_candidates(question)
                distribution = gold[qid]
                row = {
                    # exactly one "::" -- schema_key() reads the question id after it
                    "case_id": f"{workflow}_{index:05d}::{qid}",
                    "source_group": source_group,
                    "workflow": workflow,
                    "state": state,
                    "question": text,
                    "candidates": candidates,
                    "target": normalized([distribution[c["id"]] for c in candidates]),
                }
                if kind != "choice":
                    row["question_type"] = kind
                row["_kind"] = kind
                rows.append(row)
    return rows


def split_by_source_group(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Stratify whole source groups by workflow, then split 70/10/10/10 with train taking the remainder."""
    by_workflow: dict[str, list[str]] = {}
    for row in rows:
        by_workflow.setdefault(row["workflow"], []).append(row["source_group"])
    groups_by_split: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for workflow, groups in sorted(by_workflow.items()):
        ordered = sorted(set(groups))
        import random
        random.Random(f"{seed}:split:{workflow}").shuffle(ordered)
        total, cursor = len(ordered), 0
        for name in SPLIT_NAMES[:-1]:
            size = round(total * SPLIT_WEIGHTS[name] / 100)
            groups_by_split[name].update(ordered[cursor:cursor + size])
            cursor += size
        groups_by_split["test"].update(ordered[cursor:])
    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    membership = {group: name for name in SPLIT_NAMES for group in groups_by_split[name]}
    for row in rows:
        splits[membership[row["source_group"]]].append(row)
    return splits


def letter_prompt(state: dict, question: str, texts: list[str]) -> str:
    options = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(texts))
    return (f"State:\n{json.dumps(state, ensure_ascii=False, sort_keys=True)}\n\n"
            f"Question: {question}\n{options}\n\nAnswer with the letter of the best option.\nAnswer:")


def cyclic_texts(row: dict, offset: int) -> list[str]:
    base = [c["text"] for c in row["candidates"]]
    return [base[(i - offset) % len(base)] for i in range(len(base))]


class Teacher:
    """A causal LM scored on the next token after a chat-formatted multiple-choice prompt."""

    def __init__(self, model_id: str, device: str, cache_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import HfApi
        self.model_id, self.device, self.torch = model_id, device, torch
        self.revision = HfApi().model_info(model_id).sha
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.revision, cache_dir=str(cache_dir))
        # The score is read at position -1, so padding must go on the left.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=self.revision, cache_dir=str(cache_dir),
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
        self.model.to(device).eval()
        self.letter_ids = {}
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, f"teacher tokenizer does not map {letter!r} to one token: {ids}"
            self.letter_ids[letter] = ids[0]

    def render(self, prompt: str) -> str:
        """Chat template when available, so the reply starts directly with the letter token."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=False, add_generation_prompt=True)
        return prompt + " "


def label_with_teacher(rows: list[dict], teacher: Teacher, batch_size: int,
                       limit: int | None, orders: int = 3) -> list[dict]:
    """Average next-letter probabilities over cyclic option orders; record disagreement."""
    torch, tokenizer, model, device = teacher.torch, teacher.tokenizer, teacher.model, teacher.device
    letter_ids = teacher.letter_ids

    work = rows if limit is None else rows[:limit]
    requests = [(row, offset) for row in work for offset in range(orders)]
    accum: dict[int, list[list[float]]] = {}
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            part = requests[start:start + batch_size]
            prompts = [teacher.render(letter_prompt(row["state"], row["question"], cyclic_texts(row, offset)))
                       for row, offset in part]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=2048, add_special_tokens=False).to(device)
            logits = model(**inputs).logits[:, -1, :].float()
            for (row, offset), row_logits in zip(part, logits):
                k = len(row["candidates"])
                valid = torch.tensor([letter_ids[LETTERS[i]] for i in range(k)], device=logits.device)
                probs = torch.softmax(row_logits[valid], dim=-1).tolist()
                accum.setdefault(id(row), [[0.0] * k for _ in range(orders)])
                # cyclic_texts puts original option (p - offset) % k at position p
                for p, prob in enumerate(probs):
                    accum[id(row)][offset][(p - offset) % k] = prob
            done = start + len(part)
            if (done // batch_size) % 20 == 0 or done == len(requests):
                print(f"teacher: {done}/{len(requests)} prompt-orders", flush=True)

    labeled = []
    for row in work:
        per_order = accum[id(row)]
        k = len(row["candidates"])
        mean = [sum(per_order[o][i] for o in range(orders)) / orders for i in range(k)]
        total = sum(mean) or 1.0
        mean = [p / total for p in mean]
        winner = max(range(k), key=mean.__getitem__)
        tops = [per_order[o][winner] for o in range(orders)]
        new_row = {key: value for key, value in row.items() if not key.startswith("_")}
        new_row["target"] = normalized(mean)
        new_row["teacher_disagreement"] = round(max(tops) - min(tops), 6)
        new_row["rule_target"] = row["target"]
        labeled.append(new_row)
    return labeled


def strip_private(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-workflow", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts" / "coding-decisions-v0")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER,
                        help=f"teacher model id; use with --no-teacher to skip (default {DEFAULT_TEACHER})")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N rows (smoke tests)")
    parser.add_argument("--no-teacher", action="store_true", help="keep the rule-derived labels, skip the teacher")
    args = parser.parse_args()
    if args.no_teacher:
        args.teacher = None
    if args.teacher and args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            args.device = "cpu"
            print("CUDA unavailable; labeling on CPU (slow; fine for smoke tests only)", flush=True)

    rows = build_rows(args.cases_per_workflow, args.seed)
    splits = split_by_source_group(rows, args.seed)
    groups = {name: {r["source_group"] for r in split} for name, split in splits.items()}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1:]:
            overlap = groups[left] & groups[right]
            assert not overlap, f"source leakage between {left} and {right}: {next(iter(overlap), '')}"

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "anarkali/synthetic-coding-decisions",
        "revision": f"seed-{args.seed}",
        "config": "coding",
        "seed": args.seed,
        "cases_per_workflow": args.cases_per_workflow,
        "question_types": ["choice", "noul", "score"],
        "prompt_version": PROMPT_VERSION,
        "teacher_id": args.teacher,
        "teacher_averaging": None if args.teacher is None else
        "3 cyclic option orders; probabilities renormalized over the valid option letters",
        "label_source": "rule-derived distributions from the latent factors (anarkali.workflows.coding.gold_answers)"
        if args.teacher is None else
        "teacher distribution averaged over option orders in target; rule-derived distribution kept in rule_target",
        "split_unit": "whole synthetic source cases, stratified by workflow",
        "split_counts": {},
    }
    teacher = Teacher(args.teacher, args.device, REPO / ".cache" / "huggingface") if args.teacher else None
    if teacher:
        manifest["teacher_revision"] = teacher.revision
    for name in SPLIT_NAMES:
        split_rows = splits[name]
        if teacher:
            split_rows = label_with_teacher(split_rows, teacher, args.batch_size, args.limit)
        else:
            split_rows = [strip_private(row) for row in split_rows]
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["split_counts"][name] = {"decision_cases": len(split_rows),
                                          "source_groups": len(groups[name]),
                                          "sha256": digest_file(path)}
        print(f"{name}: {len(split_rows)} decisions from {len(groups[name])} source groups -> {path}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
