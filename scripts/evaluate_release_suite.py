"""Evaluate a released Anarkali model on a JSONL typed-decision suite."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows, start=1):
        missing = {"id", "state", "questions", "expected"} - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing required keys: {sorted(missing)}")
    return rows


def _answer_value(answer: dict[str, Any]) -> Any:
    for key in ("choice", "value", "score", "noul"):
        if key in answer:
            return answer[key]
    return None


def evaluate(engine: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    results = []
    tag_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    question_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for case in cases:
        prediction = engine.predict(case["state"], case["questions"])
        answers = prediction["answers"]
        mismatches = {}
        for qid, gold in case["expected"].items():
            got = _answer_value(answers.get(qid, {}))
            if got != gold:
                mismatches[qid] = {"expected": gold, "actual": got}
            question_totals[qid][0] += 1
            question_totals[qid][1] += got == gold
        correct = not mismatches
        for tag in case.get("tags", ["untagged"]):
            tag_totals[tag][0] += 1
            tag_totals[tag][1] += correct
        results.append({
            "id": case["id"],
            "expected": case["expected"],
            "answers": answers,
            "correct": correct,
            "mismatches": mismatches,
        })

    correct = sum(row["correct"] for row in results)
    total = len(results)
    return {
        "model": getattr(engine, "name", "unknown"),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "by_tag": {tag: {"total": n, "correct": c, "accuracy": c / n if n else 0.0}
                   for tag, (n, c) in sorted(tag_totals.items())},
        "by_question": {qid: {"total": n, "correct": c, "accuracy": c / n if n else 0.0}
                        for qid, (n, c) in sorted(question_totals.items())},
        "failures": [row for row in results if not row["correct"]],
        "results": results,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, default=REPO / "benchmarks" / "anarkali-routing-v1" / "cases.jsonl")
    parser.add_argument("--orders", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    from anarkali.engine import Engine

    report = evaluate(Engine.load(args.model, orders=args.orders, threads=args.threads), load_cases(args.cases))
    report.update({"model_id": args.model, "cases": str(args.cases), "min_accuracy": args.min_accuracy})
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["accuracy"] < args.min_accuracy:
        raise SystemExit(f"accuracy {report['accuracy']:.3f} is below required {args.min_accuracy:.3f}")
    return report


if __name__ == "__main__":
    main()
