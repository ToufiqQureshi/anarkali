"""Score one or more Anarkali models on hand-labelled real-world cases and write an evidence report.

Each case is a JSON line:

    {"case_id": "...", "workflow": "coding_ci_failure", "state": {...},
     "evidence": {"url": "https://github.com/.../job/123"},
     "gold": {"cause": "real_regression"}, "note": "why the label is right"}

`gold` names the correct option for each labelled question (for `noul`, "true" or "false"). Only
labelled questions are asked. The questions come from the workflow definitions in
`anarkali.workflows.coding`, so a model is asked exactly what it is asked in production.

    python scripts/realworld_benchmark.py --cases benchmarks/realworld-ci-v0/cases.jsonl \
        --model anarkali=toufiqqureshi651/anarkali --model large=path/to/release --output out

Writes out/report.json (every prediction) and out/report.md (summary, then one row per case with
the evidence link, the key error line, the gold answer and each model's answer and probability).
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anarkali.engine import Engine  # noqa: E402
from anarkali.workflows.coding import WORKFLOWS  # noqa: E402


def load_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            questions = WORKFLOWS.get(case.get("workflow"))
            if questions is None:
                raise SystemExit(f"line {number}: unknown workflow {case.get('workflow')!r}")
            for qid, answer in case.get("gold", {}).items():
                question = questions.get(qid)
                if question is None:
                    raise SystemExit(f"line {number}: {case['workflow']} has no question {qid!r}")
                allowed = allowed_answers(question)
                if answer not in allowed:
                    raise SystemExit(f"line {number}: {qid}={answer!r} is not one of {allowed}")
            if not case.get("gold"):
                raise SystemExit(f"line {number}: no gold labels")
            cases.append(case)
    return cases


def allowed_answers(question):
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "score":
        return [str(level) for level in range(len(question["criteria"]))]
    return list(question["criteria"])


def picked(answer, kind):
    """The option a model's answer selects, in the same form as the gold label."""
    if kind == "noul":
        return "true" if answer["noul"] >= 0.5 else "false"
    if kind == "score":
        probabilities = answer["probabilities"]
        return max(probabilities, key=probabilities.get)
    return answer["choice"]


def gold_probability(answer, kind, gold):
    if kind == "noul":
        return answer["noul"] if gold == "true" else 1 - answer["noul"]
    return answer["probabilities"][gold]


def run_model(engine, cases):
    results, latencies = [], []
    for case in cases:
        questions = {qid: WORKFLOWS[case["workflow"]][qid] for qid in case["gold"]}
        started = time.perf_counter()
        answers = engine.predict(case["state"], questions)["answers"]
        latencies.append((time.perf_counter() - started) * 1000)
        for qid, gold in case["gold"].items():
            kind = questions[qid]["type"]
            answer = answers[qid]
            choice = picked(answer, kind)
            results.append({"case_id": case["case_id"], "question": qid, "gold": gold, "choice": choice,
                            "correct": choice == gold, "confidence": answer["confidence"],
                            "gold_probability": round(gold_probability(answer, kind, gold), 4)})
    latencies.sort()
    return results, {"p50_ms": statistics.median(latencies),
                     "p95_ms": latencies[int(0.95 * (len(latencies) - 1))], "cases": len(latencies)}


def summarize(results):
    by_question = {}
    for r in results:
        by_question.setdefault(r["question"], []).append(r)
    out = {"all": accuracy(results)}
    out.update({q: accuracy(rows) for q, rows in sorted(by_question.items())})
    # per gold label, so a skewed benchmark cannot hide a model that only knows the common answer
    by_label = {}
    for r in results:
        by_label.setdefault(f"{r['question']}={r['gold']}", []).append(r)
    out.update({key: accuracy(rows) for key, rows in sorted(by_label.items())})
    return out


def majority_baseline(cases):
    """Accuracy of always answering each question's most common gold label."""
    golds = {}
    for case in cases:
        for qid, gold in case["gold"].items():
            golds.setdefault(qid, []).append(gold)
    hits = sum(max(labels.count(v) for v in set(labels)) for labels in golds.values())
    return hits / sum(len(labels) for labels in golds.values())


def accuracy(rows):
    return {"accuracy": sum(r["correct"] for r in rows) / len(rows), "n": len(rows),
            "mean_gold_probability": sum(r["gold_probability"] for r in rows) / len(rows)}


def key_line(state):
    for field in ("error_lines", "signal_lines"):
        lines = [line for line in state.get(field) or [] if "exit code" not in line.lower()]
        if lines:
            return lines[0]
    return (state.get("error_lines") or [""])[0]


def cell(text, limit=90):
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def write_markdown(path, cases, report):
    names = list(report["models"])
    lines = ["# Real-world benchmark", "",
             f"{len(cases)} hand-labelled cases. Accuracy is the share of labelled questions where the "
             "model's top answer matches the label; `p(gold)` is the probability it gave the right answer.", "",
             "| Model | Accuracy | Mean p(gold) | p50 ms | p95 ms |", "|---|---:|---:|---:|---:|"]
    for name in names:
        s, lat = report["models"][name]["summary"]["all"], report["models"][name]["latency"]
        lines.append(f"| {name} | {s['accuracy']:.1%} ({s['n']}) | {s['mean_gold_probability']:.3f} | "
                     f"{lat['p50_ms']:.0f} | {lat['p95_ms']:.0f} |")
    lines.append(f"| always the most common label | {report['majority_baseline']:.1%} | | | |")
    keys = [k for k in report["models"][names[0]]["summary"] if "=" in k]
    lines += ["", "## By gold label", "", "| Label | n | " + " | ".join(names) + " |",
              "|---|---:|" + "---:|" * len(names)]
    for key in keys:
        n = report["models"][names[0]]["summary"][key]["n"]
        lines.append(f"| {key} | {n} | " + " | ".join(
            f"{report['models'][name]['summary'][key]['accuracy']:.0%}" for name in names) + " |")
    lines += ["", "## Every case", "",
              "| Case | Key line | Question | Gold | " + " | ".join(names) + " |",
              "|---|---|---|---|" + "---|" * len(names)]
    by_model = {name: {(r["case_id"], r["question"]): r for r in report["models"][name]["results"]}
                for name in names}
    for case in cases:
        url = case.get("evidence", {}).get("url")
        label = f"[{cell(case['case_id'], 50)}]({url})" if url else cell(case["case_id"], 50)
        for qid, gold in case["gold"].items():
            answers = []
            for name in names:
                r = by_model[name][(case["case_id"], qid)]
                answers.append(f"{'✅' if r['correct'] else '❌'} {r['choice']} ({r['gold_probability']:.2f})")
            lines.append(f"| {label} | {cell(key_line(case['state']))} | {qid} | {gold} | " + " | ".join(answers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                        help="a Hugging Face id, release folder or .pt checkpoint; repeat to compare")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--orders", type=int, default=None)
    args = parser.parse_args(argv)
    cases = load_cases(args.cases)
    report = {"cases": len(cases), "majority_baseline": majority_baseline(cases), "models": {}}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model needs NAME=PATH, got {spec!r}")
        engine = Engine.load(path, orders=args.orders)
        results, latency = run_model(engine, cases)
        report["models"][name] = {"path": path, "summary": summarize(results), "latency": latency,
                                  "results": results}
        print(f"{name}: {report['models'][name]['summary']['all']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output / "report.md", cases, report)
    return report


if __name__ == "__main__":
    main()
