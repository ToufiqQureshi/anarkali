"""Benchmark a released Anarkali model on prepared typed decisions, under inference variants.

    python scripts/benchmark_release.py --model toufiqqureshi651/anarkali \\
        --data artifacts/typed-decisions-v2 --variants orders=1 orders=2 orders=3 orders=1,max_tokens=1024

For each variant the calibration split is scored first and one temperature per question type
is fitted on it (grid search on soft cross-entropy, Guo et al. 2017, arXiv:1706.04599). The test
split is then scored and reported both raw and with those temperatures, so the calibrated numbers
never see test labels. Metrics follow the README: argmax accuracy against the teacher's top
option, 15-bin ECE, Brier, soft CE against the teacher distribution, selective accuracy, and the
share of decisions whose state was cut to fit the token budget.

Writes <output>/benchmark.json and a Markdown table (also to $GITHUB_STEP_SUMMARY when set).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anarkali.packing import pack_row  # noqa: E402

GRID = [round(0.3 + 0.02 * i, 2) for i in range(186)]  # 0.30 .. 4.00
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def kind_of(row: dict) -> str:
    return row.get("question_type", "choice")


def rescale(probs: list[float], temperature: float) -> list[float]:
    """Temperature on log-probabilities; equals logit scaling for a single order."""
    logs = [math.log(max(p, 1e-12)) / temperature for p in probs]
    top = max(logs)
    exps = [math.exp(x - top) for x in logs]
    total = sum(exps)
    return [x / total for x in exps]


def soft_ce(probs: list[float], target: list[float]) -> float:
    return -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, probs))


def fit_temperatures(rows: list[dict], probs: list[list[float]]) -> dict[str, float]:
    by_kind = defaultdict(list)
    for row, p in zip(rows, probs):
        by_kind[kind_of(row)].append((p, row["target"]))
    return {kind: min(GRID, key=lambda t: sum(soft_ce(rescale(p, t), target) for p, target in pairs))
            for kind, pairs in sorted(by_kind.items())}


def metrics(rows: list[dict], probs: list[list[float]]) -> dict[str, dict]:
    groups = defaultdict(lambda: {"n": 0, "correct": 0, "ce": 0.0, "kl": 0.0, "brier": 0.0, "conf": [], "hit": []})
    for row, p in zip(rows, probs):
        pred = max(range(len(p)), key=p.__getitem__)
        gold = max(range(len(row["target"])), key=row["target"].__getitem__)
        for key in ("all", f"type:{kind_of(row)}", f"workflow:{row.get('workflow', '?')}"):
            g = groups[key]
            g["n"] += 1
            g["correct"] += pred == gold
            g["ce"] += soft_ce(p, row["target"])
            # KL(gold || predicted): soft CE minus the gold distribution's entropy, as the leaderboard reports
            g["kl"] += soft_ce(p, row["target"]) + sum(t * math.log(t) for t in row["target"] if t > 0)
            g["brier"] += sum((q - (i == gold)) ** 2 for i, q in enumerate(p))
            g["conf"].append(p[pred])
            g["hit"].append(pred == gold)
    out = {}
    for key, g in sorted(groups.items()):
        bins = [[] for _ in range(15)]
        for c, h in zip(g["conf"], g["hit"]):
            bins[min(14, int(c * 15))].append((c, h))
        ece = sum(len(b) / g["n"] * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
                  for b in bins if b)
        selective = {}
        for threshold in THRESHOLDS:
            kept = [h for c, h in zip(g["conf"], g["hit"]) if c >= threshold]
            selective[str(threshold)] = {"coverage": len(kept) / g["n"],
                                         "accuracy": sum(kept) / len(kept) if kept else None}
        out[key] = {"decisions": g["n"], "accuracy": g["correct"] / g["n"], "ece_15_bins": ece,
                    "brier": g["brier"] / g["n"], "soft_ce": g["ce"] / g["n"], "kl_from_gold": g["kl"] / g["n"],
                    "selective": selective}
    return out


def parse_variant(text: str) -> dict:
    settings = {"orders": 1, "max_tokens": None}
    for part in filter(None, text.split(",")):
        key, sep, value = part.partition("=")
        if not sep or key not in settings:
            raise SystemExit(f"bad variant {text!r}; use orders=N[,max_tokens=M]")
        settings[key] = int(value)
    return settings


def score(engine, rows: list[dict], orders: int, chunk: int = 64) -> tuple[list[list[float]], float]:
    items = [(kind_of(r), r["state"], r["question"], r["candidates"]) for r in rows]
    started, probs = time.perf_counter(), []
    for start in range(0, len(items), chunk):
        part, _tokens = engine.score(items[start:start + chunk], orders=orders)
        probs.extend(part)
    return probs, (time.perf_counter() - started) * 1000 / max(1, len(rows))


def truncated_share(engine, rows: list[dict]) -> float:
    cut = sum(pack_row({"state": r["state"], "question": r["question"], "candidates": r["candidates"]},
                       engine.backend.tokenizer, engine.max_tokens)[2]["state_tokens_dropped"] > 0 for r in rows)
    return cut / max(1, len(rows))


def markdown(report: dict) -> str:
    if not report["variants"]:
        return "No variant finished: " + json.dumps(report.get("failed_variants", {})) + "\n"
    lines = ["| Variant | Accuracy | KL from gold (cal.) | ECE raw | ECE calibrated | Brier cal. | p≥0.7 coverage / acc. "
             "| State cut | ms / decision |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, v in report["variants"].items():
        raw, cal = v["test_raw"]["all"], v["test_calibrated"]["all"]
        sel = cal["selective"]["0.7"]
        acc = "–" if sel["accuracy"] is None else f"{sel['accuracy']:.1%}"
        lines.append(f"| `{name}` | {raw['accuracy']:.1%} | {cal['kl_from_gold']:.3f} | {raw['ece_15_bins']:.3f} | "
                     f"{cal['ece_15_bins']:.3f} | {cal['brier']:.3f} | {sel['coverage']:.0%} / {acc} | {v['state_truncated_share']:.1%} | "
                     f"{v['test_ms_per_decision']:.0f} |")
    first = next(iter(report["variants"].values()))
    lines += ["", "Per question type, first variant, calibrated:", "",
              "| Slice | n | Accuracy | ECE |", "|---|---:|---:|---:|"]
    for key, m in first["test_calibrated"].items():
        if key != "all":
            lines.append(f"| {key} | {m['decisions']} | {m['accuracy']:.1%} | {m['ece_15_bins']:.3f} |")
    lines += ["", "Temperatures fitted on the calibration split: "
              + "; ".join(f"`{n}` {v['temperatures']}" for n, v in report["variants"].items())]
    for name, error in report.get("failed_variants", {}).items():
        lines.append(f"\nVariant `{name}` failed: {error}")
    return "\n".join(lines) + "\n"


def recommend(report: dict, tolerance: float = 0.002) -> dict | None:
    """Release settings: the cheapest variant within `tolerance` of the best accuracy, ties by lower KL.

    Accuracy is raw argmax (temperatures do not change it); KL is after the per-type temperatures.
    """
    variants = report["variants"]
    if not variants:
        return None
    best = max(v["test_raw"]["all"]["accuracy"] for v in variants.values())
    close = [(name, v) for name, v in variants.items() if v["test_raw"]["all"]["accuracy"] >= best - tolerance]
    name, chosen = min(close, key=lambda item: (item[1]["orders"], item[1]["test_calibrated"]["all"]["kl_from_gold"]))
    config = {"orders": chosen["orders"], "temperature_by_type": chosen["temperatures"]}
    if chosen["max_tokens"] != next(iter(variants.values()))["max_tokens"]:
        config["max_tokens"] = chosen["max_tokens"]
    return {"variant": name, "anarkali_json": config,
            "accuracy": chosen["test_raw"]["all"]["accuracy"],
            "kl_from_gold_calibrated": chosen["test_calibrated"]["all"]["kl_from_gold"],
            "ece_calibrated": chosen["test_calibrated"]["all"]["ece_15_bins"]}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="released model directory or Hugging Face repo id")
    parser.add_argument("--data", type=Path, required=True, help="prepared set with calibration.jsonl and test.jsonl")
    parser.add_argument("--variants", nargs="+", default=["orders=1", "orders=2", "orders=3"])
    parser.add_argument("--limit", type=int, default=None, help="use only the first N rows of each split")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts" / "benchmark")
    args = parser.parse_args(argv)

    from anarkali.engine import Engine
    calibration = load_rows(args.data / "calibration.jsonl")[:args.limit]
    test = load_rows(args.data / "test.jsonl")[:args.limit]
    report = {"model": args.model, "data": str(args.data), "calibration_decisions": len(calibration),
              "test_decisions": len(test), "variants": {}}
    engines = {}
    for text in args.variants:
        settings = parse_variant(text)
        key = settings["max_tokens"]
        if key not in engines:
            engines[key] = Engine.load(args.model, threads=args.threads, max_tokens=key)
        engine = engines[key]
        try:
            cal_probs, _ = score(engine, calibration, settings["orders"])
            temperatures = fit_temperatures(calibration, cal_probs)
            test_probs, ms = score(engine, test, settings["orders"])
        except Exception as exc:  # e.g. a graph that cannot run past its traced length
            report.setdefault("failed_variants", {})[text] = repr(exc)[:500]
            print(json.dumps({"variant": text, "failed": repr(exc)[:500]}), flush=True)
            continue
        calibrated = [rescale(p, temperatures[kind_of(r)]) for r, p in zip(test, test_probs)]
        report["variants"][text] = {
            "orders": settings["orders"], "max_tokens": engine.max_tokens, "temperatures": temperatures,
            "test_raw": metrics(test, test_probs), "test_calibrated": metrics(test, calibrated),
            "state_truncated_share": truncated_share(engine, test), "test_ms_per_decision": ms,
        }
        summary = report["variants"][text]
        print(json.dumps({"variant": text, "accuracy": summary["test_raw"]["all"]["accuracy"],
                          "ece_raw": summary["test_raw"]["all"]["ece_15_bins"],
                          "ece_calibrated": summary["test_calibrated"]["all"]["ece_15_bins"],
                          "temperatures": temperatures, "ms_per_decision": round(ms, 1)}), flush=True)
    report["recommended"] = recommend(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = markdown(report)
    if report["recommended"]:
        table += ("\nRecommended release settings (merge into `anarkali.json`): `"
                  + json.dumps(report["recommended"]["anarkali_json"]) + f"` from `{report['recommended']['variant']}`\n")
    (args.output / "benchmark.md").write_text(table, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(f"## Anarkali benchmark: {args.model}\n\n{table}")
    print(table)
    return report


if __name__ == "__main__":
    main()
