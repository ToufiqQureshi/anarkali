"""Mix every teacher's distribution, and the human label where there is one, into one training target.

Input rows carry teacher_targets {name: probabilities} from label_with_checkpoint.py and/or
relabel_with_teachers.py (the Brio teacher included). Per teacher, a temperature is first fitted
on the calibration split's human-labelled rows, so an overconfident LLM and an underconfident
encoder are put on the same honest scale (Guo et al. 2017) before they are averaged. Then:

    teacher mix  m = weighted mean of the calibrated teachers present on the row
    gold row     target = g * gold + (1 - g) * m        (g = --gold-weight)
    pool row     target = m, dropped if the teachers' winners agree less than --min-agreement

teacher_agreement (share of teachers whose top option matches the mix) is written on every row,
for train_anarkali.py --weight-field teacher_agreement. The test split is copied unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SPLIT_NAMES = ("train", "development", "calibration", "test")
GRID = [round(0.3 + 0.02 * i, 2) for i in range(136)]  # 0.30 .. 3.00


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_gold(row: dict) -> bool:
    return str(row.get("label_source", "")).startswith("gold")


def tempered(probs: list[float], temperature: float) -> list[float]:
    logs = [math.log(max(p, 1e-9)) / temperature for p in probs]
    top = max(logs)
    exps = [math.exp(v - top) for v in logs]
    total = sum(exps)
    return [v / total for v in exps]


def fit_temperatures(rows: list[dict]) -> dict[str, float]:
    """Per teacher, the temperature minimising cross-entropy against human labels."""
    by_teacher: dict[str, list[tuple[list[float], list[float]]]] = {}
    for row in rows:
        if is_gold(row):
            for name, probs in row.get("teacher_targets", {}).items():
                if len(probs) == len(row["target"]):
                    by_teacher.setdefault(name, []).append((probs, row["target"]))
    fitted = {}
    for name, pairs in by_teacher.items():
        def loss(t):
            return sum(-sum(g * math.log(max(p, 1e-12)) for g, p in zip(gold, tempered(probs, t)))
                       for probs, gold in pairs)
        fitted[name] = min(GRID, key=loss)
    return fitted


def combine_row(row: dict, temperatures: dict[str, float], weights: dict[str, float], gold_weight: float):
    """(new row, keep?) for one row; rows without teachers keep their target if it is human."""
    teachers = {n: p for n, p in row.get("teacher_targets", {}).items()
                if len(p) == len(row["candidates"]) and weights.get(n, 1.0) > 0}
    if not teachers:
        return row, is_gold(row)
    k = len(row["candidates"])
    total_w = sum(weights.get(n, 1.0) for n in teachers)
    calibrated = {n: tempered(p, temperatures.get(n, 1.0)) for n, p in teachers.items()}
    mix = [sum(weights.get(n, 1.0) * calibrated[n][i] for n in teachers) / total_w for i in range(k)]
    winner = max(range(k), key=mix.__getitem__)
    agreement = sum(max(range(k), key=p.__getitem__) == winner for p in calibrated.values()) / len(calibrated)
    new = dict(row, teacher_agreement=round(agreement, 6))
    if is_gold(row):
        new["original_target"] = row["target"]
        target = [gold_weight * g + (1 - gold_weight) * m for g, m in zip(row["target"], mix)]
    else:
        target = mix
    total = sum(target)
    new["target"] = [v / total for v in target]
    return new, True


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-weight", type=float, default=0.5,
                        help="share of the human label in a gold row's target (1 = ignore teachers there)")
    parser.add_argument("--weight", action="append", default=[], help="NAME=W teacher weight (repeat; default 1)")
    parser.add_argument("--min-agreement", type=float, default=0.0,
                        help="drop pool rows whose teachers' winners agree less than this")
    parser.add_argument("--no-temperature", action="store_true", help="average raw teacher distributions")
    args = parser.parse_args(argv)
    weights = {}
    for item in args.weight:
        name, sep, value = item.partition("=")
        try:
            weights[name] = float(value)
        except ValueError:
            sep = ""
        if not sep or weights[name] < 0:
            raise SystemExit(f"--weight must be NAME=W with W >= 0, got {item!r}")
    if not 0 <= args.gold_weight <= 1 or not 0 <= args.min_agreement <= 1:
        raise SystemExit("--gold-weight and --min-agreement must be in [0, 1]")

    source = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    calibration = []
    with (args.input / "calibration.jsonl").open(encoding="utf-8") as stream:
        calibration = [json.loads(line) for line in stream if line.strip()]
    temperatures = {} if args.no_temperature else fit_temperatures(calibration)
    print(json.dumps({"teacher_temperatures": temperatures}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    counts, stats = {}, {}
    for name in SPLIT_NAMES:
        kept = dropped = 0
        groups = set()
        with (args.input / f"{name}.jsonl").open(encoding="utf-8") as reader, \
                (args.output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as writer:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                if name != "test":
                    row, keep = combine_row(row, temperatures, weights, args.gold_weight)
                    if keep and not is_gold(row) and row.get("teacher_agreement", 1.0) + 1e-9 < args.min_agreement:
                        keep = False
                    if not keep:
                        dropped += 1
                        continue
                groups.add(row["source_group"])
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
        path = args.output / f"{name}.jsonl"
        counts[name] = {"decision_cases": kept, "source_groups": len(groups), "sha256": digest_file(path)}
        stats[name] = {"kept": kept, "dropped": dropped}
    settings = {"gold_weight": args.gold_weight, "weights": weights, "min_agreement": args.min_agreement,
                "temperatures": temperatures}
    manifest = {
        **{k: v for k, v in source.items() if k not in ("split_counts", "revision", "dataset")},
        "dataset": f"{source.get('dataset', args.input.name)}+combined",
        "revision": hashlib.sha256(json.dumps([source.get("revision"), settings], sort_keys=True)
                                   .encode()).hexdigest()[:16],
        "label_source": "calibrated teacher mix; gold rows mixed with their human label (see combine)",
        "combine": {**settings, "stats": stats},
        "split_counts": counts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stats": stats}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
