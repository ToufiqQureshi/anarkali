"""Paired prediction reports. A synthetic report cannot establish model quality."""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"invalid {name}")
    return float(value)


def validate_prediction(row: dict) -> dict:
    if not isinstance(row, dict):
        raise ValueError("prediction must be an object")
    required = {"case_id", "source_group", "request_sha256", "expected", "raw_choice",
                "choice", "probabilities", "latency_ms"}
    if not required <= row.keys():
        raise ValueError(f"missing fields: {sorted(required - row.keys())}")
    for key in ("case_id", "source_group", "request_sha256", "expected", "raw_choice"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"invalid {key}")
    digest = row["request_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("request_sha256 must be a lowercase SHA256 digest")
    probabilities = row["probabilities"]
    if not isinstance(probabilities, dict) or len(probabilities) < 2:
        raise ValueError("at least two candidate probabilities are required")
    if any(not isinstance(k, str) or not k.strip() for k in probabilities):
        raise ValueError("candidate probability keys must be nonempty strings")
    p = {key: _finite(value, "probability") for key, value in probabilities.items()}
    if any(not 0 <= value <= 1 for value in p.values()):
        raise ValueError("probabilities must be in [0, 1]")
    total = math.fsum(p.values())
    # Upstream rounds published probabilities to four decimal places.
    if abs(total - 1) > 1e-3:
        raise ValueError("candidate probabilities must sum to one (tolerance 1e-3)")
    p = {key: value / total for key, value in p.items()}
    if row["expected"] not in p or row["raw_choice"] not in p:
        raise ValueError("expected and raw_choice must be candidate ids")
    if p[row["raw_choice"]] < max(p.values()) - 1e-3:
        raise ValueError("raw_choice is inconsistent with probabilities")
    if row["choice"] is not None and row["choice"] != row["raw_choice"]:
        raise ValueError("accepted choice must equal raw_choice")
    latency = _finite(row["latency_ms"], "latency_ms")
    if latency < 0:
        raise ValueError("latency_ms cannot be negative")
    return dict(row, probabilities=p, latency_ms=latency)


def load_predictions(path: str | Path) -> list[dict]:
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = validate_prediction(json.loads(line))
                if row["case_id"] in seen:
                    raise ValueError("duplicate case_id")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(row["case_id"])
            records.append(row)
    if not records:
        raise ValueError("prediction ledger is empty")
    return records


def _quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("records must not be empty")
    rows = [validate_prediction(r) for r in records]
    accepted = [r for r in rows if r["choice"] is not None]
    nll, brier = [], []
    bins = [[] for _ in range(15)]
    for row in rows:
        p = row["probabilities"]
        correct = row["raw_choice"] == row["expected"]
        nll.append(-math.log(max(p[row["expected"]], 1e-12)))
        brier.append(math.fsum((value - int(key == row["expected"])) ** 2 for key, value in p.items()))
        confidence = p[row["raw_choice"]]
        bins[min(14, int(confidence * 15))].append((confidence, correct))
    ece = math.fsum(len(b) / len(rows) * abs(statistics.mean(p for p, _ in b)
                                          - statistics.mean(ok for _, ok in b)) for b in bins if b)
    latencies = [r["latency_ms"] for r in rows]
    return {
        "cases": len(rows), "source_groups": len({r["source_group"] for r in rows}),
        "accuracy": statistics.mean(r["raw_choice"] == r["expected"] for r in rows),
        "coverage": len(accepted) / len(rows),
        "accepted_accuracy": statistics.mean(r["choice"] == r["expected"] for r in accepted) if accepted else None,
        "nll": statistics.mean(nll), "brier": statistics.mean(brier), "ece_15_bins": ece,
        "latency_p50_ms": _quantile(latencies, .50), "latency_p95_ms": _quantile(latencies, .95),
    }


def compare_predictions(baseline: list[dict], candidate: list[dict], *,
                        bootstrap_samples: int = 2000, seed: int = 17) -> dict:
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 100_000:
        raise ValueError("bootstrap_samples must be between 100 and 100000")
    baseline = [validate_prediction(r) for r in baseline]
    candidate = [validate_prediction(r) for r in candidate]
    left = {r["case_id"]: r for r in baseline}
    right = {r["case_id"]: r for r in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate):
        raise ValueError("duplicate case_id")
    if not left or left.keys() != right.keys():
        raise ValueError("both ledgers must contain exactly the same case ids")
    groups = defaultdict(list)
    for identity in sorted(left):
        a, b = left[identity], right[identity]
        for field in ("source_group", "request_sha256", "expected"):
            if a[field] != b[field]:
                raise ValueError(f"{identity}: incompatible {field}")
        if a["probabilities"].keys() != b["probabilities"].keys():
            raise ValueError(f"{identity}: incompatible candidate ids")
        groups[a["source_group"]].append(int(b["raw_choice"] == b["expected"]) - int(a["raw_choice"] == a["expected"]))
    observed = statistics.mean(d for values in groups.values() for d in values)
    interval = None
    if len(groups) >= 2:
        grouped = list(groups.values())
        rng = random.Random(seed)
        samples = []
        for _ in range(bootstrap_samples):
            selected = [grouped[rng.randrange(len(grouped))] for _ in grouped]
            samples.append(statistics.mean(d for values in selected for d in values))
        interval = [_quantile(samples, .025), _quantile(samples, .975)]
    return {
        "baseline": summarize(baseline), "candidate": summarize(candidate),
        "accuracy_delta": observed, "accuracy_delta_cluster_bootstrap_95_interval": interval,
        "bootstrap_samples": bootstrap_samples, "seed": seed,
        "superiority_established": False,
        "note": "Accuracy alone does not establish release gates; check provenance, coverage, calibration and hardware budgets separately.",
    }
