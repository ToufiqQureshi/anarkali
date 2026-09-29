"""Score a decision set with a trained packed checkpoint and store its distribution as a teacher.

Offline distillation: run the 400M winner once over millions of rows on a GPU, then train the
68M student for many epochs without paying for the teacher again. Each row gets
teacher_targets[NAME], averaged over --orders cyclic option orders (the same position-bias
fix the engine uses). Unlabelled rows (label_source "none") can take it as their target
(--fill-unlabelled); rows with human labels keep theirs, and combine_teachers.py mixes the two.

    python scripts/label_with_checkpoint.py --checkpoint kaggle/bb-ettin-400m-best.pt \\
        --input artifacts/public-decisions-v0/pool --output artifacts/public-pool-400m \\
        --name anarkali400m --fill-unlabelled --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SPLIT_NAMES = ("train", "development", "calibration", "test")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cyclic(row: dict, offset: int) -> dict:
    k = len(row["candidates"])
    order = [(p - offset) % k for p in range(k)]  # position p holds original option (p - offset) % k
    return dict(row, candidates=[row["candidates"][i] for i in order], target=[row["target"][i] for i in order])


def score_rows(checkpoint, rows: list[dict], orders: int, batch_size: int, device, torch) -> list[list[float]]:
    """Mean probabilities over cyclic option orders, in each row's original option order."""
    use_amp = getattr(device, "type", str(device)) == "cuda"
    sums = [[0.0] * len(r["candidates"]) for r in rows]
    counts = [0] * len(rows)
    with torch.inference_mode():
        for offset in range(orders):
            for start in range(0, len(rows), batch_size):
                part = rows[start:start + batch_size]
                active = [i for i, r in enumerate(part) if offset < len(r["candidates"])]
                if not active:
                    continue
                shifted = [cyclic(part[i], offset) for i in active]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = checkpoint.model(*checkpoint.batch(shifted, device))
                probs = torch.softmax(out.logits.float(), -1).cpu().tolist()
                for i, row_probs in zip(active, probs):
                    k = len(part[i]["candidates"])
                    for p in range(k):
                        sums[start + i][(p - offset) % k] += row_probs[p]
                    counts[start + i] += 1
    return [[v / c for v in s] for s, c in zip(sums, counts)]


def main(argv: list[str] | None = None, loader=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="decision directory with manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="anarkali_teacher", help="key under teacher_targets")
    parser.add_argument("--splits", default="train,development,calibration")
    parser.add_argument("--orders", type=int, default=2, help="cyclic option orders averaged per row")
    parser.add_argument("--fill-unlabelled", action="store_true",
                        help="rows with label_source 'none' take the teacher distribution as their target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-rows", type=int, default=20_000, help="rows read and scored at a time")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=Path, default=REPO / ".cache" / "huggingface")
    args = parser.parse_args(argv)
    if args.orders < 1 or args.batch_size < 1 or args.chunk_rows < 1:
        raise SystemExit("--orders, --batch-size and --chunk-rows must be positive")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"--splits must be a subset of {', '.join(SPLIT_NAMES)}")
    if "test" in splits:
        raise SystemExit("the test split is never teacher-labelled; it must stay an honest measurement")

    import torch
    from anarkali.checkpoint import load_packed_checkpoint
    device = torch.device(args.device)
    checkpoint = (loader or load_packed_checkpoint)(args.checkpoint, device, args.cache)
    source = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    counts, filled, started = {}, {}, time.perf_counter()
    for name in SPLIT_NAMES:
        src, dst = args.input / f"{name}.jsonl", args.output / f"{name}.jsonl"
        rows_done = filled[name] = 0
        groups = set()
        with src.open(encoding="utf-8") as reader, dst.open("w", encoding="utf-8", newline="\n") as writer:
            while True:
                chunk = [json.loads(line) for _, line in zip(range(args.chunk_rows), reader) if line.strip()]
                if not chunk:
                    break
                if name in splits:
                    for row, probs in zip(chunk, score_rows(checkpoint, chunk, args.orders, args.batch_size,
                                                            device, torch)):
                        row.setdefault("teacher_targets", {})[args.name] = probs
                        if args.fill_unlabelled and row.get("label_source") == "none":
                            row["target"] = probs
                            row["label_source"] = f"teacher:{args.name}"
                            filled[name] += 1
                for row in chunk:
                    groups.add(row["source_group"])
                    writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_done += len(chunk)
                if name in splits:
                    print(json.dumps({"split": name, "rows": rows_done, "filled": filled[name],
                                      "elapsed_s": round(time.perf_counter() - started)}), flush=True)
        counts[name] = {"decision_cases": rows_done, "source_groups": len(groups), "sha256": digest_file(dst)}
    ckpt_sha = digest_file(args.checkpoint)
    manifest = {
        **{k: v for k, v in source.items() if k not in ("split_counts", "revision", "dataset")},
        "dataset": f"{source.get('dataset', args.input.name)}+{args.name}",
        "revision": hashlib.sha256(json.dumps([source.get("revision"), ckpt_sha, splits, args.orders,
                                               args.fill_unlabelled]).encode()).hexdigest()[:16],
        "teacher_checkpoints": [*source.get("teacher_checkpoints", []),
                                {"name": args.name, "checkpoint_sha256": ckpt_sha,
                                 "model_id": checkpoint.raw.get("model_id"), "orders": args.orders,
                                 "splits": splits, "filled_unlabelled": filled}],
        "split_counts": counts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "filled": filled}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
