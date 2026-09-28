"""Relabel a prepared decision set with several open teacher LLMs and keep only rows they agree on.

Any OpenAI-compatible chat endpoint works: vLLM on a Kaggle/Colab GPU, Groq, OpenRouter.
Each teacher sees the same letter prompt as generate_coding_decisions.py, over 3 cyclic
option orders to cancel position bias. Probabilities come from first-token logprobs when
the endpoint returns them, otherwise from repeated samples (coarser: steps of 1/samples).

The new target averages every teacher (and, by default, the existing target). Rows whose
teachers pick different winners are dropped from the relabelled split and written to
dropped-<split>.jsonl for review. A labelled test split is always copied untouched, so
benchmark scores stay comparable; only a generated test split whose rows all have
label_source "none" can be labelled, and only with --no-original.

Use non-thinking models (e.g. Qwen3-*-Instruct-2507): a leading <think> token hides the
letter. For hybrid Qwen3 on vLLM pass --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'.

Teacher spec: NAME=MODEL@BASE_URL, with the API key read from $<NAME>_API_KEY (upper case,
optional for local vLLM). Example (model ids are illustrative; check each provider's list):

    QWEN_API_KEY=... MISTRAL_API_KEY=... python scripts/relabel_with_teachers.py \\
        --input artifacts/typed-decisions-v2 --output artifacts/typed-decisions-v2-relabel \\
        --teacher qwen=Qwen/Qwen3-30B-A3B-Instruct-2507@http://localhost:8000/v1 \\
        --teacher mistral=mistralai/mistral-small-3.2-24b-instruct:free@https://openrouter.ai/api/v1
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_coding_decisions import LETTERS, cyclic_texts, digest_file, letter_prompt, normalized  # noqa: E402

PROMPT_VERSION = "relabel-teachers-v0"
SPLIT_NAMES = ("train", "development", "calibration", "test")
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class Teacher:
    name: str
    model: str
    base_url: str
    api_key: str | None = None
    rpm: float = 0.0
    mode: str = "auto"  # auto | logprobs | sample
    extra_body: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _next_slot: float = 0.0

    @classmethod
    def parse(cls, spec: str, rpm: float, mode: str, extra_body: dict | None = None) -> "Teacher":
        name, sep, rest = spec.partition("=")
        model, sep2, base_url = rest.rpartition("@")
        if not sep or not sep2 or not name or not model or not base_url.startswith(("http://", "https://")):
            raise argparse.ArgumentTypeError(f"teacher must be NAME=MODEL@BASE_URL, got {spec!r}")
        return cls(name, model, base_url.rstrip("/"), os.environ.get(f"{name.upper()}_API_KEY"), rpm, mode,
                   dict(extra_body or {}))

    def wait_for_slot(self):
        if self.rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + 60.0 / self.rpm
        time.sleep(max(0.0, slot - now))


def post_chat(teacher: Teacher, body: dict, retries: int = 6, timeout: float = 120.0) -> dict:
    """POST /chat/completions with backoff on rate limits and server errors."""
    headers = {"Content-Type": "application/json"}
    if teacher.api_key:
        headers["Authorization"] = f"Bearer {teacher.api_key}"
    data = json.dumps({**teacher.extra_body, "model": teacher.model, **body}).encode("utf-8")
    delay = 2.0
    for attempt in range(retries + 1):
        teacher.wait_for_slot()
        request = urllib.request.Request(f"{teacher.base_url}/chat/completions", data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == retries:
                detail = exc.read().decode("utf-8", "replace")[:500]
                raise RuntimeError(f"{teacher.name}: HTTP {exc.code}: {detail}") from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else delay
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise RuntimeError(f"{teacher.name}: {exc}") from exc
            wait = delay
        time.sleep(min(wait, 120.0))
        delay *= 2
    raise AssertionError("unreachable")


def first_letter(text: str, k: int) -> str | None:
    for char in text.strip():
        if char in LETTERS[:k]:
            return char
        if not char.isspace() and char not in "*([\"'":
            return None
    return None


def letter_probs_from_logprobs(response: dict, k: int) -> list[float] | None:
    """Renormalize first-token probabilities over the valid option letters; None if absent."""
    try:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    mass = [0.0] * k
    for entry in top or []:
        token = str(entry.get("token", "")).strip()
        if len(token) == 1 and token in LETTERS[:k]:
            mass[LETTERS.index(token)] += math.exp(float(entry["logprob"]))
    return normalized(mass) if sum(mass) > 0 else None


def score_prompt(teacher: Teacher, prompt: str, k: int, samples: int) -> tuple[list[float], str]:
    messages = [{"role": "user", "content": prompt}]
    if teacher.mode in ("auto", "logprobs"):
        response = post_chat(teacher, {"messages": messages, "max_tokens": 1, "temperature": 0,
                                       "logprobs": True, "top_logprobs": 20})
        probs = letter_probs_from_logprobs(response, k)
        if probs is not None:
            return probs, "logprobs"
        if teacher.mode == "logprobs":
            raise RuntimeError(f"{teacher.name}: endpoint returned no letter logprobs")
    counts = [0] * k
    for _ in range(samples):
        response = post_chat(teacher, {"messages": messages, "max_tokens": 4, "temperature": 1.0})
        letter = first_letter(response["choices"][0]["message"].get("content") or "", k)
        if letter is not None:
            counts[LETTERS.index(letter)] += 1
    if not any(counts):
        raise RuntimeError(f"{teacher.name}: no valid option letter in {samples} samples")
    return normalized([float(c) for c in counts]), "sample"


class CacheMiss(RuntimeError):
    pass


class Cache:
    """Directory of append-only JSONL shards of prompt scores, so interrupted runs resume.

    Every shard is read; new scores go to <shard>.jsonl only, so workers on different
    machines never write the same file. A line cut off by a killed session is skipped.
    """

    def __init__(self, directory: Path, shard: str = "local", offline: bool = False):
        self.lock, self.entries, self.offline = threading.Lock(), {}, offline
        if not shard or not all(ch.isalnum() or ch in "-_." for ch in shard):
            raise ValueError(f"cache shard must be a plain file name, got {shard!r}")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{shard}.jsonl"
        # A torn final line from a killed session must not swallow the next entry.
        if self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n"):
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n")
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    @staticmethod
    def key(teacher: Teacher, prompt: str, samples: int) -> str:
        raw = json.dumps([PROMPT_VERSION, teacher.model, teacher.mode, samples, teacher.extra_body, prompt],
                         sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        return self.entries.get(key)

    def put(self, entry: dict):
        with self.lock:
            self.entries[entry["key"]] = entry
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")


def teacher_distribution(teacher: Teacher, row: dict, cache: Cache, samples: int, orders: int):
    """Average one teacher over cyclic option orders, mapped back to the original option order."""
    k = len(row["candidates"])
    per_order, sources = [], set()
    for offset in range(min(orders, k)):
        prompt = letter_prompt(row["state"], row["question"], cyclic_texts(row, offset))
        key = Cache.key(teacher, prompt, samples)
        entry = cache.get(key)
        if entry is None:
            if cache.offline:
                raise CacheMiss(f"{teacher.name}: no cached score for row {row.get('case_id')} (offline run)")
            probs, source = score_prompt(teacher, prompt, k, samples)
            entry = {"key": key, "teacher": teacher.name, "probs": probs, "source": source}
            cache.put(entry)
        sources.add(entry["source"])
        # cyclic_texts puts original option (p - offset) % k at position p
        original = [0.0] * k
        for p, prob in enumerate(entry["probs"]):
            original[(p - offset) % k] = prob
        per_order.append(original)
    mean = normalized([sum(order[i] for order in per_order) / len(per_order) for i in range(k)])
    winner = max(range(k), key=mean.__getitem__)
    tops = [order[winner] for order in per_order]
    return mean, round(max(tops) - min(tops), 6), sorted(sources)


def relabel_row(row: dict, teachers: list[Teacher], cache: Cache, *, samples: int, orders: int,
                include_original: bool, min_agreement: float) -> tuple[dict, bool]:
    k = len(row["candidates"])
    votes = {}
    new_row = dict(row)
    new_row["teacher_targets"], new_row["teacher_position_disagreement"] = {}, {}
    for teacher in teachers:
        dist, position_gap, sources = teacher_distribution(teacher, row, cache, samples, orders)
        votes[teacher.name] = dist
        new_row["teacher_targets"][teacher.name] = dist
        new_row["teacher_position_disagreement"][teacher.name] = position_gap
        new_row.setdefault("teacher_sources", {})[teacher.name] = sources
    if include_original:
        votes["original"] = row["target"]
    target = normalized([sum(dist[i] for dist in votes.values()) / len(votes) for i in range(k)])
    winner = max(range(k), key=target.__getitem__)
    agreeing = sum(1 for dist in votes.values() if max(range(k), key=dist.__getitem__) == winner)
    agreement = agreeing / len(votes)
    new_row["original_target"] = row["target"]
    new_row["target"] = target
    new_row["teacher_agreement"] = round(agreement, 6)
    return new_row, agreement + 1e-9 >= min_agreement


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def score_only(args, teachers: list[Teacher], cache: Cache, splits: list[str]) -> dict:
    """Fill the cache for the given teachers; relabelling happens in a later offline run."""
    scored = {}
    for name in splits:
        rows = load_jsonl(args.input / f"{name}.jsonl")
        rows = [row for row in (rows if args.limit is None else rows[:args.limit])
                if len(row["candidates"]) <= len(LETTERS)]

        def job(row):
            for teacher in teachers:
                teacher_distribution(teacher, row, cache, args.samples, args.orders)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, _ in enumerate(pool.map(job, rows), 1):
                if index % 50 == 0 or index == len(rows):
                    print(f"{name}: scored {index}/{len(rows)} rows", flush=True)
        scored[name] = len(rows)
    print(json.dumps({"scored": scored, "cache": str(cache.path)}), flush=True)
    return {"scored": scored}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="prepared decision directory with manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", action="append", required=True, help="NAME=MODEL@BASE_URL (repeat)")
    parser.add_argument("--splits", default="train",
                        help="comma-separated splits to relabel; test only for unlabelled generated sets")
    parser.add_argument("--mode", choices=("auto", "logprobs", "sample"), default="auto")
    parser.add_argument("--samples", type=int, default=5, help="samples per prompt when logprobs are unavailable")
    parser.add_argument("--orders", type=int, default=3, help="cyclic option orders per row")
    parser.add_argument("--min-agreement", type=float, default=2 / 3,
                        help="share of voters whose winner matches the averaged winner (default 2/3)")
    parser.add_argument("--no-original", action="store_true", help="do not count the existing target as a voter")
    parser.add_argument("--rpm", type=float, default=0.0, help="max requests per minute per teacher (0 = no limit)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="relabel only the first N rows per split")
    parser.add_argument("--teacher-extra", action="append", default=[],
                        help="NAME=JSON request fields for one teacher, overriding --extra-body (repeat)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="shared score cache directory (default <output>/teacher-cache)")
    parser.add_argument("--cache-shard", default="local", help="file name this run appends to inside --cache-dir")
    parser.add_argument("--offline", action="store_true",
                        help="never call a teacher; fail on any score missing from the cache")
    parser.add_argument("--only-score", action="store_true",
                        help="fill the cache for these teachers and write nothing else (one GPU, one teacher at a time)")
    parser.add_argument("--extra-body", type=json.loads, default={},
                        help='JSON merged into every request, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    args = parser.parse_args(argv)

    teachers = [Teacher.parse(spec, args.rpm, args.mode, args.extra_body) for spec in args.teacher]
    by_name = {t.name: t for t in teachers}
    for item in args.teacher_extra:
        name, sep, raw = item.partition("=")
        if not sep or name not in by_name:
            raise SystemExit(f"--teacher-extra must be NAME=JSON for a given teacher, got {item!r}")
        by_name[name].extra_body = json.loads(raw)
    if len({t.name for t in teachers}) != len(teachers) or "original" in {t.name for t in teachers}:
        raise SystemExit("teacher names must be unique and must not be 'original'")
    splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    if not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"--splits must be a subset of {', '.join(SPLIT_NAMES)}")
    if "test" in splits:
        # A test split with real labels is never relabelled, so benchmark scores stay comparable.
        # One that was generated without labels (label_source "none") has nothing to protect.
        test_rows = load_jsonl(args.input / "test.jsonl")
        if not args.no_original or not test_rows or any(r.get("label_source") != "none" for r in test_rows):
            raise SystemExit("the test split is relabelled only when every row has label_source 'none' "
                             "and --no-original is set; benchmark scores must stay comparable")
    if not 0 < args.min_agreement <= 1 or args.samples < 1 or args.orders < 1 or args.workers < 1:
        raise SystemExit("invalid --min-agreement, --samples, --orders or --workers")

    source_manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache_dir or args.output / "teacher-cache", args.cache_shard, args.offline)
    if args.only_score:
        return score_only(args, teachers, cache, splits)
    counts, stats = {}, {}
    for name in SPLIT_NAMES:
        path = args.output / f"{name}.jsonl"
        if name not in splits:
            shutil.copyfile(args.input / f"{name}.jsonl", path)
            rows = load_jsonl(path)
            counts[name] = {"decision_cases": len(rows), "source_groups": len({r["source_group"] for r in rows}),
                            "sha256": digest_file(path)}
            continue
        rows = load_jsonl(args.input / f"{name}.jsonl")
        work = rows if args.limit is None else rows[:args.limit]
        rest = rows[len(work):]
        too_wide = [row for row in work if len(row["candidates"]) > len(LETTERS)]
        work = [row for row in work if len(row["candidates"]) <= len(LETTERS)]

        def job(row):
            return relabel_row(row, teachers, cache, samples=args.samples, orders=args.orders,
                               include_original=not args.no_original, min_agreement=args.min_agreement)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = []
            for index, result in enumerate(pool.map(job, work), 1):
                results.append(result)
                if index % 50 == 0 or index == len(work):
                    print(f"{name}: {index}/{len(work)} rows", flush=True)
        kept = [row for row, keep in results if keep]
        dropped = [row for row, keep in results if not keep]
        write_jsonl(args.output / f"dropped-{name}.jsonl", dropped)
        stats[name] = {"relabelled": len(results), "kept": len(kept), "dropped": len(dropped),
                       "kept_original_too_many_options": len(too_wide), "not_relabelled": len(rest)}
        rows = kept + too_wide + rest
        write_jsonl(path, rows)
        counts[name] = {"decision_cases": len(rows), "source_groups": len({r["source_group"] for r in rows}),
                        "sha256": digest_file(path)}

    teacher_ids = [{"name": t.name, "model": t.model, "base_url": t.base_url, "mode": t.mode} for t in teachers]
    revision_raw = json.dumps([source_manifest.get("revision"), PROMPT_VERSION, teacher_ids, splits,
                               args.min_agreement, not args.no_original, args.orders, args.samples])
    manifest = {
        "dataset": f"{source_manifest.get('dataset', args.input.name)}+teachers",
        "revision": hashlib.sha256(revision_raw.encode("utf-8")).hexdigest()[:16],
        "source": {"path": str(args.input), "dataset": source_manifest.get("dataset"),
                   "revision": source_manifest.get("revision")},
        "prompt_version": PROMPT_VERSION,
        "teachers": teacher_ids,
        "relabelled_splits": splits,
        "label_source": (f"average of teachers {', '.join(t.name for t in teachers)}"
                         + ("" if args.no_original else " and the original target")
                         + f" over {args.orders} cyclic option orders; rows below {args.min_agreement:.3f}"
                           " winner agreement dropped. Unrelabelled splits copied unchanged."),
        "relabel_stats": stats,
        "split_unit": source_manifest.get("split_unit"),
        "split_counts": counts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"relabel_stats": stats, "output": str(args.output)}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
