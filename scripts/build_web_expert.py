"""Build the web-expert dataset (v1) from real Common Crawl pages. Standard library only.

Every crawled HTML page becomes up to four universal decisions (see anarkali/webpage.py): page_type,
page_status, language and published_date. Error pages (404, 403, 500, ...) come from Common Crawl's
non-200 responses. Whole hosts stay inside one split, so the test set measures sites the model
never saw. WARC files are streamed, never stored, and processed in parallel.

    python scripts/build_web_expert.py --warcs 12 --error-warcs 3 --output artifacts/web-expert-v1
    python scripts/build_web_expert.py --warc-files a.warc.gz --output ...      # offline
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time
import urllib.request
from urllib.parse import urlsplit
import zlib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from anarkali.webpage import page_rows  # noqa: E402

CC = "https://data.commoncrawl.org/"
UA = {"User-Agent": "anarkali-web-expert/1 (research dataset builder)"}
SPLITS = ("train", "development", "calibration", "test")
SPLIT_SHARES = (("train", 0.80), ("development", 0.08), ("calibration", 0.06), ("test", 0.06))
MAX_HTML_BYTES = 1_500_000


def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def split_of(host: str, seed: int) -> str:
    point = int(hashlib.sha256(f"{seed}:{host}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    total = 0.0
    for name, share in SPLIT_SHARES:
        total += share
        if point <= total:
            return name
    return "test"


def _get(url: str, timeout: int = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def latest_crawl() -> str:
    return json.loads(_get("https://index.commoncrawl.org/collinfo.json", 60).read())[0]["id"]


def crawl_paths(crawl: str, kind: str) -> list[str]:
    """kind: 'warc' (normal pages) or 'non200responses' (404s, 403s, 500s, ...)."""
    return gzip.decompress(_get(f"{CC}crawl-data/{crawl}/{kind}.paths.gz").read()).decode().split()

# ------------------------------------------------------------------------------------------- WARC


def warc_records(stream, max_bytes: int = 10**12):
    """(warc headers, http status, http headers, body) for each response record of a .warc.gz stream.

    Metadata records come through as (headers, "metadata", {}, detected ISO 639-3 language).
    """
    raw_read, pending, decomp, data = 0, b"", zlib.decompressobj(31), b""
    while raw_read < max_bytes:
        chunk = stream.read(1 << 16)
        if not chunk:
            break
        raw_read += len(chunk)
        pending += chunk
        while pending:
            try:
                data += decomp.decompress(pending)
            except zlib.error:
                return
            pending = decomp.unused_data
            if not decomp.eof:
                break
            yield from _parse_record(data)
            data, decomp = b"", zlib.decompressobj(31)


def _headers(block: str) -> dict[str, str]:
    out = {}
    for line in block.split("\r\n"):
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip()
    return out


def _parse_record(data: bytes):
    head, _, rest = data.partition(b"\r\n\r\n")
    headers = _headers(head.decode("utf-8", "replace").split("\r\n", 1)[-1])
    kind = headers.get("warc-type")
    if kind == "metadata":
        code = ""
        match = re.search(rb"languages-cld2: (\{.*?\})\r?\n", rest)
        if match:
            try:
                langs = json.loads(match.group(1)).get("languages") or []
                code = langs[0].get("code-iso-639-3", "") if langs else ""
            except (ValueError, AttributeError):
                pass
        yield headers, "metadata", {}, code
        return
    if kind != "response":
        return
    http_head, _, body = rest.partition(b"\r\n\r\n")
    status_line, _, header_block = http_head.decode("latin-1", "replace").partition("\r\n")
    try:
        status = int(status_line.split()[1])
    except (IndexError, ValueError):
        return
    yield headers, status, _headers(header_block), body[:MAX_HTML_BYTES]


def with_language(records):
    """Attach Common Crawl's detected language (from the metadata record after a response) as 'cld2'."""
    pending = None
    for headers, status, http, body in records:
        if status == "metadata":
            if pending and pending[0].get("warc-target-uri") == headers.get("warc-target-uri"):
                pending[0]["cld2"] = body
                yield pending
                pending = None
            continue
        if pending:
            yield pending
        pending = (headers, status, http, body)
    if pending:
        yield pending


def decode_html(body: bytes, http: dict[str, str]) -> str | None:
    content_type = http.get("content-type", "").lower()
    if "html" not in content_type:
        return None
    charset = content_type.split("charset=")[-1].split(";")[0].strip() if "charset=" in content_type else "utf-8"
    try:
        return body.decode(charset or "utf-8", "replace")
    except LookupError:
        return body.decode("utf-8", "replace")

# ---------------------------------------------------------------------------------------- collect


def collect_stream(stream, *, max_pages: int, errors_only: bool = False) -> tuple[list[dict], Counter]:
    """Rows and stats from one WARC stream. errors_only: keep only non-200 pages (for error WARCs)."""
    rows, stats, seen = [], Counter(), set()
    for headers, status, http, body in with_language(warc_records(stream)):
        url = headers.get("warc-target-uri", "")
        if not url or url in seen or (errors_only and status == 200) or (not errors_only and status != 200):
            continue
        html = decode_html(body, http)
        if html is None:
            continue
        seen.add(url)
        stats["html_pages"] += 1
        got, dropped = page_rows(html, url, host_of(url), http_status=status,
                                 detected_language=headers.get("cld2", ""))
        stats.update(dropped)
        rows.extend(got)
        if stats["html_pages"] >= max_pages:
            break
    return rows, stats


def collect_url(job: tuple[str, int, bool]) -> tuple[str, list[dict], Counter]:
    url, max_pages, errors_only = job
    started = time.time()
    try:
        rows, stats = collect_stream(_get(url, 300), max_pages=max_pages, errors_only=errors_only)
    except Exception as exc:  # one broken WARC must not lose the others
        print(json.dumps({"warc": url.rsplit("/", 1)[-1], "error": repr(exc)[:300]}), flush=True)
        return url, [], Counter({"failed_warcs": 1})
    stats["seconds"] = round(time.time() - started)
    print(json.dumps({"warc": url.rsplit("/", 1)[-1], "rows": len(rows), **stats}), flush=True)
    return url, rows, stats

# ------------------------------------------------------------------------------------------ write


def decision(row) -> str:
    return row["case_id"].rsplit("::", 1)[-1]


def gold_label(row) -> str:
    return row["candidates"][max(range(len(row["target"])), key=row["target"].__getitem__)]["id"]


def cap_per_host(rows, max_pages_per_host: int, seed: int):
    """Keep every row of at most max_pages_per_host pages per host, so no site dominates."""
    pages = defaultdict(set)
    for row in rows:
        pages[row["source_group"]].add(row["case_id"].rsplit("::", 1)[0])
    rng = random.Random(seed)
    keep = set()
    for host, urls in pages.items():
        urls = sorted(urls)
        rng.shuffle(urls)
        keep.update(urls[:max_pages_per_host])
    return [r for r in rows if r["case_id"].rsplit("::", 1)[0] in keep]


def balance(rows, *, seed: int, page_type_ratio: float, ok_ratio: float, per_language: int):
    """Keep any one label from swamping a decision.

    page_type: each label at most page_type_ratio x the median label count.
    page_status: 'ok' at most ok_ratio x all the error labels together.
    language: at most per_language rows per language.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(decision(row), gold_label(row))].append(row)
    for group in groups.values():
        rng.shuffle(group)
    caps = {}
    types = [len(g) for (d, _), g in groups.items() if d == "page_type"]
    if types:
        cap = max(1, int(page_type_ratio * statistics.median(types)))
        caps.update({k: cap for k in groups if k[0] == "page_type"})
    errors = sum(len(g) for (d, lab), g in groups.items() if d == "page_status" and lab != "ok")
    if errors:
        caps[("page_status", "ok")] = max(1, int(ok_ratio * errors))
    caps.update({k: per_language for k in groups if k[0] == "language"})
    out = [r for key, group in groups.items() for r in group[:caps.get(key, len(group))]]
    rng.shuffle(out)
    return out


def write_dataset(rows, output: Path, *, seed: int, balance_args: dict, extra: dict):
    splits = {name: [] for name in SPLITS}
    for row in rows:
        splits[split_of(row["source_group"], seed)].append(row)
    for name in SPLITS:
        splits[name] = balance(splits[name], seed=seed, **balance_args)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in part:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[name] = {"decision_cases": len(part), "source_groups": len({r["source_group"] for r in part}),
                        "labels": dict(sorted(Counter(f"{decision(r)}={gold_label(r)}" for r in part).items())),
                        "label_quality": dict(Counter(r["label_quality"] for r in part)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    revision = hashlib.sha256("".join(counts[n]["sha256"] for n in SPLITS).encode()).hexdigest()[:16]
    manifest = {"dataset": "web-expert-v1", "revision": revision,
                "state": "anarkali.webpage.page_state (structural outline)",
                "label_source": "page_type: schema.org checked against og:type and URL, conflicts dropped; "
                                "page_status: HTTP status; language: <html lang> agreeing with Common Crawl's "
                                "detector; published_date: schema.org/article metadata matched to visible text. "
                                "No label source is part of the model input.",
                "split_unit": "whole hosts; no host appears in two splits",
                "balance": balance_args, **extra, "split_counts": counts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "artifacts" / "web-expert-v1")
    p.add_argument("--crawl", default="latest", help="Common Crawl id such as CC-MAIN-2026-39, or 'latest'")
    p.add_argument("--warcs", type=int, default=12, help="random WARC files of normal pages (~30-40k each)")
    p.add_argument("--error-warcs", type=int, default=3, help="random WARC files of non-200 responses")
    p.add_argument("--warc-files", nargs="*", default=None, help="local .warc.gz files instead of downloading")
    p.add_argument("--max-pages-per-warc", type=int, default=40_000)
    p.add_argument("--max-pages-per-host", type=int, default=20)
    p.add_argument("--page-type-ratio", type=float, default=3.0)
    p.add_argument("--ok-ratio", type=float, default=3.0)
    p.add_argument("--per-language", type=int, default=1500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261006)
    args = p.parse_args()

    started = time.time()
    rows, stats = [], Counter()
    if args.warc_files:
        crawl = "local"
        for path in args.warc_files:
            with open(path, "rb") as stream:
                got, st = collect_stream(stream, max_pages=args.max_pages_per_warc)
            rows.extend(got)
            stats.update(st)
    else:
        crawl = latest_crawl() if args.crawl == "latest" else args.crawl
        rng = random.Random(args.seed)
        jobs = [(CC + u, args.max_pages_per_warc, False) for u in rng.sample(crawl_paths(crawl, "warc"), args.warcs)]
        jobs += [(CC + u, args.max_pages_per_warc, True)
                 for u in rng.sample(crawl_paths(crawl, "non200responses"), args.error_warcs)]
        print(json.dumps({"crawl": crawl, "warcs": [j[0] for j in jobs]}, indent=1), flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            for _, got, st in pool.map(collect_url, jobs):
                rows.extend(got)
                stats.update({k: v for k, v in st.items() if k != "seconds"})
    rows = cap_per_host(rows, args.max_pages_per_host, args.seed)
    if not rows:
        raise SystemExit("no labelled pages found; stream more WARCs")
    manifest = write_dataset(rows, args.output, seed=args.seed,
                             balance_args={"page_type_ratio": args.page_type_ratio, "ok_ratio": args.ok_ratio,
                                           "per_language": args.per_language},
                             extra={"crawl": crawl, "collection_stats": dict(stats),
                                    "build_seconds": round(time.time() - started)})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
