"""Build the web-decisions dataset from real Common Crawl pages, labelled by their own schema.org markup.

Each HTML page with JSON-LD becomes up to three decisions (see anarkali/web.py):
page_type for every page, and price_field and in_stock for product pages. Whole hosts are kept
inside one split, so the test set measures pages from sites the model never saw.

    python scripts/build_web_decisions.py --warcs 4 --output artifacts/web-decisions-v0
    python scripts/build_web_decisions.py --warc-files a.warc.gz b.warc.gz --output ...   # offline
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from anarkali.web import page_decisions  # noqa: E402

CC = "https://data.commoncrawl.org/"
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


def latest_crawl(session) -> str:
    return session.get("https://index.commoncrawl.org/collinfo.json", timeout=60).json()[0]["id"]


def warc_urls(session, crawl: str, count: int, seed: int) -> list[str]:
    raw = session.get(f"{CC}crawl-data/{crawl}/warc.paths.gz", timeout=120).content
    paths = gzip.decompress(raw).decode().split()
    return [CC + p for p in random.Random(seed).sample(paths, count)]


def html_records(stream, max_pages: int):
    """(url, html) for successful HTML responses in one WARC stream."""
    from warcio.archiveiterator import ArchiveIterator
    seen = 0
    for record in ArchiveIterator(stream):
        if record.rec_type != "response" or record.http_headers is None:
            continue
        if record.http_headers.get_statuscode() != "200":
            continue
        content_type = (record.http_headers.get_header("Content-Type") or "").lower()
        if "html" not in content_type:
            continue
        url = record.rec_headers.get_header("WARC-Target-URI") or ""
        body = record.content_stream().read(MAX_HTML_BYTES)
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        try:
            html = body.decode(charset, errors="replace")
        except LookupError:
            html = body.decode("utf-8", errors="replace")
        yield url, html
        seen += 1
        if seen >= max_pages:
            return


def collect(sources, *, max_pages_per_warc, max_pages_per_host, log_every=2000):
    """sources: iterable of (name, opener) where opener() returns a binary WARC stream."""
    rows_by_url, per_host, stats = {}, Counter(), Counter()
    for name, opener in sources:
        started = time.time()
        try:
            stream = opener()
            for url, html in html_records(stream, max_pages_per_warc):
                stats["html_pages"] += 1
                if stats["html_pages"] % log_every == 0:
                    print(json.dumps({"warc": name.rsplit("/", 1)[-1], **stats,
                                      "seconds": round(time.time() - started)}), flush=True)
                host = host_of(url)
                if not host or url in rows_by_url or per_host[host] >= max_pages_per_host:
                    continue
                rows = page_decisions(html, url, host)
                if rows:
                    rows_by_url[url] = rows
                    per_host[host] += 1
                    stats["labelled_pages"] += 1
        except Exception as exc:  # one broken or interrupted WARC must not lose the others
            print(json.dumps({"warc": name, "error": repr(exc)[:300]}), flush=True)
    return [row for rows in rows_by_url.values() for row in rows], stats


def question_key(row) -> str:
    return row["case_id"].rsplit("::", 1)[-1]


def gold_label(row) -> str:
    return row["candidates"][max(range(len(row["target"])), key=row["target"].__getitem__)]["id"]


def balance(rows, *, max_ratio: float, seed: int):
    """Cap every page_type / in_stock label at max_ratio x its rarest label. Train only."""
    rng = random.Random(seed)
    out, groups = [], defaultdict(list)
    for row in rows:
        key = question_key(row)
        if key in ("page_type", "in_stock"):
            groups[(key, gold_label(row))].append(row)
        else:
            out.append(row)
    for key in ("page_type", "in_stock"):
        labels = {lab: grp for (k, lab), grp in groups.items() if k == key}
        if not labels:
            continue
        cap = int(max_ratio * min(len(g) for g in labels.values()))
        for grp in labels.values():
            rng.shuffle(grp)
            out.extend(grp[:max(cap, 1)])
    rng.shuffle(out)
    return out


def mix_rows(directory: Path, count: int, seed: int):
    """Train-split rows from an older decision set, so the model keeps the general format."""
    with (directory / "train.jsonl").open(encoding="utf-8", newline="") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = random.Random(seed).sample(rows, min(count, len(rows)))
    for row in rows:
        row["source_group"] = "mix::" + row["source_group"]
    return rows


def write_dataset(rows, output: Path, *, seed: int, max_ratio: float, mix_dir: Path | None,
                  mix_ratio: float, extra: dict):
    splits = {name: [] for name in SPLITS}
    for row in rows:
        splits[split_of(row["source_group"], seed)].append(row)
    if max_ratio > 0:
        splits["train"] = balance(splits["train"], max_ratio=max_ratio, seed=seed)
    mixed = 0
    if mix_dir and mix_ratio > 0:
        count = int(len(splits["train"]) * mix_ratio / (1 - mix_ratio))
        extra_rows = mix_rows(mix_dir, count, seed)
        mixed = len(extra_rows)
        splits["train"] = splits["train"] + extra_rows
        random.Random(seed + 1).shuffle(splits["train"])
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in part:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        labels = Counter(f"{question_key(r)}={gold_label(r)}" for r in part if r["workflow"] == "web")
        counts[name] = {"decision_cases": len(part), "source_groups": len({r["source_group"] for r in part}),
                        "labels": dict(sorted(labels.items())),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    revision = hashlib.sha256("".join(counts[n]["sha256"] for n in SPLITS).encode()).hexdigest()[:16]
    manifest = {"dataset": "web-decisions-v0", "revision": revision,
                "label_source": "each page's own schema.org JSON-LD; noisy but real, never seen by the model",
                "split_unit": "whole hosts; no host appears in two splits",
                "mixed_train_rows": mixed, "mix_source": str(mix_dir) if mixed else None,
                **extra, "split_counts": counts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "artifacts" / "web-decisions-v0")
    p.add_argument("--crawl", default="latest", help="Common Crawl id such as CC-MAIN-2025-38, or 'latest'")
    p.add_argument("--warcs", type=int, default=4, help="random WARC files to stream (~25-40k HTML pages each)")
    p.add_argument("--warc-files", nargs="*", default=None, help="local .warc.gz files instead of downloading")
    p.add_argument("--max-pages-per-warc", type=int, default=40_000)
    p.add_argument("--max-pages-per-host", type=int, default=20)
    p.add_argument("--max-label-ratio", type=float, default=3.0,
                   help="cap each page_type/in_stock label in train at this x the rarest; 0 disables")
    p.add_argument("--mix", type=Path, default=None, help="older prepared decision set to mix into train")
    p.add_argument("--mix-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20261004)
    args = p.parse_args()
    if not 0 <= args.mix_ratio < 1:
        raise SystemExit("--mix-ratio must be in [0, 1)")

    if args.warc_files:
        sources = [(f, (lambda f=f: open(f, "rb"))) for f in args.warc_files]
        crawl = "local"
    else:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "anarkali-web-decisions/0 (research dataset builder)"
        crawl = latest_crawl(session) if args.crawl == "latest" else args.crawl
        urls = warc_urls(session, crawl, args.warcs, args.seed)
        print(json.dumps({"crawl": crawl, "warcs": urls}, indent=2), flush=True)

        def opener(url):
            response = session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            return response.raw
        sources = [(u, (lambda u=u: opener(u))) for u in urls]
    rows, stats = collect(sources, max_pages_per_warc=args.max_pages_per_warc,
                          max_pages_per_host=args.max_pages_per_host)
    if not rows:
        raise SystemExit("no labelled pages found; stream more WARCs")
    manifest = write_dataset(rows, args.output, seed=args.seed, max_ratio=args.max_label_ratio,
                             mix_dir=args.mix, mix_ratio=args.mix_ratio,
                             extra={"crawl": crawl, "collection_stats": dict(stats)})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
