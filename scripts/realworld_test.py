"""Real-world check of a finished run on fresh pages from sites the run never saw. Runs on a CPU.

Streams random WARC files from a Common Crawl crawl (by default the newest one, so the pages are
newer than the training data), skips every host that appears in the run's dataset, and scores the
universal decisions (page_type, page_status, language, published_date) on every page:
- where the page has an automatic label (schema.org, HTTP status, <html lang>, article dates), it
  reports accuracy and coverage/accuracy at confidence >= 0.9, next to the majority baseline;
- on all pages, it reports how often the model is confident enough to answer without an LLM;
- it writes a review sample that a person must read: the automatic labels are not the gold test.

    python scripts/realworld_test.py --model release-dir-or-best.pt --data run-dataset-dir --output out
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from anarkali.webpage import (PAGE_STATUS, PAGE_STATUS_QUESTION, PAGE_TYPE_QUESTION, PAGE_TYPES,  # noqa: E402
                              page_rows, page_state)
from build_web_expert import (CC, _get, crawl_paths, decode_html, host_of, latest_crawl,  # noqa: E402
                              warc_records, with_language)

CONFIDENT = 0.9
PT_OPTIONS = [{"id": k, "text": v} for k, v in PAGE_TYPES.items()]
ST_OPTIONS = [{"id": k, "text": v} for k, v in PAGE_STATUS.items()]


def known_hosts(data: Path) -> set[str]:
    hosts = set()
    for split in ("train", "development", "calibration", "test"):
        with open(data / f"{split}.jsonl", encoding="utf-8") as stream:
            for line in stream:
                hosts.add(json.loads(line)["source_group"])
    return hosts


def fresh_pages(url: str, *, errors_only: bool, limit: int, per_host: int, skip: set[str], say) -> list[dict]:
    """Pages from one WARC on hosts outside `skip`, with their automatic labels (if any)."""
    pages, hosts = [], Counter()
    try:
        for headers, status, http, body in with_language(warc_records(_get(url, 300))):
            page_url = headers.get("warc-target-uri", "")
            if not page_url or (status == 200) == errors_only:
                continue
            host = host_of(page_url)
            if not host or host in skip or hosts[host] >= per_host:
                continue
            html = decode_html(body, http)
            if html is None:
                continue
            state = page_state(html, page_url)
            if len(state["page"]) < 200:
                continue
            rows, _ = page_rows(html, page_url, host, http_status=status, detected_language=headers.get("cld2", ""))
            pages.append({"url": page_url, "host": host, "http_status": status, "state": state,
                          "labelled": {r["case_id"].rsplit("::", 1)[-1]: r for r in rows}})
            hosts[host] += 1
            if len(pages) >= limit:
                break
    except Exception as exc:  # one broken WARC must not stop the test
        say("WARC failed:", url.rsplit("/", 1)[-1], repr(exc)[:200])
    say(f"{url.rsplit('/', 1)[-1]}: {len(pages)} pages from {len(hosts)} new sites")
    return pages


def gold(row) -> set[str]:
    return {c["id"] for c, t in zip(row["candidates"], row["target"]) if t > 0}


def summarise(results: list[tuple[bool, float, str]]) -> dict:
    """results: (correct, confidence, gold label)."""
    n = len(results)
    sure = [ok for ok, conf, _ in results if conf >= CONFIDENT]
    majority = Counter(g for _, _, g in results).most_common(1)
    return {"n": n,
            "accuracy": round(sum(ok for ok, _, _ in results) / n, 4) if n else None,
            "majority_baseline": round(majority[0][1] / n, 4) if n else None,
            "confident_coverage": round(len(sure) / n, 4) if n else None,
            "confident_accuracy": round(sum(sure) / len(sure), 4) if sure else None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="an ONNX release directory or a best.pt")
    p.add_argument("--graph", default=None, help="ONNX graph inside the release (default: int8 if shipped)")
    p.add_argument("--temperature-by-type", type=json.loads, default=None,
                   help="needed for a best.pt; a release reads it from anarkali.json")
    p.add_argument("--data", type=Path, required=True, help="the run's dataset: its hosts are skipped")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crawl", default="latest")
    p.add_argument("--warcs", type=int, default=2)
    p.add_argument("--error-warcs", type=int, default=1)
    p.add_argument("--pages-per-warc", type=int, default=800)
    p.add_argument("--per-host", type=int, default=2)
    p.add_argument("--review", type=int, default=150)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    from anarkali.engine import Engine
    args.output.mkdir(parents=True, exist_ok=True)
    log = open(args.output / "log.txt", "a", encoding="utf-8")

    def say(*parts):
        text = " ".join(str(x) for x in parts)
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    engine = Engine.load(args.model, graph=args.graph, threads=args.threads)
    if args.temperature_by_type:
        engine.temperature_by_type = args.temperature_by_type
    say("model:", args.model, "| graph:", getattr(engine.backend, "graph", "torch"),
        "| temperatures:", engine.temperature_by_type)
    skip = known_hosts(args.data)
    crawl = latest_crawl() if args.crawl == "latest" else args.crawl
    rng = random.Random(args.seed)
    jobs = [(CC + u, False) for u in rng.sample(crawl_paths(crawl, "warc"), args.warcs)]
    jobs += [(CC + u, True) for u in rng.sample(crawl_paths(crawl, "non200responses"), args.error_warcs)]
    say("crawl:", crawl, "| hosts skipped (in the run's dataset):", len(skip))

    pages_file = args.output / "pages.jsonl"   # kept, so a stopped run resumes without streaming again
    if pages_file.exists():
        pages = [json.loads(line) for line in open(pages_file, encoding="utf-8")]
        say("resuming with", len(pages), "saved pages")
    else:
        pages = []
        for url, errors_only in jobs:
            pages += fresh_pages(url, errors_only=errors_only, limit=args.pages_per_warc, per_host=args.per_host,
                                 skip=skip | {p["host"] for p in pages}, say=say)
        pages_file.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages), encoding="utf-8")
    say("pages:", len(pages), "| non-200:", sum(p["http_status"] != 200 for p in pages))

    # every page gets page_type and page_status; language and date only where the page offers options
    items, where = [], []
    for i, page in enumerate(pages):
        items.append(("choice", page["state"], PAGE_TYPE_QUESTION, PT_OPTIONS))
        where.append((i, "page_type", PT_OPTIONS))
        items.append(("choice", page["state"], PAGE_STATUS_QUESTION, ST_OPTIONS))
        where.append((i, "page_status", ST_OPTIONS))
        for name in ("language", "published_date"):
            row = page["labelled"].get(name)
            if row:
                items.append(("choice", page["state"], row["question"], row["candidates"]))
                where.append((i, name, row["candidates"]))
    scores_file = args.output / "scores.jsonl"   # one line per batch, appended as it is scored
    probs = [pr for line in open(scores_file, encoding="utf-8") for pr in json.loads(line)] \
        if scores_file.exists() else []
    if probs:
        say("resuming after", len(probs), "scored decisions")
    started, done_before = time.time(), len(probs)
    with open(scores_file, "a", encoding="utf-8") as out:
        for start in range(len(probs), len(items), 64):
            batch = engine.score(items[start:start + 64])[0]
            probs += batch
            out.write(json.dumps(batch) + "\n")
            out.flush()
            if (start // 64) % 10 == 0:
                say(f"scored {len(probs)}/{len(items)} in {time.time() - started:.0f}s")
    seconds = (time.time() - started) * len(items) / max(1, len(items) - done_before)
    for (i, name, options), pr in zip(where, probs):
        best = max(range(len(pr)), key=pr.__getitem__)
        pages[i].setdefault("model", {})[name] = {"answer": options[best]["id"], "confidence": round(pr[best], 4)}

    report = {"crawl": crawl, "pages": len(pages), "sites": len({p["host"] for p in pages}),
              "decisions_scored": len(items), "seconds_per_decision": round(seconds / max(1, len(items)), 3),
              "against_automatic_labels": {}, "all_pages": {}}
    by_decision, confusions = defaultdict(list), defaultdict(Counter)
    for page in pages:
        for name, row in page["labelled"].items():
            got = page["model"].get(name)
            if not got:
                continue
            g = gold(row)
            label = sorted(g)[0]
            by_decision[name].append((got["answer"] in g, got["confidence"], label))
            if got["answer"] not in g:
                confusions[name][f"{label} -> {got['answer']}"] += 1
    for name, results in sorted(by_decision.items()):
        report["against_automatic_labels"][name] = {**summarise(results),
                                                    "top_confusions": dict(confusions[name].most_common(8))}
    for name in ("page_type", "page_status"):
        answers = [p["model"][name] for p in pages if p["http_status"] == 200 or name == "page_status"]
        report["all_pages"][name] = {
            "n": len(answers),
            "confident_share": round(sum(a["confidence"] >= CONFIDENT for a in answers) / max(1, len(answers)), 4),
            "answers": dict(Counter(a["answer"] for a in answers).most_common()),
            "confident_answers": dict(Counter(a["answer"] for a in answers if a["confidence"] >= CONFIDENT).most_common()),
        }

    review = []
    for page in rng.sample(pages, min(args.review, len(pages))):
        review.append({"url": page["url"], "http_status": page["http_status"], "title": page["state"]["title"],
                       "page": page["state"]["page"][:700], "model": page["model"],
                       "automatic_labels": {k: sorted(gold(r)) for k, r in page["labelled"].items()}})
    (args.output / "review-sample.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    say(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
