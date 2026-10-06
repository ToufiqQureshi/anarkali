"""The web-expert dataset builder, offline: a tiny .warc.gz in memory, splits, balance and manifest."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_web_expert", REPO / "scripts" / "build_web_expert.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

TEXT = "<p>" + "The museum opens a new wing for local history and hosts school visits every week. " * 4 + "</p>"


def html(kind, lang="en"):
    ld = json.dumps({"@context": "https://schema.org", "@type": kind, "datePublished": "2026-02-01"})
    return (f'<html lang="{lang}"><head><title>{kind}</title><script type="application/ld+json">{ld}</script>'
            f"</head><body><nav><a href='/'>Home</a></nav><main><h1>{kind}</h1><time>1 February 2026</time>"
            f"<p>Edited 2026-03-05.</p>{TEXT}</main></body></html>")


def record(url, status, body, lang_code="eng"):
    http = (f"HTTP/1.1 {status} X\r\nContent-Type: text/html; charset=utf-8\r\n\r\n").encode() + body.encode()
    response = (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: {url}\r\n"
                f"Content-Length: {len(http)}\r\n\r\n").encode() + http + b"\r\n\r\n"
    meta_body = ('languages-cld2: {"reliable":true,"languages":[{"code":"en","code-iso-639-3":"%s"}]}\r\n'
                 % lang_code).encode()
    metadata = (f"WARC/1.0\r\nWARC-Type: metadata\r\nWARC-Target-URI: {url}\r\n"
                f"Content-Length: {len(meta_body)}\r\n\r\n").encode() + meta_body + b"\r\n\r\n"
    return gzip.compress(response) + gzip.compress(metadata)


class BuilderTest(unittest.TestCase):
    def test_stream_to_rows_and_dataset(self):
        pages = [(f"https://site{i}.example/a/{i}", 200, html(kind)) for i, kind in
                 enumerate(["NewsArticle", "BlogPosting", "Recipe", "Event", "FAQPage", "JobPosting"] * 6)]
        pages.append(("https://gone.example/x", 404, html("WebPage")))
        stream = io.BytesIO(b"".join(record(*p) for p in pages))
        rows, stats = builder.collect_stream(stream, max_pages=1000)
        self.assertEqual(stats["html_pages"], len(pages) - 1)   # the 404 is skipped outside errors_only
        kinds = {builder.decision(r) for r in rows}
        self.assertEqual(kinds, {"page_type", "page_status", "language", "published_date"})
        self.assertTrue(all(r["source_group"].startswith("site") for r in rows))

        errors, _ = builder.collect_stream(io.BytesIO(b"".join(record(*p) for p in pages)), max_pages=1000,
                                           errors_only=True)
        self.assertEqual([(builder.decision(r), builder.gold_label(r)) for r in errors],
                         [("page_status", "not_found")])

        with tempfile.TemporaryDirectory() as tmp:
            manifest = builder.write_dataset(rows + errors, Path(tmp), seed=1,
                                             balance_args={"page_type_ratio": 3.0, "ok_ratio": 3.0,
                                                           "per_language": 1000}, extra={})
            total = sum(c["decision_cases"] for c in manifest["split_counts"].values())
            self.assertGreater(total, 0)
            hosts = {}
            for split in builder.SPLITS:
                for line in (Path(tmp) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    host = json.loads(line)["source_group"]
                    self.assertEqual(hosts.setdefault(host, split), split)   # one host, one split

    def test_balance_caps_ok_and_languages(self):
        def row(decision, label, i):
            return {"case_id": f"https://h{i}.example/::{decision}", "source_group": f"h{i}.example",
                    "candidates": [{"id": label, "text": "x"}, {"id": "other", "text": "y"}], "target": [1.0, 0.0]}
        rows = [row("page_status", "ok", i) for i in range(50)] + [row("page_status", "not_found", i) for i in range(5)]
        rows += [row("language", "en", i) for i in range(30)]
        out = builder.balance(rows, seed=0, page_type_ratio=3.0, ok_ratio=2.0, per_language=10)
        counts = {}
        for r in out:
            key = (builder.decision(r), builder.gold_label(r))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts, {("page_status", "ok"): 10, ("page_status", "not_found"): 5, ("language", "en"): 10})


if __name__ == "__main__":
    unittest.main()
