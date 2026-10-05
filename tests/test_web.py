"""Web decisions: page states, schema.org labels and the Common Crawl dataset builder (offline)."""
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anarkali.typed import question_candidates  # noqa: E402
from anarkali.web import (IN_STOCK_QUESTION, PAGE_TYPE_QUESTION, page_decisions, page_state,  # noqa: E402
                          parse_amount, price_candidates)

FILLER = "<p>" + "Free delivery across India on orders above Rs. 499. Easy 7 day returns. " * 5 + "</p>"


def product_html(price="1299.00", availability="https://schema.org/InStock"):
    ld = {"@context": "https://schema.org", "@type": "Product", "name": "Steel Bottle 1L",
          "offers": {"@type": "Offer", "price": price, "priceCurrency": "INR", "availability": availability}}
    return f"""<html><head><title>Steel Bottle 1L | Shop</title>
<script type="application/ld+json">{json.dumps(ld)}</script>
<style>.x{{color:red}}</style></head><body>
<nav><a href="/">Home</a> <a href="/bottles">Bottles</a></nav>
<h1>Steel Bottle 1L</h1>
<div class="price">MRP <s>₹1,999.00</s> Sale price <span>₹1,299.00</span></div>
<p>EMI from ₹108/month. Add to cart.</p>{FILLER}
<script>var secret = "₹5.00";</script></body></html>"""


def listing_html():
    items = [{"@type": "ListItem", "position": i, "url": f"/p/{i}"} for i in range(3)]
    ld = {"@context": "https://schema.org", "@type": "ItemList", "itemListElement": items}
    return (f'<html><head><title>Bottles</title><script type="application/ld+json">{json.dumps(ld)}</script>'
            f"</head><body><h1>All bottles</h1>{FILLER}</body></html>")


def article_html():
    ld = {"@context": "https://schema.org", "@graph": [{"@type": "WebSite"}, {"@type": "NewsArticle"}]}
    return (f'<html><head><script type="application/ld+json">{json.dumps(ld)}</script></head>'
            f"<body><h1>Monsoon arrives early</h1>{FILLER}</body></html>")


def target_of(row):
    return dict(zip([c["id"] for c in row["candidates"]], row["target"]))


class PageStateTests(unittest.TestCase):
    def test_state_keeps_visible_text_and_drops_scripts_and_markup(self):
        state = page_state(product_html(), "https://shop.example/p/1")
        self.assertEqual(state["title"], "Steel Bottle 1L | Shop")
        self.assertIn("Sale price ₹1,299.00", state["text"])
        self.assertNotIn("schema.org", json.dumps(state))
        self.assertNotIn("secret", state["text"])
        self.assertNotIn("color:red", state["text"])
        self.assertEqual(state["link_count"], 2)

    def test_state_text_is_capped(self):
        state = page_state("<p>" + "word " * 5000 + "</p>", max_text_chars=500)
        self.assertLessEqual(len(state["text"]), 500)

    def test_malformed_html_does_not_raise(self):
        self.assertIsInstance(page_state("<div><p>unclosed <b>tags & stuff"), dict)


class AmountTests(unittest.TestCase):
    def test_common_formats(self):
        for text, value in [("₹1,299.00", 1299.0), ("1.299,50 €", 1299.5), ("$12", 12.0), ("1299", 1299.0),
                            ("Rs. 2,49,999", 249999.0), ("€1.299", 1299.0), ("19,99 €", 19.99)]:
            self.assertEqual(parse_amount(text), value, text)
        self.assertIsNone(parse_amount("free"))

    def test_price_candidates_mark_each_match(self):
        found = price_candidates(["MRP ₹1,999.00 Sale price ₹1,299.00"])
        self.assertEqual([c["value"] for c in found], [1999.0, 1299.0])
        self.assertIn("[₹1,299.00]", found[1]["text"])


class DecisionTests(unittest.TestCase):
    def test_product_page_yields_three_labelled_decisions(self):
        rows = page_decisions(product_html(), "https://shop.example/p/1", "shop.example")
        by = {r["case_id"].rsplit("::", 1)[-1]: r for r in rows}
        self.assertEqual(set(by), {"page_type", "price_field", "in_stock"})
        self.assertEqual(target_of(by["page_type"])["product"], 1.0)
        price = by["price_field"]
        gold = [c["text"] for c, t in zip(price["candidates"], price["target"]) if t > 0]
        self.assertEqual(len(gold), 1)
        self.assertIn("[₹1,299.00]", gold[0])
        self.assertAlmostEqual(sum(price["target"]), 1.0)
        self.assertEqual(target_of(by["in_stock"]), {"false": 0.0, "true": 1.0})
        self.assertEqual(by["in_stock"]["question_type"], "noul")
        for row in rows:
            self.assertNotIn("InStock", json.dumps(row["state"]))

    def test_rows_match_the_inference_question_format(self):
        rows = page_decisions(product_html(), "https://shop.example/p/1", "shop.example")
        by = {r["case_id"].rsplit("::", 1)[-1]: r for r in rows}
        for key, spec in (("page_type", PAGE_TYPE_QUESTION), ("in_stock", IN_STOCK_QUESTION)):
            _, question, options = question_candidates(spec)
            self.assertEqual(by[key]["question"], question)
            self.assertEqual(by[key]["candidates"], options)

    def test_out_of_stock_and_unknown_availability(self):
        rows = page_decisions(product_html(availability="http://schema.org/OutOfStock"), "https://a.example/x", "a")
        stock = [r for r in rows if r["case_id"].endswith("in_stock")][0]
        self.assertEqual(target_of(stock), {"false": 1.0, "true": 0.0})
        rows = page_decisions(product_html(availability="PreOrder"), "https://a.example/y", "a")
        self.assertFalse(any(r["case_id"].endswith("in_stock") for r in rows))

    def test_price_question_skipped_when_price_not_visible(self):
        rows = page_decisions(product_html(price="777"), "https://a.example/z", "a")
        self.assertFalse(any(r["case_id"].endswith("price_field") for r in rows))

    def test_listing_article_and_untyped_pages(self):
        listing = page_decisions(listing_html(), "https://a.example/c", "a")
        self.assertEqual([target_of(r)["listing"] for r in listing], [1.0])
        article = page_decisions(article_html(), "https://news.example/a", "news.example")
        self.assertEqual(target_of(article[0])["article"], 1.0)
        self.assertEqual(page_decisions(f"<html><body>{FILLER}</body></html>", "https://a.example/u", "a"), [])


class BuilderTests(unittest.TestCase):
    def setUp(self):
        try:
            from warcio.warcwriter import WARCWriter  # noqa: F401
        except ImportError:
            self.skipTest("warcio not installed")
        spec = importlib.util.spec_from_file_location("build_web_decisions", REPO / "scripts" / "build_web_decisions.py")
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)

    def warc(self, pages):
        from warcio.statusandheaders import StatusAndHeaders
        from warcio.warcwriter import WARCWriter
        buffer = io.BytesIO()
        writer = WARCWriter(buffer, gzip=True)
        for url, html, status in pages:
            headers = StatusAndHeaders(f"{status} OK", [("Content-Type", "text/html; charset=utf-8")],
                                       protocol="HTTP/1.1")
            writer.write_record(writer.create_warc_record(url, "response", payload=io.BytesIO(html.encode()),
                                                          http_headers=headers))
        buffer.seek(0)
        return buffer

    def test_end_to_end_dataset_has_host_disjoint_splits_and_valid_manifest(self):
        pages = []
        for h in range(40):
            for i in range(3):
                pages.append((f"https://www.shop{h}.example/p/{i}", product_html(), 200))
            pages.append((f"https://news{h}.example/a", article_html(), 200))
            pages.append((f"https://shop{h}.example/c", listing_html(), 200))
        pages.append(("https://shop0.example/missing", product_html(), 404))
        rows, stats = self.builder.collect([("mem", lambda: self.warc(pages))], max_pages_per_warc=10_000,
                                           max_pages_per_host=2)
        self.assertEqual(stats["html_pages"], len(pages) - 1)
        self.assertEqual(stats["labelled_pages"], 40 * 2 + 40)  # per-host cap: 2 per shop, 1 per news site
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = self.builder.write_dataset(rows, out, seed=1, max_ratio=3.0, mix_dir=None, mix_ratio=0,
                                                  extra={})
            owners = {}
            import hashlib
            for name in self.builder.SPLITS:
                data = (out / f"{name}.jsonl").read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["split_counts"][name]["sha256"])
                for line in data.decode().splitlines():
                    group = json.loads(line)["source_group"]
                    self.assertEqual(owners.setdefault(group, name), name)
            self.assertGreater(manifest["split_counts"]["train"]["decision_cases"], 0)

    def test_mix_adds_only_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old"
            old.mkdir()
            old_row = {"case_id": "o::q", "source_group": "g", "workflow": "support", "state": {"x": 1},
                       "question": "Route?", "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
                       "target": [1.0, 0.0]}
            (old / "train.jsonl").write_text("\n".join(json.dumps({**old_row, "source_group": f"g{i}"})
                                                       for i in range(50)) + "\n")
            rows = page_decisions(product_html(), "https://shop.example/p/1", "shop.example")
            out = Path(tmp) / "new"
            manifest = self.builder.write_dataset(rows * 10, out, seed=3, max_ratio=0, mix_dir=old, mix_ratio=0.5,
                                                  extra={})
            self.assertGreater(manifest["mixed_train_rows"], 0)
            for name in ("development", "calibration", "test"):
                for line in (out / f"{name}.jsonl").read_text().splitlines():
                    self.assertEqual(json.loads(line)["workflow"], "web")


class DownsampleTests(unittest.TestCase):
    """Runs without warcio: the WARC reader is replaced by a list of pages."""

    def test_non_product_pages_are_downsampled_and_products_kept(self):
        spec = importlib.util.spec_from_file_location("build_web_decisions", REPO / "scripts" / "build_web_decisions.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        pages = []
        for h in range(200):
            pages.append((f"https://shop{h}.example/p", product_html()))
            pages.append((f"https://news{h}.example/a", article_html()))
        builder.html_records = lambda stream, max_pages: iter(pages)
        rows, stats = builder.collect([("mem", lambda: None)], max_pages_per_warc=10_000,
                                      max_pages_per_host=5, non_product_keep=0.3)
        page_types = Counter(builder.gold_label(r) for r in rows if r["case_id"].endswith("page_type"))
        self.assertEqual(page_types["product"], 200)
        self.assertTrue(30 <= page_types["article"] <= 90, page_types)
        self.assertEqual(stats["skipped_non_product_pages"], 200 - page_types["article"])
        self.assertEqual(builder.collect([("mem", lambda: None)], max_pages_per_warc=10_000,
                                         max_pages_per_host=5)[1]["labelled_pages"], 400)


if __name__ == "__main__":
    unittest.main()
