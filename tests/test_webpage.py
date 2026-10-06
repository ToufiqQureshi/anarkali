"""Web-expert page state (structure outline) and universal labels, offline."""
import json
from pathlib import Path
import random
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anarkali.webpage import (PAGE_TYPES, page_rows, page_state, page_type_label,  # noqa: E402
                              page_type_votes, parse_page)

BODY = "<p>" + "The council met on Tuesday to discuss the new library opening hours and budget. " * 4 + "</p>"


def page(ld=None, og=None, lang="en", body=BODY, title="Library hours"):
    head = f"<title>{title}</title>"
    if ld:
        head += f'<script type="application/ld+json">{json.dumps(ld)}</script>'
    if og:
        head += f'<meta property="og:type" content="{og}">'
    return f"""<html lang="{lang}"><head>{head}<style>.x{{color:red}}</style></head><body>
<div class="site-navigation"><a href="/">Home</a> <a href="/news">News</a> <a href="/contact">Contact</a></div>
<main><h1>{title}</h1><time datetime="2026-03-14">14 March 2026</time>{body}
<p>Updated on 2026-04-02 after the vote.</p></main>
<form><input type="email" name="email"><input type="password" name="pw"><button>Sign in</button></form>
<footer><a href="/privacy">Privacy</a> <a href="/terms">Terms</a></footer>
<script>var secret = "NewsArticle";</script></body></html>"""


NEWS = {"@context": "https://schema.org", "@type": "NewsArticle", "headline": "Library hours",
        "datePublished": "2026-03-14T09:00:00Z"}


class PageStateTest(unittest.TestCase):
    def test_outline_keeps_structure_and_drops_label_sources(self):
        state = page_state(page(NEWS, og="article"), "https://town.example/news/library")
        text = state["page"]
        self.assertIn("<nav links=3> Home | News | Contact", text)
        self.assertIn("<main>", text)
        self.assertIn("<h1> Library hours", text)
        self.assertIn("<time> 14 March 2026", text)
        self.assertIn("<form fields=email,password buttons=Sign in>", text)
        self.assertIn("<footer links=2> Privacy | Terms", text)
        for leak in ("NewsArticle", "og:type", "datePublished", "lang=", "color:red", "secret"):
            self.assertNotIn(leak, json.dumps(state))

    def test_tables_are_rows_and_capped(self):
        rows = "".join(f"<tr><td>Item {i}</td><td>{i * 10}</td></tr>" for i in range(10))
        text = page_state(page(body=f"<table>{rows}</table>" + BODY), "https://a.example/t")["page"]
        self.assertIn("<row> Item 0 | 0", text)
        self.assertNotIn("Item 7", text)
        self.assertIn("... 10 rows in all", text)

    def test_broken_markup_does_not_raise(self):
        state = page_state("<html><body><div><p>Unclosed <b>bold " + "text " * 80, "https://a.example/")
        self.assertIn("Unclosed bold", state["page"])


class LabelTest(unittest.TestCase):
    def test_page_type_agreement_and_conflict(self):
        votes = page_type_votes(parse_page(page(NEWS, og="article")), "https://town.example/news/library")
        self.assertEqual(page_type_label(votes), ("news_article", "agreed"))
        votes = page_type_votes(parse_page(page({"@type": "Recipe"}, og="product")), "https://a.example/r")
        self.assertEqual(page_type_label(votes), (None, "conflict"))
        votes = page_type_votes(parse_page(page({"@type": "Recipe"})), "https://a.example/tag/soups/")
        self.assertEqual(page_type_label(votes)[0], None)
        votes = page_type_votes(parse_page(page()), "https://a.example/")
        self.assertEqual(page_type_label(votes), ("home", "single"))

    def test_graph_and_list_types(self):
        ld = {"@context": "https://schema.org", "@graph": [{"@type": "WebPage"}, {"@type": ["FAQPage"]}]}
        votes = page_type_votes(parse_page(page(ld)), "https://a.example/help")
        self.assertEqual(votes["schema.org"], "faq")

    def test_page_rows_for_a_news_page(self):
        rows, dropped = page_rows(page(NEWS, og="article"), "https://town.example/news/library", "town.example",
                                  detected_language="eng", rng=random.Random(0))
        by = {r["case_id"].rsplit("::", 1)[-1]: r for r in rows}
        self.assertEqual(set(by), {"page_type", "page_status", "language", "published_date"})
        gold = lambda r: [c["id"] for c, t in zip(r["candidates"], r["target"]) if t > 0]
        self.assertEqual(gold(by["page_type"]), ["news_article"])
        self.assertEqual(len(by["page_type"]["candidates"]), len(PAGE_TYPES))
        self.assertEqual(gold(by["page_status"]), ["ok"])
        self.assertEqual(gold(by["language"]), ["en"])
        dates = {c["id"]: c["text"] for c in by["published_date"]["candidates"]}
        self.assertEqual([dates[g] for g in gold(by["published_date"])], ["[14 March 2026]"])
        self.assertAlmostEqual(sum(by["published_date"]["target"]), 1.0)
        self.assertEqual(dropped, {})

    def test_language_needs_both_sources(self):
        rows, dropped = page_rows(page(NEWS, lang="pt"), "https://a.example/n", "a.example", detected_language="glg")
        self.assertNotIn("language", {r["case_id"].rsplit("::", 1)[-1] for r in rows})
        self.assertEqual(dropped.get("language_disagreement"), 1)

    def test_error_pages_get_only_a_status_row(self):
        html = page(title="Page not found", body="<p>" + "Sorry, this page does not exist. " * 10 + "</p>")
        rows, _ = page_rows(html, "https://a.example/old", "a.example", http_status=404, detected_language="eng")
        self.assertEqual([r["case_id"].rsplit("::", 1)[-1] for r in rows], ["page_status"])
        self.assertEqual(rows[0]["target"][[c["id"] for c in rows[0]["candidates"]].index("not_found")], 1.0)

    def test_thin_pages_are_skipped(self):
        rows, dropped = page_rows("<html><body><p>hi</p></body></html>", "https://a.example/", "a.example")
        self.assertEqual((rows, dropped), ([], {"too_little_text": 1}))


if __name__ == "__main__":
    unittest.main()
