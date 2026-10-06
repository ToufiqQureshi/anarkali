"""Web-expert page state and labels (v1): the page's structure, and universal decisions any site has.

`page_state(html, url)` is what the model reads: an outline of the page in which menus, footers,
cookie banners and forms are one line each, content blocks keep their tag (h1, p, li, time, ...),
and the text is grouped under header / main / article / aside / comments when the page marks them.
Scripts, styles, <head>, classes, ids and attributes such as lang are never part of it, so the
label sources below (schema.org JSON-LD, OpenGraph, <html lang>, the HTTP status) never leak into
the input. Standard library only; importing this module does not load torch.

Decisions (all `choice` questions):
- page_type: ~20 kinds of page, from schema.org types, checked against og:type and the URL.
- page_status: ok / not_found / login_required / blocked_or_denied / server_error, from the HTTP status.
- language: from <html lang>, kept only where Common Crawl's own language detector agrees.
- published_date: which visible date is the publish date, from schema.org or article metadata.
"""
from __future__ import annotations

from html.parser import HTMLParser
import json
import random
import re
from typing import Any
from urllib.parse import urlsplit

MAX_OUTLINE_CHARS = 3000
MAX_TABLE_ROWS = 6
MAX_DATE_OPTIONS = 8
LANGUAGE_OPTIONS = 5

_SKIP = {"script", "style", "noscript", "svg", "template", "head", "iframe", "canvas", "object"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_SUMMARISED = {"nav", "footer", "cookie", "form"}               # one line each
_CONTEXTS = {"header", "main", "article", "aside", "comments"}   # shown as a marker when text moves into them
_ROLE_HINTS = [("cookie", ("cookie", "consent", "gdpr")), ("nav", ("navbar", "navigation", "menu", "breadcrumb")),
               ("footer", ("footer",)), ("header", ("masthead", "site-header", "topbar")),
               ("aside", ("sidebar", "widget")), ("comments", ("comment",))]
_TEXT_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "blockquote", "pre",
              "figcaption", "label", "button", "time", "caption", "summary", "legend"}


class PageParser(HTMLParser):
    """Reads a page into (context, kind, text) lines, plus the label sources (title, JSON-LD, meta, lang)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.stack: list[tuple[str, str | None]] = []
        self.regions: list[dict[str, Any]] = []
        self.lines: list[tuple[str, str, Any]] = []
        self.buf: list[str] = []
        self.buf_tag: str | None = None
        self.title, self._in_title = "", False
        self.jsonld: list[str] = []
        self._in_jsonld = False
        self.meta: dict[str, str] = {}
        self.html_lang = ""
        self._closed = False

    def _summarising(self):
        return next((r for r in reversed(self.regions) if r["name"] in _SUMMARISED), None)

    def _context(self) -> str:
        return next((r["name"] for r in reversed(self.regions) if r["name"] in _CONTEXTS), "")

    def _table(self):
        return next((r for r in reversed(self.regions) if r["name"] == "table"), None)

    @staticmethod
    def _region_of(tag: str, attrs: dict[str, str]) -> str | None:
        if tag in ("header", "nav", "main", "article", "aside", "footer", "form", "table"):
            return tag
        mapped = {"navigation": "nav", "banner": "header", "contentinfo": "footer", "main": "main",
                  "complementary": "aside", "search": "form"}.get((attrs.get("role") or "").lower())
        if mapped:
            return mapped
        if tag in ("div", "ul", "section"):
            hint = f"{attrs.get('id', '')} {attrs.get('class', '')}".lower()
            for region, words in _ROLE_HINTS:
                if any(w in hint for w in words):
                    return region
        return None

    def _flush(self):
        text = " ".join("".join(self.buf).split())
        kind = self.buf_tag
        self.buf, self.buf_tag = [], None
        if not text:
            return
        summary, table = self._summarising(), self._table()
        if summary:
            (summary["buttons"] if kind == "button" else summary["texts"]).append(text)
        elif table is not None and kind in (None, "cell"):
            table["cells"].append(text)
        else:
            self.lines.append((self._context(), kind or "text", text))

    def handle_starttag(self, tag, attrs_list):
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag == "html":
            self.html_lang = attrs.get("lang", "")
        elif tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            if key:
                self.meta[key.lower()] = attrs.get("content", "")
        elif tag == "title":
            self._in_title = True
        elif tag == "script" and "ld+json" in attrs.get("type", "").lower():
            self._in_jsonld = True
            self.jsonld.append("")
        if tag in _SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in _VOID:
            summary = self._summarising()
            if tag == "input" and summary is not None:
                kind = attrs.get("type", "text").lower()
                if kind in ("submit", "button") and attrs.get("value"):
                    summary["buttons"].append(attrs["value"][:30])
                elif kind not in ("hidden", "submit", "button"):
                    summary["fields"].append(kind if kind not in ("text", "") else (attrs.get("name") or "text")[:20])
            elif tag == "br":
                self.buf.append(" ")
            return
        region = None
        if not self._summarising():
            region = self._region_of(tag, attrs)
            if region:
                self._flush()
                self.regions.append({"name": region, "texts": [], "buttons": [], "fields": [], "links": 0,
                                     "cells": [], "rows": 0})
        if tag == "a":
            for r in self.regions:
                r["links"] += 1
        if tag in _TEXT_TAGS:
            self._flush()
            self.buf_tag = tag
        elif tag in ("td", "th"):
            self._flush()
            self.buf_tag = "cell"
        self.stack.append((tag, region))

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_jsonld = False
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag in _VOID:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] != tag:
                continue
            popped = self.stack[i:]
            del self.stack[i:]
            self._flush()
            for name, region in reversed(popped):
                if name == "tr":
                    self._end_row()
                if region:
                    self._close_region()
            break

    def _end_row(self):
        table = self._table()
        if table is not None and table["cells"]:
            table["rows"] += 1
            if table["rows"] <= MAX_TABLE_ROWS:
                self.lines.append((self._context(), "row", " | ".join(c[:40] for c in table["cells"][:8])))
            table["cells"] = []

    def _close_region(self):
        r = self.regions.pop()
        ctx = self._context()
        if r["name"] == "table":
            if r["cells"]:
                self.lines.append((ctx, "row", " | ".join(c[:40] for c in r["cells"][:8])))
            if r["rows"] > MAX_TABLE_ROWS:
                self.lines.append((ctx, "table", f"... {r['rows']} rows in all"))
        elif r["name"] in _SUMMARISED and (r["texts"] or r["fields"] or r["buttons"] or r["links"]):
            info = [f"links={r['links']}"] if r["links"] else []
            if r["fields"]:
                info.append("fields=" + ",".join(dict.fromkeys(r["fields"]))[:80])
            if r["buttons"]:
                info.append("buttons=" + ",".join(dict.fromkeys(b[:20] for b in r["buttons"]))[:80])
            self.lines.append((ctx, r["name"], (" ".join(info), " | ".join(t[:30] for t in r["texts"][:8]))))

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_jsonld and self.jsonld:
            self.jsonld[-1] += data
        if not self.skip:
            self.buf.append(data)

    def close(self):
        if self._closed:
            return
        self._closed = True
        super().close()
        self._flush()
        while self.regions:
            self._close_region()


def parse_page(html: str) -> PageParser:
    page = PageParser()
    try:
        page.feed(html)
    except Exception:  # broken markup: keep what was read
        pass
    page.close()
    return page


def outline(page: PageParser, limit: int = MAX_OUTLINE_CHARS) -> str:
    """The page as the model reads it."""
    out, size, shown = [], 0, ""
    for ctx, kind, text in page.lines:
        lines = []
        if ctx != shown:
            if ctx:
                lines.append(f"<{ctx}>")
            shown = ctx
        pad = "  " if ctx else ""
        if kind in _SUMMARISED:
            info, sample = text
            lines.append(f"{pad}<{kind}{' ' + info if info else ''}> {sample}".rstrip())
        else:
            lines.append(f"{pad}<{kind}> {text[:300]}" if kind != "text" else f"{pad}{text[:300]}")
        for line in lines:
            if size + len(line) > limit:
                return "\n".join(out)
            out.append(line)
            size += len(line) + 1
    return "\n".join(out)


def state_from(page: PageParser, url: str) -> dict[str, str]:
    return {"url": url[:300], "title": " ".join(page.title.split())[:300], "page": outline(page)}


def page_state(html: str, url: str = "") -> dict[str, str]:
    """The model input for one page: the same function runs at training and inference time."""
    return state_from(parse_page(html), url)

# ---------------------------------------------------------------------------------------------- labels


PAGE_TYPES = {
    "home": "home page of a site",
    "news_article": "news story",
    "blog_post": "blog post",
    "article": "other article, guide or report",
    "docs": "technical documentation or reference",
    "forum_thread": "forum or discussion thread",
    "qa": "question and answers",
    "faq": "frequently asked questions",
    "listing": "list or category of items, posts or results",
    "search_results": "search results",
    "product": "single product for sale",
    "local_business": "local business, place or venue",
    "event": "event",
    "job_post": "job opening",
    "recipe": "recipe",
    "how_to": "step-by-step instructions",
    "profile": "profile of a person or account",
    "about": "about page",
    "contact": "contact page",
    "video": "page built around a video",
    "course": "course or class",
    "software": "software or app",
}
PAGE_TYPE_QUESTION = "What kind of page is this?"
PAGE_STATUS = {
    "ok": "a real page with usable content",
    "not_found": "not found or deleted",
    "login_required": "login or sign-up wall",
    "blocked_or_denied": "access denied or blocked",
    "server_error": "server error",
}
PAGE_STATUS_QUESTION = "Is this a real, usable page? If not, what is wrong with it?"
LANGUAGE_QUESTION = "What language is this page written in?"
DATE_QUESTION = "Which date is when this page was published?"

_SCHEMA_PAGE_TYPES = [  # first match wins, most specific first
    ("job_post", {"JobPosting"}), ("recipe", {"Recipe"}),
    ("event", {"Event", "MusicEvent", "SportsEvent", "TheaterEvent", "BusinessEvent", "EducationEvent",
               "Festival", "ExhibitionEvent", "SocialEvent"}),
    ("faq", {"FAQPage"}), ("qa", {"QAPage"}), ("forum_thread", {"DiscussionForumPosting"}),
    ("how_to", {"HowTo"}), ("course", {"Course"}),
    ("product", {"Product", "ProductGroup", "IndividualProduct"}),
    ("search_results", {"SearchResultsPage"}), ("listing", {"CollectionPage", "ItemList", "OfferCatalog"}),
    ("news_article", {"NewsArticle", "ReportageNewsArticle", "AnalysisNewsArticle", "OpinionNewsArticle"}),
    ("blog_post", {"BlogPosting", "LiveBlogPosting"}),
    ("docs", {"TechArticle", "APIReference"}),
    ("article", {"Article", "ScholarlyArticle", "Report"}),
    ("profile", {"ProfilePage"}), ("about", {"AboutPage"}), ("contact", {"ContactPage"}),
    ("video", {"VideoObject"}),
    ("software", {"SoftwareApplication", "MobileApplication", "WebApplication"}),
    ("local_business", {"LocalBusiness", "Restaurant", "Hotel", "Store", "Dentist", "MedicalClinic",
                        "AutoRepair", "RealEstateAgent", "LegalService", "BeautySalon", "LodgingBusiness"}),
]
_OG_TYPES = {"article": {"news_article", "blog_post", "article", "docs", "how_to", "recipe"},
             "product": {"product"}, "profile": {"profile"}, "video.movie": {"video"},
             "video.other": {"video"}, "video.episode": {"video"}}
_URL_HINTS = [("search_results", r"/search\b|[?&](q|s|query)="), ("listing", r"/(tag|tags|category|categories)/"),
              ("contact", r"/contact"), ("about", r"/about"), ("job_post", r"/jobs?/|/careers?/"),
              ("forum_thread", r"/forum|/threads?/|/topic/"), ("docs", r"/docs?/|/documentation/")]
# a URL hint only conflicts with these labels; e.g. a /tag/ page can hold an article list but not a recipe
_URL_CONFLICTS = {"listing": {"product", "recipe", "event", "job_post", "news_article", "blog_post", "article"},
                  "search_results": {"product", "recipe", "event", "job_post", "news_article", "blog_post", "article"}}

HTTP_STATUS = {404: "not_found", 410: "not_found", 401: "login_required", 403: "blocked_or_denied",
               451: "blocked_or_denied", 500: "server_error", 502: "server_error", 503: "server_error"}

ISO3 = {"eng": "en", "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "por": "pt", "nld": "nl", "rus": "ru",
        "pol": "pl", "jpn": "ja", "zho": "zh", "kor": "ko", "ara": "ar", "tur": "tr", "ukr": "uk", "ces": "cs",
        "swe": "sv", "dan": "da", "nor": "no", "fin": "fi", "hun": "hu", "ron": "ro", "ell": "el", "heb": "he",
        "hin": "hi", "ind": "id", "vie": "vi", "tha": "th", "fas": "fa", "bul": "bg", "slk": "sk", "hrv": "hr",
        "srp": "sr", "lit": "lt", "lav": "lv", "est": "et", "slv": "sl", "cat": "ca", "msa": "ms", "ben": "bn"}
LANGUAGES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
             "nl": "Dutch", "ru": "Russian", "pl": "Polish", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
             "ar": "Arabic", "tr": "Turkish", "uk": "Ukrainian", "cs": "Czech", "sv": "Swedish", "da": "Danish",
             "no": "Norwegian", "fi": "Finnish", "hu": "Hungarian", "ro": "Romanian", "el": "Greek", "he": "Hebrew",
             "hi": "Hindi", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "fa": "Persian", "bg": "Bulgarian",
             "sk": "Slovak", "hr": "Croatian", "sr": "Serbian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
             "sl": "Slovenian", "ca": "Catalan", "ms": "Malay", "bn": "Bengali"}

_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_DATE_PATTERNS = [
    (re.compile(r"\b((?:19|20)\d\d)-(\d{1,2})-(\d{1,2})\b"), lambda g: [(int(g[0]), int(g[1]), int(g[2]))]),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./]((?:19|20)\d\d)\b"),
     lambda g: [(int(g[2]), int(g[1]), int(g[0])), (int(g[2]), int(g[0]), int(g[1]))]),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d\d)\b"),
     lambda g: [(int(g[2]), _MONTHS.get(g[1].lower(), 0), int(g[0]))]),
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d\d)\b"),
     lambda g: [(int(g[2]), _MONTHS.get(g[0].lower(), 0), int(g[1]))]),
    (re.compile(r"((?:19|20)\d\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"), lambda g: [(int(g[0]), int(g[1]), int(g[2]))]),
]


def schema_types(jsonld: list[str]) -> set[str]:
    found: set[str] = set()

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kinds = node.get("@type")
            for kind in kinds if isinstance(kinds, list) else [kinds]:
                if isinstance(kind, str):
                    found.add(kind.rsplit("/", 1)[-1])
            for key in ("@graph", "mainEntity", "mainEntityOfPage"):
                if key in node:
                    walk(node[key])
    for raw in jsonld:
        try:
            walk(json.loads(raw.strip()))
        except (ValueError, RecursionError):
            pass
    return found


def page_type_votes(page: PageParser, url: str) -> dict[str, Any]:
    """Each source's opinion of the page type. og:type gives a set of labels it is compatible with."""
    votes: dict[str, Any] = {}
    types = schema_types(page.jsonld)
    for label, names in _SCHEMA_PAGE_TYPES:
        if types & names:
            votes["schema.org"] = label
            break
    og = page.meta.get("og:type", "").lower()
    if og in _OG_TYPES:
        votes["og:type"] = sorted(_OG_TYPES[og])
    parts = urlsplit(url)
    if parts.path in ("", "/") and not parts.query:
        votes["url"] = "home"
    else:
        for label, pattern in _URL_HINTS:
            if re.search(pattern, url, re.I):
                votes["url"] = label
                break
    return votes


def page_type_label(votes: dict[str, Any]) -> tuple[str | None, str]:
    """(label, quality). quality: 'agreed' (2+ sources), 'single' or 'conflict' (label is None)."""
    label = votes.get("schema.org")
    if votes.get("url") == "home":
        if label and label not in ("listing", "local_business", "software", "about"):
            return None, "conflict"   # a home URL with an article/product/... schema is ambiguous
        return "home", "agreed" if label else "single"
    if not label:
        return None, "none"
    agreeing = 1
    og = votes.get("og:type")
    if og is not None:
        if label not in og:
            return None, "conflict"
        agreeing += 1
    hint = votes.get("url")
    if hint:
        if label in _URL_CONFLICTS.get(hint, set()) or (hint not in _URL_CONFLICTS and hint != label):
            return None, "conflict"
        agreeing += hint == label
    return label, "agreed" if agreeing >= 2 else "single"


def language_label(page: PageParser, detected_iso3: str) -> str | None:
    """The page language when <html lang> and Common Crawl's detector agree; None otherwise."""
    declared = page.html_lang.lower().replace("_", "-").split("-")[0]
    return declared if declared in LANGUAGES and ISO3.get(detected_iso3) == declared else None


def published_date(page: PageParser) -> tuple[tuple[int, int, int] | None, str | None]:
    for raw in page.jsonld:
        m = re.search(r'"datePublished"\s*:\s*"(\d{4})-(\d{2})-(\d{2})', raw)
        if m:
            return tuple(int(x) for x in m.groups()), "schema.org datePublished"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", page.meta.get("article:published_time", ""))
    if m:
        return tuple(int(x) for x in m.groups()), "meta article:published_time"
    return None, None


def date_candidates(page: PageParser) -> list[dict[str, Any]]:
    """Every distinct visible date, with a little context, in page order."""
    seen, out = set(), []
    for _, _, text in page.lines:
        if not isinstance(text, str):
            continue
        for pattern, parse in _DATE_PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.span()
                snippet = " ".join((text[max(0, start - 30):start] + "[" + m.group(0) + "]" + text[end:end + 15]).split())
                if snippet in seen:
                    continue
                seen.add(snippet)
                dates = [d for d in parse(m.groups()) if 1 <= d[1] <= 12 and 1 <= d[2] <= 31]
                out.append({"text": snippet, "dates": dates})
    return out

# ------------------------------------------------------------------------------------------------ rows


def _row(url, group, state, decision, question, options: list[tuple[str, str]], gold: set[str], votes, quality):
    share = 1.0 / len(gold)
    return {"case_id": f"{url}::{decision}", "source_group": group, "workflow": "web", "state": state,
            "question": question, "candidates": [{"id": i, "text": t} for i, t in options],
            "target": [share if i in gold else 0.0 for i, _ in options],
            "label_source": json.dumps(votes, ensure_ascii=False, sort_keys=True), "label_quality": quality}


def page_rows(html: str, url: str, group: str, *, http_status: int = 200, detected_language: str = "",
              rng: random.Random | None = None, min_chars: int = 200) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Labelled rows for one crawled page, and counts of labels that were dropped (and why)."""
    rng = rng or random.Random(url)
    page = parse_page(html)
    state = state_from(page, url)
    dropped: dict[str, int] = {}
    if len(state["page"]) < min_chars:
        return [], {"too_little_text": 1}
    rows = []
    if http_status != 200:
        label = HTTP_STATUS.get(http_status)
        if label:
            rows.append(_row(url, group, state, "page_status", PAGE_STATUS_QUESTION, list(PAGE_STATUS.items()),
                             {label}, {"http_status": http_status}, "single"))
        return rows, dropped
    votes = page_type_votes(page, url)
    label, quality = page_type_label(votes)
    if label:
        rows.append(_row(url, group, state, "page_type", PAGE_TYPE_QUESTION, list(PAGE_TYPES.items()),
                         {label}, votes, quality))
        if votes.get("schema.org"):  # a 200 page with a real type: a positive example of a usable page
            rows.append(_row(url, group, state, "page_status", PAGE_STATUS_QUESTION, list(PAGE_STATUS.items()),
                             {"ok"}, {"http_status": 200, "schema.org": votes["schema.org"]}, "agreed"))
    elif quality == "conflict":
        dropped["page_type_conflict"] = 1
    lang = language_label(page, detected_language)
    if lang:
        others = rng.sample([x for x in LANGUAGES if x != lang], LANGUAGE_OPTIONS - 1)
        options = [lang] + others
        rng.shuffle(options)
        rows.append(_row(url, group, state, "language", LANGUAGE_QUESTION, [(x, LANGUAGES[x]) for x in options],
                         {lang}, {"html lang": page.html_lang, "common crawl detector": detected_language}, "agreed"))
    elif page.html_lang and detected_language:
        dropped["language_disagreement"] = 1
    date, source = published_date(page)
    if date:
        found = date_candidates(page)
        gold = [i for i, c in enumerate(found) if date in c["dates"]]
        if found and gold and len(found) >= 2:
            chosen = found[:MAX_DATE_OPTIONS]
            if any(date in c["dates"] for c in chosen):
                rng.shuffle(chosen)
                options = [(f"d{i}", c["text"]) for i, c in enumerate(chosen)]
                rows.append(_row(url, group, state, "published_date", DATE_QUESTION, options,
                                 {f"d{i}" for i, c in enumerate(chosen) if date in c["dates"]},
                                 {source: "%04d-%02d-%02d" % date}, "single"))
    return rows, dropped
