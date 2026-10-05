"""Web page decisions: turn raw HTML into an Anarkali state, and read free labels from schema.org.

The same `page_state` runs at training and inference time, so a deployed model sees pages in
exactly the format it was trained on. Labels come from a page's own JSON-LD (schema.org), which
lives inside <script> tags and therefore never reaches the state the model reads.
Standard library only; importing this module does not load torch.
"""
from __future__ import annotations

from html.parser import HTMLParser
import json
import re
from typing import Any

MAX_TEXT_CHARS = 3000
MAX_PRICE_OPTIONS = 8

PAGE_TYPE_QUESTION = {
    "type": "choice",
    "instructions": "What kind of web page is this?",
    "criteria": {
        "product": "A single product page that sells one item",
        "listing": "A listing, category or search results page showing many items",
        "article": "An article, blog post or news story",
        "other": "Another kind of page, such as a home, about, contact, event or recipe page",
    },
}
PRICE_QUESTION_TEXT = "Which text shows this product's current selling price?"
IN_STOCK_QUESTION = {
    "type": "noul",
    "instructions": "The product on this page is in stock and can be bought now.",
    "criteria": {"false": "The product is out of stock or unavailable.",
                 "true": "The product is in stock and can be bought now."},
}

_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "head", "iframe", "canvas"}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "ul", "ol", "tr", "td", "th", "table", "h1", "h2",
               "h3", "h4", "h5", "h6", "header", "footer", "nav", "main", "aside", "br", "form", "label",
               "button", "option", "dd", "dt", "figcaption"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_LISTING_TYPES = {"ItemList", "CollectionPage", "SearchResultsPage", "OfferCatalog"}
_ARTICLE_TYPES = {"Article", "NewsArticle", "BlogPosting", "Report", "TechArticle", "ScholarlyArticle",
                  "LiveBlogPosting", "OpinionNewsArticle", "AnalysisNewsArticle"}
_IN_STOCK = {"InStock", "LimitedAvailability", "OnlineOnly", "InStoreOnly"}
_OUT_OF_STOCK = {"OutOfStock", "SoldOut", "Discontinued"}

_CURRENCY = r"(?:₹|Rs\.?|INR|\$|US\$|USD|€|EUR|£|GBP|¥|JPY|CNY|A\$|AUD|C\$|CAD|CHF|kr|SEK|NOK|DKK|zł|PLN|R\$|BRL|₩|KRW|₺|TRY|₽|RUB)"
_NUMBER = r"\d+(?:[.,]\d{2,3}|[\u00a0 ]\d{3})*"
PRICE_PATTERN = re.compile(rf"(?:{_CURRENCY}\s?(?:{_NUMBER})|(?:{_NUMBER})\s?{_CURRENCY})")


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.jsonld: list[str] = []
        self.title = ""
        self.links = 0
        self._current: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_jsonld = False
        self._jsonld_buf: list[str] = []

    def _flush(self):
        text = " ".join("".join(self._current).split())
        if text:
            self.blocks.append(text)
        self._current = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag == "script" and (dict(attrs).get("type") or "").strip().lower() == "application/ld+json":
            self._in_jsonld = True
            self._jsonld_buf = []
        if tag in _SKIP_TAGS and tag not in _VOID_TAGS:
            self._skip_depth += 1
            return
        if tag == "a":
            self.links += 1
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "script" and self._in_jsonld:
            self.jsonld.append("".join(self._jsonld_buf))
            self._in_jsonld = False
        if tag in _SKIP_TAGS and tag not in _VOID_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._in_jsonld:
            self._jsonld_buf.append(data)
        elif self._in_title:
            self.title += data
        elif not self._skip_depth:
            self._current.append(data)

    def close(self):
        super().close()
        self._flush()


def parse_page(html: str) -> _PageParser:
    parser = _PageParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep whatever was parsed before the error
        parser._flush()
    return parser


def _visible_text(blocks: list[str], limit: int) -> str:
    out, used = [], 0
    for block in blocks:
        if used + len(block) + 1 > limit:
            out.append(block[:max(0, limit - used)])
            break
        out.append(block)
        used += len(block) + 1
    return "\n".join(x for x in out if x)


def page_state(html: str, url: str = "", *, max_text_chars: int = MAX_TEXT_CHARS) -> dict[str, Any]:
    """The compact, model-ready view of one page: url, title, visible text and link count."""
    page = parse_page(html)
    return _state_from(page, url, max_text_chars)


def _state_from(page: _PageParser, url: str, max_text_chars: int) -> dict[str, Any]:
    return {"url": url[:300], "title": " ".join(page.title.split())[:300],
            "text": _visible_text(page.blocks, max_text_chars), "link_count": page.links}


# ---- schema.org labels -------------------------------------------------------------------------

def _walk(node, out):
    if isinstance(node, list):
        for item in node:
            _walk(item, out)
    elif isinstance(node, dict):
        out.append(node)
        for key, value in node.items():
            if key != "@context":
                _walk(value, out)


def _types(node: dict) -> set[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {str(v).rsplit("/", 1)[-1].rsplit(":", 1)[-1] for v in values if isinstance(v, str) and v}


def schema_nodes(jsonld_blocks: list[str]) -> list[dict]:
    nodes: list[dict] = []
    for block in jsonld_blocks:
        try:
            data = json.loads(block.strip().rstrip(";"))
        except (ValueError, RecursionError):
            continue
        _walk(data, nodes)
    return nodes


def page_type_label(nodes: list[dict]) -> str | None:
    """product / listing / article / other from the page's top-level schema types; None if untyped."""
    types: list[set[str]] = [_types(n) for n in nodes]
    flat = set().union(*types) if types else set()
    if not flat:
        return None
    products = sum("Product" in t or "ProductGroup" in t for t in types)
    if flat & _LISTING_TYPES or products >= 3:
        return "listing"
    if products:
        return "product"
    if flat & _ARTICLE_TYPES:
        return "article"
    return "other"


def _offers(nodes: list[dict]) -> list[dict]:
    offers = []
    for node in nodes:
        if "Product" not in _types(node):
            continue
        raw = node.get("offers")
        for offer in raw if isinstance(raw, list) else [raw]:
            if isinstance(offer, dict):
                offers.append(offer)
    return offers


def parse_amount(text: str) -> float | None:
    """'₹1,299.00' -> 1299.0, '1.299,50 €' -> 1299.5; None when no number is present."""
    match = re.search(_NUMBER, str(text))
    if not match:
        return None
    raw = re.sub(r"[\s ]", "", match.group(0))
    if "," in raw and "." in raw:
        decimal = "," if raw.rfind(",") > raw.rfind(".") else "."
        raw = raw.replace("." if decimal == "," else ",", "").replace(decimal, ".")
    elif "," in raw:
        head, _, tail = raw.rpartition(",")
        raw = head.replace(",", "") + ("." + tail if len(tail) <= 2 else tail)
    elif raw.count(".") > 1 or (raw.count(".") == 1 and len(raw.rpartition(".")[2]) == 3):
        raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def offer_price(nodes: list[dict]) -> float | None:
    for offer in _offers(nodes):
        for key in ("price", "lowPrice"):
            if key in offer and offer[key] not in (None, ""):
                value = parse_amount(offer[key])
                if value and value > 0:
                    return value
        spec = offer.get("priceSpecification")
        for item in spec if isinstance(spec, list) else [spec]:
            if isinstance(item, dict) and item.get("price") not in (None, ""):
                value = parse_amount(item["price"])
                if value and value > 0:
                    return value
    return None


def offer_in_stock(nodes: list[dict]) -> bool | None:
    seen = set()
    for offer in _offers(nodes):
        availability = offer.get("availability")
        if isinstance(availability, str) and availability:
            seen.add(availability.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1])
    if seen and seen <= _IN_STOCK:
        return True
    if seen and seen <= _OUT_OF_STOCK:
        return False
    return None  # missing or mixed availability is not a clean label


def price_candidates(blocks: list[str], *, context: int = 40) -> list[dict[str, Any]]:
    """Every distinct price-looking string in the visible text, with a little surrounding context."""
    seen, out = set(), []
    for block in blocks:
        for match in PRICE_PATTERN.finditer(block):
            start, end = match.span()
            snippet = block[max(0, start - context):start] + "[" + match.group(0) + "]" + block[end:end + 20]
            snippet = " ".join(snippet.split())
            if snippet in seen:
                continue
            seen.add(snippet)
            out.append({"text": snippet, "value": parse_amount(match.group(0))})
    return out


# ---- decision rows -----------------------------------------------------------------------------

def _row(case_id, group, state, question, labels, target, kind="choice"):
    row = {"case_id": case_id, "source_group": group, "workflow": "web", "state": state,
           "question": question, "candidates": labels, "target": target, "label_source": "schema.org"}
    if kind != "choice":
        row["question_type"] = kind
    return row


def page_decisions(html: str, url: str, group: str, *, rng=None) -> list[dict[str, Any]]:
    """Labelled decision rows for one page: page_type, and for products price_field and in_stock."""
    import random
    rng = rng or random.Random(url)
    page = parse_page(html)
    nodes = schema_nodes(page.jsonld)
    label = page_type_label(nodes)
    if label is None:
        return []
    state = _state_from(page, url, MAX_TEXT_CHARS)
    if len(state["text"]) < 200:
        return []
    from .typed import question_candidates
    rows = []
    _, question, options = question_candidates(PAGE_TYPE_QUESTION)
    rows.append(_row(f"{url}::page_type", group, state, question, options,
                     [1.0 if o["id"] == label else 0.0 for o in options]))
    if label != "product":
        return rows
    price = offer_price(nodes)
    if price:
        found = price_candidates(page.blocks)
        gold = [c for c in found if c["value"] is not None and abs(c["value"] - price) < 0.005]
        other = [c for c in found if c not in gold]
        if gold and other:
            rng.shuffle(other)
            chosen = gold[:3] + other[:MAX_PRICE_OPTIONS - min(3, len(gold))]
            rng.shuffle(chosen)
            options = [{"id": f"p{i}", "text": c["text"]} for i, c in enumerate(chosen)]
            share = 1.0 / sum(c in gold for c in chosen)
            rows.append(_row(f"{url}::price_field", group, state, PRICE_QUESTION_TEXT, options,
                             [share if c in gold else 0.0 for c in chosen]))
    stock = offer_in_stock(nodes)
    if stock is not None:
        _, question, options = question_candidates(IN_STOCK_QUESTION)
        rows.append(_row(f"{url}::in_stock", group, state, question, options,
                         [0.0, 1.0] if stock else [1.0, 0.0], kind="noul"))
    return rows
