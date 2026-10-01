"""Generate decision cases for many domains with an open LLM (DeepSeek, Qwen, ...), labelled by design.

Breadth is what makes a typed-decision model general. This script asks an OpenAI-compatible
model (DeepSeek's API, vLLM, Groq, OpenRouter) to write realistic situations for every domain in
scripts/domains/catalog.json (20 general domains) and scripts/domains/benchmark.json (the four
typed-decisions workflows, with one example state from their train split so the structure
matches). What makes the data train well, following the synthetic-data literature:

- Variety (Yu et al. 2023, "attributed" prompts; Li et al. 2023): every call draws an industry,
  region, organisation size, writer's tone, length and format, so cases do not collapse into one
  template.
- Hard cases (Swayamdipta et al. 2020, dataset cartography): each call has one difficulty:
  clear-cut, borderline, misleading, incomplete or conflicting.
- Balance: each call steers the answer to the domain's first choice question toward one option,
  cycling through all of them, so rare outcomes are not missing.
- Soft labels (Peterson et al. 2019): the generator also states, per question, the probability a
  careful expert panel would give each option. They are kept as teacher_targets["generator"], one
  teacher among several: label the rows with the 400M checkpoint too, and keep only rows where
  the teachers agree (combine_teachers.py --min-agreement).

Cases hold only the situation. Keys that look like answers are dropped, oversized and duplicate
cases are skipped, and every case becomes one row per question with a uniform target and
label_source "none" until a teacher labels it.

    DEEPSEEK_API_KEY=... python scripts/generate_domain_decisions.py \\
        --teacher deepseek=deepseek-chat@https://api.deepseek.com/v1 \\
        --catalog scripts/domains/benchmark.json --catalog scripts/domains/catalog.json \\
        --domains customer_service --rows-per-domain 20000 --workers 8 --output artifacts/synth-customer_service
    python scripts/generate_domain_decisions.py --print-prompt customer_service   # the exact prompt, no API call

Replies pasted from a chat window work too: save each reply as <domain>__<anything>.txt in a
folder and pass --import-replies FOLDER (no API key needed).

Splits are by generated case and stratified by domain. Responses are cached, so a
rate-limited run resumes where it stopped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import threading

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anarkali.typed import question_candidates  # noqa: E402
from generate_coding_decisions import SPLIT_NAMES, digest_file, normalized, split_by_source_group  # noqa: E402
from relabel_with_teachers import Teacher, post_chat  # noqa: E402

CATALOG = Path(__file__).resolve().parent / "domains" / "catalog.json"
BENCHMARK_CATALOG = Path(__file__).resolve().parent / "domains" / "benchmark.json"
WIDE_CATALOG = Path(__file__).resolve().parent / "domains" / "catalog_wide.json"  # 220 domains, see build_wide_catalog.py
PROMPT_VERSION = "domain-cases-v1"
DIFFICULTIES = [
    "clear-cut, where a careful reader reaches the answer quickly",
    "borderline, where two options are both defensible and the details decide",
    "misleading, where a surface cue points one way but a detail points the other",
    "incomplete, where an important fact is missing and the reader must notice it",
    "conflicting, where two sources in the case disagree",
]
STYLES = ["terse system records", "a long free-text note", "mixed records and quoted messages",
          "a forwarded thread", "structured fields with one messy comment"]
INDUSTRIES = ["retail", "SaaS software", "banking", "healthcare", "logistics", "telecom", "travel", "education",
              "manufacturing", "public sector", "gaming", "insurance", "food delivery", "real estate", "energy",
              "media", "automotive", "pharmaceuticals", "fintech", "e-commerce marketplace"]
REGIONS = ["India", "the United States", "the United Kingdom", "Germany", "France", "Brazil", "Mexico", "Nigeria",
           "Kenya", "South Africa", "the UAE", "Saudi Arabia", "Singapore", "Indonesia", "Japan", "Australia",
           "Canada", "Poland", "Turkey", "the Philippines"]
SIZES = ["a five-person startup", "a mid-size company", "a large enterprise", "a family business",
         "a government agency", "a fast-growing scale-up"]
TONES = ["calm and precise", "angry", "confused", "terse", "very polite", "written in non-native English",
         "sarcastic", "anxious and urgent", "rambling", "formal"]
LENGTHS = ["short (a few fields)", "medium", "long, with several records and messages"]
LEAK_KEY = re.compile(r"answer|label|correct|expected|recommend|verdict|ground.?truth|outcome_decision|intended",
                      re.I)
MAX_CASE_CHARS = 2400


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    for name, domain in catalog["domains"].items():
        if not domain.get("description") or not domain.get("questions"):
            raise ValueError(f"domain {name} needs a description and questions")
        for qid, question in domain["questions"].items():
            question_candidates(question)  # raises on a malformed question
    return catalog


def load_catalogs(paths: list[Path]) -> dict:
    """Several catalogs as one; a domain name may appear only once."""
    merged = {"version": "+".join(load_catalog(p)["version"] for p in paths), "domains": {}}
    for path in paths:
        for name, domain in load_catalog(path)["domains"].items():
            if name in merged["domains"]:
                raise ValueError(f"domain {name} appears in two catalogs")
            merged["domains"][name] = domain
    return merged


def describe_questions(domain: dict) -> str:
    """One line per question with its id and every option id, so the reply can name them."""
    lines = []
    for qid, question in domain["questions"].items():
        kind, _, candidates = question_candidates(question)
        options = "; ".join(f'"{c["id"]}" = {c["text"]}' for c in candidates)
        lines.append(f'- id "{qid}" ({kind}): {question["instructions"]} Options: {options}')
    return "\n".join(lines)


def steering(domain: dict, call_index: int) -> str:
    """Point this call at one option of the first choice question, cycling through all of them."""
    for qid, question in domain["questions"].items():
        kind, _, candidates = question_candidates(question)
        if kind == "choice":
            option = candidates[call_index % len(candidates)]
            return (f'Aim for about half of the cases to have "{option["id"]}" as the right answer to question '
                    f'"{qid}", and spread the rest across its other options.')
    return ""


def build_prompt(name: str, domain: dict, batch: int, difficulty: str, style: str, hint: int,
                 attributes: dict | None = None, steer: str = "") -> str:
    attributes = attributes or {}
    setting = ""
    if attributes:
        setting = (f"Set the cases in {attributes['industry']} in {attributes['region']}, at {attributes['size']}. "
                   f"People in the cases write in a tone that is {attributes['tone']}. "
                   f"Case length: {attributes['length']}. ")
    example = ""
    if domain.get("example_state"):
        example = ("Use the same structure as this example (the same field names and nesting), but invent "
                   "entirely new content, names, numbers and wording:\n"
                   f"{json.dumps(domain['example_state'], ensure_ascii=False)}\n\n")
    return (
        "You write realistic, varied cases for training a model that makes business decisions.\n"
        f"Domain: {name.replace('_', ' ')}. Each case is {domain['description']}\n\n"
        "A reader of each case will have to answer these questions:\n"
        f"{describe_questions(domain)}\n\n"
        f"{example}"
        f"Write {batch} different cases. Make them {difficulty}. "
        f"{'' if domain.get('example_state') else f'Write them as {style}. '}{setting}{steer} "
        f"Use different names, organisations, amounts, dates and wording in every case (variety seed {hint}). "
        "Numbers, dates and totals inside a case must be consistent with each other unless the case is meant to "
        "contain a discrepancy.\n\n"
        "Reply with only a JSON array. Each element is an object with two keys:\n"
        '- "case": the situation, holding only what the reader would see: records, messages, numbers and any '
        "policy text needed to decide. Never put the answers, a recommendation, or fields that give them away "
        f"inside the case. Keep each case under {MAX_CASE_CHARS - 400} characters.\n"
        '- "intended": for every question id above, the probability that a panel of careful expert reviewers '
        'would pick each option, as {"<question id>": {"<option id>": probability, ...}, ...}. Each question\'s '
        "probabilities sum to 1. Use 0.9 or more only when the case is unambiguous; borderline, incomplete and "
        "conflicting cases should spread their probability honestly."
    )


def parse_cases(text: str) -> list[dict]:
    """The JSON array in a reply, tolerant of code fences and prose around it."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [case for case in value if isinstance(case, dict)] if isinstance(value, list) else []


def split_item(item: dict) -> tuple[dict, dict | None]:
    """(case, intended) from one reply element; a bare case object has no intended labels."""
    if isinstance(item.get("case"), dict):
        intended = item.get("intended")
        return item["case"], intended if isinstance(intended, dict) else None
    return item, None


def intended_probs(intended: dict | None, qid: str, candidates: list[dict]) -> list[float] | None:
    """The generator's distribution over this question's options, or None if it is missing or malformed."""
    if not intended or not isinstance(intended.get(qid), dict):
        return None
    given = intended[qid]
    try:
        values = [float(given[c["id"]]) for c in candidates]
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        return None
    return normalized(values)


def scrub(case: dict) -> dict | None:
    """Drop answer-like keys at any depth; None if the case is empty or too long."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not LEAK_KEY.search(str(k))}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    cleaned = clean(case)
    size = len(json.dumps(cleaned, ensure_ascii=False))
    return cleaned if cleaned and 40 <= size <= MAX_CASE_CHARS else None


def case_hash(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


class ResponseCache:
    def __init__(self, path: Path):
        self.path, self.entries, self.lock = path, {}, threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        entry = json.loads(line)
                        self.entries[entry["key"]] = entry["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        path.parent.mkdir(parents=True, exist_ok=True)

    def get_or_call(self, key: str, call) -> str:
        if key in self.entries:
            return self.entries[key]
        text = call()
        with self.lock:
            self.entries[key] = text
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        return text


def call_plan(name: str, domain: dict, index: int, seed: int, batch: int, attributes: bool = True) -> dict:
    """Everything that varies per call, derived from (seed, domain, call index) so a resumed run matches."""
    rng = random.Random(f"{seed}:{name}:{index}")
    difficulty = DIFFICULTIES[index % len(DIFFICULTIES)]
    style, hint = rng.choice(STYLES), rng.randrange(10 ** 6)
    attrs = ({"industry": rng.choice(INDUSTRIES), "region": rng.choice(REGIONS), "size": rng.choice(SIZES),
              "tone": rng.choice(TONES), "length": rng.choice(LENGTHS)} if attributes else {})
    if attributes:
        attrs.update(domain.get("attributes", {}))  # a domain may pin some, e.g. GST cases are always in India
    steer = steering(domain, index // len(DIFFICULTIES))
    return {"difficulty": difficulty, "style": style, "attributes": attrs, "steer": steer,
            "prompt": build_prompt(name, domain, batch, difficulty, style, hint, attrs, steer)}


def rows_for_case(name: str, domain: dict, case: dict, intended: dict | None, generation: dict) -> list[dict]:
    digest = case_hash(case)
    rows = []
    for qid, question in domain["questions"].items():
        kind, qtext, candidates = question_candidates(question)
        row = {"case_id": f"{name}_{digest}::{qid}", "source_group": f"{name}::{digest}",
               "workflow": name, "state": case, "question": qtext, "candidates": candidates,
               "target": normalized([1.0] * len(candidates)), "label_source": "none", "generation": generation}
        probs = intended_probs(intended, qid, candidates)
        if probs is not None:
            row["teacher_targets"] = {"generator": probs}
        if kind != "choice":
            row["question_type"] = kind
        rows.append(row)
    return rows


def collect(name: str, domain: dict, replies: list[tuple[str, dict]], seen: set, limit: int) -> tuple[list, int, int]:
    """Rows from replies, in order, until `limit` cases; (rows, cases kept, cases dropped)."""
    rows, kept, dropped = [], 0, 0
    for text, generation in replies:
        for item in parse_cases(text):
            if kept >= limit:
                return rows, kept, dropped
            raw, intended = split_item(item)
            case = scrub(raw)
            digest = case and case_hash(case)
            if case is None or digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            kept += 1
            rows.extend(rows_for_case(name, domain, case, intended, generation))
    return rows, kept, dropped


def generate(catalog: dict, domains: list[str], teacher: Teacher, cache: ResponseCache, *, cases_per_domain: dict,
             batch: int, seed: int, max_tokens: int, workers: int = 1, attributes: bool = True) -> tuple[list, dict]:
    rows, stats = [], {}
    for name in domains:
        domain, target = catalog["domains"][name], cases_per_domain[name]
        seen, kept, calls, dropped, labelled, next_index = set(), 0, 0, 0, 0, 0
        max_calls = math.ceil(target / batch) * 3  # extra calls cover dropped and short replies
        while kept < target and next_index < max_calls:
            # Waves of at most 8 calls per worker: progress shows (and is cached) as it goes.
            size = min(max(1, math.ceil((target - kept) / batch * 1.15)), workers * 8)
            wave = range(next_index, min(max_calls, next_index + size))
            next_index = wave.stop
            plans = [call_plan(name, domain, i, seed, batch, attributes) for i in wave]

            def ask(plan):
                key = hashlib.sha256(json.dumps([PROMPT_VERSION, teacher.model, plan["prompt"]]).encode()).hexdigest()
                return cache.get_or_call(key, lambda: post_chat(teacher, {
                    "messages": [{"role": "user", "content": plan["prompt"]}], "max_tokens": max_tokens,
                    "temperature": 0.9})["choices"][0]["message"].get("content") or "")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                texts = list(pool.map(ask, plans))
            calls += len(plans)
            generation = [{"difficulty": p["difficulty"].split(",")[0], "style": p["style"], **p["attributes"]}
                          for p in plans]
            new_rows, got, lost = collect(name, domain, list(zip(texts, generation)), seen, target - kept)
            rows.extend(new_rows)
            kept, dropped = kept + got, dropped + lost
            labelled += sum("teacher_targets" in r for r in new_rows)
            print(json.dumps({"domain": name, "cases": kept, "target": target, "calls": calls}), flush=True)
        stats[name] = {"cases": kept, "calls": calls, "dropped": dropped, "rows_with_generator_labels": labelled}
        print(json.dumps({"domain": name, **stats[name]}), flush=True)
    return rows, stats


def import_replies(catalog: dict, folder: Path, domains: list[str] | None) -> tuple[list, dict]:
    """Replies saved from a chat window as <domain>__<anything>.txt (or .json/.md)."""
    rows, stats, seen = [], {}, {}
    files = sorted(p for p in folder.iterdir() if p.suffix in (".txt", ".json", ".md"))
    for path in files:
        name = path.name.split("__", 1)[0]
        if name not in catalog["domains"] or (domains and name not in domains):
            raise SystemExit(f"{path.name}: name it <domain>__<anything>{path.suffix} with a known domain")
        generation = {"difficulty": "unknown", "style": "imported", "file": path.name}
        new_rows, got, lost = collect(name, catalog["domains"][name],
                                      [(path.read_text(encoding="utf-8"), generation)], seen.setdefault(name, set()),
                                      10 ** 9)
        rows.extend(new_rows)
        entry = stats.setdefault(name, {"cases": 0, "calls": 0, "dropped": 0, "rows_with_generator_labels": 0})
        entry["cases"] += got
        entry["calls"] += 1
        entry["dropped"] += lost
        entry["rows_with_generator_labels"] += sum("teacher_targets" in r for r in new_rows)
    return rows, stats


def write(output: Path, rows: list[dict], stats: dict, *, seed: int, generator: dict, catalog: dict) -> dict:
    if not rows:
        raise SystemExit("no cases generated; check the teacher endpoint or the imported replies")
    splits = split_by_source_group(rows, seed)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in SPLIT_NAMES:
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in splits[name]:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[name] = {"decision_cases": len(splits[name]),
                        "source_groups": len({r["source_group"] for r in splits[name]}),
                        "sha256": digest_file(path)}
    manifest = {
        "dataset": "anarkali/domain-decisions",
        "revision": hashlib.sha256(json.dumps([PROMPT_VERSION, catalog["version"], generator, seed,
                                               sorted(stats.items())]).encode()).hexdigest()[:16],
        "config": "domains", "catalog_version": catalog["version"], "prompt_version": PROMPT_VERSION,
        "generator": generator,
        "domains": stats,
        "label_source": ("none: uniform targets. The generator's own distribution is in teacher_targets.generator; "
                         "add another teacher (label_with_checkpoint.py or relabel_with_teachers.py) and mix "
                         "them with combine_teachers.py --min-agreement"),
        "split_unit": "generated case, stratified by domain",
        "split_counts": counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", help="NAME=MODEL@BASE_URL of the generating model (key in $NAME_API_KEY)")
    parser.add_argument("--catalog", type=Path, action="append",
                        help=f"domain catalog (repeat); default {CATALOG.name}. The benchmark's four workflows "
                             f"are in {BENCHMARK_CATALOG.name}")
    parser.add_argument("--domains", default=None, help="comma-separated subset of catalog domains")
    parser.add_argument("--cases-per-domain", type=int, default=100)
    parser.add_argument("--rows-per-domain", type=int, default=0,
                        help="aim for this many rows per domain (cases x questions); overrides --cases-per-domain")
    parser.add_argument("--batch", type=int, default=5, help="cases requested per call")
    parser.add_argument("--workers", type=int, default=1, help="parallel requests")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--rpm", type=float, default=0.0)
    parser.add_argument("--extra-body", type=json.loads, default={},
                        help='JSON merged into every request, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\' for Qwen3')
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--no-attributes", action="store_true",
                        help="leave industry, region, size, tone and length out of the prompt")
    parser.add_argument("--print-prompt", metavar="DOMAIN", help="print one generation prompt and exit")
    parser.add_argument("--import-replies", type=Path, metavar="FOLDER",
                        help="build the set from saved chat replies named <domain>__*.txt instead of calling an API")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts" / "domain-decisions-v0")
    args = parser.parse_args(argv)
    catalog = load_catalogs(args.catalog or [CATALOG])
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else list(catalog["domains"])
    unknown = set(domains) - set(catalog["domains"])
    if unknown:
        raise SystemExit(f"unknown domains: {', '.join(sorted(unknown))}")
    if args.cases_per_domain < 1 or args.batch < 1 or args.workers < 1 or args.rows_per_domain < 0:
        raise SystemExit("--cases-per-domain, --batch and --workers must be positive")
    if args.print_prompt:
        if args.print_prompt not in catalog["domains"]:
            raise SystemExit(f"unknown domain {args.print_prompt}")
        plan = call_plan(args.print_prompt, catalog["domains"][args.print_prompt], 1, args.seed, args.batch,
                         not args.no_attributes)
        print(plan["prompt"])
        return {"prompt": plan["prompt"]}
    if args.import_replies:
        rows, stats = import_replies(catalog, args.import_replies, domains if args.domains else None)
        generator = {"imported_from": str(args.import_replies)}
    else:
        if not args.teacher:
            raise SystemExit("--teacher is required unless --print-prompt or --import-replies is given")
        teacher = Teacher.parse(args.teacher, args.rpm, "sample", args.extra_body)
        cases = {name: (math.ceil(args.rows_per_domain / len(catalog["domains"][name]["questions"]))
                        if args.rows_per_domain else args.cases_per_domain) for name in domains}
        cache = ResponseCache(args.output / "generation-cache.jsonl")
        rows, stats = generate(catalog, domains, teacher, cache, cases_per_domain=cases, batch=args.batch,
                               seed=args.seed, max_tokens=args.max_tokens, workers=args.workers,
                               attributes=not args.no_attributes)
        generator = {"model": teacher.model, "base_url": teacher.base_url}
    manifest = write(args.output, rows, stats, seed=args.seed, generator=generator, catalog=catalog)
    print(json.dumps({"split_counts": manifest["split_counts"]}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
