"""Generate unlabelled decision cases for many business domains with a teacher LLM.

Breadth is what makes a typed-decision model general: the public benchmark covers four
workflows. This script asks an OpenAI-compatible model (vLLM, Groq, OpenRouter) to write
realistic situations for every domain in scripts/domains/catalog.json. Each call asks for a
batch of cases at one difficulty (clear-cut, borderline, a misleading surface cue, missing
information, conflicting signals), so the set is not only easy cases.

Cases hold only the situation, never an answer. Keys that look like answers are dropped,
oversized and duplicate cases are skipped, and every case becomes one row per question with a
uniform target and label_source "none". Label them afterwards with several teachers:

    python scripts/generate_domain_decisions.py --teacher gen=Qwen/Qwen3-30B-A3B-Instruct-2507@http://localhost:8000/v1 \\
        --cases-per-domain 200 --output artifacts/domain-decisions-v0
    python scripts/relabel_with_teachers.py --input artifacts/domain-decisions-v0 --output artifacts/domain-decisions-v0-labelled \\
        --splits train,development,calibration,test --no-original --teacher qwen=...@... --teacher mistral=...@...

Splits are by generated case and stratified by domain. Responses are cached, so a
rate-limited run resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anarkali.typed import question_candidates  # noqa: E402
from generate_coding_decisions import SPLIT_NAMES, digest_file, normalized, split_by_source_group  # noqa: E402
from relabel_with_teachers import Teacher, post_chat  # noqa: E402

CATALOG = Path(__file__).resolve().parent / "domains" / "catalog.json"
PROMPT_VERSION = "domain-cases-v0"
DIFFICULTIES = [
    "clear-cut, where a careful reader reaches the answer quickly",
    "borderline, where two options are both defensible and the details decide",
    "misleading, where a surface cue points one way but a detail points the other",
    "incomplete, where an important fact is missing and the reader must notice it",
    "conflicting, where two sources in the case disagree",
]
STYLES = ["terse system records", "a long free-text note", "mixed records and quoted messages",
          "a forwarded thread", "structured fields with one messy comment"]
LEAK_KEY = re.compile(r"answer|label|correct|expected|recommend|verdict|ground.?truth|outcome_decision", re.I)
MAX_CASE_CHARS = 2000


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    for name, domain in catalog["domains"].items():
        if not domain.get("description") or not domain.get("questions"):
            raise ValueError(f"domain {name} needs a description and questions")
        for qid, question in domain["questions"].items():
            question_candidates(question)  # raises on a malformed question
    return catalog


def describe_questions(domain: dict) -> str:
    lines = []
    for question in domain["questions"].values():
        kind, text, candidates = question_candidates(question)
        options = "; ".join(c["text"] for c in candidates)
        lines.append(f"- ({kind}) {question['instructions']} Options: {options}")
    return "\n".join(lines)


def build_prompt(name: str, domain: dict, batch: int, difficulty: str, style: str, hint: int) -> str:
    return (
        "You write realistic, varied cases for training a model that makes business decisions.\n"
        f"Domain: {name.replace('_', ' ')}. Each case is {domain['description']}\n\n"
        "A reader of each case will have to answer these questions:\n"
        f"{describe_questions(domain)}\n\n"
        f"Write {batch} different cases. Make them {difficulty}. Write them as {style}. "
        f"Use different names, organisations, amounts, dates and wording in every case (variety seed {hint}). "
        "Each case is a JSON object holding only what the reader would see: records, messages, numbers and any "
        "policy text needed to decide. Never include the answers, a recommendation, or fields that give them away. "
        f"Keep each case under {MAX_CASE_CHARS - 400} characters.\n\n"
        "Reply with only a JSON array of the case objects."
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
        self.path, self.entries = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
        path.parent.mkdir(parents=True, exist_ok=True)

    def get_or_call(self, key: str, call) -> str:
        if key not in self.entries:
            text = call()
            self.entries[key] = text
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        return self.entries[key]


def generate(catalog: dict, domains: list[str], teacher: Teacher, cache: ResponseCache, *, cases_per_domain: int,
             batch: int, seed: int, max_tokens: int) -> tuple[list[dict], dict]:
    rows, stats = [], {}
    for name in domains:
        domain = catalog["domains"][name]
        rng = random.Random(f"{seed}:{name}")
        seen, kept, calls, dropped = set(), 0, 0, 0
        for call_index in range(math.ceil(cases_per_domain / batch) * 2):  # extra calls cover dropped cases
            if kept >= cases_per_domain:
                break
            difficulty = DIFFICULTIES[call_index % len(DIFFICULTIES)]
            style, hint = rng.choice(STYLES), rng.randrange(10 ** 6)
            prompt = build_prompt(name, domain, batch, difficulty, style, hint)
            key = hashlib.sha256(json.dumps([PROMPT_VERSION, teacher.model, prompt]).encode()).hexdigest()
            text = cache.get_or_call(key, lambda: post_chat(teacher, {
                "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0.9})["choices"][0]["message"].get("content") or "")
            calls += 1
            for raw in parse_cases(text):
                case = scrub(raw)
                digest = case and case_hash(case)
                if case is None or digest in seen:
                    dropped += 1
                    continue
                seen.add(digest)
                kept += 1
                for qid, question in domain["questions"].items():
                    kind, qtext, candidates = question_candidates(question)
                    row = {"case_id": f"{name}_{digest}::{qid}", "source_group": f"{name}::{digest}",
                           "workflow": name, "state": case, "question": qtext, "candidates": candidates,
                           "target": normalized([1.0] * len(candidates)), "label_source": "none",
                           "generation": {"difficulty": difficulty.split(",")[0], "style": style}}
                    if kind != "choice":
                        row["question_type"] = kind
                    rows.append(row)
                if kept >= cases_per_domain:
                    break
        stats[name] = {"cases": kept, "calls": calls, "dropped": dropped}
        print(json.dumps({"domain": name, **stats[name]}), flush=True)
    return rows, stats


def write(output: Path, rows: list[dict], stats: dict, *, seed: int, teacher: Teacher, catalog: dict) -> dict:
    if not rows:
        raise SystemExit("no cases generated; check the teacher endpoint")
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
        "revision": hashlib.sha256(json.dumps([PROMPT_VERSION, catalog["version"], teacher.model, seed,
                                               sorted(stats.items())]).encode()).hexdigest()[:16],
        "config": "domains", "catalog_version": catalog["version"], "prompt_version": PROMPT_VERSION,
        "generator": {"model": teacher.model, "base_url": teacher.base_url},
        "domains": stats,
        "label_source": "none: uniform targets; label with relabel_with_teachers.py --no-original",
        "split_unit": "generated case, stratified by domain",
        "split_counts": counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, help="NAME=MODEL@BASE_URL of the generating model")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--domains", default=None, help="comma-separated subset of catalog domains")
    parser.add_argument("--cases-per-domain", type=int, default=100)
    parser.add_argument("--batch", type=int, default=5, help="cases requested per call")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--rpm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=REPO / "artifacts" / "domain-decisions-v0")
    args = parser.parse_args(argv)
    catalog = load_catalog(args.catalog)
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else list(catalog["domains"])
    unknown = set(domains) - set(catalog["domains"])
    if unknown:
        raise SystemExit(f"unknown domains: {', '.join(sorted(unknown))}")
    if args.cases_per_domain < 1 or args.batch < 1:
        raise SystemExit("--cases-per-domain and --batch must be positive")
    teacher = Teacher.parse(args.teacher, args.rpm, "sample")
    cache = ResponseCache(args.output / "generation-cache.jsonl")
    rows, stats = generate(catalog, domains, teacher, cache, cases_per_domain=args.cases_per_domain,
                           batch=args.batch, seed=args.seed, max_tokens=args.max_tokens)
    manifest = write(args.output, rows, stats, seed=args.seed, teacher=teacher, catalog=catalog)
    print(json.dumps({"split_counts": manifest["split_counts"]}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
