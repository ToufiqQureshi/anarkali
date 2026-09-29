"""Turn large public, openly licensed datasets into typed decisions: gold rows plus a teacher pool.

Two directories come out, both in the typed-decisions row format and split by source group:

    <output>/gold   human labels from the dataset itself (hard, or soft where annotators disagree,
                    as Civil Comments records the share of raters who marked a comment toxic).
                    Real outcomes, not a model's opinion: the calibration signal a teacher cannot give.
    <output>/pool   the same real texts with in-domain questions from scripts/domains/catalog.json,
                    unlabelled (uniform target, label_source "none"). Label them with teachers:
                    label_with_checkpoint.py (the 400M Anarkali), relabel_with_teachers.py
                    (--brio-teacher for colibri, or any OpenAI-compatible LLM), then
                    combine_teachers.py.

Only permissive licences are on by default (CC0, Apache-2.0, MIT, CC-BY). Share-alike sets
(SNLI, BoolQ, DBpedia, MultiNLI) need --allow-share-alike; non-commercial sets are never listed.
Every source is credited in <output>/CREDITS.md. Upstream test splits are never read, so the
datasets' own benchmarks stay clean. Our own benchmarks (typed-decisions, realworld-ci-v0) are
not touched either.

Rows stream from the Hub, so millions of rows need no more memory than the dedup set:

    python scripts/harvest_public_decisions.py --output artifacts/public-decisions-v0
    python scripts/harvest_public_decisions.py --output /tmp/smoke --max-per-source 200   # quick look
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Callable, Iterable, Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anarkali.typed import question_candidates  # noqa: E402

CATALOG = Path(__file__).resolve().parent / "domains" / "catalog.json"
SPLIT_NAMES = ("train", "development", "calibration", "test")
# Per mille. Large sets need little held out: 1% is still thousands of rows per source.
SPLIT_CUTS = (("train", 960), ("development", 980), ("calibration", 990), ("test", 1000))
MAX_TEXT_CHARS = 4000
VERSION = "public-decisions-v0"


def noul(instructions: str, false: str, true: str, p_true: float) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"false": false, "true": true},
            "target": {"false": 1.0 - p_true, "true": p_true}}


def choice(instructions: str, criteria: dict[str, str], gold: str) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria,
            "target": {key: float(key == gold) for key in criteria}}


def sample_options(names: list[str], gold: str, rng: random.Random, low: int, high: int) -> list[str]:
    """The gold label plus distractors, shuffled: a model has to read the options, not memorise a list."""
    others = [n for n in names if n != gold]
    picked = rng.sample(others, min(len(others), rng.randint(low, high) - 1)) + [gold]
    rng.shuffle(picked)
    return picked


def clip(text) -> str:
    text = str(text or "").strip()
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS] + " [...]"


def human(label: str) -> str:
    return label.replace("_", " ")


# ---- converters: one upstream record -> (group key, state, gold questions) ----------------------

CIVIL_ASPECTS = [
    ("toxicity", "This comment is toxic: rude, disrespectful or unreasonable enough to make someone leave the discussion.",
     "The comment is civil.", "The comment is toxic."),
    ("insult", "This comment insults a person or a group.", "No insult.", "It contains an insult."),
    ("threat", "This comment threatens violence or harm.", "No threat.", "It contains a threat."),
    ("identity_attack", "This comment attacks people for their identity (race, religion, gender, and so on).",
     "No identity attack.", "It attacks an identity group."),
    ("obscene", "This comment is obscene or uses sexually explicit language.", "Not obscene.", "Obscene or explicit."),
]


def civil_comments(record: dict, rng: random.Random):
    text = clip(record.get("text"))
    toxicity = float(record.get("toxicity") or 0.0)
    # 92% of comments are clean; keeping a third of the clearly clean ones balances the set
    # without touching the soft labels of the rest.
    if not text or (toxicity < 0.1 and rng.random() > 0.33):
        return None
    questions = [noul(CIVIL_ASPECTS[0][1], CIVIL_ASPECTS[0][2], CIVIL_ASPECTS[0][3], toxicity)]
    field, instructions, false, true = rng.choice(CIVIL_ASPECTS[1:])
    questions.append(noul(instructions, false, true, float(record.get(field) or 0.0)))
    return text, {"platform": "comments section of a news site", "comment": text}, questions, "gold-soft"


SENTIMENT_QUESTIONS = [
    ("How does the customer feel about the product?",
     {"negative": "Dissatisfied: the product disappointed them.", "positive": "Satisfied: they like the product."}),
    ("Is this review a complaint or praise?",
     {"negative": "A complaint about the product.", "positive": "Praise for the product."}),
]


def amazon_polarity(record: dict, rng: random.Random):
    title, content = clip(record.get("title")), clip(record.get("content"))
    if not content:
        return None
    gold = "positive" if int(record["label"]) == 1 else "negative"
    if rng.random() < 0.5:
        instructions, criteria = rng.choice(SENTIMENT_QUESTIONS)
        question = choice(instructions, criteria, gold)
    else:
        question = noul("The customer is happy with the purchase.", "Unhappy with the purchase.",
                        "Happy with the purchase.", float(gold == "positive"))
    return title + "\n" + content, {"review": {"title": title, "text": content}}, [question], "gold"


def clinc_oos(record: dict, rng: random.Random, names: list[str]):
    text = clip(record.get("text"))
    gold = names[int(record["intent"])]
    in_scope = [n for n in names if n != "oos"]
    if gold == "oos":
        options = rng.sample(in_scope, rng.randint(3, 8)) + ["oos"]
    else:
        options = sample_options(in_scope, gold, rng, 4, 9)
        if rng.random() < 0.5:
            options.append("oos")
    rng.shuffle(options)
    criteria = {o: ("None of these; the request is out of scope." if o == "oos" else f"The user wants: {human(o)}.")
                for o in options}
    return text, {"channel": "voice assistant", "user_message": text}, \
        [choice("What does the user want?", criteria, gold)], "gold"


def go_emotions(record: dict, rng: random.Random, names: list[str]):
    labels = record.get("labels") or []
    if len(labels) != 1:
        return None
    text = clip(record.get("text"))
    gold = names[int(labels[0])]
    options = sample_options(names, gold, rng, 4, 8)
    criteria = {o: f"Mostly {human(o)}." if o != "neutral" else "No clear emotion." for o in options}
    return text, {"platform": "Reddit", "comment": text}, \
        [choice("Which emotion does the writer express most?", criteria, gold)], "gold"


def commonsense_qa(record: dict, rng: random.Random):
    labels, texts = record["choices"]["label"], record["choices"]["text"]
    if record.get("answerKey") not in labels or len(set(texts)) != len(texts):
        return None
    criteria = {label: text for label, text in zip(labels, texts)}
    return record["question"], {"kind": "commonsense question", "concept": record.get("question_concept", "")}, \
        [choice(record["question"], criteria, record["answerKey"])], "gold"


def prompt_injections(record: dict, rng: random.Random):
    text = clip(record.get("text"))
    return text, {"channel": "chat message sent to an AI assistant", "message": text}, \
        [noul("The message tries to override the assistant's instructions (a prompt injection or jailbreak).",
              "An ordinary request.", "A prompt injection.", float(int(record["label"]) == 1))], "gold"


def nli(record: dict, rng: random.Random):
    if int(record["label"]) not in (0, 1, 2):
        return None
    gold = ("entails", "neutral", "contradicts")[int(record["label"])]
    criteria = {"entails": "The statement follows from the passage.",
                "neutral": "The passage neither supports nor contradicts it.",
                "contradicts": "The passage contradicts the statement."}
    state = {"passage": clip(record["premise"]), "statement": clip(record["hypothesis"])}
    return state["passage"] + "\n" + state["statement"], state, \
        [choice("How does the passage relate to the statement?", criteria, gold)], "gold"


def boolq(record: dict, rng: random.Random):
    question = str(record["question"]).strip().rstrip("?") + "?"
    return question + record["passage"], {"passage": clip(record["passage"])}, \
        [noul(question[0].upper() + question[1:], "No.", "Yes.", float(bool(record["answer"])))], "gold"


def dbpedia(record: dict, rng: random.Random, names: list[str]):
    gold = names[int(record["label"])]
    options = sample_options(names, gold, rng, 4, 7)
    state = {"title": clip(record.get("title")), "text": clip(record.get("content"))}
    return state["title"] + state["text"], state, \
        [choice("What kind of thing is this article about?", {o: human(o) for o in options}, gold)], "gold"


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str | None
    splits: tuple[str, ...]
    license: str
    credit: str
    convert: Callable
    cap: int
    pool_domain: str | None = None  # catalog domain whose questions go to the teacher pool
    share_alike: bool = False
    label_names: str | None = None  # feature holding ClassLabel names, passed to convert


SOURCES = [
    Source("civil_comments", "google/civil_comments", None, ("train", "validation"), "CC0-1.0",
           "Borkan et al. 2019, Nuanced Metrics for Measuring Unintended Bias (Jigsaw / Civil Comments)",
           civil_comments, 600_000, "content_moderation"),
    Source("amazon_polarity", "fancyzhx/amazon_polarity", "amazon_polarity", ("train",), "Apache-2.0",
           "Zhang, Zhao and LeCun 2015, Character-level Convolutional Networks for Text Classification",
           amazon_polarity, 600_000, "app_review_triage"),
    Source("clinc_oos", "clinc/clinc_oos", "plus", ("train", "validation"), "CC-BY-3.0",
           "Larson et al. 2019, An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
           clinc_oos, 100_000, label_names="intent"),
    Source("go_emotions", "google-research-datasets/go_emotions", "simplified", ("train", "validation"),
           "Apache-2.0", "Demszky et al. 2020, GoEmotions: A Dataset of Fine-Grained Emotions",
           go_emotions, 100_000, label_names="labels"),
    Source("commonsense_qa", "tau/commonsense_qa", None, ("train", "validation"), "MIT",
           "Talmor et al. 2019, CommonsenseQA", commonsense_qa, 100_000),
    Source("prompt_injections", "deepset/prompt-injections", None, ("train",), "Apache-2.0",
           "deepset, prompt-injections", prompt_injections, 100_000),
    Source("snli", "stanfordnlp/snli", "plain_text", ("train", "validation"), "CC-BY-SA-4.0",
           "Bowman et al. 2015, A large annotated corpus for learning natural language inference",
           nli, 300_000, share_alike=True),
    Source("multi_nli", "nyu-mll/multi_nli", None, ("train",), "CC-BY-3.0 / CC-BY-SA-3.0 / MIT / other (per genre)",
           "Williams, Nangia and Bowman 2018, MultiNLI", nli, 300_000, share_alike=True),
    Source("boolq", "google/boolq", None, ("train", "validation"), "CC-BY-SA-3.0",
           "Clark et al. 2019, BoolQ", boolq, 100_000, share_alike=True),
    Source("dbpedia_14", "fancyzhx/dbpedia_14", "dbpedia_14", ("train",), "CC-BY-SA-3.0",
           "Zhang, Zhao and LeCun 2015 (DBpedia ontology classes)", dbpedia, 200_000,
           share_alike=True, label_names="label"),
]
BY_NAME = {source.name: source for source in SOURCES}


# ---- plumbing -------------------------------------------------------------------------------------

def split_of(group: str, seed: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()[:8], 16) % 1000
    return next(name for name, cut in SPLIT_CUTS if bucket < cut)


def to_row(source: Source, group: str, index: int, state: dict, question: dict, label_source: str) -> dict:
    kind, text, candidates = question_candidates(question)
    ids = [c["id"] for c in candidates]
    raw = question.get("target")
    if raw is None:
        target = [1.0 / len(ids)] * len(ids)
    else:
        # score questions come as a list of levels; their ids are "0".."n-1"
        values = [float(raw[i] if isinstance(raw, dict) else raw[int(i)]) for i in ids]
        total = sum(values)
        target = [v / total for v in values] if total > 0 else [1.0 / len(ids)] * len(ids)
    row = {"case_id": f"{group}::{source.name}_{index}", "source_group": group, "workflow": f"public:{source.name}",
           "state": state, "question": text, "candidates": candidates, "target": target,
           "label_source": label_source, "source_dataset": source.dataset, "license": source.license}
    if kind != "choice":
        row["question_type"] = kind
    return row


def default_loader(source: Source, split: str, seed: int):
    """Stream one split from the Hub, shuffled in a buffer so a cap samples the whole set."""
    from datasets import load_dataset
    stream = load_dataset(source.dataset, source.config, split=split, streaming=True)
    names = None
    if source.label_names:
        feature = stream.features[source.label_names]
        names = list(getattr(feature, "feature", feature).names)
    return stream.shuffle(seed=seed, buffer_size=10_000), names


def harvest(sources: list[Source], output: Path, *, seed: int, caps: dict[str, int], pool_rate: float,
            loader=default_loader, log=print) -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["domains"]
    rng = random.Random(seed)
    streams, seen = {}, set()
    counts = {kind: {name: 0 for name in SPLIT_NAMES} for kind in ("gold", "pool")}
    per_source = {}
    for kind in ("gold", "pool"):
        (output / kind).mkdir(parents=True, exist_ok=True)
        streams[kind] = {name: (output / kind / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
                         for name in SPLIT_NAMES}
    try:
        for source in sources:
            cap, kept, index = caps.get(source.name, source.cap), 0, 0
            pool_questions = catalog[source.pool_domain]["questions"] if source.pool_domain else {}
            stats = {"records_read": 0, "states_kept": 0, "gold_rows": 0, "pool_rows": 0, "duplicates": 0}
            for split in source.splits:
                if kept >= cap:
                    break
                records, names = loader(source, split, seed)
                for record in records:
                    if kept >= cap:
                        break
                    stats["records_read"] += 1
                    converted = source.convert(record, rng, names) if source.label_names else source.convert(record, rng)
                    if converted is None:
                        continue
                    key_text, state, questions, label_source = converted
                    key = hashlib.sha256(" ".join(str(key_text).lower().split()).encode("utf-8")).hexdigest()[:16]
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                    group = f"public:{source.name}::{key}"
                    split_name = split_of(group, seed)
                    for question in questions:
                        streams["gold"][split_name].write(json.dumps(
                            to_row(source, group, index, state, question, label_source),
                            ensure_ascii=False, separators=(",", ":")) + "\n")
                        index += 1
                        stats["gold_rows"] += 1
                        counts["gold"][split_name] += 1
                    if pool_questions and rng.random() < pool_rate:
                        for qid, question in pool_questions.items():
                            row = to_row(source, group, index, state, question, "none")
                            row["case_id"] = f"{group}::pool_{qid}"
                            streams["pool"][split_name].write(json.dumps(row, ensure_ascii=False,
                                                                         separators=(",", ":")) + "\n")
                            index += 1
                            stats["pool_rows"] += 1
                            counts["pool"][split_name] += 1
                    kept += 1
                    stats["states_kept"] = kept
                    if kept % 50_000 == 0:
                        log(json.dumps({"source": source.name, **stats}), flush=True)
            per_source[source.name] = stats
            log(json.dumps({"source": source.name, "done": True, **stats}), flush=True)
    finally:
        for kind in streams.values():
            for stream in kind.values():
                stream.close()
    return {"counts": counts, "per_source": per_source}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifests(output: Path, sources: list[Source], result: dict, seed: int):
    for kind in ("gold", "pool"):
        split_counts = {}
        for name in SPLIT_NAMES:
            path = output / kind / f"{name}.jsonl"
            groups = set()
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    groups.add(json.loads(line)["source_group"])
            split_counts[name] = {"decision_cases": result["counts"][kind][name], "source_groups": len(groups),
                                  "sha256": digest_file(path)}
        revision = hashlib.sha256(json.dumps([VERSION, kind, seed, split_counts], sort_keys=True).encode()).hexdigest()[:16]
        manifest = {
            "dataset": f"{VERSION}-{kind}", "revision": revision, "seed": seed,
            "label_source": ("human labels from each dataset: hard, or the share of raters (civil_comments)"
                             if kind == "gold" else "none: uniform targets; label with teachers before training"),
            "split_unit": "source groups (one deduplicated upstream text), hashed 96/2/1/1; gold and pool share groups",
            "sources": [{"name": s.name, "dataset": s.dataset, "license": s.license, "credit": s.credit,
                         **result["per_source"].get(s.name, {})} for s in sources],
            "split_counts": split_counts,
        }
        (output / kind / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Credits", "", f"Rows in this directory were derived from these datasets ({VERSION}).", "",
             "| Source | Dataset | Licence | Credit |", "|---|---|---|---|"]
    lines += [f"| {s.name} | https://huggingface.co/datasets/{s.dataset} | {s.license} | {s.credit} |" for s in sources]
    (output / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_caps(items: Iterable[str]) -> dict[str, int]:
    caps = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in BY_NAME or not value.isdigit():
            raise SystemExit(f"--cap must be SOURCE=N for a known source, got {item!r}")
        caps[name] = int(value)
    return caps


def main(argv: list[str] | None = None, loader=default_loader):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", default="",
                        help="comma-separated source names; default every permissive source "
                             f"({', '.join(s.name for s in SOURCES if not s.share_alike)})")
    parser.add_argument("--allow-share-alike", action="store_true",
                        help=f"also use CC-BY-SA sets ({', '.join(s.name for s in SOURCES if s.share_alike)})")
    parser.add_argument("--max-per-source", type=int, default=0, help="cap every source at N states (0 = own caps)")
    parser.add_argument("--cap", action="append", default=[], help="SOURCE=N, overriding one source's cap (repeat)")
    parser.add_argument("--pool-rate", type=float, default=0.25,
                        help="share of states that also get the catalog's questions for teachers")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise SystemExit(f"unknown sources: {', '.join(unknown)}; known: {', '.join(BY_NAME)}")
    chosen = [BY_NAME[n] for n in names] if names else [s for s in SOURCES if not s.share_alike or args.allow_share_alike]
    blocked = [s.name for s in chosen if s.share_alike and not args.allow_share_alike]
    if blocked:
        raise SystemExit(f"{', '.join(blocked)} are share-alike; pass --allow-share-alike to use them")
    if not 0 <= args.pool_rate <= 1 or args.max_per_source < 0:
        raise SystemExit("--pool-rate must be in [0, 1] and --max-per-source nonnegative")
    caps = {s.name: args.max_per_source for s in chosen} if args.max_per_source else {}
    caps.update(parse_caps(args.cap))
    result = harvest(chosen, args.output, seed=args.seed, caps=caps, pool_rate=args.pool_rate, loader=loader)
    write_manifests(args.output, chosen, result, args.seed)
    print(json.dumps({"output": str(args.output), **result["counts"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
