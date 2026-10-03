"""Typed questions (choice / noul / score) as candidate choices, plus Jev-shaped answers.

Training and inference share this mapping, so a served question is packed exactly as its
training rows were. Importing this module does not load torch.
"""
from __future__ import annotations

import json
import math
from typing import Any

QUESTION_TYPES = ("choice", "noul", "score")
QUESTION_PREFIX = {"choice": "", "noul": "True or false: ", "score": "Pick the level that fits best: "}
DEFAULT_NOUL_CRITERIA = {"false": "The statement is false.", "true": "The statement is true."}
MAX_OPTIONS = 255


def validate_state(state: Any, *, max_chars: int = 50_000) -> Any:
    """Validate a Jev/Laya-style state payload is JSON-serializable and within limits."""
    if state is None:
        return state
    if isinstance(state, str):
        if len(state) > max_chars:
            raise ValueError(f"state longer than {max_chars} characters")
        return state
    try:
        encoded = json.dumps(state, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive validation path
        raise ValueError("state must be a JSON-serializable object or primitive") from exc
    if len(encoded) > max_chars:
        raise ValueError(f"state longer than {max_chars} characters")
    return state


def validate_question(question: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Jev/Laya-style question into a strict, validated schema."""
    kind, text, candidates = question_candidates(question)
    normalized = {"type": kind, "instructions": text, "criteria": {c["id"]: c["text"] for c in candidates}}
    if kind == "score":
        normalized["criteria"] = [c["text"] for c in candidates]
    return normalized


def question_candidates(question: dict[str, Any]) -> tuple[str, str, list[dict[str, str]]]:
    """Return (type, packed question text, candidates) for one Jev-style question object."""
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    kind = question.get("type", "choice")
    if not isinstance(kind, str):
        raise ValueError("question type must be a string")
    kind = kind.strip().lower()
    if kind not in QUESTION_TYPES:
        raise ValueError(f"unsupported question type {kind!r}; use choice, noul or score")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("question needs non-empty 'instructions'")
    criteria = question.get("criteria")
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must be a list of 2-10 level descriptions")
        if any(not isinstance(level, str) or not level.strip() for level in criteria):
            raise ValueError("score criteria must contain only non-empty strings")
        criteria = {str(level): text for level, text in enumerate(criteria)}
    elif kind == "noul":
        if criteria is None:
            criteria = DEFAULT_NOUL_CRITERIA
        if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
            raise ValueError("noul criteria must be an object with exactly 'false' and 'true'")
        criteria = {"false": criteria["false"], "true": criteria["true"]}
    elif not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError(f"choice criteria must be an object with 2-{MAX_OPTIONS} options")
    candidates = []
    for label, text in criteria.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"option {label!r} needs a non-empty description")
        candidates.append({"id": str(label), "text": text})
    return kind, QUESTION_PREFIX[kind] + instructions, candidates


def concentration(probabilities: list[float]) -> float:
    """Jev's confidence: (N*pmax - 1)/(N - 1); 0 is uniform, 1 is certain."""
    n = len(probabilities)
    return (n * max(probabilities) - 1) / (n - 1)


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def format_answer(kind: str, candidates: list[dict[str, str]], probabilities: list[float],
                  abstain_below: float | None = None) -> dict[str, Any]:
    """Build a Jev/Laya-compatible answer; `abstain` is Anarkali's extra field."""
    ids = [c["id"] for c in candidates]
    probs = {label: round(float(p), 4) for label, p in zip(ids, probabilities)}
    top = max(range(len(ids)), key=probabilities.__getitem__)
    confidence = round(concentration(probabilities), 4)
    if kind == "choice":
        answer = {"type": "choice", "choice": ids[top], "probabilities": probs, "confidence": confidence}
    elif kind == "score":
        answer = {"type": "score", "score": round(sum(i * p for i, p in enumerate(probabilities)), 4),
                  "legend": {c["id"]: c["text"] for c in candidates}, "probabilities": probs,
                  "confidence": confidence}
    else:
        p_true = float(probabilities[ids.index("true")])
        answer = {"type": "noul", "noul": round(p_true, 4), "confidence": round(max(p_true, 1 - p_true), 4)}
    if abstain_below is not None:
        answer["abstain"] = max(probabilities) < abstain_below
    return answer
