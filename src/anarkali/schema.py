"""Candidate identity is separate from text fed to the encoder."""

from dataclasses import dataclass
from typing import Any, Mapping


def _text(value: Any, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.id, "candidate id", 128)
        _text(self.text, "candidate text", 4096)
        if self.id != self.id.strip():
            raise ValueError("candidate id must not have surrounding whitespace")


@dataclass(frozen=True)
class ChoiceRequest:
    state: str
    question: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        _text(self.state, "state", 200_000)
        _text(self.question, "question", 4096)
        if not isinstance(self.candidates, tuple) or not 2 <= len(self.candidates) <= 128:
            raise ValueError("candidates must be a tuple containing 2 to 128 candidates")
        if not all(isinstance(c, Candidate) for c in self.candidates):
            raise ValueError("each candidate must be a Candidate")
        if len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("candidate ids must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChoiceRequest":
        if not isinstance(data, Mapping) or set(data) != {"state", "question", "candidates"}:
            raise ValueError("request must contain exactly state, question and candidates")
        candidates = data["candidates"]
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        parsed = []
        for item in candidates:
            if not isinstance(item, Mapping) or set(item) != {"id", "text"}:
                raise ValueError("candidate must contain exactly id and text")
            parsed.append(Candidate(item["id"], item["text"]))
        return cls(data["state"], data["question"], tuple(parsed))

    def to_dict(self) -> dict:
        return {"state": self.state, "question": self.question,
                "candidates": [{"id": c.id, "text": c.text} for c in self.candidates]}
