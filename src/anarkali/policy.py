"""Selection policy; entropy is not called probability of correctness."""

from dataclasses import dataclass
import math
from typing import Sequence

from .schema import ChoiceRequest


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class DecisionPolicy:
    min_probability: float = 0.70
    min_margin: float = 0.05
    min_answerability: float = 0.50
    tie_tolerance: float = 1e-7

    def __post_init__(self):
        for name in ("min_probability", "min_margin", "min_answerability", "tie_tolerance"):
            value = _number(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")

    def decide(self, request: ChoiceRequest, logits: Sequence[float], *,
               trained: bool = False, temperature: float = 1.0,
               calibration_fitted: bool = False, answerability: float | None = None) -> dict:
        if not isinstance(trained, bool) or not isinstance(calibration_fitted, bool):
            raise ValueError("trained and calibration_fitted must be booleans")
        if len(logits) != len(request.candidates):
            raise ValueError("one logit is required for each candidate")
        temperature = _number(temperature, "temperature")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        raw = [_number(z, "logit") for z in logits]
        # Subtract before dividing, avoiding overflow from an extremely small temperature.
        largest = max(raw)
        exponentials = [math.exp((z - largest) / temperature) for z in raw]
        total = math.fsum(exponentials)
        probabilities = [p / total for p in exponentials]
        ranked = sorted(range(len(raw)), key=lambda i: (-probabilities[i], request.candidates[i].id))
        best, second = ranked[:2]
        top = probabilities[best]
        margin = top - probabilities[second]
        entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
        if answerability is not None:
            answerability = _number(answerability, "answerability")
            if not 0 <= answerability <= 1:
                raise ValueError("answerability must be between zero and one")
        reason = None
        if not trained:
            reason = "untrained_model"
        elif margin <= self.tie_tolerance:
            reason = "ambiguous_tie"
        elif answerability is not None and answerability < self.min_answerability:
            reason = "no_supported_candidate"
        elif top < self.min_probability:
            reason = "low_probability"
        elif margin < self.min_margin:
            reason = "small_margin"
        return {
            "schema_version": 1,
            "choice": request.candidates[best].id if reason is None else None,
            "raw_choice": request.candidates[best].id,
            "abstain": reason is not None,
            "reason": reason,
            "probabilities": {c.id: p for c, p in zip(request.candidates, probabilities)},
            "top_probability": top,
            "margin": margin,
            "normalized_entropy": entropy / math.log(len(raw)),
            "answerability": answerability,
            "calibration_fitted": calibration_fitted,
            "trained": trained,
        }
