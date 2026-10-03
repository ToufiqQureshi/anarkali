"""Training objectives and optimiser helpers for the packed decision model.

All of these are off by default; with the defaults `decision_losses` equals the soft
cross-entropy of `neural.training_loss`.

- Brier score: a strictly proper scoring rule, added to soft cross-entropy. Like the
  log score it is minimised only by honest probabilities, but it is bounded, so a few
  confident mistakes cannot dominate the gradient.
- Ranked probability score for ordinal ("score") questions: compares cumulative
  distributions, so predicting level 2 when the teacher says 3 costs less than
  predicting 0 (Epstein 1969; Murphy 1971).
- Permutation consistency: the same rows under two option orders, with a symmetric KL
  between the two predictions mapped back to one order. This is R-Drop (Liang et al.
  2021, arXiv:2106.14448) with the second view coming from a different option order
  rather than only a different dropout mask, aimed at the packed encoder's position bias.
- Layer-wise learning-rate decay (Sun et al. 2019, arXiv:1905.05583), warmup with a
  linear or cosine decay, and an exponential moving average of the weights (Polyak and
  Juditsky 1992; related to SWA, Izmailov et al. 2018, arXiv:1803.05407).
"""
from __future__ import annotations

import math
import random
import re

import torch
from torch import nn


def ordinal_index(rows: list[dict], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row, candidate positions sorted by level for score questions.

    Returns (index [B, K] long, is_ordinal [B] bool). Non-score rows and padded slots keep
    their own position; they are masked out of the ranked probability score.
    """
    index = torch.arange(width).repeat(len(rows), 1)
    ordinal = torch.zeros(len(rows), dtype=torch.bool)
    for i, row in enumerate(rows):
        if row.get("question_type") != "score":
            continue
        ids = [c["id"] for c in row["candidates"]]
        try:
            order = sorted(range(len(ids)), key=lambda p: int(ids[p]))
        except ValueError:
            continue
        index[i, :len(order)] = torch.tensor(order)
        ordinal[i] = True
    return index, ordinal


def row_weights(rows: list[dict], field: str | None, floor: float = 0.1) -> torch.Tensor | None:
    """Per-row loss weights from a numeric field such as relabel's teacher_agreement."""
    if not field:
        return None
    values = []
    for row in rows:
        value = row.get(field, 1.0)
        values.append(max(floor, float(value)) if isinstance(value, (int, float)) and math.isfinite(value) else 1.0)
    return torch.tensor(values, dtype=torch.float32)


def decision_losses(logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor, *,
                    brier_weight: float = 0.0, rps_weight: float = 0.0,
                    ordinal: tuple[torch.Tensor, torch.Tensor] | None = None,
                    weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Soft CE plus optional Brier and ranked probability score, averaged over rows."""
    logits = logits.float()
    targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0)
    probs = logp.exp() * mask
    per_row = -(targets * logp).sum(-1)
    parts = {"soft_ce": per_row}
    if brier_weight:
        brier = ((probs - targets) ** 2 * mask).sum(-1)
        parts["brier"] = brier
        per_row = per_row + brier_weight * brier
    if rps_weight and ordinal is not None:
        index, is_ordinal = (t.to(logits.device) for t in ordinal)
        valid = mask.gather(1, index).float()
        cdf_p = (probs.gather(1, index) * valid).cumsum(-1)
        cdf_t = (targets.gather(1, index) * valid).cumsum(-1)
        levels = (mask.sum(-1) - 1).clamp_min(1).float()
        rps = ((cdf_p - cdf_t) ** 2 * valid).sum(-1) / levels
        rps = rps * is_ordinal.float()
        parts["rps"] = rps
        per_row = per_row + rps_weight * rps
    if weights is not None:
        weights = weights.to(logits.device)
        loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
    else:
        loss = per_row.mean()
    return {"loss": loss, **{name: value.mean().detach() for name, value in parts.items()}}


def shuffle_with_index(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[list[int]]]:
    """Shuffle each row's candidates; index[i][p] is the original position of new position p."""
    shuffled, index = [], []
    for row in rows:
        order = list(range(len(row["candidates"])))
        rng.shuffle(order)
        shuffled.append(dict(row, candidates=[row["candidates"][i] for i in order],
                             target=[row["target"][i] for i in order]))
        index.append(order)
    return shuffled, index


def to_original_order(probs: torch.Tensor, index: list[list[int]]) -> torch.Tensor:
    """Scatter [B, K] probabilities from shuffled positions back to original positions."""
    width = probs.shape[1]
    full = torch.arange(width).repeat(len(index), 1)
    for i, order in enumerate(index):
        full[i, :len(order)] = torch.tensor(order)
    return torch.zeros_like(probs).scatter(1, full.to(probs.device), probs)


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over rows of (KL(p||q) + KL(q||p)) / 2 over valid candidates."""
    p, q = p.float().clamp_min(1e-8), q.float().clamp_min(1e-8)
    kl = ((p * (p.log() - q.log())) + (q * (q.log() - p.log()))) * mask
    return 0.5 * kl.sum(-1).mean()


LAYER = re.compile(r"\.(?:layers|layer)\.(\d+)\.")


def layerwise_groups(model: nn.Module, encoder_lr: float, head_lr: float, decay: float,
                     weight_decay: float = 0.01) -> list[dict]:
    """AdamW parameter groups: the top encoder layer gets encoder_lr, each lower one `decay` times less."""
    if not 0 < decay <= 1:
        raise ValueError("layer-wise decay must be in (0, 1]")
    named = list(model.encoder.named_parameters())
    depths = [int(m.group(1)) for n, _ in named if (m := LAYER.search("." + n))]
    top = max(depths) if depths else 0
    groups: dict[float, list] = {}
    for name, param in named:
        match = LAYER.search("." + name)
        if match:
            scale = decay ** (top - int(match.group(1)))
        elif "embed" in name:
            scale = decay ** (top + 1)
        else:
            scale = 1.0  # final norm and anything above the layers
        groups.setdefault(encoder_lr * scale, []).append(param)
    out = [{"params": params, "lr": lr, "weight_decay": weight_decay} for lr, params in sorted(groups.items())]
    out.append({"params": list(model.head.parameters()), "lr": head_lr, "weight_decay": weight_decay})
    return out


def lr_lambda(total_steps: int, warmup_ratio: float, schedule: str):
    """Multiplier for LambdaLR: linear warmup, then constant, linear or cosine decay to zero."""
    if schedule not in ("constant", "linear", "cosine"):
        raise ValueError("schedule must be constant, linear or cosine")
    warmup = int(total_steps * warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 1.0 - progress if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * progress))
    return factor


class EMA:
    """Exponential moving average of floating-point weights, swapped in for evaluation."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        state = model.state_dict()
        self.backup = {k: state[k].detach().clone() for k in self.shadow}
        for key, value in self.shadow.items():
            state[key].copy_(value.to(state[key].dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup is None:
            return
        state = model.state_dict()
        for key, value in self.backup.items():
            state[key].copy_(value)
        self.backup = None


def distillation_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor,
                    temperature: float = 2.0) -> torch.Tensor:
    """KL(teacher || student) at a temperature, times T^2 so its gradients match the hard loss's scale.

    Hinton, Vinyals and Dean 2015 (arXiv:1503.02531). Both inputs are raw scores over the same
    options in the same order; padded options are masked out.
    """
    t = float(temperature)
    s = torch.log_softmax(student_logits.float().masked_fill(~mask, -1e4) / t, dim=-1)
    q = torch.log_softmax(teacher_logits.float().masked_fill(~mask, -1e4) / t, dim=-1)
    kl = (q.exp() * (q - s)).masked_fill(~mask, 0).sum(-1)
    return kl.mean() * t * t


def option_vector_loss(student_vectors: torch.Tensor, teacher_vectors: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity between projected student and teacher option vectors, over real options.

    A light form of feature distillation (FitNets, Romero et al. 2015, arXiv:1412.6550; MiniLM,
    Wang et al. 2020, arXiv:2002.10957): the student learns what the teacher sees in each option,
    not only its final score. Cosine, not MSE, so the two models' scales need not match.
    """
    cos = torch.nn.functional.cosine_similarity(student_vectors.float(), teacher_vectors.float(), dim=-1)
    valid = mask.float()
    return ((1 - cos) * valid).sum() / valid.sum().clamp_min(1)
