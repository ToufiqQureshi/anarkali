"""Original prototype: instruction fusion and an unordered candidate-set head.

The head consumes encoder representations, not candidate IDs or global candidate
positions. It is permutation equivariant in evaluation mode, up to numerical
roundoff. Random initialization does not produce useful semantic predictions.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class HeadConfig:
    encoder_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    fusion_layers: int = 2
    set_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("encoder_dim", "hidden_dim", "num_heads", "fusion_layers", "set_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "set_layers" else 1):
                raise ValueError(f"invalid {name}")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # masked_fill also prevents an ignored NaN from contaminating a pooled result.
    selected = values.masked_fill(~mask.unsqueeze(-1), 0)
    return selected.sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, queries, memory, memory_mask, query_mask):
        normalized = self.query_norm(queries)
        keys = self.memory_norm(memory)
        delta, _ = self.attention(normalized, keys, keys,
                                  key_padding_mask=~memory_mask, need_weights=False)
        result = queries + self.drop(delta)
        result = result + self.drop(self.ff(self.ff_norm(result)))
        return result.masked_fill(~query_mask.unsqueeze(-1), 0)


@dataclass
class HeadOutput:
    logits: torch.Tensor
    answerability_logits: torch.Tensor
    candidate_mask: torch.Tensor

    def probabilities(self, temperature: float = 1.0):
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not torch.isfinite(self.logits[self.candidate_mask]).all().item():
            raise ValueError("nonfinite candidate logits")
        z = self.logits
        if temperature < torch.finfo(z.dtype).tiny or temperature > torch.finfo(z.dtype).max:
            z = z.double()
        z = z - z.amax(dim=-1, keepdim=True)
        return torch.softmax(z / temperature, dim=-1).to(self.logits.dtype)


class ChoiceHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.project = nn.Linear(config.encoder_dim, d)
        self.memory_types = nn.Embedding(2, d)
        self.instruction_fusion = AttentionBlock(d, config.num_heads, config.dropout)
        self.candidate_fusion = nn.ModuleList(
            AttentionBlock(d, config.num_heads, config.dropout) for _ in range(config.fusion_layers))
        self.set_mixing = nn.ModuleList(
            AttentionBlock(d, config.num_heads, config.dropout) for _ in range(config.set_layers))
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.answerability = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, state, state_mask, question, question_mask, candidates, candidate_mask):
        tensors = ((state, state_mask, "state"), (question, question_mask, "question"),
                   (candidates, candidate_mask, "candidates"))
        batch = state.shape[0] if state.ndim else 0
        if batch == 0:
            raise ValueError("batch cannot be empty")
        for values, mask, name in tensors:
            if values.ndim != 3 or values.shape[0] != batch or values.shape[-1] != self.config.encoder_dim:
                raise ValueError(f"invalid {name} embedding shape")
            if mask.dtype != torch.bool or mask.shape != values.shape[:2]:
                raise ValueError(f"invalid {name} mask")
            if not mask.any(dim=-1).all().item():
                raise ValueError(f"every row needs at least one valid {name} item")
            if not torch.isfinite(values[mask]).all().item():
                raise ValueError(f"nonfinite {name} embeddings")
        if not (candidate_mask.sum(-1) >= 2).all().item():
            raise ValueError("every choice question needs at least two candidates")

        s = self.project(state.masked_fill(~state_mask.unsqueeze(-1), 0))
        q = self.project(question.masked_fill(~question_mask.unsqueeze(-1), 0))
        c = self.project(candidates.masked_fill(~candidate_mask.unsqueeze(-1), 0))
        q = self.instruction_fusion(q, s, state_mask, question_mask)
        memory = torch.cat((s + self.memory_types.weight[0], q + self.memory_types.weight[1]), dim=1)
        memory_mask = torch.cat((state_mask, question_mask), dim=1)
        for block in self.candidate_fusion:
            c = block(c, memory, memory_mask, candidate_mask)
        # Self-attention contains no option-position embeddings. Candidates communicate
        # as a set while retaining an output per candidate (equivariance, not invariance).
        for block in self.set_mixing:
            c = block(c, c, candidate_mask, candidate_mask)
        logits = self.scorer(c).squeeze(-1).masked_fill(~candidate_mask, float("-inf"))
        pooled = torch.cat((masked_mean(s, state_mask), masked_mean(q, question_mask),
                            masked_mean(c, candidate_mask)), dim=-1)
        supported = self.answerability(pooled).squeeze(-1)
        return HeadOutput(logits, supported, candidate_mask)


def training_loss(output: HeadOutput, targets: torch.Tensor, *,
                  answerable: torch.Tensor | None = None,
                  answerability_weight: float = 0.25) -> dict[str, torch.Tensor]:
    """Soft CE for labeled supported choices; BCE only with explicit support labels.

    Unsupported examples use an all-zero choice target and answerable=False.
    Soft targets are normalized per row, supporting annotator distributions.
    The answerability head is not trained unless explicit labels are supplied.
    """
    if isinstance(answerability_weight, bool) or not math.isfinite(answerability_weight) or answerability_weight < 0:
        raise ValueError("answerability_weight must be nonnegative and finite")
    if targets.shape != output.logits.shape or not torch.isfinite(targets).all().item() or (targets < 0).any().item():
        raise ValueError("invalid targets")
    if targets[~output.candidate_mask].count_nonzero().item():
        raise ValueError("target mass on a padded candidate")
    sums = targets.sum(-1)
    if not torch.isfinite(sums).all().item():
        raise ValueError("nonfinite target mass")
    if answerable is None:
        if (sums <= 0).any().item():
            raise ValueError("zero-mass targets need explicit answerability labels")
        active = torch.ones_like(sums, dtype=torch.bool)
    else:
        if answerable.dtype != torch.bool or answerable.shape != sums.shape:
            raise ValueError("answerable must be a boolean per row")
        if (sums[answerable] <= 0).any().item() or (sums[~answerable] != 0).any().item():
            raise ValueError("target mass must match answerability labels")
        active = answerable
    # Mask padded log-probabilities before multiplication: 0 * -inf would be NaN.
    logp = torch.log_softmax(output.logits, dim=-1).masked_fill(~output.candidate_mask, 0)
    normalized = targets / sums.clamp_min(1e-12).unsqueeze(-1)
    choice = -(normalized[active] * logp[active]).sum(-1).mean() if active.any().item() else output.answerability_logits.sum() * 0
    support = output.answerability_logits.sum() * 0
    if answerable is not None:
        support = nn.functional.binary_cross_entropy_with_logits(output.answerability_logits, answerable.float())
    return {"loss": choice + answerability_weight * support, "choice_loss": choice,
            "answerability_loss": support}
