"""Compact decision scorer: one joint encoder sequence per question.

Candidates share the state encoding. Absolute positions mean this architecture
is NOT permutation invariant; train-time shuffling and order audits are needed.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .neural import HeadOutput
from .packing import pack_row  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class PackedHeadConfig:
    encoder_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1
    architecture: str = 'packed'


class PackedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scorer = nn.Sequential(nn.LayerNorm(config.encoder_dim),
                                    nn.Linear(config.encoder_dim, config.hidden_dim),
                                    nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 1))

    def forward(self, tokens, spans):
        weights = spans.to(tokens.dtype)
        pooled = torch.bmm(weights, tokens) / weights.sum(-1, keepdim=True).clamp_min(1)
        return self.scorer(pooled).squeeze(-1)


class PackedChoiceModel(nn.Module):
    def __init__(self, encoder, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.head = PackedHead(PackedHeadConfig(encoder.config.hidden_size, hidden_dim, dropout))

    def forward(self, input_ids, attention_mask, candidate_spans):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError('packed IDs must be int64 [batch, tokens]')
        if attention_mask.dtype != torch.bool or attention_mask.shape != input_ids.shape:
            raise ValueError('attention mask must be boolean and match IDs')
        if (candidate_spans.ndim != 3 or candidate_spans.dtype != torch.bool
                or candidate_spans.shape[0] != input_ids.shape[0]
                or candidate_spans.shape[2] != input_ids.shape[1]):
            raise ValueError('candidate spans must be boolean [batch, candidates, tokens]')
        valid = candidate_spans.any(-1)
        if input_ids.shape[0] == 0 or not (valid.sum(-1) >= 2).all().item():
            raise ValueError('at least two candidates required')
        if (candidate_spans & ~attention_mask[:, None]).any().item():
            raise ValueError('candidate span includes padding')
        tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask.long()).last_hidden_state
        logits = self.head(tokens, candidate_spans).masked_fill(~valid, float('-inf'))
        return HeadOutput(logits, logits.new_zeros(input_ids.shape[0]), valid)


def collate_packed(rows, tokenizer, max_tokens=512):
    if not rows:
        raise ValueError('empty batch')
    packed = [pack_row(r, tokenizer, max_tokens) for r in rows]
    width, k = max(len(x[0]) for x in packed), max(len(x[1]) for x in packed)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    spans = torch.zeros((len(rows), k, width), dtype=torch.bool)
    targets = torch.zeros((len(rows), k), dtype=torch.float32)
    for i, (row, (tokens, ranges, _)) in enumerate(zip(rows, packed)):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        mask[i, :len(tokens)] = True
        for j, (start, end) in enumerate(ranges):
            spans[i, j, start:end] = True
        target = torch.tensor(row['target'], dtype=torch.float32)
        if (len(target) != len(ranges) or not torch.isfinite(target).all()
                or (target < 0).any() or not torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-5)):
            raise ValueError('invalid candidate targets')
        targets[i, :len(ranges)] = target
    return ids, mask, spans, targets


def shuffle_candidates(rows, rng):
    result = []
    for row in rows:
        order = list(range(len(row['candidates'])))
        rng.shuffle(order)
        result.append(dict(row, candidates=[row['candidates'][i] for i in order],
                           target=[row['target'][i] for i in order]))
    return result
