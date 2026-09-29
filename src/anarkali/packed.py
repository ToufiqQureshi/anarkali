"""Compact decision scorer: one joint encoder sequence per question.

Candidates share the state encoding. Absolute positions mean this architecture
is NOT permutation invariant; train-time shuffling and order audits are needed.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .neural import HeadOutput
from .packing import pack_row, shared_option_positions  # noqa: F401  (pack_row re-exported)


@dataclass(frozen=True)
class PackedHeadConfig:
    encoder_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1
    architecture: str = 'packed'
    shared_option_positions: bool = False


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


def _is_modernbert(encoder):
    return getattr(encoder.config, 'model_type', None) == 'modernbert'


def _builds_own_masks(encoder):
    """Older ModernBERT builds its global and sliding masks from a 2D mask in _update_attention_mask.

    Newer releases drop that method and accept one mask per layer type instead. Checked by
    behaviour, not version number: some transformers 5 builds (Kaggle's, 2026-09) still have it.
    """
    return callable(getattr(encoder, '_update_attention_mask', None))


def position_window_masks(encoder, attention_mask, position_ids, dtype):
    """Additive masks per ModernBERT layer type, with the local window measured in position IDs.

    ModernBERT's own sliding window uses token index, which would let local layers see option
    order even when options share positions. Measuring the window in position IDs instead makes
    every layer order-blind: tokens of different options at the same position are symmetric.
    """
    config = encoder.config
    half = getattr(config, 'sliding_window', None) or config.local_attention // 2
    valid = attention_mask.bool()
    both = valid[:, None, :, None] & valid[:, None, None, :]
    near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= half
    # -1e4, not finfo.min: it stays finite in fp16, so padded rows get a uniform softmax
    # instead of NaN, and exp(-1e4) is still exactly zero for every real key.
    blocked = -1e4

    def additive(allowed):
        return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, blocked)
    return {'full_attention': additive(both), 'sliding_attention': additive(both & near[:, None])}


class PackedChoiceModel(nn.Module):
    def __init__(self, encoder, hidden_dim=128, dropout=0.1, shared_option_positions=False):
        super().__init__()
        self.encoder = encoder
        self.head = PackedHead(PackedHeadConfig(encoder.config.hidden_size, hidden_dim, dropout,
                                                shared_option_positions=shared_option_positions))

    @property
    def shared_option_positions(self):
        return self.head.config.shared_option_positions

    def encode(self, input_ids, attention_mask, position_ids=None):
        """Token states; with shared positions, ModernBERT also gets position-measured local windows."""
        if position_ids is None:
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask.long()).last_hidden_state
        if _is_modernbert(self.encoder):
            dtype = self.encoder.embeddings.tok_embeddings.weight.dtype
            device_type = input_ids.device.type
            if torch.is_autocast_enabled(device_type):
                dtype = torch.get_autocast_dtype(device_type)
            masks = position_window_masks(self.encoder, attention_mask, position_ids, dtype)
            if not _builds_own_masks(self.encoder):
                return self.encoder(input_ids=input_ids, attention_mask=masks,
                                    position_ids=position_ids).last_hidden_state
            # Older ModernBERT would measure the local window in token index; hand it our masks instead.
            self.encoder._update_attention_mask = lambda *args, **kwargs: (masks['full_attention'],
                                                                           masks['sliding_attention'])
            try:
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask.long(),
                                    position_ids=position_ids).last_hidden_state
            finally:
                del self.encoder._update_attention_mask
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask.long(),
                            position_ids=position_ids).last_hidden_state

    def forward(self, input_ids, attention_mask, candidate_spans, position_ids=None):
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
        if self.shared_option_positions and position_ids is None:
            raise ValueError('this model was trained with shared option positions; pass position_ids')
        tokens = self.encode(input_ids, attention_mask, position_ids)
        logits = self.head(tokens, candidate_spans).masked_fill(~valid, float('-inf'))
        return HeadOutput(logits, logits.new_zeros(input_ids.shape[0]), valid)


def collate_packed(rows, tokenizer, max_tokens=512, shared_positions=False):
    """(ids, mask, spans, targets), or (ids, mask, spans, position_ids, targets) with shared positions."""
    if not rows:
        raise ValueError('empty batch')
    packed = [pack_row(r, tokenizer, max_tokens) for r in rows]
    width, k = max(len(x[0]) for x in packed), max(len(x[1]) for x in packed)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    positions = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    spans = torch.zeros((len(rows), k, width), dtype=torch.bool)
    targets = torch.zeros((len(rows), k), dtype=torch.float32)
    for i, (row, (tokens, ranges, _)) in enumerate(zip(rows, packed)):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        mask[i, :len(tokens)] = True
        if shared_positions:
            positions[i, :len(tokens)] = torch.tensor(shared_option_positions(len(tokens), ranges))
        for j, (start, end) in enumerate(ranges):
            spans[i, j, start:end] = True
        target = torch.tensor(row['target'], dtype=torch.float32)
        if (len(target) != len(ranges) or not torch.isfinite(target).all()
                or (target < 0).any() or not torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-5)):
            raise ValueError('invalid candidate targets')
        targets[i, :len(ranges)] = target
    return (ids, mask, spans, positions, targets) if shared_positions else (ids, mask, spans, targets)


def shuffle_candidates(rows, rng):
    result = []
    for row in rows:
        order = list(range(len(row['candidates'])))
        rng.shuffle(order)
        result.append(dict(row, candidates=[row['candidates'][i] for i in order],
                           target=[row['target'][i] for i in order]))
    return result
