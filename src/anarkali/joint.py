"""Joint state/question/candidate control for architecture experiments."""
from dataclasses import dataclass
import json
import torch
from torch import nn
from .neural import HeadOutput, masked_mean


@dataclass(frozen=True)
class JointHeadConfig:
    encoder_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1
    architecture: str = 'joint'


class JointHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scorer = nn.Sequential(nn.LayerNorm(config.encoder_dim * 2),
                                    nn.Linear(config.encoder_dim * 2, config.hidden_dim),
                                    nn.GELU(), nn.Dropout(config.dropout),
                                    nn.Linear(config.hidden_dim, 1))

    def forward(self, tokens, mask):
        return self.scorer(torch.cat((tokens[:, 0], masked_mean(tokens, mask)), dim=-1)).squeeze(-1)


class JointChoiceModel(nn.Module):
    """Encode each complete context independently, then normalize candidate scores.

    This costs a state encoding per option. It is an experimental control, not an
    asserted replacement for the shared-state model or an efficient serving path.
    """
    def __init__(self, encoder, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.head = JointHead(JointHeadConfig(encoder.config.hidden_size, hidden_dim, dropout))

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 3 or input_ids.dtype != torch.long:
            raise ValueError('joint IDs must be int64 [batch, candidates, tokens]')
        if attention_mask.dtype != torch.bool or attention_mask.shape != input_ids.shape:
            raise ValueError('joint mask must be boolean and match IDs')
        b, k, n = input_ids.shape
        valid = attention_mask.any(-1)
        if b == 0 or not (valid.sum(-1) >= 2).all().item():
            raise ValueError('joint rows require at least two candidates')
        selected = valid.reshape(-1)
        ids = input_ids.reshape(b*k, n)[selected]
        mask = attention_mask.reshape(b*k, n)[selected]
        encoded = self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        scores = self.head(encoded, mask)
        logits = scores.new_full((b*k,), float('-inf'))
        logits[selected] = scores
        # No supported/unsupported supervision is supplied by this benchmark.
        answerability = scores.new_zeros((b,))
        return HeadOutput(logits.reshape(b, k), answerability, valid)


def collate_joint(rows, tokenizer, max_tokens):
    if max_tokens < 8:
        raise ValueError('joint token budget too small')
    first, second = [], []
    for row in rows:
        state = json.dumps(row['state'], ensure_ascii=False, sort_keys=True)
        for candidate in row['candidates']:
            first.append(f"Question: {row['question']}\nCandidate: {candidate['text']}")
            second.append(state)
    encoded = tokenizer(first, text_pair=second, padding=True, truncation='only_second',
                        max_length=max_tokens, return_tensors='pt')
    b, k, width = len(rows), max(len(r['candidates']) for r in rows), encoded['input_ids'].shape[-1]
    ids = torch.full((b, k, width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((b, k, width), dtype=torch.bool)
    targets = torch.zeros((b, k), dtype=torch.float32)
    offset = 0
    for i, row in enumerate(rows):
        count = len(row['candidates'])
        ids[i, :count] = encoded['input_ids'][offset:offset+count]
        mask[i, :count] = encoded['attention_mask'][offset:offset+count].bool()
        targets[i, :count] = torch.tensor(row['target'])
        offset += count
    return ids, mask, targets
