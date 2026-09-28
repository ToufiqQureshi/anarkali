"""Shared encoder adapter. No downloads occur unless from_pretrained is called."""

from dataclasses import dataclass

import torch
from torch import nn

from .neural import ChoiceHead, HeadConfig, masked_mean


@dataclass
class EncodedState:
    embeddings: torch.Tensor
    mask: torch.Tensor


class EncoderChoiceModel(nn.Module):
    def __init__(self, encoder: nn.Module, head_config: HeadConfig):
        super().__init__()
        self.encoder = encoder
        self.head = ChoiceHead(head_config)

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str,
                        local_files_only: bool = True, **head_options):
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("an explicit encoder revision is required")
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(model_id, revision=revision,
                                            local_files_only=local_files_only,
                                            trust_remote_code=False)
        return cls(encoder, HeadConfig(encoder_dim=encoder.config.hidden_size, **head_options))

    def _encode(self, ids, mask):
        if ids.ndim != 2 or mask.shape != ids.shape or mask.dtype != torch.bool or ids.dtype != torch.long:
            raise ValueError("encoder inputs require rank-2 int64 ids and a boolean mask")
        if not mask.any(-1).all().item():
            raise ValueError("encoder input cannot contain an all-padding row")
        return self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state

    def encode_state(self, ids, mask):
        return EncodedState(self._encode(ids, mask), mask)

    def score_encoded(self, state: EncodedState, question_ids, question_mask,
                      candidate_ids, candidate_token_mask):
        if candidate_ids.ndim != 3 or candidate_token_mask.shape != candidate_ids.shape:
            raise ValueError("candidate tokens must have shape [batch, candidates, tokens]")
        if candidate_token_mask.dtype != torch.bool or candidate_ids.dtype != torch.long:
            raise ValueError("candidate inputs require int64 ids and a boolean mask")
        b, k, length = candidate_ids.shape
        if state.embeddings.shape[0] != b or question_ids.shape[0] != b:
            raise ValueError("state, question and candidate batch sizes must match")
        valid = candidate_token_mask.any(-1)
        if not (valid.sum(-1) >= 2).all().item():
            raise ValueError("at least two candidates are required per row")
        question = self._encode(question_ids, question_mask)
        flat_ids = candidate_ids.reshape(b * k, length)
        flat_mask = candidate_token_mask.reshape(b * k, length)
        # Do not send padding-only candidates through the pretrained encoder.
        selected = valid.reshape(-1)
        encoded = self._encode(flat_ids[selected], flat_mask[selected])
        pooled = masked_mean(encoded, flat_mask[selected])
        candidates = pooled.new_zeros((b * k, pooled.shape[-1]))
        candidates[selected] = pooled
        return self.head(state.embeddings, state.mask, question, question_mask,
                         candidates.reshape(b, k, -1), valid)

    def forward(self, state_ids, state_mask, question_ids, question_mask,
                candidate_ids, candidate_token_mask):
        return self.score_encoded(self.encode_state(state_ids, state_mask), question_ids,
                                  question_mask, candidate_ids, candidate_token_mask)
