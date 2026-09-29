"""Load a packed Anarkali checkpoint (best.pt from train_anarkali.py) for scoring or as a teacher."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PackedCheckpoint:
    model: object
    tokenizer: object
    max_tokens: int
    shared_positions: bool
    raw: dict

    def batch(self, rows, device):
        """Collated inputs for this model (targets dropped), on device."""
        from .packed import collate_packed
        values = collate_packed(rows, self.tokenizer, self.max_tokens, self.shared_positions)
        return tuple(v.to(device) for v in values[:-1])


def load_packed_checkpoint(path: Path, device="cpu", cache_dir: Path | None = None) -> PackedCheckpoint:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .packed import PackedChoiceModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("run_config", {})
    if config.get("architecture") != "packed":
        raise ValueError(f"{path}: only packed checkpoints are supported")
    model_id, revision = checkpoint["model_id"], checkpoint["model_revision"]
    kwargs = {"revision": revision, **({"cache_dir": str(cache_dir)} if cache_dir else {})}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_id, **kwargs))
    head = checkpoint["head_config"]
    shared = head.get("shared_option_positions", False)
    model = PackedChoiceModel(encoder, hidden_dim=head["hidden_dim"], dropout=head["dropout"],
                              shared_option_positions=shared)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return PackedCheckpoint(model, tokenizer, config.get("packed_max_tokens", 512), shared, checkpoint)
