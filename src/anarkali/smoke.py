"""Exercise neural mechanics with synthetic embeddings, not a semantic benchmark."""

import itertools
import platform


def run_smoke() -> dict:
    import torch
    from .neural import ChoiceHead, HeadConfig, training_loss

    torch.set_num_threads(1)
    torch.manual_seed(73)
    head = ChoiceHead(HeadConfig(encoder_dim=24, hidden_dim=32, num_heads=4, dropout=0))
    state = torch.randn(4, 7, 24)
    question = torch.randn(4, 3, 24)
    options = torch.randn(4, 4, 24)
    sm = torch.ones(4, 7, dtype=torch.bool)
    qm = torch.ones(4, 3, dtype=torch.bool)
    cm = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    args = (state, sm, question, qm)
    head.eval()
    worst = 0.0
    with torch.no_grad():
        reference = head(*args, options, cm)
        for order in itertools.permutations(range(4)):
            index = torch.tensor(order)
            permuted = head(*args, options[:, index], cm[:, index])
            delta = (permuted.probabilities() - reference.probabilities()[:, index]).abs().max().item()
            worst = max(worst, delta)
            worst = max(worst, (permuted.answerability_logits - reference.answerability_logits).abs().max().item())
    targets = torch.zeros(4, 4)
    targets[0, 0] = targets[1, 1] = targets[2, 0] = 1
    answerable = torch.tensor([True, True, True, False])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
    initial = training_loss(head(*args, options, cm), targets, answerable=answerable)["loss"].item()
    head.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = training_loss(head(*args, options, cm), targets, answerable=answerable)["loss"]
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite synthetic training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        optimizer.step()
    head.eval()
    final = training_loss(head(*args, options, cm), targets, answerable=answerable)["loss"].item()
    if worst > 2e-6 or final >= initial:
        raise RuntimeError("neural smoke checks failed")
    return {
        "scope": "synthetic neural mechanics only", "semantic_benchmark": False,
        "pretrained_encoder_loaded": False, "superiority_established": False,
        "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "permutations_checked": 24, "max_permutation_difference": worst,
        "synthetic_training_steps": 40, "initial_loss": initial, "final_loss": final,
        "passed": True,
    }
