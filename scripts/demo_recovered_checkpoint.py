"""Run one visible Anarkali decision with a recovered training checkpoint.

Example (Colab GPU):
  python scripts/demo_recovered_checkpoint.py \
    --checkpoint /content/anarkali/artifacts/distill-run/distilled-150m/best.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


EXAMPLE_STATE = {
    "task": "Prepare a release of Anarkali.",
    "constraints": ["Do not expose credentials.", "Run the test suite before release."],
    "recent_actions": ["Recovered the saved model checkpoints from Kaggle."],
}
EXAMPLE_QUESTION = {
    "type": "choice",
    "instructions": "What should happen next?",
    "criteria": {
        "A": "Run the held-out evaluation and inspect calibration before release.",
        "B": "Publish immediately without evaluating the recovered checkpoint.",
        "C": "Delete the saved checkpoints and train again from scratch.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=Path, default=REPO / ".cache" / "huggingface")
    parser.add_argument("--state-json", type=Path, help="optional state JSON object")
    parser.add_argument("--question-json", type=Path, help="optional Jev-shaped question JSON object")
    args = parser.parse_args()

    import torch
    from anarkali.checkpoint import load_packed_checkpoint
    from anarkali.typed import format_answer, question_candidates

    state = json.loads(args.state_json.read_text(encoding="utf-8")) if args.state_json else EXAMPLE_STATE
    question = json.loads(args.question_json.read_text(encoding="utf-8")) if args.question_json else EXAMPLE_QUESTION
    kind, instructions, candidates = question_candidates(question)
    row = {"state": state, "question": instructions, "candidates": candidates,
           "target": [1.0 / len(candidates)] * len(candidates)}
    device = torch.device(args.device)
    model = load_packed_checkpoint(args.checkpoint, device=device, cache_dir=args.cache)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        logits = model.model(*model.batch([row], device)).logits[0, :len(candidates)].float()
    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
    print(json.dumps({"checkpoint": str(args.checkpoint), "model_id": model.raw["model_id"],
                      "answer": format_answer(kind, candidates, probabilities)}, indent=2))


if __name__ == "__main__":
    main()
