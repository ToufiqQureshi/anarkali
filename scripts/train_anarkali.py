"""Fine-tune Anarkali's set-aware choice model on the pinned Typed Decisions train split."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO / ".cache" / "huggingface" / "hub"))
MODEL_ID = "answerdotai/ModernBERT-base"


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def collate(rows, tokenizer, torch, max_state, max_question, max_candidate):
    batch = len(rows)
    states = [json.dumps(r["state"], ensure_ascii=False, sort_keys=True) for r in rows]
    questions = [r["question"] for r in rows]
    state_enc = tokenizer(states, padding=True, truncation=True, max_length=max_state, return_tensors="pt")
    question_enc = tokenizer(questions, padding=True, truncation=True, max_length=max_question, return_tensors="pt")
    max_k = max(len(r["candidates"]) for r in rows)
    flat = [c["text"] for r in rows for c in r["candidates"]]
    encoded = tokenizer(flat, padding=True, truncation=True, max_length=max_candidate, return_tensors="pt")
    width = encoded["input_ids"].shape[-1]
    ids = torch.full((batch, max_k, width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch, max_k, width), dtype=torch.bool)
    targets = torch.zeros((batch, max_k), dtype=torch.float32)
    for i, row in enumerate(rows):
        k = len(row["candidates"])
        ids[i, :k] = encoded["input_ids"][sum(len(x["candidates"]) for x in rows[:i]):sum(len(x["candidates"]) for x in rows[:i+1])]
        mask[i, :k] = encoded["attention_mask"][sum(len(x["candidates"]) for x in rows[:i]):sum(len(x["candidates"]) for x in rows[:i+1])].bool()
        targets[i, :k] = torch.tensor(row["target"], dtype=torch.float32)
    return (state_enc["input_ids"], state_enc["attention_mask"].bool(),
            question_enc["input_ids"], question_enc["attention_mask"].bool(), ids, mask, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "artifacts" / "typed-decisions-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", help="Pin encoder revision across architecture experiments")
    parser.add_argument("--cache-dir", type=Path, default=REPO / '.cache' / 'huggingface')
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-state-tokens", type=int, default=384)
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--max-candidate-tokens", type=int, default=64)
    parser.add_argument("--architecture", choices=("set", "joint", "packed"), default="set")
    parser.add_argument("--joint-max-tokens", type=int, default=640)
    parser.add_argument("--packed-max-tokens", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overfit-source-cases", type=int, default=0,
                        help="Train on a small whole-source subset and report its fit; no final-test evaluation")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Explicit final evaluation, after architecture selection on development")
    parser.add_argument("--state-ablation", action="store_true", help="Compare blank/shuffled states on development")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--target-power", type=float, default=1.0,
                        help="Sharpen soft teacher targets for training only; 1.0 preserves them")
    parser.add_argument("--selection-metric", choices=("soft_ce", "accuracy_then_ce"), default="soft_ce")
    parser.add_argument("--no-amp", action="store_true",
                        help="Train in float32 on CUDA; DeBERTa-v3 can overflow under fp16 autocast")
    # V4 objectives and optimisation (packed architecture; all off by default, see anarkali/objectives.py)
    parser.add_argument("--brier-weight", type=float, default=0.0, help="add w * Brier score to soft CE")
    parser.add_argument("--rps-weight", type=float, default=0.0,
                        help="add w * ranked probability score on score (ordinal) questions")
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="second pass under another option order; add w * symmetric KL between them")
    parser.add_argument("--weight-field", default=None,
                        help="weight each row's loss by this numeric field, e.g. teacher_agreement")
    parser.add_argument("--llrd", type=float, default=1.0, help="layer-wise learning-rate decay per encoder layer")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--schedule", choices=("constant", "linear", "cosine"), default="constant")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="evaluate and save an exponential moving average of the weights (e.g. 0.999)")
    args = parser.parse_args()
    args.output = args.output or REPO / "artifacts" / f"anarkali-{args.architecture}-v2"
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.overfit_source_cases < 0 or args.log_every < 1:
        raise ValueError("invalid overfit/log settings")
    if args.overfit_source_cases and args.evaluate_test:
        raise ValueError("overfit diagnostics must not evaluate the final test")
    if any(not math.isfinite(x) or x <= 0 for x in (args.encoder_lr, args.head_lr)):
        raise ValueError("learning rates must be finite and positive")
    if not math.isfinite(args.target_power) or args.target_power < 1 or args.target_power > 4:
        raise ValueError("target-power must be finite and in [1, 4]")
    if min(args.brier_weight, args.rps_weight, args.consistency_weight) < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("loss weights must be nonnegative and warmup-ratio in [0, 1)")
    if not 0 < args.llrd <= 1 or not 0 <= args.ema_decay < 1:
        raise ValueError("llrd must be in (0, 1] and ema-decay in [0, 1)")
    use_objectives = bool(args.brier_weight or args.rps_weight or args.consistency_weight or args.weight_field)
    if use_objectives and args.architecture != "packed":
        raise ValueError("--brier/--rps/--consistency/--weight-field need --architecture packed")

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    from anarkali.encoder import EncoderChoiceModel
    from anarkali.neural import HeadConfig, training_loss
    from anarkali.joint import JointChoiceModel, collate_joint
    from anarkali.packed import PackedChoiceModel, collate_packed, shuffle_candidates
    from anarkali.diagnostics import development_controls, state_ablations
    from anarkali.objectives import (EMA, decision_losses, layerwise_groups, lr_lambda, ordinal_index,
                                     row_weights, shuffle_with_index, symmetric_kl, to_original_order)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; select a GPU runtime or explicitly use --device cpu")
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    except RuntimeError:
        pass

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train", "development") + (("test",) if args.evaluate_test else ()):
        path = args.data / f"{name}.jsonl"
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != manifest['split_counts'][name]['sha256']:
            raise ValueError(f"dataset manifest hash mismatch: {name}")
    train_rows, dev_rows = (load_rows(args.data / f"{name}.jsonl") for name in ("train", "development"))
    prior_report = development_controls(train_rows, dev_rows)
    if args.overfit_source_cases:
        groups = sorted({r["source_group"] for r in train_rows})
        random.Random(args.seed).shuffle(groups)
        selected = set(groups[:args.overfit_source_cases])
        train_rows = [r for r in train_rows if r["source_group"] in selected]
    info = HfApi().model_info(args.model, revision=args.revision)
    revision = info.sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=str(args.cache_dir))
    encoder = AutoModel.from_pretrained(args.model, revision=revision, cache_dir=str(args.cache_dir), trust_remote_code=False)
    if args.architecture == "packed":
        if args.packed_max_tokens > encoder.config.max_position_embeddings:
            raise ValueError("packed budget exceeds encoder position limit")
        model = PackedChoiceModel(encoder).to(device)
    elif args.architecture == "joint":
        model = JointChoiceModel(encoder).to(device)
    else:
        model = EncoderChoiceModel(encoder, HeadConfig(encoder_dim=encoder.config.hidden_size)).to(device)
    def make_batch(rows):
        values = (collate_packed(rows, tokenizer, args.packed_max_tokens) if args.architecture == "packed"
                  else collate_joint(rows, tokenizer, args.joint_max_tokens) if args.architecture == "joint"
                  else collate(rows, tokenizer, torch, args.max_state_tokens,
                               args.max_question_tokens, args.max_candidate_tokens))
        return tuple(v.to(device) for v in values)
    run_config = {**vars(args), "data": str(args.data), "output": str(args.output), "cache_dir": str(args.cache_dir),
                  "model_revision": revision, "torch": torch.__version__,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "candidate_order_augmentation": args.architecture == "packed",
                  "train_decisions": len(train_rows), "development_decisions": len(dev_rows),
                  "precision": "cuda_fp16_autocast" if use_amp else "float32"}
    if args.llrd < 1:
        optimizer = torch.optim.AdamW(layerwise_groups(model, args.encoder_lr, args.head_lr, args.llrd))
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ], weight_decay=0.01)
    steps_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(args.epochs * steps_per_epoch, args.warmup_ratio, args.schedule))
    ema = EMA(model, args.ema_decay) if args.ema_decay else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def objective(rows, out, targets):
        return decision_losses(out.logits, out.candidate_mask, targets, brier_weight=args.brier_weight,
                               rps_weight=args.rps_weight,
                               ordinal=ordinal_index(rows, targets.shape[1]) if args.rps_weight else None,
                               weights=row_weights(rows, args.weight_field))["loss"]

    def sharpen(targets):
        if args.target_power == 1:
            return targets
        targets = targets.pow(args.target_power)
        return targets / targets.sum(-1, keepdim=True)

    def evaluate(rows):
        model.eval()
        losses, correct = [], 0
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                part = rows[start:start + args.batch_size]
                values = make_batch(part)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(*values[:-1])
                    loss = training_loss(out, values[-1])["choice_loss"]
                losses.append(float(loss) * len(part))
                predicted = out.logits.argmax(-1)
                correct += int((predicted == values[-1].argmax(-1)).sum())
        return sum(losses) / max(1, len(rows)), correct / max(1, len(rows))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'prior-controls.json').write_text(json.dumps(prior_report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'architecture': args.architecture, 'development_prior': prior_report['controls']['all']}), flush=True)
    best_loss = float("inf")
    best_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        running, steps, skipped_steps = 0.0, 0, 0
        for start in range(0, len(order), args.batch_size):
            original = [train_rows[i] for i in order[start:start + args.batch_size]]
            part = original
            if args.architecture == "packed":
                shuffle_rng = random.Random(args.seed + epoch * 100000 + start)
                if use_objectives:
                    part, index_a = shuffle_with_index(original, shuffle_rng)
                else:
                    part = shuffle_candidates(original, shuffle_rng)
            values = make_batch(part)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(*values[:-1])
                targets = sharpen(values[-1])
                if not use_objectives:
                    loss = training_loss(out, targets)["loss"]
                else:
                    loss = objective(part, out, targets)
                    if args.consistency_weight:
                        part_b, index_b = shuffle_with_index(original, random.Random(shuffle_rng.random()))
                        values_b = make_batch(part_b)
                        out_b = model(*values_b[:-1])
                        loss = 0.5 * (loss + objective(part_b, out_b, sharpen(values_b[-1])))
                        probs_a = to_original_order(torch.softmax(out.logits.float(), -1), index_a)
                        probs_b = to_original_order(torch.softmax(out_b.logits.float(), -1), index_b)
                        valid = torch.arange(probs_a.shape[1], device=probs_a.device)[None, :] < \
                            torch.tensor([len(r["candidates"]) for r in original], device=probs_a.device)[:, None]
                        loss = loss + args.consistency_weight * symmetric_kl(probs_a, probs_b, valid)
            if not torch.isfinite(loss.detach()).item():
                raise RuntimeError("nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            skipped_steps += int(scaler.get_scale() < previous_scale)
            running += float(loss.detach()) * len(part)
            steps += 1
            if steps % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": steps,
                                  "steps_total": (len(order)+args.batch_size-1)//args.batch_size,
                                  "batch_loss": float(loss.detach()),
                                  "skipped_optimizer_steps": skipped_steps,
                                  "elapsed_seconds": time.perf_counter()-epoch_started}), flush=True)
        if ema is not None:
            ema.apply_to(model)
        dev_loss, dev_accuracy = evaluate(dev_rows)
        record = {"epoch": epoch, "train_loss": running / len(train_rows),
                  "development_soft_ce": dev_loss, "development_argmax_accuracy": dev_accuracy,
                  "epoch_seconds": time.perf_counter()-epoch_started,
                  "skipped_optimizer_steps": skipped_steps}
        if args.overfit_source_cases:
            fit_loss, fit_accuracy = evaluate(train_rows)
            record.update(overfit_soft_ce=fit_loss, overfit_argmax_accuracy=fit_accuracy)
        history.append(record)
        print(json.dumps(record), flush=True)
        selected = (dev_loss < best_loss if args.selection_metric == "soft_ce"
                    else dev_accuracy > best_accuracy or
                    (dev_accuracy == best_accuracy and dev_loss < best_loss))
        if selected:
            best_loss = dev_loss
            best_accuracy = dev_accuracy
            torch.save({"state_dict": model.state_dict(), "model_id": args.model,
                        "model_revision": revision, "head_config": model.head.config.__dict__,
                        "seed": args.seed, "epoch": epoch, "manifest": manifest,
                        "run_config": run_config}, args.output / "best.pt")
        if ema is not None:
            ema.restore(model)
    (args.output / "training.json").write_text(json.dumps({
        "model_id": args.model, "model_revision": revision, "device": str(device),
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "best_development_soft_ce": best_loss, "best_development_argmax_accuracy": best_accuracy,
        "history": history, "run_config": run_config,
    }, indent=2) + "\n", encoding="utf-8")

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if args.state_ablation:
        ablations = {}
        for condition, values in state_ablations(dev_rows, args.seed).items():
            ce, accuracy = evaluate(values)
            ablations[condition] = {'soft_ce': ce, 'argmax_accuracy': accuracy}
        (args.output / 'state-ablation.json').write_text(json.dumps(ablations, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'development_state_ablation': ablations}), flush=True)
    if not args.evaluate_test:
        print(json.dumps({"final_test_evaluated": False, "output": str(args.output),
                          "best_development_soft_ce": best_loss}), flush=True)
        return
    predictions = []
    test_rows = load_rows(args.data / "test.jsonl")
    with torch.inference_mode():
        for start in range(0, len(test_rows), args.batch_size):
            part = test_rows[start:start + args.batch_size]
            batch_started = time.perf_counter()
            values = make_batch(part)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(*values[:-1])
            probabilities = out.probabilities().float().cpu().tolist()
            latency_per_decision = (time.perf_counter() - batch_started) * 1000 / len(part)
            for row, probs in zip(part, probabilities):
                ids = [c["id"] for c in row["candidates"]]
                expected = ids[max(range(len(row["target"])), key=row["target"].__getitem__)]
                spec = {"type": "choice", "instructions": row["question"],
                        "criteria": {c["id"]: c["text"] for c in row["candidates"]}}
                pred_index = max(range(len(ids)), key=probs.__getitem__)
                prediction = {
                    "case_id": row["case_id"], "source_group": row["source_group"],
                    "request_sha256": digest({"state": row["state"], "question": spec}),
                    "expected": expected, "raw_choice": ids[pred_index], "choice": ids[pred_index],
                    "probabilities": dict(zip(ids, probs[:len(ids)])), "latency_ms": latency_per_decision,
                    "teacher_target": dict(zip(ids, row["target"])),
                }
                predictions.append(prediction)
    with (args.output / "anarkali-test.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"predictions": len(predictions), "output": str(args.output / "anarkali-test.jsonl"),
                      "checkpoint": str(args.output / "best.pt"), "test_decisions": len(test_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
