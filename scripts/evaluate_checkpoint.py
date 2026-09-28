"""Evaluate a packed checkpoint on the final test split with a temperature fitted on the calibration split."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO/'scripts'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def softmax(logits, temperature):
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def soft_ce(logits, target, temperature):
    probs = softmax(logits, temperature)
    return -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, probs))


def fit_temperature(rows):
    grid = [round(0.5 + 0.01 * i, 2) for i in range(451)]
    return min(grid, key=lambda t: sum(soft_ce(r['logits'], r['target'], t) for r in rows))


def metrics(rows, temperature):
    out = defaultdict(lambda: {'n': 0, 'correct': 0, 'ce': 0.0, 'brier': 0.0, 'conf': [], 'hit': []})
    for r in rows:
        probs = softmax(r['logits'], temperature)
        pred = max(range(len(probs)), key=probs.__getitem__)
        gold = max(range(len(r['target'])), key=r['target'].__getitem__)
        for key in ('all', r['workflow'], 'type:' + r.get('question_type', 'choice')):
            s = out[key]
            s['n'] += 1
            s['correct'] += pred == gold
            s['ce'] += soft_ce(r['logits'], r['target'], temperature)
            s['brier'] += sum((p - (i == gold)) ** 2 for i, p in enumerate(probs))
            s['conf'].append(probs[pred])
            s['hit'].append(pred == gold)
    result = {}
    for key, s in out.items():
        bins = [[] for _ in range(15)]
        for c, h in zip(s['conf'], s['hit']):
            bins[min(14, int(c * 15))].append((c, h))
        ece = sum(len(b) / s['n'] * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
                  for b in bins if b)
        result[key] = {'decisions': s['n'], 'accuracy': s['correct'] / s['n'], 'soft_ce': s['ce'] / s['n'],
                       'brier': s['brier'] / s['n'], 'ece_15_bins': ece}
    return result


def selective(rows, temperature, thresholds=(0.4, 0.5, 0.6, 0.7)):
    """Coverage and accuracy when decisions below a top-probability threshold abstain."""
    out = {}
    for threshold in thresholds:
        accepted = correct = 0
        for r in rows:
            probs = softmax(r['logits'], temperature)
            pred = max(range(len(probs)), key=probs.__getitem__)
            if probs[pred] >= threshold:
                accepted += 1
                correct += pred == max(range(len(r['target'])), key=r['target'].__getitem__)
        out[str(threshold)] = {'coverage': accepted / len(rows),
                               'accepted_accuracy': correct / accepted if accepted else None}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=REPO/'artifacts'/'typed-decisions-v1')
    p.add_argument('--cache', type=Path, default=REPO/'.cache'/'huggingface')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--latency-samples', type=int, default=100)
    p.add_argument('--threads', type=int, default=0, help='torch CPU threads; 0 keeps the default')
    p.add_argument('--output', type=Path)
    p.add_argument('--out-of-domain', '--skip-revision-check', dest='out_of_domain', action='store_true',
                   help='evaluate on a dataset the checkpoint was not trained on (skips the revision match)')
    args = p.parse_args()
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from anarkali.packed import PackedChoiceModel, collate_packed
    from train_anarkali import load_rows
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint.get('run_config', {})
    if config.get('architecture') != 'packed':
        raise ValueError('only packed checkpoints are supported')
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if not args.out_of_domain and checkpoint['manifest'].get('revision') != manifest.get('revision'):
        raise ValueError('checkpoint and dataset revisions differ')
    for name in ('calibration', 'test'):
        if hashlib.sha256((args.data/f'{name}.jsonl').read_bytes()).hexdigest() != manifest['split_counts'][name]['sha256']:
            raise ValueError(f'dataset hash mismatch: {name}')
    model_id, revision = checkpoint['model_id'], checkpoint['model_revision']
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=str(args.cache))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_id, revision=revision, cache_dir=str(args.cache)))
    head = checkpoint['head_config']
    model = PackedChoiceModel(encoder, hidden_dim=head['hidden_dim'], dropout=head['dropout'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    max_tokens = config.get('packed_max_tokens', 512)

    def score(rows):
        scored = []
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                part = rows[start:start+args.batch_size]
                values = tuple(v.to(device) for v in collate_packed(part, tokenizer, max_tokens))
                logits = model(*values[:-1]).logits.float().cpu().tolist()
                for row, row_logits in zip(part, logits):
                    n = len(row['candidates'])
                    scored.append(dict(row, logits=row_logits[:n]))
        return scored

    calibration = score(load_rows(args.data/'calibration.jsonl'))
    temperature = fit_temperature(calibration)
    test_rows = load_rows(args.data/'test.jsonl')
    test = score(test_rows)
    reversed_rows = [dict(r, candidates=r['candidates'][::-1], target=r['target'][::-1]) for r in test_rows]
    flips = 0
    for original, rev in zip(test, score(reversed_rows)):
        ids = [c['id'] for c in original['candidates']]
        rev_ids = [c['id'] for c in rev['candidates']]
        flips += (ids[max(range(len(ids)), key=original['logits'].__getitem__)]
                  != rev_ids[max(range(len(rev_ids)), key=rev['logits'].__getitem__)])

    latencies = []
    with torch.inference_mode():
        sample = test[:args.latency_samples]
        for row in sample[:10]:
            model(*tuple(v.to(device) for v in collate_packed([row], tokenizer, max_tokens))[:-1])
        for row in sample:
            started = time.perf_counter()
            values = tuple(v.to(device) for v in collate_packed([row], tokenizer, max_tokens))
            model(*values[:-1]).logits.cpu()  # copy back so GPU timing includes the kernels
            latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()

    output = args.output or args.checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    with (output/'anarkali-test.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in test:
            ids = [c['id'] for c in row['candidates']]
            probs = softmax(row['logits'], temperature)
            spec = {'type': 'choice', 'instructions': row['question'],
                    'criteria': {c['id']: c['text'] for c in row['candidates']}}
            choice = ids[max(range(len(ids)), key=probs.__getitem__)]
            stream.write(json.dumps({
                'case_id': row['case_id'], 'source_group': row['source_group'],
                'request_sha256': digest({'state': row['state'], 'question': spec}),
                'expected': ids[max(range(len(ids)), key=row['target'].__getitem__)],
                'raw_choice': choice, 'choice': choice, 'probabilities': dict(zip(ids, probs)),
                'teacher_target': dict(zip(ids, row['target'])),
            }, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'model_id': model_id, 'model_revision': revision, 'dataset_revision': manifest['revision'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'temperature_fitted_on': 'calibration split (source-group disjoint from train/dev/test)',
        'temperature': temperature,
        'test_uncalibrated': metrics(test, 1.0), 'test_calibrated': metrics(test, temperature),
        'calibration_uncalibrated': metrics(calibration, 1.0)['all'],
        'order_reversal_argmax_change_fraction': flips / len(test),
        'selective_uncalibrated': selective(test, 1.0),
        'latency': {'device': str(device), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
                    'batch_size': 1, 'samples': len(latencies), 'includes_tokenization': True,
                    'p50_ms': statistics.median(latencies), 'p95_ms': latencies[int(0.95 * (len(latencies) - 1))]},
    }
    (output/'test-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
