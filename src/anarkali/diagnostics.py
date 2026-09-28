"""Development-only controls; priors cannot establish semantic model quality."""
from collections import defaultdict, Counter
import json
import math
import random


def state_ablations(rows, seed):
    """Change source state while preserving question, options, labels and workflow."""
    by_workflow = {}
    for row in rows:
        by_workflow.setdefault(row['workflow'], {})[row['source_group']] = row['state']
    state_map = {}
    for workflow, states in by_workflow.items():
        keys = sorted(states)
        random.Random(f'{seed}:{workflow}').shuffle(keys)
        if len(keys) < 2:
            raise ValueError('state ablation requires multiple sources per workflow')
        state_map.update({key: states[keys[(i+1) % len(keys)]] for i, key in enumerate(keys)})
    return {'original': rows, 'blank': [dict(row, state={}) for row in rows],
            'shuffled_within_workflow': [dict(row, state=state_map[row['source_group']]) for row in rows]}


def schema_key(row):
    return (row['workflow'], row['case_id'].split('::', 1)[-1],
            tuple(sorted((c['id'], c['text']) for c in row['candidates'])))


def development_controls(train, development):
    if not train or not development:
        raise ValueError('train and development rows are required')
    train_groups = {r['source_group'] for r in train}
    development_groups = {r['source_group'] for r in development}
    if train_groups & development_groups:
        raise ValueError('source-group leakage')
    mass, votes, counts = defaultdict(Counter), defaultdict(Counter), Counter()
    for row in train + development:
        ids, target = [c['id'] for c in row['candidates']], row['target']
        if len(ids) < 2 or len(set(ids)) != len(ids) or len(target) != len(ids):
            raise ValueError('invalid candidate/target alignment')
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in target):
            raise ValueError('invalid target')
        if abs(math.fsum(target) - 1) > 1e-6:
            raise ValueError('target must sum to one')
    for row in train:
        key = schema_key(row)
        counts[key] += 1
        ids = [c['id'] for c in row['candidates']]
        winner = max(range(len(ids)), key=row['target'].__getitem__)
        votes[key][ids[winner]] += 1
        for cid, probability in zip(ids, row['target']):
            mass[key][cid] += probability
    buckets = defaultdict(list)
    for row in development:
        key = schema_key(row)
        if key not in counts:
            raise ValueError('development schema not found in training')
        ids = [c['id'] for c in row['candidates']]
        target = dict(zip(ids, row['target']))
        expected = ids[max(range(len(ids)), key=row['target'].__getitem__)]
        hard_choice = min(ids, key=lambda cid: (-votes[key][cid], cid))
        soft_choice = min(ids, key=lambda cid: (-mass[key][cid], cid))
        prior = {cid: mass[key][cid] / counts[key] for cid in ids}
        metrics = {
            'hard_majority_accuracy': float(hard_choice == expected),
            'soft_prior_accuracy': float(soft_choice == expected),
            'random_choice_expected_accuracy': 1 / len(ids),
            'teacher_entropy': -math.fsum(p * math.log(p) for p in target.values() if p > 0),
            'soft_prior_ce': -math.fsum(target[cid] * math.log(max(prior[cid], 1e-12)) for cid in ids),
            'soft_prior_brier': math.fsum((target[cid] - prior[cid]) ** 2 for cid in ids),
        }
        buckets['all'].append(metrics)
        buckets[row['workflow']].append(metrics)
    def aggregate(items):
        return {'decisions': len(items), **{k: math.fsum(x[k] for x in items) / len(items) for k in items[0]}}
    return {'split': 'development', 'train_source_groups': len(train_groups),
            'development_source_groups': len(development_groups),
            'controls': {key: aggregate(items) for key, items in buckets.items()},
            'note': 'Priors fit only on training labels; teacher entropy is a soft-CE floor, not a correctness ceiling.'}


def token_budget_report(rows, tokenizer, *, state_limit=384, question_limit=96, candidate_limit=64):
    lengths = {'state': [], 'question': [], 'candidate': []}
    collisions = 0
    for row in rows:
        state = json.dumps(row['state'], ensure_ascii=False, sort_keys=True)
        lengths['state'].append(len(tokenizer(state, truncation=False)['input_ids']))
        lengths['question'].append(len(tokenizer(row['question'], truncation=False)['input_ids']))
        candidates = []
        for c in row['candidates']:
            lengths['candidate'].append(len(tokenizer(c['text'], truncation=False)['input_ids']))
            candidates.append(tuple(tokenizer(c['text'], truncation=True, max_length=candidate_limit)['input_ids']))
        collisions += len(set(candidates)) != len(candidates)
    result = {}
    for name, limit in [('state', state_limit), ('question', question_limit), ('candidate', candidate_limit)]:
        values = sorted(lengths[name])
        if not values:
            raise ValueError('empty token audit')
        result[name] = {'items': len(values), 'limit': limit, 'truncated_items': sum(n > limit for n in values),
                        'truncated_fraction': sum(n > limit for n in values) / len(values),
                        'median_tokens': values[len(values)//2], 'max_tokens': max(values)}
    result['candidate_token_collisions'] = collisions
    return result
