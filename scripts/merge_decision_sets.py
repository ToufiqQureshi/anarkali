"""Concatenate prepared decision sets split-by-split into one training directory.

Default: typed-decisions-v2 (public benchmark) + coding-decisions-v0 (synthetic coding
workflows) -> anarkali-decisions-v3. Source groups must not collide across inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "calibration", "test")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, nargs='+',
                   default=[REPO/'artifacts'/'typed-decisions-v2', REPO/'artifacts'/'coding-decisions-v0'])
    p.add_argument('--output', type=Path, default=REPO/'artifacts'/'anarkali-decisions-v3')
    args = p.parse_args()
    manifests = [json.loads((d/'manifest.json').read_text(encoding='utf-8')) for d in args.inputs]
    for directory, manifest in zip(args.inputs, manifests):
        for name in SPLITS:
            actual = hashlib.sha256((directory/f'{name}.jsonl').read_bytes()).hexdigest()
            if actual != manifest['split_counts'][name]['sha256']:
                raise ValueError(f'{directory.name}/{name}.jsonl does not match its manifest')
    args.output.mkdir(parents=True, exist_ok=True)
    seen_groups = {}
    counts = {}
    for name in SPLITS:
        lines = []
        groups = set()
        for directory in args.inputs:
            for line in (directory/f'{name}.jsonl').read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                group = json.loads(line)['source_group']
                owner = seen_groups.setdefault(group, (directory.name, name))
                if owner != (directory.name, name):
                    raise ValueError(f'source group {group} appears in {owner} and {(directory.name, name)}')
                groups.add(group)
                lines.append(line)
        path = args.output/f'{name}.jsonl'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        counts[name] = {'decision_cases': len(lines), 'source_groups': len(groups),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    revision = hashlib.sha256(json.dumps([m.get('revision') for m in manifests]).encode()).hexdigest()[:16]
    manifest = {'dataset': 'anarkali-decisions-v3', 'revision': revision,
                'sources': [{'path': str(d.relative_to(REPO)), 'dataset': m.get('dataset'), 'revision': m.get('revision'),
                             'label_source': m.get('label_source')} for d, m in zip(args.inputs, manifests)],
                'split_unit': 'source groups inherited from each input; no group appears in two splits',
                'split_counts': counts}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
