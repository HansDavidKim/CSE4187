#!/usr/bin/env python3
"""Summarize user-bootstrap lagged edge selection; exploratory, not a causal verdict."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/backend-ts-bootstrap'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/backend-ts-bootstrap/summary.json'))
    args = parser.parse_args()
    files = sorted(args.input.glob('seed-*/nts_dag.json'))
    if not files:
        raise ValueError('no bootstrap runs')
    runs = [json.loads(path.read_text()) for path in files]
    if any(run['sequence_sampling'] != 'user bootstrap with replacement' for run in runs):
        raise ValueError('all runs must use user bootstrap')
    counts = Counter((edge['source'], edge['target'], edge['lag']) for run in runs for edge in run['edges'])
    result = {'runs': len(runs), 'converged_runs': sum(run['converged'] for run in runs),
        'same_bin_edges_forbidden': all(run['forbid_instantaneous'] for run in runs),
        'edge_frequencies': [{'source': source, 'target': target, 'lag': lag, 'count': count,
                              'frequency': count / len(runs)}
                             for (source, target, lag), count in sorted(counts.items())],
        'warning': 'Exploratory small bootstrap; threshold-sensitive associations in logged histories, not identified causal effects.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
