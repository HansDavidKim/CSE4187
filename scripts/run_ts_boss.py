#!/usr/bin/env python3
"""Run the authors' TS-BOSS on boundary-safe, pooled customer windows."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ecommerce_sim.preprocessing.nts_adapter import load_sequences, make_lag_windows


def graph_edges(graph: np.ndarray, features: list[str]) -> list[dict]:
    """Keep ambiguous contemporaneous endpoints instead of inventing directions."""
    edges = []
    for lag in range(graph.shape[2]):
        for i, source in enumerate(features):
            for j, target in enumerate(features):
                marker = graph[i, j, lag]
                if marker == '-->' or (lag == 0 and marker == 'o-o' and i < j):
                    edges.append({'source': source, 'target': target, 'lag': lag,
                                  'orientation': 'undirected' if marker == 'o-o' else 'directed'})
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--source', type=Path, default=Path('external/TS-BOSS'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-lag', type=int, default=2)
    parser.add_argument('--max-sequences', type=int)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--pd', type=int, default=2, help='BIC penalty multiplier')
    parser.add_argument('--drop-features', nargs='*', default=[])
    parser.add_argument('--nts-result', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str((args.source / 'src').resolve()))
    from tsboss.ts_boss import TSBOSS

    sequences, features = load_sequences(args.sequences)
    missing = set(args.drop_features) - set(features)
    if missing:
        parser.error(f'unknown features: {sorted(missing)}')
    keep = [i for i, name in enumerate(features) if name not in args.drop_features]
    features = [features[i] for i in keep]
    if args.max_sequences is not None:
        if not 1 <= args.max_sequences <= len(sequences):
            parser.error('max-sequences must be within available sequence count')
        indices = np.random.default_rng(args.seed).choice(len(sequences), args.max_sequences, replace=False)
        sequences = [sequences[i] for i in indices]
    sequences = [sequence[:, keep] for sequence in sequences]
    windows, _ = make_lag_windows(sequences, args.max_lag)
    # The official IID input accepts already-unrolled rows: oldest slice first.
    # This does NOT imply that overlapping windows are statistically independent.
    data = windows.transpose(0, 2, 1).reshape(len(windows), -1)
    if not np.isfinite(data).all() or np.any(np.std(data, axis=0) == 0):
        raise ValueError('TS-BOSS BIC requires finite, nonconstant columns')
    model = TSBOSS(lag_max=args.max_lag, pd=args.pd, rng=np.random.default_rng(args.seed))
    # Upstream currently accesses an unset self.verbose when verbose is None.
    model.run_tsboss(data, iid_data=True, get_mpdag=True, verbose=False)
    edges = graph_edges(model.mpdag['graph'], features)
    commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'method': 'official TS-BOSS with pooled pre-unrolled customer windows',
              'official_source_commit': commit, 'features': features,
              'sequence_count': len(sequences), 'window_count': len(windows),
              'max_lag': args.max_lag, 'seed': args.seed, 'pd': args.pd,
              'sampling': 'seeded customer sample without replacement' if args.max_sequences else 'all sequences',
              'caveats': ['Overlapping windows are not iid; pooled BIC is an approximation.',
                          'Linear Gaussian BIC; no domain prior constraints supplied.',
                          'MPDAG directions depend on assumptions and are not identified causal effects.'],
              'representative_dag_edges': graph_edges(model.dag['graph'], features),
              'mpdag_edges': edges}
    if args.nts_result:
        nts = json.loads(args.nts_result.read_text())
        if nts['features'] != features or nts['seed'] != args.seed or nts['max_lag'] != args.max_lag or nts['sequence_count'] != len(sequences):
            raise ValueError('NTS comparison does not match features, seed, lag or sample size')
        def key(edge):
            s, t, lag = edge['source'], edge['target'], edge['lag']
            return (tuple(sorted((s, t))) if lag == 0 else (s, t), lag)
        nts_adj = {key(e) for e in nts['edges']}
        boss_adj = {key(e) for e in edges}
        nts_directed = {(e['source'], e['target'], e['lag']) for e in nts['edges']}
        boss_directed = {(e['source'], e['target'], e['lag']) for e in edges if e['orientation'] == 'directed'}
        result['nts_comparison'] = {'nts_path': str(args.nts_result), 'nts_adjacencies': len(nts_adj),
            'tsboss_adjacencies': len(boss_adj), 'shared_adjacencies': len(nts_adj & boss_adj),
            'adjacency_jaccard': len(nts_adj & boss_adj) / len(nts_adj | boss_adj) if nts_adj | boss_adj else 1.0,
            'shared_directed_edges': len(nts_directed & boss_directed),
            'note': 'NTS uses same-week domain constraints; TS-BOSS does not. Model and scoring assumptions differ.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'tsboss_graph.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'mpdag_edges': len(edges),
                      'undirected': sum(e['orientation'] == 'undirected' for e in edges),
                      'nts_comparison': result.get('nts_comparison')}, indent=2))


if __name__ == '__main__':
    main()
