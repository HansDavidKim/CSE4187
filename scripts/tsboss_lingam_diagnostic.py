#!/usr/bin/env python3
"""Exploratory lag-residual DirectLiNGAM orientation of a TS-BOSS skeleton."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis, normaltest, skew
from ecommerce_sim.preprocessing.nts_adapter import load_sequences, make_lag_windows

# gCastle changes the process-wide dtype while importing its unrelated torch models.
_previous_dtype = torch.get_default_dtype()
try:
    from castle.algorithms import DirectLiNGAM
finally:
    torch.set_default_dtype(_previous_dtype)


def orient_skeleton(edges, order):
    rank = {name: i for i, name in enumerate(order)}
    result = []
    for edge in edges:
        e = dict(edge)
        if e['lag'] == 0:
            a, b = e['source'], e['target']
            e['source'], e['target'] = (a, b) if rank[a] < rank[b] else (b, a)
            e['orientation'] = 'directed_by_lingam_order_hypothesis'
        result.append(e)
    return result


def hsic_gram(x):
    d = (x[:, None] - x[None, :]) ** 2
    nonzero = d[d > 0]
    if len(nonzero) == 0:
        return None
    k = np.exp(-d / (2 * np.median(nonzero)))
    return k - k.mean(axis=0) - k.mean(axis=1)[:, None] + k.mean()


def hsic_permutation(x, y, rng, permutations=199):
    k, l = hsic_gram(x), hsic_gram(y)
    if k is None or l is None:
        return None
    observed = np.sum(k * l)
    null = [np.sum(k * l[np.ix_(p, p)]) for p in (rng.permutation(len(x)) for _ in range(permutations))]
    return float((1 + sum(v >= observed for v in null)) / (permutations + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout-customers', type=int, default=200)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    sequences, original_features = load_sequences(args.sequences)
    features = graph['features']
    keep = [original_features.index(f) for f in features]
    sequences = [s[:, keep] for s in sequences]
    seed, n, lag = graph['seed'], graph['sequence_count'], graph['max_lag']
    train_ids = np.random.default_rng(seed).choice(len(sequences), n, replace=False)
    available = np.setdiff1d(np.arange(len(sequences)), train_ids)
    if args.holdout_customers > len(available):
        parser.error('not enough distinct held-out customers')
    rng = np.random.default_rng(seed + 10000)
    heldout_ids = rng.choice(available, args.holdout_customers, replace=False)
    train = [sequences[i] for i in train_ids]
    mean, scale = np.concatenate(train).mean(axis=0), np.concatenate(train).std(axis=0)
    if np.any(scale == 0):
        raise ValueError('constant training feature')
    train = [(s - mean) / scale for s in train]
    windows, _ = make_lag_windows(train, lag)
    x = windows[:, :, :-1].transpose(0, 2, 1).reshape(len(windows), -1)
    y = windows[:, :, -1]
    x = np.column_stack([np.ones(len(x)), x])
    coeff = np.linalg.lstsq(x, y, rcond=None)[0]
    innovation = y - x @ coeff
    innovation_scale = innovation.std(axis=0)
    if np.any(innovation_scale < 1e-10):
        raise ValueError('degenerate innovation')
    model = DirectLiNGAM(measure='pwling')
    model.fit(innovation / innovation_scale)
    order = [features[i] for i in model._causal_order]
    oriented = orient_skeleton(graph['mpdag_edges'], order)

    # One randomly chosen window per never-seen customer for diagnostics.
    heldout = []
    for i in heldout_ids:
        s = (sequences[i] - mean) / scale
        t = int(rng.integers(lag, len(s)))
        heldout.append(s[t-lag:t+1].T)
    heldout = np.stack(heldout)
    hx = heldout[:, :, :-1].transpose(0, 2, 1).reshape(len(heldout), -1)
    hx = np.column_stack([np.ones(len(hx)), hx])
    innov = (heldout[:, :, -1] - hx @ coeff) / innovation_scale
    errors = innov - innov @ np.asarray(model.adjacency_matrix_).T
    normality = {f: float(normaltest(errors[:, i]).pvalue) for i, f in enumerate(features)}
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            p = hsic_permutation(errors[:, i], errors[:, j], rng)
            pairs.append({'a': features[i], 'b': features[j], 'p': p})
    same_week = [e for e in graph['mpdag_edges'] if e['lag'] == 0]
    changed = [e for e, new in zip(graph['mpdag_edges'], oriented) if e['lag'] == 0 and e['orientation'] == 'directed' and (e['source'], e['target']) != (new['source'], new['target'])]
    result = {'seed': seed, 'features': features, 'source_graph': str(args.graph),
              'training_customers': n, 'training_overlapping_windows': len(windows),
              'heldout_customers': len(heldout), 'heldout_sampling': 'one random window per disjoint customer',
              'method': 'OLS on all lagged variables then gcastle DirectLiNGAM on pooled innovations; orient TS-BOSS skeleton by estimated order',
              'causal_order_hypothesis': order, 'same_week_edges': len(same_week),
              'reversed_vs_tsboss': len(changed), 'reversed_edges': changed,
              'heldout_error_moments_standardized': {f: {'variance': float(np.var(errors[:, i])),
                  'skewness': float(skew(errors[:, i])), 'excess_kurtosis': float(kurtosis(errors[:, i]))}
                  for i, f in enumerate(features)},
              'heldout_error_normaltest_pvalues_descriptive': normality,
              'heldout_error_hsic_permutation_pvalues_exploratory': pairs,
              'oriented_candidate_edges': oriented,
              'limitations': ['Linear no-hidden-confounding independent non-Gaussian errors assumed, not established.',
                              'Overlapping training windows; shared temporal shocks and multiple testing affect diagnostics.',
                              'Counts, zero inflation, aggregation and nonlinear dynamics violate or challenge model assumptions.',
                              'Estimated order orients skeleton but does not test direct effects; no domain prior imposed.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'lingam_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'order': order, 'same_week': len(same_week), 'reversed': len(changed),
                      'normality_p_below_0.05': sum(p < .05 for p in normality.values()),
                      'hsic_p_below_0.05': sum(p['p'] is not None and p['p'] < .05 for p in pairs),
                      'hsic_pairs': len(pairs)}, indent=2))


if __name__ == '__main__':
    main()
