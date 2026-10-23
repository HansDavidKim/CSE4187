"""Temporal, administrative-censoring-aware evaluation of next-session hazards."""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ecommerce_sim.analysis.session_return import EDGES, FEATURES, durations, expand_hazards, horizon_prob, sessions
from scripts.audit_observed_days import ROOT, load_events


def scores(model, s, cutoff, baseline):
    elapsed, event, available = durations(s, cutoff)
    output = {}
    for h in (24, 168, 336):
        # A full horizon must be observable for every selected session.
        keep = available >= h
        target = (event & (elapsed < h))[keep].astype(int)
        if len(target) == 0:
            continue
        p = np.clip(horizon_prob(model, s.loc[keep, FEATURES].to_numpy(dtype=float), h, baseline), 1e-8, 1 - 1e-8)
        calibration = []
        for low, high in zip((0, .05, .1, .2, .4), (.05, .1, .2, .4, 1.01)):
            group = (p >= low) & (p < high)
            if group.any():
                calibration.append({'range': [low, min(high, 1)], 'sessions': int(group.sum()),
                                    'predicted': float(p[group].mean()), 'observed': float(target[group].mean())})
        output[str(h)] = {'sessions': len(target), 'returns': int(target.sum()),
                          'observed_rate': float(target.mean()), 'mean_prediction': float(p.mean()),
                          'brier': float(brier_score_loss(target, p)),
                          'log_loss': float(log_loss(target, p, labels=[0, 1])),
                          'auc': float(roc_auc_score(target, p)) if len(np.unique(target)) == 2 else None,
                          'calibration_bins': calibration}
    return output


def run(source):
    e, loader = load_events(source)
    if source != 'cosmetics':
        ids = e['user'].drop_duplicates()
        selected = ids[pd.util.hash_pandas_object(ids, index=False).to_numpy() % 20 == 0]
        e = e[e['user'].isin(selected)].copy()
    start = e['at'].min().ceil('D')
    end = e['at'].max().floor('D')
    e = e[(e['at'] >= start) & (e['at'] < end)]
    train_end = start + (end - start) * .6
    valid_end = start + (end - start) * .8
    s = sessions(e, end)
    train = s[s['end'] < train_end]
    valid = s[(s['end'] >= train_end) & (s['end'] < valid_end)]
    test = s[(s['end'] >= valid_end) & (s['end'] < end)]
    x, y = expand_hazards(train, train_end)
    model = LogisticRegression(max_iter=200, random_state=0).fit(x, y)
    baseline = LogisticRegression(max_iter=200, random_state=0).fit(x[:, :len(EDGES) - 1], y)
    result = {'source': source, 'sampling': 'user hash modulo 20 == 0' if source != 'cosmetics' else 'user id modulo 100 == 0',
              'loader': loader, 'start': start.isoformat(), 'train_end': train_end.isoformat(),
              'validation_end': valid_end.isoformat(), 'end': end.isoformat(),
              'session_rule': 'new session when gap >30 min; prior end is last event +30 min; 30-minute inactivity assumed observed',
              'sessions': {k: len(v) for k, v in [('train', train), ('validation', valid), ('test', test)]},
              'train_hazard_rows': len(y), 'train_return_bin_events': int(y.sum()),
              'train_bin_support': [{'range_hours': [float(EDGES[i]), float(EDGES[i + 1])],
                                     'rows': int(x[:, i].sum()), 'returns': int(y[x[:, i] == 1].sum())}
                                    for i in range(len(EDGES) - 1)],
              'features': FEATURES, 'hazard_bin_edges_hours': EDGES.tolist(),
              'evaluation': {name: {'baseline': scores(baseline, part, cutoff, True),
                                    'features': scores(model, part, cutoff, False)}
                             for name, part, cutoff in [('validation', valid, valid_end), ('test', test, end)]},
              'limitations': ['A session end and observation completeness are operational assumptions, not observed logout.',
                 'Timeout defines minimum 30-minute gap and can bias session semantics; same-day returns are possible.',
                 'Only bins with full administrative follow-up enter training, including when a return happened earlier in the bin.',
                 'Scores use sessions with full horizon observed, not all test sessions; time split is not user-disjoint.',
                 'Only one global split, user subsample, and no GRU persona embeddings; no survival-tail or permanent-churn validation.',
                 'Train cutoffs censor future sessions; features contain only prior and current completed sessions.',
                 'No period sampler selected, fitted model saved, or IRL run.']}
    out = ROOT / 'artifacts/session-return-evaluation'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, json.dumps(result['evaluation']), flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
