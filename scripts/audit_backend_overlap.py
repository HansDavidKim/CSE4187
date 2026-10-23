#!/usr/bin/env python3
"""Cross-fitted, user-grouped propensity overlap diagnostic; not causal validation."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecommerce_sim.backend_ts_v2 import history_summaries


def effective_size(weights):
    return float(weights.sum() ** 2 / np.square(weights).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/backend-ts-features-v2/train.npz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/backend-ts-overlap-audit.json'))
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        x = history_summaries(data['history'])
        t = data['treatment'].astype(int)
        users = data['user_ids']
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=2026))
    p = cross_val_predict(model, x, t, groups=users, cv=GroupKFold(n_splits=5), method='predict_proba')[:, 1]
    clipped = np.clip(p, .01, .99)
    weights = np.where(t == 1, 1 / clipped, 1 / (1 - clipped))
    metrics = {'episodes': len(t), 'users': len(np.unique(users)),
        'cross_fitted_propensity_auc': float(roc_auc_score(t, p)),
        'propensity_quantiles': np.quantile(p, [0, .05, .25, .5, .75, .95, 1]).tolist(),
        'outside_0_05_0_95_fraction': float(np.mean((p < .05) | (p > .95))),
        'clipped_ipw_ess_treated': effective_size(weights[t == 1]),
        'clipped_ipw_ess_control': effective_size(weights[t == 0]),
        'warning': 'Diagnostic only: history-selected observational episodes, time confounding, sampling and positivity remain. ESS is episode-weight based, not user-cluster effective size.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
