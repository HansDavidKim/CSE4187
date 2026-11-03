#!/usr/bin/env python3
"""Predict observed 3h login from prehistory and episode treatment; not uplift."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing.backend_ts_v2 import history_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/backend-ts-features-v2'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/backend-ts-outcome-baseline'))
    args = parser.parse_args()
    data = {}
    for split in ['train', 'validation', 'test']:
        with np.load(args.input / f'{split}.npz', allow_pickle=False) as d:
            if not d['endpoint_mask'].all() or not d['history_grid_mask'].all():
                raise ValueError('baseline does not support missing fields')
            x = np.column_stack((history_summaries(d['history']), d['treatment']))
            data[split] = x, d['outcome_y3'].copy()
    preprocess = ColumnTransformer([('history', StandardScaler(), list(range(10))),
                                   ('treatment', 'passthrough', [10])])
    model = make_pipeline(preprocess, LogisticRegression(max_iter=1000, random_state=2026))
    model.fit(*data['train'])
    metrics = {}
    for split, (x, y) in data.items():
        p = model.predict_proba(x)[:, 1]
        metrics[split] = {'episodes': len(y), 'observed_login_rate': float(y.mean()),
            'log_loss': float(log_loss(y, p, labels=[0, 1])), 'brier': float(brier_score_loss(y, p)),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    result = {'method': 'train-only-scaled logistic outcome prediction',
        'features': ['log1p method counts for pre-index hours -168:-24', 'log1p method counts for hours -24:0', 'episode treatment'],
        'metrics': metrics, 'warning': 'Predictive baseline, not an uplift or causal effect estimator; pseudo-control and confounding remain.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
