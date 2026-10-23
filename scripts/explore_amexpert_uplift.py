#!/usr/bin/env python3
"""Sensitivity-only AIPW: assumes missing offer records mean no offer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('artifacts/amexpert-campaign-audit/local_customer_periods.csv.gz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-campaign-audit/uplift_sensitivity.json'))
    parser.add_argument('--assume-missing-unassigned', action='store_true', help='required, not verified from source')
    args = parser.parse_args()
    if not args.assume_missing_unassigned:
        parser.error('requires explicit --assume-missing-unassigned sensitivity assumption')
    p = pd.read_csv(args.panel)
    # The last observation interval ends before the active campaign has ended.
    p = p[(p.period_id > 0) & (p.period_id < p.period_id.max())].copy().reset_index(drop=True)
    x = p[['period_id', 'duration_days', 'previous_spend_per_day', 'previous_transaction_rows_per_day',
           'previous_has_offer_record', 'previous_period_days']].astype(float)
    a = p.has_train_offer_record.astype(int).to_numpy()
    y = p.spend_per_day.to_numpy()
    ids = p.customer_id.to_numpy()
    if x.isna().any().any() or len(np.unique(a)) != 2:
        raise ValueError('missing pretreatment features or a treatment arm')
    propensity = np.empty(len(p))
    mu0 = np.empty(len(p))
    mu1 = np.empty(len(p))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(x, a, ids)):
        classifier = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1, random_state=fold)
        classifier.fit(x.iloc[train], a[train])
        propensity[test] = classifier.predict_proba(x.iloc[test])[:, 1]
        for target, out in ((0, mu0), (1, mu1)):
            subset = train[a[train] == target]
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, l2_regularization=1, random_state=fold)
            model.fit(x.iloc[subset], y[subset])
            out[test] = model.predict(x.iloc[test])
    overlap = (propensity >= .05) & (propensity <= .95)
    e, ay, yy = propensity[overlap], a[overlap], y[overlap]
    influence = mu1[overlap] - mu0[overlap] + ay*(yy-mu1[overlap])/e - (1-ay)*(yy-mu0[overlap])/(1-e)
    estimate = influence.mean()
    clusters = pd.DataFrame({'customer_id': ids[overlap], 'centered': influence-estimate}).groupby('customer_id').centered.sum()
    n_groups = len(clusters)
    se = float(np.sqrt((n_groups/(n_groups-1)) * np.square(clusters).sum()) / len(influence))
    period = p.groupby('period_id').has_train_offer_record.agg(['size', 'mean'])
    result = {'classification': 'assumption-dependent observational sensitivity; NOT verified causal uplift',
        'assumption': 'All absent offer records denote non-receipt, including prior-period indicators; not established.',
        'target': 'difference in average daily spend, restricted to estimated propensity in [0.05, 0.95]',
        'sample': 'periods 1..31; excludes initial history and final partially observed campaign interval',
        'rows': len(p), 'customers': int(p.customer_id.nunique()), 'recorded_treated_share': float(a.mean()),
        'raw_recorded_vs_absent_spend_difference': float(y[a==1].mean()-y[a==0].mean()),
        'cross_fit': 'five folds grouped by customer; not a temporal holdout',
        'features': list(x.columns), 'propensity_quantiles': np.quantile(propensity, [.01,.05,.5,.95,.99]).tolist(),
        'rows_in_overlap': int(overlap.sum()), 'overlap_share': float(overlap.mean()),
        'treated_overlap': int(ay.sum()), 'untreated_overlap': int((1-ay).sum()),
        'aipw_assumption_dependent_estimate': float(estimate), 'customer_cluster_only_se': se,
        'customer_cluster_only_approx_95ci': [float(estimate-1.96*se), float(estimate+1.96*se)],
        'periods_without_both_recorded_groups': [int(i) for i, r in period.iterrows() if r['mean'] in (0, 1)],
        'limitations': ['Treatment status for absent records unverified; differential misclassification can bias all estimates.',
            'No randomized assignment; exchangeability, positivity and interference assumptions unverified.',
            'Previous-period offer can reflect continuing treatment; current treatment conflates new and ongoing offers.',
            'Coupon versions and overlapping assignments are bundled; decision time only approximated.',
            'Unmeasured user traits, demographics and calendar shocks are not fully controlled.',
            'Interval-level overlap trimming changes the target population and is not proof of positivity.',
            'CI clusters only by customer and ignores common calendar shocks, selection and model specification.',
            'Previous-period summaries span variable durations; model is a sensitivity diagnostic, not the PLOS analysis.',
            'No causal graph edge or simulator reward is updated from this estimate.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
