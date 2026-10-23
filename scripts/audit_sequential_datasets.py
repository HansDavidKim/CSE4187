#!/usr/bin/env python3
"""Audit sequential intervention data; absence of a campaign record is not a control."""
import argparse
import json
from pathlib import Path

import pandas as pd


def heartsteps(path):
    d = pd.read_csv(path / 'data_files' / 'suggestions.csv', low_memory=False)
    required = ['user.index', 'decision.index', 'sugg.select.utime', 'avail', 'connect',
                'is.randomized', 'send', 'jbsteps30pre', 'jbsteps30']
    d = d[required].copy()
    if d[['user.index', 'decision.index']].isna().any().any():
        raise ValueError('missing user/decision identifier')
    if d.duplicated(['user.index', 'decision.index']).any():
        raise ValueError('duplicate user/decision identifier')
    d['sugg.select.utime'] = pd.to_datetime(d['sugg.select.utime'], errors='coerce')
    d = d.sort_values(['user.index', 'decision.index'])
    eligible = d.avail.eq(True) & d.connect.eq(True) & d['send'].notna()
    known = d[eligible]
    observed = known[known.jbsteps30.notna()]
    counts = d.groupby('user.index').size()
    # is.randomized is the draw result, not an indicator that randomization was attempted.
    report = {'rows': len(d), 'users': int(d['user.index'].nunique()),
              'duplicate_user_decisions': 0, 'missing_selected_time': int(d['sugg.select.utime'].isna().sum()),
              'decisions_per_user_min': int(counts.min()), 'decisions_per_user_median': float(counts.median()),
              'decisions_per_user_max': int(counts.max()),
              'available_connected_known_action': int(eligible.sum()),
              'available_sent': int(known['send'].eq(True).sum()),
              'available_not_sent': int(known['send'].eq(False).sum()),
              'unavailable_sent_inconsistencies': int((d.avail.eq(False) & d['send'].eq(True)).sum()),
              'missing_action': int(d['send'].isna().sum()),
              'available_missing_outcome': int(known.jbsteps30.isna().sum()),
              'observed_outcome_by_action': {str(k): {'n': len(g), 'mean_steps30': float(g.jbsteps30.mean())}
                                             for k, g in observed.groupby('send')},
              'limitations': ['Observed-outcome group means are descriptive, not an unbiased causal effect.',
                 'is.randomized is not an eligibility flag; filtering it removes the comparison arm.',
                 'Missing step measurements are not zero; eligibility and delivery inconsistencies need review.',
                 'Dataset is a randomized intervention policy, not demonstrations of a user choosing notifications.']}
    return d, report


def journey(path):
    frames = {name: pd.read_csv(path / (name + '.csv'), low_memory=False)
              for name in ('transaction_data', 'campaign_table', 'campaign_desc')}
    tx, assign, campaigns = (frames[k] for k in ('transaction_data', 'campaign_table', 'campaign_desc'))
    for frame, cols in [(tx, ['household_key', 'DAY', 'BASKET_ID', 'SALES_VALUE']),
                        (assign, ['household_key', 'CAMPAIGN']),
                        (campaigns, ['CAMPAIGN', 'START_DAY', 'END_DAY'])]:
        if not set(cols).issubset(frame.columns):
            raise ValueError('missing columns: ' + str(set(cols) - set(frame.columns)))
    if campaigns.CAMPAIGN.duplicated().any():
        raise ValueError('campaign metadata has duplicate IDs')
    report = {'transaction_rows': len(tx), 'transaction_households': int(tx.household_key.nunique()),
              'transaction_days_min': int(tx.DAY.min()), 'transaction_days_max': int(tx.DAY.max()),
              'campaign_rows': len(campaigns), 'assignment_rows': len(assign),
              'assigned_households': int(assign.household_key.nunique()),
              'assignment_duplicate_keys': int(assign.duplicated(['household_key', 'CAMPAIGN']).sum()),
              'assignment_rows_missing_campaign': int((~assign.CAMPAIGN.isin(campaigns.CAMPAIGN)).sum()),
              'assignment_households_absent_from_transactions': int((~assign.household_key.isin(tx.household_key)).sum()),
              'households_with_multiple_recorded_campaigns': int((assign.groupby('household_key').CAMPAIGN.nunique() > 1).sum()),
              'limitations': ['No assignment record is unknown, not a verified control.',
                              'Campaign start date is not a verified household contact time.',
                              'No assumption of randomized targeting; coupon redemption is not assignment.']}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heartsteps', type=Path, default=Path('data/raw/heartsteps-v1'))
    p.add_argument('--journey', type=Path)
    p.add_argument('--output', type=Path, default=Path('artifacts/sequential-data-audit'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, report = heartsteps(args.heartsteps)
    data.to_csv(args.output / 'heartsteps_local_decisions.csv.gz', index=False)
    (args.output / 'heartsteps_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'heartsteps': report}, indent=2))
    if args.journey:
        report = journey(args.journey)
        (args.output / 'journey_audit.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'complete_journey': report}, indent=2))


if __name__ == '__main__':
    main()
