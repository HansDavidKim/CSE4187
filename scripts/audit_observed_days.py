"""Audit observed event-days without inferring user inactivity or training IRL."""
import json
import sys

import pandas as pd

from ecommerce_sim.preprocessing.observed_day_features import build_days, gap_bin, history_rows, user_split
from ecommerce_sim.paths import PROJECT_ROOT, CAPSTONE_ROOT
CAPSTONE = CAPSTONE_ROOT
ROOT = PROJECT_ROOT


def load_events(source):
    root = CAPSTONE / 'data/raw/kaggle'
    paths = sorted((root / 'cosmetics').glob('20??-*.csv')) if source == 'cosmetics' else [root / source / 'events.csv']
    if not paths:
        raise ValueError('no input files')
    retail = source == 'retailrocket'
    columns = ['visitorid', 'timestamp', 'event'] if retail else ['user_id', 'event_time', 'event_type', 'user_session']
    user_col, time_col, action_col = columns[:3]
    mapping = {'view': 'view', 'addtocart': 'cart', 'transaction': 'purchase'} if retail else {
        'view': 'view', 'cart': 'cart', 'purchase': 'purchase', 'remove_from_cart': 'remove'}
    parts = []
    audit = {'raw_rows': 0, 'sample_rows': 0, 'missing_session_rows': 0,
             'valid_rows_missing_session': 0, 'invalid_user_time_action_rows': 0}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=columns, dtype={user_col: 'string'}, chunksize=500_000):
            audit['raw_rows'] += len(chunk)
            if source == 'cosmetics':
                uid = pd.to_numeric(chunk[user_col], errors='coerce').astype('Int64')
                chunk = chunk.loc[uid.mod(100).eq(0).fillna(False)].copy()
            audit['sample_rows'] += len(chunk)
            at = pd.to_datetime(chunk[time_col], unit='ms' if retail else None, utc=True, errors='coerce')
            action = chunk[action_col].map(mapping)
            valid = chunk[user_col].notna() & at.notna() & action.notna()
            audit['invalid_user_time_action_rows'] += int((~valid).sum())
            if not retail:
                missing = chunk['user_session'].isna()
                audit['missing_session_rows'] += int(missing.sum())
                audit['valid_rows_missing_session'] += int((missing & valid).sum())
            parts.append(pd.DataFrame({'user': chunk.loc[valid, user_col], 'at': at[valid],
                                       'action': action[valid]}))
    if not parts:
        raise ValueError('no events')
    return pd.concat(parts, ignore_index=True), audit


def run(source):
    events, audit = load_events(source)
    first, last = events['at'].min(), events['at'].max()
    # Exclude whole boundary days because logging starts/ends mid-day.
    first_complete, last_exclusive = first.ceil('D'), last.floor('D')
    eligible_events = events[(events['at'] >= first_complete) & (events['at'] < last_exclusive)].copy()
    users = eligible_events['user'].drop_duplicates()
    split_map = {u: user_split(source, u) for u in users}
    days = build_days(eligible_events)
    days['split'] = days['user'].map(split_map)
    rows = history_rows(days, max_lag=3)
    splits = {}
    for split in ('train', 'validation', 'test'):
        frame = days[days['split'].eq(split)]
        relevant = [r for r in rows if split_map[r['user']] == split]
        actions = frame.loc[frame.eligible, 'action_code'].value_counts().sort_index()
        gaps = pd.Series([gap_bin(r['state_gap_days']) for r in relevant], dtype='string').value_counts()
        splits[split] = {'users': int(frame.user.nunique()), 'event_days': len(frame),
                         'eligible_event_days': int(frame.eligible.sum()),
                         'excluded_remove_days': int(frame.excluded_remove_day.sum()),
                         'eligible_actions': {str(i): int(actions.get(i, 0)) for i in range(1, 8)},
                         'history_targets': len(relevant),
                         'targets_with_lag3': sum(r['lags'][3] != -1 for r in relevant),
                         'known_state_gap_bins': {k: int(v) for k, v in gaps.items()}}
    missing_ids = eligible_events.loc[eligible_events['user'].notna(), 'user'].nunique()
    result = {'source': source, 'loader_audit': audit, 'first_event_utc': first.isoformat(),
              'last_event_utc': last.isoformat(), 'full_day_start_inclusive': first_complete.isoformat(),
              'full_day_end_exclusive': last_exclusive.isoformat(), 'users_in_complete_days': int(missing_ids),
              'split_rule': 'SHA256(v1:source:user) %100; train <70, validation <85, test otherwise',
              'cosmetics_sampling': 'user id modulo 100 == 0' if source == 'cosmetics' else None,
              'splits': splits, 'no_irl_trained': True,
              'limitations': ['Only observed event-days are considered; gaps are not evidence of no activity.',
                 'Remove-containing days excluded and break history, causing selection bias.',
                 'First and final partial UTC days excluded; per-user coverage and day-level completeness not verified.',
                 'Train/validation/test user hash split only; calendar cutoff and untouched final evaluation not yet defined.',
                 'Gap before the target is unavailable; only gaps between prior observed days are state features.',
                 'Cosmetics modulo sample cannot establish population prevalence.']}
    out = ROOT / 'artifacts/observed-day-audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, 'loader', audit, 'split counts', {k: (v['eligible_event_days'], v['history_targets'])
                                                   for k, v in splits.items()}, flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
