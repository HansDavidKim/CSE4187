"""Audit observed event-days without inferring user inactivity or training IRL."""
import json
import sys

import pandas as pd

from preprocessing.observed_day_features import build_days, gap_bin, history_rows, user_split
from utils.paths import PROJECT_ROOT, CAPSTONE_ROOT
CAPSTONE = CAPSTONE_ROOT
ROOT = PROJECT_ROOT


from utils.events import load_events


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
