"""Audit calendar-day features on a deterministic user subsample; no training."""
import hashlib
import json

from scripts.audit_observed_days import ROOT, load_events
from ecommerce_sim.calendar_day_features import calendar_days
from ecommerce_sim.observed_day_features import build_days, history_rows, user_split


def run(source):
    events, loader = load_events(source)
    start = events['at'].min().ceil('D')
    end = events['at'].max().floor('D')
    events = events[(events['at'] >= start) & (events['at'] < end)].copy()
    # Small deterministic subsample for an inexpensive feature audit, not a population estimate.
    threshold = 100 if source == 'cosmetics' else 10
    users = events.user.drop_duplicates()
    selected = set(u for u in users if int.from_bytes(
        hashlib.sha256(f'calendar-audit:{source}:{u}'.encode()).digest()[:8], 'big') % 1000 < threshold)
    events = events[events.user.isin(selected)]
    split_map = {u: user_split(source, u) for u in selected}
    panel = calendar_days(build_days(events))
    panel['split'] = panel.user.map(split_map)
    rows = history_rows(panel, 3)
    splits = {}
    for split in ('train', 'validation', 'test'):
        part = panel[panel['split'].eq(split)]
        targets = [r for r in rows if split_map[r['user']] == split]
        action_counts = part.loc[part.eligible, 'action_code'].value_counts()
        splits[split] = {'users': int(part.user.nunique()), 'calendar_days': len(part),
                         'assumed_inactive_days': int(part.assumed_inactive.sum()),
                         'excluded_remove_days': int(part.excluded_remove_day.sum()),
                         'eligible_actions': {str(i): int(action_counts.get(i, 0)) for i in range(8)},
                         'targets': len(targets), 'targets_with_lag3': sum(r['lags'][3] != -1 for r in targets)}
    result = {'source': source, 'loader': loader, 'subsample': f'hash threshold {threshold}/1000'
              + (' after existing cosmetics user-id modulo-100 sample' if source == 'cosmetics' else ''),
              'full_day_start_inclusive': start.isoformat(), 'full_day_end_exclusive': end.isoformat(),
              'assumption': 'complete target-platform event logging between each user first and last observed full day',
              'splits': splits, 'no_irl_trained': True,
              'limitations': ['Per-user first and last observed days are selected using future data and are not known at online prediction time.',
                  'Outside the first/last observed interval is censored, not inactive.',
                  'A lack of logged events means no recorded platform action under the assumption, not intentional rest.',
                  'Sample audit only; no calendar time split, model evaluation, or population prevalence estimate.',
                  'Remove-containing days are excluded and reset lag histories.']}
    out = ROOT / 'artifacts/calendar-day-audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, splits, flush=True)


if __name__ == '__main__':
    import sys
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
