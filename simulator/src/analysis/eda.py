"""Exploratory analyses, intervention audits, and causal feature extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from utils.events import load_events
from utils.paths import CAPSTONE_ROOT, PROJECT_ROOT as ROOT


def split_intervals(campaigns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Split inclusive observation window at campaign starts and day after ends."""
    if start > end or campaigns[['start', 'end']].isna().any().any() or (campaigns.start > campaigns.end).any():
        raise ValueError('invalid campaign or observation dates')
    stop = end + pd.Timedelta(days=1)
    cuts = {start, stop}
    for row in campaigns.itertuples():
        if row.start <= end and row.end >= start:
            cuts.add(max(row.start, start))
            cuts.add(min(row.end + pd.Timedelta(days=1), stop))
    bounds = sorted(cuts)
    rows = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        active = campaigns.loc[(campaigns.start <= left) & (campaigns.end >= left), 'campaign_id']
        rows.append({'period_id': len(rows), 'start': left, 'end_exclusive': right,
                     'duration_days': (right-left).days, 'campaign_ids': tuple(sorted(active.tolist()))})
    return pd.DataFrame(rows)


def reconstruct(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    campaign = pd.read_csv(root / 'campaign_data.csv')
    campaign['start'] = pd.to_datetime(campaign.start_date, format='%d/%m/%y')
    campaign['end'] = pd.to_datetime(campaign.end_date, format='%d/%m/%y')
    train = pd.read_csv(root / 'train.csv')
    test = pd.read_csv(root / 'test.csv')
    tx = pd.read_csv(root / 'customer_transaction_data.csv')
    tx['date'] = pd.to_datetime(tx.date, format='%Y-%m-%d')
    if campaign.campaign_id.duplicated().any() or train.id.duplicated().any() or test.id.duplicated().any():
        raise ValueError('duplicate campaign or assignment ID')
    if set(train.id) & set(test.id):
        raise ValueError('train/test assignment IDs overlap')
    if not set(train.campaign_id).union(test.campaign_id).issubset(set(campaign.campaign_id)):
        raise ValueError('assignment campaign missing from campaign metadata')
    train_ids, test_ids = set(train.campaign_id), set(test.campaign_id)
    intervals = split_intervals(campaign, tx.date.min(), tx.date.max())
    customers = np.union1d(tx.customer_id.unique(), np.union1d(train.customer_id.unique(), test.customer_id.unique()))
    interval_count = len(intervals)
    lookup = pd.DataFrame({'customer_id': customers})
    panel = lookup.merge(intervals.drop(columns='campaign_ids'), how='cross')
    bounds = pd.DatetimeIndex(intervals.start).to_numpy()
    period_index = np.searchsorted(bounds, tx.date.to_numpy(), side='right') - 1
    if (period_index < 0).any() or (period_index >= interval_count).any():
        raise ValueError('transaction outside reconstructed intervals')
    tx['period_id'] = period_index
    spend = tx.groupby(['customer_id', 'period_id']).agg(spend=('selling_price', 'sum'),
        transaction_rows=('item_id', 'size'), coupon_discount=('coupon_discount', 'sum')).reset_index()
    panel = panel.merge(spend, on=['customer_id', 'period_id'], how='left', validate='one_to_one')
    for col in ('spend', 'transaction_rows', 'coupon_discount'):
        panel[col] = panel[col].fillna(0)
    panel['spend_per_day'] = panel.spend / panel.duration_days
    panel['has_train_offer_record'] = False
    panel['has_test_offer_record'] = False
    for source, frame in [('train', train), ('test', test)]:
        counts = frame.groupby(['customer_id', 'campaign_id']).size().rename('count').reset_index()
        occurrences = []
        for period in intervals.itertuples():
            if period.campaign_ids:
                segment = counts[counts.campaign_id.isin(period.campaign_ids)][['customer_id']].drop_duplicates().copy()
                segment['period_id'] = period.period_id
                occurrences.append(segment)
        if occurrences:
            idx = pd.MultiIndex.from_frame(pd.concat(occurrences).drop_duplicates())
            panel[f'has_{source}_offer_record'] = pd.MultiIndex.from_frame(panel[['customer_id','period_id']]).isin(idx)
    panel['offer_status'] = np.where(panel.has_train_offer_record | panel.has_test_offer_record,
                                     'recorded_offer', 'unknown_not_verified_untreated')
    panel = panel.sort_values(['customer_id', 'period_id']).reset_index(drop=True)
    panel['previous_spend_per_day'] = panel.groupby('customer_id').spend_per_day.shift(1)
    panel['previous_transaction_rows_per_day'] = (panel.transaction_rows / panel.duration_days).groupby(panel.customer_id).shift(1)
    panel['previous_has_offer_record'] = panel.groupby('customer_id').has_train_offer_record.shift(1)
    panel['previous_period_days'] = panel.groupby('customer_id').duration_days.shift(1)
    exposure = train[['campaign_id','customer_id','coupon_id','redemption_status']]
    test_overlap = set(test.campaign_id) & train_ids
    unknown_ids = set(campaign.campaign_id) - train_ids - test_ids
    report = {'observation_start': str(tx.date.min().date()), 'observation_end': str(tx.date.max().date()),
        'customers': len(customers), 'periods': interval_count, 'active_periods': int(intervals.campaign_ids.map(bool).sum()),
        'rows': len(panel), 'duration_days_min': int(intervals.duration_days.min()),
        'duration_days_max': int(intervals.duration_days.max()),
        'campaigns_total': len(campaign), 'train_campaigns': sorted(map(int, train_ids)),
        'test_campaigns': sorted(map(int, test_ids)), 'train_test_campaign_overlap': sorted(map(int, test_overlap)),
        'campaigns_without_assignment_records': sorted(map(int, unknown_ids)),
        'assignment_ids_train': len(train), 'assignment_ids_test': len(test),
        'train_duplicate_customer_campaign_coupon_rows': int(exposure.duplicated(['customer_id','campaign_id','coupon_id']).sum()),
        'test_duplicate_customer_campaign_coupon_rows': int(test.duplicated(['customer_id','campaign_id','coupon_id']).sum()),
        'recorded_train_offer_customer_periods': int(panel.has_train_offer_record.sum()),
        'recorded_test_offer_customer_periods': int(panel.has_test_offer_record.sum()),
        'recorded_either_offer_customer_periods': int((panel.offer_status == 'recorded_offer').sum()),
        'unknown_treatment_customer_periods': int((panel.offer_status != 'recorded_offer').sum()),
        'redemption_positive_train_rows': int(train.redemption_status.sum()),
        'analysis_rows_after_initial_history_period': int((panel.period_id > 0).sum()),
        'recorded_offer_share_after_initial_period': float(panel.loc[panel.period_id > 0, 'has_train_offer_record'].mean()),
        'last_period_clipped_by_observation': bool(((campaign.campaign_id.isin(intervals.iloc[-1].campaign_ids)) & (campaign.end > tx.date.max())).any()),
        'test_redemption_labels_available': False,
        'caveats': ['Observed test offers are used only as recorded assignments; test redemption labels are absent.',
            'No assignment record is not verified as no exposure; do not interpret unknown as an untreated control.',
            'Interval cut points use campaign metadata, not verified actual assignment timestamps.',
            'Observation boundaries clip campaigns; incomplete edge intervals are not full treatment windows.',
            'No randomized assignment or conditional exchangeability has been established.',
            'Raw and row-level derivatives remain local pending a data rights review.']}
    return intervals, panel, report


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


def audit_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--amexpert', type=Path, help='audit AmExpert campaign-validity periods')
    p.add_argument('--heartsteps', type=Path, default=Path('data/raw/heartsteps-v1'))
    p.add_argument('--journey', type=Path)
    p.add_argument('--output', type=Path, default=Path('artifacts/sequential-data-audit'))
    args = p.parse_args(argv)
    if args.amexpert:
        output = args.output if args.output != Path('artifacts/sequential-data-audit') else Path('artifacts/amexpert-campaign-audit')
        periods, panel, report = reconstruct(args.amexpert)
        output.mkdir(parents=True, exist_ok=True)
        periods.assign(campaign_ids=periods.campaign_ids.map(lambda ids: ','.join(map(str, ids)))).to_csv(output / 'periods.csv', index=False)
        panel.to_csv(output / 'local_customer_periods.csv.gz', index=False)
        (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    data, report = heartsteps(args.heartsteps)
    data.to_csv(args.output / 'heartsteps_local_decisions.csv.gz', index=False)
    (args.output / 'heartsteps_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'heartsteps': report}, indent=2))
    if args.journey:
        report = journey(args.journey)
        (args.output / 'journey_audit.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'complete_journey': report}, indent=2))



def extract_markov_boundary(
    edges: list[dict[str, object]],
    target: str,
    *,
    known_at_decision: set[str] | None = None,
) -> dict[str, object]:
    """Extract a temporal Markov-boundary hypothesis and leakage-safe state set.

    Lagged parents are safe state candidates. Contemporaneous parents are only
    included in the state set when explicitly declared known before the action.
    Children and spouses are reported for analysis but excluded from state by
    default because their measurement order can be ambiguous.
    """
    known_at_decision = known_at_decision or set()
    parents: set[tuple[str, int]] = set()
    children: set[str] = set()
    spouses: set[tuple[str, int]] = set()

    for edge in edges:
        source, child, lag = str(edge["source"]), str(edge["target"]), int(edge["lag"])
        if child == target:
            parents.add((source, lag))
        if source == target and lag == 0:
            children.add(child)

    for child in children:
        for edge in edges:
            if str(edge["target"]) != child:
                continue
            source, lag = str(edge["source"]), int(edge["lag"])
            if source != target:
                spouses.add((source, lag))

    boundary = parents | spouses | {(child, 0) for child in children}
    state = {
        node for node in parents
        if node[1] > 0 or (node[1] == 0 and node[0] in known_at_decision)
    }

    def records(nodes: set[tuple[str, int]]) -> list[dict[str, object]]:
        return [
            {"feature": feature, "lag": lag}
            for feature, lag in sorted(nodes, key=lambda value: (value[1], value[0]))
        ]

    return {
        "target": target,
        "parents": records(parents),
        "children": sorted(children),
        "spouses": records(spouses),
        "markov_boundary": records(boundary),
        "state_candidates": records(state),
        "state_rule": "lagged parents plus explicitly pre-decision contemporaneous parents",
    }


def save_feature_sets(
    dag_file: str | Path,
    output: str | Path,
    targets: list[str],
    known_at_decision: set[str] | None = None,
) -> dict[str, object]:
    dag = json.loads(Path(dag_file).read_text())
    if dag.get("converged") is not True or dag.get("instantaneous_is_dag") is not True:
        raise ValueError("NTS graph must converge and be acyclic before extracting features")
    result = {
        "source_dag": str(dag_file),
        "warning": "Markov boundaries inherit uncertainty from the learned observational graph.",
        "targets": {
            target: extract_markov_boundary(
                dag["edges"], target, known_at_decision=known_at_decision
            )
            for target in targets
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result



def inactivity_label(landmark, next_start, observation_end, days):
    if days <= 0:
        raise ValueError('days must be positive')
    landmark = pd.to_datetime(landmark, utc=True)
    next_start = pd.to_datetime(next_start, utc=True)
    end = pd.Timestamp(observation_end)
    if end.tzinfo is None:
        raise ValueError('observation end must be timezone-aware')
    cutoff = landmark + pd.Timedelta(days=days)
    if (next_start.notna() & next_start.lt(landmark)).any():
        raise ValueError('next session precedes landmark')
    labels = pd.Series(pd.NA, index=landmark.index, dtype='boolean')
    observed = cutoff.le(end)
    labels.loc[observed] = (next_start.loc[observed].isna() |
                            next_start.loc[observed].gt(cutoff.loc[observed]))
    return labels


def summarize_long_inactivity(events, first_complete, end_exclusive):
    anchor = end_exclusive - pd.Timedelta(days=60)
    if anchor <= first_complete + pd.Timedelta(days=30):
        raise ValueError('need 30 days of history and 60 days of follow-up')
    e = events[(events['at'] >= first_complete) & (events['at'] < end_exclusive)].copy()
    e['day'] = e['at'].dt.floor('D')
    past = e[e['day'] < anchor].copy()
    future = e[e['day'] >= anchor]
    days = past[['user', 'day']].drop_duplicates().sort_values(['user', 'day'])
    users = days.groupby('user', sort=False)['day'].agg(['min', 'max', 'size'])
    users['days_since_visit'] = (anchor - users['max']).dt.days
    for period in (7, 30):
        cnt = days[days['day'] >= anchor - pd.Timedelta(days=period)].groupby('user').size()
        users[f'visits_{period}'] = cnt.reindex(users.index, fill_value=0)
    future_day = future[['user', 'day']].drop_duplicates()
    returned_30 = set(future_day.loc[future_day.day < anchor + pd.Timedelta(days=30), 'user'])
    returned_60 = set(future_day.user)
    users['status'] = 'inactive60'
    users.loc[users.index.isin(returned_60), 'status'] = 'late_return_31_60'
    users.loc[users.index.isin(returned_30), 'status'] = 'return_1_30'
    last = days.groupby('user', sort=False).tail(1)
    last_events = past.merge(last, on=['user', 'day'], how='inner')
    flags = pd.crosstab(last_events['user'], last_events['action']).gt(0)
    users = users.join(flags, how='left')
    recent = days.groupby('user', sort=False).tail(4).copy()
    recent['gap'] = recent.groupby('user', sort=False)['day'].diff().dt.days
    recent['rank'] = recent.groupby('user', sort=False).cumcount()
    gaps = recent[recent['rank'] > 0].pivot(index='user', columns='rank', values='gap').reindex(columns=[1, 2, 3]).dropna()
    # Only users with four pre-anchor days have three valid within-history gaps.
    assert all(users.loc[gaps.index, 'size'] >= 4)
    def group_summary(u):
        g = gaps.loc[gaps.index.intersection(u.index)]
        n = len(u)
        return {'users': n,
            'past_single_visit_day': int(u['size'].eq(1).sum()),
            'days_since_visit_median': float(u['days_since_visit'].median()) if n else None,
            'visit_days_last_7_mean': float(u['visits_7'].mean()) if n else None,
            'visit_days_last_30_mean': float(u['visits_30'].mean()) if n else None,
            'last_visit_action_fraction': {a: float(u[a].fillna(False).mean()) if a in u and n else 0.0
                                           for a in ('view', 'cart', 'purchase', 'remove')},
            'four_or_more_pre_anchor_visit_days': len(g),
            'last_three_gaps_median': [float(g[i].median()) for i in (1, 2, 3)] if len(g) else None,
            'gap_increasing_fraction': float(((g[1] < g[2]) & (g[2] < g[3])).mean()) if len(g) else None,
            'third_gap_greater_first_fraction': float((g[3] > g[1]).mean()) if len(g) else None,
            'third_gap_less_first_fraction': float((g[3] < g[1]).mean()) if len(g) else None}
    groups, recent_groups = {}, {}
    for label in ('inactive60', 'late_return_31_60', 'return_1_30'):
        u = users[users.status.eq(label)]
        groups[label] = group_summary(u)
        recent_groups[label] = group_summary(u[u['visits_30'] > 0])
    return {'anchor_utc': anchor.isoformat(), 'end_exclusive_utc': end_exclusive.isoformat(),
            'start_complete_utc': first_complete.isoformat(), 'groups': groups,
            'groups_with_pre_anchor_visit_in_last_30_days': recent_groups}


def run_long_inactivity(source):
    events, loader = load_events(source)
    result = summarize_long_inactivity(events, events['at'].min().ceil('D'), events['at'].max().floor('D'))
    result.update({'source': source, 'loader': loader,
                   'cosmetics_sampling': 'user id modulo 100 == 0' if source == 'cosmetics' else None,
                   'definition': 'Anchor is start of final 60 full UTC days. Cohort has at least one prior logged event; groups are mutually exclusive by first return within 60 days.',
                   'caveats': ['60-day inactivity is not verified permanent churn.',
                     'All features use only pre-anchor events; last visit means latest pre-anchor visit, not necessarily final visit.',
                     'Single fixed anchor; observation length and user-entry timing differ; not matched or causal.',
                     'Some users already had long inactivity before anchor; compare recency and consider first-seen cohorts.',
                     'Only completed observed pre-anchor gaps are used; requiring >=4 days selects repeat visitors.',
                     'Platform logging/identity completeness is assumed; cosmetics is a modulo sample.']})
    out = ROOT / 'artifacts/long-inactivity-analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, json.dumps(result), flush=True)


def summarize_visit_gaps(events):
    days = events[['user', 'at']].copy()
    days['day'] = days['at'].dt.floor('D')
    days = days[['user', 'day']].drop_duplicates().sort_values(['user', 'day'])
    counts = days.groupby('user', sort=False).size()
    first = days[days.groupby('user', sort=False).cumcount() < 4].copy()
    first['gap'] = first.groupby('user', sort=False)['day'].diff().dt.days
    first['ordinal'] = first.groupby('user', sort=False).cumcount()
    first = first[first['ordinal'] > 0]
    gaps = first.pivot(index='user', columns='ordinal', values='gap').reindex(columns=[1, 2, 3]).dropna()
    a, b, c = (gaps[i] for i in (1, 2, 3))
    assert (gaps.to_numpy() >= 1).all()
    masks = {'strictly_increasing': (a < b) & (b < c),
             'nondecreasing_with_increase': (a <= b) & (b <= c) & (a < c),
             'third_greater_than_first': c > a,
             'third_equal_first': c == a, 'third_less_than_first': c < a,
             'strictly_decreasing': (a > b) & (b > c)}
    n = len(gaps)
    examples = gaps[masks['strictly_increasing']].value_counts().head(10)
    return {'users_total': int(len(counts)), 'users_with_4plus_visit_days': n,
            'visit_day_count_distribution': {k: int(v) for k, v in {
                'one': (counts == 1).sum(), 'two': (counts == 2).sum(),
                'three': (counts == 3).sum(), 'four_or_more': (counts >= 4).sum()}.items()},
            'first_three_gap_days': {str(i): {'median': float(gaps[i].median()) if n else None,
                                               'mean': float(gaps[i].mean()) if n else None,
                                               'p75': float(gaps[i].quantile(.75)) if n else None}
                                     for i in (1, 2, 3)},
            'patterns': {k: {'users': int(v.sum()), 'fraction_of_4plus_users': float(v.mean()) if n else None}
                         for k, v in masks.items()},
            'strict_increase_examples': [{'gaps_days': [int(x) for x in key], 'users': int(value)}
                                         for key, value in examples.items()],
            'gap_bins': {str(i): {k: int(v) for k, v in {
                '1_day': (gaps[i] == 1).sum(), '2_to_7': gaps[i].between(2, 7).sum(),
                '8_to_30': gaps[i].between(8, 30).sum(), 'over_30': (gaps[i] > 30).sum()}.items()}
                         for i in (1, 2, 3)}}


def run_visit_gaps(source):
    events, loader = load_events(source)
    result = summarize_visit_gaps(events)
    result.update({'source': source, 'loader': loader,
                   'cosmetics_sampling': 'user id modulo 100 == 0' if source == 'cosmetics' else None,
                   'first_utc': events['at'].min().isoformat(), 'last_utc': events['at'].max().isoformat(),
                   'definition': 'unique UTC days with any mapped platform event, including cosmetics remove; first three gaps among users with at least four observed visit days',
                   'caveats': ['Not representative of users with fewer than 4 observed visit days.',
                               'Only completed observed gaps; last-visit-to-end is right-censored and not used.',
                               'Short observation windows and requiring 4 visits select frequent early users; comparison is descriptive, not a churn or latent-satisfaction estimate.',
                               'One observed event day is not necessarily one session; within-day gaps are not measured.']})
    output = ROOT / 'artifacts/visit-gap-analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, json.dumps(result), flush=True)


LATE_ROOT = CAPSTONE_ROOT


LATE_OUT = Path('artifacts/late-inactivity')


def session_type(frame):
    return np.select([frame.purchases.gt(0), (frame.carts + frame.removes).gt(0)],
                     ['purchase', 'cart_or_remove_no_purchase'], default='view_only')


def run_late_session(source):
    from analysis.persona.sequence import source_data

    path = LATE_ROOT / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                   else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    end = events['at'].max()
    for name in ('view', 'cart', 'remove', 'purchase'):
        events[name + 's'] = (events.action.eq(actions.index(name)).astype('int8')
                              if name in actions else 0)
    sessions = events.groupby(['user', 'session'], sort=False).agg(
        start=('at', 'min'), end=('at', 'max'), views=('views', 'sum'),
        carts=('carts', 'sum'), removes=('removes', 'sum'), purchases=('purchases', 'sum')).reset_index()
    sessions = sessions.sort_values(['user', 'start', 'end', 'session'], kind='stable')
    sessions['rank'] = sessions.groupby('user', sort=False).cumcount()
    second = sessions[sessions['rank'].eq(1)].set_index('user')
    third = sessions[sessions['rank'].eq(2)].set_index('user').copy()
    fourth = sessions[sessions['rank'].eq(3)].set_index('user')
    third['previous_type'] = session_type(second.reindex(third.index))
    third['type'] = session_type(third)
    third['next_start'] = fourth['start'].reindex(third.index)
    labels = inactivity_label(third.end, third.next_start, end, 30)
    panel = third[labels.notna()].copy()
    panel['inactive'] = labels[labels.notna()].astype(bool)
    profiles = []
    for label, group in panel.groupby('inactive'):
        transitions = group.groupby(['previous_type', 'type']).size().to_dict()
        profiles.append({'inactive_30d': bool(label), 'users': len(group),
                         'third_session_types': group.type.value_counts().to_dict(),
                         'third_view_any': int(group.views.gt(0).sum()),
                         'third_cart_any': int(group.carts.gt(0).sum()),
                         'third_purchase_any': int(group.purchases.gt(0).sum()),
                         'third_remove_any': int(group.removes.gt(0).sum()) if 'remove' in actions else None,
                         'third_events_median': float(group[['views', 'carts', 'removes', 'purchases']].sum(axis=1).median()),
                         'second_to_third': [{'second': a, 'third': b, 'users': int(n)}
                                             for (a,b), n in transitions.items()]})
    result = {'source': source, 'quality': quality, 'observation_end': end.isoformat(),
              'at_least_three_sessions': len(third), 'eligible': len(panel),
              'censored': int(labels.isna().sum()), 'profiles': profiles,
              'limitations': ['Third observed session is not necessarily the final lifetime session.',
                              'No observed event is not proof of churn or complete logging.',
                              '30-day eligibility uses last observed timestamp as cutoff.',
                              'Cosmetics uses deterministic user_id modulo 100 sampling.' if source == 'cosmetics' else
                              'Electronics has no remove event in the input action set.']}
    LATE_OUT.mkdir(parents=True, exist_ok=True)
    (LATE_OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))



TIMEOUT = pd.Timedelta(minutes=30)


EDGES = np.array([0, 1, 6, 24, 72, 168, 336, 720, 1440], dtype=float)


FEATURES = ['view', 'cart', 'purchase', 'remove', 'events', 'duration_hours',
            'previous_sessions', 'previous_gap_hours', 'has_previous', 'hour_sin', 'hour_cos']


def sessions(events, observation_end):
    e = events.sort_values(['user', 'at'], kind='stable').copy()
    delta = e.groupby('user', sort=False)['at'].diff()
    e['sid'] = delta.isna() | delta.gt(TIMEOUT)
    e['sid'] = e.groupby('user', sort=False)['sid'].cumsum()
    for action in ('view', 'cart', 'purchase', 'remove'):
        e[action] = e['action'].eq(action).astype('int8')
    e['events'] = 1
    grouped = e.groupby(['user', 'sid'], sort=False)
    s = grouped.agg(start=('at', 'min'), last=('at', 'max'),
                    **{a: (a, 'sum') for a in ('view', 'cart', 'purchase', 'remove', 'events')}).reset_index()
    s['end'] = s['last'] + TIMEOUT
    s['next_start'] = s.groupby('user', sort=False)['start'].shift(-1)
    # Session ends are observed only after the inactivity timeout has elapsed.
    s = s[s['end'] <= observation_end].copy()
    g = s.groupby('user', sort=False)
    s['previous_end'] = g['end'].shift(1)
    s['has_previous'] = s['previous_end'].notna().astype(int)
    s['previous_gap_hours'] = ((s['start'] - s['previous_end']).dt.total_seconds() / 3600).fillna(0).clip(0, 720) / 720
    s['previous_sessions'] = g.cumcount().clip(0, 20) / 20
    s['duration_hours'] = ((s['last'] - s['start']).dt.total_seconds() / 3600).clip(0, 24) / 24
    for col in ('view', 'cart', 'purchase', 'remove', 'events'):
        s[col] = s[col].clip(0, 20) / 20
    hour = s['end'].dt.hour + s['end'].dt.minute / 60
    s['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    s['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return s


def durations(s, cutoff):
    available = (cutoff - s['end']).dt.total_seconds().to_numpy() / 3600
    next_hours = (s['next_start'] - s['end']).dt.total_seconds().to_numpy() / 3600
    event = np.isfinite(next_hours) & (next_hours <= available)
    return np.where(event, next_hours, available), event, available


def expand_hazards(s, cutoff):
    elapsed, event, available = durations(s, cutoff)
    arrays, labels = [], []
    x = s[FEATURES].to_numpy(dtype=float)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        # Require full administrative follow-up for the entire interval for
        # both outcomes; otherwise early observed returns are overrepresented.
        use = (available >= hi) & (elapsed >= lo)
        occurred = event & (elapsed < hi)
        if not use.any():
            continue
        bin_columns = np.zeros((use.sum(), len(EDGES) - 1))
        bin_columns[:, i] = 1
        arrays.append(np.column_stack((bin_columns, x[use])))
        labels.append(occurred[use].astype(int))
    return np.concatenate(arrays), np.concatenate(labels)


def horizon_prob(model, x, horizon, baseline=False):
    survival = np.ones(len(x))
    for i, hi in enumerate(EDGES[1:]):
        if hi > horizon:
            break
        bins = np.zeros((len(x), len(EDGES) - 1))
        bins[:, i] = 1
        inp = bins if baseline else np.column_stack((bins, x))
        survival *= 1 - model.predict_proba(inp)[:, 1]
    return 1 - survival


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


def evaluate(source):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('audit', 'long-inactivity', 'visit-gaps', 'late-session', 'session-return'))
    args, remaining = parser.parse_known_args()
    if args.task == 'audit':
        audit_main(remaining)
        return
    sources_parser = argparse.ArgumentParser(description=f'Run {args.task} analysis.')
    sources_parser.add_argument('sources', nargs='+', choices=('electronics', 'cosmetics', 'retailrocket'))
    sources = sources_parser.parse_args(remaining).sources
    if args.task == 'late-session' and 'retailrocket' in sources:
        sources_parser.error('late-session supports electronics and cosmetics only')
    runner = {'long-inactivity': run_long_inactivity, 'visit-gaps': run_visit_gaps,
              'late-session': run_late_session, 'session-return': evaluate}[args.task]
    for source in sources:
        runner(source)


if __name__ == '__main__':
    main()

