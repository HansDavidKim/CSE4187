"""Compare first three observed UTC visit-day gaps; writes aggregate results only."""
import json
import sys


from utils.paths import PROJECT_ROOT as ROOT
from utils.events import load_events


def analyze(events):
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


def run(source):
    events, loader = load_events(source)
    result = analyze(events)
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


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
