"""Retrospective persona-labelled second-session gaps; labels use future prefix."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from ecommerce_sim.paths import CAPSTONE_ROOT


CAPSTONE = CAPSTONE_ROOT
OUT = Path('artifacts/persona-delays')
FIG = Path('research/figures/analysis')
HOURS = np.array([.5, 1, 6, 24, 72, 168, 336, 720])


def summarize(frame):
    n = len(frame)
    if not n:
        return None
    gap = frame.gap_hours
    returned = gap.notna() & gap.le(720)
    result = {'n': n, 'returns_30d': int(returned.sum()),
              'return_counts': [int((gap.notna() & gap.le(h)).sum()) for h in HOURS],
              'returner_gap_hours_quantiles': (gap[returned].quantile([.25, .5, .75]).tolist()
                                              if returned.any() else None)}
    if returned.any():
        previous = frame.loc[returned, 'previous_gap_hours']
        following = gap[returned]
        result['paired_previous_median_hours'] = float(previous.median())
        result['paired_next_median_hours'] = float(following.median())
        result['paired_next_greater_count'] = int((following > previous).sum())
    return result


def run(source):
    from src.analysis.masked_embedding_clusters import rows_from_events, source_data

    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    observation_end = events['at'].max()
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if emb.user_id.tolist() != [r[0] for r in rows]:
        raise ValueError('saved embeddings do not match reconstructed users')
    z = emb[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions_per_user = np.array([r[2] for r in rows])
    order = np.random.default_rng(42).permutation(len(rows))
    train, test = order[:int(.8*len(rows))], order[int(.8*len(rows)):]
    train, test = train[sessions_per_user[train] >= 2], test[sessions_per_user[test] >= 2]
    subset = np.random.default_rng(97).choice(train, min(1800, len(train)), replace=False)
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=6, n_init=10, random_state=42).fit(z[subset]).predict(z[test])
    labels_df = pd.DataFrame({'user': emb.user_id.iloc[test].to_numpy(), 'cluster': labels})
    baseline = json.loads(Path(f'artifacts/two-dataset-gru-clusters/{source}.json').read_text())
    expected = next(r['groups'] for r in baseline['runs'] if r['method'] == 'kmeans' and r['k'] == 6)
    for group in expected:
        assert int((labels == group['cluster']).sum()) == group['users']

    events = events[events.user.isin(labels_df.user)].copy()
    for action in ('view', 'cart', 'remove', 'purchase'):
        events[action] = (events.action.eq(actions.index(action)).astype('int8') if action in actions else 0)
    s = events.groupby(['user', 'session'], sort=False).agg(
        start=('at', 'min'), end=('at', 'max'), view=('view', 'sum'), cart=('cart', 'sum'),
        remove=('remove', 'sum'), purchase=('purchase', 'sum')).reset_index()
    s = s.sort_values(['user', 'start', 'end', 'session'], kind='stable')
    s['rank'] = s.groupby('user', sort=False).cumcount()
    first = s[s['rank'].eq(0)].set_index('user')
    second = s[s['rank'].eq(1)].set_index('user').copy()
    third = s[s['rank'].eq(2)].set_index('user')
    second['previous_gap_hours'] = (second.start - first.end.reindex(second.index)).dt.total_seconds()/3600
    second['next_start'] = third.start.reindex(second.index)
    second['gap_hours'] = (second.next_start - second.end).dt.total_seconds()/3600
    assert second.previous_gap_hours.ge(0).all()
    assert second.gap_hours.dropna().ge(0).all()
    second['scenario'] = np.select([second.purchase.gt(0), second.cart.gt(0) | second.remove.gt(0)],
                                   ['purchase', 'cart_or_remove_no_purchase'], default='view_only')
    second = second.reset_index().merge(labels_df, on='user', validate='one_to_one')
    panel = second[second.end + pd.Timedelta(days=30) <= observation_end].copy()
    result = {'source': source, 'quality': quality, 'observation_end': observation_end.isoformat(),
              'heldout_repeat_users': len(second), 'fully_observed_30d': len(panel),
              'censored': len(second)-len(panel), 'horizons_hours': HOURS.tolist(), 'clusters': []}
    for cluster in range(6):
        part = panel[panel.cluster.eq(cluster)]
        record = {'cluster': cluster, 'all': summarize(part), 'scenarios': {}}
        for scenario in ('view_only', 'cart_or_remove_no_purchase', 'purchase'):
            record['scenarios'][scenario] = summarize(part[part.scenario.eq(scenario)])
        result['clusters'].append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#6d6d6d', '#3572a2', '#d58b39', '#6c8d5d', '#a65a72', '#665b9e']
    for row, color in zip(result['clusters'], colors):
        item = row['all']
        if item and item['n'] >= 30:
            ax.plot(HOURS/24, np.array(item['return_counts'])/item['n']*100, marker='o',
                    color=color, label=f"c{row['cluster']} (n={item['n']})")
    ax.set_xscale('log')
    ax.set_xticks([.5/24, 1/24, .25, 1, 3, 7, 14, 30], ['30m', '1h', '6h', '1d', '3d', '7d', '14d', '30d'])
    ax.set_ylim(0, 100)
    ax.set(xlabel='Time since end of second observed session', ylabel='Next session recorded by time (%)',
           title=f'{source.title()}: retrospective K=6 labels, complete 30-day follow-up')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / f'{source}_persona_delay.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(6)
    for offset, scenario, marker, color in [(-.18, 'view_only', 'o', '#3572a2'),
                                             (0, 'cart_or_remove_no_purchase', 's', '#d58b39'),
                                             (.18, 'purchase', '^', '#a65a72')]:
        for row in result['clusters']:
            item = row['scenarios'][scenario]
            if item and item['n'] >= 20:
                x = row['cluster'] + offset
                ax.scatter(x, 100*item['returns_30d']/item['n'], marker=marker, color=color)
                ax.annotate(f"{item['n']}", (x, 100*item['returns_30d']/item['n']),
                            xytext=(3, 5), textcoords='offset points', fontsize=8)
        ax.scatter([], [], marker=marker, color=color, label=scenario.replace('_', ' '))
    ax.set_xticks(positions, [f'c{i}' for i in positions])
    ax.set(ylim=(0, 105), ylabel='Next session recorded within 30 days (%)',
           xlabel='Dataset-specific K=6 cluster',
           title=f'{source.title()}: second-session type (labels show n, only n≥20)')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f'{source}_persona_delay_scenarios.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'cosmetics'):
            raise ValueError(source)
        run(source)
