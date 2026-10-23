"""Within-cluster variation in observed prefixes, sessions and return gaps."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from ecommerce_sim.paths import CAPSTONE_ROOT

BASE = CAPSTONE_ROOT
OUT = Path('artifacts/persona-heterogeneity')
FIG = Path('research/figures/heterogeneity')
ACTION_NAMES = ['view', 'cart', 'remove', 'purchase']


def qvalues(values):
    a = np.asarray(values, dtype=float)
    return np.quantile(a, [.25, .5, .75]).tolist() if len(a) else None


def action_bins(count, total):
    share = np.asarray(count) / np.asarray(total)
    return [int((share == 0).sum()), int(((share > 0) & (share <= .25)).sum()),
            int(((share > .25) & (share <= .5)).sum()), int((share > .5).sum())]


def delay_bins(gaps):
    gap = np.asarray(gaps, dtype=float)
    return [int((gap <= .5).sum()), int(((gap > .5) & (gap <= 24)).sum()),
            int(((gap > 24) & (gap <= 168)).sum()), int(((gap > 168) & (gap <= 720)).sum()),
            int((np.isnan(gap) | (gap > 720)).sum())]


def run(source):
    from src.analysis.masked_embedding_clusters import rows_from_events, source_data

    path = BASE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                   else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    end = events['at'].max()
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(BASE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if emb.user_id.tolist() != [r[0] for r in rows]:
        raise ValueError('saved embedding users differ from reconstructed prefixes')
    z = emb[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions_per_user = np.array([r[2] for r in rows])
    order = np.random.default_rng(42).permutation(len(rows))
    train, test = order[:int(.8*len(rows))], order[int(.8*len(rows)):]
    train, test = train[sessions_per_user[train] >= 2], test[sessions_per_user[test] >= 2]
    subset = np.random.default_rng(97).choice(train, min(1800, len(train)), replace=False)
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=6, n_init=10, random_state=42).fit(z[subset]).predict(z[test])
    users = pd.DataFrame({'user': emb.user_id.iloc[test].to_numpy(), 'cluster': labels})
    for name in actions:
        i = actions.index(name)
        users[name] = [rows[t][1].count(i) for t in test]
    users['events'] = users[actions].sum(axis=1)
    users['tokens_at_limit'] = [len(rows[t][1]) == 64 for t in test]
    assert users.events.gt(0).all()
    reference = json.loads(Path(f'artifacts/two-dataset-gru-clusters/{source}.json').read_text())
    groups = next(r['groups'] for r in reference['runs'] if r['method'] == 'kmeans' and r['k'] == 6)
    for g in groups:
        u = users[users.cluster.eq(g['cluster'])]
        assert len(u) == g['users']
        for action in actions:
            assert abs(float(u[action].gt(0).mean()) - g['users_with_action'][action]) < 1e-9

    e = events[events.user.isin(users.user)].copy()
    for a in ACTION_NAMES:
        e[a] = e.action.eq(actions.index(a)).astype('int8') if a in actions else 0
    sess = e.groupby(['user', 'session'], sort=False).agg(
        start=('at', 'min'), end=('at', 'max'), view=('view', 'sum'), cart=('cart', 'sum'),
        purchase=('purchase', 'sum'), remove=('remove', 'sum')).reset_index()
    sess = sess.sort_values(['user', 'start', 'end', 'session'], kind='stable')
    sess['rank'] = sess.groupby('user', sort=False).cumcount()
    second = sess[sess['rank'].eq(1)].set_index('user').copy()
    third = sess[sess['rank'].eq(2)].set_index('user')
    second['next_start'] = third.start.reindex(second.index)
    second['gap_h'] = (second.next_start - second.end).dt.total_seconds() / 3600
    assert second.gap_h.dropna().ge(0).all()
    second['events'] = second[ACTION_NAMES].sum(axis=1)
    second = second.reset_index().merge(users[['user', 'cluster']], on='user', validate='one_to_one')
    full = second[second.end + pd.Timedelta(days=30) <= end].copy()
    prior_delay = json.loads(Path(f'artifacts/persona-delays/{source}.json').read_text())
    assert len(full) == prior_delay['fully_observed_30d']
    result = {'source': source, 'quality': quality, 'actions': actions, 'users': len(users),
              'followup_30d': len(full), 'observation_end': end.isoformat(), 'clusters': []}
    for c in range(6):
        u = users[users.cluster.eq(c)]
        s = second[second.cluster.eq(c)]
        f = full[full.cluster.eq(c)]
        observed = f[f.gap_h.notna() & f.gap_h.le(720)]
        r = {'cluster': c, 'n': len(u), 'prefix_action_count_quartiles': qvalues(u.events),
             'prefix_tokens_at_limit_n': int(u.tokens_at_limit.sum()),
             'user_action_presence_counts': {a: int(u[a].gt(0).sum()) for a in actions},
             'cart_user_share_bins': action_bins(u.cart, u.events),
             'purchase_user_share_bins': action_bins(u.purchase, u.events),
             'second_session_event_count_quartiles': qvalues(s.events),
             'second_session_duration_minutes_quartiles': qvalues((s.end-s.start).dt.total_seconds()/60),
             'second_session_action_presence_counts': {a: int(s[a].gt(0).sum()) for a in actions},
             'followup_n': len(f), 'delay_bins': delay_bins(f.gap_h),
             'returner_gap_hours_quartiles': qvalues(observed.gap_h)}
        assert sum(r['delay_bins']) == len(f)
        assert sum(r['cart_user_share_bins']) == len(u)
        assert r['delay_bins'][-1] == len(f) - prior_delay['clusters'][c]['all']['returns_30d']
        result['clusters'].append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result, users, second


def stacked(results, key, title, labels, colors, filename, denominator):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, data in zip(axes, results):
        bottom = np.zeros(6)
        for idx, (label, color) in enumerate(zip(labels, colors)):
            values = np.array([r[key][idx] / r[denominator] * 100 if r[denominator] else 0 for r in data['clusters']])
            ax.bar(np.arange(6), values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.set_xticks(range(6), [f"c{r['cluster']}\nn={r[denominator]}" for r in data['clusters']])
        ax.set_title(data['source'].title())
        ax.set_ylim(0, 100)
        ax.set_xlabel('Dataset-specific K=6 cluster')
    axes[0].set_ylabel('Users in cluster (%)')
    fig.suptitle(title)
    fig.legend(labels, frameon=False, fontsize=9, loc='lower center', ncol=len(labels))
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(FIG / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)


def boxplots(panels, column, title, filename, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (data, frame) in zip(axes, panels):
        values = [frame.loc[frame.cluster.eq(c), column].to_numpy() for c in range(6)]
        ax.boxplot(values, showfliers=False, patch_artist=True,
                   boxprops={'facecolor': '#a8c2d8'}, medianprops={'color': '#a04a28'})
        ax.set_yscale('log')
        ax.set_xticks(range(1, 7), [f"c{c}\nn={len(v)}" for c, v in enumerate(values)])
        ax.set_title(data['source'].title())
        ax.set_xlabel('Dataset-specific K=6 cluster')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    outputs = [run(s) for s in ('electronics', 'cosmetics')]
    results = [o[0] for o in outputs]
    boxplots([(a, b) for a, b, _ in outputs], 'events',
             'Prefix action counts per user (first 64 tokens; log scale)', 'prefix_lengths.png', 'Action events in prefix')
    boxplots([(a, c) for a, _, c in outputs], 'events',
             'Second session action counts per user (log scale)', 'second_session_lengths.png', 'Action events in second session')
    stacked(results, 'cart_user_share_bins', 'Cart action share within each user prefix',
            ['0%', '(0, 25%]', '(25, 50%]', '>50%'], ['#d9e0e5', '#9ab9cf', '#557f9f', '#d58b39'],
            'cart_share_distribution.png', 'n')
    stacked(results, 'purchase_user_share_bins', 'Purchase action share within each user prefix',
            ['0%', '(0, 25%]', '(25, 50%]', '>50%'], ['#d9e0e5', '#9ab9cf', '#557f9f', '#d58b39'],
            'purchase_share_distribution.png', 'n')
    stacked(results, 'delay_bins', 'Observed next session after second session (30-day complete follow-up)',
            ['≤30m', '30m–1d', '1–7d', '7–30d', 'No session in 30d'],
            ['#4e789c', '#8fb0c8', '#d58b39', '#b37985', '#d9e0e5'], 'return_gap_bins.png', 'followup_n')
