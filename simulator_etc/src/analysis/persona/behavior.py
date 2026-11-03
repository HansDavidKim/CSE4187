"""Persona behavior windows, changes, session delays and heterogeneity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from utils.paths import CAPSTONE_ROOT, PROJECT_ROOT


WINDOW_CAPSTONE = CAPSTONE_ROOT
WINDOW_OUT = PROJECT_ROOT / 'artifacts/early-later-two-datasets'
CROSS_LIFECYCLE_OUT = PROJECT_ROOT / 'artifacts/cross-domain-lifecycle'
CROSS_LIFECYCLE_FIG = PROJECT_ROOT / 'img/persona/cross_domain_lifecycle.png'

COMMON_LIFECYCLE_FEATURES = (
    'log_session_index', 'log_events', 'view_share', 'cart_share',
    'purchase_share', 'log_duration_hours', 'log_previous_gap_hours',
    'has_previous_gap', 'gap_acceleration', 'profile_s1', 'profile_s2',
)

SHARED_LIFECYCLE_PROFILE = {
    'cosmetics': {0: 'S0', 1: 'UNRESOLVED', 2: 'S1', 3: 'S2'},
    'electronics': {0: 'S1', 1: 'S0', 2: 'S1', 3: 'S2'},
}
COSMETICS_SIMULATOR_INTERCEPT_OFFSET = 0.1


def session_tokens(sessions, max_tokens=64):
    if not sessions or max_tokens < 1:
        raise ValueError('nonempty sessions and positive token limit required')
    tokens = []
    for actions in sessions:
        if not actions or any(action not in (0, 1, 2) for action in actions):
            raise ValueError('sessions must contain view/cart/purchase actions')
        tokens.extend(actions)
        tokens.append(3)
    return tokens[:max_tokens]


def summarize_tokens(tokens):
    events = sum(tokens.count(i) for i in range(3))
    if not events:
        raise ValueError('at least one event required')
    return {'views': tokens.count(0), 'carts': tokens.count(1), 'purchases': tokens.count(2),
            'events': events, 'tokens': len(tokens)}


def summarize_windows(events, actions):
    frame = events.copy()
    frame['rank'] = frame.groupby('user', sort=False)['session'].transform(lambda s: pd.factorize(s, sort=False)[0])
    frame = frame[frame['rank'] < 4].copy()
    frame['period'] = np.where(frame['rank'] < 2, 'early', 'later')
    flags = pd.crosstab([frame['user'], frame['period']], frame['action']).reindex(columns=range(len(actions)), fill_value=0)
    flags.columns = actions
    early = flags.xs('early', level='period').sort_index()
    later = flags.xs('later', level='period').reindex(early.index)
    if later.isna().any().any():
        raise ValueError('each user must have four sessions')
    early_view_only = early['view'].gt(0) & early.drop(columns='view').eq(0).all(axis=1)
    later_view_only = later['view'].gt(0) & later.drop(columns='view').eq(0).all(axis=1)
    early_cartbuy = early['cart'].gt(0) | early['purchase'].gt(0)
    later_cartbuy = later['cart'].gt(0) | later['purchase'].gt(0)
    categories = {'early_view_only': early_view_only, 'later_view_only': later_view_only,
                  'early_view_only_later_cart_or_purchase': early_view_only & later_cartbuy,
                  'early_cart_or_purchase_later_view_only': early_cartbuy & later_view_only,
                  'view_only_both_windows': early_view_only & later_view_only,
                  'cart_or_purchase_both_windows': early_cartbuy & later_cartbuy}
    n = len(early)
    outcomes = {}
    for action in actions:
        a, b = early[action].gt(0), later[action].gt(0)
        outcomes[action] = {'early_users': int(a.sum()), 'later_users': int(b.sum()),
                            'neither_then_yes': int((~a & b).sum()),
                            'yes_then_neither': int((a & ~b).sum())}
    return {'users': n, 'counts': {k: int(v.sum()) for k, v in categories.items()},
            'early_view_only_to_later_cart_or_purchase_fraction':
                float((early_view_only & later_cartbuy).sum() / early_view_only.sum()) if early_view_only.sum() else None,
            'actions': outcomes,
            'early_median_events': float(early.sum(axis=1).median()),
            'later_median_events': float(later.sum(axis=1).median())}


def run_window_comparison(source):
    from analysis.persona.sequence import rows_from_events, source_data

    path = WINDOW_CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics' else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(WINDOW_CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if emb.user_id.tolist() != [r[0] for r in rows]:
        raise ValueError('saved embedding cohort differs')
    order = np.random.default_rng(42).permutation(len(rows))
    heldout = order[int(.8 * len(rows)):]
    users = {rows[i][0] for i in heldout if rows[i][2] >= 4}
    result = summarize_windows(events[events.user.isin(users)], actions)
    if result['users'] != len(users):
        raise ValueError('four-session user cohort mismatch')
    result.update({'source': source, 'cohort': 'held-out users in existing 12000 stratified masked-GRU cohort with >=4 observed sessions',
                   'actions_mapping': actions, 'source_quality': quality,
                   'limitations': ['Observed first sessions may not be true onboarding.',
                       'Four sessions required and sampled by total sessions: selected retrospectively, not whole population.',
                       'No proof of causal change, stable preference, uplift or return after observation end.',
                       'Two windows use complete event sequences; prior saved embeddings used first 64 tokens.',
                       'Separate window GRU embeddings unavailable for cosmetics checkpoint in existing files.']})
    WINDOW_OUT.mkdir(parents=True, exist_ok=True)
    (WINDOW_OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, json.dumps({k: result[k] for k in ('users', 'counts', 'actions')}, ensure_ascii=False), flush=True)








DYNAMICS_CAPSTONE = CAPSTONE_ROOT
DYNAMICS_OUT = Path('artifacts/persona-evolution')


def classify(model, checkpoint, sequences):
    from src.irl.masked_sequence_personas import embeddings

    from src.irl.sequence_personas import padded

    tokens, lengths = padded(sequences, max_tokens=64)
    z = embeddings(model, tokens, lengths)
    z = (z - checkpoint['scaler_mean'].numpy()) / checkpoint['scaler_scale'].numpy()
    distances = ((z[:, None, :] - checkpoint['centers'].numpy()[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def profile(frame, label):
    count = len(frame)
    if not count:
        return {'users': 0}
    return {'users': count,
            'early_buyer_share': float(frame.early_purchases.gt(0).mean()),
            'later_buyer_share': float(frame.later_purchases.gt(0).mean()),
            'early_cart_share': float(frame.early_carts.gt(0).mean()),
            'later_cart_share': float(frame.later_carts.gt(0).mean()),
            'early_view_event_share': float(frame.early_views.sum() / frame.early_events.sum()),
            'later_view_event_share': float(frame.later_views.sum() / frame.later_events.sum()),
            'early_median_events': float(frame.early_events.median()),
            'later_median_events': float(frame.later_events.median()),
            'early_at_limit_share': float(frame.early_tokens.eq(64).mean()),
            'later_at_limit_share': float(frame.later_tokens.eq(64).mean()),
            'label': label}


def run_dynamics(source):
    from src.irl.fit import load_events

    from src.irl.masked_sequence_personas import MaskedGRUEncoder

    from src.irl.sequence_personas import sequence_rows

    torch.set_num_threads(1)
    checkpoint = torch.load(DYNAMICS_CAPSTONE / f'data/models/{source}_masked_sequence_personas.pt',
                            map_location='cpu', weights_only=True)
    model = MaskedGRUEncoder()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    events, quality = load_events(source, DYNAMICS_CAPSTONE / f'data/raw/kaggle/{source}/events.csv')
    rows = sequence_rows(events, 12000, 64, 42)
    _, test = train_test_split(np.arange(len(rows)), test_size=.2, random_state=42)
    saved = pd.read_parquet(DYNAMICS_CAPSTONE / f'data/models/{source}_masked_sequence_personas.parquet').set_index('user_id')
    if not saved.index.is_unique or set(saved.index) != {r[0] for r in rows}:
        raise ValueError('Reconstructed cohort differs from checkpoint cohort')
    full_labels = classify(model, checkpoint, [r[1] for r in rows])
    if not np.array_equal(full_labels, saved.loc[[r[0] for r in rows], 'cluster'].to_numpy()):
        raise ValueError('Checkpoint does not reproduce saved full-prefix labels')
    test_users = {rows[i][0] for i in test if rows[i][3] >= 4}
    selected = events[events.user.isin(test_users)]
    early, later, cumulative, metadata = [], [], [], []
    for user, group in selected.groupby('user', sort=False):
        sessions = [g.action.tolist() for _, g in group.groupby('session', sort=False)]
        if len(sessions) < 4:
            raise ValueError('repeat-session count mismatch')
        a, b = session_tokens(sessions[:2]), session_tokens(sessions[2:4])
        both = session_tokens(sessions[:4])
        early.append(a)
        later.append(b)
        cumulative.append(both)
        x, y = summarize_tokens(a), summarize_tokens(b)
        metadata.append({**{'early_' + k: v for k, v in x.items()},
                         **{'later_' + k: v for k, v in y.items()},
                         'cumulative_contains_later': len(a) < 64})
    if len(metadata) != len(test_users):
        raise ValueError('Missing selected held-out users')
    frame = pd.DataFrame(metadata)
    frame['early_cluster'] = classify(model, checkpoint, early)
    frame['later_cluster'] = classify(model, checkpoint, later)
    frame['cumulative_cluster'] = classify(model, checkpoint, cumulative)
    k = int(checkpoint['cluster_count'])
    matrix = pd.crosstab(frame.early_cluster, frame.later_cluster).reindex(
        index=range(k), columns=range(k), fill_value=0)
    cum_matrix = pd.crosstab(frame.early_cluster, frame.cumulative_cluster).reindex(
        index=range(k), columns=range(k), fill_value=0)
    changed = frame.early_cluster.ne(frame.later_cluster)
    paths = []
    for (a, b), subset in frame.groupby(['early_cluster', 'later_cluster']):
        paths.append({'early_cluster': int(a), 'later_cluster': int(b),
                      **profile(subset, f'{a}->{b}')})
    result = {'source': source, 'source_quality': quality, 'checkpoint': str(
        DYNAMICS_CAPSTONE / f'data/models/{source}_masked_sequence_personas.pt'),
        'users': len(frame), 'heldout_user_disjoint_from_encoder_train': True,
        'early_sessions': [1, 2], 'later_sessions': [3, 4],
        'sample_restriction': 'held-out sampled users with >=4 sessions in complete observed record',
        'cluster_count': k, 'early_to_later_counts': matrix.to_numpy().tolist(),
        'early_to_cumulative_counts': cum_matrix.to_numpy().tolist(),
        'changed_early_to_later': int(changed.sum()),
        'changed_early_to_later_share': float(changed.mean()),
        'changed_early_to_cumulative_share': float(frame.early_cluster.ne(frame.cumulative_cluster).mean()),
        'cumulative_contains_later_share': float(frame.cumulative_contains_later.mean()),
        'changed_profile': profile(frame[changed], 'changed'),
        'unchanged_profile': profile(frame[~changed], 'unchanged'),
        'paths': paths,
        'limitations': ['Conditional on 4 observed sessions, not the whole population; selection uses future.',
                        'Separate windows can receive different labels solely because of finite length, shifting observations or model uncertainty.',
                        'Cluster centroids and encoder trained on first-64 full prefixes, not separately on two-session windows.',
                        'Cumulative input is first 64 tokens of first 4 sessions and may not include later sessions.',
                        'Clusters learned separately by dataset; cluster IDs are not equivalent across datasets.',
                        'Session end is inferred and date intervals between windows are unequal.',
                        'Same windows determine clusters and are summarized; no prospective forecast or causal preference change established.']}
    DYNAMICS_OUT.mkdir(parents=True, exist_ok=True)
    (DYNAMICS_OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'source': source, 'users': len(frame), 'matrix': result['early_to_later_counts'],
                      'changed': result['changed_early_to_later_share'],
                      'early_to_cumulative_changed': result['changed_early_to_cumulative_share']}, indent=2))





DELAY_CAPSTONE = CAPSTONE_ROOT
DELAY_OUT = Path('artifacts/persona-delays')
DELAY_FIG = Path('img/analysis')
HOURS = np.array([.5, 1, 6, 24, 72, 168, 336, 720])


def summarize_delays(frame):
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


def run_delays(source):
    import matplotlib.pyplot as plt
    from analysis.persona.sequence import rows_from_events, source_data

    path = DELAY_CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    observation_end = events['at'].max()
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(DELAY_CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
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
        record = {'cluster': cluster, 'all': summarize_delays(part), 'scenarios': {}}
        for scenario in ('view_only', 'cart_or_remove_no_purchase', 'purchase'):
            record['scenarios'][scenario] = summarize_delays(part[part.scenario.eq(scenario)])
        result['clusters'].append(record)
    DELAY_OUT.mkdir(parents=True, exist_ok=True)
    DELAY_FIG.mkdir(parents=True, exist_ok=True)
    (DELAY_OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
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
    fig.savefig(DELAY_FIG / f'{source}_persona_delay.png', dpi=160, bbox_inches='tight')
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
    fig.savefig(DELAY_FIG / f'{source}_persona_delay_scenarios.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)




HETERO_BASE = CAPSTONE_ROOT
HETERO_OUT = Path('artifacts/persona-heterogeneity')
HETERO_FIG = Path('img/heterogeneity')
ACTION_NAMES = ['view', 'cart', 'remove', 'purchase']


def run_temporal_profiles(sources=('electronics', 'cosmetics'), k=2):
    """Describe early/middle/late observed-session behavior for held-out GRU clusters."""
    import matplotlib.pyplot as plt
    def selected_events(source, users):
        root = CAPSTONE_ROOT / 'data/raw/kaggle'
        files = sorted((root / 'cosmetics').glob('20??-*.csv')) if source == 'cosmetics' else [root / 'electronics/events.csv']
        actions = ['view', 'cart', 'remove', 'purchase'] if source == 'cosmetics' else ['view', 'cart', 'purchase']
        mapping = {'view': 0, 'cart': 1, 'remove_from_cart': 2, 'purchase': 3} if source == 'cosmetics' else {'view': 0, 'cart': 1, 'purchase': 2}
        parts, raw = [], 0
        for file in files:
            for chunk in pd.read_csv(file, usecols=['user_id', 'event_time', 'event_type', 'user_session'],
                                     dtype={'user_id': 'Int64', 'user_session': 'string'}, chunksize=500_000):
                raw += len(chunk)
                chunk = chunk[chunk.user_id.isin(users)].copy()
                at, action = pd.to_datetime(chunk.event_time, utc=True, errors='coerce'), chunk.event_type.map(mapping)
                valid = chunk.user_session.notna() & at.notna() & action.notna()
                parts.append(pd.DataFrame({'user': chunk.loc[valid, 'user_id'].astype('int64'),
                                           'session': chunk.loc[valid, 'user_session'], 'at': at[valid],
                                           'action': action[valid].astype('int8')}))
        return pd.concat(parts, ignore_index=True), {'raw_rows': raw, 'selected_rows': sum(map(len, parts))}, actions

    output = PROJECT_ROOT / f'artifacts/persona-temporal-profiles-k{k}'
    figures = PROJECT_ROOT / 'img/persona'
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports, panels = [], []
    for source in sources:
        members = pd.read_parquet(PROJECT_ROOT / f'artifacts/three-ecommerce-personas/{source}_memberships.parquet')
        members = members[(members.split == 'test') & members.full_history_sessions.ge(3)][['user_id', f'k{k}']]
        members = members.rename(columns={'user_id': 'user', f'k{k}': 'cluster'})
        events, quality, actions = selected_events(source, set(members.user))
        frame = events.merge(members, on='user', how='inner', validate='many_to_one')
        observed_repeat = frame.groupby('user', sort=False).session.nunique()
        frame = frame[frame.user.isin(observed_repeat[observed_repeat.ge(3)].index)].copy()
        frame = frame.sort_values(['user', 'at', 'session'], kind='stable')
        sessions = frame[['user', 'session', 'at', 'cluster']].drop_duplicates(['user', 'session'], keep='first')
        sessions = sessions.sort_values(['user', 'at', 'session'], kind='stable')
        sessions['session_rank'] = sessions.groupby('user', sort=False).cumcount()
        sessions['session_total'] = sessions.groupby('user', sort=False).session.transform('size')
        sessions['stage'] = np.minimum(2, (3 * sessions.session_rank / sessions.session_total).astype(int))
        frame = frame.merge(sessions[['user', 'session', 'stage']], on=['user', 'session'], validate='many_to_one')
        rows = []
        for cluster in range(k):
            for stage, name in enumerate(('early', 'middle', 'late')):
                part = frame[(frame.cluster == cluster) & (frame.stage == stage)]
                session_rows = part.groupby(['user', 'session'], sort=False).action.agg(list)
                total = len(part)
                row = {'cluster': cluster, 'stage': name, 'events': int(total),
                       'sessions': int(len(session_rows)), 'users': int(part.user.nunique())}
                for index, action in enumerate(actions):
                    row[f'{action}_share'] = float(part.action.eq(index).mean()) if total else None
                row['purchase_session_rate'] = float(session_rows.map(lambda x: actions.index('purchase') in x).mean()) if len(session_rows) else None
                row['cart_session_rate'] = float(session_rows.map(lambda x: actions.index('cart') in x).mean()) if len(session_rows) else None
                row['median_events_per_session'] = float(session_rows.map(len).median()) if len(session_rows) else None
                rows.append(row)
        profile = pd.DataFrame(rows)
        report = {'source': source, 'k': k, 'source_quality': quality,
                  'cohort': 'held-out users from the existing masked-GRU K=2 clustering with at least three sessions in the currently available raw log',
                  'stage_definition': 'each user\'s observed sessions divided into equal-count early/middle/late thirds',
                  'clusters': profile.to_dict('records'),
                  'limitations': ['Cluster labels are inferred from full observed prefixes, so this is descriptive rather than a prospective validation.',
                                  'Observed first session need not be true onboarding; later stage is not a fixed calendar horizon.',
                                  'Datasets have separate embedding spaces and cluster IDs are not comparable across datasets.']}
        (output / f'{source}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        reports.append(report)
        panels.append((source, profile, actions))

    colors = {'view': '#4e79a7', 'cart': '#f28e2b', 'remove': '#e15759', 'purchase': '#59a14f'}
    fig, axes = plt.subplots(len(panels), k, figsize=(3.2 * k, 4.6 * len(panels)), sharey=False, squeeze=False)
    for row, (source, profile, actions) in enumerate(panels):
        for cluster in range(k):
            ax = axes[row, cluster]
            subset = profile[profile.cluster.eq(cluster)].set_index('stage').loc[['early', 'middle', 'late']]
            bottom = np.zeros(3)
            for action in actions:
                values = subset[f'{action}_share'].to_numpy() * 100
                ax.bar(['Early', 'Middle', 'Late'], values, bottom=bottom, color=colors[action], label=action)
                bottom += values
            ax.set(title=f'{source.title()} · cluster {cluster}', ylim=(0, 100), ylabel='Event share (%)')
            ax.grid(axis='y', alpha=.2)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=colors[action]) for action in ('view', 'cart', 'remove', 'purchase') if action in colors],
               [action for action in ('view', 'cart', 'remove', 'purchase') if action in {a for _, _, acts in panels for a in acts}],
               loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Action composition changes across each user\'s observed session stages')
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.savefig(figures / f'persona_k{k}_stage_action_mix.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(len(panels), k, figsize=(3.2 * k, 4.6 * len(panels)), sharey=True, squeeze=False)
    for row, (source, profile, _) in enumerate(panels):
        for cluster, ax in enumerate(axes[row]):
            subset = profile[profile.cluster.eq(cluster)].set_index('stage').loc[['early', 'middle', 'late']]
            ax.plot(['Early', 'Middle', 'Late'], subset.purchase_session_rate * 100, marker='o', lw=2, label='purchase session')
            ax.plot(['Early', 'Middle', 'Late'], subset.cart_session_rate * 100, marker='s', lw=1.2, ls='--', label='cart session')
            ax.set(title=f'{source.title()} · c{cluster}', ylabel='Sessions containing action (%)', ylim=(0, 100))
            ax.grid(axis='y', alpha=.2)
            if row == 0 and cluster == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Stage-specific session incidence: purchase versus cart')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(figures / f'persona_k{k}_stage_purchase_patterns.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports


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


def run_heterogeneity(source):
    from analysis.persona.sequence import rows_from_events, source_data

    path = HETERO_BASE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                   else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    end = events['at'].max()
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(HETERO_BASE / f'data/models/{source}_masked_embedding_embeddings.parquet')
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
    HETERO_OUT.mkdir(parents=True, exist_ok=True)
    HETERO_FIG.mkdir(parents=True, exist_ok=True)
    (HETERO_OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result, users, second


def stacked(results, key, title, labels, colors, filename, denominator):
    import matplotlib.pyplot as plt
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
    fig.savefig(HETERO_FIG / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)


def boxplots(panels, column, title, filename, ylabel):
    import matplotlib.pyplot as plt
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
    fig.savefig(HETERO_FIG / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_heterogeneity():
    outputs = [run_heterogeneity(s) for s in ('electronics', 'cosmetics')]
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


def lifecycle_window_features(frame):
    """Behavior features for a complete set of observed sessions."""
    actions = ['view', 'cart', 'remove', 'purchase']
    total = float(frame[actions].to_numpy().sum())
    if frame.empty or total <= 0:
        raise ValueError('lifecycle window needs at least one event')
    shares = frame[actions].sum() / total
    gaps = frame.previous_gap_hours.dropna()
    return np.r_[
        shares.to_numpy(), np.log1p(total / len(frame)),
        frame.purchase.gt(0).mean(), frame.cart.gt(0).mean(), frame.remove.gt(0).mean(),
        np.log1p(gaps.median() if len(gaps) else 0),
    ]


def shared_lifecycle_window_features(frame):
    """Window features observed in both Cosmetics and Electronics."""
    actions = ['view', 'cart', 'purchase']
    total = float(frame[actions].to_numpy().sum())
    if frame.empty or total <= 0:
        raise ValueError('shared lifecycle window needs at least one shared event')
    shares = frame[actions].sum() / total
    gaps = frame.previous_gap_hours.dropna()
    return {
        **{f'{action}_share': float(shares[action]) for action in actions},
        'events': int(total),
        'median_gap_hours': float(gaps.median() if len(gaps) else 0),
    }


def lifecycle_decisions(sessions, observation_end):
    """Build past-only 30-day return decisions with right-censoring."""
    frame = sessions.sort_values(['user', 'start'], kind='stable').copy()
    grouped = frame.groupby('user', sort=False)
    frame['session_index'] = grouped.cumcount()
    shared_total = frame[['view', 'cart', 'purchase']].sum(axis=1).clip(lower=1)
    for action in ('view', 'cart', 'purchase'):
        frame[f'{action}_share'] = frame[action] / shared_total
    frame['log_session_index'] = np.log1p(frame.session_index)
    frame['log_events'] = np.log1p(shared_total)
    frame['log_duration_hours'] = np.log1p(frame.duration_hours.clip(lower=0))
    frame['has_previous_gap'] = frame.previous_gap_hours.notna().astype(float)
    frame['log_previous_gap_hours'] = np.log1p(
        frame.previous_gap_hours.fillna(0).clip(lower=0)
    )
    prior_log_gap = grouped['previous_gap_hours'].shift().fillna(0).clip(lower=0).pipe(np.log1p)
    frame['gap_acceleration'] = (
        frame.log_previous_gap_hours - prior_log_gap
    ).clip(lower=-np.log1p(720), upper=np.log1p(720))
    followup_hours = (observation_end - frame.end).dt.total_seconds() / 3600
    observed_return = frame.next_gap_hours.notna()
    frame['no_return_30d'] = np.where(
        observed_return, frame.next_gap_hours.ge(720),
        np.where(followup_hours.ge(720), True, np.nan),
    )
    return frame.loc[frame.no_return_30d.notna()].assign(
        no_return_30d=lambda x: x.no_return_30d.astype(int)
    )


def _lifecycle_metrics(actual, probability):
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = probability >= .5
    return {
        'rows': int(len(actual)),
        'prevalence': float(actual.mean()),
        'predicted_prevalence': float(probability.mean()),
        'roc_auc': float(roc_auc_score(actual, probability)),
        'average_precision': float(average_precision_score(actual, probability)),
        'brier': float(brier_score_loss(actual, probability)),
        'nll': float(log_loss(actual, np.c_[1 - probability, probability], labels=[0, 1])),
        'accuracy': float((predicted == actual).mean()),
    }


def _fit_lifecycle_head(train, test, sample_weight=None, seed=42):
    scaler = StandardScaler().fit(train[list(COMMON_LIFECYCLE_FEATURES)])
    estimator = LogisticRegression(max_iter=2000, random_state=seed)
    estimator.fit(
        scaler.transform(train[list(COMMON_LIFECYCLE_FEATURES)]),
        train.no_return_30d,
        sample_weight=sample_weight,
    )
    probability = estimator.predict_proba(
        scaler.transform(test[list(COMMON_LIFECYCLE_FEATURES)])
    )[:, 1]
    return estimator, scaler, _lifecycle_metrics(test.no_return_30d, probability)


def _calibrate_intercept(linear_predictor, target_rate):
    """Match one domain's observed prevalence while preserving shared slopes."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        middle = (lo + hi) / 2
        probability = 1 / (1 + np.exp(-np.clip(linear_predictor + middle, -30, 30)))
        if probability.mean() < target_rate:
            lo = middle
        else:
            hi = middle
    return float((lo + hi) / 2)


def _logit(value):
    value = np.clip(float(value), 1e-9, 1 - 1e-9)
    return float(np.log(value / (1 - value)))


def long_term_buyer_analysis(decisions):
    """Calibrate a rare repeated-purchase subtype without redefining home personas."""
    user_rows, profile_rows = [], []
    for source, frame in decisions.groupby('dataset', sort=False):
        users = frame.groupby('user', sort=False).agg(
            sessions=('session', 'size'),
            purchase_sessions=('purchase', lambda values: values.gt(0).sum()),
            purchases=('purchase', 'sum'), first_seen=('start', 'min'), last_seen=('end', 'max'),
        )
        users['span_days'] = (
            users.last_seen - users.first_seen
        ).dt.total_seconds() / 86400
        users['long_term_buyer'] = (
            users.sessions.ge(10) & users.purchase_sessions.ge(3) & users.span_days.ge(30)
        )
        users = users.reset_index().assign(dataset=source)
        user_rows.append(users)
        joined = frame.merge(users[['user', 'long_term_buyer']], on='user')
        profile = (joined.groupby('long_term_buyer', as_index=False)
                   .agg(users=('user', 'nunique'), sessions=('session', 'size'),
                        purchase_session_rate=('purchase', lambda values: values.gt(0).mean()),
                        no_return_30d_rate=('no_return_30d', 'mean'),
                        median_previous_gap_hours=('previous_gap_hours', 'median')))
        profile.insert(0, 'dataset', source)
        profile_rows.append(profile)
    users = pd.concat(user_rows, ignore_index=True)
    profiles = pd.concat(profile_rows, ignore_index=True)

    original = pd.read_parquet(
        PROJECT_ROOT / 'artifacts/cosmetics-shop-personas/clusters.parquet'
    )[['user_id', 'cluster']].rename(columns={'user_id': 'user', 'cluster': 'original_persona'})
    cosmetics = decisions.loc[decisions.dataset.eq('cosmetics')].merge(original, on='user')
    cosmetics_users = users.loc[users.dataset.eq('cosmetics'), ['user', 'long_term_buyer']]
    cosmetics = cosmetics.merge(cosmetics_users, on='user')
    p2 = cosmetics.loc[cosmetics.original_persona.eq(2)]
    p2_profiles = (p2.groupby('long_term_buyer', as_index=False)
                   .agg(users=('user', 'nunique'), sessions=('session', 'size'),
                        purchase_session_rate=('purchase', lambda values: values.gt(0).mean()),
                        no_return_30d_rate=('no_return_30d', 'mean'),
                        median_previous_gap_hours=('previous_gap_hours', 'median')))
    by_type = p2_profiles.set_index('long_term_buyer')
    purchase_offset = _logit(by_type.loc[True, 'purchase_session_rate']) - _logit(
        by_type.loc[False, 'purchase_session_rate']
    )
    nonreturn_offset = _logit(by_type.loc[True, 'no_return_30d_rate']) - _logit(
        by_type.loc[False, 'no_return_30d_rate']
    )
    p2_users = cosmetics[['user', 'original_persona', 'long_term_buyer']].drop_duplicates()
    p2_users = p2_users.loc[p2_users.original_persona.eq(2)]
    observed_conditional_probability = float(p2_users.long_term_buyer.mean())
    cosmetics_users = users.loc[users.dataset.eq('cosmetics')]
    population_probability = float(cosmetics_users.long_term_buyer.mean())
    gap_model = json.loads((
        PROJECT_ROOT / 'artifacts/cosmetics-shop-persona-quadratic-irl/gap_token_irl_model.json'
    ).read_text(encoding='utf-8'))
    simulator_p2_probability = float(gap_model['lifecycle']['home_persona_prior']['2'])
    sampling_probability = min(population_probability / simulator_p2_probability, 1.0)
    return users, profiles, p2_profiles, {
        'name': 'P_LT', 'parent_home_persona': 2,
        'operational_definition': 'at least 10 sessions, 3 purchase sessions, and 30 observed days',
        'population_probability': population_probability,
        'observed_conditional_probability_within_p2': observed_conditional_probability,
        'sampling_probability_within_p2': sampling_probability,
        'purchase_logit_offset': purchase_offset,
        'no_return_logit_offset': nonreturn_offset,
        'rollout_validation': {
            'users': 10000, 'seed': 20261027,
            'generated_users': 82, 'generated_share': 0.0082,
            'generated_purchase_session_rate': 0.052283,
            'generated_no_return_30d_rate': 0.043680,
        },
        'calibration_only': (
            'retrospective full-history subtype used as a simulator latent prior; '
            'not available as an online observed label'
        ),
    }


def _long_term_stage_row(dataset, user, frame):
    """Summarize early, middle, and late thirds of one LT user's eligible sessions."""
    frame = frame.sort_values('start', kind='stable')
    stage_index = np.minimum(np.arange(len(frame)) * 3 // len(frame), 2)
    row = {'dataset': dataset, 'user': user, 'eligible_sessions': len(frame)}
    for index, stage in enumerate(('early', 'middle', 'late')):
        part = frame.iloc[np.flatnonzero(stage_index == index)]
        action_total = float(part[list(ACTION_NAMES)].to_numpy().sum())
        for action in ACTION_NAMES:
            row[f'{stage}_{action}_share'] = float(part[action].sum() / action_total)
        row[f'{stage}_events_per_session'] = float(part.events.mean())
        row[f'{stage}_median_previous_gap_hours'] = float(part.previous_gap_hours.median())
        row[f'{stage}_purchase_session_rate'] = float(part.purchase.gt(0).mean())
    final = frame.iloc[-1]
    row['terminal_state'] = (
        'DISSATISFIED_CHURNING' if final.no_return_30d == 1 and final.purchase == 0 else
        'SATIATED_EXIT' if final.no_return_30d == 1 else 'OBSERVED_RETURN'
    )
    row['view_share_change'] = row['late_view_share'] - row['early_view_share']
    row['purchase_share_change'] = row['late_purchase_share'] - row['early_purchase_share']
    row['event_volume_ratio'] = (
        row['late_events_per_session'] / row['early_events_per_session']
    )
    early_gap = max(row['early_median_previous_gap_hours'], 1 / 60)
    row['gap_ratio'] = row['late_median_previous_gap_hours'] / early_gap
    row['view_shift_10pp'] = row['view_share_change'] >= .10
    row['purchase_share_decline'] = row['purchase_share_change'] < 0
    row['event_volume_decline'] = row['event_volume_ratio'] < 1
    row['prior_gap_doubled'] = row['gap_ratio'] >= 2
    row['view_led_churning_proxy'] = (
        row['terminal_state'] == 'DISSATISFIED_CHURNING'
        and row['view_shift_10pp'] and row['event_volume_decline']
    )
    return row


def run_long_term_churn_analysis(output=CROSS_LIFECYCLE_OUT):
    """Check whether retail LT users become view-heavy before observed non-return."""
    import matplotlib.pyplot as plt

    output = Path(output)
    decisions = pd.read_parquet(output / 'return_decisions.parquet')
    users = pd.read_parquet(output / 'long_term_buyer_users.parquet')
    long_term = users.loc[users.long_term_buyer, ['dataset', 'user']]
    frame = decisions.merge(long_term, on=['dataset', 'user'], validate='many_to_one')
    rows = [
        _long_term_stage_row(dataset, user, part)
        for (dataset, user), part in frame.groupby(['dataset', 'user'], sort=True)
    ]
    changes = pd.DataFrame(rows)

    stages = ('early', 'middle', 'late')
    profiles = []
    for (dataset, terminal), part in changes.groupby(['dataset', 'terminal_state'], sort=True):
        for stage in stages:
            profiles.append({
                'dataset': dataset, 'terminal_state': terminal, 'stage': stage,
                'users': len(part),
                'median_view_share': float(part[f'{stage}_view_share'].median()),
                'median_purchase_share': float(part[f'{stage}_purchase_share'].median()),
                'median_events_per_session': float(
                    part[f'{stage}_events_per_session'].median()
                ),
                'median_previous_gap_hours': float(
                    part[f'{stage}_median_previous_gap_hours'].median()
                ),
            })
    profiles = pd.DataFrame(profiles)

    signal_rows = []
    for dataset, part in changes.groupby('dataset', sort=True):
        churn = part.loc[part.terminal_state.eq('DISSATISFIED_CHURNING')]
        signal_rows.append({
            'dataset': dataset,
            'long_term_users': len(part),
            'dissatisfied_churning_users': len(churn),
            'satiated_exit_users': int(part.terminal_state.eq('SATIATED_EXIT').sum()),
            'view_shift_10pp_all_users': int(part.view_shift_10pp.sum()),
            'event_volume_decline_all_users': int(part.event_volume_decline.sum()),
            'view_shift_10pp_among_churning': int(churn.view_shift_10pp.sum()),
            'event_volume_decline_among_churning': int(churn.event_volume_decline.sum()),
            'view_led_churning_proxy_users': int(churn.view_led_churning_proxy.sum()),
            'view_led_churning_proxy_share_of_churning': (
                float(churn.view_led_churning_proxy.mean()) if len(churn) else None
            ),
            'purchase_share_decline_among_churning': int(
                churn.purchase_share_decline.sum()
            ),
            'prior_gap_doubled_among_churning': int(churn.prior_gap_doubled.sum()),
        })
    signals = pd.DataFrame(signal_rows)
    report = {
        'long_term_definition': (
            'at least 10 eligible sessions, 3 purchase sessions, and 30 observed days'
        ),
        'stage_definition': 'chronological eligible sessions split into user-level thirds',
        'terminal_definition': {
            'DISSATISFIED_CHURNING': 'non-purchase session followed by observed 30-day non-return',
            'SATIATED_EXIT': 'purchase session followed by observed 30-day non-return',
            'OBSERVED_RETURN': 'a subsequent session is observed within 30 days',
        },
        'view_led_churning_proxy': (
            'DISSATISFIED_CHURNING plus a late-minus-early view share increase of at least '
            '10 percentage points and lower late events per session'
        ),
        'signals': signals.to_dict(orient='records'),
        'limitations': [
            'The LT label and stage split use retrospective history and are simulator calibration labels, not online states.',
            'The terminal label requires observable 30-day follow-up and does not establish permanent churn.',
            'Electronics has only seven LT users, so counts are reported without population inference.',
            'A view-share increase alone is not a reliable churn signal; terminal non-return and activity contraction are required by the proxy.',
        ],
    }
    changes.to_csv(output / 'long_term_stage_changes.csv', index=False)
    profiles.to_csv(output / 'long_term_stage_profiles.csv', index=False)
    signals.to_csv(output / 'long_term_churn_signals.csv', index=False)
    (output / 'long_term_churn_summary.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )

    figure = PROJECT_ROOT / 'img/persona/long_term_churn_progression.png'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    colors = {
        'DISSATISFIED_CHURNING': '#e15759',
        'SATIATED_EXIT': '#59a14f',
        'OBSERVED_RETURN': '#4e79a7',
    }
    for column, dataset in enumerate(('cosmetics', 'electronics')):
        source = profiles.loc[profiles.dataset.eq(dataset)]
        for terminal, part in source.groupby('terminal_state', sort=True):
            part = part.set_index('stage').reindex(stages)
            label = f"{terminal.replace('_', ' ').title()} (n={int(part.users.iloc[0])})"
            axes[0, column].plot(stages, part.median_view_share * 100, marker='o',
                                 color=colors[terminal], label=label)
            axes[1, column].plot(stages, part.median_events_per_session, marker='o',
                                 color=colors[terminal], label=label)
        axes[0, column].set_title(dataset.title())
        axes[0, column].set_ylabel('Median view share (%)')
        axes[1, column].set_ylabel('Median events per session')
        axes[1, column].set_xlabel('Sequence stage')
        for axis in axes[:, column]:
            axis.grid(axis='y', alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
            axis.legend(frameon=False, fontsize=7)
    fig.savefig(figure, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return report


def _lifecycle_source(source, raw, memberships_path, sample_modulus):
    """Apply one lifecycle definition to one retail log."""
    from analysis.irl.persona import session_summaries
    from analysis.persona.sequence import source_data

    events, quality, actions = source_data(source, Path(raw), sample_modulus)
    canonical = {name: index for index, name in enumerate(('view', 'cart', 'remove', 'purchase'))}
    action_map = {index: canonical[name] for index, name in enumerate(actions)}
    events = events.copy()
    events['action'] = events.action.map(action_map).astype('int8')
    memberships = pd.read_parquet(memberships_path)[['user_id', 'cluster']]
    sessions, observation_end = session_summaries(events, memberships)
    sessions = sessions.sort_values(['user', 'start'], kind='stable')
    observation_start = events['at'].min()
    groups = {int(user): frame for user, frame in sessions.groupby('user', sort=False)}
    users = sessions.groupby('user', sort=False).agg(
        first_seen=('start', 'min'), last_seen=('end', 'max'), sessions=('session', 'size')
    ).reset_index()
    cohort = users.loc[
        users.first_seen.ge(observation_start + pd.Timedelta(days=30))
        & users.first_seen.le(observation_end - pd.Timedelta(days=60))
        & users.sessions.ge(6)
    ]
    trajectory_rows = []
    for user_row in cohort.itertuples(index=False):
        frame = groups[int(user_row.user)]
        cluster = int(frame.persona.iloc[0])
        shared_profile = SHARED_LIFECYCLE_PROFILE[source][cluster]
        windows = {'early': frame.iloc[:2], 'middle': frame.iloc[2:4], 'late': frame.iloc[-2:]}
        no_return = observation_end - frame.end.iloc[-1] >= pd.Timedelta(days=30)
        final_purchase = bool(frame.purchase.iloc[-1] > 0)
        terminal = (
            'SATIATED_EXIT_CANDIDATE' if no_return and final_purchase else
            'DISSATISFIED_CHURNING' if no_return else 'RIGHT_CENSORED'
        )
        for stage, window in windows.items():
            trajectory_rows.append({
                'dataset': source, 'user': int(user_row.user), 'stage': stage,
                'cluster': cluster, 'shared_profile': shared_profile,
                'terminal_state': terminal,
                **shared_lifecycle_window_features(window),
            })
    trajectories = pd.DataFrame(trajectory_rows)
    profiles = (trajectories.groupby(['dataset', 'terminal_state', 'stage'], as_index=False)
                 .agg(users=('user', 'nunique'), view_share=('view_share', 'mean'),
                      cart_share=('cart_share', 'mean'),
                      purchase_share=('purchase_share', 'mean'),
                      median_events=('events', 'median'),
                      median_gap_hours=('median_gap_hours', 'median')))
    decisions = lifecycle_decisions(sessions, observation_end)
    decisions.insert(0, 'dataset', source)
    decisions['shared_profile'] = decisions.persona.map(SHARED_LIFECYCLE_PROFILE[source])
    decisions['profile_s1'] = decisions.shared_profile.eq('S1').astype(float)
    decisions['profile_s2'] = decisions.shared_profile.eq('S2').astype(float)
    delayed = decisions.loc[decisions.next_gap_hours.ge(720)]
    terminal_counts = (trajectories[['user', 'terminal_state']].drop_duplicates()
                       .terminal_state.value_counts().to_dict())
    return {
        'quality': quality,
        'observation_start': observation_start,
        'observation_end': observation_end,
        'cohort_users': int(len(cohort)),
        'terminal_counts': {key: int(value) for key, value in terminal_counts.items()},
        'delayed_return_sessions': int(len(delayed)),
        'delayed_return_purchase_sessions': int(delayed.purchase.gt(0).sum()),
    }, trajectories, profiles, decisions


def run_cross_domain_lifecycle(raw_paths=None, output=CROSS_LIFECYCLE_OUT, seed=42):
    """Validate and fit the coupon-free lifecycle layer across both retail logs."""
    raw_paths = raw_paths or {
        'cosmetics': PROJECT_ROOT / 'data/raw/kaggle/cosmetics-shop',
        'electronics': PROJECT_ROOT / 'data/raw/kaggle/electronics/events.csv',
    }
    membership_paths = {
        source: PROJECT_ROOT / f'artifacts/cross-dataset-persona-similarity/{source}_k4_memberships.parquet'
        for source in ('cosmetics', 'electronics')
    }
    source_results = {}
    trajectories, profiles, decisions = [], [], []
    for source in ('cosmetics', 'electronics'):
        result = _lifecycle_source(
            source, raw_paths[source], membership_paths[source],
            100 if source == 'cosmetics' else 1,
        )
        source_results[source] = result[0]
        trajectories.append(result[1])
        profiles.append(result[2])
        decisions.append(result[3])
    trajectories = pd.concat(trajectories, ignore_index=True)
    profiles = pd.concat(profiles, ignore_index=True)
    decisions = pd.concat(decisions, ignore_index=True)
    model_decisions = decisions.loc[decisions.shared_profile.ne('UNRESOLVED')].copy()

    persona_purchase_profiles = (decisions.loc[decisions.shared_profile.ne('UNRESOLVED')]
        .assign(purchase_session=lambda x: x.purchase.gt(0),
                satiated_exit_candidate=lambda x: x.no_return_30d.eq(1) & x.purchase.gt(0),
                dissatisfied_churning=lambda x: x.no_return_30d.eq(1) & x.purchase.eq(0))
        .groupby(['dataset', 'shared_profile'], as_index=False)
        .agg(sessions=('session', 'size'), users=('user', 'nunique'),
             purchase_session_rate=('purchase_session', 'mean'),
             no_return_30d_rate=('no_return_30d', 'mean'),
             satiated_exit_candidate_rate=('satiated_exit_candidate', 'mean'),
             dissatisfied_churning_rate=('dissatisfied_churning', 'mean')))
    persona_terminal_profiles = (trajectories[
        ['dataset', 'user', 'shared_profile', 'terminal_state']
    ].drop_duplicates()
        .groupby(['dataset', 'shared_profile', 'terminal_state'], as_index=False)
        .size().rename(columns={'size': 'users'}))
    persona_terminal_profiles['share'] = (
        persona_terminal_profiles.users
        / persona_terminal_profiles.groupby(['dataset', 'shared_profile']).users.transform('sum')
    )
    long_term_users, long_term_profiles, long_term_p2_profiles, long_term_specification = (
        long_term_buyer_analysis(decisions)
    )

    evaluations = {}
    for train_source, test_source in (
        ('cosmetics', 'electronics'), ('electronics', 'cosmetics')
    ):
        _, _, metrics = _fit_lifecycle_head(
            model_decisions.loc[model_decisions.dataset.eq(train_source)],
            model_decisions.loc[model_decisions.dataset.eq(test_source)], seed=seed,
        )
        evaluations[f'{train_source}_to_{test_source}'] = metrics
    for source in ('cosmetics', 'electronics'):
        frame = model_decisions.loc[model_decisions.dataset.eq(source)].copy()
        heldout = pd.util.hash_pandas_object(
            source + ':' + frame.user.astype(str), index=False
        ).mod(5).eq(0)
        _, _, metrics = _fit_lifecycle_head(frame.loc[~heldout], frame.loc[heldout], seed=seed)
        evaluations[f'{source}_heldout'] = metrics

    domain_weight = model_decisions.dataset.map(
        model_decisions.dataset.value_counts().rdiv(len(model_decisions) / 2)
    ).to_numpy()
    estimator, scaler, pooled_metrics = _fit_lifecycle_head(
        model_decisions, model_decisions, sample_weight=domain_weight, seed=seed
    )
    coefficients = dict(zip(COMMON_LIFECYCLE_FEATURES, estimator.coef_[0], strict=True))
    standardized = scaler.transform(model_decisions[list(COMMON_LIFECYCLE_FEATURES)])
    linear_predictor = standardized @ estimator.coef_[0]
    domain_intercepts = {
        source: _calibrate_intercept(
            linear_predictor[model_decisions.dataset.eq(source).to_numpy()],
            model_decisions.loc[model_decisions.dataset.eq(source), 'no_return_30d'].mean(),
        )
        for source in ('cosmetics', 'electronics')
    }
    model = {
        'states': ['NEW', 'HOME_PERSONA', 'DISSATISFIED_CHURNING', 'SATIATED_EXIT', 'EXIT'],
        'home_persona_rule': 'sampled once at birth and restored after reactivation',
        'new_rule': 'first two simulated sessions; observed first sessions are only a proxy',
        'dissatisfied_churning_rule': 'no session within 30 days after a non-purchase session',
        'satiated_exit_rule': (
            'purchase followed by no session within 30 days; satisfaction is a latent simulator '
            'interpretation and the observed label remains SATIATED_EXIT_CANDIDATE'
        ),
        'simulator_compatibility_alias': {'SATIATED_EXIT': 'MISSION_EXIT'},
        'transitions': {
            'NEW': ['NEW', 'HOME_PERSONA', 'DISSATISFIED_CHURNING', 'SATIATED_EXIT'],
            'HOME_PERSONA': ['HOME_PERSONA', 'DISSATISFIED_CHURNING', 'SATIATED_EXIT'],
            'DISSATISFIED_CHURNING': ['HOME_PERSONA', 'EXIT'],
            'SATIATED_EXIT': ['SATIATED_EXIT'],
            'EXIT': ['EXIT'],
        },
        'no_return_head': {
            'features': list(COMMON_LIFECYCLE_FEATURES),
            'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
            'pooled_intercept': float(estimator.intercept_[0]),
            'domain_intercepts': domain_intercepts,
            'simulator_intercepts': {
                'cosmetics': domain_intercepts['cosmetics']
                + COSMETICS_SIMULATOR_INTERCEPT_OFFSET,
            },
            'coefficients': coefficients,
            'fit': 'dataset-balanced pooled logistic slopes with prevalence-calibrated domain intercepts',
            'simulator_calibration': {
                'cosmetics_intercept_offset': COSMETICS_SIMULATOR_INTERCEPT_OFFSET,
                'selection_users': 2000, 'selection_seed': 20261026,
                'target_no_return_30d_rate': 0.26243093922651933,
                'validation_users': 10000, 'validation_seed': 20261027,
                'validation_no_return_30d_rate': 0.26324740898219506,
                'validation_absolute_error': 0.0008164697556757305,
                'validation_satiated_share_among_nonreturns': 0.05007066424389259,
                'target_satiated_share_among_nonreturns': 0.0491514991646883,
            },
        },
        'long_term_buyer': long_term_specification,
    }
    summary = {
        'cohort_definition': 'first observed 30d after log start, 60d before log end, >=6 sessions',
        'sources': {
            source: {
                **{key: value for key, value in result.items()
                   if key not in ('observation_start', 'observation_end')},
                'observation_start': result['observation_start'].isoformat(),
                'observation_end': result['observation_end'].isoformat(),
            }
            for source, result in source_results.items()
        },
        'evaluation': evaluations,
        'pooled_in_sample_diagnostic': pooled_metrics,
        'shared_profile_mapping': {
            'rule': 'Electronics clusters map to their nearest Cosmetics shared-action profile; '
                    'S0/S1/S2 are cross-domain validation profiles, not simulator home-persona labels; '
                    'Cosmetics C1 is unresolved because it has no distinct counterpart',
            'mapping': SHARED_LIFECYCLE_PROFILE,
        },
        'persona_purchase_profiles': persona_purchase_profiles.to_dict(orient='records'),
        'persona_terminal_profiles': persona_terminal_profiles.to_dict(orient='records'),
        'long_term_buyer_profiles': long_term_profiles.to_dict(orient='records'),
        'long_term_buyer_p2_profiles': long_term_p2_profiles.to_dict(orient='records'),
        'model': model,
        'limitations': [
            'NEW denotes first observed sessions because neither dataset records signup.',
            'Electronics has no remove event, so the shared transition head excludes removal.',
            'SATIATED_EXIT_CANDIDATE is purchase plus 30-day non-return, not observed satisfaction.',
            'The pooled fit is predictive and observational; it does not identify causal lifecycle effects.',
            'Pooled in-sample metrics diagnose fit and are not a generalization estimate.',
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trajectories.to_parquet(output / 'trajectories.parquet', index=False)
    profiles.to_csv(output / 'stage_profiles.csv', index=False)
    decisions.to_parquet(output / 'return_decisions.parquet', index=False)
    persona_purchase_profiles.to_csv(output / 'persona_purchase_profiles.csv', index=False)
    persona_terminal_profiles.to_csv(output / 'persona_terminal_profiles.csv', index=False)
    long_term_users.to_parquet(output / 'long_term_buyer_users.parquet', index=False)
    long_term_profiles.to_csv(output / 'long_term_buyer_profiles.csv', index=False)
    long_term_p2_profiles.to_csv(output / 'long_term_buyer_p2_profiles.csv', index=False)
    (output / 'lifecycle_model.json').write_text(json.dumps(model, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _plot_cross_domain_lifecycle(profiles, persona_purchase_profiles, evaluations)
    return summary


def _plot_cross_domain_lifecycle(profiles, persona_purchase_profiles, evaluations):
    import matplotlib.pyplot as plt

    stages = ['early', 'middle', 'late']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    axes = axes.ravel()
    colors = {'cosmetics': '#4e79a7', 'electronics': '#e15759'}
    for source in ('cosmetics', 'electronics'):
        frame = profiles.loc[
            profiles.dataset.eq(source) & profiles.terminal_state.eq('DISSATISFIED_CHURNING')
        ].set_index('stage').reindex(stages)
        axes[0].plot(stages, frame.median_gap_hours, marker='o', linewidth=2,
                     color=colors[source], label=source.title())
    axes[0].set_yscale('symlog', linthresh=1)
    axes[0].set(ylabel='Median previous gap (hours)', title='Pre-churn gap progression')
    axes[0].legend(frameon=False)
    terminal_order = ['DISSATISFIED_CHURNING', 'SATIATED_EXIT_CANDIDATE']
    x = np.arange(len(terminal_order))
    for offset, source in zip((-.18, .18), ('cosmetics', 'electronics'), strict=True):
        frame = profiles.loc[
            profiles.dataset.eq(source) & profiles.stage.eq('late')
        ].set_index('terminal_state').reindex(terminal_order)
        axes[1].bar(x + offset, frame.purchase_share, width=.34, color=colors[source],
                    label=source.title())
    axes[1].set_xticks(x, ['Dissatisfied\nchurning', 'Satiated-exit\ncandidate'])
    axes[1].set(ylabel='Late-window purchase share', title='Operational exit separation')
    axes[1].legend(frameon=False)

    persona_order = ['S0', 'S1', 'S2']
    x = np.arange(len(persona_order))
    for offset, source in zip((-.18, .18), ('cosmetics', 'electronics'), strict=True):
        frame = persona_purchase_profiles.loc[
            persona_purchase_profiles.dataset.eq(source)
        ].set_index('shared_profile').reindex(persona_order)
        axes[2].bar(x + offset, frame.purchase_session_rate, width=.34,
                    color=colors[source], label=source.title())
    axes[2].set_xticks(x, persona_order)
    axes[2].set(ylabel='Purchase-session rate', title='Cross-domain profile purchase pattern')

    names = ['Cosmetics\nheld-out', 'Electronics\nheld-out',
             'Cosmetics→\nElectronics', 'Electronics→\nCosmetics']
    keys = ['cosmetics_heldout', 'electronics_heldout',
            'cosmetics_to_electronics', 'electronics_to_cosmetics']
    values = [evaluations[key]['roc_auc'] for key in keys]
    axes[3].bar(names, values, color=['#4e79a7', '#e15759', '#76b7b2', '#f28e2b'])
    axes[3].axhline(.5, color='#666666', linestyle='--', linewidth=1)
    axes[3].set_ylim(.45, 1)
    axes[3].set(ylabel='ROC-AUC', title='30-day non-return transfer')
    for index, value in enumerate(values):
        axes[3].text(index, value + .015, f'{value:.2f}', ha='center', fontsize=8)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    CROSS_LIFECYCLE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CROSS_LIFECYCLE_FIG, dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_lifecycle(
    raw=PROJECT_ROOT / 'data/raw/kaggle/cosmetics-shop',
    embeddings_path=PROJECT_ROOT / 'artifacts/cosmetics-shop-personas/clusters.parquet',
    output=PROJECT_ROOT / 'artifacts/cosmetics-lifecycle',
    sample_modulus=100,
    seed=42,
):
    """Describe observed entry, mature behavior windows, and terminal non-return."""
    from analysis.irl.persona import session_summaries
    from analysis.persona.sequence import source_data

    feature_names = [
        'view_share', 'cart_share', 'remove_share', 'purchase_share',
        'log_events_per_session', 'purchase_session_rate', 'cart_session_rate',
        'remove_session_rate', 'log_median_gap',
    ]
    events, quality, _ = source_data('cosmetics', Path(raw), sample_modulus)
    embeddings = pd.read_parquet(embeddings_path)
    sessions, observation_end = session_summaries(events, embeddings[['user_id', 'cluster']])
    sessions = sessions.sort_values(['user', 'start'], kind='stable')
    groups = {int(user): frame for user, frame in sessions.groupby('user', sort=False)}
    users = pd.DataFrame([
        {
            'user': user, 'stored_cluster': int(frame.persona.iloc[0]),
            'sessions': len(frame), 'first_seen': frame.start.iloc[0],
            'last_seen': frame.end.iloc[-1],
            **dict(zip(feature_names, lifecycle_window_features(frame))),
        }
        for user, frame in groups.items()
    ])
    mature = users.loc[users.stored_cluster.isin((0, 2, 3))]
    scaler = StandardScaler().fit(mature[feature_names])
    x, y = scaler.transform(mature[feature_names]), mature.stored_cluster.to_numpy()
    estimator = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    predicted = cross_val_predict(estimator, x, y, cv=folds)
    estimator.fit(x, y)
    observation_start = events['at'].min()
    cohort = users.loc[
        users.first_seen.ge(observation_start + pd.Timedelta(days=30))
        & users.first_seen.le(observation_end - pd.Timedelta(days=60))
        & users.sessions.ge(6)
    ]
    rows = []
    for user_row in cohort.itertuples(index=False):
        frame = groups[int(user_row.user)]
        windows = {'early': frame.iloc[:2], 'middle': frame.iloc[2:4], 'late': frame.iloc[-2:]}
        no_return = observation_end - frame.end.iloc[-1] >= pd.Timedelta(days=30)
        final_purchase = bool(frame.purchase.iloc[-1] > 0)
        row = {
            'user': int(user_row.user), 'stored_cluster': int(user_row.stored_cluster),
            'sessions': len(frame), 'terminal_state': (
                'PURCHASE_EXIT_CANDIDATE' if no_return and final_purchase else
                'CHURNING' if no_return else 'RIGHT_CENSORED'
            ),
        }
        for stage, window in windows.items():
            values = lifecycle_window_features(window)
            probability = estimator.predict_proba(scaler.transform(
                pd.DataFrame([values], columns=feature_names)
            ))[0]
            row[f'{stage}_state'] = f'P{estimator.classes_[probability.argmax()]}'
            row[f'{stage}_confidence'] = float(probability.max())
            for name, value in zip(feature_names, values):
                row[f'{stage}_{name}'] = float(value)
            row[f'{stage}_events'] = int(window[ACTION_NAMES].to_numpy().sum())
        rows.append(row)
    trajectories = pd.DataFrame(rows)
    transitions = []
    for left, right in (('early', 'middle'), ('middle', 'late')):
        table = trajectories.groupby([f'{left}_state', f'{right}_state']).size().rename('users').reset_index()
        table['probability'] = table.users / table.groupby(f'{left}_state').users.transform('sum')
        table.insert(0, 'transition', f'{left}_to_{right}')
        transitions.append(table.rename(columns={f'{left}_state': 'from_state', f'{right}_state': 'to_state'}))
    transitions = pd.concat(transitions, ignore_index=True)
    stage_profiles = []
    for terminal, terminal_frame in trajectories.groupby('terminal_state'):
        for stage in ('early', 'middle', 'late'):
            stage_profiles.append({
                'terminal_state': terminal, 'stage': stage, 'users': len(terminal_frame),
                **{f'{action}_share': float(terminal_frame[f'{stage}_{action}_share'].mean())
                   for action in ACTION_NAMES},
                'median_events': float(terminal_frame[f'{stage}_events'].median()),
                'median_gap_hours': float(np.expm1(
                    terminal_frame[f'{stage}_log_median_gap'].median()
                )),
            })
    stage_profiles = pd.DataFrame(stage_profiles)
    terminal_counts = trajectories.terminal_state.value_counts()
    summary = {
        'data': quality, 'observation_start': observation_start.isoformat(),
        'observation_end': observation_end.isoformat(), 'cohort_users': len(trajectories),
        'cohort_definition': (
            'first observed at least 30 days after log start and 60 days before log end, '
            'with at least six observed sessions'
        ),
        'mature_window_classifier': {
            'classes': estimator.classes_.tolist(),
            'cross_validated_accuracy': float(accuracy_score(y, predicted)),
            'confusion_matrix_labels_0_2_3': confusion_matrix(y, predicted, labels=[0, 2, 3]).tolist(),
            'mean_window_confidence': {
                stage: float(trajectories[f'{stage}_confidence'].mean())
                for stage in ('early', 'middle', 'late')
            },
        },
        'early_to_late_changed_share': float(
            trajectories.early_state.ne(trajectories.late_state).mean()
        ),
        'terminal_counts': {key: int(value) for key, value in terminal_counts.items()},
        'transitions': transitions.to_dict(orient='records'),
        'stage_profiles': stage_profiles.to_dict(orient='records'),
        'limitations': [
            'First observed activity is not a recorded signup and may be a returning customer.',
            'Requiring six sessions selects retained and more active users using future information.',
            'P0/P2/P3 window states are predictions from behavior summaries, not observed identities.',
            'Two-session windows are shorter than the full histories used to train the classifier.',
            'PURCHASE_EXIT_CANDIDATE means final-session purchase followed by observed 30-day non-return; intent is unknown.',
            'RIGHT_CENSORED means the final session lacks 30 days of follow-up, not confirmed activity.',
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trajectories.to_parquet(output / 'trajectories.parquet', index=False)
    transitions.to_csv(output / 'transitions.csv', index=False)
    stage_profiles.to_csv(output / 'stage_profiles.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze persona behavior, delays and heterogeneity.')
    parser.add_argument('task', choices=(
        'windows', 'dynamics', 'delays', 'heterogeneity', 'temporal',
        'lifecycle', 'lifecycle-cross-domain', 'long-term-churn',
    ))
    parser.add_argument('sources', nargs='*', choices=('electronics', 'cosmetics'))
    parser.add_argument('--k', type=int, default=2)
    args = parser.parse_args()
    if args.task == 'lifecycle-cross-domain':
        if args.sources:
            parser.error('cross-domain lifecycle uses Cosmetics and Electronics together')
        run_cross_domain_lifecycle()
    elif args.task == 'long-term-churn':
        if args.sources:
            parser.error('long-term churn uses saved Cosmetics and Electronics lifecycle data')
        run_long_term_churn_analysis()
    elif args.task == 'lifecycle':
        if args.sources:
            parser.error('lifecycle currently uses the Cosmetics source only')
        run_lifecycle()
    elif args.task == 'heterogeneity':
        if args.sources and args.sources != ['electronics', 'cosmetics']:
            parser.error('heterogeneity plots require electronics and cosmetics in that order')
        plot_heterogeneity()
    elif args.task == 'temporal':
        if args.sources and tuple(args.sources) != ('electronics', 'cosmetics'):
            parser.error('temporal profiles require electronics and cosmetics in that order')
        if args.k < 2 or args.k > 8:
            parser.error('--k must be between 2 and 8')
        run_temporal_profiles(k=args.k)
    else:
        if not args.sources:
            parser.error('specify at least one dataset')
        if args.task == 'dynamics' and args.sources != ['electronics']:
            parser.error('the available dynamics checkpoint is for electronics only')
        runner = {'windows': run_window_comparison, 'dynamics': run_dynamics, 'delays': run_delays}[args.task]
        for source in args.sources:
            runner(source)
