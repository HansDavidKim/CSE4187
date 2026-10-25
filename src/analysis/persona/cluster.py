"""Clustering, K-sweeps, and stability of persona embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from utils.paths import CAPSTONE_ROOT, PROJECT_ROOT


CLUSTER_ROOT = PROJECT_ROOT
CLUSTER_CAPSTONE = CAPSTONE_ROOT
CLUSTER_OUT = CLUSTER_ROOT / 'artifacts/two-dataset-gru-clusters'
FOUR_MODE_OUT = CLUSTER_ROOT / 'artifacts/four-behavior-modes'


def _capstone_models() -> Path:
    """Return the external model cache without silently retraining an encoder."""
    candidate = CLUSTER_CAPSTONE / 'data/models'
    if candidate.exists():
        return candidate
    fallback = Path('/Users/kimdaewon/Desktop/캡스톤디자인/data/models')
    if fallback.exists():
        return fallback
    raise FileNotFoundError('Set ECOMMERCE_CAPSTONE_ROOT to the project holding saved embeddings.')


def _selected_events(source: str, user_ids: set[int]) -> pd.DataFrame:
    """Load only saved-embedding users, keeping chronological within-session events."""
    root = _capstone_models().parent.parent
    path = root / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                   else 'data/raw/kaggle/electronics/events.csv')
    files = sorted(path.glob('20??-*.csv')) if source == 'cosmetics' else [path]
    action_map = ({'view': 'view', 'cart': 'cart', 'remove_from_cart': 'remove', 'purchase': 'purchase'}
                  if source == 'cosmetics' else {'view': 'view', 'cart': 'cart', 'purchase': 'purchase'})
    parts = []
    for file in files:
        for chunk in pd.read_csv(file, usecols=['user_id', 'event_time', 'event_type', 'user_session'],
                                 dtype={'user_id': 'Int64', 'user_session': 'string'}, chunksize=500_000):
            selected = chunk.user_id.isin(user_ids)
            if not selected.any():
                continue
            chunk = chunk.loc[selected].copy()
            at = pd.to_datetime(chunk.event_time, utc=True, errors='coerce')
            action = chunk.event_type.map(action_map)
            valid = chunk.user_id.notna() & chunk.user_session.notna() & at.notna() & action.notna()
            parts.append(pd.DataFrame({'user_id': chunk.loc[valid, 'user_id'].astype('int64'),
                                       'at': at[valid], 'session': chunk.loc[valid, 'user_session'],
                                       'action': action[valid]}))
    events = pd.concat(parts, ignore_index=True)
    return events.sort_values(['user_id', 'at'], kind='stable')


def _prefix_profile(events: pd.DataFrame, membership: pd.DataFrame, source: str) -> dict[int, dict]:
    """Profile the same first-64-token action prefix used by the saved encoder."""
    actions = ['view', 'cart', 'remove', 'purchase']
    action_index = {action: index for index, action in enumerate(actions)}
    labels = dict(zip(membership.user_id.astype('int64'), membership.stored_k6.astype(int)))
    counts = {cluster: np.zeros(len(actions), dtype=int) for cluster in sorted(membership.stored_k6.unique())}
    pairs = {cluster: np.zeros((len(actions), len(actions)), dtype=int)
             for cluster in sorted(membership.stored_k6.unique())}
    per_user = {cluster: [] for cluster in counts}
    for user_id, frame in events.groupby('user_id', sort=False):
        cluster = labels.get(int(user_id))
        if cluster is None:
            continue
        tokens, previous = [], None
        for session, action in zip(frame.session, frame.action):
            if previous is not None and session != previous:
                tokens.append(None)
            tokens.append(action)
            previous = session
        tokens = tokens[:64]
        sequence = [action_index[token] if token is not None else None for token in tokens]
        actual = [token for token in sequence if token is not None]
        if actual:
            counts[cluster] += np.bincount(actual, minlength=len(actions))
        per_user[cluster].append(len(actual))
        for left, right in zip(sequence, sequence[1:]):
            if left is not None and right is not None:
                pairs[cluster][left, right] += 1
    result = {}
    for cluster in counts:
        total = int(counts[cluster].sum())
        total_pairs = int(pairs[cluster].sum())
        ranked = sorted(((int(pairs[cluster][i, j]), actions[i], actions[j])
                         for i in range(len(actions)) for j in range(len(actions))), reverse=True)
        result[int(cluster)] = {
            'prefix_action_share': {action: float(counts[cluster][index] / total) if total else 0.0
                                    for index, action in enumerate(actions)},
            'prefix_median_actions': float(np.median(per_user[cluster])) if per_user[cluster] else 0.0,
            'top_transitions': [{'from': left, 'to': right, 'share': float(count / total_pairs), 'count': count}
                                for count, left, right in ranked if count][:5],
            'cart_to_purchase_share': float(pairs[cluster][1, 3] / total_pairs) if total_pairs else 0.0,
        }
    return result


def build_four_modes() -> dict:
    """Persist a K-sweep and choose four supported, behaviorally separated active modes.

    The GRU is never fitted here: this consumes the existing user_id--48D embedding
    mapping and the existing K=6 assignments.  K=6 is used only to define profile
    atoms after K=2..8 stability/support checks, not because silhouette alone chose it.
    """
    import matplotlib.pyplot as plt

    models = _capstone_models()
    FOUR_MODE_OUT.mkdir(parents=True, exist_ok=True)
    figures = PROJECT_ROOT / 'img/persona'
    figures.mkdir(parents=True, exist_ok=True)
    # Each atom has >= 400 users. Purchase path explicitly requires observed purchase.
    mode_atoms = {
        'P1_short_observation': {'electronics': [3], 'cosmetics': [0]},
        'P2_deep_consideration': {'cosmetics': [1]},
        'P3_cart_revision': {'cosmetics': [2]},
        'P4_purchase_path': {'electronics': [0, 4], 'cosmetics': [5]},
    }
    profile_reports, memberships, sweeps = {}, {}, {}
    for source in ('electronics', 'cosmetics'):
        emb = pd.read_parquet(models / f'{source}_masked_embedding_embeddings.parquet')
        stored = pd.read_parquet(models / f'{source}_masked_embedding_embeddings_k6.parquet')
        if not emb.user_id.equals(stored.user_id):
            raise ValueError(f'{source}: saved embedding and K=6 mapping do not align')
        z_columns = [f'z_{index}' for index in range(48)]
        z = StandardScaler().fit_transform(emb[z_columns].to_numpy())
        mature = np.flatnonzero(emb.sessions.to_numpy() >= 2)
        mapping = emb[['user_id', 'sessions']].copy()
        mapping['stored_k6'] = stored.cluster.to_numpy()
        source_sweep = []
        with threadpool_limits(limits=1):
            for k in range(2, 9):
                first = KMeans(n_clusters=k, n_init=20, random_state=42).fit(z[mature])
                alternate = KMeans(n_clusters=k, n_init=20, random_state=43).fit(z[mature])
                labels = first.predict(z)
                mapping[f'k{k}'] = labels
                sample = mature[:min(2000, len(mature))]
                source_sweep.append({
                    'k': k,
                    'seed_ari': float(adjusted_rand_score(labels, alternate.predict(z))),
                    'silhouette_mature_sample': float(silhouette_score(z[sample], first.predict(z[sample]))),
                    'smallest_cluster': int(np.bincount(labels, minlength=k).min()),
                })
        mapping['mode'] = 'unassigned_active'
        for mode, by_source in mode_atoms.items():
            if source in by_source:
                mapping.loc[mapping.stored_k6.isin(by_source[source]), 'mode'] = mode
        mapping.to_parquet(FOUR_MODE_OUT / f'{source}_embedding_memberships.parquet', index=False)
        memberships[source] = mapping
        sweeps[source] = source_sweep
        report = json.loads((models / f'{source}_masked_embedding_clusters_k6.json').read_text(encoding='utf-8'))
        profile_reports[source] = {profile['cluster']: profile for profile in report['profiles']}

    prefix = {}
    for source, mapping in memberships.items():
        needed = set(mapping.loc[mapping['mode'].ne('unassigned_active'), 'user_id'].astype(int))
        prefix[source] = _prefix_profile(_selected_events(source, needed), mapping, source)

    selected = []
    for mode, by_source in mode_atoms.items():
        for source, clusters in by_source.items():
            for cluster in clusters:
                saved = profile_reports[source][cluster]
                row = {'mode': mode, 'source': source, 'stored_k6_cluster': cluster,
                       'users': saved['users'], 'median_events': saved['median_events'],
                       'mean_sessions': saved['mean_sessions'], 'buyer_share': saved['buyer_share'],
                       'event_share': saved['event_share'], **prefix[source][cluster]}
                selected.append(row)
    # A weighted behavioral centroid, not an embedding-space comparison across datasets.
    rows = []
    for mode in mode_atoms:
        pieces = [row for row in selected if row['mode'] == mode]
        weight = sum(row['users'] for row in pieces)
        def weighted(key): return sum(row['users'] * row[key] for row in pieces) / weight
        rows.append({'mode': mode, 'users': weight,
                     'view': sum(row['users'] * row['event_share'].get('view', 0) for row in pieces) / weight,
                     'cart': sum(row['users'] * row['event_share'].get('cart', 0) for row in pieces) / weight,
                     'remove': sum(row['users'] * row['event_share'].get('remove', 0) for row in pieces) / weight,
                     'purchase': sum(row['users'] * row['event_share'].get('purchase', 0) for row in pieces) / weight,
                     'buyer_share': weighted('buyer_share'),
                     'log_median_events': sum(row['users'] * np.log1p(row['median_events']) for row in pieces) / weight})
    mode_frame = pd.DataFrame(rows)
    feature_columns = ['view', 'cart', 'remove', 'purchase', 'buyer_share', 'log_median_events']
    standardized = StandardScaler().fit_transform(mode_frame[feature_columns])
    distance = np.sqrt(((standardized[:, None, :] - standardized[None, :, :]) ** 2).sum(axis=2))

    # Action composition: show the actual cluster atoms rather than hiding source differences.
    action_order = ['view', 'cart', 'remove', 'purchase']
    colors = {'view': '#4e79a7', 'cart': '#f28e2b', 'remove': '#e15759', 'purchase': '#59a14f'}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    labels = [f"{row['mode']}\n{row['source'][0].upper()}-c{row['stored_k6_cluster']} (n={row['users']:,})" for row in selected]
    left = np.zeros(len(selected))
    for action in action_order:
        values = np.array([row['event_share'].get(action, 0) for row in selected]) * 100
        ax.barh(labels, values, left=left, color=colors[action], label=action)
        left += values
    ax.invert_yaxis(); ax.set(xlim=(0, 100), xlabel='Action share in saved embedding profile (%)',
                               title='Four active modes: K=6 centroid atoms and action composition')
    ax.grid(axis='x', alpha=.2); ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(.5, -0.22))
    fig.tight_layout(); fig.savefig(figures / 'four_mode_action_profiles.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # Include cart→purchase even when it is not a top-three edge, so purchase is auditable.
    fig, ax = plt.subplots(figsize=(11, 6.4))
    transition_rows = []
    for row in selected:
        edges = row['top_transitions'][:3]
        if row['mode'] == 'P4_purchase_path':
            purchase_edge = next((edge for edge in row['top_transitions'] if edge['from'] == 'cart' and edge['to'] == 'purchase'), None)
            if purchase_edge and purchase_edge not in edges:
                edges = edges[:2] + [purchase_edge]
        for edge in edges:
            transition_rows.append((f"{row['mode']} · {row['source'][0].upper()}c{row['stored_k6_cluster']}",
                                    f"{edge['from']}→{edge['to']}", edge['share'] * 100))
    y = np.arange(len(transition_rows))
    ax.barh(y, [item[2] for item in transition_rows], color='#4e79a7')
    ax.set_yticks(y, [f'{item[0]}: {item[1]}' for item in transition_rows], fontsize=8)
    ax.invert_yaxis(); ax.set(xlabel='Share of within-session adjacent pairs (%)',
                               title='Observed transitions of the selected K=6 atoms')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout(); fig.savefig(figures / 'four_mode_transitions.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(distance, cmap='magma')
    names = mode_frame['mode'].str.replace('_', ' ').tolist()
    ax.set_xticks(range(4), names, rotation=25, ha='right'); ax.set_yticks(range(4), names)
    for i in range(4):
        for j in range(4): ax.text(j, i, f'{distance[i, j]:.2f}', ha='center', va='center', color='white' if distance[i, j] > 1.5 else 'black')
    fig.colorbar(image, ax=ax, label='Euclidean distance after feature standardization')
    ax.set_title('Mode separation in observed behavioral-profile space')
    fig.tight_layout(); fig.savefig(figures / 'four_mode_separation.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    result = {'method': {'embedding': 'existing saved 48D masked-GRU embedding; no retraining',
                         'sweep': 'K-means K=2..8 over saved embeddings; seed stability and support recorded',
                         'selection': 'K=6 profile atoms with >=400 users; selected modes must differ in action mix, purchase/cart-removal behavior, and/or activity depth'},
              'sweep': sweeps, 'mode_atoms': mode_atoms, 'selected_atoms': selected,
              'mode_behavior_centroids': mode_frame.to_dict(orient='records'),
              'pairwise_behavior_distance': distance.round(4).tolist(),
              'unassigned_users_are_not_forced_into_a_mode': True,
              'reserved_future_modes': ['churn_progression', 'new_entrant']}
    (FOUR_MODE_OUT / 'selection.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def render_cosmetics_mode_distribution() -> dict:
    """Report Cosmetics as the primary dataset without re-fitting embeddings or clusters."""
    import matplotlib.pyplot as plt

    membership = pd.read_parquet(FOUR_MODE_OUT / 'cosmetics_embedding_memberships.parquet')
    ordered = ['P1_short_observation', 'P2_deep_consideration',
               'P3_cart_revision', 'P4_purchase_path']
    counts = membership['mode'].value_counts()
    total = len(membership)
    assigned = int(counts.reindex(ordered, fill_value=0).sum())
    result = {
        'dataset': 'cosmetics', 'total_saved_embedding_users': total,
        'assigned_users': assigned, 'assigned_coverage': assigned / total,
        'all_users': {mode: int(counts.get(mode, 0)) for mode in ordered + ['unassigned_active']},
        'assigned_only_share': {mode: float(counts.get(mode, 0) / assigned) for mode in ordered},
        'note': 'unassigned_active is retained rather than forcing weakly matched users into one of four core modes',
    }
    FOUR_MODE_OUT.mkdir(parents=True, exist_ok=True)
    (FOUR_MODE_OUT / 'cosmetics_distribution.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    labels = ['P1\nshort observation', 'P2\ndeep consideration',
              'P3\ncart revision', 'P4\npurchase path', 'unassigned\nactive']
    values = [result['all_users'][mode] for mode in ordered] + [result['all_users']['unassigned_active']]
    colors = ['#4e79a7', '#76b7b2', '#f28e2b', '#59a14f', '#bab0ab']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(labels, values, color=colors)
    axes[0].set(title='All saved Cosmetics users', ylabel='Users')
    for index, value in enumerate(values):
        axes[0].text(index, value + total * .012, f'{value:,}\n{value / total:.1%}', ha='center', fontsize=9)
    selected_values = values[:-1]
    axes[1].bar(labels[:-1], selected_values, color=colors[:-1])
    axes[1].set(title=f'Four-mode subset (coverage {assigned / total:.1%})', ylabel='Users')
    for index, value in enumerate(selected_values):
        axes[1].text(index, value + assigned * .018, f'{value / assigned:.1%}', ha='center', fontsize=9)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(axis='x', labelsize=8)
    fig.suptitle('Cosmetics is the primary persona dataset; Electronics is supplementary')
    fig.tight_layout()
    figures = PROJECT_ROOT / 'img/persona'
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / 'cosmetics_four_mode_distribution.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return result


def _fixed_mahalanobis_assignment(z: np.ndarray, core_labels: np.ndarray, core_mask: np.ndarray,
                                  modes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-prior nearest-centroid assignment using a pooled shrinkage covariance."""
    scaler = StandardScaler().fit(z[core_mask])
    scaled = scaler.transform(z)
    core_scaled = scaled[core_mask]
    centroids = np.vstack([core_scaled[core_labels == mode].mean(axis=0) for mode in modes])
    prototype_index = np.array([modes.index(mode) for mode in core_labels])
    residuals = core_scaled - centroids[prototype_index]
    precision = LedoitWolf(assume_centered=True).fit(residuals).precision_
    deltas = scaled[:, None, :] - centroids[None, :, :]
    distance = np.einsum('nkd,de,nke->nk', deltas, precision, deltas)
    predicted = distance.argmin(axis=1)
    return predicted, distance, centroids, precision


def assign_all_cosmetics_to_fixed_personas() -> dict:
    """Force all Cosmetics embeddings to one fixed persona with regularized Mahalanobis distance."""
    modes = ['P1_short_observation', 'P2_deep_consideration',
             'P3_cart_revision', 'P4_purchase_path']
    mapping = pd.read_parquet(FOUR_MODE_OUT / 'cosmetics_embedding_memberships.parquet')
    models = _capstone_models()
    embedding = pd.read_parquet(models / 'cosmetics_masked_embedding_embeddings.parquet')
    if not mapping.user_id.equals(embedding.user_id):
        raise ValueError('Cosmetics membership and saved embedding user order do not align')
    z = embedding[[f'z_{index}' for index in range(48)]].to_numpy()
    core_mask = mapping['mode'].isin(modes).to_numpy()
    core_labels = mapping.loc[core_mask, 'mode'].to_numpy()
    predicted, distance, centroids, precision = _fixed_mahalanobis_assignment(z, core_labels, core_mask, modes)
    ordered = np.sort(distance, axis=1)
    result = mapping[['user_id', 'sessions', 'mode']].copy()
    result['forced_persona'] = np.array(modes, dtype=object)[predicted]
    result['distance_margin'] = ordered[:, 1] - ordered[:, 0]
    for index, mode in enumerate(modes):
        result[f'd2_{mode}'] = distance[:, index]
    FOUR_MODE_OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(FOUR_MODE_OUT / 'cosmetics_fixed_persona_assignment.parquet', index=False)

    core_prediction = result.loc[core_mask, 'forced_persona'].to_numpy()
    metadata = {
        'classification': 'equal-prior nearest fixed centroid under a pooled Ledoit-Wolf shrinkage Mahalanobis metric',
        'core_anchor_users': int(core_mask.sum()),
        'core_anchor_recovery_in_sample': float(np.mean(core_prediction == core_labels)),
        'forced_distribution': result.forced_persona.value_counts().reindex(modes, fill_value=0).to_dict(),
        'median_distance_margin': float(result.distance_margin.median()),
    }
    (FOUR_MODE_OUT / 'cosmetics_fixed_persona_assignment.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return metadata


def render_representative_profiles(sources=('electronics', 'cosmetics'), k=6):
    """Render qualitatively distinct held-out K-means profiles across two datasets."""
    import matplotlib.pyplot as plt

    figures = PROJECT_ROOT / 'img/persona'
    figures.mkdir(parents=True, exist_ok=True)
    selected = []
    selected_ids = {'electronics': (1, 5, 4, 2), 'cosmetics': (2, 0)}
    for source in sources:
        report = json.loads((CLUSTER_OUT / f'{source}.json').read_text(encoding='utf-8'))
        run = next(item for item in report['runs'] if item['method'] == 'kmeans' and item['k'] == k)
        by_cluster = {group['cluster']: group for group in run['groups']}
        groups = [by_cluster[cluster] for cluster in selected_ids[source]]
        selected.append({'source': source, 'k': k, 'holdout_seed_ari': run['holdout_seed_ari'], 'groups': groups})
    colors = {'view': '#4e79a7', 'cart': '#f28e2b', 'remove': '#e15759', 'purchase': '#59a14f'}
    fig, axes = plt.subplots(1, len(selected), figsize=(12.5, 4.7), sharex=True)
    for ax, result in zip(np.atleast_1d(axes), selected):
        groups = result['groups']
        labels = [f"c{group['cluster']} (n={group['users']})" for group in groups]
        left = np.zeros(len(groups))
        actions = list(groups[0]['event_share'])
        for action in actions:
            values = np.array([group['event_share'][action] for group in groups]) * 100
            ax.barh(labels, values, left=left, color=colors[action], label=action)
            left += values
        ax.invert_yaxis()
        ax.set(title=f"{result['source'].title()} · selected K=6 profiles", xlabel='Event share (%)', xlim=(0, 100))
        ax.grid(axis='x', alpha=.2)
    from matplotlib.patches import Patch
    visible = [action for action in ('view', 'cart', 'remove', 'purchase')
               if any(action in group['event_share'] for result in selected for group in result['groups'])]
    fig.legend([Patch(color=colors[action]) for action in visible], visible,
               loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Qualitatively distinct held-out GRU-cluster behavior profiles')
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(figures / 'persona_representative_k6_profiles.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, len(selected), figsize=(12.5, 5.2), sharex=True)
    rank_colors = ['#4e79a7', '#f28e2b', '#59a14f']
    for ax, result in zip(np.atleast_1d(axes), selected):
        groups = result['groups']
        base = np.arange(len(groups))
        for rank, color in enumerate(rank_colors):
            values = [group['top_pairs'][rank]['share'] * 100 if len(group['top_pairs']) > rank else 0 for group in groups]
            labels = [f"{group['top_pairs'][rank]['from']}→{group['top_pairs'][rank]['to']}" if len(group['top_pairs']) > rank else '' for group in groups]
            ypos = base + (rank - 1) * .23
            ax.barh(ypos, values, height=.2, color=color, label=f'top {rank + 1}')
            for y, value, label in zip(ypos, values, labels):
                if value:
                    ax.text(value + .8, y, label, va='center', fontsize=8)
        ax.set_yticks(base, [f"c{group['cluster']} (n={group['users']})" for group in groups])
        ax.invert_yaxis()
        ax.set(title=result['source'].title(), xlabel='Share of all within-session adjacent pairs (%)')
        ax.grid(axis='x', alpha=.2)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Top three within-session transitions for each representative profile')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(figures / 'persona_representative_k6_transitions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    destination = PROJECT_ROOT / 'artifacts/persona-representatives.json'
    destination.write_text(json.dumps(selected, indent=2), encoding='utf-8')
    return selected


def profile(tokens, actions, labels, k):
    records = np.array([[seq.count(i) for i in range(len(actions))] for seq in tokens], dtype=int)
    result = []
    for cluster in range(k):
        group = records[labels == cluster]
        n = len(group)
        total = group.sum()
        pairs = np.zeros((len(actions), len(actions)), dtype=int)
        for seq, label in zip(tokens, labels):
            if label == cluster:
                for a, b in zip(seq, seq[1:]):
                    if a < len(actions) and b < len(actions):
                        pairs[a, b] += 1
        top_pairs = sorted(((int(pairs[i, j]), actions[i], actions[j])
                            for i in range(len(actions)) for j in range(len(actions))), reverse=True)[:3]
        result.append({'cluster': cluster, 'users': n,
                       'event_share': {a: float(group[:, i].sum() / total) if total else None
                                       for i, a in enumerate(actions)},
                       'users_with_action': {a: float((group[:, i] > 0).mean()) if n else None
                                             for i, a in enumerate(actions)},
                       'median_actions': float(np.median(group.sum(axis=1))) if n else None,
                       'within_session_adjacent_pairs': int(pairs.sum()),
                       'top_pairs': [{'from': a, 'to': b, 'count': count,
                                      'share': float(count / pairs.sum())} for count, a, b in top_pairs if count]})
    assert sum(x['users'] for x in result) == len(tokens)
    return result


def run_clustering(source):
    from analysis.persona.sequence import rows_from_events, source_data

    path = CLUSTER_CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics' else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(CLUSTER_CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if emb.user_id.tolist() != [r[0] for r in rows]:
        raise ValueError('saved embedding rows do not match reconstructed tokens')
    z = emb[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions = np.array([r[2] for r in rows])
    rng = np.random.default_rng(42)
    order = rng.permutation(len(rows))
    train, test = order[:int(.8 * len(rows))], order[int(.8 * len(rows)):]
    train, test = train[sessions[train] >= 2], test[sessions[test] >= 2]
    subset = np.random.default_rng(97).choice(train, min(1800, len(train)), replace=False)
    ztrain, ztest = z[subset], z[test]
    tokens = [rows[i][1] for i in test]
    graph = kneighbors_graph(ztrain, n_neighbors=10, include_self=False)
    result = {'source': source, 'source_quality': quality, 'actions': actions,
              'selected_users': len(rows), 'training_repeat_users': len(train),
              'fit_sample_users': len(subset), 'heldout_repeat_users': len(test), 'runs': []}
    with threadpool_limits(limits=1):
        for k in (3, 4, 5, 6):
            for method in ('kmeans', 'gmm_diag', 'knn_graph_ward'):
                stability = None
                if method == 'kmeans':
                    first = KMeans(n_clusters=k, n_init=10, random_state=42).fit(ztrain)
                    alt = KMeans(n_clusters=k, n_init=10, random_state=43).fit(ztrain)
                    labels = first.predict(ztest)
                    stability = adjusted_rand_score(labels, alt.predict(ztest))
                elif method == 'gmm_diag':
                    first = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3,
                                            n_init=3, max_iter=150, random_state=42).fit(ztrain)
                    alt = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3,
                                          n_init=3, max_iter=150, random_state=43).fit(ztrain)
                    labels = first.predict(ztest)
                    stability = adjusted_rand_score(labels, alt.predict(ztest))
                else:
                    fitted = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=graph).fit_predict(ztrain)
                    labels = KNeighborsClassifier(n_neighbors=1).fit(ztrain, fitted).predict(ztest)
                result['runs'].append({'method': method, 'k': k,
                                       'holdout_seed_ari': float(stability) if stability is not None else None,
                                       'groups': profile(tokens, actions, labels, k)})
                print(source, method, k, [g['users'] for g in result['runs'][-1]['groups']], flush=True)
    CLUSTER_OUT.mkdir(parents=True, exist_ok=True)
    (CLUSTER_OUT / f'{source}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


def report():
    docs = ['# Electronics·Cosmetics GRU 임베딩 기반 사용자 행동 군집', '',
            '## 목적과 방법', '',
            '기존 양방향 GRU masked-token 인코더의 저장된 48차원 임베딩을 데이터셋별로 재사용했다. '
            'BERT 자체가 아니라 양방향 GRU에 masked-token 복원 목적을 적용한 모델이다. '
            '입력은 행동 토큰과 세션 경계 토큰이며 앞 최대 64개 토큰이다. 모델 학습과 임베딩은 새로 수행하지 않았다.', '',
            '기존 사용자 표본 및 80/20 사용자 분할을 재구성하고, 전체 이력 기준 세션 2개 이상인 표본에서 '
            '학습 1,800명 이하로 군집을 학습해 보류 사용자에게 할당했다. '
            'K-means, 대각 공분산 GMM, 10-NN 연결 그래프 Ward 계층 군집을 K=3,4,5,6으로 비교한다. '
            'KNN 자체는 비지도 군집이 아니며 Ward의 보류 사용자 할당에는 1-NN을 사용한다. '
            'ARI는 동일 사용자에서 두 초기화의 할당 일치도이며 encoder·표본 변동 안정성은 아니다.', '',
            '각 비율은 보류 표본의 입력 prefix에서 측정한 기술 통계다. 행동 비율은 이벤트 가중이고, '
            '행동 경험 비율은 사용자 단위로 서로 중복 가능하다. 인접 행동 비율은 해당 군집의 세션 내 인접 행동쌍 전체가 분모다. '
            '이 prefix 자체가 군집에도 사용되므로 독립적 미래 검증이 아니다.', '']
    for source in ('electronics', 'cosmetics'):
        d = json.loads((CLUSTER_OUT / f'{source}.json').read_text())
        docs += [f'## {source.title()}', '',
                 f"표본 {d['selected_users']:,}명 중 반복 세션 조건 충족 보류 사용자 {d['heldout_repeat_users']:,}명; 군집 학습 {d['fit_sample_users']:,}명.", '']
        reference = next(r for r in d['runs'] if r['method'] == 'kmeans' and r['k'] == 4)
        substantial = [g for g in reference['groups'] if g['users'] >= 30]
        most_view = max(substantial, key=lambda g: g['event_share']['view'])
        most_buy = max(substantial, key=lambda g: g['users_with_action']['purchase'])
        docs += [f"K-means K=4의 기술적 예시: 군집 {most_view['cluster']}은 조회가 이벤트의 "
                 f"{100*most_view['event_share']['view']:.1f}%인 조회 중심 군집이며, "
                 f"군집 {most_buy['cluster']}은 관측 prefix 구매 경험 비율이 "
                 f"{100*most_buy['users_with_action']['purchase']:.1f}%로 30명 이상 군집 중 가장 높다. "
                 '이 기준은 패턴 설명을 위한 예시이지 K=4가 최적이라는 주장은 아니다.', '']
        if source == 'cosmetics':
            most_remove = max(substantial, key=lambda g: g['users_with_action']['remove'])
            docs += [f"장바구니 삭제 경험 비율은 군집 {most_remove['cluster']}에서 "
                     f"{100*most_remove['users_with_action']['remove']:.1f}%로 높았다. "
                     '삭제 원인이나 불만족 여부는 관측하지 않았다.', '']
        for run in d['runs']:
            ari = '해당 없음' if run['holdout_seed_ari'] is None else f"{run['holdout_seed_ari']:.3f}"
            docs += [f"### {run['method']}, K={run['k']} (초기화 ARI: {ari})", '',
                     '| 군집 | 사용자 수 | 중앙 행동 수 | 조회 비율 | 장바구니 경험 | 구매 경험 | 삭제 경험 | 주요 세션 내 인접 행동 |',
                     '|---|---:|---:|---:|---:|---:|---:|---|']
            for g in run['groups']:
                def pct(x): return '—' if x is None else f'{100*x:.1f}%'
                pairs = ', '.join(f"{p['from']}→{p['to']} {pct(p['share'])}" for p in g['top_pairs'][:2]) or '—'
                docs.append(f"| {g['cluster']} | {g['users']} | {g['median_actions'] if g['median_actions'] is not None else '—'} | "
                            f"{pct(g['event_share'].get('view'))} | {pct(g['users_with_action'].get('cart'))} | "
                            f"{pct(g['users_with_action'].get('purchase'))} | {pct(g['users_with_action'].get('remove'))} | {pairs} |")
            docs.append('')
    docs += ['## 해석 및 논문 수록 시 주의', '',
             '- 표본은 전체 모집단 대표 표본이 아니다. 세션 수에 따른 층화 표본이며 Cosmetics는 사용자 ID modulo 100 표본이다.',
             '- 2개 이상 세션 조건은 사용자 전체 관측 이력을 활용한 사후 조건이다. 온라인 시점에 이 정보를 미리 사용할 수 없다.',
             '- 기존 로더는 session ID 결측 행을 제외한다. 중복·세션 정의 및 64개 토큰 절단도 분포에 영향을 준다.',
             '- 데이터셋별 인코더와 임베딩 공간이 달라 군집 번호나 좌표를 직접 대응시키지 않는다.',
             '- 10-NN 그래프에 연결되지 않은 컴포넌트가 있으면 scikit-learn이 연결을 보완한다. 그래프 군집의 독립성·안정성을 확인한 것은 아니다.',
             '- 사용자 임베딩과 행동 프로파일이 동일 prefix를 사용하므로 차이는 기술적 요약일 뿐 독립 검증 또는 원인 설명이 아니다.',
             '- 그룹을 진짜 선호·만족도·구매 의도나 이탈, uplift 유형으로 해석하지 않는다. 같은 사용자 반복/시간별/인코더 재학습 안정성과 미래 행동 검증은 수행하지 않았다.',
             '- 논문에는 데이터 탐색과 시뮬레이터 동기 부여로 포함할 수 있으나, 표본 선정·토큰화·군집 설정과 한계를 명시하고 가능하면 후속 시점 평가를 추가한다.', '']
    (CLUSTER_OUT / 'REPORT.md').write_text('\n'.join(docs), encoding='utf-8')







SWEEP_CAPSTONE = CAPSTONE_ROOT
SWEEP_OUT = Path('artifacts/three-ecommerce-personas')


def run_sweep(source: str):
    from analysis.persona.sequence import rows_from_events, source_data

    path = SWEEP_CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    embeddings = pd.read_parquet(SWEEP_CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if embeddings.user_id.tolist() != [row[0] for row in rows]:
        raise ValueError(f'{source}: reconstructed sample differs from saved embeddings')
    z = embeddings[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions = np.array([row[2] for row in rows])
    permutation = np.random.default_rng(42).permutation(len(rows))
    train, test = permutation[:int(.8 * len(rows))], permutation[int(.8 * len(rows)):]
    split = np.full(len(rows), 'train', dtype=object)
    split[test] = 'test'
    train = train[sessions[train] >= 2]
    test = test[sessions[test] >= 2]
    profiles = []
    for i in test:
        tokens = rows[i][1]
        record = {name: tokens.count(j) for j, name in enumerate(actions)}
        record['tokens'] = len(tokens)
        for origin, dest in [('view', 'cart'), ('cart', 'purchase'), ('cart', 'remove')]:
            if dest not in actions:
                continue
            a, b = actions.index(origin), actions.index(dest)
            pairs = list(zip(tokens, tokens[1:]))
            record[origin + '_next_count'] = sum(x == a for x, _ in pairs)
            record[origin + '_to_' + dest] = sum(x == a and y == b for x, y in pairs)
        profiles.append(record)
    profiles = pd.DataFrame(profiles)
    members = pd.DataFrame({'user_id': embeddings.user_id, 'split': split,
                            'full_history_sessions': sessions})
    sweep = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[train])
        alt = KMeans(n_clusters=k, n_init=10, random_state=43).fit(z[train])
        labels = model.predict(z[test])
        members[f'k{k}'] = model.predict(z)
        groups = []
        for c in range(k):
            frame = profiles[labels == c]
            counts = frame[actions].sum()
            group = {'cluster': c, 'train_users': int((model.labels_ == c).sum()),
                     'heldout_users': len(frame),
                     'event_shares': (counts / counts.sum()).to_dict() if counts.sum() else None,
                     'user_action_shares': {a: float(frame[a].gt(0).mean()) if len(frame) else None
                                            for a in actions},
                     'median_tokens': float(frame.tokens.median()) if len(frame) else None}
            for origin, dest in [('view', 'cart'), ('cart', 'purchase'), ('cart', 'remove')]:
                if dest not in actions:
                    continue
                n = int(frame[origin + '_next_count'].sum())
                group[origin + '_to_' + dest] = {'rate': float(frame[origin + '_to_' + dest].sum() / n)
                                                  if n else None, 'denominator': n}
            groups.append(group)
        assert sum(g['heldout_users'] for g in groups) == len(test)
        sweep.append({'k': k, 'kmeans_seed_ari': float(adjusted_rand_score(labels, alt.predict(z[test]))),
                      'clusters': groups})
    SWEEP_OUT.mkdir(parents=True, exist_ok=True)
    result = {'source': source, 'quality': quality, 'actions': actions, 'users': len(rows),
              'mature_train_users': len(train), 'mature_test_users': len(test),
              'sweep': sweep}
    (SWEEP_OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    members.to_parquet(SWEEP_OUT / f'{source}_memberships.parquet', index=False)
    print(f'{source}: users={len(rows)}, train/test repeat={len(train)}/{len(test)}', flush=True)
    for item in sweep:
        print('k=', item['k'], 'test sizes=', [g['heldout_users'] for g in item['clusters']], flush=True)





STABILITY_ROOT = CAPSTONE_ROOT
STABILITY_OUT = Path('artifacts/three-ecommerce-personas')


def run_stability(source):
    embeddings = pd.read_parquet(STABILITY_ROOT / f'data/models/{source}_masked_embedding_embeddings.parquet')
    members = pd.read_parquet(STABILITY_OUT / f'{source}_memberships.parquet')
    if not embeddings.user_id.equals(members.user_id):
        raise ValueError('Membership and embedding user order mismatch')
    z = embeddings[[f'z_{i}' for i in range(48)]].to_numpy()
    permutation = np.random.default_rng(42).permutation(len(members))
    cutoff = int(.8 * len(members))
    train, test = permutation[:cutoff], permutation[cutoff:]
    train = train[members.full_history_sessions.to_numpy()[train] >= 2]
    test = test[members.full_history_sessions.to_numpy()[test] >= 2]
    results = []
    for k in range(2, 9):
        baseline = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[train])
        reference = baseline.predict(z[test])
        if not np.array_equal(reference, members.loc[test, f'k{k}'].to_numpy()):
            raise ValueError('Reference KMeans differs from saved K sweep')
        overlap = np.zeros((5, k))
        aris = []
        for repeat in range(5):
            rng = np.random.default_rng(100 + repeat)
            sample = rng.choice(train, int(.8 * len(train)), replace=False)
            model = KMeans(n_clusters=k, n_init=10, random_state=100 + repeat).fit(z[sample])
            alternative = model.predict(z[test])
            aris.append(float(adjusted_rand_score(reference, alternative)))
            scores = np.zeros((k, k))
            for a in range(k):
                for b in range(k):
                    x, y = reference == a, alternative == b
                    union = np.logical_or(x, y).sum()
                    scores[a, b] = np.logical_and(x, y).sum() / union if union else 0
            left, right = linear_sum_assignment(-scores)
            overlap[repeat, left] = scores[left, right]
        results.append({'k': k, 'ari_replicates': aris,
                        'groups': [{'cluster': c, 'heldout_users': int((reference == c).sum()),
                                    'jaccard_replicates': overlap[:, c].tolist(),
                                    'min_jaccard': float(overlap[:, c].min()),
                                    'mean_jaccard': float(overlap[:, c].mean())} for c in range(k)]})
    (STABILITY_OUT / f'{source}_stability.json').write_text(json.dumps({
        'source': source, 'train_repeat_users': len(train), 'test_repeat_users': len(test),
        'method': 'five 80% train-user subsamples without replacement and five KMeans seeds; '
                  'matched held-out clusters via maximum total Jaccard',
        'limitation': 'encoder, split, sample and representation held fixed; not independent time validation',
        'results': results}, indent=2), encoding='utf-8')
    print(source, flush=True)
    for result in results:
        print('k', result['k'], 'ARI mean', round(np.mean(result['ari_replicates']), 3),
              [(g['cluster'], g['heldout_users'], round(g['mean_jaccard'], 2),
                round(g['min_jaccard'], 2)) for g in result['groups']], flush=True)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare, sweep or audit persona clusters.')
    parser.add_argument('task', choices=('compare', 'sweep', 'stability', 'representatives'))
    parser.add_argument('sources', nargs='+', choices=('electronics', 'cosmetics'))
    args = parser.parse_args()
    if args.task == 'representatives':
        if args.sources and args.sources != ['electronics', 'cosmetics']:
            parser.error('representative profiles require electronics and cosmetics in that order')
        render_representative_profiles()
        raise SystemExit(0)
    runner = {'compare': run_clustering, 'sweep': run_sweep, 'stability': run_stability}[args.task]
    for source in args.sources:
        runner(source)
    if args.task == 'compare':
        report()
