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
SIMILARITY_OUT = CLUSTER_ROOT / 'artifacts/cross-dataset-persona-similarity'
ELECTRONICS_DROP_OUT = CLUSTER_ROOT / 'artifacts/electronics-drop-alignment'
RETAIL_COMBINED_OUT = CLUSTER_ROOT / 'artifacts/retail-combined-personas'
THREE_DOMAIN_ALIGNMENT_OUT = CLUSTER_ROOT / 'artifacts/three-domain-cluster-alignment'


SIMILARITY_FEATURES = (
    'view_share', 'cart_share', 'remove_share', 'purchase_share',
    'buyer_share', 'log_activity_depth', 'log_sessions',
)

ALIGNMENT_FEATURES = (
    'purchase_share', 'log_total_events', 'log_active_days', 'log_events_per_day',
    'purchase_day_share', 'log_median_gap', 'first_day_purchase', 'last_day_purchase',
    'explore_to_purchase', 'purchase_to_explore', 'explore_persistence',
    'purchase_persistence', 'has_transition',
)

RETAIL_ALIGNMENT_FEATURES = (
    'cart_share', 'purchase_share', 'log_total_events', 'log_sessions',
    'log_events_per_session', 'log_median_session_gap', 'has_cart', 'has_purchase',
    'first_cart', 'first_purchase', 'last_cart', 'last_purchase',
    'view_to_cart', 'cart_to_purchase', 'view_persistence', 'cart_persistence',
    'purchase_persistence',
)


def behavioral_profile_similarity(reference: dict, target: dict, scales: dict) -> tuple[float, float, float]:
    """Return shape similarity, feature coverage, and the underlying scaled distance."""
    shared = [name for name in SIMILARITY_FEATURES if name in reference and name in target]
    if not shared:
        raise ValueError('behavior profiles have no shared features')
    distance = float(np.sqrt(np.mean([
        ((reference[name] - target[name]) / scales[name]) ** 2 for name in shared
    ])))
    return float(np.exp(-distance)), len(shared) / len(SIMILARITY_FEATURES), distance


def _shared_behavior_features(events: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Build coupon/cart-free user features shared by all three domains."""
    events = events.sort_values(['user', 'at'], kind='stable').copy()
    counts = (pd.crosstab(events.user, events.action)
                .reindex(columns=['explore', 'purchase'], fill_value=0))
    daily = (events.assign(day=events['at'].dt.floor('D'), present=1)
                   .pivot_table(index=['user', 'day'], columns='action', values='present',
                                aggfunc='max', fill_value=0)
                   .reindex(columns=['explore', 'purchase'], fill_value=0).reset_index()
                   .sort_values(['user', 'day'], kind='stable'))
    grouped = daily.groupby('user', sort=False)
    active_days = grouped.size()
    purchase_days = grouped.purchase.sum()
    gaps = grouped.day.diff().dt.days.groupby(daily.user).median().fillna(0)
    first = grouped.head(1).set_index('user')
    last = grouped.tail(1).set_index('user')
    daily['next_explore'] = grouped.explore.shift(-1)
    daily['next_purchase'] = grouped.purchase.shift(-1)
    valid = daily.next_explore.notna()
    transitions = daily.loc[valid].copy()
    transitions['explore_to_purchase'] = (
        transitions.explore.eq(1) & transitions.purchase.eq(0) & transitions.next_purchase.eq(1)
    )
    transitions['purchase_to_explore'] = (
        transitions.purchase.eq(1) & transitions.next_explore.eq(1) & transitions.next_purchase.eq(0)
    )
    transitions['explore_persistence'] = (
        transitions.explore.eq(1) & transitions.purchase.eq(0) & transitions.next_explore.eq(1)
    )
    transitions['purchase_persistence'] = transitions.purchase.eq(1) & transitions.next_purchase.eq(1)
    transition_rates = transitions.groupby('user')[list(ALIGNMENT_FEATURES[8:12])].mean()

    total = counts.sum(axis=1)
    features = pd.DataFrame(index=counts.index)
    features['purchase_share'] = counts.purchase.div(total)
    features['log_total_events'] = np.log1p(total)
    features['log_active_days'] = np.log1p(active_days)
    features['log_events_per_day'] = np.log1p(total.div(active_days))
    features['purchase_day_share'] = purchase_days.div(active_days)
    features['log_median_gap'] = np.log1p(gaps)
    features['first_day_purchase'] = first.purchase
    features['last_day_purchase'] = last.purchase
    features = features.join(transition_rates).fillna(0)
    features['has_transition'] = active_days.gt(1).astype(float)
    features = membership[['user_id', 'cluster']].merge(
        features.reset_index().rename(columns={'user': 'user_id'}), on='user_id', how='inner'
    )
    if features[list(ALIGNMENT_FEATURES)].isna().any().any():
        raise ValueError('shared behavior features contain missing values')
    return features


def _retail_behavior_features(events: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Build user features after both retail domains share view/cart/purchase actions."""
    events = events.loc[events.action.isin(['view', 'cart', 'purchase'])].copy()
    events = events.sort_values(['user_id', 'at'], kind='stable')
    counts = (pd.crosstab(events.user_id, events.action)
                .reindex(columns=['view', 'cart', 'purchase'], fill_value=0))
    total = counts.sum(axis=1)
    sessions = events.groupby('user_id').session.nunique()
    starts = (events.groupby(['user_id', 'session'], as_index=False).at.min()
                    .sort_values(['user_id', 'at'], kind='stable'))
    gaps = starts.groupby('user_id').at.diff().dt.total_seconds().div(86400)
    median_gap = gaps.groupby(starts.user_id).median().fillna(0)
    first = events.groupby('user_id', sort=False).head(1).set_index('user_id').action
    last = events.groupby('user_id', sort=False).tail(1).set_index('user_id').action

    events['next_action'] = events.groupby(['user_id', 'session'], sort=False).action.shift(-1)
    pairs = events.loc[events.next_action.notna(), ['user_id', 'action', 'next_action']].copy()
    transitions = {}
    for name, origin, destination in (
        ('view_to_cart', 'view', 'cart'),
        ('cart_to_purchase', 'cart', 'purchase'),
        ('view_persistence', 'view', 'view'),
        ('cart_persistence', 'cart', 'cart'),
        ('purchase_persistence', 'purchase', 'purchase'),
    ):
        origin_pairs = pairs.loc[pairs.action.eq(origin)]
        transitions[name] = origin_pairs.next_action.eq(destination).groupby(origin_pairs.user_id).mean()

    features = pd.DataFrame(index=counts.index)
    features['cart_share'] = counts.cart.div(total)
    features['purchase_share'] = counts.purchase.div(total)
    features['log_total_events'] = np.log1p(total)
    features['log_sessions'] = np.log1p(sessions)
    features['log_events_per_session'] = np.log1p(total.div(sessions))
    features['log_median_session_gap'] = np.log1p(median_gap)
    features['has_cart'] = counts.cart.gt(0).astype(float)
    features['has_purchase'] = counts.purchase.gt(0).astype(float)
    features['first_cart'] = first.eq('cart').astype(float)
    features['first_purchase'] = first.eq('purchase').astype(float)
    features['last_cart'] = last.eq('cart').astype(float)
    features['last_purchase'] = last.eq('purchase').astype(float)
    for name, values in transitions.items():
        features[name] = values
    features = features.fillna(0)
    features = membership[['user_id', 'cluster']].merge(
        features.reset_index(), on='user_id', how='inner'
    )
    if features[list(RETAIL_ALIGNMENT_FEATURES)].isna().any().any():
        raise ValueError('retail behavior features contain missing values')
    return features


def _fixed_k4_membership(source: str) -> tuple[pd.DataFrame, dict]:
    """Create once and reuse a deterministic K=4 mapping on the stored GRU embedding."""
    membership_path = SIMILARITY_OUT / f'{source}_k4_memberships.parquet'
    report_path = SIMILARITY_OUT / f'{source}_k4_selection.json'
    if membership_path.exists() and report_path.exists():
        return pd.read_parquet(membership_path), json.loads(report_path.read_text())

    embeddings = pd.read_parquet(
        _capstone_models() / f'{source}_masked_embedding_embeddings.parquet'
    )
    z = embeddings[[f'z_{index}' for index in range(48)]].to_numpy()
    mature = np.flatnonzero(embeddings.sessions.to_numpy() >= 2)
    permutation = np.random.default_rng(2026).permutation(mature)
    train = permutation[:int(.8 * len(permutation))]
    test = permutation[int(.8 * len(permutation)):]
    first = KMeans(n_clusters=4, n_init=50, random_state=42).fit(z[train])
    alternate = KMeans(n_clusters=4, n_init=50, random_state=43).fit(z[train])
    labels = first.predict(z)
    train_sample = np.random.default_rng(1).choice(len(train), min(3000, len(train)), False)
    test_sample = np.random.default_rng(2).choice(len(test), min(3000, len(test)), False)
    report = {
        'cluster_count_mode': 'fixed_operational_k4',
        'selected_k': 4,
        'train_silhouette': float(
            silhouette_score(z[train][train_sample], labels[train][train_sample])
        ),
        'heldout_silhouette': float(
            silhouette_score(z[test][test_sample], labels[test][test_sample])
        ),
        'seed_ari': float(adjusted_rand_score(labels[train], alternate.predict(z[train]))),
        'cluster_sizes': np.bincount(labels, minlength=4).tolist(),
        'mature_cluster_sizes': np.bincount(labels[mature], minlength=4).tolist(),
    }
    membership = embeddings[['user_id', 'sessions']].assign(cluster=labels)
    SIMILARITY_OUT.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(membership_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return membership, report


def _tianchi_shared_features() -> tuple[pd.DataFrame, dict]:
    membership = pd.read_parquet(PROJECT_ROOT / 'artifacts/tianchi-online-personas/clusters.parquet')
    user_ids = set(membership.user_id.astype('int64'))
    parts = []
    source = PROJECT_ROOT / 'data/raw/tianchi-o2o/ccf_online_stage1_train.csv'
    for chunk in pd.read_csv(source, usecols=['User_id', 'Action', 'Date'],
                             dtype={'User_id': 'Int64', 'Action': 'Int8', 'Date': 'string'},
                             chunksize=500_000):
        chunk = chunk.loc[chunk.User_id.isin(user_ids) & chunk.Action.isin([0, 1])]
        if chunk.empty:
            continue
        parts.append(pd.DataFrame({
            'user': chunk.User_id.astype('int64'),
            'at': pd.to_datetime(chunk.Date, format='%Y%m%d', utc=True, errors='coerce'),
            'action': chunk.Action.map({0: 'explore', 1: 'purchase'}),
        }).dropna())
    features = _shared_behavior_features(pd.concat(parts, ignore_index=True), membership)
    report = json.loads(
        (PROJECT_ROOT / 'artifacts/tianchi-online-personas/clusters.json').read_text()
    )
    return features, report


def _alignment_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    inputs, selection_reports = {}, {}
    for source in ('cosmetics', 'electronics'):
        membership, selection_reports[source] = _fixed_k4_membership(source)
        events = _selected_events(source, set(membership.user_id.astype('int64')))
        events = (events.loc[events.action.isin(['view', 'purchase']), ['user_id', 'at', 'action']]
                        .rename(columns={'user_id': 'user'}))
        events['action'] = events.action.replace({'view': 'explore'})
        inputs[source] = _shared_behavior_features(events, membership)

    inputs['tianchi'], selection_reports['tianchi'] = _tianchi_shared_features()
    return inputs, selection_reports


def _profile_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = np.sqrt(np.mean((left[:, None, :] - right[None, :, :]) ** 2, axis=2))
    return np.exp(-distance)


def _nearest_profile_mapping(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each source profile independently; duplicate targets are intentional."""
    order = np.argsort(-scores, axis=1)
    nearest = order[:, 0]
    margin = scores[np.arange(len(scores)), order[:, 0]] - scores[np.arange(len(scores)), order[:, 1]]
    return nearest, margin


def cross_dataset_similarity(bootstrap_repeats: int = 200) -> dict:
    """Map Tianchi clusters onto independently selected target-domain profiles."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler

    inputs, selection_reports = _alignment_inputs()
    combined = pd.concat([frame.assign(dataset=source) for source, frame in inputs.items()],
                         ignore_index=True)
    scaler = RobustScaler().fit(combined[list(ALIGNMENT_FEATURES)])
    combined[list(ALIGNMENT_FEATURES)] = scaler.transform(combined[list(ALIGNMENT_FEATURES)])
    profiles = (combined.groupby(['dataset', 'cluster'])[list(ALIGNMENT_FEATURES)]
                         .median().sort_index())
    profile_arrays = {source: profiles.loc[source].to_numpy() for source in inputs}
    cluster_ids = {source: profiles.loc[source].index.to_numpy() for source in inputs}
    cluster_counts = {source: len(ids) for source, ids in cluster_ids.items()}
    targets = ('cosmetics', 'electronics')
    matrices = {target: _profile_similarity(profile_arrays['tianchi'], profile_arrays[target])
                for target in targets}
    direct_scores = _profile_similarity(profile_arrays['cosmetics'], profile_arrays['electronics'])

    rng = np.random.default_rng(2026)
    boot_counts = {
        target: np.zeros((cluster_counts['tianchi'], cluster_counts[target]), dtype=int)
        for target in targets
    }
    direct_boot_counts = {
        'cosmetics_to_electronics': np.zeros(
            (cluster_counts['cosmetics'], cluster_counts['electronics']), dtype=int
        ),
        'electronics_to_cosmetics': np.zeros(
            (cluster_counts['electronics'], cluster_counts['cosmetics']), dtype=int
        ),
    }
    grouped = {(source, cluster): group[list(ALIGNMENT_FEATURES)].to_numpy()
               for (source, cluster), group in combined.groupby(['dataset', 'cluster'])}
    for _ in range(bootstrap_repeats):
        boot = {}
        for source in inputs:
            boot[source] = np.vstack([
                np.median(values[rng.integers(0, len(values), len(values))], axis=0)
                for cluster in cluster_ids[source] for values in [grouped[(source, cluster)]]
            ])
        for target in targets:
            score = _profile_similarity(boot['tianchi'], boot[target])
            nearest, _ = _nearest_profile_mapping(score)
            boot_counts[target][np.arange(cluster_counts['tianchi']), nearest] += 1
        score = _profile_similarity(boot['cosmetics'], boot['electronics'])
        nearest, _ = _nearest_profile_mapping(score)
        direct_boot_counts['cosmetics_to_electronics'][
            np.arange(cluster_counts['cosmetics']), nearest
        ] += 1
        nearest, _ = _nearest_profile_mapping(score.T)
        direct_boot_counts['electronics_to_cosmetics'][
            np.arange(cluster_counts['electronics']), nearest
        ] += 1

    comparisons, assignments, margins = {}, {}, {}
    for target, scores in matrices.items():
        nearest, margin = _nearest_profile_mapping(scores)
        assignments[target] = nearest
        margins[target] = margin
        matches = []
        for i, tianchi_cluster in enumerate(cluster_ids['tianchi']):
            j = int(nearest[i])
            matches.append({
                'tianchi_cluster': int(tianchi_cluster),
                f'{target}_cluster': int(cluster_ids[target][j]),
                'similarity': float(scores[i, j]),
                'nearest_margin': float(margin[i]),
                'bootstrap_nearest_probability': float(boot_counts[target][i, j] / bootstrap_repeats),
            })
        comparisons[target] = {'similarity_matrix': scores.tolist(), 'matches': matches}

    mapping = []
    for i, cluster in enumerate(cluster_ids['tianchi']):
        c = int(assignments['cosmetics'][i])
        e = int(assignments['electronics'][i])
        mapping.append({
            'tianchi_cluster': int(cluster),
            'cosmetics_cluster': int(cluster_ids['cosmetics'][c]),
            'cosmetics_similarity': float(matrices['cosmetics'][i, c]),
            'cosmetics_margin': float(margins['cosmetics'][i]),
            'cosmetics_bootstrap_nearest_probability': float(
                boot_counts['cosmetics'][i, c] / bootstrap_repeats
            ),
            'electronics_cluster': int(cluster_ids['electronics'][e]),
            'electronics_similarity': float(matrices['electronics'][i, e]),
            'electronics_margin': float(margins['electronics'][i]),
            'electronics_bootstrap_nearest_probability': float(
                boot_counts['electronics'][i, e] / bootstrap_repeats
            ),
        })
    collapsed_counts = {
        target: len(set(assignments[target].tolist())) for target in targets
    }
    direct_mapping = {}
    for name, scores, source, target in (
        ('cosmetics_to_electronics', direct_scores, 'cosmetics', 'electronics'),
        ('electronics_to_cosmetics', direct_scores.T, 'electronics', 'cosmetics'),
    ):
        nearest, margin = _nearest_profile_mapping(scores)
        direct_mapping[name] = [
            {
                f'{source}_cluster': int(cluster_ids[source][i]),
                f'{target}_cluster': int(cluster_ids[target][j]),
                'similarity': float(scores[i, j]),
                'nearest_margin': float(margin[i]),
                'bootstrap_nearest_probability': float(
                    direct_boot_counts[name][i, j] / bootstrap_repeats
                ),
            }
            for i, j in enumerate(nearest)
        ]
    result = {
        'method': 'each Tianchi K=6 cluster independently maps to its nearest pooled-robust-scaled '
                  'shared behavioral-profile median in each coupon-unobserved domain; many-to-one '
                  'targets are allowed',
        'features': list(ALIGNMENT_FEATURES),
        'similarity': 'exp(-RMS distance); descriptive score, not a probability',
        'bootstrap_repeats': bootstrap_repeats,
        'cluster_counts': cluster_counts,
        'cluster_selection': {
            source: {
                'method': report.get('cluster_count_mode') or 'silhouette_with_min_size',
                'selected_k': int(report['selected_k']),
                'train_silhouette': float(report['train_silhouette']),
                'heldout_silhouette': float(report['heldout_silhouette']),
                'seed_ari': report.get('seed_ari'),
                'cluster_sizes': report.get('cluster_sizes'),
            }
            for source, report in selection_reports.items()
        },
        'comparisons': comparisons,
        'mapping': mapping,
        'distinct_target_clusters': collapsed_counts,
        'direct_cosmetics_electronics': {
            'similarity_matrix': direct_scores.tolist(),
            **direct_mapping,
        },
        'limitations': [
            'Cart, removal and coupon receipt are excluded from alignment features.',
            'Cluster similarity is descriptive and does not identify shared latent users or causal effects.',
            'Cosmetics and Electronics do not record coupon exposure; treating them as coupon-free is a '
            'simulation assumption rather than an observed treatment condition.',
            'Many-to-one nearest mapping can show profile collapse but cannot attribute that collapse '
            'causally to coupons.',
        ],
    }

    SIMILARITY_OUT.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SIMILARITY_OUT / 'shared_user_features.parquet', index=False)
    profiles.reset_index().to_parquet(SIMILARITY_OUT / 'cluster_profiles.parquet', index=False)
    (SIMILARITY_OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, target in zip(axes[:2], targets, strict=True):
        scores = matrices[target]
        image = ax.imshow(scores, vmin=0, vmax=1, cmap='YlGnBu')
        ax.set_xticks(range(cluster_counts[target]),
                      [f'{target[0].upper()}{i}' for i in cluster_ids[target]])
        ax.set_yticks(range(cluster_counts['tianchi']),
                      [f'T{i}' for i in cluster_ids['tianchi']])
        ax.set(xlabel=target.title(), ylabel='Tianchi',
               title=f'Tianchi $\\rightarrow$ {target.title()}')
        for i in range(cluster_counts['tianchi']):
            for j in range(cluster_counts[target]):
                ax.text(j, i, f'{scores[i, j]:.2f}', ha='center', va='center',
                        color='white' if scores[i, j] > .62 else 'black', fontsize=8)
    colors = plt.cm.tab10(np.arange(cluster_counts['tianchi']))
    source_y_positions = np.linspace(5, 0, cluster_counts['tianchi'])
    target_y_positions = {
        target: np.linspace(5, 0, cluster_counts[target]) for target in targets
    }
    for i, cluster in enumerate(cluster_ids['tianchi']):
        source_y = source_y_positions[i]
        axes[2].scatter(0, source_y, color=colors[cluster], s=55, zorder=3)
        axes[2].text(-.08, source_y, f'T{cluster}', ha='right', va='center', fontsize=8)
        for x, source in enumerate(targets, start=1):
            target_index = int(assignments[source][i])
            target_y = target_y_positions[source][target_index]
            axes[2].plot([0.05, x - .05], [source_y, target_y], color=colors[cluster],
                         alpha=.55, linewidth=1.2, zorder=1)
    for x, source in enumerate(targets, start=1):
        used = set(assignments[source].tolist())
        for index, cluster in enumerate(cluster_ids[source]):
            target_y = target_y_positions[source][index]
            axes[2].scatter(x, target_y, facecolor='white',
                            edgecolor='#333333' if index in used else '#BBBBBB',
                            linewidth=1.2, s=55, zorder=3)
            axes[2].text(x + .08, target_y, f'{source[0].upper()}{cluster}',
                         ha='left', va='center', fontsize=8,
                         color='#222222' if index in used else '#AAAAAA')
    axes[2].set_xlim(-.35, 2.35)
    axes[2].set_ylim(-.6, 5.6)
    axes[2].set_xticks([0, 1, 2], ['Tianchi', 'Cosmetics', 'Electronics'])
    axes[2].set_yticks([])
    axes[2].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    axes[2].tick_params(axis='x', length=0)
    axes[2].set_title('Many-to-one nearest-profile mapping')
    fig.colorbar(image, ax=axes[:2], label='Shared-profile similarity', shrink=.78)
    figure_dir = PROJECT_ROOT / 'img/persona'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'cross_domain_cluster_alignment.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    image = axes[0].imshow(direct_scores, vmin=0, vmax=1, cmap='YlGnBu')
    axes[0].set_xticks(range(cluster_counts['electronics']),
                       [f'E{i}' for i in cluster_ids['electronics']])
    axes[0].set_yticks(range(cluster_counts['cosmetics']),
                       [f'C{i}' for i in cluster_ids['cosmetics']])
    axes[0].set(xlabel='Electronics', ylabel='Cosmetics',
                title='Cosmetics--Electronics similarity')
    for i in range(cluster_counts['cosmetics']):
        for j in range(cluster_counts['electronics']):
            axes[0].text(j, i, f'{direct_scores[i, j]:.2f}', ha='center', va='center',
                         color='white' if direct_scores[i, j] > .62 else 'black', fontsize=9)
    fig.colorbar(image, ax=axes[0], label='Shared-profile similarity', shrink=.8)

    c_y = np.linspace(3, 0, cluster_counts['cosmetics'])
    e_y = np.linspace(3, 0, cluster_counts['electronics'])
    for index, cluster in enumerate(cluster_ids['cosmetics']):
        axes[1].scatter(0, c_y[index], color=plt.cm.tab10(index), s=60, zorder=3)
        axes[1].text(-.08, c_y[index], f'C{cluster}', ha='right', va='center')
    for index, cluster in enumerate(cluster_ids['electronics']):
        axes[1].scatter(1, e_y[index], facecolor='white', edgecolor='#333333', s=60, zorder=3)
        axes[1].text(1.08, e_y[index], f'E{cluster}', ha='left', va='center')
    c_to_e, _ = _nearest_profile_mapping(direct_scores)
    e_to_c, _ = _nearest_profile_mapping(direct_scores.T)
    for c_index, e_index in enumerate(c_to_e):
        axes[1].annotate('', xy=(.95, e_y[e_index]), xytext=(.05, c_y[c_index]),
                         arrowprops={'arrowstyle': '->', 'color': plt.cm.tab10(c_index),
                                     'alpha': .7, 'linewidth': 1.4})
    for e_index, c_index in enumerate(e_to_c):
        axes[1].annotate('', xy=(.05, c_y[c_index]), xytext=(.95, e_y[e_index]),
                         arrowprops={'arrowstyle': '->', 'color': '#777777',
                                     'linestyle': '--', 'alpha': .55, 'linewidth': 1})
    axes[1].plot([], [], color='#333333', label='C→E nearest')
    axes[1].plot([], [], color='#777777', linestyle='--', label='E→C nearest')
    axes[1].set_xlim(-.3, 1.3)
    axes[1].set_ylim(-.5, 3.5)
    axes[1].set_xticks([0, 1], ['Cosmetics', 'Electronics'])
    axes[1].set_yticks([])
    axes[1].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    axes[1].tick_params(axis='x', length=0)
    axes[1].set_title('Bidirectional nearest-profile mapping')
    axes[1].legend(frameon=False, loc='lower center', fontsize=8)
    fig.savefig(figure_dir / 'cosmetics_electronics_k4_alignment.png',
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    return result


def electronics_drop_similarity(bootstrap_repeats: int = 200) -> dict:
    """Compare Electronics with Cosmetics retrained after dropping remove events."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler

    drop_membership = pd.read_parquet(
        PROJECT_ROOT / 'artifacts/electronics-drop-personas/electronics_drop_embeddings_k4.parquet'
    )
    electronics_membership, electronics_selection = _fixed_k4_membership('electronics')
    original_membership, original_selection = _fixed_k4_membership('cosmetics')
    memberships = {
        'cosmetics_original': original_membership,
        'electronics_drop': drop_membership[['user_id', 'cluster', 'sessions']],
        'electronics': electronics_membership,
    }
    cosmetics_events = _selected_events(
        'cosmetics', set(drop_membership.user_id.astype('int64'))
    )
    cosmetics_events = cosmetics_events.loc[cosmetics_events.action.ne('remove')]
    events = {
        'cosmetics_original': cosmetics_events,
        'electronics_drop': cosmetics_events,
        'electronics': _selected_events(
            'electronics', set(electronics_membership.user_id.astype('int64'))
        ),
    }
    inputs = {
        source: _retail_behavior_features(events[source], memberships[source])
        for source in memberships
    }
    combined = pd.concat(
        [frame.assign(dataset=source) for source, frame in inputs.items()], ignore_index=True
    )
    baseline = combined.dataset.isin(['cosmetics_original', 'electronics'])
    scaler = RobustScaler().fit(combined.loc[baseline, list(RETAIL_ALIGNMENT_FEATURES)])
    combined[list(RETAIL_ALIGNMENT_FEATURES)] = scaler.transform(
        combined[list(RETAIL_ALIGNMENT_FEATURES)]
    )
    profiles = (combined.groupby(['dataset', 'cluster'])[list(RETAIL_ALIGNMENT_FEATURES)]
                         .median().sort_index())
    drop_profile = profiles.loc['electronics_drop'].to_numpy()
    original_profile = profiles.loc['cosmetics_original'].to_numpy()
    electronics_profile = profiles.loc['electronics'].to_numpy()
    score_matrices = {
        'cosmetics_original': _profile_similarity(original_profile, electronics_profile),
        'electronics_drop': _profile_similarity(drop_profile, electronics_profile),
    }
    scores = score_matrices['electronics_drop']

    rng = np.random.default_rng(2026)
    grouped = {(source, cluster): group[list(RETAIL_ALIGNMENT_FEATURES)].to_numpy()
               for (source, cluster), group in combined.groupby(['dataset', 'cluster'])}
    forward_counts = {name: np.zeros(matrix.shape, dtype=int)
                      for name, matrix in score_matrices.items()}
    reverse_counts = {name: np.zeros(matrix.T.shape, dtype=int)
                      for name, matrix in score_matrices.items()}
    for _ in range(bootstrap_repeats):
        boot = {}
        for source in memberships:
            boot[source] = np.vstack([
                np.median(values[rng.integers(0, len(values), len(values))], axis=0)
                for cluster in range(4) for values in [grouped[(source, cluster)]]
            ])
        for name in score_matrices:
            boot_scores = _profile_similarity(boot[name], boot['electronics'])
            nearest, _ = _nearest_profile_mapping(boot_scores)
            forward_counts[name][np.arange(4), nearest] += 1
            nearest, _ = _nearest_profile_mapping(boot_scores.T)
            reverse_counts[name][np.arange(4), nearest] += 1

    comparisons = {}
    for name, matrix in score_matrices.items():
        forward, forward_margin = _nearest_profile_mapping(matrix)
        reverse, reverse_margin = _nearest_profile_mapping(matrix.T)
        mutual = [(int(i), int(j)) for i, j in enumerate(forward) if reverse[j] == i]
        hungarian_left, hungarian_right = linear_sum_assignment(-matrix)
        comparisons[name] = {
            'similarity_matrix': matrix.tolist(),
            'to_electronics': [
                {
                    'source_cluster': i,
                    'electronics_cluster': int(j),
                    'similarity': float(matrix[i, j]),
                    'nearest_margin': float(forward_margin[i]),
                    'bootstrap_nearest_probability': float(
                        forward_counts[name][i, j] / bootstrap_repeats
                    ),
                }
                for i, j in enumerate(forward)
            ],
            'electronics_to_source': [
                {
                    'electronics_cluster': i,
                    'source_cluster': int(j),
                    'similarity': float(matrix[j, i]),
                    'nearest_margin': float(reverse_margin[i]),
                    'bootstrap_nearest_probability': float(
                        reverse_counts[name][i, j] / bootstrap_repeats
                    ),
                }
                for i, j in enumerate(reverse)
            ],
            'mutual_nearest_pairs': mutual,
            'forward_is_one_to_one': bool(len(set(forward.tolist())) == 4),
            'is_bijective_nearest_mapping': bool(
                len(set(forward.tolist())) == 4
                and len(set(reverse.tolist())) == 4
                and len(mutual) == 4
            ),
            'hungarian_mapping': [
                {'source_cluster': int(i), 'electronics_cluster': int(j),
                 'similarity': float(matrix[i, j])}
                for i, j in zip(hungarian_left, hungarian_right, strict=True)
            ],
            'hungarian_mean_similarity': float(
                matrix[hungarian_left, hungarian_right].mean()
            ),
        }
    drop_comparison = comparisons['electronics_drop']
    forward = np.array([
        row['electronics_cluster'] for row in drop_comparison['to_electronics']
    ])
    action_profiles = {}
    for source in memberships:
        joined = events[source].merge(memberships[source][['user_id', 'cluster']], on='user_id')
        rows = []
        for cluster, frame in joined.groupby('cluster'):
            shares = (frame.action.value_counts(normalize=True)
                           .reindex(['view', 'cart', 'purchase'], fill_value=0))
            rows.append({
                'cluster': int(cluster),
                'users': int(memberships[source].cluster.eq(cluster).sum()),
                'event_share': {name: float(shares[name]) for name in shares.index},
                'buyer_share': float(
                    frame.loc[frame.action.eq('purchase'), 'user_id'].nunique()
                    / memberships[source].cluster.eq(cluster).sum()
                ),
            })
        action_profiles[source] = rows

    drop_selection = json.loads(
        (PROJECT_ROOT / 'artifacts/electronics-drop-personas/electronics_drop_clusters_k4.json')
        .read_text()
    )
    result = {
        'experiment': 'electronics_drop: Cosmetics remove_from_cart events are deleted before '
                      'masked-GRU retraining; both domains use view/cart/purchase and K=4',
        'features': list(RETAIL_ALIGNMENT_FEATURES),
        'similarity': 'exp(-RMS distance) on robust-scaled cluster-median profiles; '
                      'the scaler is fitted on original Cosmetics and Electronics only, '
                      'then applied to electronics_drop out of sample',
        'selection': {
            'cosmetics_original': original_selection,
            'electronics_drop': {
                'train_silhouette': drop_selection['train_silhouette'],
                'heldout_silhouette': drop_selection['heldout_silhouette'],
                'seed_ari': next(
                    row['seed_ari'] for row in drop_selection['cluster_search'] if row['k'] == 4
                ),
            },
            'electronics': electronics_selection,
        },
        'baseline_cosmetics_original': comparisons['cosmetics_original'],
        'ablation_effect': {
            'original_hungarian_mean_similarity': comparisons['cosmetics_original'][
                'hungarian_mean_similarity'
            ],
            'drop_hungarian_mean_similarity': drop_comparison['hungarian_mean_similarity'],
            'hungarian_mean_similarity_delta': (
                drop_comparison['hungarian_mean_similarity']
                - comparisons['cosmetics_original']['hungarian_mean_similarity']
            ),
            'original_forward_is_one_to_one': comparisons['cosmetics_original'][
                'forward_is_one_to_one'
            ],
            'drop_forward_is_one_to_one': drop_comparison['forward_is_one_to_one'],
        },
        'similarity_matrix': drop_comparison['similarity_matrix'],
        'electronics_drop_to_electronics': [
            {
                'electronics_drop_cluster': row['source_cluster'],
                **{key: value for key, value in row.items() if key != 'source_cluster'},
            }
            for row in drop_comparison['to_electronics']
        ],
        'electronics_to_electronics_drop': [
            {
                'electronics_drop_cluster': row['source_cluster'],
                **{key: value for key, value in row.items() if key != 'source_cluster'},
            }
            for row in drop_comparison['electronics_to_source']
        ],
        'mutual_nearest_pairs': drop_comparison['mutual_nearest_pairs'],
        'forward_is_one_to_one': drop_comparison['forward_is_one_to_one'],
        'hungarian_mapping': [
            {
                'electronics_drop_cluster': row['source_cluster'],
                **{key: value for key, value in row.items() if key != 'source_cluster'},
            }
            for row in drop_comparison['hungarian_mapping']
        ],
        'hungarian_mean_similarity': drop_comparison['hungarian_mean_similarity'],
        'is_bijective_nearest_mapping': drop_comparison['is_bijective_nearest_mapping'],
        'action_profiles': action_profiles,
        'limitations': [
            'Dropping remove_from_cart changes the observed sequence rather than recovering '
            'the counterfactual Cosmetics behavior under Electronics logging.',
            'The encoders are trained independently, so comparison uses observable behavioral '
            'profiles rather than embedding coordinates.',
            'A scaler refitted on electronics_drop and Electronics makes the nearest mapping '
            'appear one-to-one; that result is normalization-sensitive and is not used. The '
            'reported ablation keeps the original Cosmetics--Electronics metric fixed.',
        ],
    }
    ELECTRONICS_DROP_OUT.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(ELECTRONICS_DROP_OUT / 'shared_user_features.parquet', index=False)
    profiles.reset_index().to_parquet(ELECTRONICS_DROP_OUT / 'cluster_profiles.parquet', index=False)
    (ELECTRONICS_DROP_OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, name, prefix, title in (
        (axes[0], 'cosmetics_original', 'C', 'Original Cosmetics'),
        (axes[1], 'electronics_drop', 'D', 'Electronics-drop'),
    ):
        matrix = score_matrices[name]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu')
        ax.set_xticks(range(4), [f'E{i}' for i in range(4)])
        ax.set_yticks(range(4), [f'{prefix}{i}' for i in range(4)])
        ax.set(xlabel='Electronics', ylabel=title, title=f'{title} similarity')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                        color='white' if matrix[i, j] > .62 else 'black', fontsize=8)
    fig.colorbar(image, ax=axes[:2], label='Similarity', shrink=.8)
    y = np.arange(3, -1, -1)
    for i in range(4):
        axes[2].scatter(0, y[i], color=plt.cm.tab10(i), s=60, zorder=3)
        axes[2].text(-.08, y[i], f'D{i}', ha='right', va='center')
        axes[2].scatter(1, y[i], facecolor='white', edgecolor='#333333', s=60, zorder=3)
        axes[2].text(1.08, y[i], f'E{i}', ha='left', va='center')
    for i, j in enumerate(forward):
        axes[2].annotate('', xy=(.95, y[j]), xytext=(.05, y[i]),
                         arrowprops={'arrowstyle': '->', 'color': plt.cm.tab10(i),
                                     'alpha': .75, 'linewidth': 1.5})
    axes[2].set_xlim(-.3, 1.3)
    axes[2].set_ylim(-.5, 3.5)
    axes[2].set_xticks([0, 1], ['Electronics-drop', 'Electronics'])
    axes[2].set_yticks([])
    axes[2].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    axes[2].tick_params(axis='x', length=0)
    axes[2].set_title('Drop-variant nearest mapping')
    figure_dir = PROJECT_ROOT / 'img/persona'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'electronics_drop_alignment.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return result


def retail_combined_summary() -> dict:
    """Summarize the shared K=4 space jointly fitted to Electronics and Cosmetics-drop."""
    import matplotlib.pyplot as plt

    report = json.loads(
        (RETAIL_COMBINED_OUT / 'retail_combined_clusters_k4.json').read_text()
    )
    profiles = pd.DataFrame(report['domain_cluster_profiles'])
    clusters = sorted(profiles.cluster.unique())
    domains = ['cosmetics_drop', 'electronics']
    domain_totals = profiles.groupby('domain').users.sum()
    distributions = {
        domain: (profiles.loc[profiles.domain.eq(domain)].set_index('cluster').users
                         .reindex(clusters, fill_value=0) / domain_totals[domain]).to_numpy()
        for domain in domains
    }
    p, q = distributions['cosmetics_drop'], distributions['electronics']
    midpoint = (p + q) / 2

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(np.where(
            left > 0, left * np.log(left / np.clip(right, 1e-12, None)), 0
        )))

    rows = []
    for cluster in clusters:
        pair = profiles.loc[profiles.cluster.eq(cluster)].set_index('domain')
        left = np.array(list(pair.loc['cosmetics_drop', 'event_share'].values()), dtype=float)
        right = np.array(list(pair.loc['electronics', 'event_share'].values()), dtype=object)
        supported = pair.loc['electronics', 'users'] >= 30
        similarity = None
        if supported and all(value is not None for value in right):
            similarity = float(np.exp(-np.sqrt(
                np.mean((left - right.astype(float)) ** 2)
            )))
        rows.append({
            'cluster': int(cluster),
            'cosmetics_drop_users': int(pair.loc['cosmetics_drop', 'users']),
            'electronics_users': int(pair.loc['electronics', 'users']),
            'cosmetics_drop_cluster_probability': float(p[cluster]),
            'electronics_cluster_probability': float(q[cluster]),
            'event_profile_similarity': similarity,
            'cross_domain_supported': bool(supported),
        })
    result = {
        'experiment': 'one masked-GRU and one K=4 KMeans jointly fitted after removing '
                      'Cosmetics remove_from_cart events',
        'selected_users': report['domain_selected_users'],
        'mature_users': report['domain_mature_users'],
        'balanced_mature_train_users_per_domain': report[
            'balanced_mature_train_users_per_domain'
        ],
        'balanced_mature_test_users_per_domain': report[
            'balanced_mature_test_users_per_domain'
        ],
        'train_silhouette': report['train_silhouette'],
        'heldout_silhouette': report['heldout_silhouette'],
        'domain_cluster_nmi': report['domain_cluster_nmi'],
        'cluster_distribution_total_variation': float(np.abs(p - q).sum() / 2),
        'cluster_distribution_jensen_shannon_divergence': float(
            .5 * kl(p, midpoint) + .5 * kl(q, midpoint)
        ),
        'clusters': rows,
        'interpretation': (
            'Clusters 0--2 have at least 30 mature users from both domains. Cluster 3 has only '
            'three mature Electronics users and is not accepted as a cross-domain match.'
        ),
    }
    (RETAIL_COMBINED_OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    x = np.arange(len(clusters))
    width = .36
    axes[0].bar(x - width / 2, 100 * p, width, label='Cosmetics-drop', color='#4C78A8')
    axes[0].bar(x + width / 2, 100 * q, width, label='Electronics', color='#F28E2B')
    axes[0].set(xticks=x, xticklabels=[f'J{i}' for i in clusters],
                ylabel='Users within domain (%)', xlabel='Joint cluster',
                title='Domain-normalized cluster distribution')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis='y', alpha=.2)

    similarities = [row['event_profile_similarity'] for row in rows]
    plotted = [value if value is not None else 0 for value in similarities]
    colors = ['#59A14F' if value is not None else '#BAB0AC' for value in similarities]
    axes[1].bar(x, plotted, color=colors)
    for i, row in enumerate(rows):
        label = f"C={row['cosmetics_drop_users']}, E={row['electronics_users']}"
        axes[1].text(i, plotted[i] + .025, label, ha='center', va='bottom', fontsize=7)
        if similarities[i] is None:
            axes[1].text(i, .05, 'insufficient', ha='center', va='bottom', fontsize=7,
                         rotation=90)
    axes[1].set(xticks=x, xticklabels=[f'J{i}' for i in clusters], ylim=(0, 1.12),
                ylabel='Event-profile similarity', xlabel='Joint cluster',
                title='Within-cluster cross-domain agreement')
    axes[1].grid(axis='y', alpha=.2)
    figure_dir = PROJECT_ROOT / 'img/persona'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'retail_combined_k4.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return result


def joint_three_domain_alignment(bootstrap_repeats: int = 200) -> dict:
    """Align Tianchi K=6 to the joint Cosmetics-drop/Electronics K=4 profiles."""
    from sklearn.preprocessing import RobustScaler

    membership = pd.read_parquet(
        RETAIL_COMBINED_OUT / 'retail_combined_embeddings_k4.parquet'
    )
    inputs = {}
    for domain, source in (('cosmetics_drop', 'cosmetics'), ('electronics', 'electronics')):
        members = (membership.loc[membership.domain.eq(domain),
                                  ['original_user_id', 'cluster']]
                             .rename(columns={'original_user_id': 'user_id'}))
        events = _selected_events(source, set(members.user_id.astype('int64')))
        events = (events.loc[events.action.isin(['view', 'purchase']),
                             ['user_id', 'at', 'action']]
                        .rename(columns={'user_id': 'user'}))
        events['action'] = events.action.replace({'view': 'explore'})
        inputs[domain] = _shared_behavior_features(events, members)

    inputs['tianchi'], tianchi_report = _tianchi_shared_features()
    combined = pd.concat([
        frame.assign(dataset=source) for source, frame in inputs.items()
    ], ignore_index=True)
    scaler = RobustScaler().fit(combined[list(ALIGNMENT_FEATURES)])
    combined[list(ALIGNMENT_FEATURES)] = scaler.transform(
        combined[list(ALIGNMENT_FEATURES)]
    )

    baseline = combined.loc[combined.dataset.isin(['cosmetics_drop', 'electronics'])]
    baseline_profiles = (baseline.groupby('cluster')[list(ALIGNMENT_FEATURES)]
                                  .median().sort_index())
    tianchi = combined.loc[combined.dataset.eq('tianchi')]
    tianchi_profiles = (tianchi.groupby('cluster')[list(ALIGNMENT_FEATURES)]
                                .median().sort_index())
    scores = _profile_similarity(tianchi_profiles.to_numpy(), baseline_profiles.to_numpy())
    nearest, margin = _nearest_profile_mapping(scores)

    rng = np.random.default_rng(2026)
    counts = np.zeros_like(scores, dtype=int)
    baseline_groups = {
        int(cluster): group[list(ALIGNMENT_FEATURES)].to_numpy()
        for cluster, group in baseline.groupby('cluster')
    }
    tianchi_groups = {
        int(cluster): group[list(ALIGNMENT_FEATURES)].to_numpy()
        for cluster, group in tianchi.groupby('cluster')
    }
    for _ in range(bootstrap_repeats):
        boot_baseline = np.vstack([
            np.median(values[rng.integers(0, len(values), len(values))], axis=0)
            for cluster in baseline_profiles.index
            for values in [baseline_groups[int(cluster)]]
        ])
        boot_tianchi = np.vstack([
            np.median(values[rng.integers(0, len(values), len(values))], axis=0)
            for cluster in tianchi_profiles.index
            for values in [tianchi_groups[int(cluster)]]
        ])
        boot_nearest, _ = _nearest_profile_mapping(
            _profile_similarity(boot_tianchi, boot_baseline)
        )
        counts[np.arange(len(boot_nearest)), boot_nearest] += 1

    retail_summary = json.loads((RETAIL_COMBINED_OUT / 'summary.json').read_text())
    cross_domain = {
        int(row['cluster']): {
            'cosmetics_drop_users': int(row['cosmetics_drop_users']),
            'electronics_users': int(row['electronics_users']),
            'event_profile_similarity': row['event_profile_similarity'],
            'supported': bool(row['cross_domain_supported']),
        }
        for row in retail_summary['clusters']
    }
    mappings = []
    for row_index, tianchi_cluster in enumerate(tianchi_profiles.index):
        target_index = int(nearest[row_index])
        target_cluster = int(baseline_profiles.index[target_index])
        mappings.append({
            'tianchi_cluster': int(tianchi_cluster),
            'joint_cluster': target_cluster,
            'similarity': float(scores[row_index, target_index]),
            'nearest_margin': float(margin[row_index]),
            'bootstrap_nearest_probability': float(
                counts[row_index, target_index] / bootstrap_repeats
            ),
        })

    result = {
        'method': (
            'Cosmetics remove_from_cart is dropped before one joint masked-GRU/K=4 fit with '
            'Electronics; Tianchi K=6 profiles then map many-to-one to pooled joint-cluster '
            'medians in the 13-feature explore/purchase space'
        ),
        'features': list(ALIGNMENT_FEATURES),
        'similarity': 'exp(-RMS distance) after pooled robust scaling',
        'bootstrap_repeats': bootstrap_repeats,
        'cluster_counts': {'cosmetics_drop': 4, 'electronics': 4,
                           'tianchi': int(tianchi_report['selected_k'])},
        'joint_retail_clusters': cross_domain,
        'tianchi_similarity_matrix': scores.tolist(),
        'tianchi_mapping': mappings,
        'distinct_joint_targets': len(set(nearest.tolist())),
        'limitations': [
            'The Cosmetics-drop/Electronics one-to-one labels arise from joint clustering.',
            'Tianchi mapping is descriptive and does not show that coupon receipt caused K=6.',
            'Joint cluster J3 has only three mature Electronics users and is shown as sparse.',
        ],
    }
    THREE_DOMAIN_ALIGNMENT_OUT.mkdir(parents=True, exist_ok=True)
    (THREE_DOMAIN_ALIGNMENT_OUT / 'summary.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8'
    )
    return result


def tianchi_centroid_action_histograms(top_n: int = 20) -> dict:
    """Plot raw Action=0/1/2 frequencies for users nearest each Tianchi centroid."""
    import matplotlib.pyplot as plt

    output = PROJECT_ROOT / 'artifacts/tianchi-online-personas'
    report = json.loads((output / 'clusters.json').read_text())
    k = int(report['selected_k'])
    embeddings = pd.read_parquet(output / 'clusters.parquet')
    z_columns = [f'z_{index}' for index in range(48)]
    mature = embeddings.sessions.ge(2)
    centers = np.asarray([row['centroid'] for row in sorted(report['profiles'], key=lambda row: row['cluster'])])
    members = embeddings.loc[mature, ['user_id', 'sessions', 'cluster']].copy()
    z = embeddings.loc[mature, z_columns].to_numpy()
    members['centroid_distance'] = np.linalg.norm(z - centers[members.cluster.to_numpy()], axis=1)
    members.to_parquet(output / f'mature_k{k}_memberships.parquet', index=False)
    representatives = (members.sort_values(['cluster', 'centroid_distance'])
                               .groupby('cluster', sort=True).head(top_n).copy())

    user_ids = set(representatives.user_id.astype('int64'))
    raw_counts = []
    for chunk in pd.read_csv(PROJECT_ROOT / 'data/raw/tianchi-o2o/ccf_online_stage1_train.csv',
                             usecols=['User_id', 'Action'], dtype={'User_id': 'Int64', 'Action': 'Int8'},
                             chunksize=500_000):
        chunk = chunk.loc[chunk.User_id.isin(user_ids) & chunk.Action.isin([0, 1, 2])]
        if not chunk.empty:
            raw_counts.append(chunk.groupby(['User_id', 'Action']).size())
    counts = (pd.concat(raw_counts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
                .reindex(index=representatives.user_id, columns=range(3), fill_value=0))
    shares = counts.div(counts.sum(axis=1), axis=0)
    action_names = ['click', 'purchase', 'coupon_receive']
    for action, name in enumerate(action_names):
        representatives[f'{name}_count'] = counts[action].to_numpy()
        representatives[f'{name}_share'] = shares[action].to_numpy()
    representatives.to_parquet(output / 'centroid_representatives.parquet', index=False)

    colors = ['#4C78A8', '#F2CF5B', '#59A14F']
    bins = {action: np.linspace(0, max(1, counts[action].max()), 11) for action in range(3)}
    fig, axes = plt.subplots(k, 3, figsize=(10.5, 2.1 * k + .5), sharex='col', sharey=True,
                             constrained_layout=True)
    summary = []
    for cluster in range(k):
        group = representatives.loc[representatives.cluster.eq(cluster)]
        row = {'cluster': cluster, 'users': len(group),
               'max_centroid_distance': float(group.centroid_distance.max()), 'actions': {}}
        for action, name in enumerate(action_names):
            values = group[f'{name}_count'].to_numpy()
            ax = axes[cluster, action]
            ax.hist(values, bins=bins[action], color=colors[action], edgecolor='white')
            ax.axvline(np.median(values), color='#333333', linestyle='--', linewidth=1)
            ax.set_title(name.replace('_', ' '))
            row['actions'][name] = {
                'mean_count': float(values.mean()), 'median_count': float(np.median(values)),
                'min_count': int(values.min()), 'max_count': int(values.max()),
                'mean_share': float(group[f'{name}_share'].mean()),
            }
            if action == 0:
                ax.set_ylabel(f'C{cluster}\nUsers')
            if cluster == k - 1:
                ax.set_xlabel('Raw event count per user')
        summary.append(row)
    fig.suptitle(f'Tianchi online: action distributions of {top_n} centroid-nearest users per cluster')
    figure = PROJECT_ROOT / 'img/persona/tianchi_centroid_action_histograms.png'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches='tight')
    plt.close(fig)
    result = {
        'selection': f'{top_n} repeat users with minimum Euclidean distance to each K={k} centroid',
        'distance_space': 'standardized 48-D masked-GRU embedding',
        'actions': action_names,
        'clusters': summary,
    }
    (output / 'centroid_action_summary.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8'
    )
    return result


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
    parser.add_argument('task', choices=('compare', 'sweep', 'stability', 'representatives',
                                         'similarity', 'electronics-drop', 'retail-combined',
                                         'joint-three-domain', 'tianchi-centroids'))
    parser.add_argument('sources', nargs='*', choices=('electronics', 'cosmetics'))
    args = parser.parse_args()
    if args.task == 'similarity':
        if args.sources:
            parser.error('similarity uses the saved Cosmetics, Electronics, and Tianchi profiles')
        cross_dataset_similarity()
        raise SystemExit(0)
    if args.task == 'electronics-drop':
        if args.sources:
            parser.error('electronics-drop uses the saved K=4 Electronics and retrained variant')
        electronics_drop_similarity()
        raise SystemExit(0)
    if args.task == 'retail-combined':
        if args.sources:
            parser.error('retail-combined uses the saved jointly trained K=4 artifacts')
        retail_combined_summary()
        raise SystemExit(0)
    if args.task == 'joint-three-domain':
        if args.sources:
            parser.error('joint-three-domain uses the saved joint K=4 and Tianchi K=6 artifacts')
        joint_three_domain_alignment()
        raise SystemExit(0)
    if args.task == 'tianchi-centroids':
        if args.sources:
            parser.error('tianchi-centroids uses the saved Tianchi online embeddings')
        tianchi_centroid_action_histograms()
        raise SystemExit(0)
    if not args.sources:
        parser.error(f'{args.task} requires at least one source')
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
