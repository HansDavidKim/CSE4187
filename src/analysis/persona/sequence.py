"""Fit masked GRU embeddings and inspect centroid behavior across event datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from utils.paths import CAPSTONE_ROOT, PROJECT_ROOT
from utils.masking import MASK, mask_tokens


def rows_from_events(events: pd.DataFrame, max_users: int, max_tokens: int,
                     n_actions: int, seed: int) -> list[tuple[int, list[int], int]]:
    sessions = events.groupby("user", sort=False).session.nunique()
    rng = np.random.default_rng(seed)
    buckets = [sessions[sessions.eq(1)].index.to_numpy(),
               sessions[sessions.eq(2)].index.to_numpy(),
               sessions[sessions.ge(3)].index.to_numpy()]
    chosen = []
    for bucket in buckets:
        chosen.extend(rng.choice(bucket, min(len(bucket), max_users // 3), replace=False).tolist())
    remaining = max_users - len(chosen)
    available = sessions.index.difference(pd.Index(chosen)).to_numpy()
    if remaining and len(available):
        chosen.extend(rng.choice(available, min(remaining, len(available)), replace=False).tolist())
    selected = events[events.user.isin(chosen)]
    rows = []
    for user, frame in selected.groupby("user", sort=False):
        tokens, previous = [], None
        for session, action in zip(frame.session.to_numpy(), frame.action.to_numpy()):
            if previous is not None and session != previous:
                tokens.append(n_actions)
            tokens.append(int(action))
            previous = session
        tokens.append(n_actions)
        rows.append((int(user), tokens[:max_tokens], frame.session.nunique()))
    return rows


def padded(rows: list[list[int]], max_tokens: int, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([min(len(row), max_tokens) for row in rows], dtype=torch.long)
    result = torch.full((len(rows), max_tokens), pad, dtype=torch.long)
    for i, row in enumerate(rows):
        result[i, :lengths[i]] = torch.tensor(row[:max_tokens], dtype=torch.long)
    return result, lengths


class MaskedGRU(nn.Module):
    def __init__(self, vocab_size: int, pad: int, hidden: int = 24):
        super().__init__()
        self.token = nn.Embedding(vocab_size, 16, padding_idx=pad)
        self.encoder = nn.GRU(16, hidden, batch_first=True, bidirectional=True)
        self.readout = nn.Linear(hidden * 2, vocab_size - 2)

    def encode(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            self.token(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        return torch.cat((hidden[0], hidden[1]), dim=-1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            self.token(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=tokens.shape[1]
        )
        return self.readout(encoded)


def source_data(source: str, path: Path, sample_modulus: int) -> tuple[pd.DataFrame, dict, list[str]]:
    if source not in ('electronics', 'cosmetics', 'electronics_drop', 'retail_combined',
                      'tianchi') or sample_modulus < 1:
        raise ValueError(
            'supported sources are electronics/cosmetics/electronics_drop/retail_combined/tianchi '
            'with a positive modulus'
        )
    if source == 'retail_combined':
        cosmetics, cosmetics_quality, actions = source_data(
            'electronics_drop', path / 'cosmetics', sample_modulus
        )
        electronics, electronics_quality, _ = source_data(
            'electronics', path / 'electronics/events.csv', sample_modulus
        )
        cosmetics = cosmetics.copy()
        electronics = electronics.copy()
        cosmetics['user'] = cosmetics.user.astype('int64') * 2
        electronics['user'] = electronics.user.astype('int64') * 2 + 1
        events = (pd.concat([cosmetics, electronics], ignore_index=True)
                    .sort_values(['user', 'at'], kind='stable'))
        return events, {
            'sample_modulus': sample_modulus,
            'domains': {
                'cosmetics_drop': cosmetics_quality,
                'electronics': electronics_quality,
            },
            'user_namespace': 'cosmetics_drop=2*user_id; electronics=2*user_id+1',
        }, actions
    if source == 'tianchi':
        parts, raw_rows, sample_action_rows = [], 0, 0
        action_counts = np.zeros(3, dtype=np.int64)
        for chunk in pd.read_csv(path, usecols=['User_id', 'Action', 'Date_received', 'Date'],
                                 dtype={'User_id': 'Int64', 'Action': 'Int8',
                                        'Date_received': 'string', 'Date': 'string'},
                                 chunksize=500_000):
            raw_rows += len(chunk)
            counts = chunk.Action.value_counts()
            action_counts += np.array([counts.get(i, 0) for i in range(3)], dtype=np.int64)
            selected = chunk.User_id.mod(sample_modulus).eq(0).fillna(False)
            chunk = chunk.loc[selected & chunk.Action.isin([0, 1, 2])].copy()
            sample_action_rows += len(chunk)
            day = chunk.Date.copy()
            receipt = chunk.Action.eq(2)
            day.loc[receipt] = chunk.loc[receipt, 'Date_received']
            chunk['at'] = pd.to_datetime(day, format='%Y%m%d', utc=True, errors='coerce')
            chunk = chunk.loc[chunk.User_id.notna() & chunk['at'].notna()]
            if chunk.empty:
                continue
            daily = (chunk.assign(click=chunk.Action.eq(0).astype('int8'),
                                  purchase=chunk.Action.eq(1).astype('int8'),
                                  receive=chunk.Action.eq(2).astype('int8'))
                           .groupby(['User_id', 'at'], as_index=False)[['click', 'purchase', 'receive']].max())
            parts.append(daily)
        daily = (pd.concat(parts, ignore_index=True)
                   .groupby(['User_id', 'at'], as_index=False)[['click', 'purchase', 'receive']].max())
        daily['action'] = (daily.click + 2 * daily.purchase + 4 * daily.receive - 1).astype('int8')
        events = pd.DataFrame({
            'user': daily.User_id.astype('int64'),
            'session': daily['at'].dt.strftime('%Y%m%d'),
            'at': daily['at'],
            'action': daily.action,
            'product': pd.Series(pd.NA, index=daily.index, dtype='Int64'),
        }).sort_values(['user', 'at'], kind='stable')
        if events.empty:
            raise ValueError('no valid Tianchi online action days after sampling')
        quality = {
            'raw_rows': raw_rows,
            'raw_action_counts': {'click': int(action_counts[0]), 'purchase': int(action_counts[1]),
                                  'coupon_receive': int(action_counts[2])},
            'sample_action_rows': sample_action_rows,
            'sample_user_days': len(events),
            'sample_modulus': sample_modulus,
            'temporal_grain': 'one seven-state action-combination token per user-day; '
                              'within-day order unavailable',
        }
        return events, quality, [
            'click_only_day', 'purchase_only_day', 'click_purchase_day',
            'coupon_receive_only_day', 'click_receive_day', 'purchase_receive_day',
            'click_purchase_receive_day',
        ]

    cosmetics_source = source in ('cosmetics', 'electronics_drop')
    files = sorted(path.glob('20??-*.csv')) if cosmetics_source else [path]
    actions = ['view', 'cart', 'remove', 'purchase'] if source == 'cosmetics' else ['view', 'cart', 'purchase']
    action_map = {'view': 0, 'cart': 1, 'remove_from_cart': 2, 'purchase': 3} if source == 'cosmetics' else {
        'view': 0, 'cart': 1, 'purchase': 2}
    parts, rows = [], 0
    for file in files:
        for chunk in pd.read_csv(file, usecols=['user_id', 'event_time', 'event_type', 'user_session', 'product_id'],
                                 dtype={'user_id': 'Int64', 'user_session': 'string', 'product_id': 'Int64'}, chunksize=500_000):
            rows += len(chunk)
            selected = chunk.user_id.mod(sample_modulus).eq(0).fillna(False)
            chunk = chunk.loc[selected].copy()
            at = pd.to_datetime(chunk.event_time, utc=True, errors='coerce')
            action = chunk.event_type.map(action_map)
            valid = chunk.user_id.notna() & chunk.user_session.notna() & at.notna() & action.notna()
            parts.append(pd.DataFrame({'user': chunk.loc[valid, 'user_id'].astype('int64'),
                                       'session': chunk.loc[valid, 'user_session'], 'at': at[valid],
                                       'action': action[valid].astype('int8'),
                                       'product': chunk.loc[valid, 'product_id']}))
    if not parts:
        raise ValueError(f'no input CSV rows found under {path}')
    events = pd.concat(parts, ignore_index=True)
    if events.empty:
        raise ValueError('no valid events after sampling')
    events = events.sort_values(['user', 'at'], kind='stable')
    return events, {
        'raw_rows': rows,
        'sample_rows': len(events),
        'sample_modulus': sample_modulus,
        'dropped_actions': ['remove_from_cart'] if source == 'electronics_drop' else [],
    }, actions


def fit(source: str, path: Path, output: Path, max_users: int = 12000,
        max_tokens: int = 64, epochs: int = 8, sample_modulus: int = 100,
        seed: int = 42, sample_seed: int | None = None,
        cluster_count: int | None = None) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    events, quality, action_names = source_data(source, path, sample_modulus)
    n_actions = len(action_names)
    end, pad, mask = n_actions, n_actions + 1, n_actions + 2
    row_seed = seed if sample_seed is None else sample_seed
    if source == 'retail_combined':
        cosmetics_events = events.loc[events.user.mod(2).eq(0)]
        electronics_events = events.loc[events.user.mod(2).eq(1)]
        users_per_domain = min(
            max_users // 2,
            cosmetics_events.user.nunique(),
            electronics_events.user.nunique(),
        )
        rows = (
            rows_from_events(cosmetics_events, users_per_domain, max_tokens, n_actions, row_seed)
            + rows_from_events(electronics_events, users_per_domain, max_tokens, n_actions, row_seed)
        )
        quality['balanced_users_per_domain'] = users_per_domain
    else:
        rows = rows_from_events(events, max_users, max_tokens, n_actions, row_seed)
    users = np.array([row[0] for row in rows])
    sessions = np.array([row[2] for row in rows])
    tokens, lengths = padded([row[1] for row in rows], max_tokens, pad)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    train, test = perm[:int(0.8 * len(rows))], perm[int(0.8 * len(rows)):]
    model = MaskedGRU(n_actions + 3, pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train) / 128)))):
            masked_tokens, selected = mask_tokens(tokens[idx], lengths[idx], 0.15, generator)
            masked_tokens[masked_tokens.eq(MASK)] = mask
            logits = model(masked_tokens, lengths[idx])
            loss = nn.functional.cross_entropy(logits[selected], tokens[idx][selected])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        z = np.concatenate([model.encode(tokens[i:i + 256], lengths[i:i + 256]).numpy()
                            for i in range(0, len(tokens), 256)])
    scaler = StandardScaler().fit(z[train])
    z_scaled = scaler.transform(z)
    mature_train = train[sessions[train] >= 2]
    mature_test = test[sessions[test] >= 2]
    if source == 'retail_combined':
        def balanced_domain_indices(indices: np.ndarray) -> np.ndarray:
            cosmetics_indices = indices[users[indices] % 2 == 0]
            electronics_indices = indices[users[indices] % 2 == 1]
            size = min(len(cosmetics_indices), len(electronics_indices))
            return np.concatenate([
                rng.choice(cosmetics_indices, size, replace=False),
                rng.choice(electronics_indices, size, replace=False),
            ])

        mature_train = balanced_domain_indices(mature_train)
        mature_test = balanced_domain_indices(mature_test)
    candidates = []
    for k in range(2, 9):
        fitted = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z_scaled[mature_train])
        alternate = KMeans(n_clusters=k, n_init=10, random_state=seed + 1).fit(z_scaled[mature_train])
        sample = rng.choice(len(mature_train), min(2000, len(mature_train)), replace=False)
        candidates.append((k, float(silhouette_score(z_scaled[mature_train][sample], fitted.labels_[sample])),
                           int(np.bincount(fitted.labels_, minlength=k).min()), fitted,
                           float(adjusted_rand_score(fitted.labels_, alternate.labels_))))
    eligible = [item for item in candidates if item[2] >= max(30, int(0.02 * len(mature_train)))]
    if cluster_count is not None:
        if cluster_count < 2 or cluster_count > 12:
            raise ValueError("cluster_count must be between 2 and 12")
        fitted = KMeans(n_clusters=cluster_count, n_init=20, random_state=seed).fit(z_scaled[mature_train])
        k = cluster_count
        train_sample = rng.choice(len(mature_train), min(2000, len(mature_train)), replace=False)
        train_silhouette = float(silhouette_score(z_scaled[mature_train][train_sample],
                                                   fitted.labels_[train_sample]))
    else:
        k, train_silhouette, _, fitted, _ = max(eligible or candidates, key=lambda item: item[1])
    labels = fitted.predict(z_scaled)
    sample = rng.choice(len(mature_test), min(2000, len(mature_test)), replace=False)
    heldout_silhouette = float(silhouette_score(z_scaled[mature_test][sample], labels[mature_test][sample]))
    profiles = []
    for cluster in range(k):
        member_users = set(users[labels == cluster])
        group = events[events.user.isin(member_users)]
        counts = group.action.value_counts().reindex(range(n_actions), fill_value=0)
        per_user = group.groupby("user").size()
        per_user_sessions = group.groupby("user").session.nunique()
        starts = group.groupby(["user", "session"], sort=False).at.min().reset_index()
        gaps = starts.groupby("user").at.diff().dt.total_seconds().div(86400).dropna()
        purchase_actions = [i for i, name in enumerate(action_names) if 'purchase' in name]
        profiles.append({
            "cluster": cluster, "users": int((labels == cluster).sum()),
            "centroid": fitted.cluster_centers_[cluster].round(5).tolist(),
            "event_share": {action_names[i]: round(float(counts[i] / len(group)), 4) for i in range(n_actions)},
            "mean_sessions": round(float(per_user_sessions.mean()), 3),
            "median_events": round(float(per_user.median()), 3),
            "buyer_share": round(float(group.loc[group.action.isin(purchase_actions), "user"].nunique() / len(member_users)), 4),
            "median_return_gap_days": round(float(gaps.median()), 3) if len(gaps) else None,
        })
    report = {
        "source": source, "source_quality": quality, "action_names": action_names,
        "selected_users": len(rows), "segment_eligible_users": int((sessions >= 2).sum()),
        "cluster_count_mode": "fixed" if cluster_count is not None else "silhouette_with_min_size",
        "selected_k": k, "train_silhouette": train_silhouette,
        "heldout_silhouette": heldout_silhouette,
        "cluster_search": [{"k": item[0], "silhouette": item[1], "smallest_cluster": item[2],
                            "seed_ari": item[4]}
                            for item in candidates],
        "profiles": profiles,
        "caveats": ["These clusters describe observed behavior, not causal persona types or churn labels."]
    }
    embedding_frame = pd.DataFrame(
        z_scaled, columns=[f"z_{i}" for i in range(z_scaled.shape[1])]
    ).assign(user_id=users, cluster=labels, sessions=sessions)
    if source == 'retail_combined':
        domains = np.where(users % 2 == 0, 'cosmetics_drop', 'electronics')
        embedding_frame['domain'] = domains
        embedding_frame['original_user_id'] = users // 2
        mature = sessions >= 2
        report['domain_cluster_nmi'] = float(
            normalized_mutual_info_score(domains[mature], labels[mature])
        )
        report['domain_selected_users'] = {
            domain: int((domains == domain).sum()) for domain in np.unique(domains)
        }
        report['domain_mature_users'] = {
            domain: int(((domains == domain) & mature).sum()) for domain in np.unique(domains)
        }
        report['balanced_mature_train_users_per_domain'] = int(len(mature_train) // 2)
        report['balanced_mature_test_users_per_domain'] = int(len(mature_test) // 2)
        report['domain_cluster_profiles'] = []
        for cluster in range(k):
            for domain, parity in (('cosmetics_drop', 0), ('electronics', 1)):
                cluster_mature = (labels == cluster) & mature
                member_users = set(users[cluster_mature & (users % 2 == parity)])
                group = events[events.user.isin(member_users)]
                counts = group.action.value_counts().reindex(range(n_actions), fill_value=0)
                report['domain_cluster_profiles'].append({
                    'cluster': cluster,
                    'domain': domain,
                    'users': len(member_users),
                    'cluster_domain_share': float(
                        len(member_users) / max(1, int(cluster_mature.sum()))
                    ),
                    'event_share': {
                        action_names[i]: float(counts[i] / len(group)) if len(group) else None
                        for i in range(n_actions)
                    },
                })
        report['caveats'].extend([
            'Cosmetics remove_from_cart events are deleted before joint training.',
            'A shared encoder and KMeans remove independent-coordinate alignment, but domain-specific '
            'sampling, calendar coverage, catalog and logging differences remain.',
        ])
    else:
        report['caveats'].extend([
            'Each dataset has its own embedding space; centroids are not numerically comparable across datasets.',
            'Cosmetics has remove_from_cart; Electronics and Tianchi do not.',
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    embedding_frame.to_parquet(
        output.with_name(output.stem.replace("_clusters", "_embeddings") + ".parquet"), index=False
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["electronics", "cosmetics", "electronics_drop", "retail_combined",
                             "tianchi"],
        required=True,
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-users", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sample-modulus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--cluster-count", type=int, default=None,
                        help="fix the operational cluster count, e.g. 6, to search for rare personas")
    args = parser.parse_args()
    root = CAPSTONE_ROOT / "data/raw/kaggle"
    if args.source == 'tianchi':
        default_path = PROJECT_ROOT / 'data/raw/tianchi-o2o/ccf_online_stage1_train.csv'
    elif args.source == 'retail_combined':
        default_path = root
    else:
        default_path = (root / "cosmetics" if args.source in ("cosmetics", "electronics_drop")
                        else root / args.source / "events.csv")
    suffix = f"_k{args.cluster_count}" if args.cluster_count else ""
    default_output = Path(f"data/models/{args.source}_masked_embedding_clusters{suffix}.json")
    result = fit(args.source, args.input or default_path, args.output or default_output,
                 max_users=args.max_users, max_tokens=args.max_tokens, epochs=args.epochs,
                 sample_modulus=args.sample_modulus,
                 seed=args.seed, sample_seed=args.sample_seed,
                 cluster_count=args.cluster_count)
    print(json.dumps({"source": args.source, "selected_k": result["selected_k"],
                      "heldout_silhouette": result["heldout_silhouette"]}))


if __name__ == "__main__":
    main()
