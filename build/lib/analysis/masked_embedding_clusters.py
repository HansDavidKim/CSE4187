"""Fit masked GRU embeddings and inspect centroid behavior across event datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from utils.paths import CAPSTONE_ROOT
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
    if source == "cosmetics":
        from src.irl.late_personas import load_cosmetics
        events, quality = load_cosmetics(path, sample_modulus)
        return events, quality, ["view", "cart", "remove", "purchase"]
    from src.irl.fit import load_events
    events, quality = load_events(source, path)
    return events, quality, ["view", "cart", "purchase"]


def fit(source: str, path: Path, output: Path, max_users: int = 12000,
        max_tokens: int = 64, epochs: int = 8, sample_modulus: int = 100,
        seed: int = 42, cluster_count: int | None = None) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    events, quality, action_names = source_data(source, path, sample_modulus)
    n_actions = len(action_names)
    end, pad, mask = n_actions, n_actions + 1, n_actions + 2
    rows = rows_from_events(events, max_users, max_tokens, n_actions, seed)
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
    candidates = []
    for k in range(2, 9):
        fitted = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z_scaled[mature_train])
        sample = rng.choice(len(mature_train), min(2000, len(mature_train)), replace=False)
        candidates.append((k, float(silhouette_score(z_scaled[mature_train][sample], fitted.labels_[sample])),
                           int(np.bincount(fitted.labels_, minlength=k).min()), fitted))
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
        k, train_silhouette, _, fitted = max(eligible or candidates, key=lambda item: item[1])
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
        profiles.append({
            "cluster": cluster, "users": int((labels == cluster).sum()),
            "centroid": fitted.cluster_centers_[cluster].round(5).tolist(),
            "event_share": {action_names[i]: round(float(counts[i] / len(group)), 4) for i in range(n_actions)},
            "mean_sessions": round(float(per_user_sessions.mean()), 3),
            "median_events": round(float(per_user.median()), 3),
            "buyer_share": round(float(group.loc[group.action.eq(n_actions - 1), "user"].nunique() / len(member_users)), 4),
            "median_return_gap_days": round(float(gaps.median()), 3) if len(gaps) else None,
        })
    report = {
        "source": source, "source_quality": quality, "action_names": action_names,
        "selected_users": len(rows), "segment_eligible_users": int((sessions >= 2).sum()),
        "cluster_count_mode": "fixed" if cluster_count is not None else "silhouette_with_min_size",
        "selected_k": k, "train_silhouette": train_silhouette,
        "heldout_silhouette": heldout_silhouette,
        "cluster_search": [{"k": item[0], "silhouette": item[1], "smallest_cluster": item[2]}
                            for item in candidates],
        "profiles": profiles,
        "caveats": ["Each dataset has its own embedding space; centroids are not numerically comparable across datasets.",
                    "Cosmetics has remove_from_cart; Electronics and Retailrocket do not.",
                    "Olist and Online Retail II are not included because they lack browse/cart event sequences.",
                    "These clusters describe observed behavior, not causal persona types or churn labels."]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(z_scaled, columns=[f"z_{i}" for i in range(z_scaled.shape[1])]).assign(
        user_id=users, cluster=labels, sessions=sessions
    ).to_parquet(output.with_name(output.stem.replace("_clusters", "_embeddings") + ".parquet"), index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["electronics", "retailrocket", "cosmetics"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-users", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sample-modulus", type=int, default=100)
    parser.add_argument("--cluster-count", type=int, default=None,
                        help="fix the operational cluster count, e.g. 6, to search for rare personas")
    args = parser.parse_args()
    root = CAPSTONE_ROOT / "data/raw/kaggle"
    default_path = root / "cosmetics" if args.source == "cosmetics" else root / args.source / "events.csv"
    suffix = f"_k{args.cluster_count}" if args.cluster_count else ""
    default_output = Path(f"data/models/{args.source}_masked_embedding_clusters{suffix}.json")
    result = fit(args.source, args.input or default_path, args.output or default_output,
                 max_users=args.max_users, epochs=args.epochs, sample_modulus=args.sample_modulus,
                 cluster_count=args.cluster_count)
    print(json.dumps({"source": args.source, "selected_k": result["selected_k"],
                      "heldout_silhouette": result["heldout_silhouette"]}))


if __name__ == "__main__":
    main()
