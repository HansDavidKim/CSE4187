"""Mature labels and purged rolling-origin splits."""
import numpy as np
import pandas as pd
from .data import timestamp
from .features import FeatureBuilder, DAY


def make_dataset(records, config, as_ofs=None, labeled=True):
    builder = FeatureBuilder(records, config.get("lookback_days", 30), config.get("max_length", 128))
    if as_ofs is None:
        as_ofs = pd.date_range(timestamp(config["first_as_of"]), timestamp(config["last_as_of"]), freq=f"{config.get('stride_days', 7)}D")
    end = timestamp(config["observation_end"]) if labeled else None
    horizon = config.get("horizon_days", 30) * DAY
    rows, xs, seqs, excluded = [], [], [], []
    for t in map(timestamp, as_ofs):
        if labeled and t + horizon > end:
            excluded.append({"customer_id": None, "as_of": t, "reason": "immature_label_window"})
            continue
        for customer in records["customers"].itertuples(index=False):
            user, signup = str(customer.customer_id), customer.signed_up_at
            s = builder.group("sessions", user)
            reason = None
            if pd.notna(signup) and (t - signup) / DAY < config.get("minimum_age_days", 30):
                reason = "not_yet_eligible_age"
            elif records.metadata.get("mode") == "canonical" and pd.isna(signup):
                reason = "unknown_signup"
            elif not ((s.started_at >= t - config.get("lookback_days", 30) * DAY) & (s.started_at < t)).any():
                reason = "no_recent_session"
            if reason:
                excluded.append({"customer_id": user, "as_of": t, "reason": reason})
                continue
            f, seq, truncated = builder.build(user, t, signup)
            row = {"customer_id": user, "as_of": t, "label_end": t + horizon, "truncated_groups": truncated}
            if labeled:
                if config.get("target", "visit") == "visit":
                    times = s.started_at
                elif config["target"] == "purchase" and records.metadata.get("mode") == "canonical":
                    times = builder.orders.get(user, records["orders"].iloc[:0]).purchased_at
                else:
                    raise ValueError("purchase target requires canonical orders")
                row["y"] = int(not ((times >= t) & (times < t + horizon)).any())
            rows.append(row); xs.append(f); seqs.append(seq)
    if not rows:
        if not labeled:
            return pd.DataFrame(columns=["customer_id", "as_of"]), pd.DataFrame(), [], pd.DataFrame(excluded)
        raise ValueError("No eligible samples; inspect period, observation end and eligibility")
    return pd.DataFrame(rows), pd.DataFrame(xs).replace([np.inf, -np.inf], np.nan), seqs, pd.DataFrame(excluded)


def rolling_folds(meta, test_start, n_folds=5):
    test_start = timestamp(test_start)
    dev = meta[meta.label_end <= test_start]
    dates = sorted(dev.as_of.unique())
    valid_dates = [t for t in dates if (dev.label_end <= t).any()]
    if len(valid_dates) < n_folds:
        raise ValueError(f"Only {len(valid_dates)} purged validation dates; need {n_folds}")
    folds = []
    for t in valid_dates[-n_folds:]:
        tr = meta.index[(meta.as_of < t) & (meta.label_end <= t)].to_numpy()
        va = meta.index[meta.as_of.eq(t) & (meta.label_end <= test_start)].to_numpy()
        if meta.loc[tr, "y"].nunique() < 2 or meta.loc[va, "y"].nunique() < 2:
            raise ValueError(f"Single-class fold at {t}")
        folds.append((tr, va))
    train = meta.index[meta.label_end <= test_start].to_numpy()
    test = meta.index[meta.as_of >= test_start].to_numpy()
    if not len(test):
        raise ValueError("No mature test samples")
    return folds, train, test


def early_stop_split(meta, train):
    dates = sorted(meta.loc[train, "as_of"].unique())
    for t in reversed(dates):
        fit = np.array([i for i in train if meta.at[i, "label_end"] <= t])
        val = np.array([i for i in train if meta.at[i, "as_of"] == t])
        if len(fit) and meta.loc[fit, "y"].nunique() == 2 and meta.loc[val, "y"].nunique() == 2:
            return fit, val
    raise ValueError("Insufficient history for a purged early-stopping split")
