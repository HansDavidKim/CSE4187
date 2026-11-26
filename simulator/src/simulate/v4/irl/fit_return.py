"""Observed point-in-time return choices, with active and inactive source strata."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from simulate.v4.irl.fit_shopping import ACTION_MAP, load_source, prepare_source
from simulate.v4.irl.source_contract import canonical_available_actions


CONTEXT_NAMES = ["visit_recency_days", "session_count_30d", "session_count_previous30d",
                 "purchase_session_count_30d", "cart_count_30d", "remove_count_30d",
                 "observed_tenure_days", "days_since_purchase", "last_session_viewed_price_minor",
                 "last_session_event_count", "has_observed_session"]
FEATURE_NAMES = ["log_visit_recency_days", "log_sessions_30d", "log_sessions_previous30d",
                 "log_purchase_sessions_30d", "log_carts_30d", "log_removes_30d",
                 "log_observed_tenure_days", "log_days_since_purchase", "has_purchase_history",
                 "log_last_session_viewed_price_minor", "has_last_viewed_price",
                 "log_last_session_event_count", "eligible_active30d"]


def _utc(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("return timestamps must be timezone-aware")
    return value.tz_convert("UTC")


def return_context(session_starts, events, as_of_at, purchase_item_units=None) -> dict:
    """Summarize strictly past session starts and event dictionaries; never reads a persona."""
    t0 = _utc(as_of_at)
    since30, since60 = t0 - pd.Timedelta(days=30), t0 - pd.Timedelta(days=60)
    starts = sorted(at for at in map(_utc, session_starts) if at < t0)
    history = []
    for index, event in enumerate(events):
        if event["event_type"] not in {"view", "cart", "remove_from_cart", "purchase"}:
            continue  # Auxiliary events do not alter the fitted four-action context.
        at = _utc(event["event_at"])
        if at < t0:
            history.append((at, index, event))
    history.sort(key=lambda record: (record[0], record[1]))
    recent = [record for record in history if record[0] >= since30]
    purchases = [record for record in history if record[2]["event_type"] == "purchase"]
    last_session = [record for record in history if starts and record[0] >= starts[-1]]
    last_session_id = last_session[-1][2].get("session_id") if last_session else None
    last_session = [record for record in last_session if record[2].get("session_id") == last_session_id]
    viewed_prices = []
    for _, _, event in last_session:
        if event["event_type"] != "view":
            continue
        price = event.get("unit_price_minor")
        if price is None or pd.isna(price):
            price = event.get("offer_unit_price_minor")
        if price is not None and pd.notna(price) and math.isfinite(float(price)) and float(price) > 0:
            viewed_prices.append(float(price))
    counts = Counter(record[2]["event_type"] for record in recent)
    last_session_units = 0
    for _, _, event in last_session:
        units = 1
        if purchase_item_units is not None and event["event_type"] == "purchase":
            units = purchase_item_units.get(event.get("event_id"))
            if isinstance(units, bool) or not isinstance(units, numbers.Integral) or units < 1:
                raise ValueError("purchase_item_units requires a positive integer for each purchase event")
        last_session_units += units
    return {
        "visit_recency_days": (t0 - starts[-1]).total_seconds() / 86400 if starts else None,
        "session_count_30d": sum(at >= since30 for at in starts),
        "session_count_previous30d": sum(since60 <= at < since30 for at in starts),
        "purchase_session_count_30d": len({record[2]["session_id"] for record in recent if record[2]["event_type"] == "purchase"}),
        "cart_count_30d": counts["cart"], "remove_count_30d": counts["remove_from_cart"],
        "observed_tenure_days": (t0 - starts[0]).total_seconds() / 86400 if starts else 0.0,
        "days_since_purchase": (t0 - purchases[-1][0]).total_seconds() / 86400 if purchases else None,
        "last_session_viewed_price_minor": float(np.median(viewed_prices)) if viewed_prices else None,
        "last_session_event_count": last_session_units, "has_observed_session": bool(starts),
    }


def feature_support(available_actions=None) -> np.ndarray:
    """Unavailable source actions do not supply zero-valued feature evidence."""
    actions = canonical_available_actions(available_actions)
    support = np.ones(len(FEATURE_NAMES), dtype=bool)
    for action, names in {
        "cart": ("log_carts_30d",), "remove": ("log_removes_30d",),
        "purchase": ("log_purchase_sessions_30d", "log_days_since_purchase", "has_purchase_history"),
        "view": ("log_last_session_viewed_price_minor", "has_last_viewed_price"),
    }.items():
        if action not in actions:
            support[[FEATURE_NAMES.index(name) for name in names]] = False
    return support


def context_features(context: dict, *, available_actions=None) -> list[float]:
    actions = canonical_available_actions(available_actions)
    if actions != canonical_available_actions():
        context = context.copy()
        for action, names in {"cart": ("cart_count_30d",), "remove": ("remove_count_30d",),
                              "purchase": ("purchase_session_count_30d", "days_since_purchase"),
                              "view": ("last_session_viewed_price_minor",)}.items():
            if action not in actions:
                for name in names:
                    context[name] = None

    def value(name):
        number = float(context[name]) if context.get(name) is not None else 0.0
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"invalid nonnegative return context: {name}")
        return number

    names = ("visit_recency_days", "session_count_30d", "session_count_previous30d",
             "purchase_session_count_30d", "cart_count_30d", "remove_count_30d", "observed_tenure_days",
             "days_since_purchase")
    return [*[math.log1p(value(name)) for name in names],
            float(context.get("days_since_purchase") is not None),
            math.log1p(value("last_session_viewed_price_minor")),
            float(context.get("last_session_viewed_price_minor") is not None),
            math.log1p(value("last_session_event_count")), float(value("session_count_30d") > 0)]


def return_probability(model: dict, context: dict) -> float:
    """Evaluate the source-fitted MaxEnt logit from JSON, without changing its intercept."""
    if model.get("schema_version") != 1 or model["feature_names"] != FEATURE_NAMES:
        raise ValueError("incompatible return model")
    actions = canonical_available_actions(model.get("available_actions"))
    support = feature_support(actions)
    if ("feature_support" in model and (not isinstance(model["feature_support"], list)
            or any(type(value) is not bool for value in model["feature_support"])
            or model["feature_support"] != support.tolist())
            or not support.all() and "feature_support" not in model):
        raise ValueError("return feature support differs from observed actions")
    if not support.all():
        for name, reference in (("coef", 0.), ("scaler_mean", 0.), ("scaler_scale", 1.),
                                ("feature_min", 0.), ("feature_max", 0.)):
            values = np.asarray(model[name], dtype=float)
            if (values.shape != support.shape or not np.isfinite(values).all()
                    or not np.all(values[~support] == reference)
                    or name == "scaler_scale" and (values <= 0).any()):
                raise ValueError("unavailable return features must remain unused reference slots")
    if not context.get("has_observed_session", False):
        return float(model["cold_start_probability"])
    if not support.all():
        values = np.asarray(context_features(context, available_actions=actions))[support]
        values = np.clip(values, np.asarray(model["feature_min"])[support], np.asarray(model["feature_max"])[support])
        values = (values - np.asarray(model["scaler_mean"])[support]) / np.asarray(model["scaler_scale"])[support]
        logit = float(np.asarray(model["coef"])[support] @ values + model["intercept"])
        return float(1 / (1 + math.exp(-max(-40, min(40, logit)))))
    values = np.clip(np.asarray(context_features(context)), model["feature_min"], model["feature_max"])
    values = (values - model["scaler_mean"]) / model["scaler_scale"]
    logit = float(np.asarray(model["coef"]) @ values + model["intercept"])
    return float(1 / (1 + math.exp(-max(-40, min(40, logit)))))


def build_panel(rows: pd.DataFrame, seed: int = 42, *, available_actions=None, cutoff_at=None):
    actions = canonical_available_actions(available_actions)
    if actions != canonical_available_actions() and not rows.event_type.map(ACTION_MAP).isin(actions).all():
        raise ValueError("return source contains an unsupported demonstrated action")
    start = rows["at"].min().floor("D")
    end = rows["at"].max().floor("D") + pd.Timedelta(days=1)
    snapshots = list(pd.date_range(start + pd.Timedelta(days=30), end - pd.Timedelta(days=30), freq="15D"))
    if len(snapshots) < 3:
        raise ValueError("return fitting requires at least three fully observed monthly horizons")
    cutoff = snapshots[-1] if cutoff_at is None else _utc(cutoff_at)
    if not start < cutoff < end:
        raise ValueError("return cutoff must be strictly inside source observation bounds")
    records, labels = [], []
    first_return_eligible = first_returned = 0
    first_return_support = Counter()
    last_first_return_window_end = None
    for uid, customer in rows.groupby("user_id", sort=True):
        starts = customer.groupby("session", sort=False)["at"].min().sort_values().tolist()
        events = [{"event_at": row.at, "event_type": row.event_type, "session_id": str(row.session),
                   "unit_price_minor": None if pd.isna(row.price_minor) else int(row.price_minor)}
                  for row in customer.itertuples(index=False)]
        train_user = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
        first_window_end = starts[0] + pd.Timedelta(days=30)
        first_return_support["source_users"] += 1
        first_return_support["training_users"] += train_user
        first_return_support["heldout_users_excluded"] += not train_user
        first_return_support["training_windows_not_available_before_cutoff"] += train_user and first_window_end >= cutoff
        first_return_support["training_windows_right_censored_at_observation_end"] += train_user and first_window_end > end
        if train_user and first_window_end < cutoff:
            first_return_eligible += 1
            first_returned += len(starts) >= 2 and starts[1] < first_window_end
            last_first_return_window_end = (first_window_end if last_first_return_window_end is None
                                          else max(last_first_return_window_end, first_window_end))
        for t0 in snapshots:
            if starts[0] >= t0:
                continue
            context = return_context(starts, events, t0)
            records.append(context_features(context, available_actions=actions))
            returned = any(t0 <= at < t0 + pd.Timedelta(days=30) for at in starts)
            labels.append({"user_id": uid, "as_of_at": t0, "outcome_end_at": t0 + pd.Timedelta(days=30),
                           "returned": int(returned), "train_user": train_user,
                           "eligible_active30d": context["session_count_30d"] > 0})
    if not first_return_eligible:
        raise ValueError("cold-start return fitting requires train users with complete first-session 30-day windows strictly before cutoff")
    return np.asarray(records), pd.DataFrame(labels), {
        "observation_start": start.isoformat(), "observation_end": end.isoformat(),
        "snapshots": [at.isoformat() for at in snapshots],
        "cold_start_probability": first_returned / first_return_eligible,
        "cold_start_eligible": first_return_eligible, "cold_start_returned": int(first_returned),
        "cold_start_fit": {"cutoff_at": cutoff.isoformat(), "horizon_days": 30,
            "user_rule": f"sha256({seed}:user_id).digest()[0] < 205",
            "label_available_rule": "first_observed_session_start + 30d < cutoff",
            "training_windows": first_return_eligible, "returned": int(first_returned),
            "max_outcome_end_at": last_first_return_window_end.isoformat(),
            **dict(first_return_support)},
    }


def _metrics(y, probability, baseline, users):
    if not len(y):
        return {"records": 0, "users": 0, "nll": None, "frequency_baseline_nll": None}
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    b = np.clip(baseline, 1e-12, 1 - 1e-12)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    null = -(y * np.log(b) + (1 - y) * np.log1p(-b))
    user_counts = users.value_counts()
    weights = users.map(1 / user_counts).to_numpy()
    return {"records": len(y), "users": len(user_counts), "return_rate": float(y.mean()),
            "mean_probability": float(p.mean()), "nll": float(loss.mean()),
            "frequency_baseline_nll": float(null.mean()),
            "customer_balanced_nll": float(np.average(loss, weights=weights)),
            "customer_balanced_frequency_baseline_nll": float(np.average(null, weights=weights)),
            "brier": float(np.mean((p - y) ** 2)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None}


def fit_return(events: pd.DataFrame, provenance: dict | None = None, seed: int = 42, *,
               available_actions=None, cutoff_at=None) -> dict:
    rows, _, source_quality = prepare_source(events)
    return _fit_prepared_return(rows, provenance, seed, source_quality=source_quality,
                                available_actions=available_actions, cutoff_at=cutoff_at)


def _fit_prepared_return(rows: pd.DataFrame, provenance: dict | None = None, seed: int = 42, *,
                         source_quality: dict, available_actions=None, cutoff_at=None) -> dict:
    actions = canonical_available_actions(available_actions)
    support = feature_support(actions)
    features, labels, observation = build_panel(rows, seed, available_actions=actions, cutoff_at=cutoff_at)
    design = features if support.all() else features[:, support]
    cutoff = pd.Timestamp(observation["cold_start_fit"]["cutoff_at"])
    early = (labels.outcome_end_at.le(cutoff) if cutoff_at is None
             else labels.outcome_end_at.lt(cutoff)).to_numpy()
    late = labels.as_of_at.ge(cutoff).to_numpy()
    train = labels.train_user.to_numpy() & early
    y = labels.returned.to_numpy()
    if train.sum() < 20 or len(np.unique(y[train])) < 2:
        raise ValueError("return fitting requires positive and negative train examples")
    scaler = StandardScaler().fit(design[train])
    x = scaler.transform(design)
    counts = labels.loc[train, "user_id"].value_counts()
    weights = labels.loc[train, "user_id"].map(1 / counts).to_numpy()
    weights *= len(weights) / weights.sum()
    estimator = LogisticRegression(C=1.0, max_iter=250, solver="lbfgs", tol=1e-6, random_state=seed)
    estimator.fit(x[train], y[train], sample_weight=weights)
    probability = estimator.predict_proba(x)[:, 1]
    feature_min, feature_max = design[train].min(axis=0), design[train].max(axis=0)
    bounded = scaler.transform(np.clip(design, feature_min, feature_max))
    bounded_probability = estimator.predict_proba(bounded)[:, 1]
    global_rate = float(y[train].mean())
    strata_rates = {}
    for active in (False, True):
        selected = train & labels.eligible_active30d.eq(active).to_numpy()
        strata_rates[str(active)] = float(y[selected].mean()) if selected.any() else global_rate
    baseline = np.array([strata_rates[str(bool(value))] for value in labels.eligible_active30d])
    validation = {"temporal_cutoff": cutoff.isoformat(),
                  "split": "hash-disjoint users; train outcomes end by temporal cutoff; final source snapshot is time holdout",
                  "purged_rows": int((~early & ~late).sum()), "training_return_frequency": global_rate,
                  "training_active_stratum_return_frequency": strata_rates,
                  "baseline": "training return frequency conditional on past active30d flag; no outcome-defined strata",
                  "unclipped_evaluation": "Main metrics use original held-out source features; runtime clipped metrics are reported separately.",
                  "splits": {}}
    if cutoff_at is not None:
        validation["split"] = "hash-disjoint users; train outcome_end_at strictly before cutoff; later windows start at/after cutoff"
        validation["fit_label_available_rule"] = "outcome_end_at < cutoff"
    masks = {"train": train, "user_holdout": ~labels.train_user.to_numpy() & early,
             "time_holdout": labels.train_user.to_numpy() & late,
             "user_and_time_holdout": ~labels.train_user.to_numpy() & late}
    for name, mask in masks.items():
        result = _metrics(y[mask], probability[mask], baseline[mask], labels.loc[mask, "user_id"])
        result["runtime_clipped"] = _metrics(y[mask], bounded_probability[mask], baseline[mask], labels.loc[mask, "user_id"])
        result["strata"] = {}
        for active in (False, True):
            selected = mask & labels.eligible_active30d.eq(active).to_numpy()
            result["strata"]["active30d" if active else "inactive30d"] = _metrics(
                y[selected], probability[selected], baseline[selected], labels.loc[selected, "user_id"])
        validation["splits"][name] = result
    coefficients, mean, scale = estimator.coef_[0], scaler.mean_, scaler.scale_
    metadata = {}
    if not support.all():
        def embedded(values, reference):
            result = np.full(len(FEATURE_NAMES), reference)
            result[support] = values
            return result
        coefficients = embedded(coefficients, 0.)
        mean, scale = embedded(mean, 0.), embedded(scale, 1.)
        feature_min, feature_max = embedded(feature_min, 0.), embedded(feature_max, 0.)
        metadata = {"available_actions": list(actions), "feature_support": support.tolist()}
    return {"schema_version": 1, "model_type": "point_in_time_binary_maxent_return_logit", **metadata,
            "seed": seed, "source": provenance or {"dataset": "explicit deterministic test fixture"},
            **observation, "source_quality": source_quality, "context_names": CONTEXT_NAMES,
            "feature_names": FEATURE_NAMES, "coef": coefficients.tolist(),
            "intercept": float(estimator.intercept_[0]), "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(), "n_iter": estimator.n_iter_.tolist(),
            "feature_min": feature_min.tolist(), "feature_max": feature_max.tolist(),
            "runtime_clipping": "Each log-transformed feature is clipped to train-only extrema before scaling; raw held-out metrics remain unmodified.",
            "calibration_reference_probabilities": np.quantile(probability[train & labels.eligible_active30d.to_numpy()],
                                                                 np.linspace(0, 1, min(2048, int((train & labels.eligible_active30d.to_numpy()).sum())))).tolist(),
            "regularization_C": estimator.C, "currency": "SOURCE_UNSPECIFIED", "minor_unit": "0.01",
            "validation": validation,
            "limitations": [
                "Observational return association, not causal effects of activity, prices or coupons.",
                "Observed tenure begins at first recorded session, not verified signup; earlier history is left-censored.",
                "Active30d and previously observed inactive users are both source populations; their validation strata remain separate.",
                "Cold-start probability is a complete-followup first-observed-session return proxy, not signup-to-first-visit probability.",
                "Snapshots every fifteen days have overlapping outcomes; purging separates train outcome windows from the final temporal test snapshot.",
                "Source purchase rows lack true order and quantity; purchase-session features avoid treating item rows as separate orders.",
                "Generated fixed prices and canonical checkout headers differ from source item events, especially last-session event counts.",
                "Five observed months cannot validate twenty-four-month behavior or permanent account exit.",
                "A generator intercept shift to a specified churn target is an explicit scenario and does not change fitted coefficients.",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmetics", type=Path, default=Path("data/raw/kaggle/cosmetics"))
    parser.add_argument("--output", type=Path, default=Path("data/models/return_model.json"))
    parser.add_argument("--sample-modulus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--available-actions", nargs="+", choices=tuple(ACTION_MAP.values()))
    parser.add_argument("--prepared-source", type=Path)
    parser.add_argument("--domain", choices=("cosmetics", "electronics", "tianchi"))
    parser.add_argument("--cutoff-at")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"candidate must use a new artifact path: {args.output}")
    if args.prepared_source:
        from simulate.v4.irl.source_contract import load_prepared_source
        if args.domain is None or args.cutoff_at is None:
            raise ValueError("prepared return fitting requires --domain and --cutoff-at")
        rows, descriptor = load_prepared_source(args.prepared_source, domain=args.domain, cutoff_at=args.cutoff_at)
        contract = descriptor["source_contract"]
        actions = canonical_available_actions(contract["observed_actions"])
        if args.available_actions is not None and canonical_available_actions(args.available_actions) != actions:
            raise ValueError("return actions differ from the prepared source contract")
        artifact = _fit_prepared_return(rows, descriptor["source_provenance"], args.seed,
            source_quality=descriptor.get("source_quality", {"source_rows": len(rows)}),
            available_actions=actions, cutoff_at=args.cutoff_at)
        artifact["source_contract"] = contract
    else:
        if args.domain is not None:
            raise ValueError("--domain requires --prepared-source")
        events, provenance = load_source(args.cosmetics, args.sample_modulus)
        artifact = fit_return(events, provenance, args.seed, available_actions=args.available_actions, cutoff_at=args.cutoff_at)
    artifact["source"]["fitter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=f".{args.output.name}-",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False))
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "validation": artifact["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
