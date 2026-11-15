"""Fit source-conditioned shopping choices and donor profiles, without named personas.

This is a one-step maximum-entropy choice model, not identified long-horizon
reward or causal price elasticity. Cosmetics rows have no order or quantity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ACTIONS = ("view", "cart", "remove", "purchase", "end")


SOURCE_COLUMNS = ("event_time", "event_type", "product_id", "category_id", "category_code",
                  "brand", "price", "user_id", "user_session")
ACTION_MAP = {"view": "view", "cart": "cart", "remove_from_cart": "remove", "purchase": "purchase"}
PREVIOUS_ACTIONS = ("start", "view", "cart", "remove", "purchase")
FEATURE_NAMES = [*[f"previous_{name}" for name in PREVIOUS_ACTIONS],
                 "log_relative_offer_price", "log_cart_quantity", "log_cart_total_quantity",
                 "log_days_since_first_observed", "log_days_since_purchase", "has_purchase_history",
                 "log_purchase_count", "log_product_purchase_count", "category_purchase_share",
                 "log_session_event_count", "log_prior_gap_seconds", "has_observed_offer"]
CONTEXT_NAMES = ["previous_action", "unit_price_minor", "reference_price_minor", "cart_quantity",
                 "cart_total_quantity", "days_since_first_observed", "days_since_purchase",
                 "purchase_count", "product_purchase_count", "category_purchase_share",
                 "session_event_count", "prior_gap_seconds"]


def price_minor(value: str) -> int | None:
    """Positive observed source price in 0.01 source units; keep original text separately."""
    try:
        price = Decimal(str(value))
        if not price.is_finite() or price <= 0:
            return None
        result = int((price * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return result if result > 0 else None
    except (InvalidOperation, ValueError, OverflowError):
        return None


def context_features(context: dict) -> list[float]:
    """Public shared feature mapping; every input describes information already observed."""
    def number(name: str) -> float:
        value = context.get(name)
        result = float(value) if value is not None else 0.0
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"invalid nonnegative choice context: {name}")
        return result

    previous = context.get("previous_action", "start")
    if previous not in PREVIOUS_ACTIONS:
        raise ValueError("unrecognized previous_action")
    unit = number("unit_price_minor")
    reference = number("reference_price_minor") or unit
    category_share = number("category_purchase_share")
    if category_share > 1:
        raise ValueError("category_purchase_share must be in [0, 1]")
    return [*[float(previous == name) for name in PREVIOUS_ACTIONS],
            math.log((unit + 1) / (reference + 1)) if unit else 0.0,
            math.log1p(number("cart_quantity")), math.log1p(number("cart_total_quantity")),
            math.log1p(number("days_since_first_observed")), math.log1p(number("days_since_purchase")),
            float(context.get("days_since_purchase") is not None),
            math.log1p(number("purchase_count")), math.log1p(number("product_purchase_count")),
            category_share, math.log1p(number("session_event_count")),
            math.log1p(number("prior_gap_seconds")), float(unit > 0)]


def choice_probabilities(model: dict, context: dict) -> dict[str, float]:
    """Evaluate JSON coefficients without pickle; return probabilities in ACTIONS order."""
    probabilities = choice_probabilities_batch(model, np.asarray([context_features(context)]))[0]
    return dict(zip(ACTIONS, probabilities.tolist()))


def choice_probabilities_batch(model: dict, features: np.ndarray) -> np.ndarray:
    """Use the same fitted policy for source likelihood checks and simulator choices."""
    block = model.get("choice_model", model)
    if block["feature_names"] != FEATURE_NAMES or block["actions"] != list(ACTIONS):
        raise ValueError("incompatible shopping choice model")
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("shopping features must be a finite record-by-feature matrix")
    scale = np.asarray(block["scaler_scale"], dtype=float)
    values = (values - np.asarray(block["scaler_mean"], dtype=float)) / scale
    logits = values @ np.asarray(block["coef"], dtype=float).T + np.asarray(block["intercept"], dtype=float)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def load_source(folder: Path, sample_modulus: int = 100) -> tuple[pd.DataFrame, dict]:
    """Stream all months and retain complete histories for the same deterministic ID sample."""
    if sample_modulus < 20:
        raise ValueError("source sampling must retain at most five percent of numeric IDs")
    pieces, sources = [], []
    for path in sorted(folder.glob("20??-*.csv")):
        before = path.stat()
        for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, chunksize=250_000):
            if not set(SOURCE_COLUMNS) <= set(chunk):
                raise ValueError("unexpected Cosmetics source schema")
            ids = pd.to_numeric(chunk.user_id, errors="raise")
            selected = chunk.loc[ids.mod(sample_modulus).eq(0), list(SOURCE_COLUMNS)].copy()
            selected["source_event_id"] = path.name + ":" + selected.index.astype(str).str.zfill(9)
            pieces.append(selected)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed during profiling")
        sources.append({"path": str(path.resolve()), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns})
    if not pieces:
        raise ValueError("no Cosmetics source CSVs found")
    return pd.concat(pieces, ignore_index=True), {
        "dataset": "Cosmetics Shop events", "files": sources,
        "sample_rule": f"numeric user_id % {sample_modulus} == 0", "sample_modulus": sample_modulus,
        "source_hashes": "not computed; source path, size and mtime_ns recorded",
    }


def _bounded(values, limit: int = 32) -> list:
    values = sorted(values)
    if len(values) > limit:
        values = [values[int(index)] for index in np.linspace(0, len(values) - 1, limit)]
    return values


def _preferences(values: Counter, limit: int = 16) -> dict[str, int]:
    return dict(sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _gaps(values) -> list[float]:
    return [(later - earlier).total_seconds() for earlier, later in zip(values, values[1:])]


def _quantiles(values) -> dict:
    if not len(values):
        return {"count": 0, "p10": None, "p50": None, "p90": None}
    return {"count": len(values), **dict(zip(("p10", "p50", "p90"),
                                           np.quantile(values, [0.1, 0.5, 0.9]).tolist()))}


def _template_rows(counts: Counter) -> list[dict]:
    return [{"items": [{"product_id": product, "quantity": quantity} for product, quantity in key],
             "observed_count": count} for key, count in sorted(counts.items())]


def _pooled_checkout_templates(profiles: list[dict]) -> list[dict]:
    counts = Counter()
    for profile in profiles:
        for template in profile["checkout_templates"]:
            key = tuple((item["product_id"], item["quantity"]) for item in template["items"])
            counts[key] += template["observed_count"]
    return _template_rows(counts)


def prepare_source(events: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    if not set(SOURCE_COLUMNS) <= set(events):
        raise ValueError("missing Cosmetics source columns")
    rows = events.copy()
    if "source_event_id" not in rows:
        rows["source_event_id"] = [f"fixture:{i:09d}" for i in range(len(rows))]
    for name in ("user_id", "product_id", "category_id", "user_session"):
        if rows[name].isna().any():
            rows[name] = rows[name].fillna("")
        rows[name] = rows[name].astype(str)
    if rows.user_id.eq("").any() or rows.product_id.eq("").any() or rows.category_id.eq("").any():
        raise ValueError("source user, product and category IDs cannot be missing")
    rows["at"] = pd.to_datetime(rows.event_time, utc=True, errors="raise")
    if rows["at"].isna().any() or not rows.event_type.isin(ACTION_MAP).all():
        raise ValueError("invalid source timestamp or event")
    rows["action"] = rows.event_type.map(ACTION_MAP)
    rows["price_minor"] = rows.price.map(price_minor)
    rows = rows.sort_values(["user_id", "at", "source_event_id"], kind="stable")
    valid = rows.loc[rows.price_minor.notna()].copy()
    catalog = []
    for product_id, product in valid.groupby("product_id", sort=True):
        counts = product.price_minor.astype("int64").value_counts()
        representative = int(counts.index[0])
        observed = product.loc[product.price_minor.eq(representative)].iloc[-1]
        brand = observed.brand if pd.notna(observed.brand) and str(observed.brand).strip() else None
        catalog.append({"product_id": str(product_id), "category_id": str(observed.category_id),
                        "category": str(observed.category_id), "brand": brand,
                        "unit_price_minor": representative, "observed_source_price": str(observed.price),
                        "observed_min_price_minor": int(product.price_minor.min()),
                        "observed_max_price_minor": int(product.price_minor.max()),
                        "observations": len(product), "currency": "SOURCE_UNSPECIFIED", "minor_unit": "0.01"})
    if not catalog:
        raise ValueError("no positive observed prices available for catalog")
    audit = {"source_rows": len(rows), "missing_session_rows": int(rows.user_session.eq("").sum()),
             "nonpositive_or_invalid_price_rows": int(rows.price_minor.isna().sum()),
             "catalog_products": len(catalog), "catalog_price_rule": "most frequent positive observed price; source text retained"}
    rows = rows.loc[rows.user_session.ne("")].copy()
    gap = rows.groupby("user_id", sort=False)["at"].diff()
    previous_session = rows.groupby("user_id", sort=False).user_session.shift()
    rows["session"] = (gap.isna() | gap.gt(pd.Timedelta(minutes=30)) |
                       rows.user_session.ne(previous_session)).cumsum().astype("int64")
    return rows, catalog, audit


def build_records(rows: pd.DataFrame, reference_price: int, observation_end: pd.Timestamp,
                  seed: int = 42, checkout_episodes: bool = False) -> tuple[np.ndarray, pd.DataFrame, list[dict], dict]:
    records, labels, profiles = [], [], []
    audit = Counter()
    catalog_ids = set(rows.loc[rows.price_minor.notna(), "product_id"])
    for uid, group in rows.groupby("user_id", sort=True):
        cart, purchased_products, purchased_categories = Counter(), Counter(), Counter()
        product_counts, category_counts, action_counts, first_action_counts = Counter(), Counter(), Counter(), Counter()
        decision_counts, checkout_items, checkout_templates = Counter(), Counter(), Counter()
        session_choice_lengths, decision_gaps = [], []
        prior_decision_at = None
        session_starts, session_ends, session_lengths, event_gaps, session_hours = [], [], [], [], []
        purchase_sessions, purchase_prices, viewed_prices = {}, [], []
        first = group["at"].iloc[0]
        prior_at = prior_session = last_purchase = offer_product = offer_price = None
        reference_sum = reference_count = purchase_count = in_session = 0
        customer_records = list(group.itertuples(index=False))
        train_user = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
        for index, event in enumerate(customer_records):
            new_session = event.session != prior_session
            if new_session:
                first_action_counts[event.action] += 1
                session_starts.append(event.at)
                session_ends.append(event.at)
                session_lengths.append(0)
                session_choice_lengths.append(0)
                session_hours.append(event.at.hour)
                in_session = 0
                offer_product = offer_price = None
                prior_decision_at = None
            session_lengths[-1] += 1
            session_ends[-1] = event.at
            in_session += 1
            prior_gap = (event.at - prior_at).total_seconds() if prior_at is not None and not new_session else 0.0
            if prior_at is not None and not new_session:
                event_gaps.append(prior_gap)
            action_counts[event.action] += 1
            product_counts[event.product_id] += 1
            category_counts[event.category_id] += 1
            positive_price = None if pd.isna(event.price_minor) else int(event.price_minor)
            if event.action == "view":
                offer_product, offer_price = event.product_id, positive_price
                if positive_price is not None:
                    viewed_prices.append(positive_price)
            elif event.action == "cart":
                cart[event.product_id] += 1
            elif event.action == "remove":
                if cart[event.product_id]:
                    cart[event.product_id] -= 1
                else:
                    audit["remove_without_known_cart_unit"] += 1
            elif event.action == "purchase":
                purchase_count += 1
                purchased_products[event.product_id] += 1
                purchased_categories[event.category_id] += 1
                purchase_sessions[event.session] = session_starts[-1]
                last_purchase = event.at
                if positive_price is not None:
                    purchase_prices.append(positive_price)
                if cart[event.product_id]:
                    cart[event.product_id] -= 1
                else:
                    audit["purchase_without_known_cart_unit"] += 1
            following = customer_records[index + 1] if index + 1 < len(customer_records) else None
            same_session = following is not None and following.session == event.session
            within_checkout = (checkout_episodes and event.action == "purchase" and same_session
                               and following.action == "purchase" and following.at == event.at)
            if checkout_episodes and event.action == "purchase":
                checkout_items[event.product_id] += 1
                audit["checkout_unit_rows"] += 1
                if not within_checkout:
                    # Aggregate units commute; keep the final source item last for the next choice context.
                    key = tuple(sorted((pid, quantity) for pid, quantity in checkout_items.items()
                                       if pid != event.product_id)) + ((event.product_id, checkout_items[event.product_id]),)
                    checkout_templates[key] += 1
                    checkout_items.clear()
                    audit["checkout_episodes"] += 1
            if within_checkout:
                audit["skipped_within_checkout_choice_records"] += 1
            else:
                if prior_decision_at is not None:
                    decision_gaps.append((event.at - prior_decision_at).total_seconds())
                prior_decision_at = event.at
                decision_counts[event.action] += 1
                session_choice_lengths[-1] += 1
                context = {
                    "previous_action": event.action, "unit_price_minor": offer_price,
                    "reference_price_minor": reference_sum / reference_count if reference_count else reference_price,
                    "cart_quantity": cart[offer_product] if offer_product is not None else 0,
                    "cart_total_quantity": sum(cart.values()),
                    "days_since_first_observed": (event.at - first).total_seconds() / 86400,
                    "days_since_purchase": (event.at - last_purchase).total_seconds() / 86400 if last_purchase is not None else None,
                    "purchase_count": purchase_count,
                    "product_purchase_count": purchased_products[offer_product] if offer_product is not None else 0,
                    "category_purchase_share": purchased_categories[event.category_id] / purchase_count if purchase_count else 0,
                    "session_event_count": in_session, "prior_gap_seconds": prior_gap,
                }
                records.append(context_features(context))
                next_action = following.action if same_session else "end"
                if next_action != "end":
                    target_at = following.at
                else:
                    target_at = event.at + pd.Timedelta(minutes=30)
                    if following is not None:
                        target_at = min(target_at, following.at)
                labels.append({"user_id": uid, "at": event.at, "target_at": target_at,
                               "is_censored": target_at > observation_end,
                               "action": ACTIONS.index(next_action), "train_user": train_user})
            if positive_price is not None:
                reference_sum += positive_price
                reference_count += 1
            prior_at, prior_session = event.at, event.session
        session_gaps = _gaps(session_starts)
        idle_gaps = [(later - earlier).total_seconds() for earlier, later in zip(session_ends, session_starts[1:])]
        purchase_gaps = _gaps(list(purchase_sessions.values()))
        eligible = returned = 0
        for index, at in enumerate(session_starts):
            if at + pd.Timedelta(days=30) <= observation_end:
                eligible += 1
                returned += index + 1 < len(session_starts) and session_starts[index + 1] <= at + pd.Timedelta(days=30)
        exposure_days = max(1.0, (observation_end - first).total_seconds() / 86400)
        active_days = group["at"].dt.floor("D").nunique()
        if len(purchase_sessions) >= 2:
            home_group = "repeat_buyer"
        elif purchase_sessions:
            home_group = "observed_buyer"
        elif action_counts["remove"]:
            home_group = "cart_remover_without_purchase"
        elif action_counts["cart"]:
            home_group = "cart_without_purchase"
        else:
            home_group = "browse_only"
        preferred_prices = purchase_prices or viewed_prices
        ordered_prices = sorted(preferred_prices)
        complete_windows = int(exposure_days // 30)
        window_counts = Counter(int((at - first).total_seconds() // (30 * 86400)) for at in session_starts
                                if int((at - first).total_seconds() // (30 * 86400)) < complete_windows)
        active_windows = len(window_counts)
        profiles.append({
            "profile_id": hashlib.sha256(f"cosmetics:{uid}".encode()).hexdigest()[:16],
            "home_group": home_group, "behavior_group": home_group,
            "source_event_count": len(group), "observed_sessions": len(session_starts),
            "observed_purchase_sessions": len(purchase_sessions), "action_counts": dict(action_counts),
            "session_first_action_counts": dict(first_action_counts),
            "product_weights": _preferences(Counter({pid: count for pid, count in product_counts.items() if pid in catalog_ids})),
            "category_weights": _preferences(category_counts),
            "purchased_product_weights": _preferences(purchased_products),
            "purchased_category_weights": _preferences(purchased_categories),
            "preferred_price_minor": ordered_prices[(len(ordered_prices) - 1) // 2] if ordered_prices else reference_price,
            "session_gap_seconds": _bounded(session_gaps), "idle_gap_seconds": _bounded(idle_gaps),
            "event_gap_seconds": _bounded(event_gaps), "purchase_gap_seconds": _bounded(purchase_gaps),
            "session_lengths": _bounded(session_lengths), "session_start_hours": _bounded(session_hours),
            "session_counts_active_30d": _bounded(window_counts.values()),
            "mean_sessions_per_active_month": sum(window_counts.values()) / active_windows if active_windows else None,
            "daily_visit_probability": min(1.0, active_days / exposure_days),
            "visit_rate_per_30d": 30 * len(session_starts) / exposure_days,
            "eligible_30d_windows": complete_windows, "active_30d_windows": active_windows,
            "eligible_return_sessions_30d": eligible, "returned_within_30d": int(returned),
            "entry_duration_days": session_gaps[0] / 86400 if session_gaps else None,
            "first_observed_at": first.isoformat(), "last_observed_at": group["at"].iloc[-1].isoformat(),
            "right_censored_days": max(0.0, (observation_end - group["at"].iloc[-1]).total_seconds() / 86400),
            "initial_cart_unknown": True,
        })
        if checkout_episodes:
            profiles[-1].update(decision_action_counts=dict(decision_counts),
                                session_choice_lengths=_bounded(session_choice_lengths),
                                choice_gap_seconds=_bounded(decision_gaps),
                                checkout_templates=_template_rows(checkout_templates))
    return np.asarray(records, dtype=np.float64), pd.DataFrame(labels), profiles, dict(audit)


def _fit_choice(features: np.ndarray, labels: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> tuple[dict, dict]:
    before = labels.target_at.lt(cutoff).to_numpy() & ~labels.is_censored.to_numpy()
    after = labels["at"].ge(cutoff).to_numpy() & ~labels.is_censored.to_numpy()
    train = labels.train_user.to_numpy() & before
    if train.sum() < 10 or labels.loc[train, "action"].nunique() < 2:
        raise ValueError("shopping choice fitting requires at least ten records and two train actions")
    scaler = StandardScaler().fit(features[train])
    transformed = scaler.transform(features)
    per_user = labels.loc[train, "user_id"].value_counts()
    weights = labels.loc[train, "user_id"].map(1 / per_user).to_numpy()
    weights *= len(weights) / weights.sum()
    estimator = LogisticRegression(max_iter=250, C=1.0, solver="lbfgs", random_state=seed, tol=1e-5)
    estimator.fit(transformed[train], labels.loc[train, "action"], sample_weight=weights)
    coefficients = np.zeros((len(ACTIONS), features.shape[1]))
    intercepts = np.full(len(ACTIONS), -30.0)
    if len(estimator.classes_) == 2:
        intercepts[estimator.classes_[0]] = 0
        coefficients[estimator.classes_[1]] = estimator.coef_[0]
        intercepts[estimator.classes_[1]] = estimator.intercept_[0]
    else:
        coefficients[estimator.classes_] = estimator.coef_
        intercepts[estimator.classes_] = estimator.intercept_
    logits = transformed @ coefficients.T + intercepts
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    balanced_frequency = np.bincount(labels.loc[train, "action"], weights=weights, minlength=len(ACTIONS)) + 0.5
    balanced_frequency /= balanced_frequency.sum()
    frequency = np.bincount(labels.loc[train, "action"], minlength=len(ACTIONS)) + 0.5
    frequency /= frequency.sum()
    y = labels.action.to_numpy()
    evaluation = {"cutoff_at_utc": cutoff.isoformat(),
                  "split": "train users with target available strictly before cutoff; scaler train-only; crossing/censored targets excluded",
                  "training_users": int(labels.loc[train, "user_id"].nunique()), "training_records": int(train.sum()),
                  "frequency_baseline_probabilities": dict(zip(ACTIONS, frequency.tolist())),
                  "customer_balanced_frequency_baseline_probabilities": dict(zip(ACTIONS, balanced_frequency.tolist())),
                  "censored_terminal_records": int(labels.is_censored.sum()),
                  "crossing_cutoff_records": int((labels["at"].lt(cutoff) & labels.target_at.ge(cutoff)).sum()),
                  "splits": {}}
    masks = {"train": train,
             "user_holdout": ~labels.train_user.to_numpy() & before,
             "time_holdout": labels.train_user.to_numpy() & after,
             "user_and_time_holdout": ~labels.train_user.to_numpy() & after}
    for name, mask in masks.items():
        if not mask.any():
            evaluation["splits"][name] = {"records": 0, "nll": None, "frequency_baseline_nll": None}
            continue
        losses = -np.log(probabilities[mask, y[mask]].clip(1e-12))
        null_losses = -np.log(frequency[y[mask]].clip(1e-12))
        counts = labels.loc[mask, "user_id"].value_counts()
        user_weights = labels.loc[mask, "user_id"].map(1 / counts).to_numpy()
        evaluation["splits"][name] = {
            "users": int(len(counts)), "records": int(mask.sum()), "nll": float(losses.mean()),
            "frequency_baseline_nll": float(null_losses.mean()),
            "customer_balanced_nll": float(np.average(losses, weights=user_weights)),
            "customer_balanced_frequency_baseline_nll": float(np.average(-np.log(balanced_frequency[y[mask]].clip(1e-12)), weights=user_weights)),
            "observed_action_share": dict(zip(ACTIONS, (np.bincount(y[mask], minlength=len(ACTIONS)) / mask.sum()).tolist())),
            "predicted_action_share": dict(zip(ACTIONS, probabilities[mask].mean(axis=0).tolist())),
        }
    return {"type": "contextual_one_step_maxent_multinomial_logit", "actions": list(ACTIONS),
            "context_names": CONTEXT_NAMES, "feature_names": FEATURE_NAMES,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": coefficients.tolist(), "intercept": intercepts.tolist(),
            "n_iter": estimator.n_iter_.tolist(), "max_iter": estimator.max_iter,
            "regularization_C": estimator.C, "price_effect": "conditional association using last observed view as offer proxy"}, evaluation


def fit_shopping(events: pd.DataFrame, provenance: dict | None = None, seed: int = 42,
                 checkout_episodes: bool = False) -> dict:
    rows, catalog, audit = prepare_source(events)
    if rows.empty:
        raise ValueError("no source sessions available")
    observation_start = rows["at"].min().floor("D")
    observation_end = rows["at"].max().floor("D") + pd.Timedelta(days=1)
    span = observation_end - observation_start
    cutoff = observation_end - min(pd.Timedelta(days=29), span / 4)
    train_users = rows.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
    training_prices = rows.loc[train_users & rows["at"].lt(cutoff) & rows.price_minor.notna(), "price_minor"]
    if training_prices.empty:
        raise ValueError("no positive train prices available")
    reference_price = int(training_prices.median())
    features, labels, profiles, cart_audit = build_records(rows, reference_price, observation_end, seed, checkout_episodes)
    model, validation = _fit_choice(features, labels, cutoff, seed)
    price_valid = rows.action.eq("purchase") & rows.price_minor.notna()
    purchase_sessions = (rows.assign(has_purchase=rows.action.eq("purchase"))
                         .groupby(["user_id", "session"], sort=False)
                         .agg(at=("at", "min"), has_purchase=("has_purchase", "max"))
                         .reset_index())
    purchase_sessions = purchase_sessions.loc[purchase_sessions.has_purchase].sort_values(["user_id", "at"])
    purchase_sessions["gap_seconds"] = purchase_sessions.groupby("user_id")["at"].diff().dt.total_seconds()
    gap_train_users = purchase_sessions.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
    validation["source_reference_distributions"] = {
        "meaning": "held-out source references for generator checks, not a generated-price forecast",
        "train_purchase_price_minor": _quantiles(rows.loc[price_valid & train_users & rows["at"].lt(cutoff), "price_minor"].astype(float)),
        "user_and_time_holdout_purchase_price_minor": _quantiles(rows.loc[price_valid & ~train_users & rows["at"].ge(cutoff), "price_minor"].astype(float)),
        "train_purchase_session_gap_seconds": _quantiles(purchase_sessions.loc[gap_train_users & purchase_sessions["at"].lt(cutoff), "gap_seconds"].dropna()),
        "user_and_time_holdout_purchase_session_gap_seconds": _quantiles(purchase_sessions.loc[~gap_train_users & purchase_sessions["at"].ge(cutoff), "gap_seconds"].dropna()),
    }
    grouped = defaultdict(list)
    for profile in profiles:
        grouped[profile["home_group"]].append(profile)
    groups = {}
    for name, donors in sorted(grouped.items()):
        eligible_returns = sum(p["eligible_return_sessions_30d"] for p in donors)
        returns = sum(p["returned_within_30d"] for p in donors)
        windows = sum(p["eligible_30d_windows"] for p in donors)
        active = sum(p["active_30d_windows"] for p in donors)
        prior_return = returns / eligible_returns if eligible_returns else 0.0
        prior_visit = active / windows if windows else 0.0
        for profile in donors:
            profile["return_probability"] = (profile["returned_within_30d"] + prior_return) / (profile["eligible_return_sessions_30d"] + 1)
            profile["visit_probability_30d"] = (profile["active_30d_windows"] + prior_visit) / (profile["eligible_30d_windows"] + 1)
        groups[name] = {"profiles": len(donors), "share": len(donors) / len(profiles),
                        "return_probability": prior_return, "visit_probability_30d": prior_visit,
                        **{key: _bounded([value for p in donors for value in p[key]], 4096)
                           for key in ("session_gap_seconds", "idle_gap_seconds", "event_gap_seconds",
                                       "purchase_gap_seconds", "session_lengths", "session_start_hours",
                                       "session_counts_active_30d")}}
        if checkout_episodes:
            groups[name].update(checkout_templates=_pooled_checkout_templates(donors),
                                **{key: _bounded([v for p in donors for v in p[key]], 4096)
                                   for key in ("session_choice_lengths", "choice_gap_seconds")})
    purchase_prices = rows.loc[rows.action.eq("purchase") & rows.price_minor.notna(), "price_minor"].astype(float).to_numpy()
    source_counts = rows.action.value_counts().to_dict()
    artifact = {
        "schema_version": 1, "model_type": "observed_shopping_profiles_and_contextual_maxent",
        "seed": seed, "currency": "SOURCE_UNSPECIFIED", "minor_unit": "0.01",
        "reference_price_minor": reference_price,
        "observation_start": observation_start.isoformat(), "observation_end": observation_end.isoformat(),
        "source": provenance or {"dataset": "explicit deterministic test fixture"},
        "source_summary": {**audit, **cart_audit, "sample_users": len(profiles),
                           "sessions": int(rows.session.nunique()), "event_counts": {name: int(source_counts.get(name, 0)) for name in ACTIONS[:-1]},
                           "session_first_action_counts": {name: sum(profile["session_first_action_counts"].get(name, 0) for profile in profiles)
                                                           for name in ACTIONS[:-1]},
                           "end_labels": "one inferred session end per observed/inactivity-split session"},
        "catalog": catalog, "choice_model": model, "profiles": profiles, "home_groups": groups,
        "distributions": {"purchase_price_minor": _quantiles(purchase_prices),
                          "purchase_gap_seconds": _quantiles(purchase_sessions.gap_seconds.dropna()),
                          **{key: _bounded([v for p in profiles for v in p[key]], 4096)
                             for key in ("session_gap_seconds", "idle_gap_seconds", "event_gap_seconds",
                                         "session_lengths", "session_start_hours", "session_counts_active_30d")}},
        "validation": validation,
        "lifecycle": {"states": ["entry", "active", "exit_proxy"],
                      "entry": "first observed session, not verified signup",
                      "generator_entry_rule": "first two generated sessions is a structural lifecycle rule, not an estimated signup transition",
                      "active": "donor's descriptive observed home_group; no fixed six-persona mixture",
                      "exit_proxy": "inactivity relative to observed gaps, not observed account closure or permanent churn",
                      "idle_gap_p95_seconds": float(np.quantile([v for p in profiles for v in p["idle_gap_seconds"]], 0.95))
                      if any(p["idle_gap_seconds"] for p in profiles) else None},
        "limitations": [
            "Cosmetics has no true order ID or quantity: one interaction contributes one unit only as a cart-replay proxy.",
            "Unknown cart inventory before observation and missing sessions prevent exact source cart reconstruction.",
            "Purchase cadence is between observed purchase sessions, not validated orders; simultaneous item events are not independent repeat orders.",
            "Contextual one-step MaxEnt imitation does not identify long-horizon rewards, causal price elasticity or marketing uplift.",
            "Last observed view is an offer proxy; unobserved choices, impressions and product availability are unknown.",
            "Sessions may begin with cart, removal or purchase; before an observed view the offered-product/price proxy is unknown, even if an item event has a recorded price.",
            "Donor home_groups and preference summaries describe full observed histories; they are not prospective persona predictions.",
            "Catalog uses modal positive source price; raw source price strings are retained and currency is unspecified.",
            "Negative/zero prices are excluded only from the positive catalog/reference; their interaction rows remain in behavioral histories.",
            "Within-session timing and source session IDs use the existing 30-minute inactivity split; session end is inferred.",
            "Per-profile timing arrays are at most 32 empirical quantiles; preference lists retain 16 most frequent observed IDs.",
            "Five observed months do not validate twelve-month history plus twelve-month future extrapolation.",
            "Thirty-day activity windows and any logit shift targeting a specified churn rate are generator scenario assumptions, not raw-source estimates.",
        ],
    }
    if checkout_episodes:
        artifact.update(model_type="observed_shopping_profiles_and_contextual_maxent_checkout_episode",
                        choice_unit="source_checkout_episode",
                        checkout_episode_definition="same customer/session, exactly equal timestamp, contiguous purchase rows; not a verified order ID",
                        context_units={"session_event_count": "raw item-interaction rows including all checkout units",
                                       "purchase_count": "raw purchase unit proxies, not checkout count",
                                       "prior_gap_seconds": "gap from prior raw item row; zero after multiunit same-time checkout",
                                       "session_choice_lengths": "nonpurchase interactions plus checkout episodes",
                                       "session_lengths": "raw item-interaction rows",
                                       "choice_gap_seconds": "between successive episode-level decisions in a session",
                                       "checkout_template_items": "one source row per unit proxy; final source product is the final template item"})
        artifact["source_summary"].update(decision_event_counts={action: sum(p["decision_action_counts"].get(action, 0) for p in profiles)
                                                               for action in ACTIONS[:-1]},
                                            decision_records=len(labels))
        artifact["distributions"].update(checkout_templates=_pooled_checkout_templates(profiles),
                                           **{key: _bounded([v for p in profiles for v in p[key]], 4096)
                                              for key in ("session_choice_lengths", "choice_gap_seconds")})
        artifact["limitations"].append("Simultaneous contiguous purchase rows define a checkout episode proxy; true order boundaries and quantities are unobserved. Unit source counts remain unchanged, and only internal purchase choice labels are removed.")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmetics", type=Path, default=Path("data/raw/kaggle/cosmetics"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkout-episodes", action="store_true")
    parser.add_argument("--item-choice", action="store_true")
    parser.add_argument("--sample-modulus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.item_choice and not args.checkout_episodes:
        parser.error("--item-choice requires --checkout-episodes")
    args.output = args.output or Path("data/models/shopping_profiles_checkout_items.json" if args.item_choice else
                                    "data/models/shopping_profiles_checkout.json" if args.checkout_episodes
                                    else "data/models/shopping_profiles.json")
    if args.checkout_episodes and args.output.exists():
        raise FileExistsError(f"checkout candidate must use a new artifact path: {args.output}")
    events, provenance = load_source(args.cosmetics, args.sample_modulus)
    artifact = fit_shopping(events, provenance, args.seed, args.checkout_episodes)
    artifact["source"]["fitter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.item_choice:
        from simulate.v4.simulator.item_choice import fit_item_choice
        rows, _, _ = prepare_source(events)
        kernel = fit_item_choice(rows, artifact)
        kernel["provenance"] = {"source_kind": "same in-memory source rows as the checkout fit; no extra coefficient fit",
                                "base_model_content_sha256": hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest(),
                                "fitter_code_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("simulator/item_choice.py").read_bytes()).hexdigest()}
        artifact["item_choice"] = kernel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "source_summary": artifact["source_summary"],
                      "validation": artifact["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
