"""Small source-sequence maximum causal entropy IRL, with an explicit observed MDP.

The finite abstraction uses source-training means of continuous past-context
features. Both learning and runtime use those same frozen means, never future
persona labels. This is within-session predictive IRL, not a causal price or
coupon effect estimator, and the logged transition model is an approximation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


ACTIONS = ("view", "cart", "remove", "purchase", "end")
PREVIOUS = ("start", "view", "cart", "remove", "purchase")
FEATURES = ("log_cart_total_quantity", "log_observed_offer_price", "has_observed_offer",
            "log_days_since_first_observed", "log_days_since_purchase", "has_purchase_history",
            "log_purchase_item_units", "log_session_decision_count", "log_prior_gap_seconds",
            "purchase_in_current_session", *[f"previous_{a}" for a in PREVIOUS])
CONTEXT_KEYS = ("previous_action", "cart_total_quantity", "unit_price_minor",
                "days_since_first_observed", "days_since_purchase", "purchase_count",
                "session_decision_count", "prior_gap_seconds", "purchase_in_current_session")
TYPE = "discounted_observed_mdp_maximum_causal_entropy_irl"
_CACHE = {}


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def context_vector(context):
    """Continuous, already-observed inputs. Progress counts decisions, not item units."""
    previous = context.get("previous_action", "start")
    if previous not in PREVIOUS:
        raise ValueError("unrecognized previous_action")

    def number(name):
        value = context.get(name)
        value = 0.0 if value is None else float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid nonnegative source context: {name}")
        return value

    progress = number("session_decision_count")
    if (previous == "start") != (progress == 0) or not progress.is_integer():
        raise ValueError("session_decision_count must match the observed start/decision context")
    price = number("unit_price_minor")
    prior = context.get("days_since_purchase") is not None
    same_session = context.get("purchase_in_current_session", False)
    if not isinstance(same_session, (bool, np.bool_)) or same_session and not prior:
        raise ValueError("purchase_in_current_session requires actual prior purchase history")
    return np.asarray([np.log1p(number("cart_total_quantity")), np.log1p(price), float(price > 0),
        np.log1p(number("days_since_first_observed")), np.log1p(number("days_since_purchase")),
        float(prior), np.log1p(number("purchase_count")), np.log1p(progress),
        np.log1p(number("prior_gap_seconds")), float(same_session),
        *[float(previous == a) for a in PREVIOUS]], dtype=float)


def build_demonstrations(rows, observation_end, seed=42):
    """Replay every source row; collapse only contiguous same-time purchase units.

    A record's context is available before its action. END is an inferred label,
    observed only after 30 minutes of follow-up. An impossible removal invalidates
    its entire session, instead of stitching the remaining actions together.
    """
    required = {"user_id", "product_id", "session", "action", "at", "source_event_id", "price_minor"}
    if not required <= set(rows):
        raise ValueError("prepared source rows are required")
    end = pd.Timestamp(observation_end)
    if end.tzinfo is None:
        raise ValueError("observation_end must be timezone aware")
    ordered = rows.sort_values(["user_id", "at", "source_event_id"], kind="stable")
    clock_sessions = ordered.groupby(["user_id", "at"]).session.nunique()
    bad_clocks = set(clock_sessions.loc[clock_sessions.gt(1)].index)
    vectors, records, session_rows = [], [], []
    audit = Counter(source_rows=len(ordered), source_users=ordered.user_id.nunique())
    for uid, user in ordered.groupby("user_id", sort=True):
        cart = Counter()
        first_at = user["at"].iloc[0]
        last_purchase = last_purchase_session = None
        purchased_units = 0
        train_user = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
        inner_user = hashlib.sha256(f"{seed}:inner:{uid}".encode()).digest()[0] < 205
        for sid, session in user.groupby("session", sort=False):
            source = list(session.itertuples(index=False))
            previous, decision_count, offer, prior_gap = "start", 0, None, 0.0
            at = source[0].at
            previous_decision_at = None
            start_index = len(records)
            invalid_product = invalid_total = False
            unmatched_positive = unmatched_one = 0
            ambiguous = any((uid, item.at) in bad_clocks for item in source)

            def current_context():
                return {"previous_action": previous, "cart_total_quantity": sum(cart.values()),
                    "unit_price_minor": offer, "days_since_first_observed": (at - first_at).total_seconds() / 86400,
                    "days_since_purchase": (at - last_purchase).total_seconds() / 86400 if last_purchase is not None else None,
                    "purchase_count": purchased_units, "session_decision_count": decision_count,
                    "prior_gap_seconds": prior_gap, "purchase_in_current_session": last_purchase_session == sid}

            index = 0
            while index < len(source):
                event = source[index]
                if event.action not in ACTIONS[:-1]:
                    raise ValueError("unknown source shopping action")
                stop = index + 1
                if event.action == "purchase":
                    while stop < len(source) and source[stop].action == "purchase" and source[stop].at == event.at:
                        stop += 1
                vectors.append(context_vector(current_context()))
                records.append({"user_id": str(uid), "session": int(sid), "at": at, "target_at": event.at,
                                "action": ACTIONS.index(event.action), "step": decision_count})
                for item in source[index:stop]:
                    if item.action == "cart":
                        cart[item.product_id] += 1
                    elif item.action == "remove":
                        known_total = sum(cart.values())
                        if not known_total:
                            invalid_total = True
                            audit["remove_without_any_known_cart_unit"] += 1
                        if not cart[item.product_id]:
                            invalid_product = True
                            audit["remove_without_matching_known_cart_unit"] += 1
                            if known_total > 0:
                                unmatched_positive += 1
                                unmatched_one += known_total == 1
                        else:
                            cart[item.product_id] -= 1
                    elif item.action == "purchase":
                        purchased_units += 1
                        last_purchase, last_purchase_session = item.at, sid
                        if cart[item.product_id]:
                            cart[item.product_id] -= 1
                        else:
                            audit["purchase_without_matching_known_cart_unit"] += 1
                    elif item.action == "view":
                        offer = None if pd.isna(item.price_minor) else int(item.price_minor)
                at = source[stop - 1].at
                prior_gap = 0.0 if previous_decision_at is None or stop - index > 1 else (at - previous_decision_at).total_seconds()
                previous_decision_at = at
                previous = event.action
                decision_count += 1
                audit["collapsed_purchase_unit_rows"] += stop - index - 1
                index = stop
            vectors.append(context_vector(current_context()))
            known_at = at + pd.Timedelta(minutes=30)
            records.append({"user_id": str(uid), "session": int(sid), "at": at, "target_at": known_at,
                            "action": 4, "step": decision_count})
            session_rows.append({"user_id": str(uid), "session": int(sid), "start_at": source[0].at,
                "last_at": at, "terminal_known_at": known_at, "record_start": start_index,
                "record_end": len(records), "decisions": decision_count, "train_user": train_user,
                "inner_user": inner_user, "invalid_matching_cart": invalid_product,
                "invalid_total_cart": invalid_total, "ambiguous_clock_sessions": ambiguous,
                "unmatched_remove_positive_known_total": unmatched_positive,
                "unmatched_remove_known_total_one": unmatched_one,
                "is_censored": known_at > end})
    info = pd.DataFrame(session_rows)
    audit.update({"sessions": len(info), "checkout_decisions": sum(info.decisions), "end_records": len(info),
        "sessions_with_unknown_matching_cart_removal": int(info.invalid_matching_cart.sum()),
        "sessions_with_empty_known_cart_removal": int(info.invalid_total_cart.sum()),
        "same_clock_multisession_clocks": len(bad_clocks),
        "same_clock_multisession_affected_sessions": int(info.ambiguous_clock_sessions.sum()),
        "unmatched_remove_positive_known_total": int(info.unmatched_remove_positive_known_total.sum()),
        "unmatched_remove_known_total_one": int(info.unmatched_remove_known_total_one.sum()),
        "administratively_censored_sessions": int(info.is_censored.sum())})
    return {"features": np.asarray(vectors), "records": pd.DataFrame(records), "sessions": info,
            "audit": dict(audit), "observation_end": end}


def _selected(data, session_mask, fit_target="customer-macro"):
    if fit_target not in {"customer-macro", "pooled-session"}:
        raise ValueError("fit target must be customer-macro or pooled-session")
    sessions = data["sessions"].loc[session_mask]
    if sessions.empty:
        raise ValueError("empty sequence panel")
    indices = np.concatenate([np.arange(r.record_start, r.record_end) for r in sessions.itertuples()])
    if not len(indices):
        raise ValueError("empty sequence panel")
    counts = sessions.user_id.value_counts()
    customer_weights = sessions.user_id.map(1 / counts).to_numpy() / len(counts)
    session_weights = np.full(len(sessions), 1 / len(sessions)) if fit_target == "pooled-session" else customer_weights
    weights = np.concatenate([np.full(r.record_end - r.record_start, w)
                              for r, w in zip(sessions.itertuples(), session_weights)])
    encoder_weights = np.concatenate([np.full(r.record_end - r.record_start, w)
                                      for r, w in zip(sessions.itertuples(), customer_weights)])
    starts = np.zeros(len(indices), dtype=bool)
    starts[np.cumsum([0, *[r.record_end - r.record_start for r in sessions.itertuples()]])[:-1]] = True
    return {"features": data["features"][indices], "records": data["records"].iloc[indices].reset_index(drop=True),
            "weights": weights, "encoder_weights": encoder_weights, "starts": starts,
            "sessions": len(sessions), "users": len(counts), "fit_target": fit_target}


def _keys(features, thresholds):
    x = np.asarray(features)
    previous = x[:, 10:].argmax(axis=1)
    progress = np.where(x[:, 7] == 0, 0, 1 + (x[:, 7] > thresholds["progress_log_median"]).astype(int))
    price = np.where(x[:, 2] == 0, 0, 1 + (x[:, 1] > thresholds["offer_log_median"]).astype(int))
    history = np.where(x[:, 5] == 0, 0, np.where(x[:, 9] > 0, 1,
        2 + (x[:, 4] > thresholds["prior_purchase_log_age_median"]).astype(int)))
    return np.column_stack([previous, x[:, 0] > 0, progress, price, history]).astype(int)


def fit_encoder(panel):
    x = panel["features"]
    def median(values):
        return float(np.median(values)) if len(values) else 0.0
    thresholds = {"progress_log_median": median(x[x[:, 7] > 0, 7]),
        "offer_log_median": median(x[x[:, 2] > 0, 1]),
        "prior_purchase_log_age_median": median(x[(x[:, 5] > 0) & (x[:, 9] == 0), 4])}
    keys = _keys(x, thresholds)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    # Keep the representation fixed when comparing occupancy/selection objectives.
    w = panel.get("encoder_weights", panel["weights"])
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.0)
    means = np.asarray([np.average(x[inverse == i], axis=0, weights=w[inverse == i]) for i in range(len(unique))])
    basis = np.column_stack([np.ones(len(unique)), (means - mean) / scale])
    return {"schema_version": 1, "feature_names": list(FEATURES), "context_keys": list(CONTEXT_KEYS),
        "state_keys": unique.tolist(), "thresholds": thresholds, "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(), "continuous_context_means": means.tolist(),
        "reward_basis": basis.tolist(), "maximum_combinatorial_states": 360,
        "meaning": "train-only context bins with frozen continuous source-training mean reward features; within-bin variation is abstracted, not a continuous-state MDP"}


def encode_features(encoder, features):
    x = np.asarray(features, dtype=float)
    states = np.asarray(encoder["state_keys"], dtype=int)
    mapping = {tuple(key): i for i, key in enumerate(states)}
    keys = _keys(x, encoder["thresholds"])
    means = np.asarray(encoder["continuous_context_means"])
    scale = np.asarray(encoder["feature_scale"])
    encoded, fallback = np.empty(len(x), dtype=int), np.zeros(len(x), dtype=bool)
    for i, key in enumerate(keys):
        found = mapping.get(tuple(key))
        if found is None:
            fallback[i] = True
            eligible = np.flatnonzero((states[:, 0] == key[0]) & (states[:, 1] == key[1]) & ((states[:, 2] == 0) == (key[2] == 0)))
            if not len(eligible):
                eligible = np.flatnonzero((states[:, 1] == key[1]) & ((states[:, 2] == 0) == (key[2] == 0)))
            if not len(eligible):
                raise ValueError("no source state supports this cart/start context")
            found = int(eligible[np.argmin(np.square((means[eligible] - x[i]) / scale).sum(axis=1))])
        encoded[i] = found
    return encoded, fallback


def _mask(encoder):
    keys = np.asarray(encoder["state_keys"])
    result = np.ones((len(keys), len(ACTIONS)), dtype=bool)
    result[keys[:, 1] == 0, 2] = False
    result[keys[:, 2] == 0, 4] = False
    return result


def reward_basis(encoder, kind="continuous"):
    if kind == "continuous":
        return np.asarray(encoder["reward_basis"], dtype=float), ["intercept", *FEATURES]
    if kind == "tabular":
        count = len(encoder["state_keys"])
        return np.eye(count), [f"source_state_{i:03d}" for i in range(count)]
    raise ValueError("reward basis must be continuous or tabular")


def estimate_dynamics(encoder, panel, gamma):
    state, fallback = encode_features(encoder, panel["features"])
    if fallback.any():
        raise ValueError("training states cannot require an encoder fallback")
    action = panel["records"].action.to_numpy()
    steps = panel["records"].step.to_numpy()
    w = panel["weights"] * gamma ** steps
    n = len(encoder["state_keys"])
    counts = np.zeros((n, len(ACTIONS), n))
    for i in np.flatnonzero(action != 4):
        if i + 1 >= len(state) or panel["starts"][i + 1]:
            raise ValueError("nonterminal action must have its actual next source state")
        counts[state[i], action[i], state[i + 1]] += w[i]
    transition = np.zeros_like(counts)
    keys = np.asarray(encoder["state_keys"])
    mask = _mask(encoder)
    backoffs = Counter()
    for s in range(n):
        for a in range(4):
            if not mask[s, a]:
                continue
            candidates = [("state_action", counts[s, a]),
                ("cart_previous_action", counts[(keys[:, 0] == keys[s, 0]) & (keys[:, 1] == keys[s, 1]), a].sum(axis=0)),
                ("cart_action", counts[keys[:, 1] == keys[s, 1], a].sum(axis=0)),
                ("pooled_action", counts[:, a].sum(axis=0))]
            for scope, values in candidates:
                if values.sum() > 0:
                    transition[s, a] = values / values.sum()
                    backoffs[scope] += 1
                    break
            else:
                raise ValueError(f"no observed dynamics for supported action {ACTIONS[a]}")
    d0 = np.bincount(state[panel["starts"]], weights=panel["weights"][panel["starts"]], minlength=n)
    occupancy = np.zeros((n, 5))
    np.add.at(occupancy, (state, action), w)
    if not mask[state, action].all():
        raise ValueError("demonstration contains an impossible cart/start action")
    return transition, mask, d0, occupancy, dict(backoffs)


def soft_value_iteration(reward, transition, mask, gamma, tolerance=1e-10, max_iterations=5000):
    """E[V(next)] soft Bellman recursion; END leads to implicit zero-value absorption."""
    if not 0 <= gamma < 1 or tolerance <= 0:
        raise ValueError("discount must be in [0,1) and tolerance positive")
    reward = np.asarray(reward, dtype=float)
    transition, mask = np.asarray(transition), np.asarray(mask, dtype=bool)
    if reward.shape != mask.shape or transition.shape != (*reward.shape, len(reward)) or not mask.any(axis=1).all():
        raise ValueError("incompatible finite MDP arrays")
    value = np.zeros(len(reward))
    for iteration in range(max_iterations):
        q = reward + gamma * np.einsum("sak,k->sa", transition, value, optimize=True)
        q = np.where(mask, q, -np.inf)
        updated = logsumexp(q, axis=1)
        if np.max(np.abs(updated - value)) <= tolerance:
            value = updated
            break
        value = updated
    else:
        raise ValueError("soft value iteration did not converge")
    q = np.where(mask, reward + gamma * np.einsum("sak,k->sa", transition, value, optimize=True), -np.inf)
    probability = np.exp(q - logsumexp(q, axis=1, keepdims=True))
    residual = float(np.max(np.abs(logsumexp(q, axis=1) - value)))
    return value, probability, {"value_iterations": iteration + 1, "bellman_residual": residual, "tolerance": tolerance}


def discounted_occupancy(transition, probability, initial, gamma):
    p = np.einsum("sa,sak->sk", probability, transition)
    visits = np.linalg.solve(np.eye(len(p)) - gamma * p.T, initial)
    if np.min(visits) < -1e-9 or not np.isfinite(visits).all():
        raise ValueError("invalid discounted state occupancy")
    return np.maximum(visits, 0)[:, None] * probability


def dual_objective(weights, basis, transition, mask, initial, expert, gamma, regularization):
    """g(theta)=E_initial[V]-theta.mu_E+lambda/2||theta||², with exact occupancy gradient."""
    theta = np.asarray(weights).reshape(4, basis.shape[1])
    reward = np.column_stack([basis @ theta.T, np.zeros(len(basis))])
    value, probability, diagnostics = soft_value_iteration(reward, transition, mask, gamma)
    visits = discounted_occupancy(transition, probability, initial, gamma)
    empirical = expert[:, :4].T @ basis
    expected = visits[:, :4].T @ basis
    loss = initial @ value - np.sum(theta * empirical) + regularization * np.sum(theta ** 2) / 2
    gradient = expected - empirical + regularization * theta
    return float(loss), gradient.ravel(), (reward, value, probability, visits, diagnostics)


def fit_mce(encoder, panel, gamma, regularization, basis_kind="continuous"):
    transition, mask, initial, expert, backoffs = estimate_dynamics(encoder, panel, gamma)
    basis, names = reward_basis(encoder, basis_kind)
    def objective(theta):
        loss, gradient, _ = dual_objective(theta, basis, transition, mask, initial, expert, gamma, regularization)
        return loss, gradient
    optimum = minimize(objective, np.zeros(4 * basis.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 800 if basis_kind == "tabular" else 200, "gtol": 1e-7, "ftol": 1e-11})
    loss, gradient, (reward, value, probability, visits, bellman) = dual_objective(
        optimum.x, basis, transition, mask, initial, expert, gamma, regularization)
    return {"schema_version": 1, "type": TYPE, "actions": list(ACTIONS), "temperature": 1.0,
        "fit_target": panel.get("fit_target", "customer-macro"),
        "discount": float(gamma), "regularization": float(regularization), "encoder": encoder,
        "reward_basis_type": basis_kind, "reward_feature_names": names,
        "reward_weights": optimum.x.reshape(4, basis.shape[1]).tolist(), "end_reward": 0.0,
        "transition": transition.tolist(), "action_mask": mask.tolist(), "value": value.tolist(),
        "policy": probability.tolist(), "terminal": "implicit absorbing state: zero reward/value/entropy after END",
        "fit": {"objective": "initial.V - theta.mu_expert + L2/2; gradient=mu_policy-mu_expert+L2*theta",
            "optimizer": "L-BFGS-B", "converged": bool(optimum.success), "optimizer_message": str(optimum.message),
            "optimizer_iterations": int(optimum.nit), "objective_value": loss, "gradient_max_abs": float(np.abs(gradient).max()),
            "training_sessions": panel["sessions"], "training_users": panel["users"], "training_records": len(panel["records"]),
            "state_count": len(basis), "reward_parameter_count": len(optimum.x), "dynamics_backoff_counts": backoffs,
            "transition_estimator": ("discounted session-uniform transition counts" if panel.get("fit_target") == "pooled-session" else "discounted user-balanced session transition counts") + "; observed MLE then source cart/previous/action pools, with no invented pseudocounts",
            "occupancy_denominator": "one unit of initial mass per session, normalized over sessions" if panel.get("fit_target") == "pooled-session" else "equal customer mass, divided equally over each customer's sessions",
            "empirical_discounted_feature_counts": (expert[:, :4].T @ basis).tolist(),
            "expected_discounted_feature_counts": (visits[:, :4].T @ basis).tolist(), **bellman}}


def fit_myopic(encoder, panel, regularization, basis_kind="continuous"):
    """Same encoder/reward basis, all observed contexts, gamma=0 behavior-cloning ablation."""
    state, _ = encode_features(encoder, panel["features"])
    basis, names = reward_basis(encoder, basis_kind)
    mask = _mask(encoder)
    counts = np.zeros_like(mask, dtype=float)
    np.add.at(counts, (state, panel["records"].action.to_numpy()), panel["weights"])
    counts /= counts.sum()
    def objective(theta):
        theta = theta.reshape(4, basis.shape[1])
        logits = np.where(mask, np.column_stack([basis @ theta.T, np.zeros(len(basis))]), -np.inf)
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        probability = np.exp(logp)
        loss = -np.sum(counts[mask] * logp[mask]) + regularization * np.sum(theta ** 2) / 2
        gradient = (probability * counts.sum(axis=1, keepdims=True) - counts)[:, :4].T @ basis + regularization * theta
        return loss, gradient.ravel()
    optimum = minimize(objective, np.zeros(4 * basis.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 800 if basis_kind == "tabular" else 200, "gtol": 1e-7})
    logits = np.where(mask, np.column_stack([basis @ optimum.x.reshape(4, basis.shape[1]).T, np.zeros(len(basis))]), -np.inf)
    return {"encoder": encoder, "policy": np.exp(logits - logsumexp(logits, axis=1, keepdims=True)).tolist(),
        "discount": 0.0, "type": "myopic_same_basis_behavior_cloning_ablation",
        "reward_basis_type": basis_kind, "reward_feature_names": names,
        "reward_weights": optimum.x.reshape(4, basis.shape[1]).tolist(),
        "regularization": regularization, "converged": bool(optimum.success)}


def evaluate(model, panel):
    state, fallback = encode_features(model["encoder"], panel["features"])
    probabilities = np.asarray(model["policy"])[state]
    y = panel["records"].action.to_numpy()
    observed = probabilities[np.arange(len(y)), y]
    losses = -np.log(np.maximum(observed, 1e-300))
    user = panel["records"].user_id
    macro = float(pd.Series(losses).groupby(user).mean().mean())
    result = {"users": panel["users"], "sessions": panel["sessions"], "records": len(y),
        "action_counts": dict(zip(ACTIONS, np.bincount(y, minlength=5).tolist())),
        "action_nll": float(losses.mean()), "customer_macro_action_nll": macro,
        "mean_session_log_likelihood": float((-pd.Series(losses)).groupby(np.cumsum(panel["starts"]) - 1).sum().mean()),
        "zero_policy_support_targets": int((observed == 0).sum()), "encoder_fallback_records": int(fallback.sum()),
        "observed_action_share": dict(zip(ACTIONS, (np.bincount(y, minlength=5) / len(y)).tolist())),
        "teacher_context_predicted_action_share": dict(zip(ACTIONS, probabilities.mean(axis=0).tolist()))}
    if "transition" in model:
        gamma = model["discount"]
        w = panel["weights"] * gamma ** panel["records"].step.to_numpy()
        actual = np.zeros((len(model["encoder"]["state_keys"]), 5))
        np.add.at(actual, (state, y), w)
        initial = np.bincount(state[panel["starts"]], weights=panel["weights"][panel["starts"]], minlength=len(actual))
        visits = discounted_occupancy(np.asarray(model["transition"]), np.asarray(model["policy"]), initial, gamma)
        basis, _ = reward_basis(model["encoder"], model.get("reward_basis_type", "continuous"))
        result["discounted_state_action_occupancy_normalized_l1"] = float(np.abs(actual / actual.sum() - visits / visits.sum()).sum())
        result["discounted_feature_matching_l2"] = float(np.linalg.norm((actual[:, :4] - visits[:, :4]).T @ basis))
        targets = np.flatnonzero(y != 4)
        actual_transitions = np.asarray(model["transition"])[state[targets], y[targets], state[targets + 1]]
        result["transition_zero_support_targets"] = int((actual_transitions == 0).sum())
        result["transition_nll_with_1e_minus12_floor"] = float(-np.log(np.maximum(actual_transitions, 1e-12)).mean())
        result["transition_likelihood_caveat"] = "floor metric only; true transition NLL is infinite if any zero-support target exists"
    return result


def split_sessions(data, cutoff, inner_cutoff, cart_support="matching-sku"):
    if cart_support not in {"matching-sku", "known-total"}:
        raise ValueError("cart support must be matching-sku or known-total")
    s = data["sessions"]
    invalid_cart = s.invalid_matching_cart if cart_support == "matching-sku" else s.invalid_total_cart
    valid = ~invalid_cart & ~s.ambiguous_clock_sessions & ~s.is_censored
    early = s.terminal_known_at.lt(cutoff)
    late = s.start_at.ge(cutoff)
    outer = {"train": valid & s.train_user & early,
        "user_holdout": valid & ~s.train_user & early,
        "time_holdout": valid & s.train_user & late,
        "user_and_time_holdout": valid & ~s.train_user & late}
    inner_early = s.terminal_known_at.lt(inner_cutoff)
    inner_late = s.start_at.ge(inner_cutoff) & early
    inner = {"train": valid & s.train_user & s.inner_user & inner_early,
        "user_holdout": valid & s.train_user & ~s.inner_user & inner_early,
        "time_holdout": valid & s.train_user & s.inner_user & inner_late,
        "user_and_time_holdout": valid & s.train_user & ~s.inner_user & inner_late}
    return outer, inner, {"support_rule": cart_support, "valid_cart_clock_complete_sessions": int(valid.sum()),
        "crossing_outer_cutoff_sessions_excluded": int((valid & s.start_at.lt(cutoff) & ~early).sum()),
        "crossing_inner_cutoff_sessions_excluded": int((valid & s.start_at.lt(inner_cutoff) & ~inner_early).sum()),
        "outer_session_counts": {k: int(v.sum()) for k, v in outer.items()},
        "inner_session_counts": {k: int(v.sum()) for k, v in inner.items()},
        "outer_unknown_sku_remove_observation_audit": {name: {
            "positive_known_total_contexts": int(s.loc[mask, "unmatched_remove_positive_known_total"].sum()),
            "known_total_one_contexts": int(s.loc[mask, "unmatched_remove_known_total_one"].sum())}
            for name, mask in outer.items()}}


def _selection_score(validation, fit_target):
    if fit_target == "pooled-session":
        return float(np.average([x["action_nll"] for x in validation.values()],
                                weights=[x["records"] for x in validation.values()]))
    return float(np.mean([x["customer_macro_action_nll"] for x in validation.values()]))


def fit_causal_shopping(rows, base, seed=42, basis_kind="continuous", cart_support="matching-sku", fit_target="customer-macro"):
    if basis_kind not in {"continuous", "tabular"}:
        raise ValueError("reward basis must be continuous or tabular")
    penalties = (.00001, .0001, .001, .01) if basis_kind == "tabular" else (.01, .1, 1.0)
    if fit_target not in {"customer-macro", "pooled-session"}:
        raise ValueError("fit target must be customer-macro or pooled-session")
    discounts = (.95, .99) if fit_target == "pooled-session" else (.8, .95)
    cutoff = pd.Timestamp(base["validation"]["cutoff_at_utc"])
    inner_cutoff = cutoff - pd.DateOffset(months=1)
    data = build_demonstrations(rows, base["observation_end"], seed)
    outer, inner, split_audit = split_sessions(data, cutoff, inner_cutoff, cart_support)
    inner_panels = {name: _selected(data, mask, fit_target) for name, mask in inner.items()}
    encoder = fit_encoder(inner_panels["train"])
    candidates = []
    for gamma in discounts:
        for regularization in penalties:
            model = fit_mce(encoder, inner_panels["train"], gamma, regularization, basis_kind)
            validation = {name: evaluate(model, panel) for name, panel in inner_panels.items() if name != "train"}
            score = _selection_score(validation, fit_target)
            candidates.append({"discount": gamma, "regularization": regularization, "selection_score": score,
                "converged": model["fit"]["converged"], "fit": model["fit"], "validation": validation})
            print(json.dumps({"stage": "inner", "discount": gamma, "regularization": regularization, "score": score, "converged": model["fit"]["converged"]}), flush=True)
    usable = [x for x in candidates if x["converged"] and np.isfinite(x["selection_score"])]
    if not usable:
        raise ValueError("no inner MCE candidate converged")
    selected = min(usable, key=lambda x: x["selection_score"])
    panels = {name: _selected(data, mask, fit_target) for name, mask in outer.items()}
    encoder = fit_encoder(panels["train"])
    model = fit_mce(encoder, panels["train"], selected["discount"], selected["regularization"], basis_kind)
    if not model["fit"]["converged"]:
        raise ValueError("final source MCE fit did not converge")
    # An all-context gamma=0 comparator isolates sequential planning from the encoder/features.
    ablations = []
    for regularization in penalties:
        clone = fit_myopic(fit_encoder(inner_panels["train"]), inner_panels["train"], regularization, basis_kind)
        score = _selection_score({name: evaluate(clone, panel) for name, panel in inner_panels.items() if name != "train"}, fit_target)
        ablations.append({"regularization": regularization, "selection_score": score, "converged": clone["converged"]})
    ablation_selected = min([x for x in ablations if x["converged"]], key=lambda x: x["selection_score"])
    myopic = fit_myopic(encoder, panels["train"], ablation_selected["regularization"], basis_kind)
    model["myopic_ablation"] = {key: value for key, value in myopic.items() if key != "encoder"}
    model["myopic_ablation"]["encoder_reference"] = "same frozen encoder as this MCE block; not the deployed policy"
    model["training"] = {"seed": seed, "cutoff_at": cutoff.isoformat(), "inner_cutoff_at": inner_cutoff.isoformat(),
        "selection": ("record-pooled causal action NLL over all three inner user/time validation panels; all outer panels untouched"
                      if fit_target == "pooled-session" else "mean customer-macro action NLL over train-user/time inner panels only; all outer panels untouched"),
        "fit_target": fit_target, "encoder_weighting": "fixed source customer-balanced session statistics, unchanged across fit targets to isolate occupancy/selection objective",
        "selected_discount": selected["discount"], "selected_regularization": selected["regularization"],
        "inner_candidates": candidates, "gamma_zero_myopic_selection": ablations,
        "source_audit": data["audit"], "split_audit": split_audit}
    model["support_rule"] = cart_support
    model["validation"] = {name: {"mce_irl": evaluate(model, panel), "gamma_zero_same_basis_ablation": evaluate(myopic, panel)} for name, panel in panels.items()}
    model["limitations"] = [
        "Within-session reward/decisions only; between-session return, 28-day lifecycle, item selection and financial execution remain separate components.",
        "The logged source-conditioned transition MDP is an observed abstraction, not identified interventional environment dynamics or recovered true personal utility.",
        ("Source state bins use continuous training-context means as their reward basis; variation inside a bin is intentionally abstracted at both fit and runtime."
         if basis_kind == "continuous" else "Source state-action indicator rewards model bin interactions; continuous contexts still determine the frozen state encoder/backoff, while within-bin variation is abstracted."),
        ("Initial source inventory is unobserved: tracked carts start at zero as a lower-bound replay. Entire sessions with unmatched SKU removals are excluded; this feasible subset has selection bias."
         if cart_support == "matching-sku" else "Initial source inventory is unobserved and no units are fabricated. Only sessions with zero-known-total removals are excluded. Unmatched-SKU removals with any known held unit remain abstract feasible actions, while the exact known-SKU ledger stays unchanged; no other SKU is deducted. This is a partially observed empirical kernel, not an exact full-inventory physical transition model. At known total one, its next cart-presence can differ from runtime; those contexts are explicitly counted."),
        "Tied within-session rows use original source_event_id order; same-clock multiple-session episodes are excluded. That file order is not verified sub-second causality.",
        "END is inferred after 30 minutes of follow-up; administratively censored and cutoff-crossing sessions are excluded rather than assigned END.",
        "Positive discount is per checkout-choice step, not elapsed real time, and selected for prediction; reward gauge/temperature do not identify psychological utility.",
        "Search/review/CS and coupon-response coefficients are not jointly fitted: no cross-dataset person-level demonstration links are observed.",
        "Five source months do not validate multi-year behavior; no empirical claim is made for unseen unsupported state/action interventions."]
    return model


def _block(model):
    return model.get("causal_shopping", model)


def validate_model(model):
    block = _block(model)
    if block.get("schema_version") != 1 or block.get("type") != TYPE or block.get("actions") != list(ACTIONS):
        raise ValueError("incompatible causal shopping model")
    if block.get("support_rule", "matching-sku") not in {"matching-sku", "known-total"}:
        raise ValueError("incompatible source cart support rule")
    if block.get("fit_target", "customer-macro") not in {"customer-macro", "pooled-session"}:
        raise ValueError("incompatible causal fit target")
    if isinstance(block["discount"], bool) or not 0 <= float(block["discount"]) < 1 or block.get("temperature") != 1.0 or block.get("end_reward") != 0 or not np.isfinite(block["regularization"]) or block["regularization"] <= 0:
        raise ValueError("invalid causal reward gauge/discount")
    encoder = block["encoder"]
    if encoder.get("schema_version") != 1 or encoder.get("feature_names") != list(FEATURES) or encoder.get("context_keys") != list(CONTEXT_KEYS):
        raise ValueError("incompatible causal shopping encoder")
    states = np.asarray(encoder["state_keys"])
    n = len(states)
    if states.shape != (n, 5) or states.dtype.kind not in "iu" or not 0 < n <= 360 or len({tuple(x) for x in states}) != n or (states < 0).any() or (states >= [5, 2, 3, 3, 4]).any():
        raise ValueError("invalid finite source states")
    means = np.asarray(encoder["continuous_context_means"], dtype=float)
    center = np.asarray(encoder["feature_mean"], dtype=float)
    scale = np.asarray(encoder["feature_scale"], dtype=float)
    thresholds = encoder["thresholds"]
    if means.shape != (n, len(FEATURES)) or center.shape != (len(FEATURES),) or scale.shape != center.shape or not all(np.isfinite(x).all() for x in [means, center, scale]) or (scale <= 0).any():
        raise ValueError("invalid frozen continuous reward features")
    if set(thresholds) != {"progress_log_median", "offer_log_median", "prior_purchase_log_age_median"} or not all(np.isfinite(v) and v >= 0 for v in thresholds.values()):
        raise ValueError("invalid source-only state thresholds")
    context_basis = np.asarray(encoder["reward_basis"], dtype=float)
    basis, names = reward_basis(encoder, block.get("reward_basis_type", "continuous"))
    if block.get("reward_feature_names", names) != names:
        raise ValueError("incompatible reward feature names")
    transition = np.asarray(block["transition"], dtype=float)
    probability = np.asarray(block["policy"], dtype=float)
    mask = np.asarray(block["action_mask"], dtype=bool)
    weights = np.asarray(block["reward_weights"], dtype=float)
    if context_basis.shape != (n, len(FEATURES) + 1) or transition.shape != (n, 5, n) or probability.shape != (n, 5) or weights.shape != (4, basis.shape[1]) or mask.shape != (n, 5):
        raise ValueError("invalid causal MDP array shapes")
    if not all(np.isfinite(x).all() for x in [basis, transition, probability, weights]) or (transition < 0).any() or (probability < 0).any():
        raise ValueError("nonfinite or negative causal model values")
    if not np.isfinite(context_basis).all() or not np.allclose(context_basis, np.column_stack([np.ones(n), (means - center) / scale])):
        raise ValueError("reward basis differs from frozen runtime encoder")
    if not np.array_equal(mask, _mask(encoder)) or not np.allclose(probability.sum(axis=1), 1) or probability[~mask].any():
        raise ValueError("invalid causal action support")
    expected_sum = mask.astype(float)
    expected_sum[:, 4] = 0
    if not np.allclose(transition.sum(axis=2), expected_sum) or transition[:, 4].any():
        raise ValueError("END must have only implicit absorbing transitions")
    reward = np.column_stack([basis @ weights.T, np.zeros(n)])
    value, expected_policy, _ = soft_value_iteration(reward, transition, mask, block["discount"])
    if not np.allclose(block["value"], value, rtol=1e-8, atol=1e-8) or not np.allclose(expected_policy, probability, rtol=1e-8, atol=1e-8):
        raise ValueError("stored causal policy does not reproduce its reward/dynamics")


def load_model(path):
    model = json.loads(Path(path).read_text())
    validate_model(model)
    return model


def policy(model, context):
    """Runtime planner policy, with no donor multiplier or forced session length."""
    block = _block(model)
    cache = _CACHE.get(id(block))
    if cache is None or cache[0] is not block:
        encoder = block["encoder"]
        states = np.asarray(encoder["state_keys"])
        cache = (block, {tuple(k): i for i, k in enumerate(states)}, states,
                 np.asarray(encoder["continuous_context_means"]), np.asarray(encoder["feature_scale"]),
                 np.asarray(block["policy"]))
        if len(_CACHE) >= 4:
            _CACHE.clear()
        _CACHE[id(block)] = cache
    _, mapping, states, means, scale, probabilities = cache
    x = context_vector(context)
    key = _keys(x[None], block["encoder"]["thresholds"])[0]
    index = mapping.get(tuple(key))
    if index is None:
        eligible = np.flatnonzero((states[:, 0] == key[0]) & (states[:, 1] == key[1]) & ((states[:, 2] == 0) == (key[2] == 0)))
        if not len(eligible):
            eligible = np.flatnonzero((states[:, 1] == key[1]) & ((states[:, 2] == 0) == (key[2] == 0)))
        if not len(eligible):
            raise ValueError("no source state supports runtime cart/start context")
        index = int(eligible[np.argmin(np.square((means[eligible] - x) / scale).sum(axis=1))])
    result = probabilities[index].copy()
    if context.get("session_decision_count", 0) == 0:
        result[4] = 0
    if context.get("cart_total_quantity", 0) == 0:
        result[2] = 0
    if not result.sum() > 0:
        raise ValueError("causal shopping policy has no feasible runtime action")
    result /= result.sum()
    return dict(zip(ACTIONS, result.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/models/shopping_profiles_aligned_v6.json"))
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/models/shopping_causal_irl_candidate.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/audit/simulator/causal_shopping_training.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-basis", choices=("continuous", "tabular"), default="continuous")
    parser.add_argument("--cart-support", choices=("matching-sku", "known-total"), default="matching-sku")
    parser.add_argument("--fit-target", choices=("customer-macro", "pooled-session"), default="customer-macro")
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        raise ValueError("model output and report must have separate paths")
    for destination in (args.output, args.report):
        if destination.exists():
            raise FileExistsError(f"immutable fitting output already exists: {destination}")
    from simulate.v4.irl.fit_donor_prior import load_prior_source
    base = json.loads(args.base.read_text())
    rows, source = load_prior_source(base, args.source_cache)
    block = fit_causal_shopping(rows, base, args.seed, args.reward_basis, args.cart_support, args.fit_target)
    block["provenance"] = {"base_path": str(args.base), "base_sha256": _digest(args.base),
        "source_cache": source, "source_cache_sha256": _digest(args.source_cache) if args.source_cache.is_file() else None,
        "fitter_sha256": _digest(__file__), "no_auxiliary_customer_identity_join": True}
    validate_model(block)
    artifact = deepcopy(base)
    artifact["causal_shopping"] = block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    evidence = {"passed": True, "model_path": str(args.output), "model_sha256": _digest(args.output),
        "base_fields_unchanged": {key: artifact[key] for key in base} == base,
        "provenance": block["provenance"], "source_audit": block["training"]["source_audit"],
        "training": block["training"], "fit": block["fit"], "validation": block["validation"],
        "reward_basis_type": block["reward_basis_type"], "reward_feature_names": block["reward_feature_names"],
        "support_rule": block["support_rule"],
        "fit_target": block["fit_target"],
        "myopic_ablation": block["myopic_ablation"],
        "limitations": block["limitations"], "adoption": "candidate only; runtime source holdout and matched rollout quality gate required"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"model_path": str(args.output), "model_sha256": evidence["model_sha256"], "fit": block["fit"]}), flush=True)


if __name__ == "__main__":
    main()
