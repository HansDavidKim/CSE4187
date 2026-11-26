"""Fit a source-derived Dirichlet prior for donor action-count personalization."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import numbers
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


ACTIONS = ("view", "cart", "remove", "purchase", "end")
TYPE = "source_empirical_bayes_dirichlet_action_prior"


def validate_donor_prior(block):
    if (not isinstance(block, dict) or type(block.get("schema_version")) is not int
            or block.get("schema_version") != 1 or block.get("type") != TYPE
            or block.get("actions") != list(ACTIONS)):
        raise ValueError("incompatible donor action prior")
    strength = block.get("strength")
    if isinstance(strength, bool) or not isinstance(strength, numbers.Real) or not math.isfinite(strength) or strength <= 0:
        raise ValueError("donor prior strength must be positive and finite")
    for key in ("base_probabilities", "alpha"):
        values = block.get(key)
        if (not isinstance(values, dict) or set(values) != set(ACTIONS)
                or any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) or v <= 0
                       for v in values.values())):
            raise ValueError("donor prior probabilities and alpha must have positive finite action support")
    if not math.isclose(sum(block["base_probabilities"].values()), 1., rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("donor prior base probabilities must sum to one")
    if (not math.isclose(sum(block["alpha"].values()), strength, rel_tol=1e-10)
            or any(not math.isclose(block["alpha"][a], strength * block["base_probabilities"][a], rel_tol=1e-10)
                   for a in ACTIONS)):
        raise ValueError("donor prior alpha must equal strength times base probabilities")
    calibration = block.get("policy_calibration")
    if calibration is not None:
        if (not isinstance(calibration, dict) or type(calibration.get("schema_version")) is not int
                or calibration.get("schema_version") not in (1, 2)
                or calibration.get("type") != "source_nonterminal_mnl_donor_joint_calibration"
                or calibration.get("actions") != list(ACTIONS[:-1]) or calibration.get("anchor_action") != "view"):
            raise ValueError("incompatible donor policy calibration")
        gamma, biases = calibration.get("gamma"), calibration.get("action_intercepts")
        if (isinstance(gamma, bool) or not isinstance(gamma, numbers.Real) or not math.isfinite(gamma) or gamma < 0
                or not isinstance(biases, dict) or set(biases) != set(ACTIONS[:-1])
                or any(isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value)
                       for value in biases.values()) or biases["view"] != 0.):
            raise ValueError("donor policy gamma/intercepts must be finite with nonnegative gamma and fixed view anchor")
        if calibration["schema_version"] == 2:
            timing = calibration.get("purchase_timing")
            if (not isinstance(timing, dict) or type(timing.get("schema_version")) is not int or timing.get("schema_version") != 1
                    or timing.get("type") != "source_purchase_age_bins"):
                raise ValueError("incompatible donor purchase timing")
            knots = timing.get("age_knots_days")
            coefficients = timing.get("different_session_log_multipliers")
            same = timing.get("same_session_log_multiplier")
            def finite(value):
                return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)
            if (not isinstance(knots, list) or len(knots) > 3 or any(not finite(v) or v <= 0 for v in knots)
                    or knots != sorted(set(knots)) or not isinstance(coefficients, list) or len(coefficients) != len(knots) + 1
                    or not all(finite(v) for v in coefficients) or not finite(same)):
                raise ValueError("purchase timing needs finite coefficients and ordered positive source age knots")
            try:
                transformed = [math.exp(v) for v in [same, *coefficients]]
            except OverflowError as exc:
                raise ValueError("purchase timing coefficients must exponentiate to finite multipliers") from exc
            if any(not math.isfinite(v) or v <= 0 for v in transformed):
                raise ValueError("purchase timing coefficients must exponentiate to positive finite multipliers")


def donor_action_weights(block, counts, observed_sessions):
    """Return posterior/prior ratios in the same count units as the donor artifact."""
    validate_donor_prior(block)
    if (not isinstance(counts, dict) or set(counts) - set(ACTIONS[:-1])
            or isinstance(observed_sessions, bool) or not isinstance(observed_sessions, numbers.Integral)
            or observed_sessions < 0
            or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0 for v in counts.values())):
        raise ValueError("donor counts need nonnegative integer decisions and observed sessions")
    total = sum(counts.values()) + observed_sessions
    strength, base = block["strength"], block["base_probabilities"]
    values = {action: ((observed_sessions if action == "end" else counts.get(action, 0))
                       + block["alpha"][action]) / (total + strength) / base[action] for action in ACTIONS}
    calibration = block.get("policy_calibration")
    if calibration is not None:
        logs = {a: calibration["gamma"] * math.log(values[a]) + calibration["action_intercepts"][a]
                for a in ACTIONS[:-1]}
        # A shared scale cancels in the shopping softmax and prevents exponential overflow.
        maximum = max(logs.values())
        values = {a: math.exp(value - maximum) for a, value in logs.items()}
        values["end"] = 1.  # END is masked, rather than learned, in the conditional runtime policy.
    if not all(math.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError("donor prior produced nonfinite action weights")
    return values


def donor_purchase_time_multiplier(block, context):
    """Apply validated source timing to purchase only; earlier policy schemas are neutral."""
    calibration = block.get("policy_calibration")
    if calibration is None or calibration.get("schema_version") == 1:
        return 1.
    timing = calibration["purchase_timing"]
    age, same = context.get("days_since_purchase"), context.get("purchase_in_current_session")
    if not isinstance(same, bool):
        raise ValueError("purchase timing requires a boolean current-session history flag")
    if age is None:
        if same:
            raise ValueError("same-session purchase history requires an observed prior purchase")
        return 1.
    if isinstance(age, bool) or not isinstance(age, numbers.Real) or not math.isfinite(age) or age < 0:
        raise ValueError("purchase age must be finite and nonnegative")
    coefficient = (timing["same_session_log_multiplier"] if same else
                   timing["different_session_log_multipliers"][int(np.searchsorted(timing["age_knots_days"], age, side="right"))])
    try:
        value = math.exp(coefficient)
    except OverflowError as exc:
        raise ValueError("purchase timing multiplier must remain finite") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("purchase timing multiplier must remain finite and positive")
    return value


def build_prior_records(rows, observation_end, seed=42):
    """Count strictly earlier decisions, and END only once its inferred clock is known."""
    required = {"user_id", "session", "at", "action"}
    if not required <= set(rows) or rows.empty or not rows.action.isin(ACTIONS[:-1]).all():
        raise ValueError("donor prior needs prepared observed shopping rows")
    if not isinstance(rows["at"].dtype, pd.DatetimeTZDtype) or rows["at"].isna().any():
        raise ValueError("donor source clocks must be timezone-aware")
    end = pd.Timestamp(observation_end)
    if pd.isna(end) or end.tzinfo is None:
        raise ValueError("donor observation end must be timezone-aware")
    source = rows.sort_values(["user_id", "at"], kind="stable").copy()
    following = source.groupby("user_id", sort=False).shift(-1)
    internal = (source.action.eq("purchase") & following.action.eq("purchase")
                & source.session.eq(following.session) & source["at"].eq(following["at"]))
    decisions = source.loc[~internal, ["user_id", "at", "action"]]
    sessions = (source.groupby(["user_id", "session"], sort=False)
                .agg(start=("at", "min"), last=("at", "max")).reset_index()
                .sort_values(["user_id", "start"], kind="stable"))
    ends = sessions["last"] + pd.Timedelta(minutes=30)
    next_start = sessions.groupby("user_id", sort=False).start.shift(-1)
    ends = ends.where(next_start.isna() | ends.le(next_start), next_start)
    observed = ends.lt(end)
    terminals = pd.DataFrame({"user_id": sessions.loc[observed, "user_id"], "at": ends.loc[observed], "action": "end"})
    events = pd.concat([decisions, terminals], ignore_index=True).sort_values(["user_id", "at"], kind="stable")
    if events["at"].ge(end).any():
        events = events.loc[events["at"].lt(end)]
    grouped = events.groupby(["user_id", "at", "action"], sort=True).size().unstack(fill_value=0).reindex(columns=ACTIONS, fill_value=0)
    prior = grouped.groupby(level="user_id", sort=False).cumsum() - grouped
    records = events.merge(prior.rename(columns={a: f"prior_{a}" for a in ACTIONS}),
                           on=["user_id", "at"], how="left", validate="many_to_one")
    records["train_user"] = records.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
    return records.reset_index(drop=True), {"observed_decisions": len(decisions), "observed_end_tokens": int(observed.sum()),
        "right_censored_end_tokens": int((~observed).sum()), "within_checkout_unit_rows_excluded": int(internal.sum()),
        "same_timestamp_count_rule": "all action counts have timestamps strictly before the predicted event; tied events do not update one another",
        "end_clock": "min(last observed session action + 30 minutes, next observed session start), only within observation"}


def _scores(records, alpha, pooled):
    if records.empty:
        return {"records": 0, "users": 0}
    counts = records[[f"prior_{a}" for a in ACTIONS]].to_numpy(dtype=float)
    y = records.action.map({a: i for i, a in enumerate(ACTIONS)}).to_numpy()
    total, true = counts.sum(axis=1), counts[np.arange(len(counts)), y]
    loss = np.log(total + alpha.sum()) - np.log(true + alpha[y])
    legacy = np.log(total + 2.5) - np.log(true + .5)
    weights = 1 / records.user_id.map(records.user_id.value_counts()).to_numpy()
    q = (counts + alpha) / (total[:, None] + alpha.sum())
    return {"records": len(records), "users": int(records.user_id.nunique()),
        "nll": float(loss.mean()), "legacy_uniform_half_nll": float(legacy.mean()),
        "pooled_training_nll": float(-np.log(pooled[y]).mean()),
        "customer_balanced_nll": float(np.average(loss, weights=weights)),
        "customer_balanced_legacy_uniform_half_nll": float(np.average(legacy, weights=weights)),
        "observed_action_share": dict(zip(ACTIONS, np.bincount(y, minlength=5) / len(y))),
        "predicted_action_share": dict(zip(ACTIONS, q.mean(axis=0)))}


def fit_donor_prior(rows, cutoff, observation_end, seed=42, source=None):
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("donor training cutoff must be timezone-aware")
    records, audit = build_prior_records(rows, observation_end, seed)
    before, known_user = records["at"].lt(cutoff), records.train_user
    train = records.loc[before & known_user]
    frequency = train.action.value_counts().reindex(ACTIONS, fill_value=0).to_numpy(dtype=float)
    if len(train) < 10 or (frequency <= 0).any():
        raise ValueError("donor fitting needs all five observed actions and at least ten training records")
    pooled = frequency / frequency.sum()
    counts = train[[f"prior_{a}" for a in ACTIONS]].to_numpy(dtype=float)
    total = counts.sum(axis=1)
    y = train.action.map({a: i for i, a in enumerate(ACTIONS)}).to_numpy()
    true = counts[np.arange(len(counts)), y]
    def objective(log_alpha):
        alpha = np.exp(log_alpha)
        loss = np.log(total + alpha.sum()) - np.log(true + alpha[y])
        gradient = np.full(5, np.mean(1 / (total + alpha.sum())))
        gradient -= np.bincount(y, weights=1 / (true + alpha[y]), minlength=5) / len(y)
        return float(loss.mean()), alpha * gradient
    initial = frequency / train.user_id.nunique()
    bounds = [(math.log(1e-8), math.log(1e8))] * len(ACTIONS)
    fitted = minimize(objective, np.log(initial), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f"donor prior optimization failed: {fitted.message}")
    alpha = np.exp(fitted.x)
    strength = float(alpha.sum())
    splits = {"train": before & known_user, "user_holdout": before & ~known_user,
              "time_holdout": ~before & known_user, "user_and_time_holdout": ~before & ~known_user}
    block = {"schema_version": 1, "type": TYPE, "actions": list(ACTIONS), "strength": strength,
        "base_probabilities": dict(zip(ACTIONS, (alpha / strength).tolist())), "alpha": dict(zip(ACTIONS, alpha.tolist())),
        "count_units": "checkout decisions plus one inferred END per observed session",
        "formula": "posterior(action)/prior(action), with posterior=(count+alpha)/(sum(counts)+sum(alpha))",
        "source": source or {"kind": "explicit deterministic source fixture"}, "source_audit": audit,
        "fit": {"seed": seed, "cutoff_at": cutoff.isoformat(), "observation_end_at": pd.Timestamp(observation_end).isoformat(),
            "objective": "event-weighted strictly-past predictive categorical negative log likelihood; no holdout parameter selection",
            "optimizer": "L-BFGS-B analytic gradient", "iterations": int(fitted.nit), "converged": bool(fitted.success),
            "per_alpha_numerical_bounds": [1e-8, 1e8], "parameter_at_bound": bool(np.isclose(alpha, 1e-8).any() or np.isclose(alpha, 1e8).any()),
            "initial_alpha": initial.tolist(), "training_action_counts": dict(zip(ACTIONS, frequency.astype(int).tolist()))},
        "validation": {name: _scores(records.loc[mask], alpha, pooled) for name, mask in splits.items()},
        "limitations": ["This validates a donor-count prior, not the joint contextual shopping-policy likelihood or rollout fidelity.",
            "Source END is inferred and may be censored. Runtime's observed-session count includes one END proxy per donor session.",
            "Runtime donor counts describe the entire donor history and are latent scenario traits, not future-safe customer predictors.",
            "Positive alpha bounds are numerical constraints, not observed pseudo-counts. Source profiles and shopping coefficients remain unchanged."]}
    validate_donor_prior(block)
    return block


def purchase_timing_contexts(rows, labels, features):
    """Recover current/last purchase sessions without reading same-clock future purchases."""
    from simulate.v4.irl.fit_shopping import FEATURE_NAMES
    clocks = rows.groupby(["user_id", "at"]).session.agg(["first", "nunique"]).reset_index()
    contexts = labels[["user_id", "at", "target_at"]].reset_index(names="feature_row")
    contexts = contexts.merge(clocks.rename(columns={"first": "context_session", "nunique": "context_session_count"}),
                              on=["user_id", "at"], how="left", validate="many_to_one")
    targets = clocks.rename(columns={"at": "target_at", "first": "target_session", "nunique": "target_session_count"})
    contexts = contexts.merge(targets, on=["user_id", "target_at"], how="left", validate="many_to_one")
    purchases = (rows.loc[rows.action.eq("purchase")].groupby(["user_id", "at"]).session.agg(["first", "nunique"])
                 .reset_index().rename(columns={"at": "last_purchase_at", "first": "last_purchase_session",
                                                "nunique": "last_purchase_session_count"}))
    contexts = pd.merge_asof(contexts.sort_values("at"), purchases.sort_values("last_purchase_at"),
                            left_on="at", right_on="last_purchase_at", by="user_id", allow_exact_matches=False)
    contexts = contexts.sort_values("feature_row")
    if contexts.context_session.isna().any():
        raise ValueError("a source shopping context has no source session")
    history = features[:, FEATURE_NAMES.index("has_purchase_history")] > 0
    ages = np.expm1(features[:, FEATURE_NAMES.index("log_days_since_purchase")])
    current_purchase = features[:, FEATURE_NAMES.index("previous_purchase")] > 0
    zero_age = history & (ages == 0)
    # The frozen MNL context already knows a processed same-time purchase; no <= source scan is used.
    observed_here = history & (zero_age | current_purchase)
    earlier_age = (contexts["at"] - contexts.last_purchase_at).dt.total_seconds().to_numpy() / 86400
    mismatch = history & ~observed_here & (~np.isfinite(earlier_age) | ~np.isclose(ages, earlier_age, rtol=1e-9, atol=1e-9))
    ambiguous_context = contexts.context_session_count.ne(1).to_numpy()
    ambiguous_target = contexts.target_session_count.gt(1).to_numpy()
    ambiguous_prior = history & ~observed_here & contexts.last_purchase_session_count.gt(1).to_numpy()
    excluded = ambiguous_context | ambiguous_target | ambiguous_prior | mismatch
    same = history & (observed_here | contexts.context_session.eq(contexts.last_purchase_session).to_numpy())
    audit = {"ambiguous_context_clock_multi_session": int(ambiguous_context.sum()),
             "ambiguous_target_clock_multi_session": int(ambiguous_target.sum()),
             "ambiguous_prior_purchase_clock_multi_session": int(ambiguous_prior.sum()),
             "prior_purchase_age_context_mismatch": int(mismatch.sum()),
             "known_same_clock_purchase_from_context": int(observed_here.sum()),
             "excluded_union": int(excluded.sum()), "reason_counts_may_overlap": True,
             "clock_policy": "raw purchase session lookup is strictly earlier than context.at; zero-age history uses already-observed frozen MNL context only"}
    return ages, history, same, excluded, audit


def fit_policy_calibration(rows, model, prior, seed=42):
    """Jointly calibrate a frozen MNL and count prior on feasible nonterminal source targets."""
    from simulate.v4.irl.fit_shopping import FEATURE_NAMES, build_records, choice_probabilities_batch
    if model.get("choice_unit") != "source_checkout_episode" or seed != model.get("seed"):
        raise ValueError("joint calibration requires the same checkout artifact and user-split seed")
    if "policy_calibration" in prior:
        raise ValueError("calibrate an unmodified source prior once")
    cutoff, end = pd.Timestamp(model["validation"]["cutoff_at_utc"]), pd.Timestamp(model["observation_end"])
    features, labels, _, _ = build_records(rows, model["reference_price_minor"], end, seed, checkout_episodes=True)
    records, _ = build_prior_records(rows, end, seed)
    count_columns = [f"prior_{a}" for a in ACTIONS]
    # Every timestamp group shares a strict prefix, including when several targets are tied.
    if (records.groupby(["user_id", "at"])[count_columns].nunique().to_numpy() > 1).any():
        raise ValueError("tied source targets have inconsistent prior counts")
    prefixes = records[["user_id", "at", *count_columns]].drop_duplicates(["user_id", "at"])
    joined = labels.reset_index(names="feature_row").merge(prefixes, left_on=["user_id", "target_at"],
        right_on=["user_id", "at"], how="left", validate="many_to_one", suffixes=("", "_prefix")).sort_values("feature_row")
    target = labels.action.to_numpy(dtype=int)
    nonterminal = target < len(ACTIONS)-1
    if joined.loc[nonterminal, count_columns].isna().any().any():
        raise ValueError("a nonterminal shopping target has no observed strict prefix")
    held = features[:, FEATURE_NAMES.index("log_cart_total_quantity")] > 0
    infeasible_remove = nonterminal & (target == ACTIONS.index("remove")) & ~held
    ages, history, same_session, timing_excluded, timing_audit = purchase_timing_contexts(rows, labels, features)
    usable = nonterminal & ~infeasible_remove & ~timing_excluded & ~labels.is_censored.to_numpy()
    counts = joined[count_columns].fillna(0).to_numpy(dtype=float)
    alpha = np.array([prior["alpha"][a] for a in ACTIONS])
    base = np.array([prior["base_probabilities"][a] for a in ACTIONS])
    ratios = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha.sum()) / base
    log_ratios = np.log(ratios[:, :-1])
    base_probabilities = choice_probabilities_batch(model, features)[:, :-1]
    base_logits = np.log(np.maximum(base_probabilities, 1e-300))
    base_logits[~held, ACTIONS.index("remove")] = -np.inf
    early = labels.target_at.lt(cutoff).to_numpy()
    later = labels["at"].ge(cutoff).to_numpy()
    train_user = labels.train_user.to_numpy()
    masks = {"train": usable & train_user & early, "user_holdout": usable & ~train_user & early,
             "time_holdout": usable & train_user & later, "user_and_time_holdout": usable & ~train_user & later}
    train = masks["train"]
    if train.sum() < 10 or len(np.unique(target[train])) != 4:
        raise ValueError("joint calibration needs four feasible observed training actions")
    repeat_positive = train & history & ~same_session & (target == ACTIONS.index("purchase"))
    observed_ages = ages[repeat_positive]
    raw_knots = np.quantile(observed_ages, [.25, .5, .75]).tolist() if len(observed_ages) else []
    knots = sorted(set(raw_knots))
    bins = np.searchsorted(knots, ages, side="right")
    categories = [history & same_session, *[history & ~same_session & (bins == k) for k in range(len(knots)+1)]]
    support = []
    active = []
    for index, category in enumerate(categories):
        local = train & category
        positives = int((local & (target == ACTIONS.index("purchase"))).sum())
        negatives = int(local.sum()) - positives
        estimated = positives > 0 and negatives > 0
        if estimated:
            active.append(index)
        support.append({"records": int(local.sum()), "purchase_targets": positives, "nonpurchase_targets": negatives,
                        "estimated": estimated, "status": "source_mle" if estimated else "not_estimated_no_positive_or_no_negative_support"})
    design = np.column_stack([categories[i] for i in active]).astype(float) if active else np.empty((len(labels), 0))
    train_design = design[train]
    train_logits, train_ratios, y = base_logits[train], log_ratios[train], target[train]
    def objective(parameters):
        logits = train_logits + parameters[0] * train_ratios + np.r_[0., parameters[1:4]]
        logits[:, ACTIONS.index("purchase")] += train_design @ parameters[4:]
        normalizer = logsumexp(logits, axis=1)
        probabilities = np.exp(logits - normalizer[:, None])
        probabilities[np.arange(len(y)), y] -= 1
        gradient = np.r_[np.mean(np.sum(probabilities * train_ratios, axis=1)), probabilities[:, 1:].mean(axis=0),
                         train_design.T @ probabilities[:, ACTIONS.index("purchase")] / len(y)]
        return float(np.mean(normalizer-logits[np.arange(len(y)), y])), gradient
    result = minimize(objective, np.zeros(4 + len(active)), jac=True, method="L-BFGS-B",
                      bounds=[(0., None), *[(None, None)] * (3 + len(active))],
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"joint donor policy calibration failed: {result.message}")
    gamma, biases = float(result.x[0]), np.r_[0., result.x[1:4]]
    timing_coefficients = np.zeros(len(categories))
    timing_coefficients[active] = result.x[4:]
    validation = {}
    def metrics(logits, mask):
        local, classes = logits[mask], target[mask]
        norm = logsumexp(local, axis=1)
        losses = norm - local[np.arange(len(classes)), classes]
        probabilities = np.exp(local - norm[:, None])
        return {"nll": float(losses.mean()),
                "observed_action_share": dict(zip(ACTIONS[:-1], (np.bincount(classes, minlength=4)/len(classes)).tolist())),
                "predicted_action_share": dict(zip(ACTIONS[:-1], probabilities.mean(axis=0).tolist()))}
    candidate_logits = base_logits + gamma * log_ratios + biases
    candidate_logits[:, ACTIONS.index("purchase")] += design @ result.x[4:]
    for name, mask in masks.items():
        validation[name] = {"records": int(mask.sum()), "users": int(labels.loc[mask, "user_id"].nunique())}
        if mask.any():
            validation[name].update({"fixed_mnl": metrics(base_logits, mask),
                                    "uncalibrated_prior": metrics(base_logits + log_ratios, mask),
                                    "joint_calibrated": metrics(candidate_logits, mask)})
    timing = {"schema_version": 1, "type": "source_purchase_age_bins", "age_knots_days": knots,
        "different_session_log_multipliers": timing_coefficients[1:].tolist(), "same_session_log_multiplier": float(timing_coefficients[0]),
        "source_quantiles": {"probabilities": [.25, .5, .75], "raw_quantile_days": raw_knots,
            "positive_different_session_repeat_targets": len(observed_ages), "collapsed_duplicate_knots": len(raw_knots)-len(knots)},
        "fit_support": {"same_session": support[0], "different_session_bins": support[1:],
            "no_purchase_history": {"records": int((train & ~history).sum()),
                "purchase_targets": int((train & ~history & (target == ACTIONS.index("purchase"))).sum()), "log_multiplier": 0.}},
        "runtime_clock": "days_since_purchase at current source/runtime context; same-session flag uses actually observed purchase session",
        "extrapolation": "last age bin is open-ended; five source months do not validate long-horizon hazards",
        "unsupported_category_policy": "coefficient fixed at neutral zero, not estimated, when training has no positive or no negative target"}
    return {"schema_version": 2, "type": "source_nonterminal_mnl_donor_joint_calibration", "actions": list(ACTIONS[:-1]),
        "anchor_action": "view", "gamma": gamma, "action_intercepts": dict(zip(ACTIONS[:-1], biases.tolist())),
        "purchase_timing": timing,
        "formula": "conditional softmax(log P_frozen_MNL + gamma * log posterior/prior + action_intercept + purchase-only source-age offset); remove masked when known cart is empty",
        "fit": {"seed": seed, "cutoff_at": cutoff.isoformat(), "training_records": int(train.sum()),
            "objective": "feasible nonterminal target categorical MLE; train users with target known strictly before cutoff only",
            "optimizer": "convex logit calibration with analytic gradient; L-BFGS-B", "iterations": int(result.nit),
            "converged": bool(result.success), "gamma_lower_bound": 0., "gamma_at_lower_bound": gamma == 0.,
            "source_choice_model_sha256": hashlib.sha256(json.dumps(model["choice_model"], sort_keys=True).encode()).hexdigest()},
        "source_audit": {"choice_records": len(labels), "nonterminal_targets": int(nonterminal.sum()),
            "end_targets_excluded": int((~nonterminal).sum()), "remove_targets_without_known_cart_excluded": int(infeasible_remove.sum()),
            "nonterminal_prefix_missing": 0, "strict_same_timestamp_prefix": True, "purchase_timing_context": timing_audit,
            "timing_exclusions_from_nonterminal_feasible_targets": int((nonterminal & ~infeasible_remove & timing_excluded).sum()),
            "crossing_cutoff_usable_targets_excluded": int((usable & ~early & ~later).sum()),
            "removed_infeasible_by_split": {name: int((infeasible_remove & mask).sum()) for name, mask in
                {"train": train_user & early, "user_holdout": ~train_user & early,
                 "time_holdout": train_user & later, "user_and_time_holdout": ~train_user & later}.items()}},
        "validation": validation,
        "limitations": ["Fixed base MNL was fitted on these same training users/times: this is in-sample stacking, not cross-fitted calibration.",
            "END is excluded as a predicted action because runtime termination uses empirical donor lengths, while END remains a past donor-count unit.",
            "Removal with no known source cart is excluded and audited; unknown initial inventory is not fabricated.",
            "Source calibration uses strictly past source counts. Runtime donor profiles are full-history latent traits, so source likelihood alone cannot establish rollout fidelity.",
            "No holdout label or generated action-share target enters this MLE; outer panels remain frozen diagnostic comparisons."]}


def load_prior_source(model, cache_path):
    """Load a matching cache or publish one from the unchanged recorded raw files."""
    binding = model["source"].get("prepared_source_cache")
    if binding is not None:
        from .source_contract import load_prepared_source
        contract = model.get("source_contract", {})
        rows, descriptor = load_prepared_source(cache_path, domain=model["source"]["domain"],
            cutoff_at=model["source"].get("training_cutoff_at", contract.get("training_cutoff_at", contract.get("outer_cutoff_at"))),
            inner_cutoff_at=contract.get("inner_cutoff_at"))
        actual = descriptor["source_provenance"]
        if (actual["prepared_source_cache"]["sha256"] != binding["sha256"]
                or actual["prepared_source_cache"]["manifest_sha256"] != binding.get("manifest_sha256")
                or actual["files"] != model["source"]["files"]
                or actual["sample_modulus"] != model["source"]["sample_modulus"]):
            raise ValueError("prepared source differs from the fitted source contract")
        return rows, descriptor
    from simulate.v4.irl.fit_shopping import load_source, prepare_source
    recorded = model["source"]["files"]
    if not recorded:
        raise ValueError("donor fitting requires recorded original source files")
    for item in recorded:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
            raise ValueError("original source file changed after base fitting")
    spec = {"files": recorded, "sample_modulus": model["source"]["sample_modulus"],
            "prepare_source_sha256": hashlib.sha256(inspect.getsource(prepare_source).encode()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = Path(cache_path)
    if path.exists():
        import pyarrow.parquet as pq
        block = pq.read_schema(path).metadata or {}
        descriptor = json.loads(block.get(b"quality_source_cache", b"{}"))
        if descriptor.get("fingerprint") == fingerprint:
            rows = pd.read_parquet(path)
            if len(rows) != descriptor["rows"]:
                raise ValueError("prepared donor source cache row count changed")
            return rows, {**descriptor, "cache_hit": True, "path": str(path)}
        raise ValueError("prepared source cache does not match the recorded raw files; use an unused cache path")
    raw, provenance = load_source(Path(recorded[0]["path"]).parent, spec["sample_modulus"])
    if provenance["files"] != recorded:
        raise ValueError("raw donor source file list or provenance differs from the base fit")
    rows, _, quality = prepare_source(raw)
    descriptor = {"fingerprint": fingerprint, "spec": spec, "source_quality": quality,
                  "source_provenance": provenance, "rows": len(rows), "cache_hit": False,
                  "path": str(path), "cache_policy": "created from the unchanged recorded raw files; never overwritten"}
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(rows, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
        b"quality_source_cache": json.dumps(descriptor, sort_keys=True, allow_nan=False).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return rows, descriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/models/shopping_profiles_checkout.json"))
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joint-policy", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("donor prior artifact output must be new")
    content = args.base.read_bytes()
    base = json.loads(content)
    if base.get("choice_unit") != "source_checkout_episode" or "donor_prior" in base:
        raise ValueError("augment an unchanged checkout artifact without a donor prior")
    rows, descriptor = load_prior_source(base, args.source_cache)
    source = {"base_artifact_path": str(args.base), "base_artifact_sha256": hashlib.sha256(content).hexdigest(),
        "prepared_source_cache": descriptor, "fitter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    block = fit_donor_prior(rows, base["validation"]["cutoff_at_utc"], base["observation_end"], args.seed, source)
    if args.joint_policy:
        block["policy_calibration"] = fit_policy_calibration(rows, base, block, args.seed)
        validate_donor_prior(block)
    candidate = copy.deepcopy(base)
    candidate["donor_prior"] = block
    assert {k: v for k, v in candidate.items() if k != "donor_prior"} == base
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(candidate, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "strength": block["strength"], "base_probabilities": block["base_probabilities"],
                      "fit": block["fit"], "validation": block["validation"], "policy_calibration": block.get("policy_calibration")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
