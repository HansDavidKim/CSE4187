"""Descriptive coupon-receipt and redemption analyses; not causal effects."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from utils.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT

REACT_OUT = ROOT / "artifacts/coupon-reactivation"


def interval(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, mid-half), min(1, mid+half)]


def cohort_from_counts(data):
    prior = data["prior_0"].gt(0) | data["prior_1"].gt(0)
    quiet = data["quiet_0"].eq(0) & data["quiet_1"].eq(0)
    cohort = data.loc[prior & quiet].copy()
    clicks, buys = cohort.prior_0.gt(0), cohort.prior_1.gt(0)
    cohort["group"] = np.select([clicks & ~buys, clicks & buys], ["click_only", "click_purchase"], default="purchase_only")
    cohort["treated"] = cohort.anchor_2.gt(0).astype(int)
    return cohort


def run_reactivation():
    REACT_OUT.mkdir(parents=True, exist_ok=True)
    chunks = []
    source = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    for c in pd.read_csv(source, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        receipt = c.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(c.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        diff = (day - pd.Timestamp("2016-05-15")).dt.days
        period = np.select([diff.between(-44, -15), diff.between(-14, -1), diff.eq(0),
                            diff.between(1, 7), diff.between(8, 14), diff.between(15, 21), diff.between(22, 28)],
                           ["prior", "quiet", "anchor", "w1", "w2", "w3", "w4"], default="outside")
        mask = (period != "outside") & c.Action.isin([0, 1, 2])
        data = c.loc[mask, ["User_id", "Action"]].copy()
        data["period"] = period[mask]
        chunks.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(chunks).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{p}_{a}" for p, a in wide.columns]
    for p in ["prior", "quiet", "anchor", "w1", "w2", "w3", "w4"]:
        for action in [0, 1, 2]:
            name = f"{p}_{action}"
            if name not in wide:
                wide[name] = 0
    cohort = cohort_from_counts(wide)
    assert cohort.index.is_unique
    result = []
    for (group, treated), part in cohort.groupby(["group", "treated"]):
        n = len(part)
        click = part[[f"w{w}_0" for w in range(1, 5)]].to_numpy() > 0
        buy = part[[f"w{w}_1" for w in range(1, 5)]].to_numpy() > 0
        any_action = click | buy
        first = np.where(any_action.any(axis=1), np.argmax(any_action, axis=1) + 1, 0)
        active28, purchased28 = int(any_action.any(axis=1).sum()), int(buy.any(axis=1).sum())
        first7 = int(any_action[:, 0].sum())
        later_receipt = part[[f"w{w}_2" for w in range(1, 5)]].gt(0).any(axis=1)
        result.append({"group": group, "anchor_receipt": int(treated), "n": n,
                       "return_7d": first7, "return_7d_ci": interval(first7, n),
                       "return_28d": active28, "return_28d_ci": interval(active28, n),
                       "purchase_28d": purchased28, "first_return_week_0_none": np.bincount(first, minlength=5).tolist(),
                       "prior_receipt_quiet": int(part.quiet_2.gt(0).sum()),
                       "later_receipt_28d": int(later_receipt.sum()),
                       "prior_click_events_mean": float(part.prior_0.mean()), "prior_purchase_events_mean": float(part.prior_1.mean())})
    (REACT_OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


PERSIST_OUT = ROOT / "artifacts/coupon-persistence"


def summarize_persistence(bits):
    assert bits.ndim == 2 and bits.shape[1] == 6
    n = len(bits)
    count = bits.sum(axis=1)
    first = np.where(count > 0, np.argmax(bits, axis=1) + 1, 0)
    last = np.where(count > 0, 6 - np.argmax(bits[:, ::-1], axis=1), 0)
    return {"n": n, "weekly_n": bits.sum(axis=0).astype(int).tolist(),
            "active_weeks_distribution": np.bincount(count, minlength=7).astype(int).tolist(),
            "first_recorded_week_distribution_0_none": np.bincount(first, minlength=7).astype(int).tolist(),
            "last_recorded_week_distribution_0_none": np.bincount(last, minlength=7).astype(int).tolist(),
            "week1_users": int(bits[:, 0].sum()),
            "week1_and_week6_users": int((bits[:, 0] & bits[:, 5]).sum()),
            "week1_and_any_week2to6_users": int((bits[:, 0] & bits[:, 1:].any(axis=1)).sum()),
            "active_week6_right_boundary": int(bits[:, 5].sum())}


def run_persistence():
    PERSIST_OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(ROOT / "artifacts/canonical-type-bounds/tianchi_cohort.csv")
    assert cohort.User_id.is_unique
    users = pd.Index(cohort.User_id)
    chunks = []
    source = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    for c in pd.read_csv(source, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        c = c[c.User_id.isin(users)].copy()
        day = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        receipt = c.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(c.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - pd.Timestamp("2016-05-15")).dt.days
        mask = delta.between(1, 42) & c.Action.isin([0, 1, 2])
        c = c[mask].copy()
        c["week"] = ((delta[mask] - 1) // 7 + 1).astype(int)
        chunks.append(c.groupby(["User_id", "week", "Action"]).size())
    totals = pd.concat(chunks).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    totals.columns = [f"w{w}_a{a}" for w, a in totals.columns]
    data = cohort.join(totals, on="User_id").fillna(0)
    result = []
    for (group, treated), part in data.groupby(["group", "treated"]):
        row = {"group": group, "receipt_on_anchor": int(treated), "n": len(part)}
        for action, label in [(0, "click"), (1, "purchase")]:
            bits = part[[f"w{w}_a{action}" for w in range(1, 7)]].to_numpy() > 0
            row[label] = summarize_persistence(bits)
            assert sum(row[label]["active_weeks_distribution"]) == len(part)
        # Receipt dates stop on June 15: week 5 is partial and week 6 unknown.
        row["later_receipt_week1to4_n"] = [int(part[f"w{w}_a2"].gt(0).sum()) for w in range(1, 5)]
        row["later_receipt_week5_partial_n"] = int(part["w5_a2"].gt(0).sum())
        row["later_receipt_week6_n"] = None
        result.append(row)
    (PERSIST_OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


LONG_OUT = ROOT / "artifacts/uplift-sequence-pilot"


def run_long_term():
    LONG_OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(ROOT / "artifacts/canonical-type-bounds/tianchi_cohort.csv")
    assert cohort.User_id.is_unique
    users = pd.Index(cohort.User_id)
    # One row per user; action and week counts include later coupon receipts.
    records = []
    source = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    for c in pd.read_csv(source, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        c = c[c.User_id.isin(users)].copy()
        date = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        receipt = c.Action.eq(2)
        date.loc[receipt] = pd.to_datetime(c.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (date - pd.Timestamp("2016-05-15")).dt.days
        mask = delta.between(1, 28) & c.Action.isin([0, 1, 2])
        c = c[mask].copy()
        c["week"] = ((delta[mask] - 1) // 7 + 1).astype(int)
        records.append(c.groupby(["User_id", "week", "Action"]).size())
    wide = pd.concat(records).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"w{week}_a{action}" for week, action in wide.columns]
    frame = cohort.join(wide, on="User_id").fillna(0)
    result = []
    for (group, treated), part in frame.groupby(["group", "treated"]):
        item = {"group": group, "receipt_on_anchor": int(treated), "n": len(part), "weeks": []}
        for week in range(1, 5):
            item["weeks"].append({"week": week, "click_any": float(part[f"w{week}_a0"].gt(0).mean()),
                                   "purchase_any": float(part[f"w{week}_a1"].gt(0).mean()),
                                   "another_receipt_any": float(part[f"w{week}_a2"].gt(0).mean())})
        result.append(item)
    (LONG_OUT / "long_term_observed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


PERSONA_OUT = ROOT / "artifacts/coupon_persona_patterns"
ONLINE = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
TIANCHI_PROXY_OUT = ROOT / "artifacts/tianchi-persona-proxies"
TIANCHI_UPLIFT_OUT = ROOT / "artifacts/tianchi-persona-uplift"
TIANCHI_SINGLE_RECEIPT_OUT = ROOT / "artifacts/tianchi-single-receipt-duration"
TIANCHI_OVERLAP_OUT = ROOT / "artifacts/tianchi-coupon-overlap"
TIANCHI_CUMULATIVE_OUT = ROOT / "artifacts/tianchi-cumulative-coupon-history"
TIANCHI_FIRST_RECEIPT_OUT = ROOT / "artifacts/tianchi-first-receipt-alignment"
TIANCHI_PERSONA_COUPON_OUT = ROOT / "artifacts/tianchi-persona-coupon-matching"
TIANCHI_LONG_TERM_OUT = ROOT / "artifacts/tianchi-long-term-lifecycle"
TIANCHI_CANONICAL_PROXY_OUT = ROOT / "artifacts/tianchi-canonical-proxies"


def nearest_cluster_matches(values, profiles, cluster_ids):
    """Return nearest profile, similarity, and first-versus-second margin."""
    distance = np.sqrt(np.mean((values[:, None, :] - profiles[None, :, :]) ** 2, axis=2))
    similarity = np.exp(-distance)
    order = np.argsort(-similarity, axis=1)
    nearest = order[:, 0]
    return (np.asarray(cluster_ids)[nearest], similarity[np.arange(len(values)), nearest],
            similarity[np.arange(len(values)), nearest] - similarity[np.arange(len(values)), order[:, 1]])


def _propensity_pairs(frame, covariates, caliper_scale=.2):
    """One-to-one nearest propensity-score pairs without replacement."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    treated = frame.treated.to_numpy(dtype=bool)
    if treated.sum() < 2 or (~treated).sum() < 2:
        return np.empty((0, 2), dtype=int), None
    x = StandardScaler().fit_transform(frame[list(covariates)].to_numpy(float))
    propensity = LogisticRegression(max_iter=1000).fit(x, treated).predict_proba(x)[:, 1]
    propensity = np.clip(propensity, 1e-6, 1 - 1e-6)
    logits = np.log(propensity / (1 - propensity))
    caliper = caliper_scale * logits.std(ddof=1)
    pairs = []
    for purchase_history in (0, 1):
        exact = frame.pre_purchase_any.to_numpy() == purchase_history
        treated_index = np.flatnonzero(treated & exact)
        control_index = np.flatnonzero(~treated & exact)
        if not len(treated_index) or not len(control_index):
            continue
        order = np.argsort(logits[control_index])
        control_index = control_index[order]
        control_logits = logits[control_index]
        for row in treated_index[np.argsort(logits[treated_index])]:
            position = np.searchsorted(control_logits, logits[row])
            candidates = []
            if position < len(control_index):
                candidates.append(control_index[position])
            if position:
                candidates.append(control_index[position - 1])
            if not candidates:
                break
            match = min(candidates, key=lambda index: abs(logits[index] - logits[row]))
            if abs(logits[match] - logits[row]) <= caliper:
                pairs.append((row, match))
                keep = control_index != match
                control_index = control_index[keep]
                control_logits = control_logits[keep]
    return np.asarray(pairs, dtype=int).reshape(-1, 2), x


def _paired_binary_summary(frame, pairs, outcome):
    if not len(pairs):
        return None
    treated = frame.iloc[pairs[:, 0]][outcome].to_numpy(float)
    control = frame.iloc[pairs[:, 1]][outcome].to_numpy(float)
    difference = treated - control
    standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 0.
    estimate = float(difference.mean())
    return {
        "treated_rate": float(treated.mean()),
        "matched_control_rate": float(control.mean()),
        "difference": estimate,
        "standard_error": standard_error,
        "ci95": [estimate - 1.96 * standard_error, estimate + 1.96 * standard_error],
    }


def _matched_outcomes(frame, pairs, outcome, group):
    """Keep pair-level outcomes so uncertainty is computed at the matching unit."""
    return pd.DataFrame({
        "group": group,
        "pair": np.arange(len(pairs), dtype=int),
        "control": frame.iloc[pairs[:, 1]][outcome].to_numpy(float),
        "treated": frame.iloc[pairs[:, 0]][outcome].to_numpy(float),
    })


def _bootstrap_pair_rates(pair_frame, draws, rng):
    values = pair_frame[["control", "treated"]].to_numpy(float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(axis=1)
    return sampled[:, 0], sampled[:, 1]


def _canonical_proxy_row(persona, components, weights, draws=5_000, seed=20261026):
    """Transport matched marginals and form one maximum-entropy joint proxy."""
    from analysis.uplift.model import maximum_entropy_types

    rng = np.random.default_rng(seed)
    p0_draws = np.zeros(draws)
    p1_draws = np.zeros(draws)
    p0 = p1 = 0.
    pairs = 0
    for group, weight in weights.items():
        part = components[group]
        control_draws, treated_draws = _bootstrap_pair_rates(part, draws, rng)
        p0_draws += weight * control_draws
        p1_draws += weight * treated_draws
        p0 += weight * part.control.mean()
        p1 += weight * part.treated.mean()
        pairs += len(part)
    canonical = maximum_entropy_types(p0, p1)
    canonical_draws = {
        "sure_thing": p0_draws * p1_draws,
        "persuadable": (1 - p0_draws) * p1_draws,
        "sleeping_dog": p0_draws * (1 - p1_draws),
        "lost_cause": (1 - p0_draws) * (1 - p1_draws),
    }
    row = {
        "persona": persona,
        "matched_pairs_available": pairs,
        "p0": p0,
        "p0_ci_low": float(np.quantile(p0_draws, .025)),
        "p0_ci_high": float(np.quantile(p0_draws, .975)),
        "p1": p1,
        "p1_ci_low": float(np.quantile(p1_draws, .025)),
        "p1_ci_high": float(np.quantile(p1_draws, .975)),
        "difference": p1 - p0,
        "difference_ci_low": float(np.quantile(p1_draws - p0_draws, .025)),
        "difference_ci_high": float(np.quantile(p1_draws - p0_draws, .975)),
    }
    for name, estimate in canonical.items():
        row[name] = estimate
        row[f"{name}_ci_low"] = float(np.quantile(canonical_draws[name], .025))
        row[f"{name}_ci_high"] = float(np.quantile(canonical_draws[name], .975))
    return row


def run_tianchi_canonical_proxy_estimation():
    """Estimate persona-level synthetic canonical proportions from saved matched cohorts."""
    TIANCHI_CANONICAL_PROXY_OUT.mkdir(parents=True, exist_ok=True)

    persona_cohort = pd.read_parquet(TIANCHI_PERSONA_COUPON_OUT / "cohort.parquet")
    profile_covariates = [
        "purchase_share", "log_total_events", "log_active_days", "log_events_per_day",
        "purchase_day_share", "log_median_gap", "first_day_purchase",
        "last_day_purchase", "explore_to_purchase", "purchase_to_explore",
        "explore_persistence", "purchase_persistence", "has_transition",
        "log_pre_receipts",
    ]
    profile_components = {}
    pair_frames = []
    for profile, part in persona_cohort.groupby("shared_profile", sort=True):
        part = part.reset_index(drop=True)
        pairs, _ = _propensity_pairs(part, profile_covariates)
        if len(pairs):
            matched = _matched_outcomes(part, pairs, "post_purchase_any", profile)
            profile_components[profile] = matched
            pair_frames.append(matched.assign(cohort="shared_profile"))

    weights = pd.read_csv(TIANCHI_PERSONA_COUPON_OUT / "persona_transport_weights.csv")
    rows = []
    for persona, group in weights.groupby("persona", sort=True):
        group = group.loc[group.shared_profile.isin(profile_components)].copy()
        group["weight"] /= group.weight.sum()
        rows.append(_canonical_proxy_row(
            persona,
            profile_components,
            dict(zip(group.shared_profile, group.weight, strict=True)),
            seed=20261026 + len(rows),
        ))

    p_lt = pd.read_parquet(TIANCHI_LONG_TERM_OUT / "anchor_p_lt_cohort.parquet")
    p_lt_covariates = [
        "log_active_days", "log_purchase_days", "log_purchases", "log_events",
        "log_span_days", "purchase_day_share", "log_prior_receipts",
    ]
    p_lt_pairs, _ = _propensity_pairs(p_lt, p_lt_covariates)
    p_lt_matched = _matched_outcomes(p_lt, p_lt_pairs, "post_purchase_any", "P_LT")
    pair_frames.append(p_lt_matched.assign(cohort="long_term_buyer"))
    rows.append(_canonical_proxy_row(
        "P_LT", {"P_LT": p_lt_matched}, {"P_LT": 1.}, seed=20261030,
    ))

    result = pd.DataFrame(rows)
    result.to_csv(TIANCHI_CANONICAL_PROXY_OUT / "canonical_type_proxies.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_parquet(
        TIANCHI_CANONICAL_PROXY_OUT / "matched_pair_outcomes.parquet", index=False
    )
    rules = {
        "estimand": "14-day purchase after the 2016-05-15 anchor",
        "treatment_proxy": "at least one Tianchi Action=2 coupon receipt on the anchor day",
        "outcome": "at least one Tianchi Action=1 purchase during anchor days 1 through 14",
        "shared_proxy_assignment": {
            "window": "click and purchase records in anchor days -28 through -1",
            "features": profile_covariates[:-1],
            "rule": (
                "robust-scale against Cosmetics and Electronics shared behavior features, then "
                "assign the nearest Cosmetics K=4 centroid; C0/C2/C3 map to S0/S1/S2 and "
                "unresolved C1 is excluded"
            ),
        },
        "persona_transport": (
            "estimate p0 and p1 within S0/S1/S2, then average them with the empirical "
            "Cosmetics P0/P2/P3-to-shared-profile overlap weights"
        ),
        "long_term_proxy": (
            "P_LT is assigned before the anchor when a user has at least 10 active days, "
            "3 purchase days, and a 30-day observed span"
        ),
        "matching": (
            "within proxy group, 1:1 logistic propensity matching without replacement, "
            "exact on prior purchase and within a 0.2-SD propensity-logit caliper"
        ),
        "canonical_joint": (
            "maximum-entropy independent coupling of matched marginals p0 and p1; "
            "pair bootstrap intervals do not identify individual potential outcomes"
        ),
        "bootstrap_draws": 5000,
        "bootstrap_seed": 20261026,
    }
    report = {"rules": rules, "results": result.to_dict(orient="records")}
    (TIANCHI_CANONICAL_PROXY_OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_tianchi_persona_coupon_matching():
    """Match anchor-day receipt proxies inside cross-domain behavioral profiles."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler

    from analysis.persona.behavior import SHARED_LIFECYCLE_PROFILE
    from analysis.persona.cluster import (
        ALIGNMENT_FEATURES, _fixed_k4_membership, _selected_events,
        _shared_behavior_features,
    )
    from analysis.uplift.model import maximum_entropy_types

    TIANCHI_PERSONA_COUPON_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    base = pd.read_csv(ROOT / "artifacts/canonical-type-bounds/tianchi_cohort.csv")
    base["pre_total"] = base.pre_0 + base.pre_1
    base = base.loc[base.pre_total.ge(2)].copy()
    treated = base.loc[base.treated.eq(1)]
    controls = base.loc[base.treated.eq(0)].copy()
    controls["selection_hash"] = pd.util.hash_pandas_object(
        "coupon-control:" + controls.User_id.astype(str), index=False
    ).to_numpy()
    controls = controls.nsmallest(min(50_000, len(controls)), "selection_hash")
    selected = pd.concat([treated, controls], ignore_index=True)
    users = set(selected.User_id.astype("int64"))

    pre_parts = []
    for chunk in pd.read_csv(
        ONLINE, usecols=["User_id", "Action", "Date"],
        dtype={"User_id": "Int64", "Action": "Int8", "Date": "string"},
        chunksize=1_000_000,
    ):
        chunk = chunk.loc[chunk.User_id.isin(users) & chunk.Action.isin([0, 1])].copy()
        if chunk.empty:
            continue
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        keep = (day - anchor).dt.days.between(-28, -1)
        if keep.any():
            pre_parts.append(pd.DataFrame({
                "user": chunk.loc[keep, "User_id"].astype("int64"),
                "at": day[keep],
                "action": chunk.loc[keep, "Action"].map({0: "explore", 1: "purchase"}),
            }).dropna())
    pre_events = pd.concat(pre_parts, ignore_index=True)
    membership = pd.DataFrame({"user_id": sorted(pre_events.user.unique()), "cluster": 0})
    pre_features = _shared_behavior_features(pre_events, membership)

    reference_frames = []
    for source in ("cosmetics", "electronics"):
        source_membership, _ = _fixed_k4_membership(source)
        events = _selected_events(source, set(source_membership.user_id.astype("int64")))
        events = events.loc[events.action.isin(["view", "purchase"]), ["user_id", "at", "action"]]
        events = events.rename(columns={"user_id": "user"})
        events["action"] = events.action.replace({"view": "explore"})
        end = events.groupby("user")["at"].transform("max")
        events = events.loc[events["at"].gt(end - pd.Timedelta(days=28))]
        reference_frames.append(_shared_behavior_features(events, source_membership).assign(dataset=source))
    reference = pd.concat(reference_frames, ignore_index=True)
    scaler = RobustScaler().fit(reference[list(ALIGNMENT_FEATURES)])
    reference[list(ALIGNMENT_FEATURES)] = scaler.transform(reference[list(ALIGNMENT_FEATURES)])
    tianchi_scaled = scaler.transform(pre_features[list(ALIGNMENT_FEATURES)])
    cosmetics_profiles = (reference.loc[reference.dataset.eq("cosmetics")]
                           .groupby("cluster")[list(ALIGNMENT_FEATURES)].median().sort_index())
    cluster, similarity, margin = nearest_cluster_matches(
        tianchi_scaled, cosmetics_profiles.to_numpy(), cosmetics_profiles.index.to_numpy()
    )
    assigned = pre_features[["user_id", *ALIGNMENT_FEATURES]].copy()
    assigned["cosmetics_cluster"] = cluster
    assigned["shared_profile"] = assigned.cosmetics_cluster.map(SHARED_LIFECYCLE_PROFILE["cosmetics"])
    assigned["profile_similarity"] = similarity
    assigned["profile_margin"] = margin

    frame = selected.merge(assigned, left_on="User_id", right_on="user_id", validate="one_to_one")
    frame["pre_purchase_any"] = frame.pre_1.gt(0).astype(int)
    frame["post_purchase_any"] = frame.post_1.gt(0).astype(int)
    frame["post_activity_any"] = (frame.post_0 + frame.post_1).gt(0).astype(int)
    frame["later_receipt_any"] = frame.post_2.gt(0).astype(int)
    frame["log_pre_receipts"] = np.log1p(frame.pre_2)
    assignment_counts = (frame.loc[frame.treated.eq(1), "shared_profile"]
                          .value_counts(dropna=False).sort_index().to_dict())
    frame = frame.loc[frame.shared_profile.ne("UNRESOLVED")].reset_index(drop=True)

    covariates = [*ALIGNMENT_FEATURES, "log_pre_receipts"]
    shared_rows = []
    for profile, part in frame.groupby("shared_profile", sort=True):
        part = part.reset_index(drop=True)
        pairs, standardized = _propensity_pairs(part, covariates)
        if not len(pairs):
            continue
        treated_x = standardized[pairs[:, 0]]
        control_x = standardized[pairs[:, 1]]
        raw_treated = standardized[part.treated.to_numpy(dtype=bool)]
        raw_control = standardized[~part.treated.to_numpy(dtype=bool)]
        raw_pooled = np.sqrt((raw_treated.var(axis=0, ddof=1) + raw_control.var(axis=0, ddof=1)) / 2)
        raw_smd = np.divide(raw_treated.mean(axis=0) - raw_control.mean(axis=0), raw_pooled,
                            out=np.zeros_like(raw_pooled), where=raw_pooled > 0)
        pooled = np.sqrt((treated_x.var(axis=0, ddof=1) + control_x.var(axis=0, ddof=1)) / 2)
        smd = np.divide(treated_x.mean(axis=0) - control_x.mean(axis=0), pooled,
                        out=np.zeros_like(pooled), where=pooled > 0)
        purchase = _paired_binary_summary(part, pairs, "post_purchase_any")
        activity = _paired_binary_summary(part, pairs, "post_activity_any")
        later_receipt = _paired_binary_summary(part, pairs, "later_receipt_any")
        shared_rows.append({
            "shared_profile": profile,
            "eligible_treated": int(part.treated.sum()),
            "eligible_controls": int(part.treated.eq(0).sum()),
            "matched_pairs": int(len(pairs)),
            "matched_fraction": float(len(pairs) / part.treated.sum()),
            "median_profile_similarity": float(part.loc[part.treated.eq(1), "profile_similarity"].median()),
            "max_abs_unmatched_smd": float(np.abs(raw_smd).max()),
            "max_abs_matched_smd": float(np.abs(smd).max()),
            "purchase": purchase,
            "activity": activity,
            "later_receipt_diagnostic": later_receipt,
        })
    shared = pd.DataFrame([{
        "shared_profile": row["shared_profile"],
        "eligible_treated": row["eligible_treated"],
        "eligible_controls": row["eligible_controls"],
        "matched_pairs": row["matched_pairs"],
        "matched_fraction": row["matched_fraction"],
        "median_profile_similarity": row["median_profile_similarity"],
        "max_abs_unmatched_smd": row["max_abs_unmatched_smd"],
        "max_abs_matched_smd": row["max_abs_matched_smd"],
        **{f"purchase_{key}": value for key, value in row["purchase"].items() if key != "ci95"},
        "purchase_ci95_low": row["purchase"]["ci95"][0],
        "purchase_ci95_high": row["purchase"]["ci95"][1],
        "activity_difference": row["activity"]["difference"],
        "later_receipt_difference": row["later_receipt_diagnostic"]["difference"],
    } for row in shared_rows])

    old = pd.read_parquet(ROOT / "artifacts/four-behavior-modes/cosmetics_embedding_memberships.parquet")
    fixed, _ = _fixed_k4_membership("cosmetics")
    overlap = old[["user_id", "mode"]].merge(fixed[["user_id", "cluster"]], on="user_id")
    overlap["shared_profile"] = overlap.cluster.map(SHARED_LIFECYCLE_PROFILE["cosmetics"])
    persona_modes = {"P0": "P4_purchase_path", "P2": "P2_deep_consideration", "P3": "P3_cart_revision"}
    weights = []
    for persona, mode in persona_modes.items():
        values = overlap.loc[(overlap["mode"].eq(mode)) & overlap.shared_profile.ne("UNRESOLVED"), "shared_profile"]
        for profile, weight in values.value_counts(normalize=True).sort_index().items():
            weights.append({"persona": persona, "shared_profile": profile, "weight": float(weight),
                            "overlap_users": int(values.eq(profile).sum())})
    weights = pd.DataFrame(weights)
    effects = shared.set_index("shared_profile")
    persona_rows = []
    for persona, group in weights.groupby("persona", sort=True):
        group = group.loc[group.shared_profile.isin(effects.index)].copy()
        group["weight"] /= group.weight.sum()
        w = group.weight.to_numpy()
        rows = effects.loc[group.shared_profile]
        p0 = float(w @ rows.purchase_matched_control_rate.to_numpy())
        p1 = float(w @ rows.purchase_treated_rate.to_numpy())
        effect = p1 - p0
        se = float(np.sqrt(np.sum(np.square(w * rows.purchase_standard_error.to_numpy()))))
        persona_rows.append({
            "persona": persona, "matched_control_purchase": p0,
            "receipt_purchase": p1, "difference": effect,
            "standard_error": se, "ci95_low": effect - 1.96 * se,
            "ci95_high": effect + 1.96 * se,
            "maximum_entropy_canonical_proxy": maximum_entropy_types(p0, p1),
        })
    persona_rows.append({**next(row for row in persona_rows if row["persona"] == "P2"),
                         "persona": "P_LT", "inherited_from": "P2"})
    persona = pd.DataFrame([{key: value for key, value in row.items()
                             if key != "maximum_entropy_canonical_proxy"} for row in persona_rows])

    frame.to_parquet(TIANCHI_PERSONA_COUPON_OUT / "cohort.parquet", index=False)
    shared.to_csv(TIANCHI_PERSONA_COUPON_OUT / "shared_profile_effects.csv", index=False)
    weights.to_csv(TIANCHI_PERSONA_COUPON_OUT / "persona_transport_weights.csv", index=False)
    persona.to_csv(TIANCHI_PERSONA_COUPON_OUT / "persona_coupon_proxies.csv", index=False)
    report = {
        "dataset": "Tianchi O2O online",
        "anchor_day": "2016-05-15",
        "treatment_proxy": "at least one Action=2 coupon-receipt record on the anchor day",
        "outcome": "at least one Action=1 purchase record in days 1 through 14",
        "eligibility": "at least two click/purchase records before the anchor; all treated and a deterministic 50,000-user control sample",
        "eligible_treated_total": int(len(treated)),
        "treated_profile_assignment": {str(key): int(value) for key, value in assignment_counts.items()},
        "profile_assignment": "28-day pre-anchor click/purchase behavior matched to Cosmetics K=4 medians; C0/C2/C3 become S0/S1/S2 and C1 is unresolved",
        "matching": "within-profile regularized-logistic propensity matching without replacement, exact on prior purchase and a 0.2-SD logit caliper",
        "shared_profile_results": shared_rows,
        "persona_transport_weights": weights.to_dict(orient="records"),
        "persona_results": persona_rows,
        "limitations": [
            "Receipt is observed, but offer eligibility and failed delivery are not; the matched difference is not an identified causal effect.",
            "Controls can receive coupons after the anchor, and the later-receipt contrast is reported as a targeting diagnostic.",
            "Tianchi click is aligned to retail view; cart, removal, product identity, and within-day event order are unavailable.",
            "P0 and P3 both mainly transport through S1, so Tianchi cannot sharply separate their coupon response.",
            "P_LT inherits P2 because Tianchi does not identify the rare long-term-buyer subtype.",
        ],
    }
    (TIANCHI_PERSONA_COUPON_OUT / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    figure_dir = ROOT / "img/uplift"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    x = np.arange(len(shared))
    axes[0].bar(x - .18, shared.purchase_matched_control_rate * 100, .36,
                label="matched no-receipt", color="#4e79a7")
    axes[0].bar(x + .18, shared.purchase_treated_rate * 100, .36,
                label="receipt", color="#f28e2b")
    axes[0].set(xticks=x, xticklabels=shared.shared_profile,
                ylabel="Users with purchase in days 1–14 (%)", title="Shared behavior profiles")
    axes[0].legend(frameon=False)
    px = np.arange(len(persona))
    estimate = persona.difference.to_numpy() * 100
    lower = persona.ci95_low.to_numpy() * 100
    upper = persona.ci95_high.to_numpy() * 100
    axes[1].bar(px, estimate, color="#59a14f")
    axes[1].errorbar(px, estimate, yerr=np.vstack([estimate - lower, upper - estimate]),
                     fmt="none", color="black", capsize=3)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xticks=px, xticklabels=persona.persona,
                ylabel="Matched purchase difference (percentage points)",
                title="Transported simulator-persona proxies")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi coupon-receipt proxy matched through cross-domain behavior", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "tianchi_persona_coupon_matching.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return report


def run_tianchi_long_term_lifecycle():
    """Check Tianchi analogues of P_LT and purchase-followed-by-nonreturn."""
    import matplotlib.pyplot as plt

    from analysis.uplift.canonical import bounds
    from analysis.uplift.model import maximum_entropy_types

    TIANCHI_LONG_TERM_OUT.mkdir(parents=True, exist_ok=True)
    behavior_parts, receipt_parts = [], []
    for chunk in pd.read_csv(
        ONLINE,
        usecols=["User_id", "Action", "Coupon_id", "Date", "Date_received"],
        dtype={"User_id": "Int64", "Action": "Int8", "Coupon_id": "string",
               "Date": "string", "Date_received": "string"},
        chunksize=1_000_000,
    ):
        behavior = chunk.loc[chunk.Action.isin([0, 1]) & chunk.User_id.notna()].copy()
        if not behavior.empty:
            behavior["day"] = pd.to_datetime(behavior.Date, format="%Y%m%d", errors="coerce")
            behavior["purchase"] = behavior.Action.eq(1).astype(int)
            behavior["coupon_purchase"] = (
                behavior.Action.eq(1) & behavior.Coupon_id.notna()
            ).astype(int)
            behavior_parts.append(
                behavior.dropna(subset=["day"])
                .groupby(["User_id", "day"])
                .agg(events=("Action", "size"), purchases=("purchase", "sum"),
                     coupon_purchases=("coupon_purchase", "sum"))
            )
        receipt = chunk.loc[chunk.Action.eq(2) & chunk.User_id.notna()].copy()
        if not receipt.empty:
            receipt["day"] = pd.to_datetime(
                receipt.Date_received, format="%Y%m%d", errors="coerce"
            )
            receipt_parts.append(
                receipt.dropna(subset=["day"]).groupby(["User_id", "day"]).size()
                .rename("receipts")
            )
    daily = pd.concat(behavior_parts).groupby(level=[0, 1]).sum().reset_index()
    receipts = pd.concat(receipt_parts).groupby(level=[0, 1]).sum().reset_index()
    daily = daily.sort_values(["User_id", "day"], kind="stable")
    observation_end = daily.day.max()
    anchor = pd.Timestamp("2016-05-15")

    def user_stats(frame):
        source = frame.assign(purchase_day=frame.purchases.gt(0).astype(int))
        result = source.groupby("User_id").agg(
            active_days=("day", "size"), purchase_days=("purchase_day", "sum"),
            purchases=("purchases", "sum"), events=("events", "sum"),
            first_day=("day", "min"), last_day=("day", "max"),
        )
        result["span_days"] = (result.last_day - result.first_day).dt.days
        return result

    pre = user_stats(daily.loc[daily.day.lt(anchor)])
    pre["p_lt"] = (
        pre.active_days.ge(10) & pre.purchase_days.ge(3) & pre.span_days.ge(30)
    )
    pre_receipts = (receipts.loc[receipts.day.lt(anchor)].groupby("User_id").receipts.sum())
    anchor_receipt = set(receipts.loc[receipts.day.eq(anchor), "User_id"].astype("int64"))
    post = daily.loc[daily.day.between(anchor + pd.Timedelta(days=1),
                                       anchor + pd.Timedelta(days=14))]
    post_purchase = post.groupby("User_id").purchases.sum().gt(0)
    later_receipt = (receipts.loc[receipts.day.between(anchor + pd.Timedelta(days=1),
                                                        anchor + pd.Timedelta(days=14))]
                     .groupby("User_id").receipts.sum().gt(0))
    anchor_frame = pre.loc[pre.p_lt].reset_index()
    anchor_frame["treated"] = anchor_frame.User_id.isin(anchor_receipt).astype(int)
    anchor_frame["pre_purchase_any"] = 1
    anchor_frame["post_purchase_any"] = (
        anchor_frame.User_id.map(post_purchase).eq(True).astype(int)
    )
    anchor_frame["later_receipt_any"] = (
        anchor_frame.User_id.map(later_receipt).eq(True).astype(int)
    )
    anchor_frame["purchase_day_share"] = anchor_frame.purchase_days / anchor_frame.active_days
    anchor_frame["log_active_days"] = np.log1p(anchor_frame.active_days)
    anchor_frame["log_purchase_days"] = np.log1p(anchor_frame.purchase_days)
    anchor_frame["log_purchases"] = np.log1p(anchor_frame.purchases)
    anchor_frame["log_events"] = np.log1p(anchor_frame.events)
    anchor_frame["log_span_days"] = np.log1p(anchor_frame.span_days)
    anchor_frame["log_prior_receipts"] = np.log1p(
        anchor_frame.User_id.map(pre_receipts).fillna(0)
    )
    covariates = [
        "log_active_days", "log_purchase_days", "log_purchases", "log_events",
        "log_span_days", "purchase_day_share", "log_prior_receipts",
    ]
    pairs, standardized = _propensity_pairs(anchor_frame, covariates)
    purchase_match = _paired_binary_summary(anchor_frame, pairs, "post_purchase_any")
    later_receipt_match = _paired_binary_summary(anchor_frame, pairs, "later_receipt_any")
    p0 = purchase_match["matched_control_rate"]
    p1 = purchase_match["treated_rate"]
    canonical_proxy = maximum_entropy_types(p0, p1)
    canonical_bounds = bounds(p0, p1)
    treated_x, control_x = standardized[pairs[:, 0]], standardized[pairs[:, 1]]
    pooled = np.sqrt((treated_x.var(axis=0, ddof=1) + control_x.var(axis=0, ddof=1)) / 2)
    smd = np.divide(treated_x.mean(axis=0) - control_x.mean(axis=0), pooled,
                    out=np.zeros_like(pooled), where=pooled > 0)

    all_users = user_stats(daily)
    final_day = daily.groupby("User_id", sort=False).tail(1).set_index("User_id")
    penultimate_day = (daily.groupby("User_id", sort=False).tail(2)
                       .groupby("User_id", sort=False).head(1).set_index("User_id").day)
    all_users["final_purchase"] = final_day.purchases.gt(0)
    all_users["final_coupon_purchase"] = final_day.coupon_purchases.gt(0)
    all_users["prior_active_days"] = all_users.active_days - 1
    all_users["prior_purchase_days"] = (
        all_users.purchase_days - all_users.final_purchase.astype(int)
    )
    all_users["prior_span_days"] = (
        penultimate_day.reindex(all_users.index) - all_users.first_day
    ).dt.days.fillna(0)
    all_users["p_lt"] = (
        all_users.active_days.ge(10) & all_users.purchase_days.ge(3)
        & all_users.span_days.ge(30)
    )
    all_users["established_p_lt_before_final"] = (
        all_users.prior_active_days.ge(10) & all_users.prior_purchase_days.ge(3)
        & all_users.prior_span_days.ge(30)
    )
    all_users["full_30d_followup"] = (
        observation_end - all_users.last_day
    ).dt.days.ge(30)
    all_users["terminal_state"] = np.select(
        [all_users.full_30d_followup & all_users.final_purchase,
         all_users.full_30d_followup & ~all_users.final_purchase],
        ["SATIATED_EXIT_CANDIDATE", "DISSATISFIED_EXIT_CANDIDATE"],
        default="RIGHT_CENSORED",
    )
    receipt_link = receipts.merge(
        all_users[["last_day"]], left_on="User_id", right_index=True, how="inner"
    )
    prior_14 = (receipt_link.loc[
        receipt_link.day.between(receipt_link.last_day - pd.Timedelta(days=14),
                                 receipt_link.last_day)
    ].groupby("User_id").receipts.sum().gt(0))
    after_final = (receipt_link.loc[receipt_link.day.gt(receipt_link.last_day)]
                   .groupby("User_id").receipts.sum().gt(0))
    all_users["receipt_14d_before_final"] = (
        all_users.index.map(prior_14).fillna(False).astype(bool)
    )
    all_users["receipt_after_final"] = (
        all_users.index.map(after_final).fillna(False).astype(bool)
    )

    established = all_users.loc[all_users.established_p_lt_before_final].copy()
    terminal_rows = []
    for state, part in established.groupby("terminal_state", sort=True):
        terminal_rows.append({
            "terminal_state": state, "users": int(len(part)),
            "share": float(len(part) / len(established)),
            "median_active_days": float(part.active_days.median()),
            "median_purchase_days": float(part.purchase_days.median()),
            "median_span_days": float(part.span_days.median()),
            "final_coupon_purchase_rate": float(part.final_coupon_purchase.mean()),
            "receipt_14d_before_final_rate": float(part.receipt_14d_before_final.mean()),
            "receipt_after_final_rate": float(part.receipt_after_final.mean()),
        })
    terminal = pd.DataFrame(terminal_rows)
    eligible_established = established.loc[established.full_30d_followup]
    satiated = eligible_established.terminal_state.eq("SATIATED_EXIT_CANDIDATE")
    eligible_all = all_users.loc[all_users.full_30d_followup]
    all_satiated_share = float(
        eligible_all.terminal_state.eq("SATIATED_EXIT_CANDIDATE").mean()
    )
    full_p_lt = all_users.loc[all_users.p_lt]

    anchor_frame.to_parquet(TIANCHI_LONG_TERM_OUT / "anchor_p_lt_cohort.parquet", index=False)
    established.reset_index().to_parquet(
        TIANCHI_LONG_TERM_OUT / "established_long_term_terminal_users.parquet", index=False
    )
    terminal.to_csv(TIANCHI_LONG_TERM_OUT / "terminal_profiles.csv", index=False)
    pd.DataFrame([{
        "persona": "P_LT", "p0": p0, "p1": p1,
        **canonical_proxy,
    }]).to_csv(TIANCHI_LONG_TERM_OUT / "canonical_type_proxy.csv", index=False)
    report = {
        "dataset": "Tianchi O2O online",
        "observation_window": [str(daily.day.min().date()), str(observation_end.date())],
        "p_lt_definition": "at least 10 active days, 3 purchase days, and 30 observed days",
        "anchor_coupon_proxy": {
            "anchor_day": str(anchor.date()), "p_lt_users": int(len(anchor_frame)),
            "treated_users": int(anchor_frame.treated.sum()), "matched_pairs": int(len(pairs)),
            "max_abs_matched_smd": float(np.abs(smd).max()),
            "purchase_1to14": purchase_match,
            "later_receipt_1to14_diagnostic": later_receipt_match,
            "canonical_types": {
                "definition": {
                    "sure_thing": "Y(0)=1, Y(1)=1",
                    "persuadable": "Y(0)=0, Y(1)=1",
                    "sleeping_dog": "Y(0)=1, Y(1)=0",
                    "lost_cause": "Y(0)=0, Y(1)=0",
                },
                "maximum_entropy_independent_coupling": canonical_proxy,
                "frechet_bounds": canonical_bounds,
                "interpretation": "synthetic joint distribution preserving matched p0 and p1; not identified individual-type proportions",
            },
        },
        "whole_log_p_lt_users": int(len(full_p_lt)),
        "whole_log_p_lt_share": float(len(full_p_lt) / len(all_users)),
        "established_before_final_users": int(len(established)),
        "terminal_profiles": terminal_rows,
        "established_with_full_followup": int(len(eligible_established)),
        "satiated_exit_candidates": int(satiated.sum()),
        "satiated_share_among_established_full_followup": float(satiated.mean()),
        "satiated_share_among_all_full_followup": all_satiated_share,
        "satiated_share_ratio_established_vs_all": float(satiated.mean() / all_satiated_share),
        "terminal_cohort_canonical_types": (
            "not estimated because terminal cohorts are defined by post-anchor purchase and return; "
            "conditioning on them would be post-treatment selection"
        ),
        "limitations": [
            "An active day is used as the Tianchi analogue of a retail session because within-day order and session boundaries are unavailable.",
            "SATIATED_EXIT_CANDIDATE means final-day purchase followed by 30 observed days without click or purchase; satisfaction and permanent churn are not observed.",
            "Anchor-day coupon receipt is observational; offer eligibility, failed delivery, and targeting rules are unavailable.",
            "A receipt or coupon-linked purchase on the final day has unknown within-day ordering.",
        ],
    }
    (TIANCHI_LONG_TERM_OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    figure_dir = ROOT / "img/uplift"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(["matched\nno receipt", "anchor-day\nreceipt"],
                [purchase_match["matched_control_rate"] * 100,
                 purchase_match["treated_rate"] * 100],
                color=["#4e79a7", "#f28e2b"])
    axes[0].set(ylabel="14-day purchase rate (%)",
                title=f"P_LT anchor-day coupon proxy (n={len(pairs):,} pairs)")
    values = np.array([satiated.mean(), 1 - satiated.mean()]) * 100
    axes[1].bar(["satiated\nexit candidate", "dissatisfied\nexit candidate"],
                values, color=["#59a14f", "#e15759"])
    axes[1].set(ylabel="Share of fully observed P_LT users (%)",
                title=f"Observed terminal proxy (n={len(eligible_established):,})", ylim=(0, 100))
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi long-term buyer and purchase-followed-by-nonreturn proxies", y=1.02)
    fig.savefig(figure_dir / "tianchi_long_term_lifecycle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return report


def coupon_overlap_band(distinct_coupon_count):
    """Map distinct receipts in an exposure window to 0, 1, or 2+ dose."""
    return np.minimum(np.asarray(distinct_coupon_count, dtype=int), 2)


def _history_bins(frame):
    """Small, pre-exposure-only strata used for descriptive standardization."""
    click = np.select([frame.pre_click.eq(0), frame.pre_click.eq(1)], ["0", "1"], default="2+")
    purchase = np.where(frame.pre_purchase.gt(0), "1+", "0")
    receipt = np.where(frame.pre_receipt.gt(0), "1+", "0")
    return (pd.Series(click, index=frame.index, dtype="string") + " click / " +
            pd.Series(purchase, index=frame.index, dtype="string") + " purchase / " +
            pd.Series(receipt, index=frame.index, dtype="string") + " prior receipt")


def run_tianchi_coupon_overlap():
    """Describe the association of one-week multiple-coupon receipt with next-week purchase.

    Tianchi records coupon receipt, not coupon expiry or redeemed inventory.  Therefore
    ``2+`` means two or more *distinct coupon IDs received in one calendar week*, not
    verified simultaneously redeemable coupons.
    """
    import matplotlib.pyplot as plt

    TIANCHI_OVERLAP_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    periods = ["pre", "exposure", "outcome"]
    counts, exposure_ids = [], []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Coupon_id", "Date", "Date_received"],
                             dtype={"Coupon_id": "string", "Date": "string", "Date_received": "string"},
                             chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.select([delta.between(-14, -1), delta.between(1, 7), delta.between(8, 14)], periods, default="outside")
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        counts.append(data.groupby(["User_id", "period", "Action"]).size())
        overlap = chunk.loc[receipt & delta.between(1, 7), ["User_id", "Coupon_id"]].dropna()
        if not overlap.empty:
            exposure_ids.append(overlap)

    wide = pd.concat(counts).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in periods:
        for action in (0, 1, 2):
            name = f"{period}_{action}"
            if name not in wide:
                wide[name] = 0
    ids = pd.concat(exposure_ids, ignore_index=True).drop_duplicates(["User_id", "Coupon_id"])
    distinct = ids.groupby("User_id").size().rename("distinct_coupon_ids")
    frame = wide.join(distinct, how="outer").fillna(0).reset_index()
    for name in wide.columns:
        frame[name] = frame[name].astype(int)
    frame["distinct_coupon_ids"] = frame.distinct_coupon_ids.astype(int)
    frame["pre_click"], frame["pre_purchase"], frame["pre_receipt"] = frame.pre_0, frame.pre_1, frame.pre_2
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["receipt_events_exposure"] = frame.exposure_2
    frame["overlap_band"] = coupon_overlap_band(frame.distinct_coupon_ids)
    frame["overlap_label"] = pd.Categorical(np.where(frame.overlap_band.eq(2), "2+", frame.overlap_band.astype(str)),
                                             categories=["0", "1", "2+"], ordered=True)
    frame["outcome_purchase_any"] = frame.outcome_1.gt(0).astype(int)
    frame["outcome_any"] = (frame.outcome_0 + frame.outcome_1).gt(0).astype(int)
    frame["outcome_receipt_any"] = frame.outcome_2.gt(0).astype(int)
    frame = frame.loc[frame.pre_total.gt(0)].copy()
    frame["stratum"] = _history_bins(frame)

    common = [name for name, part in frame.groupby("stratum", observed=True)
              if set(part.overlap_band.unique()) == {0, 1, 2}]
    frame = frame.loc[frame.stratum.isin(common)].copy()
    reference_counts = frame.loc[frame.overlap_band.eq(1), "stratum"].value_counts()
    weights = reference_counts / reference_counts.sum()
    results = []
    for band, label in [(0, "0"), (1, "1"), (2, "2+")]:
        part = frame.loc[frame.overlap_band.eq(band)]
        table = part.groupby("stratum").agg(purchase=("outcome_purchase_any", "mean"),
                                             purchase_n=("outcome_purchase_any", "count"),
                                             any_action=("outcome_any", "mean"),
                                             receipt=("outcome_receipt_any", "mean"))
        table = table.loc[weights.index]
        purchase = float((table.purchase * weights).sum())
        purchase_se = float(np.sqrt((weights ** 2 * table.purchase * (1 - table.purchase) / table.purchase_n).sum()))
        results.append({"overlap_band": label, "n": int(len(part)),
                        "distinct_coupon_ids_mean": float(part.distinct_coupon_ids.mean()),
                        "receipt_events_exposure_mean": float(part.receipt_events_exposure.mean()),
                        "standardized_purchase_8to14": purchase,
                        "standardized_purchase_ci95": [purchase - 1.96 * purchase_se, purchase + 1.96 * purchase_se],
                        "standardized_any_action_8to14": float((table.any_action * weights).sum()),
                        "outcome_receipt_rate_8to14": float((table.receipt * weights).sum())})
    baseline, one, multiple = results
    for item, reference, name in [(one, baseline, "difference_vs_0"), (multiple, one, "increment_2plus_vs_1")]:
        difference = item["standardized_purchase_8to14"] - reference["standardized_purchase_8to14"]
        item[name] = difference
        item[f"{name}_ci95"] = [
            difference - 1.96 * np.sqrt(((item["standardized_purchase_ci95"][1] - item["standardized_purchase_ci95"][0]) / 3.92) ** 2 +
                                         ((reference["standardized_purchase_ci95"][1] - reference["standardized_purchase_ci95"][0]) / 3.92) ** 2),
            difference + 1.96 * np.sqrt(((item["standardized_purchase_ci95"][1] - item["standardized_purchase_ci95"][0]) / 3.92) ** 2 +
                                         ((reference["standardized_purchase_ci95"][1] - reference["standardized_purchase_ci95"][0]) / 3.92) ** 2),
        ]
    keep = ["User_id", "stratum", "pre_click", "pre_purchase", "pre_receipt", "pre_total", "distinct_coupon_ids",
            "receipt_events_exposure", "overlap_label", "outcome_purchase_any", "outcome_any", "outcome_receipt_any"]
    frame[keep].to_parquet(TIANCHI_OVERLAP_OUT / "cohort.parquet", index=False)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "exposure_window": "days 1-7: number of distinct Coupon_id values with Action=2",
              "outcome_window": "days 8-14: any purchase record (Action=1)",
              "overlap_definition": "0, 1, or 2+ distinct coupon IDs received in the exposure window; active/expiry state is unavailable",
              "eligibility": "at least one pre-anchor click or purchase; common support standardized over pre click, purchase, and receipt-history strata",
              "n_common_support": int(len(frame)), "stratum_weights_from_one_coupon_arm": {str(k): float(v) for k, v in weights.items()},
              "results": results,
              "caveat": "This is an observational dose-response description. Higher multiple-receipt rates and outcome-window receipt rates can reflect continuing targeting, user demand, or unobserved offer rules; it is not a causal coupon-stacking estimate."}
    (TIANCHI_OVERLAP_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["overlap_band"] for row in results]
    x = np.arange(len(results))
    rate = np.array([row["standardized_purchase_8to14"] for row in results]) * 100
    ci = np.array([row["standardized_purchase_ci95"] for row in results]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x, rate, color=["#4e79a7", "#f28e2b", "#59a14f"], width=.62)
    axes[0].errorbar(x, rate, yerr=np.vstack([rate - ci[:, 0], ci[:, 1] - rate]), fmt="none", color="black", capsize=4)
    axes[0].set(xticks=x, xticklabels=labels, xlabel="Distinct coupon IDs received in days 1–7", ylabel="Purchase record in days 8–14 (%)",
                title="Standardized next-week purchase rate")
    later = [row["outcome_receipt_rate_8to14"] * 100 for row in results]
    axes[1].plot(x, later, marker="o", linewidth=2, color="#b07aa1")
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Distinct coupon IDs received in days 1–7", ylabel="Received another coupon in days 8–14 (%)",
                title="Continued coupon targeting diagnostic")
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi multiple-coupon receipt: descriptive overlap proxy", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_coupon_overlap.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_cumulative_coupon_history():
    """Describe next-week purchase by coupon receipts accumulated before an anchor."""
    import matplotlib.pyplot as plt

    TIANCHI_CUMULATIVE_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.where(delta.lt(0), "history", np.where(delta.between(1, 14), "outcome", "outside"))
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        pieces.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in ("history", "outcome"):
        for action in (0, 1, 2):
            name = f"{period}_{action}"
            if name not in wide:
                wide[name] = 0
    frame = wide.reset_index()
    frame["pre_click"], frame["pre_purchase"], frame["cumulative_receipts"] = frame.history_0, frame.history_1, frame.history_2
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame = frame.loc[frame.pre_total.gt(0)].copy()
    frame["coupon_history_band"] = pd.cut(frame.cumulative_receipts, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3–4", "5+"])
    frame["outcome_purchase_any"] = frame.outcome_1.gt(0).astype(int)
    frame["outcome_receipt_any"] = frame.outcome_2.gt(0).astype(int)
    click_bin = np.select([frame.pre_click.eq(0), frame.pre_click.eq(1)], ["0", "1"], default="2+")
    purchase_bin = np.where(frame.pre_purchase.gt(0), "1+", "0")
    frame["stratum"] = pd.Series(click_bin, index=frame.index, dtype="string") + " click / " + pd.Series(purchase_bin, index=frame.index, dtype="string") + " purchase"
    bands = ["0", "1", "2", "3–4", "5+"]
    common = [key for key, part in frame.groupby("stratum", observed=True) if set(part.coupon_history_band.dropna()) == set(bands)]
    frame = frame.loc[frame.stratum.isin(common)].copy()
    weights = frame.loc[frame.coupon_history_band.eq("1"), "stratum"].value_counts()
    weights = weights / weights.sum()
    results = []
    for band in bands:
        part = frame.loc[frame.coupon_history_band.eq(band)]
        table = part.groupby("stratum").agg(purchase=("outcome_purchase_any", "mean"), n=("outcome_purchase_any", "count"),
                                             later_receipt=("outcome_receipt_any", "mean")).loc[weights.index]
        rate = float((table.purchase * weights).sum())
        se = float(np.sqrt((weights ** 2 * table.purchase * (1 - table.purchase) / table.n).sum()))
        results.append({"cumulative_receipt_band": band, "n": int(len(part)), "mean_cumulative_receipts": float(part.cumulative_receipts.mean()),
                        "standardized_purchase_1to14": rate, "standardized_purchase_ci95": [rate - 1.96 * se, rate + 1.96 * se],
                        "outcome_receipt_rate_1to14": float((table.later_receipt * weights).sum())})
    frame[["User_id", "stratum", "pre_click", "pre_purchase", "cumulative_receipts", "coupon_history_band", "outcome_purchase_any", "outcome_receipt_any"]].to_parquet(TIANCHI_CUMULATIVE_OUT / "cohort.parquet", index=False)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "exposure": "all Action=2 coupon-receipt events observed before the anchor, from 2016-01-01 onward",
              "outcome": "any purchase record in days 1-14 after anchor", "eligibility": "at least one pre-anchor click or purchase; standardized over pre click and purchase strata with common support",
              "interpretation": "Cumulative receipts measure historical targeting/exposure, not an active coupon inventory because expiry and redemption balance are unobserved.",
              "results": results,
              "caveat": "Observational association only: accumulated receipt count is confounded by user activity, eligibility, and targeting."}
    (TIANCHI_CUMULATIVE_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures = ROOT / "img/uplift"; figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(results)); rates = np.array([row["standardized_purchase_1to14"] for row in results]) * 100
    ci = np.array([row["standardized_purchase_ci95"] for row in results]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(x, rates, marker="o", linewidth=2, color="#4e79a7")
    axes[0].fill_between(x, ci[:, 0], ci[:, 1], color="#4e79a7", alpha=.18)
    axes[0].set(xticks=x, xticklabels=bands, xlabel="Cumulative coupon receipts before anchor", ylabel="Purchase record, days 1–14 (%)", title="Standardized purchase association")
    axes[1].plot(x, [row["outcome_receipt_rate_1to14"] * 100 for row in results], marker="o", linewidth=2, color="#b07aa1")
    axes[1].set(xticks=x, xticklabels=bands, xlabel="Cumulative coupon receipts before anchor", ylabel="Another receipt, days 1–14 (%)", title="Continued targeting diagnostic")
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi cumulative coupon-receipt history: descriptive association", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_cumulative_coupon_history.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_uplift_audit():
    """Multi-anchor balance, placebo, and temporal-stability audit for receipt uplift."""
    import matplotlib.pyplot as plt

    out = ROOT / "artifacts/tianchi-uplift-audit"; out.mkdir(parents=True, exist_ok=True)
    anchors = pd.to_datetime(["2016-05-01", "2016-05-05", "2016-05-09", "2016-05-13", "2016-05-17"])
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        for anchor in anchors:
            delta = (day - anchor).dt.days
            period = np.select([delta.between(-28, -15), delta.between(-14, -1), delta.eq(0), delta.between(1, 14)],
                               ["placebo", "pre", "anchor", "outcome"], default="outside")
            keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
            data = chunk.loc[keep, ["User_id", "Action"]].copy()
            data["period"], data["anchor_day"] = period[keep], anchor.strftime("%Y-%m-%d")
            pieces.append(data.groupby(["anchor_day", "User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2, 3]).sum().unstack([2, 3], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in ("placebo", "pre", "anchor", "outcome"):
        for action in (0, 1, 2):
            if f"{period}_{action}" not in wide: wide[f"{period}_{action}"] = 0
    rows = []
    for anchor_day, part in wide.groupby(level=0):
        f = part.droplevel(0).reset_index()
        f["treated"] = f.anchor_2.gt(0).astype(int)
        f["pre_click"], f["pre_purchase"] = f.pre_0, f.pre_1
        click = np.select([f.pre_click.eq(0), f.pre_click.eq(1)], ["0", "1"], default="2+")
        purchase = np.where(f.pre_purchase.gt(0), "1+", "0")
        f["stratum"] = pd.Series(click, index=f.index, dtype="string") + "_click__" + pd.Series(purchase, index=f.index, dtype="string")
        f = f.loc[(f.pre_click + f.pre_purchase).gt(0)].copy()
        common = [s for s, g in f.groupby("stratum") if g.treated.nunique() == 2]
        f = f[f.stratum.isin(common)]
        weights = f[f.treated.eq(1)].stratum.value_counts(); weights = weights / weights.sum()
        def standardized(column):
            tab = f.groupby(["stratum", "treated"])[column].mean().unstack().loc[weights.index]
            return float((tab[1] * weights).sum() - (tab[0] * weights).sum())
        f["outcome_purchase"] = f.outcome_1.gt(0).astype(int)
        f["placebo_purchase"] = f.placebo_1.gt(0).astype(int)
        f["later_receipt"] = f.outcome_2.gt(0).astype(int)
        pooled_sd = np.sqrt((f.loc[f.treated.eq(1), "pre_purchase"].var() + f.loc[f.treated.eq(0), "pre_purchase"].var()) / 2)
        rows.append({"anchor_day": str(anchor_day), "treated_n": int(f.treated.sum()), "control_n": int((1-f.treated).sum()),
                     "purchase_difference": standardized("outcome_purchase"), "placebo_purchase_difference": standardized("placebo_purchase"),
                     "later_receipt_difference": standardized("later_receipt"),
                     "pre_purchase_smd": None if not pooled_sd else float((f.loc[f.treated.eq(1), "pre_purchase"].mean()-f.loc[f.treated.eq(0), "pre_purchase"].mean())/pooled_sd)})
    report = {"dataset": "Tianchi O2O online", "anchors": rows,
              "design": "five fixed anchors; 14-day pre-period stratification, 14-day post purchase outcome, and earlier 14-day placebo outcome",
              "interpretation": "Nonzero placebo or baseline imbalance indicates receipt is not conditionally randomized under these observed covariates."}
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = pd.DataFrame(rows); table.to_csv(out / "anchor_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table)); ax.plot(x, table.purchase_difference*100, marker="o", label="post purchase difference", color="#4e79a7")
    ax.plot(x, table.placebo_purchase_difference*100, marker="o", label="pre-treatment placebo difference", color="#e15759")
    ax.axhline(0, color="black", linewidth=.8); ax.set(xticks=x, xticklabels=table.anchor_day.str[5:], ylabel="Treated − control (pp)", xlabel="Anchor date (2016)", title="Receipt uplift audit: temporal stability and placebo")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); (ROOT / "img/uplift").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / "img/uplift/tianchi_uplift_audit.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def tianchi_persona_proxy_labels(frame):
    """Return disjoint observable proxies; Tianchi does not observe carts or removals."""
    active = frame.pre_total.gt(0)
    q25, q75, q95 = frame.loc[active, "pre_total"].quantile([.25, .75, .95]).astype(int)
    labels = np.full(len(frame), "unassigned", dtype=object)
    labels[(frame.pre_purchase.eq(0) & frame.pre_total.between(1, q25)).to_numpy()] = "P1_observation_proxy"
    labels[(frame.pre_purchase.gt(0) & frame.pre_total.ge(q95)).to_numpy()] = "P2_deep_consideration_proxy"
    labels[(frame.pre_purchase.eq(0) & frame.pre_total.ge(q75)).to_numpy()] = "P3_cart_revision_proxy"
    labels[(frame.pre_purchase.gt(0) & frame.pre_total.lt(q95)).to_numpy()] = "P4_purchase_path_proxy"
    return labels, {"pre_total_q25": int(q25), "pre_total_q75": int(q75), "pre_total_q95": int(q95)}


def run_tianchi_persona_proxies():
    """Find Tianchi coupon-anchor users similar on the actions shared with Cosmetics."""
    TIANCHI_PROXY_OUT.mkdir(parents=True, exist_ok=True)
    anchors = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date_received"],
                             dtype={"Date_received": "string"}, chunksize=1_000_000):
        received = pd.to_datetime(chunk.Date_received, format="%Y%m%d", errors="coerce")
        mask = chunk.Action.eq(2) & received.between("2016-05-01", "2016-05-31")
        anchors.append(pd.DataFrame({"User_id": chunk.loc[mask, "User_id"], "anchor": received[mask]}))
    anchors = (pd.concat(anchors, ignore_index=True).sort_values(["User_id", "anchor"])
               .drop_duplicates("User_id").set_index("User_id")["anchor"])
    counts, days = [], []
    users = set(anchors.index)
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date"], dtype={"Date": "string"},
                             chunksize=1_000_000):
        chunk = chunk.loc[chunk.User_id.isin(users) & chunk.Action.isin([0, 1])].copy()
        if chunk.empty:
            continue
        date = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        delta = (date - chunk.User_id.map(anchors)).dt.days
        period = np.select([delta.between(-14, -1), delta.between(1, 14)], ["pre", "post"], default="outside")
        chunk = chunk.loc[period != "outside", ["User_id", "Action"]].copy()
        chunk["period"] = period[period != "outside"]
        chunk["date"] = date[period != "outside"].to_numpy()
        counts.append(chunk.groupby(["User_id", "period", "Action"]).size())
        days.append(chunk[["User_id", "period", "date"]].drop_duplicates())
    wide = pd.concat(counts).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    active_days = pd.concat(days).drop_duplicates().groupby(["User_id", "period"]).size().unstack(fill_value=0)
    frame = anchors.rename("anchor").to_frame().join(wide, how="left").join(active_days.add_suffix("_active_days"), how="left").fillna(0)
    for period in ("pre", "post"):
        for action in (0, 1):
            if f"{period}_{action}" not in frame:
                frame[f"{period}_{action}"] = 0
        if f"{period}_active_days" not in frame:
            frame[f"{period}_active_days"] = 0
    frame = frame.reset_index()
    frame["pre_click"], frame["pre_purchase"] = frame.pop("pre_0").astype(int), frame.pop("pre_1").astype(int)
    frame["post_click"], frame["post_purchase"] = frame.pop("post_0").astype(int), frame.pop("post_1").astype(int)
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["pre_purchase_share"] = np.divide(frame.pre_purchase, frame.pre_total,
                                               out=np.zeros(len(frame)), where=frame.pre_total.gt(0))
    frame["proxy_persona"], quantiles = tianchi_persona_proxy_labels(frame)
    frame["post_any"] = (frame.post_click + frame.post_purchase).gt(0).astype(int)
    frame["post_purchase_any"] = frame.post_purchase.gt(0).astype(int)
    keep = ["User_id", "anchor", "proxy_persona", "pre_click", "pre_purchase", "pre_total",
            "pre_active_days", "pre_purchase_share", "post_click", "post_purchase", "post_active_days",
            "post_any", "post_purchase_any"]
    frame[keep].to_parquet(TIANCHI_PROXY_OUT / "tianchi_coupon_anchor_persona_proxies.parquet", index=False)
    summary = []
    for persona, part in frame.groupby("proxy_persona", sort=True):
        summary.append({"proxy_persona": persona, "users": int(len(part)),
                        "pre_click_median": float(part.pre_click.median()),
                        "pre_purchase_median": float(part.pre_purchase.median()),
                        "pre_active_days_median": float(part.pre_active_days.median()),
                        "post_any_14d_rate": float(part.post_any.mean()),
                        "post_purchase_14d_rate": float(part.post_purchase_any.mean()),
                        "sample_user_ids": part.User_id.sort_values().head(10).astype(int).tolist()})
    result = {"dataset": "Tianchi O2O online", "anchor": "first coupon receipt in May 2016",
              "matching_basis": "pre-anchor click, purchase, total activity, and active days only",
              "missing_from_tianchi": ["cart", "remove_from_cart", "session boundaries"],
              "quantiles": quantiles, "proxies": summary,
              "causal_caveat": "receipt is observational; post rates are not coupon uplift estimates"}
    (TIANCHI_PROXY_OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_tianchi_first_receipt_alignment():
    """Match pre-first-receipt Tianchi behavior to fixed retail clusters."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler

    from analysis.persona.cluster import (
        ALIGNMENT_FEATURES, _fixed_k4_membership, _selected_events,
        _shared_behavior_features,
    )

    TIANCHI_FIRST_RECEIPT_OUT.mkdir(parents=True, exist_ok=True)

    receipt_parts = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date_received"],
                             dtype={"User_id": "Int64", "Action": "Int8",
                                    "Date_received": "string"}, chunksize=1_000_000):
        chunk = chunk.loc[chunk.Action.eq(2) & chunk.User_id.notna()]
        if chunk.empty:
            continue
        day = pd.to_datetime(chunk.Date_received, format="%Y%m%d", errors="coerce")
        receipt_parts.append(pd.DataFrame({"user": chunk.User_id.astype("int64"), "day": day})
                             .dropna().groupby("user").day.min())
    first_receipt = pd.concat(receipt_parts).groupby(level=0).min()
    anchors = first_receipt[first_receipt.between("2016-05-01", "2016-05-31")]
    anchor_users = set(anchors.index)

    behavior_parts, later_receipt_parts = [], []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"User_id": "Int64", "Action": "Int8", "Date": "string",
                                    "Date_received": "string"}, chunksize=1_000_000):
        chunk = chunk.loc[chunk.User_id.isin(anchor_users) & chunk.Action.isin([0, 1, 2])].copy()
        if chunk.empty:
            continue
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        is_receipt = chunk.Action.eq(2)
        day.loc[is_receipt] = pd.to_datetime(
            chunk.loc[is_receipt, "Date_received"], format="%Y%m%d", errors="coerce"
        )
        delta = (day - chunk.User_id.map(anchors)).dt.days
        behavior = chunk.Action.isin([0, 1]) & (delta.between(-28, -1) | delta.between(1, 28))
        if behavior.any():
            part = pd.DataFrame({
                "user": chunk.loc[behavior, "User_id"].astype("int64"),
                "at": day[behavior],
                "action": chunk.loc[behavior, "Action"].map({0: "explore", 1: "purchase"}),
                "period": np.where(delta[behavior].lt(0), "pre", "post"),
            }).dropna()
            behavior_parts.append(part)
        later = is_receipt & delta.between(1, 28)
        if later.any():
            later_receipt_parts.append(chunk.loc[later].groupby("User_id").size())
    behavior = pd.concat(behavior_parts, ignore_index=True)
    later_receipts = (pd.concat(later_receipt_parts).groupby(level=0).sum()
                      if later_receipt_parts else pd.Series(dtype=int))

    counts = (behavior.groupby(["user", "period", "action"]).size()
                      .unstack(["period", "action"], fill_value=0))
    for period in ("pre", "post"):
        for action in ("explore", "purchase"):
            if (period, action) not in counts:
                counts[(period, action)] = 0
    counts = counts.sort_index(axis=1)
    active_days = (behavior.assign(day=behavior["at"].dt.floor("D"))
                           .drop_duplicates(["user", "period", "day"])
                           .groupby(["user", "period"]).size().unstack(fill_value=0))
    cohort = pd.DataFrame(index=anchors.index)
    cohort.index.name = "user_id"
    for period in ("pre", "post"):
        cohort[f"{period}_click"] = counts[(period, "explore")].reindex(cohort.index, fill_value=0)
        cohort[f"{period}_purchase"] = counts[(period, "purchase")].reindex(cohort.index, fill_value=0)
        cohort[f"{period}_active_days"] = active_days.get(period, pd.Series(dtype=int)).reindex(
            cohort.index, fill_value=0
        )
    cohort["later_receipts_28d"] = later_receipts.reindex(cohort.index, fill_value=0).astype(int)
    cohort["anchor"] = anchors
    cohort["pre_total"] = cohort.pre_click + cohort.pre_purchase
    cohort["post_total"] = cohort.post_click + cohort.post_purchase
    eligible = cohort.pre_total.ge(2)
    cohort = cohort.loc[eligible].copy()

    reference_features = {}
    for source in ("cosmetics", "electronics"):
        membership, _ = _fixed_k4_membership(source)
        events = _selected_events(source, set(membership.user_id.astype("int64")))
        events = (events.loc[events.action.isin(["view", "purchase"]),
                             ["user_id", "at", "action"]]
                        .rename(columns={"user_id": "user"}))
        events["action"] = events.action.replace({"view": "explore"})
        end = events.groupby("user").at.transform("max")
        events = events.loc[events["at"].gt(end - pd.Timedelta(days=28))]
        reference_features[source] = _shared_behavior_features(events, membership)

    pre_events = behavior.loc[behavior.period.eq("pre") & behavior.user.isin(cohort.index),
                              ["user", "at", "action"]]
    post_events = behavior.loc[behavior.period.eq("post") & behavior.user.isin(cohort.index),
                               ["user", "at", "action"]]
    dummy = pd.DataFrame({"user_id": cohort.index, "cluster": 0})
    pre_features = _shared_behavior_features(pre_events, dummy)
    post_users = np.sort(post_events.user.unique())
    post_features = _shared_behavior_features(
        post_events, pd.DataFrame({"user_id": post_users, "cluster": 0})
    ) if len(post_users) else pd.DataFrame()

    reference = pd.concat([
        frame.assign(dataset=source) for source, frame in reference_features.items()
    ], ignore_index=True)
    scaler = RobustScaler().fit(reference[list(ALIGNMENT_FEATURES)])
    reference[list(ALIGNMENT_FEATURES)] = scaler.transform(
        reference[list(ALIGNMENT_FEATURES)]
    )
    pre_scaled = scaler.transform(pre_features[list(ALIGNMENT_FEATURES)])
    post_scaled = (scaler.transform(post_features[list(ALIGNMENT_FEATURES)])
                   if len(post_features) else np.empty((0, len(ALIGNMENT_FEATURES))))

    matches = pre_features[["user_id"]].copy()
    matches["user_id"] = matches.user_id.astype("int64")
    for source in ("cosmetics", "electronics"):
        profiles = (reference.loc[reference.dataset.eq(source)]
                     .groupby("cluster")[list(ALIGNMENT_FEATURES)].median().sort_index())
        cluster_ids = profiles.index.to_numpy()
        pre_cluster, similarity, margin = nearest_cluster_matches(
            pre_scaled, profiles.to_numpy(), cluster_ids
        )
        matches[f"{source}_pre_cluster"] = pre_cluster
        matches[f"{source}_similarity"] = similarity
        matches[f"{source}_margin"] = margin
        post_cluster = pd.Series("inactive", index=cohort.index, dtype=object)
        if len(post_features):
            assigned, _, _ = nearest_cluster_matches(post_scaled, profiles.to_numpy(), cluster_ids)
            post_cluster.loc[post_features.user_id.astype("int64")] = assigned.astype(str)
        matches[f"{source}_post_cluster"] = matches.user_id.map(post_cluster)

    cohort = cohort.reset_index().merge(matches, on="user_id", how="inner", validate="one_to_one")
    cohort.to_parquet(TIANCHI_FIRST_RECEIPT_OUT / "cohort.parquet", index=False)

    summaries, transitions = {}, {}
    for source in ("cosmetics", "electronics"):
        cluster_col = f"{source}_pre_cluster"
        post_col = f"{source}_post_cluster"
        rows = []
        for cluster, part in cohort.groupby(cluster_col, sort=True):
            no_later = part.loc[part.later_receipts_28d.eq(0)]
            rows.append({
                "cluster": int(cluster), "users": int(len(part)),
                "share": float(len(part) / len(cohort)),
                "median_similarity": float(part[f"{source}_similarity"].median()),
                "median_margin": float(part[f"{source}_margin"].median()),
                "pre_click_mean": float(part.pre_click.mean()),
                "post_click_mean": float(part.post_click.mean()),
                "pre_purchase_any": float(part.pre_purchase.gt(0).mean()),
                "post_purchase_any": float(part.post_purchase.gt(0).mean()),
                "purchase_any_change": float(
                    part.post_purchase.gt(0).mean() - part.pre_purchase.gt(0).mean()
                ),
                "post_inactive": float(part.post_total.eq(0).mean()),
                "later_receipt_rate": float(part.later_receipts_28d.gt(0).mean()),
                "no_later_receipt_users": int(len(no_later)),
                "no_later_receipt_purchase_change": float(
                    no_later.post_purchase.gt(0).mean() - no_later.pre_purchase.gt(0).mean()
                ) if len(no_later) else None,
            })
        summaries[source] = rows
        table = pd.crosstab(cohort[cluster_col], cohort[post_col], normalize="index")
        transitions[source] = {
            str(index): {str(column): float(value) for column, value in row.items()}
            for index, row in table.iterrows()
        }

    cross_reference = pd.crosstab(
        cohort.cosmetics_pre_cluster, cohort.electronics_pre_cluster, normalize="index"
    )
    result = {
        "dataset": "Tianchi O2O online",
        "anchor": "first observed coupon receipt, 2016-05-01 through 2016-05-31",
        "windows": "days -28..-1 and +1..+28; anchor day excluded because within-day order is unknown",
        "eligibility": "at least two observed click/purchase events before first receipt",
        "first_receipt_users": int(len(anchors)),
        "matched_users": int(len(cohort)),
        "matching_features": list(ALIGNMENT_FEATURES),
        "reference_window": "trailing 28 days of each fixed Cosmetics/Electronics K=4 member",
        "overall_change": {
            "pre_click_mean": float(cohort.pre_click.mean()),
            "post_click_mean": float(cohort.post_click.mean()),
            "pre_purchase_mean": float(cohort.pre_purchase.mean()),
            "post_purchase_mean": float(cohort.post_purchase.mean()),
            "pre_purchase_any": float(cohort.pre_purchase.gt(0).mean()),
            "post_purchase_any": float(cohort.post_purchase.gt(0).mean()),
            "post_inactive": float(cohort.post_total.eq(0).mean()),
            "later_receipt_rate": float(cohort.later_receipts_28d.gt(0).mean()),
        },
        "cross_reference_match": {
            str(index): {str(column): float(value) for column, value in row.items()}
            for index, row in cross_reference.iterrows()
        },
        "summaries": summaries,
        "transitions": transitions,
        "limitations": [
            "Tianchi click is treated as the analogue of retail view; cart and removal are unavailable.",
            "The first observed receipt need not be the user's lifetime first coupon receipt.",
            "Pre/post changes are observational and can reflect targeting, seasonality, regression to the mean, or later receipts.",
            "Cosmetics and Electronics do not observe coupon exposure, so their clusters are behavioral references rather than untreated controls.",
        ],
    }
    (TIANCHI_FIRST_RECEIPT_OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for column, source in enumerate(("cosmetics", "electronics")):
        rows = summaries[source]
        x = np.arange(len(rows))
        labels = [f"{source[0].upper()}{row['cluster']}" for row in rows]
        axes[0, column].bar(x, [100 * row["share"] for row in rows], color="#4C78A8")
        axes[0, column].set(xticks=x, xticklabels=labels, ylabel="Matched users (%)",
                            title=f"Pre-receipt match to {source.title()} K=4")
        axes[1, column].bar(x - .18, [100 * row["pre_purchase_any"] for row in rows], .36,
                            label="pre", color="#9ECAE1")
        axes[1, column].bar(x + .18, [100 * row["post_purchase_any"] for row in rows], .36,
                            label="post", color="#F28E2B")
        axes[1, column].set(xticks=x, xticklabels=labels, ylabel="Users with purchase record (%)",
                            title="Observed change around first receipt")
        for ax in axes[:, column]:
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].legend(frameon=False)
    figure_dir = ROOT / "img/uplift"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "tianchi_first_receipt_cluster_alignment.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result


def run_tianchi_persona_uplift():
    """Describe post-receipt trajectories and canonical-type bounds by Tianchi proxy."""
    from analysis.uplift.canonical import bounds
    import matplotlib.pyplot as plt

    TIANCHI_UPLIFT_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    periods = ["pre", "anchor", "w1", "w2", "w3", "w4", "w5", "w6"]
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.select([delta.between(-14, -1), delta.eq(0), delta.between(1, 7), delta.between(8, 14),
                            delta.between(15, 21), delta.between(22, 28), delta.between(29, 35), delta.between(36, 42)],
                           periods, default="outside")
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        pieces.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in periods:
        for action in (0, 1, 2):
            if f"{period}_{action}" not in wide:
                wide[f"{period}_{action}"] = 0
    frame = wide.reset_index()
    frame["pre_click"], frame["pre_purchase"] = frame["pre_0"], frame["pre_1"]
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["proxy_persona"], quantiles = tianchi_persona_proxy_labels(frame)
    frame["treated"] = frame["anchor_2"].gt(0).astype(int)
    for week in range(1, 7):
        frame[f"w{week}_any"] = (frame[f"w{week}_0"] + frame[f"w{week}_1"]).gt(0).astype(int)
        frame[f"w{week}_purchase_any"] = frame[f"w{week}_1"].gt(0).astype(int)
        frame[f"w{week}_later_receipt"] = frame[f"w{week}_2"].gt(0).astype(int)
    frame["purchase_14d"] = frame[["w1_purchase_any", "w2_purchase_any"]].any(axis=1).astype(int)
    cohort = frame.loc[frame.proxy_persona.ne("unassigned")].copy()
    cohort.to_parquet(TIANCHI_UPLIFT_OUT / "tianchi_persona_uplift_cohort.parquet", index=False)
    result = []
    for persona, part in cohort.groupby("proxy_persona", sort=True):
        arms = part.groupby("treated").purchase_14d.agg(["count", "sum", "mean"])
        item = {"proxy_persona": persona, "n": int(len(part)), "quantiles": quantiles}
        if set(arms.index) == {0, 1}:
            p0, p1 = float(arms.loc[0, "mean"]), float(arms.loc[1, "mean"])
            item.update({"n_control": int(arms.loc[0, "count"]), "n_treated": int(arms.loc[1, "count"]),
                         "purchase_14d_control": p0, "purchase_14d_treated": p1,
                         "observed_difference": p1 - p0, "canonical_type_bounds": bounds(p0, p1)})
        item["weekly"] = []
        for week in range(1, 7):
            arms_week = part.groupby("treated")[[f"w{week}_any", f"w{week}_purchase_any", f"w{week}_later_receipt"]].mean()
            if set(arms_week.index) != {0, 1}:
                continue
            item["weekly"].append({"week": week,
                                   "any_control": float(arms_week.loc[0, f"w{week}_any"]),
                                   "any_treated": float(arms_week.loc[1, f"w{week}_any"]),
                                   "any_difference": float(arms_week.loc[1, f"w{week}_any"] - arms_week.loc[0, f"w{week}_any"]),
                                   "purchase_control": float(arms_week.loc[0, f"w{week}_purchase_any"]),
                                   "purchase_treated": float(arms_week.loc[1, f"w{week}_purchase_any"]),
                                   "purchase_difference": float(arms_week.loc[1, f"w{week}_purchase_any"] - arms_week.loc[0, f"w{week}_purchase_any"]),
                                   "later_receipt_treated": float(arms_week.loc[1, f"w{week}_later_receipt"]),
                                   "later_receipt_control": float(arms_week.loc[0, f"w{week}_later_receipt"])})
        result.append(item)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "treatment": "coupon receipt record on anchor day", "outcome": "any purchase record in days 1-14",
              "proxy_rule": "pre-anchor click/purchase depth only; carts, removal, sessions are unavailable",
              "canonical_interpretation": "Fréchet bounds under the additional assumption that arm-specific observed rates equal comparable potential-outcome marginals; no individual type labels",
              "duration_interpretation": "weekly treated-control differences are observational and can be confounded by later coupon receipts",
              "results": result}
    (TIANCHI_UPLIFT_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [item["proxy_persona"].replace("_proxy", "") for item in result]
    control = [item["purchase_14d_control"] * 100 for item in result]
    treated = [item["purchase_14d_treated"] * 100 for item in result]
    x = np.arange(len(result))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - .18, control, .36, label="no anchor-day receipt", color="#4e79a7")
    ax.bar(x + .18, treated, .36, label="anchor-day receipt", color="#f28e2b")
    for index, item in enumerate(result):
        ax.text(index, max(control[index], treated[index]) + 1, f"Δ {item['observed_difference'] * 100:+.1f}pp",
                ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="Purchase record in days 1–14 (%)",
           title="Tianchi proxy cohorts: observed 14-day purchase contrast", ylim=(0, max(treated) + 12))
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(figures / "tianchi_proxy_14d_contrast.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    types = ["sure_thing", "persuadable", "sleeping_dog", "lost_cause"]
    type_labels = ["sure thing", "persuadable", "sleeping dog", "lost cause"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, item in zip(axes.flat, result):
        intervals = item["canonical_type_bounds"]
        for index, key in enumerate(types):
            low, high = np.array(intervals[key]) * 100
            ax.hlines(index, low, high, color="#59a14f" if key == "persuadable" else "#4e79a7", linewidth=5)
            ax.plot([low, high], [index, index], "o", color="black", ms=3)
            ax.text(high + 1, index, f"{low:.1f}–{high:.1f}%", va="center", fontsize=8)
        ax.set(title=item["proxy_persona"].replace("_proxy", ""), yticks=range(4), yticklabels=type_labels,
               xlim=(0, 100), xlabel="Possible share (%)")
        ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Canonical-type Fréchet bounds under the comparability assumption", y=1.02)
    fig.tight_layout(); fig.savefig(figures / "tianchi_proxy_canonical_bounds.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    weeks = np.arange(1, 7)
    colors = ["#4e79a7", "#76b7b2", "#f28e2b", "#59a14f"]
    for item, color, label in zip(result, colors, labels):
        weekly = item["weekly"]
        axes[0].plot(weeks, [row["purchase_difference"] * 100 for row in weekly], marker="o", color=color, label=label)
        # Receipt dates are right-censored after mid-June; only weeks 1–5 are comparable.
        axes[1].plot(weeks[:5], [(row["later_receipt_treated"] - row["later_receipt_control"]) * 100 for row in weekly[:5]],
                     marker="o", color=color, label=label)
    axes[0].axhline(0, color="black", linewidth=.8); axes[1].axhline(0, color="black", linewidth=.8)
    axes[0].set(title="Observed weekly purchase-record difference", xlabel="Week after anchor", ylabel="Treated − control (percentage points)", xticks=weeks)
    axes[1].set(title="Difference in subsequent coupon receipt", xlabel="Week after anchor", ylabel="Treated − control (percentage points)", xticks=weeks[:5])
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Duration is confounded by later coupon receipt; week 6 receipt data are censored", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_proxy_duration.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_single_receipt_duration():
    """Descriptive duration analysis with no other recorded coupon receipt in the window."""
    import matplotlib.pyplot as plt

    TIANCHI_SINGLE_RECEIPT_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    periods = ["pre", "anchor", "w1", "w2", "w3", "w4", "w5", "w6"]
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.select([delta.between(-14, -1), delta.eq(0), delta.between(1, 7), delta.between(8, 14),
                            delta.between(15, 21), delta.between(22, 28), delta.between(29, 35), delta.between(36, 42)],
                           periods, default="outside")
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        pieces.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in periods:
        for action in (0, 1, 2):
            if f"{period}_{action}" not in wide:
                wide[f"{period}_{action}"] = 0
    frame = wide.reset_index()
    frame["pre_click"], frame["pre_purchase"] = frame["pre_0"], frame["pre_1"]
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["receipt_total"] = frame[[f"{period}_2" for period in periods]].sum(axis=1)
    frame["single_anchor_receipt"] = frame.anchor_2.eq(1) & frame.receipt_total.eq(1)
    frame["never_receipt_window"] = frame.receipt_total.eq(0)
    frame["treated"] = frame.single_anchor_receipt.astype(int)
    frame["arm"] = np.where(frame.single_anchor_receipt, "single_anchor_receipt",
                            np.where(frame.never_receipt_window, "never_receipt", "excluded_other_receipt"))
    frame = frame.loc[frame.pre_total.gt(0) & frame.arm.ne("excluded_other_receipt")].copy()
    frame["pre_click_bin"] = np.select([frame.pre_click.eq(0), frame.pre_click.eq(1)], ["0", "1"], default="2+")
    frame["pre_purchase_bin"] = np.where(frame.pre_purchase.gt(0), "1+", "0")
    frame["stratum"] = frame.pre_click_bin + "_click__" + frame.pre_purchase_bin + "_purchase"
    for week in range(1, 7):
        frame[f"w{week}_purchase_any"] = frame[f"w{week}_1"].gt(0).astype(int)
        frame[f"w{week}_any"] = (frame[f"w{week}_0"] + frame[f"w{week}_1"]).gt(0).astype(int)
    eligible_strata = []
    for stratum, part in frame.groupby("stratum"):
        if part.treated.nunique() == 2:
            eligible_strata.append(stratum)
    frame = frame[frame.stratum.isin(eligible_strata)].copy()
    treated_counts = frame[frame.treated.eq(1)].stratum.value_counts()
    weights = treated_counts / treated_counts.sum()
    weekly = []
    for week in range(1, 7):
        values = {}
        for outcome in ("purchase_any", "any"):
            table = frame.groupby(["stratum", "treated"])[f"w{week}_{outcome}"].agg(["mean", "count"]).unstack()
            control = float((table[("mean", 0)] * weights.loc[table.index]).sum())
            treated = float((table[("mean", 1)] * weights.loc[table.index]).sum())
            variance = float((weights.loc[table.index] ** 2 * (
                table[("mean", 0)] * (1 - table[("mean", 0)]) / table[("count", 0)] +
                table[("mean", 1)] * (1 - table[("mean", 1)]) / table[("count", 1)])).sum())
            values[f"{outcome}_control"] = control
            values[f"{outcome}_treated"] = treated
            values[f"{outcome}_difference"] = treated - control
            values[f"{outcome}_difference_ci95"] = [treated - control - 1.96 * np.sqrt(variance),
                                                       treated - control + 1.96 * np.sqrt(variance)]
        weekly.append({"week": week, **values})
    keep = ["User_id", "arm", "treated", "stratum", "pre_click", "pre_purchase", "pre_total", "receipt_total"] + [
        f"w{week}_{kind}" for week in range(1, 7) for kind in ("purchase_any", "any")]
    frame[keep].to_parquet(TIANCHI_SINGLE_RECEIPT_OUT / "cohort.parquet", index=False)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "treated_definition": "exactly one receipt event, on anchor day, and no other receipt in days -14 through 42",
              "control_definition": "no receipt event in days -14 through 42",
              "eligibility": "at least one pre-anchor click or purchase; standardized within click-count and purchase-history strata",
              "n_treated": int(frame.treated.sum()), "n_control": int((1 - frame.treated).sum()),
              "stratum_weights": {str(key): float(value) for key, value in weights.items()},
              "weekly_standardized": weekly,
              "caveat": "conditioning on no future receipt is post-treatment selection; results describe a single-receipt trajectory, not an intervention effect"}
    (TIANCHI_SINGLE_RECEIPT_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    weeks = np.arange(1, 7)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    axes[0].plot(weeks, [row["purchase_any_control"] * 100 for row in weekly], marker="o", color="#4e79a7", label="never receipt")
    axes[0].plot(weeks, [row["purchase_any_treated"] * 100 for row in weekly], marker="o", color="#f28e2b", label="single anchor receipt")
    axes[0].set(title="Standardized weekly purchase-record rate", xlabel="Week after anchor", ylabel="Purchase record (%)", xticks=weeks)
    axes[0].legend(frameon=False)
    axes[1].plot(weeks, [row["purchase_any_difference"] * 100 for row in weekly], marker="o", color="#59a14f", label="purchase")
    purchase_ci = np.array([row["purchase_any_difference_ci95"] for row in weekly]) * 100
    axes[1].fill_between(weeks, purchase_ci[:, 0], purchase_ci[:, 1], color="#59a14f", alpha=.18, label="purchase 95% normal CI")
    axes[1].plot(weeks, [row["any_difference"] * 100 for row in weekly], marker="o", color="#b07aa1", label="any click/purchase")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(title="Standardized treated − control difference", xlabel="Week after anchor", ylabel="Percentage points", xticks=weeks)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi single-receipt descriptive trajectory (no later coupon receipt)", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_single_receipt_duration.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_coupon_redemption_ordering():
    """Audit whether Tianchi can support an oldest-received-coupon choice rule.

    A coupon purchase has a ``Date_received`` field, but this need not have a
    matching Action=2 receipt record.  We consequently report a deliberately
    strict, directly linkable subset rather than projecting its pattern to all
    coupon purchases.
    """
    import matplotlib.pyplot as plt

    out = ROOT / "artifacts/tianchi-coupon-redemption-ordering"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    action_counts = {0: 0, 1: 0, 2: 0}
    coupon_purchase_count = 0
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Merchant_id", "Action", "Coupon_id", "Date_received", "Date"],
                             dtype={"Coupon_id": "string", "Date_received": "string", "Date": "string"}, chunksize=1_000_000):
        for action, n in chunk.Action.value_counts().items():
            action_counts[int(action)] += int(n)
        coupon_purchase_count += int((chunk.Action.eq(1) & chunk.Coupon_id.notna()).sum())
        records.append(chunk.loc[chunk.Action.eq(2) | (chunk.Action.eq(1) & chunk.Coupon_id.notna())])
    events = pd.concat(records, ignore_index=True)
    events["received"] = pd.to_datetime(events.Date_received, format="%Y%m%d", errors="coerce")
    events["event_date"] = pd.to_datetime(events.Date, format="%Y%m%d", errors="coerce")
    keys = ["User_id", "Merchant_id", "Coupon_id", "received"]
    receipts = events.loc[events.Action.eq(2), keys].copy()
    receipt_counts = receipts.groupby(keys, dropna=False).size().rename("receipt_record_count")
    issued = receipts.drop_duplicates(keys)
    redemptions = events.loc[events.Action.eq(1) & events.Coupon_id.notna()].copy()
    redemptions = redemptions.merge(receipt_counts.reset_index(), on=keys, how="left")
    direct_unique = redemptions.loc[redemptions.receipt_record_count.eq(1)].copy()
    direct_unique["token"] = direct_unique[keys].astype(str).agg("|".join, axis=1)
    # Multiple purchase records against one token are not an orderable single-coupon use.
    first_claim = direct_unique.groupby("token").event_date.transform("min")
    direct_unique = direct_unique.loc[direct_unique.event_date.eq(first_claim)].drop_duplicates(["token", "event_date"])

    issues_by_user = {user: group.sort_values("received") for user, group in issued.groupby("User_id", sort=False)}
    choice_rows = []
    for user, claims in direct_unique.groupby("User_id", sort=False):
        user_issues = issues_by_user.get(user)
        if user_issues is None:
            continue
        issue_rows = [(row.received, (row.Merchant_id, row.Coupon_id, row.received))
                      for row in user_issues.itertuples(index=False)]
        used = set()
        # Within-day ordering is unavailable: retain only a single, uniquely linked claim per user-day.
        for purchase_day, claims_on_day in claims.groupby("event_date", sort=True):
            if len(claims_on_day) != 1:
                continue
            claim = claims_on_day.iloc[0]
            token = (claim.Merchant_id, claim.Coupon_id, claim.received)
            candidates = [(received, issue) for received, issue in issue_rows if received < purchase_day and issue not in used]
            if token not in {issue for _, issue in candidates}:
                continue
            receipt_days = np.array([received for received, _ in candidates])
            oldest_day = receipt_days.min()
            older_count = int((receipt_days < claim.received).sum())
            candidate_count = len(candidates)
            choice_rows.append({"User_id": user, "purchase_day": purchase_day, "candidate_count": candidate_count,
                                "used_coupon_age_days": int((purchase_day - claim.received).days),
                                "selected_oldest": int(older_count == 0),
                                "random_oldest_probability": float((receipt_days == oldest_day).mean()),
                                "relative_age_rank_0_oldest_1_youngest": None if candidate_count == 1 else older_count / (candidate_count - 1)})
            used.add(token)
    choices = pd.DataFrame(choice_rows)
    choices.to_parquet(out / "orderable_coupon_choices.parquet", index=False)
    multi = choices.loc[choices.candidate_count.ge(2)].copy()
    multi["choice_set_bin"] = pd.cut(multi.candidate_count, [1, 2, 3, 5, 10, np.inf], labels=["2", "3", "4–5", "6–10", "11+"])
    by_size = []
    for label, part in multi.groupby("choice_set_bin", observed=True):
        by_size.append({"candidate_count": str(label), "n": int(len(part)),
                        "selected_oldest_rate": float(part.selected_oldest.mean()),
                        "random_oldest_rate": float(part.random_oldest_probability.mean()),
                        "mean_relative_age_rank": float(part.relative_age_rank_0_oldest_1_youngest.mean())})
    report = {
        "dataset": "Tianchi O2O online",
        "action_counts": action_counts,
        "coupon_purchase_records": coupon_purchase_count,
        "coupon_purchase_has_direct_unique_receipt": int(len(direct_unique)),
        "direct_unique_receipt_coverage": float(len(direct_unique) / coupon_purchase_count),
        "same_day_receipt_and_purchase_among_coupon_purchases": int(redemptions.event_date.eq(redemptions.received).sum()),
        "same_day_receipt_and_purchase_rate": float(redemptions.event_date.eq(redemptions.received).mean()),
        "strictly_orderable_choice_sets": int(len(choices)),
        "strictly_orderable_with_2plus_candidates": int(len(multi)),
        "multi_candidate_summary": {
            "selected_oldest_rate": float(multi.selected_oldest.mean()),
            "random_oldest_rate": float(multi.random_oldest_probability.mean()),
            "mean_relative_age_rank_0_oldest_1_youngest": float(multi.relative_age_rank_0_oldest_1_youngest.mean()),
            "median_used_coupon_age_days": float(multi.used_coupon_age_days.median())},
        "by_choice_set_size": by_size,
        "interpretation": "The strict subset is not representative of all coupon purchases. It is an orderability audit, not a population FIFO estimate.",
        "modeling_implication": "Do not impose FIFO from Tianchi. If coupon-level inventory is simulated, estimate/sweep an age coefficient in a conditional coupon-choice model and retain a no-age-effect case."}
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    age_bins = [-.5, .5, 1.5, 3.5, 7.5, 14.5, 30.5, choices.used_coupon_age_days.max() + 1]
    axes[0].hist(choices.used_coupon_age_days, bins=age_bins, color="#4e79a7", edgecolor="white")
    axes[0].set(title="Used coupon age in strict orderable subset", xlabel="Days from receipt to purchase", ylabel="Coupon purchases")
    labels = [row["candidate_count"] for row in by_size]
    x = np.arange(len(labels))
    axes[1].plot(x, [row["selected_oldest_rate"] * 100 for row in by_size], marker="o", linewidth=2, color="#f28e2b", label="observed oldest selected")
    axes[1].plot(x, [row["random_oldest_rate"] * 100 for row in by_size], marker="o", linewidth=2, color="#7f7f7f", linestyle="--", label="uniform-choice benchmark")
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Observed available coupons in choice set", ylabel="Oldest coupon selected (%)", title="FIFO diagnostic; strict subset only")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi coupon ordering audit", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_coupon_ordering_audit.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def online():
    PERSONA_OUT.mkdir(parents=True, exist_ok=True)
    # Pick a receipt in May; both 14-day windows then lie within the logged Jan–June period.
    receipts, activities = [], []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date_received"], dtype={"Date_received": "string"}, chunksize=1_000_000):
        dates = pd.to_datetime(chunk.Date_received, format="%Y%m%d", errors="coerce")
        chosen = chunk.Action.eq(2) & dates.between("2016-05-01", "2016-05-31")
        receipts.append(pd.DataFrame({"user": chunk.loc[chosen, "User_id"], "anchor": dates[chosen]}))
    candidates = pd.concat(receipts, ignore_index=True).sort_values(["user", "anchor"])
    anchors = candidates.drop_duplicates("user").reset_index(drop=True)
    assert len(anchors) > 0, "No eligible receipts: check dates and schema"
    users = set(anchors.user)
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"], dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        chunk = chunk[chunk.User_id.isin(users)].copy()
        chunk["day"] = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        is_receipt = chunk.Action.eq(2)
        chunk.loc[is_receipt, "day"] = pd.to_datetime(chunk.loc[is_receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        chunk = chunk.merge(anchors, left_on="User_id", right_on="user")
        delta = (chunk.day - chunk.anchor).dt.days
        chunk["period"] = np.select([delta.between(-14, -1), delta.between(1, 14)], ["pre", "post"], default="outside")
        chunk = chunk[chunk.period.ne("outside") & chunk.Action.isin([0, 1, 2])]
        activities.append(chunk.groupby(["user", "period", "Action"]).size())
    counts = pd.concat(activities).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    counts.columns = [f'{period}_{action}' for period, action in counts.columns]
    pairs = anchors.join(counts, on="user").fillna(0)
    for period in ["pre", "post"]:
        for action in [0, 1, 2]:
            col = f"{period}_{action}"
            if col not in pairs:
                pairs[col] = 0
            pairs[col] = pairs[col].astype(int)
    pre_click, pre_buy = pairs.pre_0.gt(0), pairs.pre_1.gt(0)
    pairs["group"] = np.select([pre_click & ~pre_buy, pre_click & pre_buy, ~pre_click & pre_buy],
        ["click_only", "click_and_purchase", "purchase_only"], default="neither_recorded")
    post_click, post_buy = pairs.post_0.gt(0), pairs.post_1.gt(0)
    pairs["post_group"] = np.select([post_click & ~post_buy, post_click & post_buy, ~post_click & post_buy],
        ["click_only", "click_and_purchase", "purchase_only"], default="neither_recorded")
    pairs.to_csv(PERSONA_OUT / "tianchi_user_pairs.csv", index=False)
    assert len(pairs) == pairs.user.nunique()
    report = []
    for name, g in pairs.groupby("group"):
        report.append({"group": name, "n": len(g), "pre_click_mean": g.pre_0.mean(), "post_click_mean": g.post_0.mean(),
                       "pre_purchase_mean": g.pre_1.mean(), "post_purchase_mean": g.post_1.mean(),
                       "post_purchase_users": int(g.post_1.gt(0).sum()), "post_click_users": int(g.post_0.gt(0).sum()),
                       "post_more_clicks": int(g.post_0.gt(g.pre_0).sum()),
                       "post_fewer_clicks": int(g.post_0.lt(g.pre_0).sum()),
                       "another_post_receipt": int(g.post_2.gt(0).sum()),
                       "post_groups": {str(k): int(v) for k, v in g.post_group.value_counts().items()},
                       "no_later_receipt_n": int(g.post_2.eq(0).sum()),
                       "no_later_receipt_post_purchase_users": int((g.post_2.eq(0) & g.post_1.gt(0)).sum())})
    return {"users": len(pairs), "groups": report, "other_post_receipt_users": int(pairs.post_2.gt(0).sum())}


def purchase_history(dataset):
    p = pd.read_csv(ROOT / f"artifacts/coupon_before_after/{dataset}_pairs.csv")
    p["prior_purchase_count"] = pd.cut(p.pre, [-1, 0, 1, float("inf")], labels=["0", "1", "2+"])
    result = []
    for name, g in p.groupby("prior_purchase_count", observed=True):
        result.append({"prior_count": str(name), "n": len(g), "post_any": int(g.post.gt(0).sum()),
                       "post_more": int(g.post.gt(g.pre).sum()), "post_fewer": int(g.post.lt(g.pre).sum()),
                       "pre_mean": g.pre.mean(), "post_mean": g.post.mean()})
    return result


PAIRS_OUT = ROOT / "artifacts" / "coupon_before_after"


def summarize_pairs(name, pairs):
    PAIRS_OUT.mkdir(parents=True, exist_ok=True)
    delta = pairs.post - pairs.pre
    result = {
        "dataset": name, "n": len(pairs),
        "pre_mean": pairs.pre.mean(), "post_mean": pairs.post.mean(),
        "decrease": int((delta < 0).sum()), "same": int((delta == 0).sum()),
        "increase": int((delta > 0).sum()),
        "zero_to_positive": int(((pairs.pre == 0) & (pairs.post > 0)).sum()),
        "positive_to_zero": int(((pairs.pre > 0) & (pairs.post == 0)).sum()),
        "no_purchase_both": int(((pairs.pre == 0) & (pairs.post == 0)).sum()),
    }
    assert result["decrease"] + result["same"] + result["increase"] == result["n"]
    pairs.to_csv(PAIRS_OUT / f"{name}_pairs.csv", index=False)
    return result


def starbucks():
    raw = ROOT / "data/raw/starbucks"
    events = pd.read_json(raw / "transcript.json", lines=True)
    portfolio = pd.read_json(raw / "portfolio.json", lines=True)
    offer_type = portfolio.set_index("id").offer_type.to_dict()
    offers = events[events.event.eq("offer received")].copy()
    offers["type"] = offers.value.map(lambda x: offer_type[x.get("offer id", x.get("offer_id"))])
    # t=168 provides a fully observed 7-day baseline and follow-up for every user.
    anchors = offers[(offers.time == 168) & offers.type.isin(["discount", "bogo"])].copy()
    assert not anchors.person.duplicated().any()
    tx = events[events.event.eq("transaction")]
    # Exclude timestamp 168: the data cannot order events with equal timestamps.
    pre = tx[(tx.time > 0) & (tx.time < 168)].groupby("person").size()
    post = tx[(tx.time > 168) & (tx.time < 336)].groupby("person").size()
    pairs = anchors[["person", "type"]].copy()
    pairs["pre"] = pairs.person.map(pre).fillna(0).astype(int)
    pairs["post"] = pairs.person.map(post).fillna(0).astype(int)
    previous = set(offers.loc[offers.time < 168, "person"])
    pairs["prior_offer"] = pairs.person.isin(previous)
    result = summarize_pairs("starbucks", pairs)
    result["prior_offer_n"] = int(pairs.prior_offer.sum())
    result["at_anchor_transaction_excluded"] = int(tx[tx.time.eq(168) & tx.person.isin(pairs.person)].shape[0])
    result["by_type"] = {k: {"n": len(v), "pre": v.pre.mean(), "post": v.post.mean()} for k, v in pairs.groupby("type")}
    users = pd.read_json(raw / "profile.json", lines=True).id
    received_at_anchor = set(offers.loc[offers.time == 168, "person"])
    controls = pd.DataFrame({"person": users[~users.isin(received_at_anchor)]})
    controls["pre"] = controls.person.map(pre).fillna(0).astype(int)
    controls["post"] = controls.person.map(post).fillna(0).astype(int)
    controls["prior_offer"] = controls.person.isin(previous)
    result["no_offer_at_anchor"] = {
        "n": len(controls), "pre_mean": controls.pre.mean(), "post_mean": controls.post.mean(),
        "prior_offer_n": int(controls.prior_offer.sum()),
    }
    return result


def journey():
    days = pd.read_csv(PAIRS_OUT / "journey_daily.csv", dtype={"household_id": str}, parse_dates=["day"])
    redeems = pd.read_csv(PAIRS_OUT / "journey_redemptions.csv", dtype={"household_id": str}, parse_dates=["redemption_date"])
    lo, hi = days.day.min(), days.day.max()
    eligible = redeems[(redeems.redemption_date >= lo + pd.Timedelta(days=14)) &
                       (redeems.redemption_date <= hi - pd.Timedelta(days=14))]
    anchors = eligible.sort_values("redemption_date").drop_duplicates("household_id")
    joined = days.merge(anchors[["household_id", "redemption_date"]], on="household_id", how="inner")
    joined["diff"] = (joined.day - joined.redemption_date).dt.days
    pre = joined[joined['diff'].between(-14, -1)].groupby("household_id").baskets.sum()
    post = joined[joined['diff'].between(1, 14)].groupby("household_id").baskets.sum()
    pairs = anchors[["household_id", "redemption_date"]].copy()
    pairs["pre"] = pairs.household_id.map(pre).fillna(0).astype(int)
    pairs["post"] = pairs.household_id.map(post).fillna(0).astype(int)
    result = summarize_pairs("complete_journey", pairs)
    result.update({"first_data_day": str(lo.date()), "last_data_day": str(hi.date()),
                   "redemption_households_total": int(redeems.household_id.nunique()),
                   "anchor_day_excluded": True})
    return result


def run_persona_patterns():
    result = {"tianchi": online(), "starbucks": purchase_history("starbucks"),
              "complete_journey": purchase_history("complete_journey")}
    (PERSONA_OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_paired_windows():
    output = [starbucks(), journey()]
    (PAIRS_OUT / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("reactivation", "persistence", "long-term", "persona", "persona-proxies", "persona-uplift", "first-receipt-alignment", "persona-coupon-matching", "tianchi-long-term-lifecycle", "canonical-proxies", "single-receipt-duration", "coupon-overlap", "cumulative-coupons", "uplift-audit", "paired"))
    args = parser.parse_args()
    {"reactivation": run_reactivation, "persistence": run_persistence,
     "long-term": run_long_term, "persona": run_persona_patterns, "persona-proxies": run_tianchi_persona_proxies,
     "persona-uplift": run_tianchi_persona_uplift,
     "first-receipt-alignment": run_tianchi_first_receipt_alignment,
     "persona-coupon-matching": run_tianchi_persona_coupon_matching,
     "tianchi-long-term-lifecycle": run_tianchi_long_term_lifecycle,
     "canonical-proxies": run_tianchi_canonical_proxy_estimation,
     "single-receipt-duration": run_tianchi_single_receipt_duration,
     "coupon-overlap": run_tianchi_coupon_overlap,
     "cumulative-coupons": run_tianchi_cumulative_coupon_history,
     "uplift-audit": run_tianchi_uplift_audit,
     "paired": run_paired_windows}[args.task]()


if __name__ == "__main__":
    main()
