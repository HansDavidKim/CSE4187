"""Re-query immutable generated records under a shorter outcome definition."""
from __future__ import annotations

import copy
import math
import numbers

import numpy as np
import pandas as pd

GENERATION_WINDOW_DAYS = 30


def experiment_window_days(metadata, key):
    """Resolve a declared experiment window or the original 30-day default."""
    value = metadata.get(key, GENERATION_WINDOW_DAYS)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value not in (28, 30):
        raise ValueError(f"{key} must be 28 or 30")
    return int(value)


def _utc(value):
    result = pd.to_datetime(value, utc=True, errors="raise")
    if isinstance(result, pd.Series) and result.isna().any():
        raise ValueError("Outcome timestamps must be known")
    return result


def _first_session_share(profile, group, outcome_days):
    """P(min(positive idle gap, 30d*U) < outcome_days), conditional on activity."""
    gaps = profile.get("idle_gap_seconds") or group.get("idle_gap_seconds") or [1.0]
    positive = [float(value) for value in gaps if not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0]
    positive = positive or [1.0]
    tail = sum(value >= outcome_days * 86400 for value in positive) / len(positive)
    return 1 - (GENERATION_WINDOW_DAYS - outcome_days) / GENERATION_WINDOW_DAYS * tail


def redefine_outcomes(tables, metadata, outcome_days=28, shopping_model=None, visit_timing_model=None, *,
                      eligibility_days=None):
    """Return new labels/diagnostics; share unchanged raw tables and assignments.

    This is an analysis view of the existing 30-day generation law, not a new
    fitted hazard, intervention or price model. Revenue30 and mature CLV targets
    retain their original definitions.
    """
    if isinstance(outcome_days, bool) or not isinstance(outcome_days, numbers.Integral) or not 1 <= outcome_days <= GENERATION_WINDOW_DAYS:
        raise ValueError("outcome_days must be an integer from 1 through 30")
    eligibility_days = experiment_window_days(
        metadata if eligibility_days is None else {"eligibility_window_days": eligibility_days},
        "eligibility_window_days",
    )
    definition = f"future_no_session_{int(outcome_days)}d_active{eligibility_days}d"
    if "labels" not in tables or "treatments" not in tables:
        raise ValueError("Outcome redefinition requires original labels and factual assignments")
    if visit_timing_model is not None:
        from simulate.v4.irl.fit_visit_timing import first_session_share, validate_model
        validate_model(visit_timing_model)
    labels = tables["labels"].copy()
    if not {"customer_id", "as_of_at", "future_revenue_30d"} <= set(labels):
        raise ValueError("Original canonical labels must retain explicit future_revenue_30d")
    labels["as_of_at"] = _utc(labels.as_of_at)
    if labels.duplicated(["customer_id", "as_of_at"]).any():
        raise ValueError("Duplicate customer/snapshot labels")
    observation_end = _utc(metadata["observation_end_at"])
    observation_start = _utc(metadata["observation_start_at"])
    if labels.as_of_at.lt(observation_start).any() or labels.as_of_at.gt(observation_end).any():
        raise ValueError("Outcome snapshot lies outside the observed history")
    session_times = _utc(tables["sessions"].started_at)
    purchases = tables["events"].loc[tables["events"].event_type.eq("purchase"), ["event_id", "customer_id", "event_at"]].copy()
    purchases["event_at"] = _utc(purchases.event_at)
    orders = tables["orders"][["purchase_event_id", "paid_minor"]].merge(purchases, left_on="purchase_event_id", right_on="event_id", validate="one_to_one")
    retained = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    revenue = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    for t0, indices in labels.groupby("as_of_at", sort=False).groups.items():
        end = t0 + pd.Timedelta(days=int(outcome_days))
        if end > observation_end:
            continue
        active = set(tables["sessions"].loc[session_times.ge(t0) & session_times.lt(end), "customer_id"])
        paid = orders.loc[orders.event_at.ge(t0) & orders.event_at.lt(end)].groupby("customer_id").paid_minor.sum()
        customers = labels.loc[indices, "customer_id"]
        retained.loc[indices] = customers.isin(active).astype("Int8")
        revenue.loc[indices] = customers.map(paid).fillna(0).astype("Int64")
    labels["retained"] = retained
    labels["future_retained"] = retained
    labels["churn"] = 1 - retained
    revenue_column = f"future_revenue_{int(outcome_days)}d"
    labels[revenue_column] = revenue
    labels["future_revenue"] = revenue
    labels["outcome_end_at"] = labels.as_of_at + pd.Timedelta(days=int(outcome_days))
    labels["outcome_window_days"] = int(outcome_days)
    labels["eligibility_window_days"] = eligibility_days
    labels["definition_id"] = definition
    labels["is_censored"] = labels.churn.isna()
    labels["observation_status"] = np.where(labels.is_censored, "immature", "complete")

    result = {**tables, "labels": labels}
    new_metadata = copy.deepcopy(metadata)
    parent = metadata.get("parent_dataset_id", metadata["dataset_id"])
    suffix = ""
    if metadata.get("session_completion") is not None:
        from ..session_observations import completion_digest, completion_exclusions
        completion_exclusions(metadata, tables["sessions"].session_id)
        suffix = "-completion-" + completion_digest(metadata["session_completion"])
    generation_definition = metadata.get("generation_definition_id", metadata.get("definition_id"))
    new_metadata.update({
        "dataset_id": f"{parent.removesuffix(suffix)}-{int(outcome_days)}d{suffix}", "parent_dataset_id": parent,
        "generation_definition_id": generation_definition,
        "generation_policy_id": metadata.get("generation_policy_id", metadata.get("policy_id")),
        "generation_target_churn_rate": metadata.get("generation_target_churn_rate", metadata.get("target_churn_rate")),
        "generation_window_days": GENERATION_WINDOW_DAYS,
        "outcome_window_days": int(outcome_days), "eligibility_window_days": eligibility_days,
        "definition_id": definition,
        "outcome_revenue_column": revenue_column,
        "analysis_view": "immutable 30d generation queried under a shorter observation window",
        "analysis_target_recalibrated": False,
        "observed_churn_rate": float(labels.churn.mean()) if labels.churn.notna().any() else None,
        "generation_observed_churn_rate": metadata.get("generation_observed_churn_rate", metadata.get("observed_churn_rate")),
        "generation_expected_control_churn_rate": metadata.get("generation_expected_control_churn_rate", metadata.get("expected_control_churn_rate")),
    })
    for field in ("expected_control_churn_rate", "expected_treatment_churn_rate", "expected_churn_rate"):
        new_metadata[f"generation_{field}"] = metadata.get(f"generation_{field}", metadata.get(field))
        new_metadata[field] = None
    assignment = tables["treatments"][["customer_id", "as_of_at", "treatment"]].copy()
    assignment["as_of_at"] = _utc(assignment.as_of_at)
    observed = labels.merge(assignment, on=["customer_id", "as_of_at"], validate="one_to_one")
    for arm, name in ((0, "control"), (1, "treatment")):
        values = observed.loc[observed.treatment.eq(arm), "churn"]
        new_metadata[f"observed_{name}_churn_rate"] = float(values.mean()) if values.notna().any() else None

    if "diagnostics" in tables:
        diagnostics = tables["diagnostics"].copy()
        contact_days = (metadata.get("coupon_effect_calibration") or {}).get("contact_days")
        if contact_days is not None and visit_timing_model is None:
            raise ValueError("calibrated return probabilities require the fitted visit clock")
        profiles = {str(profile["profile_id"]): profile for profile in (shopping_model or {}).get("profiles", [])}
        groups = (shopping_model or {}).get("home_groups", {})
        timing_key = ("home_persona" if (visit_timing_model or {}).get("profile_key_scope") == "synthetic_home_group"
                      else "source_profile_id")
        shares, short_contact_fractions, fallbacks = [], [], 0
        for row in diagnostics.itertuples():
            if visit_timing_model is not None:
                key = str(getattr(row, timing_key, ""))
                share = first_session_share(visit_timing_model, key, True, int(outcome_days))
                shares.append(share)
                if contact_days is not None and outcome_days < contact_days:
                    contact_share = first_session_share(visit_timing_model, key, True, contact_days)
                    short_contact_fractions.append(share / contact_share if contact_share else 0.)
                continue
            profile = profiles.get(str(getattr(row, "source_profile_id", "")), {})
            group_name = profile.get("home_group", getattr(row, "home_persona", None))
            group = groups.get(group_name, {})
            if not profile.get("idle_gap_seconds") and not group.get("idle_gap_seconds"):
                fallbacks += 1
            shares.append(_first_session_share(profile, group, int(outcome_days)))
        diagnostics[f"first_session_within_{int(outcome_days)}d_share"] = shares
        for prefix in ("no_coupon", "coupon"):
            original = f"{prefix}_return_probability_30d"
            if original in diagnostics:
                probability = pd.to_numeric(diagnostics[original], errors="raise").astype(float)
                if probability.isna().any() or not probability.between(0, 1).all():
                    raise ValueError("Original analytic return probabilities must be in [0,1]")
                marginal = probability * np.asarray(shares)
                if prefix == "coupon" and contact_days is not None:
                    baseline = diagnostics["no_coupon_return_probability_30d"]
                    fraction = np.asarray(short_contact_fractions) if outcome_days < contact_days else 1.
                    marginal = (baseline * np.asarray(shares) + (probability - baseline) * fraction).clip(0., 1.)
                diagnostics[f"{prefix}_return_probability_{int(outcome_days)}d"] = marginal
        result["diagnostics"] = diagnostics
        required = {f"no_coupon_return_probability_{int(outcome_days)}d", f"coupon_return_probability_{int(outcome_days)}d"}
        if required <= set(diagnostics):
            analytic = assignment.merge(diagnostics[["customer_id", *sorted(required)]], on="customer_id", how="left", validate="one_to_one")
            if analytic[list(required)].isna().any().any():
                raise ValueError("Analytic diagnostics must cover every assigned customer")
            control = analytic[f"no_coupon_return_probability_{int(outcome_days)}d"]
            treated = analytic[f"coupon_return_probability_{int(outcome_days)}d"]
            factual = np.where(analytic.treatment.eq(1), treated, control)
            new_metadata["expected_control_churn_rate"] = float(1 - control.mean())
            new_metadata["expected_treatment_churn_rate"] = float(1 - treated.mean())
            new_metadata["expected_churn_rate"] = float(np.mean(1 - factual))
        new_metadata["analytic_outcome_view"] = ({
            "formula": ("p(days)=p30*P(first_synthetic_session_offset<days | home_group,active_previous30=True)"
                        if timing_key == "home_persona" else "p(days)=p30*P(first_native_session_offset<days | source_profile,active_previous30=True)"),
            "generation_window_days": GENERATION_WINDOW_DAYS, "outcome_window_days": int(outcome_days),
            "timing_source": ("the same synthetic home-group/active30 schedules as the runtime sampler"
                              if timing_key == "home_persona" else "exact same profile/active30 source-frequency-weighted regular bank as runtime sampler"),
            "timing_model_type": visit_timing_model['type'],
            "interpretation": "native conditional first-offset CDF of the actual30d timing law; not newly learned28d hazard coefficients",
        } if visit_timing_model is not None else {
            "formula": "p(days)=p30*[1-(30-days)/30*P(positive_idle_gap>=days)]",
            "generation_window_days": GENERATION_WINDOW_DAYS, "outcome_window_days": int(outcome_days),
            "gap_source": "profile idle gaps, else home-group gaps, else explicit one-second fallback",
            "one_second_fallback_customers": fallbacks,
            "interpretation": "analytic query of original 30d activity law; not newly learned 28d coefficients",
        })
        if contact_days is not None:
            new_metadata["analytic_outcome_view"]["formula"] = (
                "p0(days)=p0(30)*F(days); p1(days)=p0(days)+(p1(30)-p0(30))*F(min(days,contact_days))/F(contact_days)"
            )
    if "_metadata" in tables:
        result["_metadata"] = new_metadata
    return result, new_metadata
