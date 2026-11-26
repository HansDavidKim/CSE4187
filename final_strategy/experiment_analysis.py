"""Analyze static five-arm ledgers using Phase 2's input contract.

Results describe the supplied records. Statistical calculations do not verify
the experiment's assignment, execution, or causal interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from decimal import Context, Decimal, localcontext
import math
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from scipy.stats import norm

from .experiment_power import PowerPlan
from .experiment_spec import ExperimentSpec
from .experiment_validation import validate_experiment_run
from .validation import InputValidationError


KEYS = ("customer_id", "snapshot_id", "evaluation_dataset_id")
CONFIDENCE_LEVEL = 0.95
CHURN_METRIC = "no_session_churn_28d"


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_copy(value):
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_copy(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentAnalysisResult:
    experiment_id: str
    experiment_version: str
    evaluation_dataset_id: str
    as_of_at: str
    currency: str
    amount_unit: str
    provenance: Mapping
    validation: Mapping
    metrics: Mapping
    by_arm: Mapping
    primary_comparisons: tuple[Mapping, ...]
    secondary_comparisons: tuple[Mapping, ...]
    exploratory_churn_comparisons: tuple[Mapping, ...]
    holm: Mapping
    causal_interpretation: str
    limitations: tuple[str, ...]

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))

    def to_dict(self) -> dict:
        """Return a detached object suitable for strict JSON serialization."""
        return {field.name: _json_copy(getattr(self, field.name)) for field in fields(self)}


def _snapshot(value):
    """Copy nested declarations and frames while preserving object-column types."""
    if isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
        # pandas deep copies its arrays, but does not recursively copy objects
        # stored inside an object array. Preserve the original dtype and position.
        for index, dtype in enumerate(frame.dtypes):
            if dtype == object:
                frame.isetitem(index, pd.Series(
                    [_snapshot(item) for item in frame.iloc[:, index]],
                    index=frame.index, dtype=object,
                ))
        return frame
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return deepcopy(value)


def _per_customer(total, denominator):
    """Decimal display with explicit precision; total and denominator stay exact."""
    if total is None or not denominator:
        return None
    with localcontext(Context(prec=34)):
        return str(Decimal(total) / Decimal(denominator))


def _arm_summary(group, validated):
    observed = group.loc[group.outcome_observed.astype(bool)]
    n_itt, n_observed = int(len(group)), int(len(observed))
    n_censored = n_itt - n_observed
    if (n_itt, n_observed, n_censored) != (
            validated["itt"], validated["observed"], validated["censored"]):
        raise InputValidationError("experiment_analysis: row counts differ from validated ITT")
    complete = n_censored == 0
    purchases = sum(bool(value) for value in observed.purchase_occurred)
    churn = sum(not bool(value) for value in observed.session_occurred)
    paid_observed = sum(int(value) for value in observed.paid_minor)
    paid_total = paid_observed if complete else None
    discount = validated["observed_discount_minor"] if complete else None
    spend = validated["realized_spend_minor"]
    expected = validated["selected_expected_cost"]
    variance = None
    if spend is not None:
        with localcontext(Context(prec=len(str(spend)) + len(expected) + 10)):
            variance = str(Decimal(spend) - Decimal(expected))
    return {
        "n_itt": n_itt, "n_observed": n_observed, "n_censored": n_censored,
        "observation_rate": n_observed / n_itt if n_itt else None,
        "itt_observation_complete": complete,
        "purchase_customers_observed": purchases, "churn_customers_observed": churn,
        "purchase_rate_itt": purchases / n_itt if complete and n_itt else None,
        "churn_rate_itt": churn / n_itt if complete and n_itt else None,
        "observed_only_descriptive": {
            "denominator": n_observed,
            "purchase_rate": purchases / n_observed if n_observed else None,
            "churn_rate": churn / n_observed if n_observed else None,
        },
        "n_requested": validated["requested"], "n_sent": validated["sent"],
        "n_failed": validated["failed"], "n_unsent": validated["unsent"],
        "sample_plan": validated["sample_plan"],
        "money": {
            "itt_denominator": n_itt, "observed_denominator": n_observed,
            "observed_paid_minor": paid_observed, "itt_paid_minor": paid_total,
            "observed_paid_minor_per_customer": _per_customer(paid_observed, n_observed),
            "itt_paid_minor_per_customer": _per_customer(paid_total, n_itt),
            "send_cost_minor": validated["send_cost_minor"],
            "observed_coupon_discount_minor": validated["observed_discount_minor"],
            "itt_coupon_discount_minor": discount,
            "known_spend_minor": validated["known_spend_minor"], "realized_spend_minor": spend,
            "realized_spend_minor_per_customer": _per_customer(spend, n_itt),
            "expected_cost_minor": expected, "budget_minor": validated["budget"],
            "realized_minus_expected_minor": variance,
            "plan_within_budget": validated["plan_within_budget"],
            "realized_budget_status": validated["realized_budget_status"],
            "per_customer_decimal_precision": 34,
        },
    }


def _wilson(successes, size, critical):
    rate = successes / size
    z2 = critical**2
    divisor = 1 + z2 / size
    center = (rate + z2 / (2 * size)) / divisor
    margin = critical * math.sqrt(rate * (1 - rate) / size + z2 / (4 * size**2)) / divisor
    return min(rate, max(0.0, center - margin)), max(rate, min(1.0, center + margin))


def _newcombe(successes_1, size_1, successes_2, size_2):
    """Newcombe method 10, with no continuity correction."""
    p1, p2 = successes_1 / size_1, successes_2 / size_2
    critical = float(norm.isf((1 - CONFIDENCE_LEVEL) / 2))
    l1, u1 = _wilson(successes_1, size_1, critical)
    l2, u2 = _wilson(successes_2, size_2, critical)
    delta = p1 - p2
    return (max(-1.0, delta - math.hypot(p1 - l1, u2 - p2)),
            min(1.0, delta + math.hypot(u1 - p1, p2 - l2)))


def _z_test(successes_1, size_1, successes_2, size_2, difference):
    size = size_1 + size_2
    events = successes_1 + successes_2
    non_events = size - events
    if not events or not non_events:
        return None, None, "pooled_rate_at_boundary"
    # Compare the four expected frequencies to 5 without rounding at the cutoff.
    if min(size_1, size_2) * min(events, non_events) < 5 * size:
        return None, None, "expected_cell_below_five"
    se = math.sqrt((events / size) * (non_events / size) * (1 / size_1 + 1 / size_2))
    if not math.isfinite(se) or se <= 0:
        return None, None, "nonfinite_statistic"
    z = difference / se
    p = float(2 * norm.sf(abs(z)))
    if not math.isfinite(z) or not 0 <= p <= 1:
        return None, None, "nonfinite_statistic"
    return z, p, None


def _comparison(pair, metric, scope, by_arm, nominal_alpha):
    baseline, comparison = pair
    a, b = by_arm[baseline], by_arm[comparison]
    churn = metric == CHURN_METRIC
    count_field = "churn_customers_observed" if churn else "purchase_customers_observed"
    rate_field = "churn_rate_itt" if churn else "purchase_rate_itt"
    direction = "baseline_minus_comparison" if churn else "comparison_minus_baseline"
    result = {
        "baseline_arm": baseline, "comparison_arm": comparison, "metric": metric, "scope": scope,
        "direction": direction,
        "baseline_n_itt": a["n_itt"], "comparison_n_itt": b["n_itt"],
        "baseline_n_observed": a["n_observed"], "comparison_n_observed": b["n_observed"],
        "baseline_events_observed": a[count_field], "comparison_events_observed": b[count_field],
        "baseline_rate_itt": a[rate_field], "comparison_rate_itt": b[rate_field],
        "itt_observation_complete": a["itt_observation_complete"] and b["itt_observation_complete"],
        "absolute_difference": None, "absolute_difference_unit": "proportion",
        "absolute_difference_display_unit": "%p", "relative_change": None,
        "relative_change_unit": "ratio", "relative_change_display_unit": "%",
        "confidence_interval": None, "confidence_level": CONFIDENCE_LEVEL,
        "confidence_interval_method": "newcombe_wilson", "multiplicity_adjusted": False,
        "test_method": "two_proportion_z_pooled_two_sided",
        "z_statistic": None, "raw_p_value": None, "test_status": "not_testable", "reason": None,
    }
    if scope == "primary":
        result.update(adjusted_p_value=None, confirmatory_decision="withheld")
    else:
        result.update(nominal_alpha=nominal_alpha, nominal_significant=None)
    if not a["n_itt"] or not b["n_itt"]:
        result["reason"] = "empty_arm"
        return result
    if not result["itt_observation_complete"]:
        result["reason"] = "incomplete_itt_observation"
        return result

    first, second = (a, b) if churn else (b, a)
    difference = first[rate_field] - second[rate_field]
    args = (first[count_field], first["n_itt"], second[count_field], second["n_itt"])
    ci = _newcombe(*args)
    if not all(math.isfinite(limit) for limit in ci) or not -1 <= ci[0] <= ci[1] <= 1:
        raise InputValidationError("experiment_analysis: invalid Newcombe–Wilson interval")
    z, p, reason = _z_test(*args, difference)
    baseline_rate = a[rate_field]
    result.update(absolute_difference=difference,
                  relative_change=difference / baseline_rate if baseline_rate else None,
                  confidence_interval=ci, z_statistic=z, raw_p_value=p, reason=reason,
                  test_status="tested" if reason is None else "not_testable")
    if scope != "primary" and p is not None:
        result["nominal_significant"] = p < nominal_alpha
    return result


def _apply_holm(primary, alpha):
    unavailable = tuple(
        {"baseline_arm": row["baseline_arm"], "comparison_arm": row["comparison_arm"],
         "reason": row["reason"]}
        for row in primary if row["raw_p_value"] is None
    )
    if unavailable:
        return {"status": "withheld", "family_alpha": alpha, "unavailable_comparisons": unavailable}
    order = sorted(range(len(primary)), key=lambda i: primary[i]["raw_p_value"])
    previous = 0.0
    for rank, i in enumerate(order):
        row = primary[i]
        adjusted = min(1.0, max(previous, (len(primary) - rank) * row["raw_p_value"]))
        row.update(adjusted_p_value=adjusted,
                   confirmatory_decision="reject" if adjusted < alpha else "do_not_reject")
        previous = adjusted
    return {"status": "applied", "family_alpha": alpha, "unavailable_comparisons": ()}


def analyze_experiment(
    spec: ExperimentSpec, manifest: Mapping, *, population: pd.DataFrame,
    assignments: pd.DataFrame, decisions: pd.DataFrame, executions: pd.DataFrame,
    outcomes: pd.DataFrame, power_plan: PowerPlan | None = None,
    features: pd.DataFrame | str | Path | None = None,
    training_customer_ids=None, policy_evidence: Mapping | None = None,
) -> ExperimentAnalysisResult:
    """Analyze the same working copies that passed the Phase 2 validator.

    Callers must prevent concurrent mutation while inputs are copied. File and
    producer authenticity stay subject to the evidence checks in Phase 2.
    """
    manifest = _snapshot(manifest)
    ledgers = {name: _snapshot(frame) for name, frame in (
        ("population", population), ("assignments", assignments),
        ("decisions", decisions), ("executions", executions), ("outcomes", outcomes),
    )}
    validated = validate_experiment_run(
        spec, manifest, **ledgers, power_plan=power_plan, features=_snapshot(features),
        training_customer_ids=_snapshot(training_customer_ids), policy_evidence=_snapshot(policy_evidence),
    )
    if validated.data_integrity != "passed":
        raise InputValidationError("experiment_analysis: ledger validation did not pass")
    joined = ledgers["assignments"][[*KEYS, "arm_id"]].merge(
        ledgers["outcomes"][[*KEYS, "outcome_observed", "purchase_occurred",
                            "session_occurred", "paid_minor"]],
        on=list(KEYS), validate="one_to_one",
    )
    groups = {arm: group for arm, group in joined.groupby("arm_id", sort=False, observed=True)}
    by_arm = {arm: _arm_summary(groups.get(arm, joined.iloc[:0]), validated.by_arm[arm])
              for arm, _ in spec.arms}
    alpha = float(spec.family_alpha)
    primary = tuple(_comparison(pair, spec.primary_metric, "primary", by_arm, alpha)
                    for pair in spec.primary_comparisons)
    holm = _apply_holm(primary, alpha)
    secondary = tuple(_comparison(pair, spec.primary_metric, "exploratory", by_arm, alpha)
                      for pair in spec.secondary_comparisons)
    churn = tuple(_comparison(pair, CHURN_METRIC, "exploratory", by_arm, alpha)
                  for pair in (*spec.primary_comparisons, *spec.secondary_comparisons))
    return ExperimentAnalysisResult(
        experiment_id=validated.experiment_id, experiment_version=validated.experiment_version,
        evaluation_dataset_id=validated.evaluation_dataset_id, as_of_at=validated.as_of_at,
        currency=validated.currency, amount_unit=validated.amount_unit,
        provenance={key: manifest[key] for key in
                    ("feature_raw_sha256", "result_raw_sha256", "ledger_sha256", "outcome_start_rule")},
        validation=validated.to_dict(), by_arm=by_arm, primary_comparisons=primary,
        secondary_comparisons=secondary, exploratory_churn_comparisons=churn, holm=holm,
        metrics={spec.primary_metric: "purchase_occurred", CHURN_METRIC: "not session_occurred",
                 "outcome_window_days": spec.outcome_days},
        causal_interpretation="unverified",
        limitations=(
            "Observed-only rates are descriptive and may be affected by informative censoring.",
            "Newcombe–Wilson intervals are per-comparison, with no multiplicity adjustment.",
            "Nominal exploratory p-values and primary statistical decisions do not establish causality.",
            "Observed money totals may be partial. Paid amounts are already net of coupon discounts.",
        ),
    )
