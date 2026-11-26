"""Held-out factual Qini and independent randomized strategy-group comparisons."""
import math
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from .experiment_validation import _flag, _sha

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import binomtest, fisher_exact, norm, t as student_t

from .weekly_inputs import join_weekly_predictions
from .weekly_validation import KEYS, keys, numeric, require, utc

QINI_COLUMNS = ["model", "fraction", "cumulative_gain", "random_targeting_gain"]
STRATEGY_COLUMNS = ["strategy_group", "customers", "conversion_rate", "churn_rate", "net_revenue_per_customer",
                    "cost", "roi", "roi_reason", "primary_metric", "primary_value", "primary_ci95_lower",
                    "primary_ci95_upper", "difference_vs_A", "difference_ci95_lower", "difference_ci95_upper",
                    "p_value", "adjusted_p_value", "test", "inference_reason"]


def _unavailable(reason):
    return {"status": "unavailable", "reason": reason, "frame": pd.DataFrame(columns=QINI_COLUMNS)}


def _evaluation_reason(frame, metadata):
    if metadata.get("randomized_assignment") is not True or metadata.get("coupon_assignment", {}).get("randomized") is False:
        return "Factual assignment was not declared randomized"
    if metadata.get("evaluation_only") is not True:
        return "Evaluation-only cohort was not declared"
    development = metadata.get("development_customer_keys")
    if not isinstance(development, list):
        return "Model development customer keys were not supplied for overlap verification"
    known = {(str(row["dataset_id"]), str(row["customer_id"])) for row in development}
    if known & set(map(tuple, frame[["dataset_id", "customer_id"]].to_numpy())):
        return "Model development and strategy evaluation customers overlap"
    stamp = metadata.get("models_frozen_at")
    if stamp is None or pd.Timestamp(stamp).tzinfo is None:
        return "Timezone-aware model/strategy freeze time was not supplied"
    if "assigned_at" not in frame:
        return "Assignment timestamps were not supplied"
    assigned = utc(frame.assigned_at, "assigned_at")
    if not assigned.eq(frame.as_of_at).all() or assigned.lt(pd.Timestamp(stamp)).any():
        return "Models/strategy must be frozen before assignment at the prediction cutoff"
    if not {"outcome_end_at", "observation_end_at"} <= set(frame):
        return "Outcome maturity timestamps were not supplied"
    end = utc(frame.outcome_end_at, "outcome_end_at")
    if not end.eq(frame.as_of_at+pd.Timedelta(weeks=4)).all():
        return "Strategy/Qini results require the same exact 4-week outcome"
    if utc(frame.observation_end_at, "observation_end_at").lt(end).any():
        return "The factual 4-week outcome is not mature"
    return None


def qini_curves(observations, predictions, metadata):
    """IPW cumulative purchase gain and its straight random-targeting reference."""
    frame = keys(observations).reset_index(drop=True)
    reason = _evaluation_reason(frame, metadata)
    required = {"treatment", "assignment_probability", "purchased_28d"}
    if reason or not required <= set(frame):
        return _unavailable(reason or "Factual purchase/assignment fields are missing")
    treatment = pd.to_numeric(frame.treatment, errors="raise")
    outcome = pd.to_numeric(frame.purchased_28d, errors="raise")
    propensity = pd.to_numeric(frame.assignment_probability, errors="raise")
    require(treatment.isin([0, 1]).all() and outcome.isin([0, 1]).all(), "Qini requires complete binary factual treatment/purchase")
    if set(treatment) != {0, 1} or not propensity.between(0, 1, inclusive="neither").all():
        return _unavailable("Both randomized arms and strict propensity overlap are required")
    pseudo = outcome.to_numpy() * (treatment.to_numpy()/propensity.to_numpy()-(1-treatment.to_numpy())/(1-propensity.to_numpy()))
    rows, coefficients = [], {}
    for name, batch in predictions.items():
        scored = join_weekly_predictions(frame, {"uplift": batch})
        if scored.uplift_score.isna().any():
            return _unavailable(f"Missing held-out uplift score for {name}")
        score = scored.uplift_score.to_numpy().round(12)
        order = np.argsort(-score, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0)+1, len(frame)]
        starts = np.r_[0, ends[:-1]]
        blocks = [math.fsum(pseudo[order[start:end]]) for start, end in zip(starts, ends)]
        gain = np.r_[0., np.cumsum(blocks)/len(frame)]
        fraction = np.r_[0., ends/len(frame)]
        reference = fraction*gain[-1]
        coefficients[name] = float(trapezoid(gain-reference, fraction))
        rows.extend({"model": name, "fraction": float(x), "cumulative_gain": float(y),
                     "random_targeting_gain": float(r)} for x, y, r in zip(fraction, gain, reference))
    if not rows:
        return _unavailable("No model purchase predictions supplied")
    return {"status": "available", "reason": None, "frame": pd.DataFrame(rows), "qini_coefficients": coefficients,
            "target": "4-week factual purchase", "gain_definition": "cumulative inverse-propensity transformed purchase outcome / cohort size",
            "ties": "one point at each tied score-block end; scores rounded to12 decimal places; no arbitrary within-tie ordering",
            "random_targeting_definition": "fraction times the full-cohort IPW purchase effect; not an RL coupon policy"}


def _mean_interval(values, binary):
    values = np.asarray(values, float)
    if len(values) == 0 or not np.isfinite(values).all():
        return None, None, None
    mean = float(values.mean())
    if binary:
        ci = binomtest(int(values.sum()), len(values)).proportion_ci(.95, method="wilson")
        return mean, float(ci.low), float(ci.high)
    if len(values) < 2:
        return mean, None, None
    margin = float(student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return mean, mean-margin, mean+margin


def _difference(a, b, binary):
    if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None, None, None, None, "At least two complete outcomes per group are required"
    ma, la, ua = _mean_interval(a, binary)
    mb, lb, ub = _mean_interval(b, binary)
    diff = ma-mb
    if binary:
        # Wilson component bounds avoid zero-width Wald intervals for all-zero arms.
        lo = diff-math.sqrt((ma-la)**2+(ub-mb)**2)
        hi = diff+math.sqrt((ua-ma)**2+(mb-lb)**2)
        pooled = (a.sum()+b.sum())/(len(a)+len(b))
        expected = np.array([len(a)*pooled, len(a)*(1-pooled), len(b)*pooled, len(b)*(1-pooled)])
        if (expected < 5).any():
            p = float(fisher_exact([[int(a.sum()), len(a)-int(a.sum())], [int(b.sum()), len(b)-int(b.sum())]]).pvalue)
            test = "Fisher exact (small expected cell count)"
        else:
            se = math.sqrt(pooled*(1-pooled)*(1/len(a)+1/len(b)))
            p = float(2*norm.sf(abs(diff/se)))
            test = "two-proportion z (same alpha as legacy ITT)"
        return diff, max(-1., lo), min(1., hi), p, test
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    if va+vb == 0:
        return diff, None, None, None, "Constant observed revenue; sampling interval/test unavailable"
    df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    se = math.sqrt(va+vb)
    margin = float(student_t.ppf(.975, df)*se)
    return diff, diff-margin, diff+margin, float(2*student_t.sf(abs(diff/se), df)), "Welch t"


def _holm(values):
    order = np.argsort([1. if x is None else x for x in values], kind="stable")
    result, previous = [None]*len(values), 0.
    for rank, index in enumerate(order):
        adjusted = min(1., max(previous, (len(values)-rank)*(1. if values[index] is None else values[index])))
        previous = adjusted
        if values[index] is not None:
            result[index] = adjusted
    return result


def _itt_reason(frame, assignments):
    if assignments is None:
        return "Original strategy assignment roster was not supplied"
    original = keys(assignments)
    identity = [*KEYS, "snapshot_id", "raw_database_sha256", "assigned_at", "strategy_group"]
    if not set(identity) <= set(original) & set(frame) or "assignment_probability" not in original:
        return "Original assignment identity or strategy assignment probability is missing"
    original["assigned_at"] = utc(original.assigned_at, "assigned_at")
    frame["assigned_at"] = utc(frame.assigned_at, "assigned_at")
    for table in (original, frame):
        require(table[identity].notna().all().all(), "Missing strategy assignment identity")
        require(table.raw_database_sha256.nunique() == 1, "Mixed assignment ledgers")
        _sha(table.raw_database_sha256.iloc[0], "assignment ledger")
        require(table.snapshot_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                "Invalid strategy assignment snapshot")
    prob = original.assignment_probability
    require(not prob.map(lambda x: isinstance(x, (bool, np.bool_))).any()
            and pd.to_numeric(prob, errors="raise").between(0, 1, inclusive="neither").all(),
            "Invalid strategy assignment probability")
    if "assignment_probability" in frame:
        identity.append("assignment_probability")
    match = frame[identity].merge(original[identity], on=identity, how="outer",
                                  indicator=True, validate="one_to_one")
    require(match._merge.eq("both").all(), "Outcomes differ from the original assignment roster")
    return None


MONEY_FIELDS = ("gross_revenue_minor", "discount_minor", "send_cost_minor", "paid_revenue_minor",
                "paid_minor", "net_revenue_minor", "net_after_marketing_minor")


def _minor(value, field):
    require(not isinstance(value, (bool, np.bool_)), f"{field} requires integer minor units")
    try:
        amount = Decimal(str(value))
    except (ValueError, InvalidOperation) as error:
        raise ValueError(f"Invalid {field}") from error
    require(amount.is_finite() and amount == amount.to_integral_value()
            and (field == "net_after_marketing_minor" or amount >= 0),
            f"{field} requires finite integer minor units with valid sign")
    return int(amount)


def _accounting(frame, observed):
    # Parse before pandas arithmetic to avoid int64 overflow and float rounding.
    for field in MONEY_FIELDS:
        if field in frame:
            amounts = [None if pd.isna(v) and not complete else _minor(v, field)
                       for v, complete in zip(frame[field], observed)]
            frame[field] = pd.Series(amounts, index=frame.index, dtype=object)
    if "send_cost_minor" in frame and "actual_sent" in frame:
        sent = frame.actual_sent.map(lambda value: _flag(value, "actual_sent"))
        require(not any(x is not None and x > 0 for x in frame.loc[~sent, "send_cost_minor"]),
                "unsupported positive charge without a successful send")
    if not set(MONEY_FIELDS[:3]) <= set(frame):
        return False
    for row in frame.loc[observed].itertuples():
        paid = row.gross_revenue_minor - row.discount_minor
        require(paid >= 0, "Discount exceeds factual gross revenue")
        for field in MONEY_FIELDS[3:]:
            expected = paid - row.send_cost_minor if field == "net_after_marketing_minor" else paid
            if field in frame:
                require(getattr(row, field) == expected,
                        f"Factual {field} contradicts gross, discount or send cost; paid/net cannot substitute for gross")
        require(pd.isna(row.purchased_28d) or row.purchased_28d != 0 or row.gross_revenue_minor == 0,
                "Factual 4-week gross revenue contradicts no purchase")
    require({"currency", "amount_unit"} <= set(frame)
            and frame.currency.notna().all() and frame.amount_unit.notna().all()
            and frame.currency.nunique() == frame.amount_unit.nunique() == 1,
            "Factual financial units must be explicit and consistent")
    return bool(observed.all() and "actual_sent" in frame)


def strategy_evaluation(observations, metadata, *, primary_metric="churn_rate", assignments=None):
    """Describe assigned groups; causal ROI, CI and p-values need a registered held-out RCT."""
    require(primary_metric in ("churn_rate", "conversion_rate", "net_revenue_per_customer"), "Unknown primary metric")
    baseline = "A"
    frame = keys(observations).reset_index(drop=True)
    require("strategy_group" in frame and frame.strategy_group.notna().all(), "Actual strategy-group assignment is required")
    planned = metadata.get("strategy_groups", sorted(frame.strategy_group.astype(str).unique()))
    require(all(isinstance(group, str) and bool(group) for group in planned)
            and len(planned) == len(set(planned)) and set(frame.strategy_group) <= set(planned)
            and baseline in planned, "Strategy group family must explicitly include baseline A")
    roster_reason = _itt_reason(frame, assignments)
    reason = _evaluation_reason(frame, metadata) or roster_reason
    observed = pd.Series(True, index=frame.index)
    if "outcome_observed" in frame:
        observed &= frame.outcome_observed.map(lambda x: _flag(x, "outcome_observed"))
    else:
        reason = reason or "Outcome completion flags were not supplied"
    if "is_censored" in frame:
        observed &= ~frame.is_censored.map(lambda x: _flag(x, "is_censored"))
    if "censoring_reason" in frame:
        observed &= frame.censoring_reason.isna()
    if not observed.all():
        reason = reason or "Censored or incomplete strategy outcomes"
    if metadata.get("strategy_randomized_assignment") is not True:
        reason = reason or "Customer-level strategy-group randomization was not declared; coupon randomization alone is insufficient"
    if "strategy_groups" not in metadata:
        reason = reason or "Predeclared strategy-group comparison family was not supplied"
    if metadata.get("primary_metric") != primary_metric:
        reason = reason or "The evaluated primary metric was not declared before assignment"
    if baseline not in set(frame.strategy_group):
        reason = "Baseline A has no observed customers"
    frame["strategy_group"] = frame.strategy_group.astype(str)
    for field in ("purchased_28d", "churn"):
        if field not in frame:
            frame[field] = np.nan
        else:
            require(frame[field].dropna().isin([0, 1]).all(), f"{field} must be binary factual outcomes")
    frame.loc[~observed, ["purchased_28d", "churn"]] = np.nan
    money = _accounting(frame, observed) if metadata.get("source_monetary_values_observed", True) is not False else False
    if frame[["purchased_28d", "churn"]].isna().any().any():
        reason = reason or "Incomplete factual strategy outcomes"
    frame["cost"] = frame.discount_minor+frame.send_cost_minor if money else np.nan
    frame["net"] = frame.gross_revenue_minor-frame.cost if money else np.nan
    column = {"churn_rate": "churn", "conversion_rate": "purchased_28d", "net_revenue_per_customer": "net"}[primary_metric]
    binary = primary_metric != "net_revenue_per_customer"
    baseline_frame = frame.loc[frame.strategy_group.eq(baseline)]
    rows, comparisons = [], []
    for group in planned:
        part = frame.loc[frame.strategy_group.eq(group)]
        mean, lo, hi = _mean_interval(part[column].to_numpy(float), binary)
        row = {"strategy_group": group, "customers": len(part),
               "conversion_rate": float(part.purchased_28d.mean()) if len(part) and part.purchased_28d.notna().all() else None,
               "churn_rate": float(part.churn.mean()) if len(part) and part.churn.notna().all() else None,
               "net_revenue_per_customer": float(part.net.mean()) if money and len(part) else None,
               "cost": sum(part.cost) if money and len(part) else None,
               "roi": None, "roi_reason": reason or ("No observed monetary outcomes" if not money else "Baseline or zero incremental cost"),
               "primary_metric": primary_metric, "primary_value": mean, "primary_ci95_lower": lo, "primary_ci95_upper": hi,
               "difference_vs_A": None, "difference_ci95_lower": None, "difference_ci95_upper": None,
               "p_value": None, "adjusted_p_value": None, "test": None, "inference_reason": reason}
        if reason:
            row["primary_ci95_lower"] = row["primary_ci95_upper"] = None
        if group == baseline:
            row["difference_vs_A"] = 0.
            row["test"] = "baseline reference; no test"
        if group != baseline:
            if not reason:
                left, right = part[column], baseline_frame[column]
                if not binary and money and len(right):
                    # Translation leaves Welch inference invariant. Subtract in integer minor
                    # units first so adjacent large cash amounts remain distinguishable.
                    origin = right.iloc[0]
                    left, right = left - origin, right - origin
                diff, dl, du, pv, test = _difference(left.to_numpy(float), right.to_numpy(float), binary)
                row.update(difference_vs_A=diff, difference_ci95_lower=dl, difference_ci95_upper=du, p_value=pv, test=test)
                if pv is None:
                    row["inference_reason"] = test
            if not reason and money and len(part) and len(baseline_frame):
                increment = Fraction(sum(part.cost), len(part)) - Fraction(sum(baseline_frame.cost), len(baseline_frame))
                if increment > 0:
                    net_change = Fraction(sum(part.net), len(part)) - Fraction(sum(baseline_frame.net), len(baseline_frame))
                    row["roi"] = float(net_change / increment)
                    row["roi_reason"] = None
            comparisons.append(row)
        rows.append(row)
    adjusted = _holm([row["p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["adjusted_p_value"] = value
    report = pd.DataFrame(rows)
    report["cost"] = pd.Series([row["cost"] for row in rows], dtype=object)
    return {"status": "available" if not reason else "descriptive_only", "reason": reason,
            "frame": report, "baseline": baseline, "primary_metric": primary_metric,
            "comparison_family": [f"{group} versus {baseline}: {primary_metric}" for group in planned if group != baseline],
            "multiple_comparison": "Holm; family includes every predeclared non-baseline group",
            "alpha": .05, "confidence_level": .95, "binary_ci_method": "Newcombe difference of Wilson intervals",
            "roi_definition": "difference in mean (gross-discount-send fee) versus A / difference in mean (discount+send fee)",
            "sampling_unit": "one factual assigned customer; no counterfactual outcome rows"}
