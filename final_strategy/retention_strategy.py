"""Development retention policies, observable history, and declared expected costs.

Targets the current v5 contract with default 28-day coupon validity.
The simulate.v4 package name is retained. Expected costs are caller-declared;
this module neither sends coupons nor changes simulator policies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Context, Decimal, DecimalException, Inexact, Rounded, localcontext
from itertools import product

import numpy as np
import pandas as pd

from . import budget as engine
from .budget import Budget, CostMetadata
from .segmentation import (SegmentationResult, CHURN_BANDS, UPLIFT_BANDS, CLV_BANDS,
                           SEGMENT_COLUMNS, SUMMARY_COLUMNS, MEAN_COLUMNS, _segment)
from .validation import CONTEXT_COLUMNS, JOIN_KEYS, InputValidationError

SEGMENT_IDS = tuple("__".join(x) for x in product(CHURN_BANDS, UPLIFT_BANDS, CLV_BANDS))
COUPON_COLUMNS = ("coupon_id", "customer_id", "sent_at", "expires_at", "discount_rate", "policy_id", "send_cost_minor")
REDEMPTION_COLUMNS = ("coupon_id", "customer_id", "purchased_at")
ELIGIBILITY_COLUMNS = (*CONTEXT_COLUMNS, "business_eligible", "business_reason", "history_complete")
COST_INPUT_COLUMNS = (*CONTEXT_COLUMNS, "send_cost", "redemption_probability", "expected_discount_if_redeemed",
                      "source_id", "source_version", "source_as_of_at")


@dataclass(frozen=True)
class SegmentPolicy:
    segment_id: str
    policy_id: str = "development-positive-uplift"
    version: str = "retention-dev-v1"
    allow_candidate: bool = True
    min_uplift: object = Decimal(0)
    exclusion_reason: str = "policy_disabled"
    send_delay: object = pd.Timedelta(0)


@dataclass(frozen=True)
class HistoryPolicy:
    # Explicit choices required; no generic resend restriction is inferred.
    block_active_unused: bool
    block_previously_redeemed: bool
    minimum_send_interval: object
    incomplete_history: str = "unknown"  # unknown, exclude, error
    expected_validity_days: int | None = 28  # v5 default; caller may declare other observed-history policies


@dataclass(frozen=True)
class HistoryContext:
    dataset_id: str
    raw_database_sha256: str
    observed_until_at: object
    source_id: str
    version: str
    known_at: object


@dataclass(frozen=True)
class RetentionResult:
    evaluation: pd.DataFrame
    cost_estimates: pd.DataFrame
    budget_inputs: SegmentationResult
    costs: pd.DataFrame
    cost_metadata: CostMetadata
    metadata: dict


def default_segment_policies():
    """All twelve labels supported; strict positive uplift, not business validation."""
    return tuple(SegmentPolicy(identifier) for identifier in SEGMENT_IDS)


def _fail(message):
    raise InputValidationError(f"retention: {message}")


def _columns(frame, required, name):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any() or set(frame) != set(required):
        _fail(f"{name} requires exactly {list(required)}")
    if "evaluation_only" in frame.attrs:
        _fail(f"{name} evaluation_only is forbidden")
    return frame.copy(deep=True).reset_index(drop=True)


def _duration(value, field):
    if isinstance(value, (bool, int, float, np.number)):
        _fail(f"{field} requires an explicit timedelta or duration string")
    try:
        result = pd.Timedelta(value)
        if pd.isna(result) or result < pd.Timedelta(0):
            raise ValueError()
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputValidationError(f"retention: invalid {field}") from exc


def _policies(policies):
    policies = default_segment_policies() if policies is None else tuple(policies)
    if any(not isinstance(p, SegmentPolicy) for p in policies):
        _fail("SegmentPolicy declarations required")
    identifiers = [p.segment_id for p in policies]
    if len(identifiers) != 12 or set(identifiers) != set(SEGMENT_IDS):
        _fail("exactly one policy for each of twelve segments required")
    for p in policies:
        for key in ("policy_id", "version", "exclusion_reason"):
            engine._text(getattr(p, key), f"policy.{key}")
        if not isinstance(p.allow_candidate, bool) or engine._decimal(p.min_uplift, "min_uplift") > 1:
            _fail("invalid candidate flag or uplift threshold")
        _duration(p.send_delay, "send_delay")
    return {p.segment_id: p for p in policies}


def _history(coupons, redemptions, context, policy, dataset_id, sha, point):
    if not isinstance(context, HistoryContext) or not isinstance(policy, HistoryPolicy):
        _fail("HistoryContext and explicit HistoryPolicy required")
    if context.dataset_id != dataset_id or context.raw_database_sha256 != sha:
        _fail("history dataset or raw SHA mismatch")
    for key in ("source_id", "version"):
        engine._text(getattr(context, key), f"history.{key}")
    if engine._clock(context.observed_until_at, "history.observed_until_at") != point:
        _fail("history coverage must end at the decision snapshot")
    if engine._clock(context.known_at, "history.known_at") > point:
        _fail("future history knowledge")
    if any(not isinstance(getattr(policy, k), bool) for k in ("block_active_unused", "block_previously_redeemed")):
        _fail("explicit boolean history restrictions required")
    interval = _duration(policy.minimum_send_interval, "minimum_send_interval")
    if policy.incomplete_history not in ("unknown", "exclude", "error"):
        _fail("invalid incomplete history policy")
    days = policy.expected_validity_days
    if days is not None and (type(days) is not int or days <= 0):
        _fail("expected history validity must be positive integer days or None")
    cp = _columns(coupons, COUPON_COLUMNS, "coupons")
    rd = _columns(redemptions, REDEMPTION_COLUMNS, "redemptions")
    for frame, keys in ((cp, ("coupon_id", "customer_id", "policy_id")), (rd, ("coupon_id", "customer_id"))):
        for key in keys:
            if not frame[key].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                _fail(f"invalid history {key}")
        if frame.coupon_id.duplicated().any():
            _fail("duplicate coupon or multiple redemptions")
    for key in ("sent_at", "expires_at"):
        cp[key] = pd.to_datetime([engine._clock(v, key) for v in cp[key]], utc=True)
    rd["purchased_at"] = pd.to_datetime([engine._clock(v, "purchased_at") for v in rd.purchased_at], utc=True)
    if not cp.expires_at.gt(cp.sent_at).all():
        _fail("coupon expiry must follow send time")
    if days is not None and not (cp.expires_at - cp.sent_at).eq(pd.Timedelta(days=days)).all():
        _fail(f"coupon history differs from declared {days}-day validity (v5 default 28)")
    for row in cp.itertuples():
        if engine._decimal(row.discount_rate, "discount_rate") != Decimal(".1"):
            _fail("coupon history must use 10% discounts")
        cost = engine._decimal(row.send_cost_minor, "send_cost_minor")
        if cost != cost.to_integral_value():
            _fail("send_cost_minor must be integral")
    linked = rd.merge(cp[["coupon_id", "customer_id", "sent_at", "expires_at"]], on="coupon_id", how="left", suffixes=("", "_coupon"), validate="one_to_one")
    if (linked.sent_at.isna().any() or not linked.customer_id.eq(linked.customer_id_coupon).all()
            or not (linked.purchased_at.ge(linked.sent_at) & linked.purchased_at.lt(linked.expires_at)).all()):
        _fail("redemption ownership, linkage or validity mismatch")
    # Actual raw tables may include future records; none participates in decisions.
    cp = cp.loc[cp.sent_at.lt(point)].copy()
    rd = rd.loc[rd.purchased_at.lt(point)].copy()
    return cp, rd, interval


def _aligned(frame, required, source, name):
    table = _columns(frame, required, name)
    table = engine._table(table, required, name)
    aligned = source[list(CONTEXT_COLUMNS)].merge(table, on=list(JOIN_KEYS), how="outer", indicator=True,
                                                suffixes=("", "_input"), validate="one_to_one")
    if not aligned._merge.eq("both").all():
        _fail(f"{name} missing or additional customers/snapshots")
    for key in set(CONTEXT_COLUMNS) - set(JOIN_KEYS):
        if key == "amount_unit":
            equal = all(engine._decimal(a, key, positive=True) == engine._decimal(b, key, positive=True)
                        for a, b in zip(aligned[key], aligned[f"{key}_input"]))
        else:
            equal = aligned[key].eq(aligned[f"{key}_input"]).all()
        if not equal:
            _fail(f"{name} {key} mismatch")
    return aligned[list(required)]


def _candidate_summary(customers, source_summary):
    summary = source_summary.copy(deep=True)
    for index, row in summary.iterrows():
        members = customers.loc[customers.segment_id.eq(row.segment_id)]
        summary.loc[index, "population_count"] = len(customers)
        summary.loc[index, "customer_count"] = len(members)
        summary.loc[index, "customer_ratio"] = len(members) / len(customers) if len(customers) else 0.
        summary.loc[index, "priority_score_sum"] = engine._score_sum(members.priority_score)
        for column in MEAN_COLUMNS:
            summary.loc[index, f"mean_{column}"] = members[column].mean()
    return summary


def audit_observed_coupon_order(items, *, purchase_at, as_of_at, actual_discount_minor, actual_paid_minor):
    """Validate a past 10% coupon order, never estimate a future basket."""
    from simulate.v4.simulator.raw_store import discount_minor
    if engine._clock(purchase_at, "purchase_at") >= engine._clock(as_of_at, "as_of_at"):
        _fail("observed order must be strictly before the snapshot")
    table = _columns(items, ("line_no", "quantity", "unit_price_minor"), "observed items")
    if table.empty: _fail("observed order has no items")
    for key in table:
        table[key] = [engine._ticks(engine._decimal(v, key, positive=key != "unit_price_minor"), 0, key) for v in table[key]]
    if table.line_no.duplicated().any(): _fail("duplicate order line")
    target = table.sort_values(["unit_price_minor", "line_no"], ascending=[False, True]).iloc[0]
    discount = discount_minor(int(target.unit_price_minor), .1)
    gross = sum(int(row.quantity) * int(row.unit_price_minor) for row in table.itertuples())
    if engine._decimal(actual_discount_minor, "discount_minor") != discount or engine._decimal(actual_paid_minor, "paid_minor") != gross - discount:
        _fail("observed coupon discount or paid amount mismatch")
    return {"line_no": int(target.line_no), "gross_minor": gross, "discount_minor": discount, "paid_minor": gross - discount}


def build_retention_strategy(segmentation, eligibility, coupons, redemptions, cost_inputs, cost_metadata, *,
                             dataset_id, as_of_at, currency, amount_unit, history_context, history_policy, policies=None):
    """Evaluate one cohort; calculate only declared candidate costs; never send."""
    return _build_retention_strategy(
        segmentation, eligibility, coupons, redemptions, cost_inputs, cost_metadata,
        dataset_id=dataset_id, as_of_at=as_of_at, currency=currency, amount_unit=amount_unit,
        history_context=history_context, history_policy=history_policy, policies=policies,
    )


def _price_cost_inputs(estimates, cost_metadata, point):
    """Price an aligned declaration using its exact costs and pre-decision source clock."""
    estimates = estimates.copy()
    for key in ("send_cost", "redemption_probability", "expected_discount_if_redeemed"):
        estimates[key] = estimates[key].astype(object)
    estimates["expected_cost"] = pd.Series([None] * len(estimates), dtype=object)
    if not isinstance(cost_metadata, CostMetadata): _fail("CostMetadata required")
    cost_clock = engine._clock(cost_metadata.as_of_at, "cost.as_of_at")
    for index, row in estimates.iterrows():
        for key in ("source_id", "source_version"): engine._text(row[key], key)
        if engine._clock(row.source_as_of_at, "source_as_of_at") > cost_clock or cost_clock > point:
            _fail("future cost knowledge")
        send = engine._decimal(row.send_cost, "send_cost")
        probability = engine._decimal(row.redemption_probability, "redemption_probability")
        discount = engine._decimal(row.expected_discount_if_redeemed, "expected_discount_if_redeemed")
        if probability > 1: _fail("redemption_probability exceeds one")
        try:
            with localcontext(Context(prec=200)) as ctx:
                ctx.traps[Inexact] = ctx.traps[Rounded] = True
                total = engine._decimal(send + probability * discount, "expected_cost")
        except DecimalException as exc:
            raise InputValidationError("retention: expected_cost exceeds exact computation precision") from exc
        for key, value in (("send_cost", send), ("redemption_probability", probability),
                           ("expected_discount_if_redeemed", discount), ("expected_cost", total)):
            estimates.loc[index, key] = value
    return estimates


def _coupon_eligibility(d, customer_id, cp, rd, interval, point, history_policy):
    if not isinstance(d.history_complete, (bool, np.bool_)):
        _fail("history_complete must be explicit bool")
    business = d.business_eligible
    if pd.notna(business) and not isinstance(business, (bool, np.bool_)):
        _fail("business_eligible must be bool or NULL")
    if business is True or isinstance(business, np.bool_) and business:
        if pd.notna(d.business_reason): _fail("eligible business row cannot have exclusion reason")
        eligible, reason = True, None
    else:
        engine._text(d.business_reason, "business_reason")
        eligible, reason = (False if pd.notna(business) else pd.NA), d.business_reason
    personal = cp.loc[cp.customer_id.eq(customer_id)]
    used = set(rd.loc[rd.customer_id.eq(customer_id), "coupon_id"])
    active_unused = personal.expires_at.gt(point) & ~personal.coupon_id.isin(used)
    if eligible is True:
        if not d.history_complete:
            if history_policy.incomplete_history == "error": _fail("incomplete history")
            eligible = False if history_policy.incomplete_history == "exclude" else pd.NA
            reason = "history_incomplete"
        elif history_policy.block_active_unused and active_unused.any(): eligible, reason = False, "active_unused_coupon"
        elif history_policy.block_previously_redeemed and used: eligible, reason = False, "previously_redeemed"
        elif len(personal) and point - personal.sent_at.max() < interval: eligible, reason = False, "minimum_send_interval"
    return eligible, reason, business, personal, used, active_unused


def _build_retention_strategy(segmentation, eligibility, coupons, redemptions, cost_inputs, cost_metadata, *,
                              dataset_id, as_of_at, currency, amount_unit, history_context, history_policy,
                              policies, common_cost_inputs=False):
    if not isinstance(segmentation, SegmentationResult):
        _fail("SegmentationResult required")
    if not isinstance(history_context, HistoryContext):
        _fail("HistoryContext required")
    point = engine._clock(as_of_at, "as_of_at")
    mapping = _policies(policies)
    source = segmentation.customers.copy(deep=True).reset_index(drop=True)
    excluded = segmentation.excluded.copy(deep=True).reset_index(drop=True)
    for frame in (source, excluded):
        required = (*engine.ROW_COLUMNS, "is_high_clv", "priority_rank") if frame is source else CONTEXT_COLUMNS
        checked = engine._table(frame, required, "segmentation")
        frame["as_of_at"] = checked.as_of_at
        if not (frame.dataset_id.eq(dataset_id) & frame.as_of_at.eq(point) & frame.currency.eq(currency)).all():
            _fail("mixed dataset, snapshot or currency")
        if any(engine._decimal(v, "amount_unit", positive=True) != engine._decimal(amount_unit, "amount_unit", positive=True) for v in frame.amount_unit):
            _fail("mixed amount unit")
    all_context = pd.concat([source[list(CONTEXT_COLUMNS)], excluded[list(CONTEXT_COLUMNS)]])
    if all_context.duplicated(list(JOIN_KEYS)).any() or all_context.raw_database_sha256.nunique() > 1:
        _fail("overlapping customers or mixed raw SHA")
    sha = all_context.raw_database_sha256.iloc[0] if len(all_context) else history_context.raw_database_sha256
    cp, rd, interval = _history(coupons, redemptions, history_context, history_policy, dataset_id, sha, point)
    declared = _aligned(eligibility, ELIGIBILITY_COLUMNS, source, "eligibility").set_index("customer_id")
    records = []
    for row in source.itertuples():
        if row.segment_id not in mapping:
            _fail("unknown segment")
        rule = mapping[row.segment_id]
        uplift = row.uplift_score
        if isinstance(uplift, (bool, np.bool_)) or not np.isfinite(uplift) or not -1 <= uplift <= 1:
            _fail("invalid uplift")
        d = declared.loc[row.customer_id]
        eligible, reason, business, personal, used, active_unused = _coupon_eligibility(
            d, row.customer_id, cp, rd, interval, point, history_policy)
        policy_candidate = rule.allow_candidate and Decimal(str(uplift)) > engine._decimal(rule.min_uplift, "min_uplift")
        candidate = eligible is True and policy_candidate
        policy_reason = None if policy_candidate else rule.exclusion_reason if not rule.allow_candidate else "uplift_not_above_threshold"
        record = source.loc[row.Index].to_dict()
        record.update(analyzed=True, business_eligible=business, business_reason=d.business_reason,
                      history_complete=bool(d.history_complete), observed_active_unused_count=int(active_unused.sum()),
                      observed_used_coupon_count=len(used),
                      observed_expired_unused_count=int((personal.expires_at.le(point) & ~personal.coupon_id.isin(used)).sum()),
                      eligible_for_coupon=eligible, eligibility_reason=reason,
                      policy_candidate=policy_candidate, budget_candidate=candidate, policy_exclusion_reason=policy_reason,
                      decision_at=point, planned_sent_at=point + _duration(rule.send_delay, "send_delay") if candidate else pd.NaT,
                      requires_eligibility_recheck=bool(candidate and _duration(rule.send_delay, "send_delay") > pd.Timedelta(0)),
                      policy_id=rule.policy_id, policy_version=rule.version,
                      exclusion_reason=None if candidate else reason or policy_reason)
        records.append(record)
    evaluation = pd.DataFrame(records, columns=[*source.columns, "analyzed", "business_eligible", "business_reason",
        "history_complete", "observed_active_unused_count", "observed_used_coupon_count", "observed_expired_unused_count",
        "eligible_for_coupon", "eligibility_reason",
        "policy_candidate", "budget_candidate", "policy_exclusion_reason", "decision_at", "planned_sent_at",
        "requires_eligibility_recheck", "policy_id", "policy_version", "exclusion_reason"])
    candidate_mask = evaluation.budget_candidate.astype(bool)
    candidates = source.loc[candidate_mask].copy()
    if common_cost_inputs:
        table = engine._table(_columns(cost_inputs, COST_INPUT_COLUMNS, "common cost_inputs"),
                              COST_INPUT_COLUMNS, "common cost_inputs")
        cost_scope = all_context.merge(table[list(JOIN_KEYS)], on=list(JOIN_KEYS), validate="one_to_one")
        estimates = _aligned(table, COST_INPUT_COLUMNS, cost_scope, "common cost_inputs")
        candidate_keys = candidates[list(JOIN_KEYS)]
        covered = candidate_keys.merge(estimates[list(JOIN_KEYS)], on=list(JOIN_KEYS), how="left", indicator=True)
        if not covered._merge.eq("both").all():
            _fail("cost_inputs missing candidate customers/snapshots")
    else:
        estimates = _aligned(cost_inputs, COST_INPUT_COLUMNS, candidates, "cost_inputs")
    estimates = _price_cost_inputs(estimates, cost_metadata, point)
    if common_cost_inputs:
        estimates = candidate_keys.merge(estimates, on=list(JOIN_KEYS), validate="one_to_one")
    costs = estimates[list(engine.COST_COLUMNS)].copy()
    policy_excluded = source.loc[~candidate_mask].copy()
    policy_excluded["exclusion_reason"] = evaluation.loc[~candidate_mask, "exclusion_reason"].values
    policy_excluded["exclusion_stage"] = "retention"
    excluded["exclusion_stage"] = "upstream"
    combined = pd.concat([excluded, policy_excluded], ignore_index=True)
    metadata = deepcopy(segmentation.metadata)
    source_metadata = deepcopy(metadata)
    metadata.update(population_scope="explicit_eligibility", input_count=len(all_context), analyzed_count=len(candidates),
                    excluded_count=len(combined), retention_source_metadata=source_metadata,
                    retention_policy_version="retention-adapter-v1",
                    high_clv_policy="original Segmentation labels preserved; not recomputed on candidates",
                    priority_policy="original scores and ranks preserved; candidate ranks may have gaps",
                    retained_original_classification_and_ranks=True)
    for cohort in metadata.get("cohorts", []):
        cohort.update(analyzed_count=len(candidates), excluded_count=len(combined),
                      high_clv_count=int(candidates.is_high_clv.sum()),
                      high_clv_ratio=float(candidates.is_high_clv.mean()) if len(candidates) else 0.)
    adapted = SegmentationResult(candidates, combined, _candidate_summary(candidates, segmentation.summary), metadata)
    # Reuse the full Budget contract validator with actual costs, without running an algorithm.
    engine._prepare(adapted, costs, cost_metadata, Budget(0, currency, amount_unit), dataset_id, point)
    previous = excluded.copy()
    previous["analyzed"], previous["budget_candidate"] = False, False
    evaluation = pd.concat([evaluation, previous], ignore_index=True)
    evaluation["eligible_for_coupon"] = pd.array(evaluation.eligible_for_coupon, dtype="boolean")
    return RetentionResult(evaluation, estimates, adapted, costs, cost_metadata, {
        "contract": "v5 default 28-day coupon contract; declared expected costs; no actual dispatch",
        "dataset_id": dataset_id, "as_of_at": point.isoformat(), "currency": currency, "amount_unit": str(amount_unit),
        "history_context": asdict(history_context), "history_policy": asdict(history_policy),
        "policies": [asdict(p) for p in mapping.values()], "cost": asdict(cost_metadata),
        "candidate_count": len(candidates), "upstream_excluded_count": len(excluded), "policy_excluded_count": len(policy_excluded),
        "development_policy": True, "actual_dispatch_performed": False})
