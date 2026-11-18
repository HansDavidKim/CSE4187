"""Independently reconcile canonical observations and describe source/simulation gaps."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


from simulate.v4.data.loader import check_generation_sidecars
from simulate.v4.irl.fit_shopping import load_source, prepare_source
from simulate.v4.simulator.coupon_response import sample_redemption_delay, scale_redemption_probability
from simulate.v4.simulator.outcomes import GENERATION_WINDOW_DAYS, experiment_window_days
from simulate.v4.simulator.raw_store import cart_at, load_raw_dataset


def _time(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("validation timestamps must be timezone-aware")
    return value.tz_convert("UTC")


def _frame_times(frame, columns):
    frame = frame.copy()
    for column in columns:
        if any(pd.Timestamp(value).tzinfo is None for value in frame[column].dropna()):
            raise ValueError(f"{column} must be timezone-aware")
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _observation_bound(metadata, name):
    key = f"observation_{name}_at"
    return _time(metadata[key] if key in metadata else metadata[f"{name}_at"])


def _outcome_contract(metadata):
    eligible_days = experiment_window_days(metadata, "eligibility_window_days")
    generation_definition = f"future_no_session_30d_active{eligible_days}d"
    definition = metadata.get("definition_id", generation_definition)
    days = {generation_definition: GENERATION_WINDOW_DAYS, f"future_no_session_28d_active{eligible_days}d": 28}.get(definition)
    if days is None:
        raise ValueError("unsupported outcome definition")
    for field, expected in (("outcome_window_days", days), ("generation_window_days", GENERATION_WINDOW_DAYS),
                            ("eligibility_window_days", eligible_days)):
        value = metadata.get(field, expected if days == 30 else None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value != expected:
            raise ValueError(f"{field} must match {expected}")
    if days == 28 and (metadata.get("generation_definition_id") != generation_definition or
                       not isinstance(metadata.get("parent_dataset_id"), str) or not metadata["parent_dataset_id"].strip()):
        raise ValueError("28-day outcomes require the original 30-day definition and parent dataset")
    return definition, days


def _numbers(values):
    return pd.to_numeric(pd.Series(values, copy=False), errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _summary(values, weights=None):
    values = _numbers(values)
    valid = np.isfinite(values)
    values = values[valid]
    weights = np.ones(len(values)) if weights is None else _numbers(weights)[valid]
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    if not len(values):
        return {"records": 0, "weight": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quantiles = values[order][np.searchsorted(cumulative, [0.1, 0.5, 0.9]).clip(max=len(values) - 1)]
    return {"records": len(values), "weight": float(weights.sum()),
            "mean": float(np.average(values, weights=weights)),
            **dict(zip(("p10", "p50", "p90"), quantiles.tolist()))}


def _distance(source, generated, unit, source_weights=None, generated_weights=None):
    left, right = _numbers(source), _numbers(generated)
    li, ri = np.isfinite(left), np.isfinite(right)
    left, right = left[li], right[ri]
    sw = None if source_weights is None else _numbers(source_weights)[li]
    gw = None if generated_weights is None else _numbers(generated_weights)[ri]
    result = {"unit": unit, "source": _summary(left, sw), "generated": _summary(right, gw),
              "ks_statistic_unweighted_records": None, "wasserstein": None,
              "pass_threshold": None}
    if len(left) and len(right):
        result["ks_statistic_unweighted_records"] = float(ks_2samp(left, right, method="auto").statistic)
        result["wasserstein"] = float(wasserstein_distance(left, right, u_weights=sw, v_weights=gw))
    return result


def _order_events(tables):
    return tables["orders"].merge(tables["events"][["event_id", "customer_id", "session_id", "event_at"]],
                                  left_on="purchase_event_id", right_on="event_id", validate="one_to_one")


def audit_coupon_usage(tables, metadata):
    """Independently pair raw3 usage records with one highest-priced purchased unit."""
    events, orders, items, coupons = (tables[name] for name in ("events", "orders", "order_items", "coupons"))
    version = metadata.get("raw_schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or version not in (1, 2, 3, 4, 5):
        raise ValueError("unsupported raw schema version for coupon usage audit")
    usage = events.loc[events.event_type.eq("coupon_usage")]
    mismatches = Counter()
    summary = {"raw_schema_version": version, "coupon_usage_events": len(usage),
               "purchase_events": int(events.event_type.eq("purchase").sum()),
               "orders_with_coupon": int(orders.coupon_id.notna().sum())}
    if version < 3:
        mismatches["usage_in_legacy_schema"] = len(usage)
        return {**summary, "passed": not len(usage), "status": "legacy_whole_order_discount",
                "mismatch_counts": dict(mismatches)}
    references = ("related_purchase_event_id", "coupon_id", "order_line_no", "discount_minor")
    if not set(references) <= set(events):
        raise ValueError("raw3 coupon usage audit requires explicit event linkage columns")
    ordinary = ~events.event_type.eq("coupon_usage")
    item_review = events.event_type.eq("review") & (version >= 5)
    forbidden = ordinary & events[["coupon_id", "discount_minor"]].notna().any(axis=1)
    forbidden |= ordinary & events.order_line_no.notna() & ~item_review
    forbidden |= ordinary & events.related_purchase_event_id.notna() & ~(events.event_type.eq("review") & (version >= 4))
    mismatches["nonusage_has_coupon_linkage"] = int(forbidden.sum())
    mismatches["duplicate_event_id"] = int(events.event_id.duplicated().sum())
    mismatches["duplicate_order_purchase"] = int(orders.purchase_event_id.duplicated().sum())
    mismatches["coupon_used_multiple_orders"] = int(orders.loc[orders.coupon_id.notna(), "coupon_id"].duplicated().sum())
    mismatches["duplicate_usage_purchase"] = int(usage.related_purchase_event_id.dropna().duplicated().sum())
    mismatches["duplicate_usage_coupon"] = int(usage.coupon_id.dropna().duplicated().sum())
    mismatches["duplicate_order_line"] = int(items.duplicated(["order_id", "line_no"]).sum())
    purchase = events.loc[events.event_type.eq("purchase")]
    mismatches["purchase_header_has_item_values"] = int(purchase[["product_id", "quantity", "unit_price_minor"]].notna().any(axis=1).sum())
    parent_map = {row.event_id: row for row in events.loc[events.event_type.isin(["purchase", "coupon_usage"])].itertuples(index=False)}
    order_map = {row.purchase_event_id: row for row in orders.itertuples(index=False)}
    coupon_map = {row.coupon_id: row for row in coupons.itertuples(index=False)}
    session_map = {row.session_id: row for row in tables["sessions"].itertuples(index=False)}
    order_lines = {}
    for line in items.itertuples(index=False):
        order_lines.setdefault(line.order_id, []).append(line)
    usage_counts = usage.related_purchase_event_id.value_counts().to_dict()
    purchase_counts = orders.purchase_event_id.value_counts().to_dict()
    mismatches["purchase_header_order_count"] = sum(purchase_counts.get(event_id, 0) != 1 for event_id in purchase.event_id)
    expected_discounts = {}
    for order in orders.itertuples(index=False):
        parent = parent_map.get(order.purchase_event_id)
        mismatches["order_not_linked_to_purchase"] += parent is None or parent.event_type != "purchase"
        lines = order_lines.get(order.order_id, [])
        mismatches["order_has_no_items"] += not lines
        gross = sum(int(line.quantity) * int(line.unit_price_minor) for line in lines)
        expected = 0
        if pd.notna(order.coupon_id):
            coupon = coupon_map.get(order.coupon_id)
            mismatches["order_coupon_fk"] += coupon is None
            if lines and coupon is not None:
                target = min(lines, key=lambda line: (-int(line.unit_price_minor), int(line.line_no)))
                expected = int((Decimal(int(target.unit_price_minor)) * Decimal(str(coupon.discount_rate))).quantize(Decimal(1), rounding=ROUND_HALF_UP))
                expected_discounts[order.purchase_event_id] = (target, expected)
        mismatches["usage_count_by_order"] += usage_counts.get(order.purchase_event_id, 0) != int(pd.notna(order.coupon_id))
        mismatches["order_unit_discount"] += order.discount_minor != expected
        mismatches["order_money"] += order.gross_minor != gross or order.paid_minor != gross - expected
    for row in usage.itertuples(index=False):
        if any(pd.isna(getattr(row, field)) for field in (*references, "product_id", "quantity", "unit_price_minor")):
            mismatches["usage_missing_fields"] += 1
            continue
        mismatches["usage_quantity_not_one"] += row.quantity != 1
        if any(pd.notna(getattr(row, field, None)) for field in ("offer_product_id", "offer_unit_price_minor")):
            mismatches["usage_has_offer_proxy"] += 1
        parent = parent_map.get(row.related_purchase_event_id)
        if parent is None or parent.event_type != "purchase":
            mismatches["usage_purchase_fk"] += 1
            continue
        mismatches["usage_event_order"] += not row.event_id > parent.event_id
        mismatches["usage_purchase_customer"] += row.customer_id != parent.customer_id
        mismatches["usage_purchase_session"] += row.session_id != parent.session_id
        at, parent_at = _time(row.event_at), _time(parent.event_at)
        mismatches["usage_purchase_time"] += at != parent_at
        session = session_map.get(row.session_id)
        mismatches["usage_session_fk"] += session is None or session.customer_id != row.customer_id
        if session is not None:
            mismatches["usage_session_clock"] += at < _time(session.started_at) or (pd.notna(session.ended_at) and at > _time(session.ended_at))
        if ("observation_start_at" in metadata or "start_at" in metadata) and ("observation_end_at" in metadata or "end_at" in metadata):
            mismatches["usage_observation_clock"] += not _observation_bound(metadata, "start") <= at < _observation_bound(metadata, "end")
        order, coupon = order_map.get(parent.event_id), coupon_map.get(row.coupon_id)
        if order is None:
            mismatches["usage_order_fk"] += 1
            continue
        mismatches["usage_order_coupon"] += pd.isna(order.coupon_id) or order.coupon_id != row.coupon_id
        if coupon is None:
            mismatches["usage_coupon_fk"] += 1
        else:
            mismatches["usage_coupon_customer"] += coupon.customer_id != row.customer_id
            mismatches["usage_coupon_clock"] += not _time(coupon.sent_at) <= at < _time(coupon.expires_at)
        target = expected_discounts.get(parent.event_id)
        if target is not None:
            line, discount = target
            mismatches["usage_highest_unit_line"] += row.order_line_no != line.line_no
            mismatches["usage_target_product"] += row.product_id != line.product_id
            mismatches["usage_target_price"] += row.unit_price_minor != line.unit_price_minor
            mismatches["usage_unit_discount"] += row.discount_minor != discount
        mismatches["usage_order_discount"] += row.discount_minor != order.discount_minor
    return {**summary, "passed": not any(mismatches.values()), "status": "checked",
            "mismatch_counts": {key: int(value) for key, value in mismatches.items()},
            "usage_discount_minor": sum(int(value) for value in usage.discount_minor.dropna()),
            "order_discount_minor": sum(int(value) for value in orders.discount_minor),
            "rule": "One usage per coupon-bearing order; highest purchased unit price, lowest line_no on ties, one unit only, decimal half-up at the actual coupon rate. This is declared policy, not a learned source preference."}


def audit_auxiliary_events(tables, metadata):
    """Independently check auxiliary clocks/FKs without turning asynchronous events into visits."""
    events = tables["events"]
    auxiliary = events.loc[events.event_type.isin(["search", "review", "cs_contact"])]
    version = metadata.get("raw_schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or version not in (1, 2, 3, 4, 5):
        raise ValueError("unsupported raw schema version for auxiliary event audit")
    allowed = {"view", "cart", "remove_from_cart", "purchase"}
    if version >= 3:
        allowed.add("coupon_usage")
    if version >= 4:
        allowed.update(["search", "review", "cs_contact"])
    mismatches = Counter(unknown_event_type=int((~events.event_type.isin(allowed)).sum()))
    counts = {name: 0 for name in sorted(allowed)}
    counts.update({str(key): int(value) for key, value in events.event_type.value_counts().items()})
    if version < 4:
        return {"passed": not any(mismatches.values()), "status": "legacy_no_auxiliary_events", "event_counts": counts,
                "mismatch_counts": dict(mismatches)}
    required = {"rating", "rating_kind", "contact_id", "related_purchase_event_id", "coupon_id", "order_line_no", "discount_minor"}
    if not required <= set(events):
        raise ValueError("raw4 auxiliary audit requires explicit review/CS metadata")
    auxiliary = _frame_times(auxiliary, ["event_at"])
    people = tables["customers"].set_index("customer_id").signed_up_at
    birth = auxiliary.customer_id.map(people)
    mismatches["unknown_customer"] = int(birth.isna().sum())
    mismatches["before_signup"] = int(auxiliary.event_at.lt(birth).sum())
    start, end = _observation_bound(metadata, "start"), _observation_bound(metadata, "end")
    mismatches["outside_observation"] = int((auxiliary.event_at.lt(start) | auxiliary.event_at.ge(end)).sum())
    item_review = auxiliary.event_type.eq("review") & (version >= 5)
    no_catalog = ["quantity", "unit_price_minor", "offer_product_id", "offer_unit_price_minor"]
    forbidden_catalog = auxiliary[no_catalog].notna().any(axis=1) | (auxiliary.product_id.notna() & ~item_review)
    mismatches["invented_catalog_interaction"] = int(forbidden_catalog.sum())
    forbidden_coupon = auxiliary[["coupon_id", "discount_minor"]].notna().any(axis=1)
    forbidden_coupon |= auxiliary.order_line_no.notna() & ~item_review
    mismatches["auxiliary_coupon_details"] = int(forbidden_coupon.sum())
    asynchronous = auxiliary.loc[auxiliary.event_type.isin(["review", "cs_contact"])]
    mismatches["async_fabricated_session"] = int(asynchronous.session_id.notna().sum())
    sessions = {row.session_id: row for row in tables["sessions"].itertuples(index=False)}
    for row in auxiliary.loc[auxiliary.event_type.eq("search")].itertuples(index=False):
        session = sessions.get(row.session_id)
        mismatches["search_session_fk"] += session is None or session.customer_id != row.customer_id
        if session is not None:
            mismatches["search_session_clock"] += row.event_at < _time(session.started_at) or (pd.notna(session.ended_at) and row.event_at > _time(session.ended_at))
        mismatches["search_has_review_or_contact"] += any(pd.notna(getattr(row, name)) for name in required - {"coupon_id", "order_line_no", "discount_minor"})
    purchase = {row.event_id: row for row in events.loc[events.event_type.eq("purchase")].itertuples(index=False)}
    order_purchases = {row.purchase_event_id: row for row in tables["orders"].itertuples(index=False)}
    order_lines = ({(row.order_id, row.line_no): row for row in tables["order_items"].itertuples(index=False)}
                   if version >= 5 else {})
    reviews = auxiliary.loc[auxiliary.event_type.eq("review")]
    mismatches["duplicate_review_purchase"] = int(reviews.related_purchase_event_id.dropna().duplicated().sum()) if version < 5 else 0
    if version >= 5:
        mismatches["duplicate_review_product"] = int(reviews.duplicated(["related_purchase_event_id", "product_id"]).sum())
    for row in reviews.itertuples(index=False):
        parent = purchase.get(row.related_purchase_event_id) if pd.notna(row.related_purchase_event_id) else None
        mismatches["review_order_fk"] += parent is None or row.related_purchase_event_id not in order_purchases
        if parent is not None:
            mismatches["review_customer"] += row.customer_id != parent.customer_id
            mismatches["review_clock"] += not (row.event_id > parent.event_id and row.event_at >= _time(parent.event_at))
        if version >= 5:
            number = row.order_line_no
            valid_line = (not isinstance(number, (bool, np.bool_))
                          and ((isinstance(number, (int, np.integer)) and 0 < number < 2 ** 63)
                               or (isinstance(number, (float, np.floating)) and np.isfinite(number)
                                   and 0 < number <= 2 ** 53 and float(number).is_integer())))
            order = order_purchases.get(row.related_purchase_event_id) if parent is not None else None
            line = order_lines.get((order.order_id, int(number))) if order is not None and valid_line else None
            mismatches["review_order_line"] += line is None
            mismatches["review_product"] += pd.isna(row.product_id) or (line is not None and row.product_id != line.product_id)
        mismatches["review_rating"] += isinstance(row.rating, bool) or pd.isna(row.rating) or row.rating not in (1, 2, 3, 4, 5)
        mismatches["review_rating_kind"] += pd.isna(row.rating_kind) or row.rating_kind not in ("product_star_scenario", "order_survey_score")
        mismatches["review_has_contact"] += pd.notna(row.contact_id)
    contacts = auxiliary.loc[auxiliary.event_type.eq("cs_contact")]
    mismatches["duplicate_contact_id"] = int(contacts.contact_id.dropna().duplicated().sum())
    mismatches["missing_contact_id"] = int(contacts.contact_id.map(lambda value: not isinstance(value, str) or not value.strip()).sum())
    mismatches["cs_has_order_or_rating"] = int(contacts[["related_purchase_event_id", "rating", "rating_kind"]].notna().any(axis=1).sum())
    return {"passed": not any(mismatches.values()), "status": "checked", "event_counts": counts,
            "mismatch_counts": {key: int(value) for key, value in mismatches.items()},
            "review_grain": "purchase_product" if version >= 5 else "purchase",
            "interpretation": "Search belongs to a shopping session. Review/CS are asynchronous records and never themselves count as sessions, retention visits, purchases, or individual causal truth."}


def audit_sidecars(tables, metadata, labels, treatments, lifecycle=None, diagnostics=None):
    """Recalculate outcomes from raw sessions/orders, never from latent diagnostics."""
    definition, outcome_days = _outcome_contract(metadata)
    eligible_days = experiment_window_days(metadata, "eligibility_window_days")
    coupon_days = experiment_window_days(metadata, "coupon_validity_days")
    required_labels = {"customer_id", "churn", "retained", "future_retained", "future_revenue_30d",
                       "future_revenue_12m", "as_of_at", "outcome_end_at", "is_censored", "definition_id"}
    required_assignments = {"customer_id", "treatment", "as_of_at", "assigned_at", "assignment_probability",
                            "actual_sent", "coupon_id", "send_cost_minor"}
    if labels.empty or not required_labels <= set(labels) or not required_assignments <= set(treatments):
        raise ValueError("explicit labels and randomized assignment sidecars are required")
    if outcome_days == 28 and not {"future_revenue_28d", "future_revenue"} <= set(labels):
        raise ValueError("28-day labels must expose future_revenue_28d and the default revenue alias")
    labels = _frame_times(labels, ["as_of_at", "outcome_end_at"])
    treatments = _frame_times(treatments, ["as_of_at", "assigned_at"])
    if labels.duplicated(["customer_id", "as_of_at"]).any() or treatments.duplicated(["customer_id", "as_of_at"]).any():
        raise ValueError("duplicate snapshot label or assignment")
    if not set(labels.customer_id) == set(tables["customers"].customer_id):
        raise ValueError("labels must cover every customer")
    joined = labels.merge(treatments, on=["customer_id", "as_of_at"], how="outer", validate="one_to_one", indicator=True,
                          suffixes=("", "_assignment"))
    if not joined._merge.eq("both").all() or not joined.treatment.isin([0, 1]).all():
        raise ValueError("ITT labels must include all treatment/control assignments")
    observation_start = _observation_bound(metadata, "start")
    observation_end = _observation_bound(metadata, "end")
    sessions, events = tables["sessions"], tables["events"]
    orders = _order_events(tables)
    signups = tables["customers"].set_index("customer_id").signed_up_at
    mismatches = Counter()
    window_reports = []
    for (t0, until), group in joined.groupby(["as_of_at", "outcome_end_at"], sort=True):
        mismatches[f"outcome_window_not_{outcome_days}d"] += int(until != t0 + pd.Timedelta(days=outcome_days)) * len(group)
        mismatches["incomplete_outcome_followup"] += int(until > observation_end) * len(group)
        mismatches["prediction_before_observation"] += int(t0 < observation_start) * len(group)
        mismatches["snapshot_differs_from_metadata"] += int(t0 != _time(metadata["as_of_at"])) * len(group)
        local = t0.tz_convert("Asia/Seoul")
        mismatches["not_monday_midnight_seoul"] += int(local.weekday() != 0 or local != local.normalize()) * len(group)
        mismatches["definition_id"] += int(group.definition_id.ne(definition).sum())
        mismatches["assignment_probability"] += int(pd.to_numeric(group.assignment_probability, errors="coerce").ne(0.5).sum())
        if "eligibility_window_days" in group:
            mismatches["eligibility_window_days"] += int(group.eligibility_window_days.ne(eligible_days).fillna(True).sum())
        history = sessions.loc[sessions.started_at.ge(t0 - pd.Timedelta(days=eligible_days)) & sessions.started_at.lt(t0)]
        eligible = group.customer_id.isin(history.customer_id)
        mismatches[f"not_eligible_active{eligible_days}d"] += int((~eligible).sum())
        mismatches["signup_at_or_after_snapshot"] += int(group.customer_id.map(signups).ge(t0).sum())
        future = sessions.loc[sessions.started_at.ge(t0) & sessions.started_at.lt(until)]
        retained = group.customer_id.isin(future.customer_id).astype(int)
        mismatches["churn_label"] += int(pd.to_numeric(group.churn, errors="coerce").ne(1 - retained).sum())
        for column in ("retained", "future_retained"):
            if column in group:
                mismatches[column] += int(pd.to_numeric(group[column], errors="coerce").ne(retained).sum())
        paid = orders.loc[orders.event_at.ge(t0) & orders.event_at.lt(until)].groupby("customer_id").paid_minor.sum()
        expected_revenue = group.customer_id.map(paid).fillna(0)
        for column in ("future_revenue", f"future_revenue_{outcome_days}d"):
            if column in group:
                mismatches[column] += int(pd.to_numeric(group[column], errors="coerce").ne(expected_revenue).sum())
        if outcome_days != 30:
            paid30 = orders.loc[orders.event_at.ge(t0) & orders.event_at.lt(t0+pd.Timedelta(days=30))].groupby("customer_id").paid_minor.sum()
            expected30 = group.customer_id.map(paid30).fillna(0)
            mismatches["future_revenue_30d"] += int(pd.to_numeric(group.future_revenue_30d, errors="coerce").ne(expected30).sum())
        if "is_censored" in group:
            mismatches["is_censored"] += int((~group.is_censored.isin([True, False]) |
                                               group.is_censored.ne(until > observation_end)).sum())
        if "snapshot_at" in group:
            mismatches["snapshot_at"] += int(pd.to_datetime(group.snapshot_at, utc=True).ne(t0).sum())
        if "observation_end_at" in group:
            mismatches["label_observation_end"] += int(pd.to_datetime(group.observation_end_at, utc=True).ne(observation_end).sum())
        if "observation_status" in group:
            mismatches["observation_status"] += int(group.observation_status.ne("complete").sum())
        highest_unit = metadata.get("raw_schema_version", 1) >= 3
        treatment_policy = f"one_coupon_10pct_{'highest_unit_' if highest_unit else ''}{coupon_days}d"
        if "policy_id" in group:
            expected_policy = group.treatment.map({0: "no_new_coupon", 1: treatment_policy})
            mismatches["policy_id"] += int(group.policy_id.ne(expected_policy).sum())
        mismatches["assignment_time"] += int(group.assigned_at.ne(t0).sum())
        policy_until = t0 + pd.Timedelta(days=coupon_days)
        sent = tables["coupons"].loc[tables["coupons"].sent_at.ge(t0)]
        sent_count = group.customer_id.map(sent.groupby("customer_id").size()).fillna(0)
        mismatches["coupon_count_by_assignment"] += int(sent_count.ne(group.treatment).sum())
        mismatches["actual_sent"] += int((~group.actual_sent.isin([True, False]) |
                                         group.actual_sent.ne(sent_count.gt(0))).sum())
        actual_cost = pd.to_numeric(group.customer_id.map(sent.groupby("customer_id").send_cost_minor.sum()), errors="coerce").fillna(0)
        mismatches["send_cost_minor"] += int(pd.to_numeric(group.send_cost_minor, errors="coerce").ne(actual_cost).sum())
        mismatches["control_coupon_reference"] += int(group.loc[group.treatment.eq(0), "coupon_id"].notna().sum())
        coupon_map = sent.set_index("coupon_id")
        for row in group.loc[group.treatment.eq(1)].itertuples(index=False):
            coupon = coupon_map.loc[row.coupon_id] if pd.notna(row.coupon_id) and row.coupon_id in coupon_map.index else None
            mismatches["treated_coupon_reference"] += int(coupon is None or coupon.customer_id != row.customer_id)
        mismatches["experiment_coupon_time"] += int((sent.sent_at.ne(t0) | sent.expires_at.ne(policy_until)).sum())
        mismatches["experiment_coupon_discount"] += int(pd.to_numeric(sent.discount_rate, errors="coerce").ne(0.1).sum())
        if highest_unit:
            mismatches["experiment_coupon_policy_id"] += int(sent.policy_id.ne(treatment_policy).sum())
        if "future_revenue_12m" in group:
            clv_end = t0 + pd.DateOffset(months=12)
            if "clv_outcome_end_at" in group:
                mismatches["clv_outcome_end"] += int(pd.to_datetime(group.clv_outcome_end_at, utc=True).ne(clv_end).sum())
            clv = orders.loc[orders.event_at.ge(t0) & orders.event_at.lt(clv_end)].groupby("customer_id").paid_minor.sum()
            complete_control = group.treatment.eq(0) & (clv_end <= observation_end)
            expected = group.customer_id.map(clv).fillna(0)
            mismatches["future_revenue_12m"] += int(pd.to_numeric(group.loc[complete_control, "future_revenue_12m"], errors="coerce").ne(expected.loc[complete_control]).sum())
            mismatches["unidentified_12m_clv_label"] += int(group.loc[~complete_control, "future_revenue_12m"].notna().sum())
            newly_sent = tables["coupons"].loc[tables["coupons"].sent_at.ge(t0) & tables["coupons"].sent_at.lt(clv_end)]
            mismatches["control_clv_new_coupon"] += int(group.loc[complete_control, "customer_id"].isin(newly_sent.customer_id).sum())
        arm_counts = {str(int(arm)): int(count) for arm, count in group.treatment.value_counts().items()}
        mismatches["unbalanced_randomized_arms"] += abs(arm_counts.get("0", 0) - arm_counts.get("1", 0))
        window_reports.append({"as_of_at": t0.isoformat(), "outcome_end_at": until.isoformat(),
                               "definition_id": definition, "outcome_window_days": outcome_days,
                               "generation_window_days": GENERATION_WINDOW_DAYS, "eligibility_window_days": eligible_days,
                               "coupon_validity_days": coupon_days,
                               "customers": len(group), "eligible": int(eligible.sum()), "ITT_arm_counts": arm_counts,
                               "recomputed_churn_rate": float((1 - retained).mean()),
                               "recomputed_churn_by_arm": {str(int(arm)): float((1 - retained.loc[part.index]).mean())
                                                           for arm, part in group.groupby("treatment")}})
    mismatches["raw_event_outside_observation"] = int((events.event_at.lt(observation_start) | events.event_at.ge(observation_end)).sum())
    mismatches["raw_session_outside_observation"] = int((sessions.started_at.lt(observation_start) | sessions.started_at.ge(observation_end)).sum())
    gross = sum(int(value) for value in tables["orders"].gross_minor)
    discount = sum(int(value) for value in tables["orders"].discount_minor)
    paid = sum(int(value) for value in tables["orders"].paid_minor)
    item_gross = sum(int(row.quantity) * int(row.unit_price_minor) for row in tables["order_items"].itertuples(index=False))
    mismatches["aggregate_money"] = int(gross != item_gross or gross != discount + paid)
    coupon_usage = audit_coupon_usage(tables, metadata)
    mismatches["coupon_usage_integrity"] = int(not coupon_usage["passed"])
    auxiliary_events = audit_auxiliary_events(tables, metadata)
    mismatches["auxiliary_event_integrity"] = int(not auxiliary_events["passed"])
    return {"passed": not any(mismatches.values()), "mismatch_counts": dict(mismatches), "windows": window_reports,
            "money_minor": {"item_gross": item_gross, "order_gross": gross, "discount": discount, "paid": paid},
            "coupon_usage": coupon_usage,
            "auxiliary_events": auxiliary_events,
            "ITT_definition": "Every assigned customer retained in the denominator regardless of coupon use or future visits.",
            "lifecycle_session_counts": lifecycle.life_cycle.value_counts().to_dict() if lifecycle is not None and "life_cycle" in lifecycle else {},
            "lifecycle_customer_end_counts": diagnostics.life_cycle_end.value_counts().to_dict() if diagnostics is not None and "life_cycle_end" in diagnostics else {}}


def audit_cart_sample(tables, as_of_at, sample_size=20):
    """Compare replay with an independent chronological journal for at most 20 customers."""
    as_of_at = _time(as_of_at)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be positive")
    ids = sorted(tables["customers"].customer_id)
    ids = [ids[int(index)] for index in np.linspace(0, len(ids) - 1, min(sample_size, 20, len(ids)))]
    scoped = dict(tables)
    scoped["customers"] = tables["customers"].loc[tables["customers"].customer_id.isin(ids)]
    scoped["events"] = tables["events"].loc[tables["events"].customer_id.isin(ids) & tables["events"].event_at.lt(as_of_at)]
    purchase_ids = scoped["events"].loc[scoped["events"].event_type.eq("purchase"), "event_id"]
    scoped["orders"] = tables["orders"].loc[tables["orders"].purchase_event_id.isin(purchase_ids)]
    scoped["order_items"] = tables["order_items"].loc[tables["order_items"].order_id.isin(scoped["orders"].order_id)]
    orders = {row.purchase_event_id: row for row in scoped["orders"].itertuples(index=False)}
    baskets = {}
    for item in scoped["order_items"].itertuples(index=False):
        basket = baskets.setdefault(item.order_id, {})
        basket[item.product_id] = basket.get(item.product_id, 0) + int(item.quantity)
    journals = {uid: {} for uid in ids}
    negative_prefix = set()
    for event in scoped["events"].sort_values(["event_at", "event_id"]).itertuples(index=False):
        inventory = journals[event.customer_id]
        if event.event_type in {"cart", "remove_from_cart"}:
            sign = 1 if event.event_type == "cart" else -1
            inventory[event.product_id] = inventory.get(event.product_id, 0) + sign * int(event.quantity)
        elif event.event_type == "purchase":
            order = orders[event.event_id]
            if order.checkout_source in {"cart", "mixed"}:
                for product, quantity in baskets[order.order_id].items():
                    held = inventory.get(product, 0)
                    consumed = quantity if order.checkout_source == "cart" else min(held, quantity)
                    inventory[product] = held - consumed
            elif order.checkout_source != "buy_now":
                raise ValueError("unknown checkout_source")
        if any(quantity < 0 for quantity in inventory.values()):
            negative_prefix.add(event.customer_id)
    prices = tables["products"].set_index("product_id").unit_price_minor
    mismatches, summaries = 0, []
    for uid in ids:
        expected = {pid: quantity for pid, quantity in journals[uid].items() if quantity > 0}
        actual = cart_at(scoped, uid, as_of_at)
        mismatches += expected != actual or uid in negative_prefix
        summaries.append({"quantity": sum(actual.values()), "skus": len(actual),
                          "gross_minor": sum(qty * int(prices.loc[pid]) for pid, qty in actual.items())})
    return {"customers": len(ids), "as_of_at": as_of_at.isoformat(), "strictly_before_snapshot": True,
            "mismatches": int(mismatches), "passed": mismatches == 0,
            "quantity": _summary([r["quantity"] for r in summaries]), "sku": _summary([r["skus"] for r in summaries]),
            "gross_minor": _summary([r["gross_minor"] for r in summaries])}


def audit_coupon_responses(tables, metadata, coupon_response_diagnostics, treatments, response_model=None):
    """Check scenario bookkeeping against raw history and realized coupon orders."""
    if coupon_response_diagnostics is None:
        if metadata.get("coupon_purchase_response") is not None:
            raise ValueError("coupon response diagnostics are required when a response model is declared")
        return {"passed": True, "status": "not_present", "customers": 0, "mismatch_counts": {}}
    required = {"customer_id", "as_of_at", "treatment", "coupon_id", "history_cohort", "source_cohort",
                "response_probability", "response_uniform", "response_selected", "source_delay_days",
                "response_ready_at", "opportunity_at", "baseline_action", "purchase_event_id",
                "promoted_purchase", "coupon_used", "response_status"}
    if not required <= set(coupon_response_diagnostics):
        raise ValueError("coupon response diagnostics are missing required fields")
    diagnostic = _frame_times(coupon_response_diagnostics, ["as_of_at", "response_ready_at", "opportunity_at"])
    assignments = _frame_times(treatments, ["as_of_at"])
    if (diagnostic.as_of_at.isna().any() or diagnostic.duplicated(["customer_id", "as_of_at"]).any()
            or not set(diagnostic.customer_id) == set(tables["customers"].customer_id)):
        raise ValueError("coupon response diagnostics must uniquely cover every customer and snapshot")
    joined = diagnostic.merge(assignments[["customer_id", "as_of_at", "treatment", "coupon_id"]],
                              on=["customer_id", "as_of_at"], how="outer", validate="one_to_one",
                              suffixes=("", "_assignment"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("coupon response diagnostics must cover all assignment snapshots")
    declaration = metadata.get("coupon_purchase_response") or {}
    if response_model is None:
        path = Path(declaration["model_path"])
        model_bytes = path.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != declaration.get("model_sha256"):
            raise ValueError("coupon response model SHA256 differs from the dataset generation model")
        response_model = json.loads(model_bytes)
    if (response_model.get("type") != "tianchi_o2o_descriptive_response" or response_model.get("inactivity_days") != 28
            or response_model.get("window_days") != 30):
        raise ValueError("unsupported coupon response scenario model")
    calibration = metadata.get("coupon_effect_calibration")
    if calibration is not None and not isinstance(calibration, dict):
        raise ValueError("coupon effect calibration must be an object")
    odds_multiplier = calibration.get("redemption_odds_multiplier", 1.) if calibration is not None else 1.
    requires_policy_purchase = calibration is not None or declaration.get("requires_policy_purchase") is True
    groups = response_model["groups"]
    orders = _order_events(tables)
    by_event = orders.set_index("purchase_event_id")
    used_order_counts = orders.groupby("coupon_id").size().to_dict()
    coupons = tables["coupons"].set_index("coupon_id")
    mismatches = Counter()
    cohorts = Counter()
    for t0, group in joined.groupby("as_of_at", sort=True):
        # Reconstruct only the observed prefix, including the exact 28-day lower boundary.
        prior = orders.loc[orders.event_at.lt(t0)].groupby("customer_id").event_at.max()
        mismatches["snapshot_differs_from_metadata"] += int(t0 != _time(metadata["as_of_at"])) * len(group)
        for row in group.itertuples(index=False):
            last = prior.get(row.customer_id)
            cohort = ("no_observed_purchase" if last is None else "purchased_within_28d"
                      if t0 - last <= pd.Timedelta(days=28) else "lapsed_purchase")
            source_cohort = cohort if groups[cohort]["receipts"] else "pooled"
            source = groups[source_cohort]
            probability = scale_redemption_probability(source["response_probability"], odds_multiplier)
            key = json.dumps(["tianchi-coupon-response-v1", int(metadata["seed"]), row.customer_id, t0.isoformat()]).encode()
            noise = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
            uniform = float(noise.random())
            selected = uniform < probability
            delay = sample_redemption_delay(source, noise) if selected else None
            cohorts[cohort] += 1
            mismatches["history_cohort"] += row.history_cohort != cohort
            mismatches["source_cohort"] += row.source_cohort != source_cohort
            mismatches["response_probability"] += not np.isclose(row.response_probability, probability, rtol=0, atol=1e-12)
            mismatches["response_uniform"] += not np.isclose(row.response_uniform, uniform, rtol=0, atol=1e-15)
            valid_flags = all(isinstance(value, (bool, np.bool_)) for value in
                              (row.response_selected, row.coupon_used, row.promoted_purchase))
            mismatches["boolean_flags"] += not valid_flags
            if not valid_flags:
                continue
            mismatches["response_selected"] += row.response_selected != selected
            mismatches["source_delay_days"] += (pd.isna(row.source_delay_days) if selected else pd.notna(row.source_delay_days)) or (selected and row.source_delay_days != delay)
            ready = t0 + pd.Timedelta(days=delay) if selected else pd.NaT
            mismatches["source_clock"] += pd.isna(row.response_ready_at) if selected else pd.notna(row.response_ready_at)
            if selected and pd.notna(row.response_ready_at):
                mismatches["source_clock"] += row.response_ready_at != ready
            treated = row.treatment_assignment == 1
            mismatches["treatment"] += row.treatment != row.treatment_assignment
            same_coupon = ((pd.isna(row.coupon_id) and pd.isna(row.coupon_id_assignment)) or
                           (pd.notna(row.coupon_id) and pd.notna(row.coupon_id_assignment) and row.coupon_id == row.coupon_id_assignment))
            mismatches["assignment_coupon"] += not same_coupon
            used_count = used_order_counts.get(row.coupon_id_assignment, 0) if pd.notna(row.coupon_id_assignment) else 0
            mismatches["used_order_count"] += used_count != int(row.coupon_used)
            status = "control" if not treated else "redeemed" if row.coupon_used else "no_eligible_opportunity" if selected else "no_response"
            mismatches["response_status"] += row.response_status != status
            if not row.coupon_used:
                mismatches["unused_response_has_purchase"] += pd.notna(row.purchase_event_id) or pd.notna(row.opportunity_at) or pd.notna(row.baseline_action) or row.promoted_purchase
                continue
            mismatches["unselected_or_control_used"] += not selected or not treated
            mismatches["promotion_flag"] += row.promoted_purchase != (row.baseline_action != "purchase")
            if requires_policy_purchase:
                mismatches["forced_purchase"] += row.promoted_purchase
            mismatches["baseline_action"] += pd.notna(row.baseline_action) and row.baseline_action not in {"view", "cart", "remove", "purchase"}
            if pd.isna(row.purchase_event_id) or row.purchase_event_id not in by_event.index:
                mismatches["used_purchase_event_fk"] += 1
                continue
            order = by_event.loc[row.purchase_event_id]
            mismatches["used_order_customer"] += order.customer_id != row.customer_id
            mismatches["used_order_coupon"] += pd.isna(order.coupon_id) or order.coupon_id != row.coupon_id
            mismatches["opportunity_time"] += pd.isna(row.opportunity_at) or row.opportunity_at != order.event_at
            if pd.isna(row.coupon_id) or row.coupon_id not in coupons.index:
                mismatches["source_coupon_fk"] += 1
                continue
            coupon = coupons.loc[row.coupon_id]
            mismatches["source_coupon_customer"] += coupon.customer_id != row.customer_id
            mismatches["used_response_clock"] += pd.isna(ready) or not (max(t0, coupon.sent_at, ready) <= order.event_at < coupon.expires_at)
    return {"passed": not any(mismatches.values()), "status": "checked", "customers": len(joined),
            "mismatch_counts": {key: int(value) for key, value in mismatches.items()}, "history_cohort_counts": dict(cohorts),
            "coupon_used": int(diagnostic.coupon_used.sum()),
            "interpretation": "Observed-history and factual-order consistency for a transferred descriptive response scenario; not individual causal truth or identified purchase uplift."}


def _generated_interactions(tables):
    events = tables["events"]
    ordinary = events.loc[events.event_type.isin(["view", "cart", "remove_from_cart"]), ["customer_id", "session_id", "event_at", "event_id", "event_type", "product_id", "quantity", "unit_price_minor"]].copy()
    purchased = tables["order_items"].merge(_order_events(tables)[["order_id", "customer_id", "session_id", "event_at", "event_id"]],
                                            on="order_id", validate="many_to_one")
    purchased["event_type"] = "purchase"
    interactions = pd.concat([ordinary, purchased[ordinary.columns]], ignore_index=True)
    interactions["quantity"] = pd.to_numeric(interactions.quantity).fillna(1)
    interactions = interactions.merge(tables["products"][["product_id", "category_id"]], on="product_id", validate="many_to_one")
    return interactions


def _source_interactions(rows):
    return pd.DataFrame({"customer_id": rows.user_id, "session_id": rows.session.astype(str), "event_at": rows["at"],
                         "event_id": rows.source_event_id, "event_type": rows.event_type, "product_id": rows.product_id,
                         "quantity": 1, "unit_price_minor": rows.price_minor, "category_id": rows.category_id})


def _source_sessions(rows):
    return rows.groupby(["user_id", "session"], sort=False).agg(started_at=("at", "min"), ended_at=("at", "max")).reset_index().rename(
        columns={"user_id": "customer_id", "session": "session_id"}).assign(session_id=lambda frame: frame.session_id.astype(str))


def removal_transitions(interactions, observation_end):
    frame = interactions.sort_values(["customer_id", "product_id", "event_at", "event_id"], kind="stable").copy()
    keys = [frame.customer_id, frame.product_id]
    for action in ("cart", "purchase"):
        frame[f"next_{action}"] = frame.event_at.where(frame.event_type.eq(action)).groupby(keys, sort=False).bfill()
    removed = frame.loc[frame.event_type.eq("remove_from_cart")]
    result = {"removals": len(removed), "definition": "same customer/product, later in event order; only removals with full followup enter rates", "horizons": {}}
    for days in (7, 30):
        until = removed.event_at + pd.Timedelta(days=days)
        eligible = until.le(observation_end)
        recart = eligible & removed.next_cart.notna() & removed.next_cart.lt(until)
        buy = eligible & removed.next_purchase.notna() & removed.next_purchase.lt(until)
        denominator = int(eligible.sum())
        result["horizons"][str(days)] = {"eligible": denominator, "right_censored": len(removed) - denominator,
            "later_cart": int(recart.sum()), "later_purchase": int(buy.sum()),
            "cart_share": float(recart.sum() / denominator) if denominator else None,
            "purchase_share": float(buy.sum() / denominator) if denominator else None}
    return result


def _metric_arrays(interactions, sessions, lower, upper):
    scoped = interactions.loc[interactions.event_type.isin(["view", "cart", "remove_from_cart", "purchase"]) &
                              interactions.event_at.ge(lower) & interactions.event_at.lt(upper)]
    complete = sessions.loc[sessions.started_at.ge(lower) & sessions.ended_at.lt(upper)].sort_values(["customer_id", "started_at"])
    lengths = scoped.groupby(["customer_id", "session_id"]).quantity.sum().reindex(
        pd.MultiIndex.from_frame(complete[["customer_id", "session_id"]])).fillna(0)
    idle = complete.started_at - complete.groupby("customer_id").ended_at.shift()
    purchased = scoped.loc[scoped.event_type.eq("purchase")]
    purchase_pairs = purchased[["customer_id", "session_id", "product_id"]].drop_duplicates()
    pair_counts = purchase_pairs.groupby(["customer_id", "product_id"]).size()
    purchase_sessions = purchased[["customer_id", "session_id", "event_at"]].groupby(["customer_id", "session_id"]).event_at.min().reset_index().sort_values(["customer_id", "event_at"])
    purchase_gap = purchase_sessions.groupby("customer_id").event_at.diff().dt.total_seconds().dropna()
    counts = scoped.groupby("event_type").quantity.sum()
    total = counts.sum()
    arrays = {"session_item_proxy_length": lengths.to_numpy(dtype=float),
              "session_duration_seconds": (complete.ended_at - complete.started_at).dt.total_seconds().to_numpy(),
              "idle_gap_seconds": idle.dt.total_seconds().dropna().to_numpy(),
              "purchase_session_gap_seconds": purchase_gap.to_numpy(),
              "purchased_price_minor": _numbers(purchased.unit_price_minor),
              "product_diversity_per_observed_customer": scoped.groupby("customer_id").product_id.nunique().to_numpy(),
              "category_diversity_per_observed_customer": scoped.groupby("customer_id").category_id.nunique().to_numpy(),
              "purchased_product_diversity_per_buyer": purchased.groupby("customer_id").product_id.nunique().to_numpy()}
    summary = {"lower_inclusive": lower.isoformat(), "upper_exclusive": upper.isoformat(),
               "observed_customers": int(scoped.customer_id.nunique()), "complete_sessions": len(complete),
               "item_interaction_rows": len(scoped), "item_proxy_weight": float(total),
               "item_proxy_action_share": {str(key): float(value / total) for key, value in counts.items()} if total else {},
               "repeat_product_across_sessions": {"customer_product_pairs": len(pair_counts),
                   "pairs_in_two_or_more_sessions": int(pair_counts.ge(2).sum()),
                   "share": float(pair_counts.ge(2).mean()) if len(pair_counts) else None},
               "remove_then_later": removal_transitions(scoped, upper)}
    return arrays, purchased.quantity.to_numpy(dtype=float), summary


def compare_source(tables, metadata, artifact, source_rows):
    _, outcome_days = _outcome_contract(metadata)
    source = _source_interactions(source_rows)
    source_sessions = _source_sessions(source_rows)
    generated = _generated_interactions(tables)
    lower = _observation_bound(metadata, "start")
    upper = _observation_bound(metadata, "end")
    t0 = _time(metadata["as_of_at"])
    cutoff = _time(artifact["validation"]["cutoff_at_utc"])
    held_users = source.customer_id.map(lambda uid: hashlib.sha256(f"{artifact['seed']}:{uid}".encode()).digest()[0] >= 205)
    source_start, source_end = _time(artifact["observation_start"]), _time(artifact["observation_end"])
    held_ids = set(source.loc[held_users, "customer_id"])
    source_views = {
        "all_sampled_source": (source, source_sessions, source_start, source_end),
        "user_holdout": (source.loc[held_users], source_sessions.loc[source_sessions.customer_id.isin(held_ids)], source_start, cutoff),
        "time_holdout": (source.loc[~held_users], source_sessions.loc[~source_sessions.customer_id.isin(held_ids)], cutoff, source_end),
        "user_and_time_holdout": (source.loc[held_users], source_sessions.loc[source_sessions.customer_id.isin(held_ids)], cutoff, source_end)}
    generated_views = {"all_observed": (lower, upper), "pre_snapshot_30d": (t0 - pd.Timedelta(days=30), t0),
                       f"outcome_{outcome_days}d": (t0, t0 + pd.Timedelta(days=outcome_days))}
    references = {}
    source_arrays = {}
    for name, (events, visits, start, end) in source_views.items():
        arrays, weights, summary = _metric_arrays(events, visits, start, end)
        references[name] = summary
        source_arrays[name] = (arrays, weights)
    results = {}
    for name, (start, end) in generated_views.items():
        arrays, weights, summary = _metric_arrays(generated, tables["sessions"], start, end)
        raw_counts = tables["events"].loc[tables["events"].event_at.ge(start) & tables["events"].event_at.lt(end)].event_type.value_counts()
        summary["canonical_business_event_counts"] = {str(key): int(value) for key, value in raw_counts.items()}
        summary["canonical_business_event_share"] = {str(key): float(value / raw_counts.sum()) for key, value in raw_counts.items()} if raw_counts.sum() else {}
        summary["canonical_purchase_is_order_header"] = True
        comparisons = {}
        for source_name, (reference, reference_weights) in source_arrays.items():
            numeric = {}
            for key in arrays:
                unit = "minor_source_unit" if key.endswith("minor") else "seconds" if key.endswith("seconds") else "count"
                numeric[key] = _distance(reference[key], arrays[key], unit,
                                         reference_weights if key == "purchased_price_minor" else None,
                                         weights if key == "purchased_price_minor" else None)
            source_share = references[source_name]["item_proxy_action_share"]
            generated_share = summary["item_proxy_action_share"]
            difference = {action: generated_share.get(action, 0) - source_share.get(action, 0)
                          for action in ("view", "cart", "remove_from_cart", "purchase")}
            comparisons[source_name] = {"numeric_distances": numeric, "action_share_difference": difference,
                                        "action_share_L1": sum(abs(value) for value in difference.values())}
        results[name] = {"summary": summary, "comparisons": comparisons}
    return {"source_scopes": references, "generated_scopes": results,
            "interpretation": [
                "Source purchase rows are product interactions without true order ID/quantity; generated canonical purchase events are order headers.",
                "Item-proxy comparisons replace each generated purchase header by its order-item quantities; source one-row-one-unit is an explicit proxy.",
                "Coupon usage, search, review and CS are excluded from four-action shopping interactions, source action shares, sequences and numeric distribution metrics.",
                "Purchased-price Wasserstein and summaries weight generated lines by quantity; KS uses unweighted item records and has no iid significance interpretation.",
                "Session metrics require completed sessions inside the scope; event metrics use event time and half-open boundaries.",
                "Repeat products require purchases in two distinct observed sessions, not multiple units in one checkout.",
                "Source holds five months and a deterministic ID sample; generated runs may span 24 months and select active-before-snapshot customers.",
                "Retention calibration, experiment policy, longer followup, source cart censoring, and fixed modal catalog prices can cause real distribution differences.",
                "Distances are descriptive. No similarity pass threshold or causal claim is inferred from them.",
            ]}


def audit_event_model_provenance(metadata):
    result = {}
    for name, declaration in metadata.get("event_models", {}).items():
        path = Path(declaration["model_path"])
        model_bytes = path.read_bytes()
        digest = hashlib.sha256(model_bytes).hexdigest()
        if digest != declaration.get("model_sha256"):
            raise ValueError(f"{name} event model SHA256 differs from the dataset generation model")
        artifact = json.loads(model_bytes)
        if declaration.get("type") != artifact.get("type"):
            raise ValueError(f"{name} event model type differs from the generation declaration")
        result[name] = {"path": str(path), "sha256": digest, "type": artifact.get("type"), "status": "verified"}
    return result


def validate_directory(data_dir: Path, model_path: Path, source_cache: Path | None = None) -> dict:
    data_dir, model_path = Path(data_dir), Path(model_path)
    tables, metadata = load_raw_dataset(data_dir / "raw.sqlite")
    check_generation_sidecars(data_dir / "raw.sqlite", metadata)
    raw_manifest = json.loads((data_dir / "raw.manifest.json").read_text())
    sidecars = {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in ("labels", "treatments")}
    optional = {name: pd.read_parquet(data_dir / f"{name}.parquet") if (data_dir / f"{name}.parquet").exists() else None
                for name in ("lifecycle", "diagnostics", "coupon_response_diagnostics")}
    model_bytes = model_path.read_bytes()
    artifact = json.loads(model_bytes)
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if metadata.get("model_sha256") != model_sha256:
        raise ValueError("supplied model SHA256 differs from the dataset generation model")
    return_provenance = {"path": metadata.get("return_model_path"),
                         "recorded_sha256": metadata.get("return_model_sha256"), "status": "not_used"}
    if return_provenance["path"]:
        return_path = Path(return_provenance["path"])
        return_provenance["status"] = "missing_file_not_verified"
        if return_path.exists():
            return_provenance["actual_sha256"] = hashlib.sha256(return_path.read_bytes()).hexdigest()
            if return_provenance["actual_sha256"] != return_provenance["recorded_sha256"]:
                raise ValueError("return model SHA256 differs from the dataset generation model")
            return_provenance["status"] = "verified"
    if metadata["currency"] != artifact["currency"] or str(metadata["amount_unit"]) != str(artifact["minor_unit"]):
        raise ValueError("source and generated currency/minor-unit declarations differ")
    observations = audit_sidecars(tables, metadata, **sidecars,
                                  lifecycle=optional["lifecycle"], diagnostics=optional["diagnostics"])
    cart = audit_cart_sample(tables, _time(metadata["as_of_at"]))
    coupon_response = audit_coupon_responses(tables, metadata, optional["coupon_response_diagnostics"], sidecars["treatments"])
    event_model_provenance = audit_event_model_provenance(metadata)
    source_paths = artifact["source"].get("files") or []
    if not source_paths:
        raise ValueError("source profile artifact has no inspected raw source files")
    if source_cache is None:
        folder = Path(source_paths[0]["path"]).parent
        source, source_provenance = load_source(folder, artifact["source"]["sample_modulus"])
        source, _, source_quality = prepare_source(source)
        if len(source_paths) != len(source_provenance["files"]):
            raise ValueError("raw source file count differs from the fitted model")
        for recorded, actual in zip(source_paths, source_provenance["files"]):
            if any(recorded[key] != actual[key] for key in ("path", "bytes", "mtime_ns")):
                raise ValueError("raw source provenance differs from the fitted model")
    else:
        from .irl.fit_donor_prior import load_prior_source
        source, descriptor = load_prior_source(artifact, source_cache)
        source_quality = {"mode": "provenance-bound prepared source cache", "rows": len(source),
                          "cache_sha256": hashlib.sha256(Path(source_cache).read_bytes()).hexdigest(),
                          "fingerprint": descriptor["fingerprint"],
                          "recorded_preparation_audit": descriptor.get("source_quality")}
    return {"schema_version": 1, "dataset_id": metadata["dataset_id"],
            "data_dir": str(data_dir.resolve()), "model_path": str(model_path.resolve()),
            "model_sha256": model_sha256,
            "dataset_model_sha256": metadata.get("model_sha256"),
            "outcome_view_provenance": {key: metadata.get(key) for key in (
                "definition_id", "outcome_window_days", "generation_window_days", "eligibility_window_days", "coupon_validity_days",
                "generation_definition_id", "generation_policy_id", "parent_dataset_id", "raw_database_sha256",
                "source_raw_database_sha256", "analysis_view", "analytic_outcome_view", "generation_observed_churn_rate")},
            "return_model_provenance": return_provenance,
            "event_model_provenance": event_model_provenance,
            "generator_calibration": {key: metadata.get(key) for key in (
                "distribution", "return_model_type", "target_churn_rate", "expected_control_churn_rate",
                "history_logit_shift", "outcome_logit_shift", "candidate_customers", "eligible_customers")},
            "generator_assumptions": metadata.get("assumptions", []),
            "currency": metadata["currency"], "amount_unit": metadata["amount_unit"],
            "raw_integrity": {"passed": True, "database_sha256": raw_manifest["database_sha256"],
                              "method": "load_raw_dataset checksum, schema, foreign keys, amounts, coupons and complete cart replay"},
            "observation_and_ITT": observations, "cart_sample": cart, "coupon_response_diagnostics": coupon_response,
            "coupon_usage": observations["coupon_usage"],
            "auxiliary_events": observations["auxiliary_events"],
            "exact_checks_passed": observations["passed"] and cart["passed"] and coupon_response["passed"],
            "source_quality": source_quality,
            "source_provenance_check": "Exact paths, file count, sizes and mtime_ns; original full-file hashes were not recorded by the fitter.",
            "distribution_comparison": compare_source(tables, metadata, artifact, source),
            "threshold_policy": "Exact record consistency only. Distribution distances have no invented acceptance threshold."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, help="Reuse unchanged prepared rows after checking original source provenance")
    parser.add_argument("--model", type=Path, help="Defaults to the source model recorded in the raw manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        model = args.model
        if model is None:
            manifest = json.loads((args.data_dir / "raw.manifest.json").read_text())
            model = Path(manifest["metadata"]["model_path"])
        report = (validate_directory(args.data_dir, model, args.source_cache) if args.source_cache is not None
                  else validate_directory(args.data_dir, model))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as exc:
        report = {"exact_checks_passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "exact_checks_passed": report["exact_checks_passed"]}))
    return 0 if report["exact_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
