"""Observed Seoul-week aggregates and the fixed, once-only experiment ledger."""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


SEOUL = "Asia/Seoul"
WEEK = pd.Timedelta(days=7)


def _utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("weekly exports require timezone-aware timestamps")
    return value.tz_convert("UTC")


def _bounds(metadata):
    start = _utc(metadata.get("observation_start_at", metadata.get("start_at")))
    end = _utc(metadata.get("observation_end_at", metadata.get("end_at")))
    if end <= start:
        raise ValueError("observation_end must follow observation_start")
    for key in ("dataset_id", "currency", "amount_unit"):
        if not str(metadata.get(key, "")).strip():
            raise ValueError(f"metadata.{key} is required")
    if isinstance(metadata.get("seed"), bool) or not isinstance(metadata.get("seed"), numbers.Integral):
        raise ValueError("metadata.seed must be an integer")
    return start, end


def _times(values):
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    if any(pd.Timestamp(value).tzinfo is None for value in values.dropna()):
        raise ValueError("weekly exports require timezone-aware timestamps")
    return pd.to_datetime(values, utc=True)


def _week(values):
    if isinstance(values, pd.Timestamp):
        local = _utc(values).tz_convert(SEOUL)
        return local.normalize() - pd.Timedelta(days=local.weekday())
    local = _times(values).dt.tz_convert(SEOUL)
    return local.dt.normalize() - pd.to_timedelta(local.dt.weekday, unit="D")


def _orders(tables):
    purchases = tables["events"].loc[tables["events"].event_type.eq("purchase"),
                                      ["event_id", "customer_id", "session_id", "event_at"]]
    return tables["orders"].merge(purchases, left_on="purchase_event_id", right_on="event_id", validate="one_to_one")


def _assignment_rows(tables):
    frame = tables["treatments"].copy()
    required = {"customer_id", "as_of_at", "assigned_at", "treatment", "assignment_probability"}
    if not required <= set(frame):
        raise ValueError(f"assignment log requires {sorted(required)}")
    if frame.customer_id.duplicated().any():
        raise ValueError("this fixed experiment requires exactly one assignment per customer")
    if not set(frame.customer_id) == set(tables["customers"].customer_id):
        raise ValueError("assignment log must retain every customer, including unsent/unredeemed customers")
    if not frame.treatment.isin([0, 1]).all():
        raise ValueError("treatment must be fixed zero/one")
    probability = pd.to_numeric(frame.assignment_probability, errors="raise")
    if not probability.eq(0.5).all():
        raise ValueError("the declared fixed 1:1 experiment requires recorded probability 0.5")
    frame["as_of_at"] = _times(frame.as_of_at)
    frame["assigned_at"] = _times(frame.assigned_at)
    return frame


def build_experiment_log(tables, metadata) -> pd.DataFrame:
    """One factual assignment row per customer, regardless of execution or redemption."""
    _, observation_end = _bounds(metadata)
    frame = _assignment_rows(tables)
    labels = tables["labels"]
    if labels.duplicated(["customer_id", "as_of_at"]).any():
        raise ValueError("duplicate experiment labels")
    label_fields = [name for name in ("customer_id", "as_of_at", "snapshot_id", "outcome_end_at", "churn",
                                      "definition_id", "is_censored") if name in labels]
    selected_labels = labels[label_fields].copy()
    selected_labels["as_of_at"] = _times(selected_labels.as_of_at)
    selected_labels = selected_labels.rename(columns={"churn": "factual_churn_label", "definition_id": "churn_definition_id"})
    frame = frame.merge(selected_labels, on=["customer_id", "as_of_at"], how="left", validate="one_to_one")
    if "outcome_end_at" not in frame:
        days = metadata.get("outcome_window_days", metadata.get("analysis_outcome_days"))
        if isinstance(days, bool) or not isinstance(days, numbers.Integral) or days <= 0:
            raise ValueError("supply labels.outcome_end_at or a positive integer metadata.outcome_window_days")
        frame["outcome_end_at"] = frame.as_of_at + pd.Timedelta(days=int(days))
    frame["outcome_end_at"] = _times(frame.outcome_end_at)
    if frame.outcome_end_at.isna().any() or not frame.outcome_end_at.gt(frame.as_of_at).all():
        raise ValueError("every assignment requires an explicit positive outcome horizon")
    frame["outcome_horizon_days"] = (frame.outcome_end_at - frame.as_of_at).dt.total_seconds() / 86400
    frame["outcome_observed"] = frame.outcome_end_at.le(observation_end)
    if "is_censored" in frame and not frame.is_censored.astype(bool).eq(~frame.outcome_observed).all():
        raise ValueError("label censoring disagrees with the observation horizon")
    if "factual_churn_label" in frame and frame.loc[~frame.outcome_observed, "factual_churn_label"].notna().any():
        raise ValueError("unobserved future outcomes cannot have factual churn labels")
    frame["group"] = np.where(frame.treatment.eq(1), "treatment", "control")
    frame["assignment_known_at"] = frame.assigned_at
    frame["group_is_retrospective_tag"] = True
    if "coupon_action" in frame:
        action = pd.to_numeric(frame.coupon_action, errors="raise")
    elif "decision_action" in frame:
        action = frame.decision_action.map({"send_coupon": 1, "no_coupon": 0})
    else:
        # This experiment assigns exactly one coupon action to Treatment; execution is separate.
        action = frame.treatment.copy()
    if not action.isin([0, 1]).all():
        raise ValueError("coupon_action must be an explicit send/no-send decision")
    frame["coupon_action"] = action.astype("Int64")
    frame["decision_action"] = np.where(action.eq(1), "send_coupon", "no_coupon")
    frame["requested_coupon_id"] = frame.get("coupon_id", pd.Series(pd.NA, index=frame.index))
    recorded_actual = frame["actual_sent"].copy() if "actual_sent" in frame else None
    raw_coupons = tables["coupons"]
    coupon_context = raw_coupons.merge(frame[["customer_id", "assigned_at", "treatment"]], on="customer_id", validate="many_to_one")
    coupon_context["sent_at"] = _times(coupon_context.sent_at)
    coupon_context["expires_at"] = _times(coupon_context.expires_at)
    prior = coupon_context.loc[coupon_context.sent_at.lt(coupon_context.assigned_at)]
    new = coupon_context.loc[coupon_context.sent_at.ge(coupon_context.assigned_at)].copy()
    if new.customer_id.duplicated().any():
        raise ValueError("the once-only policy forbids additional coupon sends")
    if new.treatment.eq(0).any():
        raise ValueError("Control must not receive a new coupon")
    if not new.sent_at.eq(new.assigned_at).all():
        raise ValueError("this policy sends only at the w0 assignment time")
    if not new.expires_at.eq(new.sent_at + pd.Timedelta(days=30)).all():
        raise ValueError("the declared coupon validity is exactly thirty days")
    execution = new.set_index("customer_id")
    frame["coupon_id"] = frame.customer_id.map(execution.coupon_id)
    frame["actual_sent"] = frame.coupon_id.notna()
    if recorded_actual is not None and not recorded_actual.astype(bool).eq(frame.actual_sent).all():
        raise ValueError("actual_sent disagrees with raw coupon records")
    if (frame.actual_sent & frame.coupon_action.eq(0)).any():
        raise ValueError("raw coupon execution has no send decision")
    for column in ("sent_at", "expires_at", "discount_rate"):
        frame[column] = frame.customer_id.map(execution[column])
    frame["send_cost_minor"] = frame.customer_id.map(execution.send_cost_minor).astype("Int64").fillna(0)
    default_status = pd.Series(np.where(frame.actual_sent, "sent", np.where(frame.coupon_action.eq(1), "not_sent", "not_requested")), index=frame.index)
    if "execution_status" in frame:
        frame["execution_status"] = frame.execution_status.fillna(default_status)
    else:
        frame["execution_status"] = default_status
    if "failure_reason" not in frame:
        frame["failure_reason"] = pd.NA
    if (frame.execution_status.eq("sent") & ~frame.actual_sent).any() or (frame.execution_status.isin(["failed", "not_sent"]) & frame.actual_sent).any():
        raise ValueError("execution_status contradicts actual raw sends")
    frame["past_coupon_count"] = frame.customer_id.map(prior.groupby("customer_id").size()).fillna(0).astype("Int64")
    frame["has_past_coupon_history"] = frame.past_coupon_count.gt(0)
    frame["past_coupon_history_scope"] = "observed raw sends strictly before assignment; unobserved pre-observation history is unknown"
    frame["additional_send_count"] = pd.Series(0, index=frame.index, dtype="Int64")
    orders = _orders(tables)
    redeemed = orders.loc[orders.coupon_id.notna()].copy()
    if redeemed.coupon_id.duplicated().any():
        raise ValueError("one coupon cannot redeem on multiple orders")
    redemption = redeemed.set_index("coupon_id")
    frame["redemption_order_id"] = frame.coupon_id.map(redemption.order_id)
    frame["redemption_at"] = frame.coupon_id.map(redemption.event_at)
    frame["redeemed"] = frame.redemption_order_id.notna()
    frame["redemption_discount_cost_minor"] = frame.coupon_id.map(redemption.discount_minor).fillna(0).astype("Int64")
    inside_outcome = frame.redemption_at.ge(frame.as_of_at) & frame.redemption_at.lt(frame.outcome_end_at)
    frame["redeemed_in_outcome"] = pd.array(inside_outcome.where(frame.outcome_observed | inside_outcome), dtype="boolean")
    frame["outcome_redemption_discount_cost_minor"] = frame.redemption_discount_cost_minor.where(inside_outcome, 0).where(frame.outcome_observed | inside_outcome).astype("Int64")
    old_redemption = redeemed.merge(frame[["customer_id", "assigned_at"]], on="customer_id", validate="many_to_one")
    old_redemption = old_redemption.loc[old_redemption.event_at.lt(old_redemption.assigned_at)]
    frame["past_coupon_redemption_count"] = frame.customer_id.map(old_redemption.groupby("customer_id").size()).fillna(0).astype("Int64")
    frame["eligible_at_assignment"] = False
    frame["factual_no_session"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["factual_no_order"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["factual_outcome_net_revenue_minor"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for (t0, until), indices in frame.groupby(["as_of_at", "outcome_end_at"]).groups.items():
        past_sessions = tables["sessions"].loc[tables["sessions"].started_at.ge(t0 - pd.Timedelta(days=30)) & tables["sessions"].started_at.lt(t0)]
        frame.loc[indices, "eligible_at_assignment"] = frame.loc[indices, "customer_id"].isin(past_sessions.customer_id)
        if observation_end >= until:
            future_sessions = tables["sessions"].loc[tables["sessions"].started_at.ge(t0) & tables["sessions"].started_at.lt(until)]
            future_orders = orders.loc[orders.event_at.ge(t0) & orders.event_at.lt(until)]
            frame.loc[indices, "factual_no_session"] = ~frame.loc[indices, "customer_id"].isin(future_sessions.customer_id)
            frame.loc[indices, "factual_no_order"] = ~frame.loc[indices, "customer_id"].isin(future_orders.customer_id)
            paid = future_orders.groupby("customer_id").paid_minor.sum()
            frame.loc[indices, "factual_outcome_net_revenue_minor"] = frame.loc[indices, "customer_id"].map(paid).fillna(0)
    frame["eligibility_definition"] = "session start in [as_of_at-30d,as_of_at)"
    frame["outcome_definition_no_session"] = "no session start in [as_of_at,outcome_end_at)"
    frame["outcome_definition_no_order"] = "no unique order in [as_of_at,outcome_end_at)"
    frame["label_known_at"] = frame.outcome_end_at.where(frame.outcome_observed)
    frame["posthoc_response_type"] = pd.NA
    for column in ("dataset_id", "seed", "currency", "amount_unit"):
        frame[column] = metadata[column]
    frame["predictor_use"] = "assignment/action/execution/redemption/outcomes/posthoc fields are excluded from pre-assignment X"
    return frame.reindex(columns=[
        "dataset_id", "seed", "customer_id", "experiment_id", "snapshot_id", "as_of_at", "assigned_at",
        "assignment_known_at", "group", "treatment", "assignment_probability", "group_is_retrospective_tag",
        "coupon_action", "decision_action", "requested_coupon_id", "actual_sent", "coupon_id", "sent_at", "expires_at",
        "discount_rate", "send_cost_minor", "execution_status", "failure_reason", "additional_send_count",
        "redeemed", "redemption_order_id", "redemption_at", "redemption_discount_cost_minor",
        "redeemed_in_outcome", "outcome_redemption_discount_cost_minor",
        "past_coupon_count", "past_coupon_redemption_count", "has_past_coupon_history", "past_coupon_history_scope",
        "eligible_at_assignment", "eligibility_definition", "outcome_end_at", "outcome_horizon_days", "outcome_observed",
        "factual_no_session", "factual_no_order", "factual_outcome_net_revenue_minor", "factual_churn_label",
        "churn_definition_id", "outcome_definition_no_session", "outcome_definition_no_order", "is_censored",
        "label_known_at", "posthoc_response_type", "currency", "amount_unit", "predictor_use"])


def _aggregate(frame, time_column, value=None):
    keys = [frame.customer_id, _week(frame[time_column])]
    result = frame.groupby(keys).size() if value is None else frame.groupby(keys)[value].sum()
    result.index.names = ["customer_id", "reference_week"]
    return result


def _session_seconds(sessions, observation_end):
    """Allocate duration to calendar-week overlap; an unknown end remains unknown."""
    start_week = _week(sessions.started_at)
    end = _times(sessions.ended_at)
    completed = end.notna()
    end_week = _week(end.where(completed, sessions.started_at) - pd.Timedelta(nanoseconds=1))
    same = completed & (end_week.eq(start_week) | end.eq(sessions.started_at))
    direct = sessions.loc[same, ["customer_id", "started_at"]].copy()
    direct["seconds"] = (end.loc[same] - sessions.loc[same, "started_at"]).dt.total_seconds()
    totals = _aggregate(direct, "started_at", "seconds")
    pieces, unknown = [], []
    for row in sessions.loc[~same].itertuples(index=False):
        first = _week(_utc(row.started_at))
        last = _week((_utc(row.ended_at) if pd.notna(row.ended_at) else observation_end) - pd.Timedelta(nanoseconds=1))
        for week in pd.date_range(first, last, freq="7D"):
            if pd.isna(row.ended_at):
                unknown.append((row.customer_id, week))
            else:
                seconds = (min(_utc(row.ended_at), (week + WEEK).tz_convert("UTC")) - max(_utc(row.started_at), week.tz_convert("UTC"))).total_seconds()
                pieces.append((row.customer_id, week, max(0.0, seconds)))
    if pieces:
        extra = pd.DataFrame(pieces, columns=["customer_id", "reference_week", "seconds"]).groupby(["customer_id", "reference_week"]).seconds.sum()
        totals = totals.add(extra, fill_value=0)
    unknown_index = pd.MultiIndex.from_tuples(unknown, names=["customer_id", "reference_week"])
    return totals, unknown_index


def build_weekly_panel(tables, metadata) -> pd.DataFrame:
    """Every customer × observed calendar week; missing coverage is never a zero."""
    start, end = _bounds(metadata)
    experiment = build_experiment_log(tables, metadata)
    customers = tables["customers"]
    if customers.customer_id.duplicated().any():
        raise ValueError("duplicate customer_id")
    weeks = pd.date_range(_week(start), _week(end - pd.Timedelta(nanoseconds=1)), freq="7D")
    index = pd.MultiIndex.from_product([customers.customer_id, weeks], names=["customer_id", "reference_week"])
    panel = index.to_frame(index=False)
    signups = _times(customers.signed_up_at)
    signup_by_id = pd.Series(signups.to_numpy(), index=customers.customer_id)
    panel["signed_up_at"] = pd.to_datetime(panel.customer_id.map(signup_by_id), utc=True)
    week_start = panel.reference_week.dt.tz_convert("UTC")
    coverage_start = week_start.where(week_start.ge(panel.signed_up_at), panel.signed_up_at).clip(lower=start)
    coverage_end = (week_start + WEEK).clip(upper=end)
    panel["coverage_seconds"] = (coverage_end - coverage_start).dt.total_seconds().clip(lower=0)
    observed = panel.coverage_seconds.gt(0)
    panel["observed_week"] = observed
    panel["partial_coverage"] = observed & panel.coverage_seconds.lt(WEEK.total_seconds())
    panel["coverage_start_at"] = coverage_start.where(observed)
    panel["coverage_end_at"] = coverage_end.where(observed)
    sessions, events, coupons = tables["sessions"], tables["events"], tables["coupons"]
    orders = _orders(tables)
    converted = orders.assign(_order_week=_week(orders.event_at)).drop_duplicates(["customer_id", "session_id", "_order_week"])
    aggregates = {"sessions": _aggregate(sessions, "started_at"),
                  "purchases": _aggregate(orders, "event_at"),
                  "converted_sessions": _aggregate(converted, "event_at"),
                  "gross_revenue_minor": _aggregate(orders, "event_at", "gross_minor"),
                  "redemption_discount_cost_minor": _aggregate(orders, "event_at", "discount_minor"),
                  "net_revenue_minor": _aggregate(orders, "event_at", "paid_minor"),
                  "coupons_sent": _aggregate(coupons, "sent_at"),
                  "marketing_send_cost_minor": _aggregate(coupons, "sent_at", "send_cost_minor"),
                  "coupon_actions": _aggregate(experiment.loc[experiment.coupon_action.eq(1)], "assigned_at")}
    event_counts = events.groupby([events.customer_id, _week(events.event_at), events.event_type]).size().unstack(fill_value=0)
    event_counts.index.names = ["customer_id", "reference_week"]
    for event_type, column in (("view", "views"), ("cart", "cart_adds"), ("remove_from_cart", "cart_removes"),
                               ("review", "reviews")):
        aggregates[column] = event_counts[event_type] if event_type in event_counts else pd.Series(dtype=int)
    for column, values in aggregates.items():
        panel[column] = pd.array(values.reindex(index, fill_value=0).to_numpy(), dtype="Int64")
        panel.loc[~observed, column] = pd.NA
    reviewed = events.loc[events.event_type.eq("review")]
    ratings = (reviewed.groupby([reviewed.customer_id, _week(reviewed.event_at)]).rating.mean()
               if "rating" in reviewed else pd.Series(dtype=float))
    panel["mean_review_rating"] = pd.array(ratings.reindex(index).to_numpy(), dtype="Float64")
    panel.loc[~observed, "mean_review_rating"] = pd.NA
    seconds, unknown = _session_seconds(sessions, end)
    panel["session_seconds"] = pd.array(seconds.reindex(index, fill_value=0).to_numpy(), dtype="Float64")
    uncertain = index.isin(unknown)
    panel["session_seconds_complete"] = observed & ~uncertain
    panel.loc[~observed | uncertain, "session_seconds"] = pd.NA
    panel["net_after_marketing_minor"] = panel.net_revenue_minor - panel.marketing_send_cost_minor
    panel["no_purchase_this_week"] = panel.purchases.eq(0).astype("boolean")
    panel["no_session_this_week"] = panel.sessions.eq(0).astype("boolean")
    tags = experiment.set_index("customer_id")
    for column in ("group", "treatment", "assignment_probability", "assigned_at", "assignment_known_at", "as_of_at", "snapshot_id", "eligible_at_assignment"):
        if column in tags:
            panel[column] = panel.customer_id.map(tags[column])
    assignment_week = _week(panel.assigned_at)
    panel["relative_week"] = ((panel.reference_week - assignment_week) / WEEK).astype("Int64")
    panel["group_is_retrospective_tag"] = True
    panel["assignment_known_at_week_start"] = panel.assignment_known_at.lt(week_start)
    panel["assignment_known_by_week_end"] = panel.assignment_known_at.lt(week_start + WEEK)
    for column in ("dataset_id", "seed", "currency", "amount_unit"):
        panel[column] = metadata[column]
    panel.attrs["coverage_assumption"] = "continuous simulated observation after signup within declared observation bounds"
    return panel
