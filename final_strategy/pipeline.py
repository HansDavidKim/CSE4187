"""Six business segments and an explicit, single-coupon 4-week planning budget."""
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from clv.bundle import file_sha256
from simulate.v4.time_spec import weekly_metadata

from .inputs import integrate_inputs
from .validation import KEYS, PROVENANCE, finite_number, keys, numeric, require

CUSTOMER_COLUMNS = ["customer_id", "churn_prob", "p0", "p1", "uplift_score", "clv", "segment",
                    "priority_score", "send", "expected_net_value"]
SEGMENTS = ("newly_observed", "high_value_persuadable", "low_value_persuadable",
            "high_value_sure_thing", "high_value_no_action", "low_value_no_action")


@dataclass(frozen=True)
class DecisionConfig:
    budget: float
    send_cost: float
    currency: str
    amount_unit: str
    high_value_threshold: float
    churn_threshold: float
    uplift_tolerance: float
    new_history_days: int
    eligible_segments: tuple[str, ...]
    baseline_purchase_threshold: float
    expected_discount_per_send: float | None = None

    def validate(self):
        for name in ("budget", "send_cost", "high_value_threshold", "uplift_tolerance"):
            finite_number(getattr(self, name), name)
        require(0 < finite_number(self.churn_threshold, "churn_threshold") < 1,
                "churn_threshold must be in (0,1)")
        require(self.uplift_tolerance < 1, "uplift_tolerance must be <1")
        require(0 < finite_number(self.baseline_purchase_threshold, "baseline_purchase_threshold") < 1,
                "baseline_purchase_threshold must be in (0,1)")
        require(type(self.new_history_days) is int and self.new_history_days >= 0,
                "new_history_days must be a nonnegative observed-history threshold")
        require(bool(self.currency) and bool(self.amount_unit), "Currency and amount unit must be explicit")
        require(len(set(self.eligible_segments)) == len(self.eligible_segments)
                and set(self.eligible_segments) <= set(SEGMENTS), "Unknown/duplicate eligible business segments")
        if self.expected_discount_per_send is not None:
            finite_number(self.expected_discount_per_send, "expected_discount_per_send")


@dataclass
class DecisionOutputs:
    customers: pd.DataFrame
    identity: pd.DataFrame
    availability: pd.DataFrame
    segment_summary: pd.DataFrame
    budget_summary: dict
    what_if: pd.DataFrame
    graph_data: dict
    metadata: dict


def _response_group(frame, config):
    tau, baseline = frame.uplift_score, frame.p0
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[tau.lt(-config.uplift_tolerance)] = "Sleeping Dogs"
    result.loc[tau.gt(config.uplift_tolerance)] = "Persuadables"
    known = tau.notna() & baseline.notna() & result.isna()
    result.loc[known & baseline.ge(config.baseline_purchase_threshold)] = "Sure Things"
    result.loc[known & baseline.lt(config.baseline_purchase_threshold)] = "Lost Causes"
    return result


def _segments(frame, response, config):
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    tenure = pd.to_numeric(frame.get("observed_tenure_days", pd.Series(np.nan, index=frame.index)), errors="raise")
    require(tenure.dropna().ge(0).all() and np.isfinite(tenure.dropna()).all(), "Observed tenure must be finite/nonnegative")
    new = tenure.lt(config.new_history_days)
    result.loc[new] = "newly_observed"
    known = tenure.notna() & ~new & frame.clv.notna() & frame.churn_prob.notna() & response.notna()
    high = frame.clv.ge(config.high_value_threshold)
    result.loc[known & high] = "high_value_no_action"
    result.loc[known & ~high] = "low_value_no_action"
    result.loc[known & high & response.eq("Sure Things")] = "high_value_sure_thing"
    at_risk = frame.churn_prob.ge(config.churn_threshold)
    result.loc[known & at_risk & high & response.eq("Persuadables")] = "high_value_persuadable"
    result.loc[known & at_risk & ~high & response.eq("Persuadables")] = "low_value_persuadable"
    return result


def _choose(frame, eligible, budget):
    chosen = pd.Series(0, index=frame.index, dtype="int8")
    remaining = budget
    ordered = frame.loc[eligible].sort_values(["priority_score", "customer_id"], ascending=[False, True], kind="stable")
    for i, row in ordered.iterrows():
        if row.expected_coupon_cost <= remaining + 1e-12:
            chosen.loc[i] = 1
            remaining -= row.expected_coupon_cost
    return chosen


def _budget(frame, selection, budget, unavailable):
    chosen = frame.loc[selection.eq(1)]
    cost = float(chosen.expected_coupon_cost.sum())
    revenue = float(chosen.expected_incremental_revenue.sum())
    unknown = unavailable == len(frame)
    return {"total_budget": float(budget), "send_count": len(chosen),
            "expected_send_cost": float(len(chosen) * frame.attrs["send_cost"]),
            "expected_discount_cost": None if unknown else float(chosen.expected_discount_if_sent.sum()),
            "expected_cost": None if unknown else cost, "expected_incremental_revenue": None if unknown else revenue,
            "expected_net_value": None if unknown else revenue - cost,
            "expected_roi": (revenue - cost) / cost if cost > 0 and not unknown else None,
            "status": "unavailable" if unknown else "available" if not unavailable else "partially_unavailable",
            "unavailable_customers": unavailable,
            "roi_reason": "financial prediction inputs unavailable" if unknown else "zero expected cost" if cost == 0 else None}


def build_decisions(features, predictions, config, *, expected_discounts=None, profile=None):
    """Use model estimates only; a missing monetary input never becomes zero profit."""
    config.validate()
    missing = [name for name in ("churn", "uplift", "clv") if predictions.get(name) is None]
    require(not missing, f"Missing required prediction batches: {missing}")
    frame = integrate_inputs(features, predictions)
    response = _response_group(frame, config)
    frame["segment"] = _segments(frame, response, config)
    frame["priority_score"] = frame.uplift_score.clip(lower=0) * frame.clv
    money = frame.attrs.get("source_monetary_values_observed", True)
    for name, expected in (("currency", config.currency), ("amount_unit", config.amount_unit)):
        if name not in frame or frame[name].isna().any():
            money = False
        else:
            require(frame[name].astype(str).eq(str(expected)).all(), f"Decision {name} differs from prediction units")
    discount = pd.Series(np.nan, index=frame.index, dtype=float)
    if expected_discounts is not None:
        supplied = keys(expected_discounts)
        require(len(supplied) == len(frame), "Expected-discount cohort differs")
        ordered = frame[KEYS].merge(supplied, on=KEYS, validate="one_to_one", how="left", indicator=True)
        require(ordered._merge.eq("both").all(), "Expected-discount customer/cutoff differs")
        require("expected_discount_if_sent" in ordered, "Expected discount must be an unconditional cost per send")
        require({"input_cutoff_at", "prediction_end_at"} <= set(ordered), "Expected-discount estimates require time endpoints")
        from .validation import utc
        require(utc(ordered.input_cutoff_at, "discount input cutoff").eq(ordered.as_of_at).all()
                and utc(ordered.prediction_end_at, "discount prediction endpoint").eq(ordered.as_of_at+pd.Timedelta(weeks=4)).all(),
                "Expected-discount estimates require the same preassignment cutoff and 4-week horizon")
        for name, expected in (("currency", config.currency), ("amount_unit", config.amount_unit)):
            require(name in ordered and ordered[name].astype(str).eq(str(expected)).all(), f"Expected-discount {name} differs")
        discount = numeric(ordered.expected_discount_if_sent, "expected_discount_if_sent")
    elif config.expected_discount_per_send is not None:
        discount[:] = config.expected_discount_per_send
    frame["expected_discount_if_sent"] = discount
    frame["expected_coupon_cost"] = config.send_cost + discount
    frame["expected_incremental_revenue"] = frame.uplift_score * frame.predicted_gross_4w_if_purchase
    frame["expected_net_value"] = frame.expected_incremental_revenue - frame.expected_coupon_cost
    if not money:
        frame[["expected_coupon_cost", "expected_incremental_revenue", "expected_net_value"]] = np.nan
    reasons = []
    for _, row in frame.iterrows():
        missing = [name for name in ("churn_prob", "p0", "p1", "clv", "segment") if pd.isna(row[name])]
        if pd.isna(row.predicted_gross_4w_if_purchase):
            missing.append("4-week conditional gross prediction")
        if pd.isna(row.expected_discount_if_sent):
            missing.append("expected discount cost")
        if not money:
            missing.append("observed monetary units")
        reasons.append("; ".join(missing) or None)
    unavailable = pd.Series(reasons, index=frame.index).notna()
    eligible = (~unavailable & frame.segment.isin(config.eligible_segments) & frame.expected_net_value.gt(0)
                & frame.priority_score.notna() & response.ne("Sleeping Dogs").fillna(False))
    profile_audit = profile_metadata = None
    if profile is not None:
        from .profile_adapter import profile_gate
        gate, profile_audit, profile_metadata = profile_gate(frame, profile)
        eligible &= gate
    frame.attrs["send_cost"] = config.send_cost
    frame["send"] = _choose(frame, eligible, config.budget)
    summary = []
    for name in SEGMENTS:
        subset = frame.loc[frame.segment.eq(name).fillna(False)]
        summary.append({"segment": name, "customers": len(subset), "share": len(subset) / len(frame),
                        "mean_clv": subset.clv.mean(), "mean_churn_prob": subset.churn_prob.mean(),
                        "mean_uplift": subset.uplift_score.mean(), "sends": int(subset.send.sum())})
    what_if = []
    for scale in (.5, 1., 2.):
        selected = _choose(frame, eligible, config.budget * scale)
        what_if.append({"budget_ratio": scale, **_budget(frame, selected, config.budget * scale, int(unavailable.sum()))})
    identity = frame[KEYS + [name for name in PROVENANCE if name in frame]].copy()
    availability = frame[KEYS].assign(status=np.where(unavailable, "unavailable", "available"), reason=reasons)
    scatter = frame[["customer_id", "churn_prob", "uplift_score", "clv", "segment"]].copy()
    quadrants = frame[["customer_id", "p0", "p1", "churn_prob", "uplift_score"]].assign(response_group=response)
    metadata = {"configuration": asdict(config), "monetary_available": bool(money),
                "prediction_provenance": frame.attrs.get("prediction_provenance", {}),
                "planning_method": "greedy positive 4-week expected net value, ordered by max(uplift,0)*52-week CLV",
                "expected_revenue_assumption": "purchase probability change times no-coupon conditional 4-week gross; same conditional gross under both actions",
                "discount_cost_assumption": "explicit unconditional expected discount per send, separate from the send fee",
                "new_customer_definition": "observed tenure below configured threshold; not inferred signup or latent lifecycle",
                "segment_definition": "six explicit operational rules; main gives examples rather than an exhaustive partition; not simulator persona labels",
                "response_group_definition": "positive purchase uplift: Persuadables; negative: Sleeping Dogs; near-zero and high/low p0: Sure Things/Lost Causes; not identified individual response types",
                "churn_gate": "configured churn threshold restricts persuadable business-target segments only; it does not redefine purchase response groups",
                "unavailable_customers": int(unavailable.sum()), "unavailable_segment_customers": int(frame.segment.isna().sum()),
                "financial_scope": "modeled incremental gross revenue minus expected sending and discount costs; not observed causal profit",
                "time_contract": weekly_metadata()}
    if profile_metadata is not None:
        metadata["human_strategy_profile"] = profile_metadata
    graph_data = {"churn_risk": frame[["customer_id", "churn_prob"]].copy(),
                  "uplift_quadrants": quadrants, "segments6": scatter}
    if profile_audit is not None:
        graph_data["human_profile"] = profile_audit
    return DecisionOutputs(frame[CUSTOMER_COLUMNS].copy(), identity, availability, pd.DataFrame(summary),
                           _budget(frame, frame.send, config.budget, int(unavailable.sum())), pd.DataFrame(what_if),
                           graph_data, metadata)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (Path, date, datetime)):
        return str(value) if isinstance(value, Path) else value.isoformat()
    if value is pd.NA or isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(result, output, *, qini=None, strategy=None):
    """Write the ten-column handoff, audit keys and six inspectable plot artifacts."""
    from .plots import render_plots
    from .evaluation import QINI_COLUMNS, STRATEGY_COLUMNS
    output = Path(output)
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    result.customers.to_csv(output / "customers.csv", index=False)
    result.identity.to_parquet(output / "customer_identity.parquet", index=False)
    result.availability.to_csv(output / "availability.csv", index=False)
    result.segment_summary.to_csv(output / "segment_summary.csv", index=False)
    result.what_if.to_csv(output / "budget_what_if.csv", index=False)
    for name, value in (("budget_summary", result.budget_summary), ("metadata", result.metadata)):
        (output / f"{name}.json").write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    data = output / "graph_data"
    data.mkdir()
    for name, frame in result.graph_data.items():
        frame.to_csv(data / f"{name}.csv", index=False)
    result.what_if.to_csv(data / "budget_roi.csv", index=False)
    for name, payload in (("qini", qini), ("strategy", strategy)):
        payload = payload or {"status": "unavailable", "reason": "No held-out randomized factual evaluation supplied",
                              "frame": pd.DataFrame(columns=QINI_COLUMNS if name=="qini" else STRATEGY_COLUMNS)}
        payload["frame"].to_csv(data / f"{name}.csv", index=False)
        (data / f"{name}_status.json").write_text(json.dumps(_jsonable({k:v for k,v in payload.items() if k!="frame"}), indent=2)+"\n")
    render_plots(result, output / "graphs", qini=qini, strategy=strategy)
    files = {str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()}
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "customer_columns": CUSTOMER_COLUMNS,
        "customers": len(result.customers), "file_sha256": files}, ensure_ascii=False, indent=2)+"\n")
    return output
