"""Report existing policy allocations on the original weekly prediction roster."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Context, Decimal, localcontext

import pandas as pd
from simulate.v4.storage import FEATURE_KEYS

from .budget import Budget, CostMetadata, _decimal
from .coupon_cost_estimation import CostEstimate
from .pipeline import CUSTOMER_COLUMNS, SEGMENTS, DecisionOutputs, _response_group, _segments
from .policy_search import PolicyCandidate, _decision_fingerprint, _project_policy_run
from .retention_strategy import (COST_INPUT_COLUMNS, ELIGIBILITY_COLUMNS, _aligned,
                                 _build_retention_strategy, _price_cost_inputs)
from .segmentation import segment_customers
from .validation import CONTEXT_COLUMNS, JOIN_KEYS
from .weekly_validation import finite_number, require
from .what_if import SCENARIOS, run_what_if


@dataclass(frozen=True)
class ReportingConfig:
    high_value_threshold: float
    churn_threshold: float
    uplift_tolerance: float
    new_history_days: int
    baseline_purchase_threshold: float

    def __post_init__(self):
        finite_number(self.high_value_threshold, "high_value_threshold")
        for field in ("churn_threshold", "baseline_purchase_threshold"):
            require(0 < finite_number(getattr(self, field), field) < 1, field)
        require(finite_number(self.uplift_tolerance, "uplift_tolerance") < 1, "uplift_tolerance")
        require(type(self.new_history_days) is int and self.new_history_days >= 0, "new_history_days")


def _scope(table, columns, roster, name):
    """Allow full-roster declarations, checking even rows outside the model cohort."""
    require(isinstance(table, pd.DataFrame) and "customer_id" in table, f"Missing {name}")
    supplied = roster.loc[roster.customer_id.isin(table.customer_id)]
    return _aligned(table, columns, supplied, name)


def _check_handoff(handoff):
    roster = handoff.roster.copy(deep=True)
    scored = roster.loc[roster.status.eq("scored")]
    excluded = roster.loc[~roster.status.eq("scored")]
    _aligned(handoff.excluded[list(CONTEXT_COLUMNS)], CONTEXT_COLUMNS, excluded, "excluded roster")
    reasons = handoff.excluded.set_index("customer_id").exclusion_reason
    require(excluded.customer_id.map(reasons).eq(excluded.status).all(), "model exclusion reasons differ")
    require((handoff.integrated is None) == scored.empty, "scored roster mismatch")
    if not scored.empty:
        data = handoff.integrated.data
        _aligned(data[list(CONTEXT_COLUMNS)], CONTEXT_COLUMNS, scored, "scored roster")
        for name in ("churn_probability", "uplift_score", "predicted_clv_12m", "observed_tenure_days"):
            actual = scored.customer_id.map(data.set_index("customer_id")[name])
            require((scored[name].eq(actual) | (scored[name].isna() & actual.isna())).all(),
                    f"scored roster differs from integrated {name}")
    roster["model_exclusion_stage"] = roster.customer_id.map(handoff.excluded.set_index("customer_id").exclusion_stage)
    roster["model_exclusion_reason"] = roster.customer_id.map(reasons)
    return roster, scored


def _finance(frame, summary):
    chosen = frame.loc[frame.send.eq(1).fillna(False)]
    valid = summary["has_valid_solution"]
    cost = summary["total_expected_cost"]
    revenue = net = roi = None
    if not valid:
        reason = summary["status"]
    elif chosen.expected_incremental_revenue.isna().any():
        reason = "four-week economics require aligned same-cutoff requests, gross and monetary units"
    else:
        revenue = sum(chosen.expected_incremental_revenue, Decimal(0))
        net = revenue - cost
        roi = net / cost if cost else None
        reason = None if cost else "zero expected cost"
    send_fees = sum(chosen.send_cost, Decimal(0)) if valid else None
    return dict(expected_send_cost=send_fees, expected_discount_cost=cost-send_fees if valid else None,
                total_budget=summary["budget"], send_count=summary["selected_customer_count"],
                expected_cost=cost, expected_incremental_revenue=revenue, expected_net_value=net,
                expected_roi=roi, roi_reason=reason)


def validate_planning_output(result):
    """Reconcile the mutable export tables before writing the first file."""
    audit = result.decision_audit
    summary = result.budget_summary
    require(result.customers.equals(audit[CUSTOMER_COLUMNS]) and
            result.identity.equals(audit[list(FEATURE_KEYS)]), "customer export differs from decision audit")
    scored = audit.status.eq("scored")
    require(audit.send.equals(audit.selected.astype("Int64").mask(~scored, 0)), "send differs from selected")
    if summary["has_valid_solution"]:
        chosen = audit.loc[audit.send.eq(1).fillna(False)]
        require(_decision_fingerprint(audit.loc[scored]) == summary["decision_fingerprint"]
                and len(chosen) == summary["selected_customer_count"]
                and chosen.expected_cost.notna().all()
                and sum(chosen.expected_cost, Decimal(0)) == summary["total_expected_cost"],
                "decision audit differs from the budget allocation")
    for field, value in _finance(audit, summary).items():
        require(value == summary[field], f"customer economics differ from budget summary: {field}")


def _cost_declarations(first, cost_estimate, cost_inputs, cost_metadata):
    evidence = None
    if cost_estimate is not None:
        require(isinstance(cost_estimate, CostEstimate) and cost_inputs is None and cost_metadata is None,
                "supply either an estimate or declared costs")
        evidence = cost_estimate.audit
        require(cost_estimate.status in ("complete", "withheld") and evidence.get("status") == cost_estimate.status
                and evidence.get("population_definition") == "all coupon requests including failures",
                "cost estimate request denominator or status mismatch")
        target = evidence["target"]
        require(all(target[k] == first[k] for k in ("dataset_id", "raw_database_sha256", "currency"))
                and pd.Timestamp(target["as_of_at"]) == first.as_of_at
                and Decimal(str(target["amount_unit"])) == Decimal(str(first.amount_unit)),
                "cost estimate target mismatch")
        cost_metadata = cost_estimate.cost_metadata
        if cost_estimate.status == "complete":
            cost_inputs = cost_estimate.cost_inputs
    return cost_inputs, cost_metadata, evidence


def build_strategy_decisions(handoff, profile, reporting, budget, *, algorithm, eligibility,
                             coupons, redemptions, history_context, history_policy,
                             cost_estimate=None, cost_inputs=None, cost_metadata=None,
                             solver_config=None, hybrid_config=None):
    """Plan one declared profile; costs are per request over the planning window."""
    return _build_strategy_decisions(
        handoff, profile, reporting, budget, algorithm=algorithm, eligibility=eligibility,
        coupons=coupons, redemptions=redemptions, history_context=history_context,
        history_policy=history_policy, cost_estimate=cost_estimate, cost_inputs=cost_inputs,
        cost_metadata=cost_metadata, solver_config=solver_config, hybrid_config=hybrid_config)


def _build_strategy_decisions(handoff, profile, reporting, budget, *, algorithm, eligibility,
                             coupons, redemptions, history_context, history_policy,
                             cost_estimate=None, cost_inputs=None, cost_metadata=None,
                             solver_config=None, hybrid_config=None, segmentation=None):
    """Plan one declared profile; costs are per request over the planning window."""
    with localcontext(Context(prec=200)):
        require(isinstance(profile, PolicyCandidate) and isinstance(reporting, ReportingConfig)
                and isinstance(budget, Budget), "explicit profile, reporting configuration and budget required")
        require(algorithm in ("greedy", "milp", "hybrid"), "choose one algorithm")
        from .weekly_prediction_adapter import WeeklyStrategyInputs
        require(isinstance(handoff, WeeklyStrategyInputs), "validated weekly producer handoff required")
        handoff = handoff.snapshot()
        roster, scored = _check_handoff(handoff)
        first = roster.iloc[0]
        amount = _decimal(budget.amount, "budget")
        require(roster.currency.eq(budget.currency).all() and
                all(_decimal(unit, "unit", positive=True) == _decimal(budget.amount_unit, "budget unit", positive=True)
                    for unit in roster.amount_unit), "budget units differ from roster")
        cost_inputs, cost_metadata, evidence = _cost_declarations(
            first, cost_estimate, cost_inputs, cost_metadata)
        runs = {}
        prices = None
        reason = None
        if scored.empty:
            reason = "no_model_predictions"
        elif cost_inputs is None:
            reason = "cost_evidence_unavailable"
        else:
            require(isinstance(cost_metadata, CostMetadata), "declared costs require CostMetadata")
            eligible = _scope(eligibility, ELIGIBILITY_COLUMNS, roster, "eligibility")
            eligible = eligible.loc[eligible.customer_id.isin(scored.customer_id)]
            declared = _scope(cost_inputs, COST_INPUT_COLUMNS, roster, "cost_inputs")
            costs = declared.loc[declared.customer_id.isin(scored.customer_id)]
            strategy = _build_retention_strategy(
                segment_customers(handoff.integrated, profile.segmentation) if segmentation is None else segmentation,
                eligible, coupons, redemptions,
                costs, cost_metadata, dataset_id=first.dataset_id, as_of_at=first.as_of_at,
                currency=budget.currency, amount_unit=budget.amount_unit, policies=profile.policies,
                history_context=history_context, history_policy=history_policy, common_cost_inputs=True)
            other = declared.loc[~declared.customer_id.isin(strategy.cost_estimates.customer_id)]
            prices = pd.concat([strategy.cost_estimates, _price_cost_inputs(other, cost_metadata, first.as_of_at)],
                               ignore_index=True).set_index("customer_id")
            scenarios = run_what_if(strategy.budget_inputs, strategy.costs, budget, cost_metadata,
                                    dataset_id=first.dataset_id, as_of_at=first.as_of_at, config=solver_config,
                                    algorithm=algorithm, hybrid_config=hybrid_config,
                                    scenarios=None if amount else ["budget_100"])
            for key, allocation in scenarios.results.items():
                runs[key] = _project_policy_run(profile, strategy, allocation, algorithm, first.dataset_id,
                                                first.as_of_at, allocation.metadata["total_execution_time"])
        return _outputs(roster, reporting, budget, profile, algorithm, runs, reason, handoff,
                        cost_metadata, evidence, prices)


def _outputs(roster, reporting, budget, profile, algorithm, runs, hold, handoff, cost_metadata, evidence, prices):
    model_fields = [*FEATURE_KEYS, "observed_tenure_days", "status", "churn_probability", "p0", "p1",
                    "uplift_score", "clv_training_role", "predicted_clv_12m", "predicted_gross_4w_if_purchase",
                    "model_exclusion_stage", "model_exclusion_reason"]
    base = roster[model_fields].rename(columns={"churn_probability": "churn_prob", "predicted_clv_12m": "clv"})
    if prices is not None:
        for field in ("send_cost", "redemption_probability", "expected_discount_if_redeemed",
                      "expected_cost", "source_id", "source_version", "source_as_of_at"):
            label = "declared_expected_cost" if field == "expected_cost" else field
            base[label] = base.customer_id.map(prices[field])
    base["response_group"] = _response_group(base, reporting)
    base["business_segment"] = base["segment"] = _segments(base, base.response_group, reporting)
    base["expected_incremental_revenue"] = [
        Decimal(str(effect)) * Decimal(str(gross)) if pd.notna(gross) else None
        for effect, gross in zip(base.uplift_score, base.predicted_gross_4w_if_purchase)]
    if (handoff.evidence.get("feature_store") or {}).get("source_monetary_values_observed") is False:
        base["expected_incremental_revenue"] = None
    records, details = [], {}
    fingerprint = profile.configuration_fingerprint()
    for key, ratio in SCENARIOS:
        run = runs.get(key, runs.get("budget_100"))
        if run is None:
            frame = base.copy()
            for column in ("segment_id", "priority_score", "budget_candidate", "selected", "expected_cost",
                           "planned_sent_at", "eligibility_reason", "policy_exclusion_reason"):
                frame[column] = pd.NA
            summary = dict(status=hold, budget=Decimal(str(budget.amount))*ratio, has_valid_solution=False,
                           accepted_for_planning=False, total_expected_cost=None, selected_customer_count=None,
                           solver_status=None, decision_fingerprint=None)
        else:
            # The authoritative rows already contain the original segment, score, costs and fingerprint.
            decisions = run.decisions
            summary = deepcopy(run.summary)
            scored = base.loc[base.status.eq("scored")]
            _aligned(decisions[list(CONTEXT_COLUMNS)], CONTEXT_COLUMNS, scored, "policy roster")
            # Preserve the reported model values and identity; source plan fields from PolicyRun.
            fields = [c for c in decisions if c not in set(CONTEXT_COLUMNS) - set(JOIN_KEYS)
                      and c not in ("churn_probability", "uplift_score", "predicted_clv_12m")]
            overlaps = base.columns.intersection(fields).difference(JOIN_KEYS)
            frame = base.drop(columns=overlaps).merge(decisions[fields], on=list(JOIN_KEYS),
                                                      how="left", validate="one_to_one")
            if summary["has_valid_solution"]:
                chosen = frame.loc[frame.selected.eq(1).fillna(False)]
                require(len(chosen) == summary["selected_customer_count"]
                        and chosen.expected_cost.notna().all()
                        and sum(chosen.expected_cost, Decimal(0)) == summary["total_expected_cost"]
                        and _decision_fingerprint(frame.loc[frame.status.eq("scored")]) == summary["decision_fingerprint"],
                        "exported decisions disagree with the allocation")
        missing = frame.status.ne("scored")
        frame["send"] = frame.selected.astype("Int64").mask(missing, 0)
        delayed = frame.planned_sent_at.notna() & frame.planned_sent_at.ne(frame.as_of_at)
        frame.loc[delayed, "expected_incremental_revenue"] = None
        frame["expected_net_value"] = [
            revenue-cost if pd.notna(revenue) and pd.notna(cost) else None
            for revenue, cost in zip(frame.expected_incremental_revenue, frame.expected_cost)]
        economics = _finance(frame, summary)
        details[key] = summary
        records.append(dict(scenario_id=key, budget_ratio=ratio, **economics, status=summary["status"],
                            has_valid_solution=summary["has_valid_solution"],
                            accepted_for_planning=summary["accepted_for_planning"],
                            decision_fingerprint=summary["decision_fingerprint"]))
        if key == "budget_100":
            audit = frame
            budget_summary = dict(deepcopy(summary), **economics)
    audit["plan_state"] = hold or ("invalid_solver_result" if not budget_summary["has_valid_solution"]
                                   else "valid_unapproved" if not budget_summary["accepted_for_planning"]
                                   else "valid_approved_for_planning")
    audit["row_state"] = "not_selected"
    audit.loc[audit.send.eq(1).fillna(False), "row_state"] = "selected"
    audit.loc[audit.policy_exclusion_reason.notna(), "row_state"] = "policy_excluded"
    audit.loc[audit.eligibility_reason.notna(), "row_state"] = "eligibility_excluded"
    audit.loc[audit.send.isna(), "row_state"] = hold or "invalid_solver_result"
    audit.loc[audit.status.ne("scored"), "row_state"] = "model_unavailable"
    audit["policy_id"], audit["policy_version"] = profile.policy_id, profile.version
    audit["configuration_fingerprint"], audit["algorithm"] = fingerprint, algorithm
    for field in ("decision_fingerprint", "has_valid_solution", "accepted_for_planning", "solver_status", "budget"):
        audit[field] = budget_summary[field]
    audit["optimality"] = budget_summary.get("optimality")
    audit["quality_status"] = budget_summary.get("allocation_metadata", {}).get("hybrid", {}).get("quality", {}).get("status")
    audit["dispatch_authorized"] = False
    budget_summary.update(currency=budget.currency, amount_unit=str(budget.amount_unit),
                          dispatch_authorized=False, policy_id=profile.policy_id, policy_version=profile.version,
                          configuration_fingerprint=fingerprint)
    groups = []
    for name in SEGMENTS:
        part = audit.loc[audit.segment.eq(name).fillna(False)]
        groups.append(dict(segment=name, customers=len(part), share=len(part)/len(audit),
                           mean_churn_prob=part.churn_prob.mean(), mean_clv=part.clv.mean(),
                           mean_uplift=part.uplift_score.mean(),
                           sends=int(part.send.sum()) if budget_summary["has_valid_solution"] else None))
    meta = dict(configuration=asdict(reporting), engine="final_strategy", policy_id=profile.policy_id,
                policy_version=profile.version, configuration_fingerprint=fingerprint, scenarios=details,
                roster_count=len(audit), analyzed_count=int(audit.status.eq("scored").sum()),
                producer_metadata=deepcopy(handoff.producer_metadata), evidence=deepcopy(handoff.evidence),
                prediction_metadata={k:asdict(v) for k,v in handoff.prediction_metadata.items()},
                cost_source="statistical_estimate" if evidence is not None else "declared_scenario" if cost_metadata else "unavailable",
                cost_metadata=asdict(cost_metadata) if cost_metadata else None, cost_estimation=evidence,
                financial_scope="planned incremental gross revenue less expected coupon request costs",
                cost_population="requests including failed sends over four weeks",
                objective_semantics="uplift times 52-week CLV", policy_effect_verified=False,
                expected_revenue_assumption="same conditional four-week gross under both actions",
                dispatch_authorized=False)
    graph = {key:audit[cols].copy() for key, cols in {
        "churn_risk":["customer_id","churn_prob"],
        "uplift_quadrants":["customer_id","p0","p1","churn_prob","uplift_score","response_group"],
        "segments6":["customer_id","churn_prob","clv","uplift_score","segment"]}.items()}
    availability = audit[[*CONTEXT_COLUMNS,"status","row_state","plan_state",
                          "model_exclusion_stage","model_exclusion_reason","eligibility_reason","policy_exclusion_reason"]].copy()
    return DecisionOutputs(audit[CUSTOMER_COLUMNS].copy(), audit[list(FEATURE_KEYS)].copy(), availability,
                           pd.DataFrame(groups), budget_summary, pd.DataFrame(records), graph, meta, audit)
