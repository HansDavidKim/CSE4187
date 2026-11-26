"""Synthetic weekly producer handoffs; exercise the actual retention/Budget and output APIs."""
from dataclasses import asdict, replace
from decimal import Context, Decimal, localcontext
from hashlib import sha256
import json
import sys

import pandas as pd
import pytest

from final_strategy import budget as engine
from final_strategy.__main__ import main
from final_strategy.budget import Budget, CostMetadata
from final_strategy.pipeline import CUSTOMER_COLUMNS
from final_strategy.policy_search import run_policy_search
from final_strategy.retention_strategy import COUPON_COLUMNS, REDEMPTION_COLUMNS, HistoryContext, HistoryPolicy
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.tests.test_coupon_cost_estimation import data as historical_data, estimate
from final_strategy.tests.test_weekly_prediction_adapter import cases, features, published
from final_strategy.validation import CONTEXT_COLUMNS
from final_strategy.weekly_planning import ReportingConfig, build_strategy_decisions
from final_strategy.weekly_prediction_adapter import adapt_weekly_predictions


DECLARATIONS = dict(version="fixture", threshold_source="synthetic", common=".2", risk_high=".1", risk_low=".3",
                    high_value=".2", standard_value=".4", h4_r=dict(a=0, b=".1", c=".3", d=".4"))
REPORT = ReportingConfig(100, .5, .01, 7, .5)


@pytest.fixture
def weekly(cases):
    features, batches = cases
    # Two scored customers, distinct per-request costs, one explicit model exclusion.
    retention = batches["retention"]
    second = retention.frame.customer_id.eq("002")
    retention.frame.loc[second, ["status", "churn_prob"]] = ["scored", .6]
    retention.producer_metadata.update(status_counts={"scored":2, "no_recent_session":1}, excluded_rows=1)
    uplift = batches["uplift"].frame
    uplift.loc[uplift.customer_id.eq("002"), ["p0","p1","uplift_score"]] = [.4,.6,.2]
    clv = batches["clv"]
    clv.gross_4w = clv.frame.drop(columns="predicted_clv_12m").assign(predicted_gross_4w_if_purchase=10)
    clv.gross_4w["prediction_end_at"] = clv.gross_4w.as_of_at + pd.Timedelta(weeks=4)
    clv.gross_4w_metadata = dict(clv.metadata, target_definition_id="gross_amount_4w_if_purchase_v1",
                                 model_version="gross", horizon_value=4)
    clv.gross_4w_status = "trained"
    clv.producer_metadata["gross_4w"] = dict(status="trained", model_version="gross")
    return adapt_weekly_predictions(features, **batches)


def options(weekly, algorithm="greedy"):
    roster = weekly.roster
    keys = roster[list(CONTEXT_COLUMNS)]
    first = keys.iloc[0]
    costs = keys.assign(send_cost=["2","3","1"], redemption_probability=".25", expected_discount_if_redeemed="10",
                        source_id="declared", source_version="v1", source_as_of_at=keys.as_of_at)
    return dict(profile=build_strategy_profiles(**DECLARATIONS).policies[1], reporting=REPORT,
                budget=Budget("6.000", first.currency, first.amount_unit), algorithm=algorithm,
                eligibility=keys.assign(business_eligible=True, business_reason=None, history_complete=True),
                coupons=pd.DataFrame(columns=COUPON_COLUMNS), redemptions=pd.DataFrame(columns=REDEMPTION_COLUMNS),
                cost_inputs=costs, cost_metadata=CostMetadata("declared-request-28d","v1",3,first.as_of_at),
                history_context=HistoryContext(first.dataset_id,first.raw_database_sha256,first.as_of_at,
                                              "history","v1",first.as_of_at),
                history_policy=HistoryPolicy(False,False,"0d"))


@pytest.mark.parametrize("algorithm", ["greedy","milp","hybrid"])
def test_canonical_decisions_and_economics(weekly, algorithm, monkeypatch):
    args = options(weekly, algorithm)
    scored = weekly.integrated.data
    canonical = run_policy_search(
        weekly.integrated, [args["profile"]], args["eligibility"].query("customer_id != '003'"),
        args["coupons"], args["redemptions"], args["cost_inputs"].query("customer_id != '003'"),
        args["cost_metadata"], args["budget"], dataset_id="dummy", as_of_at=scored.as_of_at.iloc[0],
        history_context=args["history_context"], history_policy=args["history_policy"], algorithm=algorithm).runs["H0"]
    calls = []
    fn = engine._execute_prepared
    def count(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)
    monkeypatch.setattr(engine, "_execute_prepared", count)
    with localcontext(Context(prec=1)):
        result = build_strategy_decisions(weekly, **args)
    assert len(calls) == 3
    audit = result.decision_audit.set_index("customer_id")
    assert audit.send.to_dict() == {"001":1,"002":0,"003":0}
    assert audit.row_state.to_list() == ["selected","not_selected","model_unavailable"]
    assert result.budget_summary["decision_fingerprint"] == canonical.summary["decision_fingerprint"]
    assert result.budget_summary["configuration_fingerprint"] == build_strategy_profiles(**DECLARATIONS).policies[0].configuration_fingerprint()
    assert audit.loc["001","expected_cost"] == Decimal("4.50") and audit.loc["002","expected_cost"] == Decimal("5.50")
    summary = result.budget_summary
    assert summary["remaining_budget"] == Decimal("1.5")
    assert summary["expected_incremental_revenue"] == 3 and summary["expected_net_value"] == Decimal("-1.50")
    assert summary["expected_roi"] < 0 and summary["accepted_for_planning"] == canonical.summary["accepted_for_planning"]
    assert summary["status"] == canonical.summary["status"]
    if algorithm == "hybrid":
        assert not summary["accepted_for_planning"] and "hybrid" in summary["allocation_metadata"]
    assert not audit.dispatch_authorized.any()
    other = build_strategy_decisions(weekly, **(args | {"reporting":replace(REPORT,new_history_days=0, high_value_threshold=200)}))
    assert not audit.business_segment.equals(other.decision_audit.set_index("customer_id").business_segment)
    assert other.budget_summary["decision_fingerprint"] == summary["decision_fingerprint"]


def test_unavailable_money_invalid_solver_and_hold(weekly, cases, monkeypatch):
    args = options(weekly)
    features, batches = cases
    clv = batches["clv"]
    clv.gross_4w = clv.gross_4w_metadata = None
    clv.gross_4w_status = "not_trained"
    clv.producer_metadata["gross_4w"] = dict(status="not_trained", model_version=None)
    weekly = adapt_weekly_predictions(features, **batches)
    weekly.roster["predicted_gross_4w_if_purchase"] = 10
    weekly.roster.loc[0, ["p0", "p1"]] = [.99, .01]
    result = build_strategy_decisions(weekly, **args)
    assert result.budget_summary["has_valid_solution"] and result.budget_summary["expected_roi"] is None
    assert result.budget_summary["send_count"] == 1
    held = build_strategy_decisions(weekly, **(args | {"cost_inputs":None,"cost_metadata":None}))
    assert held.customers.send.isna().tolist() == [True, True, False]
    assert held.budget_summary["status"] == "cost_evidence_unavailable"
    def failed(*args):
        raise RuntimeError("synthetic solver failure")
    monkeypatch.setattr(engine, "_solve_milp", failed)
    result = build_strategy_decisions(weekly, **(args | {"algorithm":"milp"}))
    assert result.customers.send.isna().tolist() == [True, True, False]
    assert result.budget_summary["send_count"] is None
    assert result.availability.row_state.tolist() == ["invalid_solver_result","invalid_solver_result","model_unavailable"]


@pytest.mark.parametrize("algorithm", ["greedy", "milp", "hybrid"])
def test_editing_handoff_views_cannot_override_solver_or_model(weekly, cases, algorithm):
    args = options(weekly, algorithm)
    view = weekly.roster
    view["selected"] = 0
    view["budget_candidate"] = False
    view["expected_cost"] = 999
    view["planned_sent_at"] = pd.NaT
    view.loc[0, ["p0", "p1", "predicted_gross_4w_if_purchase", "uplift_score"]] = [.99, .01, 100, -.98]
    weekly.integrated.data.loc[0, "uplift_score"] = -.98
    weekly.excluded["exclusion_reason"] = "changed"
    weekly.producer_metadata["clv"]["inputs"].clear()
    weekly.evidence.clear()
    _, batches = cases
    batches["clv"].gross_4w["predicted_gross_4w_if_purchase"] = 100
    result = build_strategy_decisions(weekly, **args)
    audit = result.decision_audit
    assert result.customers.send.tolist() == audit.send.tolist() == [1, 0, 0]
    assert audit.loc[0, ["p0", "p1", "uplift_score"]].tolist() == [.2, .5, .3]
    assert audit.loc[0, "planned_sent_at"] == audit.loc[0, "as_of_at"]
    assert audit.loc[2, "model_exclusion_reason"] == "no_recent_session"
    assert sum(audit.loc[audit.send.eq(1), "expected_cost"]) == result.budget_summary["expected_cost"] == Decimal("4.50")
    assert result.budget_summary["expected_net_value"] == Decimal("-1.5")
    assert float(result.budget_summary["expected_roi"]) == pytest.approx(-1/3)
    args["cost_inputs"].loc[0, "send_cost"] = "20"
    new = build_strategy_decisions(weekly, **args)
    assert new.customers.send.tolist() == [0, 1, 0] and result.customers.send.tolist() == [1, 0, 0]
    assert new.budget_summary["expected_cost"] == Decimal("5.50")


def test_changed_customer_export_is_rejected_before_writing(weekly, tmp_path):
    from final_strategy.pipeline import write_outputs
    result = build_strategy_decisions(weekly, **options(weekly))
    result.customers["send"] = 0
    destination = tmp_path / "inconsistent"
    with pytest.raises(ValueError, match="differs from decision audit"):
        write_outputs(result, destination)
    assert not destination.exists()


def test_zero_cost_and_all_excluded(weekly, cases):
    args = options(weekly)
    args["cost_inputs"][["send_cost", "redemption_probability"]] = "0"
    result = build_strategy_decisions(weekly, **args)
    assert result.budget_summary["expected_cost"] == 0 and result.budget_summary["expected_roi"] is None
    features, batches = cases
    r = batches["retention"]
    r.frame["status"], r.frame["churn_prob"] = "no_recent_session", float("nan")
    r.producer_metadata.update(status_counts={"no_recent_session":3}, excluded_rows=3)
    empty = build_strategy_decisions(adapt_weekly_predictions(features, **batches), **args)
    assert empty.metadata["analyzed_count"] == 0 and len(empty.customers) == 3
    assert empty.budget_summary["status"] == "no_model_predictions" and empty.budget_summary["send_count"] is None


@pytest.mark.parametrize("artifact", ["parquet", "official_trained", "official_without_gross"])
def test_cli_exports_engine_results_and_hashes(weekly, cases, published, tmp_path, monkeypatch, artifact):
    _, batches = cases
    plan = options(weekly)
    declarations = {}
    for name, batch in batches.items():
        role = "churn" if name == "retention" else name
        batch.frame.to_parquet(tmp_path/f"{role}.parquet")
        declarations[role] = dict(frame=f"{role}.parquet", metadata=batch.metadata, producer_metadata=batch.producer_metadata)
    clv = batches["clv"]
    if artifact == "parquet":
        clv.audit.to_parquet(tmp_path/"audit.parquet")
        declarations["clv"]["audit"] = "audit.parquet"
        clv.gross_4w.to_parquet(tmp_path/"gross.parquet")
        declarations["gross_4w"] = dict(frame="gross.parquet",metadata=clv.gross_4w_metadata)
    else:
        from clv.inference import StrategyPredictions
        if artifact == "official_without_gross":
            clv.gross_4w = clv.gross_4w_metadata = None
            clv.gross_4w_status = "not_trained"
            clv.producer_metadata["gross_4w"] = dict(status="not_trained", model_version=None)
        StrategyPredictions(**vars(clv)).save(tmp_path/"clv.parquet")
        declarations["clv"]["audit"] = "clv_audit.csv"
        if clv.gross_4w is not None:
            declarations["gross_4w"] = dict(frame="clv_gross_4w.parquet", metadata=clv.gross_4w_metadata)
    config = dict(algorithm="greedy",retention_policy="observed_experiment_mix")
    for name in ("budget","history_context","history_policy","cost_metadata"):
        config[name] = asdict(plan[name])
    for name in ("eligibility","cost_inputs","coupons","redemptions"):
        config[name] = f"{name}.parquet"
        plan[name].to_parquet(tmp_path/config[name])
    for name, payload in (("batches",declarations),("report",asdict(REPORT)),("profiles",DECLARATIONS),("plan",config)):
        (tmp_path/f"{name}.json").write_text(json.dumps(payload,default=str), encoding="utf-8")
    out = tmp_path/"output"
    monkeypatch.setattr(sys, "argv", ["final_strategy","--features",str(published),"--predictions",str(tmp_path/"batches.json"),
                        "--config",str(tmp_path/"report.json"),"--profile","H0","--profile-declarations",str(tmp_path/"profiles.json"),
                        "--engine-plan",str(tmp_path/"plan.json"),"--output",str(out)])
    if artifact == "official_trained":
        from final_strategy.tests.test_contract import assigned_roster, strategy_factual
        outcomes, evaluation = strategy_factual()
        outcomes.to_parquet(tmp_path/"outcomes.parquet", index=False)
        assigned_roster(outcomes).to_parquet(tmp_path/"assigned.parquet", index=False)
        (tmp_path/"eval.json").write_text(json.dumps(evaluation), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", sys.argv + ["--strategy-outcomes", str(tmp_path/"outcomes.parquet"),
                            "--strategy-assignments", str(tmp_path/"assigned.parquet"),
                            "--evaluation-metadata", str(tmp_path/"eval.json")])
    main()
    assert pd.read_csv(out/"customers.csv").columns.tolist() == CUSTOMER_COLUMNS
    audit = pd.read_csv(out/"decision_audit.csv", dtype={"customer_id": str})
    assert audit.customer_id.tolist() == weekly.roster.customer_id.tolist()
    assert audit.raw_database_sha256.tolist() == weekly.roster.raw_database_sha256.tolist()
    assert audit.send.tolist() == pd.read_csv(out/"customers.csv").send.tolist() == [1, 0, 0]
    if artifact == "official_without_gross":
        assert json.loads((out/"budget_summary.json").read_text())["expected_roi"] is None
    assert json.loads((out/"budget_summary.json").read_text())["expected_cost"] == "4.50"
    manifest = json.loads((out/"manifest.json").read_text())
    assert "graph_data/qini.csv" in manifest["file_sha256"]
    for key, digest in manifest["file_sha256"].items():
        assert "\\" not in key and sha256((out/key).read_bytes()).hexdigest() == digest
    assert len(list((out/"graphs").glob("*.png"))) == 6
    assert json.loads((out/"graph_data/strategy_status.json").read_text())["status"] == ("available" if artifact == "official_trained" else "unavailable")
    if artifact == "official_trained":
        outcomes["paid_minor"] = outcomes.gross_revenue_minor - outcomes.discount_minor
        outcomes.loc[0, "paid_minor"] += 1
        outcomes.to_parquet(tmp_path/"outcomes.parquet", index=False)
        with pytest.raises(ValueError, match="contradicts"):
            main()
        outcomes.iloc[1:].to_parquet(tmp_path/"outcomes.parquet", index=False)
        with pytest.raises(ValueError, match="original assignment"):
            main()
    if artifact.startswith("official"):
        with (tmp_path/"clv_audit.csv").open("ab") as f:
            f.write(b"\n")
        with pytest.raises(ValueError, match="checksum"):
            main()


def test_historical_estimate_withheld_and_future_source(weekly, cases, historical_data):
    features, _ = cases
    inputs = historical_data | {"target_features":features}
    args = options(weekly)
    del args["cost_metadata"], args["cost_inputs"]
    result = build_strategy_decisions(weekly, cost_estimate=estimate(inputs), **args)
    assert result.metadata["cost_source"] == "statistical_estimate"
    assert result.decision_audit.declared_expected_cost.tolist() == [Decimal("2.672")]*3
    assert result.decision_audit.loc[result.decision_audit.send.eq(1), "expected_cost"].eq(Decimal("2.672")).all()
    assert result.budget_summary["expected_send_cost"] == Decimal("3.34")
    assert result.budget_summary["expected_discount_cost"] == Decimal("2.004")
    held = build_strategy_decisions(weekly, cost_estimate=estimate(inputs,min_requests=4), **args)
    assert held.budget_summary["status"] == "cost_evidence_unavailable"
    assert held.metadata["cost_estimation"]["training"]["failure_charge_allowed"] is True
    args = options(weekly)
    args["cost_inputs"].loc[2,"source_as_of_at"] += pd.Timedelta(days=3)
    with pytest.raises(ValueError, match="future cost knowledge"):
        build_strategy_decisions(weekly, **args)


def test_eligible_business_exclusion_keeps_original_reason_and_price(weekly):
    args = options(weekly)
    args["eligibility"].loc[0, ["business_eligible", "business_reason"]] = [False, "opt_out"]
    result = build_strategy_decisions(weekly, **args)
    row = result.decision_audit.iloc[0]
    assert row.row_state == "eligibility_excluded" and row.eligibility_reason == "opt_out"
    assert row.send == 0 and row.declared_expected_cost == Decimal("4.5") and pd.isna(row.expected_cost)


def test_integrated_evaluation_requires_original_assignment_maturity_and_billing(tmp_path):
    from final_strategy.__main__ import _evaluation_input
    cutoff = pd.Timestamp("2024-03-03T15:00:00Z")
    assigned = pd.DataFrame(dict(dataset_id="eval", customer_id=["001","NA"], snapshot_id=["s1","s2"],
                                 raw_database_sha256="a"*64, as_of_at=cutoff, assigned_at=cutoff,
                                 strategy_group=["A","D2"]))
    assigned.to_parquet(tmp_path/"roster.parquet")
    outcomes = assigned.assign(outcome_end_at=cutoff+pd.Timedelta(weeks=4),
                               observation_end_at=cutoff+pd.Timedelta(weeks=4), actual_sent=False,
                               outcome_observed=True, send_cost_minor=0)
    def check(rows):
        rows.to_parquet(tmp_path/"outcomes.parquet")
        return _evaluation_input(tmp_path/"outcomes.parquet", tmp_path/"roster.parquet", ["strategy_group"])
    assert check(outcomes.iloc[::-1]).customer_id.tolist() == ["NA","001"]
    for rows, reason in ((outcomes.iloc[:1],"original assignment"), (outcomes.assign(send_cost_minor=1),"positive charge"),
                         (outcomes.assign(outcome_observed=False),"censored"),
                         (outcomes.assign(observation_end_at=cutoff),"follow-up")):
        with pytest.raises(ValueError, match=reason):
            check(rows)


def test_delayed_plan_keeps_budget_but_withholds_cutoff_economics(weekly):
    args = options(weekly)
    p = args["profile"]
    args["profile"] = replace(p, policies=tuple(replace(r,send_delay=pd.Timedelta(hours=1)) for r in p.policies))
    result = build_strategy_decisions(weekly, **args)
    assert result.budget_summary["send_count"] == 1 and result.budget_summary["expected_roi"] is None
    row = result.decision_audit.query("send == 1").iloc[0]
    assert row.requires_eligibility_recheck and row.planned_sent_at == row.as_of_at + pd.Timedelta(hours=1)
    assert pd.isna(row.expected_net_value)


def test_untouched_decimal_unit_and_cli_string_ids(cases, tmp_path):
    from final_strategy.__main__ import read
    features, batches = cases
    unit = "0.012345678901234567890123456789"
    features["amount_unit"] = unit
    for batch in batches.values():
        batch.frame["amount_unit"] = unit
    with localcontext(Context(prec=2)):
        assert adapt_weekly_predictions(features, **batches).roster.amount_unit.eq(unit).all()
    path = tmp_path / "ids.csv"
    path.write_text("customer_id,p0\nNA,\n001,0.5\n")
    table = read(path)
    assert table.customer_id.tolist() == ["NA", "001"] and pd.isna(table.p0.iloc[0])

def test_independent_comparison_and_sensitivity_details(weekly, cases, historical_data, tmp_path):
    from final_strategy.pipeline import write_outputs
    from final_strategy.policy_comparison import compare_policies
    from final_strategy.policy_sensitivity import run_policy_sensitivity
    features, _ = cases
    args = options(weekly)
    costs = estimate(historical_data | {"target_features":features})
    del args["cost_inputs"], args["cost_metadata"]
    result = build_strategy_decisions(weekly, cost_estimate=costs, **args)
    shared = {k:v for k,v in args.items() if k not in ("reporting","profile")}
    shared.update(inputs=weekly.integrated, dataset_id="dummy", as_of_at=weekly.roster.as_of_at.iloc[0])
    shared["eligibility"] = shared["eligibility"].query("customer_id != '003'")
    # Compare on the scored cohort, as the policy engine requires.
    scored_costs = estimate(historical_data | {"target_features":features.query("customer_id != '003'")})
    directions = {"purchase_rate_28d":"max"}
    comparison = compare_policies(build_strategy_profiles(**DECLARATIONS), scenario_id="synthetic",
        search_inputs=shared | {"cost_inputs":scored_costs.cost_inputs, "cost_metadata":scored_costs.cost_metadata},
        pareto_directions=directions, pairs=[])
    sensitivity = run_policy_sensitivity(profile_settings=DECLARATIONS, search_inputs=shared,
        cost_estimate=scored_costs, scenarios=[], max_scenarios=1, pareto_directions=directions)
    out = write_outputs(result, tmp_path/"details", comparison=comparison, sensitivity=sensitivity)
    for name, detail in (("policy_comparison",comparison),("policy_sensitivity",sensitivity)):
        assert json.loads((out/f"{name}.json").read_text(encoding="utf-8")) == detail.to_dict()
