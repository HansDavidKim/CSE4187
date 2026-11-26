"""Explicit production Hybrid tests; no publisher or optional solver dependency."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget as engine, budget_quality as quality
from final_strategy.budget import Budget, CostMetadata, SolverConfig
from final_strategy.hybrid import HybridConfig
from final_strategy.budget_quality_integration import certify_budget_result
from final_strategy.tests.test_budget import population, run, chosen, DATASET, T0
from final_strategy.tests.test_integration import cohort, retain, assert_unchanged, EXPECTED_SCENARIOS
from final_strategy.validation import InputValidationError
from final_strategy.what_if import run_what_if, export_what_if, plot_what_if


def exact(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def what_if(seg, costs, amount=50, places=0, **kwargs):
    return run_what_if(seg, costs, Budget(amount, "SOURCE_UNSPECIFIED", ".01"),
                      CostMetadata("explicit", "test", places, T0), dataset_id=DATASET, as_of_at=T0,
                      algorithm="hybrid", **kwargs)


def test_counterexample_improves_without_claiming_exact_optimality():
    result = run(algorithm="hybrid")
    assert list(result.results) == ["hybrid"]
    summary = result.results["hybrid"].summary
    evidence = result.metadata["hybrid"]
    assert chosen(result, "hybrid") == {"00001", "00002"}
    assert exact(evidence["selection"]["initial_greedy_score"]) == 160
    assert exact(evidence["selection"]["final_score"]) == 220
    assert exact(evidence["selection"]["improvement"]) == 60
    assert exact(evidence["quality"]["certificate"]["upper_bound"]) == 240
    assert summary["status"] == "feasible_uncertified"
    assert summary["total_expected_cost"] == Decimal(50)
    assert summary["has_valid_solution"] and not summary["accepted_for_planning"]
    assert summary["solver_status"] is summary["optimality_gap"] is None
    assert result.comparison["objective_difference"] is None
    assert not evidence["additional_solver_used"]


@pytest.mark.parametrize("scores,costs,amount,places", [
    ([], [], 0, 0), ([0,-1], [0,1], 0, 0), ([3,4], [0,1], 0, 0),
    ([3,4], [0,0], 0, 0), ([10,20,30], ["0.050000025"]*3, ".1", 9),
    ([10,20], ["0.050000025"]*2, ".1", 9),
    ([1,2], [".000000001",".000000002"], ".000000002", 9),
    ([2,3], [2,3], 3, 0), ([2,4], [2,4], 5, 0), ([10], [100], 1, 0),
])
def test_boundaries_and_early_exit(scores, costs, amount, places, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no improvement/solver required")
    monkeypatch.setattr(quality, "_improve_selection", forbidden)
    monkeypatch.setattr(engine, "_solve_milp", forbidden)
    result = run(*population(scores, costs), amount=amount, places=places, algorithm="hybrid")
    evidence = result.metadata["hybrid"]
    assert evidence["quality"]["status"] == "proven_optimal"
    assert not evidence["selection"]["local_improvement_executed"]
    assert evidence["timings"]["local_improvement_seconds"] == 0
    assert result.results["hybrid"].summary["total_expected_cost"] <= Decimal(str(amount))


def test_epsilon_exit_and_allow_unproven_does_not_approve(monkeypatch):
    monkeypatch.setattr(quality, "_improve_selection", lambda *a, **k: pytest.fail("target already met"))
    result = run(algorithm="hybrid", hybrid_config=HybridConfig(epsilon="0.34"),
                 config=SolverConfig(allow_unproven=True))
    assert result.results["hybrid"].summary["status"] == "epsilon_certified"
    assert not result.results["hybrid"].summary["accepted_for_planning"]


@pytest.mark.parametrize("mode", ["lower", "tie", "unknown", "duplicate", "overbudget", "exception"])
def test_improvement_cannot_damage_baseline(mode, monkeypatch):
    def substitute(problem, ids, **kwargs):
        if mode == "exception":
            raise InputValidationError("controlled invalid improvement")
        selected = {"lower": ["00000"], "tie": list(ids), "unknown": ["unknown"],
                    "duplicate": ["00000", "00000"], "overbudget": list(problem.customer_ids)}[mode]
        return SimpleNamespace(selected_customer_ids=selected, lower_bound=10**99), {"controlled": mode}
    monkeypatch.setattr(quality, "_improve_selection", substitute)
    result = run(algorithm="hybrid")
    assert chosen(result, "hybrid") == {"00000", "00001"}
    assert result.metadata["hybrid"]["selection"]["source"] == "greedy"
    assert exact(result.metadata["hybrid"]["quality"]["certificate"]["lower_bound"]) == 160


def test_equal_score_different_selection_keeps_original(monkeypatch):
    monkeypatch.setattr(quality, "_improve_selection", lambda p, ids, **kw:
                       (SimpleNamespace(selected_customer_ids=["00001"]), {}))
    result = run(*population([10,10,9], [6,6,6]), amount=10, algorithm="hybrid")
    assert chosen(result, "hybrid") == {"00000"}


@pytest.mark.parametrize("setting", [
    {"epsilon": .01}, {"epsilon": True}, {"epsilon": "nan"}, {"epsilon": "-1"}, {"epsilon": "1.01"},
    {"local_improvement": 1}, {"pool_size": True}, {"pool_size": 65}, {"pool_size": -1},
    {"max_passes": 9}, {"max_passes": 1.5}, {"time_limit": float("inf")},
    {"time_limit": float("nan")}, {"time_limit": -1}, {"time_limit": 6}, {"time_limit": True},
])
def test_config_limits(setting):
    with pytest.raises(InputValidationError):
        HybridConfig(**setting)


@pytest.mark.parametrize("algorithm", ["both", "milp", "greedy"])
def test_config_not_silently_ignored(algorithm):
    with pytest.raises(InputValidationError, match="requires algorithm"):
        run(algorithm=algorithm, hybrid_config=HybridConfig())
    seg, costs = population()
    with pytest.raises(InputValidationError, match="requires algorithm"):
        run_what_if(seg, costs, Budget(50,"SOURCE_UNSPECIFIED",".01"), CostMetadata("x","v1",0,T0),
                    dataset_id=DATASET, as_of_at=T0, algorithm=algorithm, hybrid_config=HybridConfig())


def test_legacy_both_still_runs_greedy_then_milp(monkeypatch):
    calls = []
    original = engine._execute
    def execute(p, algorithm, config):
        calls.append(algorithm)
        return original(p, algorithm, config)
    monkeypatch.setattr(engine, "_execute", execute)
    monkeypatch.setattr(quality, "fractional_upper_bound", lambda p: pytest.fail("legacy mode must not certify"))
    result = run()
    assert calls == ["greedy", "milp"]
    assert list(result.results) == ["milp", "greedy"]
    assert result.comparison["objective_difference"] == 60
    assert "hybrid" not in result.metadata


def test_one_preparation_and_one_upper_bound_per_hybrid(monkeypatch):
    counts = {"prepare": 0, "upper": 0}
    prepare, upper = engine._prepare, quality.fractional_upper_bound
    def prep(*args):
        counts["prepare"] += 1
        return prepare(*args)
    def bound(problem):
        counts["upper"] += 1
        return upper(problem)
    monkeypatch.setattr(engine, "_prepare", prep)
    monkeypatch.setattr(quality, "fractional_upper_bound", bound)
    run(algorithm="hybrid")
    assert counts == {"prepare": 1, "upper": 1}


@pytest.mark.parametrize("seed", range(12))
def test_exact_bounds_against_all_subsets(seed):
    rng = np.random.default_rng(seed)
    costs = rng.integers(0, 11, 8).tolist()
    scores = (rng.integers(-2, 30, 8) / 7).tolist()
    result = run(*population(scores, costs), amount=19, algorithm="hybrid",
                 hybrid_config=HybridConfig(epsilon="0", time_limit=1))
    evidence = result.metadata["hybrid"]["quality"]
    lower, upper = (exact(evidence["certificate"][field]) for field in ("lower_bound", "upper_bound"))
    optimum = max(sum((Fraction.from_float(s)*b for s,b in zip(scores,bits)), Fraction())
                  for bits in product((0,1), repeat=8) if sum(c*b for c,b in zip(costs,bits)) <= 19)
    assert lower <= optimum <= upper
    assert evidence["proven_optimal"] == (lower == upper)
    if evidence["proven_optimal"]:
        assert lower == optimum


@pytest.mark.parametrize("change", ["duplicate", "source", "time", "cost_missing", "nonfinite", "overflow"])
def test_contract_errors_still_rejected(change):
    seg, costs = population()
    if change == "duplicate":
        seg = replace(seg, customers=pd.concat([seg.customers, seg.customers.iloc[:1]]))
    elif change == "source":
        costs.loc[0,"raw_database_sha256"] = "b"*64
    elif change == "time":
        costs.loc[0,"as_of_at"] = T0 + pd.Timedelta(days=1)
    elif change == "cost_missing":
        costs = costs.iloc[1:]
    elif change == "nonfinite":
        costs.loc[0,"expected_cost"] = float("inf")
    else:
        costs.expected_cost = [1, 10**13, 10**13-1]
    with pytest.raises(InputValidationError):
        run(seg, costs, amount=10**13 if change == "overflow" else 50, algorithm="hybrid")


def test_no_external_problem_or_stale_state_and_no_mutation():
    seg, costs = population()
    before = deepcopy((seg, costs))
    result = run(seg, costs, algorithm="hybrid")
    assert_unchanged((seg, costs), before)
    shuffled = run(replace(seg, customers=seg.customers.iloc[::-1]), costs.iloc[::-1], algorithm="hybrid")
    assert chosen(result,"hybrid") == chosen(shuffled,"hybrid")
    result.metadata["segmentation"]["models"]["churn"]["model_version"] = "mutated"
    result.metadata["hybrid"]["quality"]["certificate"]["lower_bound"]["numerator"] = "0"
    result.results["hybrid"].decisions.loc[0,"priority_score"] = 0
    assert_unchanged((seg, costs), before)
    again = run(seg, costs, algorithm="hybrid")
    assert exact(again.metadata["hybrid"]["quality"]["certificate"]["lower_bound"]) == 220
    with pytest.raises(InputValidationError):
        run(seg, costs, algorithm="hybrid", hybrid_config=SimpleNamespace(input_sha256="forged"))


def test_phase_a_still_only_certifies_external_greedy():
    seg, costs = population()
    budget = Budget(50,"SOURCE_UNSPECIFIED",".01")
    meta = CostMetadata("explicit-external-cost","dummy-v1",0,T0)
    comparison = run(seg,costs,algorithm="hybrid")
    evidence = certify_budget_result(seg,costs,budget,meta,comparison,dataset_id=DATASET,as_of_at=T0)
    assert evidence.certification_status == "unavailable"
    assert evidence.reason == "greedy_result_missing"


def test_whatif_preflight_independent_bounds_and_no_shared_state(monkeypatch):
    seg, costs = population([10,20,30], [6,10,14])
    counts = []
    original = engine._prepare
    def prepare(*args):
        p = original(*args)
        counts.append((p["budget"],p["scaled_budget"],p["metadata"]["constraint_cost_gcd"]))
        return p
    monkeypatch.setattr(engine,"_prepare",prepare)
    result = what_if(seg,costs,amount=10)
    assert counts == [(Decimal(5),1,None),(Decimal(10),5,2),(Decimal(20),10,2)]
    certs = [r.metadata["hybrid"]["quality"]["certificate"] for r in result.results.values()]
    assert len({c["input_sha256"] for c in certs}) == 3
    assert len({exact(c["upper_bound"]) for c in certs}) == 3
    assert result.summary.objective_difference.isna().all()
    assert result.summary.relative_improvement.isna().all()
    certs[0]["lower_bound"]["numerator"] = "999"
    assert certs[1]["lower_bound"]["numerator"] != "999"


def test_subset_still_preflights_all_three_before_greedy(monkeypatch):
    seg,costs = population()
    calls=[]
    prepare,execute=engine._prepare,engine._execute
    def prep(*args):
        calls.append("prepare")
        return prepare(*args)
    def run_stage(*args):
        calls.append("execute")
        return execute(*args)
    monkeypatch.setattr(engine,"_prepare",prep)
    monkeypatch.setattr(engine,"_execute",run_stage)
    result=what_if(seg,costs,scenarios=["budget_100"])
    assert calls == ["prepare"]*3+["execute"]
    assert list(result.results)==["budget_100"]


def test_exports_exact_values_ids_nulls_and_plot(tmp_path):
    seg,costs=population([.1,.2,.3],[".05"]*3,ids=["001","NA","003"])
    result=what_if(seg,costs,amount=".2",places=3)
    paths=export_what_if(result,tmp_path)
    rows=pd.read_csv(paths["decisions"],dtype=str,keep_default_na=False)
    summary=pd.read_csv(paths["summary"],dtype=str,keep_default_na=False)
    metadata=json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert set(rows.customer_id)=={"001","NA","003"}
    assert set(rows.expected_cost)=={"0.05"}
    assert all(pd.Timestamp(value)==T0 for value in rows.as_of_at)
    assert summary.solver_status.eq("").all()
    assert summary.accepted_for_planning.eq("False").all()
    for row in summary.to_dict("records"):
        cert=metadata["scenario_metadata"][row["scenario_id"]]["hybrid"]["quality"]["certificate"]
        assert Fraction(int(row["quality_lower_bound_numerator"]),int(row["quality_lower_bound_denominator"])) == exact(cert["lower_bound"])
        assert Fraction(int(row["quality_epsilon_numerator"]),int(row["quality_epsilon_denominator"]))==Fraction(".0001")
    figure=plot_what_if(result)
    figure.savefig(tmp_path/"hybrid.png")
    assert len(figure.axes)==3 and (tmp_path/"hybrid.png").stat().st_size>0
    assert "Hybrid" in figure._suptitle.get_text()


def test_actual_retention_cohort(cohort):
    original=deepcopy(cohort)
    retention=retain(cohort)
    retained=deepcopy(retention)
    assert len(cohort["inputs"].data)==60
    assert len(cohort["segmentation"].customers)==59
    assert len(retention.budget_inputs.customers)==18 and len(retention.budget_inputs.excluded)==42
    result=run_what_if(retention.budget_inputs,retention.costs,cohort["budget"],retention.cost_metadata,
                      dataset_id="dummy-segmentation",as_of_at=cohort["cost_metadata"].as_of_at,
                      algorithm="hybrid")
    for name,(amount,count,cost,score) in EXPECTED_SCENARIOS.items():
        allocation=result.results[name].results["hybrid"]
        assert len(allocation.decisions)==60
        assert allocation.summary["selected_customer_count"]==count
        assert allocation.summary["total_expected_cost"]==cost<=amount
        assert allocation.summary["objective_value"]==score
        assert allocation.summary["status"]=="proven_optimal"
        assert not allocation.summary["accepted_for_planning"]
        candidates=allocation.decisions.loc[allocation.decisions.customer_id.isin(retention.costs.customer_id)]
        # The existing Budget row contract exports bands/segment/score, not the
        # upstream is_high_clv/priority_rank columns. Those stay in the input.
        columns = [*engine.SEGMENT_COLUMNS, "priority_score"]
        pd.testing.assert_frame_equal(candidates[columns].reset_index(drop=True),
            retention.budget_inputs.customers.sort_values("customer_id")[columns].reset_index(drop=True),check_dtype=False)
    assert_unchanged(retention,retained)
    assert_unchanged(cohort,original)


def test_improvement_reaches_epsilon_but_is_not_approved():
    result = run(algorithm="hybrid", hybrid_config=HybridConfig(epsilon="0.1"))
    assert result.results["hybrid"].summary["status"] == "epsilon_certified"
    assert result.metadata["hybrid"]["selection"]["source"] == "local_improvement"
    assert not result.results["hybrid"].summary["accepted_for_planning"]


def test_disabled_improvement_returns_valid_uncertified_baseline(monkeypatch):
    monkeypatch.setattr(quality, "_improve_selection", lambda *a, **kw: pytest.fail("disabled"))
    result = run(algorithm="hybrid", hybrid_config=HybridConfig(local_improvement=False))
    assert result.results["hybrid"].summary["status"] == "feasible_uncertified"
    assert chosen(result,"hybrid") == {"00000", "00001"}


def test_unavailable_has_no_fabricated_selection(monkeypatch, tmp_path):
    original = engine._execute
    def invalid(*args):
        allocation=original(*args)
        allocation.summary["has_valid_solution"]=False
        return allocation
    monkeypatch.setattr(engine,"_execute",invalid)
    result=what_if(*population())
    assert result.summary.status.eq("unavailable").all()
    assert result.summary.total_expected_cost.isna().all()
    assert result.decisions.selected.isna().all()
    assert not result.summary.accepted_for_planning.any()
    paths=export_what_if(result,tmp_path)
    restored=json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert restored["scenario_metadata"]["budget_100"]["hybrid"]["quality"]["certificate"] is None


def test_future_fields_not_forwarded_by_hybrid():
    seg,costs=population()
    seg.customers["future_churn_label"]=1
    seg.customers["experiment_assignment"]="treatment"
    result=run(seg,costs,algorithm="hybrid")
    assert "future_churn_label" not in result.results["hybrid"].decisions
    assert "experiment_assignment" not in result.results["hybrid"].decisions
    seg.customers.attrs["evaluation_only"]=True
    with pytest.raises(InputValidationError,match="evaluation_only"):
        run(seg,costs,algorithm="hybrid")


@pytest.mark.parametrize("change", ["binary", "cost", "context", "score", "summary"])
def test_actual_decisions_are_checked_before_certification(change, monkeypatch):
    execute=engine._execute
    def tamper(*args):
        result=execute(*args)
        if change=="binary": result.decisions.loc[0,"selected"]=2
        elif change=="cost": result.decisions.loc[0,"expected_cost"]=Decimal(0)
        elif change=="context": result.decisions.loc[0,"snapshot_id"]="tampered"
        elif change=="score": result.decisions.loc[0,"priority_score"]=999
        else: result.summary["objective_value"]=999
        return result
    monkeypatch.setattr(engine,"_execute",tamper)
    with pytest.raises(InputValidationError,match="hybrid:"):
        run(algorithm="hybrid")
