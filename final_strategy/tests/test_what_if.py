"""Scenario tests with in-memory inputs; publisher and real models are unused."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget as engine
from final_strategy import what_if
from final_strategy.budget import Budget, CostMetadata, SolverConfig
from final_strategy.tests.test_budget import population, fake_solver, T0, DATASET
from final_strategy.validation import CONTEXT_COLUMNS, InputValidationError
from final_strategy.what_if import build_budget_scenarios, run_what_if, export_what_if, plot_what_if


def run(segmentation=None, costs=None, amount=100, places=0, **kwargs):
    if segmentation is None:
        segmentation, costs = population([100, 24, 24, 24, 24], [100, 25, 25, 25, 25], ids=["a", "b", "c", "d", "e"])
    return run_what_if(segmentation, costs, Budget(amount, "SOURCE_UNSPECIFIED", ".01"),
                       CostMetadata("external-dummy", "v1", places, T0), dataset_id=DATASET, as_of_at=T0, **kwargs)


def test_real_scenarios_and_non_nested_selection():
    result = run()
    assert result.summary.budget.drop_duplicates().tolist() == [Decimal(50), Decimal(100), Decimal(200)]
    for algorithm in ("milp", "greedy"):
        rows = result.summary.loc[result.summary.algorithm.eq(algorithm)]
        assert rows.selected_customer_count.tolist() == [2, 1, 5]
        assert rows.objective_value.tolist() == [48, 100, 196]
        assert rows.total_expected_cost.tolist() == [50, 100, 200]
        assert rows.budget_utilization.tolist() == [1, 1, 1]
    assert result.summary.budget_compliant.all()
    assert result.comparisons.objective_difference.eq(0).all()


def test_budget_zero_free_customers_and_null_utilization():
    result = run(*population([1, -1, 2], [0, 0, 1]), amount=0)
    assert result.summary.selected_customer_count.eq(1).all()
    assert result.summary.total_expected_cost.eq(0).all()
    assert result.summary.budget_utilization.isna().all()
    assert result.summary.budget_compliant.all()


def test_decimal_scenarios_ignore_ambient_context_precision():
    with localcontext() as context:
        context.prec = 6
        budgets = build_budget_scenarios(Budget("12345.678", "SOURCE_UNSPECIFIED", ".01"),
                                          CostMetadata("d", "v", 3, T0))
    assert [value.amount for value in budgets.values()] == [Decimal("6172.8390"), Decimal("12345.678"), Decimal("24691.356")]


@pytest.mark.parametrize("amount,places", [(".05", 2), ("1", 0), ("600000000000000000", 0),
                                           (-1, 2), (float("inf"), 2), (True, 2)])
def test_invalid_scenario_budget_fails_before_any_allocation(monkeypatch, amount, places):
    def forbidden(*args, **kwargs): pytest.fail("allocation started before preflight completed")
    monkeypatch.setattr(engine, "optimize_budget", forbidden)
    with pytest.raises(InputValidationError): run(amount=amount, places=places, scenarios=["budget_100"])


@pytest.mark.parametrize("change", ["currency", "unit", "dataset", "snapshot", "duplicate", "missing"])
def test_cost_contract_errors_preflight_before_any_allocation(monkeypatch, change):
    segmented, costs = population()
    if change == "currency": costs.loc[0, "currency"] = "KRW"
    if change == "unit": costs.loc[0, "amount_unit"] = "1"
    if change == "dataset": costs.loc[0, "dataset_id"] = "other"
    if change == "snapshot": costs.loc[0, "snapshot_id"] = "other"
    if change == "duplicate": costs = pd.concat([costs, costs.iloc[:1]])
    if change == "missing": costs = costs.iloc[1:]
    def forbidden(*args, **kwargs): pytest.fail("allocation started on invalid inputs")
    monkeypatch.setattr(engine, "optimize_budget", forbidden)
    with pytest.raises(InputValidationError): run(segmented, costs)


def test_all_budget_dependent_scales_preflight_even_for_subset(monkeypatch):
    segmented, costs = population([1], ["1"])
    def forbidden(*args, **kwargs): pytest.fail("allocation started before scale validation")
    monkeypatch.setattr(engine, "optimize_budget", forbidden)
    # 050 and 100 fit binary64's exact integer range; 200 exceeds it.
    with pytest.raises(InputValidationError):
        run(segmented, costs, amount="9007199254740990", scenarios=["budget_050"])


@pytest.mark.parametrize("scenarios", [[], ["unknown"], ["budget_100", "budget_100"], "budget_100", [1]])
def test_invalid_scenario_selection(scenarios):
    with pytest.raises(InputValidationError): run(scenarios=scenarios)


@pytest.mark.parametrize("algorithm", ["invalid", None])
def test_invalid_algorithm(algorithm):
    with pytest.raises(InputValidationError): run(algorithm=algorithm)


def test_independent_calls_same_objects_and_config(monkeypatch):
    segmented, costs = population()
    original = deepcopy(segmented); cost_original = costs.copy(deep=True)
    config = SolverConfig(presolve=False)
    calls = []
    real = engine.optimize_budget
    def observe(seg, table, budget, metadata, **kwargs):
        assert seg is segmented and table is costs and kwargs["config"] is config
        calls.append(budget.amount)
        return real(seg, table, budget, metadata, **kwargs)
    monkeypatch.setattr(engine, "optimize_budget", observe)
    first = run(segmented, costs, config=config, algorithm="greedy")
    second = run(segmented, costs, config=config, algorithm="greedy",
                 scenarios=["budget_200", "budget_100", "budget_050"])
    assert calls == [50, 100, 200, 200, 100, 50]
    keys = ["scenario_id", "algorithm", "customer_id"]
    pd.testing.assert_frame_equal(first.decisions.sort_values(keys).reset_index(drop=True),
                                  second.decisions.sort_values(keys).reset_index(drop=True))
    pd.testing.assert_frame_equal(segmented.customers, original.customers)
    pd.testing.assert_frame_equal(segmented.excluded, original.excluded)
    pd.testing.assert_frame_equal(costs, cost_original)
    assert segmented.metadata == original.metadata


@pytest.mark.parametrize("algorithm", ["milp", "greedy", "both"])
def test_selected_scenario_and_algorithm(algorithm):
    result = run(algorithm=algorithm, scenarios=["budget_100"])
    assert set(result.summary.scenario_id) == {"budget_100"}
    assert set(result.summary.algorithm) == ({"milp", "greedy"} if algorithm == "both" else {algorithm})
    assert set(result.metadata["scenario_budgets"]) == {"budget_050", "budget_100", "budget_200"}


def test_mock_one_milp_failure_preserves_other_scenarios_and_greedy(monkeypatch):
    def controlled(prepared, config):
        return fake_solver(prepared, config, status=1, success=False, x=None) if prepared["budget"] == 100 else fake_solver(prepared, config)
    monkeypatch.setattr(engine, "_solve_milp", controlled)
    result = run(*population([60, 100, 120], [10, 20, 30]))
    rows = result.summary.set_index(["scenario_id", "algorithm"])
    missing = rows.loc[("budget_100", "milp")]
    assert missing.status == "limit_no_incumbent" and not missing.has_valid_solution
    assert pd.isna(missing.objective_value) and pd.isna(missing.selected_customer_count)
    decisions = result.decisions.loc[result.decisions.scenario_id.eq("budget_100") & result.decisions.algorithm.eq("milp")]
    assert decisions.selected.isna().all()
    assert rows.loc[("budget_200", "milp"), "has_valid_solution"]
    assert result.summary.loc[result.summary.algorithm.eq("greedy"), "has_valid_solution"].all()


def test_mock_low_quality_limit_incumbent_preserves_difference_and_approval(monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, status=1, success=False, mip_gap=.5))
    result = run(*population([60, 100, 120], [10, 20, 30]), scenarios=["budget_100"])
    row = result.summary.loc[result.summary.algorithm.eq("milp")].iloc[0]
    assert row.status == "limit_feasible" and row.objective_value == 60 and row.optimality_gap == .5
    assert row.objective_difference < 0 and not row.accepted_for_planning and row.optimality == "not_proven"
    allowed = run(*population(), scenarios=["budget_100"], config=SolverConfig(allow_unproven=True))
    assert allowed.summary.loc[allowed.summary.algorithm.eq("milp"), "accepted_for_planning"].all()


def test_mock_solver_exception_is_partial_but_unexpected_wrapper_exception_propagates(monkeypatch):
    def solver_error(*args): raise RuntimeError("solver error")
    monkeypatch.setattr(engine, "_solve_milp", solver_error)
    result = run()
    assert result.summary.loc[result.summary.algorithm.eq("milp"), "status"].eq("solver_error").all()
    assert result.summary.loc[result.summary.algorithm.eq("greedy"), "has_valid_solution"].all()
    def code_error(*args, **kwargs): raise RuntimeError("unexpected wrapper bug")
    monkeypatch.setattr(engine, "optimize_budget", code_error)
    with pytest.raises(RuntimeError, match="unexpected wrapper bug"): run()


def test_exclusions_and_full_source_metadata_separate_from_selected_cohort():
    segmented, costs = population()
    excluded = segmented.customers.iloc[:1][list(CONTEXT_COLUMNS)].copy()
    excluded["exclusion_reason"] = "explicitly_ineligible"
    segmented = replace(segmented, customers=segmented.customers.iloc[1:], excluded=excluded)
    segmented.metadata["original_other_cohort"] = "source context only"
    result = run(segmented, costs.iloc[1:], algorithm="greedy")
    rows = result.decisions.loc[result.decisions.customer_id.eq("00000")]
    assert len(rows) == 3 and rows.selected.eq(0).all()
    assert rows.exclusion_reason.eq("explicitly_ineligible").all()
    assert rows.expected_cost.isna().all()
    assert result.metadata["selected_cohort"]["candidate_count"] == 2
    assert result.metadata["selected_cohort"]["excluded_count"] == 1
    assert result.metadata["source_segmentation"] == segmented.metadata


def test_empty_population_and_zero_denominator_comparison():
    result = run(*population([], []), algorithm="both")
    assert result.decisions.empty and len(result.summary) == 6
    assert result.summary.objective_value.eq(0).all()
    assert result.comparisons.relative_improvement.isna().all()


@pytest.mark.parametrize("seed", range(4))
def test_real_milp_scenarios_match_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    scores, costs = rng.integers(-2, 21, 8), rng.integers(0, 11, 8)
    result = run(*population(scores, costs), amount=20)
    for row in result.summary.loc[result.summary.algorithm.eq("milp")].itertuples():
        optimum = max(sum(s * x for s, x in zip(scores, bits)) for bits in product((0, 1), repeat=8)
                      if sum(c * x for c, x in zip(costs, bits)) <= row.budget)
        assert row.objective_value == optimum and row.budget_compliant


@pytest.mark.parametrize("presolve", [True, False])
def test_customer_totals_and_decimal_costs_match_summary(presolve):
    result = run(*population([10, 20, 30], [".050000025"] * 3), amount=".2", places=9,
                 config=SolverConfig(presolve=presolve))
    for row in result.summary.itertuples():
        assert row.has_valid_solution and row.budget_compliant
        if row.algorithm == "milp":
            assert row.status == "optimal_numeric" and row.optimality_gap == 0
        selected = result.decisions.loc[result.decisions.scenario_id.eq(row.scenario_id)
                                        & result.decisions.algorithm.eq(row.algorithm) & result.decisions.selected.eq(1)]
        assert len(selected) == row.selected_customer_count
        assert sum(selected.expected_cost, Decimal(0)) == row.total_expected_cost <= row.budget
        assert sum(selected.priority_score) == row.objective_value
    assert result.summary.loc[result.summary.algorithm.eq("greedy"), "has_valid_solution"].all()
    first = result.results["budget_050"].results["milp"]
    assert first.summary["objective_value"] == 30
    assert first.summary["total_expected_cost"] == Decimal(".050000025")
    assert first.decisions.loc[first.decisions.selected.eq(1), "customer_id"].tolist() == ["00002"]


def test_inconsistent_customer_summary_is_not_silently_published(monkeypatch):
    real = engine.optimize_budget
    def inconsistent(*args, **kwargs):
        result = real(*args, **kwargs)
        result.results["greedy"].summary["selected_customer_count"] += 1
        return result
    monkeypatch.setattr(engine, "optimize_budget", inconsistent)
    with pytest.raises(RuntimeError, match="disagree with allocation totals"):
        run(algorithm="greedy")


def test_export_round_trip_preserves_ids_decimals_utc_null_and_state(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, status=1, success=False, x=None))
    result = run(*population([1, 2], [".05", ".02"], ids=["001", "NA"]), amount=".2", places=2)
    paths = export_what_if(result, tmp_path)
    summary = pd.read_csv(paths["summary"], dtype=str, keep_default_na=False)
    decisions = pd.read_csv(paths["decisions"], dtype=str, keep_default_na=False)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert set(decisions.customer_id) == {"001", "NA"}
    assert Decimal(summary.budget.iloc[0]) == Decimal(".10")
    assert summary.loc[summary.algorithm.eq("milp"), "objective_value"].eq("").all()
    assert decisions.loc[decisions.algorithm.eq("milp"), "selected"].eq("").all()
    assert decisions.as_of_at.str.endswith("+00:00").all()
    assert Decimal(decisions.expected_cost.iloc[0]) == Decimal(".05")
    assert metadata["selected_cohort"]["dataset_id"] == DATASET
    assert metadata["scenario_budgets"]["budget_050"]["budget_multiplier"] == "0.5"
    assert metadata["source_segmentation"]["models"] == result.metadata["source_segmentation"]["models"]
    assert set(metadata["algorithm_versions"]) == {"milp", "greedy"}
    before = {key: path.read_bytes() for key, path in paths.items()}
    with pytest.raises(FileExistsError): export_what_if(result, tmp_path)
    assert {key: path.read_bytes() for key, path in paths.items()} == before


def test_plot_has_three_axes_and_leaves_missing_milp_values_unplotted(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, status=1, success=False, x=None))
    result = run()
    figure = plot_what_if(result)
    assert len(figure.axes) == 3
    for axis in figure.axes:
        milp_lines = [line for line in axis.lines if line.get_label() == "milp"]
        assert all(np.isnan(line.get_ydata()).all() for line in milp_lines)
    figure.savefig(tmp_path / "what_if.png", dpi=80)
    assert (tmp_path / "what_if.png").stat().st_size > 1000


def test_plot_distinguishes_unproven_incumbent_and_rejects_mixed_context(monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, status=1, success=False, mip_gap=.5))
    result = run()
    assert any(line.get_marker() == "x" for line in plot_what_if(result).axes[0].lines)
    result.summary.loc[0, "currency"] = "KRW"
    with pytest.raises(InputValidationError, match="mixed currency"): plot_what_if(result)
