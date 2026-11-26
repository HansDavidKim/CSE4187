"""Independent allocation tests and an opt-in fresh-process benchmark CLI."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from itertools import product
from types import SimpleNamespace
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget as engine
from final_strategy.budget import Budget, CostMetadata, SolverConfig, optimize_budget
from final_strategy.segmentation import SegmentationResult, segment_customers
from final_strategy.validation import CONTEXT_COLUMNS, InputValidationError


T0 = pd.Timestamp("2024-03-03T15:00:00Z")
DATASET = "dummy-budget"


def population(scores=(60, 100, 120), costs=(10, 20, 30), ids=None):
    """Minimal already-segmented contract, not a simulator/model publication."""
    ids = [f"{index:05}" for index in range(len(scores))] if ids is None else ids
    rows = []
    for customer, score in zip(ids, scores):
        row = dict.fromkeys(engine.ROW_COLUMNS, "dummy-segment")
        row.update(dataset_id=DATASET, customer_id=customer, as_of_at=T0,
                   reference_week="2024-03-04", snapshot_id=f"opaque:{customer}",
                   raw_database_sha256="a" * 64, currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
                   churn_probability=.5, uplift_score=-1.0 if score < 0 else 1.0,
                   predicted_clv_12m=abs(float(score)), priority_score=float(score))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=engine.ROW_COLUMNS)
    models = {
        "churn": dict(model_version="dummy-churn", target_definition_id="no_visit", horizon_value=28,
                      horizon_unit="days", value_semantics="probability"),
        "uplift": dict(model_version="dummy-uplift", target_definition_id="purchase_retention", horizon_value=4,
                       horizon_unit="weeks", value_semantics="probability_difference"),
        "clv": dict(model_version="dummy-clv", target_definition_id="paid_revenue", horizon_value=52,
                    horizon_unit="weeks", value_semantics="amount"),
    }
    excluded = pd.DataFrame(columns=[*CONTEXT_COLUMNS, "exclusion_reason"])
    segmented = SegmentationResult(frame, excluded, pd.DataFrame(), {
        "population_scope": "validated_inputs", "config": {"segmentation_version": "dummy-v1"},
        "models": models, "input_snapshots": [{"opaque": "preserved"}],
    })
    table = frame[[key for key in engine.COST_COLUMNS if key != "expected_cost"]].copy()
    table["expected_cost"] = list(costs)
    return segmented, table


def run(segmented=None, costs=None, amount=50, places=0, **kwargs):
    if segmented is None:
        segmented, costs = population()
    return optimize_budget(segmented, costs, Budget(amount, "SOURCE_UNSPECIFIED", "0.01"),
                           CostMetadata("explicit-external-cost", "dummy-v1", places, T0),
                           dataset_id=DATASET, as_of_at=T0, **kwargs)


def chosen(result, name):
    frame = result.results[name].decisions
    return set(frame.loc[frame.selected.eq(1), "customer_id"])


def fake_solver(prepared, config, **overrides):
    vector = np.zeros(len(prepared["active"]))
    vector[0] = 1
    value = -float(prepared["objective_coefficients"] @ vector)
    values = dict(status=0, success=True, message="controlled solver response", x=vector,
                  fun=value, mip_dual_bound=value, mip_gap=0., mip_node_count=1)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_real_solver_greedy_counterexample_and_original_objective():
    result = run()
    milp, greedy = (result.results[name].summary for name in ("milp", "greedy"))
    assert milp["status"] == "optimal_numeric" and milp["accepted_for_planning"]
    assert milp["objective_value"] == 220 and milp["total_expected_cost"] == Decimal(50)
    assert greedy["objective_value"] == 160 and greedy["remaining_budget"] == Decimal(20)
    assert result.comparison["objective_difference"] == 60
    assert result.comparison["relative_improvement"] == .375
    assert greedy["optimality_gap"] is None
    assert result.comparison["memory_usage_difference"] is None


@pytest.mark.parametrize("seed", range(8))
def test_real_milp_matches_exhaustive_search_up_to_ten(seed):
    rng = np.random.default_rng(seed)
    n = seed + 3
    scores, costs = rng.integers(-2, 30, n), rng.integers(0, 20, n)
    limit = int(sum(costs) // 3)
    result = run(*population(scores, costs), amount=limit)
    optimum = max(sum(score * bit for score, bit in zip(scores, bits))
                  for bits in product((0, 1), repeat=n) if sum(cost * bit for cost, bit in zip(costs, bits)) <= limit)
    assert result.results["milp"].summary["objective_value"] == optimum
    for item in result.results.values():
        assert item.summary["total_expected_cost"] <= limit
        assert not item.decisions.customer_id.duplicated().any()


@pytest.mark.parametrize("amount", [0, 1, 100])
def test_free_nonpositive_overbudget_and_zero_budget(amount):
    result = run(*population([10, 0, -5, 12], [0, 0, 0, 101]), amount=amount)
    for name, item in result.results.items():
        assert chosen(result, name) == {"00000"}
        assert item.summary["total_expected_cost"] == 0 and item.summary["objective_value"] == 10


def test_empty_population_and_no_affordable_customer():
    for scores, costs in (([], []), ([10], [51])):
        result = run(*population(scores, costs))
        for item in result.results.values():
            assert item.summary["selected_customer_count"] == 0
            assert item.summary["remaining_budget"] == 50
        assert result.comparison["relative_improvement"] is None


def test_free_score_is_added_back_to_solver_objective_and_bound():
    result = run(*population([7, 60, 100, 120], [0, 10, 20, 30]))
    item = result.results["milp"].summary
    assert item["objective_value"] == 227 and item["objective_upper_bound"] == pytest.approx(227)
    assert item["total_expected_cost"] == 50


def test_duplicate_source_customer_rejected():
    segmented, costs = population()
    segmented = replace(segmented, customers=pd.concat([segmented.customers, segmented.customers.iloc[:1]]))
    with pytest.raises(InputValidationError, match="duplicate"): run(segmented, costs)


def test_actual_segmentation_output_and_exclusions():
    from final_strategy.tests.test_segmentation import make_inputs, eligibility_for
    inputs = make_inputs([("001", .5, .5, 100), ("002", .5, .2, 100), ("003", .5, .1, 100)])
    segmented = segment_customers(inputs, eligibility=eligibility_for(inputs, {"003": "not_eligible"}))
    costs = segmented.customers[[key for key in engine.COST_COLUMNS if key != "expected_cost"]].copy()
    costs["expected_cost"] = [5, 10]
    result = optimize_budget(segmented, costs, Budget(10, "SOURCE_UNSPECIFIED", ".01"),
                             CostMetadata("declared", "v1", 0, T0), dataset_id="dummy-segmentation", as_of_at=T0)
    for item in result.results.values():
        row = item.decisions.set_index("customer_id").loc["003"]
        assert row.selected == 0 and row.exclusion_reason == "not_eligible"
        assert row.selection_reason == "upstream_excluded" and pd.isna(row.expected_cost)
        assert "mean_net_order_value_30d" not in item.decisions
    assert result.metadata["segmentation"] == segmented.metadata


def test_all_excluded():
    segmented, costs = population([10], [1])
    excluded = segmented.customers[list(CONTEXT_COLUMNS)].copy()
    excluded["exclusion_reason"] = "ineligible"
    segmented = replace(segmented, customers=segmented.customers.iloc[:0], excluded=excluded)
    result = run(segmented, costs.iloc[:0])
    assert all(item.summary["selected_customer_count"] == 0 for item in result.results.values())


def test_greedy_skip_and_deterministic_ties():
    segmented, costs = population([20, 10, 10, 2], [10, 5, 5, 1], ids=["b", "02", "01", "a"])
    result = run(segmented, costs, amount=11)
    assert chosen(result, "greedy") == {"b", "a"}  # ratio tie -> score first, then skip
    tied, tied_cost = population([10, 10], [5, 5], ids=["2", "10"])
    assert chosen(run(tied, tied_cost, amount=5), "greedy") == {"10"}


def test_row_shuffle_and_source_nonmutation():
    segmented, costs = population()
    snapshot = deepcopy(segmented)
    original_costs = costs.copy(deep=True)
    first = run(segmented, costs)
    shuffled = replace(segmented, customers=segmented.customers.sample(frac=1, random_state=2))
    second = run(shuffled, costs.sample(frac=1, random_state=3))
    for name in first.results:
        pd.testing.assert_frame_equal(first.results[name].decisions, second.results[name].decisions)
    pd.testing.assert_frame_equal(segmented.customers, snapshot.customers)
    pd.testing.assert_frame_equal(segmented.excluded, snapshot.excluded)
    pd.testing.assert_frame_equal(costs, original_costs)
    assert segmented.metadata == snapshot.metadata
    first.metadata["segmentation"]["models"]["churn"]["model_version"] = "changed"
    assert segmented.metadata == snapshot.metadata


def test_multiple_optima_preserve_objective_without_id_perturbation():
    result = run(*population([10, 10], [5, 5]), amount=5)
    assert result.results["milp"].summary["objective_value"] == 10
    assert len(chosen(result, "milp")) == 1  # no promise about HiGHS' tie choice


def test_exact_fractional_cost_and_original_decimal_overspend_case():
    result = run(*population([1, 1], ["0.050000025"] * 2), amount="0.1", places=9)
    for item in result.results.values():
        assert item.summary["total_expected_cost"] == Decimal("0.050000025")
        assert item.summary["selected_customer_count"] == 1
    exact = run(*population([1, 2], ["0.1", "0.2"]), amount="0.3", places=1)
    assert exact.results["milp"].summary["remaining_budget"] == Decimal(0)


def test_float_decimal_policy_does_not_recover_unknown_precision():
    with pytest.raises(InputValidationError, match="represent exactly"):
        run(*population([1], [0.1 + 0.2]), amount=1, places=2)
    assert run(*population([1], [Decimal(".30")]), amount=".30", places=2).results["milp"].summary["remaining_budget"] == 0


@pytest.mark.parametrize("field", ["amount", "cost"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "bad", Decimal("-Infinity")])
def test_invalid_money(field, value):
    segmented, costs = population()
    if field == "cost":
        costs["expected_cost"] = costs.expected_cost.astype(object)
        costs.loc[0, "expected_cost"] = value
    with pytest.raises(InputValidationError):
        run(segmented, costs, amount=value if field == "amount" else 50)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1j, pd.Timestamp("2024-01-01")])
def test_invalid_score(value):
    segmented, costs = population()
    segmented.customers["priority_score"] = segmented.customers.priority_score.astype(object)
    segmented.customers.loc[0, "priority_score"] = value
    with pytest.raises(InputValidationError):
        run(segmented, costs)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "missing_column", "extra_column"])
def test_cost_coverage(change):
    segmented, costs = population()
    if change == "missing": costs = costs.iloc[1:]
    if change == "duplicate": costs = pd.concat([costs, costs.iloc[:1]])
    if change == "extra":
        extra = costs.iloc[:1].copy(); extra["customer_id"] = "unexpected"
        costs = pd.concat([costs, extra])
    if change == "missing_column": costs = costs.drop(columns="expected_cost")
    if change == "extra_column": costs["future_paid_revenue"] = 999
    with pytest.raises(InputValidationError): run(segmented, costs)


@pytest.mark.parametrize("field,value", [("dataset_id", "other"), ("as_of_at", T0 + pd.Timedelta(days=1)),
                                         ("snapshot_id", "other"), ("raw_database_sha256", "b" * 64),
                                         ("currency", "KRW"), ("amount_unit", "1")])
def test_cost_context_mismatch(field, value):
    segmented, costs = population()
    costs.loc[0, field] = value
    with pytest.raises(InputValidationError): run(segmented, costs)


def test_cohort_selection_is_explicit_and_same_week_snapshots_are_distinct():
    segmented, costs = population()
    other = segmented.customers.copy(); other["as_of_at"] = T0 + pd.Timedelta(hours=1)
    other["snapshot_id"] = "another-opaque-snapshot"
    segmented = replace(segmented, customers=pd.concat([segmented.customers, other]))
    assert run(segmented, costs).metadata["candidate_count"] == 3
    with pytest.raises(InputValidationError):
        run(segmented, pd.concat([costs, costs.assign(as_of_at=T0 + pd.Timedelta(hours=1))]))


def test_provenance_and_score_consistency_revalidated():
    segmented, costs = population()
    segmented.customers.loc[0, "raw_database_sha256"] = "b" * 64
    costs.loc[0, "raw_database_sha256"] = "b" * 64
    with pytest.raises(InputValidationError, match="mixed source"): run(segmented, costs)
    segmented, costs = population()
    segmented.customers.loc[0, "priority_score"] = 999
    with pytest.raises(InputValidationError, match="differs from uplift"): run(segmented, costs)


@pytest.mark.parametrize("places", [-1, 10, 1.5, True])
def test_precision_declaration_rejected(places):
    with pytest.raises(InputValidationError): run(places=places)


def test_precision_and_safe_ranges_and_exact_gcd_reduction():
    for amount, cost, places in ((".0000000001", "0", 9), ("1e18", "1", 0)):
        with pytest.raises(InputValidationError): run(*population([1], [cost]), amount=amount, places=places)
    # Coprime costs keep the actual solver coefficient above the safe range.
    with pytest.raises(InputValidationError, match="safe coefficient range"):
        run(*population([1, 1], ["1000000000001", "1"]), amount="1000000000002")
    result = run(*population([1], ["1e18"]), amount="1e18")
    assert result.results["milp"].summary["total_expected_cost"] == Decimal("1e18")
    boundary = run(*population([1, 1], ["1000000000000", "1"]), amount="1000000000001")
    assert boundary.results["milp"].summary["total_expected_cost"] == Decimal("1000000000001")
    with pytest.raises(InputValidationError, match="dynamic range"):
        run(*population([1e-12, 1], [1, 1]), amount=2)


@pytest.mark.parametrize("kwargs", [{"time_limit": -1}, {"mip_rel_gap": 2}, {"binary_tolerance": .1}, {"allow_unproven": 1}])
def test_solver_config_rejected(kwargs):
    with pytest.raises(InputValidationError): SolverConfig(**kwargs)


def test_future_cost_timestamp_and_naive_timestamp():
    segmented, costs = population()
    for clock in (T0 + pd.Timedelta(seconds=1), "2024-03-03"):
        with pytest.raises(InputValidationError):
            optimize_budget(segmented, costs, Budget(50, "SOURCE_UNSPECIFIED", ".01"),
                            CostMetadata("d", "v", 0, clock), dataset_id=DATASET, as_of_at=T0)


@pytest.mark.parametrize("status,vector,expected", [(1, None, "limit_no_incumbent"), (2, None, "infeasible"),
                                                   (3, None, "unbounded"), (4, None, "solver_error"),
                                                   (0, [.5, 0, 0], "invalid_binary_solution"),
                                                   (0, [float("nan"), 0, 0], "invalid_binary_solution"),
                                                   (0, [1, 0], "invalid_binary_solution")])
def test_mock_invalid_or_missing_solver_plan(monkeypatch, status, vector, expected):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, status=status, x=vector))
    result = run()
    item = result.results["milp"]
    assert item.summary["status"] == expected
    assert not item.summary["has_valid_solution"] and not item.summary["accepted_for_planning"]
    assert item.summary["objective_value"] is None and item.decisions.selected.isna().all()
    assert result.comparison["relative_improvement"] is None
    assert result.results["greedy"].summary["objective_value"] == 160


@pytest.mark.parametrize("overrides,expected", [({"status": 1, "success": False, "mip_gap": .2}, "limit_feasible"),
                                              ({"mip_gap": .1}, "gap_feasible"),
                                              ({"mip_dual_bound": None}, "feasible_uncertified"),
                                              ({"fun": -999}, "feasible_uncertified")])
def test_mock_unproven_incumbents_retained_but_not_automatically_accepted(monkeypatch, overrides, expected):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, **overrides))
    default = run().results["milp"].summary
    assert default["status"] == expected and default["has_valid_solution"]
    assert not default["accepted_for_planning"] and default["optimality"] == "not_proven"
    assert run(config=SolverConfig(allow_unproven=True)).results["milp"].summary["accepted_for_planning"]


def test_mock_solver_exception_and_missing_capability_preserve_greedy(monkeypatch):
    def fail(*args): raise RuntimeError("controlled exception")
    monkeypatch.setattr(engine, "_solve_milp", fail)
    assert run().results["milp"].summary["status"] == "solver_error"
    monkeypatch.undo()
    monkeypatch.delattr(engine.optimize, "milp")
    result = run()
    assert "SciPy >= 1.9" in result.results["milp"].summary["solver_message"]
    assert result.results["greedy"].summary["has_valid_solution"]


def test_mock_decimal_postcheck_invalidates_overspend_without_repair(monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, x=np.ones(2), fun=-2., mip_dual_bound=-2.))
    result = run(*population([1, 1], [".050000025"] * 2), amount=".1", places=9)
    item = result.results["milp"]
    assert item.summary["status"] == "invalid_budget_solution"
    assert not item.summary["has_valid_solution"] and item.summary["total_expected_cost"] is None
    assert "0.100000050" in item.summary["solver_message"]
    assert item.decisions.selected.isna().all()
    assert result.results["greedy"].summary["selected_customer_count"] == 1


def test_mock_binary_tolerance_is_explicit_and_followed_by_cost_check(monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, x=[1 - 1e-8, 0, 0]))
    assert run().results["milp"].summary["has_valid_solution"]
    assert not run(config=SolverConfig(binary_tolerance=0)).results["milp"].summary["has_valid_solution"]


def test_future_columns_not_forwarded_and_model_periods_preserved():
    segmented, costs = population()
    segmented.customers["future_churn_label"] = 1
    segmented.customers["experiment_assignment"] = "treatment"
    result = run(segmented, costs)
    assert result.metadata["segmentation"]["models"] == segmented.metadata["models"]
    assert result.metadata["segmentation"]["models"]["uplift"]["horizon_unit"] == "weeks"
    for item in result.results.values():
        assert "future_churn_label" not in item.decisions and "experiment_assignment" not in item.decisions
    segmented.customers.attrs["evaluation_only"] = True
    with pytest.raises(InputValidationError, match="evaluation_only"): run(segmented, costs)


@pytest.mark.parametrize("failure", ["limit", "no_incumbent", "error"])
def test_greedy_completed_and_preserved_before_milp(monkeypatch, failure):
    events, saved = [], {}
    execute = engine._execute

    def observed(prepared, algorithm, config):
        events.append(algorithm)
        result = execute(prepared, algorithm, config)
        saved[algorithm] = deepcopy(result)
        return result

    def controlled(prepared, config):
        assert "greedy" in saved
        assert saved["greedy"].summary["objective_value"] == 160
        if failure == "error":
            raise RuntimeError("controlled solver failure after Greedy")
        return fake_solver(prepared, config, status=1, success=False, mip_gap=.5,
                           **({"x": None} if failure == "no_incumbent" else {}))

    monkeypatch.setattr(engine, "_execute", observed)
    monkeypatch.setattr(engine, "_solve_milp", controlled)
    result = run()
    assert events == ["greedy", "milp"]
    assert list(result.results) == ["milp", "greedy"]
    assert result.metadata["execution_order"] == events
    pd.testing.assert_frame_equal(saved["greedy"].decisions, result.results["greedy"].decisions)
    assert saved["greedy"].summary == result.results["greedy"].summary
    assert not result.results["milp"].summary["accepted_for_planning"]


@pytest.mark.parametrize("presolve", [None, True, False])
def test_public_presolve_option_forwarded_and_recorded(monkeypatch, presolve):
    seen = []

    def capture(**kwargs):
        seen.append(kwargs["options"].copy())
        return SimpleNamespace(status=1, success=False, x=None, message="controlled no incumbent")

    monkeypatch.setattr(engine.optimize, "milp", capture)
    config = SolverConfig() if presolve is None else SolverConfig(presolve=presolve)
    result = run(config=config)
    expected = True if presolve is None else presolve
    assert seen == [{"time_limit": 10., "mip_rel_gap": 0., "presolve": expected}]
    assert result.metadata["solver_config"]["presolve"] is expected
    assert SolverConfig(10., 0., 1e-7, False).presolve is True


@pytest.mark.parametrize("presolve", [None, 0, 1, "False", np.bool_(True)])
def test_presolve_requires_explicit_python_bool(presolve):
    with pytest.raises(InputValidationError, match="presolve"):
        SolverConfig(presolve=presolve)


def test_prepared_arrays_preserve_fixed_preoptimization_contract():
    segmented, costs = population([9, -2, 0, 7, 5], [".3", "0", "0", "0", ".2"],
                                 ids=["c", "n", "z", "a", "b"])
    before, cost_before = deepcopy(segmented), costs.copy(deep=True)
    prepared = engine._prepare(segmented, costs, CostMetadata("d", "v", 1, T0),
                               Budget(".4", "SOURCE_UNSPECIFIED", ".01"), DATASET, T0)
    assert prepared["customer_ids"].tolist() == ["a", "b", "c", "n", "z"]
    np.testing.assert_array_equal(prepared["scores"], [7., 5., 9., -2., 0.])
    assert prepared["ticks"] == [0, 2, 3, 0, 0]
    assert prepared["active"] == [1, 2] and prepared["free"] == [0]
    assert prepared["scaled_budget"] == 4 and prepared["scaled_costs"] == [2, 3]
    assert prepared["objective_scale"] == 9.
    np.testing.assert_array_equal(prepared["objective_coefficients"], [5. / 9., 1.])
    result = run(segmented, costs, amount=".4", places=1)
    assert chosen(result, "greedy") == {"a", "c"}
    assert chosen(result, "milp") == {"a", "c"}
    pd.testing.assert_frame_equal(before.customers, segmented.customers)
    pd.testing.assert_frame_equal(cost_before, costs)
    assert before.metadata == segmented.metadata


@pytest.mark.parametrize("presolve", [True, False])
def test_real_presolve_modes_preserve_decimal_budget_check(presolve):
    result = run(*population([1, 1], [".050000025"] * 2), amount=".1", places=9,
                 config=SolverConfig(presolve=presolve))
    for item in result.results.values():
        assert item.summary["total_expected_cost"] == Decimal(".050000025")
        assert item.summary["budget_compliant"] is True


@pytest.mark.parametrize("allowed_gap,within", [(0., False), (.25, True)])
def test_lower_quality_milp_is_reported_without_greedy_replacement(monkeypatch, allowed_gap, within):
    monkeypatch.setattr(engine, "_solve_milp", lambda p, c: fake_solver(p, c, mip_gap=.2, mip_dual_bound=-.6))
    result = run(config=SolverConfig(mip_rel_gap=allowed_gap))
    milp, greedy = result.results["milp"].summary, result.results["greedy"].summary
    assert milp["objective_value"] == 60 and greedy["objective_value"] == 160
    assert result.comparison["objective_difference"] == -100
    assert result.comparison["relative_improvement"] == -.625
    assert milp["status"] == "gap_feasible" and milp["optimality_gap"] == .2
    assert milp["gap_within_tolerance"] is within
    assert not milp["accepted_for_planning"] and milp["budget_compliant"]
    assert milp["total_expected_cost"] == 10 and greedy["total_expected_cost"] == 30


@pytest.mark.parametrize("algorithm", ["milp", "greedy"])
def test_single_algorithm_execution_remains_independent(algorithm):
    result = run(algorithm=algorithm)
    assert list(result.results) == [algorithm]
    assert result.metadata["execution_order"] == [algorithm]


@pytest.mark.parametrize("presolve", [True, False])
def test_real_three_decimal_costs_have_exact_binary_optimum(monkeypatch, presolve):
    real = engine.optimize.milp
    returned = []
    def observe(**kwargs):
        result = real(**kwargs)
        returned.append(result)
        return result
    monkeypatch.setattr(engine.optimize, "milp", observe)
    result = run(*population([10, 20, 30], [".050000025"] * 3), amount=".1", places=9,
                 config=SolverConfig(presolve=presolve))
    native = returned[0]
    np.testing.assert_array_equal(native.x, [0, 0, 1])
    assert native.status == 0 and native.success and native.mip_gap == 0
    allocation = result.results["milp"]
    assert chosen(result, "milp") == {"00002"}
    assert allocation.summary["objective_value"] == 30
    assert allocation.summary["total_expected_cost"] == Decimal(".050000025")
    assert allocation.summary["status"] == "optimal_numeric"
    assert allocation.summary["accepted_for_planning"] and allocation.summary["budget_compliant"]
    assert result.metadata["integer_gcd"] == 25  # legacy field definition retained
    assert result.metadata["constraint_cost_gcd"] == 50000025
    assert result.metadata["constraint_budget_tick_remainder"] == 49999975


@pytest.mark.parametrize("costs,amount,places,expected_gcd,expected_rhs", [
    ([6, 10], 20, 0, 2, 10), ([6, 10], 21, 0, 2, 10),
    ([3, 5], 10, 0, 1, 10), ([0, 6, 10], 21, 0, 2, 10),
    ([0, 0], 5, 0, None, 1), ([9], 8, 0, None, 1), ([1], 0, 0, None, 0),
    ([".000000001", ".000000003"], ".000000005", 9, 1, 5),
    ([".050000025"] * 3, ".1", 9, 50000025, 1),
])
def test_integer_constraint_reduction_exact_feasible_sets(costs, amount, places, expected_gcd, expected_rhs):
    segmented, table = population(list(range(1, len(costs) + 1)), costs)
    prepared = engine._prepare(segmented, table, CostMetadata("d", "v", places, T0),
                               Budget(amount, "SOURCE_UNSPECIFIED", ".01"), DATASET, T0)
    assert prepared["metadata"]["constraint_cost_gcd"] == expected_gcd
    assert prepared["scaled_budget"] == expected_rhs
    active_costs = [prepared["ticks"][i] for i in prepared["active"]]
    for original, reduced in zip(active_costs, prepared["scaled_costs"]):
        assert original % expected_gcd == 0 and original == reduced * expected_gcd
    for bits in product((0, 1), repeat=len(active_costs)):
        original_cost = sum(c * x for c, x in zip(active_costs, bits))
        reduced_cost = sum(c * x for c, x in zip(prepared["scaled_costs"], bits))
        assert (original_cost <= prepared["budget_ticks"]) == (reduced_cost <= prepared["scaled_budget"])


@pytest.mark.parametrize("seed", range(8))
def test_cost_gcd_floor_equivalence_for_all_random_subsets(seed):
    rng = np.random.default_rng(seed)
    factor = int(rng.integers(2, 1000))
    costs = [int(x) * factor for x in rng.integers(1, 20, 8)]
    amount = int(sum(costs) // 2 + 1)
    segmented, table = population([1] * 8, costs)
    p = engine._prepare(segmented, table, CostMetadata("d", "v", 0, T0),
                        Budget(amount, "SOURCE_UNSPECIFIED", ".01"), DATASET, T0)
    for bits in product((0, 1), repeat=len(p["active"])):
        original = sum(p["ticks"][i] * bit for i, bit in zip(p["active"], bits))
        reduced = sum(c * bit for c, bit in zip(p["scaled_costs"], bits))
        assert (original <= p["budget_ticks"]) == (reduced <= p["scaled_budget"])


def benchmark_worker(size, algorithm, time_limit, presolve=True):
    rng = np.random.default_rng(541)
    scores, costs = rng.integers(1, 1001, size), rng.integers(1, 1001, size)
    start = perf_counter()
    segmented, table = population(scores, costs)
    generation = perf_counter() - start
    amount = int(sum(int(value) for value in costs) * 3 // 10)
    result = run(segmented, table, amount=amount, algorithm=algorithm,
                 config=SolverConfig(time_limit=time_limit, presolve=presolve))
    summary = result.results[algorithm].summary
    return {"size": size, "algorithm": algorithm, "seed": 541, "cost_distribution": "integer uniform [1,1000]",
            "score_distribution": "integer uniform [1,1000]", "budget_fraction": ".30 (integer floor)",
            "time_limit": time_limit, "presolve": presolve, "input_generation_seconds": generation,
            "preparation_seconds": result.metadata["preparation_time"], "total_seconds": result.metadata["total_execution_time"],
            **summary, "budget_compliant": None if not summary["has_valid_solution"] else summary["total_expected_cost"] <= amount}


def benchmark_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--algorithm", choices=["milp", "greedy"])
    parser.add_argument("--time-limit", type=float, default=3.)
    parser.add_argument("--presolve", choices=["true", "false"], default="true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(benchmark_worker(args.worker, args.algorithm, args.time_limit,
                                          presolve=args.presolve == "true"), default=str))
    elif args.benchmark:
        records = []
        for size in (10, 100, 1000, 5000, 20000):
            for algorithm in ("milp", "greedy"):
                process = subprocess.run([sys.executable, "-B", "-m", "final_strategy.tests.test_budget",
                                          "--worker", str(size), "--algorithm", algorithm, "--time-limit", str(args.time_limit),
                                          "--presolve", args.presolve],
                                         capture_output=True, text=True, check=True, timeout=120)
                record = json.loads(process.stdout)
                records.append(record)
                print(json.dumps(record, default=str), flush=True)
        if args.output:
            args.output.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    else:
        parser.error("--benchmark or --worker is required")


if __name__ == "__main__":
    benchmark_main()
