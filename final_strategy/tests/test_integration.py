"""Public-API integration with declared dummy costs, no publisher or real models."""
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass, replace
from decimal import Decimal
from itertools import product
import json
from math import fsum

import pandas as pd
import pytest

from simulate.v4.features import EXPANDED_FEATURE_VERSION
from final_strategy import budget as engine
from final_strategy.budget import Budget, CostMetadata, SolverConfig, optimize_budget
from final_strategy.inputs import PredictionBatch, integrate_inputs
from final_strategy.retention_strategy import (
    COUPON_COLUMNS, REDEMPTION_COLUMNS, HistoryContext, HistoryPolicy,
    build_retention_strategy, default_segment_policies,
)
from final_strategy.segmentation import SegmentationConfig, export_segments, segment_customers
from final_strategy.tests.test_budget import fake_solver
from final_strategy.tests.test_retention_strategy import coupon, setup as small_retention_inputs
from final_strategy.tests.test_segmentation import T0, eligibility_for, make_inputs
from final_strategy.validation import CONTEXT_COLUMNS, JOIN_KEYS, PREDICTION_COLUMNS, InputValidationError
from final_strategy.what_if import export_what_if, run_what_if


# No RNG/seed: fixed enumeration, including all twelve segment combinations.
EXPECTED_CANDIDATES = frozenset(f"{i:03}" for i in (*range(20, 29), *range(50, 59)))
CLASSIFICATION = ("segment_id", "churn_band", "uplift_band", "clv_band",
                  "is_high_clv", "priority_score", "priority_rank")
EXPECTED_SCENARIOS = {"budget_050": (Decimal(5), 1, Decimal("4.50"), 200),
                      "budget_100": (Decimal(10), 2, Decimal("9.00"), 400),
                      "budget_200": (Decimal(20), 4, Decimal("18.00"), 800)}


def normalized_frames(seed):
    """Reuse the established fixture; recover its declared normalized batches."""
    features = seed.data.drop(columns=list(PREDICTION_COLUMNS.values())).copy()
    predictions = {}
    for model, column in PREDICTION_COLUMNS.items():
        frame = seed.data[[*CONTEXT_COLUMNS, column]].merge(
            seed.prediction_context[model], on=list(JOIN_KEYS), validate="one_to_one")
        predictions[model] = PredictionBatch(frame, seed.prediction_metadata[model])
    return features, predictions


def assert_unchanged(current, original):
    if isinstance(current, pd.DataFrame):
        pd.testing.assert_frame_equal(current, original, check_exact=True)
    elif is_dataclass(current):
        for field in fields(current):
            assert_unchanged(getattr(current, field.name), getattr(original, field.name))
    elif isinstance(current, dict):
        assert current.keys() == original.keys()
        for key in current:
            assert_unchanged(current[key], original[key])
    elif isinstance(current, (tuple, list)):
        assert len(current) == len(original)
        for value, before in zip(current, original):
            assert_unchanged(value, before)
    else:
        assert current == original


@pytest.fixture
def cohort():
    records = [(f"{i:03}", churn, uplift, 1000 if j < 2 else 100)
               for i, (churn, uplift, j) in enumerate(
                   (c, u, j) for c, u in product((.2, .8), (-.1, 0, .2)) for j in range(10))]
    features, predictions = normalized_frames(make_inputs(records))
    original = deepcopy((features, predictions))
    inputs = integrate_inputs(features, predictions)
    assert_unchanged((features, predictions), original)
    analysis_eligibility = eligibility_for(inputs, {"009": "dummy_upstream_exclusion"})
    segmentation_config = SegmentationConfig(churn_threshold=.5, uplift_threshold=0,
                                             high_clv_fraction=.2, segmentation_version="segmentation-12-v1")
    segmentation = segment_customers(inputs, segmentation_config, analysis_eligibility)
    eligibility = segmentation.customers[list(CONTEXT_COLUMNS)].copy()
    eligibility["business_eligible"] = pd.Series(True, index=eligibility.index, dtype=object)
    eligibility["business_reason"] = None
    eligibility["history_complete"] = True
    eligibility.loc[eligibility.customer_id.eq("029"), ["business_eligible", "business_reason"]] = [
        False, "dummy_business_exclusion"]
    eligibility.loc[eligibility.customer_id.eq("059"), "history_complete"] = False
    # Declared expected IDs for this fixture, not a duplicate of policy evaluation.
    cost_inputs = segmentation.customers.loc[
        segmentation.customers.customer_id.isin(EXPECTED_CANDIDATES), list(CONTEXT_COLUMNS)].reset_index(drop=True)
    cost_inputs["send_cost"], cost_inputs["redemption_probability"] = "2", ".25"
    cost_inputs["expected_discount_if_redeemed"] = "10"
    cost_inputs["source_id"], cost_inputs["source_version"], cost_inputs["source_as_of_at"] = "explicit-dummy", "v1", T0
    return dict(features=features, predictions=predictions, inputs=inputs,
                analysis_eligibility=analysis_eligibility, segmentation=segmentation,
                segmentation_config=segmentation_config, eligibility=eligibility, cost_inputs=cost_inputs,
                coupons=pd.DataFrame(columns=COUPON_COLUMNS), redemptions=pd.DataFrame(columns=REDEMPTION_COLUMNS),
                cost_metadata=CostMetadata("expected-send-plus-redemption-28d", "v1", 3, T0),
                history_context=HistoryContext("dummy-segmentation", "a" * 64, T0, "dummy-history", "v1", T0),
                history_policy=HistoryPolicy(False, False, pd.Timedelta(0)),
                policies=default_segment_policies(), budget=Budget("10", "SOURCE_UNSPECIFIED", ".01"),
                solver_config=SolverConfig(time_limit=2, presolve=True))


def retain(cohort, **overrides):
    arguments = {key: cohort[key] for key in ("segmentation", "eligibility", "coupons", "redemptions",
                 "cost_inputs", "cost_metadata", "history_context", "history_policy", "policies")}
    arguments.update(dataset_id="dummy-segmentation", as_of_at=T0, currency="SOURCE_UNSPECIFIED", amount_unit=".01")
    arguments.update(overrides)
    return build_retention_strategy(**arguments)


@pytest.fixture
def pipeline(cohort):
    original = deepcopy(cohort)
    retention = retain(cohort)
    retention_original = deepcopy(retention)
    options = dict(dataset_id="dummy-segmentation", as_of_at=T0, config=cohort["solver_config"])
    allocation = optimize_budget(retention.budget_inputs, retention.costs, cohort["budget"],
                                 retention.cost_metadata, **options)
    scenarios = run_what_if(retention.budget_inputs, retention.costs, cohort["budget"],
                            retention.cost_metadata, **options)
    return cohort, retention, allocation, scenarios, original, retention_original


def check_allocation(allocation, retention, budget):
    decisions, summary = allocation.decisions, allocation.summary
    assert not decisions.duplicated(list(JOIN_KEYS)).any()
    assert set(decisions.customer_id) == set(retention.evaluation.customer_id)
    assert decisions.selected.isin([0, 1]).all()
    selected = decisions.loc[decisions.selected.eq(1)]
    candidates = retention.costs.set_index("customer_id")
    assert set(selected.customer_id) <= set(candidates.index)
    candidate_rows = decisions.loc[decisions.customer_id.isin(candidates.index)]
    assert candidate_rows.expected_cost.eq(candidate_rows.customer_id.map(candidates.expected_cost)).all()
    total = sum((candidates.loc[uid, "expected_cost"] for uid in selected.customer_id), Decimal(0))
    assert total == summary["total_expected_cost"] <= budget
    assert summary["remaining_budget"] == budget - total
    assert summary["selected_customer_count"] == len(selected)
    assert summary["objective_value"] == fsum(selected.priority_score)
    assert summary["has_valid_solution"] and summary["budget_compliant"]
    assert summary["accepted_for_planning"]
    expected_context = retention.evaluation.set_index("customer_id", drop=False)
    for column in CONTEXT_COLUMNS:
        assert decisions[column].eq(decisions.customer_id.map(expected_context[column])).all()
    excluded = retention.budget_inputs.excluded.set_index("customer_id")
    excluded_rows = decisions.loc[decisions.customer_id.isin(excluded.index)]
    assert excluded_rows.selected.eq(0).all() and excluded_rows.expected_cost.isna().all()
    assert excluded_rows.exclusion_reason.eq(excluded_rows.customer_id.map(excluded.exclusion_reason)).all()
    assert decisions.loc[decisions.selected.eq(1), "exclusion_reason"].isna().all()
    if summary["algorithm"] == "milp":
        assert summary["status"] in ("optimal_numeric", "optimal_preprocessed")
        assert summary["optimality"] in ("numeric_optimal", "analytic_optimal")
        if summary["status"] == "optimal_numeric":
            assert summary["optimality_gap"] == 0 and summary["solver_status"] == 0
    else:
        assert summary["status"] == "baseline_feasible" and summary["optimality"] == "not_applicable"


def test_population_classification_and_declared_candidate_costs(pipeline):
    cohort, retention, _, _, _, _ = pipeline
    inputs, segmentation = cohort["inputs"], cohort["segmentation"]
    assert len(inputs.data) == 60 == len(segmentation.customers) + len(segmentation.excluded)
    assert len(segmentation.customers) == 59 and len(segmentation.excluded) == 1
    assert segmentation.customers.segment_id.nunique() == 12
    assert segmentation.summary.customer_count.sum() == 59
    assert segmentation.summary.customer_ratio.sum() == pytest.approx(1)
    assert retention.metadata["candidate_count"] == 18
    assert retention.metadata["policy_excluded_count"] == 41
    assert len(retention.budget_inputs.excluded) == 42
    assert 59 == 18 + 41 and 60 == 18 + 42
    actual_ids = set(retention.evaluation.loc[retention.evaluation.budget_candidate.eq(True), "customer_id"])
    assert actual_ids == EXPECTED_CANDIDATES == set(retention.costs.customer_id)
    assert actual_ids == set(retention.budget_inputs.customers.customer_id)
    source = segmentation.customers.set_index("customer_id")
    adapted = retention.budget_inputs.customers.set_index("customer_id")
    pd.testing.assert_frame_equal(adapted[list(CLASSIFICATION)], source.loc[adapted.index, list(CLASSIFICATION)])
    assert set(adapted.index[adapted.is_high_clv]) == set(source.index[source.is_high_clv]) & actual_ids
    assert source.is_high_clv.sum() == 12 and adapted.is_high_clv.sum() == 4
    assert retention.budget_inputs.summary.customer_count.sum() == 18
    assert retention.budget_inputs.summary.population_count.eq(18).all()
    assert segmentation.summary.population_count.eq(59).all()
    assert set(retention.cost_estimates.expected_cost) == {Decimal("4.50")}
    assert set(retention.cost_estimates.send_cost) == {Decimal(2)}
    assert set(retention.cost_estimates.redemption_probability) == {Decimal(".25")}
    assert set(retention.cost_estimates.expected_discount_if_redeemed) == {Decimal(10)}
    assert inputs.data.mean_net_order_value_30d.isna().all()
    assert adapted.mean_net_order_value_30d.isna().all()
    evaluated = retention.evaluation.set_index("customer_id")
    assert evaluated.loc["009", "exclusion_reason"] == "dummy_upstream_exclusion"
    assert evaluated.loc["029", "exclusion_reason"] == "dummy_business_exclusion"
    assert pd.isna(evaluated.loc["059", "eligible_for_coupon"])
    nonpositive = source.index[source.uplift_score.le(0)]
    assert not evaluated.loc[nonpositive, "policy_candidate"].any()
    assert source.loc[source.uplift_score.lt(0), "priority_score"].lt(0).all()
    assert "sent_at" not in evaluated and not retention.metadata["actual_dispatch_performed"]


def test_real_milp_greedy_and_three_independent_budget_scenarios(pipeline):
    _, retention, allocation, scenarios, _, _ = pipeline
    assert allocation.metadata["execution_order"] == ["greedy", "milp"]
    assert len(scenarios.summary) == 6 and len(scenarios.decisions) == 360
    for result in allocation.results.values():
        check_allocation(result, retention, Decimal(10))
        assert result.summary["objective_value"] == 400
    for identifier, (budget, count, cost, objective) in EXPECTED_SCENARIOS.items():
        result = scenarios.results[identifier]
        assert result.metadata["candidate_count"] == 18 and result.metadata["excluded_count"] == 42
        for allocation in result.results.values():
            check_allocation(allocation, retention, budget)
            assert allocation.summary["selected_customer_count"] == count
            assert allocation.summary["total_expected_cost"] == cost
            assert allocation.summary["objective_value"] == objective
        assert result.comparison["objective_difference"] == 0


def test_policy_filter_does_not_recompute_high_clv_membership(cohort):
    removed = {"020", "021", "050"}  # explicit fixture exclusions among original high-CLV candidates
    eligibility = cohort["eligibility"].copy(deep=True)
    eligibility.loc[eligibility.customer_id.isin(removed), ["business_eligible", "business_reason"]] = [
        False, "dummy_high_clv_exclusion"]
    costs = cohort["cost_inputs"].loc[~cohort["cost_inputs"].customer_id.isin(removed)].copy()
    retention = retain(cohort, eligibility=eligibility, cost_inputs=costs)
    candidates = retention.budget_inputs.customers.set_index("customer_id")
    source = cohort["segmentation"].customers.set_index("customer_id")
    assert len(candidates) == 15 and candidates.is_high_clv.sum() == 1
    assert set(candidates.index[candidates.is_high_clv]) == {"051"}
    pd.testing.assert_frame_equal(candidates[list(CLASSIFICATION)], source.loc[candidates.index, list(CLASSIFICATION)])
    assert cohort["segmentation"].customers.is_high_clv.sum() == 12


def test_what_if_passes_original_candidates_and_costs_for_every_call(pipeline, monkeypatch):
    cohort, retention, _, scenarios, _, _ = pipeline
    calls, real = [], engine.optimize_budget
    def observe(segmentation, costs, budget, cost_metadata, **kwargs):
        assert segmentation is retention.budget_inputs and costs is retention.costs
        assert cost_metadata is retention.cost_metadata and kwargs["config"] is cohort["solver_config"]
        calls.append(budget.amount)
        return real(segmentation, costs, budget, cost_metadata, **kwargs)
    monkeypatch.setattr(engine, "optimize_budget", observe)
    reverse = run_what_if(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                          dataset_id="dummy-segmentation", as_of_at=T0, config=cohort["solver_config"],
                          algorithm="greedy", scenarios=["budget_200", "budget_100", "budget_050"])
    assert calls == [20, 10, 5]
    for identifier in EXPECTED_SCENARIOS:
        pd.testing.assert_frame_equal(reverse.results[identifier].results["greedy"].decisions,
                                      scenarios.results[identifier].results["greedy"].decisions)


def test_traceability_is_kept_in_the_original_stage_objects(pipeline):
    cohort, retention, allocation, scenarios, _, _ = pipeline
    source = cohort["segmentation"]
    assert source.metadata["config"] == asdict(cohort["segmentation_config"])
    assert {row["expanded_feature_set_version"] for row in source.metadata["input_snapshots"]} == {EXPANDED_FEATURE_VERSION}
    assert {row["raw_database_sha256"] for row in source.metadata["input_snapshots"]} == {"a" * 64}
    assert {row["source_model_sha256"] for row in source.metadata["input_snapshots"]} == {"b" * 64}
    assert source.metadata["models"] == allocation.metadata["segmentation"]["models"]
    assert source.metadata["prediction_context"] == scenarios.metadata["source_segmentation"]["prediction_context"]
    assert [source.metadata["models"][model]["horizon_unit"] for model in ("churn", "uplift", "clv")] == [
        "days", "weeks", "weeks"]
    assert len({item["target_definition_id"] for item in source.metadata["models"].values()}) == 3
    assert retention.metadata["policies"] == [asdict(p) for p in cohort["policies"]]
    assert retention.metadata["history_context"] == asdict(cohort["history_context"])
    assert retention.metadata["history_policy"] == asdict(cohort["history_policy"])
    assert retention.metadata["cost"] == asdict(retention.cost_metadata)
    assert retention.budget_inputs.metadata["retention_source_metadata"] == source.metadata
    assert set(retention.cost_estimates.source_id) == {"explicit-dummy"}
    assert set(retention.cost_estimates.source_version) == {"v1"}
    assert retention.cost_estimates.source_as_of_at.eq(T0).all()
    assert allocation.metadata["cost"]["decimal_places"] == 3
    assert allocation.metadata["cost"]["definition_id"] == retention.cost_metadata.definition_id
    assert scenarios.metadata["source_segmentation"]["retention_policy_version"] == "retention-adapter-v1"
    assert "policies" not in scenarios.metadata["source_segmentation"]
    assert "history_context" not in allocation.metadata["segmentation"]
    assert "source_id" not in retention.costs
    assert set(retention.budget_inputs.excluded.exclusion_stage) == {"upstream", "retention"}
    assert "exclusion_stage" not in allocation.results["greedy"].decisions
    assert "priority_rank" not in allocation.results["greedy"].decisions
    assert set(scenarios.metadata["algorithm_versions"]) == {"milp", "greedy"}


def test_all_pipeline_inputs_and_original_metadata_are_unchanged(pipeline):
    cohort, retention, _, _, original, retention_original = pipeline
    assert_unchanged(cohort, original)
    assert_unchanged(retention, retention_original)


@pytest.mark.parametrize("change,match", [
    ("missing", "missing or additional"), ("extra", "missing or additional"), ("duplicate", "duplicate"),
    ("dataset", "missing or additional"), ("snapshot", "snapshot_id mismatch"),
    ("time", "missing or additional"), ("currency", "currency mismatch"),
    ("unit", "amount_unit mismatch"), ("precision", "cannot represent exactly"),
    ("raw_sha", "raw_database_sha256 mismatch"), ("future_source", "future cost knowledge"),
    ("missing_source", "source_id")])
def test_candidate_cost_contract_errors_are_not_imputed_or_ignored(cohort, change, match):
    table = cohort["cost_inputs"].copy(deep=True)
    if change == "missing": table = table.iloc[1:].copy()
    elif change == "extra":
        extra = table.iloc[:1].copy()
        context = cohort["segmentation"].customers.set_index("customer_id").loc["010"]
        extra["customer_id"] = "010"
        for column in set(CONTEXT_COLUMNS) - {"customer_id"}: extra[column] = context[column]
        table = pd.concat([table, extra], ignore_index=True)
    elif change == "duplicate": table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    else:
        column, value = {"dataset": ("dataset_id", "other"), "snapshot": ("snapshot_id", "other"),
                         "time": ("as_of_at", T0 + pd.Timedelta(hours=1)), "currency": ("currency", "KRW"),
                         "unit": ("amount_unit", "1"), "precision": ("send_cost", "2.0001"),
                         "raw_sha": ("raw_database_sha256", "c" * 64),
                         "future_source": ("source_as_of_at", T0 + pd.Timedelta(seconds=1)),
                         "missing_source": ("source_id", "")}[change]
        table.loc[0, column] = value
    original = deepcopy((cohort, table))
    with pytest.raises(InputValidationError, match=match): retain(cohort, cost_inputs=table)
    assert_unchanged((cohort, table), original)


@pytest.mark.parametrize("column,value,match", [
    ("dataset_id", "other", "dataset_id"), ("raw_database_sha256", "c" * 64, "raw_database_sha256"),
    ("as_of_at", T0 + pd.Timedelta(hours=1), "coverage mismatch")])
def test_mixed_input_contexts_fail_before_segmentation(cohort, column, value, match):
    features = cohort["features"].copy(deep=True)
    features.loc[0, column] = value
    with pytest.raises(InputValidationError, match=match): integrate_inputs(features, cohort["predictions"])


def test_same_week_different_snapshots_are_distinct_and_retention_requires_one_cohort(cohort):
    second = T0 + pd.Timedelta(hours=1)
    features, predictions = normalized_frames(make_inputs([("001", .8, .2, 100, T0), ("001", .8, .3, 100, second)]))
    inputs = integrate_inputs(features, predictions)
    segmentation = segment_customers(inputs)
    assert len(inputs.data) == 2 and inputs.data.reference_week.nunique() == 1
    assert inputs.data.snapshot_id.nunique() == 2 and inputs.data.as_of_at.nunique() == 2
    assert len(segmentation.metadata["cohorts"]) == 2 and segmentation.summary.customer_count.sum() == 2
    with pytest.raises(InputValidationError, match="mixed dataset, snapshot or currency"):
        retain(cohort, segmentation=segmentation)


@pytest.mark.parametrize("state,candidate", [
    ("active", False), ("expired", True), ("redeemed", True), ("future_redemption", False), ("future_send", True)])
def test_coupon_time_boundaries_affect_candidates_before_budget(cohort, state, candidate):
    sent = T0 - pd.Timedelta(days=28) if state == "expired" else T0 if state == "future_send" else T0 - pd.Timedelta(days=2)
    coupons = coupon(sent)
    coupons["customer_id"] = "020"
    redemptions = cohort["redemptions"]
    if state in ("redeemed", "future_redemption"):
        at = T0 - pd.Timedelta(days=1) if state == "redeemed" else T0 + pd.Timedelta(days=1)
        redemptions = pd.DataFrame([dict(coupon_id="C1", customer_id="020", purchased_at=at)])
    cost_inputs = cohort["cost_inputs"]
    if not candidate: cost_inputs = cost_inputs.loc[cost_inputs.customer_id.ne("020")].copy()
    retention = retain(cohort, coupons=coupons, redemptions=redemptions, cost_inputs=cost_inputs,
                       history_policy=HistoryPolicy(True, False, pd.Timedelta(0)))
    row = retention.evaluation.set_index("customer_id").loc["020"]
    assert bool(row.budget_candidate) == candidate
    assert row.observed_used_coupon_count == (1 if state == "redeemed" else 0)
    if not candidate: assert row.exclusion_reason == "active_unused_coupon"
    allocation = optimize_budget(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                                 dataset_id="dummy-segmentation", as_of_at=T0, algorithm="greedy")
    check_allocation(allocation.results["greedy"], retention, Decimal(10))
    assert ("020" in set(retention.costs.customer_id)) == candidate


@pytest.mark.parametrize("case", ["unknown_end", "short_coverage", "incomplete_error"])
def test_unconfirmed_history_is_not_assumed_complete(cohort, case):
    options = {}
    if case == "incomplete_error":
        options["history_policy"] = replace(cohort["history_policy"], incomplete_history="error")
    else:
        options["history_context"] = replace(cohort["history_context"], observed_until_at=(
            None if case == "unknown_end" else T0 - pd.Timedelta(seconds=1)))
    with pytest.raises(InputValidationError, match="history"):
        retain(cohort, **options)


def test_no_candidates_preserves_every_exclusion_without_zero_cost_imputation(cohort):
    eligibility = cohort["eligibility"].copy(deep=True)
    eligibility["business_eligible"], eligibility["business_reason"] = False, "dummy_no_candidates"
    retention = retain(cohort, eligibility=eligibility, cost_inputs=cohort["cost_inputs"].iloc[:0].copy())
    assert retention.costs.empty and retention.cost_estimates.empty and retention.budget_inputs.customers.empty
    assert len(retention.budget_inputs.excluded) == 60
    scenarios = run_what_if(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                            dataset_id="dummy-segmentation", as_of_at=T0)
    assert scenarios.summary.selected_customer_count.eq(0).all()
    assert scenarios.summary.objective_value.eq(0).all()
    assert scenarios.decisions.expected_cost.isna().all()
    for scenario in scenarios.results.values():
        for result in scenario.results.values(): check_allocation(result, retention, result.summary["budget"])


def test_zero_budget_keeps_candidates_and_exclusions_separate(cohort):
    retention = retain(cohort)
    scenarios = run_what_if(retention.budget_inputs, retention.costs, replace(cohort["budget"], amount="0"),
                            retention.cost_metadata, dataset_id="dummy-segmentation", as_of_at=T0)
    assert scenarios.summary.selected_customer_count.eq(0).all()
    assert scenarios.summary.budget_utilization.isna().all()
    for scenario in scenarios.results.values():
        for result in scenario.results.values(): check_allocation(result, retention, Decimal(0))


@pytest.mark.parametrize("status", ["limit_no_incumbent", "solver_error", "limit_feasible"])
def test_controlled_milp_failure_or_unproven_solution_preserves_greedy(cohort, monkeypatch, tmp_path, status):
    retention = retain(cohort)
    def controlled(prepared, config):
        if status == "solver_error": raise RuntimeError("controlled integration failure")
        if status == "limit_no_incumbent": return fake_solver(prepared, config, status=1, success=False, x=None)
        return fake_solver(prepared, config, status=1, success=False, mip_gap=.5)
    monkeypatch.setattr(engine, "_solve_milp", controlled)
    allocation = optimize_budget(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                                 dataset_id="dummy-segmentation", as_of_at=T0)
    scenarios = run_what_if(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                            dataset_id="dummy-segmentation", as_of_at=T0)
    for comparison in (allocation, *scenarios.results.values()):
        assert comparison.metadata["execution_order"] == ["greedy", "milp"]
        milp = comparison.results["milp"]
        assert milp.summary["status"] == status and not milp.summary["accepted_for_planning"]
        assert milp.summary["optimality"] == "not_proven"
        candidates = milp.decisions.loc[milp.decisions.customer_id.isin(EXPECTED_CANDIDATES)]
        assert milp.summary["has_valid_solution"] == (status == "limit_feasible")
        if status != "limit_feasible":
            assert candidates.selected.isna().all() and milp.summary["objective_value"] is None
            assert comparison.comparison["objective_difference"] is None
        else:
            assert candidates.selected.sum() == 1 and milp.summary["optimality_gap"] == .5
            total = sum(milp.decisions.loc[milp.decisions.selected.eq(1), "expected_cost"], Decimal(0))
            assert total == milp.summary["total_expected_cost"] <= milp.summary["budget"]
        check_allocation(comparison.results["greedy"], retention, comparison.results["greedy"].summary["budget"])
    if status == "limit_feasible": assert allocation.comparison["objective_difference"] < 0
    paths = export_what_if(scenarios, tmp_path)
    reloaded = pd.read_csv(paths["decisions"], dtype=str, keep_default_na=False)
    if status != "limit_feasible":
        missing = reloaded.loc[reloaded.algorithm.eq("milp") & reloaded.customer_id.isin(EXPECTED_CANDIDATES)]
        assert missing.selected.eq("").all()  # unavailable incumbent is NULL, never fabricated zero
        assert reloaded.loc[reloaded.algorithm.eq("greedy"), "selected"].isin(["0", "1"]).all()


@pytest.mark.parametrize("presolve", [True, False])
def test_small_public_pipeline_matches_exhaustive_decimal_optimum(cohort, presolve):
    bundle = small_retention_inputs([("001", .8, .2, 50), ("002", .2, .2, 100), ("003", .8, .2, 150)])
    retention = retain(cohort, **dict(zip(("segmentation", "eligibility", "coupons", "redemptions", "cost_inputs", "cost_metadata"), bundle)))
    candidates = retention.budget_inputs.customers.merge(retention.costs[list(JOIN_KEYS) + ["expected_cost"]],
                                                         on=list(JOIN_KEYS), validate="one_to_one")
    allocation = optimize_budget(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                                 dataset_id="dummy-segmentation", as_of_at=T0, config=SolverConfig(presolve=presolve))
    optimum = max(fsum(row.priority_score * bit for row, bit in zip(candidates.itertuples(), bits))
                  for bits in product((0, 1), repeat=len(candidates))
                  if sum((row.expected_cost * bit for row, bit in zip(candidates.itertuples(), bits)), Decimal(0)) <= 10)
    assert allocation.results["milp"].summary["objective_value"] == optimum == 50
    check_allocation(allocation.results["milp"], retention, Decimal(10))


def test_public_exports_reload_decimal_utc_exclusions_and_metadata(pipeline, tmp_path):
    cohort, retention, _, scenarios, _, _ = pipeline
    before = deepcopy((cohort, retention, scenarios))
    paths = export_what_if(scenarios, tmp_path / "what-if")
    summary = pd.read_csv(paths["summary"], dtype=str, keep_default_na=False)
    decisions = pd.read_csv(paths["decisions"], dtype=str, keep_default_na=False)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert len(summary) == 6 and len(decisions) == 360 and "001" in set(decisions.customer_id)
    assert pd.to_datetime(decisions.as_of_at, utc=True).eq(T0).all()
    selected = pd.array(decisions.selected.map(int), dtype="Int64")
    assert set(selected) == {0, 1}
    for identifier, (budget, count, cost, objective) in EXPECTED_SCENARIOS.items():
        rows = summary.loc[summary.scenario_id.eq(identifier)]
        assert rows.budget.map(Decimal).eq(budget).all()
        assert rows.total_expected_cost.map(Decimal).eq(cost).all()
        assert rows.selected_customer_count.map(int).eq(count).all()
        assert rows.objective_value.map(float).eq(objective).all()
        for algorithm in ("milp", "greedy"):
            chosen = decisions.loc[decisions.scenario_id.eq(identifier) & decisions.algorithm.eq(algorithm) & decisions.selected.eq("1")]
            assert sum(chosen.expected_cost.map(Decimal), Decimal(0)) == cost
    assert set(decisions.loc[decisions.customer_id.isin(EXPECTED_CANDIDATES), "expected_cost"]) == {"4.50"}
    assert decisions.loc[decisions.customer_id.eq("009"), "expected_cost"].eq("").all()
    assert decisions.loc[decisions.customer_id.eq("029"), "exclusion_reason"].eq("dummy_business_exclusion").all()
    assert metadata["source_segmentation"]["models"] == cohort["segmentation"].metadata["models"]
    assert metadata["scenario_budgets"]["budget_050"]["budget"] == "5.0"
    assert metadata["scenario_metadata"]["budget_100"]["cost"]["decimal_places"] == 3
    segment_paths = export_segments(cohort["segmentation"], tmp_path / "segments_6plus.csv")
    customers = pd.read_csv(segment_paths["customers"], dtype=str, keep_default_na=False)
    assert len(customers) == 59 and customers.segment_id.nunique() == 12
    expected_ranks = cohort["segmentation"].customers.set_index("customer_id").priority_rank
    assert customers.priority_rank.map(int).eq(customers.customer_id.map(expected_ranks)).all()
    assert_unchanged((cohort, retention, scenarios), before)


def test_export_keeps_numeric_looking_and_na_customer_ids(cohort, tmp_path):
    bundle = small_retention_inputs([("001", .8, .2, 100), ("NA", .2, 0, 200)])
    retention = retain(cohort, **dict(zip(("segmentation", "eligibility", "coupons", "redemptions", "cost_inputs", "cost_metadata"), bundle)))
    result = run_what_if(retention.budget_inputs, retention.costs, cohort["budget"], retention.cost_metadata,
                         dataset_id="dummy-segmentation", as_of_at=T0)
    paths = export_what_if(result, tmp_path)
    rows = pd.read_csv(paths["decisions"], dtype=str, keep_default_na=False)
    assert set(rows.customer_id) == {"001", "NA"}
    assert rows.loc[rows.customer_id.eq("NA"), "selected"].eq("0").all()
    assert rows.loc[rows.customer_id.eq("NA"), "expected_cost"].eq("").all()
    assert rows.loc[rows.customer_id.eq("001"), "expected_cost"].map(Decimal).eq(Decimal("4.50")).all()


def test_retention_timedelta_export_limit_is_explicit_and_does_not_mutate_results(pipeline, tmp_path):
    _, retention, _, scenarios, _, _ = pipeline
    before = deepcopy((retention, scenarios))
    interval = retention.metadata["history_policy"]["minimum_send_interval"]
    assert interval == pd.Timedelta(0)
    assert all(policy["send_delay"] == pd.Timedelta(0) for policy in retention.metadata["policies"])
    # Isolated negative probe, never merge this metadata into the actual result.
    unsupported = replace(scenarios, metadata=deepcopy(retention.metadata))
    output = tmp_path / "unsupported-retention"
    with pytest.raises(TypeError, match="unsupported metadata type Timedelta"):
        export_what_if(unsupported, output)
    assert not output.exists()
    assert_unchanged((retention, scenarios), before)
