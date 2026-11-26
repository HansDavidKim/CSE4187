"""Synthetic policy comparisons exercise the production scoring and allocation APIs."""
from dataclasses import replace
from decimal import Decimal
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget as engine
from final_strategy.budget import Budget, CostMetadata, optimize_budget
from final_strategy.policy_search import PolicyCandidate, run_policy_search
from final_strategy.retention_strategy import CONTEXT_COLUMNS, HistoryContext, HistoryPolicy, default_segment_policies
from final_strategy.segmentation import SegmentationConfig
from final_strategy.validation import InputValidationError
from final_strategy.tests.test_retention_strategy import run, setup
from final_strategy.tests.test_segmentation import T0, eligibility_for, make_inputs


# All prices, scores and thresholds here are synthetic.
RECORDS = [("001", .8, .2, 100), ("002", .2, .1, 200),
           ("003", .8, -.1, 100), ("004", .2, .3, 100)]
BASE = PolicyCandidate("base", "v1", "Development baseline", "synthetic fixture")


def _inputs(records=RECORDS, *, upstream=None):
    inputs = make_inputs(records)
    seg, eligible, coupons, redemptions, _, _ = setup(records, upstream=upstream)
    costs = seg.customers[list(CONTEXT_COLUMNS)].copy()
    costs["send_cost"] = "0.050000025"
    costs["redemption_probability"] = ".25"
    costs["expected_discount_if_redeemed"] = ".02"
    costs["source_id"], costs["source_version"], costs["source_as_of_at"] = "fixture", "v1", T0
    return dict(
        inputs=inputs, eligibility=eligible, coupons=coupons, redemptions=redemptions, cost_inputs=costs,
        cost_metadata=CostMetadata("declared-expected-cost", "v1", 9, T0),
        budget=Budget(".110000050", "SOURCE_UNSPECIFIED", ".01"), dataset_id="dummy-segmentation",
        as_of_at=T0, history_context=HistoryContext("dummy-segmentation", "a" * 64, T0, "fixture", "v1", T0),
        history_policy=HistoryPolicy(False, False, pd.Timedelta(0)),
        segmentation_eligibility=eligibility_for(inputs, upstream) if upstream else None,
    )


def _policy(name, change):
    return replace(BASE, policy_id=name, policies=tuple(change(p) for p in default_segment_policies()))


def test_default_policy_matches_public_pipeline_and_decimal_budget():
    kwargs = _inputs()
    result = run_policy_search(candidates=[BASE], **kwargs).runs["base"]
    bundle = setup(RECORDS)
    bundle[4] = kwargs["cost_inputs"].query("customer_id != '003'")
    bundle[5] = kwargs["cost_metadata"]
    retention = run(bundle)
    expected = optimize_budget(retention.budget_inputs, retention.costs, kwargs["budget"],
                               kwargs["cost_metadata"], dataset_id=kwargs["dataset_id"], as_of_at=T0,
                               algorithm="greedy").results["greedy"]
    assert result.decisions.set_index("customer_id").selected.to_dict() == {"001": 1, "002": 0, "003": 0, "004": 1}
    assert result.summary["objective_value"] == expected.summary["objective_value"]
    assert result.summary["total_expected_cost"] == Decimal(".110000050")
    assert result.summary["remaining_budget"] == 0
    assert result.summary["budget_compliant"] is True
    assert result.summary["population_count"] == 4
    assert result.summary["excluded_count"] == 1


@pytest.mark.parametrize(("prefixes", "expected"), [
    ((), {"001", "004"}),
    (("high",), {"001"}),
    (("high_value",), {"002"}),
    (("low", "high_value"), {"002"}),
])
def test_policy_variants_change_candidate_set(prefixes, expected):
    def change(p):
        churn, _, clv = p.segment_id.split("__")
        if not prefixes:
            return replace(p, min_uplift=".15")
        if "high_value" in prefixes:
            allow = clv == "high_value" and ("low" not in prefixes or churn == "low")
        else:
            allow = churn == "high"
        return replace(p, allow_candidate=allow)

    kwargs = _inputs()
    runs = run_policy_search(candidates=[BASE, _policy("alternative", change)], **kwargs).runs
    alternate = runs["alternative"]
    assert set(alternate.decisions.query("budget_candidate").customer_id) == expected
    base_rows = runs["base"].decisions.set_index("customer_id")
    rows = alternate.decisions.set_index("customer_id")
    assert set(rows.index) == set(base_rows.index)
    for key in expected:
        assert rows.loc[key, "expected_cost"] == base_rows.loc[key, "expected_cost"]
    assert alternate.summary["budget"] == runs["base"].summary["budget"]


def test_setting_equality_candidate_changes_and_decision_equality_are_separate():
    equivalent = _policy("same", lambda p: replace(p, min_uplift="0.000"))
    other_churn = replace(BASE, policy_id="relabel", segmentation=SegmentationConfig(churn_threshold=.9))
    fewer = _policy("fewer", lambda p: replace(p, min_uplift=".15"))
    different = _policy("different", lambda p: replace(p, min_uplift=".25"))
    result = run_policy_search(candidates=[BASE, equivalent, other_churn, fewer, different], **_inputs())
    a, b, c, d = result.comparisons[:4]
    assert (a["same_configuration"], a["same_candidates"], a["same_decisions"]) == (True, True, True)
    assert (b["same_configuration"], b["same_candidates"], b["same_decisions"]) == (False, True, True)
    assert (c["same_candidates"], c["same_decisions"]) == (False, True)
    assert d["same_decisions"] is False
    assert d["left_only_selected"] == ["001"]
    assert d["selected_intersection_count"] == 1
    assert d["jaccard_similarity"] == .5


def test_plan_hash_and_tie_break_are_independent_of_input_order():
    a = run_policy_search(candidates=[BASE], **_inputs()).runs["base"]
    b = run_policy_search(candidates=[BASE], **_inputs(list(reversed(RECORDS)))).runs["base"]
    assert a.summary["decision_fingerprint"] == b.summary["decision_fingerprint"]
    assert a.decisions.set_index("customer_id").selected.to_dict() == b.decisions.set_index("customer_id").selected.to_dict()
    assert a.decisions.set_index("customer_id").selected.loc["001"] == 1


def test_same_customer_with_different_send_time_is_a_different_plan():
    later = _policy("later", lambda p: replace(p, send_delay="1 day"))
    result = run_policy_search(candidates=[BASE, later], **_inputs())
    comparison = result.comparisons[0]
    assert comparison["same_decisions"] is False
    assert comparison["jaccard_similarity"] == 1
    planned = result.runs["later"].decisions
    assert planned.loc[planned.selected.eq(1), "planned_sent_at"].eq(T0 + pd.Timedelta(days=1)).all()
    assert planned.loc[planned.selected.eq(0), "planned_sent_at"].isna().all()


@pytest.mark.parametrize(("change", "message"), [
    ("missing", "missing candidate"),
    ("duplicate", "duplicate customer"),
    ("snapshot_id", "snapshot_id mismatch"),
    ("raw_database_sha256", "raw_database_sha256 mismatch"),
    ("source_as_of_at", "future cost"),
    ("as_of_at", "missing or additional"),
    ("future_column", "requires exactly"),
])
def test_bad_common_cost_inputs_rejected(change, message):
    kwargs = _inputs()
    costs = kwargs["cost_inputs"]
    if change == "missing":
        costs = costs.iloc[1:]
    elif change == "duplicate":
        costs = pd.concat([costs, costs.iloc[:1]], ignore_index=True)
    elif change in ("source_as_of_at", "as_of_at"):
        costs.loc[costs.customer_id.eq("003"), change] = T0 + pd.Timedelta(days=1)
    else:
        costs.loc[costs.customer_id.eq("001"), change] = "different"
    with pytest.raises(InputValidationError, match=message):
        run_policy_search(candidates=[BASE], **(kwargs | {"cost_inputs": costs}))


def test_empty_candidates_and_upstream_exclusions_are_preserved():
    disabled = _policy("none", lambda p: replace(p, allow_candidate=False))
    result = run_policy_search(candidates=[disabled], **_inputs(upstream={"003": "fixture exclusion"})).runs["none"]
    assert result.summary["candidate_count"] == result.summary["selected_customer_count"] == 0
    assert result.summary["exclusion_counts"]["policy_disabled"] == 3
    assert len(result.decisions) == 4
    assert pd.notna(result.decisions.set_index("customer_id").exclusion_reason.loc["003"])
    assert result.summary["total_expected_cost"] == 0
    assert result.summary["decision_fingerprint"]


def test_solver_failure_is_preserved_without_fallback(monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda *args: SimpleNamespace(status=1, success=False, x=None))
    no_candidates = _policy("none", lambda p: replace(p, allow_candidate=False))
    result = run_policy_search(candidates=[BASE, no_candidates], algorithm="milp", **_inputs())
    failed = result.runs["base"]
    assert failed.summary["has_valid_solution"] is False
    assert failed.summary["algorithm"] == "milp"
    assert failed.summary["decision_fingerprint"] is None
    assert failed.decisions.loc[failed.decisions.budget_candidate, "selected"].isna().all()
    assert result.runs["none"].summary["has_valid_solution"] is True
    assert result.comparisons[0]["comparable"] is False


def test_serialization_and_caller_inputs_are_independent():
    kwargs = _inputs()
    before = {k: v.copy(deep=True) for k, v in kwargs.items() if isinstance(v, pd.DataFrame)}
    inputs_before = kwargs["inputs"].data.copy(deep=True)
    same = replace(BASE, policy_id="same", segmentation=SegmentationConfig(churn_threshold=np.float32(.5)))
    result = run_policy_search(candidates=[BASE, same], **kwargs)
    encoded = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert Decimal(encoded["runs"]["base"]["summary"]["total_expected_cost"]) == Decimal(".110000050")
    result.runs["base"].decisions.loc[:, "selected"] = 0
    assert result.runs["same"].decisions.selected.sum() == 2
    for key, frame in before.items():
        pd.testing.assert_frame_equal(kwargs[key], frame)
    pd.testing.assert_frame_equal(kwargs["inputs"].data, inputs_before)


def test_duplicate_id_and_invalid_configuration_rejected():
    with pytest.raises(InputValidationError, match="duplicate policy ID"):
        run_policy_search(candidates=[BASE, replace(BASE, version="v2")], **_inputs())
    with pytest.raises(InputValidationError, match="twelve"):
        replace(BASE, policies=BASE.policies[:-1])
    with pytest.raises(InputValidationError, match="uplift threshold"):
        _policy("invalid", lambda p: replace(p, min_uplift="1.01"))
    with pytest.raises(InputValidationError, match="policy_id"):
        replace(BASE, policy_id=" ")


def test_config_fingerprint_normalizes_numbers_without_expanding_small_decimals():
    same = replace(BASE, segmentation=SegmentationConfig(uplift_threshold=0))
    assert same.configuration_fingerprint() == BASE.configuration_fingerprint()
    tiny = _policy("tiny", lambda p: replace(p, min_uplift="1e-100000000"))
    padded = _policy("padded", lambda p: replace(p, min_uplift="10e-100000001"))
    assert tiny.configuration_fingerprint() == padded.configuration_fingerprint()
    assert tiny.configuration_fingerprint() != BASE.configuration_fingerprint()


def test_hybrid_valid_plan_keeps_the_engine_approval_state():
    result = run_policy_search(candidates=[BASE], algorithm="hybrid", **_inputs()).runs["base"]
    assert result.summary["algorithm"] == "hybrid"
    assert result.summary["has_valid_solution"] is True
    assert result.summary["accepted_for_planning"] is False
    assert result.summary["budget_compliant"] is True
    assert result.summary["decision_fingerprint"]


def test_future_result_in_model_inputs_is_rejected():
    kwargs = _inputs()
    kwargs["inputs"].data["future_purchase"] = 1
    with pytest.raises(InputValidationError, match=r"segmentation.data: requires exactly"):
        run_policy_search(candidates=[BASE], **kwargs)
