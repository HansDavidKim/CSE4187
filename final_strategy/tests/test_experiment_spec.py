"""Planning declarations only; these defaults are not business approval."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from final_strategy.experiment_spec import ARMS, PRIMARY, UNRESOLVED, ExperimentSpec
from final_strategy.validation import InputValidationError


def test_contract_and_detached_serialization():
    spec = ExperimentSpec("dummy-study", "v1", (1, 1, 1, 1, 1))
    assert spec.arms == (("A", "No Action"), ("B", "Random"), ("C", "Uplift-only"),
                         ("D1", "Full Strategy + Greedy"), ("D2", "Full Strategy + MILP"))
    assert spec.primary_comparisons == PRIMARY
    assert spec.secondary_comparisons == (("D1", "D2"),)
    assert spec.unresolved_policy_items == UNRESOLVED
    assert spec.coupon_discount == .1 and spec.coupon_validity_days == spec.outcome_days == 28
    data = spec.to_dict()
    assert data["comparison_alpha"] == .025
    assert data["planning_correction"] == "bonferroni" and data["analysis_correction"] == "holm"
    assert data["solver_by_arm"] == {"D1": "greedy", "D2": "milp"}
    assert data["d1_d2_same_policy_score_cost"] and not data["equal_actual_spend_required"]
    assert data["design_status"] == "planning_unapproved"
    data["solver_by_arm"]["D2"] = "greedy"
    assert spec.to_dict()["solver_by_arm"]["D2"] == "milp"
    with pytest.raises(FrozenInstanceError): spec.version = "other"


@pytest.mark.parametrize("field,value", [
    ("experiment_id", ""), ("version", None), ("arms", ARMS[:-1] + (ARMS[0],)),
    ("allocation_weights", (1, 1, 1, 1)), ("allocation_weights", [1]*5),
    ("allocation_weights", (1, 0, 1, 1, 1)), ("allocation_weights", (1, -1, 1, 1, 1)),
    ("allocation_weights", (1, True, 1, 1, 1)), ("allocation_weights", (1, .5, 1, 1, 1)),
    ("allocation_weights", (1, float("nan"), 1, 1, 1)),
    ("primary_comparisons", (("B", "D2"),)),
    ("secondary_comparisons", (("D2", "B"),)),
    ("secondary_comparisons", (("A", "D2"), ("D2", "A"))),
    ("secondary_comparisons", (("X", "A"),)),
    ("coupon_discount", .2), ("coupon_validity_days", 30), ("outcome_days", 30),
    ("family_alpha", 0), ("family_alpha", 1), ("family_alpha", float("inf")),
    ("target_power", .7), ("target_power", 1), ("target_power", True),
    ("primary_metric", "roi"), ("metric_unit", "currency"),
    ("alternative", "one-sided"), ("analysis_correction", "none"),
    ("independent_evaluation_required", False), ("no_action_new_coupon", True),
    ("common_eligibility_required", False), ("common_per_customer_budget_required", False),
    ("d1_d2_same_policy_score_cost", False), ("unresolved_policy_items", ()),
    ("design_status", "approved"),
])
def test_invalid_spec(field, value):
    with pytest.raises(InputValidationError):
        replace(ExperimentSpec("dummy", "v1", (1,)*5), **{field: value})


def test_explicit_secondary_and_unequal_weights():
    spec = ExperimentSpec("dummy", "v1", (1, 2, 2, 1, 3),
                          secondary_comparisons=(("D1", "D2"), ("A", "B")))
    assert spec.allocation_weights == (1, 2, 2, 1, 3)
    assert spec.to_dict()["secondary_plan_status"] == "proposed_exploratory_not_confirmatory"


@pytest.mark.parametrize("scalar", [np.float32, np.float64])
def test_numpy_spec_strict_json_and_immutability(scalar):
    alpha, power = scalar(.05), scalar(.8)
    spec = ExperimentSpec("dummy", "v1", (1,)*5, family_alpha=alpha,
                          target_power=power, coupon_discount=np.float64(.1))
    data = spec.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False))["family_alpha"] == float(alpha)
    for name in ("family_alpha", "target_power", "coupon_discount", "comparison_alpha"):
        assert type(data[name]) is float
    assert type(data["allocation_weights"]) is tuple
    assert type(data["independent_evaluation_required"]) is bool
    assert spec.family_alpha is alpha and spec.target_power is power
    data["solver_by_arm"]["D2"] = "changed"
    assert spec.to_dict()["solver_by_arm"]["D2"] == "milp"
    with pytest.raises(FrozenInstanceError):
        spec.family_alpha = .1


@pytest.mark.parametrize("field,value", [
    ("family_alpha", np.float32("nan")), ("family_alpha", np.float64("inf")),
    ("family_alpha", True), ("target_power", np.bool_(True)),
    ("allocation_weights", (np.int64(1),)*5), ("outcome_days", np.int64(28)),
    ("coupon_discount", np.float32(.1)),
])
def test_numpy_does_not_relax_spec_contract(field, value):
    with pytest.raises(InputValidationError):
        replace(ExperimentSpec("dummy", "v1", (1,)*5), **{field: value})
