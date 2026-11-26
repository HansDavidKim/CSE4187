import copy

import numpy as np
import pandas as pd
import pytest

from simulate.v4.simulator.coupon_assignment import (
    assign_coupons, resolve_coupon_assignment, validate_coupon_assignments)


def test_assignment_preserves_auto_and_randomizes_fixed_counts_without_behavior_rng():
    ids = [f"u{i}" for i in range(5)]
    business = np.random.default_rng(19)
    before = copy.deepcopy(business.bit_generator.state)
    actions, propensity, metadata = assign_coupons(ids, {"mode":"assign","ratio":.5,"seed":7})
    assert actions.sum() == 2 and np.array_equal(propensity, np.full(5,.4))
    assert metadata["randomized"] and metadata["ratio_rounding"] == "floor(n*ratio)"
    assert business.bit_generator.state == before
    reverse, _, _ = assign_coupons(ids[::-1], {"mode":"assign","count":2,"seed":7})
    assert dict(zip(ids,actions)) == dict(zip(ids[::-1],reverse))
    legacy = [1,0,0,1]
    auto, probability, design = assign_coupons(ids[:4], legacy=legacy)
    np.testing.assert_array_equal(auto,legacy)
    np.testing.assert_array_equal(probability,[.5]*4)
    assert design["mode"] == "auto"


@pytest.mark.parametrize("ratio,total", [(0,0),(1,5),(.2,1)])
def test_zero_one_and_fractional_ratios_are_replayable(ratio,total):
    ids = [f"u{i}" for i in range(5)]
    actions, probability, design = assign_coupons(ids,{"mode":"assign","ratio":ratio},seed=17)
    assert actions.sum() == total
    frame = pd.DataFrame({"customer_id":ids,"treatment":actions,"assignment_probability":probability})
    validate_coupon_assignments(frame,{"seed":17,"coupon_assignment":design})
    bad = frame.copy();bad.loc[0,"assignment_probability"] = .123
    with pytest.raises(ValueError,match="probabilities"):
        validate_coupon_assignments(bad,{"seed":17,"coupon_assignment":design})


def test_none_and_explicit_ids_are_deterministic_and_allow_one_customer():
    none, p, metadata = assign_coupons(["u"],{"mode":"none"})
    assert none.tolist() == [0] and p.tolist() == [0] and not metadata["randomized"]
    action, p, metadata = assign_coupons(["u","v"],{"mode":"assign","customer_ids":["v"]})
    assert action.tolist() == p.tolist() == [0,1]
    assert metadata["design"] == "specified_customers" and not metadata["randomized"]
    assert resolve_coupon_assignment({"count":1})["mode"]=="assign"
    array,_,_=assign_coupons(pd.Series(["u","v"]),{"mode":"assign","customer_ids":np.array(["v"])})
    np.testing.assert_array_equal(array,action)
    with pytest.raises(ValueError,match="unknown"):
        assign_coupons(["u"],{"mode":"assign","customer_ids":["v"]})
    with pytest.raises(ValueError,match="exceeds"):
        assign_coupons(["u"],{"mode":"assign","count":2})


def test_decimal_ratio_does_not_lose_a_customer_to_binary_rounding():
    actions, probability, _ = assign_coupons([str(i) for i in range(100)], {"ratio": .29})
    assert actions.sum() == 29
    assert np.all(probability == .29)


@pytest.mark.parametrize("spec", [
    {"mode":"unknown"}, {"mode":[]}, {"mode":"none","ratio":0}, {"mode":"auto","seed":1},
    {"mode":"assign"}, {"mode":"assign","count":1,"ratio":.5},
    {"mode":"assign","count":True}, {"mode":"assign","count":-1},
    {"mode":"assign","ratio":True}, {"mode":"assign","ratio":np.nan},
    {"mode":"assign","ratio":np.inf}, {"mode":"assign","ratio":1.1},
    {"mode":"assign","ratio":-.1}, {"mode":"assign","count":1,"seed":True},
    {"mode":"assign","customer_ids":"u"}, {"mode":"assign","customer_ids":["u","u"]},
    {"mode":"assign","customer_ids":[None]}, {"mode":"assign","customer_ids":[np.nan]},
    {"mode":"assign","ratio":.5,"future":1},
])
def test_assignment_rejects_ambiguous_or_invalid_specifications(spec):
    with pytest.raises(ValueError):resolve_coupon_assignment(spec)
