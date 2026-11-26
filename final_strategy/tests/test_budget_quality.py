"""Exact certificate tests independent of simulator publishers and new solvers."""
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget as engine
from final_strategy.budget_quality import (prepare_quality_problem, fractional_upper_bound,
    certify_selection, improve_selection, exact_integer_objective)
from final_strategy.tests.test_budget import population, DATASET, T0
from final_strategy.validation import InputValidationError


def problem(scores=(60, 100, 120), costs=(10, 20, 30), amount=50, places=0):
    seg, table = population(scores, costs)
    return prepare_quality_problem(seg, table, engine.Budget(amount, "SOURCE_UNSPECIFIED", "0.01"),
        engine.CostMetadata("explicit", "test", places, T0), dataset_id=DATASET, as_of_at=T0)


@pytest.mark.parametrize("seed", range(20))
def test_fractional_bound_and_certificates_against_every_subset(seed):
    rng = np.random.default_rng(seed)
    costs = rng.integers(0, 11, 8).tolist()
    scores = (rng.integers(-2, 30, 8) / 7).tolist()
    p = problem(scores, costs, 19)
    feasible = []
    for bits in product((0, 1), repeat=8):
        ids = [p.customer_ids[i] for i, bit in enumerate(bits) if bit]
        admissible = all(not bit or i in (*p.active, *p.free) for i, bit in enumerate(bits))
        if admissible and sum(c * bit for c, bit in zip(costs, bits)) <= 19:
            feasible.append(certify_selection(p, ids))
    optimum = max(c.lower_bound for c in feasible)
    upper = fractional_upper_bound(p)
    assert all(c.lower_bound <= optimum <= c.upper_bound == upper for c in feasible)
    for c in feasible:
        for epsilon in ("0.01", "0.001", "0.0001"):
            assert c.meets(epsilon) == (c.lower_bound >= (1 - Fraction(epsilon)) * upper)


@pytest.mark.parametrize("scores,costs,amount,places,expected", [
    ([], [], 0, 0, 0), ([0, -1], [0, 1], 0, 0, 0),
    ([3, 4], [0, 1], 0, 0, 3), ([3, 4], [0, 0], 0, 0, 7),
    ([10, 20, 30], ["0.050000025"] * 3, "0.1", 9, 30),
    ([1, 2], ["0.000000001", "0.000000002"], "0.000000002", 9, 2),
    ([2, 4], [2, 4], 5, 0, 4), ([2, 3], [2, 3], 3, 0, 3),
    ([10, 99], [1, 100], 1, 0, 10), ([1] * 5, [2] * 5, 5, 0, 2),
])
def test_boundary_exact_upper(scores, costs, amount, places, expected):
    assert fractional_upper_bound(problem(scores, costs, amount, places)) == expected


def test_binary64_not_decimal_string_and_no_downward_bound_rounding():
    p = problem([0.1, 0.2, 0.3], [1, 1, 1], 3)
    c = certify_selection(p, p.customer_ids)
    assert c.lower_bound == sum(map(Fraction.from_float, [0.1, 0.2, 0.3]))
    assert c.lower_bound != Fraction("0.6")
    restored = json.loads(json.dumps(c.to_dict()))
    assert Fraction(int(restored["upper_bound"]["numerator"]), int(restored["upper_bound"]["denominator"])) == c.upper_bound
    assert c.optimality_proven


@pytest.mark.parametrize("ids", [["00000", "00001", "00002"], ["unknown"], ["00000", "00000"], [1]])
def test_invalid_selection_rejected(ids):
    with pytest.raises(InputValidationError):
        certify_selection(problem(), ids)


def test_decimal_overbudget_never_certified():
    p = problem([10, 20, 30], ["0.050000025"] * 3, "0.1", 9)
    with pytest.raises(InputValidationError, match="budget exceeded"):
        certify_selection(p, p.customer_ids[:2])
    assert certify_selection(p, [p.customer_ids[2]]).total_expected_cost == Decimal("0.050000025")


@pytest.mark.parametrize("epsilon", [True, 0.01, "nan", "-0.01", "1.01"])
def test_bad_epsilon_rejected(epsilon):
    with pytest.raises(InputValidationError):
        certify_selection(problem(), []).meets(epsilon)


def test_exact_integer_conversion_and_explicit_overflow():
    p = problem([0.1, 0.2, 0.3], [1, 2, 3], 4)
    coefficients, factor = exact_integer_objective(p)
    assert [v * factor for v in coefficients] == list(p.scores)
    with pytest.raises(InputValidationError, match="safe integer range"):
        exact_integer_objective(p, max_integer=100)
    with pytest.raises(InputValidationError, match="safe integer range"):
        exact_integer_objective(problem([0.1, 0.3, 1e7], [1, 2, 3], 4))


def test_local_exchange_reaches_known_counterexample_without_relabeling():
    p = problem()
    baseline = certify_selection(p, p.customer_ids[:2])
    improved, details = improve_selection(p, baseline.selected_customer_ids, time_limit=5)
    assert baseline.lower_bound == 160
    assert improved.lower_bound == 220 and improved.total_expected_cost == 50
    assert details["examined_exchanges"] > 0


def test_best_single_and_free_zero_search_time():
    p = problem([1, 2, 100, 7], [1, 2, 10, 0], 10)
    improved, _ = improve_selection(p, p.customer_ids[:2], time_limit=0)
    assert improved.lower_bound == 107


def test_one_for_two_exchange():
    p = problem([5, 4, 4], [6, 3, 3], 6)
    c, _ = improve_selection(p, [p.customer_ids[0]], time_limit=5)
    assert c.lower_bound == 8


@pytest.mark.parametrize("field,value", [("currency", "USD"), ("raw_database_sha256", "bad"),
    ("snapshot_id", "bad"), ("as_of_at", T0 + pd.Timedelta(days=1)), ("amount_unit", "1")])
def test_existing_contract_checks_reused(field, value):
    seg, costs = population()
    costs.loc[0, field] = value
    with pytest.raises(InputValidationError):
        prepare_quality_problem(seg, costs, engine.Budget(50, "SOURCE_UNSPECIFIED", "0.01"),
            engine.CostMetadata("explicit", "test", 0, T0), dataset_id=DATASET, as_of_at=T0)


def test_nonfinite_score_rejected():
    seg, costs = population()
    seg.customers.loc[0, "priority_score"] = np.inf
    with pytest.raises(InputValidationError):
        prepare_quality_problem(seg, costs, engine.Budget(50, "SOURCE_UNSPECIFIED", "0.01"),
            engine.CostMetadata("explicit", "test", 0, T0), dataset_id=DATASET, as_of_at=T0)


def test_inputs_not_changed_and_row_order_deterministic():
    seg, costs = population()
    old = deepcopy((seg, costs))
    def prepare(s, c):
        return prepare_quality_problem(s, c, engine.Budget(50, "SOURCE_UNSPECIFIED", "0.01"),
            engine.CostMetadata("explicit", "test", 0, T0), dataset_id=DATASET, as_of_at=T0)
    a = prepare(seg, costs)
    pd.testing.assert_frame_equal(seg.customers, old[0].customers)
    pd.testing.assert_frame_equal(costs, old[1])
    assert seg.metadata == old[0].metadata
    seg.customers.sort_index(ascending=False, inplace=True)
    b = prepare(seg, costs.iloc[::-1])
    assert a == b
    assert improve_selection(a, [], time_limit=5)[0] == improve_selection(b, [], time_limit=5)[0]


@pytest.mark.parametrize("limit", [True, 0, 1.5, 2**63])
def test_integer_guard_cannot_be_relaxed_beyond_int64(limit):
    with pytest.raises(InputValidationError):
        exact_integer_objective(problem(), max_integer=limit)


def test_certificate_bound_to_original_metadata_and_budget():
    a = problem(amount=50)
    b = problem(amount=51)
    assert a.input_sha256 != b.input_sha256
    assert json.loads(a.input_metadata_json)["cost"]["version"] == "test"
    assert certify_selection(a, []).input_sha256 == a.input_sha256
