"""Synthetic sensitivity scenarios exercise the existing decision and IPS engines."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, Inexact, ROUND_DOWN, ROUND_UP, getcontext, localcontext
from fractions import Fraction
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget, policy_sensitivity as sensitivity
from final_strategy.policy_sensitivity import SensitivityScenario, run_policy_sensitivity
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.tests.test_coupon_cost_estimation import data, estimate
from final_strategy.tests.test_policy_comparison import scoped
from final_strategy.tests.test_policy_search import _inputs
from final_strategy.tests.test_policy_value import inputs, runs
from final_strategy.tests.test_strategy_profiles import CUTS
from final_strategy.validation import InputValidationError


def options(data, limit=20, **settings):
    shared = _inputs()
    del shared["cost_inputs"], shared["cost_metadata"]
    shared["budget"] = replace(shared["budget"], amount=limit)
    return dict(search_inputs=shared, profile_settings=deepcopy(CUTS), cost_estimate=estimate(data, **settings),
                max_scenarios=10, pareto_directions={"purchase_rate_28d": "max"})


def change(field, value, name=None):
    return SensitivityScenario(name or field, field, value, "synthetic assumption")


def ratio(value):
    return Fraction(value["numerator"], value["denominator"])


def evaluate(opts, *scenarios, **kwargs):
    return run_policy_sensitivity(**(opts | kwargs), scenarios=scenarios)


def test_budget_reference_and_non_nested_optima(data, monkeypatch):
    opts = options(data, 5, purchase_count_30d_threshold=2)
    opts["search_inputs"]["algorithm"] = "milp"
    called = []
    compare = sensitivity.compare_policies

    def record(*args, **kwargs):
        called.append((kwargs["scenario_id"], kwargs["pairs"]))
        return compare(*args, **kwargs)

    monkeypatch.setattr(sensitivity, "compare_policies", record)
    report = evaluate(opts, *(change("budget", k, k) for k in ("budget_050", "budget_100", "budget_200"))).to_dict()
    assert called == [("baseline", []), ("budget_050", []), ("budget_200", [])]
    assert report["comparison_calls"] == 3
    assert report["scenarios"]["budget_100"]["comparison"] == report["scenarios"]["baseline"]["comparison"]
    assert report["scenarios"]["budget_200"]["after"] == "10"
    swaps = report["differences"]["budget_050"]["H0"]["selected"]
    assert swaps["added"] == swaps["removed"] == 1 and ratio(swaps["jaccard"]) == 0


def test_components_and_fixed_log(data, inputs):
    opts = options(data, 100)
    original = opts["cost_estimate"].cost_inputs
    log, evidence = inputs
    policy = next(p for p in build_strategy_profiles(**CUTS).policies if p.policy_id == "H0")
    result = evaluate(opts, change("send_cost", np.float32(.5)), change("redemption_probability", "2"),
                      change("expected_discount_if_redeemed", "2"), change("monetary_components", "2"),
                      experiment_log=log, evidence=evidence, allow_unverified_log=True,
                      baseline_fixations={"H0": scoped(policy, log, evidence)})
    report = result.to_dict()
    json.dumps(report, allow_nan=False)
    # Historical estimator emits 1.67 + .334 * 3 per request after upward rounding.
    for field, expected in (("send_cost", "1.837"), ("redemption_probability", "3.674"),
                            ("expected_discount_if_redeemed", "3.674"), ("monetary_components", "5.344")):
        comparison = report["scenarios"][field]["comparison"]
        chosen = [row for row in comparison["decisions"] if row["representative_id"] == "P0" and row["selected"]]
        assert len(chosen) == 3 and all(Decimal(r["expected_cost"]) == Decimal(expected) for r in chosen)
        assert comparison["audit"]["fixations"] == {}
        diff = report["differences"][field]["H0"]
        assert ratio(diff["selected"]["jaccard"]) == 1
        assert ratio(diff["plan"]["total_expected_cost"]["change"]) != 0
        assert ratio(diff["ips"]["marketing_cost_minor_per_customer"]["change"]) == 0
    assert report["scenarios"]["baseline"]["comparison"]["audit"]["fixations"]["H0"]["binding"] == "scope_matched"
    pd.testing.assert_frame_equal(original, opts["cost_estimate"].cost_inputs)
    opts["profile_settings"]["common"] = .95
    report["original_cost_inputs"][0]["send_cost"] = 0
    assert result.to_dict()["original_cost_inputs"][0]["send_cost"] == "1.67"


def test_profiles_reclassify_gates_keep_members_and_aliases(data):
    report = evaluate(options(data), change("common", np.float32(.3)), change("risk_low", ".4"),
                      change("standard_value", ".4"), change("h4_r", dict(a=".1", b=".1", c=".3", d=".3"))).to_dict()
    assert report["scenarios"]["common"]["after"] == "0.3"
    assert report["differences"]["common"]["H1"]["candidates"]["removed"] == 1
    for label, owners in (("baseline", 6), ("h4_r", 4)):
        record = report["scenarios"][label]["comparison"]
        assert len(record["registration"]["arms"]) == owners
        assert next(p for p in record["plans"] if p["policy_id"] == "H0")["representative_id"] == "P0"


def test_algorithm_holds_and_partial_ips(data, inputs, monkeypatch):
    log, evidence = inputs
    log.loc[0, "outcome_observed"] = False
    log.loc[0, ["factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"]] = pd.NA
    monkeypatch.setattr(budget, "_solve_milp", lambda *args: SimpleNamespace(status=1, success=False, x=None))
    report = evaluate(options(data), change("algorithm", "hybrid", "H"), change("algorithm", "milp", "M"),
                      experiment_log=log, evidence=evidence, allow_unverified_log=True).to_dict()
    diff = report["differences"]["H"]["H0"]
    assert diff["accepted_for_planning"] == [True, False]
    assert diff["ips"]["purchase_rate_28d"]["hold_reason"] == ["censored_outcomes", "unapproved_plan"]
    assert diff["ips"]["send_cost_minor_per_customer"]["estimable"] == [True, False]
    failed = report["differences"]["M"]["H0"]
    assert failed["has_valid_solution"] == [True, False] and failed["selected"]["added"] is None
    assert failed["selected"]["jaccard"] is None and failed["plan"]["selected_customer_count"]["after"] is None


def test_hand_counts_and_empty_sets():
    assert sensitivity._sets({"a", "b"}, {"b", "c"}) == dict(
        intersection=1, added=1, removed=1, jaccard=Fraction(1, 3))
    assert sensitivity._sets(set(), set())["jaccard"] == 1
    assert sensitivity._difference(0, 1)["relative_change"] is None


@pytest.mark.parametrize("field,value", [("send_cost", "-1"), ("redemption_probability", 4),
                         ("send_cost", ".000001"), ("common", False), ("risk_high", ".4"),
                         ("high_value", ".5"), ("h4_r", dict(a=0, b=.4, c=.3, d=.5)),
                         ("churn_threshold", .4), ("algorithm", "both"), ("budget", "custom")])
def test_invalid_changes(data, field, value):
    with pytest.raises(InputValidationError):
        evaluate(options(data), change(field, value))


def test_cap_ids_and_conflicting_cost_inputs(data):
    opts = options(data)
    case = change("algorithm", "milp")
    with pytest.raises(InputValidationError, match="cap"):
        evaluate(opts, case, max_scenarios=1)
    with pytest.raises(InputValidationError, match="duplicate"):
        evaluate(opts, case, case)
    opts["search_inputs"]["cost_inputs"] = opts["cost_estimate"].cost_inputs
    with pytest.raises(InputValidationError, match="only through"):
        evaluate(opts, case)


def test_withheld_and_inexact_budget(data):
    report = evaluate(options(data, min_requests=10), change("send_cost", 2)).to_dict()
    assert report["status"] == "withheld" and report["comparison_calls"] == 0 and not report["scenarios"]
    with pytest.raises(InputValidationError, match="exactly"):
        evaluate(options(data, ".00001"), change("budget", "budget_050"))


def test_zero_baseline_and_bad_producer_charge(data, inputs):
    log, evidence = inputs
    log.loc[3, "send_cost_minor"] = 1  # failed request with a positive charge
    report = evaluate(options(data, 0), change("budget", "budget_200"), experiment_log=log,
                      evidence=evidence, allow_unverified_log=True).to_dict()
    row = report["differences"]["budget"]["H0"]
    assert row["has_valid_solution"] == [True, True]
    assert ratio(row["selected"]["jaccard"]) == 1 and row["plan"]["selected_customer_count"]["before"] == 0
    assert row["plan"]["selected_customer_count"]["relative_change"] is None
    assert row["ips"]["purchase_rate_28d"]["change"] is None
    assert "realized experimental coupon charge" in row["ips"]["purchase_rate_28d"]["hold_reason"][1]


def test_context_independent_remaining_budget_and_ips(data, inputs):
    opts = options(data)
    log, evidence = inputs
    snapshots = []
    for precision, rounding in ((28, ROUND_DOWN), (2, ROUND_UP)):
        with localcontext() as ctx:
            ctx.prec, ctx.rounding, ctx.Emax = precision, rounding, 2
            ctx.traps[Inexact] = True
            original = repr(ctx)
            result = evaluate(opts, change("send_cost", ".5"), experiment_log=log, evidence=evidence,
                              allow_unverified_log=True).to_dict()
            assert getcontext() is ctx and repr(ctx) == original
        ips = {}
        for scenario, spent, remaining in (("baseline", "8.016", "11.984"), ("send_cost", "5.511", "14.489")):
            comparison = result["scenarios"][scenario]["comparison"]
            plan = next(row for row in comparison["plans"] if row["policy_id"] == "H0")
            assert plan["budget_compliant"] and plan["accepted_for_planning"]
            assert Fraction(plan["total_expected_cost"]) == Fraction(spent)
            assert Fraction(plan["remaining_budget"]) == Fraction(remaining)
            ips[scenario] = {row["metric"]: (row["readiness"], row["estimable"], row["estimate"], row["hold_reason"])
                             for row in comparison["ips"] if row["policy_id"] == "H0"}
            assert len(ips[scenario]) == 6
            assert all(status == "point_estimates_available" and valid and value is not None and reason is None
                       for status, valid, value, reason in ips[scenario].values())
        delta = result["differences"]["send_cost"]["H0"]["plan"]["total_expected_cost"]
        assert ratio(delta["change"]) == Fraction("-2.505")
        snapshots.append(ips)
    assert snapshots[0] == snapshots[1]


def test_max_budget_and_ninth_decimal_tick(data):
    data["exposures"].loc[:2, "charged_minor"] = 1000
    opts = options(data, budget.MAX_AMOUNT, decimal_places=9, probability_places=9, discount_places=0)
    with localcontext() as ctx:
        ctx.prec, ctx.Emax = 2, 2
        ctx.traps[Inexact] = True
        report = evaluate(opts).to_dict()
    plan = next(r for r in report["scenarios"]["baseline"]["comparison"]["plans"] if r["policy_id"] == "H0")
    spent = 3 * Fraction("1001.000000002")
    assert Fraction(plan["total_expected_cost"]) == spent
    assert Fraction(plan["remaining_budget"]) == Fraction(10**18) - spent


@pytest.mark.parametrize("scenarios", [
    [change("algorithm", "both")], [change("budget", "custom")], [change("common", Decimal("NaN"))],
    [change("send_cost", Decimal("Infinity"))], [change("send_cost", -1)], [change("send_cost", True)],
    [change("risk_high", ".4")], [change("standard_value", ".01")],
    [change("h4_r", dict(a=0, b=".5", c=".3", d=".6"))], [change("common", {"risk_low": ".1"})],
    [change("churn_threshold", ".3")], [change("algorithm", "milp"), change("budget", "budget_050", "algorithm")],
])
def test_withheld_rejects_invalid_declarations_before_comparison(data, monkeypatch, scenarios):
    monkeypatch.setattr(sensitivity, "compare_policies", lambda *a, **kw: pytest.fail("unexpected comparison"))
    opts = options(data, min_requests=10)
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 2, ROUND_DOWN
        ctx.traps[Inexact] = True
        original = repr(ctx)
        with pytest.raises(InputValidationError):
            evaluate(opts, *scenarios)
        assert getcontext() is ctx and repr(ctx) == original


def test_withheld_defers_only_data_dependent_checks(data, monkeypatch):
    monkeypatch.setattr(sensitivity, "compare_policies", lambda *a, **kw: pytest.fail("unexpected comparison"))
    opts = options(data, min_requests=10)
    result = evaluate(opts, change("common", ".3"), change("algorithm", "hybrid"),
                      change("redemption_probability", 4), change("budget", "budget_050")).to_dict()
    assert result["reason"] == "cost_estimate_withheld" and result["comparison_calls"] == 0
