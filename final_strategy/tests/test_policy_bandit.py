"""Phase D contract tests; every outcome and policy comparison here is synthetic."""
from dataclasses import replace
from fractions import Fraction
import json
import math
import random

import pytest

from final_strategy.policy_bandit import (
    EvaluationUnit, METHODS, Observation, PURCHASE, SYNTHETIC, compare_synthetic_search,
    run_policy_bandit, search_fixed_log,
)
from final_strategy.policy_candidates import CandidateSet
from final_strategy.tests.test_policy_candidates import generate
from final_strategy.tests.test_policy_model import analyze, case
from final_strategy.tests.test_policy_search import BASE, _inputs, _policy
from final_strategy.tests.test_policy_value import inputs, runs


P0 = replace(BASE, policy_id="P0")
OTHER = _policy("other", lambda r: replace(r, allow_candidate=False))
CANDIDATES = CandidateSet((OTHER, P0), {})


def unit(i):
    return EvaluationUnit(f"u{i}", f"s{i}", f"{i:064x}", "demo", i)


def observe(policy, req):
    i = req.unit.seed
    return Observation(req, "observed", f"run{i}", f"out{i}", f"{i:064x}",
                       reward=.75 if policy.policy_id == "P0" else .25)


def search(fn=observe, **options):
    settings = dict(method="uniform", evaluation_units=map(unit, range(20)), evaluator=fn,
                    run_id="test", seed=7, max_attempts=6, environment="synthetic",
                    conditions={}, reward_definition=SYNTHETIC)
    return run_policy_bandit(options.pop("candidates", CANDIDATES), **(settings | options)).to_dict()


def test_actual_phase_c_offline_path(case, inputs):
    candidates = generate(analyze(case), max_candidates=3)
    log, evidence = inputs
    args = dict(search_inputs=_inputs(), experiment_log=log, evidence=evidence, allow_unverified_log=True,
                method="exhaustive", max_evaluations=3, seed=1)
    result = search_fixed_log(candidates, **args).to_dict()
    assert result["coverage_complete"] and result["evaluated_count"] == 3
    assert result["policy_run_calls"] == result["value_calls"] == result["comparison_calls"] == 3
    assert 0 <= result["comparison_seconds"] <= result["elapsed_seconds"]
    assert result["order"][0] == "P0"
    metric = result["results"]["P0"]["result"]["metrics"]["purchase_rate_28d"]
    assert metric["estimate"] == {"numerator": 3, "denominator": 2}
    assert result["comparisons_to_p0"]["P0"]["metrics"]["purchase_rate_28d"]["difference"]["numerator"] == 0
    assert result["independent_observations"] == 0
    json.dumps(result, allow_nan=False)
    args.update(method="random", max_evaluations=2)
    partial = search_fixed_log(candidates, **args).report
    assert not partial["coverage_complete"] and len(set(partial["order"])) == 2
    assert partial["order"] == search_fixed_log(candidates, **args).report["order"]
    assert search(lambda p, req: result)["arm_statistics"]["P0"]["observations"] == 0


def test_offline_hold(inputs):
    log, evidence = inputs
    result = search_fixed_log(CANDIDATES, method="exhaustive", max_evaluations=1, seed=2,
                              experiment_log=log, evidence=evidence, search_inputs=_inputs()).report
    assert result["results"]["P0"]["status"] == "held"


def test_arm_freeze_aliases_and_order():
    alias = replace(P0, policy_id="alias", description="alternate derivation")
    data = {"policies": {P0.configuration_fingerprint(): {"evidence": ["discovery"]}}}
    candidates = CandidateSet((OTHER, alias, P0), data)

    def callback(p, req):
        data["policies"].clear()
        return observe(p, req)

    result = search(callback, candidates=candidates)
    entry = result["arms"][0]
    assert entry["policy_id"] == "P0" and entry["evidence"]["evidence"] == ["discovery"]
    assert [p["policy_id"] for p in entry["aliases"]] == ["P0", "alias"]
    assert len(entry["aliases"][0]["policies"]) == 12
    assert len(result["arm_statistics"]) == 2


@pytest.mark.parametrize("policies", [(), (P0, replace(P0, version="changed")),
                                      (P0, replace(OTHER, policy_id="P0"))])
def test_registration_rejects_conflicts(policies):
    with pytest.raises(ValueError):
        search(candidates=CandidateSet(policies, {}))


def test_uniform_and_single_arm():
    result = search(max_attempts=5)
    assert [r["policy_id"] for r in result["attempts"]] == ["P0", "other", "P0", "other", "P0"]
    assert result["arm_statistics"]["P0"]["cumulative_reward"] == 2.25
    assert result["recommendation"] == "P0"
    assert search(candidates=CandidateSet((OTHER,), {}), method="ucb1")["recommendation"] == "other"


@pytest.mark.parametrize("method,eps", [("random", None), ("epsilon_greedy", 0), ("epsilon_greedy", 1)])
def test_sequential_draws(method, eps):
    result = search(method=method, epsilon=eps)
    keys = ["P0", "other"]
    rng = random.Random(7)
    expected = []
    for _ in range(4):
        explore = method == "random" or rng.random() < eps
        expected.append(rng.choice(keys) if explore else "P0")
    assert [r["policy_id"] for r in result["attempts"]] == keys + expected


def test_ucb_after_failed_initialization_attempt():
    def callback(p, req):
        if req.unit.seed == 0:
            return Observation(req, "external_failure")
        return observe(p, req)
    result = search(callback, method="ucb1")
    fourth = result["attempts"][3]
    assert fourth["selection_scores"]["other"] == pytest.approx(.25 + math.sqrt(2 * math.log(2)))
    assert fourth["selection_scores"]["P0"] > 1
    assert result["recommendation_reason"] == "incomplete_evidence"
    assert result["arm_statistics"]["P0"]["failed"] == 1


def test_ucb_selection_differs_from_recommendation():
    result = search(method="ucb1", max_attempts=5)
    assert result["attempts"][-1]["policy_id"] == "other"
    assert result["recommendation"] == "P0"


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), -.01, 1.01, True, None])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        search(method="epsilon_greedy", epsilon=eps)


def test_initialization_limits_and_insufficient_evidence():
    with pytest.raises(ValueError):
        search(method="ucb1", max_attempts=1)
    result = search(lambda p, r: Observation(r, "pending"), method="ucb1", max_retries=0)
    assert result["stop_reason"] == "initialization_failed" and result["recommendation"] is None
    result = search(evaluation_units=[unit(0)], min_observations=2)
    assert result["stop_reason"] == "units_exhausted" and result["recommendation"] is None


@pytest.mark.parametrize("change", [
    lambda r: replace(r, attempt_id="replay"), lambda r: replace(r, policy_id="other-policy"),
    lambda r: replace(r, version="v2"), lambda r: replace(r, configuration_fingerprint="wrong"),
    lambda r: replace(r, conditions={"budget": 999}), lambda r: replace(r, reward_definition=PURCHASE),
])
def test_receipt_binding(change):
    result = search(lambda p, r: replace(observe(p, r), request=change(r)))
    assert all(a["status"] == "contract_violation" for a in result["attempts"])


@pytest.mark.parametrize("name", ["producer_run_id", "outcome_id", "ledger_sha256"])
def test_replayed_observation(name):
    value = "a" * 64
    result = search(lambda p, r: replace(observe(p, r), **{name: value}), max_attempts=2)
    assert [a["status"] for a in result["attempts"]] == ["observed", "contract_violation"]


@pytest.mark.parametrize("second", [unit(0), replace(unit(1), state_id="s0"),
                                    replace(unit(1), roster_fingerprint=unit(0).roster_fingerprint),
                                    replace(unit(1), role="final")])
def test_reject_reused_or_reserved_unit(second):
    result = search(evaluation_units=[unit(0), second])
    assert result["attempts"][1]["status"] == "contract_violation"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2, True])
def test_invalid_bounded_observation(value):
    result = search(lambda p, r: replace(observe(p, r), reward=value))
    assert all(a["status"] == "contract_violation" for a in result["attempts"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("status", ["solver_failure", "invalid_plan", "unapproved_plan",
                                    "contract_violation", "external_failure", "pending", "censored"])
def test_no_implicit_zero(status):
    result = search(lambda p, r: Observation(r, status))
    assert all(a["reward"] is None for a in result["attempts"])
    assert all(s["mean"] is None for s in result["arm_statistics"].values())


def purchase(change=None):
    def callback(p, req):
        obs = replace(observe(p, req), valid_plan=True, approved_plan=True,
                      receipt="execution", decision_fingerprint="f" * 64, assigned=8,
                      observed=8, censored=0, purchasing=2, reward=.25, horizon_days=28)
        return replace(obs, **(change or {}))
    return search(callback, environment="synthetic", reward_definition=PURCHASE,
                  conditions=dict(unit_size=8, solver="greedy", budget_per_customer="0.1",
                                  population="declared", coupon_terms="10%", cost_definition="v1"))


def test_purchase_counts_tie_and_provenance():
    result = purchase()
    assert result["recommendation"] == "P0" and result["independence"] == "unverified"
    assert result["attempts"][0]["evidence"]["purchasing"] == 2
    assert result["dispatch_authorized"] is False
    assert result["attempts"][0]["reward"] == .25


@pytest.mark.parametrize("fields,status", [
    ({"approved_plan": False}, "unapproved_plan"), ({"valid_plan": False}, "invalid_plan"),
    ({"observed": 7, "censored": 1}, "contract_violation"), ({"reward": 1}, "contract_violation"),
    ({"purchasing": 9}, "contract_violation"), ({"assigned": 7}, "contract_violation"),
    ({"receipt": None}, "contract_violation"), ({"horizon_days": 30}, "contract_violation"),
])
def test_purchase_contract(fields, status):
    result = purchase(fields)
    assert result["attempts"][0]["status"] == status and result["recommendation"] is None


def test_synthetic_comparison_and_regret():
    means = {"P0": 0., "other": 1.}
    result = compare_synthetic_search(CANDIDATES, true_means=means, seeds=[0, 1, 2],
                                      max_attempts=5, epsilon=.3, unit_size=4).to_dict()
    assert len(result["runs"]) == 12
    for run in result["runs"]:
        assert len(run["attempts"]) == 5
        assert run["simple_regret"] == 0 and run["recommendation"] == "other"
        assert run["cumulative_expected_regret"] == sum(a["policy_id"] == "P0" for a in run["attempts"])
        assert "true_means" not in run and run["cumulative_reward"] == run["arm_statistics"]["other"]["attempts"]
    assert result["recommendation_counts"] == {m: {"other": 3} for m in METHODS}
    json.dumps(result, allow_nan=False)


def test_unverified_independent_holds_repackaged_ips_before_dispatch(inputs, runs):
    from final_strategy.policy_search import run_policy_search
    from final_strategy.policy_value import evaluate_policy_value

    hybrid = run_policy_search(candidates=[P0], **(_inputs() | {"algorithm": "hybrid"})).runs["P0"]
    assert hybrid.summary["has_valid_solution"] and hybrid.summary["budget_compliant"]
    assert not hybrid.summary["accepted_for_planning"]
    log, evidence = inputs
    ips = evaluate_policy_value(runs["none"], log, evidence=evidence, allow_unverified_log=True)
    value = ips.metrics["purchase_rate_28d"]["estimate"]
    assert value == .5
    calls = []

    def repackaged(p, req):
        calls.append(req.unit.unit_id)
        return replace(observe(p, req), decision_fingerprint=hybrid.summary["decision_fingerprint"],
                       receipt="declared", valid_plan=True, approved_plan=True, reward=float(value),
                       assigned=4, observed=4, censored=0, purchasing=int(4 * value), horizon_days=28)

    result = search(repackaged, environment="independent", reward_definition=PURCHASE, max_attempts=2,
                    conditions=dict(unit_size=4, solver="hybrid", budget_per_customer=".1",
                                    population="fixture", coupon_terms="10%", cost_definition="v1"))
    assert calls == []
    assert result["stop_reason"] == result["recommendation_reason"] == "producer_verification_unavailable"
    assert result["attempt_count"] == result["observation_count"] == result["cumulative_reward"] == 0
    assert result["recommendation"] is None and not result["dispatch_authorized"]
    assert all(v["mean"] is None for v in result["arm_statistics"].values())


@pytest.mark.parametrize("bad", ["reward", "auxiliary", "late_json", "version", "pending"])
def test_only_valid_rewards_register_identifiers(bad):
    def callback(p, req):
        obs = replace(observe(p, req), producer_run_id="shared", outcome_id="shared", ledger_sha256="a"*64)
        if req.unit.seed:
            return obs
        if bad == "version":
            return replace(obs, request=replace(req, version="changed"))
        if bad == "pending":
            return replace(obs, status="pending", reward=None)
        if bad == "late_json":
            return replace(obs, reward=Fraction(1, 2))
        if bad == "auxiliary":
            return replace(obs, auxiliary={"x": math.inf})
        return replace(obs, reward=2)
    result = search(callback, evaluation_units=[unit(0), unit(0), unit(1), unit(2)], max_retries=4)
    statuses = [r["status"] for r in result["attempts"]]
    assert statuses == ["pending" if bad == "pending" else "contract_violation", "contract_violation",
                        "observed", "contract_violation"]
    assert result["attempt_count"] == 4 and result["observation_count"] == 1
    assert result["cumulative_reward"] == .75 and result["recommendation"] is None


@pytest.mark.parametrize("purchasing,status", [(999, "contract_violation"), (True, "contract_violation"),
                                              (-1, "contract_violation"), (1, "censored"), (None, "censored")])
def test_partial_purchases_validated(purchasing, status):
    r = purchase(dict(status="censored", observed=1, censored=7, purchasing=purchasing, reward=None))
    assert r["attempts"][0]["status"] == status and r["observation_count"] == 0
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("fields", [dict(assigned=1, observed=0, censored=1), dict(purchasing=0)])
def test_bounded_has_no_purchase_fields(fields):
    r = search(lambda p, req: replace(observe(p, req), **fields))
    assert r["attempts"][0]["status"] == "contract_violation" and r["recommendation"] is None


@pytest.mark.parametrize("fields", [dict(assigned=True), dict(observed=False), dict(censored=False),
                                   dict(observed=1, censored=2)])
def test_invalid_roster_counts(fields):
    assert purchase(fields)["attempts"][0]["status"] == "contract_violation"


@pytest.mark.parametrize("n", [0, 8])
def test_complete_purchase_endpoints(n):
    r = purchase(dict(purchasing=n, reward=n / 8))
    assert r["observation_count"] == 6 and r["attempts"][0]["reward"] == n / 8


@pytest.mark.parametrize("method", ["uniform", "epsilon_greedy"])
def test_exact_decimal_tie(method):
    r = search(lambda p, req: replace(observe(p, req), reward=.7), method=method, max_attempts=5,
               epsilon=0 if method == "epsilon_greedy" else None)
    assert all(s["mean"] == .7 for s in r["arm_statistics"].values())
    assert r["arm_statistics"]["P0"]["observations"] == (3 if method == "uniform" else 4)
    assert r["recommendation"] == "P0" and r["cumulative_reward"] == 3.5
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("extra,winner", [(0, "P0"), (1, "other")])
def test_purchase_ranking_preserves_integer_precision(extra, winner):
    size = 10**18
    def callback(p, req):
        # P0 observes one quarter then three quarters. Other observes one half.
        count = (size//4 if req.unit.seed == 0 else 3*size//4) if p.policy_id == "P0" else size//2 + extra
        return replace(observe(p, req), reward=None, assigned=size, observed=size, censored=0,
                       purchasing=count, horizon_days=28)
    r = search(callback, max_attempts=4, reward_definition=PURCHASE, conditions={"unit_size": size})
    assert r["arm_statistics"]["other"]["mean"] == r["arm_statistics"]["P0"]["mean"] == .5
    assert r["recommendation"] == winner
