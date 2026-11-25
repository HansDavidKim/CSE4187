"""Human profiles on synthetic cohorts. Numerical cutoffs are test inputs only."""
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from final_strategy import budget as engine
from final_strategy.policy_bandit import run_policy_bandit, search_fixed_log
from final_strategy.policy_candidates import CandidateSet
from final_strategy.policy_search import PolicyCandidate, run_policy_search
from final_strategy.policy_value import compare_policy_values, evaluate_policy_value
from final_strategy.retention_strategy import default_segment_policies
from final_strategy.segmentation import SegmentationConfig
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.tests.test_policy_search import T0, _inputs
from final_strategy.tests.test_policy_value import inputs, runs  # pytest log fixtures
from final_strategy.validation import InputValidationError


CUTS = dict(version="fixture-v1", threshold_source="synthetic", common=".2", risk_high=".1", risk_low=".3",
            high_value=".1", standard_value=".3", h4_r=dict(a=0, b=".1", c=".3", d=".4"))
RECORDS = [
    ("001", .5, .2, 100), ("002", .2, .2, 100), ("003", .8, .2, 100), ("004", .2, .4, 50),
    ("005", .8, -.1, 50), ("006", .2, 0, 50), ("007", .8, .0001, 50), ("008", .2, .5, 0),
    ("009", .8, .3, 50), ("010", .2, .31, 50),
]


@pytest.fixture(scope="module")
def profiles():
    return build_strategy_profiles(**CUTS)


def test_declarations_and_evidence(profiles):
    assert isinstance(profiles, CandidateSet)
    by_id = {p.policy_id: p for p in profiles.policies}
    ids = {p.segment_id for p in default_segment_policies()}
    expected = {
        "P0": [0] * 4, "H0": [0] * 4, "H1": [.2] * 4, "H2": [.1, .1, .3, .3], "H3": [.1, .3, .1, .3],
        "H4-R": [0, .1, .3, .4], "H4-V": [0, .3, .1, .4],
    }
    assert set(expected) == set(by_id)
    groups = ["high__high_value", "high__standard_value", "low__high_value", "low__standard_value"]
    report = json.loads(json.dumps(profiles.to_dict(), allow_nan=False))
    for name, levels in expected.items():
        candidate = by_id[name]
        assert candidate.segmentation == SegmentationConfig()
        assert len(candidate.policies) == 12 and {r.segment_id for r in candidate.policies} == ids
        entry = report["policies"][candidate.configuration_fingerprint()][name]
        assert entry["version"] == CUTS["version"] and entry["hypothesis"] and entry["rationale"]
        assert entry["threshold_source"] == candidate.source
        by_group = {group: Decimal(str(level)) for group, level in zip(groups, levels, strict=True)}
        for rule in candidate.policies:
            risk, _, value = rule.segment_id.split("__")
            group = f"{risk}__{value}"
            assert rule.min_uplift == Decimal(entry["thresholds"][group]) == by_group[group]
            assert rule.allow_candidate and rule.send_delay == pd.Timedelta(0)
    p0 = PolicyCandidate("P0", "other", "Development positive-uplift baseline", "default_segment_policies")
    assert by_id["H0"].policies == p0.policies
    assert by_id["H0"].configuration_fingerprint() == by_id["P0"].configuration_fingerprint() == p0.configuration_fingerprint()


@pytest.mark.parametrize("patch", [
    {"common": True}, {"risk_low": np.bool_(True)}, {"common": float("nan")}, {"common": Decimal("Infinity")},
    {"common": None}, {"common": 0}, {"common": "1.00001"}, {"risk_high": -1},
    {"risk_high": ".3"}, {"risk_low": 0}, {"standard_value": ".1"}, {"high_value": 1},
    {"h4_r": {"a": 0}}, {"h4_r": dict(a=0, b=.3, c=.3, d=1)},
    {"h4_r": dict(a=.2, b=.1, c=.3, d=1)}, {"h4_r": dict(a=0, b=.1, c=.3, d=.2)},
    {"threshold_source": " "}, {"segmentation": SegmentationConfig(uplift_threshold=.1)},
    {"segmentation": SegmentationConfig(churn_threshold=.6)}, {"segmentation": SegmentationConfig(high_clv_fraction=.3)},
])
def test_invalid_declarations(patch):
    with pytest.raises(InputValidationError):
        build_strategy_profiles(**(CUTS | patch))


def test_thresholds_are_required_and_endpoints_are_supported():
    with pytest.raises(TypeError, match="risk_low"):
        build_strategy_profiles(**{k: v for k, v in CUTS.items() if k != "risk_low"})
    profiles = build_strategy_profiles(**(CUTS | dict(common=1, risk_high=0, risk_low=1,
                                                      h4_r=dict(a=0, b=0, c=1, d=1))))
    assert len(profiles.policies) == 7


def test_actions_and_cohort_rank(profiles):
    options = _inputs(RECORDS)
    options["budget"] = replace(options["budget"], amount=10)
    plans = run_policy_search(candidates=profiles.policies, **options).runs
    baseline_ids = {"001", "002", "003", "004", "007", "008", "009", "010"}
    expected = {
        "P0": baseline_ids, "H0": baseline_ids, "H1": {"004", "008", "009", "010"},
        "H2": {"001", "003", "004", "008", "009", "010"}, "H3": {"001", "002", "004", "008", "010"},
        "H4-R": {"001", "003", "008", "009"}, "H4-V": {"001", "002", "008"},
    }
    for name, plan in plans.items():
        rows = plan.decisions.set_index("customer_id")
        assert len(rows) == plan.summary["population_count"] == len(RECORDS)
        assert set(rows.index[rows.budget_candidate]) == expected[name]
        assert set(rows.index[rows.selected.eq(1)]) == expected[name] - {"008"}
        assert rows.loc[rows.budget_candidate, "expected_cost"].eq(Decimal(".055000025")).all()
        assert plan.summary["budget"] == 10 and plan.summary["budget_compliant"]
        assert rows.loc[rows.selected.eq(1), "planned_sent_at"].eq(T0).all()
    cols = ["customer_id", "budget_candidate", "selected", "planned_sent_at"]
    pd.testing.assert_frame_equal(plans["P0"].decisions[cols], plans["H0"].decisions[cols])
    # Same CLV on 001, 002 and 003. Only the first two of the original ten are high-value.
    reversed_plan = run_policy_search(candidates=[profiles.policies[-1]],
                                      **(_inputs(RECORDS[::-1]) | {"budget": options["budget"]})).runs["H4-V"]
    assert reversed_plan.summary["decision_fingerprint"] == plans["H4-V"].summary["decision_fingerprint"]
    assert reversed_plan.decisions.query("budget_candidate").customer_id.tolist() == ["001", "002", "008"]


def test_common_eligibility_history_and_costs(profiles):
    options = _inputs()
    eligibility = options["eligibility"]
    eligibility.loc[eligibility.customer_id.eq("001"), ["business_eligible", "business_reason"]] = [False, "business-exclusion"]
    eligibility.loc[eligibility.customer_id.eq("002"), "history_complete"] = False
    plan = run_policy_search(candidates=profiles.policies, **options)
    for run in plan.runs.values():
        rows = run.decisions.set_index("customer_id")
        assert rows.loc[["001", "002"], "selected"].tolist() == [0, 0]
        assert rows.loc[["001", "002"], "eligibility_reason"].tolist() == ["business-exclusion", "history_incomplete"]
    options["cost_inputs"] = options["cost_inputs"].query("customer_id != '004'")
    with pytest.raises(InputValidationError, match="missing candidate"):
        run_policy_search(candidates=profiles.policies, **options)


def test_restricted_optimum_against_enumeration(profiles):
    options = _inputs(RECORDS)
    plans = run_policy_search(candidates=profiles.policies, algorithm="milp", **options).runs
    for plan in plans.values():
        rows = list(plan.decisions.query("budget_candidate").itertuples())
        best = max(sum(r.priority_score for r in subset)
                   for size in range(len(rows) + 1) for subset in combinations(rows, size)
                   if sum(r.expected_cost for r in subset) <= Decimal(options["budget"].amount))
        assert plan.summary["objective_value"] == pytest.approx(best)
        assert plan.summary["budget_compliant"] and plan.summary["accepted_for_planning"]
        assert best <= plans["H0"].summary["objective_value"]
    h2 = plans["H2"]
    assert h2.summary["selected_customer_count"] < h2.summary["candidate_count"]


def test_invalid_solver_plan_is_preserved(profiles, monkeypatch):
    monkeypatch.setattr(engine, "_solve_milp", lambda *_: SimpleNamespace(status=1, x=None))
    plan = run_policy_search(candidates=[profiles.policies[-1]], algorithm="milp", **_inputs(RECORDS)).runs["H4-V"]
    assert not plan.summary["has_valid_solution"] and not plan.summary["accepted_for_planning"]
    assert plan.decisions.loc[plan.decisions.budget_candidate, "selected"].isna().all()


def test_fixed_log_and_paired_ips(profiles, inputs):
    log, evidence = inputs
    args = dict(evidence=evidence, allow_unverified_log=True)
    result = search_fixed_log(profiles, method="exhaustive", seed=11, max_evaluations=7,
                              search_inputs=_inputs(), experiment_log=log, **args).to_dict()
    assert result["coverage_complete"] and result["target_count"] == result["evaluated_count"] == 6
    assert result["independent_observations"] == 0 and not result["dispatch_authorized"]
    assert [p["policy_id"] for p in result["arms"][0]["aliases"]] == ["P0", "H0"]
    assert set(result["comparisons_to_p0"]) == {"P0", "H1", "H2", "H3", "H4-R", "H4-V"}
    plans = run_policy_search(candidates=profiles.policies, **_inputs()).runs
    diff = compare_policy_values(plans["H2"], plans["H3"], log, **args)
    assert {k: m["difference"] for k, m in diff.metrics.items()} == {
        "purchase_rate_28d": Fraction(1, 2), "no_session_rate_28d": 0, "paid_minor_per_customer": 45,
        "send_cost_minor_per_customer": Fraction(5, 2), "coupon_discount_minor_per_customer": 5,
        "marketing_cost_minor_per_customer": Fraction(15, 2),
    }
    unapproved = run_policy_search(candidates=[profiles.policies[-1]], algorithm="hybrid", **_inputs()).runs["H4-V"]
    assert evaluate_policy_value(unapproved, log, **args).metrics["purchase_rate_28d"]["hold_reason"] == "unapproved_plan"
    log = log.copy()
    log.loc[0, ["factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"]] = pd.NA
    log.loc[0, "outcome_observed"] = False
    value = evaluate_policy_value(plans["H2"], log, **args)
    assert value.metrics["no_session_rate_28d"]["hold_reason"] == "censored_outcomes"
    assert value.metrics["send_cost_minor_per_customer"]["estimable"]


def test_report_reads_and_input_mapping_are_independent():
    levels = CUTS["h4_r"].copy()
    profiles = build_strategy_profiles(**(CUTS | {"h4_r": levels}))
    h4 = profiles.policies[-2]
    key = h4.configuration_fingerprint()
    original = profiles.to_dict()
    levels["b"] = ".99"
    first, second = profiles.report, profiles.report
    for exported in (first, profiles.to_dict()):
        exported["policies"][key]["H4-R"]["thresholds"]["high__standard_value"] = ".99"
        exported["policies"][key]["H4-R"]["threshold_source"] = "changed"
        exported["segmentation"].clear()
        exported["dispatch_authorized"] = True
    assert second == profiles.report == original
    assert json.loads(json.dumps(profiles.to_dict(), allow_nan=False)) == original
    assert next(r for r in h4.policies if r.segment_id == "high__positive__standard_value").min_uplift == Decimal(".1")
    assert h4.configuration_fingerprint() == key


@pytest.mark.parametrize("edge", [False, True])
def test_fixed_log_uses_original_evidence_after_report_edit(edge, inputs):
    options = CUTS | {"h4_r": dict(a=".1", b=".1", c=".3", d=".3")} if edge else CUTS
    profiles = build_strategy_profiles(**options)
    h1 = next(p for p in profiles.policies if p.policy_id == "H1")
    key = h1.configuration_fingerprint()
    profiles.report["policies"][key]["H1"]["thresholds"]["high__high_value"] = "0.99"
    log, evidence = inputs
    shared = _inputs()
    result = search_fixed_log(profiles, method="exhaustive", seed=1, max_evaluations=7,
                              search_inputs=shared, experiment_log=log, evidence=evidence, allow_unverified_log=True).report
    groups = {arm["policy_id"]: arm for arm in result["arms"]}
    assert result["evaluated_count"] == (4 if edge else 6)
    assert [p["policy_id"] for p in groups["P0"]["aliases"]] == ["P0", "H0"]
    if edge:
        for representative, alias in (("H2", "H4-R"), ("H3", "H4-V")):
            assert [p["policy_id"] for p in groups[representative]["aliases"]] == [representative, alias]
    arm = groups["H1"]
    assert arm["configuration_fingerprint"] == key
    assert set(arm["evidence"]["H1"]["thresholds"].values()) == {"0.2"}
    assert all(rule["min_uplift"] == "0.2" for rule in arm["aliases"][0]["policies"])
    plan = run_policy_search(candidates=[h1], **shared).runs["H1"]
    assert plan.decisions.query("selected == 1").customer_id.tolist() == ["004"]
    assert result["results"]["H1"]["status"] == "observed_ips"
    assert result["results"]["H1"]["result"]["summary"]["decision_fingerprint"] == plan.summary["decision_fingerprint"]
    assert not result["dispatch_authorized"]
    json.dumps(result, allow_nan=False)


def test_independent_bandit_stays_held(profiles):
    report = run_policy_bandit(profiles, method="uniform", evaluation_units=(), evaluator=pytest.fail,
                               run_id="fixture", seed=2, max_attempts=7, environment="independent",
                               conditions=dict(population="fixture", unit_size=4, budget_per_customer=1,
                                               cost_definition="declared", coupon_terms="10%", solver="greedy")).report
    assert report["stop_reason"] == "producer_verification_unavailable"
    assert not report["attempts"] and not report["dispatch_authorized"]
