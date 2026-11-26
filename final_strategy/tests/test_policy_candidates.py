"""Rule candidates use the same policy runner, budget and logged actions as P0."""
import importlib.util
import json
from dataclasses import replace

import pytest

from final_strategy.policy_candidates import generate_candidates
from final_strategy.policy_model import ResidualConfig
from final_strategy.policy_search import run_policy_search
from final_strategy.policy_value import compare_policy_values, evaluate_policy_value
from final_strategy.tests.test_policy_model import analyze, case
from final_strategy.tests.test_policy_search import BASE, _inputs
from final_strategy.tests.test_policy_value import inputs, runs


def generate(report, **options):
    return generate_candidates(report, **(dict(version="synthetic-v1", max_candidates=64,
                               max_splits_per_feature=3, max_rule_groups=2) | options))


def test_rule_families_and_evidence(case):
    report = analyze(case, cuts=(.1, .1), churn_cuts=(.8,))
    result = generate(report)
    assert generate(report).to_dict() == result.to_dict()
    assert {p.policy_id.split("-")[0] for p in result.policies} == {"P0", "P1", "P2", "P3", "P4"}
    assert result.policies[0].configuration_fingerprint() == BASE.configuration_fingerprint()
    keys = [p.configuration_fingerprint() for p in result.policies]
    assert len(keys) == len(set(keys))
    assert all(len(p.policies) == 12 and p.segmentation.high_clv_fraction == .2 for p in result.policies)
    for policy in result.policies[1:]:
        family = policy.policy_id.split("-")[0]
        basis = result.report["policies"][policy.configuration_fingerprint()]["evidence"][0]
        for rule in policy.policies:
            risk, _, value = rule.segment_id.split("__")
            applies = basis["risk"] in (None, risk) and basis["clv"] in (None, value)
            assert float(rule.min_uplift) == (basis["min_uplift"] if applies else 0)
        if family == "P1" and basis["min_uplift"] == .1:
            assert len(basis["thresholds"]) == 2
    json.dumps(result.to_dict(), allow_nan=False)


def test_caps_and_support(case):
    report = analyze(case)
    small = generate(report, max_candidates=2, max_splits_per_feature=1)
    assert len(small.policies) == 2
    assert small.report["omitted"]["candidate_limit"] and small.report["omitted"]["per_feature_limit"]
    assert all(p.policy_id.startswith(("P0", "P1")) for p in generate(report, max_rule_groups=1).policies)
    assert len(generate(analyze(case, min_events=15)).policies) == 1
    with pytest.raises(ValueError):
        generate(report, max_candidates=0)
    with pytest.raises(ValueError):
        analyze(case, churn_cuts=(1.1,))


@pytest.mark.parametrize("use_model", [False, pytest.param(True, marks=pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None, reason="optional xgboost"))])
def test_new_cohort_phase_a_b(case, inputs, use_model):
    report = analyze(case, model_config=ResidualConfig() if use_model else None)
    candidates = generate(report)
    if use_model:
        assert any(t["source"] == "xgboost_split" for p in candidates.report["policies"].values()
                   for e in p["evidence"] for t in e["thresholds"])
    options = _inputs()
    log, evidence = inputs
    assert set(log.customer_id).isdisjoint(case[2]["roles"].customer_id)
    planned = run_policy_search(candidates=candidates.policies, **options)
    kwargs = dict(evidence=evidence, allow_unverified_log=True)
    base = planned.runs["P0"]
    reference = evaluate_policy_value(base, log, **kwargs)
    for policy in candidates.policies:
        run = planned.runs[policy.policy_id]
        assert run.summary["budget"] == base.summary["budget"] and run.summary["budget_compliant"]
        if policy.policy_id.startswith("P1"):
            cutoff = float(policy.policies[0].min_uplift)
            rows = run.decisions
            assert not rows.loc[rows.uplift_score.le(cutoff), "budget_candidate"].any()
        result = evaluate_policy_value(run, log, **kwargs)
        delta = compare_policy_values(run, base, log, **kwargs)
        assert result.summary["exploratory"] and not result.summary["dispatch_authorized"]
        for metric, item in result.metrics.items():
            if item["estimable"]:
                assert delta.metrics[metric]["difference"] == item["estimate"] - reference.metrics[metric]["estimate"]


@pytest.mark.parametrize("role", ["selection", "final"])
@pytest.mark.parametrize("field,value", [("predicted_clv_12m", 100000), ("churn_probability", .99),
                                         ("uplift_score", -.9), ("mean_net_order_value_30d", 123.)])
def test_reserved_values_leave_development_unchanged(case, role, field, value):
    inputs, log, opts = case
    changed = inputs.data.copy()
    changed.loc[changed.customer_id.str.startswith(role + "-"), field] = value
    features = ("mean_net_order_value_30d",)
    base = analyze(case, extra_features=features)
    result = analyze((replace(inputs, data=changed), log, opts), extra_features=features)
    assert result == base
    assert result["split"]["development_count"] == 120
    assert generate(result).to_dict() == generate(base).to_dict()


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="optional xgboost")
def test_actual_booster_features_and_reserved_values(case, monkeypatch):
    import xgboost
    feature = "mean_net_order_value_30d"
    names = ["churn_probability", "uplift_score", "predicted_clv_12m", feature]
    for method in ("fit", "predict"):
        original = getattr(xgboost.XGBRegressor, method)
        def checked(self, X, *args, original=original, **kwargs):
            assert list(X.columns) == names
            return original(self, X, *args, **kwargs)
        monkeypatch.setattr(xgboost.XGBRegressor, method, checked)
    inputs, log, opts = case
    changed = inputs.data.copy()
    held = changed.customer_id.str.startswith(("selection-", "final-"))
    changed.loc[held, names] = [.99, -.9, 100000, 123.]
    changed_log = log.copy()
    reserved = changed_log.customer_id.str.startswith(("selection-", "final-"))
    changed_log.loc[reserved, "factual_no_order"] = True
    changed_log.loc[reserved, "factual_outcome_net_revenue_minor"] = 0
    base = analyze(case, model_config=ResidualConfig(), extra_features=iter([feature]))
    result = analyze((replace(inputs, data=changed), changed_log, opts),
                     model_config=ResidualConfig(), extra_features=iter([feature]))
    assert result == base
    assert generate(result).report == generate(base).report


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="optional xgboost")
def test_tree_adds_supported_rules_beyond_quantiles(case):
    baseline = generate(analyze(case, cuts=())).report["policies"]
    trained = generate(analyze(case, cuts=(), model_config=ResidualConfig()))
    additions = trained.report["policies"].keys() - baseline.keys()
    assert additions
    for key in additions:
        policy = next(p for p in trained.policies if p.configuration_fingerprint() == key)
        assert len(policy.policies) == 12 and policy.segmentation.high_clv_fraction == .2
        assert any(t["supported"] and t["source"] == "xgboost_split"
                   and t["feature"] in ("uplift_score", "churn_probability") and 0 <= t["threshold"] <= 1
                   for b in trained.report["policies"][key]["evidence"] for t in b["thresholds"])
    shared = generate(analyze(case, model_config=ResidualConfig())).report["policies"]
    assert any({"declared_bin", "xgboost_split"} <= {t["source"] for t in b["thresholds"]}
               for policy in shared.values() for b in policy["evidence"])
    no_split = analyze(case, cuts=(), model_config=ResidualConfig(min_child_weight=100000))
    assert not no_split["residual_model"]["splits"]
    assert generate(no_split).report["policies"].keys() == baseline.keys()


def test_return_limit_and_search_diagnostics(case):
    result = generate(analyze(case), max_candidates=1)
    assert len(result.policies) == 1
    assert 1 < result.report["support_checks"] <= result.report["combinations_examined"]
    assert any(e["reason"] == "candidate_limit" and e["uplift_sources"] for e in result.report["exclusions"])
