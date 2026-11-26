"""Synthetic fixed-cohort comparisons. No marketing outcomes are inferred from these fixtures."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from types import SimpleNamespace
import json

import pandas as pd
import numpy as np
import pytest

from final_strategy import budget
from final_strategy.coupon_cost_estimation import estimate_coupon_costs
from final_strategy.policy_candidates import CandidateSet
from final_strategy.policy_comparison import H_PAIRS, ScopedPolicyFixation, _pareto, compare_policies, evaluation_fingerprint
from final_strategy.policy_value import METRICS, PolicyFixation
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.tests.test_coupon_cost_estimation import data as cost_ledger
from final_strategy.tests.test_policy_candidates import generate
from final_strategy.tests.test_policy_model import analyze, case
from final_strategy.tests.test_policy_search import BASE, T0, _inputs, _policy
from final_strategy.tests.test_policy_value import inputs, runs  # shared producer log fixtures
from final_strategy.tests.test_strategy_profiles import CUTS
from final_strategy.validation import InputValidationError


def compare(candidates=None, shared=None, **kwargs):
    return compare_policies(candidates if candidates is not None else CandidateSet((BASE,), {}),
                            search_inputs=_inputs() if shared is None else shared, scenario_id="synthetic",
                            pareto_directions={METRICS[0]: "max"}, **kwargs)


def observed(inputs, **kwargs):
    log, evidence = inputs
    return compare(experiment_log=log, evidence=evidence, allow_unverified_log=True, **kwargs)


def scoped(policy, log, evidence):
    return ScopedPolicyFixation(policy.policy_id, policy.version, policy.configuration_fingerprint(),
                                evaluation_fingerprint(log, evidence), PolicyFixation(T0, False))


@pytest.fixture(scope="module")
def profiles():
    return build_strategy_profiles(**CUTS)


def test_profiles_direct_contrasts_and_aliases(profiles, inputs):
    result = observed(inputs, candidates=profiles, include_exploratory=True)
    report = result.to_dict()
    assert report["audit"]["evaluation_calls"] == len(report["registration"]["arms"]) == 6
    assert report["audit"]["paired_reevaluations"] == 14
    plans = result.table("plans").set_index("policy_id")
    assert plans.loc["H0","representative_id"] == "P0"
    assert plans.loc["H0","configuration_fingerprint"] == plans.loc["P0","configuration_fingerprint"]
    assert len(plans) == 7 and plans.loc["H0","decision_fingerprint"] == plans.loc["P0","decision_fingerprint"]
    assert plans.loc["H0","population_count"] == 4 and plans.loc["H0","candidate_count"] == 3
    assert plans.loc["H0","population_request_rate"] == Fraction(1,2)
    assert plans.loc["H0","candidate_selection_rate"] == Fraction(2,3)
    assert plans.loc["H0","budget_utilization"] == 1
    pairs = {(r.left_policy_id,r.right_policy_id):r for r in result.table("comparisons").itertuples()}
    assert set(pairs) == set(H_PAIRS)
    contrast = pairs["H2","H3"]
    assert contrast.ips[METRICS[0]]["difference"] == Fraction(1,2)
    assert contrast.ips[METRICS[0]]["percentage_points"] == 50
    assert contrast.ips[METRICS[-1]]["difference"] == Fraction(15,2)
    assert contrast.jaccard_similarity == 0 and contrast.left_only_selected == ["001"]
    ips = result.table("ips").query("policy_id == 'H0'").set_index("metric")
    assert ips.loc[METRICS[0],"estimate"] == Fraction(3,2)
    assert set(ips.index) == set(METRICS) and ips.effective_sample_size.eq(4).all()
    assert not ips.dispatch_authorized.any()
    assert report["audit"]["causal_interpretation"] == "unverified"
    assert len(result.table("segments")) == 6*12


def test_edge_aliases_and_separate_fixation(inputs):
    profiles = build_strategy_profiles(**(CUTS | {"h4_r":dict(a=".1", b=".1", c=".3", d=".3")}))
    alias = next(p for p in profiles.policies if p.policy_id == "H0")
    result = observed(inputs, candidates=profiles, fixations={"H0": scoped(alias, *inputs)},
                      pairs=[("H0","P0"),("H2","H4-R")], include_exploratory=True)
    assert len(result.to_dict()["registration"]["arms"]) == 4
    assert result.to_dict()["audit"]["evaluation_calls"] == 5
    ips = result.table("ips").query("metric == 'purchase_rate_28d'").set_index("policy_id")
    assert ips.loc["P0","policy_fixation"] == "unverified"
    assert ips.loc["H0","policy_fixation"] == "caller_declared_pre_outcome"
    assert all(x[METRICS[0]]["difference"] == 0 for x in result.table("comparisons").ips)
    assert result.table("comparisons").same_decisions.all()


def test_invalid_ids_and_fixation():
    with pytest.raises(InputValidationError):
        compare(CandidateSet((BASE,replace(BASE,version="other")),{}))
    with pytest.raises(InputValidationError):
        compare(fixations={"typo":PolicyFixation(T0,False)})
    with pytest.raises(InputValidationError):
        compare(pairs=[("base","missing")])
    with pytest.raises(InputValidationError, match="conflicting policy ID"):
        compare(CandidateSet((BASE, _policy(BASE.policy_id, lambda r: replace(r, min_uplift=.75))), {}))


def test_estimator_handoff_with_separate_budget_scenarios(cost_ledger, profiles):
    estimate = estimate_coupon_costs(**cost_ledger)
    shared = dict(_inputs(), cost_inputs=estimate.cost_inputs, cost_metadata=estimate.cost_metadata)
    reports = []
    for scenario_id, limit in (("baseline", 10), ("empty-budget", 0)):
        result = compare_policies(profiles, scenario_id=scenario_id,
                                  search_inputs=dict(shared, budget=replace(shared["budget"], amount=limit)),
                                  pareto_directions={METRICS[0]: "max"})
        reports.append(result)
        assert result.to_dict()["audit"]["scenario_id"] == scenario_id
        decisions = result.table("decisions").query("representative_id == 'P0' and budget_candidate")
        costs = estimate.cost_inputs.set_index("customer_id")
        for row in decisions.itertuples():
            components = costs.loc[row.customer_id]
            assert row.expected_cost == components.send_cost + (
                components.redemption_probability * components.expected_discount_if_redeemed)
        assert result.table("pareto").status.eq("held").all()
    assert reports[0].to_dict()["audit"]["common_input_fingerprint"] != reports[1].to_dict()["audit"]["common_input_fingerprint"]
    assert reports[0].table("plans").selected_customer_count.max() > reports[1].table("plans").selected_customer_count.max() == 0


def test_verified_file_and_corrupt_log_hold(tmp_path, inputs):
    log, evidence = inputs
    path = tmp_path / "producer.parquet"
    log.to_parquet(path)
    def read():
        declared = replace(evidence, log_sha256=sha256(path.read_bytes()).hexdigest())
        return compare(experiment_log=path, evidence=declared, fixations={"base": scoped(BASE, path, declared)})
    verified = read()
    assert set(verified.table("ips").log_checksum_status) == {"verified_against_supplied_manifest"}
    assert verified.table("pareto").iloc[0].status == "non_dominated"
    path.write_bytes(b"not a parquet stream")
    invalid = read()
    assert set(invalid.table("ips").readiness) == {"contract_violation"}
    assert invalid.table("plans").iloc[0].has_valid_solution


def test_input_digest_order_and_values():
    a = _inputs()
    ref = compare(shared=a)
    sig = ref.to_dict()["audit"]["common_input_fingerprint"]
    a["inputs"] = replace(a["inputs"], data=a["inputs"].data.iloc[::-1],
                          prediction_context={k:v.iloc[::-1] for k,v in a["inputs"].prediction_context.items()})
    for k in ("cost_inputs","eligibility"):
        a[k] = a[k].iloc[::-1].copy()
    assert compare(shared=a).to_dict()["audit"]["common_input_fingerprint"] == sig
    a["cost_inputs"].loc[a["cost_inputs"].customer_id.eq("003"),"send_cost"] = ".9"
    changed = compare(shared=a)
    assert changed.table("plans").decision_fingerprint.equals(ref.table("plans").decision_fingerprint)
    assert changed.to_dict()["audit"]["common_input_fingerprint"] != sig


@pytest.mark.parametrize("kind", ["prediction","history","eligibility","budget"])
def test_different_scenario_inputs_affect_digest(kind):
    a = _inputs()
    first = compare(shared=a).to_dict()["audit"]["common_input_fingerprint"]
    if kind == "prediction":
        a["inputs"].data.loc[0,"churn_probability"] = .7
    elif kind == "history":
        a["history_policy"] = replace(a["history_policy"],block_active_unused=True)
    elif kind == "eligibility":
        a["eligibility"].loc[0,["business_eligible","business_reason"]] = [False,"explicit"]
    else:
        a["budget"] = replace(a["budget"],amount=3)
    assert compare(shared=a).to_dict()["audit"]["common_input_fingerprint"] != first


def test_unit_comparison_before_normalization():
    shared = _inputs()
    shared["cost_inputs"]["amount_unit"] = "0.0100000000000000000000000000001"
    with pytest.raises(InputValidationError, match="amount_unit"):
        compare(shared=shared)


def test_empty_candidates_upstream_and_zero_budget():
    a = _inputs(upstream={"003":"excluded"})
    a["budget"] = replace(a["budget"], amount=0)
    result = compare(CandidateSet((_policy("none",lambda r:replace(r,allow_candidate=False)),),{}), a)
    p = result.table("plans").iloc[0]
    assert (p.population_count,p.analyzed_count,p.candidate_count,p.selected_customer_count) == (4,3,0,0)
    assert p.budget_compliant and p.total_expected_cost == 0
    assert p.budget_utilization is None and p.candidate_selection_rate is None
    segments = result.table("segments").set_index("segment_id")
    assert len(segments) == 13 and segments.loc["unclassified","population_count"] == 1
    assert segments.population_count.sum() == 4 and segments.selected_customer_count.sum() == 0
    assert segments.loc[segments.population_count.eq(0),"population_selection_rate"].isna().all()


def test_invalid_solution_is_not_empty(monkeypatch,inputs):
    monkeypatch.setattr(budget,"_solve_milp",lambda *args:SimpleNamespace(status=1, success=False, x=None))
    r = observed(inputs, shared=_inputs() | {"algorithm":"milp"}, pairs=[("base","base")])
    assert r.table("plans").iloc[0].selected_customer_count is None
    assert r.table("segments").selected_customer_count.isna().all()
    assert r.table("comparisons").iloc[0].jaccard_similarity is None
    assert set(r.table("ips").hold_reason) == {"invalid_plan"}
    assert set(r.table("pareto").status) == {"held"}


def test_hybrid_requires_explicit_exploration(inputs):
    shared = _inputs() | {"algorithm":"hybrid"}
    held = observed(inputs, shared=shared)
    assert not held.table("plans").iloc[0].accepted_for_planning
    assert set(held.table("ips").hold_reason) == {"unapproved_plan"}
    explored = observed(inputs, shared=shared, allow_unapproved=True)
    assert explored.table("ips").estimable.all()
    assert explored.table("pareto").iloc[0].reasons == ["exploratory_excluded"]


def test_partial_censor_and_failed_charge(inputs):
    log,e = inputs
    log.loc[0,["factual_no_order","factual_no_session","factual_outcome_net_revenue_minor"]] = pd.NA
    log.loc[0,"outcome_observed"] = False
    r = observed(inputs, pairs=[("base","base")])
    ips = r.table("ips").set_index("metric")
    assert ips.loc[METRICS[0],"hold_reason"] == "censored_outcomes" and ips.loc[METRICS[3],"estimable"]
    assert ips.loc[METRICS[3],"censored_count"] == 1 and ips.loc[METRICS[0],"metric_observed_count"] == 3
    delta = r.table("comparisons").iloc[0].ips
    assert delta[METRICS[0]]["right_hold_reason"] == "censored_outcomes" and delta[METRICS[3]]["difference"] == 0
    log.loc[3,"send_cost_minor"] = 1
    bad = observed((log,e))
    assert set(bad.table("ips").readiness) == {"contract_violation"}
    assert "realized experimental coupon charge" in bad.table("ips").hold_reason.iloc[0]
    assert bad.table("plans").iloc[0].has_valid_solution
    assert set(compare().table("ips").readiness) == {"not_evaluated"}


def test_zero_reference_and_mutation(inputs):
    a = _inputs()
    inputs[0]["send_cost_minor"] = 0
    r = observed(inputs, shared=a, pairs=[("base","base")], include_exploratory=True)
    before = json.dumps(r.to_dict(),allow_nan=False,sort_keys=True)
    assert r.table("comparisons").ips.iloc[0][METRICS[3]]["relative_change"] is None
    assert r.table("plans").total_expected_cost.iloc[0] == Decimal(".110000050")
    inputs[0].loc[0,"send_cost_minor"] = 99
    a["cost_inputs"].loc[0,"send_cost"] = 100
    r.table("comparisons").ips.iloc[0][METRICS[3]]["difference"] = -100
    r.to_dict()["registration"]["arms"].clear()
    assert json.dumps(r.to_dict(),allow_nan=False,sort_keys=True) == before


def test_exact_pareto_ties_conflicts_evidence_and_missing():
    def sample(gain,cost,grade="verified_against_supplied_manifest"):
        s = dict(accepted_for_planning=True,exploratory=True,log_checksum_status=grade)
        return s,{m:dict(estimate=v,estimable=v is not None,hold_reason=None if v is not None else "missing")
                  for m,v in zip(METRICS[:2],(gain,cost))}
    close = 1 + Fraction(1,10**50)
    data = {"base":sample(1,1),"better":sample(close,1),"tie":sample(close,1),"trade":sample(2,2),
            "other":sample(0,10,"unverified"),"missing":sample(None,0)}
    result = {r["policy_id"]:r for r in _pareto(data,dict(zip(METRICS[:2],("max","min"))),True)}
    assert result["base"]["dominated_by"] == ["better","tie"]
    assert result["missing"]["status"] == "held" and result["missing"]["reasons"]
    assert all(result[k]["status"] == "non_dominated" for k in ("better","tie","trade","other"))


def test_execution_amount_fingerprint_at_allocation_boundary():
    amount = np.float32(1000000.0625)
    def run(value, cost):
        s = _inputs()
        s["budget"] = replace(s["budget"], amount=value)
        s["cost_inputs"]["send_cost"] = cost
        s["cost_inputs"]["redemption_probability"] = np.int32(0)
        r = compare(shared=s)
        row = r.table("plans").iloc[0]
        return row.selected_customer_count, row.objective_value, r.to_dict()["audit"]["common_input_fingerprint"]
    with localcontext() as ctx:
        ctx.prec = 3
        a = run(amount, "1000000.061")
        b = run(float(amount), Decimal("1000000.061"))
        assert a[:2] == (0, 0) and b[:2] == (1, 30) and a[2] != b[2]
        assert a == run(Decimal("1000000.060000000"), Decimal("1000000.0610"))
        assert b == run("1000000.062500", "1000000.0610")
        assert run(np.int64(1000001), "1000000.061") == run(1000001, "1000000.061")


@pytest.mark.parametrize("field", ["policy_id", "policy_version", "configuration_fingerprint", "evaluation_fingerprint"])
def test_fixation_scope_mismatch(inputs, field):
    declaration = replace(scoped(BASE, *inputs), **{field: "wrong"})
    with pytest.raises(InputValidationError, match="scope mismatch"):
        observed(inputs, fixations={"base": declaration})


def test_fixation_version_replay_and_legacy_grade(tmp_path, inputs):
    log, evidence = inputs
    path = tmp_path / "verified.parquet"
    log.to_parquet(path)
    evidence = replace(evidence, log_sha256=sha256(path.read_bytes()).hexdigest())
    version2 = CandidateSet((replace(BASE, version="v2"),), {})
    with pytest.raises(InputValidationError, match="scope mismatch"):
        compare(version2, experiment_log=path, evidence=evidence, fixations={"base": scoped(BASE, path, evidence)})
    legacy = compare(version2, experiment_log=path, evidence=evidence, fixations={"base": PolicyFixation(T0, False)})
    assert legacy.table("ips").exploratory.all() and set(legacy.table("ips").policy_fixation) == {"unverified"}
    assert legacy.table("pareto").iloc[0].status == "held"
    assert legacy.to_dict()["audit"]["fixations"]["base"]["binding"] == "scope_unavailable"
    class Misbound(PolicyFixation):
        policy_id = "other"
    with pytest.raises(InputValidationError, match="unexpected fixation"):
        observed(inputs, fixations={"base": Misbound(T0, False)})
    with pytest.raises(InputValidationError, match="post-assignment"):
        observed(inputs, fixations={"base": PolicyFixation(T0, True)})


def test_scoped_representative_and_versioned_alias(inputs):
    alias = replace(BASE, policy_id="second", version="v2")
    family = CandidateSet((BASE, alias), {})
    for policy, other in ((BASE, alias), (alias, BASE)):
        result = observed(inputs, candidates=family, fixations={policy.policy_id: scoped(policy, *inputs)},
                          include_exploratory=True)
        grades = result.table("pareto").set_index("policy_id").evidence_grade
        assert grades[policy.policy_id]["policy_fixation"] == "caller_declared_pre_outcome"
        assert grades[other.policy_id]["policy_fixation"] == "unverified"
        assert result.to_dict()["audit"]["evaluation_calls"] == 2


@pytest.mark.parametrize("fault", ["H1", "H4-V", "version", "fingerprint", "id", "alias", "rule"])
def test_structured_design_matches_executable_family(profiles, fault):
    policies = profiles.policies
    target = next(p for p in policies if p.policy_id == (fault if fault in ("H1","H4-V") else "H0"))
    fp = target.configuration_fingerprint()
    report = profiles.report
    group = report["policies"][fp]
    if fault in ("H1", "H4-V"):
        group[fault]["thresholds"]["high__standard_value"] = "0.99"
    elif fault == "fingerprint":
        report["policies"]["wrong"] = report["policies"].pop(fp)
    elif fault == "id":
        group["wrong"] = group.pop("H0")
    elif fault == "alias":
        del group["P0"]
    elif fault == "rule":
        policies = tuple(replace(p, policies=tuple(replace(r, allow_candidate=False) for r in p.policies))
                         if p.policy_id == "H0" else p for p in policies)
    else:
        group["H0"]["version"] = "other"
    with pytest.raises(InputValidationError):
        compare(CandidateSet(policies, report))


@pytest.mark.parametrize("directions", [
    [(METRICS[0], "max"), (METRICS[0], "min")], [(METRICS[0], "min"), (METRICS[0], "max")],
    [(METRICS[0], "max")] * 2, {}, {"unsupported": "max"}, {METRICS[0]: "up"}, None,
])
def test_invalid_pareto_declaration(directions):
    with pytest.raises(InputValidationError, match="metrics|directions"):
        compare_policies(CandidateSet((BASE,), {}), search_inputs=_inputs(), scenario_id="demo",
                         pareto_directions=directions)


def test_multi_metric_order_and_audit(inputs, profiles):
    opts = dict(search_inputs=_inputs(), scenario_id="order", experiment_log=inputs[0], evidence=inputs[1],
                allow_unverified_log=True, include_exploratory=True, pairs=[])
    directions = {METRICS[0]: "max", METRICS[-1]: "min"}
    first, second = (compare_policies(profiles, pareto_directions=d, **opts) for d in
                     (directions, dict(reversed(directions.items()))))
    assert first.table("pareto").equals(second.table("pareto"))
    assert second.to_dict()["audit"]["pareto_directions"] == directions


@pytest.mark.parametrize("file_input", [True, False])
def test_internal_value_error_propagates(monkeypatch, inputs, tmp_path, file_input):
    log, evidence = inputs
    if file_input:
        path = tmp_path / "valid.parquet"
        log.to_parquet(path)
        evidence = replace(evidence, log_sha256=sha256(path.read_bytes()).hexdigest())
        log = path
    def broken(*args, **kwargs):
        raise ValueError("injected calculation failure")
    monkeypatch.setattr("final_strategy.policy_comparison.evaluate_policy_value", broken)
    with pytest.raises(ValueError, match="injected calculation failure"):
        compare(experiment_log=log, evidence=evidence)


def test_missing_file_and_checksum_report(tmp_path, inputs):
    log, evidence = inputs
    path = tmp_path / "absent.parquet"
    evidence = replace(evidence, log_sha256="0"*64)
    missing = compare(experiment_log=path, evidence=evidence)
    assert set(missing.table("ips").readiness) == {"contract_violation"}
    log.to_parquet(path)
    corrupt = compare(experiment_log=path, evidence=evidence)
    assert "checksum mismatch" in corrupt.table("ips").iloc[0].hold_reason


def test_generated_phase_c_and_generic_evidence_status(case):
    family = generate(analyze(case), max_candidates=2)
    result = compare(family)
    assert set(result.table("plans").policy_id) == {p.policy_id for p in family.policies}
    assert result.to_dict()["registration"]["design_evidence_status"] == "producer_metadata_only"
    assert compare().to_dict()["registration"]["design_evidence_status"] == "producer_metadata_only"


def test_fixation_cannot_move_between_logs_or_rosters(inputs):
    log, evidence = inputs
    frozen = scoped(BASE, log, evidence)
    for changed, declaration in ((log.assign(customer_id=log.customer_id + "-new"), evidence),
                                 (log, replace(evidence, raw_database_sha256="b"*64))):
        with pytest.raises(InputValidationError, match="scope mismatch"):
            observed((changed, declaration), fixations={"base": frozen})
