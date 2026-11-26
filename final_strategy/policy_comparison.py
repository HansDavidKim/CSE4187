"""Compare fixed retention plans on one frozen scenario and one optional IPS log."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import budget
from .policy_bandit import _register
from .policy_candidates import CandidateSet
from .policy_search import _decimal_key, _json_value, run_policy_search
from .policy_value import (METRICS, ExperimentEvidence, PolicyFixation, _fixation_status, _read_log,
                           compare_policy_values, evaluate_policy_value)
from .retention_strategy import SEGMENT_IDS
from .segmentation import segment_customers
from .strategy_profiles import build_strategy_profiles
from .validation import InputValidationError, JOIN_KEYS


H_PAIRS = tuple((name, "H0") for name in ("H1", "H2", "H3", "H4-R", "H4-V")) + (
    ("H2", "H3"), ("H4-R", "H4-V"),
)


def _require(ok, message):
    if not ok:
        raise InputValidationError("policy comparison: " + message)


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical(obj):
    """Type-tag non-JSON scalars and containers. Frame indexes are not customer keys."""
    if isinstance(obj, pd.DataFrame):
        _require(obj.columns.is_unique and all(isinstance(c, str) for c in obj), "invalid frame columns")
        cols = sorted(obj)
        rows = sorted(_dump([_canonical(v) for v in row])
                      for row in obj[cols].itertuples(index=False, name=None))
        return ["frame", cols, [str(obj[c].dtype) for c in cols], rows, _canonical(obj.attrs)]
    if is_dataclass(obj):
        return [type(obj).__name__, {f.name: _canonical(getattr(obj, f.name)) for f in fields(obj)}]
    if isinstance(obj, dict):
        _require(all(isinstance(k, str) for k in obj), "expected string mapping keys")
        return ["mapping", {k: _canonical(v) for k, v in obj.items()}]
    if isinstance(obj, (list, tuple)):
        return ["sequence", [_canonical(v) for v in obj]]
    if isinstance(obj, np.floating):
        # Other engines may interpret a setting through str(value) or its binary value.
        ratio = obj.as_integer_ratio() if np.isfinite(obj) else None
        return ["numpy_float", obj.dtype.name, str(obj), ratio]
    if isinstance(obj, np.generic):
        return _canonical(obj.item())
    if obj is pd.NA or obj is pd.NaT:
        return ["missing", str(obj)]
    if isinstance(obj, Decimal):
        return ["decimal", str(obj)]
    if isinstance(obj, Fraction):
        return ["fraction", obj.numerator, obj.denominator]
    if isinstance(obj, float):
        return ["float", obj.hex()]
    if isinstance(obj, (date, pd.Timedelta, timedelta)):
        return [type(obj).__name__, str(obj)]
    if isinstance(obj, Path):
        return ["path", str(obj)]
    return obj


def _digest(obj):
    return sha256(_dump(_canonical(obj)).encode()).hexdigest()


def _execution_digest(shared):
    def amount(value):
        return _decimal_key(budget._decimal(value, "fingerprint.amount"))

    def frame_values(frame, decimal_columns):
        if frame is None:
            return None
        frame = frame.copy()
        for col in decimal_columns:
            if col in frame:
                frame[col] = [amount(v) for v in frame[col].astype(object)]
        return frame

    declaration = shared["budget"]
    inputs = shared["inputs"]
    values = dict(shared, budget=replace(declaration, amount=amount(declaration.amount),
                                         amount_unit=amount(declaration.amount_unit)),
                  inputs=replace(inputs, data=frame_values(inputs.data, ["amount_unit"]),
                                 prediction_context={k: frame_values(v, ["amount_unit"])
                                                     for k, v in inputs.prediction_context.items()}))
    for key, columns in (("cost_inputs", ("amount_unit", "send_cost", "redemption_probability",
                                          "expected_discount_if_redeemed")),
                         ("eligibility", ("amount_unit",)), ("segmentation_eligibility", ("amount_unit",)),
                         ("coupons", ("discount_rate", "send_cost_minor"))):
        if key in values:
            values[key] = frame_values(values[key], columns)
    return _digest(values)


@dataclass(frozen=True)
class ScopedPolicyFixation:
    policy_id: str
    policy_version: str
    configuration_fingerprint: str
    evaluation_fingerprint: str
    fixation: PolicyFixation


def evaluation_fingerprint(experiment_log, evidence):
    """Identify the declared experiment and roster-bearing log; file bytes are checked by Phase B."""
    _require(type(evidence) is ExperimentEvidence, "ExperimentEvidence required for fixation scope")
    if isinstance(experiment_log, pd.DataFrame):
        source = ("frame", _digest(experiment_log))
    else:
        _require(isinstance(experiment_log, (str, Path)) and evidence.log_sha256,
                 "file scope requires a declared checksum")
        source = ("parquet", evidence.log_sha256)
    return _digest((source, evidence))


def _effective_fixations(declared, policies, log, evidence, point):
    effective, audit = {}, {}
    scope = None
    for name, item in declared.items():
        _require(name in policies, "unknown policy fixation")
        if type(item) is ScopedPolicyFixation:
            _require(set(vars(item)) == {f.name for f in fields(ScopedPolicyFixation)}, "unexpected scope fields")
            policy = policies[name]
            if scope is None:
                scope = evaluation_fingerprint(log, evidence)
            _require((item.policy_id, item.policy_version, item.configuration_fingerprint, item.evaluation_fingerprint)
                     == (name, policy.version, policy.configuration_fingerprint(), scope), "fixation scope mismatch")
            fixation = item.fixation
            binding = "scope_matched"
        else:
            fixation = item
            binding = "scope_unavailable"
        _require(type(fixation) is PolicyFixation and set(vars(fixation)) == {f.name for f in fields(PolicyFixation)},
                 "unexpected fixation type or fields")
        _fixation_status(fixation, budget._clock(point, "as_of_at"))
        effective[name] = fixation if binding == "scope_matched" else None
        audit[name] = dict(binding=binding, declaration=asdict(item))
    return effective, audit


def _validate_design(candidate_set):
    report = candidate_set.report
    entries = report.get("policies", {})
    human_ids = {"H0", "H1", "H2", "H3", "H4-R", "H4-V"}
    if "segmentation" not in report and not any(isinstance(group, dict) and human_ids.intersection(group)
                                                 for group in entries.values()):
        return "producer_metadata_only"
    # Rebuild the declared family with its own validator, including H4-V's middle-group swap.
    try:
        by_id = {name: entry for group in entries.values() for name, entry in group.items()}
        cuts = {name: by_id[name]["thresholds"] for name in human_ids}
        hh, hs, lh, ls = ("high__high_value", "high__standard_value", "low__high_value", "low__standard_value")
        expected = build_strategy_profiles(
            version=by_id["H1"]["version"], threshold_source=by_id["H1"]["threshold_source"],
            common=cuts["H1"][hh], risk_high=cuts["H2"][hh], risk_low=cuts["H2"][lh],
            high_value=cuts["H3"][hh], standard_value=cuts["H3"][hs],
            h4_r=dict(zip("abcd", (cuts["H4-R"][key] for key in (hh, hs, lh, ls)))),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise InputValidationError("incomplete human strategy declaration") from exc
    originals = {p.policy_id: p for p in expected.policies}
    _require(report == expected.report, "human design evidence differs from executable strategy")
    for policy in candidate_set.policies:
        aliases = entries.get(policy.configuration_fingerprint(), {})
        _require(policy.policy_id in aliases and originals.get(policy.policy_id) == policy,
                 "human design evidence differs from executable strategy")
    return "human_profile_rules_matched"


def _json(obj):
    if isinstance(obj, Fraction):
        return {"numerator": obj.numerator, "denominator": obj.denominator}
    if isinstance(obj, dict):
        return {k: _json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json(v) for v in obj]
    return _json_value(obj)


class PolicyComparisonResult:
    """Read independent typed table copies or a strict JSON-compatible report."""
    def __init__(self, report):
        self._report = deepcopy(report)

    def to_dict(self):
        return _json(self._report)

    def table(self, name):
        _require(name in ("plans", "segments", "ips", "comparisons", "pareto", "decisions"), "unknown table")
        return pd.DataFrame(deepcopy(self._report[name]), dtype=object)


def _ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return Fraction(numerator) / Fraction(denominator)


def _plan(run):
    s = run.summary
    allocation = s["allocation_metadata"]
    retention = s["retention_metadata"]
    n, candidates, selected = s["population_count"], s["candidate_count"], s["selected_customer_count"]
    keys = ("population_count", "candidate_count", "selected_customer_count", "budget", "remaining_budget",
            "total_expected_cost", "objective_value", "algorithm", "algorithm_version", "status", "optimality",
            "has_valid_solution", "accepted_for_planning", "budget_compliant", "decision_fingerprint",
            "configuration_fingerprint", "optimality_gap", "objective_upper_bound", "solver_status", "solver_message")
    return {**{k: s[k] for k in keys}, "analyzed_count": n - retention["upstream_excluded_count"],
            "eligibility_counts_status": "not_provided", "unselected_count": None if selected is None else n-selected,
            "population_request_rate": _ratio(selected, n), "candidate_selection_rate": _ratio(selected, candidates),
            "budget_utilization": _ratio(s["total_expected_cost"], s["budget"]), "solver_config": allocation["solver_config"],
            "hybrid_quality": allocation.get("hybrid", {}).get("quality"), "dispatch_authorized": False,
            "actual_spend": None, "actual_budget_compliant": None,
            "planning_hold_reason": None if s["accepted_for_planning"] else s["status"]}


def _segments(frame, s):
    labels = frame.segment_id.fillna("unclassified")
    groups = list(SEGMENT_IDS)
    if labels.eq("unclassified").any():
        groups.append("unclassified")
    rows = []
    for group in groups:
        members = frame.loc[labels.eq(group)]
        candidates = int(members.budget_candidate.sum())
        selected = members.loc[members.selected.eq(1)]
        valid = s["has_valid_solution"]
        count = len(selected) if valid else None
        rows.append(dict(segment_id=group, population_count=len(members), candidate_count=candidates,
                         selected_customer_count=count, population_selection_rate=_ratio(count, len(members)),
                         candidate_selection_rate=_ratio(count, candidates),
                         total_expected_cost=budget._decimal_sum(selected.expected_cost) if valid else None,
                         objective_value=budget._score_sum(selected.priority_score) if valid else None))
    for key in ("population_count", "candidate_count"):
        _require(sum(r[key] for r in rows) == s[key], "segment reconciliation: " + key)
    if s["has_valid_solution"]:
        _require(sum(r["selected_customer_count"] for r in rows) == s["selected_customer_count"]
                 and budget._decimal_sum(r["total_expected_cost"] for r in rows) == s["total_expected_cost"]
                 and math.isclose(budget._score_sum(r["objective_value"] for r in rows), s["objective_value"],
                                  rel_tol=1e-15), "selected segment totals differ")
    return rows


def _classified_decisions(run, labels):
    frame = run.decisions.merge(labels, on=list(JOIN_KEYS), how="left",
                                validate="one_to_one", suffixes=("_budget", ""))
    supplied = frame.segment_id_budget.notna()
    _require(frame.loc[supplied, "segment_id_budget"].eq(frame.loc[supplied, "segment_id"]).all()
             and frame.segment_id.isna().sum() == run.summary["retention_metadata"]["upstream_excluded_count"],
             "full cohort classification differs from plan")
    return frame.drop(columns="segment_id_budget")


def _held(reason, readiness):
    return (dict(readiness=readiness, causal_interpretation="unverified", dispatch_authorized=False),
            {m: dict(estimable=False, estimate=None, hold_reason=reason) for m in METRICS})


def _pareto(values, directions, include_exploratory):
    rows, competitors = {}, {}
    for name, (summary, metrics) in values.items():
        reasons = [f"{m}: {metrics[m]['hold_reason']}" for m in directions if not metrics[m]["estimable"]]
        grade = {k: summary.get(k) for k in ("accepted_for_planning", "exploratory", "policy_fixation",
                                               "log_checksum_status", "assignment_evidence", "causal_interpretation")}
        if summary.get("exploratory") and not include_exploratory:
            reasons.append("exploratory_excluded")
        if not reasons:
            competitors[name] = [metrics[m]["estimate"] * (1 if direction == "max" else -1)
                                 for m, direction in directions.items()]
        rows[name] = dict(policy_id=name, evidence_grade=grade, status="held", reasons=reasons, dominated_by=[])
    for name, point in competitors.items():
        dominators = []
        for other, rival in competitors.items():
            if rows[other]["evidence_grade"] == rows[name]["evidence_grade"]:
                if all(a >= b for a, b in zip(rival, point)) and any(a > b for a, b in zip(rival, point)):
                    dominators.append(other)
        rows[name].update(status="dominated" if dominators else "non_dominated", dominated_by=dominators)
    return list(rows.values())


def _plan_difference(left, right, phase_a):
    a, b = left.summary, right.summary
    valid = a["has_valid_solution"] and b["has_valid_solution"]
    ca, cb = (set(run.decisions.loc[run.decisions.budget_candidate, "customer_id"]) for run in (left, right))
    out = dict(candidate_count_difference=len(ca)-len(cb), candidate_intersection_count=len(ca & cb),
               left_only_candidates=sorted(ca-cb), right_only_candidates=sorted(cb-ca),
               candidate_jaccard=_ratio(len(ca & cb), len(ca | cb)) if ca or cb else Fraction(1))
    # Phase A already computed the selected sets in one orientation for every distinct pair.
    if phase_a is None:  # aliases of the same run
        out.update(same_decisions=True if valid else None, selected_intersection_count=a["selected_customer_count"],
                   left_only_selected=[] if valid else None, right_only_selected=[] if valid else None,
                   jaccard_similarity=1.0 if valid else None)
    else:
        forward = phase_a["left_policy_id"] == a["policy_id"]
        for key in ("same_decisions", "selected_intersection_count", "jaccard_similarity"):
            out[key] = phase_a[key]
        for side, reverse in (("left", "right"), ("right", "left")):
            out[side+"_only_selected"] = phase_a[(side if forward else reverse)+"_only_selected"]
    for key in ("selected_customer_count", "total_expected_cost", "objective_value"):
        out[key+"_difference"] = Fraction(a[key])-Fraction(b[key]) if valid else None
    la, rb = (_ratio(s["total_expected_cost"], s["budget"]) for s in (a, b))
    out["budget_utilization_difference"] = la-rb if la is not None and rb is not None else None
    return out


def compare_policies(candidate_set, *, search_inputs, scenario_id, pareto_directions, pairs=None,
                     experiment_log=None, evidence=None, fixations=None, allow_unapproved=False,
                     allow_unverified_log=False, include_exploratory=False):
    """Execute each configuration once. Pairwise IPS always uses the original two plans."""
    budget._text(scenario_id, "scenario_id")
    _require(isinstance(pareto_directions, Mapping), "pareto_directions requires a mapping with unique metrics")
    directions = dict(pareto_directions)
    _require(directions and all(m in METRICS and d in ("min", "max") for m, d in directions.items()),
             "declare IPS metrics and min/max directions")
    for flag in (allow_unapproved, allow_unverified_log, include_exploratory):
        _require(isinstance(flag, bool), "comparison options must be Boolean")
    shared = deepcopy(search_inputs)
    _require(isinstance(candidate_set, CandidateSet), "CandidateSet required")
    candidate_set = CandidateSet(deepcopy(candidate_set.policies), deepcopy(candidate_set.report))
    arms, registration = _register(candidate_set)
    registration["design_evidence_status"] = _validate_design(candidate_set)
    declarations = {alias["policy_id"]: (arm["policy_id"], alias["version"]) for arm in registration["arms"]
                    for alias in arm["aliases"]}
    pairs = tuple(pair for pair in H_PAIRS if all(n in declarations for n in pair)) if pairs is None else tuple(pairs)
    _require(all(isinstance(pair, (list, tuple)) and len(pair) == 2 and all(n in declarations for n in pair)
                 for pair in pairs), "unknown policy comparison")
    _require(fixations is None or isinstance(fixations, Mapping), "fixations requires a policy mapping")
    unit = budget._decimal(shared["budget"].amount_unit, "budget.amount_unit", positive=True)
    for frame in (shared["inputs"].data, *shared["inputs"].prediction_context.values(),
                  shared["eligibility"], shared["cost_inputs"], shared.get("segmentation_eligibility")):
        if frame is not None and "amount_unit" in frame:
            _require(all(budget._decimal(v, "amount_unit", positive=True) == unit for v in frame.amount_unit),
                     "supplied amount_unit differs from budget")
    log = deepcopy(experiment_log)
    evidence = deepcopy(evidence)
    fixed, fixation_audit = _effective_fixations(deepcopy(dict(fixations or {})),
                                                {p.policy_id: p for p in candidate_set.policies},
                                                log, evidence, shared["as_of_at"])
    results = run_policy_search(candidates=arms, **shared)
    execution_digest = _execution_digest(shared)
    file_error = None
    if isinstance(log, (str, Path)) and isinstance(evidence, ExperimentEvidence):
        # Limit decoder/IO exception handling to the producer's actual read boundary.
        # Phase B still receives the path and rechecks the bytes on each evaluation.
        try:
            _read_log(log, evidence)
        except (OSError, ValueError) as exc:
            file_error = str(exc)
    plans, segments, decisions = [], [], []
    classifications = {}
    for arm in arms:
        rep = arm.policy_id
        run = results.runs[rep]
        _require(all(budget._decimal(v, "amount_unit", positive=True) == unit for v in run.decisions.amount_unit),
                 "normalized plan lost original amount_unit")
        # Budget strips labels from excluded rows. Recover the original full-cohort
        # classification once per configuration, without changing the run used by IPS.
        config_key = _digest(arm.segmentation)
        if config_key not in classifications:
            classification = segment_customers(shared["inputs"], arm.segmentation,
                                               eligibility=shared.get("segmentation_eligibility"))
            classifications[config_key] = classification.customers[[*JOIN_KEYS, "segment_id"]]
        frame = _classified_decisions(run, classifications[config_key])
        segments.extend(dict(representative_id=rep, **row) for row in _segments(frame, run.summary))
        decisions.extend(dict(representative_id=rep, **row) for row in frame.to_dict("records"))
    values, cache, keys = {}, {}, {}
    evaluation_calls = pair_calls = 0
    start = perf_counter()
    options = dict(evidence=evidence, allow_unapproved=allow_unapproved, allow_unverified_log=allow_unverified_log)
    for name, (rep, version) in declarations.items():
        run = results.runs[rep]
        aliases = [policy_id for policy_id, (owner, _) in declarations.items() if owner == rep]
        plans.append(dict(policy_id=name, policy_version=version, representative_id=rep,
                          aliases=aliases, **_plan(run)))
        key = rep, _digest(fixed.get(name))
        keys[name] = key
        if key not in cache:
            if log is None:
                cache[key] = _held("log_not_supplied", "not_evaluated")
            elif file_error is not None:
                cache[key] = _held(file_error, "contract_violation")
            else:
                evaluation_calls += 1
                try:
                    v = evaluate_policy_value(run, log, fixation=fixed.get(name), **options)
                    cache[key] = v.summary, v.metrics
                except InputValidationError as exc:
                    cache[key] = _held(str(exc), "contract_violation")
        values[name] = cache[key]
    evaluation_seconds = perf_counter() - start
    start = perf_counter()
    comparisons, paired = [], {}
    sets = {frozenset((c["left_policy_id"], c["right_policy_id"])): c for c in results.comparisons}
    for left, right in pairs:
        a, b = keys[left], keys[right]
        delta = _plan_difference(results.runs[a[0]], results.runs[b[0]], sets.get(frozenset((a[0], b[0]))))
        pair = a, b
        if pair not in paired:
            metrics = {}
            if all(values[n][0]["readiness"] not in ("contract_violation", "not_evaluated") for n in (left, right)):
                pair_calls += 1
                comparison = compare_policy_values(results.runs[a[0]], results.runs[b[0]], log,
                                                   left_fixation=fixed.get(left), right_fixation=fixed.get(right), **options)
                metrics = comparison.metrics
            paired[pair] = metrics
        differences = {}
        for m in METRICS:
            lm, rm = values[left][1][m], values[right][1][m]
            item = dict(paired[pair].get(m, dict(estimable=False, difference=None,
                                                left_hold_reason=lm["hold_reason"], right_hold_reason=rm["hold_reason"])))
            diff = item["difference"]
            item.update(relative_change=_ratio(diff, rm["estimate"]),
                        percentage_points=100*diff if diff is not None and m in METRICS[:2] else None)
            differences[m] = item
        comparisons.append(dict(left_policy_id=left, right_policy_id=right, **delta, ips=differences))
    ips = []
    for name, (summary, metrics) in values.items():
        rep, version = declarations[name]
        for m, metric in metrics.items():
            ips.append({**summary, **metric, "policy_id": name, "policy_version": version,
                        "representative_id": rep, "metric": m})
    audit = dict(report_version="policy-comparison-v2", scenario_id=scenario_id,
                 created_at=pd.Timestamp.now(tz="UTC").isoformat(), common_input_fingerprint=execution_digest,
                 dataset_id=shared["dataset_id"], as_of_at=shared["as_of_at"], currency=shared["budget"].currency,
                 amount_unit=str(unit), cost_metadata=vars(shared["cost_metadata"]),
                 models={k: vars(v) for k, v in shared["inputs"].prediction_metadata.items()},
                 raw_database_sha256=sorted(set(shared["inputs"].data.raw_database_sha256)),
                 history_context=vars(shared["history_context"]), history_policy=vars(shared["history_policy"]),
                 log_fingerprint=_digest(log) if isinstance(log, pd.DataFrame) else None,
                 log_path=str(log) if isinstance(log, (str, Path)) else None,
                 evidence=vars(evidence) if isinstance(evidence, ExperimentEvidence) else None,
                 fixations=fixation_audit,
                 allow_unapproved=allow_unapproved, allow_unverified_log=allow_unverified_log,
                 pareto_directions=directions, include_exploratory=include_exploratory,
                 evaluation_calls=evaluation_calls, paired_calls=pair_calls, paired_reevaluations=2*pair_calls,
                 evaluation_seconds=evaluation_seconds, paired_seconds=perf_counter()-start,
                 causal_interpretation="unverified", dispatch_authorized=False, policy_effect_verified=False)
    return PolicyComparisonResult(dict(plans=plans, segments=segments, decisions=decisions, ips=ips,
                                      comparisons=comparisons, pareto=_pareto(values, directions, include_exploratory),
                                      registration=registration, audit=audit))
