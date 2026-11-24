"""Translate supported calibration thresholds into bounded, fixed Phase A rules."""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import chain
import pandas as pd

from .policy_search import PolicyCandidate
from .retention_strategy import default_segment_policies
from .segmentation import SegmentationConfig
from .validation import InputValidationError


@dataclass(frozen=True)
class CandidateSet:
    policies: tuple[PolicyCandidate, ...]
    report: dict

    def to_dict(self):
        return deepcopy(self.report)


def _rule_support(rows, risk, value, cut, uplift, min_arm, min_events):
    if risk is not None:
        rows = rows.loc[rows.churn_probability.ge(cut).eq(risk == "high")]
    if value is not None:
        rows = rows.loc[rows.clv_band.eq(value)]
    totals = rows.groupby(["role", rows.uplift_score.gt(uplift).rename("above_cut"), "treatment"])[
        ["n", "purchases"]].sum()
    grid = pd.MultiIndex.from_product([("discovery", "confirmation"), (False, True), (0, 1)],
                                      names=totals.index.names)
    totals = totals.reindex(grid, fill_value=0)
    return bool((totals.n.ge(min_arm) & totals.purchases.ge(min_events)).all()), totals.reset_index().to_dict("records")


def generate_candidates(analysis, *, version, max_candidates, max_splits_per_feature, max_rule_groups):
    """Rank supported thresholds by occurrence, gain, then value. Always retain P0.

    Each variant tightens one target group's uplift cutoff, retaining the baseline
    elsewhere. Risk-based rules can also vary the churn cut. Absolute CLV and
    additional KIT39 splits stay in the analysis report.
    """
    for limit in (max_candidates, max_splits_per_feature, max_rule_groups):
        if type(limit) is not int or limit < 1:
            raise InputValidationError("candidate limits must be positive integers")
    default = SegmentationConfig()
    baseline = PolicyCandidate("P0", version, "Development positive-uplift baseline", "default_segment_policies")
    policies = [baseline]
    support_rows = pd.DataFrame(analysis["rule_support"])
    baseline_key = baseline.configuration_fingerprint()
    details = {baseline_key: dict(policy_id="P0", version=version, complexity=1, evidence=[])}
    skipped = {}
    exclusions = []
    combinations_examined = support_checks = 0

    def omit(reason, **context):
        skipped[reason] = skipped.get(reason, 0) + 1
        exclusions.append(dict(reason=reason, **context))

    groups = {"uplift_score": {}, "churn_probability": {}}
    for item in analysis["thresholds"]:
        feature, value = item["feature"], item["threshold"]
        if feature not in groups or not 0 <= value <= 1:
            omit("unrepresentable", feature=feature, threshold=value, source=item["source"])
        elif not item["supported"]:
            omit("insufficient_support", feature=feature, threshold=value, source=item["source"])
        else:
            groups[feature].setdefault(value, []).append(item)
    for feature, values in groups.items():
        ranking = sorted(values, key=lambda k: (-sum(i.get("occurrences", 0) for i in values[k]),
                                                -sum(i.get("gain", 0) for i in values[k]), k))
        for k in ranking[max_splits_per_feature:]:
            omit("per_feature_limit", feature=feature, threshold=k, sources=[i["source"] for i in values[k]])
        groups[feature] = {k: values[k] for k in ranking[:max_splits_per_feature]}
    churn = {default.churn_threshold: [], **groups["churn_probability"]}
    # Iterate without materializing a Cartesian product of twelve segment rules.
    for family, risks, values in (
        ("P1", (None,), (None,)), ("P2", ("high", "low"), (None,)),
        ("P3", (None,), ("high_value", "standard_value")),
        ("P4", ("high", "low"), ("high_value", "standard_value")),
    ):
        for uplift, sources in groups["uplift_score"].items():
            for risk in risks:
                for value in values:
                    for cut in churn if risk else (default.churn_threshold,):
                        combinations_examined += 1
                        context = dict(family=family, risk=risk, clv=value, churn_cut=cut, min_uplift=uplift,
                                       uplift_sources=sorted({i["source"] for i in sources}),
                                       churn_sources=sorted({i["source"] for i in churn[cut]}) if risk else [])
                        complexity = 1 if risk is None and value is None else 2
                        if complexity > max_rule_groups:
                            omit("complexity_limit", **context)
                            continue
                        support_checks += 1
                        supported, support = _rule_support(support_rows, risk, value, cut, uplift,
                                                           analysis["min_arm"], analysis["min_events"])
                        if not supported:
                            omit("insufficient_rule_group_support", **context)
                            continue
                        rules = []
                        for rule in default_segment_policies():
                            r, _, v = rule.segment_id.split("__")
                            applies = (risk is None or r == risk) and (value is None or v == value)
                            rules.append(replace(rule, min_uplift=uplift if applies else 0))
                        candidate = PolicyCandidate(family, version, "Exploratory minimum uplift rule", "phase_c",
                                                    replace(default, churn_threshold=cut), tuple(rules))
                        key = candidate.configuration_fingerprint()
                        basis = dict(family=family, risk=risk, clv=value, churn_cut=cut, min_uplift=uplift,
                                     group_support=support,
                                     thresholds=deepcopy(list(chain(sources, churn.get(cut, []) if risk else []))))
                        if key in details:
                            details[key]["evidence"].append(basis)
                            omit("duplicate_configuration", fingerprint=key, **context)
                        elif len(policies) >= max_candidates:
                            omit("candidate_limit", fingerprint=key, **context)
                        else:
                            candidate = replace(candidate, policy_id=f"{family}-{key[:16]}")
                            policies.append(candidate)
                            details[key] = dict(policy_id=candidate.policy_id, version=version,
                                                complexity=complexity, evidence=[basis])
    report = dict(policies=details, omitted=skipped, exclusions=exclusions, combinations_examined=combinations_examined,
                  support_checks=support_checks, limits=dict(max_candidates=max_candidates,
                  max_splits_per_feature=max_splits_per_feature, max_rule_groups=max_rule_groups),
                  source=deepcopy(analysis["provenance"]), split=deepcopy(analysis["split"]),
                  model_version=(analysis["residual_model"] or {}).get("version"),
                  stability=analysis["stability"], policy_effect_verified=False)
    return CandidateSet(tuple(policies), report)
