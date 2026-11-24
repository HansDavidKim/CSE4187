"""Run fixed retention policies on one cohort and compare their planned decisions.

This module compares allocations and priority-score proxies. It does not rank
policies by marketing effectiveness or authorize a coupon send.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
from itertools import combinations
import json
import math
from time import perf_counter

import numpy as np
import pandas as pd

from . import budget as engine
from .budget import Budget, CostMetadata, SolverConfig, optimize_budget
from .inputs import IntegratedInputs
from .retention_strategy import (
    HistoryContext, HistoryPolicy, SegmentPolicy, _build_retention_strategy,
    _duration, _policies, default_segment_policies,
)
from .segmentation import SegmentationConfig, segment_customers
from .validation import JOIN_KEYS, InputValidationError


@dataclass(frozen=True)
class PolicyCandidate:
    """One complete policy. Use per-segment rules to express differential targeting."""

    policy_id: str
    version: str
    description: str
    source: str
    segmentation: SegmentationConfig = SegmentationConfig()
    policies: tuple[SegmentPolicy, ...] = default_segment_policies()

    def __post_init__(self):
        for field in ("policy_id", "version", "description", "source"):
            engine._text(getattr(self, field), f"policy_candidate.{field}")
        if not isinstance(self.segmentation, SegmentationConfig):
            raise InputValidationError("policy_candidate: requires SegmentationConfig")
        if self.policies is None:
            raise InputValidationError("policy_candidate: declare all twelve segment policies")
        try:
            rules = _policies(self.policies)
        except TypeError as exc:
            raise InputValidationError("policy_candidate: policies must be iterable") from exc
        object.__setattr__(self, "policies", tuple(rules[key] for key in sorted(rules)))

    def configuration_fingerprint(self) -> str:
        """Hash behavior and exclusion rules, independent of declaration IDs and order."""
        config = asdict(self.segmentation)
        for key in ("churn_threshold", "uplift_threshold"):
            config[key] = float(config[key])
        config["high_clv_fraction"] = _decimal_key(Decimal(str(config["high_clv_fraction"])))
        rules = []
        for rule in self.policies:
            rules.append([
                rule.segment_id, rule.allow_candidate,
                _decimal_key(engine._decimal(rule.min_uplift, "min_uplift")),
                rule.exclusion_reason, _duration(rule.send_delay, "send_delay").isoformat(),
            ])
        return _fingerprint([config, rules])


@dataclass(frozen=True)
class PolicyRun:
    decisions: pd.DataFrame
    summary: dict


@dataclass(frozen=True)
class PolicySearchResult:
    runs: dict[str, PolicyRun]
    comparisons: tuple[dict, ...]

    def to_dict(self) -> dict:
        """Export all planned decisions. Exact Decimal amounts become strings."""
        return _json_value({
            "runs": {
                key: {"summary": run.summary, "decisions": run.decisions.to_dict("records")}
                for key, run in self.runs.items()
            },
            "comparisons": self.comparisons,
        })


def _decimal_key(value):
    if value.is_zero():
        return ("0", 0)
    digits = "".join(map(str, value.as_tuple().digits)).rstrip("0")
    return (digits, value.adjusted())


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, np.floating):
        return _json_value(float(value))
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else (0.0 if value == 0 else value)
    return value


def _fingerprint(value):
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _compare(left, right):
    a, b = left.summary, right.summary
    same_config = a["configuration_fingerprint"] == b["configuration_fingerprint"]
    comparison = {"left_policy_id": a["policy_id"], "right_policy_id": b["policy_id"],
                  "same_configuration": same_config, "comparable": False,
                  "same_decisions": None, "same_candidates": None, "selected_intersection_count": None,
                  "left_only_selected": None, "right_only_selected": None, "jaccard_similarity": None}
    if a["decision_fingerprint"] is None or b["decision_fingerprint"] is None:
        return comparison
    left_selected = set(left.decisions.loc[left.decisions.selected.eq(1), "customer_id"])
    right_selected = set(right.decisions.loc[right.decisions.selected.eq(1), "customer_id"])
    common, union = left_selected & right_selected, left_selected | right_selected
    comparison.update(
        comparable=True, same_decisions=a["decision_fingerprint"] == b["decision_fingerprint"],
        same_candidates=set(left.decisions.loc[left.decisions.budget_candidate, "customer_id"])
        == set(right.decisions.loc[right.decisions.budget_candidate, "customer_id"]),
        selected_intersection_count=len(common), left_only_selected=sorted(left_selected - common),
        right_only_selected=sorted(right_selected - common),
        jaccard_similarity=len(common) / len(union) if union else 1.0,
    )
    return comparison


def run_policy_search(
    inputs: IntegratedInputs, candidates, eligibility: pd.DataFrame, coupons: pd.DataFrame,
    redemptions: pd.DataFrame, cost_inputs: pd.DataFrame, cost_metadata: CostMetadata, budget: Budget,
    *, dataset_id: str, as_of_at, history_context: HistoryContext, history_policy: HistoryPolicy,
    algorithm: str = "greedy", solver_config: SolverConfig | None = None, hybrid_config=None,
    segmentation_eligibility: pd.DataFrame | None = None,
) -> PolicySearchResult:
    """Evaluate declared candidates with one algorithm and shared inputs.

    The common cost table may cover a superset of the union of policy
    candidates within this cohort. Each candidate needs an exact cost row.
    Validation failures raise. Solver failures are retained as unavailable plans.
    """
    policies = tuple(candidates)
    if not policies or any(not isinstance(policy, PolicyCandidate) for policy in policies):
        raise InputValidationError("policy_search: requires at least one PolicyCandidate")
    if len({policy.policy_id for policy in policies}) != len(policies):
        raise InputValidationError("policy_search: duplicate policy ID")
    if algorithm not in ("greedy", "milp", "hybrid"):
        raise InputValidationError("policy_search: choose one budget algorithm")
    if not isinstance(budget, Budget):
        raise InputValidationError("policy_search: requires Budget")
    point = engine._clock(as_of_at, "as_of_at")
    runs = {}
    for policy in policies:
        start = perf_counter()
        seg = segment_customers(inputs, policy.segmentation, eligibility=segmentation_eligibility)
        retention = _build_retention_strategy(
            seg, eligibility, coupons, redemptions, cost_inputs, cost_metadata,
            dataset_id=dataset_id, as_of_at=point, currency=budget.currency, amount_unit=budget.amount_unit,
            history_context=history_context, history_policy=history_policy, policies=policy.policies,
            common_cost_inputs=True,
        )
        optimized = optimize_budget(
            retention.budget_inputs, retention.costs, budget, cost_metadata, dataset_id=dataset_id,
            as_of_at=point, algorithm=algorithm, config=solver_config, hybrid_config=hybrid_config,
        )
        allocation = optimized.results[algorithm]
        planning_fields = [*JOIN_KEYS, "budget_candidate", "planned_sent_at", "requires_eligibility_recheck",
                           "eligibility_reason", "policy_exclusion_reason"]
        provenance_fields = [*JOIN_KEYS, "source_id", "source_version", "source_as_of_at"]
        decisions = allocation.decisions.merge(retention.evaluation[planning_fields], on=list(JOIN_KEYS),
                                               how="left", validate="one_to_one").merge(
            retention.cost_estimates[provenance_fields], on=list(JOIN_KEYS), how="left", validate="one_to_one",
        )
        decisions["planned_sent_at"] = decisions["planned_sent_at"].where(decisions.selected.eq(1))
        decisions["requires_eligibility_recheck"] = (
            decisions.requires_eligibility_recheck.eq(True) & decisions.selected.eq(1).fillna(False)
        )
        plan_hash = None
        if allocation.summary["has_valid_solution"]:
            identity = ["dataset_id", "as_of_at", "customer_id", "snapshot_id", "raw_database_sha256"]
            plan_hash = _fingerprint(decisions.sort_values(identity)[
                [*identity, "selected", "planned_sent_at"]
            ].to_dict("records"))
        excluded = decisions.loc[~decisions.budget_candidate, "exclusion_reason"]
        summary = dict(
            allocation.summary, policy_id=policy.policy_id, policy_version=policy.version,
            configuration=asdict(policy), configuration_fingerprint=policy.configuration_fingerprint(),
            decision_fingerprint=plan_hash, dataset_id=dataset_id, as_of_at=point.isoformat(),
            population_count=len(decisions), candidate_count=int(decisions.budget_candidate.sum()),
            excluded_count=len(excluded), exclusion_counts=excluded.value_counts().to_dict(),
            elapsed_seconds=perf_counter() - start, retention_metadata=retention.metadata,
            allocation_metadata=optimized.metadata, objective_semantics="priority_score_proxy",
        )
        runs[policy.policy_id] = PolicyRun(decisions, summary)
    return PolicySearchResult(runs, tuple(_compare(a, b) for a, b in combinations(runs.values(), 2)))
