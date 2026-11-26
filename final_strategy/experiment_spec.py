"""Planning-only five-arm experiment contract; no assignment or dispatch.

Allocation weights and the purchase endpoint are planning declarations, not team
approval. Policy decisions remain unresolved even when a power plan is computed.
"""
from dataclasses import asdict, dataclass
import math
import numbers

from .validation import InputValidationError

ARMS = (("A", "No Action"), ("B", "Random"), ("C", "Uplift-only"),
        ("D1", "Full Strategy + Greedy"), ("D2", "Full Strategy + MILP"))
PRIMARY = (("B", "D2"), ("C", "D2"))
UNRESOLVED = ("random_cost_selection", "uplift_nonpositive_and_cost_selection",
              "milp_failure_policy", "per_customer_budget", "business_eligibility", "roi_definition")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InputValidationError(f"experiment.{name}: finite real required")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise InputValidationError(f"experiment.{name}: unsupported numeric range") from exc
    if not math.isfinite(result):
        raise InputValidationError(f"experiment.{name}: finite real required")
    return result


def _json_numbers(value):
    """Copy containers and expose numeric scalars as finite JSON-native values.

    Normalize only at export: retain validated inputs and all calculation paths,
    including their original precision. Preserve tuple/list and boolean metadata.
    """
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_numbers(item) for item in value)
    if isinstance(value, list):
        return [_json_numbers(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return _number(value, "serialized_number")
    return value


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    version: str
    allocation_weights: tuple[int, ...]  # A, B, C, D1, D2; explicit, no default decision
    arms: tuple = ARMS
    primary_comparisons: tuple = PRIMARY
    secondary_comparisons: tuple = (("D1", "D2"),)
    primary_metric: str = "purchase_conversion_28d"
    metric_unit: str = "proportion"
    coupon_discount: float = 0.10
    coupon_validity_days: int = 28
    outcome_days: int = 28
    family_alpha: float = 0.05
    target_power: float = 0.8
    alternative: str = "two-sided"
    analysis_correction: str = "holm"
    independent_evaluation_required: bool = True
    no_action_new_coupon: bool = False
    common_eligibility_required: bool = True
    common_per_customer_budget_required: bool = True
    d1_d2_same_policy_score_cost: bool = True
    unresolved_policy_items: tuple = UNRESOLVED
    design_status: str = "planning_unapproved"

    def __post_init__(self):
        for name in ("experiment_id", "version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise InputValidationError(f"experiment.{name}: nonempty string required")
        for name, expected in (("arms", ARMS), ("primary_comparisons", PRIMARY),
                               ("unresolved_policy_items", UNRESOLVED)):
            if getattr(self, name) != expected or type(getattr(self, name)) is not tuple:
                raise InputValidationError(f"experiment.{name}: fixed Phase 1 contract required")
        weights = self.allocation_weights
        if type(weights) is not tuple or len(weights) != 5 or any(type(w) is not int or not 1 <= w <= 10000 for w in weights):
            raise InputValidationError("experiment.allocation_weights: five positive integer weights <=10000 required")
        comparisons = self.secondary_comparisons
        if type(comparisons) is not tuple:
            raise InputValidationError("experiment.secondary_comparisons: immutable tuple required")
        seen = set()
        for pair in comparisons:
            if (type(pair) is not tuple or len(pair) != 2 or any(type(x) is not str for x in pair)
                    or not set(pair) <= {a for a, _ in ARMS} or len(set(pair)) != 2):
                raise InputValidationError("experiment.secondary_comparisons: invalid pair")
            key = frozenset(pair)
            if key in seen or key in {frozenset(p) for p in PRIMARY}:
                raise InputValidationError("experiment.secondary_comparisons: duplicate or primary pair")
            seen.add(key)
        if not 0 < _number(self.family_alpha, "family_alpha") < 1:
            raise InputValidationError("experiment.family_alpha: must be in (0,1)")
        if not 0.8 <= _number(self.target_power, "target_power") < 1:
            raise InputValidationError("experiment.target_power: must be in [0.8,1)")
        if _number(self.coupon_discount, "coupon_discount") != 0.1:
            raise InputValidationError("experiment.coupon_discount: requires 10 percent")
        for name in ("coupon_validity_days", "outcome_days"):
            if type(getattr(self, name)) is not int or getattr(self, name) != 28:
                raise InputValidationError(f"experiment.{name}: requires 28 days")
        fixed = {"primary_metric": "purchase_conversion_28d", "metric_unit": "proportion",
                 "alternative": "two-sided", "analysis_correction": "holm",
                 "design_status": "planning_unapproved", "independent_evaluation_required": True,
                 "no_action_new_coupon": False, "common_eligibility_required": True,
                 "common_per_customer_budget_required": True, "d1_d2_same_policy_score_cost": True}
        for name, value in fixed.items():
            if type(getattr(self, name)) is not type(value) or getattr(self, name) != value:
                raise InputValidationError(f"experiment.{name}: fixed Phase 1 condition")

    def to_dict(self):
        result = asdict(self)
        result.update(planning_correction="bonferroni", comparison_alpha=self.family_alpha / 2,
                      power_scope="per_comparison_not_joint", equal_actual_spend_required=False,
                      solver_by_arm={"D1": "greedy", "D2": "milp"},
                      secondary_plan_status="proposed_exploratory_not_confirmatory")
        return _json_numbers(result)
