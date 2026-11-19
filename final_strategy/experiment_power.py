"""Two-sided arcsine-normal planning approximation for 28-day conversion.

Cohen h = 2 asin(sqrt(p1)) - 2 asin(sqrt(p0)); variance = 1/n0+1/n1.
This is not an exact finite-sample test, a p-value, or ROI power analysis.
Near-boundary/small-cell designs need simulation/exact-test validation before
deployment. Censor inflation assumes nondifferential loss, not guaranteed counts.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import math

from scipy.special import ndtr
from scipy.stats import norm

from .experiment_spec import ARMS, PRIMARY, ExperimentSpec, _number, _json_numbers
from .validation import InputValidationError


@dataclass(frozen=True)
class PowerScenario:
    comparison: tuple[str, str]
    baseline_rate: float
    mde_absolute: float
    direction: str
    censoring_rate: float
    assumption_source: str
    baseline_is_assumed: bool = True

    def __post_init__(self):
        if type(self.comparison) is not tuple or self.comparison not in PRIMARY:
            raise InputValidationError("power.comparison: B/D2 or C/D2 required")
        p = _number(self.baseline_rate, "baseline_rate")
        d = _number(self.mde_absolute, "mde_absolute")
        c = _number(self.censoring_rate, "censoring_rate")
        if not 0 <= p <= 1 or d <= 0 or not 0 <= c < 1:
            raise InputValidationError("power: rate [0,1], MDE >0, censoring [0,1) required")
        if self.direction not in ("increase", "decrease"):
            raise InputValidationError("power.direction: increase or decrease required")
        if not 0 <= self.alternative_rate <= 1 or self.alternative_rate == p:
            raise InputValidationError("power: alternative outside [0,1] or MDE below floating precision")
        if type(self.baseline_is_assumed) is not bool or not isinstance(self.assumption_source, str) or not self.assumption_source.strip():
            raise InputValidationError("power: explicit assumption source and boolean provenance required")

    @property
    def alternative_rate(self):
        return float(self.baseline_rate) + (1 if self.direction == "increase" else -1) * float(self.mde_absolute)


def approximate_power(baseline_rate, alternative_rate, n_baseline, n_alternative, alpha):
    """Arcsine variance-stabilized normal power, including both rejection tails."""
    p0, p1, a = (_number(v, k) for v, k in ((baseline_rate, "baseline_rate"),
                  (alternative_rate, "alternative_rate"), (alpha, "alpha")))
    if not 0 <= p0 <= 1 or not 0 <= p1 <= 1 or not 0 < a < 1:
        raise InputValidationError("power: probabilities [0,1] and alpha (0,1) required")
    if any(type(n) is not int or not 1 <= n <= 10**12 for n in (n_baseline, n_alternative)):
        raise InputValidationError("power: integer sample sizes in [1,10**12] required")
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0)))
    z = norm.isf(a / 2)
    shift = h / math.sqrt(1 / n_baseline + 1 / n_alternative)
    result = float(ndtr(shift - z) + ndtr(-shift - z))
    if not math.isfinite(z) or not math.isfinite(result):
        raise InputValidationError("power: numerically unsupported scenario")
    return result


@dataclass(frozen=True)
class PowerPlan:
    experiment_id: str
    version: str
    primary_metric: str
    family_alpha: float
    comparison_alpha: float
    target_power: float
    scenarios: tuple[PowerScenario, ...]
    minimum_analyzable_by_arm: tuple[tuple[str, int], ...]
    required_assigned_by_arm: tuple[tuple[str, int], ...]
    achieved_power: tuple[tuple[str, float], ...]
    method: str = "two-sided Cohen-h arcsine normal; integer allocation-block search"

    def to_dict(self):
        result = asdict(self)
        result.update(total_required_evaluation_customers=sum(n for _, n in self.required_assigned_by_arm),
                      alternative_rates=[s.alternative_rate for s in self.scenarios],
                      design_status="planning_unapproved", planning_correction="bonferroni",
                      analysis_correction="holm", power_scope="per_comparison_not_joint",
                      effect_is_scenario_assumption=True, censoring_assumption="common nondifferential loss",
                      population_scope="independent eligible evaluation customers, not raw generated population",
                      limitations="Approximation only; no assignment, effect test, revenue/ROI power or deployment approval")
        return _json_numbers(result)


def plan_sample_size(spec: ExperimentSpec, scenarios: tuple[PowerScenario, ...], *, max_total=100_000_000):
    """Find the smallest integer allocation block meeting both approximate powers.

    A/D1 counts follow the allocation, not a claim of powered secondary contrasts.
    One common loss rate is supported; differing per-comparison losses are rejected.
    """
    if not isinstance(spec, ExperimentSpec) or type(scenarios) is not tuple or len(scenarios) != 2:
        raise InputValidationError("power: ExperimentSpec and two immutable scenarios required")
    if any(not isinstance(s, PowerScenario) for s in scenarios) or {s.comparison for s in scenarios} != set(PRIMARY):
        raise InputValidationError("power: exactly the two primary comparisons required")
    scenarios = tuple(next(s for s in scenarios if s.comparison == pair) for pair in PRIMARY)
    if scenarios[0].censoring_rate != scenarios[1].censoring_rate:
        raise InputValidationError("power: common censoring rate required")
    if type(max_total) is not int or not 5 <= max_total <= 10**12:
        raise InputValidationError("power.max_total: integer in [5,10**12] required")
    weights = dict(zip((a for a, _ in ARMS), spec.allocation_weights))
    alpha = spec.family_alpha / 2
    def powers(k):
        return tuple(approximate_power(s.baseline_rate, s.alternative_rate,
                     k * weights[s.comparison[0]], k * weights['D2'], alpha) for s in scenarios)
    limit = max_total // sum(weights.values())
    if limit < 1 or min(powers(limit)) < spec.target_power:
        raise InputValidationError("power: target cannot be reached within max_total")
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi) // 2
        if min(powers(mid)) >= spec.target_power: hi = mid
        else: lo = mid + 1
    analyzed = tuple((a, lo * w) for a, w in weights.items())
    retention = Decimal(1) - Decimal(str(scenarios[0].censoring_rate))
    assigned_block = int((Decimal(lo) / retention).to_integral_value(rounding=ROUND_CEILING))
    assigned = tuple((a, assigned_block * w) for a, w in weights.items())
    if sum(n for _, n in assigned) > max_total:
        raise InputValidationError("power: censor-adjusted allocation exceeds max_total")
    actual_power = powers(lo)
    if min(actual_power) < spec.target_power:
        raise InputValidationError("power: integer sample verification failed")
    return PowerPlan(spec.experiment_id, spec.version, spec.primary_metric, spec.family_alpha,
                     alpha, spec.target_power, scenarios, analyzed, assigned,
                     tuple((f'{s.comparison[0]}_vs_D2', p) for s, p in zip(scenarios, actual_power)))
