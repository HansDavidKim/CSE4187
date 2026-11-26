"""Search frozen Phase C policies. Offline estimates and fresh observations use separate APIs."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
from itertools import islice
import json
import math
from numbers import Real
import random
import re
from time import perf_counter

from .policy_candidates import CandidateSet
from .policy_search import PolicyCandidate, _json_value, run_policy_search
from .policy_value import compare_policy_values, evaluate_policy_value
from .validation import InputValidationError


PURCHASE = "itt_purchase_rate_28d_v1"
SYNTHETIC = "synthetic_bounded_v1"
METHODS = ("uniform", "random", "epsilon_greedy", "ucb1")


def _check(condition, message):
    if not condition:
        raise InputValidationError(message)


def _count(value, name, minimum=0):
    _check(type(value) is int and value >= minimum, f"invalid {name}")


def _bounded(value, name):
    _check(isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1,
           f"{name} must be a finite number in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class SearchResult:
    report: dict

    def to_dict(self):
        return deepcopy(self.report)


def _register(candidate_set):
    _check(isinstance(candidate_set, CandidateSet) and candidate_set.policies, "empty or invalid CandidateSet")
    policies = deepcopy(candidate_set.policies)
    _check(all(isinstance(p, PolicyCandidate) for p in policies), "invalid policy")
    report = deepcopy(candidate_set.report)
    ids, groups, arms = {}, {}, {}
    for p in sorted(policies, key=lambda p: (p.policy_id != "P0", p.policy_id)):
        key = p.configuration_fingerprint()
        _check(ids.setdefault(p.policy_id, (key, p.version)) == (key, p.version), "conflicting policy ID")
        entry = groups.setdefault(key, dict(policy_id=p.policy_id, configuration_fingerprint=key,
                                            aliases=[], evidence=report.get("policies", {}).get(key, {})))
        entry["aliases"].append(_json_value(asdict(p)))
        arms.setdefault(key, p)
    keys = sorted(arms, key=lambda k: (arms[k].policy_id != "P0", k))
    registration = dict(arms=[groups[k] for k in keys], candidate_report=report,
                        dispatch_authorized=False, policy_effect_verified=False)
    json.dumps(registration, allow_nan=False)
    return [arms[k] for k in keys], registration


def search_fixed_log(candidate_set, *, method, seed, max_evaluations, search_inputs, experiment_log,
                     evidence, fixation=None, allow_unverified_log=False):
    """Run A and B once per chosen configuration. Compare against P0 only if evaluated."""
    start = perf_counter()
    value_calls = 0
    arms, report = _register(candidate_set)
    _check(method in ("exhaustive", "random"), "invalid fixed-log method")
    _count(seed, "seed")
    _count(max_evaluations, "max_evaluations")
    if method == "random":
        random.Random(seed).shuffle(arms)
    order = [p.policy_id for p in arms[:max_evaluations]]
    runs, results, comparisons = {}, {}, {}
    options = dict(evidence=evidence, fixation=fixation, allow_unverified_log=allow_unverified_log)
    for policy in arms[:max_evaluations]:
        key = policy.policy_id
        try:
            plan = run_policy_search(candidates=[policy], **search_inputs).runs[key]
            value_calls += 1
            value = evaluate_policy_value(plan, experiment_log, **options)
        except InputValidationError as exc:
            results[key] = dict(status="contract_violation", reason=str(exc))
            continue
        runs[key] = plan
        metric = value.metrics["purchase_rate_28d"]
        results[key] = dict(status="observed_ips" if metric["estimable"] else "held",
                            reason=metric["hold_reason"], result=value.to_dict())
    comparison_start = perf_counter()
    if "P0" in runs:
        for key, plan in runs.items():
            comparisons[key] = compare_policy_values(
                plan, runs["P0"], experiment_log, evidence=evidence, left_fixation=fixation,
                right_fixation=fixation, allow_unverified_log=allow_unverified_log).to_dict()
    return SearchResult(dict(report, environment="fixed_log_ips", method=method, seed=seed,
                             policy_run_calls=len(order), value_calls=value_calls, comparison_calls=len(comparisons),
                             comparison_seconds=perf_counter() - comparison_start if comparisons else 0.,
                             elapsed_seconds=perf_counter() - start, limit=max_evaluations, order=order, target_count=len(arms),
                             attempted_count=len(order), evaluated_count=len(runs),
                             coverage_complete=len(order) == len(arms), results=results,
                             comparisons_to_p0=comparisons, p0_evaluated="P0" in runs, independent_observations=0))


@dataclass(frozen=True)
class EvaluationUnit:
    unit_id: str
    state_id: str
    roster_fingerprint: str
    dataset_id: str
    seed: int
    # Producer declarations, carried for audit. Only search units are accepted.
    role: str = "search"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    run_id: str
    attempt_id: str
    unit: EvaluationUnit
    policy_id: str
    version: str
    configuration_fingerprint: str
    environment: str
    reward_definition: str
    conditions: dict


@dataclass(frozen=True)
class Observation:
    """Echo the request and attach producer evidence. No simulator transport is assumed."""
    request: EvaluationRequest
    status: str
    producer_run_id: str | None = None
    outcome_id: str | None = None
    ledger_sha256: str | None = None
    valid_plan: bool | None = None
    approved_plan: bool | None = None
    decision_fingerprint: str | None = None
    receipt: str | None = None
    reward: float | None = None
    assigned: int | None = None
    observed: int | None = None
    censored: int | None = None
    purchasing: int | None = None
    horizon_days: int | None = None
    auxiliary: dict = field(default_factory=dict)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_unused(seen, values):
    for name, value in values.items():
        _check(value not in seen.get(name, ()), f"reused {name}")


def _register_ids(seen, values):
    for name, value in values.items():
        seen.setdefault(name, set()).add(value)


def _observe(obs, request, seen):
    _check(isinstance(obs, Observation) and obs.request == request, "observation/request mismatch")
    statuses = ("observed", "solver_failure", "invalid_plan", "unapproved_plan", "contract_violation",
                "external_failure", "censored", "pending")
    _check(obs.status in statuses, "invalid observation status")
    json.dumps(obs.auxiliary, allow_nan=False)
    identifiers = {}
    for name in ("producer_run_id", "outcome_id", "ledger_sha256"):
        value = getattr(obs, name)
        if value is not None or obs.status == "observed":
            _check(_digest(value) if name == "ledger_sha256" else _text(value), f"invalid {name}")
            identifiers[name] = value
    _check_unused(seen, identifiers)
    counts = (obs.assigned, obs.observed, obs.censored, obs.purchasing, obs.horizon_days)
    if request.reward_definition == SYNTHETIC:
        _check(all(v is None for v in counts), "bounded rewards cannot declare purchase counts or horizon")
    elif obs.status in ("observed", "censored") or any(v is not None for v in counts):
        _count(obs.assigned, "assigned", 1)
        _count(obs.observed, "observed")
        _count(obs.censored, "censored")
        _check(obs.horizon_days == 28 and obs.assigned == request.conditions["unit_size"]
               and obs.observed + obs.censored == obs.assigned, "invalid observation window or roster counts")
        if obs.purchasing is not None or obs.status == "observed":
            _count(obs.purchasing, "purchasing")
            _check(obs.purchasing <= obs.observed, "purchasing exceeds observed")
        if obs.status == "censored":
            _check(obs.censored > 0, "missing censored customers")
        if obs.status == "observed":
            _check(obs.censored == 0, "incomplete purchase observation")
    if obs.status != "observed":
        _check(obs.reward is None, "unsuccessful observation contains reward")
        return obs.status, None, identifiers
    if obs.valid_plan is not None or obs.approved_plan is not None:
        if obs.valid_plan is not True:
            return "invalid_plan", None, identifiers
        if obs.approved_plan is not True:
            return "unapproved_plan", None, identifiers
    if obs.decision_fingerprint is not None or obs.receipt is not None:
        _check(_digest(obs.decision_fingerprint) and _text(obs.receipt), "missing decision evidence")
    if request.reward_definition == SYNTHETIC:
        _bounded(obs.reward, "reward")
        value = Fraction(str(obs.reward))
    else:
        value = Fraction(obs.purchasing, obs.assigned)
        if obs.reward is not None:
            _check(_bounded(obs.reward, "reward") == float(value), "purchase count/reward mismatch")
    return "observed", value, identifiers


def _choose(method, stats, rng, epsilon, step):
    keys = list(stats)
    if method == "uniform":
        return keys[step % len(keys)], {}
    missing = [key for key in keys if not stats[key]["observations"]]
    if missing:
        return min(missing, key=lambda key: stats[key]["attempts"]), {}
    if method == "random" or (method == "epsilon_greedy" and rng.random() < epsilon):
        return rng.choice(keys), {}
    scores = {key: item["mean"] for key, item in stats.items()}
    if method == "ucb1":
        total = sum(item["observations"] for item in stats.values())
        for key in keys:
            scores[key] += math.sqrt(2 * math.log(total) / stats[key]["observations"])
    return max(keys, key=scores.get), {key: float(score) for key, score in scores.items()}


def run_policy_bandit(candidate_set, *, method, evaluation_units, evaluator, run_id, seed, max_attempts,
                      environment, conditions, reward_definition=PURCHASE, epsilon=None,
                      min_observations=1, max_retries=1):
    """Sequential evaluation. A retry is a fresh unit; stop after exhausting an arm's failure allowance.

    Conditions are shared declarations. Producers must isolate search customers from
    discovery, confirmation and final evaluation. Independent requests are held until
    the original approved PolicyRun, configuration, applied decision fingerprint,
    receipt, pre-state/roster, complete 28-day ledger, integrity and cohort separation
    can be verified. The synchronous callback has no execution timeout.
    """
    arms, report = _register(candidate_set)
    _check(_text(run_id) and method in METHODS, "invalid run ID or method")
    for name, value, minimum in (("seed", seed, 0), ("max_attempts", max_attempts, 1),
                                 ("min_observations", min_observations, 1), ("max_retries", max_retries, 0)):
        _count(value, name, minimum)
    _check(environment in ("synthetic", "independent") and
           (reward_definition == PURCHASE or environment == "synthetic" and reward_definition == SYNTHETIC),
           "unsupported environment/reward")
    conditions = deepcopy(conditions)
    _check(isinstance(conditions, dict), "invalid conditions")
    json.dumps(conditions, allow_nan=False)
    if reward_definition == PURCHASE:
        _count(conditions.get("unit_size"), "unit_size", 1)
    if environment == "independent":
        for name in ("population", "cost_definition", "coupon_terms", "solver"):
            _check(_text(conditions.get(name)), f"missing {name}")
        try:
            budget = Decimal(str(conditions.get("budget_per_customer")))
        except InvalidOperation as exc:
            raise InputValidationError("invalid budget_per_customer") from exc
        _check(budget.is_finite() and budget >= 0, "invalid budget_per_customer")
    if method == "epsilon_greedy":
        epsilon = _bounded(epsilon, "epsilon")
    else:
        _check(epsilon is None, "epsilon applies only to epsilon_greedy")
    _check(method == "uniform" or max_attempts >= len(arms), "insufficient initialization budget")
    by_id = {p.policy_id: p for p in arms}
    stats = {key: dict(attempts=0, observations=0, cumulative_reward=Fraction(0), mean=None,
                       failed=0, incomplete=0) for key in by_id}
    rng = random.Random(seed)
    used_units, observed_ids, records = {}, {}, []
    start = perf_counter()
    stop = "units_exhausted"
    if environment == "independent":
        stop = "producer_verification_unavailable"
        evaluation_units = ()
    for unit in islice(evaluation_units, max_attempts):
        key, scores = _choose(method, stats, rng, epsilon, len(records))
        policy = by_id[key]
        attempt_id = f"{run_id}/{len(records) + 1}"
        item = stats[key]
        item["attempts"] += 1
        tick = perf_counter()
        request = None
        evidence = {}
        try:
            _check(isinstance(unit, EvaluationUnit) and unit.role == "search", "requires search unit")
            _check(all(_text(s) for s in (unit.unit_id, unit.state_id, unit.dataset_id))
                   and _digest(unit.roster_fingerprint), "invalid evaluation unit")
            _count(unit.seed, "unit seed")
            json.dumps(unit.metadata, allow_nan=False)
            unit_keys = dict(unit=unit.unit_id, state=unit.state_id, roster=unit.roster_fingerprint)
            _check_unused(used_units, unit_keys)
            _register_ids(used_units, unit_keys)
            request = EvaluationRequest(run_id, attempt_id, deepcopy(unit), key, policy.version,
                                        policy.configuration_fingerprint(), environment, reward_definition, conditions)
            try:
                obs = evaluator(deepcopy(policy), deepcopy(request))
            except Exception as exc:
                obs = Observation(request, "external_failure", auxiliary={"error": str(exc)})
            status, value, identifiers = _observe(obs, request, observed_ids)
            evidence = asdict(obs)
            del evidence["request"]
            json.dumps(evidence, allow_nan=False)
        except (InputValidationError, ValueError, TypeError) as exc:
            status, value = "contract_violation", None
            evidence = {"error": str(exc)}
        if value is not None:
            _register_ids(observed_ids, identifiers)
            item["observations"] += 1
            item["cumulative_reward"] += value
            item["mean"] = item["cumulative_reward"] / item["observations"]
        elif status in ("pending", "censored"):
            item["incomplete"] += 1
        else:
            item["failed"] += 1
        records.append(dict(attempt_id=attempt_id, policy_id=key, status=status,
                            reward=float(value) if value is not None else None,
                            request=asdict(request) if request else None, evidence=evidence,
                            selection_scores=scores, elapsed_seconds=perf_counter() - tick))
        if item["failed"] + item["incomplete"] > max_retries:
            stop = "initialization_failed" if not item["observations"] else "arm_retry_limit"
            break
        if len(records) == max_attempts:
            stop = "attempt_limit"
    eligible = [key for key, item in stats.items() if item["observations"] >= min_observations]
    missing = [key for key in stats if key not in eligible]
    biased = any(item["failed"] or item["incomplete"] for item in stats.values())
    reason = "insufficient_observations" if missing else "incomplete_evidence" if biased else "highest_mean"
    recommended = max(eligible, key=lambda k: stats[k]["mean"]) if reason == "highest_mean" else None
    if environment == "independent":
        reason = stop
    # Keep exact fractions through selection and recommendation; export JSON numbers.
    total_reward = sum(item["cumulative_reward"] for item in stats.values())
    for item in stats.values():
        item["cumulative_reward"] = float(item["cumulative_reward"])
        if item["mean"] is not None:
            item["mean"] = float(item["mean"])
    return SearchResult(dict(report, method=method, environment=environment, conditions=conditions,
                             run_id=run_id, seed=seed, reward_definition=reward_definition, epsilon=epsilon,
                             max_attempts=max_attempts, max_retries=max_retries, min_observations=min_observations,
                             attempts=records, arm_statistics=stats, stop_reason=stop, recommendation=recommended,
                             recommendation_reason=reason, insufficient_arms=missing, independence="unverified",
                             attempt_count=len(records), observation_count=sum(s["observations"] for s in stats.values()),
                             cumulative_reward=float(total_reward),
                             failed_count=sum(s["failed"] for s in stats.values()),
                             incomplete_count=sum(s["incomplete"] for s in stats.values()),
                             elapsed_seconds=perf_counter() - start))


def compare_synthetic_search(candidate_set, *, true_means, seeds, max_attempts, epsilon, unit_size=1):
    """Compare Bernoulli cohort rewards using paired simulation blocks across methods only.

    Truth stays inside the environment and the retrospective regret calculation.
    Cumulative expected regret counts every attempted allocation, including failed ones.
    """
    arms, _ = _register(candidate_set)
    keys = [p.policy_id for p in arms]
    _check(set(true_means) == set(keys), "synthetic means must match registered arms")
    means = {key: _bounded(true_means[key], "true mean") for key in keys}
    _count(unit_size, "unit_size", 1)
    seeds = tuple(seeds)
    _check(seeds and len(set(seeds)) == len(seeds), "requires distinct seeds")
    best = max(means.values())
    results = []
    frequencies = {method: Counter() for method in METHODS}
    selection_counts = {method: Counter() for method in METHODS}

    def evaluate(policy, request):
        unit = request.unit
        env_rng = random.Random(f"environment/{unit.seed}/{unit.unit_id}")
        outcomes = bytes(env_rng.random() < means[policy.policy_id] for _ in range(unit_size))
        purchases = sum(outcomes)
        ledger = sha256(unit.roster_fingerprint.encode() + outcomes).hexdigest()
        return Observation(request, "observed", producer_run_id=request.attempt_id,
                           outcome_id=request.attempt_id, ledger_sha256=ledger,
                           assigned=unit_size, observed=unit_size, censored=0, purchasing=purchases, horizon_days=28)

    for seed in seeds:
        _count(seed, "seed")
        for method in METHODS:
            units = (EvaluationUnit(str(i), str(i), sha256(f"{seed}/{i}".encode()).hexdigest(),
                                    "synthetic", seed) for i in range(max_attempts))
            run = run_policy_bandit(candidate_set, method=method, evaluation_units=units, evaluator=evaluate,
                                    run_id=f"synthetic/{seed}/{method}", seed=seed, max_attempts=max_attempts,
                                    environment="synthetic", conditions={"unit_size": unit_size},
                                    epsilon=epsilon if method == "epsilon_greedy" else None).to_dict()
            chosen = run["recommendation"]
            run["simple_regret"] = None if chosen is None else best - means[chosen]
            run["cumulative_expected_regret"] = sum(best - means[row["policy_id"]] for row in run["attempts"])
            if chosen is not None:
                frequencies[method][chosen] += 1
            selection_counts[method].update(row["policy_id"] for row in run["attempts"])
            results.append(run)
    return SearchResult(dict(environment="synthetic", runs=results, true_means=means,
                             recommendation_counts={m: dict(freq) for m, freq in frequencies.items()},
                             selection_counts={m: dict(counts) for m, counts in selection_counts.items()},
                             common_random_blocks_across_methods=True, dispatch_authorized=False,
                             policy_effect_verified=False))
