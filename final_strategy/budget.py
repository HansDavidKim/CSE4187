"""One-cohort MILP allocation and a Greedy baseline using declared external costs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from functools import reduce
import math
import numbers
import os
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy import optimize, sparse

from .segmentation import SegmentationResult, SEGMENT_COLUMNS
from .validation import CONTEXT_COLUMNS, JOIN_KEYS, PREDICTION_COLUMNS, InputValidationError


MAX_DECIMAL_PLACES = 9
MAX_AMOUNT = Decimal("1e18")
MAX_EXACT_INTEGER = 2**53 - 1
MAX_MATRIX_COEFFICIENT = 10**12
MIN_OBJECTIVE_COEFFICIENT = 1e-9
COST_COLUMNS = (*JOIN_KEYS, "snapshot_id", "raw_database_sha256", "currency", "amount_unit", "expected_cost")
ROW_COLUMNS = (*CONTEXT_COLUMNS, *PREDICTION_COLUMNS.values(), "priority_score", *SEGMENT_COLUMNS)
VERSIONS = {"milp": "budget-milp-v1", "greedy": "budget-greedy-density-v1"}


@dataclass(frozen=True)
class Budget:
    """Amount and expected_cost use counts of the same declared amount_unit."""

    amount: object
    currency: str
    amount_unit: object


@dataclass(frozen=True)
class CostMetadata:
    """Internal declaration; no coupon cost estimation or external API is defined."""

    definition_id: str
    version: str
    decimal_places: int
    as_of_at: object


@dataclass(frozen=True)
class SolverConfig:
    time_limit: float = 10.0
    mip_rel_gap: float = 0.0
    binary_tolerance: float = 1e-7
    allow_unproven: bool = False
    presolve: bool = True

    def __post_init__(self):
        for field, upper in (("time_limit", None), ("mip_rel_gap", 1), ("binary_tolerance", 1e-6)):
            value = getattr(self, field)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value) or value < 0 or (upper is not None and value > upper)):
                raise InputValidationError(f"budget.{field}: invalid finite nonnegative solver option")
        if not isinstance(self.allow_unproven, bool):
            raise InputValidationError("budget.allow_unproven: requires bool")
        if not isinstance(self.presolve, bool):
            raise InputValidationError("budget.presolve: requires bool")


@dataclass(frozen=True)
class AllocationResult:
    decisions: pd.DataFrame
    summary: dict


@dataclass(frozen=True)
class BudgetComparison:
    results: dict[str, AllocationResult]
    comparison: dict
    metadata: dict


def _clock(value, name):
    try:
        point = pd.Timestamp(value)
        if pd.isna(point) or point.tzinfo is None:
            raise ValueError("unknown/naive clock")
        return point.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"budget.{name}: requires timezone-aware timestamp") from exc


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"budget.{name}: requires nonempty string")
    return value


def _decimal(value, name, positive=False):
    try:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, Decimal, numbers.Real)):
            raise ValueError("unsupported decimal value")
        # For floats this preserves str(float), not an unknown original decimal.
        result = value if isinstance(value, Decimal) else Decimal(str(value))
        if not result.is_finite() or result < 0 or (positive and result == 0):
            raise ValueError("invalid sign or non-finite")
        if result > MAX_AMOUNT or len(result.as_tuple().digits) > 64:
            raise ValueError("unsupported decimal magnitude/length")
        return result
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise InputValidationError(f"budget.{name}: requires finite nonnegative decimal <= 1e18 (at most 64 digits)") from exc


def _ticks(value, places, name):
    if value and value.adjusted() < -places:
        raise InputValidationError(f"budget.{name}: cannot represent exactly at decimal_places={places}")
    numerator, denominator = value.as_integer_ratio()
    integer, remainder = divmod(numerator * 10**places, denominator)
    if remainder:
        raise InputValidationError(f"budget.{name}: cannot represent exactly at decimal_places={places}")
    return integer


def _decimal_sum(values):
    # Declared values have <=18 integral and <=9 fractional places. This covers
    # exact sums even for very large in-memory populations without context rounding.
    values = list(values)
    with localcontext() as context:
        context.prec = 64 + len(str(len(values)))
        return sum(values, Decimal(0))


def _score_sum(values):
    try:
        result = math.fsum(float(value) for value in values)
        if not math.isfinite(result):
            raise ValueError("non-finite sum")
        return result
    except (ValueError, OverflowError) as exc:
        raise InputValidationError("budget.priority_score: unsupported aggregate magnitude") from exc


def _table(frame, required, name):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise InputValidationError(f"budget.{name}: requires DataFrame without duplicate columns")
    missing = set(required) - set(frame)
    if missing:
        raise InputValidationError(f"budget.{name}: missing columns {sorted(missing)}")
    if frame.attrs.get("evaluation_only") is not None or "evaluation_only" in frame:
        raise InputValidationError(f"budget.{name}: evaluation_only input is forbidden")
    copied = frame[list(required)].copy(deep=True).reset_index(drop=True)
    for column in ("dataset_id", "customer_id"):
        if not copied[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise InputValidationError(f"budget.{name}.{column}: requires string IDs")
    copied["as_of_at"] = pd.Series([_clock(value, f"{name}.as_of_at") for value in copied.as_of_at],
                                  index=copied.index, dtype="datetime64[ns, UTC]")
    if copied.duplicated(list(JOIN_KEYS)).any():
        raise InputValidationError(f"budget.{name}: duplicate customer/snapshot keys")
    return copied


def _prepare(segmentation, costs, cost_metadata, budget, dataset_id, as_of_at):
    preparation_started = perf_counter()
    if not isinstance(segmentation, SegmentationResult):
        raise InputValidationError("budget: requires SegmentationResult")
    if not isinstance(cost_metadata, CostMetadata) or not isinstance(budget, Budget):
        raise InputValidationError("budget: requires CostMetadata and Budget")
    dataset_id, point = _text(dataset_id, "dataset_id"), _clock(as_of_at, "as_of_at")
    for field in ("definition_id", "version"):
        _text(getattr(cost_metadata, field), f"cost_metadata.{field}")
    places = cost_metadata.decimal_places
    if isinstance(places, bool) or not isinstance(places, numbers.Integral) or not 0 <= places <= MAX_DECIMAL_PLACES:
        raise InputValidationError("budget.cost_metadata.decimal_places: integer from 0 through 9 required")
    places = int(places)
    cost_clock = _clock(cost_metadata.as_of_at, "cost_metadata.as_of_at")
    if cost_clock > point:
        raise InputValidationError("budget.cost_metadata.as_of_at: future cost information")
    currency = _text(budget.currency, "currency")
    unit = _decimal(budget.amount_unit, "amount_unit", positive=True)
    amount = _decimal(budget.amount, "amount")
    budget_ticks = _ticks(amount, places, "amount")
    metadata = deepcopy(segmentation.metadata)
    if not isinstance(metadata, dict) or metadata.get("population_scope") not in ("validated_inputs", "explicit_eligibility"):
        raise InputValidationError("budget.segmentation.metadata: population_scope required")
    if not isinstance(metadata.get("config"), dict):
        raise InputValidationError("budget.segmentation.metadata: config required")
    _text(metadata["config"].get("segmentation_version"), "segmentation_version")
    if not isinstance(metadata.get("models"), dict) or set(metadata["models"]) != set(PREDICTION_COLUMNS):
        raise InputValidationError("budget.segmentation.metadata: all model declarations required")
    for model, declaration in metadata["models"].items():
        if not isinstance(declaration, dict):
            raise InputValidationError(f"budget.{model}: invalid model declaration")
        for field in ("model_version", "target_definition_id", "horizon_unit", "value_semantics"):
            _text(declaration.get(field), f"{model}.{field}")
        horizon = declaration.get("horizon_value")
        if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon <= 0:
            raise InputValidationError(f"budget.{model}.horizon_value: positive integer required")
    all_rows = _table(segmentation.customers, ROW_COLUMNS, "customers")
    excluded = _table(segmentation.excluded, (*CONTEXT_COLUMNS, "exclusion_reason"), "excluded")
    choose = lambda frame: frame.loc[frame.dataset_id.eq(dataset_id) & frame.as_of_at.eq(point)].copy()
    rows, excluded = choose(all_rows), choose(excluded)
    if (len(all_rows) or len(segmentation.excluded)) and rows.empty and excluded.empty:
        raise InputValidationError("budget: requested dataset/as_of_at cohort not found")
    overlap = pd.MultiIndex.from_frame(rows[list(JOIN_KEYS)]).intersection(pd.MultiIndex.from_frame(excluded[list(JOIN_KEYS)]))
    if len(overlap):
        raise InputValidationError("budget: customers and excluded overlap")
    if not excluded.exclusion_reason.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise InputValidationError("budget.excluded.exclusion_reason: explicit reason required")
    cost_frame = _table(costs, COST_COLUMNS, "costs")
    if set(costs) != set(COST_COLUMNS):
        raise InputValidationError("budget.costs: unexpected columns in internal cost table")
    if not (cost_frame.dataset_id.eq(dataset_id) & cost_frame.as_of_at.eq(point)).all():
        raise InputValidationError("budget.costs: mixed dataset/as_of_at")
    candidate_keys, cost_keys = (pd.MultiIndex.from_frame(frame[list(JOIN_KEYS)]) for frame in (rows, cost_frame))
    if len(candidate_keys.difference(cost_keys)) or len(cost_keys.difference(candidate_keys)):
        raise InputValidationError("budget.costs: missing or unexpected customer/snapshot keys")
    for frame, name in ((rows, "customers"), (excluded, "excluded"), (cost_frame, "costs")):
        if not frame.currency.eq(currency).all():
            raise InputValidationError(f"budget.{name}.currency: differs from Budget")
        units = [_decimal(value, f"{name}.amount_unit", positive=True) for value in frame.amount_unit]
        if any(value != unit for value in units):
            raise InputValidationError(f"budget.{name}.amount_unit: differs from Budget")
        for column in ("snapshot_id", "raw_database_sha256"):
            if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise InputValidationError(f"budget.{name}.{column}: missing provenance")
    if pd.concat([rows.raw_database_sha256, excluded.raw_database_sha256]).nunique() > 1:
        raise InputValidationError("budget.raw_database_sha256: mixed source database")
    joined = rows[list(ROW_COLUMNS)].merge(cost_frame, on=list(JOIN_KEYS), validate="one_to_one", suffixes=("", "_cost"))
    for column in set(COST_COLUMNS) - set(JOIN_KEYS) - {"amount_unit", "expected_cost"}:
        if not joined[column].eq(joined[f"{column}_cost"]).all():
            raise InputValidationError(f"budget.costs.{column}: differs from SegmentationResult")
    joined = joined[[*ROW_COLUMNS, "expected_cost"]].sort_values("customer_id", kind="stable").reset_index(drop=True)
    for column in (*PREDICTION_COLUMNS.values(), "priority_score"):
        if joined[column].map(lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating, pd.Timestamp, np.datetime64))).any():
            raise InputValidationError(f"budget.{column}: finite real score required")
        try:
            values = pd.to_numeric(joined[column], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise InputValidationError(f"budget.{column}: finite real numeric values required") from exc
        if not np.isfinite(values).all():
            raise InputValidationError(f"budget.{column}: finite real numeric values required")
        joined[column] = values
    if (not joined.churn_probability.between(0, 1).all() or not joined.uplift_score.between(-1, 1).all()
            or joined.predicted_clv_12m.lt(0).any()):
        raise InputValidationError("budget: invalid prediction range")
    expected_scores = joined.uplift_score * joined.predicted_clv_12m
    if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=0) for a, b in zip(joined.priority_score, expected_scores)):
        raise InputValidationError("budget.priority_score: differs from uplift_score * predicted_clv_12m")
    validated_at = perf_counter()
    return _prepare_allocation(
        joined, excluded, amount, places, budget_ticks, cost_metadata, cost_clock,
        dataset_id, point, currency, unit, metadata, preparation_started, validated_at,
        objective_column="priority_score")


def _prepare_allocation(joined, excluded, amount, places, budget_ticks, cost_metadata, cost_clock,
                        dataset_id, point, currency, unit, metadata, preparation_started, validated_at,
                        *, objective_column):
    """Scale costs and objective coefficients after caller-side input validation."""
    declared_costs = [_decimal(value, "expected_cost") for value in joined.expected_cost]
    ticks = [_ticks(value, places, "expected_cost") for value in declared_costs]
    joined["expected_cost"] = declared_costs
    scores = joined[objective_column].to_numpy(dtype=float, copy=True)
    customer_ids = joined.customer_id.to_numpy(copy=True)
    active = [i for i, (value, tick) in enumerate(zip(scores, ticks)) if value > 0 and 0 < tick <= budget_ticks]
    free = [i for i, (value, tick) in enumerate(zip(scores, ticks)) if value > 0 and tick == 0]
    # Preserve the legacy metadata definition (gcd including the budget).
    common_divisor = reduce(math.gcd, [budget_ticks, *[ticks[i] for i in active]], 0) or 1
    # Integer spending is a multiple of the cost gcd. Flooring the integer RHS
    # is therefore equivalent for binary x; original Decimal amounts stay intact.
    divisor = reduce(math.gcd, (ticks[i] for i in active), 0) if active else common_divisor
    scaled_budget = budget_ticks // divisor
    scaled_costs = [ticks[i] // divisor for i in active]
    if (scaled_budget > MAX_EXACT_INTEGER or sum(scaled_costs) > MAX_EXACT_INTEGER
            or any(tick > MAX_MATRIX_COEFFICIENT for tick in scaled_costs)):
        raise InputValidationError("budget: integer scale exceeds supported safe coefficient range")
    active_scores = scores[active]
    scale = float(active_scores.max()) if len(active_scores) else 1.0
    objectives = active_scores / scale
    if len(objectives) and min(objectives) < MIN_OBJECTIVE_COEFFICIENT:
        raise InputValidationError(f"budget.{objective_column}: unsupported objective coefficient dynamic range")
    _score_sum(scores[scores > 0])
    return {"rows": joined, "excluded": excluded.sort_values("customer_id", kind="stable").reset_index(drop=True),
            "preparation_timings": {"input_validation_seconds": validated_at-preparation_started,
                                    "numeric_preparation_seconds": perf_counter()-validated_at},
            "costs": declared_costs, "ticks": ticks, "scores": scores, "customer_ids": customer_ids,
            "budget_ticks": budget_ticks, "budget": amount,
            "active": active, "free": free, "scaled_costs": scaled_costs, "scaled_budget": scaled_budget,
            "objective_coefficients": objectives, "objective_scale": scale,
            "metadata": {"dataset_id": dataset_id, "as_of_at": point.isoformat(), "currency": currency,
                         "amount_unit": str(unit), "cost": {**asdict(cost_metadata), "as_of_at": cost_clock.isoformat()},
                         "segmentation": metadata, "candidate_count": len(joined), "excluded_count": len(excluded),
                         "tick_scale": 10**places, "integer_gcd": common_divisor,
                         "constraint_cost_gcd": divisor if active else None,
                         "constraint_budget_tick_remainder": budget_ticks % divisor if active else None,
                         "constraint_reduction": "cost_gcd_floor_rhs" if active else "no_active_variables",
                         "scipy_version": scipy.__version__,
                         "objective_scale": scale, "solver_variable_count": len(active),
                         "score_consistency_relative_tolerance": 1e-12,
                         "interpretation": "planning proxy, not ROI/profit or authorization to send coupons"}}


def _memory():
    """Actual process RSS/peak on Windows; includes native libraries and baseline."""
    if os.name != "nt":
        return {"method": "unavailable"}
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(current(), ctypes.byref(counters), counters.cb):
            return {"method": "unavailable"}
        return {"method": "windows_process_working_set", "working_set_bytes": counters.working,
                "peak_working_set_bytes": counters.peak, "scope": "whole_process_not_algorithm_attributable"}
    except (AttributeError, OSError):
        return {"method": "unavailable"}


def _raw_number(result, field):
    value = getattr(result, field, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _solve_milp(prepared, config):
    solver = getattr(optimize, "milp", None)
    if solver is None:
        raise RuntimeError("scipy.optimize.milp unavailable; SciPy >= 1.9 is required")
    n = len(prepared["active"])
    return solver(c=-prepared["objective_coefficients"], integrality=np.ones(n, dtype=int),
                  bounds=optimize.Bounds(np.zeros(n), np.ones(n)),
                  constraints=optimize.LinearConstraint(sparse.csc_array(np.array([prepared["scaled_costs"]], dtype=float)),
                                                        -np.inf, prepared["scaled_budget"]),
                  options={"time_limit": float(config.time_limit), "mip_rel_gap": float(config.mip_rel_gap),
                           "presolve": config.presolve})


def _execute(prepared, algorithm, config):
    start, memory_before = perf_counter(), _memory()
    rows = prepared["rows"]
    scores, customer_ids = prepared["scores"], prepared["customer_ids"]
    selected = np.zeros(len(rows), dtype=int)
    selected[prepared["free"]] = 1
    reasons = np.array(["non_positive_score" if score <= 0 else "cost_exceeds_budget" for score in scores], dtype=object)
    reasons[prepared["free"]] = "selected_zero_cost"
    raw, valid, accepted, status, message = None, True, True, "optimal_preprocessed", "No solver variables remain"
    solver_time = 0.0
    if algorithm == "greedy":
        status, message = "baseline_feasible", "Greedy baseline; global optimality is not guaranteed"
        order = sorted(prepared["active"], key=lambda i: (
            -Fraction(Decimal(str(scores[i]))) / Fraction(prepared["costs"][i]),
            -scores[i], customer_ids[i]))
        remaining = prepared["budget_ticks"]
        for index in order:
            if prepared["ticks"][index] <= remaining:
                selected[index] = 1
                remaining -= prepared["ticks"][index]
                reasons[index] = "selected_greedy"
            else:
                reasons[index] = "insufficient_remaining_budget"
    elif prepared["active"]:
        try:
            solver_start = perf_counter()
            raw = _solve_milp(prepared, config)
            solver_time = perf_counter() - solver_start
            raw_status = getattr(raw, "status", None)
            message = str(getattr(raw, "message", "No solver message"))
            if raw_status not in (0, 1):
                status = {2: "infeasible", 3: "unbounded", 4: "solver_error"}.get(raw_status, "solver_error")
                valid = accepted = False
            elif getattr(raw, "x", None) is None:
                status, valid, accepted = "limit_no_incumbent" if raw_status == 1 else "solver_error", False, False
            else:
                vector = np.asarray(raw.x, dtype=float)
                binary = np.rint(vector)
                if (vector.shape != (len(prepared["active"]),) or not np.isfinite(vector).all()
                        or not np.isin(binary, [0, 1]).all()
                        or not np.less_equal(np.abs(vector - binary), config.binary_tolerance).all()):
                    status, valid, accepted = "invalid_binary_solution", False, False
                else:
                    selected[prepared["active"]] = binary.astype(int)
                    reasons[prepared["active"]] = np.where(binary == 1, "selected_milp", "not_selected_in_returned_solution")
                    gap, bound, fun = (_raw_number(raw, field) for field in ("mip_gap", "mip_dual_bound", "fun"))
                    actual_fun = -float(prepared["objective_coefficients"] @ binary)
                    certified = (raw_status == 0 and getattr(raw, "success", False) is True
                                 and gap == 0 and bound is not None and fun is not None
                                 and math.isfinite(bound) and math.isfinite(fun)
                                 and math.isclose(fun, actual_fun, rel_tol=1e-9, abs_tol=1e-9)
                                 and math.isclose(bound, actual_fun, rel_tol=1e-9, abs_tol=1e-9))
                    status = ("optimal_numeric" if certified else "limit_feasible" if raw_status == 1
                              else "gap_feasible" if gap is not None and math.isfinite(gap) and gap > 0
                              else "feasible_uncertified")
                    accepted = certified or config.allow_unproven
        except Exception as exc:
            solver_time = perf_counter() - solver_start
            status, valid, accepted, message = "solver_error", False, False, f"{type(exc).__name__}: {exc}"
    total = _decimal_sum(cost for cost, choice in zip(prepared["costs"], selected) if choice) if valid else None
    if valid and total > prepared["budget"]:
        status, valid, accepted = "invalid_budget_solution", False, False
        message = f"Exact Decimal cost {total} exceeds budget {prepared['budget']}; no repair applied"
    objective = _score_sum(score for score, choice in zip(scores, selected) if choice) if valid else None
    decisions = rows.copy(deep=True)
    decisions["exclusion_reason"] = pd.NA
    decisions["selected"] = pd.array(selected if valid else [pd.NA] * len(rows), dtype="Int64")
    decisions["selection_reason"] = reasons if valid else status
    excluded = prepared["excluded"].reindex(columns=[*ROW_COLUMNS, "exclusion_reason"]).copy()
    excluded["expected_cost"], excluded["selected"], excluded["selection_reason"] = pd.NA, 0, "upstream_excluded"
    decisions = pd.concat([decisions, excluded], ignore_index=True)
    decisions["algorithm"] = algorithm
    gap, bound = (_raw_number(raw, field) if raw is not None else None for field in ("mip_gap", "mip_dual_bound"))
    free_score = _score_sum(scores[i] for i in prepared["free"])
    upper_bound = None if bound is None or not math.isfinite(bound) else free_score - bound * prepared["objective_scale"]
    summary = {"algorithm": algorithm, "algorithm_version": VERSIONS[algorithm], "status": status,
               "solver_status": getattr(raw, "status", None), "solver_success": getattr(raw, "success", None),
               "solver_message": message, "has_valid_solution": valid, "accepted_for_planning": accepted,
               "optimality": "numeric_optimal" if status == "optimal_numeric" else "analytic_optimal" if status == "optimal_preprocessed"
               else "not_proven" if algorithm == "milp" else "not_applicable",
               "objective_value": objective, "selected_customer_count": int(selected.sum()) if valid else None,
               "total_expected_cost": total if valid else None,
               "remaining_budget": prepared["budget"] - total if valid else None, "budget": prepared["budget"],
               "budget_compliant": True if valid else None,
               "gap_within_tolerance": (bool(0 <= gap <= config.mip_rel_gap)
                                        if valid and gap is not None and math.isfinite(gap) else None),
               "optimality_gap": gap, "solver_dual_bound": bound, "objective_upper_bound": upper_bound,
               "solver_node_count": getattr(raw, "mip_node_count", None), "solver_time": solver_time,
               "execution_time": perf_counter() - start, "memory_before": memory_before, "memory_after": _memory()}
    return AllocationResult(decisions, summary)


def optimize_budget(segmentation: SegmentationResult, costs: pd.DataFrame, budget: Budget,
                    cost_metadata: CostMetadata, *, dataset_id: str, as_of_at,
                    config: SolverConfig | None = None, algorithm: str = "both", hybrid_config=None) -> BudgetComparison:
    """Compare independent plans; no allocation authorizes an actual coupon send.

    An unproven incumbent is retained but not accepted_for_planning by default.
    Missing/invalid MILP plans have nullable selections, never fabricated zeros.
    """
    start = perf_counter()
    config = SolverConfig() if config is None else config
    if not isinstance(config, SolverConfig) or algorithm not in ("both", "milp", "greedy", "hybrid"):
        raise InputValidationError("budget: invalid SolverConfig or algorithm")
    hybrid_config = _hybrid_config(algorithm, hybrid_config)
    prepared = _prepare(segmentation, costs, cost_metadata, budget, dataset_id, as_of_at)
    preparation_time = perf_counter() - start
    return _execute_prepared(prepared, config, algorithm, hybrid_config, preparation_time, start)


def _hybrid_config(algorithm, config):
    if algorithm != "hybrid":
        if config is not None:
            raise InputValidationError("budget.hybrid_config: requires algorithm='hybrid'")
        return None
    # Local import avoids budget -> quality -> budget initialization cycles.
    from .hybrid import HybridConfig
    if config is None:
        return HybridConfig()
    if not isinstance(config, HybridConfig):
        raise InputValidationError("budget.hybrid_config: requires HybridConfig")
    config.__post_init__()
    return config


def _execute_prepared(prepared, config, algorithm, hybrid_config, preparation_time, start):
    """Private execution of call-owned validated inputs; never accepts external cached state."""
    algorithms = ("milp", "greedy") if algorithm == "both" else (algorithm,)
    execution_order = ("greedy", "milp") if algorithm == "both" else algorithms
    hybrid_metadata = None
    if algorithm == "hybrid":
        from .hybrid import _run_hybrid
        allocation, hybrid_metadata = _run_hybrid(prepared, config, hybrid_config)
        executed = {"hybrid": allocation}
        execution_order = tuple(item["stage"] for item in hybrid_metadata["attempts"] if item["status"] != "skipped")
    else:
        executed = {name: _execute(prepared, name, config) for name in execution_order}
    results = {name: executed[name] for name in algorithms}
    comparison = {"objective_difference": None, "relative_improvement": None,
                  "selected_customer_count_difference": None, "total_expected_cost_difference": None,
                  "execution_time_difference": None, "memory_usage_difference": None,
                  "memory_comparison_reason": "shared process peaks are not independently attributable"}
    if len(results) == 2:
        milp, greedy = (results[name].summary for name in algorithms)
        comparison["execution_time_difference"] = milp["execution_time"] - greedy["execution_time"]
        if milp["has_valid_solution"] and greedy["has_valid_solution"]:
            delta = milp["objective_value"] - greedy["objective_value"]
            comparison.update(objective_difference=delta,
                              relative_improvement=delta / greedy["objective_value"] if greedy["objective_value"] > 0 else None,
                              selected_customer_count_difference=milp["selected_customer_count"] - greedy["selected_customer_count"],
                              total_expected_cost_difference=milp["total_expected_cost"] - greedy["total_expected_cost"])
    metadata = deepcopy(prepared["metadata"])
    metadata.update(solver_config=asdict(config), execution_order=list(execution_order),
                    preparation_time=preparation_time, total_execution_time=perf_counter() - start,
                    comparison_interpretation="priority-score proxy difference, not ROI or incremental profit")
    if hybrid_metadata is not None:
        hybrid_metadata["timings"].update(prepared["preparation_timings"])
        hybrid_metadata["timings"].update(input_preparation_seconds=preparation_time,
                                         total_api_seconds=metadata["total_execution_time"])
        first = hybrid_metadata["timings"].get("first_feasible_seconds")
        hybrid_metadata["timings"]["first_feasible_from_api_start_seconds"] = None if first is None else first+preparation_time
        metadata["hybrid"] = hybrid_metadata
    return BudgetComparison(results, comparison, metadata)
