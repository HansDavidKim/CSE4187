"""Independent budget scenarios over the existing allocation engine."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Context, Decimal, localcontext
import json
import math
import numbers
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import budget as engine
from .budget import Budget, BudgetComparison, CostMetadata, SolverConfig
from .segmentation import SegmentationResult
from .validation import CONTEXT_COLUMNS, InputValidationError


SCENARIOS = (("budget_050", Decimal("0.5")), ("budget_100", Decimal("1")),
             ("budget_200", Decimal("2")))
SUMMARY_FIELDS = (
    "algorithm", "algorithm_version", "selected_customer_count", "total_expected_cost", "remaining_budget",
    "objective_value", "status", "solver_status", "solver_success", "solver_message", "optimality",
    "optimality_gap", "gap_within_tolerance", "has_valid_solution", "accepted_for_planning", "budget_compliant",
    "solver_dual_bound", "objective_upper_bound", "solver_node_count", "solver_time", "execution_time",
)
CONTEXT = ("dataset_id", "as_of_at", "currency", "amount_unit")


@dataclass(frozen=True)
class WhatIfResult:
    summary: pd.DataFrame
    decisions: pd.DataFrame
    comparisons: pd.DataFrame
    metadata: dict
    results: dict[str, BudgetComparison]


def build_budget_scenarios(base_budget: Budget, cost_metadata: CostMetadata) -> dict[str, Budget]:
    """Create all three exact budgets; no allocation or implicit precision change."""
    if not isinstance(base_budget, Budget) or not isinstance(cost_metadata, CostMetadata):
        raise InputValidationError("what_if: Budget and CostMetadata required")
    places = cost_metadata.decimal_places
    if isinstance(places, (bool, np.bool_)) or not isinstance(places, numbers.Integral) or not 0 <= places <= engine.MAX_DECIMAL_PLACES:
        raise InputValidationError("what_if.decimal_places: integer from 0 through 9 required")
    # Internal validators are reused because budget has no public validation-only API.
    amount = engine._decimal(base_budget.amount, "base_budget.amount")
    engine._ticks(amount, int(places), "base_budget.amount")
    engine._text(base_budget.currency, "currency")
    engine._decimal(base_budget.amount_unit, "amount_unit", positive=True)
    amount = amount if amount else Decimal(0)
    budgets = {}
    with localcontext(Context(prec=max(64, len(amount.as_tuple().digits) + 2))):
        for identifier, multiplier in SCENARIOS:
            value = engine._decimal(amount * multiplier, f"{identifier}.amount")
            engine._ticks(value, int(places), f"{identifier}.amount")
            budgets[identifier] = replace(base_budget, amount=value)
    return budgets


def _reconcile(allocation):
    """Assert that published totals describe the actual customer decisions."""
    summary, rows = allocation.summary, allocation.decisions
    if not summary["has_valid_solution"]:
        candidates = rows.selection_reason.ne("upstream_excluded")
        if rows.loc[candidates, "selected"].notna().any():
            raise RuntimeError("what_if: invalid allocation has fabricated customer selections")
        return
    selected = rows.loc[rows.selected.eq(1)]
    cost = engine._decimal_sum(selected.expected_cost)
    score = engine._score_sum(selected.priority_score)
    if (len(selected) != summary["selected_customer_count"] or cost != summary["total_expected_cost"]
            or score != summary["objective_value"] or cost > summary["budget"]):
        raise RuntimeError("what_if: customer decisions disagree with allocation totals")


def run_what_if(segmentation: SegmentationResult, costs: pd.DataFrame, base_budget: Budget,
                cost_metadata: CostMetadata, *, dataset_id: str, as_of_at,
                config: SolverConfig | None = None, algorithm: str = "both", scenarios=None,
                hybrid_config=None, backend_time_budget=None) -> WhatIfResult:
    """Preflight every defined budget, then independently execute selected IDs.

    Input/preflight errors abort before allocation. Returned solver failures are
    retained; unexpected exceptions propagate. No prepared solver state is cached.
    """
    started = perf_counter()
    config = SolverConfig() if config is None else config
    if not isinstance(config, SolverConfig) or algorithm not in ("both", "milp", "greedy", "hybrid"):
        raise InputValidationError("what_if: invalid SolverConfig or algorithm")
    hybrid_config = engine._hybrid_config(algorithm, hybrid_config)
    if backend_time_budget is not None:
        if (algorithm != "hybrid" or hybrid_config.backend is None
                or isinstance(backend_time_budget, bool) or not isinstance(backend_time_budget, numbers.Real)
                or not math.isfinite(backend_time_budget) or backend_time_budget < 0):
            raise InputValidationError("what_if.backend_time_budget: finite nonnegative seconds require explicit Hybrid backend")
    backend_spent = 0.0
    definitions = dict(SCENARIOS)
    if scenarios is None:
        selected_ids = tuple(definitions)
    else:
        if not isinstance(scenarios, (list, tuple)) or not scenarios or any(not isinstance(item, str) for item in scenarios):
            raise InputValidationError("what_if.scenarios: nonempty list/tuple of scenario IDs required")
        selected_ids = tuple(scenarios)
        if len(set(selected_ids)) != len(selected_ids) or any(item not in definitions for item in selected_ids):
            raise InputValidationError("what_if.scenarios: unknown or duplicate scenario ID")
    budgets = build_budget_scenarios(base_budget, cost_metadata)
    # All three definitions are validated before execution. Only explicit Hybrid
    # retains call-owned prepared states, without sharing budget-dependent data.
    prepared_scenarios = {}
    for identifier, budget in budgets.items():
        try:
            tick = perf_counter()
            prepared = engine._prepare(segmentation, costs, cost_metadata, budget, dataset_id, as_of_at)
            if algorithm == "hybrid" and identifier in selected_ids:
                prepared_scenarios[identifier] = (prepared, perf_counter()-tick)
        except InputValidationError as exc:
            raise InputValidationError(f"what_if preflight {identifier}: {exc}") from exc
    preflight_time = perf_counter() - started
    results, summaries, decisions, comparisons = {}, [], [], []
    for identifier in selected_ids:
        if algorithm == "hybrid":
            prepared, preparation_time = prepared_scenarios.pop(identifier)
            scenario_config = hybrid_config
            if backend_time_budget is not None:
                scenario_config = replace(hybrid_config, external_timeout=min(hybrid_config.external_timeout,
                                           max(0.0, float(backend_time_budget)-backend_spent)))
            result = engine._execute_prepared(prepared, config, algorithm, scenario_config,
                                             preparation_time, perf_counter()-preparation_time)
            backend_spent += result.metadata["hybrid"].get("backend", {}).get("elapsed_seconds", 0.0)
        else:
            result = engine.optimize_budget(segmentation, costs, budgets[identifier], cost_metadata,
                                            dataset_id=dataset_id, as_of_at=as_of_at, config=config, algorithm=algorithm)
        results[identifier] = result
        prefix = {"scenario_id": identifier, "budget_multiplier": definitions[identifier], "budget": budgets[identifier].amount}
        comparison = {**prefix, **deepcopy(result.comparison),
                      "preparation_time": result.metadata["preparation_time"],
                      "scenario_execution_time": result.metadata["total_execution_time"]}
        comparisons.append(comparison)
        for allocation in result.results.values():
            _reconcile(allocation)
            record = {**prefix, **{key: result.metadata[key] for key in CONTEXT},
                      **{key: allocation.summary[key] for key in SUMMARY_FIELDS},
                      "objective_difference": comparison["objective_difference"],
                      "relative_improvement": comparison["relative_improvement"]}
            if algorithm == "hybrid":
                hybrid = result.metadata["hybrid"]
                evidence, selection = hybrid["quality"], hybrid["selection"]
                certificate = evidence["certificate"]
                record.update(final_selection_source=selection["source"], quality_status=evidence["status"],
                    local_improvement_executed=selection["local_improvement_executed"])
                if "backend" in hybrid:
                    backend = hybrid["backend"]
                    record.update(backend_requested=backend["requested"], backend_status=backend["status"],
                        backend_seconds=backend["elapsed_seconds"], backend_child_created=backend["child_created"],
                        backend_solver_executed=backend["solver_executed"],
                        initial_solution_submitted=backend["initial_solution_submitted"],
                        initial_solution_acceptance=backend["initial_solution_acceptance"],
                        backend_candidate_adopted=backend["candidate_adopted"])
                for field, value in (("quality_epsilon", evidence["epsilon"]),
                                     ("quality_relative_error", certificate["relative_error_bound"] if certificate else None),
                                     ("hybrid_objective_improvement", selection["improvement"]),
                                     ("quality_lower_bound", certificate["lower_bound"] if certificate else None),
                                     ("quality_upper_bound", certificate["upper_bound"] if certificate else None)):
                    for part in ("numerator", "denominator"):
                        record[f"{field}_{part}"] = value[part] if value else None
            with localcontext(Context(prec=34)):
                record["budget_utilization"] = (allocation.summary["total_expected_cost"] / prefix["budget"]
                                                if allocation.summary["has_valid_solution"] and prefix["budget"] else None)
            summaries.append(record)
            frame = allocation.decisions.copy(deep=True)
            for key, value in prefix.items():
                frame[key] = value
            decisions.append(frame)
    summary_frame = pd.DataFrame(summaries)
    for column in ("selected_customer_count", "solver_status", "solver_node_count"):
        summary_frame[column] = pd.array(summary_frame[column], dtype="Int64")
    decision_frame = pd.concat(decisions, ignore_index=True)
    decision_frame["selected"] = pd.array(decision_frame.selected, dtype="Int64")
    first = next(iter(results.values()))
    snapshots = next(iter(first.results.values())).decisions[list(CONTEXT_COLUMNS)].drop_duplicates()
    metadata = {
        "what_if_version": "budget-what-if-v1", "algorithm": algorithm, "executed_scenarios": list(selected_ids),
        "scenario_budgets": {key: {"budget_multiplier": definitions[key], "budget": value.amount} for key, value in budgets.items()},
        "selected_cohort": {key: first.metadata[key] for key in (*CONTEXT, "candidate_count", "excluded_count")},
        "selected_snapshots": snapshots.to_dict("records"),
        "source_segmentation": deepcopy(first.metadata["segmentation"]),
        "algorithm_versions": {key: value.summary["algorithm_version"] for key, value in first.results.items()},
        "scenario_metadata": {key: deepcopy({field: value for field, value in item.metadata.items() if field != "segmentation"})
                              for key, item in results.items()},
        "comparisons": deepcopy(comparisons), "preflight_time": preflight_time,
        "total_execution_time": perf_counter() - started,
        "interpretation": "priority proxy, not incremental revenue/profit or ROI; allocations are independent",
        "utilization_precision": "34 significant decimal digits; display ratio only, never a monetary feasibility check",
    }
    if algorithm == "hybrid" and hybrid_config.backend is not None:
        metadata.update(backend_time_budget=backend_time_budget, backend_time_spent=backend_spent,
                        backend_budget_scope="additional attempts only; cleanup/validation may exceed remaining allowance")
    return WhatIfResult(summary_frame, decision_frame, pd.DataFrame(comparisons), metadata, results)


def _json_ready(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise TypeError("what_if export: naive timestamp")
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"what_if export: unsupported metadata type {type(value).__name__}")


def _csv_text(frame):
    copied = frame.copy(deep=True)
    for column in copied:
        copied[column] = pd.Series([_json_ready(value) for value in copied[column]], index=copied.index, dtype=object)
    return copied.to_csv(index=False, na_rep="")


def export_what_if(result: WhatIfResult, output_dir: str | Path) -> dict[str, Path]:
    """Write three fixed names in a caller directory; never overwrite files."""
    directory = Path(output_dir)
    paths = {"summary": directory / "what_if_summary.csv", "decisions": directory / "what_if_decisions.csv",
             "metadata": directory / "what_if_metadata.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("what_if export preserves existing output files")
    content = {"summary": _csv_text(result.summary), "decisions": _csv_text(result.decisions),
               "metadata": json.dumps(_json_ready(result.metadata), ensure_ascii=False, indent=2, allow_nan=False)}
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = {}
            for key, path in paths.items():
                streams[key] = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
                created.append(path)
            for key, stream in streams.items():
                stream.write(content[key])
    except Exception:
        for path in created:
            path.unlink()
        raise
    return paths


def plot_what_if(result: WhatIfResult):
    """Three proxy/count/cost plots for a single cohort; unavailable is not zero."""
    from matplotlib.figure import Figure

    rows = result.summary
    cohort = result.metadata["selected_cohort"]
    for key in CONTEXT:
        if not rows[key].eq(cohort[key]).all():
            raise InputValidationError(f"what_if plot: mixed {key}")
    ids = [key for key, _ in SCENARIOS if key in set(rows.scenario_id)]
    positions = {key: index for index, key in enumerate(ids)}
    figure = Figure(figsize=(15, 5), layout="constrained")
    axes = figure.subplots(1, 3)
    metrics = (("selected_customer_count", "Selected customers"), ("total_expected_cost", "Expected intervention cost"),
               ("objective_value", "Selected priority proxy sum"))
    for axis, (column, title) in zip(axes, metrics):
        for algorithm, color in (("greedy", "#237B9B"), ("milp", "#BD5A27"), ("hybrid", "#487F35")):
            values = rows.loc[rows.algorithm.eq(algorithm)].set_index("scenario_id").reindex(ids)
            if values.algorithm.isna().all():
                continue
            y = [float(value) if pd.notna(value) else np.nan for value in values[column]]
            axis.plot(range(len(ids)), y, color=color, linestyle="-" if algorithm == "greedy" else "--", label=algorithm)
            for identifier, row in values.iterrows():
                if pd.isna(row[column]):
                    axis.annotate(f"{algorithm}: unavailable", (positions[identifier], .95), xycoords=("data", "axes fraction"),
                                  rotation=90, fontsize=7, va="top")
                    continue
                proven = row.optimality in ("numeric_optimal", "analytic_optimal")
                if algorithm == "hybrid":
                    markers = {"proven_optimal": "s", "epsilon_certified": "D", "feasible_uncertified": "x"}
                    axis.plot(positions[identifier], float(row[column]), color=color,
                              marker=markers.get(row.quality_status, "x"), linestyle="none", markersize=6)
                    axis.annotate(row.quality_status, (positions[identifier], float(row[column])),
                                  xytext=(0, 8), textcoords="offset points", fontsize=6)
                    continue
                axis.plot(positions[identifier], float(row[column]), color=color,
                          marker="o" if algorithm == "greedy" else "s" if proven else "x", linestyle="none",
                          markersize=9 if algorithm == "greedy" else 5,
                          markerfacecolor="none" if algorithm == "greedy" else color)
        labels = [f"{dict(SCENARIOS)[key] * 100}%\n{result.metadata['scenario_budgets'][key]['budget']}" for key in ids]
        axis.set_xticks(range(len(ids)), labels)
        axis.set(title=title, xlabel="Budget scenario")
        axis.legend(fontsize=8)
    legend = ("Hybrid squares: exact optimal; diamonds: epsilon certified; crosses: feasible uncertified"
              if result.metadata["algorithm"] == "hybrid"
              else "MILP squares: numeric/analytic optimal; crosses: unproven incumbent")
    figure.suptitle(f"{cohort['dataset_id']} | {cohort['as_of_at']} | {cohort['currency']}; amount unit {cohort['amount_unit']}\n" + legend)
    return figure
