"""Execute the five-arm ledger contract with one shared Simulator replay.

The input provider sees the validated baseline Features and strict-past CLV
history. It returns the weekly prediction batches and the official planning
arguments (profile, reporting, eligibility, history and cost declarations).
As with single-policy replay, the same-process callable is trusted.
"""
from dataclasses import asdict, fields, replace
from decimal import Context, localcontext
from numbers import Integral
from pathlib import Path
from random import Random
from time import perf_counter
from uuid import uuid4

import pandas as pd
from scipy import __version__ as scipy_version

from simulate.v4.configuration import POPULATION_CONFIG
from simulate.v4.simulator.engine import EcommerceSimulator

from .budget import (MAX_DECIMAL_PLACES, Budget, CostMetadata, SolverConfig, _clock, _decimal, _decimal_sum,
                     _execute, _prepare_allocation, _ticks)
from .experiment_analysis import analyze_experiment
from .experiment_power import PowerPlan, PowerScenario
from .experiment_spec import ARMS, ExperimentSpec
from .experiment_validation import CONTEXT, CONTRACT_VERSION, LEDGERS, _quotas, validate_experiment_run
from .pipeline import write_outputs
from .policy_execution import (_compare_replay, _generate, _json, _outcomes, _plan_manifest,
                               _publish, _frozen_plan, _read_generation, _same, file_sha256)
from .policy_search import _decision_fingerprint, _fingerprint, _json_value
from .retention_strategy import (COST_INPUT_COLUMNS, ELIGIBILITY_COLUMNS, _aligned, _coupon_eligibility,
                                 _history, _price_cost_inputs)
from .segmentation import segment_customers
from .validation import CONTEXT_COLUMNS
from .weekly_planning import _build_strategy_decisions, _cost_declarations
from .weekly_prediction_adapter import WeeklyStrategyInputs, adapt_weekly_predictions
from .weekly_validation import require


MODE = "simulation_verification"
STATUS = "five_arm_contract_verified"
RANDOM_VERSION = "random_order_skip_over_budget_v1"
UPLIFT_VERSION = "positive_uplift_milp_v1"
CLASSIFICATION = ("churn_band", "uplift_band", "clv_band", "segment_id", "priority_score")
REFERENCE_FIELDS = (*CONTEXT_COLUMNS, *CLASSIFICATION, "is_high_clv", "priority_rank",
                    "churn_probability", "uplift_score", "predicted_clv_12m")
DECISION_FIELDS = (*CONTEXT_COLUMNS, "arm_id", "candidate", "selected", "send", "planned_sent_at",
                   "decision_at", "expected_cost", "row_state")
FROZEN_FILES = ("planning_inputs.parquet", "segmentation_reference.parquet", "common_costs.parquet",
                "ledgers/population.parquet", "ledgers/assignments.parquet")


def _assign(eligible, spec, seed):
    require(type(seed) is int and seed >= 0, "strategy assignment seed must be a nonnegative integer")
    ids = sorted(eligible)
    require(len(ids) == len(set(ids)), "duplicate eligible customers")
    quotas = _quotas(tuple(arm for arm, _ in spec.arms), spec.allocation_weights, len(ids))
    require(all(quotas.values()), "contract verification requires five nonempty arms")
    Random(seed).shuffle(ids)
    assignment = {}
    offset = 0
    for arm, quota in quotas.items():
        for uid in ids[offset:offset+quota]:
            assignment[uid] = (arm, quota / len(ids))
        offset += quota
    return assignment


def _scope_handoff(handoff, ids):
    contents = handoff.snapshot()
    roster = contents.roster.loc[contents.roster.customer_id.isin(ids)].copy()
    require(set(roster.customer_id) == set(ids), "unknown arm customer")
    excluded = contents.excluded.loc[contents.excluded.customer_id.isin(ids)].copy()
    integrated = contents.integrated
    if integrated is not None:
        data = integrated.data.loc[integrated.data.customer_id.isin(ids)].copy()
        context = {key: frame.loc[frame.customer_id.isin(ids)].copy()
                   for key, frame in integrated.prediction_context.items()}
        integrated = replace(integrated, data=data, prediction_context=context) if len(data) else None
    return WeeklyStrategyInputs(replace(contents, roster=roster, excluded=excluded, integrated=integrated))


def _prepare_common(handoff, options):
    roster = handoff.roster
    first = roster.iloc[0]
    eligibility = _aligned(options["eligibility"], ELIGIBILITY_COLUMNS, roster, "experiment eligibility")
    costs, metadata, evidence = _cost_declarations(
        first, options.get("cost_estimate"), options.get("cost_inputs"), options.get("cost_metadata"))
    require(costs is not None, "missing common experiment costs")
    costs = _aligned(costs, COST_INPUT_COLUMNS, roster, "experiment costs")
    priced = _price_cost_inputs(costs, metadata, first.as_of_at)
    require(isinstance(metadata.decimal_places, Integral) and not isinstance(metadata.decimal_places, bool)
            and 0 <= metadata.decimal_places <= MAX_DECIMAL_PLACES,
            "invalid common cost precision")
    for cost in priced.expected_cost:
        _ticks(cost, metadata.decimal_places, "common expected cost")
    history_policy = options["history_policy"]
    coupons, redemptions, interval = _history(
        options["coupons"], options["redemptions"], options["history_context"], history_policy,
        first.dataset_id, first.raw_database_sha256, first.as_of_at)
    safety = {}
    for row in eligibility.itertuples(index=False):
        eligible, reason, *_ = _coupon_eligibility(
            row, row.customer_id, coupons, redemptions, interval, first.as_of_at, history_policy)
        safety[row.customer_id] = (eligible is True, reason)
    return eligibility, costs, priced, metadata, evidence, safety


def _random_selection(candidates, budget, places, seed):
    require(type(seed) is int and seed >= 0, "Random policy seed must be a nonnegative integer")
    ordered = sorted(candidates.customer_id)
    Random(seed).shuffle(ordered)
    remaining = _ticks(budget, places, "Random arm budget")
    costs = candidates.set_index("customer_id").expected_cost
    selected = set()
    for uid in ordered:
        ticks = _ticks(_decimal(costs.at[uid], "Random expected cost"), places, "Random expected cost")
        if ticks <= remaining:
            selected.add(uid)
            remaining -= ticks
    return selected, ordered


def _uplift_selection(candidates, budget, metadata, first, solver):
    start = perf_counter()
    candidates = candidates.sort_values("customer_id", kind="stable").reset_index(drop=True).copy()
    amount_unit = _decimal(first.amount_unit, "amount_unit", positive=True)
    capacity = _ticks(budget, metadata.decimal_places, "Uplift-only arm budget")
    prepared = _prepare_allocation(
        candidates, candidates.iloc[:0], budget, metadata.decimal_places, capacity,
        metadata, _clock(metadata.as_of_at, "cost.as_of_at"), first.dataset_id, first.as_of_at,
        first.currency, amount_unit, {"objective": "uplift_score"}, start, perf_counter(),
        objective_column="uplift_score")
    result = _execute(prepared, "milp", solver)
    _approved(result.summary, "C")
    return set(result.decisions.loc[result.decisions.selected.eq(1), "customer_id"]), result.summary


def _approved(summary, arm):
    require(summary["has_valid_solution"] is True and summary["accepted_for_planning"] is True
            and summary["budget_compliant"] is True, f"{arm}: invalid or unapproved plan ({summary['status']})")
    if summary["algorithm"] == "milp":
        require(summary["optimality"] in ("numeric_optimal", "analytic_optimal"),
                f"{arm}: unproven MILP plan")


def _baseline_arm(arm, rows, safety, limit, metadata, first, random_seed, solver):
    rows = rows.copy()
    rows["eligibility_reason"] = rows.customer_id.map(lambda uid: safety[uid][1])
    safe = rows.customer_id.map(lambda uid: safety[uid][0])
    rows["candidate"] = False
    if arm == "B":
        rows["candidate"] = safe
    if arm == "C":
        rows["candidate"] = safe & rows.uplift_score.gt(0)
    candidates = rows.loc[rows.candidate]
    solver_summary = None
    order = []
    if arm == "A":
        selected, version, kind = set(), "no_action_v1", "none"
    elif arm == "B":
        selected, order = _random_selection(candidates, limit, metadata.decimal_places, random_seed)
        version, kind = RANDOM_VERSION, "random_budget_scan"
    else:
        selected, solver_summary = _uplift_selection(candidates, limit, metadata, first, solver)
        version, kind = UPLIFT_VERSION, "milp"
    rows["selected"] = rows.customer_id.isin(selected)
    rows["planned_sent_at"] = rows.as_of_at.where(rows.selected)
    rows["selection_reason"] = "not_selected"
    rows.loc[~rows.candidate, "selection_reason"] = "no_action" if arm == "A" else "ineligible_or_nonpositive"
    if arm == "B":
        rows.loc[rows.candidate & ~rows.selected, "selection_reason"] = "insufficient_remaining_budget"
        rows["selection_order"] = rows.customer_id.map({uid: rank for rank, uid in enumerate(order, 1)}).astype("Int64")
    rows.loc[rows.selected, "selection_reason"] = "selected"
    if arm == "A":
        rows["expected_cost"] = None
    configuration = {"version": version, "solver": asdict(solver) if arm == "C" else None,
                     "random_seed": random_seed if arm == "B" else None,
                     "tie_policy": "stable customer-ID columns; retain official solver result" if arm == "C" else None,
                     "model_exclusion": "not_required" if arm in ("A", "B") else "validated_uplift_required"}
    summary = dict(
        solver_kind=kind, solver_version=solver_summary["algorithm_version"] if solver_summary else version,
        strategy_version=version, configuration=configuration, configuration_fingerprint=_fingerprint(configuration),
        decision_fingerprint=_decision_fingerprint(rows), accepted_for_planning=True,
        solver_summary=solver_summary, budget=str(limit))
    return rows, summary


def _plan_arms(root, spec, predictors, features, assignments, options, common, per_customer, random_seed):
    eligibility, costs, priced, metadata, _, safety = common
    roster = predictors.roster
    first = roster.iloc[0]
    solver = options.get("solver_config") or SolverConfig()
    require(isinstance(solver, SolverConfig) and not solver.allow_unproven, "experiment requires a proven MILP plan")
    profile = options["profile"]
    reference = _scope_handoff(predictors, assignments.keys()).integrated
    require(reference is not None, "no scored customers for Full Strategy")
    segmentation = segment_customers(reference, profile.segmentation)
    reference_path = root / "segmentation_reference.parquet"
    _write_table(segmentation.customers[list(REFERENCE_FIELDS)], reference_path)
    reference_evidence = {"sha256": file_sha256(reference_path), "config": segmentation.metadata["config"]}
    model_context = predictors.snapshot()
    model_version = _fingerprint({role: meta.model_version for role, meta in model_context.prediction_metadata.items()})
    cost_version = _fingerprint(asdict(metadata))
    audit, summaries, references = [], {}, {}
    for arm, _ in spec.arms:
        ids = {uid for uid, (assigned, _) in assignments.items() if assigned == arm}
        limit = _decimal(0 if arm == "A" else per_customer * len(ids), "arm budget")
        _ticks(limit, metadata.decimal_places, "arm budget")
        scoped = roster.loc[roster.customer_id.isin(ids)].copy()
        scoped["expected_cost"] = scoped.customer_id.map(priced.set_index("customer_id").expected_cost)
        if arm in ("D1", "D2"):
            algorithm = "greedy" if arm == "D1" else "milp"
            handoff = _scope_handoff(predictors, ids)
            scoped_segments = replace(segmentation, customers=segmentation.customers.loc[
                segmentation.customers.customer_id.isin(ids)].copy())
            plan_options = dict(
                profile=profile, reporting=options["reporting"], budget=Budget(limit, first.currency, first.amount_unit),
                algorithm=algorithm, eligibility=eligibility.loc[eligibility.customer_id.isin(ids)],
                cost_inputs=costs.loc[costs.customer_id.isin(ids)], cost_metadata=metadata, solver_config=solver,
                **{key: options[key] for key in ("coupons", "redemptions", "history_context", "history_policy")})
            plan = _build_strategy_decisions(handoff, **plan_options, segmentation=scoped_segments)
            _approved(plan.budget_summary, arm)
            selected_ids = _frozen_plan(plan, features.loc[features.customer_id.isin(ids)], plan_options)
            rows = plan.decision_audit.copy()
            rows["candidate"] = rows.budget_candidate.eq(True).fillna(False)
            rows["send"] = rows.customer_id.isin(selected_ids)
            plan.metadata["classification_reference"] = reference_evidence
            directory = write_outputs(plan, root / "plans" / arm)
            digest = file_sha256(directory / "manifest.json")
            references[arm] = digest
            # The official Budget audit can omit classifications for customers
            # excluded before optimization. Preserve the input used by Retention.
            classified = scoped_segments.customers.set_index("customer_id")
            for field in CLASSIFICATION:
                rows[field] = rows.customer_id.map(classified[field])
            selected = set(selected_ids)
            budget_summary = plan.budget_summary
            summary = dict(
                solver_kind=algorithm, solver_version=budget_summary["algorithm_version"],
                strategy_version=profile.version, configuration_fingerprint=budget_summary["configuration_fingerprint"],
                decision_fingerprint=budget_summary["decision_fingerprint"], accepted_for_planning=True,
                solver_summary=budget_summary, budget=str(limit),
                classification_reference_sha256=reference_evidence["sha256"])
        else:
            rows, summary = _baseline_arm(arm, scoped, safety, limit, metadata, first, random_seed, solver)
            selected = set(rows.loc[rows.selected, "customer_id"])
            rows["send"] = rows.selected
            rows["row_state"] = rows.selected.map({True: "selected", False: "not_selected"})
        require(not rows.customer_id.duplicated().any() and set(rows.customer_id) == ids and selected <= ids,
                f"{arm}: invalid arm roster")
        require(rows.loc[rows.send, "planned_sent_at"].eq(first.as_of_at).all(), f"{arm}: only t0 sends supported")
        expected = _decimal_sum(rows.loc[rows.send, "expected_cost"])
        require(expected <= limit and (arm != "A" or not selected), f"{arm}: invalid budget or No Action selection")
        summary.update(model_version=model_version if arm in ("C", "D1", "D2") else "none",
                       cost_model_version=cost_version, expected_cost=str(expected),
                       score_definition_id={"A": "none", "B": "random_candidate_order", "C": "uplift_score"}.get(
                           arm, "uplift_times_clv"),
                       decision_artifact_id=f"plans/{arm}/decisions.parquet", selected_ids=sorted(selected))
        rows["arm_id"] = arm
        rows["candidate"] = rows.candidate.astype(bool)
        rows["decision_at"] = first.as_of_at
        audit.append(rows[list(DECISION_FIELDS)])
        artifact = root / summary["decision_artifact_id"]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        audit_fields = [*DECISION_FIELDS, "status", "eligibility_reason"]
        if arm in ("D1", "D2"):
            audit_fields += ["model_exclusion_reason", *CLASSIFICATION]
        else:
            audit_fields += ["selection_reason"] + (["selection_order"] if arm == "B" else [])
            if arm == "C":
                audit_fields += ["uplift_score"]
        _write_table(rows[audit_fields], artifact)
        summaries[arm] = summary
    decisions = pd.concat(audit, ignore_index=True)
    decisions["selected"] = decisions.selected.astype("Int64")
    _selection_union(decisions, assignments)
    return decisions, summaries, references, reference_evidence


def _selection_union(decisions, assignments):
    require(not decisions.customer_id.duplicated().any() and set(decisions.customer_id) == set(assignments),
            "duplicate or missing arm decision")
    require(all(assignments[uid][0] == arm for uid, arm in zip(decisions.customer_id, decisions.arm_id)),
            "arm decision moved customer")
    require(not (decisions.arm_id.eq("A") & decisions.send).any(), "No Action cannot request coupons")
    return sorted(decisions.loc[decisions.send, "customer_id"])


def _serialized_table(frame):
    # Keep exact variable-precision Decimal expectations portable in Parquet.
    export = frame.copy()
    for column in ("expected_cost", "send_cost", "redemption_probability", "expected_discount_if_redeemed"):
        if column in export:
            export[column] = export[column].map(
                lambda cost: _format_cost(cost) if pd.notna(cost) else None)
    return export


def _format_cost(cost):
    number = format(_decimal(cost, "cost"), "f")
    return number.rstrip("0").rstrip(".") if "." in number else number


def _write_table(frame, path):
    _serialized_table(frame).to_parquet(path, index=False)


def _outcome_audit(roster, decisions):
    audit = roster[[*CONTEXT_COLUMNS, "model_exclusion_reason"]].merge(
        decisions[["customer_id", "selected", "send", "planned_sent_at", "row_state"]],
        on="customer_id", how="left", validate="one_to_one")
    audit["send"] = audit.send.eq(True).fillna(False)
    audit["row_state"] = audit.row_state.fillna("population_ineligible")
    return audit


def _population_ledgers(features, eligibility, assignments):
    base = features.rename(columns={"dataset_id": "evaluation_dataset_id"})[list(CONTEXT)]
    t0 = base.as_of_at.iloc[0]
    population = base.assign(eligible=base.customer_id.isin(assignments), eligibility_at=t0,
                             exclusion_reason=base.customer_id.map(eligibility.set_index("customer_id").business_reason))
    assigned = base.loc[base.customer_id.isin(assignments)].copy()
    assigned["arm_id"] = assigned.customer_id.map(lambda uid: assignments[uid][0])
    assigned["assignment_probability"] = assigned.customer_id.map(lambda uid: assignments[uid][1])
    assigned["assigned_at"] = t0
    return population[[*CONTEXT, *LEDGERS["population"]]], assigned[[*CONTEXT, *LEDGERS["assignments"]]]


def _ledgers(population, assigned, decisions, outcomes, baseline, replay):
    t0 = population.as_of_at.iloc[0]
    decision = decisions.rename(columns={"dataset_id": "evaluation_dataset_id", "send": "send_requested"}).copy()
    decision["selected"] = decision.selected.eq(1).fillna(False).astype(bool)
    facts = outcomes.drop(columns=["dataset_id", "as_of_at", "snapshot_id", "currency", "amount_unit"])
    executed = assigned[[*CONTEXT, "arm_id"]].merge(facts, on="customer_id", validate="one_to_one")
    executed["raw_database_sha256"] = replay["raw_database_sha256"]
    censoring_reason = executed.get("censoring_reason")
    require(censoring_reason is not None or executed.outcome_observed.all(),
            "current producer has no verified censoring reason")
    observed = executed.assign(
        outcome_start_at=t0, purchase_occurred=~executed.factual_no_order,
        session_occurred=~executed.factual_no_session, paid_minor=executed.factual_outcome_net_revenue_minor,
        coupon_discount_minor=executed.outcome_redemption_discount_cost_minor.where(executed.outcome_observed),
        censoring_reason=censoring_reason)
    require(population.raw_database_sha256.eq(baseline["raw_database_sha256"]).all(), "invalid experiment baseline SHA")
    frames = dict(population=population, assignments=assigned, decisions=decision,
                  executions=executed, outcomes=observed)
    return {name: frames[name][[*CONTEXT, *columns]].copy() for name, columns in LEDGERS.items()}


def _rows(label, expected, actual):
    keys = ["customer_id", "snapshot_id"]
    require(not expected.duplicated(keys).any() and not actual.duplicated(keys).any(),
            f"{label}: duplicate customer/snapshot")
    _same(label, expected.set_index(keys).sort_index(), actual.set_index(keys).sort_index())


def _read_spec(document):
    values = {field.name: document[field.name] for field in fields(ExperimentSpec)}
    for name in ("allocation_weights", "unresolved_policy_items"):
        values[name] = tuple(values[name])
    for name in ("arms", "primary_comparisons", "secondary_comparisons"):
        values[name] = tuple(map(tuple, values[name]))
    spec = ExperimentSpec(**values)
    require(_json_value(spec.to_dict()) == document, "published ExperimentSpec fields differ")
    return spec


def _read_power_plan(document):
    if document is None:
        return None
    values = dict(document)
    values["scenarios"] = tuple(PowerScenario(**(scenario | {"comparison": tuple(scenario["comparison"])}))
                                for scenario in values["scenarios"])
    for name in ("minimum_analyzable_by_arm", "required_assigned_by_arm", "achieved_power"):
        values[name] = tuple(map(tuple, values[name]))
    return PowerPlan(**values)


def _official_arm(root, arm, rows, summary, reference, reference_evidence, label):
    directory = root / "plans" / arm
    budget = _json(directory / "budget_summary.json")
    meta = _json(directory / "metadata.json")
    _approved(budget, arm)
    require(budget == summary["solver_summary"] and budget["decision_fingerprint"] == summary["decision_fingerprint"]
            and budget["configuration_fingerprint"] == summary["configuration_fingerprint"]
            and budget["algorithm"] == summary["solver_kind"] and budget["algorithm_version"] == summary["solver_version"]
            and _decimal(budget["budget"], "budget") == _decimal(summary["budget"], "budget"),
            f"{label}: official budget_100 or Solver evidence differs")
    require(meta["classification_reference"] == reference_evidence
            and summary["classification_reference_sha256"] == reference_evidence["sha256"],
            f"{label}: official common Segmentation reference differs")
    source = pd.read_csv(directory / "decision_audit.csv", dtype=str, keep_default_na=False)
    source = source.rename(columns={"budget_candidate": "candidate"})
    require(source.candidate.isin(["", "True", "False"]).all(), f"{label}: invalid official candidate")
    source["candidate"] = source.candidate.eq("True")
    source["send"] = source.send.eq("1")
    source["selected"] = pd.to_numeric(source.selected.replace("", None)).astype("Int64")
    source["planned_sent_at"] = pd.to_datetime(source.planned_sent_at, utc=True)
    source["as_of_at"] = pd.to_datetime(source.as_of_at, utc=True)
    source["expected_cost"] = source.expected_cost.map(lambda cost: _format_cost(cost) if cost else "")
    columns = [*CONTEXT_COLUMNS, "candidate", "selected", "send", "planned_sent_at", "expected_cost",
               "row_state", "status", "model_exclusion_reason", "eligibility_reason"]
    actual = rows[columns].copy()
    for field in ("expected_cost", "model_exclusion_reason", "eligibility_reason"):
        actual[field] = actual[field].fillna("").astype(str)
    _rows(f"{label} versus official decision_audit.csv", source[columns], actual)
    with_classification = source.loc[source.segment_id.ne(""), ["customer_id", "snapshot_id", *CLASSIFICATION]]
    reported = rows.loc[rows.customer_id.isin(with_classification.customer_id),
                        ["customer_id", "snapshot_id", *CLASSIFICATION]].copy()
    reported["priority_score"] = reported.priority_score.astype(str)
    _rows(f"{label} versus official decision_audit.csv classification", with_classification, reported)
    scored = rows.loc[rows.status.eq("scored"), [*CONTEXT_COLUMNS, *CLASSIFICATION]]
    _rows(f"{label} versus common Segmentation", reference.loc[reference.customer_id.isin(scored.customer_id),
                                                   [*CONTEXT_COLUMNS, *CLASSIFICATION]], scored)


def _read_arm_decisions(root, spec, manifest, assignments, planning, costs, reference, label):
    arms = manifest["arms"]
    require(set(arms) == {arm for arm, _ in spec.arms}, f"{label}: incomplete five-arm plans")
    reference_evidence = manifest["segmentation_reference"]
    per_customer = _decimal(manifest["budget_per_customer"], "budget")
    cost_metadata = manifest["cost_provenance"]["metadata"]
    found = []
    for arm, _ in spec.arms:
        summary = arms[arm]
        path = f"plans/{arm}/decisions.parquet"
        require(summary["decision_artifact_id"] == path and summary["accepted_for_planning"] is True,
                f"{label}/{arm}: invalid decision artifact")
        require(summary["cost_model_version"] == _fingerprint(cost_metadata),
                f"{label}/{arm}: inconsistent common cost model")
        rows = pd.read_parquet(root / path)
        ids = {uid for uid, (assigned, _) in assignments.items() if assigned == arm}
        require(not rows.customer_id.duplicated().any() and set(rows.customer_id) == ids
                and rows.arm_id.eq(arm).all(), f"{label}/{arm}: decision roster differs from assignment")
        _rows(f"{label}/{arm} decision identity", planning.loc[planning.customer_id.isin(ids),
              [*CONTEXT_COLUMNS, "status"]], rows[[*CONTEXT_COLUMNS, "status"]])
        scope = rows.loc[rows.status.eq("scored")] if arm in ("D1", "D2") else rows
        require(_decision_fingerprint(scope) == summary["decision_fingerprint"],
                f"{label}/{arm}: decision fingerprint differs from rows")
        selected = rows.selected.eq(1).fillna(False)
        require(rows.send.eq(selected).all() and not (selected & ~rows.candidate).any()
                and rows.loc[selected, "planned_sent_at"].eq(rows.loc[selected, "as_of_at"]).all()
                and rows.loc[~selected, "planned_sent_at"].isna().all()
                and rows.decision_at.eq(rows.as_of_at).all(), f"{label}/{arm}: invalid selection or request")
        require(sorted(rows.loc[selected, "customer_id"]) == summary["selected_ids"],
                f"{label}/{arm}: selected IDs differ")
        actual = rows.loc[selected]
        projected = actual.customer_id.map(costs)
        require(all(pd.notna(cost) and pd.notna(expected) and _decimal(cost, "expected_cost") == expected
                    for cost, expected in zip(actual.expected_cost, projected)), f"{label}/{arm}: wrong selected cost")
        limit = _decimal(summary["budget"], "budget")
        total = _decimal_sum(_decimal(cost, "expected_cost") for cost in actual.expected_cost)
        require(limit == (0 if arm == "A" else per_customer * len(ids))
                and total == _decimal(summary["expected_cost"], "expected_cost") <= limit,
                f"{label}/{arm}: inconsistent arm budget or expected cost")
        if arm in ("D1", "D2"):
            _official_arm(root, arm, rows, summary, reference, reference_evidence, f"{label}/{arm}")
        else:
            require(_fingerprint(summary["configuration"]) == summary["configuration_fingerprint"],
                    f"{label}/{arm}: policy configuration fingerprint differs")
            declared = (rows.customer_id.map(costs).map(_format_cost) if arm != "A"
                        else pd.Series(None, index=rows.index, dtype=object))
            require(rows.expected_cost.fillna("").eq(declared.fillna("")).all(),
                    f"{label}/{arm}: expected costs differ from common costs")
            common = planning.set_index("customer_id")
            safe = rows.customer_id.map(common.safe_to_send)
            uplifts = rows.customer_id.map(common.uplift_score)
            expected = safe & uplifts.gt(0) if arm == "C" else safe if arm == "B" else False
            require(rows.candidate.eq(expected).all(), f"{label}/{arm}: candidate differs from frozen input")
            if arm == "B":
                candidates = rows.loc[rows.candidate].copy()
                candidates["expected_cost"] = candidates.customer_id.map(costs)
                chosen, order = _random_selection(candidates, limit, cost_metadata["decimal_places"],
                                                  manifest["seeds"]["random_policy"])
                ranks = rows.customer_id.map({uid: rank for rank, uid in enumerate(order, 1)}).astype("Int64")
                require(chosen == set(summary["selected_ids"]) and ranks.equals(rows.selection_order),
                        f"{label}/B: Random selection or order differs from seed")
            if arm == "C":
                _approved(summary["solver_summary"], arm)
                require(summary["solver_summary"]["algorithm"] == "milp"
                        and rows.uplift_score.eq(uplifts).all()
                        and summary["solver_summary"]["selected_customer_count"] == len(actual)
                        and _decimal(summary["solver_summary"]["total_expected_cost"], "C cost") == total
                        and _decimal(summary["solver_summary"]["budget"], "C budget") == limit,
                        f"{label}/C: inconsistent Uplift Solver or candidate")
        found.append(rows[list(DECISION_FIELDS)])
    decisions = pd.concat(found, ignore_index=True)
    decisions["selected"] = decisions.selected.astype("Int64")
    return decisions


def _verify_semantics(root, receipt, manifest, spec, analysis):
    label = f"experiment {spec.experiment_id}"
    seeds = manifest["seeds"]
    selected = receipt["selected_ids"]
    require(manifest["experiment_id"] == receipt["experiment_id"] == spec.experiment_id
            and manifest["version"] == receipt["experiment_version"] == spec.version
            and manifest["execution_id"] == receipt["execution_id"]
            and seeds == receipt["seeds"] and manifest["assignment_seed"] == seeds["strategy_assignment"]
            and manifest["assignment_method"] == "fixed_largest_remainder"
            and manifest["assignment_algorithm"] == "python_random_shuffle_sorted_customer_ids"
            and manifest["assignment_version"] == "1"
            and manifest["arms"]["B"]["configuration"]["random_seed"] == seeds["random_policy"]
            and manifest["selected_ids"] == selected and manifest["request_sha256"] == receipt["request_sha256"]
            == _fingerprint(selected), f"{label}: inconsistent experiment identity or frozen design")
    baseline_data = _read_generation(root / "baseline", manifest["n_customers"], seeds["simulator"], {"mode": "none"})
    replay_data = _read_generation(root / "replay", manifest["n_customers"], seeds["simulator"],
                                   {"mode": "assign", "customer_ids": selected})
    past, features, history, inputs, baseline, _, _ = baseline_data
    other, after, after_history, replay_inputs, replay, log, source = replay_data
    require(baseline == receipt["baseline"] and replay == receipt["replay"]
            and inputs == receipt["generation_inputs"]
            and manifest["evaluation_dataset_id"] == baseline["dataset_id"]
            and manifest["replay_dataset_id"] == replay["dataset_id"]
            and manifest["feature_raw_sha256"] == baseline["raw_database_sha256"]
            and manifest["result_raw_sha256"] == replay["raw_database_sha256"]
            and all(manifest[key] == inputs[key] for key in
                    ("as_of_at", "observation_start_at", "observation_end_at", "currency", "amount_unit")),
            f"{label}: generation or ledger provenance differs")
    pre_state = _compare_replay(baseline, replay, selected, past, other, features, after,
                                history, after_history, inputs, replay_inputs)
    require(all(receipt[key] == value for key, value in pre_state.items()), f"{label}: public pre-state differs")
    planning = pd.read_parquet(root / "planning_inputs.parquet")
    _rows(f"{label} planning input identity", features[list(CONTEXT_COLUMNS)], planning[list(CONTEXT_COLUMNS)])
    excluded = planning.loc[planning.status.ne("scored")]
    excluded_columns = [*CONTEXT_COLUMNS, "status", "exclusion_stage", "exclusion_reason"]
    excluded = excluded.assign(exclusion_stage=excluded.model_exclusion_stage, exclusion_reason=excluded.model_exclusion_reason)
    require(_fingerprint(excluded[excluded_columns].sort_values("customer_id").to_dict("records"))
            == _fingerprint(sorted(manifest["model_exclusions"], key=lambda row: row["customer_id"])),
            f"{label}: model exclusion audit differs from planning inputs")
    eligible = planning.loc[planning.business_eligible.eq(True), "customer_id"]
    assignments = _assign(eligible, spec, seeds["strategy_assignment"])
    population, assigned = _population_ledgers(features, planning, assignments)
    reference = pd.read_parquet(root / "segmentation_reference.parquet")
    scored = planning.loc[planning.status.eq("scored") & planning.customer_id.isin(eligible)]
    _rows(f"{label} common Segmentation roster", scored[list(CONTEXT_COLUMNS)], reference[list(CONTEXT_COLUMNS)])
    require(manifest["segmentation_reference"]["sha256"] == receipt["files"]["segmentation_reference.parquet"],
            f"{label}: common Segmentation artifact hash differs")
    cost_table = pd.read_parquet(root / "common_costs.parquet")
    metadata = CostMetadata(**manifest["cost_provenance"]["metadata"])
    cost_inputs = _aligned(cost_table[list(COST_INPUT_COLUMNS)], COST_INPUT_COLUMNS, features, "experiment costs")
    priced = _price_cost_inputs(cost_inputs, metadata, features.as_of_at.iloc[0])
    _rows(f"{label} common expected cost", _serialized_table(priced), cost_table)
    decisions = _read_arm_decisions(root, spec, manifest, assignments, planning, priced.set_index("customer_id").expected_cost,
                                    reference, label)
    require(_selection_union(decisions, assignments) == selected, f"{label}: selected/requested roster differs")
    outcomes = _outcomes(log, _outcome_audit(planning, decisions), replay, selected, source, include_discount_rate=True)
    require(sorted(outcomes.loc[outcomes.actual_sent, "customer_id"]) == selected,
            f"{label}: sent roster differs from requested")
    _rows(f"{label} execution_outcomes versus official log", outcomes,
          pd.read_parquet(root / "execution_outcomes.parquet"))
    expected = _ledgers(population, assigned, decisions, outcomes, baseline, replay)
    saved = {name: pd.read_parquet(root / "ledgers" / f"{name}.parquet") for name in LEDGERS}
    for name in LEDGERS:
        _rows(f"{label}/{name} versus frozen plans and official log", _serialized_table(expected[name]), saved[name])
    analysis_inputs = manifest["analysis_inputs"]
    snapshots = features.set_index("customer_id").snapshot_id
    policy_evidence = {arm: {"artifact_id": summary["decision_artifact_id"],
                             "selected_keys": [(uid, snapshots.at[uid]) for uid in summary["selected_ids"]]}
                       for arm, summary in manifest["arms"].items() if arm != "A"}
    reproduced = analyze_experiment(spec, manifest, **saved, features=features, policy_evidence=policy_evidence,
                                    power_plan=_read_power_plan(analysis_inputs["power_plan"]),
                                    training_customer_ids=analysis_inputs["training_customer_ids"])
    require(reproduced.to_dict() == analysis, f"{label}: analysis differs from revalidated ledgers")


def _verify_bundle(root, receipt):
    require(set(receipt["official_plan_manifests"]) == {"D1", "D2"}, "incomplete official plans")
    for arm, digest in receipt["official_plan_manifests"].items():
        _plan_manifest(root / "plans" / arm, digest)
    files = receipt["files"]
    required = {"spec.json", "manifest.json", "analysis.json", "execution_outcomes.parquet", *FROZEN_FILES}
    required |= {f"ledgers/{name}.parquet" for name in LEDGERS}
    required |= {f"plans/{arm}/decisions.parquet" for arm, _ in ARMS}
    require(set(files) == required, "incomplete experiment artifacts")
    for name, digest in files.items():
        require(file_sha256(root / name) == digest, f"changed experiment artifact: {name}")
    manifest = _json(root / "manifest.json")
    analysis = _json(root / "analysis.json")
    spec_document = _json(root / "spec.json")
    require(manifest["ledger_sha256"] == {name: files[f"ledgers/{name}.parquet"] for name in LEDGERS}
            and _fingerprint(spec_document) == manifest["spec_fingerprint"] == receipt["spec_fingerprint"]
            and analysis["validation"]["by_arm"] == receipt["by_arm"]
            and analysis["holm"]["status"] == receipt["analysis_status"]
            and analysis["validation"]["design_status"] == receipt["design_status"]
            and analysis["causal_interpretation"] == receipt["causal_interpretation"],
            "experiment manifest, specification or ledger hashes differ")
    with localcontext(Context(prec=200)):
        _verify_semantics(root, receipt, manifest, _read_spec(spec_document), analysis)


def verify_experiment_receipt(output):
    """Verify a completed bundle against its saved sources before using its ledgers.

    Bundles without frozen planning inputs and a Segmentation reference lack
    the evidence required by this verifier.
    """
    root = Path(output).resolve()
    require((root / "receipt.json").is_file() and not (root / "failure.json").exists(),
            "no successful experiment receipt")
    receipt = _json(root / "receipt.json")
    require(receipt["status"] == STATUS and receipt["dispatch_authorized"] is False
            and receipt["confirmatory_approved"] is False and receipt["mode"] == MODE
            and receipt["design_status"] == "planning_unapproved"
            and receipt["internal_state_status"] == "producer_verification_unavailable"
            and receipt["causal_interpretation"] == "unverified"
            and receipt["public_pre_state"] == "verified", "invalid experiment receipt status or public pre-state")
    _verify_bundle(root, receipt)
    return receipt


def run_experiment(output, *, spec, prepare_inputs, n_customers, simulator_seed, assignment_seed, random_seed,
                   budget_per_customer, mode, config=POPULATION_CONFIG, power_plan=None, training_customer_ids=None):
    """Run a simulation contract check and publish the verified five-arm ledger bundle."""
    require(mode == MODE and isinstance(spec, ExperimentSpec), "explicit simulation verification and ExperimentSpec required")
    require(all(type(seed) is int and seed >= 0 for seed in (simulator_seed, assignment_seed, random_seed)),
            "experiment seeds must be nonnegative integers")
    per_customer = _decimal(budget_per_customer, "budget_per_customer")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    stage = "baseline"
    receipt = dict(execution_id=uuid4().hex, experiment_id=spec.experiment_id, experiment_version=spec.version,
                   spec_fingerprint=_fingerprint(spec.to_dict()), design_status=spec.design_status,
                   mode=mode, confirmatory_approved=False, dispatch_authorized=False,
                   internal_state_status="producer_verification_unavailable",
                   causal_interpretation="unverified")
    try:
        settings = EcommerceSimulator(config).config
        past, features, history, inputs, baseline, untreated, _ = _generate(
            root / "baseline", settings, n_customers, simulator_seed, {"mode": "none"})
        require(untreated.treatment.eq(0).all(), "baseline must be untreated")
        stage = "planning"
        batches, options = prepare_inputs(features.copy(deep=True), history.copy(deep=True))
        predictors = adapt_weekly_predictions(features, **batches)
        first = predictors.roster.iloc[0]
        with localcontext(Context(prec=200)):
            common = _prepare_common(predictors, options)
            eligibility, _, priced, cost_metadata, cost_evidence, safety = common
            eligible = eligibility.loc[eligibility.business_eligible.eq(True), "customer_id"].copy()
            assignments = _assign(eligible, spec, assignment_seed)
            population, assigned = _population_ledgers(features, eligibility, assignments)
            (root / "ledgers").mkdir()
            _write_table(population, root / "ledgers/population.parquet")
            _write_table(assigned, root / "ledgers/assignments.parquet")
            decisions, arms, plan_manifests, reference_evidence = _plan_arms(
                root, spec, predictors, features, assignments, options, common, per_customer, random_seed)
        excluded = predictors.excluded.set_index("customer_id")
        planning = predictors.roster[[*CONTEXT_COLUMNS, "status", "uplift_score"]].merge(
            eligibility[["customer_id", "business_eligible", "business_reason", "history_complete"]],
            on="customer_id", validate="one_to_one")
        planning["model_exclusion_reason"] = planning.customer_id.map(excluded.exclusion_reason)
        planning["model_exclusion_stage"] = planning.customer_id.map(excluded.exclusion_stage)
        planning["safe_to_send"] = planning.customer_id.map(lambda uid: safety[uid][0])
        _write_table(planning, root / "planning_inputs.parquet")
        selected = _selection_union(decisions, assignments)
        _write_table(priced, root / "common_costs.parquet")
        frozen_files = [*FROZEN_FILES,
                        *[summary["decision_artifact_id"] for summary in arms.values()]]
        frozen_hashes = {name: file_sha256(root / name) for name in frozen_files}
        receipt.update(baseline=baseline, selected_ids=selected, request_sha256=_fingerprint(selected),
                       official_plan_manifests=plan_manifests, generation_inputs=inputs,
                       seeds=dict(simulator=simulator_seed, strategy_assignment=assignment_seed, random_policy=random_seed))
        stage = "replay"
        other, after, after_history, replay_inputs, replay, log, source = _generate(
            root / "replay", settings, n_customers, simulator_seed, {"mode": "assign", "customer_ids": selected})
        receipt["replay"] = replay
        stage = "public_pre_state"
        receipt.update(_compare_replay(
            baseline, replay, selected, past, other, features, after, history, after_history, inputs, replay_inputs))
        stage = "outcomes"
        audit = _outcome_audit(planning, decisions)
        outcomes = _outcomes(log, audit, replay, selected, source, include_discount_rate=True)
        require(sorted(outcomes.loc[outcomes.actual_sent, "customer_id"]) == selected,
                "current Simulator did not send to the exact requested roster")
        _write_table(outcomes, root / "execution_outcomes.parquet")
        stage = "ledgers"
        for name, digest in frozen_hashes.items():
            require(file_sha256(root / name) == digest, f"changed frozen experiment artifact: {name}")
        ledgers = _ledgers(population, assigned, decisions, outcomes, baseline, replay)
        for name in ("decisions", "executions", "outcomes"):
            _write_table(ledgers[name], root / "ledgers" / f"{name}.parquet")
        digests = {name: file_sha256(root / "ledgers" / f"{name}.parquet") for name in ledgers}
        manifest = dict(experiment_id=spec.experiment_id, version=spec.version, contract_version=CONTRACT_VERSION,
                        execution_id=receipt["execution_id"], spec_fingerprint=receipt["spec_fingerprint"],
                        seeds=receipt["seeds"], n_customers=n_customers, selected_ids=selected,
                        request_sha256=receipt["request_sha256"], segmentation_reference=reference_evidence,
                        evaluation_dataset_id=first.dataset_id, as_of_at=inputs["as_of_at"],
                        replay_dataset_id=replay["dataset_id"],
                        population_frozen_at=inputs["as_of_at"], observation_start_at=inputs["observation_start_at"],
                        observation_end_at=inputs["observation_end_at"], outcome_start_rule="as_of_at",
                        currency=first.currency, amount_unit=str(first.amount_unit),
                        eligibility_policy_version=options["history_context"].version,
                        coupon_policy_version="simulator_t0_10pct_28d", assignment_method="fixed_largest_remainder",
                        assignment_algorithm="python_random_shuffle_sorted_customer_ids", assignment_version="1",
                        assignment_seed=assignment_seed, feature_raw_sha256=baseline["raw_database_sha256"],
                        result_raw_sha256=replay["raw_database_sha256"], budget_per_customer=str(per_customer),
                        ledger_sha256=digests, arms=arms, model_provenance=predictors.producer_metadata,
                        scipy_version=scipy_version, cohort_prediction_evidence=predictors.evidence,
                        analysis_inputs=dict(power_plan=asdict(power_plan) if isinstance(power_plan, PowerPlan) else None,
                                             training_customer_ids=sorted(training_customer_ids)
                                             if isinstance(training_customer_ids, (set, frozenset))
                                             else training_customer_ids),
                        cost_provenance=dict(metadata=asdict(cost_metadata), evidence=cost_evidence),
                        model_exclusions=predictors.excluded.to_dict("records"))
        snapshots = features.set_index("customer_id").snapshot_id
        policy_evidence = {arm: dict(artifact_id=summary["decision_artifact_id"],
                                     selected_keys=[(uid, snapshots.at[uid])
                                                    for uid in summary["selected_ids"]])
                           for arm, summary in arms.items() if arm != "A"}
        evidence = dict(power_plan=power_plan, features=features,
                        training_customer_ids=training_customer_ids, policy_evidence=policy_evidence)
        saved = {name: pd.read_parquet(root / "ledgers" / f"{name}.parquet") for name in ledgers}
        validated = validate_experiment_run(spec, manifest, **saved, **evidence)
        analysis = analyze_experiment(spec, manifest, **saved, **evidence)
        _publish(root / "spec.json", spec.to_dict())
        _publish(root / "manifest.json", _json_value(manifest))
        _publish(root / "analysis.json", analysis.to_dict())
        stage = "publication"
        files = ["spec.json", "manifest.json", "analysis.json", "execution_outcomes.parquet",
                 "ledgers/decisions.parquet", "ledgers/executions.parquet", "ledgers/outcomes.parquet"]
        completed = dict(receipt, status=STATUS, by_arm=validated.by_arm, analysis_status=analysis.holm["status"],
                         elapsed_seconds=perf_counter()-started,
                         files={**frozen_hashes, **{name: file_sha256(root / name) for name in files}})
        _verify_bundle(root, completed)
        _publish(root / "receipt.json", completed)
        return _json(root / "receipt.json")
    except Exception as exc:
        _publish(root / "failure.json", dict(receipt, status="failed", stage=stage, reason=str(exc)))
        raise
