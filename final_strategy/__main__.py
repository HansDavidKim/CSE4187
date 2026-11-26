"""Assemble existing model predictions; this command never trains a replacement model."""
import argparse
from dataclasses import asdict
from decimal import Context, localcontext
from types import SimpleNamespace
import json
from pathlib import Path

import pandas as pd

from .weekly_inputs import PredictionBatch, load_feature_store
from .pipeline import DecisionConfig, build_decisions, write_outputs
from .evaluation import MONEY_FIELDS, qini_curves, strategy_evaluation
from .strategy_profiles import build_strategy_profiles


def read(path, *, exact_money=False):
    identifiers = ("dataset_id", "customer_id", "snapshot_id", "raw_database_sha256", "currency", "amount_unit")
    return (pd.read_parquet(path) if Path(path).suffix == ".parquet"
            else pd.read_csv(path, converters={name: str for name in identifiers},
                             dtype={name: "string" for name in MONEY_FIELDS} if exact_money else None))


def read_batches(path):
    batches = {}
    for role, entry in json.loads(path.read_text(encoding="utf-8")).items():
        source = Path(entry["frame"])
        source = source if source.is_absolute() else path.parent/source
        batches[role] = PredictionBatch(read(source), entry["metadata"], entry.get("producer_metadata", {}))
    return batches


def _engine_output(args, predictions, profile):
    from .budget import Budget, CostMetadata, SolverConfig
    from .hybrid import HybridConfig
    from .inputs import load_feature_store as load_verified_store
    from .retention_strategy import HistoryContext, HistoryPolicy
    from .weekly_prediction_adapter import adapt_weekly_predictions
    from .weekly_planning import ReportingConfig, build_strategy_decisions

    config = json.loads(args.engine_plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.predictions.read_text(encoding="utf-8"))
    def table(name):
        # Keep exact declared costs, identifier strings and nullable Boolean types.
        return pd.read_parquet(args.engine_plan.parent / config[name])

    batches = {name: SimpleNamespace(frame=batch.frame, metadata=asdict(batch.metadata),
                                     producer_metadata=batch.producer_metadata)
               for name, batch in predictions.items()}
    clv = batches["clv"]
    audit_path = args.predictions.parent / manifest["clv"]["audit"]
    if audit_path.suffix.lower() == ".csv":
        from clv.bundle import file_sha256
        from .weekly_validation import require
        source = args.predictions.parent / manifest["clv"]["frame"]
        sidecar = json.loads(source.with_name(source.stem + "_metadata.json").read_text(encoding="utf-8"))
        paths = [source, audit_path]
        require(audit_path.resolve() == source.with_name(source.stem + "_audit.csv").resolve(),
                "CLV producer audit path mismatch")
        gross = batches.get("gross_4w")
        if gross is not None:
            gross_path = args.predictions.parent / manifest["gross_4w"]["frame"]
            require(gross_path.resolve() == source.with_name(source.stem + "_gross_4w.parquet").resolve(),
                    "CLV gross batch path mismatch")
            paths.append(gross_path)
        require({path.name: file_sha256(path) for path in paths} == sidecar["files_sha256"]
                and clv.metadata == sidecar["prediction_metadata"]
                and clv.producer_metadata == sidecar["producer_metadata"]
                and (gross.metadata if gross else None) == sidecar["gross_4w_metadata"]
                and sidecar["gross_4w_status"] == clv.producer_metadata["gross_4w"]["status"]
                and list(clv.frame) == sidecar["columns"] and len(clv.frame) == sidecar["rows"],
                "CLV producer checksum or metadata mismatch")
    clv.audit = read(audit_path)
    clv.gross_4w_status = clv.producer_metadata["gross_4w"]["status"]
    gross = batches.get("gross_4w")
    clv.gross_4w = gross.frame if gross else None
    clv.gross_4w_metadata = gross.metadata if gross else None
    with localcontext(Context(prec=200)):
        handoff = adapt_weekly_predictions(load_verified_store(args.features), retention=batches["churn"],
                                          uplift=batches["uplift"], clv=clv,
                                          expected_retention_policy=config["retention_policy"])
        return build_strategy_decisions(
            handoff, profile, ReportingConfig(**json.loads(args.config.read_text(encoding="utf-8"))),
            Budget(**config["budget"]), algorithm=config["algorithm"], eligibility=table("eligibility"),
            coupons=table("coupons"), redemptions=table("redemptions"),
            history_context=HistoryContext(**config["history_context"]), history_policy=HistoryPolicy(**config["history_policy"]),
            cost_inputs=table("cost_inputs") if config.get("cost_inputs") else None,
            cost_metadata=CostMetadata(**config["cost_metadata"]) if config.get("cost_metadata") else None,
            solver_config=SolverConfig(**config.get("solver_config", {})),
            hybrid_config=HybridConfig(**config["hybrid_config"]) if "hybrid_config" in config else None)


def _evaluation_input(path, assignment_path, fields):
    from .weekly_validation import KEYS, keys, require, utc
    outcomes = keys(read(path))
    assigned = keys(read(assignment_path))
    columns = [*KEYS, "snapshot_id", "raw_database_sha256", "assigned_at", *fields]
    for frame in (assigned, outcomes):
        frame["assigned_at"] = utc(frame.assigned_at, "assigned_at")
        require(frame[columns].notna().all().all(), "missing assignment identity")
    linked = outcomes[columns].merge(assigned[columns], on=columns, how="outer", indicator=True, validate="one_to_one")
    require(linked._merge.eq("both").all(), "outcomes differ from the original assignment roster")
    require(utc(outcomes.outcome_end_at, "end").eq(outcomes.as_of_at + pd.Timedelta(weeks=4)).all()
            and utc(outcomes.observation_end_at, "observed").ge(utc(outcomes.outcome_end_at, "end")).all(),
            "evaluation requires complete four-week follow-up")
    if "outcome_observed" in outcomes:
        require(outcomes.outcome_observed.eq(True).all(), "censored outcomes")
    if "is_censored" in outcomes:
        require(outcomes.is_censored.eq(False).all(), "censored outcomes")
    if "send_cost_minor" in outcomes:
        require("actual_sent" in outcomes and outcomes.actual_sent.isin([True, False]).all(),
                "realized charges require actual send status")
        require(outcomes.loc[outcomes.actual_sent.eq(False), "send_cost_minor"].eq(0).all(),
                "unsupported positive charge without a successful send")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON role -> frame path, metadata and optional producer_metadata")
    parser.add_argument("--config", type=Path, required=True, help="DecisionConfig JSON, or descriptive ReportingConfig with --engine-plan")
    parser.add_argument("--expected-discounts", type=Path)
    parser.add_argument("--profile-declarations", type=Path, help="Explicit human profile version, source and thresholds JSON")
    parser.add_argument("--profile", help="P0, H0, H1, H2, H3, H4-R or H4-V from the declared profiles")
    parser.add_argument("--strategy-outcomes", type=Path)
    parser.add_argument("--qini-outcomes", type=Path)
    parser.add_argument("--qini-predictions", type=Path, help="Optional JSON model name -> purchase probability batch for multi-model curves")
    parser.add_argument("--evaluation-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-plan", type=Path, help="Use canonical policy/Budget planning with explicit JSON inputs")
    parser.add_argument("--strategy-assignments", type=Path, help="Original strategy ITT roster")
    parser.add_argument("--qini-assignments", type=Path, help="Original randomized coupon roster")
    args = parser.parse_args()
    if args.engine_plan and (not args.profile or args.expected_discounts):
        parser.error("--engine-plan requires an explicit profile and per-request cost inputs")
    if args.engine_plan and ((args.strategy_outcomes and not args.strategy_assignments)
                             or (args.qini_outcomes and not args.qini_assignments)):
        parser.error("integrated evaluation requires the original assignment roster for each analysis")
    if bool(args.profile_declarations) != bool(args.profile):
        parser.error("--profile-declarations and --profile must be supplied together")
    profile = None
    if args.profile:
        profiles = build_strategy_profiles(**json.loads(args.profile_declarations.read_text(encoding="utf-8")))
        profile = next((item for item in profiles.policies if item.policy_id == args.profile), None)
        if profile is None:
            parser.error("Unknown declared human profile")
    predictions = read_batches(args.predictions)
    if args.engine_plan:
        result = _engine_output(args, predictions, profile)
    else:
        features = load_feature_store(args.features)
        result = build_decisions(features, predictions, DecisionConfig(**json.loads(args.config.read_text(encoding="utf-8"))),
                                 expected_discounts=read(args.expected_discounts) if args.expected_discounts else None,
                                 profile=profile)
    metadata = json.loads(args.evaluation_metadata.read_text(encoding="utf-8")) if args.evaluation_metadata else {}
    strategy = None
    if args.strategy_outcomes:
        strategy = strategy_evaluation(
            read(args.strategy_outcomes, exact_money=True), metadata,
            assignments=read(args.strategy_assignments) if args.strategy_assignments else None,
            primary_metric=metadata["primary_metric"] if args.engine_plan else "churn_rate")
    qini_models = read_batches(args.qini_predictions) if args.qini_predictions else (
        {"supplied_purchase_model": predictions["uplift"]} if "uplift" in predictions else {})
    qini = None
    if args.qini_outcomes:
        frame = (_evaluation_input(args.qini_outcomes, args.qini_assignments, ["treatment", "assignment_probability"])
                 if args.engine_plan else read(args.qini_outcomes))
        qini = qini_curves(frame, qini_models, metadata)
    output = write_outputs(result, args.output, qini=qini, strategy=strategy)
    from .pipeline import _jsonable
    print(json.dumps(_jsonable({"output": str(output), "customers": len(result.customers),
                                "budget": result.budget_summary}), allow_nan=False))


if __name__ == "__main__":
    main()
