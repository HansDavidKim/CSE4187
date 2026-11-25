"""Assemble existing model predictions; this command never trains a replacement model."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .inputs import PredictionBatch, load_feature_store
from .pipeline import DecisionConfig, build_decisions, write_outputs
from .evaluation import qini_curves, strategy_evaluation
from .strategy_profiles import build_strategy_profiles


def read(path):
    identifiers = ("dataset_id", "customer_id", "snapshot_id", "raw_database_sha256", "currency", "amount_unit")
    return (pd.read_parquet(path) if Path(path).suffix == ".parquet"
            else pd.read_csv(path, dtype={name: "string" for name in identifiers}))


def read_batches(path):
    batches = {}
    for role, entry in json.loads(path.read_text()).items():
        source = Path(entry["frame"])
        source = source if source.is_absolute() else path.parent/source
        batches[role] = PredictionBatch(read(source), entry["metadata"], entry.get("producer_metadata", {}))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON role -> frame path, metadata and optional producer_metadata")
    parser.add_argument("--config", type=Path, required=True, help="Explicit DecisionConfig JSON")
    parser.add_argument("--expected-discounts", type=Path)
    parser.add_argument("--profile-declarations", type=Path, help="Explicit human profile version, source and thresholds JSON")
    parser.add_argument("--profile", help="P0, H0, H1, H2, H3, H4-R or H4-V from the declared profiles")
    parser.add_argument("--strategy-outcomes", type=Path)
    parser.add_argument("--qini-outcomes", type=Path)
    parser.add_argument("--qini-predictions", type=Path, help="Optional JSON model name -> purchase probability batch for multi-model curves")
    parser.add_argument("--evaluation-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.profile_declarations) != bool(args.profile):
        parser.error("--profile-declarations and --profile must be supplied together")
    profile = None
    if args.profile:
        profiles = build_strategy_profiles(**json.loads(args.profile_declarations.read_text()))
        profile = next((item for item in profiles.policies if item.policy_id == args.profile), None)
        if profile is None:
            parser.error("Unknown declared human profile")
    predictions = read_batches(args.predictions)
    features = load_feature_store(args.features)
    result = build_decisions(features, predictions, DecisionConfig(**json.loads(args.config.read_text())),
                             expected_discounts=read(args.expected_discounts) if args.expected_discounts else None,
                             profile=profile)
    metadata = json.loads(args.evaluation_metadata.read_text()) if args.evaluation_metadata else {}
    strategy = strategy_evaluation(read(args.strategy_outcomes), metadata) if args.strategy_outcomes else None
    qini_models = read_batches(args.qini_predictions) if args.qini_predictions else (
        {"supplied_purchase_model": predictions["uplift"]} if "uplift" in predictions else {})
    qini = qini_curves(read(args.qini_outcomes), qini_models, metadata) if args.qini_outcomes else None
    output = write_outputs(result, args.output, qini=qini, strategy=strategy)
    print(json.dumps({"output": str(output), "customers": len(result.customers), "budget": result.budget_summary}, default=str))


if __name__ == "__main__":
    main()
