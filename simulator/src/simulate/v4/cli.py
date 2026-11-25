"""Canonical six-event simulator commands, alongside the original research CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulate.v4.configuration import DEFAULT_CONFIG, load_config
from simulate.v4.simulator.coupon_assignment import add_coupon_assignment_arguments, coupon_assignment_from_args


def add_parser(subparsers):
    parser = subparsers.add_parser("v4", help="six-event simulator, KIT39 features and SQLite")
    commands = parser.add_subparsers(dest="v4_command", required=True)
    generate = commands.add_parser("generate", help="generate immutable raw.sqlite and explicit sidecars")
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate.add_argument("--assets-root", type=Path, help="root containing data/models and source datasets")
    generate.add_argument("--output", type=Path, required=True, help="new or empty dataset directory")
    generate.add_argument("--mode", choices=("small", "full"), default="small")
    generate.add_argument("--n-customers", type=int)
    calendar = generate.add_mutually_exclusive_group()
    calendar.add_argument("--weeks", type=int, choices=(104,), help="52 past +52 future weeks (default)")
    calendar.add_argument("--months", type=int, help="explicit legacy calendar export")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--customers", type=Path, help="canonical customer CSV or Parquet")
    generate.add_argument("--products", type=Path, help="canonical fixed-price catalog CSV or Parquet")
    generate.add_argument("--fit-missing", action="store_true", help="fit missing components from configured real sources")
    add_coupon_assignment_arguments(generate)
    features = commands.add_parser("features", help="materialize KIT39 and available weekly/experiment tables")
    features.add_argument("--data", type=Path, required=True, help="dataset directory or raw.sqlite")
    features.add_argument("--output", type=Path, required=True, help="new or empty output directory")
    features.add_argument("--as-of-at", help="timezone-aware prediction time for a raw-only dataset")
    validate = commands.add_parser("validate", help="validate raw foreign keys, cart, money and event links")
    validate.add_argument("--data", type=Path, required=True)
    export = commands.add_parser("export-experiment", help="export factual weekly logs and isolated paired evaluation")
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--evaluation-output", type=Path, required=True, help="separate path containing evaluation_only")
    export.add_argument("--repetitions", type=int, default=32)


def run(args):
    if args.v4_command == "generate":
        from simulate.v4.data.make_dataset import ensure_source_models
        from simulate.v4.simulator import EcommerceSimulator

        assignment = coupon_assignment_from_args(args)
        simulator = EcommerceSimulator(load_config(args.config, args.assets_root))
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f"dataset output must be new or empty: {args.output}")
        if not args.fit_missing:
            required = [Path(value) for key, value in simulator.settings.items()
                        if key.endswith("model_path") and value]
            required += [Path(simulator.config["toy"]["message_model_path"])]
            if not simulator.settings.get("coupon_response_model_path"):
                required.append(Path(simulator.config["toy"]["coupon_model_path"]))
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing trained assets; set --assets-root or use --fit-missing with real sources:\n"
                                        + "\n".join(missing))
        ensure_source_models(simulator)
        result = simulator.generate(mode=args.mode, n_customers=args.n_customers, months=args.months,
                                    seed=args.seed, customers=args.customers, products=args.products,
                                    weeks=getattr(args,"weeks",None), coupon_assignment=assignment)
        result.save(args.output)
        summary = {"dataset": str(args.output), "customers": len(result.tables["customers"]),
                   "events": len(result.tables["events"]), "dataset_id": result.metadata["dataset_id"]}
    elif args.v4_command == "features":
        from simulate.v4.storage import materialize_features

        summary = materialize_features(args.data, args.output, as_of_at=args.as_of_at)
    elif args.v4_command == "validate":
        from simulate.v4.data.loader import check_generation_sidecars
        from simulate.v4.simulator.raw_store import load_raw_dataset

        database = args.data / "raw.sqlite" if args.data.is_dir() else args.data
        tables, metadata = load_raw_dataset(database)
        check_generation_sidecars(database, metadata)
        summary = {"valid": True, "dataset_id": metadata.get("dataset_id"),
                   "rows": {name: len(frame) for name, frame in tables.items()}}
    else:
        import pandas as pd
        from simulate.v4.data.experiment_exports import export_tables
        from simulate.v4.data.loader import load_tables

        tables = load_tables(args.data)
        directory = args.data if args.data.is_dir() else args.data.parent
        tables["diagnostics"] = pd.read_parquet(directory / "diagnostics.parquet")
        summary = export_tables(tables, tables["_metadata"], args.output,
                                args.evaluation_output, args.repetitions)
    print(json.dumps(summary, ensure_ascii=False, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_parser(commands)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
