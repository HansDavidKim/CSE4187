"""CLI for validation, training, inference, independent evaluation and experiments."""
import argparse
import sys
from .pipeline import train, infer, read_config, write_json, local_output
from .data import load_canonical
from .fixture import generate
from .simulator_input import training_inputs, simulator_imports


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train"); p.add_argument("--config", required=True); p.add_argument("--output")
    p = commands.add_parser("validate"); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--features", help="Materialized simulator features directory paired with --input raw SQLite")
    p = commands.add_parser("fixture"); p.add_argument("--output", required=True); p.add_argument("--customers", type=int, default=120)
    p = commands.add_parser("predict"); p.add_argument("--bundle", required=True); p.add_argument("--input", required=True)
    p.add_argument("--as-of", required=True); p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--bundle", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--features")
    p.add_argument("--as-of")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-training-data", action="store_true")
    p = commands.add_parser("experiment")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--customers", type=int, default=20000)
    clock = p.add_mutually_exclusive_group()
    clock.add_argument("--months", type=int, help="explicit legacy calendar-month generation")
    clock.add_argument("--weeks", type=int, choices=(104,), help="52 past +52 future weeks (default)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--integrate", action="store_true")
    if argv[:1] == ["experiment"]:
        simulator_imports()
        from simulate.v4.simulator.coupon_assignment import add_coupon_assignment_arguments, coupon_assignment_from_args
        add_coupon_assignment_arguments(p)
    args = parser.parse_args(argv)
    if args.command == "train":
        config = read_config(args.config)
        if args.output:
            config["output_dir"] = args.output
        print(train(config))
    elif args.command == "validate":
        if args.features:
            meta, x, _, _, metadata = training_inputs({"input_dir": args.input, "feature_dir": args.features})
            print(f"Verified {len(meta)} simulator samples and {len(x.columns)} predictors")
        else:
            metadata = load_canonical(args.input).metadata
            print(metadata["validation"])
        write_json(local_output(args.output) / "input_validation.json", metadata)
    elif args.command == "fixture":
        print(generate(args.output, args.customers))
    elif args.command == "evaluate":
        from .api import evaluate_retention
        result = evaluate_retention(args.bundle, args.input, args.output, args.features, args.as_of,
                                    require_independent=not args.allow_training_data)
        print(f"Evaluation status {result['label_status']}; ML AUC target {result['test_auc_target_met']}")
    elif args.command == "experiment":
        from .experiments import run_experiment
        print(run_experiment(read_config(args.config), args.output, args.customers, args.months, args.seeds, args.integrate,
                             weeks=args.weeks, coupon_assignment=coupon_assignment_from_args(args)))
    else:
        result = infer(args.bundle, args.input, args.as_of, args.output)
        print(f"Wrote {len(result)} prediction/status rows")


if __name__ == "__main__":
    main()
