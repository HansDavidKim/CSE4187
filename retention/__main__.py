"""CLI: python -m retention {validate,train,predict,fixture}."""
import argparse
from .pipeline import train, infer, read_config, write_json, local_output
from .data import load_canonical
from .fixture import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train"); p.add_argument("--config", required=True); p.add_argument("--output")
    p = commands.add_parser("validate"); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p = commands.add_parser("fixture"); p.add_argument("--output", required=True); p.add_argument("--customers", type=int, default=120)
    p = commands.add_parser("predict"); p.add_argument("--bundle", required=True); p.add_argument("--input", required=True)
    p.add_argument("--as-of", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        config = read_config(args.config)
        if args.output:
            config["output_dir"] = args.output
        print(train(config))
    elif args.command == "validate":
        records = load_canonical(args.input)
        write_json(local_output(args.output) / "input_validation.json", records.metadata)
        print(records.metadata["validation"])
    elif args.command == "fixture":
        print(generate(args.output, args.customers))
    else:
        result = infer(args.bundle, args.input, args.as_of, args.output)
        print(f"Wrote {len(result)} prediction/status rows")


if __name__ == "__main__":
    main()
