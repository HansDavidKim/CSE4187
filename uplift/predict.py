"""Export one saved Uplift model as a final_strategy PredictionBatch."""
import argparse
import json
from pathlib import Path

from .api import predict_for_strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-training-customers", action="store_true", help="development checks only")
    args = parser.parse_args()
    result = predict_for_strategy(args.model, args.features, allow_training_customers=args.allow_training_customers)
    args.output.mkdir(parents=True, exist_ok=False)
    file = args.output / "predictions.parquet"
    result.frame.to_parquet(file, index=False)
    declaration = {"uplift": {"frame": str(file.resolve()), "metadata": result.metadata,
                              "producer_metadata": result.producer_metadata}}
    (args.output / "prediction_batches.json").write_text(json.dumps(declaration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
