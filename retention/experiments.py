"""Repeat public simulator runs and evaluate fixed Retention and optional peer models."""
import time
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .api import evaluate_retention
from .pipeline import train, local_output, write_json
from .simulator_input import simulator_imports


def generate_dataset(output, customers=20000, months=24, seed=42, start_at=None):
    """Generate a target-20% scenario using a resolved in-memory simulator config."""
    simulator_imports()
    from simulate.v4.configuration import POPULATION_CONFIG
    from simulate.v4.simulator.engine import EcommerceSimulator
    from simulate.v4.storage import materialize_features
    out = local_output(output)
    print(f"Generating {customers} customers, {months} months, seed {seed}", flush=True)
    config = EcommerceSimulator(POPULATION_CONFIG).config
    config["source_simulation"]["retention_mode"] = "target"
    config["churn"]["target_rate"] = .2
    if start_at is not None:
        config["simulation"]["start_at"] = str(start_at)
    manifest = out / "dataset" / "raw.manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))["metadata"]
        expected_start = pd.Timestamp(config["simulation"]["start_at"])
        if expected_start.tzinfo is None:
            expected_start = expected_start.tz_localize("UTC")
        if (metadata.get("seed") != seed or metadata.get("months") != months
                or metadata.get("eligible_customers") != customers
                or metadata.get("retention_mode") != "target"
                or metadata.get("configured_target_churn_rate") != .2
                or pd.Timestamp(metadata["start_at"]) != expected_start):
            raise ValueError("Existing experiment data differs from the requested generation settings")
        if not (out / "features" / "manifest.json").exists():
            materialize_features(out / "dataset", out / "features")
        print("Reusing immutable dataset", metadata["dataset_id"], flush=True)
        return out
    started = time.monotonic()
    result = EcommerceSimulator(config).generate(n_customers=customers, months=months, seed=seed)
    result.save(out / "dataset")
    generated_at = time.monotonic()
    materialize_features(out / "dataset", out / "features")
    summary = {"customers": customers, "months": months, "seed": seed,
               "churn": float(result.tables["labels"].churn.mean()),
               "generation_seconds": generated_at-started,
               "materialization_seconds": time.monotonic()-generated_at}
    write_json(out / "generation_summary.json", summary)
    print("Generated", summary, flush=True)
    return out


def train_peer_models(feature_dir, raw_dir, output):
    """Use existing Uplift/CLV APIs and keep all model files under Retention."""
    simulator_imports()
    from simulate.v4.data.loader import load_uplift_inputs
    from uplift.common.features import build_dataset
    from uplift.common import schema
    from uplift.common.evaluate import qini_coefficient
    from uplift.Tlearner import TLearner
    from uplift.Xlearner import XLearner
    from uplift.Hlearner import HLearner
    from uplift.common.base import BaseUpliftLearner
    from clv.api import train_clv
    out = local_output(output)
    customers, weekly, cart = load_uplift_inputs(feature_dir)
    selection_path = out / "uplift_selection.json"
    registry_path = out / "clv" / "registry.json"
    if selection_path.exists() and registry_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        dataset_id = customers.attrs.get("dataset_id")
        if (selection.get("dataset_id") == dataset_id
                and registry["provenance"].get("dataset_id") == dataset_id):
            return {"uplift": BaseUpliftLearner.load(out / (selection["winner"] + ".pkl")),
                    "clv_dir": out / "clv", "clv_registry": registry}
    data = build_dataset(customers, weekly, cart).dropna(subset=[schema.LABEL])
    fit, selection = train_test_split(data, test_size=.3, stratify=data[schema.TREATMENT], random_state=0)
    models, scores = {}, {}
    for name, cls in (("tlearner", TLearner), ("xlearner", XLearner), ("hlearner", HLearner)):
        model = cls().fit(fit[schema.FEATURES], fit[schema.TREATMENT], fit[schema.LABEL], .5)
        model.save(out / (name + ".pkl"))
        prediction = model.predict(selection[schema.FEATURES])
        scores[name] = qini_coefficient(prediction.uplift_score, selection[schema.TREATMENT], selection[schema.LABEL])
        models[name] = model
    winner = max(scores, key=scores.get)
    data[["customer_id", "w0"]].assign(partition=np.where(data.index.isin(fit.index), "fit", "selection")).to_csv(
        out / "uplift_splits.csv", index=False)
    write_json(out / "uplift_selection.json", {"winner": winner, "selection_qini": scores,
               "dataset_id": customers.attrs.get("dataset_id"), "target": "purchase_in_28d", "seed": 0})
    registry = train_clv(feature_dir, ledger=raw_dir, output_dir=out / "clv")
    return {"uplift": models[winner], "clv_dir": out / "clv", "clv_registry": registry}


def predict_peer_models(peers, feature_dir, raw_dir, retention_predictions, output):
    """Join scores at the same source run and cutoff, keeping the target meanings explicit."""
    from simulate.v4.data.loader import load_uplift_inputs
    from uplift.common.features import build_dataset
    from uplift.common import schema
    from clv.api import predict_clv, evaluate_clv
    out = local_output(output)
    data = build_dataset(*load_uplift_inputs(feature_dir))
    predictions = peers["uplift"].predict(data[schema.FEATURES]).reset_index(drop=True)
    predictions["customer_id"] = data.customer_id.to_numpy()
    predictions["as_of_at"] = pd.to_datetime(data.w0, utc=True).to_numpy()
    predictions["as_of_at"] = pd.to_datetime(predictions.as_of_at, utc=True)
    predictions = predictions.rename(columns={"p0": "purchase_probability_no_coupon", "p1": "purchase_probability_coupon"})
    cutoff = data.w0.iloc[0]
    clv = predict_clv(feature_dir, ledger=raw_dir, model_dir=peers["clv_dir"], reference_week=cutoff)
    clv["as_of_at"] = pd.to_datetime(clv.reference_week, utc=True)
    retained = retention_predictions.copy()
    retained["as_of_at"] = pd.to_datetime(retained.as_of_at, utc=True)
    merged = retained.merge(predictions, on=["customer_id", "as_of_at"], validate="one_to_one")
    merged = merged.merge(clv[["customer_id", "as_of_at", "predicted_clv_12m", "currency", "amount_unit", "dataset_id"]],
                          on=["customer_id", "as_of_at", "currency", "amount_unit", "dataset_id"], validate="one_to_one")
    if len(merged) != len(retained):
        raise ValueError("Peer model outputs do not cover the same eligible customers and cutoff")
    merged.to_parquet(out / "combined_predictions.parquet", index=False)
    clv_evaluation = evaluate_clv(feature_dir, ledger=raw_dir, model_dir=peers["clv_dir"], cutoff=cutoff, population="all")
    write_json(out / "clv_independent_evaluation.json", clv_evaluation)
    write_json(out / "integration_contract.json", {
        "rows": len(merged), "retention": "28-day no-visit probability under the named policy",
        "uplift": "28-day purchase probability increase", "clv": "52-week paid revenue",
        "scope": "Aligned fixed-model predictions; final strategy/policy ROI requires the strategy module"})
    return merged


def run_experiment(config, output, customers=20000, months=24, seeds=(42, 43, 44), integrate=False):
    """Train once, then monitor distinct generated cohorts without refitting."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct generation seeds")
    out = local_output(output)
    runs = []
    for index, seed in enumerate(seeds):
        start = pd.Timestamp("2024-01-01") + pd.DateOffset(months=3*index)
        folder = generate_dataset(out / f"seed_{seed}", customers, months, seed, start.isoformat())
        runs.append(folder)
    development = runs[0]
    training_config = {**config, "input_mode": "simulator", "input_dir": str(development / "dataset"),
                       "feature_dir": str(development / "features"), "output_dir": str(out / "model")}
    model = out / "model"
    if (model / "run_status.json").exists() and json.loads((model / "run_status.json").read_text())["status"] == "complete":
        existing = joblib.load(model / "bundle.joblib")
        if existing["config"] != training_config:
            raise ValueError("Existing model has different training settings; choose a new experiment output")
        manifest = json.loads((development / "dataset" / "raw.manifest.json").read_text(encoding="utf-8"))
        if existing["training_provenance"]["raw_database_sha256"] != manifest["database_sha256"]:
            raise ValueError("Existing model was trained on a different raw dataset")
    else:
        model = train(training_config)
    peers = train_peer_models(development / "features", development / "dataset", out / "peers") if integrate else None
    history = []
    for folder in runs[1:]:
        evaluation = evaluate_retention(model / "bundle.joblib", folder / "dataset", folder / "evaluation", folder / "features")
        history.append({"dataset_id": evaluation["dataset_id"], "as_of_at": evaluation["as_of_at"],
                        "model_version": evaluation["model_version"], **evaluation["models"][evaluation["winner"]]})
        if peers is not None:
            bundle = joblib.load(model / "bundle.joblib")
            frame = pd.read_parquet(folder / "evaluation" / "evaluated_predictions.parquet")
            frame["churn_probability"] = frame[bundle["winner"]]
            predict_peer_models(peers, folder / "features", folder / "dataset", frame, folder / "integration")
    pd.DataFrame(history).to_csv(out / "monitoring_history.csv", index=False)
    write_json(out / "experiment_manifest.json", {"seeds": list(seeds), "customers": customers,
        "months": months, "integrated_peers": integrate, "model_dir": str(model), "evaluation_runs": len(history)})
    return out
