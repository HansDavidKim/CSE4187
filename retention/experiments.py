"""Repeat public simulator runs and evaluate fixed Retention and optional peer models."""
import time
import json
import joblib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .api import evaluate_retention
from .pipeline import train, local_output, write_json
from .simulator_input import simulator_imports


def generate_dataset(output, customers=20000, months=None, seed=42, start_at=None, *, weeks=None,
                     coupon_assignment=None):
    """Generate a target-20% scenario using a resolved in-memory simulator config."""
    simulator_imports()
    from simulate.v4.configuration import POPULATION_CONFIG
    from simulate.v4.simulator import engine
    from simulate.v4.storage import materialize_features
    from simulate.v4.simulator.coupon_assignment import resolve_coupon_assignment, assign_coupons
    from simulate.v4.time_spec import is_weekly
    if months is not None and weeks is not None:
        raise ValueError("Choose weekly104w or explicit legacy calendar months")
    weeks = (104 if weeks is None else weeks) if months is None else None
    if weeks is not None and (type(weeks) is not int or weeks != 104):
        raise ValueError("Weekly experiments require exactly104weeks")
    out = local_output(output)
    duration = f"{weeks} weeks" if weeks is not None else f"{months} legacy calendar months"
    print(f"Generating {customers} customers, {duration}, seed {seed}", flush=True)
    config = engine.EcommerceSimulator(POPULATION_CONFIG).config
    config["source_simulation"]["retention_mode"] = "target"
    config["churn"]["target_rate"] = .2
    if start_at is not None:
        config["simulation"]["start_at"] = str(start_at)
    assignment = resolve_coupon_assignment(config["source_simulation"].get("coupon_assignment")
        if coupon_assignment is None else coupon_assignment)
    manifest = out / "dataset" / "raw.manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))["metadata"]
        expected_start = pd.Timestamp(config["simulation"]["start_at"])
        if expected_start.tzinfo is None:
            expected_start = expected_start.tz_localize("Asia/Seoul" if weeks is not None else "UTC")
        duration_matches = (is_weekly(metadata) and metadata.get("weeks") == weeks) if weeks is not None \
            else not is_weekly(metadata) and metadata.get("months") == months
        if (metadata.get("seed") != seed or not duration_matches
                or metadata.get("eligible_customers") != customers
                or metadata.get("retention_mode") != "target"
                or metadata.get("configured_target_churn_rate") != .2
                or pd.Timestamp(metadata["start_at"]) != expected_start):
            raise ValueError("Existing experiment data differs from the requested generation settings")
        settings = config["source_simulation"]
        bindings = [(engine.__file__, metadata.get("simulator_code_sha256")),
                    (settings["model_path"], metadata.get("model_sha256")),
                    (settings["return_model_path"], metadata.get("return_model_sha256")),
                    (settings["coupon_response_model_path"], metadata.get("coupon_model_sha256")),
                    (settings["visit_timing_model_path"], (metadata.get("visit_timing") or {}).get("model_sha256")),
                    (settings["reward_gmm_model_path"], (metadata.get("reward_gmm") or {}).get("model_sha256")),
                    (settings["review_model_path"], metadata.get("event_models", {}).get("review", {}).get("model_sha256")),
                    (config["toy"]["message_model_path"], metadata.get("marketing_proxy_sha256"))]
        runtime_root = Path(engine.__file__).parents[1]
        runtime = metadata.get("runtime_code_sha256", {})
        for name, digest in runtime.items():
            path = next((runtime_root / domain / f"{name}.py" for domain in ("simulator", "irl", "")
                         if (runtime_root / domain / f"{name}.py").is_file()), None)
            bindings.append((path, digest))
        if (metadata.get("configuration_sha256") != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                or not runtime or any(path is None or not Path(path).is_file()
                    or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in bindings)):
            raise ValueError("Existing experiment generator inputs changed; choose a new experiment output")
        treatments = pd.read_parquet(out / "dataset" / "treatments.parquet")
        actions, probabilities, design = assign_coupons(treatments.customer_id, assignment, seed=seed,
                                                      legacy=treatments.treatment.to_numpy())
        if (metadata.get("coupon_assignment", {"spec": {"mode": "auto"}})["spec"] != design["spec"]
                or not np.array_equal(treatments.treatment.to_numpy(), actions)
                or not np.array_equal(treatments.assignment_probability.to_numpy(float), probabilities)):
            raise ValueError("Existing experiment coupon assignment differs from the requested design")
        if not (out / "features" / "manifest.json").exists():
            materialize_features(out / "dataset", out / "features")
        print("Reusing immutable dataset", metadata["dataset_id"], flush=True)
        return out
    started = time.monotonic()
    result = engine.EcommerceSimulator(config).generate(n_customers=customers, months=months, weeks=weeks, seed=seed,
                                                      coupon_assignment=assignment)
    result.save(out / "dataset")
    generated_at = time.monotonic()
    materialize_features(out / "dataset", out / "features")
    summary = {"customers": customers, "months": months, "weeks": weeks, "seed": seed,
               "coupon_assignment": result.metadata.get("coupon_assignment"),
               "churn": float(result.tables["labels"].churn.mean()),
               "generation_seconds": generated_at-started,
               "materialization_seconds": time.monotonic()-generated_at}
    write_json(out / "generation_summary.json", summary)
    print("Generated", summary, flush=True)
    return out


def _uplift_propensity(assignments):
    probabilities = pd.to_numeric(assignments.assignment_probability, errors="raise").to_numpy(float)
    if (set(assignments.treatment) != {0, 1} or not np.isfinite(probabilities).all()
            or len(np.unique(probabilities)) != 1 or not 0 < probabilities[0] < 1):
        raise ValueError("Causal Uplift fitting requires randomized coupon assignment with both arms and positive overlap; "
                         "none, all-treated and specified-customer assignments cannot identify both outcomes")
    return float(probabilities[0])


def train_peer_models(feature_dir, raw_dir, output):
    """Use existing Uplift/CLV APIs and keep all model files under Retention."""
    simulator_imports()
    try:
        from uplift.common.simulator import load_feature_dataset
        from uplift.common import schema
        from uplift.common.evaluate import qini_coefficient
        from uplift.Tlearner import TLearner
        from uplift.Xlearner import XLearner
        from uplift.Hlearner import HLearner
        from uplift.common.base import BaseUpliftLearner
        from uplift.api import bind_training_contract, predict_for_strategy
    except ModuleNotFoundError as error:
        if error.name == "uplift" or error.name.startswith("uplift."):
            raise ModuleNotFoundError("Peer integration requires the external uplift package (T/X/H learners); "
                                      "install that project before using --integrate") from error
        raise
    from clv.api import train_clv
    data = load_feature_dataset(feature_dir, ledger=raw_dir)
    propensity = _uplift_propensity(data[[schema.TREATMENT]].assign(
        assignment_probability=data.attrs.get("assignment_probability")))
    dataset_id = data.attrs.get("dataset_id")
    raw_sha = data.raw_database_sha256.iloc[0] if "raw_database_sha256" in data else None
    data = data.dropna(subset=[schema.LABEL])
    out = local_output(output)
    selection_path = out / "uplift_selection.json"
    registry_path = out / "clv" / "registry.json"
    if selection_path.exists() and registry_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if (selection.get("dataset_id") == dataset_id
                and registry["provenance"].get("dataset_id") == dataset_id):
            if (raw_sha is None or selection.get("raw_database_sha256") != raw_sha
                    or registry["provenance"].get("raw_database_sha256") != raw_sha):
                raise ValueError("Existing peer model raw dataset changed or is unverified; choose a new experiment output")
            cached = BaseUpliftLearner.load(out / (selection["winner"] + ".pkl"))
            predict_for_strategy(cached, feature_dir, allow_training_customers=True)
            return {"uplift": cached,
                    "clv_dir": out / "clv", "clv_registry": registry}
    if any(out.iterdir()):
        raise ValueError("Existing peer output cannot be reused for this dataset; choose a new experiment output")
    fit, selection = train_test_split(data, test_size=.3, stratify=data[schema.TREATMENT], random_state=0)
    models, scores = {}, {}
    for name, cls in (("tlearner", TLearner), ("xlearner", XLearner), ("hlearner", HLearner)):
        model = cls().fit(fit[schema.FEATURES], fit[schema.TREATMENT], fit[schema.LABEL], propensity)
        bind_training_contract(model, fit, selected_customers=selection[schema.CUSTOMER_ID].tolist())
        model.save(out / (name + ".pkl"))
        prediction = model.predict(selection[schema.FEATURES])
        scores[name] = qini_coefficient(prediction.uplift_score, selection[schema.TREATMENT], selection[schema.LABEL])
        models[name] = model
    winner = max(scores, key=scores.get)
    data[["customer_id", schema.T0]].assign(partition=np.where(data.index.isin(fit.index), "fit", "selection")).to_csv(
        out / "uplift_splits.csv", index=False)
    write_json(out / "uplift_selection.json", {"winner": winner, "selection_qini": scores,
               "dataset_id": dataset_id, "target": "purchase_in_28d", "seed": 0,
               "raw_database_sha256": raw_sha,
               "assignment_probability": propensity})
    registry = train_clv(feature_dir, ledger=raw_dir, output_dir=out / "clv")
    return {"uplift": models[winner], "clv_dir": out / "clv", "clv_registry": registry}


def predict_peer_models(peers, feature_dir, raw_dir, retention_predictions, output):
    """Join scores at the same source run and cutoff, keeping the target meanings explicit."""
    from uplift.api import predict_for_strategy
    from clv.api import predict_clv, evaluate_clv
    out = local_output(output)
    keys = ["dataset_id", "customer_id", "as_of_at", "currency", "amount_unit"]
    batch = predict_for_strategy(peers["uplift"], feature_dir)
    predictions = batch.frame[[*keys, "p0", "p1", "uplift_score"]].copy()
    predictions["as_of_at"] = pd.to_datetime(predictions.as_of_at, utc=True)
    predictions = predictions.rename(columns={"p0": "purchase_probability_no_coupon", "p1": "purchase_probability_coupon"})
    cutoff = predictions.as_of_at.iloc[0]
    clv = predict_clv(feature_dir, ledger=raw_dir, model_dir=peers["clv_dir"], reference_week=cutoff)
    clv["as_of_at"] = pd.to_datetime(clv.reference_week, utc=True)
    retained = retention_predictions.copy()
    retained["as_of_at"] = pd.to_datetime(retained.as_of_at, utc=True)
    merged = retained.merge(predictions, on=keys, validate="one_to_one")
    merged = merged.merge(clv[["customer_id", "as_of_at", "predicted_clv_12m", "currency", "amount_unit", "dataset_id"]],
                          on=["customer_id", "as_of_at", "currency", "amount_unit", "dataset_id"], validate="one_to_one")
    if len(merged) != len(retained):
        raise ValueError("Peer model outputs do not cover the same eligible customers and cutoff")
    merged.to_parquet(out / "combined_predictions.parquet", index=False)
    try:
        clv_evaluation = evaluate_clv(feature_dir, ledger=raw_dir, model_dir=peers["clv_dir"], cutoff=cutoff, population="all")
    except ValueError as error:
        if str(error) != "no labelled customers to evaluate at this cutoff":
            raise
        clv_evaluation = {"status": "unavailable", "reason": str(error),
                          "scope": "No mature outcomes under the frozen CLV model's population policy"}
    write_json(out / "clv_independent_evaluation.json", clv_evaluation)
    write_json(out / "integration_contract.json", {
        "rows": len(merged), "retention": "28-day no-visit probability under the named policy",
        "uplift": "28-day purchase probability increase", "clv": "52-week paid revenue",
        "scope": "Aligned fixed-model predictions; final strategy/policy ROI requires the strategy module"})
    return merged


def run_experiment(config, output, customers=20000, months=None, seeds=(42, 43, 44), integrate=False,
                   *, weeks=None, coupon_assignment=None):
    """Train once, then monitor distinct generated cohorts without refitting."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct generation seeds")
    if months is not None and weeks is not None:
        raise ValueError("Choose weekly104w or explicit legacy calendar months")
    if integrate and months is not None:
        raise ValueError("Peer strategy integration requires weekly104w; use individual model APIs for explicit legacy calendar data")
    weeks = (104 if weeks is None else weeks) if months is None else None
    if weeks is not None and (type(weeks) is not int or weeks != 104):
        raise ValueError("Weekly experiments require exactly104weeks")
    assignment = config.get("coupon_assignment") if coupon_assignment is None else coupon_assignment
    out = local_output(output)
    runs = []
    for index, seed in enumerate(seeds):
        start = (pd.Timestamp("2024-01-01T00:00:00+09:00") + pd.Timedelta(weeks=13*index)
                 if weeks is not None else pd.Timestamp("2024-01-01") + pd.DateOffset(months=3*index))
        folder = generate_dataset(out / f"seed_{seed}", customers, months, seed, start.isoformat(),
                                  weeks=weeks, coupon_assignment=assignment)
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
        "months": months, "weeks": weeks, "coupon_assignment": assignment,
        "integrated_peers": integrate, "model_dir": str(model), "evaluation_runs": len(history)})
    return out
