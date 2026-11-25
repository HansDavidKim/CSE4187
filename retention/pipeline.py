"""Retention training, comparison, persistence and batch inference."""
from pathlib import Path
import json
import platform
import sys
import time
import importlib.metadata
import hashlib
import tempfile
import uuid
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml

from .data import load_canonical, timestamp
from .dataset import make_dataset, rolling_folds, early_stop_split, experiment_splits
from .simulator_input import (training_inputs, prediction_inputs, DEFINITION, DAYS, CONTROL_POLICY,
                              KEYS, PROVENANCE, EXTRAS, HISTORY_FEATURES, aware_utc)
from .models import make_model, input_columns, predict, fit_model, calibrated, probability_input
from .sequence import SequenceModel
from .evaluate import metrics, cluster_auc_interval, threshold_on_validation, probability_plots, shap_artifacts, subgroup_metrics
from .feature_dictionary import write_dictionary, feature_quality
from .monitoring import build_reference


def local_output(path):
    root = Path(__file__).resolve().parent
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("All outputs must stay inside retention/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def process_memory():
    """Return process-wide RSS and lifetime peak RSS using native platform counters."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged",
                                                   "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": counters.rss, "lifetime_peak_rss_bytes": counters.peak_rss}
    import resource
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"lifetime_peak_rss_bytes": int(maximum if sys.platform == "darwin" else maximum*1024)}


def validate_config(config):
    """Validate direct Python calls and YAML-based training identically."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    config = dict(config)
    if config.get("calibration", "none") not in ("none", "auto", "sigmoid"):
        raise ValueError("calibration must be none, auto or sigmoid")
    if (config.get("psi_threshold", .2) <= 0 or not 0 < config.get("ks_alpha", .01) < 1
            or not 0 < config.get("max_metric_drop", .05) <= 1):
        raise ValueError("Invalid monitoring thresholds")
    mode = config.get("input_mode", "legacy")
    if mode not in ("legacy", "simulator"):
        raise ValueError("input_mode must be simulator or legacy")
    required = ["input_dir", "output_dir"]
    required += ["feature_dir"] if mode == "simulator" else [
        "first_as_of", "last_as_of", "observation_end", "test_start", "observation_assumption"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing config: {missing}")
    if str(config.get("observation_assumption")).startswith("REPLACE"):
        raise ValueError("Set observation_assumption explicitly before training")
    if config.get("target", "visit") not in ("visit", "purchase"):
        raise ValueError("target must be visit or purchase")
    if any(config.get(k, default) <= 0 for k, default in (("horizon_days", 30), ("stride_days", 7), ("lookback_days", 30), ("max_length", 128), ("epochs", 12), ("trials", 3), ("iterations", 120), ("batch_size", 128))):
        raise ValueError("Window, length and training counts must be positive")
    if mode == "simulator":
        if config.get("target", "visit") != "visit" or config.get("horizon_days", DAYS) != DAYS:
            raise ValueError("Simulator mode uses the official 28-day no-visit definition")
        if config.get("policy_condition", "observed_experiment_mix") not in ("observed_experiment_mix", CONTROL_POLICY):
            raise ValueError("policy_condition must be observed_experiment_mix or no_new_coupon")
        config.update(target="visit", horizon_days=DAYS)
    elif timestamp(config["last_as_of"]) < timestamp(config["first_as_of"]):
        raise ValueError("last_as_of precedes first_as_of")
    return config


def read_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")))


def selected_columns(x):
    # A column entirely missing in TRAIN is excluded, irrespective of test contents.
    return [c for c in x if x[c].notna().any()]


def subset_sequences(seqs, idx):
    return [seqs[int(i)] for i in idx]


def train(config):
    """Publish a complete run atomically and permit safe retries after a failure."""
    config = validate_config(config)
    out = local_output(config["output_dir"])
    lock = out.parent / (out.name + ".lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(out))
    stage = None
    try:
        if any(out.iterdir()):
            raise ValueError("Output is nonempty; choose a new run directory")
        stage = Path(tempfile.mkdtemp(prefix="." + out.name + "-", dir=out.parent))
        staged_config = {**config, "output_dir": str(stage)}
        write_json(stage / "run_status.json", {"status": "running"})
        _train(staged_config)
        bundle = joblib.load(stage / "bundle.joblib")
        bundle["config"] = config
        joblib.dump(bundle, stage / "bundle.joblib")
        write_json(stage / "config.json", config)
        write_json(stage / "run_status.json", {"status": "complete", "model_version": bundle["model_version"]})
        out.rmdir()
        stage.replace(out)
        return out
    except Exception as exc:
        if stage is not None and stage.exists():
            write_json(stage / "run_status.json", {"status": "failed", "error": str(exc), "requested_output": str(out)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def _train(config):
    out = local_output(config["output_dir"])
    if (out / "bundle.joblib").exists():
        raise ValueError("Output already contains a model; choose a new run directory")
    started = time.monotonic(); seed = config.get("seed", 42)
    simulator = config.get("input_mode") == "simulator"
    if simulator:
        meta, x, seqs, excluded, input_validation = training_inputs(config)
        folds, train_idx, test_idx, outer_train, validation, inner_fit, inner_stop = experiment_splits(meta, seed)
    else:
        records = load_canonical(config["input_dir"])
        input_validation = records.metadata
        meta, x, seqs, excluded = make_dataset(records, config)
        folds, train_idx, test_idx = rolling_folds(meta, config["test_start"], 5)
        outer_train, validation = folds[-1]
        inner_fit, inner_stop = early_stop_split(meta, outer_train)
    write_json(out / "input_validation.json", input_validation)
    write_json(out / "config.json", config)
    print(f"Validated {len(meta)} samples and {len(x.columns)} predictors", flush=True)
    meta.to_parquet(out / "samples.parquet", index=False)
    x.to_parquet(out / "features.parquet", index=False)
    excluded.to_csv(out / "excluded.csv", index=False)
    seq_lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    np.savez_compressed(out / "sequences.npz", values=np.concatenate(seqs), lengths=seq_lengths)
    split_report = [{"fold": k, "train_rows": len(tr), "validation_rows": len(va),
                     "train_label_end": str(meta.loc[tr, "label_end"].max()),
                     "validation_start": str(meta.loc[va, "as_of"].min())}
                    for k, (tr, va) in enumerate(folds)]
    write_json(out / "splits.json", {"folds": split_report, "train_rows": len(outer_train), "development_rows": len(train_idx), "test_rows": len(test_idx),
        "type": "fixed_snapshot_customer_holdouts" if simulator else "purged_rolling_origin",
        "test_start": config.get("test_start"), "tuning_rows": len(outer_train), "selection_rows": len(validation)})
    if simulator:
        splits = meta[KEYS + ["dataset_id"]].assign(partition="test", tuning_fold=pd.NA, early_stopping=pd.NA)
        splits.loc[outer_train, "partition"] = "tuning"
        splits.loc[validation, "partition"] = "selection"
        for fold_number, (_, va) in enumerate(folds):
            splits.loc[va, "tuning_fold"] = fold_number
        splits.loc[inner_fit, "early_stopping"] = "fit"
        splits.loc[inner_stop, "early_stopping"] = "validation"
        splits.to_parquet(out / "split_membership.parquet", index=False)
    y = meta.y.to_numpy()
    feature_quality(x, outer_train).to_csv(out / "feature_quality.csv", index=False)
    input_seconds = time.monotonic() - started
    calibration_idx = np.array([], dtype=int)
    if config.get("calibration", "none") != "none":
        calibration_idx, validation = train_test_split(
            validation, test_size=.5, stratify=y[validation], random_state=seed)
        if min(np.unique(y[calibration_idx], return_counts=True)[1]) < 2:
            raise ValueError("Probability calibration requires more labelled customers")
        if simulator:
            splits.loc[calibration_idx, "partition"] = "calibration"
            splits.to_parquet(out / "split_membership.parquet", index=False)
    split_summary = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    split_summary.update(calibration_rows=len(calibration_idx), selection_rows=len(validation))
    write_json(out / "splits.json", split_summary)
    cv_rows, best_params = [], {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    iterations = config.get("iterations", 120)
    # Search only development folds. Their scores are selection evidence, not unbiased OOF.
    for name in ("lightgbm", "catboost", "xgboost"):
        def objective(trial):
            params = {"iterations": iterations,
                      "learning_rate": trial.suggest_float("learning_rate", .025, .10, log=True)}
            if name == "lightgbm":
                params["num_leaves"] = trial.suggest_categorical("num_leaves", [7, 15, 31])
                params["min_child_samples"] = trial.suggest_categorical("min_child_samples", [10, 30])
            else:
                params["depth"] = trial.suggest_int("depth", 3, 6)
            scores = []
            for fold, (tr, va) in enumerate(folds):
                cols = selected_columns(x.iloc[tr])
                model = make_model(name, params, seed)
                fit_model(model, x.iloc[tr][cols], y[tr], name)
                score = metrics(y[va], predict(model, x.iloc[va][cols]))
                cv_rows.append({"model": name, "trial": trial.number, "fold": fold, **score})
                scores.append(score["roc_auc"])
            return float(np.mean(scores))
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=config.get("trials", 3))
        best_params[name] = {"iterations": iterations, **study.best_params}
        print(name, "development 5-fold AUC", study.best_value, flush=True)
    pd.DataFrame(cv_rows).to_csv(out / "cross_validation.csv", index=False)
    write_json(out / "best_params.json", best_params)
    # Use the same selection partition for every candidate and threshold tuning.
    dev_predictions, final_models, columns, calibration_predictions = {}, {}, {}, {}
    for name in ("constant", "recency", "lightgbm", "catboost", "xgboost"):
        cols = input_columns(name, selected_columns(x.iloc[outer_train]))
        model = make_model(name, best_params.get(name), seed)
        fit_model(model, x.iloc[outer_train][cols], y[outer_train], name)
        final_models[name], columns[name] = model, cols
        dev_predictions[name] = predict(model, x.iloc[validation][cols])
        calibration_predictions[name] = predict(model, x.iloc[calibration_idx][cols]) if len(calibration_idx) else np.array([])
    # Early stopping uses only a split inside the fitting partition.
    seq_settings = {"seed": seed, "hidden": config.get("hidden", 24), "batch_size": config.get("batch_size", 128)}
    epoch_counts, sequence_models = {}, {}
    for kind in ("lstm", "gru"):
        pilot = SequenceModel(**seq_settings, kind=kind).fit(subset_sequences(seqs, inner_fit), y[inner_fit],
            validation=(subset_sequences(seqs, inner_stop), y[inner_stop]), epochs=config.get("epochs", 12), patience=3)
        epoch_counts[kind] = pilot.best_epoch
        write_json(out / f"{kind}_early_stopping.json", {"fit_max_label_end": meta.loc[inner_fit, "label_end"].max(),
            "stop_start": meta.loc[inner_stop, "as_of"].min(), "outer_validation_start": meta.loc[validation, "as_of"].min(),
            "best_epoch": pilot.best_epoch, "history": pilot.history})
        dev_seq = SequenceModel(**seq_settings, kind=kind).fit(subset_sequences(seqs, outer_train), y[outer_train], epochs=pilot.best_epoch)
        sequence_models[kind] = dev_seq
        dev_predictions[kind] = dev_seq.predict_proba(subset_sequences(seqs, validation))
        calibration_predictions[kind] = dev_seq.predict_proba(subset_sequences(seqs, calibration_idx)) if len(calibration_idx) else np.array([])
        dev_seq.save(out / f"{kind}.pt")
        write_json(out / f"{kind}_training_log.json", dev_seq.history)
    calibrators, calibration_report = {}, {}
    def calibrate_candidate(name):
        calibrator = None
        before = metrics(y[validation], dev_predictions[name])
        if len(calibration_idx):
            candidate = LogisticRegression(max_iter=500, random_state=seed).fit(
                probability_input(calibration_predictions[name]), y[calibration_idx])
            proposed = calibrated(dev_predictions[name], candidate)
            after = metrics(y[validation], proposed)
            if config["calibration"] == "sigmoid" or (after["log_loss"] < before["log_loss"] and after["brier"] < before["brier"]):
                calibrator = candidate
                dev_predictions[name] = proposed
        calibrators[name] = calibrator
        calibration_report[name] = {"method": "sigmoid" if calibrator else "none", "raw_validation": before,
                                    "selected_validation": metrics(y[validation], dev_predictions[name])}
    for name in list(dev_predictions):
        calibrate_candidate(name)
    best_tree = max(("lightgbm", "catboost", "xgboost"), key=lambda n: metrics(y[validation], dev_predictions[n])["roc_auc"])
    best_sequence = max(("lstm", "gru"), key=lambda n: metrics(y[validation], dev_predictions[n])["roc_auc"])
    # Ensemble averages component probabilities after their individually selected calibration.
    dev_predictions["ensemble"] = (dev_predictions[best_tree] + dev_predictions[best_sequence]) / 2
    calibrators["ensemble"] = None
    dev_scores = {n: metrics(y[validation], p) for n, p in dev_predictions.items()}
    if simulator and config.get("ablation", True):
        kit_columns = [name for name in x if name not in EXTRAS and not name.endswith("_relative_change_30d")]
        groups = {"kit39": kit_columns, "all_candidates": list(x),
                  "without_reviews": [name for name in x if "review" not in name]}
        if config.get("history_features", False):
            groups["without_history"] = [name for name in x if name not in HISTORY_FEATURES]
        ablations = []
        for group_name, candidates in groups.items():
            cols = selected_columns(x.iloc[outer_train][candidates])
            candidate = make_model(best_tree, best_params[best_tree], seed)
            fit_model(candidate, x.iloc[outer_train][cols], y[outer_train], best_tree)
            ablations.append({"group": group_name, "model": best_tree, "columns": len(cols),
                              "probability_scale": "uncalibrated", **metrics(y[validation], predict(candidate, x.iloc[validation][cols]))})
        ablations.append({"group": "recency", "model": "recency", "columns": 1,
                          "probability_scale": "uncalibrated", **calibration_report["recency"]["raw_validation"]})
        pd.DataFrame(ablations).to_csv(out / "feature_ablation.csv", index=False)
    winner = max(dev_scores, key=lambda n: (dev_scores[n]["roc_auc"], -dev_scores[n]["log_loss"]))
    threshold = threshold_on_validation(y[validation], dev_predictions[winner])
    thresholds = {name: threshold_on_validation(y[validation], p) for name, p in dev_predictions.items()}
    write_json(out / "selection.json", {"winner": winner, "best_tree": best_tree, "best_sequence": best_sequence,
        "threshold": threshold, "thresholds": thresholds, "calibration": calibration_report,
        "threshold_objective": "validation F1 proxy", "fitting_rows": len(outer_train),
        "calibration_rows": len(calibration_idx), "selection_rows": len(validation),
        "final_fit": "Preserve models fitted before independent selection; no train+validation refit",
        "criterion": "validation ROC-AUC then log loss", "validation": dev_scores})
    bundle = {"models": final_models, "columns": columns, "sequence_models": sequence_models, "winner": winner,
              "best_tree": best_tree, "best_sequence": best_sequence, "threshold": threshold, "config": config, "schema_version": 3, "model_version": uuid.uuid4().hex,
              "calibrators": calibrators, "thresholds": thresholds, "training_provenance": input_validation}
    if simulator:
        bundle["input_contract"] = {key: meta[key].iloc[0] for key in (
            "definition_id", "expanded_feature_set_version", "currency", "amount_unit")}
    bundle["monitoring_reference"] = build_reference(x.iloc[outer_train],
        metrics(y[validation], dev_predictions[winner], threshold), config)
    joblib.dump(bundle, out / "bundle.joblib")
    test_predictions = score(bundle, x.iloc[test_idx], subset_sequences(seqs, test_idx))
    # Verify exact score parity with the persisted complete preprocessing/model bundle.
    loaded = joblib.load(out / "bundle.joblib")
    replay = score(loaded, x.iloc[test_idx], subset_sequences(seqs, test_idx))
    for name in test_predictions:
        np.testing.assert_allclose(test_predictions[name], replay[name], atol=1e-7)
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    results = {name: {**metrics(test_meta.y, p, thresholds[name]), "auc_customer_bootstrap_95": cluster_auc_interval(test_meta, p, seed, config.get("bootstrap_repeats", 100))}
               for name, p in test_predictions.items()}
    result_frame = test_meta.copy()
    for name, p in test_predictions.items():
        result_frame[name] = p
    result_frame.to_parquet(out / "test_predictions.parquet", index=False)
    dev_frame = meta.iloc[validation].copy()
    for name, p in dev_predictions.items():
        dev_frame[name] = p
    dev_frame.to_parquet(out / "validation_predictions.parquet", index=False)
    probability_plots(test_meta, test_predictions, out)
    top_features = shap_artifacts(final_models[best_tree], x.iloc[test_idx][columns[best_tree]], out, meta=meta,
                                  model_name=best_tree, scored_model=winner, seed=seed)
    write_json(out / "metrics.json", {"validation": dev_scores, "test": results,
               "selected_on_validation": winner, "top10_shap": top_features,
               "selected_tree": best_tree, "test_auc_target_met": results[best_tree]["roc_auc"] is not None and results[best_tree]["roc_auc"] >= .78,
               "subgroups": {name: subgroup_metrics(test_meta, p, thresholds[name]) for name, p in test_predictions.items()}})
    if simulator:
        requirement_checks = {
            "at_least_30_defined_features": len(x.columns) >= 30,
            "five_relative_changes": sum(c.endswith("_relative_change_30d") for c in x) >= 5,
            "three_session_quality_features": all(c in x for c in (
                "mean_session_duration_minutes_30d", "views_per_session_30d", "orders_per_session_30d")),
            "two_sequence_summaries": all(c in x for c in ("action_group_transitions", "last_action_group_run")),
            "three_tree_comparisons": all(c in results for c in ("lightgbm", "catboost", "xgboost")),
            "ml_test_auc_078": results[best_tree]["roc_auc"] >= .78,
            "five_tuning_folds": len(folds) == 5,
            "shared_ml_dl_test": all(len(p) == len(test_idx) for p in test_predictions.values())}
        write_json(out / "requirements_report.json", {"checks": requirement_checks, "all_met": all(requirement_checks.values())})
    dictionary(x.columns, out, simulator, input_validation.get("time_windows_days"))
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "duckdb", "scikit-learn", "lightgbm", "catboost", "xgboost", "torch", "optuna", "shap")}
    write_json(out / "run_manifest.json", {"python": sys.executable, "platform": platform.platform(), "versions": versions,
        "seconds": time.monotonic() - started, "input_seconds": input_seconds,
        "model_version": bundle["model_version"], "training_provenance": input_validation,
        "process_memory": process_memory(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("pipeline.py", "models.py", "sequence.py", "simulator_input.py", "monitoring.py", "evaluate.py")}, "seed": seed, "data_role": config.get("data_role", "provided_records"),
        "observation_assumption": config.get("observation_assumption", "Verified simulator raw and labels"),
        "record_policy": config.get("policy_condition", "observed_experiment_mix" if simulator else "observed_logging_policy"),
        "reload_score_parity": True, "sequence_truncation_fraction": float(meta.truncated_groups.gt(0).mean())})
    report(config, out, results, dev_scores, winner, best_tree, best_sequence, len(x.columns), len(meta), top_features)
    print("Selected", winner, "test", results[winner], flush=True)
    return out


def score(bundle, x, seqs, names=None):
    """Score only requested models and required ensemble components."""
    requested = list(names) if names is not None else [*bundle["models"], *bundle.get("sequence_models", {"lstm": None}), "ensemble"]
    needed = set(requested)
    if "ensemble" in needed:
        needed.update((bundle["best_tree"], bundle.get("best_sequence", "lstm")))
    output = {}
    calibrators = bundle.get("calibrators", {})
    sequence_models = bundle.get("sequence_models", {"lstm": bundle.get("sequence_model")})
    for name in needed - {"ensemble"}:
        if name in bundle["models"]:
            raw = predict(bundle["models"][name], x[bundle["columns"][name]])
        else:
            raw = sequence_models[name].predict_proba(seqs)
        output[name] = calibrated(raw, calibrators.get(name))
    if "ensemble" in needed:
        output["ensemble"] = (output[bundle["best_tree"]] + output[bundle.get("best_sequence", "lstm")]) / 2
    return {name: output[name] for name in requested}


def infer(bundle_path, input_dir, as_of, output_dir=None):
    # Load only trusted, locally generated joblib files.
    bundle = joblib.load(bundle_path)
    config = dict(bundle["config"])
    if bundle.get("schema_version", 1) >= 3:
        status = Path(bundle_path).parent / "run_status.json"
        if not status.exists() or json.loads(status.read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError("The model run is incomplete; inspect run_status.json")
    config["_include_sequences"] = bundle["winner"] in ("lstm", "gru", "ensemble")
    profile = {}
    if config.get("input_mode") == "simulator":
        meta, x, seqs, excluded = prediction_inputs(input_dir, as_of, config, profile=profile)
        for key in ("expanded_feature_set_version", "currency", "amount_unit"):
            if len(meta) and not meta[key].eq(bundle["input_contract"][key]).all():
                raise ValueError(f"Prediction data differs from the trained model's {key}")
        result = meta[KEYS + PROVENANCE].copy()
        result["as_of"] = result["as_of_at"]
        result["definition_id"] = bundle["input_contract"]["definition_id"]
    else:
        records = load_canonical(input_dir)
        meta, x, seqs, excluded = make_dataset(records, config, as_ofs=[timestamp(as_of)], labeled=False)
        result = meta[["customer_id", "as_of"]].copy()
    scoring_started = time.perf_counter()
    profile["memory_after_input"] = process_memory()
    predictions = score(bundle, x, seqs, names=[bundle["winner"]]) if len(meta) else {bundle["winner"]: np.array([])}
    profile["model_scoring_seconds"] = time.perf_counter()-scoring_started
    profile["memory_after_scoring"] = process_memory()
    result["churn_probability"] = predictions[bundle["winner"]]
    result["predicted_churn"] = result.churn_probability.ge(bundle["threshold"]).astype(int)
    result["status"] = "scored"
    if len(excluded):
        excluded = excluded.rename(columns={"reason": "status"})
        result = pd.concat([result, excluded], ignore_index=True) if len(result) else excluded.reindex(columns=result.columns)
    result["model"] = bundle["winner"]
    result["target"] = config.get("target", "visit")
    result["horizon_days"] = config.get("horizon_days", 30)
    simulator = config.get("input_mode") == "simulator"
    result["policy_condition"] = config.get("policy_condition", "observed_experiment_mix" if simulator else "observed_logging_policy")
    if simulator:
        result["as_of"] = result["as_of_at"]
        result["definition_id"] = DEFINITION
    result["model_version"] = bundle.get("model_version", Path(bundle_path).resolve().parent.name)
    result["threshold"] = bundle["threshold"]
    if simulator:
        reference_point = bundle.get("training_provenance", {}).get("as_of_at")
        result["snapshot_scope"] = "other_snapshot"
        if reference_point is not None:
            result.loc[result.as_of_at.eq(aware_utc(reference_point)), "snapshot_scope"] = "training_experiment_time"
    result["predicted_churn"] = result.predicted_churn.astype("Int64")
    result.attrs["runtime_profile"] = profile
    if simulator:
        result.attrs["prediction_metadata"] = {
            "model_version": result["model_version"].iloc[0] if len(result) else bundle.get("model_version", Path(bundle_path).parent.name),
            "target_definition_id": DEFINITION, "horizon_value": 4, "horizon_unit": "weeks",
            "value_semantics": "probability"}
        result.attrs["producer_metadata"] = {
            "component": "retention", "policy_condition": config.get("policy_condition", "observed_experiment_mix"),
            "status_counts": result.status.value_counts().to_dict(), "excluded_rows": int(len(excluded)),
            "time_spec": bundle.get("training_provenance", {}).get("time_spec"),
            "time_windows_days": bundle.get("training_provenance", {}).get("time_windows_days")}
    if output_dir is not None:
        out = local_output(output_dir)
        result.to_parquet(out / "predictions.parquet", index=False)
        write_json(out / "runtime_profile.json", profile)
    return result


def dictionary(columns, out, simulator=False, time_windows_days=None):
    if simulator:
        write_dictionary(columns, out, time_windows_days=time_windows_days)
        return
    lines = ["# Retention feature dictionary", "", "Computed strictly before as_of. UTC time; default level window 30 days, change windows two adjacent 14-day periods.",
             "Null: insufficient/unknown history or undefined denominator; counts: zero for known no activity. Median imputation and indicators fit inside training only.", "", "| Feature | Definition |", "| --- | --- |"]
    definitions = {"visit_recency_days": "Days since last session start", "purchase_recency_days": "Days since last order",
        "age_days": "Days since actual signup", "journey_stage": "Past order count capped at 3: no purchase / first / repeat / loyal",
        "stage_tenure_days": "Days since signup or first/second/third past order that entered stage",
        "purchase_cycle_anomaly": "Purchase recency / mean past purchase interval; null for nonpositive/undefined gap",
        "duration_minutes_30d": "Mean duration of sessions started in window and completed before as_of",
        "views_per_session_30d": "Views in completed recent sessions / completed recent sessions",
        "purchase_session_rate_30d": "Completed recent sessions with purchase / completed recent sessions",
        "cart_session_conversion_30d": "Completed sessions with cart and purchase / completed sessions with cart",
        "action_group_transitions": "Adjacent timestamp groups with different action-presence vectors",
        "last_action_group_run": "Consecutive most-recent timestamp groups with identical action-presence vectors",
        "valid_unused_coupons": "Known prior sends not expired and not redeemed before as_of"}
    for col in columns:
        text = definitions.get(col)
        if text is None:
            if col.endswith("_change"):
                text = "(Recent 14-day value - previous 14-day value) / previous value; null when previous <= 0"
            elif col.endswith("_previous_zero"):
                text = "Indicator that previous 14-day value is zero"
            elif "_recent_14d" in col or "_previous_14d" in col:
                text = "Named activity count or gross order amount in indicated adjacent 14-day window"
            else:
                text = col.replace("_", " ") + "; see retention/features.py for executable definition"
        lines.append(f"| `{col}` | {text} |")
    (out / "feature_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(config, out, results, dev_scores, winner, tree, recurrent, features, rows, shap_top):
    simulator = config.get("input_mode") == "simulator"
    lines = ["# Retention model comparison", "", f"Data role: **{config.get('data_role', 'provided_records')}**. Target: {config.get('target', 'visit')} inactivity for {config.get('horizon_days', 30)} days.",
        "", f"Samples: {rows}; computed features: {features}. Winner selected on development validation: **{winner}**; ensemble: {tree} + {recurrent}.",
        "", "| Model | Validation AUC | Test AUC | Test PR-AUC | Test log loss | Test Brier |", "| --- | --- | --- | --- | --- | --- |"]
    for name, row in results.items():
        lines.append(f"| {name} | {dev_scores[name]['roc_auc']:.4f} | {row['roc_auc'] if row['roc_auc'] is not None else 'undefined'} | {row['pr_auc']} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines += ["", "## Interpretation and limitations", "",
        "- Smoke fixtures test schema and execution only; their metrics do not establish real-data accuracy or the assignment's AUC target.",
        ("- At the experiment cutoff, held-out customers are reserved for selection and final test. Five stratified folds tune trees on the remaining customers. Both sequence models use an inner customer split for early stopping." if simulator else
         "- Five purged rolling-origin folds tune all three trees. LSTM and GRU use the same purged inner early-stopping split and common outer validation date."),
        "- Final test was excluded from parameter, epoch, ensemble, calibration and threshold selection. Final models preserve the fitting partition used to generate validation scores.",
        "- Class weighting can impair probability calibration; inspect Brier/log loss and calibration plot before business use. No unvalidated calibration is silently applied.",
        "- ML/DL have different representations; comparisons do not isolate the causal value of event ordering.",
        f"- Prediction policy: {config.get('policy_condition', 'observed_experiment_mix' if simulator else 'observed_logging_policy')}.",
        "- SHAP explains predictive contributions, not causes of churn or coupon harm.",
        "- Customer-bootstrap intervals account for repeated customer rows but not arbitrary shared calendar shocks.",
        f"- Observation source: {config.get('observation_assumption', 'Verified simulator raw and label manifests')}", "", "## Requirements and paper alignment", "",
        "| Source requirement | Evidence |", "| --- | --- |",
        "| Verified input contract | input_validation.json and table adapters |",
        "| 30+ features, five changes, three quality features, two sequence features | feature_dictionary.md, features.parquet |",
        "| Three tree models, imbalance, tuning, 5-fold | cross_validation.csv, best_params.json |",
        "| LSTM + requested GRU, padding/embedding, Early Stopping | lstm.pt, gru.pt, *_early_stopping.json, *_training_log.json |",
        "| Same test / ML+DL ensemble | test_predictions.parquet and metrics.json |",
        "| SHAP global/local/top10 | shap_summary.png, shap_importance.csv, shap_customer.csv |",
        "| PR threshold / probability evaluation | selection.json, probability_and_pr.png |",
        "| Operating-point precision/recall/counts and subgroups | metrics.json |",
        "| Feature coverage and development ablations | feature_quality.csv, feature_ablation.csv |",
        "| Tree explanation identity and scale | shap_metadata.json, shap_customer.csv |",
        "| Paper §3.3 synthetic response != learned uplift | No latent persona/type/gap labels used as model input |",
        "| Paper §4 fixed 10% coupon, purchase-time redemption | coupon/order joins; no invented refund or discount optimization |",
        "| Root docs/results path | Local equivalents only; external directories intentionally untouched |",
        "", "Top predictive SHAP features: " + ", ".join(shap_top), ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
