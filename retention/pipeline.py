"""Canonical seven-table training, comparison, persistence and batch inference."""
from pathlib import Path
import json
import platform
import sys
import time
import importlib.metadata

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml

from .data import load_canonical, timestamp
from .dataset import make_dataset, rolling_folds, early_stop_split
from .models import make_model, input_columns, predict, fit_model
from .sequence import SequenceModel
from .evaluate import metrics, cluster_auc_interval, threshold_on_validation, probability_plots, shap_artifacts


def local_output(path):
    root = Path(__file__).resolve().parent
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("All outputs must stay inside retention/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    required = ["input_dir", "output_dir", "first_as_of", "last_as_of", "observation_end", "test_start", "observation_assumption"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing config: {missing}")
    if str(config["observation_assumption"]).startswith("REPLACE"):
        raise ValueError("Set observation_assumption explicitly before training")
    if config.get("target", "visit") not in ("visit", "purchase"):
        raise ValueError("target must be visit or purchase")
    if any(config.get(k, default) <= 0 for k, default in (("horizon_days", 30), ("stride_days", 7), ("lookback_days", 30), ("max_length", 128), ("epochs", 12), ("trials", 3), ("iterations", 120), ("batch_size", 128))):
        raise ValueError("Window, length and training counts must be positive")
    if timestamp(config["last_as_of"]) < timestamp(config["first_as_of"]):
        raise ValueError("last_as_of precedes first_as_of")
    return config


def selected_columns(x):
    # A column entirely missing in TRAIN is excluded, irrespective of test contents.
    return [c for c in x if x[c].notna().any()]


def subset_sequences(seqs, idx):
    return [seqs[int(i)] for i in idx]


def train(config):
    out = local_output(config["output_dir"])
    if (out / "bundle.joblib").exists():
        raise ValueError("Output already contains a model; choose a new run directory")
    started = time.monotonic(); seed = config.get("seed", 42)
    records = load_canonical(config["input_dir"])
    write_json(out / "input_validation.json", records.metadata)
    write_json(out / "config.json", config)
    print("Input validated", records.metadata["validation"]["counts"], flush=True)
    meta, x, seqs, excluded = make_dataset(records, config)
    meta.to_parquet(out / "samples.parquet", index=False)
    x.to_parquet(out / "features.parquet", index=False)
    excluded.to_csv(out / "excluded.csv", index=False)
    seq_lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    np.savez_compressed(out / "sequences.npz", values=np.concatenate(seqs), lengths=seq_lengths)
    folds, train_idx, test_idx = rolling_folds(meta, config["test_start"], 5)
    split_report = [{"fold": k, "train_rows": len(tr), "validation_rows": len(va),
                     "train_label_end": str(meta.loc[tr, "label_end"].max()),
                     "validation_start": str(meta.loc[va, "as_of"].min())}
                    for k, (tr, va) in enumerate(folds)]
    write_json(out / "splits.json", {"folds": split_report, "train_rows": len(train_idx), "test_rows": len(test_idx),
                                    "test_start": config["test_start"]})
    y = meta.y.to_numpy()
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
    # One common outer validation date for all final candidates and threshold selection.
    outer_train, validation = folds[-1]
    dev_predictions, validation_models, validation_cols = {}, {}, {}
    for name in ("constant", "recency", "lightgbm", "catboost", "xgboost"):
        cols = input_columns(name, selected_columns(x.iloc[outer_train]))
        model = make_model(name, best_params.get(name), seed)
        fit_model(model, x.iloc[outer_train][cols], y[outer_train], name)
        dev_predictions[name] = predict(model, x.iloc[validation][cols])
        validation_models[name], validation_cols[name] = model, cols
    # Early stopping entirely INSIDE outer training; outer validation never stops epochs.
    inner_fit, inner_stop = early_stop_split(meta, outer_train)
    seq_settings = {"seed": seed, "hidden": config.get("hidden", 24), "batch_size": config.get("batch_size", 128)}
    epoch_counts = {}
    for kind in ("lstm", "gru"):
        pilot = SequenceModel(**seq_settings, kind=kind).fit(subset_sequences(seqs, inner_fit), y[inner_fit],
            validation=(subset_sequences(seqs, inner_stop), y[inner_stop]), epochs=config.get("epochs", 12), patience=3)
        epoch_counts[kind] = pilot.best_epoch
        write_json(out / f"{kind}_early_stopping.json", {"fit_max_label_end": meta.loc[inner_fit, "label_end"].max(),
            "stop_start": meta.loc[inner_stop, "as_of"].min(), "outer_validation_start": meta.loc[validation, "as_of"].min(),
            "best_epoch": pilot.best_epoch, "history": pilot.history})
        dev_seq = SequenceModel(**seq_settings, kind=kind).fit(subset_sequences(seqs, outer_train), y[outer_train], epochs=pilot.best_epoch)
        dev_predictions[kind] = dev_seq.predict_proba(subset_sequences(seqs, validation))
    best_tree = max(("lightgbm", "catboost", "xgboost"), key=lambda n: metrics(y[validation], dev_predictions[n])["roc_auc"])
    best_sequence = max(("lstm", "gru"), key=lambda n: metrics(y[validation], dev_predictions[n])["roc_auc"])
    dev_predictions["ensemble"] = (dev_predictions[best_tree] + dev_predictions[best_sequence]) / 2
    dev_scores = {n: metrics(y[validation], p) for n, p in dev_predictions.items()}
    winner = max(dev_scores, key=lambda n: (dev_scores[n]["roc_auc"], -dev_scores[n]["log_loss"]))
    threshold = threshold_on_validation(y[validation], dev_predictions[winner])
    write_json(out / "selection.json", {"winner": winner, "best_tree": best_tree, "best_sequence": best_sequence, "threshold": threshold,
        "threshold_objective": "validation F1 proxy; no business cost function supplied",
        "criterion": "common validation ROC-AUC, then log loss", "validation": dev_scores,
        "warning": "Development scores reused for tuning/selection; not unbiased or strict OOF."})
    final_models, columns, test_predictions = {}, {}, {}
    for name in ("constant", "recency", "lightgbm", "catboost", "xgboost"):
        cols = input_columns(name, selected_columns(x.iloc[train_idx]))
        model = make_model(name, best_params.get(name), seed)
        fit_model(model, x.iloc[train_idx][cols], y[train_idx], name)
        final_models[name], columns[name] = model, cols
        test_predictions[name] = predict(model, x.iloc[test_idx][cols])
    sequence_models = {}
    for kind in ("lstm", "gru"):
        final_seq = SequenceModel(**seq_settings, kind=kind).fit(subset_sequences(seqs, train_idx), y[train_idx], epochs=epoch_counts[kind])
        sequence_models[kind] = final_seq
        test_predictions[kind] = final_seq.predict_proba(subset_sequences(seqs, test_idx))
        final_seq.save(out / f"{kind}.pt")
        write_json(out / f"{kind}_training_log.json", final_seq.history)
    test_predictions["ensemble"] = (test_predictions[best_tree] + test_predictions[best_sequence]) / 2
    bundle = {"models": final_models, "columns": columns, "sequence_models": sequence_models, "winner": winner,
              "best_tree": best_tree, "best_sequence": best_sequence, "threshold": threshold, "config": config, "schema_version": 2}
    joblib.dump(bundle, out / "bundle.joblib")
    # Verify exact score parity with the persisted complete preprocessing/model bundle.
    loaded = joblib.load(out / "bundle.joblib")
    replay = score(loaded, x.iloc[test_idx], subset_sequences(seqs, test_idx))
    for name in test_predictions:
        np.testing.assert_allclose(test_predictions[name], replay[name], atol=1e-7)
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    results = {name: {**metrics(test_meta.y, p), "auc_customer_bootstrap_95": cluster_auc_interval(test_meta, p, seed, config.get("bootstrap_repeats", 100))}
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
    top_features = shap_artifacts(final_models[best_tree], x.iloc[test_idx][columns[best_tree]], out)
    write_json(out / "metrics.json", {"validation": dev_scores, "test": results,
               "selected_on_validation": winner, "top10_shap": top_features,
               "test_auc_target_met": results[winner]["roc_auc"] is not None and results[winner]["roc_auc"] >= .78})
    dictionary(x.columns, out)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "duckdb", "scikit-learn", "lightgbm", "catboost", "xgboost", "torch", "optuna", "shap")}
    write_json(out / "run_manifest.json", {"python": sys.executable, "platform": platform.platform(), "versions": versions,
        "seconds": time.monotonic() - started, "seed": seed, "data_role": config.get("data_role", "provided_records"),
        "observation_assumption": config["observation_assumption"], "record_policy": "observed logging policy, not no-send causal risk",
        "reload_score_parity": True, "sequence_truncation_fraction": float(meta.truncated_groups.gt(0).mean())})
    report(config, out, results, dev_scores, winner, best_tree, best_sequence, len(x.columns), len(meta), top_features)
    print("Selected", winner, "test", results[winner], flush=True)
    return out


def score(bundle, x, seqs):
    output = {name: predict(model, x[bundle["columns"][name]]) for name, model in bundle["models"].items()}
    sequence_models = bundle.get("sequence_models", {"lstm": bundle.get("sequence_model")})
    for kind, model in sequence_models.items():
        output[kind] = model.predict_proba(seqs)
    output["ensemble"] = (output[bundle["best_tree"]] + output[bundle.get("best_sequence", "lstm")]) / 2
    return output


def infer(bundle_path, input_dir, as_of, output_dir):
    # Load only trusted, locally generated joblib files.
    bundle = joblib.load(bundle_path)
    records = load_canonical(input_dir)
    config = bundle["config"]
    meta, x, seqs, excluded = make_dataset(records, config, as_ofs=[timestamp(as_of)], labeled=False)
    predictions = score(bundle, x, seqs) if len(meta) else {bundle["winner"]: np.array([])}
    result = meta[["customer_id", "as_of"]].copy()
    result["churn_probability"] = predictions[bundle["winner"]]
    result["predicted_churn"] = result.churn_probability.ge(bundle["threshold"]).astype(int)
    result["status"] = "scored"
    result["model"] = bundle["winner"]
    result["target"] = config.get("target", "visit")
    result["horizon_days"] = config.get("horizon_days", 30)
    result["policy_condition"] = "observed_logging_policy"
    result["model_version"] = Path(bundle_path).resolve().parent.name
    if len(excluded):
        excluded = excluded.rename(columns={"reason": "status"})
        result = pd.concat([result, excluded], ignore_index=True) if len(result) else excluded.reindex(columns=result.columns)
    result["model"] = bundle["winner"]
    result["target"] = config.get("target", "visit")
    result["horizon_days"] = config.get("horizon_days", 30)
    result["policy_condition"] = "observed_logging_policy"
    result["model_version"] = Path(bundle_path).resolve().parent.name
    result["predicted_churn"] = result.predicted_churn.astype("Int64")
    out = local_output(output_dir)
    result.to_parquet(out / "predictions.parquet", index=False)
    return result


def dictionary(columns, out):
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
    lines = ["# Retention model comparison", "", f"Data role: **{config.get('data_role', 'provided_records')}**. Target: {config.get('target', 'visit')} inactivity for {config.get('horizon_days', 30)} days.",
        "", f"Samples: {rows}; computed features: {features}. Winner selected on development validation: **{winner}**; ensemble: {tree} + {recurrent}.",
        "", "| Model | Validation AUC | Test AUC | Test PR-AUC | Test log loss | Test Brier |", "| --- | --- | --- | --- | --- | --- |"]
    for name, row in results.items():
        lines.append(f"| {name} | {dev_scores[name]['roc_auc']:.4f} | {row['roc_auc'] if row['roc_auc'] is not None else 'undefined'} | {row['pr_auc']} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines += ["", "## Interpretation and limitations", "",
        "- Smoke fixtures test schema and execution only; their metrics do not establish real-data accuracy or the assignment's AUC target.",
        "- Five purged rolling-origin folds tune all three trees. LSTM and GRU use the same purged inner early-stopping split and common outer validation date.",
        "- Final test was excluded from all parameter, epoch, ensemble and threshold selection. Development scores can be optimistic after selection.",
        "- Class weighting can impair probability calibration; inspect Brier/log loss and calibration plot before business use. No unvalidated calibration is silently applied.",
        "- ML/DL have different representations; comparisons do not isolate the causal value of event ordering.",
        "- Predictions describe observed logging policy, not identified no-coupon counterfactual risk.",
        "- SHAP explains predictive contributions, not causes of churn or coupon harm.",
        "- Customer-bootstrap intervals account for repeated customer rows but not arbitrary shared calendar shocks.",
        f"- Observation assumption: {config['observation_assumption']}", "", "## Requirements and paper alignment", "",
        "| Source requirement | Evidence |", "| --- | --- |",
        "| Root Feature contract: seven tables | input_validation.json and table adapters |",
        "| 30+ features, five changes, three quality features, two sequence features | feature_dictionary.md, features.parquet |",
        "| Three tree models, imbalance, tuning, 5-fold | cross_validation.csv, best_params.json |",
        "| LSTM + requested GRU, padding/embedding, Early Stopping | lstm.pt, gru.pt, *_early_stopping.json, *_training_log.json |",
        "| Same test / ML+DL ensemble | test_predictions.parquet and metrics.json |",
        "| SHAP global/local/top10 | shap_summary.png, shap_importance.csv, shap_customer.csv |",
        "| PR threshold / probability evaluation | selection.json, probability_and_pr.png |",
        "| Paper §3.3 synthetic response != learned uplift | No latent persona/type/gap labels used as model input |",
        "| Paper §4 fixed 10% coupon, purchase-time redemption | coupon/order joins; no invented refund or discount optimization |",
        "| Root docs/results path | Local equivalents only; external directories intentionally untouched |",
        "", "Top predictive SHAP features: " + ", ".join(shap_top), ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
