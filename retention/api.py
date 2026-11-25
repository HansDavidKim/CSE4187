"""Public Retention prediction and independent-run evaluation APIs."""
from pathlib import Path
from dataclasses import dataclass
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, fisher_exact, chi2_contingency

from .evaluate import metrics, subgroup_metrics, cluster_auc_interval, probability_plots
from .monitoring import compare
from .pipeline import infer, local_output, score, write_json, process_memory
from .pipeline import train as train_retention
from .simulator_input import (training_inputs, prediction_inputs, read_training_raw, aware_utc,
                              CONTROL_POLICY, DAYS, DEFINITION, KEYS)


def predict_retention(bundle_path, input_dir, as_of, output_dir=None):
    """Return customer probabilities and exclusion statuses without requiring a file output."""
    return infer(bundle_path, input_dir, as_of, output_dir)


@dataclass(frozen=True)
class StrategyPredictions:
    """Same snapshot and declaration fields as the CLV strategy handoff."""
    frame: pd.DataFrame
    metadata: dict
    producer_metadata: dict


def predict_for_strategy(bundle_path, input_dir, as_of):
    """Expose the 4-week churn probability without changing the legacy prediction API."""
    predictions = infer(bundle_path, input_dir, as_of)
    if "prediction_metadata" not in predictions.attrs:
        raise ValueError("Strategy handoff requires the simulator 28-day no-visit contract")
    frame = predictions[["dataset_id", *KEYS, "raw_database_sha256", "currency", "amount_unit",
                         "expanded_feature_set_version", "churn_probability", "status"]].copy()
    frame = frame.rename(columns={"churn_probability": "churn_prob"})
    frame["reference_week"] = frame["as_of_at"]
    frame["input_cutoff_at"] = frame["as_of_at"]
    frame["prediction_end_at"] = frame["as_of_at"] + pd.Timedelta(days=DAYS)
    return StrategyPredictions(frame, predictions.attrs["prediction_metadata"],
                               predictions.attrs["producer_metadata"])


def _evaluation_inputs(bundle, input_dir, feature_dir, as_of):
    config = {**bundle["config"], "input_dir": input_dir, "feature_dir": feature_dir}
    if feature_dir is not None:
        meta, x, seqs, excluded, provenance = training_inputs(config)
        if as_of is not None and not meta.as_of_at.eq(aware_utc(as_of)).all():
            raise ValueError("Evaluation as_of differs from the official feature snapshot")
        return meta, x, seqs, excluded, provenance
    path = Path(input_dir)
    if path.is_dir():
        path = path / "raw.sqlite"
    declared = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))["metadata"]
    point = aware_utc(as_of if as_of is not None else declared["as_of_at"])
    tables, provenance = read_training_raw(input_dir, point)
    meta, x, seqs, excluded = prediction_inputs(input_dir, point, config)
    end = point + pd.Timedelta(days=DAYS)
    mature = end <= aware_utc(provenance["observation_end_at"])
    meta["label_status"] = "mature" if mature else "immature"
    meta["y"] = np.nan
    if mature:
        sessions = tables["sessions"]
        returned = sessions.loc[sessions.started_at.ge(point) & sessions.started_at.lt(end), "customer_id"]
        meta["y"] = (~meta.customer_id.isin(returned)).astype(int)
        coupons = tables["coupons"]
        sends = coupons.loc[coupons.sent_at.ge(point) & coupons.sent_at.lt(end)]
        policies = sends.groupby("customer_id").policy_id.first()
        meta["policy_id"] = meta.customer_id.map(policies).fillna(CONTROL_POLICY)
        if config.get("policy_condition") == CONTROL_POLICY:
            indices = np.flatnonzero(meta.policy_id.eq(CONTROL_POLICY))
            meta, x = meta.iloc[indices].reset_index(drop=True), x.iloc[indices].reset_index(drop=True)
            seqs = [seqs[i] for i in indices]
    return meta, x, seqs, excluded, provenance


def ab_analysis(meta, baseline_rate=.2, absolute_effect=.05):
    """Describe actual randomized groups and a predeclared two-proportion sample design."""
    if not 0 < absolute_effect < baseline_rate < 1:
        raise ValueError("A/B design requires 0 < absolute_effect < baseline_rate < 1")
    p0, p1 = baseline_rate, baseline_rate-absolute_effect
    mean = (p0+p1)/2
    n = int(np.ceil(((norm.ppf(.975)*np.sqrt(2*mean*(1-mean))
                      +norm.ppf(.8)*np.sqrt(p0*(1-p0)+p1*(1-p1)))/absolute_effect)**2))
    result = {"design": {"alpha": .05, "target_power": .8, "baseline_churn": p0,
                          "absolute_churn_reduction": absolute_effect, "required_per_arm": n,
                          "method": "normal approximation, two-sided proportions"}, "status": "groups_unavailable"}
    if "policy_id" not in meta or meta.y.isna().any():
        return result
    control = meta.loc[meta.policy_id.eq(CONTROL_POLICY), "y"].astype(int)
    treated = meta.loc[~meta.policy_id.eq(CONTROL_POLICY), "y"].astype(int)
    if not len(control) or not len(treated):
        return result
    counts = [[int(treated.sum()), int(len(treated)-treated.sum())],
              [int(control.sum()), int(len(control)-control.sum())]]
    table = np.asarray(counts)
    if (table < 5).any() or (table.sum(axis=0) == 0).any():
        test, p_value = "fisher_exact", float(fisher_exact(table).pvalue)
    else:
        test, p_value = "chi_square", float(chi2_contingency(table, correction=False).pvalue)
    difference = float(treated.mean()-control.mean())
    error = float(np.sqrt(treated.mean()*(1-treated.mean())/len(treated)
                          +control.mean()*(1-control.mean())/len(control)))
    result.update(status="evaluated", control_rows=len(control), treatment_rows=len(treated),
                  control_churn=float(control.mean()), treatment_churn=float(treated.mean()),
                  treatment_minus_control=difference, difference_95_ci=[difference-1.96*error, difference+1.96*error],
                  test=test, p_value=p_value, significant=bool(p_value < .05),
                  difference_ci_method="Normal approximation for difference in proportions",
                  design_sample_size_met=bool(min(len(control), len(treated)) >= n),
                  definition_id=DEFINITION, horizon_days=DAYS,
                  scope="Whole randomized cohort; strategy targeting needs its own independent policy evaluation")
    return result


def evaluate_retention(bundle_path, input_dir, output_dir, feature_dir=None, as_of=None,
                       require_independent=True):
    """Evaluate frozen models on a new run and save PSI/KS, subgroup and A/B evidence."""
    started = time.monotonic()
    bundle_path = Path(bundle_path)
    bundle = joblib.load(bundle_path)
    if bundle.get("schema_version", 1) >= 3:
        state = json.loads((bundle_path.parent / "run_status.json").read_text(encoding="utf-8"))
        if state["status"] != "complete":
            raise ValueError("Evaluation requires a complete model run")
    if bundle["config"].get("input_mode") != "simulator":
        raise ValueError("Independent-run evaluation requires simulator inputs")
    meta, x, seqs, excluded, provenance = _evaluation_inputs(bundle, input_dir, feature_dir, as_of)
    training_id = bundle.get("training_provenance", {}).get("dataset_id")
    independent = provenance["dataset_id"] != training_id
    training_seed = bundle.get("training_provenance", {}).get("seed")
    if training_seed is not None and provenance.get("seed") == training_seed:
        independent = False
    if require_independent and not independent:
        raise ValueError("Independent evaluation needs a new dataset and generation seed")
    for key in ("currency", "amount_unit"):
        if str(provenance[key]) != str(bundle["input_contract"][key]):
            raise ValueError(f"Evaluation {key} differs from the model")
    if len(meta) and not meta.expanded_feature_set_version.eq(bundle["input_contract"]["expanded_feature_set_version"]).all():
        raise ValueError("Evaluation feature version differs from the model")
    loaded_at = time.monotonic()
    predictions = score(bundle, x, seqs) if len(meta) else {}
    mature = bool(len(meta) and meta.y.notna().all())
    thresholds = bundle.get("thresholds", {})
    by_model = {name: {**metrics(meta.y, p, thresholds.get(name, bundle["threshold"])),
                       "auc_customer_bootstrap_95": cluster_auc_interval(meta, p, repeats=bundle["config"].get("bootstrap_repeats", 100))}
                for name, p in predictions.items()} if mature else {}
    winner = bundle["winner"]
    if "monitoring_reference" not in bundle:
        raise ValueError("Re-train this legacy bundle to establish a monitoring reference")
    monitoring = compare(bundle["monitoring_reference"], x, by_model.get(winner))
    snapshot = str(meta.as_of_at.iloc[0]) if len(meta) else str(as_of or provenance.get("as_of_at"))
    monitoring.update(dataset_id=provenance["dataset_id"], as_of_at=snapshot,
                      model_version=bundle.get("model_version"), independent_run=independent)
    selected_auc = by_model.get(bundle["best_tree"], {}).get("roc_auc")
    report = {"dataset_id": provenance["dataset_id"], "training_dataset_id": training_id,
              "independent_run": independent, "as_of_at": snapshot, "provenance": provenance,
              "model_version": bundle.get("model_version"), "winner": winner, "selected_tree": bundle["best_tree"],
              "models": by_model, "excluded_rows": len(excluded), "label_status": "mature" if mature else "pending_or_unavailable",
              "test_auc_target_met": selected_auc is not None and selected_auc >= .78,
              "subgroups": subgroup_metrics(meta, predictions[winner], bundle["threshold"]) if mature else {},
              "input_seconds": loaded_at-started, "seconds": time.monotonic()-started}
    report["process_memory"] = process_memory()
    out = local_output(output_dir)
    frame = meta.copy()
    for name, p in predictions.items():
        frame[name] = p
    frame.to_parquet(out / "evaluated_predictions.parquet", index=False)
    if mature:
        probability_plots(meta, predictions, out)
    excluded.to_csv(out / "excluded.csv", index=False)
    write_json(out / "evaluation.json", report)
    write_json(out / "monitoring_report.json", monitoring)
    ab_meta = meta
    if feature_dir is not None and bundle["config"].get("policy_condition") == CONTROL_POLICY:
        from .simulator_input import simulator_imports
        loader, _, _, _, _ = simulator_imports()
        ab_meta, _, ab_y = loader(feature_dir)
        ab_meta = ab_meta.assign(y=ab_y.to_numpy())
    ab_report = ab_analysis(ab_meta)
    write_json(out / "ab_test_report.json", ab_report)
    lines = ["# Observed randomized-cohort A/B analysis", "",
             "Target is actual 28-day no-visit churn. Model predictions do not define the outcome.", "",
             "```json", json.dumps(ab_report, indent=2), "```", "",
             "Sample design uses baseline 20% churn, a five-percentage-point reduction, alpha 0.05 and target power 0.8.",
             "Strategy-specific targeting and ROI are evaluated by the strategy module."]
    (out / "ab_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
