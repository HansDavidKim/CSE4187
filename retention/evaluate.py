"""Probability quality, customer-cluster intervals, PR and SHAP artifacts."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss, brier_score_loss,
                             precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.calibration import calibration_curve


def metrics(y, p, threshold=None):
    """Report probability quality and the operating point, including small cohorts."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not len(y):
        return {"rows": 0, "prevalence": None, "roc_auc": None, "pr_auc": None, "log_loss": None, "brier": None}
    result = {"rows": len(y), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
        "prevalence": float(np.mean(y)), "mean_probability": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if np.sum(y) else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}
    if threshold is not None:
        decisions = p >= threshold
        result.update(threshold=float(threshold), precision=float(precision_score(y, decisions, zero_division=0)),
                      recall=float(recall_score(y, decisions, zero_division=0)),
                      f1=float(f1_score(y, decisions, zero_division=0)),
                      flagged=int(decisions.sum()), flagged_fraction=float(decisions.mean()),
                      confusion_matrix=confusion_matrix(y, decisions, labels=[0, 1]).tolist())
    return result


def subgroup_metrics(meta, p, threshold):
    """Keep policy and observed-history subgroup sizes visible."""
    output = {}
    for column in ("policy_id", "history_group"):
        if column in meta:
            output[column] = {}
            for key, idx in meta.groupby(column, dropna=False).indices.items():
                output[column][str(key)] = metrics(meta.iloc[idx].y, np.asarray(p)[idx], threshold)
    return output


def cluster_auc_interval(meta, p, seed=42, repeats=100):
    groups = list(meta.groupby("customer_id", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    values, y = [], meta.y.to_numpy()
    for _ in range(repeats):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return list(map(float, np.quantile(values, [.025, .975]))) if values else None


def threshold_on_validation(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def probability_plots(meta, predictions, out):
    out = Path(out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in predictions.items():
        observed, predicted = calibration_curve(meta.y, p, n_bins=10, strategy="quantile")
        axes[0].plot(predicted, observed, marker=".", label=name)
        precision, recall, _ = precision_recall_curve(meta.y, p)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set(xlabel="Predicted churn", ylabel="Observed churn")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "probability_and_pr.png", dpi=150); plt.close(fig)


def shap_artifacts(model, x, out, meta=None, model_name="tree", scored_model=None, seed=42):
    """Explain a reproducible tree sample and identify the explained customer."""
    import shap
    sample = x.sample(n=min(200, len(x)), random_state=seed)
    learner = model.steps[-1][1]
    data = model[:-1].transform(sample)
    names = list(model[:-1].get_feature_names_out(x.columns))
    explainer = shap.TreeExplainer(learner)
    values = explainer.shap_values(data)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    ranking = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    ranking.to_csv(Path(out) / "shap_importance.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        shap.summary_plot(values, data, feature_names=names, max_display=10, show=False)
    plt.tight_layout(); plt.savefig(Path(out) / "shap_summary.png", dpi=150); plt.close()
    expected = np.asarray(explainer.expected_value).reshape(-1)[-1]
    local = pd.DataFrame({"feature": names, "value": data[0], "shap_value": values[0],
                          "explained_model": model_name, "scored_model": scored_model or model_name,
                          "expected_value": float(expected), "output_scale": str(explainer.model.model_output)})
    if meta is not None:
        row = meta.loc[sample.index[0]]
        for key in ("dataset_id", "customer_id", "as_of_at", "as_of", "snapshot_id"):
            if key in row:
                local[key] = row[key]
    local.to_csv(Path(out) / "shap_customer.csv", index=False)
    (Path(out) / "shap_metadata.json").write_text(json.dumps({
        "explained_model": model_name, "scored_model": scored_model or model_name,
        "scope": "Tree raw output before optional probability calibration",
        "feature_values": "After train-fitted clipping and imputation",
        "output_scale": str(explainer.model.model_output), "sample_rows": len(sample), "seed": seed,
        "expected_value": float(expected)}, indent=2), encoding="utf-8")
    return ranking.head(10).feature.tolist()
