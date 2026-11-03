"""Probability quality, customer-cluster intervals, PR and SHAP artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss, brier_score_loss,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.calibration import calibration_curve


def metrics(y, p):
    return {"rows": len(y), "prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if np.sum(y) else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}


def cluster_auc_interval(meta, p, seed=42, repeats=100):
    users = meta.customer_id.unique(); rng = np.random.default_rng(seed)
    groups = {u: np.flatnonzero(meta.customer_id.to_numpy() == u) for u in users}
    values, y = [], meta.y.to_numpy()
    for _ in range(repeats):
        idx = np.concatenate([groups[u] for u in rng.choice(users, len(users), replace=True)])
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


def shap_artifacts(model, x, out):
    import shap
    imputer, learner = model.steps[0][1], model.steps[-1][1]
    data = imputer.transform(x.iloc[:200])
    names = list(imputer.get_feature_names_out(x.columns))
    explainer = shap.TreeExplainer(learner)
    values = explainer.shap_values(data)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    ranking = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    ranking.to_csv(Path(out) / "shap_importance.csv", index=False)
    shap.summary_plot(values, data, feature_names=names, max_display=10, show=False)
    plt.tight_layout(); plt.savefig(Path(out) / "shap_summary.png", dpi=150); plt.close()
    pd.DataFrame({"feature": names, "value": data[0], "shap_value": values[0]}).to_csv(Path(out) / "shap_customer.csv", index=False)
    return ranking.head(10).feature.tolist()
