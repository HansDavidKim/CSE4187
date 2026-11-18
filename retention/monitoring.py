"""Training-reference PSI/KS and mature-outcome monitoring for fixed models."""
import numpy as np
from scipy.stats import ks_2samp


def _fractions(values, cuts):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    counts = np.histogram(finite, [-np.inf, *cuts, np.inf])[0]
    counts = np.append(counts, len(values) - len(finite)).astype(float)
    return counts / max(len(values), 1)


def build_reference(x, validation_metrics, config=None):
    """Fix bin edges and reference samples using fitting rows only."""
    config = config or {}
    features = {}
    for name in x:
        finite = x[name].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        cuts = np.unique(np.quantile(finite, np.linspace(.1, .9, 9))).tolist() if len(finite) else []
        # A deterministic bounded reference keeps saved bundles small.
        sample = finite[np.linspace(0, len(finite)-1, min(4096, len(finite)), dtype=int)].tolist() if len(finite) else []
        features[name] = {"cuts": cuts, "fractions": _fractions(x[name], cuts).tolist(), "sample": sample,
                          "missing_fraction": float(1 - len(finite) / max(len(x), 1))}
    return {"features": features, "validation_metrics": validation_metrics,
            "thresholds": {"psi": float(config.get("psi_threshold", .2)),
                           "ks_alpha": float(config.get("ks_alpha", .01)),
                           "max_metric_drop": float(config.get("max_metric_drop", .05))}}


def compare(reference, x, current_metrics=None):
    """Include missingness in PSI and control KS false discoveries across features."""
    if x.empty:
        return {"status": "no_eligible_customers", "features": {}, "metrics": None,
                "thresholds": reference["thresholds"], "label_status": "pending_or_unavailable", "alerts": []}
    rows = {}
    for name, expected in reference["features"].items():
        if name not in x:
            raise ValueError(f"Monitoring input lacks training feature {name}")
        actual = _fractions(x[name], expected["cuts"])
        prior = np.asarray(expected["fractions"])
        # Smoothing is only for the distribution distance, never a model feature.
        p, q = np.clip(prior, 1e-6, 1), np.clip(actual, 1e-6, 1)
        p, q = p / p.sum(), q / q.sum()
        finite = x[name].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        ks = ks_2samp(expected["sample"], finite, method="asymp") if len(finite) and expected["sample"] else None
        rows[name] = {"rows": len(x), "psi": float(np.sum((q-p) * np.log(q/p))),
                      "missing_fraction": float(actual[-1]), "reference_missing_fraction": expected["missing_fraction"],
                      "ks_statistic": float(ks.statistic) if ks else None,
                      "ks_p_value": float(ks.pvalue) if ks else None, "ks_adjusted_p_value": None}
    ordered = sorted((row["ks_p_value"], name) for name, row in rows.items() if row["ks_p_value"] is not None)
    adjusted = 1.
    for index in range(len(ordered)-1, -1, -1):
        p_value, name = ordered[index]
        adjusted = min(adjusted, p_value * len(ordered) / (index + 1))
        rows[name]["ks_adjusted_p_value"] = float(adjusted)
    limits = reference["thresholds"]
    alerts = []
    for name, row in rows.items():
        row["psi_exceeded"] = row["psi"] >= limits["psi"]
        row["ks_exceeded"] = row["ks_adjusted_p_value"] is not None and row["ks_adjusted_p_value"] < limits["ks_alpha"]
        if row["psi_exceeded"] or row["ks_exceeded"]:
            alerts.append({"kind": "feature_drift", "feature": name})
    if current_metrics:
        for name in ("roc_auc", "precision", "recall"):
            past, current = reference["validation_metrics"].get(name), current_metrics.get(name)
            if past is not None and current is not None and past-current > limits["max_metric_drop"]:
                alerts.append({"kind": "performance_drop", "metric": name, "reference": past, "current": current})
    return {"status": "evaluated" if len(x) else "no_eligible_customers", "features": rows,
            "thresholds": limits, "metrics": current_metrics,
            "label_status": "mature" if current_metrics else "pending_or_unavailable", "alerts": alerts,
            "ks_correction": "Benjamini-Hochberg over observed feature tests"}
