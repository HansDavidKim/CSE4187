"""Fit a Gaussian mixture by EM on source-customer MAP reward and return effects."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from sklearn.mixture import GaussianMixture

from simulate.v4.irl.fit_donor_prior import load_prior_source
from simulate.v4.irl.fit_gaussian_rewards import FIT_SCOPE, PARAMETERS, _fit_map_effects, _sha256
from simulate.v4.irl.fit_joint_latent_shopping import validate_model
from simulate.v4.irl.reward_gmm import validate_gmm


TYPE = "source_map_reward_gmm"
MAX_COMPONENTS = 6
REG_COVAR = 1e-6


def fit_mixture(effects, seed):
    """Select covariance family and number of components by BIC, then refit."""
    effects = np.asarray(effects, dtype=float)
    if (effects.ndim != 2 or effects.shape[1] != len(PARAMETERS)
            or len(effects) < 5 or not np.isfinite(effects).all()):
        raise ValueError("reward GMM requires at least five finite source effects")
    indices = np.random.default_rng(seed).permutation(len(effects))
    split = len(indices) // 5
    train, check = effects[indices[split:]], effects[indices[:split]]
    trials = []
    for family in ("diag", "tied", "full"):
        for k in range(1, min(MAX_COMPONENTS, len(train)) + 1):
            trial = GaussianMixture(
                n_components=k, covariance_type=family, reg_covar=REG_COVAR, random_state=seed, n_init=8)
            trial.fit(train)
            trials.append({
                "components": k, "covariance_type": family, "converged": bool(trial.converged_),
                "bic": float(trial.bic(train)), "check_vector_log_density": float(trial.score(check)),
            })
    candidates = [row for row in trials if row["components"] >= 2 and row["converged"]]
    if not candidates:
        raise ValueError("no multi-component reward GMM candidate converged")
    selected = min(candidates, key=lambda row: row["bic"])
    fitted = GaussianMixture(
        n_components=selected["components"], covariance_type=selected["covariance_type"],
        reg_covar=REG_COVAR, random_state=seed, n_init=8).fit(effects)
    if not fitted.converged_:
        raise ValueError("final reward GMM EM did not converge")
    covariances = fitted.covariances_
    if fitted.covariance_type == "diag":
        covariances = np.stack([np.diag(values) for values in covariances])
    elif fitted.covariance_type == "tied":
        covariances = np.repeat(covariances[None, :, :], fitted.n_components, axis=0)
    return fitted, covariances, {
        "criterion": "minimum BIC among converged mixtures with at least two components",
        "vector_partitions": {"selection": len(train), "density_check": len(check), "final_fit": len(effects)},
        "covariance_regularization": REG_COVAR, "seed": seed, "initializations": fitted.n_init,
        "trials": trials, "selected": selected, "final_iterations": int(fitted.n_iter_),
        "final_mean_vector_log_density": float(fitted.score(effects)),
    }


def fit_reward_gmm(rows, model, return_model, max_users=256):
    _, effects, map_report = _fit_map_effects(rows, model, return_model, max_users)
    fitted, covariances, selection = fit_mixture(effects, int(model.get("seed", 42)))
    return {
        "schema_version": 1, "type": TYPE, "purpose": "synthesis_only",
        "parameter_names": PARAMETERS, "component_count": fitted.n_components,
        "mixture_weights": fitted.weights_.tolist(), "means": fitted.means_.tolist(),
        "covariances": covariances.tolist(),
        "fit": {
            "scope": FIT_SCOPE, "method": "maximum-likelihood EM on fitted source-customer MAP effects",
            "assignment": "learned mixture weights", "forecast_validation_uses_gaussian": False,
            **map_report, "selection": selection,
        },
        "limitations": [
            "The GMM describes shrinkage-affected customer effect estimates in a frozen MDP.",
            "The vector density check shares the shrinkage selection and is distinct from behavior validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "return-model", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--max-users", type=int, default=256)
    args = parser.parse_args()
    paths = (args.base, args.return_model, args.source_cache, args.output, args.report)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("GMM source, models, sidecar and report paths must differ")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"immutable GMM output exists: {path}")

    started = time.perf_counter()
    base_bytes = args.base.read_bytes()
    return_bytes = args.return_model.read_bytes()
    base_sha, return_sha = (sha256(content).hexdigest() for content in (base_bytes, return_bytes))
    model = json.loads(base_bytes)
    validate_model(model, return_sha)
    rows, source = load_prior_source(model, args.source_cache)
    result = fit_reward_gmm(rows, model, json.loads(return_bytes), args.max_users)
    result.update(base_model_sha256=base_sha, return_model_sha256=return_sha,
        provenance={"source_cache_sha256": _sha256(args.source_cache), "source_fingerprint": source["fingerprint"],
            "fitter_sha256": _sha256(__file__)})
    validate_gmm(result, base_sha, return_sha)
    report = {"passed": True, **result, "source_descriptor": source, "elapsed_seconds": time.perf_counter() - started}
    payloads = [(path, json.dumps(payload, indent=2, allow_nan=False))
                for path, payload in ((args.output, result), (args.report, report))]
    created = []
    try:
        for path, text in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(text)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    print(json.dumps({"model_path": str(args.output), "users": result["fit"]["supported_users"],
        "components": result["component_count"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
