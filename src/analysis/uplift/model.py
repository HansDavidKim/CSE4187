"""Calibrate the simulator's explicit coupon assumptions from Tianchi summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from utils.paths import PROJECT_ROOT

PERSONA_PROXY = {
    0: "P4_purchase_path_proxy",
    2: "P2_deep_consideration_proxy",
    3: "P3_cart_revision_proxy",
}
PERSONA_RETURN_GROUP = {0: "purchase_only", 2: "click_purchase", 3: "click_only"}
CANONICAL_TYPES = ("sure_thing", "persuadable", "sleeping_dog", "lost_cause")


def _logit(value: float) -> float:
    value = np.clip(value, 1e-6, 1 - 1e-6)
    return float(np.log(value / (1 - value)))


def maximum_entropy_types(p0: float, p1: float) -> dict[str, float]:
    """Independent potential outcomes: maximum-entropy joint for fixed marginals."""
    return {
        "sure_thing": p0 * p1,
        "persuadable": (1 - p0) * p1,
        "sleeping_dog": p0 * (1 - p1),
        "lost_cause": (1 - p0) * (1 - p1),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(values - logsumexp(values, axis=1, keepdims=True))


def fit_latent_canonical_assignment(
    model: dict, embeddings: pd.DataFrame, seed: int = 42
) -> dict:
    """Fit one regularized latent-to-canonical matrix from persona-level targets."""
    z_columns = sorted(
        (column for column in embeddings if column.startswith("z_")),
        key=lambda column: int(column[2:]),
    )
    frame = embeddings.loc[embeddings.cluster.isin(PERSONA_PROXY)].copy()
    raw = frame[z_columns].to_numpy(float)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.
    latent = (raw - mean) / scale
    x = np.column_stack((np.ones(len(latent)), latent))
    personas = frame.cluster.to_numpy(int)
    heldout = pd.util.hash_pandas_object(
        "canonical:" + frame.user_id.astype(str), index=False
    ).to_numpy() % 5 == 0
    for persona in PERSONA_PROXY:
        if not np.any(heldout & (personas == persona)) or not np.any(~heldout & (personas == persona)):
            raise ValueError(f"persona {persona} needs users in both latent fit splits")

    def full_matrix(free):
        free = free.reshape(x.shape[1], len(CANONICAL_TYPES) - 1)
        return np.column_stack((free, -free.sum(axis=1)))

    def fit(train, l1, l2):
        def objective(free):
            matrix = full_matrix(free)
            probability = _softmax(x @ matrix)
            gradient = np.zeros_like(matrix)
            loss = 0.
            for persona in PERSONA_PROXY:
                index = train & (personas == persona)
                group = probability[index]
                target = np.array([
                    model["personas"][str(persona)]["canonical_type_probabilities"][name]
                    for name in CANONICAL_TYPES
                ])
                average = group.mean(axis=0)
                loss -= np.sum(target * np.log(average + 1e-15)) / len(PERSONA_PROXY)
                derivative = -target / (average + 1e-15)
                logits_gradient = group * (
                    derivative - (group * derivative).sum(axis=1, keepdims=True)
                ) / len(group) / len(PERSONA_PROXY)
                gradient += x[index].T @ logits_gradient
            penalized = matrix[1:]
            loss += l1 * np.sqrt(np.square(penalized) + 1e-8).sum()
            loss += .5 * l2 * np.square(penalized).sum()
            penalty_gradient = np.zeros_like(matrix)
            penalty_gradient[1:] = (
                l1 * penalized / np.sqrt(np.square(penalized) + 1e-8)
                + l2 * penalized
            )
            gradient += penalty_gradient
            free_gradient = gradient[:, :-1] - gradient[:, [-1]]
            return float(loss), free_gradient.ravel()

        result = minimize(
            objective, np.zeros(x.shape[1] * (len(CANONICAL_TYPES) - 1)),
            method="L-BFGS-B", jac=True, options={"maxiter": 1000, "ftol": 1e-11},
        )
        if not result.success:
            raise RuntimeError(f"canonical assignment failed: {result.message}")
        return full_matrix(result.x)

    def distribution_rmse(matrix, subset):
        probability = _softmax(x @ matrix)
        errors = []
        rows = []
        for persona in PERSONA_PROXY:
            predicted = probability[subset & (personas == persona)].mean(axis=0)
            target = np.array([
                model["personas"][str(persona)]["canonical_type_probabilities"][name]
                for name in CANONICAL_TYPES
            ])
            errors.extend(predicted - target)
            rows.append((persona, predicted, target))
        return float(np.sqrt(np.mean(np.square(errors)))), rows

    grid = []
    candidates = ((1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2))
    for l1, l2 in candidates:
        matrix = fit(~heldout, l1, l2)
        rmse, _ = distribution_rmse(matrix, heldout)
        grid.append({"l1": l1, "l2": l2, "heldout_rmse": rmse})
    acceptable = [row for row in grid if row["heldout_rmse"] <= .01]
    selected = max(acceptable, key=lambda row: row["l1"] + row["l2"]) if acceptable else min(
        grid, key=lambda row: row["heldout_rmse"]
    )
    matrix = fit(np.ones(len(frame), dtype=bool), selected["l1"], selected["l2"])
    final_rmse, rows = distribution_rmse(matrix, np.ones(len(frame), dtype=bool))
    distributions = {}
    for persona in PERSONA_PROXY:
        values = latent[personas == persona]
        covariance = np.cov(values, rowvar=False) + np.eye(values.shape[1]) * 1e-4
        distributions[str(persona)] = {
            "mean": values.mean(axis=0).tolist(),
            "covariance": covariance.tolist(),
            "cholesky": np.linalg.cholesky(covariance).tolist(),
        }
    model["canonical_assignment"] = {
        "model": "standardized 48D GRU latent plus intercept to four canonical logits",
        "types": list(CANONICAL_TYPES),
        "identification": "four matrix columns sum to zero",
        "matrix": matrix.tolist(),
        "latent_mean": mean.tolist(), "latent_scale": scale.tolist(),
        "l1": selected["l1"], "l2": selected["l2"],
        "selection": "strongest penalty pair with held-out persona-distribution RMSE <= 0.01",
        "grid": grid, "final_persona_distribution_rmse": final_rmse,
        "persona_fit": [
            {
                "persona": int(persona),
                "predicted": dict(zip(CANONICAL_TYPES, predicted.tolist())),
                "target": dict(zip(CANONICAL_TYPES, target.tolist())),
            }
            for persona, predicted, target in rows
        ],
        "persona_latent_gaussians": distributions,
        "limitation": (
            "weak supervision uses persona-level canonical targets; individual canonical "
            "labels are unavailable and within-persona latent gradients are regularized assumptions"
        ),
    }
    return model


def calibrate(persona_summary: dict, reactivation_summary: list[dict]) -> dict:
    proxy = {row["proxy_persona"]: row for row in persona_summary["results"]}
    return_rows = {(row["group"], row["anchor_receipt"]): row for row in reactivation_summary}
    personas = {}
    for persona, proxy_name in PERSONA_PROXY.items():
        row = proxy[proxy_name]
        p0, p1 = row["purchase_14d_control"], row["purchase_14d_treated"]
        group = PERSONA_RETURN_GROUP[persona]
        r0 = return_rows[(group, 0)]["return_7d"] / return_rows[(group, 0)]["n"]
        r1 = return_rows[(group, 1)]["return_7d"] / return_rows[(group, 1)]["n"]
        personas[str(persona)] = {
            "tianchi_proxy": proxy_name,
            "reactivation_proxy": group,
            "purchase_14d_control": p0,
            "purchase_14d_receipt": p1,
            "purchase_logit_shift": _logit(p1) - _logit(p0),
            "return_7d_control": r0,
            "return_7d_receipt": r1,
            "return_logit_shift": _logit(r1) - _logit(r0),
            "canonical_type_probabilities": maximum_entropy_types(p0, p1),
        }
    default_strength = 4.0
    for specification in personas.values():
        specification["initial_offset_by_type"] = {
            "sure_thing": {"purchase": 0.0, "return": 0.0},
            "persuadable": {
                "purchase": default_strength * specification["purchase_logit_shift"],
                "return": default_strength * specification["return_logit_shift"],
            },
            "sleeping_dog": {
                "purchase": -default_strength * specification["purchase_logit_shift"],
                "return": -default_strength * specification["return_logit_shift"],
            },
            "lost_cause": {"purchase": 0.0, "return": 0.0},
        }
    return {
        "model": "Tianchi-calibrated coupon reward and return offsets",
        "canonical_coupling": "maximum-entropy independent Y(0),Y(1) joint at fixed marginals",
        "type_multipliers": {
            "sure_thing": 0.0, "persuadable": 1.0,
            "sleeping_dog": -1.0, "lost_cause": 0.0,
        },
        "decay": {
            "kernel": "[max(0, 1-(week-1)/5)]^rho for weeks 1..6; zero afterwards",
            "rho": 2.0, "horizon_days": 42,
        },
        "repeated_receipts": "sqrt(sum(decay(age_days)))",
        "default_strength": default_strength,
        "strength_selection": (
            "4 is only the common initializer; final persona strengths are learned "
            "by the separate simulated-method-of-moments calibration"
        ),
        "personas": personas,
        "baseline_invariance": "no receipts or zero strength gives exactly zero offset",
        "interpretation": (
            "simulation calibration from observational Tianchi contrasts; canonical labels "
            "and offset magnitudes are explicit assumptions, not identified causal effects"
        ),
    }


def _baseline_models():
    baseline_dir = PROJECT_ROOT / "artifacts/cosmetics-shop-persona-quadratic-irl"
    coefficient_model = json.loads((
        PROJECT_ROOT / "artifacts/cosmetics-shop-persona-maxent-irl/coefficient_gmm.json"
    ).read_text())
    action_model = json.loads((baseline_dir / "lifecycle_irl_model.json").read_text())
    gap_model = json.loads((baseline_dir / "gap_token_irl_model.json").read_text())
    return coefficient_model, action_model, gap_model


def _persona_purchase_14d(frame: pd.DataFrame) -> pd.Series:
    eligible = frame.loc[frame.calendar_day.lt(14)]
    purchase = eligible.groupby(["home_persona", "user_id"]).purchase.sum().gt(0)
    return purchase.groupby(level=0).mean()


def _set_persona_strengths(model: dict, strengths: dict[str, float]) -> None:
    model["strength_by_persona"] = {key: float(value) for key, value in strengths.items()}
    for persona, specification in model["personas"].items():
        strength = strengths[persona]
        for canonical_type, multiplier in model["type_multipliers"].items():
            specification["initial_offset_by_type"][canonical_type] = {
                "purchase": multiplier * strength * specification["purchase_logit_shift"],
                "return": multiplier * strength * specification["return_logit_shift"],
            }


def fit_persona_strengths(
    model: dict, users: int = 6_000, seed: int = 2026,
    max_iterations: int = 4, tolerance: float = .01,
) -> dict:
    """Calibrate persona strengths by simulated method of moments on a separate seed."""
    from simulate.latent_session_model import simulate_no_uplift_users, simulate_uplift_users

    coefficient_model, action_model, gap_model = _baseline_models()
    baseline = pd.DataFrame(simulate_no_uplift_users(
        coefficient_model, action_model, gap_model, users=users, seed=seed
    )["sessions"])
    baseline_rate = _persona_purchase_14d(baseline)
    targets = {
        persona: specification["purchase_14d_receipt"] - specification["purchase_14d_control"]
        for persona, specification in model["personas"].items()
    }
    strengths = {persona: model["default_strength"] for persona in model["personas"]}
    history = []
    for iteration in range(max_iterations):
        treated = pd.DataFrame(simulate_uplift_users(
            coefficient_model, action_model, gap_model, model,
            coupon_strength=strengths, users=users, seed=seed,
        )["sessions"])
        receipt_rate = _persona_purchase_14d(treated)
        simulated = {
            persona: float(receipt_rate.loc[int(persona)] - baseline_rate.loc[int(persona)])
            for persona in strengths
        }
        errors = {persona: simulated[persona] - targets[persona] for persona in strengths}
        history.append({
            "iteration": iteration,
            "strengths": strengths.copy(), "simulated_uplift": simulated,
            "target_uplift": targets, "errors": errors,
            "rmse": float(np.sqrt(np.mean(np.square(list(errors.values()))))),
        })
        if max(abs(value) for value in errors.values()) <= tolerance:
            break
        for persona in strengths:
            if simulated[persona] <= 0:
                raise RuntimeError(f"non-positive simulated uplift for persona {persona}")
            strengths[persona] *= targets[persona] / simulated[persona]
    selected = min(history, key=lambda row: row["rmse"])
    strengths = selected["strengths"]
    _set_persona_strengths(model, strengths)
    model["strength_training"] = {
        "method": "simulated method of moments with multiplicative updates",
        "calibration_users": users, "seed": seed,
        "target": "persona-specific Tianchi proxy 14-day purchase difference",
        "tolerance": tolerance, "max_iterations": max_iterations,
        "history": history,
        "selected_iteration": selected["iteration"],
        "final_strengths": strengths,
        "validation_seed": 42,
    }
    return model


def run(output: Path) -> dict:
    persona = json.loads((PROJECT_ROOT / "artifacts/tianchi-persona-uplift/summary.json").read_text())
    reactivation = json.loads((PROJECT_ROOT / "artifacts/coupon-reactivation/summary.json").read_text())
    model = calibrate(persona, reactivation)
    embeddings = pd.read_parquet(
        PROJECT_ROOT / "artifacts/cosmetics-shop-personas/clusters.parquet"
    )
    model = fit_latent_canonical_assignment(model, embeddings)
    model = fit_persona_strengths(model)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model


def validate_rollout(model: dict, output: Path, users: int = 10_000, seed: int = 42) -> dict:
    from simulate.latent_session_model import simulate_no_uplift_users, simulate_uplift_users

    coefficient_model, action_model, gap_model = _baseline_models()
    baseline = pd.DataFrame(simulate_no_uplift_users(
        coefficient_model, action_model, gap_model, users=users, seed=seed
    )["sessions"])
    treated_rollout = simulate_uplift_users(
        coefficient_model, action_model, gap_model, model,
        users=users, seed=seed,
    )
    treated = pd.DataFrame(treated_rollout["sessions"])
    sampled_profiles = pd.DataFrame(treated_rollout["profiles"])
    probabilities = np.stack(sampled_profiles.pop("canonical_probabilities"))
    latents = np.stack(sampled_profiles.pop("normalized_latent"))
    for index, name in enumerate(model["canonical_assignment"]["types"]):
        sampled_profiles[f"probability_{name}"] = probabilities[:, index]
    for index in range(latents.shape[1]):
        sampled_profiles[f"z_{index}"] = latents[:, index]
    sampled_profiles.to_parquet(output / "sampled_user_uplift_profiles.parquet", index=False)
    zero_users = min(users, 1_000)
    zero = pd.DataFrame(simulate_uplift_users(
        coefficient_model, action_model, gap_model, model,
        coupon_strength=0., users=zero_users, seed=seed,
    )["sessions"])
    treated.to_parquet(output / "single_receipt_rollout.parquet", index=False)

    def user_purchase_14d(frame):
        eligible = frame.loc[frame.calendar_day.lt(14)]
        return eligible.groupby(["home_persona", "user_id"]).purchase.sum().gt(0)

    p0, p1 = user_purchase_14d(baseline), user_purchase_14d(treated)
    target, rows = 0., []
    home_prior = gap_model["lifecycle"]["home_persona_prior"]
    for persona in map(int, model["personas"]):
        observed = (
            model["personas"][str(persona)]["purchase_14d_receipt"]
            - model["personas"][str(persona)]["purchase_14d_control"]
        )
        simulated = float(p1.loc[persona].mean() - p0.loc[persona].mean())
        weight = home_prior[str(persona)]
        target += weight * observed
        rows.append({
            "persona": persona, "population_weight": weight,
            "tianchi_observed_difference": observed,
            "simulated_difference": simulated,
            "baseline_purchase_14d": float(p0.loc[persona].mean()),
            "receipt_purchase_14d": float(p1.loc[persona].mean()),
        })
    simulated_population = sum(row["population_weight"] * row["simulated_difference"] for row in rows)
    maximum_persona_error = max(
        abs(row["simulated_difference"] - row["tianchi_observed_difference"])
        for row in rows
    )
    terminal = lambda frame: float(frame.gap_hours.isna().mean())
    canonical = sampled_profiles.groupby(["home_persona", "canonical_type"]).user_id.nunique()
    canonical = (canonical / canonical.groupby(level=0).sum()).rename("share").reset_index()
    canonical.to_csv(output / "sampled_canonical_type_distribution.csv", index=False)
    invariant_columns = [
        "user_id", "home_persona", "session_index", "lifecycle_state",
        "view", "cart", "remove", "purchase", "events", "gap_hours",
        "gap_token", "next_lifecycle_state",
    ]
    baseline_prefix = baseline.loc[baseline.user_id.lt(zero_users), invariant_columns]
    zero_invariant = baseline_prefix.reset_index(drop=True).equals(
        zero[invariant_columns].reset_index(drop=True)
    )
    result = {
        "users_per_arm": users, "coupon_receipt_days": [0.0],
        "strength_by_persona": model["strength_by_persona"],
        "persona_results": rows,
        "population_weighted_purchase_14d_difference": {
            "tianchi_observed_target": target,
            "simulated": simulated_population,
            "absolute_error": abs(simulated_population - target),
        },
        "purchase_session_rate": {
            "baseline": float(baseline.purchase.gt(0).mean()),
            "receipt": float(treated.purchase.gt(0).mean()),
        },
        "no_return_30d_rate": {
            "baseline": terminal(baseline), "receipt": terminal(treated),
        },
        "checks": {
            "aggregate_purchase_difference_error_le_0_03": abs(simulated_population - target) <= .03,
            "maximum_persona_difference_error_le_0_01": maximum_persona_error <= .01,
            "receipt_increases_purchase_sessions": bool(
                treated.purchase.gt(0).mean() > baseline.purchase.gt(0).mean()
            ),
            "receipt_reduces_no_return": terminal(treated) < terminal(baseline),
            "zero_uplift_exactly_matches_baseline": zero_invariant,
        },
        "interpretation": model["interpretation"],
    }
    result["ready"] = all(result["checks"].values())
    (output / "rollout_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "artifacts/tianchi-uplift-model",
    )
    parser.add_argument("--rollout-users", type=int, default=10_000)
    args = parser.parse_args()
    model = run(args.output)
    print(json.dumps(validate_rollout(model, args.output, args.rollout_users),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
