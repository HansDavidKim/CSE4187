"""Cross-domain sampling adapters for the shared session simulator.

The Cosmetics reward and gap models remain the behavioural backbone.  These
adapters describe which backbone persona is sampled in each domain and the
small set of observable action/gap shifts needed to reproduce that domain.
They are descriptive transfer parameters, not causal effects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from analysis.persona.sequence import source_data
from utils.paths import PROJECT_ROOT


SHARED_FEATURES = (
    "purchase_share", "log_total_events", "log_active_days",
    "log_events_per_day", "purchase_day_share", "log_median_gap",
    "first_day_purchase", "last_day_purchase", "explore_to_purchase",
    "purchase_to_explore", "explore_persistence", "purchase_persistence",
    "has_transition",
)
MODE_TO_PERSONA = {
    "P4_purchase_path": 0,
    "P2_deep_consideration": 2,
    "P3_cart_revision": 3,
}


def _fit_gmm(values: np.ndarray, seed: int) -> tuple[GaussianMixture, list[dict]]:
    grid = []
    upper = min(6, max(1, len(values) // 100))
    for components in range(1, upper + 1):
        mixture = GaussianMixture(
            components, covariance_type="diag", reg_covar=1e-4,
            n_init=5, max_iter=500, random_state=seed,
        ).fit(values)
        labels = mixture.predict(values)
        smallest = int(np.bincount(labels, minlength=components).min())
        grid.append({
            "components": components,
            "bic": float(mixture.bic(values)),
            "smallest_component": smallest,
            "model": mixture,
        })
    eligible = [row for row in grid if row["smallest_component"] >= 50]
    selected = min(eligible or grid, key=lambda row: row["bic"])
    return selected.pop("model"), [
        {key: value for key, value in row.items() if key != "model"} for row in grid
    ]


def _action_offsets(decisions: pd.DataFrame) -> dict[str, list[float]]:
    counts = decisions.groupby("dataset")[["view", "cart", "remove", "purchase"]].sum()
    shares = counts.div(counts.sum(axis=1), axis=0).clip(lower=1e-6)
    reference = np.log(shares.loc["cosmetics"].to_numpy())
    result = {}
    for domain in ("cosmetics", "electronics"):
        offset = np.log(shares.loc[domain].to_numpy()) - reference
        result[domain] = np.r_[np.clip(offset, -8, 8), 0.].tolist()

    # Tianchi observes click and purchase but not cart/removal.  Missing actions
    # are disabled at rollout rather than interpreted as observed zero demand.
    result["tianchi"] = [0., -8., -8., 0., 0.]
    return result


def _gap_statistics(decisions: pd.DataFrame, tianchi_path: Path) -> dict[str, dict]:
    result = {}
    for domain in ("cosmetics", "electronics"):
        frame = decisions.loc[decisions.dataset.eq(domain)]
        gaps = frame.next_gap_hours.dropna()
        result[domain] = {
            "median_return_gap_hours": float(gaps.loc[gaps.le(720)].median()),
            "no_return_30d_rate": float(frame.no_return_30d.mean()),
            "observed_decisions": int(len(frame)),
        }

    events, quality, _ = source_data("tianchi", tianchi_path, sample_modulus=50)
    daily = events[["user", "at"]].drop_duplicates().sort_values(["user", "at"])
    daily["next"] = daily.groupby("user", sort=False).at.shift(-1)
    daily["gap"] = (daily["next"] - daily["at"]).dt.total_seconds() / 3600
    observation_end = daily["at"].max()
    observed_30d = (observation_end - daily["at"]).dt.total_seconds().div(3600).ge(720)
    no_return = daily.gap.isna() & observed_30d
    eligible = daily.gap.notna() | observed_30d
    result["tianchi"] = {
        "median_return_gap_hours": float(daily.loc[daily.gap.between(0, 720), "gap"].median()),
        "no_return_30d_rate": float(no_return.loc[eligible].mean()),
        "observed_user_days": int(len(daily)),
        "sample_modulus": int(quality["sample_modulus"]),
    }
    base = result["cosmetics"]["median_return_gap_hours"]
    for value in result.values():
        ratio = value["median_return_gap_hours"] / base
        value["observed_median_gap_ratio"] = float(ratio)
        # Session definitions differ (Tianchi is day-grained).  A square-root
        # transport avoids treating that measurement-grain gap as behavioural.
        value["gap_time_scale"] = float(np.clip(np.sqrt(ratio), .5, 4.))
    return result


def build_domain_adapters(
    output: Path = PROJECT_ROOT / "artifacts/domain-simulator-profiles",
    seed: int = 42,
) -> dict:
    """Fit three domain GMMs and map every component to Cosmetics personas."""
    shared = pd.read_parquet(
        PROJECT_ROOT / "artifacts/cross-dataset-persona-similarity/shared_user_features.parquet"
    )
    modes = pd.read_parquet(
        PROJECT_ROOT / "artifacts/four-behavior-modes/cosmetics_embedding_memberships.parquet"
    )[["user_id", "mode"]]
    known = shared.loc[shared.dataset.eq("cosmetics")].merge(modes, on="user_id")
    known["persona"] = known["mode"].map(MODE_TO_PERSONA)
    known = known.dropna(subset=["persona"])
    centroids = known.groupby("persona")[list(SHARED_FEATURES)].mean()
    personas = centroids.index.astype(int).to_numpy()

    values = shared[list(SHARED_FEATURES)].to_numpy(float)
    distance = np.square(values[:, None] - centroids.to_numpy()[None]).mean(axis=2)
    shared["transferred_persona"] = personas[distance.argmin(axis=1)]

    adapters = {}
    for domain, frame in shared.groupby("dataset", sort=True):
        scaler = StandardScaler().fit(frame[list(SHARED_FEATURES)])
        standardized = scaler.transform(frame[list(SHARED_FEATURES)])
        mixture, grid = _fit_gmm(standardized, seed)
        labels = mixture.predict(standardized)
        composition = []
        for component in range(mixture.n_components):
            group = frame.loc[labels == component, "transferred_persona"]
            counts = group.value_counts().reindex(personas, fill_value=0).astype(float)
            probabilities = ((counts + 1) / (counts.sum() + len(counts))).to_numpy()
            composition.append({
                "component": component,
                "users": int(len(group)),
                "persona_probabilities": {
                    str(persona): float(probability)
                    for persona, probability in zip(personas, probabilities)
                },
            })
        adapters[domain] = {
            "users": int(len(frame)),
            "features": list(SHARED_FEATURES),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "gmm": {
                "weights": mixture.weights_.tolist(),
                "means": mixture.means_.tolist(),
                "variances": mixture.covariances_.tolist(),
                "bic_grid": grid,
            },
            "component_persona_composition": composition,
        }

    decisions = pd.read_parquet(
        PROJECT_ROOT / "artifacts/cross-domain-lifecycle/return_decisions.parquet"
    )
    action_offsets = _action_offsets(decisions)
    gap = _gap_statistics(
        decisions, PROJECT_ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    )
    for domain, adapter in adapters.items():
        adapter["action_logit_offsets"] = action_offsets[domain]
        adapter.update(gap[domain])

    result = {
        "model": "shared-feature domain GMM with nearest Cosmetics-persona transfer",
        "seed": seed,
        "persona_centroids": {
            str(int(persona)): row.to_dict() for persona, row in centroids.iterrows()
        },
        "domains": adapters,
        "interpretation": (
            "GMM components reproduce domain heterogeneity; component-conditioned persona "
            "probabilities transfer the Cosmetics MaxEnt-IRL backbone."
        ),
        "limitations": [
            "The nearest-profile assignment is descriptive and is not a causal identification result.",
            "Tianchi does not observe cart/removal, so those simulator actions are disabled.",
            "Coupon receipt is an intervention and is excluded from the user action alphabet.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    shared[["dataset", "user_id", "cluster", "transferred_persona"]].to_parquet(
        output / "user_persona_transfer.parquet", index=False
    )
    (output / "model.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def sample_domain_user(adapter: dict, rng: np.random.Generator) -> dict:
    """Sample a domain GMM component and its transferred home persona."""
    mixture = adapter["gmm"]
    component = int(rng.choice(len(mixture["weights"]), p=mixture["weights"]))
    row = adapter["component_persona_composition"][component]
    names = list(row["persona_probabilities"])
    probabilities = np.asarray(list(row["persona_probabilities"].values()), dtype=float)
    persona = int(names[int(rng.choice(len(names), p=probabilities))])
    mean = np.asarray(mixture["means"][component])
    variance = np.asarray(mixture["variances"][component])
    latent = mean + rng.normal(size=len(mean)) * np.sqrt(variance)
    return {"component": component, "persona": persona, "domain_latent": latent}

