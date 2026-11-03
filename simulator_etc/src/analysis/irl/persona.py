"""Persona-conditioned maximum-causal-entropy IRL for event sequences.

The model is a contextual MaxEnt policy, not a long-horizon transition model:
the reward for the next action is linear in state features available strictly
before that action.  A persona reward is fitted first, then each representative
user receives a MAP estimate centred on that persona reward.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from analysis.persona.sequence import source_data
from simulate.latent_session_model import (
    GAP_STATE_NAMES,
    churn_sequence_features,
    persona_gap_features,
    simulate_no_uplift_users,
    update_gap_internal_state,
)

ACTIONS = ("view", "cart", "remove", "purchase", "end")
FEATURES = (
    "intercept",
    "log_session_position",
    "log_session_elapsed_minutes",
    "log_inter_event_gap_minutes",
    "prior_view_count",
    "prior_cart_count",
    "prior_remove_count",
    "prior_purchase_count",
    "previous_view",
    "previous_cart",
    "previous_remove",
    "previous_purchase",
)
GAP_EDGES = np.array([0.0, 1.0, 6.0, 24.0, 72.0, 168.0, 336.0, 720.0])
GAP_EVAL_HORIZONS = (1.0, 6.0, 24.0, 72.0, 168.0)
GAP_TOKENS = (
    "GAP_0_1H", "GAP_1_6H", "GAP_6_24H", "GAP_1_3D",
    "GAP_3_7D", "GAP_7_14D", "GAP_14_30D", "NO_RETURN_30D",
)
GAP_FEATURES = (
    "log_view", "log_cart", "log_remove", "log_purchase",
    "log_duration_hours", "log_events", "log_previous_gap_hours",
    "log_user_mean_gap_hours", "has_previous_gap",
)
MATURE_PERSONAS = (0, 2, 3)


def select_representatives(
    embeddings: pd.DataFrame,
    centroids: np.ndarray,
    per_persona: int = 20,
    min_sessions: int = 2,
    personas: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Return the nearest eligible users to each fitted persona centroid."""
    z_columns = sorted((c for c in embeddings if c.startswith("z_")), key=lambda c: int(c[2:]))
    rows = []
    eligible = embeddings.loc[embeddings.sessions.ge(min_sessions)].copy()
    for persona in personas if personas is not None else range(len(centroids)):
        group = eligible.loc[eligible.cluster.eq(persona)].copy()
        if len(group) < per_persona:
            raise ValueError(f"persona {persona} has only {len(group)} eligible users")
        delta = group[z_columns].to_numpy(dtype=float) - centroids[persona]
        group["centroid_distance"] = np.sqrt(np.square(delta).sum(axis=1))
        rows.append(group.nsmallest(per_persona, "centroid_distance"))
    return pd.concat(rows, ignore_index=True)


def fit_coefficient_gmm(
    coefficients: pd.DataFrame,
    output: Path,
    seed: int = 42,
    max_components: int = 6,
    persona_population_weights: dict[int, float] | None = None,
) -> dict:
    """Fit a diagonal GMM to identifiable user reward coefficients."""
    free = coefficients.loc[coefficients.action.ne(ACTIONS[-1])].copy()
    free["term"] = free.action + ":" + free.feature
    matrix = free.pivot(index="user_id", columns="term", values="coefficient").sort_index(axis=1)
    personas = coefficients.groupby("user_id").persona.first().reindex(matrix.index)
    if matrix.isna().any().any() or len(matrix) < 10:
        raise ValueError("coefficient GMM needs at least ten complete users")
    scaler = StandardScaler().fit(matrix)
    x = scaler.transform(matrix)
    user_hash = pd.util.hash_pandas_object(
        "coefficient-gmm:" + matrix.index.to_series().astype(str), index=False
    ).to_numpy()
    test = user_hash % 5 == 0
    if test.sum() < 2 or (~test).sum() < 8:
        test[np.argsort(user_hash)[:max(2, len(matrix) // 5)]] = True
    train = ~test
    grid_rows = []
    for components in range(1, min(max_components, int(train.sum() // 5)) + 1):
        candidate = GaussianMixture(
            n_components=components, covariance_type="diag", reg_covar=1e-3,
            n_init=20, random_state=seed,
        ).fit(x[train])
        effective = candidate.weights_ * train.sum()
        grid_rows.append({
            "components": components,
            "bic": float(candidate.bic(x[train])),
            "train_log_likelihood": float(candidate.score(x[train])),
            "heldout_log_likelihood": float(candidate.score(x[test])),
            "minimum_effective_train_users": float(effective.min()),
            "eligible": bool(effective.min() >= 5),
        })
    grid = pd.DataFrame(grid_rows)
    eligible_grid = grid.loc[grid.eligible]
    selected = int((eligible_grid if len(eligible_grid) else grid).sort_values("bic").iloc[0].components)
    final = GaussianMixture(
        n_components=selected, covariance_type="diag", reg_covar=1e-3,
        n_init=50, random_state=seed,
    ).fit(x)
    component = final.predict(x)
    membership = pd.DataFrame({
        "user_id": matrix.index.astype(int), "persona": personas.to_numpy(dtype=int),
        "gmm_component": component,
        "posterior_probability": final.predict_proba(x).max(axis=1),
    })
    composition = membership.groupby(
        ["gmm_component", "persona"], as_index=False
    ).size()
    composition["persona_share_within_component"] = composition["size"] / composition.groupby(
        "gmm_component"
    )["size"].transform("sum")
    calibrated_weights = final.weights_.copy()
    if persona_population_weights is not None:
        calibrated_weights = np.zeros(selected)
        for persona, population_weight in persona_population_weights.items():
            persona_components = membership.loc[membership.persona.eq(persona), "gmm_component"]
            conditional = np.bincount(persona_components, minlength=selected) / len(persona_components)
            calibrated_weights += population_weight * conditional
    means = scaler.mean_[None, :] + final.means_ * scaler.scale_[None, :]
    variances = final.covariances_ * np.square(scaler.scale_)[None, :]
    statistics = []
    for k in range(selected):
        for j, term in enumerate(matrix.columns):
            statistics.append({
                "component": k, "weight": float(calibrated_weights[k]), "term": term,
                "mean": float(means[k, j]), "variance": float(variances[k, j]),
                "standard_deviation": float(np.sqrt(variances[k, j])),
            })
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "coefficient_gmm_grid.csv", index=False)
    membership.to_csv(output / "coefficient_gmm_memberships.csv", index=False)
    composition.to_csv(output / "coefficient_gmm_composition.csv", index=False)
    pd.DataFrame(statistics).to_csv(output / "coefficient_gmm_parameters.csv", index=False)
    payload = {
        "model": "diagonal Gaussian mixture over identifiable user MaxEnt-IRL coefficients",
        "selection": "minimum train BIC among components with at least five effective train users",
        "selected_components": selected,
        "weights": calibrated_weights.tolist(),
        "sample_fit_weights": final.weights_.tolist(),
        "weight_calibration": (
            "post-stratified to mature-persona population shares"
            if persona_population_weights is not None else "raw representative-sample fit"
        ),
        "standardized_means": final.means_.tolist(),
        "standardized_variances": final.covariances_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "terms": matrix.columns.tolist(),
        "dependent_action": ACTIONS[-1],
        "dependent_action_rule": "negative sum of the four stored action coefficients",
        "grid": grid.to_dict(orient="records"),
        "component_persona_composition": composition.to_dict(orient="records"),
    }
    (output / "coefficient_gmm.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def make_decisions(events: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Create past-only action decisions and conservatively inferred END actions."""
    keep = events.merge(assignments[["user_id", "cluster"]], left_on="user", right_on="user_id")
    keep = keep.sort_values(["user", "at"], kind="stable")
    observation_end = keep["at"].max()
    records: list[tuple] = []
    for user, user_frame in keep.groupby("user", sort=False):
        sessions = [frame.sort_values("at", kind="stable")
                    for _, frame in user_frame.groupby("session", sort=False)]
        for session_index, frame in enumerate(sessions):
            start = frame["at"].iloc[0]
            counts = np.zeros(4, dtype=float)
            previous = -1
            previous_at = start
            persona = int(frame.cluster.iloc[0])
            for position, row in enumerate(frame.itertuples(index=False)):
                elapsed = max(0.0, (row.at - start).total_seconds() / 60)
                gap = 0.0 if position == 0 else max(0.0, (row.at - previous_at).total_seconds() / 60)
                features = _state_features(position, elapsed, gap, counts, previous)
                records.append((int(user), persona, row.session, row.at, int(row.action), *features))
                counts[int(row.action)] += 1
                previous = int(row.action)
                previous_at = row.at
            boundary = previous_at + pd.Timedelta(minutes=30)
            next_start = (sessions[session_index + 1]["at"].iloc[0]
                          if session_index + 1 < len(sessions) else None)
            # END is not logged. Add it only if the inactivity boundary is
            # observed and no later session has already started before it.
            if boundary <= observation_end and (next_start is None or next_start > boundary):
                elapsed = max(0.0, (previous_at - start).total_seconds() / 60)
                features = _state_features(len(frame), elapsed, 0.0, counts, previous)
                records.append((int(user), persona, frame.session.iloc[0], boundary, 4, *features))
    columns = ["user_id", "persona", "session", "at", "action", *FEATURES]
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["user_id", "at"], kind="stable"
    ).reset_index(drop=True)


def _state_features(position: int, elapsed: float, gap: float,
                    counts: np.ndarray, previous: int) -> tuple[float, ...]:
    previous_one_hot = np.zeros(4, dtype=float)
    if 0 <= previous < 4:
        previous_one_hot[previous] = 1.0
    return (
        1.0,
        np.log1p(position),
        np.log1p(elapsed),
        np.log1p(gap),
        *np.log1p(counts).tolist(),
        *previous_one_hot.tolist(),
    )


def maxent_objective(
    flat: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    centre: np.ndarray | None = None,
    penalty: float = 0.0,
    average: bool = True,
) -> tuple[float, np.ndarray]:
    """Multinomial MaxEnt loss under a symmetric sum-to-zero constraint."""
    free = flat.reshape(len(ACTIONS) - 1, x.shape[1])
    theta = np.vstack((free, -free.sum(axis=0)))
    logits = x @ theta.T
    normalizer = logsumexp(logits, axis=1)
    loss = normalizer.sum() - logits[np.arange(len(y)), y].sum()
    probabilities = np.exp(logits - normalizer[:, None])
    residual = probabilities.copy()
    residual[np.arange(len(y)), y] -= 1.0
    gradient = residual.T @ x
    if average:
        loss /= len(x)
        gradient /= len(x)
    if centre is not None and penalty:
        difference = theta - centre
        loss += 0.5 * penalty * np.square(difference).sum()
        gradient += penalty * difference
    # theta[-1] is -sum(theta[:-1]), hence the chain-rule subtraction.
    return float(loss), (gradient[:-1] - gradient[-1]).ravel()


def fit_reward(
    x: np.ndarray,
    y: np.ndarray,
    centre: np.ndarray | None = None,
    penalty: float = 1e-3,
    average: bool = True,
) -> np.ndarray:
    initial = np.zeros((len(ACTIONS) - 1, x.shape[1])) if centre is None else centre[:-1].copy()
    penalty_centre = np.zeros((len(ACTIONS), x.shape[1])) if centre is None else centre
    result = minimize(
        maxent_objective,
        initial.ravel(),
        args=(x, y, penalty_centre, penalty, average),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"MaxEnt optimisation failed: {result.message}")
    free = result.x.reshape(len(ACTIONS) - 1, x.shape[1])
    return np.vstack((free, -free.sum(axis=0)))


def action_probabilities(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    logits = x @ theta.T
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


def _scale(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.loc[:, FEATURES[1:]].mean().to_numpy(dtype=float)
    scale = train.loc[:, FEATURES[1:]].std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    x = frame.loc[:, FEATURES].to_numpy(dtype=float)
    x[:, 1:] = (x[:, 1:] - mean) / scale
    return x, mean, scale


def quadratic_design(x: np.ndarray, training_index: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Add standardized squares and pairwise interactions of non-intercept features."""
    base = np.clip(x[:, 1:], -5.0, 5.0)
    columns, names = [], []
    for i, name in enumerate(FEATURES[1:]):
        columns.append(np.square(base[:, i]))
        names.append(f"square:{name}")
    for i, j in combinations(range(base.shape[1]), 2):
        columns.append(base[:, i] * base[:, j])
        names.append(f"interaction:{FEATURES[i + 1]}*{FEATURES[j + 1]}")
    derived = np.column_stack(columns)
    mean = derived[training_index].mean(axis=0)
    scale = derived[training_index].std(axis=0)
    scale[scale == 0] = 1.0
    return np.column_stack((x, (derived - mean) / scale)), (*FEATURES, *names)


def compare_persona_rewards(
    raw_path: Path,
    cluster_report: Path,
    embeddings_path: Path,
    output: Path,
    per_persona: int = 20,
    sample_modulus: int = 100,
    seed: int = 42,
) -> dict:
    """Compare linear and quadratic persona rewards on untouched late decisions."""
    report = json.loads(cluster_report.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    centroids = np.asarray([p["centroid"] for p in sorted(report["profiles"], key=lambda p: p["cluster"])])
    representatives = select_representatives(embeddings, centroids, per_persona)
    events, quality, _ = source_data("cosmetics", raw_path, sample_modulus)
    decisions = make_decisions(events, embeddings[["user_id", "cluster"]])
    representative_ids = set(representatives.user_id.astype(int))
    prior_index = decisions.index[~decisions.user_id.isin(representative_ids)].to_numpy()
    x_linear, _, _ = _scale(decisions.loc[prior_index], decisions)
    x_quadratic, quadratic_names = quadratic_design(x_linear, prior_index)
    designs = {"linear": (x_linear, FEATURES), "quadratic": (x_quadratic, quadratic_names)}
    rng = np.random.default_rng(seed)
    penalties = (1e-3, 1e-2, 1e-1)
    results, prior_payload = [], {}

    for specification, (design, names) in designs.items():
        prior_payload[specification] = {}
        for persona in range(len(centroids)):
            train_index = prior_index[decisions.loc[prior_index, "persona"].to_numpy() == persona]
            if len(train_index) > 100_000:
                train_index = rng.choice(train_index, 100_000, replace=False)
            validation_index, test_index = [], []
            users = representatives.loc[representatives.cluster.eq(persona), "user_id"]
            for user in users:
                frame = decisions.loc[decisions.user_id.eq(int(user))]
                validation_start, test_start = _temporal_boundaries(len(frame))
                validation_index.extend(frame.index[validation_start:test_start])
                test_index.extend(frame.index[test_start:])
            validation_index = np.asarray(validation_index, dtype=int)
            test_index = np.asarray(test_index, dtype=int)
            candidates = []
            for penalty in penalties:
                theta = fit_reward(
                    design[train_index], decisions.loc[train_index, "action"].to_numpy(dtype=int),
                    penalty=penalty,
                )
                probability = action_probabilities(design[validation_index], theta)
                actual = decisions.loc[validation_index, "action"].to_numpy(dtype=int)
                nll = float(-np.log(probability[np.arange(len(actual)), actual] + 1e-15).mean())
                candidates.append((nll, penalty, theta))
            validation_nll, penalty, theta = min(candidates, key=lambda item: item[0])
            probability = action_probabilities(design[test_index], theta)
            actual = decisions.loc[test_index, "action"].to_numpy(dtype=int)
            observed_mix = np.bincount(actual, minlength=len(ACTIONS)) / len(actual)
            predicted_mix = probability.mean(axis=0)
            results.append({
                "specification": specification, "persona": persona,
                "features": len(names), "penalty": penalty,
                "validation_rows": len(validation_index), "test_rows": len(test_index),
                "validation_nll": validation_nll,
                "test_nll": float(-np.log(probability[np.arange(len(actual)), actual] + 1e-15).mean()),
                "test_accuracy": float((probability.argmax(axis=1) == actual).mean()),
                "action_mix_rmse": float(np.sqrt(np.square(predicted_mix - observed_mix).mean())),
            })
            prior_payload[specification][str(persona)] = {
                action: dict(zip(names, theta[i].tolist())) for i, action in enumerate(ACTIONS)
            }

    metrics = pd.DataFrame(results)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "persona_reward_comparison.csv", index=False)
    (output / "persona_reward_priors.json").write_text(
        json.dumps(prior_payload, indent=2), encoding="utf-8"
    )
    summary = {
        "data": quality,
        "representatives_per_persona": per_persona,
        "temperature": 1.0,
        "identification": "coefficients sum to zero across actions for every feature",
        "feature_counts": {name: len(columns) for name, (_, columns) in designs.items()},
        "metrics": results,
        "mean_by_specification": metrics.groupby("specification")[[
            "test_nll", "test_accuracy", "action_mix_rmse"
        ]].mean().to_dict(orient="index"),
        "limitations": [
            "Quadratic terms are predictive reward features, not causal effects.",
            "Higher-order terms use standardized base features clipped to [-5, 5].",
            "Regularisation is selected on each persona's middle decisions; final decisions are untouched test data.",
            "A selected penalty of 0.001 is the smallest reliably converged candidate, not an interior optimum.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def session_summaries(events: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create session-end states and right-censored next-session gaps."""
    data = events.merge(assignments[["user_id", "cluster"]], left_on="user", right_on="user_id")
    for action in range(4):
        data[f"action_{action}"] = data.action.eq(action).astype("int8")
    sessions = data.groupby(["user", "session"], sort=False).agg(
        persona=("cluster", "first"), start=("at", "min"), end=("at", "max"),
        unique_products=("product", "nunique"),
        view=("action_0", "sum"), cart=("action_1", "sum"),
        remove=("action_2", "sum"), purchase=("action_3", "sum"),
        events=("action", "size"),
    ).reset_index().sort_values(["user", "start", "end"], kind="stable").reset_index(drop=True)
    grouped = sessions.groupby("user", sort=False)
    previous_end, next_start = grouped.end.shift(), grouped.start.shift(-1)
    sessions["duration_hours"] = (sessions.end - sessions.start).dt.total_seconds() / 3600
    sessions["previous_gap_hours"] = (sessions.start - previous_end).dt.total_seconds() / 3600
    sessions["next_gap_hours"] = (next_start - sessions.end).dt.total_seconds() / 3600
    sessions.loc[sessions.previous_gap_hours.lt(0), "previous_gap_hours"] = np.nan
    sessions.loc[sessions.next_gap_hours.lt(0), "next_gap_hours"] = np.nan
    return sessions, data["at"].max()


def brief_browser_subtypes(
    sessions: pd.DataFrame, observation_end: pd.Timestamp, persona: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Operationally split the brief-browser persona into birth/lifecycle roles."""
    frame = sessions.loc[sessions.persona.eq(persona)].sort_values(["user", "start"])
    grouped = frame.groupby("user", sort=False)
    users = grouped.agg(
        first_seen=("start", "min"), last_seen=("end", "max"),
        sessions=("session", "size"), purchase_events=("purchase", "sum"),
        friction_events=("cart", "sum"), total_events=("events", "sum"),
        first_unique_products=("unique_products", "first"),
    ).reset_index()
    users["friction_events"] += grouped["remove"].sum().to_numpy()
    users["observed_age_days"] = (
        observation_end - users.first_seen
    ).dt.total_seconds() / 86400
    users["tail_days"] = (observation_end - users.last_seen).dt.total_seconds() / 86400
    users["median_gap_days"] = users.user.map(grouped["previous_gap_hours"].median()) / 24
    is_new = users.observed_age_days.lt(30) & users.sessions.le(2)
    is_mission = (
        ~is_new & users.sessions.eq(1) & users.first_unique_products.eq(1)
        & users.tail_days.ge(30)
    )
    is_churning = ~is_new & ~is_mission & users.tail_days.ge(30)
    users["subtype"] = np.select(
        [is_new, is_mission, is_churning],
        ["NEW", "AMBIGUOUS_SINGLE_VIEW_EXIT", "DISSATISFIED_CHURNING"],
        default="REGULAR",
    )
    profiles = users.groupby("subtype", as_index=False).agg(
        users=("user", "size"), median_sessions=("sessions", "median"),
        purchase_user_rate=("purchase_events", lambda value: float(value.gt(0).mean())),
        median_events=("total_events", "median"),
        first_one_product_rate=("first_unique_products", lambda value: float(value.eq(1).mean())),
        median_gap_days=("median_gap_days", "median"), median_tail_days=("tail_days", "median"),
        friction_user_rate=("friction_events", lambda value: float(value.gt(0).mean())),
    )
    return users, profiles


def subtype_embedding_gmm(
    embeddings: pd.DataFrame, users: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, float]:
    """Describe subtype density in an 8D PCA space; not used as an identified label model."""
    z_columns = sorted((c for c in embeddings if c.startswith("z_")), key=lambda c: int(c[2:]))
    scaled = StandardScaler().fit_transform(embeddings[z_columns])
    pca = PCA(n_components=8, random_state=seed).fit(scaled)
    reduced = pca.transform(scaled)
    frame = pd.DataFrame(reduced, columns=[f"pc_{i}" for i in range(8)])
    frame["user"] = embeddings.user_id.to_numpy()
    frame = frame.merge(users[["user", "subtype"]], on="user")
    rows = []
    for subtype, group in frame.groupby("subtype"):
        values = group[[f"pc_{i}" for i in range(8)]].to_numpy()
        candidates = []
        for components in range(1, min(4, len(values)) + 1):
            mixture = GaussianMixture(
                components, covariance_type="diag", reg_covar=1e-3,
                n_init=10, max_iter=300, random_state=seed,
            ).fit(values)
            candidates.append((mixture.bic(values), mixture))
        bic, mixture = min(candidates, key=lambda item: item[0])
        for component in range(mixture.n_components):
            variance = mixture.covariances_[component]
            rows.append({
                "subtype": subtype, "users": len(values), "selected_components": mixture.n_components,
                "component": component, "weight": float(mixture.weights_[component]),
                "mean_variance": float(variance.mean()),
                "weighted_log_peak": float(
                    np.log(mixture.weights_[component]) - .5 * np.log(2 * np.pi * variance).sum()
                ),
                "bic": float(bic),
            })
    return pd.DataFrame(rows), float(pca.explained_variance_ratio_.sum())


def estimate_mission_prior(
    events: pd.DataFrame, embeddings: pd.DataFrame, browser_users: pd.DataFrame, seed: int
) -> dict:
    """Transport a successful pre-purchase search pattern to sufficiently exposed failures."""
    observation_end = events["at"].max()
    rows = []
    for user, frame in events.sort_values(["user", "at"]).groupby("user", sort=False):
        purchase = np.flatnonzero(frame.action.eq(3).to_numpy())
        success = len(purchase) > 0
        prefix = frame.iloc[:purchase[0]] if success else frame
        if prefix.empty:
            continue
        product_counts = prefix["product"].value_counts(dropna=True)
        starts = prefix.groupby("session")["at"].min().sort_values()
        gaps = starts.diff().dt.total_seconds().dropna() / 3600
        span = (prefix["at"].iloc[-1] - prefix["at"].iloc[0]).total_seconds() / 3600
        counts = prefix.action.value_counts()
        rows.append({
            "user": int(user), "success": success, "events": len(prefix),
            "sessions": prefix.session.nunique(), "span_hours": span,
            "unique_products": prefix["product"].nunique(),
            "dominant_product_share": float(product_counts.iloc[0] / len(prefix)) if len(product_counts) else 0,
            "view_share": counts.get(0, 0) / len(prefix), "cart_share": counts.get(1, 0) / len(prefix),
            "remove_share": counts.get(2, 0) / len(prefix),
            "median_gap_hours": float(gaps.median()) if len(gaps) else 0,
            "tail_days": (observation_end - frame["at"].iloc[-1]).total_seconds() / 86400,
        })
    frame = pd.DataFrame(rows).merge(
        embeddings[["user_id", "cluster"]], left_on="user", right_on="user_id"
    )
    anchor = frame.loc[
        frame.success & frame.events.ge(3) & frame.sessions.ge(2) & frame.span_hours.ge(24)
    ].copy()
    failed = frame.loc[
        ~frame.success & frame.events.ge(3) & frame.sessions.ge(2) & frame.tail_days.ge(30)
    ].copy()
    features = [
        "events", "sessions", "span_hours", "unique_products", "dominant_product_share",
        "view_share", "cart_share", "remove_share", "median_gap_hours",
    ]
    for column in ("events", "sessions", "span_hours", "unique_products", "median_gap_hours"):
        anchor[column] = np.log1p(anchor[column])
        failed[column] = np.log1p(failed[column])
    scaler = StandardScaler().fit(anchor[features])
    anchor_x, failed_x = scaler.transform(anchor[features]), scaler.transform(failed[features])
    candidates = []
    for components in range(1, 5):
        mixture = GaussianMixture(
            components, covariance_type="diag", reg_covar=1e-3,
            n_init=10, max_iter=300, random_state=seed,
        ).fit(anchor_x)
        candidates.append((mixture.bic(anchor_x), mixture))
    bic, mixture = min(candidates, key=lambda item: item[0])
    threshold = float(np.quantile(mixture.score_samples(anchor_x), .05))
    failed["mission_match"] = mixture.score_samples(failed_x) >= threshold
    p1_failed = failed.loc[failed.cluster.eq(1)]
    match_rate = float(p1_failed.mission_match.mean())
    ambiguous = int(browser_users.subtype.eq("AMBIGUOUS_SINGLE_VIEW_EXIT").sum())
    matched = int(p1_failed.mission_match.sum())
    p1_users = int(embeddings.cluster.eq(1).sum())
    prior = (matched + ambiguous * match_rate) / p1_users
    return {
        "successful_anchors": len(anchor), "selected_components": mixture.n_components,
        "bic": float(bic), "anchor_support_quantile": .05, "log_density_threshold": threshold,
        "eligible_p1_failures": len(p1_failed), "matched_p1_failures": matched,
        "matched_failure_rate": match_rate, "ambiguous_single_view_exits": ambiguous,
        "transported_ambiguous_mission_count": float(ambiguous * match_rate),
        "p1_users": p1_users, "mission_birth_prior": float(prior),
        "assumption": "the long-search matched-failure rate transports to ambiguous single-view exits",
    }


def _gap_choice_labels(frame: pd.DataFrame, observation_end: pd.Timestamp) -> np.ndarray:
    """Observed gap token, with 30-day no-return included only when fully observed."""
    gap = frame.next_gap_hours.to_numpy(dtype=float)
    available = (observation_end - frame.end).dt.total_seconds().to_numpy() / 3600
    labels = np.full(len(frame), -1, dtype=int)
    returned = np.isfinite(gap) & (gap <= GAP_EDGES[-1])
    labels[returned] = np.minimum(
        np.searchsorted(GAP_EDGES[1:], gap[returned], side="right"), len(GAP_EDGES) - 2
    )
    no_return = ~returned & (available >= GAP_EDGES[-1])
    labels[no_return] = len(GAP_TOKENS) - 1
    return labels


def _gap_choice_design(
    frame: pd.DataFrame, memory: float, initial_gap_hours: float
) -> tuple[np.ndarray, np.ndarray]:
    signals = np.zeros((len(frame), len(GAP_STATE_NAMES)))
    states = np.zeros_like(signals)
    initial_inactivity = np.log1p(initial_gap_hours) / np.log1p(GAP_EDGES[-1])
    for _, indices in frame.groupby("user", sort=False).groups.items():
        state = np.array([0.0, 0.0, 0.0, initial_inactivity, 0.0, 0.0])
        for index in indices:
            signal, state = update_gap_internal_state(
                state, frame.loc[index].to_dict(), memory, initial_gap_hours
            )
            signals[index], states[index] = signal, state
    return signals, states


def _churn_sequence_design(frame: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
    design = np.zeros((len(frame), 14))
    for _, indices in frame.groupby("user", sort=False).groups.items():
        history = []
        for index in indices:
            design[index] = churn_sequence_features(signals[index], history)
            history.append(signals[index, :4])
    return design


def _choice_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    observed = np.bincount(actual, minlength=len(GAP_TOKENS)) / len(actual)
    predicted = probability.mean(axis=0)
    churn = actual == len(GAP_TOKENS) - 1
    churn_probability = probability[:, -1]
    returning = ~churn
    conditional_timing = probability[returning, :-1]
    conditional_timing /= conditional_timing.sum(axis=1, keepdims=True)
    return {
        "nll": float(-np.log(probability[np.arange(len(actual)), actual] + 1e-15).mean()),
        "accuracy": float((probability.argmax(axis=1) == actual).mean()),
        "token_mix_rmse": float(np.sqrt(np.square(observed - predicted).mean())),
        "timing_nll": float(-np.log(
            conditional_timing[np.arange(returning.sum()), actual[returning]] + 1e-15
        ).mean()),
        "churn_brier": float(brier_score_loss(churn, churn_probability)),
        "churn_nll": float(-np.mean(
            churn * np.log(churn_probability + 1e-15)
            + ~churn * np.log(1 - churn_probability + 1e-15)
        )),
        "churn_roc_auc": float(roc_auc_score(churn, churn_probability))
        if np.unique(churn).size == 2 else np.nan,
        "churn_average_precision": float(average_precision_score(churn, churn_probability))
        if churn.any() else np.nan,
        "observed_churn_rate": float(churn.mean()),
        "predicted_churn_rate": float(churn_probability.mean()),
    }


def _fit_nested_gap_choice(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int
) -> tuple[LogisticRegression, LogisticRegression, np.ndarray]:
    churn = train_y == len(GAP_TOKENS) - 1
    churn_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(train_x, churn)
    timing_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        train_x[~churn], train_y[~churn]
    )
    probability = _nested_gap_probabilities(churn_model, test_x, timing_model, test_x)
    return churn_model, timing_model, probability


def _nested_gap_probabilities(
    churn_model: LogisticRegression,
    churn_x: np.ndarray,
    timing_model: LogisticRegression,
    timing_x: np.ndarray,
) -> np.ndarray:
    churn_probability = churn_model.predict_proba(churn_x)[:, 1]
    timing_probability = np.zeros((len(timing_x), len(GAP_TOKENS) - 1))
    timing_probability[:, timing_model.classes_.astype(int)] = timing_model.predict_proba(timing_x)
    return np.column_stack((
        timing_probability * (1 - churn_probability[:, None]), churn_probability
    ))


def fit_gap_token_irl(
    raw_path: Path,
    cluster_report: Path,
    embeddings_path: Path,
    output: Path,
    per_persona: int = 20,
    sample_modulus: int = 100,
    seed: int = 42,
) -> dict:
    """Fit a contextual MaxEnt gap-token choice model with recursive latent state."""
    report = json.loads(cluster_report.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    centroids = np.asarray([p["centroid"] for p in sorted(report["profiles"], key=lambda p: p["cluster"])])
    representatives = select_representatives(embeddings, centroids, per_persona)
    events, quality, _ = source_data("cosmetics", raw_path, sample_modulus)
    sessions, observation_end = session_summaries(events, embeddings[["user_id", "cluster"]])
    sessions["gap_action"] = _gap_choice_labels(sessions, observation_end)
    representative_ids = set(representatives.user_id.astype(int))
    initial_gap_hours = float(sessions.loc[
        ~sessions.user.isin(representative_ids)
        & sessions.previous_gap_hours.between(0, GAP_EDGES[-1]), "previous_gap_hours"
    ].mean())
    memories = (0.0, 0.25, 0.5, 0.75, 0.9)
    designs = {memory: _gap_choice_design(sessions, memory, initial_gap_hours)
               for memory in memories}
    model = {
        "model": "persona contextual MaxEnt gap-token choice with recursive internal state",
        "gap_tokens": list(GAP_TOKENS), "edges_hours": GAP_EDGES.tolist(),
        "state_names": list(GAP_STATE_NAMES), "initial_gap_hours": initial_gap_hours,
        "personas": {},
    }
    interaction_features = (
        tuple(f"current_{name}" for name in GAP_STATE_NAMES)
        + tuple(f"state_{name}" for name in GAP_STATE_NAMES)
        + tuple(f"interaction_{name}" for name in GAP_STATE_NAMES)
    )
    metric_rows, coefficient_rows = [], []
    pooled_selected_actual, pooled_selected_probability = [], []
    for persona in range(len(centroids)):
        train = sessions.persona.eq(persona) & ~sessions.user.isin(representative_ids) & sessions.gap_action.ge(0)
        test = sessions.persona.eq(persona) & sessions.user.isin(representative_ids) & sessions.gap_action.ge(0)
        train_users = sessions.loc[train, "user"].drop_duplicates()
        hashes = pd.util.hash_pandas_object("gap-choice:" + train_users.astype(str), index=False).to_numpy()
        validation_users = set(train_users[hashes % 5 == 0])
        fit = train & ~sessions.user.isin(validation_users)
        validation = train & sessions.user.isin(validation_users)
        memory_scores = []
        for memory in memories:
            signal, state = designs[memory]
            interaction = np.column_stack((signal, state, signal * state))
            mean, scale = interaction[fit].mean(axis=0), interaction[fit].std(axis=0)
            scale[scale == 0] = 1.0
            _, _, probability = _fit_nested_gap_choice(
                (interaction[fit] - mean) / scale,
                sessions.loc[fit, "gap_action"].to_numpy(dtype=int),
                (interaction[validation] - mean) / scale,
                seed,
            )
            actual = sessions.loc[validation, "gap_action"].to_numpy(dtype=int)
            memory_scores.append((_choice_metrics(actual, probability)["timing_nll"], memory))
        memory = min(memory_scores)[1]
        signal, state = designs[memory]
        churn_sequence = _churn_sequence_design(sessions, signal)
        actual_train = sessions.loc[train, "gap_action"].to_numpy(dtype=int)
        actual_test = sessions.loc[test, "gap_action"].to_numpy(dtype=int)
        specifications = {
            "persona_only": np.zeros((len(sessions), 1)),
            "core_session_signal": signal[:, :4],
            "churn_sequence": churn_sequence,
            "session_signal": signal,
            "internal_state": state,
            "state_interaction": np.column_stack((signal, state, signal * state)),
        }
        churn_candidates = []
        fit_churn = sessions.loc[fit, "gap_action"].eq(len(GAP_TOKENS) - 1).to_numpy()
        validation_churn = sessions.loc[validation, "gap_action"].eq(
            len(GAP_TOKENS) - 1
        ).to_numpy()
        name = "core_session_signal"
        design = specifications[name]
        candidate_mean, candidate_scale = design[fit].mean(axis=0), design[fit].std(axis=0)
        candidate_scale[candidate_scale == 0] = 1.0
        for c in (0.03, 0.1, 0.3, 1.0):
            candidate = LogisticRegression(
                C=c, max_iter=2000, random_state=seed
            ).fit((design[fit] - candidate_mean) / candidate_scale, fit_churn)
            probability = candidate.predict_proba(
                (design[validation] - candidate_mean) / candidate_scale
            )[:, 1]
            churn_candidates.append((
                brier_score_loss(validation_churn, probability), name, c
            ))
        _, selected_churn_features, selected_churn_c = min(churn_candidates)
        fitted = {}
        for name, design in specifications.items():
            mean, scale = design[train].mean(axis=0), design[train].std(axis=0)
            scale[scale == 0] = 1.0
            churn_choice, timing_choice, probability = _fit_nested_gap_choice(
                (design[train] - mean) / scale,
                actual_train,
                (design[test] - mean) / scale,
                seed,
            )
            metrics = _choice_metrics(actual_test, probability)
            metric_rows.append({
                "persona": persona, "specification": name, "memory": memory,
                "train_choices": int(train.sum()), "test_choices": int(test.sum()), **metrics,
            })
            fitted[name] = (churn_choice, timing_choice, mean, scale)
        churn_design = specifications[selected_churn_features]
        churn_mean, churn_scale = churn_design[train].mean(axis=0), churn_design[train].std(axis=0)
        churn_scale[churn_scale == 0] = 1.0
        churn_choice = LogisticRegression(
            C=selected_churn_c, max_iter=2000, random_state=seed
        ).fit(
            (churn_design[train] - churn_mean) / churn_scale,
            actual_train == len(GAP_TOKENS) - 1,
        )
        _, timing_choice, mean, scale = fitted["state_interaction"]
        timing_design = specifications["state_interaction"]
        hybrid_probability = _nested_gap_probabilities(
            churn_choice, (churn_design[test] - churn_mean) / churn_scale,
            timing_choice, (timing_design[test] - mean) / scale,
        )
        metric_rows.append({
            "persona": persona, "specification": "selected_churn_state_timing",
            "memory": memory, "train_choices": int(train.sum()),
            "test_choices": int(test.sum()), **_choice_metrics(actual_test, hybrid_probability),
        })
        pooled_selected_actual.append(actual_test)
        pooled_selected_probability.append(hybrid_probability)
        coefficient = np.zeros((len(GAP_TOKENS) - 1, len(interaction_features)))
        intercept = np.full(len(GAP_TOKENS) - 1, -30.0)
        coefficient[timing_choice.classes_.astype(int)] = timing_choice.coef_
        intercept[timing_choice.classes_.astype(int)] = timing_choice.intercept_
        model["personas"][str(persona)] = {
            "memory": memory, "features": "state_interaction",
            "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": coefficient.tolist(), "intercept": intercept.tolist(),
            "churn_features": selected_churn_features,
            "churn_c": selected_churn_c,
            "churn_mean": churn_mean.tolist(), "churn_scale": churn_scale.tolist(),
            "churn_coefficient": churn_choice.coef_[0].tolist(),
            "churn_intercept": float(churn_choice.intercept_[0]),
        }
        for class_row, action in enumerate(timing_choice.classes_.astype(int)):
            for feature, value in zip(interaction_features, timing_choice.coef_[class_row]):
                coefficient_rows.append({
                    "persona": persona, "gap_token": GAP_TOKENS[action],
                    "state": feature, "coefficient": float(value),
                })
        churn_feature_names = {
            "persona_only": ("intercept_only",),
            "session_signal": tuple(f"current_{name}" for name in GAP_STATE_NAMES),
            "core_session_signal": tuple(f"current_{name}" for name in GAP_STATE_NAMES[:4]),
            "churn_sequence": (
                tuple(f"current_{name}" for name in GAP_STATE_NAMES[:4])
                + tuple(f"lag1_{name}" for name in GAP_STATE_NAMES[:4])
                + tuple(f"lag2_{name}" for name in GAP_STATE_NAMES[:4])
                + ("has_lag1", "has_lag2")
            ),
            "internal_state": tuple(f"state_{name}" for name in GAP_STATE_NAMES),
            "state_interaction": interaction_features,
        }[selected_churn_features]
        for feature, value in zip(churn_feature_names, churn_choice.coef_[0]):
            coefficient_rows.append({
                "persona": persona, "gap_token": "NO_RETURN_30D",
                "state": feature, "coefficient": float(value),
            })
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "gap_token_irl_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "gap_token_irl_coefficients.csv", index=False)
    profile_source = sessions.loc[
        ~sessions.user.isin(representative_ids) & sessions.gap_action.ge(0)
    ].copy()
    profile_source["churn"] = profile_source.gap_action.eq(len(GAP_TOKENS) - 1)
    profile_source["view_only"] = profile_source.view.eq(profile_source.events)
    profile_source["cart_without_purchase"] = profile_source.cart.gt(0) & profile_source.purchase.eq(0)
    profile_source["remove_any"] = profile_source.remove.gt(0)
    profile_source["purchase_any"] = profile_source.purchase.gt(0)
    profiles = profile_source.groupby(["persona", "churn"], as_index=False).agg(
        sessions=("session", "size"), median_events=("events", "median"),
        median_duration_hours=("duration_hours", "median"),
        median_previous_gap_hours=("previous_gap_hours", "median"),
        view_only_rate=("view_only", "mean"),
        cart_without_purchase_rate=("cart_without_purchase", "mean"),
        remove_rate=("remove_any", "mean"), purchase_rate=("purchase_any", "mean"),
    )
    profiles.to_csv(output / "pre_churn_profiles.csv", index=False)
    browser_users, browser_profiles = brief_browser_subtypes(sessions, observation_end)
    browser_users.to_csv(output / "brief_browser_subtypes.csv", index=False)
    browser_profiles.to_csv(output / "brief_browser_subtype_profiles.csv", index=False)
    browser_gmm, browser_pca_variance = subtype_embedding_gmm(embeddings, browser_users, seed)
    browser_gmm.to_csv(output / "brief_browser_embedding_gmm.csv", index=False)
    mission_proxy = estimate_mission_prior(events, embeddings, browser_users, seed)
    (output / "mission_prior_proxy.json").write_text(
        json.dumps(mission_proxy, indent=2), encoding="utf-8"
    )
    mature_counts = embeddings.loc[embeddings.cluster.isin(MATURE_PERSONAS), "cluster"].value_counts()
    model["lifecycle"] = {
        "mature_personas": list(MATURE_PERSONAS),
        "home_persona_prior": {
            str(persona): float(mature_counts.get(persona, 0) / mature_counts.sum())
            for persona in MATURE_PERSONAS
        },
        "no_coupon_transitions": {
            "NEW": ["HOME_PERSONA", "CHURNING", "MISSION_EXIT"],
            "HOME_PERSONA": ["HOME_PERSONA", "CHURNING", "MISSION_EXIT"],
            "CHURNING": ["HOME_PERSONA", "EXIT", "MISSION_EXIT"],
            "MISSION_EXIT": ["MISSION_EXIT"],
            "EXIT": ["EXIT"],
        },
        "note": (
            "home persona is fixed at birth; MISSION_EXIT is assigned only after "
            "a purchase followed by fitted no-return, without a fixed birth prior"
        ),
    }
    sessions["session_index"] = sessions.groupby("user", sort=False).cumcount()
    sessions["lifecycle_state"] = np.select(
        [
            sessions.gap_action.eq(len(GAP_TOKENS) - 1) & sessions.purchase.gt(0),
            sessions.gap_action.eq(len(GAP_TOKENS) - 1), sessions.session_index.lt(2),
            sessions.persona.isin(MATURE_PERSONAS),
        ],
        ["MISSION_EXIT_CANDIDATE", "CHURNING", "NEW", "P" + sessions.persona.astype(str)],
        default="UNRESOLVED_SHORT_BROWSER",
    )
    lifecycle_source = sessions.loc[
        ~sessions.user.isin(representative_ids) & sessions.gap_action.ge(0)
    ].copy()
    lifecycle_source["purchase_any"] = lifecycle_source.purchase.gt(0)
    lifecycle_profiles = lifecycle_source.groupby("lifecycle_state", as_index=False).agg(
        sessions=("session", "size"), users=("user", "nunique"),
        view=("view", "sum"), cart=("cart", "sum"),
        remove=("remove", "sum"), purchase=("purchase", "sum"),
        median_events=("events", "median"),
        median_previous_gap_hours=("previous_gap_hours", "median"),
        purchase_session_rate=("purchase_any", "mean"),
    )
    total_actions = lifecycle_profiles[["view", "cart", "remove", "purchase"]].sum(axis=1)
    for action in ("view", "cart", "remove", "purchase"):
        lifecycle_profiles[f"{action}_share"] = lifecycle_profiles[action] / total_actions
    lifecycle_profiles.to_csv(output / "lifecycle_state_profiles.csv", index=False)
    ordered = sessions.sort_values(["user", "start"], kind="stable")
    ordered["next_lifecycle_state"] = ordered.groupby("user", sort=False).lifecycle_state.shift(-1)
    transitions = ordered.loc[
        ~ordered.user.isin(representative_ids) & ordered.next_lifecycle_state.notna()
    ].groupby(["lifecycle_state", "next_lifecycle_state"], as_index=False).size()
    transitions["probability"] = transitions["size"] / transitions.groupby(
        "lifecycle_state"
    )["size"].transform("sum")
    transitions.to_csv(output / "lifecycle_state_transitions.csv", index=False)
    (output / "gap_token_irl_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    trajectory_rows = []
    history_columns = [
        "events", "duration_hours", "previous_gap_hours", "view", "cart", "remove", "purchase"
    ]
    trajectory_source = ordered.loc[
        ~ordered.user.isin(representative_ids) & ordered.gap_action.ge(0)
    ].copy()
    trajectory_source["outcome"] = np.where(
        trajectory_source.gap_action.eq(len(GAP_TOKENS) - 1), "CHURNING", "RETURNED"
    )
    grouped_history = trajectory_source.groupby("user", sort=False)[history_columns]
    for lag in (2, 1, 0):
        history = grouped_history.shift(lag) if lag else trajectory_source[history_columns]
        history = history.assign(
            persona=trajectory_source.persona.to_numpy(),
            outcome=trajectory_source.outcome.to_numpy(),
        ).dropna(subset=["events"])
        history["view_only"] = history.view.eq(history.events)
        history["cart_without_purchase"] = history.cart.gt(0) & history.purchase.eq(0)
        history["remove_any"] = history.remove.gt(0)
        history["purchase_any"] = history.purchase.gt(0)
        aggregated = history.groupby(["persona", "outcome"], as_index=False).agg(
            sessions=("events", "size"), median_events=("events", "median"),
            median_duration_hours=("duration_hours", "median"),
            median_previous_gap_hours=("previous_gap_hours", "median"),
            view_only_rate=("view_only", "mean"),
            cart_without_purchase_rate=("cart_without_purchase", "mean"),
            remove_rate=("remove_any", "mean"), purchase_rate=("purchase_any", "mean"),
        )
        aggregated["relative_session"] = -lag
        pooled = history.groupby("outcome", as_index=False).agg(
            sessions=("events", "size"), median_events=("events", "median"),
            median_duration_hours=("duration_hours", "median"),
            median_previous_gap_hours=("previous_gap_hours", "median"),
            view_only_rate=("view_only", "mean"),
            cart_without_purchase_rate=("cart_without_purchase", "mean"),
            remove_rate=("remove_any", "mean"), purchase_rate=("purchase_any", "mean"),
        )
        pooled["persona"] = "ALL"
        pooled["relative_session"] = -lag
        trajectory_rows.extend([aggregated, pooled])
    pre_churn_trajectory = pd.concat(trajectory_rows, ignore_index=True)
    pre_churn_trajectory.to_csv(output / "pre_churn_trajectory.csv", index=False)
    weighted = {}
    for name, frame in metrics.groupby("specification"):
        weighted[name] = {
            key: float(np.average(frame[key], weights=frame.test_choices))
            for key in (
                "nll", "accuracy", "token_mix_rmse", "timing_nll", "churn_brier", "churn_nll",
                "churn_roc_auc", "churn_average_precision", "observed_churn_rate",
                "predicted_churn_rate",
            )
        }
    pooled_metrics = _choice_metrics(
        np.concatenate(pooled_selected_actual), np.concatenate(pooled_selected_probability)
    )
    summary = {
        "data": quality, "observation_end": observation_end.isoformat(),
        "gap_tokens": list(GAP_TOKENS), "state_names": list(GAP_STATE_NAMES),
        "state_hypotheses": {
            "engagement": "exponentially smoothed session activity intensity",
            "friction": "smoothed removals and unconverted cart activity",
            "satiation": "smoothed purchase share after consumption",
            "inactivity_momentum": "smoothed prior inter-session delay",
            "gap_acceleration": "positive increase of prior delay above its smoothed level",
            "habit_strength": "smoothed count-like persistence across completed sessions",
        },
        "selected_memory_by_persona": {
            persona: specification["memory"] for persona, specification in model["personas"].items()
        },
        "selected_churn_features_by_persona": {
            persona: specification["churn_features"]
            for persona, specification in model["personas"].items()
        },
        "selected_churn_c_by_persona": {
            persona: specification["churn_c"]
            for persona, specification in model["personas"].items()
        },
        "weighted_test_metrics": weighted,
        "pooled_selected_test_metrics": pooled_metrics,
        "pre_churn_profiles": profiles.to_dict(orient="records"),
        "lifecycle_state_profiles": lifecycle_profiles.to_dict(orient="records"),
        "lifecycle_state_transitions": transitions.to_dict(orient="records"),
        "brief_browser_subtype_profiles": browser_profiles.to_dict(orient="records"),
        "brief_browser_embedding_gmm": {
            "pca_explained_variance": browser_pca_variance,
            "components": browser_gmm.to_dict(orient="records"),
        },
        "lifecycle_model": model["lifecycle"],
        "mission_prior_proxy": mission_proxy,
        "pre_churn_trajectory": pre_churn_trajectory.to_dict(orient="records"),
        "limitations": [
            "This is contextual maximum-entropy reward recovery, not identified causal utility.",
            "Internal states are interpretable deterministic hypotheses rather than directly observed variables.",
            "Single-view exits have unknown intent; MISSION_EXIT prevalence is transported from matched long-search failures.",
            "P1 is not a simulator persona; its observations remain unresolved unless NEW or an observed terminal state applies.",
            "MISSION_EXIT is an outcome after purchase and fitted non-return, not a fixed birth archetype.",
            "Subtype rules use retrospective full histories and are simulator calibration, not leakage-free prediction.",
            "This baseline contains no coupon/uplift adjustment.",
            "NO_RETURN_30D is right-censoring-aware non-return within 30 days, not permanent churn.",
        ],
    }
    (output / "gap_token_irl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _gap_states(
    frame: pd.DataFrame,
    initial_gap_hours: float = 0.0,
    user_initial_gaps: dict[int, float] | None = None,
) -> np.ndarray:
    return np.stack([
        persona_gap_features({
            **row._asdict(),
            "initial_gap_hours": (
                user_initial_gaps.get(int(row.user), initial_gap_hours)
                if user_initial_gaps else initial_gap_hours
            ),
            "user_mean_gap_hours": (
                user_initial_gaps.get(int(row.user), initial_gap_hours)
                if user_initial_gaps else initial_gap_hours
            ),
        })
        for row in frame.itertuples(index=False)
    ])


def _hazard_rows(frame: pd.DataFrame, observation_end: pd.Timestamp,
                 state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = (observation_end - frame.end).dt.total_seconds().to_numpy() / 3600
    gap = frame.next_gap_hours.to_numpy(dtype=float)
    event = np.isfinite(gap) & (gap <= available)
    elapsed = np.where(event, gap, available)
    x, y = [], []
    for i, (lo, hi) in enumerate(zip(GAP_EDGES[:-1], GAP_EDGES[1:])):
        valid = (available >= hi) & (elapsed >= lo)
        if not valid.any():
            continue
        bins = np.zeros((valid.sum(), len(GAP_EDGES) - 1)); bins[:, i] = 1
        x.append(np.column_stack((bins, state[valid])))
        y.append((event[valid] & (elapsed[valid] < hi)).astype(int))
    return np.concatenate(x), np.concatenate(y)


def _return_probability(coefficient: np.ndarray, state: np.ndarray, horizon: float) -> np.ndarray:
    survival = np.ones(len(state))
    for i, hi in enumerate(GAP_EDGES[1:]):
        if hi > horizon:
            break
        bins = np.zeros((len(state), len(GAP_EDGES) - 1)); bins[:, i] = 1
        logit = np.column_stack((bins, state)) @ coefficient
        survival *= 1 / (1 + np.exp(np.clip(logit, -30, 30)))
    return 1 - survival


def _mean_gap_brier(coefficient: np.ndarray, frame: pd.DataFrame, state: np.ndarray,
                    observation_end: pd.Timestamp) -> float:
    available = (observation_end - frame.end).dt.total_seconds().to_numpy() / 3600
    gap = frame.next_gap_hours.to_numpy(dtype=float)
    scores = []
    for horizon in GAP_EVAL_HORIZONS:
        eligible = available >= horizon
        if eligible.any():
            truth = (np.isfinite(gap[eligible]) & (gap[eligible] <= horizon)).astype(int)
            scores.append(brier_score_loss(
                truth, _return_probability(coefficient, state[eligible], horizon)
            ))
    return float(np.mean(scores))


def fit_persona_session_gaps(
    raw_path: Path,
    cluster_report: Path,
    embeddings_path: Path,
    output: Path,
    per_persona: int = 20,
    sample_modulus: int = 100,
    seed: int = 42,
) -> dict:
    """Fit persona-specific discrete return hazards with censoring."""
    report = json.loads(cluster_report.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    centroids = np.asarray([p["centroid"] for p in sorted(report["profiles"], key=lambda p: p["cluster"])])
    representatives = select_representatives(embeddings, centroids, per_persona)
    events, quality, _ = source_data("cosmetics", raw_path, sample_modulus)
    sessions, observation_end = session_summaries(events, embeddings[["user_id", "cluster"]])
    representative_ids = set(representatives.user_id.astype(int))
    initial_pool = sessions.loc[
        ~sessions.user.isin(representative_ids)
        & sessions.previous_gap_hours.between(0, GAP_EDGES[-1]), "previous_gap_hours"
    ]
    initial_gap_hours = float(initial_pool.mean())
    valid_personal_gaps = sessions.loc[
        sessions.previous_gap_hours.between(0, GAP_EDGES[-1]),
        ["user", "persona", "previous_gap_hours"],
    ]
    personal_priors = valid_personal_gaps.groupby(["user", "persona"], as_index=False).agg(
        initial_gap_hours=("previous_gap_hours", "mean"),
        observed_gaps=("previous_gap_hours", "size"),
    )
    user_initial_gaps = dict(zip(
        personal_priors.user.astype(int), personal_priors.initial_gap_hours.astype(float)
    ))
    model = {"edges_hours": GAP_EDGES.tolist(), "feature_names": list(GAP_FEATURES),
             "initial_gap_hours": initial_gap_hours,
             "default_user_mean_gap_hours": initial_gap_hours, "personas": {}}
    metrics = []
    for persona in range(len(centroids)):
        train = sessions.loc[sessions.persona.eq(persona) & ~sessions.user.isin(representative_ids)].copy()
        test = sessions.loc[sessions.persona.eq(persona) & sessions.user.isin(representative_ids)].copy()
        validation_users = train.user.drop_duplicates()
        hashes = pd.util.hash_pandas_object(
            "gap:" + validation_users.astype(str), index=False
        ).to_numpy()
        validation_users = set(validation_users[hashes % 5 == 0])
        fit = train.loc[~train.user.isin(validation_users)]
        validation = train.loc[train.user.isin(validation_users)]
        fit_state = _gap_states(fit, initial_gap_hours)
        validation_state = _gap_states(validation, initial_gap_hours)
        fit_mean, fit_scale = fit_state.mean(axis=0), fit_state.std(axis=0)
        fit_scale[fit_scale == 0] = 1.0
        fit_state = (fit_state - fit_mean) / fit_scale
        validation_state = (validation_state - fit_mean) / fit_scale
        fit_x, fit_y = _hazard_rows(fit, observation_end, fit_state)
        candidate_full = LogisticRegression(
            C=1.0, fit_intercept=False, max_iter=1000, random_state=seed
        ).fit(fit_x, fit_y)
        candidate_baseline = LogisticRegression(
            C=1.0, fit_intercept=False, max_iter=1000, random_state=seed
        ).fit(fit_x[:, :len(GAP_EDGES) - 1], fit_y)
        full_validation_brier = _mean_gap_brier(
            candidate_full.coef_[0], validation, validation_state, observation_end
        )
        padded_baseline = np.r_[candidate_baseline.coef_[0], np.zeros(fit_state.shape[1])]
        baseline_validation_brier = _mean_gap_brier(
            padded_baseline, validation, validation_state, observation_end
        )
        selected_model = ("state_conditional" if full_validation_brier <= baseline_validation_brier
                          else "bin_only")

        train_state = _gap_states(train, initial_gap_hours)
        test_state = _gap_states(test, initial_gap_hours)
        mean, scale = train_state.mean(axis=0), train_state.std(axis=0)
        scale[scale == 0] = 1.0
        train_state, test_state = (train_state - mean) / scale, (test_state - mean) / scale
        x, y = _hazard_rows(train, observation_end, train_state)
        hazard = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000,
                                    random_state=seed).fit(x, y)
        baseline = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000,
                                      random_state=seed).fit(x[:, :len(GAP_EDGES) - 1], y)
        zero_train_state, zero_test_state = _gap_states(train), _gap_states(test)
        zero_mean, zero_scale = zero_train_state.mean(axis=0), zero_train_state.std(axis=0)
        zero_scale[zero_scale == 0] = 1.0
        zero_train_state = (zero_train_state - zero_mean) / zero_scale
        zero_test_state = (zero_test_state - zero_mean) / zero_scale
        zero_x, zero_y = _hazard_rows(train, observation_end, zero_train_state)
        zero_hazard = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000,
                                         random_state=seed).fit(zero_x, zero_y)
        oracle_train_state = _gap_states(train, initial_gap_hours, user_initial_gaps)
        oracle_test_state = _gap_states(test, initial_gap_hours, user_initial_gaps)
        oracle_mean = oracle_train_state.mean(axis=0)
        oracle_scale = oracle_train_state.std(axis=0)
        oracle_scale[oracle_scale == 0] = 1.0
        oracle_train_state = (oracle_train_state - oracle_mean) / oracle_scale
        oracle_test_state = (oracle_test_state - oracle_mean) / oracle_scale
        oracle_x, oracle_y = _hazard_rows(train, observation_end, oracle_train_state)
        oracle_hazard = LogisticRegression(
            C=1.0, fit_intercept=False, max_iter=1000, random_state=seed
        ).fit(oracle_x, oracle_y)
        coefficient = (hazard.coef_[0] if selected_model == "state_conditional" else
                       np.r_[baseline.coef_[0], np.zeros(train_state.shape[1])])
        model["personas"][str(persona)] = {
            "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": coefficient.tolist(), "selected_model": selected_model,
            "validation_brier": full_validation_brier,
            "validation_bin_only_brier": baseline_validation_brier,
            "training_sessions": len(train), "hazard_rows": len(y),
        }
        available = (observation_end - test.end).dt.total_seconds().to_numpy() / 3600
        gap = test.next_gap_hours.to_numpy(dtype=float)
        for horizon in GAP_EVAL_HORIZONS:
            eligible = available >= horizon
            if not eligible.any():
                continue
            truth = (np.isfinite(gap[eligible]) & (gap[eligible] <= horizon)).astype(int)
            probability = _return_probability(coefficient, test_state[eligible], horizon)
            baseline_probability = _return_probability(
                np.r_[baseline.coef_[0], np.zeros(test_state.shape[1])], test_state[eligible], horizon
            )
            zero_probability = _return_probability(
                zero_hazard.coef_[0], zero_test_state[eligible], horizon
            )
            oracle_probability = _return_probability(
                oracle_hazard.coef_[0], oracle_test_state[eligible], horizon
            )
            first_session = test.previous_gap_hours.isna().to_numpy()[eligible]
            first_global_brier = (
                brier_score_loss(truth[first_session], probability[first_session])
                if first_session.any() else np.nan
            )
            first_oracle_brier = (
                brier_score_loss(truth[first_session], oracle_probability[first_session])
                if first_session.any() else np.nan
            )
            metrics.append({
                "persona": persona, "horizon_hours": int(horizon), "sessions": int(eligible.sum()),
                "selected_model": selected_model,
                "observed_return_rate": float(truth.mean()),
                "predicted_return_rate": float(probability.mean()),
                "brier": float(brier_score_loss(truth, probability)),
                "bin_only_brier": float(brier_score_loss(truth, baseline_probability)),
                "zero_magnitude_brier": float(brier_score_loss(truth, zero_probability)),
                "oracle_personal_magnitude_brier": float(
                    brier_score_loss(truth, oracle_probability)
                ),
                "first_sessions": int(first_session.sum()),
                "first_session_global_initial_brier": float(first_global_brier),
                "first_session_oracle_personal_magnitude_brier": float(first_oracle_brier),
            })
    output.mkdir(parents=True, exist_ok=True)
    (output / "session_gap_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "session_gap_metrics.csv", index=False)
    personal_priors.to_csv(output / "session_gap_user_oracle_priors.csv", index=False)
    summary = {
        "data": quality, "observation_end": observation_end.isoformat(),
        "initial_gap_hours": initial_gap_hours,
        "initial_gap_rule": "mean observed nonnegative inter-session gap up to 30 days among training users",
        "oracle_personal_magnitude_users": int(len(personal_priors)),
        "oracle_single_gap_user_share": float(personal_priors.observed_gaps.eq(1).mean()),
        "rule": "persona-specific discrete hazard; sequential Bernoulli bins then uniform time within selected bin",
        "edges_hours": GAP_EDGES.tolist(), "features": list(GAP_FEATURES),
        "evaluation_horizons_hours": list(GAP_EVAL_HORIZONS),
        "metrics": metrics,
        "mean_brier": float(metric_frame.brier.mean()),
        "mean_bin_only_brier": float(metric_frame.bin_only_brier.mean()),
        "mean_zero_magnitude_brier": float(metric_frame.zero_magnitude_brier.mean()),
        "mean_oracle_personal_magnitude_brier": float(
            metric_frame.oracle_personal_magnitude_brier.mean()
        ),
        "mean_first_session_global_initial_brier": float(
            metric_frame.first_session_global_initial_brier.mean()
        ),
        "mean_first_session_oracle_personal_magnitude_brier": float(
            metric_frame.first_session_oracle_personal_magnitude_brier.mean()
        ),
        "limitations": [
            "None from the sampler means no return within 30 days, not permanent churn.",
            "Within-bin return time is sampled uniformly for computational convenience.",
            "Persona and session associations are observational and not causal effects.",
            "The initial gap is a population magnitude for left-truncated first sessions, not an observed personal history.",
            "Oracle personal mean gaps use each user's complete observed timeline, including future gaps, as a persistent latent magnitude and are an upper-bound ablation only.",
            "Nearly half of oracle users have only one observed gap, so their magnitude directly contains that future interval.",
        ],
    }
    (output / "session_gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _session_prior_states(
    sessions: pd.DataFrame, memories: dict[int, float], initial_gap_hours: float
) -> pd.DataFrame:
    """Return recursive state known before each session starts."""
    initial_inactivity = np.log1p(initial_gap_hours) / np.log1p(GAP_EDGES[-1])
    rows = []
    for user, frame in sessions.groupby("user", sort=False):
        state = np.array([0.0, 0.0, 0.0, initial_inactivity, 0.0, 0.0])
        for session_index, row in enumerate(frame.sort_values("start").itertuples(index=False)):
            rows.append({
                "user_id": int(user), "session": row.session,
                "new_state": float(session_index < 2),
                **{f"prior_{name}": float(value) for name, value in zip(GAP_STATE_NAMES, state)},
            })
            _, state = update_gap_internal_state(
                state, row._asdict(), memories[int(row.persona)], initial_gap_hours
            )
    return pd.DataFrame(rows)


def fit_lifecycle_irl(
    raw_path: Path,
    cluster_report: Path,
    embeddings_path: Path,
    output: Path,
    per_persona: int = 20,
    sample_modulus: int = 100,
    seed: int = 42,
) -> dict:
    """Fit immediate action rewards with past-only recursive lifecycle state."""
    gap_path = output / "gap_token_irl_model.json"
    if not gap_path.exists():
        raise FileNotFoundError("fit the gap-token IRL before lifecycle IRL")
    gap_model = json.loads(gap_path.read_text(encoding="utf-8"))
    report = json.loads(cluster_report.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    centroids = np.asarray([p["centroid"] for p in sorted(report["profiles"], key=lambda p: p["cluster"])])
    representatives = select_representatives(
        embeddings, centroids, per_persona, personas=MATURE_PERSONAS
    )
    events, quality, _ = source_data("cosmetics", raw_path, sample_modulus)
    assignments = embeddings[["user_id", "cluster"]]
    sessions, _ = session_summaries(events, assignments)
    memories = {
        persona: float(gap_model["personas"][str(persona)]["memory"])
        for persona in map(int, gap_model["personas"])
    }
    prior_states = _session_prior_states(sessions, memories, gap_model["initial_gap_hours"])
    decisions = make_decisions(events, assignments).merge(
        prior_states, on=["user_id", "session"], how="left", validate="many_to_one"
    )
    state_features = tuple(f"prior_{name}" for name in GAP_STATE_NAMES) + ("new_state",)
    lifecycle_features = (*FEATURES, *state_features)
    representative_ids = set(representatives.user_id.astype(int))
    train = ~decisions.user_id.isin(representative_ids) & decisions.persona.isin(MATURE_PERSONAS)
    base_reference = ~decisions.user_id.isin(representative_ids)
    raw_design = decisions.loc[:, lifecycle_features].to_numpy(dtype=float)
    base_mean = decisions.loc[base_reference, FEATURES[1:]].mean().to_numpy(dtype=float)
    base_scale = decisions.loc[base_reference, FEATURES[1:]].std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    state_mean = decisions.loc[train, state_features].mean().to_numpy(dtype=float)
    state_scale = decisions.loc[train, state_features].std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    mean, scale = np.r_[base_mean, state_mean], np.r_[base_scale, state_scale]
    design = raw_design.copy()
    design[:, 1:] = (design[:, 1:] - mean) / scale
    base_design, _, _ = _scale(decisions.loc[base_reference], decisions)
    rng = np.random.default_rng(seed)
    metric_rows, model_personas = [], {}
    for persona in MATURE_PERSONAS:
        train_index = decisions.index[train & decisions.persona.eq(persona)].to_numpy()
        if len(train_index) > 100_000:
            train_index = rng.choice(train_index, 100_000, replace=False)
        base_theta = fit_reward(
            base_design[train_index], decisions.loc[train_index, "action"].to_numpy(dtype=int),
            penalty=1e-3,
        )
        centre = np.column_stack((base_theta, np.zeros((len(ACTIONS), len(state_features)))))
        validation, test = [], []
        for user in representatives.loc[representatives.cluster.eq(persona), "user_id"]:
            frame = decisions.loc[decisions.user_id.eq(int(user))]
            validation_start, test_start = _temporal_boundaries(len(frame))
            validation.extend(frame.index[validation_start:test_start])
            test.extend(frame.index[test_start:])
        validation, test = np.asarray(validation), np.asarray(test)
        candidates = []
        for penalty in (0.001, 0.01, 0.1, 1.0):
            theta = fit_reward(
                design[train_index], decisions.loc[train_index, "action"].to_numpy(dtype=int),
                centre=centre, penalty=penalty,
            )
            probability = action_probabilities(design[validation], theta)
            actual = decisions.loc[validation, "action"].to_numpy(dtype=int)
            candidates.append((
                float(-np.log(probability[np.arange(len(actual)), actual] + 1e-15).mean()),
                penalty, theta,
            ))
        _, penalty, theta = min(candidates, key=lambda item: item[0])
        probability = action_probabilities(design[test], theta)
        base_probability = action_probabilities(base_design[test], base_theta)
        actual = decisions.loc[test, "action"].to_numpy(dtype=int)
        observed = np.bincount(actual, minlength=len(ACTIONS)) / len(actual)
        metric_rows.append({
            "persona": persona, "test_decisions": len(test), "penalty": penalty,
            "lifecycle_nll": float(-np.log(probability[np.arange(len(actual)), actual] + 1e-15).mean()),
            "base_nll": float(-np.log(base_probability[np.arange(len(actual)), actual] + 1e-15).mean()),
            "lifecycle_accuracy": float((probability.argmax(axis=1) == actual).mean()),
            "base_accuracy": float((base_probability.argmax(axis=1) == actual).mean()),
            "lifecycle_mix_rmse": float(np.sqrt(np.square(probability.mean(axis=0) - observed).mean())),
            "base_mix_rmse": float(np.sqrt(np.square(base_probability.mean(axis=0) - observed).mean())),
        })
        model_personas[str(persona)] = {
            "memory": memories[persona],
            "coefficient": theta.tolist(),
            "base_coefficient": base_theta.tolist(),
        }
    metrics = pd.DataFrame(metric_rows)
    model = {
        "model": "immediate MaxEnt action reward with recursive lifecycle-state offsets",
        "formula": "r(a_t)=theta[a_t]^T phi_t + beta[a_t]^T z_t + nu[a_t] I(NEW)",
        "actions": list(ACTIONS), "features": list(lifecycle_features),
        "state_features": list(state_features), "mean": mean.tolist(), "scale": scale.tolist(),
        "temperature": 1.0, "personas": model_personas,
        "transition_model": "gap_token_irl_model.json",
    }
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "lifecycle_irl_metrics.csv", index=False)
    (output / "lifecycle_irl_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    summary = {
        "data": quality, "representatives": len(representatives),
        "coefficient_dynamics": (
            "user baseline remains fixed; recursive state changes smoothly and adds action-specific reward offsets"
        ),
        "metrics": metric_rows,
        "mean_lifecycle_nll": float(metrics.lifecycle_nll.mean()),
        "mean_base_nll": float(metrics.base_nll.mean()),
        "mean_lifecycle_accuracy": float(metrics.lifecycle_accuracy.mean()),
        "mean_base_accuracy": float(metrics.base_accuracy.mean()),
        "limitations": [
            "Lifecycle state is deterministic recursive feature engineering, not an identified psychological state.",
            "The action reward is immediate contextual MaxEnt; transitions are fitted separately by the gap model.",
            "MISSION_EXIT is assigned after purchase plus modeled 30-day non-return, not observed intent.",
        ],
    }
    (output / "lifecycle_irl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_no_uplift_rollout(
    output: Path, coefficient_gmm_path: Path, users: int = 10_000, seed: int = 42
) -> dict:
    """Validate a fully generated baseline against held-out aggregate targets."""
    coefficient_model = json.loads(coefficient_gmm_path.read_text(encoding="utf-8"))
    action_model = json.loads((output / "lifecycle_irl_model.json").read_text(encoding="utf-8"))
    gap_model = json.loads((output / "gap_token_irl_model.json").read_text(encoding="utf-8"))
    rollout = simulate_no_uplift_users(
        coefficient_model, action_model, gap_model, users=users, seed=seed
    )
    generated = pd.DataFrame(rollout["sessions"])
    generated.to_parquet(output / "no_uplift_rollout_sessions.parquet", index=False)
    transitions = pd.DataFrame(
        rollout["transitions"], columns=["lifecycle_state", "next_lifecycle_state"]
    ).value_counts().rename("sessions").reset_index()
    transitions["probability"] = transitions.sessions / transitions.groupby(
        "lifecycle_state"
    ).sessions.transform("sum")
    transitions.to_csv(output / "no_uplift_rollout_transitions.csv", index=False)
    gap_mix = generated.gap_token.value_counts(normalize=True).rename("probability").reset_index()
    gap_mix.to_csv(output / "no_uplift_rollout_gap_mix.csv", index=False)

    profiles = pd.read_csv(output / "lifecycle_state_profiles.csv")
    profiles = profiles.loc[profiles.lifecycle_state.ne("UNRESOLVED_SHORT_BROWSER")]
    target_counts = profiles[["view", "cart", "remove", "purchase"]].sum().to_numpy(float)
    generated_counts = generated[["view", "cart", "remove", "purchase"]].sum().to_numpy(float)
    target_mix, generated_mix = target_counts / target_counts.sum(), generated_counts / generated_counts.sum()
    target_purchase_session = float(np.average(
        profiles.purchase_session_rate, weights=profiles.sessions
    ))
    generated_purchase_session = float(generated.purchase.gt(0).mean())
    gap_summary = json.loads((output / "gap_token_irl_summary.json").read_text(encoding="utf-8"))
    target_churn = gap_summary["pooled_selected_test_metrics"]["observed_churn_rate"]
    generated_churn = float(generated.gap_hours.isna().mean())
    terminal_profiles = profiles.set_index("lifecycle_state").sessions
    target_mission = float(
        terminal_profiles.get("MISSION_EXIT_CANDIDATE", 0)
        / (terminal_profiles.get("MISSION_EXIT_CANDIDATE", 0)
           + terminal_profiles.get("CHURNING", 0))
    )
    generated_nonreturn = generated.loc[generated.gap_hours.isna()]
    generated_mission = float(
        generated_nonreturn.next_lifecycle_state.eq("MISSION_EXIT").mean()
    )
    action_rmse = float(np.sqrt(np.square(generated_mix - target_mix).mean()))
    checks = {
        "action_mix_rmse_le_0_10": action_rmse <= .10,
        "churn_rate_absolute_error_le_0_05": abs(generated_churn - target_churn) <= .05,
        "mission_share_absolute_error_le_0_05": abs(generated_mission - target_mission) <= .05,
        "forced_end_rate_le_0_01": rollout["forced_end_rate"] <= .01,
        "truncated_user_rate_le_0_01": rollout["truncated_user_rate"] <= .01,
    }
    result = {
        "model": "end-to-end no-uplift rollout",
        "users": users, "sessions": len(generated),
        "mean_sessions_per_user": float(len(generated) / users),
        "action_mix": {
            action: {"observed": float(target_mix[i]), "generated": float(generated_mix[i])}
            for i, action in enumerate(ACTIONS[:-1])
        },
        "action_mix_rmse": action_rmse,
        "purchase_session_rate": {
            "observed": target_purchase_session, "generated": generated_purchase_session,
        },
        "no_return_30d_rate": {"observed": target_churn, "generated": generated_churn},
        "mission_exit_share_among_nonreturns": {
            "observed_proxy": target_mission, "generated": generated_mission,
        },
        "forced_end_rate": rollout["forced_end_rate"],
        "truncated_user_rate": rollout["truncated_user_rate"],
        "typical_event_gap_minutes": rollout["typical_event_gap_minutes"],
        "checks": checks, "ready": all(checks.values()),
        "limitations": [
            "Observed targets are aggregate held-out diagnostics, not causal quantities.",
            "Within-session event time uses an exponential approximation calibrated from the training scaler.",
            "MISSION_EXIT remains purchase plus 30-day non-return; customer intent is unobserved.",
        ],
    }
    (output / "no_uplift_rollout_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run(
    raw_path: Path,
    cluster_report: Path,
    embeddings_path: Path,
    output: Path,
    per_persona: int = 20,
    sample_modulus: int = 100,
    seed: int = 42,
) -> dict:
    report = json.loads(cluster_report.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    centroids = np.asarray([p["centroid"] for p in sorted(report["profiles"], key=lambda p: p["cluster"])])
    representatives = select_representatives(
        embeddings, centroids, per_persona, personas=MATURE_PERSONAS
    )
    events, quality, _ = source_data("cosmetics", raw_path, sample_modulus)
    assignments = embeddings[["user_id", "cluster"]]
    decisions = make_decisions(events, assignments)
    representative_ids = set(representatives.user_id.astype(int))
    prior_rows = decisions.loc[~decisions.user_id.isin(representative_ids)].copy()
    x_all, mean, scale = _scale(prior_rows, decisions)
    decisions["_row"] = np.arange(len(decisions))
    rng = np.random.default_rng(seed)
    priors: dict[int, np.ndarray] = {}
    for persona in MATURE_PERSONAS:
        index = prior_rows.index[prior_rows.persona.eq(persona)].to_numpy()
        if len(index) > 100_000:
            index = rng.choice(index, 100_000, replace=False)
        priors[persona] = fit_reward(
            x_all[index], decisions.loc[index, "action"].to_numpy(dtype=int), penalty=1e-3
        )

    representative_frames = {
        int(user): decisions.loc[decisions.user_id.eq(int(user))]
        for user in representatives.user_id
    }
    penalty_candidates = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    selected_penalties: dict[int, float] = {}
    for persona in MATURE_PERSONAS:
        users = representatives.loc[representatives.cluster.eq(persona), "user_id"]
        scores = []
        for penalty in penalty_candidates:
            loss, rows = 0.0, 0
            for user in users:
                frame = representative_frames[int(user)]
                validation_start, test_start = _temporal_boundaries(len(frame))
                development, validation = frame.iloc[:validation_start], frame.iloc[validation_start:test_start]
                theta = fit_reward(
                    x_all[development.index], development.action.to_numpy(dtype=int),
                    centre=priors[persona], penalty=penalty, average=False,
                )
                probability = action_probabilities(x_all[validation.index], theta)
                actual = validation.action.to_numpy(dtype=int)
                loss += float(-np.log(probability[np.arange(len(actual)), actual] + 1e-15).sum())
                rows += len(actual)
            scores.append((loss / rows, penalty))
        selected_penalties[persona] = min(scores)[1]

    coefficient_rows, metric_rows = [], []
    for rep in representatives.itertuples(index=False):
        user_rows = representative_frames[int(rep.user_id)]
        _, test_start = _temporal_boundaries(len(user_rows))
        train_rows, test_rows = user_rows.iloc[:test_start], user_rows.iloc[test_start:]
        persona = int(rep.cluster)
        penalty = selected_penalties[persona]
        theta = fit_reward(
            x_all[train_rows.index], train_rows.action.to_numpy(dtype=int),
            centre=priors[persona], penalty=penalty, average=False,
        )
        evaluation = test_rows
        probabilities = action_probabilities(x_all[evaluation.index], theta)
        prior_probabilities = action_probabilities(x_all[evaluation.index], priors[persona])
        actual = evaluation.action.to_numpy(dtype=int)
        observed_mix = np.bincount(actual, minlength=len(ACTIONS)) / len(actual)
        predicted_mix = probabilities.mean(axis=0)
        prior_mix = prior_probabilities.mean(axis=0)
        metric_rows.append({
            "user_id": int(rep.user_id), "persona": persona,
            "map_penalty": penalty,
            "centroid_distance": float(rep.centroid_distance),
            "train_decisions": len(train_rows), "test_decisions": len(test_rows),
            "test_nll": float(-np.log(probabilities[np.arange(len(actual)), actual] + 1e-15).mean()),
            "persona_prior_test_nll": float(
                -np.log(prior_probabilities[np.arange(len(actual)), actual] + 1e-15).mean()
            ),
            "test_accuracy": float((probabilities.argmax(axis=1) == actual).mean()),
            "persona_prior_test_accuracy": float(
                (prior_probabilities.argmax(axis=1) == actual).mean()
            ),
            "action_mix_rmse": float(np.sqrt(np.square(predicted_mix - observed_mix).mean())),
            "persona_prior_action_mix_rmse": float(
                np.sqrt(np.square(prior_mix - observed_mix).mean())
            ),
            "coefficient_shift_l2": float(np.linalg.norm(theta - priors[persona])),
        })
        for action_index, action in enumerate(ACTIONS):
            for feature_index, feature in enumerate(FEATURES):
                coefficient_rows.append({
                    "user_id": int(rep.user_id), "persona": persona,
                    "action": action, "feature": feature,
                    "coefficient": float(theta[action_index, feature_index]),
                    "persona_prior": float(priors[persona][action_index, feature_index]),
                })

    output.mkdir(parents=True, exist_ok=True)
    representatives.to_parquet(output / "representative_users.parquet", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_parquet(output / "user_coefficients.parquet", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "user_fit_metrics.csv", index=False)
    prior_payload = {
        str(persona): {
            action: dict(zip(FEATURES, priors[persona][i].tolist()))
            for i, action in enumerate(ACTIONS)
        }
        for persona in priors
    }
    (output / "persona_priors.json").write_text(json.dumps(prior_payload, indent=2), encoding="utf-8")
    mature_counts = embeddings.loc[embeddings.cluster.isin(MATURE_PERSONAS), "cluster"].value_counts()
    home_persona_prior = {
        str(persona): float(mature_counts.get(persona, 0) / mature_counts.sum())
        for persona in MATURE_PERSONAS
    }
    coefficient_gmm = fit_coefficient_gmm(
        coefficients, output, seed,
        persona_population_weights={int(k): v for k, v in home_persona_prior.items()},
    )
    summary = {
        "data": quality,
        "model": "contextual maximum-causal-entropy policy with persona-centred user MAP",
        "identification": "coefficients sum to zero across actions for every feature",
        "temperature": 1.0,
        "mature_personas": list(MATURE_PERSONAS),
        "home_persona_prior": home_persona_prior,
        "lifecycle_states": ["NEW", *[f"P{x}" for x in MATURE_PERSONAS],
                             "CHURNING", "MISSION_EXIT", "EXIT"],
        "features": list(FEATURES),
        "representatives_per_persona": per_persona,
        "representative_users": len(representatives),
        "selected_map_penalty_by_persona": selected_penalties,
        "decisions": len(decisions),
        "inferred_end_decisions": int(decisions.action.eq(4).sum()),
        "mean_test_nll": float(metrics.test_nll.mean()),
        "mean_persona_prior_test_nll": float(metrics.persona_prior_test_nll.mean()),
        "mean_test_accuracy": float(metrics.test_accuracy.mean()),
        "mean_persona_prior_test_accuracy": float(metrics.persona_prior_test_accuracy.mean()),
        "mean_action_mix_rmse": float(metrics.action_mix_rmse.mean()),
        "mean_persona_prior_action_mix_rmse": float(metrics.persona_prior_action_mix_rmse.mean()),
        "coefficient_gmm": {
            "selected_components": coefficient_gmm["selected_components"],
            "weights": coefficient_gmm["weights"],
            "selection": coefficient_gmm["selection"],
        },
        "limitations": [
            "END is inferred from an observable 30-minute inactivity boundary, not logged directly.",
            "The model recovers rewards that rationalise observed choices; it does not identify causal utility.",
            "Coupon response and canonical uplift type are absent from Cosmetics and are not inferred here.",
        ],
        "scaling": {"features": list(FEATURES[1:]), "mean": mean.tolist(), "scale": scale.tolist()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _temporal_boundaries(rows: int) -> tuple[int, int]:
    """Return non-empty 60/20/20 development, validation and test boundaries."""
    if rows < 3:
        raise ValueError("representative users need at least three decisions")
    test_start = min(rows - 1, max(2, int(rows * 0.8)))
    validation_start = min(test_start - 1, max(1, int(rows * 0.6)))
    return validation_start, test_start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--clusters", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-persona", type=int, default=20)
    parser.add_argument("--sample-modulus", type=int, default=100)
    parser.add_argument("--compare-persona-rewards", action="store_true")
    parser.add_argument("--fit-session-gap", action="store_true")
    parser.add_argument("--fit-gap-token-irl", action="store_true")
    parser.add_argument("--fit-lifecycle-irl", action="store_true")
    parser.add_argument("--validate-no-uplift-rollout", action="store_true")
    parser.add_argument("--coefficient-gmm", type=Path)
    parser.add_argument("--rollout-users", type=int, default=10_000)
    args = parser.parse_args()
    if sum((args.compare_persona_rewards, args.fit_session_gap, args.fit_gap_token_irl,
            args.fit_lifecycle_irl, args.validate_no_uplift_rollout)) > 1:
        parser.error("choose one experiment mode")
    if args.validate_no_uplift_rollout:
        if args.coefficient_gmm is None:
            parser.error("--coefficient-gmm is required for rollout validation")
        print(json.dumps(validate_no_uplift_rollout(
            args.output, args.coefficient_gmm, args.rollout_users
        ), indent=2))
        return
    if any(value is None for value in (args.raw, args.clusters, args.embeddings)):
        parser.error("--raw, --clusters and --embeddings are required for fitting")
    function = (fit_lifecycle_irl if args.fit_lifecycle_irl else
                fit_gap_token_irl if args.fit_gap_token_irl else
                fit_persona_session_gaps if args.fit_session_gap else
                compare_persona_rewards if args.compare_persona_rewards else run)
    print(json.dumps(function(args.raw, args.clusters, args.embeddings, args.output,
                              args.per_persona, args.sample_modulus), indent=2))


if __name__ == "__main__":
    main()
