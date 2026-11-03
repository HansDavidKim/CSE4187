from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ACTIONS = ("none", "informational", "discount", "bogo")
OUTCOMES = ("transactions", "spend", "offer_completed", "reward")
STATE_FEATURES = ("transactions", "spend", "offer_completed", "reward")


def _read_json_lines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def load_starbucks_panel(root: str | Path, max_users: int | None = None) -> pd.DataFrame:
    """Convert the Kaggle Starbucks event log to a dense user-day panel."""
    root = Path(root)
    portfolio = _read_json_lines(root / "portfolio.json").set_index("id")
    profile = _read_json_lines(root / "profile.json").rename(columns={"id": "user_id"})
    events = _read_json_lines(root / "transcript.json").rename(columns={"person": "user_id"})
    if max_users:
        users = sorted(events.user_id.unique())[:max_users]
        events = events[events.user_id.isin(users)]
        profile = profile[profile.user_id.isin(users)]

    events["day"] = (events["time"] // 24).astype(int)
    events["offer_id"] = events["value"].map(
        lambda value: value.get("offer id", value.get("offer_id")) if isinstance(value, dict) else None
    )
    events["amount"] = events["value"].map(
        lambda value: float(value.get("amount", 0.0)) if isinstance(value, dict) else 0.0
    )
    events["reward"] = events["value"].map(
        lambda value: float(value.get("reward", 0.0)) if isinstance(value, dict) else 0.0
    )
    offer_type = portfolio["offer_type"].to_dict()
    received = events[events.event.eq("offer received")].copy()
    received["action_name"] = received.offer_id.map(offer_type).fillna("informational")
    # Multiple offers in a day are reduced deterministically; this is an explicit modeling shortcut.
    action = received.groupby(["user_id", "day"])["action_name"].agg(lambda x: sorted(x)[0])

    base = events.assign(
        transactions=events.event.eq("transaction").astype(int),
        offer_viewed=events.event.eq("offer viewed").astype(int),
        offer_completed=events.event.eq("offer completed").astype(int),
    ).groupby(["user_id", "day"]).agg(
        transactions=("transactions", "sum"),
        spend=("amount", "sum"),
        offer_viewed=("offer_viewed", "sum"),
        offer_completed=("offer_completed", "sum"),
        reward=("reward", "sum"),
    )
    users = sorted(events.user_id.unique())
    days = range(int(events.day.min()), int(events.day.max()) + 1)
    index = pd.MultiIndex.from_product([users, days], names=["user_id", "day"])
    panel = base.reindex(index, fill_value=0).reset_index()
    panel["action_name"] = action.reindex(index, fill_value="none").to_numpy()
    panel["action"] = pd.Categorical(panel.action_name, categories=ACTIONS).codes
    for name in ACTIONS[1:]:
        panel[f"action_{name}"] = panel.action_name.eq(name).astype(int)

    profile["age"] = profile.age.where(profile.age.between(18, 100))
    profile["member_year"] = pd.to_datetime(
        profile.became_member_on.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    ).dt.year
    profile = profile[["user_id", "age", "income", "member_year"]]
    panel = panel.merge(profile, on="user_id", how="left")
    for column in ("age", "income", "member_year"):
        panel[column] = panel[column].fillna(panel[column].median())
    return panel.sort_values(["user_id", "day"]).reset_index(drop=True)


def add_lags(panel: pd.DataFrame, columns=STATE_FEATURES, lags: int = 1) -> pd.DataFrame:
    result = panel.copy()
    grouped = result.groupby("user_id", sort=False)
    for lag in range(1, lags + 1):
        for column in columns:
            result[f"{column}_lag{lag}"] = grouped[column].shift(lag)
    return result.dropna().reset_index(drop=True)


@dataclass
class DiscoveryResult:
    nodes: list[str]
    adjacency: np.ndarray
    evidence: list[dict]

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(self.nodes[i], self.nodes[j]) for i, j in zip(*np.where(self.adjacency != 0))]

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges, "intervention_evidence": self.evidence}


def discover_dag(
    frame: pd.DataFrame,
    method: str = "notears-mlp",
    max_rows: int = 5000,
    max_iter: int = 10,
    threshold: float = 0.3,
    seed: int = 7,
) -> DiscoveryResult:
    action_columns = [f"action_{name}" for name in ACTIONS[1:]]
    nodes = [f"{name}_lag1" for name in STATE_FEATURES] + action_columns + list(OUTCOMES)
    nodes = [name for name in nodes if name in frame and frame[name].nunique() > 1]
    sample = frame[nodes].sample(min(max_rows, len(frame)), random_state=seed)
    values = StandardScaler().fit_transform(sample)
    if method == "notears-mlp":
        os.environ.setdefault("CASTLE_BACKEND", "pytorch")
        from castle.algorithms import NotearsNonlinear

        random.seed(seed)
        np.random.seed(seed)
        model = NotearsNonlinear(
            lambda1=0.01,
            lambda2=0.01,
            max_iter=max_iter,
            w_threshold=threshold,
            hidden_layers=(10, 1),
            device_type="cpu",
        )
        model.learn(values, columns=nodes)
        adjacency = np.asarray(model.causal_matrix, dtype=int)
    elif method == "correlation-smoke":
        # shortcut: test-only deterministic graph; use NOTEARS-MLP for substantive runs.
        corr = np.nan_to_num(np.corrcoef(values, rowvar=False))
        adjacency = (np.abs(corr) > 0.2).astype(int)
        adjacency[np.tril_indices_from(adjacency)] = 0
    else:
        raise ValueError(f"unknown discovery method: {method}")

    # Temporal/action tiers encode what the event log actually establishes.
    tier = {name: (0 if "_lag" in name else 1 if name.startswith("action_") else 2) for name in nodes}
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if tier[source] > tier[target] or i == j:
                adjacency[i, j] = 0

    evidence = []
    for action_name in ACTIONS[1:]:
        action_col = f"action_{action_name}"
        if action_col not in nodes:
            continue
        treated = frame[action_col].eq(1)
        for outcome in OUTCOMES:
            if outcome not in nodes or treated.sum() < 20 or (~treated).sum() < 20:
                continue
            effect = float(frame.loc[treated, outcome].mean() - frame.loc[~treated, outcome].mean())
            scale = float(frame[outcome].std() or 1.0)
            standardized = effect / scale
            evidence.append({
                "action": action_name,
                "outcome": outcome,
                "mean_difference": effect,
                "standardized_difference": standardized,
                "treated_rows": int(treated.sum()),
                "identified_as": "associational intervention evidence",
            })
            if abs(standardized) >= 0.05:
                adjacency[nodes.index(action_col), nodes.index(outcome)] = 1
                adjacency[nodes.index(outcome), nodes.index(action_col)] = 0
    return DiscoveryResult(nodes, adjacency, evidence)


class ContextualMaxEntIRL:
    """One-step maximum-entropy IRL: P(a|s) is softmax of a linear reward."""

    def fit(self, states: np.ndarray, actions: np.ndarray) -> "ContextualMaxEntIRL":
        self.scaler = StandardScaler().fit(states)
        self.model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=7)
        self.model.fit(self.scaler.transform(states), actions)
        return self

    def probabilities(self, states: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(self.scaler.transform(states))
        result = np.zeros((len(states), len(ACTIONS)))
        result[:, self.model.classes_.astype(int)] = raw
        return result

    @property
    def reward_weights(self) -> np.ndarray:
        result = np.zeros((len(ACTIONS), self.model.coef_.shape[1]))
        result[self.model.classes_.astype(int)] = self.model.coef_
        return result


class EcommercePipeline:
    def __init__(self, segments: int = 4, seed: int = 7):
        self.segments = segments
        self.seed = seed

    def fit(
        self,
        panel: pd.DataFrame,
        discovery_method: str = "notears-mlp",
        max_iter: int = 10,
        max_rows: int = 5000,
    ) -> "EcommercePipeline":
        self.panel = add_lags(panel)
        user_aggregates = panel.groupby("user_id")[list(STATE_FEATURES)].mean()
        self.user_scaler = StandardScaler().fit(user_aggregates)
        scaled_users = self.user_scaler.transform(user_aggregates)
        components = min(self.segments, len(user_aggregates))
        self.gmm = GaussianMixture(components, covariance_type="full", random_state=self.seed).fit(scaled_users)
        responsibilities = self.gmm.predict_proba(scaled_users)
        self.user_ids = user_aggregates.index.to_numpy()
        self.user_state = user_aggregates
        self.user_responsibilities = responsibilities

        resp = pd.DataFrame(
            responsibilities,
            index=user_aggregates.index,
            columns=[f"segment_p{i}" for i in range(components)],
        )
        joined = self.panel.join(resp, on="user_id", rsuffix="_segment")
        state_columns = [f"{name}_lag1" for name in STATE_FEATURES] + list(resp.columns)
        self.state_columns = state_columns
        self.irl = ContextualMaxEntIRL().fit(joined[state_columns].to_numpy(), joined.action.to_numpy())
        self.discovery = discover_dag(
            self.panel,
            method=discovery_method,
            max_rows=max_rows,
            max_iter=max_iter,
            seed=self.seed,
        )

        self.transition_models = {}
        self.residual_sd = {}
        transition_inputs = state_columns + [f"action_{name}" for name in ACTIONS[1:]]
        self.transition_inputs = transition_inputs
        transition_values = joined[transition_inputs].to_numpy()
        for outcome in OUTCOMES:
            model = Ridge(alpha=1.0).fit(transition_values, joined[outcome].to_numpy())
            residual = joined[outcome].to_numpy() - model.predict(transition_values)
            self.transition_models[outcome] = model
            self.residual_sd[outcome] = float(np.std(residual))
        return self

    def user_features(self) -> pd.DataFrame:
        columns = {f"segment_p{i}": self.user_responsibilities[:, i] for i in range(self.user_responsibilities.shape[1])}
        result = pd.DataFrame({"user_id": self.user_ids, **columns})
        result["segment"] = self.user_responsibilities.argmax(axis=1)
        reward = self.irl.reward_weights
        for action, weights in zip(ACTIONS, reward):
            result[f"irl_reward_{action}"] = self.user_responsibilities @ weights[-self.user_responsibilities.shape[1]:]
        return result

    def generate(self, n_users: int = 100, periods: int = 14) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        sampled = rng.choice(len(self.user_state), size=n_users, replace=True)
        previous = self.user_state.iloc[sampled].to_numpy().clip(min=0)
        responsibilities = self.user_responsibilities[sampled]
        rows = []
        for day in range(periods):
            states = np.column_stack([previous, responsibilities])
            action_probabilities = self.irl.probabilities(states)
            actions = np.array([rng.choice(len(ACTIONS), p=p / p.sum()) for p in action_probabilities])
            action_matrix = np.column_stack([actions == i for i in range(1, len(ACTIONS))]).astype(int)
            model_input = np.column_stack([states, action_matrix])
            current = np.zeros_like(previous)
            for j, outcome in enumerate(OUTCOMES):
                prediction = self.transition_models[outcome].predict(model_input)
                noise = rng.normal(0, self.residual_sd[outcome], n_users)
                current[:, j] = np.maximum(0, prediction + noise)
            current[:, 0] = np.rint(current[:, 0])
            current[:, 2] = np.rint(current[:, 2])
            for user in range(n_users):
                rows.append({
                    "user_id": f"synthetic_{user:05d}",
                    "day": day,
                    "action": int(actions[user]),
                    "action_name": ACTIONS[actions[user]],
                    **{name: float(current[user, j]) for j, name in enumerate(OUTCOMES)},
                    "segment": int(responsibilities[user].argmax()),
                })
            previous = current
        return pd.DataFrame(rows)

    def save(self, output: str | Path, synthetic_users: int = 100, periods: int = 14) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.user_features().to_csv(output / "user_features.csv", index=False)
        self.generate(synthetic_users, periods).to_csv(output / "synthetic_panel.csv", index=False)
        (output / "causal_dag.json").write_text(json.dumps(self.discovery.to_dict(), indent=2) + "\n")
        summary = {
            "observed_rows": len(self.panel),
            "observed_users": int(self.panel.user_id.nunique()),
            "gmm_segments": self.gmm.n_components,
            "dag_edges": len(self.discovery.edges),
            "irl_type": "one-step contextual maximum-entropy",
            "identification_warning": "Starbucks offer assignment is treated as a logged action, not a proven randomized intervention.",
            "data_use_warning": "The Kaggle mirror has no clear reuse license; use this run for development, not KSC headline results.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

