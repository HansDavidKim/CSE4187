from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"V{i}" for i in range(1, 8)]
SCHEMA = ["ID", "Promotion", "purchase", *FEATURES]


def load_uplift_data(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    train, test = pd.read_csv(root / "training.csv"), pd.read_csv(root / "Test.csv")
    for name, frame, rows in (("training.csv", train, 84_534), ("Test.csv", test, 41_650)):
        if list(frame.columns) != SCHEMA or len(frame) != rows:
            raise ValueError(f"unexpected {name} schema or row count")
        if not set(frame.Promotion.unique()) <= {"Yes", "No"} or not set(frame.purchase.unique()) <= {0, 1}:
            raise ValueError(f"invalid treatment/outcome values in {name}")
    return train, test


def _design(frame: pd.DataFrame, treatment: np.ndarray | None = None) -> np.ndarray:
    x = frame[FEATURES].to_numpy(dtype=float)
    a = frame.Promotion.eq("Yes").to_numpy(dtype=float) if treatment is None else treatment.astype(float)
    return np.column_stack([x, a, x * a[:, None]])


def _smd(frame: pd.DataFrame) -> dict[str, float]:
    treated = frame.Promotion.eq("Yes")
    result = {}
    for column in FEATURES:
        pooled = np.sqrt((frame.loc[treated, column].var() + frame.loc[~treated, column].var()) / 2)
        result[column] = float((frame.loc[treated, column].mean() - frame.loc[~treated, column].mean()) / pooled)
    return result


class StarbucksUpliftSimulator:
    """Calibrated one-step contextual-bandit simulator for promotion response."""

    def __init__(self, segments: int = 6, seed: int = 7):
        self.segments = segments
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "StarbucksUpliftSimulator":
        self.train = train.copy()
        x = train[FEATURES].to_numpy(dtype=float)
        self.feature_support = {name: np.sort(train[name].unique()) for name in FEATURES}
        self.feature_scaler = StandardScaler().fit(x)
        self.gmm = GaussianMixture(
            min(self.segments, len(train)), covariance_type="full", reg_covar=1e-5, random_state=self.seed
        ).fit(self.feature_scaler.transform(x))
        self.response_model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)
        ).fit(_design(train), train.purchase.to_numpy())
        return self

    def potential_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        zeros, ones = np.zeros(len(frame)), np.ones(len(frame))
        p0 = self.response_model.predict_proba(_design(frame, zeros))[:, 1]
        p1 = self.response_model.predict_proba(_design(frame, ones))[:, 1]
        return p0, p1

    def evaluate(self, test: pd.DataFrame, policy_fraction: float = 0.30) -> dict:
        a = test.Promotion.eq("Yes").to_numpy(dtype=int)
        y = test.purchase.to_numpy(dtype=float)
        propensity = float(self.train.Promotion.eq("Yes").mean())
        p0, p1 = self.potential_probabilities(test)
        uplift = p1 - p0
        cutoff = np.quantile(uplift, 1 - policy_fraction)
        policy = (uplift >= cutoff).astype(int)
        observed_ate = float(y[a == 1].mean() - y[a == 0].mean())
        ipw_ate = float(np.mean(a * y / propensity - (1 - a) * y / (1 - propensity)))

        observed_model = np.where(a == 1, p1, p0)
        policy_model = np.where(policy == 1, p1, p0)
        policy_propensity = np.where(policy == 1, propensity, 1 - propensity)
        dr_policy = float(np.mean(policy_model + (a == policy) / policy_propensity * (y - observed_model)))
        control_value = float(np.mean(p0 + (a == 0) / (1 - propensity) * (y - p0)))
        balance = _smd(self.train)
        return {
            "train_rows": len(self.train),
            "test_rows": len(test),
            "promotion_rate_train": propensity,
            "purchase_rate_treatment_test": float(y[a == 1].mean()),
            "purchase_rate_control_test": float(y[a == 0].mean()),
            "observed_difference_test": observed_ate,
            "ipw_ate_test": ipw_ate,
            "model_mean_uplift_test": float(uplift.mean()),
            "target_fraction": float(policy.mean()),
            "dr_target_policy_value": dr_policy,
            "dr_all_control_value": control_value,
            "dr_incremental_policy_value": dr_policy - control_value,
            "standardized_mean_differences_train": balance,
            "max_abs_smd_train": max(map(abs, balance.values())),
            "assignment_note": "Promotion assignment mechanism is not documented as randomized in the checked Udacity notebook; inspect balance and avoid an RCT claim without an additional source.",
        }

    def representations(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[FEATURES].to_numpy(dtype=float)
        posterior = self.gmm.predict_proba(self.feature_scaler.transform(x))
        p0, p1 = self.potential_probabilities(frame)
        result = pd.DataFrame({
            "row_id": np.arange(len(frame)),
            "p_purchase_control": p0,
            "p_purchase_promotion": p1,
            "uplift": p1 - p0,
        })
        for i in range(posterior.shape[1]):
            result[f"segment_p{i}"] = posterior[:, i]
        result["segment"] = posterior.argmax(axis=1)
        return result

    def generate(self, n: int = 100_000, policy: str = "random", target_fraction: float = 0.30) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        scaled, segments = self.gmm.sample(n)
        x = self.feature_scaler.inverse_transform(scaled)
        frame = pd.DataFrame(x, columns=FEATURES)
        for name, support in self.feature_support.items():
            if len(support) <= 100:
                values = frame[name].to_numpy()
                nearest = np.abs(values[:, None] - support[None, :]).argmin(axis=1)
                frame[name] = support[nearest]
            else:
                frame[name] = frame[name].clip(support.min(), support.max())
        p0, p1 = self.potential_probabilities(frame)
        uplift = p1 - p0
        if policy == "random":
            action = rng.binomial(1, self.train.Promotion.eq("Yes").mean(), n)
        elif policy == "targeted":
            action = (uplift >= np.quantile(uplift, 1 - target_fraction)).astype(int)
        else:
            raise ValueError("policy must be random or targeted")
        probability = np.where(action == 1, p1, p0)
        frame.insert(0, "ID", [f"synthetic_{i:07d}" for i in range(n)])
        frame.insert(1, "Promotion", np.where(action == 1, "Yes", "No"))
        frame.insert(2, "purchase", rng.binomial(1, probability))
        frame["purchase_probability"] = probability
        frame["individual_uplift"] = uplift
        frame["segment"] = segments
        return frame

    def save(self, output: str | Path, test: pd.DataFrame, synthetic_rows: int = 100_000) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        metrics = self.evaluate(test)
        (output / "uplift_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        self.representations(test).to_csv(output / "test_user_representations.csv", index=False)
        self.generate(synthetic_rows, policy="random").to_csv(output / "synthetic_random.csv", index=False)
        self.generate(synthetic_rows, policy="targeted").to_csv(output / "synthetic_targeted.csv", index=False)
        provenance = {
            "data_role": "one-step promotion-response calibration",
            "model": "logistic S-learner with treatment interactions + Gaussian mixture covariates",
            "not_claimed": ["multi-step dynamics", "item recommendation", "documented random assignment"],
            "raw_data_redistributed": False,
            "derived_row_level_outputs": "local-only; do not redistribute without a separate rights review",
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

