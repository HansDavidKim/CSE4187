from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class CalibratedMultinomialLogit:
    """Temperature-scaled MNL with fixed utility scale for identification.

    The action columns may include a no-purchase/outside option. Fixing the
    centered utility standard deviation to one separates utility scale from
    temperature; bounded likelihood calibration prevents degenerate extremes.
    """

    min_temperature: float = 0.1
    max_temperature: float = 5.0
    temperature_: float | None = None
    utility_scale_: float | None = None

    def fit(self, utilities: np.ndarray, choices: np.ndarray) -> "CalibratedMultinomialLogit":
        utilities = np.asarray(utilities, dtype=float)
        choices = np.asarray(choices, dtype=int)
        if utilities.ndim != 2 or len(utilities) != len(choices):
            raise ValueError("utilities must be [rows, actions] and align with choices")
        if np.any(choices < 0) or np.any(choices >= utilities.shape[1]):
            raise ValueError("choice index outside the action set")
        centered = utilities - utilities.mean(axis=1, keepdims=True)
        scale = float(centered.std())
        self.utility_scale_ = scale if scale > 1e-12 else 1.0
        normalized = centered / self.utility_scale_

        def negative_log_likelihood(temperature: float) -> float:
            probability = _softmax(normalized / temperature)
            selected = probability[np.arange(len(choices)), choices]
            return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(self.min_temperature, self.max_temperature),
            method="bounded",
        )
        self.temperature_ = float(result.x)
        return self

    def probabilities(self, utilities: np.ndarray) -> np.ndarray:
        if self.temperature_ is None or self.utility_scale_ is None:
            raise RuntimeError("fit must be called before probabilities")
        utilities = np.asarray(utilities, dtype=float)
        centered = utilities - utilities.mean(axis=1, keepdims=True)
        return _softmax(centered / self.utility_scale_ / self.temperature_)

    def diagnostics(self, utilities: np.ndarray) -> dict[str, float]:
        probability = self.probabilities(utilities)
        action_count = probability.shape[1]
        entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)
        return {
            "temperature": float(self.temperature_),
            "utility_scale": float(self.utility_scale_),
            "mean_normalized_entropy": float(entropy.mean() / np.log(action_count)),
            "mean_top1_probability": float(probability.max(axis=1).mean()),
        }

    def sample(self, utilities: np.ndarray, seed: int = 2026) -> np.ndarray:
        probability = self.probabilities(utilities)
        rng = np.random.default_rng(seed)
        return np.asarray([rng.choice(probability.shape[1], p=row) for row in probability])
