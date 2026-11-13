"""세 learner의 공통 인터페이스와 기본 base model."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# 결측(NaN)을 그대로 처리하는 트리 모델을 쓴다. 대체값·스케일러가 필요 없다.
_GBM = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, random_state=0)


def make_classifier(**overrides):
    """유지 확률 P(Y=1 | X) 모델."""
    return HistGradientBoostingClassifier(**{**_GBM, **overrides})


def make_regressor(**overrides):
    """가상 효과(pseudo-outcome) 회귀 모델."""
    return HistGradientBoostingRegressor(**{**_GBM, **overrides})


def as_array(X) -> np.ndarray:
    return np.asarray(X, dtype=float)


def fit_outcome_models(X, t, y, make_model=make_classifier):
    """대조군·처치군의 결과 모델 mu0, mu1을 각각 학습한다."""
    return make_model().fit(X[t == 0], y[t == 0]), make_model().fit(X[t == 1], y[t == 1])


class BaseUpliftLearner(ABC):
    """fit(X, t, y) 후 predict(X)로 p0, p1, uplift_score를 낸다.

    p0: 미발송 시 유지 확률, p1: 발송 시 유지 확률, uplift_score = p1 - p0.
    """

    name: str

    @abstractmethod
    def fit(self, X, t, y, propensity: float = 0.5) -> "BaseUpliftLearner":
        """X: 처치 전 feature, t: 배정군(0/1), y: 결과(0/1), propensity: 배정 확률."""

    @abstractmethod
    def predict(self, X) -> pd.DataFrame:
        """컬럼 p0, p1, uplift_score."""

    def predict_uplift(self, X) -> np.ndarray:
        return self.predict(X)["uplift_score"].to_numpy()

    @staticmethod
    def _frame(p0, p1, X) -> pd.DataFrame:
        p0, p1 = np.clip(p0, 0, 1), np.clip(p1, 0, 1)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame({"p0": p0, "p1": p1, "uplift_score": p1 - p0}, index=index)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "BaseUpliftLearner":
        with open(path, "rb") as f:
            return pickle.load(f)
