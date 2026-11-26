"""세 learner의 공통 인터페이스와 기본 base model."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

_GBM = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, random_state=0)


def make_classifier(**overrides):
    """4주 내 구매 확률 P(Y=1 | X) 모델."""
    return HistGradientBoostingClassifier(**{**_GBM, **overrides})


def make_regressor(**overrides):
    """가상 효과(pseudo-outcome) 회귀 모델."""
    return HistGradientBoostingRegressor(**{**_GBM, **overrides})


def as_array(X) -> np.ndarray:
    return np.asarray(X, dtype=float)


def fit_base_model(model, X, target):
    """결측 지표와 대체값은 해당 군의 학습 행으로만 결정한다."""
    if isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        # 소표본에서는 내부 검증 분할 뒤 특정 KIT 열의 관측값이 하나도 없을 수 있다.
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True,
                                            add_indicator=True), model)
    return model.fit(X, target)


def fit_outcome_models(X, t, y, make_model=make_classifier):
    """대조군·처치군의 결과 모델 mu0, mu1을 각각 학습한다."""
    models = []
    for arm in (0, 1):
        selected = t == arm
        outcomes, counts = np.unique(y[selected], return_counts=True)
        if len(outcomes) == 0:
            raise ValueError("학습에 대조군과 처치군 고객이 모두 필요하다")
        model = make_model()
        if len(outcomes) == 1:
            model = DummyClassifier(strategy="prior")
        elif isinstance(model, HistGradientBoostingClassifier) and (counts.min() < 2 or counts.sum() < 11):
            # sklearn의 내부 층화 검증 분할을 만들 수 없는 소표본.
            model.set_params(early_stopping=False)
        models.append(fit_base_model(model, X[selected], y[selected]))
    return tuple(models)


def purchase_probability(model, X):
    """처치군에 구매자 또는 미구매자만 있는 소표본도 처리한다."""
    if isinstance(model, DummyClassifier) and len(model.classes_) == 1:
        return np.full(len(X), float(model.classes_[0]))
    return model.predict_proba(X)[:, 1]


class BaseUpliftLearner(ABC):
    """fit(X, t, y) 후 predict(X)로 p0, p1, uplift_score를 낸다.

    p0: 미발송 시 4주 구매 확률, p1: 발송 시 4주 구매 확률, uplift_score = p1 - p0.
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
