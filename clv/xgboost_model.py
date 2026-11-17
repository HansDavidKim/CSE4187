"""WBS 4.5: 13주 CLV 예측 변수로 학습하는 XGBoost 회귀 모델.

검증 분할 그리드 탐색(조기 종료)으로 설정을 고르고, seed·스레드 수를 고정해 같은 입력이면 같은 모델이 나온다.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .schema import PREDICTOR_FEATURES, check_predictors

NAME = "xgboost"
FEATURE_DTYPE = "float64"


def xgboost_status() -> tuple[bool, str]:
    """(사용 가능 여부, 버전 또는 import 오류 메시지)를 반환한다."""
    try:
        import xgboost
    except ImportError as error:
        return False, f"xgboost is not installed: {error}"
    return True, xgboost.__version__


def _errors(actual, predicted) -> dict:
    """예측 오차의 RMSE·MAE를 dict로 반환한다."""
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}


class XGBoostCLV:
    """검증 분할로 설정을 고르고 예측 변수 계약(이름·순서·자료형)을 고정한 XGBoost 회귀 모델."""

    name = NAME

    def __init__(self, seed: int = 42, grid: dict | None = None, fixed: dict | None = None,
                 selection_metric: str = "rmse", feature_names=PREDICTOR_FEATURES):
        from .config import DEFAULT_XGB_FIXED, DEFAULT_XGB_GRID

        self.feature_names = list(feature_names)
        # 고객 ID·정답·t0 이후 기록 등 금지 열이 예측 변수에 섞이지 않았는지 확인한다.
        check_predictors(self.feature_names)
        self.seed, self.selection_metric = int(seed), selection_metric
        self.grid = {key: list(value) for key, value in (grid or DEFAULT_XGB_GRID).items()}
        self.fixed = dict(fixed or DEFAULT_XGB_FIXED)
        self.model = None
        self.best_params: dict = {}
        self.grid_results: list[dict] = []
        self.last_clipped = 0

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """예측 변수를 저장된 순서와 자료형(float64)으로 뽑는다. 빠진 열이 있으면 오류를 낸다."""
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"XGBoost input is missing features: {missing}")
        return frame.loc[:, self.feature_names].astype(FEATURE_DTYPE)

    def _regressor(self, params: dict, n_estimators: int, early_stopping: int | None):
        """고정 seed로 ``XGBRegressor``를 만든다. 트리 수와 조기 종료는 ``params``가 아니라 인자로 받는다."""
        import xgboost as xgb

        settings = {key: value for key, value in params.items() if key not in ("n_estimators", "early_stopping_rounds")}
        return xgb.XGBRegressor(n_estimators=int(n_estimators), early_stopping_rounds=early_stopping,
                                eval_metric="rmse", random_state=self.seed, **settings)

    def fit(self, train: pd.DataFrame, y_train, validation: pd.DataFrame, y_validation) -> "XGBoostCLV":
        """검증 분할 ``selection_metric``으로 그리드 설정과 트리 수를 고르고, 학습 분할에서 다시 학습한다."""
        X_train, X_validation = self.matrix(train), self.matrix(validation)
        y_train, y_validation = np.asarray(y_train, dtype=float), np.asarray(y_validation, dtype=float)
        keys = sorted(self.grid)
        self.grid_results = []
        # 그리드 조합마다 학습 분할로 학습하고 검증 분할 조기 종료로 최적 트리 수를 찾는다.
        for values in itertools.product(*(self.grid[key] for key in keys)):
            params = {**self.fixed, **dict(zip(keys, values))}
            model = self._regressor(params, params["n_estimators"], params.get("early_stopping_rounds"))
            model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
            trees = int(model.best_iteration) + 1
            # 검증 점수도 predict와 같이 음수 예측을 0으로 자른 뒤 계산한다.
            predicted = np.clip(model.predict(X_validation, iteration_range=(0, trees)), 0, None)
            scores = _errors(y_validation, predicted)
            self.grid_results.append({**dict(zip(keys, values)), "n_estimators": trees,
                                      **{f"validation_{name}": value for name, value in scores.items()}})
        # 점수가 같으면 먼저 나온 조합을 고른다.
        best = min(range(len(self.grid_results)),
                   key=lambda index: (self.grid_results[index][f"validation_{self.selection_metric}"], index))
        chosen = self.grid_results[best]
        self.best_params = {**self.fixed, **{key: chosen[key] for key in keys}, "n_estimators": chosen["n_estimators"]}
        self.best_params.pop("early_stopping_rounds", None)
        # 최종 모델은 조기 종료 없이 고른 트리 수로 학습 분할에서만 다시 학습한다.
        self.model = self._regressor(self.best_params, chosen["n_estimators"], None)
        self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """예측 변수 계약에 맞춰 예측한다. CLV는 음수일 수 없어 음수 예측은 0으로 자르고 건수를 ``last_clipped``에 남긴다."""
        if self.model is None:
            raise RuntimeError("XGBoost model is not fitted")
        raw = self.model.predict(self.matrix(frame)).astype(float)
        self.last_clipped = int((raw < 0).sum())
        return np.clip(raw, 0.0, None)

    def save(self, directory: str | Path) -> Path:
        """booster(``xgboost_model.json``)와 메타데이터(``xgboost_meta.json``: 예측 변수 계약, 그리드 결과)를 저장한다."""
        import xgboost

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(directory / "xgboost_model.json")
        return write_json(directory / "xgboost_meta.json", {
            "model": NAME, "feature_names": self.feature_names,
            "feature_dtypes": {name: FEATURE_DTYPE for name in self.feature_names},
            "seed": self.seed, "selection_metric": self.selection_metric, "grid": self.grid, "fixed": self.fixed,
            "best_params": self.best_params, "grid_results": self.grid_results,
            "negative_prediction_policy": "clip_at_zero", "xgboost_version": xgboost.__version__,
        })

    @classmethod
    def load(cls, directory: str | Path) -> "XGBoostCLV":
        """저장된 모델을 불러오고 booster의 예측 변수 순서가 메타데이터와 같은지 확인한다."""
        import xgboost as xgb

        directory = Path(directory)
        meta = read_json(directory / "xgboost_meta.json")
        if meta.get("model") != NAME:
            raise ValueError("not an XGBoost CLV model directory")
        model = cls(meta["seed"], meta["grid"], meta["fixed"], meta["selection_metric"], meta["feature_names"])
        model.model = xgb.XGBRegressor()
        model.model.load_model(directory / "xgboost_model.json")
        stored = list(model.model.get_booster().feature_names or [])
        if stored != model.feature_names:
            raise ValueError("saved booster feature order differs from its metadata")
        model.best_params, model.grid_results = meta["best_params"], meta["grid_results"]
        return model
