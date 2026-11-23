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


TWO_PART_NAME = "xgboost_two_part"
# 두 파트 모델의 최소 학습 인원. 구매 고객이 너무 적으면 금액 파트를 학습하지 않는다.
MIN_TWO_PART_BUYERS = 20
PROBABILITY_EPSILON = 1e-15


def _logloss(actual, probability) -> float:
    """이진 로그 손실. 확률을 (ε, 1 - ε)로 잘라 log(0)을 막는다."""
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def classification_metrics(actual, probability) -> dict:
    """구매 여부 예측의 AUC(동점은 평균 순위), 로그 손실, Brier 점수, 실제 구매율."""
    y = np.asarray(actual, dtype=bool)
    p = np.asarray(probability, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    auc = None
    if positives and negatives:
        # Mann-Whitney U로 계산한 AUC: 무작위 구매 고객의 확률이 무작위 비구매 고객보다 높을 확률
        order = pd.Series(p).rank(method="average").to_numpy()
        auc = float((order[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {"n": int(len(y)), "buyer_share": float(y.mean()) if len(y) else None, "auc": auc,
            "logloss": _logloss(y, p), "brier": float(np.mean((p - y) ** 2))}


def _search(make, grid: dict, fixed: dict, data: tuple, score, metric: str):
    """그리드 조합마다 검증 분할 조기 종료로 트리 수를 찾고, ``metric``이 가장 낮은 조합으로 다시 학습한다.

    Returns:
        (최종 모델, 고른 설정, 그리드 결과 목록). 최종 모델은 조기 종료 없이 학습 분할에서만 학습한다.
    """
    X_train, y_train, X_validation, y_validation = data
    keys = sorted(grid)
    results = []
    for values in itertools.product(*(grid[key] for key in keys)):
        params = {**fixed, **dict(zip(keys, values))}
        model = make(params, params["n_estimators"], params.get("early_stopping_rounds"))
        model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
        trees = int(model.best_iteration) + 1
        results.append({**dict(zip(keys, values)), "n_estimators": trees,
                        **{f"validation_{name}": value for name, value in score(model, trees).items()}})
    # 점수가 같으면 먼저 나온 조합을 고른다.
    best = min(range(len(results)), key=lambda index: (results[index][f"validation_{metric}"], index))
    chosen = results[best]
    params = {**fixed, **{key: chosen[key] for key in keys}, "n_estimators": chosen["n_estimators"]}
    params.pop("early_stopping_rounds", None)
    final = make(params, chosen["n_estimators"], None)
    final.fit(X_train, y_train, verbose=False)
    return final, params, results


class TwoPartXGBoostCLV:
    """두 파트(hurdle) XGBoost: CLV = P(52주 안에 구매) × E[52주 결제액 | 구매].

    1파트는 모든 학습 고객으로 구매 여부(``actual_clv_12m > 0``)를 분류하고, 2파트는 구매한 학습 고객만으로
    결제액을 회귀한다. 두 파트는 각자 검증 분할(1파트 로그 손실, 2파트 구매 고객 RMSE)로 설정을 고른다.
    예측 변수 계약(이름·순서·float64)은 ``XGBoostCLV``와 같다.
    """

    name = TWO_PART_NAME

    def __init__(self, seed: int = 42, classifier_grid: dict | None = None, regressor_grid: dict | None = None,
                 fixed: dict | None = None, feature_names=PREDICTOR_FEATURES):
        from .config import DEFAULT_TWO_PART_CLASSIFIER_GRID, DEFAULT_TWO_PART_REGRESSOR_GRID, DEFAULT_XGB_FIXED

        self.feature_names = list(feature_names)
        check_predictors(self.feature_names)
        self.seed = int(seed)
        self.classifier_grid = {key: list(value) for key, value in
                                (classifier_grid or DEFAULT_TWO_PART_CLASSIFIER_GRID).items()}
        self.regressor_grid = {key: list(value) for key, value in
                               (regressor_grid or DEFAULT_TWO_PART_REGRESSOR_GRID).items()}
        self.fixed = dict(fixed or DEFAULT_XGB_FIXED)
        self.classifier = self.regressor = None
        self.best_params: dict = {}
        self.grid_results: dict = {}
        self.train_counts: dict = {}
        self.last_clipped = 0

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """예측 변수를 저장된 순서와 자료형(float64)으로 뽑는다. 빠진 열이 있으면 오류를 낸다."""
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"XGBoost input is missing features: {missing}")
        return frame.loc[:, self.feature_names].astype(FEATURE_DTYPE)

    def _make(self, kind: str):
        """고정 seed로 분류기(``classifier``) 또는 회귀기(``regressor``)를 만드는 함수를 반환한다."""
        import xgboost as xgb

        def make(params: dict, n_estimators: int, early_stopping: int | None):
            settings = {key: value for key, value in params.items()
                        if key not in ("n_estimators", "early_stopping_rounds")}
            if kind == "classifier":
                settings.pop("objective", None)
                return xgb.XGBClassifier(n_estimators=int(n_estimators), early_stopping_rounds=early_stopping,
                                         objective="binary:logistic", eval_metric="logloss",
                                         random_state=self.seed, **settings)
            return xgb.XGBRegressor(n_estimators=int(n_estimators), early_stopping_rounds=early_stopping,
                                    eval_metric="rmse", random_state=self.seed, **settings)
        return make

    def fit(self, train: pd.DataFrame, y_train, validation: pd.DataFrame, y_validation) -> "TwoPartXGBoostCLV":
        """1파트(구매 분류)와 2파트(구매 고객 결제액 회귀)를 각각 검증 분할로 고르고 학습한다."""
        X_train, X_validation = self.matrix(train), self.matrix(validation)
        y_train, y_validation = np.asarray(y_train, dtype=float), np.asarray(y_validation, dtype=float)
        bought_train, bought_validation = y_train > 0, y_validation > 0
        self.train_counts = {"train_customers": int(len(y_train)), "train_buyers": int(bought_train.sum()),
                             "validation_customers": int(len(y_validation)),
                             "validation_buyers": int(bought_validation.sum())}
        if bought_train.sum() < MIN_TWO_PART_BUYERS or bought_train.all():
            raise ValueError(f"two-part XGBoost needs at least {MIN_TWO_PART_BUYERS} buyers and some non-buyers in train")
        if bought_validation.sum() < 2 or bought_validation.all():
            raise ValueError("two-part XGBoost needs buyers and non-buyers in validation")

        def classifier_score(model, trees):
            probability = model.predict_proba(X_validation, iteration_range=(0, trees))[:, 1]
            return classification_metrics(bought_validation, probability)

        self.classifier, classifier_params, classifier_results = _search(
            self._make("classifier"), self.classifier_grid, self.fixed,
            (X_train, bought_train.astype(int), X_validation, bought_validation.astype(int)), classifier_score, "logloss")

        positive_validation = X_validation.loc[bought_validation]

        def regressor_score(model, trees):
            predicted = np.clip(model.predict(positive_validation, iteration_range=(0, trees)), 0, None)
            return _errors(y_validation[bought_validation], predicted)

        self.regressor, regressor_params, regressor_results = _search(
            self._make("regressor"), self.regressor_grid, self.fixed,
            (X_train.loc[bought_train], y_train[bought_train], positive_validation, y_validation[bought_validation]),
            regressor_score, "rmse")
        self.best_params = {"classifier": classifier_params, "regressor": regressor_params}
        self.grid_results = {"classifier": classifier_results, "regressor": regressor_results}
        return self

    def predict_parts(self, frame: pd.DataFrame) -> pd.DataFrame:
        """고객별 구매 확률, 구매 시 기대 결제액, 둘의 곱(``predicted_clv_12m``)을 반환한다."""
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("two-part XGBoost model is not fitted")
        X = self.matrix(frame)
        probability = self.classifier.predict_proba(X)[:, 1].astype(float)
        raw = self.regressor.predict(X).astype(float)
        # 결제액은 음수일 수 없어 음수 금액 예측은 0으로 자르고 건수를 남긴다.
        self.last_clipped = int((raw < 0).sum())
        amount = np.clip(raw, 0.0, None)
        return pd.DataFrame({"purchase_probability_52w": probability, "expected_amount_if_purchase": amount,
                             "predicted_clv_12m": probability * amount}, index=frame.index)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """52주 CLV = 구매 확률 × 구매 시 기대 결제액."""
        return self.predict_parts(frame)["predicted_clv_12m"].to_numpy(dtype=float)

    def summary(self) -> dict:
        """registry에 남길 설정·그리드 결과·학습 인원."""
        return {"best_params": self.best_params, "grid_results": self.grid_results, **self.train_counts,
                "clipped_negative_amounts": self.last_clipped}

    def save(self, directory: str | Path) -> Path:
        """``classifier.json``, ``regressor.json``과 메타데이터(``two_part_meta.json``)를 저장한다."""
        import xgboost

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(directory / "classifier.json")
        self.regressor.save_model(directory / "regressor.json")
        return write_json(directory / "two_part_meta.json", {
            "model": TWO_PART_NAME, "feature_names": self.feature_names,
            "feature_dtypes": {name: FEATURE_DTYPE for name in self.feature_names}, "seed": self.seed,
            "classifier_grid": self.classifier_grid, "regressor_grid": self.regressor_grid, "fixed": self.fixed,
            "best_params": self.best_params, "grid_results": self.grid_results, "train_counts": self.train_counts,
            "combination": "purchase_probability_52w * expected_amount_if_purchase",
            "negative_prediction_policy": "clip_amount_at_zero", "xgboost_version": xgboost.__version__,
        })

    @classmethod
    def load(cls, directory: str | Path) -> "TwoPartXGBoostCLV":
        """저장된 두 파트를 불러오고 각 booster의 예측 변수 순서가 메타데이터와 같은지 확인한다."""
        import xgboost as xgb

        directory = Path(directory)
        meta = read_json(directory / "two_part_meta.json")
        if meta.get("model") != TWO_PART_NAME:
            raise ValueError("not a two-part XGBoost CLV model directory")
        model = cls(meta["seed"], meta["classifier_grid"], meta["regressor_grid"], meta["fixed"], meta["feature_names"])
        model.classifier, model.regressor = xgb.XGBClassifier(), xgb.XGBRegressor()
        model.classifier.load_model(directory / "classifier.json")
        model.regressor.load_model(directory / "regressor.json")
        for part in (model.classifier, model.regressor):
            if list(part.get_booster().feature_names or []) != model.feature_names:
                raise ValueError("saved booster feature order differs from its metadata")
        model.best_params, model.grid_results = meta["best_params"], meta["grid_results"]
        model.train_counts = meta.get("train_counts", {})
        return model


GROSS_NAME = "xgboost_gross_4w"
# 4주 금액 모델의 최소 학습 구매 고객 수(Train). 검증 분할에도 구매 고객이 2명 이상 있어야 한다.
MIN_GROSS_BUYERS = 20


class InsufficientBuyersError(ValueError):
    """4주 모델을 학습할 구매 고객이 부족하다. 학습은 이 경우만 ``failed``로 기록하고 나머지 오류는 그대로 낸다."""


class GrossXGBoost4w:
    """4주 조건부 할인 전 총구매액 XGBoost: E[``[t0, t0 + 4주)`` 총구매액 | 4주 안에 1회 이상 구매, X(t0)].

    52주 모델과 독립적으로 학습·저장·평가한다. 4주 안에 구매한 학습 고객(학습 모집단 정책을 거친 Control
    고객)의 할인 전 총구매액(``actual_gross_4w_if_purchase``)만 회귀하며, 비구매 고객을 0 정답으로 쓰지 않는다.
    예측 변수 계약(13주 이력, 이름·순서·float64)은 52주 XGBoost와 같다.
    """

    name = GROSS_NAME

    def __init__(self, seed: int = 42, grid: dict | None = None, fixed: dict | None = None,
                 feature_names=PREDICTOR_FEATURES):
        from .config import DEFAULT_GROSS_4W_GRID, DEFAULT_XGB_FIXED

        self.feature_names = list(feature_names)
        check_predictors(self.feature_names)
        self.seed = int(seed)
        self.grid = {key: list(value) for key, value in (grid or DEFAULT_GROSS_4W_GRID).items()}
        self.fixed = dict(fixed or DEFAULT_XGB_FIXED)
        self.model = None
        self.best_params: dict = {}
        self.grid_results: list[dict] = []
        self.train_counts: dict = {}
        self.skipped_objectives: list[str] = []
        self.last_clipped = 0

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """예측 변수를 저장된 순서와 자료형(float64)으로 뽑는다. 빠진 열이 있으면 오류를 낸다."""
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"XGBoost input is missing features: {missing}")
        return frame.loc[:, self.feature_names].astype(FEATURE_DTYPE)

    def _make(self, params: dict, n_estimators: int, early_stopping: int | None):
        """고정 seed로 ``XGBRegressor``를 만든다(``_search``가 부른다)."""
        import xgboost as xgb

        settings = {key: value for key, value in params.items() if key not in ("n_estimators", "early_stopping_rounds")}
        return xgb.XGBRegressor(n_estimators=int(n_estimators), early_stopping_rounds=early_stopping,
                                eval_metric="rmse", random_state=self.seed, **settings)

    def fit(self, train: pd.DataFrame, y_train, validation: pd.DataFrame, y_validation) -> "GrossXGBoost4w":
        """구매 고객의 4주 총구매액으로 학습하고 검증 분할 RMSE로 설정과 트리 수를 고른다.

        행은 호출자가 구매 고객(4주 구매 1회 이상)으로 고른다. 정답은 유한한 0 이상이어야 하며(결측·음수면
        ValueError), 금액이 0인 구매 고객도 정답으로 쓴다. 이때 0을 다룰 수 없는 ``reg:gamma`` 후보는 그리드에서
        빼고 ``skipped_objectives``에 남긴다.
        """
        X_train, X_validation = self.matrix(train), self.matrix(validation)
        y_train, y_validation = np.asarray(y_train, dtype=float), np.asarray(y_validation, dtype=float)
        for name, values in (("train", y_train), ("validation", y_validation)):
            if not (np.isfinite(values).all() and (values >= 0).all()):
                raise ValueError(f"{GROSS_NAME} {name} labels must be finite nonnegative gross amounts of buyers")
        self.train_counts = {"train_buyers": int(len(y_train)), "validation_buyers": int(len(y_validation)),
                             "zero_gross_buyers": int((y_train == 0).sum() + (y_validation == 0).sum())}
        if len(y_train) < MIN_GROSS_BUYERS or len(y_validation) < 2:
            raise InsufficientBuyersError(f"{GROSS_NAME} needs at least {MIN_GROSS_BUYERS} train buyers and 2 validation buyers "
                             f"(got {len(y_train)} and {len(y_validation)})")

        def score(model, trees):
            predicted = np.clip(model.predict(X_validation, iteration_range=(0, trees)), 0, None)
            return _errors(y_validation, predicted)

        grid = dict(self.grid)
        self.skipped_objectives = []
        if (y_train == 0).any() or (y_validation == 0).any():
            # gamma 회귀는 정답이 양수여야 하므로 0 금액 구매 고객이 있으면 후보에서 뺀다.
            objectives = list(grid.get("objective", ["reg:squarederror"]))
            self.skipped_objectives = [value for value in objectives if value == "reg:gamma"]
            grid["objective"] = [value for value in objectives if value != "reg:gamma"]
            if not grid["objective"]:
                raise ValueError(f"{GROSS_NAME} grid has only reg:gamma but some buyers have zero gross")
        self.model, self.best_params, self.grid_results = _search(
            self._make, grid, self.fixed, (X_train, y_train, X_validation, y_validation), score, "rmse")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """구매한다면 기대되는 4주 할인 전 총구매액. 음수 예측은 0으로 자르고 건수를 ``last_clipped``에 남긴다."""
        if self.model is None:
            raise RuntimeError(f"{GROSS_NAME} model is not fitted")
        raw = self.model.predict(self.matrix(frame)).astype(float)
        self.last_clipped = int((raw < 0).sum())
        return np.clip(raw, 0.0, None)

    def summary(self) -> dict:
        """registry에 남길 설정·그리드 결과·학습 인원."""
        return {"best_params": self.best_params, "grid_results": self.grid_results,
                "skipped_objectives": self.skipped_objectives, **self.train_counts}

    def save(self, directory: str | Path) -> Path:
        """``regressor.json``과 메타데이터(``gross_4w_meta.json``)를 저장한다."""
        import xgboost

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(directory / "regressor.json")
        return write_json(directory / "gross_4w_meta.json", {
            "model": GROSS_NAME, "feature_names": self.feature_names,
            "feature_dtypes": {name: FEATURE_DTYPE for name in self.feature_names}, "seed": self.seed,
            "grid": self.grid, "fixed": self.fixed, "best_params": self.best_params,
            "grid_results": self.grid_results, "train_counts": self.train_counts,
            "skipped_objectives": self.skipped_objectives,
            "target": "gross amount over [t0, t0 + 4 weeks) given at least one purchase",
            "negative_prediction_policy": "clip_at_zero", "xgboost_version": xgboost.__version__,
        })

    @classmethod
    def load(cls, directory: str | Path) -> "GrossXGBoost4w":
        """저장된 모델을 불러오고 booster의 예측 변수 순서가 메타데이터와 같은지 확인한다."""
        import xgboost as xgb

        directory = Path(directory)
        meta = read_json(directory / "gross_4w_meta.json")
        if meta.get("model") != GROSS_NAME:
            raise ValueError("not a 4-week gross XGBoost model directory")
        model = cls(meta["seed"], meta["grid"], meta["fixed"], meta["feature_names"])
        model.model = xgb.XGBRegressor()
        model.model.load_model(directory / "regressor.json")
        if list(model.model.get_booster().feature_names or []) != model.feature_names:
            raise ValueError("saved booster feature order differs from its metadata")
        model.best_params, model.grid_results = meta["best_params"], meta["grid_results"]
        model.train_counts = meta.get("train_counts", {})
        model.skipped_objectives = meta.get("skipped_objectives", [])
        return model
