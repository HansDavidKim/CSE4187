"""CLV 공개 API: 다른 KIT 모듈(전략·예산 최적화, 대시보드)은 이 모듈만 import하면 된다 (WBS 4.7).

예측 대상은 두 가지이며 같은 고객·같은 기준시점 ``t0``를 쓴다(정의는 ``model_spec.md``).

* ``predicted_clv_12m``: ``t0``부터 **52주(364일)** 동안의 할인 후 실결제액 합. 열 이름은 호환성 때문에 유지하는
  이름이며 달력 12개월이 아니다. 새 학습(``control_no_coupon``)에서는 쿠폰을 보내지 않았을 때의 값이다.
* ``predicted_gross_4w_if_purchase``: ``t0``부터 **4주** 안에 한 번 이상 구매한다면 기대되는 할인 전 총구매액
  (반복 주문 포함). 4주 모델이 없는 번들(기존 전달 번들)에서는 결측이며 ``gross_4w_status``로 상태를 알린다.

결과 파일만 읽을 때::

    from clv.api import load_clv_predictions
    clv = load_clv_predictions("results/clv_predictions.csv")

학습된 번들로 직접 추론할 때::

    from clv.api import HANDOFF_BUNDLE, load_model
    model = load_model(HANDOFF_BUNDLE)            # 저장소의 전달 번들, 체크섬 대조 후 최종 모델
    clv = model.predict("<features 디렉터리>", reference_week="2025-01-06")

전략 모듈 입력 batch(Feature Store와 같은 고객·기준 시각)를 만들 때::

    from clv.api import HANDOFF_BUNDLE, predict_for_strategy
    result = predict_for_strategy("<features 디렉터리>", ledger="<dataset 디렉터리>", model_dir=HANDOFF_BUNDLE)
    # result.frame(11열), result.metadata(PredictionMetadata 5항목), result.audit, result.producer_metadata
    # result.gross_4w(같은 행의 4주 batch 또는 None), result.gross_4w_metadata, result.gross_4w_status
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bundle import (
    GROSS_NOT_TRAINED, GROSS_TARGET, GROSS_TARGET_DEFINITION_ID, HANDOFF_BUNDLE, TARGET, TARGET_DEFINITION_ID,
    VALUE_SEMANTICS,
)
from .config import POPULATION_POLICIES, CLVConfig, load_config
from .inference import (
    GROSS_STRATEGY_COLUMNS, STRATEGY_COLUMNS, CLVModel, StrategyPredictions, load_strategy_predictions,
)
from .pipeline import evaluate, run, train, write_predictions
from .schema import GROSS_PREDICTION_COLUMN, HIGH_VALUE_COLUMN, OUTPUT_COLUMNS, PREDICTION_COLUMN, SEOUL

__all__ = [
    # 학습·평가
    "train_clv", "evaluate_clv", "run_clv", "CLVConfig", "load_config", "POPULATION_POLICIES",
    # 추론(전달 번들 위치는 HANDOFF_BUNDLE)
    "load_model", "CLVModel", "predict_clv", "model_info", "HANDOFF_BUNDLE",
    # 전략 모듈 입력
    "predict_for_strategy", "StrategyPredictions", "load_strategy_predictions", "STRATEGY_COLUMNS",
    "GROSS_STRATEGY_COLUMNS",
    # 결과 파일
    "load_clv_predictions", "write_predictions",
    # 계약 상수
    "PREDICTION_COLUMN", "HIGH_VALUE_COLUMN", "TARGET", "TARGET_DEFINITION_ID", "VALUE_SEMANTICS",
    "GROSS_PREDICTION_COLUMN", "GROSS_TARGET", "GROSS_TARGET_DEFINITION_ID", "GROSS_NOT_TRAINED",
]


def _config(config) -> CLVConfig | None:
    """``CLVConfig``, dict, JSON/YAML 경로를 ``CLVConfig``로 바꾼다(None은 그대로 둔다)."""
    if config is None or isinstance(config, CLVConfig):
        return config
    if isinstance(config, dict):
        return CLVConfig.from_dict(config)
    return load_config(config)


def train_clv(weekly, ledger=None, output_dir: str | Path = "results/clv", config=None, cutoff=None, **options) -> dict:
    """52주 CLV 모델 4종(Baseline, BG/NBD + GG, XGBoost, 두 파트 XGBoost)과 4주 조건부 총구매액 XGBoost를
    학습·평가하고 번들로 저장한다. 학습 모집단은 ``config.population_policy``(기본 Control-only)다.

    Args:
        weekly: ``weekly_panel.parquet``, 시뮬레이터 features 디렉터리, CSV 또는 DataFrame.
        ledger: ``raw.sqlite``, 데이터셋 디렉터리, 주문 CSV/Parquet 또는 DataFrame.
            BG/NBD + Gamma-Gamma와 원장 대조에 필요하다.
        output_dir: 번들(모델, ``registry.json``, ``model_card.json``, 보고서)을 저장할 위치.
        config: ``CLVConfig``, dict 또는 JSON/YAML 경로.
        cutoff: 학습 기준시점 t0(서울 월요일). 기본값은 적격 고객이 가장 많은 기준시점.
        options: ``overwrite=True``면 이미 번들이 있는 ``output_dir``을 덮어쓴다(기본은 거절).

    Returns:
        저장한 registry(dict).
    """
    return train(weekly, ledger, output_dir, _config(config), cutoff, **options)


def evaluate_clv(weekly, ledger=None, model_dir: str | Path = "results/clv", cutoff=None, population: str = "test",
                 output=None, **options) -> dict:
    """저장된 모델을 ``cutoff``(기본: 학습 기준시점)의 정답 고객으로 평가한다."""
    return evaluate(weekly, ledger, model_dir, cutoff, population, output, **options)


def run_clv(weekly, ledger=None, output_dir: str | Path = "results/clv",
            predictions_path: str | Path = "results/clv_predictions.csv", config=None, cutoff=None,
            reference_week=None, **options) -> dict:
    """학습, 평가, ``results/clv_predictions.csv`` 쓰기를 한 번에 실행한다."""
    return run(weekly, ledger, output_dir, predictions_path, _config(config), cutoff, reference_week, **options)


def load_model(model_dir: str | Path = "results/clv", model: str | None = None) -> CLVModel:
    """번들을 체크섬 대조 후 불러온다. ``model``을 주지 않으면 registry의 최종 모델이다."""
    return CLVModel(model_dir, model)


def model_info(model_dir: str | Path = "results/clv", model: str | None = None) -> dict:
    """모델 이름·버전, 예측 대상, 코드·학습 데이터 버전, 입력 요구, Test 지표 요약."""
    return CLVModel(model_dir, model).info()


def predict_clv(weekly, ledger=None, model_dir: str | Path = "results/clv", reference_week=None, model=None,
                output: str | Path | None = None, customers=None, require_gross_4w: bool = False,
                **options) -> pd.DataFrame:
    """52주 CLV와 4주 조건부 총구매액을 같은 행으로 예측한다(``CLVModel.predict``). ``output``을 주면 CSV도 쓴다.

    번들에 4주 모델이 없으면 ``predicted_gross_4w_if_purchase``는 결측이다(``require_gross_4w=True``면 오류).
    """
    predictions = load_model(model_dir, model).predict(weekly, ledger, reference_week=reference_week,
                                                       customers=customers, require_gross_4w=require_gross_4w,
                                                       **options)
    if output is not None:
        write_predictions(predictions, output)
    return predictions


def predict_for_strategy(features, weekly=None, ledger=None, model_dir: str | Path = "results/clv", model=None,
                         output: str | Path | None = None, allow_training_customers: bool = False,
                         require_gross_4w: bool = False, **options) -> StrategyPredictions:
    """Feature Store의 모든 (고객, 기준 시각)에 대한 전략 모듈 입력 batch(``CLVModel.predict_for_strategy``).

    ``output``(``.parquet``)을 주면 frame, ``_metadata.json``, ``_audit.csv``(4주 모델이 있으면 ``_gross_4w.parquet``)도
    쓴다. CLV 학습·모델 선택에 쓴 고객이 있으면 기본으로 오류다(``allow_training_customers``).
    """
    result = load_model(model_dir, model).predict_for_strategy(
        features, weekly, ledger, allow_training_customers=allow_training_customers,
        require_gross_4w=require_gross_4w, **options)
    if output is not None:
        result.save(output)
    return result


def load_clv_predictions(path: str | Path = "results/clv_predictions.csv") -> pd.DataFrame:
    """전략·예산 모듈용으로 CLV CSV를 열별 자료형을 맞춰 읽는다.

    Returns:
        ``customer_id``(str), ``reference_week``(Asia/Seoul 시각, 예측 기준시점),
        ``predicted_clv_12m``(float, 원천 최소 단위, 52주 합), ``is_high_value_customer``(bool)와
        보조 열을 담은 DataFrame.
    """
    frame = pd.read_csv(path, dtype={"customer_id": str, "dataset_id": str, "currency": str, "amount_unit": str,
                                     "model_version": str})
    missing = [name for name in OUTPUT_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"{path} is missing CLV columns: {missing}")
    frame["reference_week"] = pd.to_datetime(frame["reference_week"], utc=True).dt.tz_convert(SEOUL)
    if "prediction_end_at" in frame:
        frame["prediction_end_at"] = pd.to_datetime(frame["prediction_end_at"], utc=True).dt.tz_convert(SEOUL)
    frame[PREDICTION_COLUMN] = frame[PREDICTION_COLUMN].astype(float)
    if GROSS_PREDICTION_COLUMN in frame:
        frame[GROSS_PREDICTION_COLUMN] = frame[GROSS_PREDICTION_COLUMN].astype(float)
    # CSV의 Boolean은 문자열로 읽힐 수 있으므로 "true"/"1"만 참으로 본다.
    frame[HIGH_VALUE_COLUMN] = frame[HIGH_VALUE_COLUMN].map(
        lambda value: str(value).strip().lower() in ("true", "1")).astype(bool)
    if "history_complete" in frame:
        frame["history_complete"] = frame["history_complete"].map(
            lambda value: str(value).strip().lower() in ("true", "1")).astype(bool)
    return frame
