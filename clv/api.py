"""WBS 4.7: 다른 KIT 모듈(전략·예산 최적화, 대시보드)이 쓰는 CLV 공개 API.

전략 모듈의 일반적인 사용::

    from clv.api import load_clv_predictions
    clv = load_clv_predictions("results/clv_predictions.csv")
    # 열: customer_id, reference_week, predicted_clv_12m, is_high_value_customer, ...

시뮬레이터 산출물에서 바로 예측하려면::

    from clv.api import predict_clv
    clv = predict_clv("artifacts/features", ledger="artifacts/small/raw.sqlite",
                      model_dir="results/clv", reference_week="2025-01-06")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CLVConfig, load_config
from .pipeline import evaluate, predict, run, train, write_predictions
from .schema import HIGH_VALUE_COLUMN, OUTPUT_COLUMNS, PREDICTION_COLUMN, SEOUL

__all__ = ["train_clv", "evaluate_clv", "predict_clv", "run_clv", "load_clv_predictions", "write_predictions",
           "CLVConfig", "load_config"]


def _config(config) -> CLVConfig | None:
    """``CLVConfig``, dict, JSON/YAML 경로를 ``CLVConfig``로 바꾼다(None은 그대로 둔다)."""
    if config is None or isinstance(config, CLVConfig):
        return config
    if isinstance(config, dict):
        return CLVConfig.from_dict(config)
    return load_config(config)


def train_clv(weekly, ledger=None, output_dir: str | Path = "results/clv", config=None, cutoff=None, **options) -> dict:
    """CLV 모델 3종을 학습·평가하고 저장한 registry를 반환한다.

    Args:
        weekly: ``weekly_panel.parquet``, features/export 디렉터리, CSV 또는 DataFrame.
        ledger: ``raw.sqlite``, 데이터셋 디렉터리, 주문 CSV/Parquet 또는 DataFrame.
            BG/NBD + Gamma-Gamma와 원장 대조에 필요하다.
        output_dir: 모델과 보고서를 저장할 위치.
        config: ``CLVConfig``, dict 또는 JSON/YAML 경로.
        cutoff: 학습 기준시점 t0(서울 월요일). 기본값은 적격 고객이 가장 많은 기준시점.
    """
    return train(weekly, ledger, output_dir, _config(config), cutoff, **options)


def evaluate_clv(weekly, ledger=None, model_dir: str | Path = "results/clv", cutoff=None, population: str = "test",
                 output=None, **options) -> dict:
    """저장된 모델을 ``cutoff``(기본: 학습 기준시점)의 정답 고객으로 평가한다."""
    return evaluate(weekly, ledger, model_dir, cutoff, population, output, **options)


def predict_clv(weekly, ledger=None, model_dir: str | Path = "results/clv", reference_week=None, model=None,
                output: str | Path | None = None, **options) -> pd.DataFrame:
    """52주 CLV를 예측한다. 미래 정답은 필요 없고, ``output``을 주면 CSV도 쓴다."""
    predictions = predict(weekly, ledger, model_dir, reference_week, model, **options)
    if output is not None:
        write_predictions(predictions, output)
    return predictions


def run_clv(weekly, ledger=None, output_dir: str | Path = "results/clv",
            predictions_path: str | Path = "results/clv_predictions.csv", config=None, cutoff=None,
            reference_week=None, **options) -> dict:
    """학습, 평가, ``results/clv_predictions.csv`` 쓰기를 한 번에 실행한다."""
    return run(weekly, ledger, output_dir, predictions_path, _config(config), cutoff, reference_week, **options)


def load_clv_predictions(path: str | Path = "results/clv_predictions.csv") -> pd.DataFrame:
    """전략·예산 모듈용으로 CLV CSV를 열별 자료형을 맞춰 읽는다.

    Returns:
        ``customer_id``(str), ``reference_week``(Asia/Seoul 시각, 예측 기준시점),
        ``predicted_clv_12m``(float, 원천 최소 단위), ``is_high_value_customer``(bool)와
        보조 열을 담은 DataFrame.
    """
    frame = pd.read_csv(path, dtype={"customer_id": str, "dataset_id": str, "currency": str, "amount_unit": str})
    missing = [name for name in OUTPUT_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"{path} is missing CLV columns: {missing}")
    frame["reference_week"] = pd.to_datetime(frame["reference_week"], utc=True).dt.tz_convert(SEOUL)
    frame[PREDICTION_COLUMN] = frame[PREDICTION_COLUMN].astype(float)
    # CSV의 Boolean은 문자열로 읽힐 수 있으므로 "true"/"1"만 참으로 본다.
    frame[HIGH_VALUE_COLUMN] = frame[HIGH_VALUE_COLUMN].map(
        lambda value: str(value).strip().lower() in ("true", "1")).astype(bool)
    if "history_complete" in frame:
        frame["history_complete"] = frame["history_complete"].map(
            lambda value: str(value).strip().lower() in ("true", "1")).astype(bool)
    return frame
