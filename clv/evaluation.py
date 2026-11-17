"""WBS 4.6: 모든 모델을 같은 고객, 같은 52주 정답으로 비교 평가한다.

MAE·RMSE·편향, 분포, 10분위, Top-N, 상위 20% 고객 일치율을 계산한다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import LABEL_COLUMN

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def regression_metrics(actual, predicted) -> dict:
    """MAE, RMSE, 편향(bias), 총액, R², Spearman 순위상관을 계산한다.

    R²·Spearman은 정의되지 않으면(분산 0, 표본 1개) None이다.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError("metrics need equally sized, non-empty arrays")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("metrics need finite labels and predictions")
    error = predicted - actual
    variance = float(np.sum((actual - actual.mean()) ** 2))
    spearman = (float(pd.Series(actual).corr(pd.Series(predicted), method="spearman"))
                if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0 else None)
    return {
        "n": int(len(actual)), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()), "mean_actual": float(actual.mean()), "mean_predicted": float(predicted.mean()),
        "total_actual": float(actual.sum()), "total_predicted": float(predicted.sum()),
        "r2": float(1 - np.sum(error ** 2) / variance) if variance > 0 else None, "spearman": spearman,
    }


def top_count(n: int, fraction: float) -> int:
    """상위 고객 수 = ceil(fraction * n), 최소 1명."""
    # 1e-9: 정수여야 할 fraction * n이 부동소수 오차로 조금 커져 한 명 더 올림되는 것을 막는다.
    return max(1, math.ceil(fraction * n - 1e-9))


def top_flags(values, customer_ids, fraction: float = 0.2) -> np.ndarray:
    """값 기준 상위 ``fraction`` 고객을 True로 표시한다. 동점은 `customer_id` 오름차순으로 정한다."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "customer_id": np.asarray(customer_ids, dtype=object)})
    order = frame.sort_values(["value", "customer_id"], ascending=[False, True], kind="mergesort").index
    flags = np.zeros(len(frame), dtype=bool)
    flags[order[:top_count(len(frame), fraction)]] = True
    return flags


def ranks(values, customer_ids) -> np.ndarray:
    """값의 내림차순 순위(1 = 최댓값). 동점 규칙은 ``top_flags``와 같다."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "customer_id": np.asarray(customer_ids, dtype=object)})
    order = frame.sort_values(["value", "customer_id"], ascending=[False, True], kind="mergesort").index
    result = np.empty(len(frame), dtype=int)
    result[order] = np.arange(1, len(frame) + 1)
    return result


def top_overlap(actual, predicted, customer_ids, fraction: float = 0.2) -> dict:
    """같은 크기 k의 예측 상위·실제 상위 집합 일치율과 포착 가치 비율을 계산한다."""
    actual = np.asarray(actual, dtype=float)
    predicted_top, actual_top = top_flags(predicted, customer_ids, fraction), top_flags(actual, customer_ids, fraction)
    k = int(actual_top.sum())
    overlap = int((predicted_top & actual_top).sum())
    boundary = actual[actual_top].min()
    best_value = actual[actual_top].sum()
    return {
        "k": k, "overlap": overlap, "overlap_rate": overlap / k,
        # 예측 상위가 가져간 실제 가치 / 실제 상위가 가진 최대 가치
        "captured_actual_value_share": float(actual[predicted_top].sum() / best_value) if best_value > 0 else None,
        "actual_boundary_value": float(boundary),
        "actual_ties_at_boundary": int((actual == boundary).sum()),
        "note": "overlap_rate = |predicted top ∩ actual top| / k (precision = recall for equal k); "
                "ties at the actual boundary are broken by customer_id",
    }


def distribution(values) -> dict:
    """분위수(``QUANTILES``), 평균, 최솟값·최댓값, 0 비율로 분포를 요약한다."""
    values = np.asarray(values, dtype=float)
    result = {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES}
    result.update(n=int(len(values)), mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                  share_zero=float(np.mean(values == 0)))
    return result


def decile_table(frame: pd.DataFrame, models) -> pd.DataFrame:
    """모델별 예측 10분위(1 = 최상위)마다 평균 예측과 실제값(평균·합계)을 비교한다."""
    tables = []
    for model in models:
        column = f"pred_{model}"
        rank = ranks(frame[column], frame["customer_id"])
        decile = np.ceil(rank / len(frame) * 10).astype(int)
        table = (frame.assign(decile=decile).groupby("decile")
                 .agg(customers=("customer_id", "size"), mean_predicted=(column, "mean"),
                      mean_actual=(LABEL_COLUMN, "mean"), total_actual=(LABEL_COLUMN, "sum"))
                 .reset_index())
        tables.append(table.assign(model=model))
    return pd.concat(tables, ignore_index=True)[["model", "decile", "customers", "mean_predicted", "mean_actual", "total_actual"]]


def top_n_table(frame: pd.DataFrame, model: str, n: int = 20) -> pd.DataFrame:
    """예측값 상위 N명과 그들의 실제 정답·실제 순위를 나란히 보여 준다."""
    column = f"pred_{model}"
    rank = ranks(frame[column], frame["customer_id"])
    table = frame.assign(predicted_rank=rank, actual_rank=ranks(frame[LABEL_COLUMN], frame["customer_id"]))
    table = table.sort_values("predicted_rank").head(n)
    return table[["predicted_rank", "customer_id", column, LABEL_COLUMN, "actual_rank"]].rename(
        columns={column: "predicted_clv_12m"}).assign(model=model)


def evaluate_predictions(frame: pd.DataFrame, models, fraction: float = 0.2) -> dict:
    """모델별 지표를 하나의 공통 고객·정답 집합에서 계산한다.

    ``frame``에는 ``customer_id``, ``actual_clv_12m``, ``pred_<model>`` 열이 있어야 하며,
    모든 모델을 정확히 이 행들로 평가한다(정답이나 예측이 빠진 행이 있으면 오류).
    """
    models = list(models)
    missing = [f"pred_{model}" for model in models if f"pred_{model}" not in frame]
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    if frame[[LABEL_COLUMN, *(f"pred_{model}" for model in models)]].isna().any().any():
        raise ValueError("every evaluated customer needs a label and a prediction from every model")
    actual = frame[LABEL_COLUMN].to_numpy(dtype=float)
    report = {"customers": int(len(frame)), "top_fraction": fraction, "actual_distribution": distribution(actual),
              "models": {}}
    for model in models:
        predicted = frame[f"pred_{model}"].to_numpy(dtype=float)
        report["models"][model] = {**regression_metrics(actual, predicted),
                                   "top": top_overlap(actual, predicted, frame["customer_id"], fraction),
                                   "prediction_distribution": distribution(predicted)}
    # 정답 구간에 Treatment 쿠폰 효과가 섞이므로 Treatment/Control별 지표도 따로 남긴다.
    if "treatment" in frame and frame["treatment"].notna().any():
        groups = {}
        for name, value in (("control", 0), ("treatment", 1)):
            part = frame.loc[frame["treatment"].eq(value)]
            if len(part) > 1:
                groups[name] = {model: regression_metrics(part[LABEL_COLUMN], part[f"pred_{model}"]) for model in models}
        report["by_assignment_group"] = groups
    return report
