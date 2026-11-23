"""WBS 4.6: 모든 모델을 같은 고객, 같은 52주 정답으로 비교 평가한다.

MAE·RMSE·편향, 분포, 10분위, Top-N, 상위 20% 고객 일치율을 계산한다.
4주 조건부 총구매액 모델은 구매 고객만으로 따로 평가하고(``evaluate_gross_4w``), Treatment·Control의
구매자당 4주 총구매액을 비교하는 진단(``gross_by_assignment``)을 만든다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import (
    ASSIGNMENT_COLUMN, CONTROL_CODE, GROSS_COMPLETE_COLUMN, GROSS_LABEL_COLUMN, GROSS_PREDICTION_COLUMN, LABEL_COLUMN,
)

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
    # Baseline 대비 비율(1보다 작으면 Baseline보다 오차가 작다). Baseline 오차가 0이면 None.
    if "baseline" in report["models"]:
        base = report["models"]["baseline"]
        for values in report["models"].values():
            values["vs_baseline"] = {f"{key}_ratio": values[key] / base[key] if base[key] > 0 else None
                                     for key in ("mae", "rmse")}
    # 정답 구간에 Treatment 쿠폰 효과가 섞이므로 Treatment/Control별 지표도 따로 남긴다.
    if "treatment" in frame and frame["treatment"].notna().any():
        groups = {}
        for name, value in (("control", 0), ("treatment", 1)):
            part = frame.loc[frame["treatment"].eq(value)]
            if len(part) > 1:
                groups[name] = {model: regression_metrics(part[LABEL_COLUMN], part[f"pred_{model}"]) for model in models}
        report["by_assignment_group"] = groups
    return report


HISTORY_SEGMENTS = {"complete": True, "incomplete": False}


def evaluate_by_history(frame: pd.DataFrame, models, fraction: float = 0.2) -> dict:
    """``history_complete``(13주 입력 완전 관측 여부)별로 ``evaluate_predictions``를 따로 계산한다.

    고객이 2명 미만인 구간은 지표 없이 고객 수만 남긴다. 상위 20%는 구간 안에서 다시 정한다.
    """
    flags = frame["history_complete"].astype(bool)
    report = {}
    for name, value in HISTORY_SEGMENTS.items():
        part = frame.loc[flags.eq(value)]
        report[name] = evaluate_predictions(part, models, fraction) if len(part) > 1 else {"customers": int(len(part))}
    return report


def evaluate_gross_4w(frame: pd.DataFrame) -> dict:
    """4주 금액 모델을 ``frame``의 구매 고객(``actual_gross_4w_if_purchase``가 있는 행)으로 평가한다.

    MAE·RMSE, 평가 고객 수(``n``), 실제·예측 평균 총구매액(``mean_actual``·``mean_predicted``) 등을 돌려준다.
    구매 고객이 없으면 ``{"n": 0}``.
    """
    buyers = frame.loc[frame[GROSS_LABEL_COLUMN].notna()]
    if buyers.empty:
        return {"n": 0}
    return regression_metrics(buyers[GROSS_LABEL_COLUMN], buyers[GROSS_PREDICTION_COLUMN])


def _gross_summary(part: pd.DataFrame, predicted: bool) -> dict:
    """한 배정군의 4주 관측 완전 고객 수, 구매 고객 수·비율, 구매자당 총구매액 요약.

    예측 요약(평균·MAE)은 ``out_of_sample`` 열이 있으면 모델이 적합·선택에 쓰지 않은 구매 고객으로만 낸다.
    """
    buyers = part.loc[part[GROSS_LABEL_COLUMN].notna()]
    amounts = buyers[GROSS_LABEL_COLUMN].astype(float)
    result = {"customers": int(len(part)), "buyers": int(len(buyers)),
              "buyer_share": float(len(buyers) / len(part)) if len(part) else None,
              "mean_gross_per_buyer": float(amounts.mean()) if len(buyers) else None,
              "median_gross_per_buyer": float(amounts.median()) if len(buyers) else None,
              "std_gross_per_buyer": float(amounts.std(ddof=1)) if len(buyers) > 1 else None}
    if predicted:
        unseen = buyers.loc[buyers["out_of_sample"].astype(bool)] if "out_of_sample" in buyers else buyers
        result["predicted_buyers"] = int(len(unseen))
        if len(unseen):
            values = unseen[GROSS_PREDICTION_COLUMN].astype(float)
            result.update(mean_predicted_gross_per_buyer=float(values.mean()),
                          mae_on_buyers=float(np.mean(np.abs(values - unseen[GROSS_LABEL_COLUMN].astype(float)))))
    return result


def gross_by_assignment(frame: pd.DataFrame) -> dict:
    """Treatment와 Control의 구매자당 4주 할인 전 총구매액을 비교한다(전략 가정 진단, 학습에는 쓰지 않음).

    최종 전략은 ``uplift_4w × predicted_gross_4w_if_purchase``를 쓰는데, 이 곱은 쿠폰을 받아 구매한 고객의
    총구매액이 Control 구매 고객과 같다고 가정한다. ``frame``은 4주 관측이 완전한 고객이며, 예측 열이 있으면
    배정군별 예측 평균과 구매 고객 MAE도 남긴다(``out_of_sample`` 고객만). 차이의 95% 구간은 Welch 표준오차의
    정규 근사다.
    """
    if ASSIGNMENT_COLUMN not in frame or frame[ASSIGNMENT_COLUMN].isna().all():
        return {"available": False, "reason": f"no {ASSIGNMENT_COLUMN} column"}
    if GROSS_COMPLETE_COLUMN in frame:
        frame = frame.loc[frame[GROSS_COMPLETE_COLUMN].astype(bool)]
    assignment = frame[ASSIGNMENT_COLUMN]
    predicted = GROSS_PREDICTION_COLUMN in frame
    control = assignment.eq(CONTROL_CODE).fillna(False).astype(bool)
    groups = {"control": _gross_summary(frame.loc[control], predicted),
              "treatment": _gross_summary(frame.loc[assignment.notna() & ~control], predicted)}
    report = {"available": True, "groups": groups,
              "note": "descriptive diagnostic over customers with a complete 4-week window; treatment customers "
                      "are never used to train the 4-week model"}
    c, t = groups["control"], groups["treatment"]
    if c["buyers"] > 1 and t["buyers"] > 1:
        difference = t["mean_gross_per_buyer"] - c["mean_gross_per_buyer"]
        se = math.sqrt(t["std_gross_per_buyer"] ** 2 / t["buyers"] + c["std_gross_per_buyer"] ** 2 / c["buyers"])
        report["treatment_minus_control"] = {
            "difference": difference,
            "ratio": (t["mean_gross_per_buyer"] / c["mean_gross_per_buyer"] if c["mean_gross_per_buyer"] else None),
            "standard_error": se, "ci95": [difference - 1.96 * se, difference + 1.96 * se],
        }
    return report
