"""WBS 4.3: 기준시점 ``t0`` 직전 13주(``t0`` 미포함)로 RFM·행동 예측 변수 17개를 만든다.

관측 주만 합산하고 미관측 주(가입 전, 관측 기간 밖, 행 없음)는 0으로 세지 않는다.
관측됐지만 활동이 없으면 0, 분모가 0인 비율과 한쪽 구간이 전혀 관측되지 않은 추세는 NaN이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    HISTORY_COLUMNS, INPUT_WEEKS, NO_EVENT_RECENCY, PREDICTOR_FEATURES, RECENT_WEEKS, SEOUL, WEEK_SECONDS,
)

WEEK = pd.Timedelta(weeks=1)


def to_cutoff(value) -> pd.Timestamp:
    """서울(Asia/Seoul) 월요일 00:00 기준시점을 반환한다. 시간대 없는 값은 서울 시각으로 읽는다."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(SEOUL) if stamp.tzinfo is None else stamp.tz_convert(SEOUL)
    if stamp.weekday() != 0 or stamp != stamp.normalize():
        raise ValueError(f"cutoff must be an Asia/Seoul Monday 00:00 week boundary: {stamp.isoformat()}")
    return stamp


def fully_observed(panel: pd.DataFrame) -> pd.Series:
    """주 전체가 관측된 행이면 True인 Series를 반환한다(부분 관측 주는 False)."""
    # 관측 시간은 부동소수 오차를 감안해 1주(초)와 비교한다.
    return (panel["observed_week"].astype(bool) & ~panel["partial_coverage"].astype(bool)
            & panel["coverage_seconds"].ge(WEEK_SECONDS - 1e-6))


def _matrix(window: pd.DataFrame, customers, column: str) -> np.ndarray:
    """``column``을 고객 × k 행렬(k = 1이 ``t0`` 직전 주, …, 13)로 편다. 행이 없는 주는 NaN이다."""
    pivot = window.pivot(index="customer_id", columns="k", values=column)
    return pivot.reindex(index=customers, columns=range(1, INPUT_WEEKS + 1)).to_numpy(dtype=float, na_value=np.nan)


def _recency(values: np.ndarray) -> np.ndarray:
    """가장 최근 활동 주의 k(직전 주 = 1)를 반환한다. 13주 안에 활동이 없으면 ``NO_EVENT_RECENCY``(14)."""
    # 열이 k = 1(최근)부터 정렬돼 있어 argmax가 가장 최근 활동 주를 찾는다.
    active = np.nan_to_num(values) > 0
    return np.where(active.any(axis=1), active.argmax(axis=1) + 1, NO_EVENT_RECENCY).astype(float)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """분모가 0이면 0이 아니라 NaN(정의되지 않음)을 돌려주는 나눗셈. XGBoost는 결측 분기로 처리한다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def input_window(panel: pd.DataFrame, t0) -> pd.DataFrame:
    """``[t0 - 13주, t0)`` 구간의 행만 반환한다. 누수를 막기 위해 ``t0`` 이후 행은 절대 넘기지 않는다."""
    t0 = to_cutoff(t0)
    window = panel.loc[panel["week"].ge(t0 - INPUT_WEEKS * WEEK) & panel["week"].lt(t0)]
    if window["week"].ge(t0).any():
        raise AssertionError("feature window reached the cutoff")
    return window


def build_features(panel: pd.DataFrame, t0) -> pd.DataFrame:
    """``t0`` 이전에 관측된 모든 고객의 CLV 예측 변수를 계산한다.

    Args:
        panel: 어댑터를 거친 주간 패널(``adapter.adapt_weekly``).
        t0: 예측 기준시점(서울 월요일 00:00).

    Returns:
        ``customer_id`` 인덱스에 ``PREDICTOR_FEATURES``(float64, 고정 순서)와 ``HISTORY_COLUMNS``를 붙인 DataFrame.
        입력 구간에 관측 주가 하나도 없는 고객은 ``t0`` 시점에 아직 고객이 아니므로 행이 없다.
    """
    t0 = to_cutoff(t0)
    window = input_window(panel, t0)
    # 관측 주만 남긴다(부분 관측 주 포함). 미관측 주는 행렬에서 NaN이 되어 0으로 세지 않는다.
    window = window.loc[window["observed_week"].astype(bool)].copy()
    window["k"] = ((t0 - window["week"]) / WEEK).round().astype(int)
    window["full"] = fully_observed(window).astype(float)
    customers = np.sort(window["customer_id"].unique())
    purchases = _matrix(window, customers, "purchases")
    sessions = _matrix(window, customers, "sessions")
    paid = _matrix(window, customers, "paid_amount")
    coupons = _matrix(window, customers, "coupons")
    actions = _matrix(window, customers, "coupon_action")
    # 주별 관측 비율(0~1). 합하면 관측 주 수가 되어 추세·Baseline의 분모로 쓴다.
    coverage = np.nan_to_num(_matrix(window, customers, "coverage_seconds")) / WEEK_SECONDS
    full = np.nan_to_num(_matrix(window, customers, "full")).astype(bool)
    observed = ~np.isnan(purchases)
    # 최근 4주(k = 1..4)와 이전 9주(k = 5..13)
    recent, previous = slice(0, RECENT_WEEKS), slice(RECENT_WEEKS, INPUT_WEEKS)
    history_4w, history_prev = coverage[:, recent].sum(axis=1), coverage[:, previous].sum(axis=1)

    def total(values, part=slice(None)):
        return np.nansum(values[:, part], axis=1)

    # 추세 = 최근 4주 주당 평균 - 이전 9주 주당 평균. 각 평균은 관측 주 수로 나누고,
    # 한쪽 구간이 전혀 관측되지 않았으면 NaN이다.
    def trend(values):
        both = (history_4w > 0) & (history_prev > 0)
        return np.where(both, _ratio(total(values, recent), history_4w) - _ratio(total(values, previous), history_prev),
                        np.nan)

    purchase_13w, session_13w, spend_13w = total(purchases), total(sessions), total(paid)
    # 최근 주부터 '관측됐고 구매 0'인 주가 이어진 수. 구매 주나 미관측 주에서 cumprod가 0이 되어 멈춘다.
    zero_streak = np.cumprod(observed & (np.nan_to_num(purchases) == 0), axis=1).sum(axis=1)
    # `coupon_action` 값이 하나도 없으면 아래에서 0이 아니라 NaN으로 둔다.
    action_seen = (~np.isnan(actions)).any(axis=1)
    columns = {
        "purchase_count_4w": total(purchases, recent),
        "purchase_count_13w": purchase_13w,
        "session_count_4w": total(sessions, recent),
        "session_count_13w": session_13w,
        "net_spend_4w": total(paid, recent),
        "net_spend_13w": spend_13w,
        "weeks_since_last_purchase": _recency(purchases),
        "purchase_active_weeks_13w": (np.nan_to_num(purchases) > 0).sum(axis=1),
        "zero_purchase_streak": zero_streak,
        "purchase_trend": trend(purchases),
        "spend_trend": trend(paid),
        "purchase_per_session_13w": _ratio(purchase_13w, session_13w),
        "weeks_since_last_session": _recency(sessions),
        "session_active_weeks_13w": (np.nan_to_num(sessions) > 0).sum(axis=1),
        "session_trend": trend(sessions),
        "avg_paid_per_purchase_13w": _ratio(spend_13w, purchase_13w),
        "coupon_sent_13w": total(coupons),
    }
    # 열 순서는 `PREDICTOR_FEATURES`, 자료형은 float64로 고정한다.
    result = pd.DataFrame({name: np.asarray(columns[name], dtype="float64") for name in PREDICTOR_FEATURES},
                          index=pd.Index(customers, name="customer_id"))
    # 이하 `HISTORY_COLUMNS`는 예측 변수가 아니라 관측 이력 표시다(Baseline 분모, 완전 이력 여부).
    result["history_weeks_13w"] = coverage.sum(axis=1)
    result["history_weeks_4w"] = history_4w
    result["history_weeks_prev9w"] = history_prev
    result["feature_window_complete"] = full.all(axis=1)
    result["coupon_action_13w"] = np.where(action_seen, total(actions), np.nan)
    return result[[*PREDICTOR_FEATURES, *HISTORY_COLUMNS]]
