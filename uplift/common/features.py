"""고객별 주간 합계 -> 고객 1명당 1행의 학습 데이터셋 (UC-01)."""

import numpy as np
import pandas as pd

from . import schema as S


def _week_matrix(weekly: pd.DataFrame, column: str, customers: pd.Index) -> pd.DataFrame:
    """고객 x 주 오프셋 행렬. 오프셋 -1이 w0 직전 주, 0이 w0."""
    offsets = range(-S.HISTORY_WEEKS, S.OUTCOME_WEEKS)
    mat = weekly.pivot(index=S.CUSTOMER_ID, columns="_offset", values=column)
    return mat.reindex(index=customers, columns=offsets)


def _weeks_since_last(active: pd.DataFrame) -> pd.Series:
    """w0 - 활동이 있던 마지막 주. 과거 12주에 활동이 없으면 NaN."""
    offsets = np.asarray(active.columns)
    last = np.where(active.to_numpy(), offsets, -np.inf).max(axis=1)
    return pd.Series(np.where(np.isfinite(last), -last, np.nan), index=active.index)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).where(den > 0)


def build_dataset(
    customers: pd.DataFrame, weekly: pd.DataFrame, cart: pd.DataFrame | None = None
) -> pd.DataFrame:
    """실험 고객 목록 + 주간 합계 (+ w0 장바구니) -> customer_id, w0, feature, T, y.

    X는 w0 이전 주만, y는 [w0, w0+4) 주만 읽는다.
    결과 4주가 다 관측되지 않은 고객은 y가 NaN이므로 학습 전에 제외한다.
    """
    if customers[S.W0].nunique() != 1:
        raise ValueError("한 데이터셋에는 기준 주 w0가 하나여야 한다. w0별로 나눠서 호출할 것.")
    w0 = pd.Timestamp(customers[S.W0].iloc[0])
    ids = pd.Index(customers[S.CUSTOMER_ID], name=S.CUSTOMER_ID)

    weekly = weekly[weekly[S.CUSTOMER_ID].isin(ids)].copy()
    days = (pd.to_datetime(weekly[S.REFERENCE_WEEK]) - w0).dt.days
    if (days % 7 != 0).any():
        raise ValueError("reference_week가 w0와 같은 요일(주 시작일)이 아니다.")
    weekly["_offset"] = days // 7

    m = {c: _week_matrix(weekly, c, ids) for c in S.WEEKLY_COLUMNS if c in weekly}
    hist = [o for o in range(-S.HISTORY_WEEKS, 0)]
    recent = [o for o in range(-S.RECENT_WEEKS, 0)]
    prior = [o for o in range(-2 * S.RECENT_WEEKS, -S.RECENT_WEEKS)]
    future = [o for o in range(0, S.OUTCOME_WEEKS)]

    sessions, purchases, revenue = m["sessions"], m["purchases"], m["net_revenue"]
    out = pd.DataFrame(index=ids)

    out["weeks_since_last_purchase"] = _weeks_since_last(purchases[hist] > 0)
    out["weeks_since_last_session"] = _weeks_since_last(sessions[hist] > 0)
    out["purchase_count_4w"] = purchases[recent].sum(axis=1)
    out["session_count_4w"] = sessions[recent].sum(axis=1)

    # 가입·관측 전 주는 NULL이므로 NULL이 아닌 최초 주부터가 관측 기간
    observed = sessions[hist].notna().to_numpy()
    first = np.where(observed, np.asarray(hist), np.inf).min(axis=1)
    out["observed_tenure_weeks"] = np.where(np.isfinite(first), -first, np.nan)
    out["net_spend_4w"] = revenue[recent].sum(axis=1)
    out["mean_net_order_value_4w"] = _ratio(out["net_spend_4w"], out["purchase_count_4w"])
    out["orders_per_session_4w"] = _ratio(out["purchase_count_4w"], out["session_count_4w"])
    out["purchase_weeks_12w"] = (purchases[hist] > 0).sum(axis=1)
    out["session_weeks_12w"] = (sessions[hist] > 0).sum(axis=1)
    out["orders_delta_4w"] = out["purchase_count_4w"] - purchases[prior].sum(axis=1)
    out["net_spend_delta_4w"] = out["net_spend_4w"] - revenue[prior].sum(axis=1)

    # 시뮬레이터가 아직 내보내지 않는 컬럼은 NaN으로 둔다
    if "views" in m:
        out["view_count_4w"] = m["views"][recent].sum(axis=1)
        out["views_per_session_4w"] = _ratio(out["view_count_4w"], out["session_count_4w"])
    if "session_minutes" in m:
        out["mean_session_minutes_4w"] = _ratio(
            m["session_minutes"][recent].sum(axis=1), out["session_count_4w"]
        )
    if cart is not None:
        c = cart.set_index(S.CUSTOMER_ID).reindex(ids)
        for col in S.CART_COLUMNS:
            out[f"{col}_w0"] = c[col]

    out = out.reindex(columns=S.FEATURES).astype(float)

    # 정답: 결과 4주가 전부 관측된 고객만
    fut = purchases[future]
    y = (fut.sum(axis=1) > 0).astype(float).where(fut.notna().all(axis=1))

    head = customers.set_index(S.CUSTOMER_ID)[[S.W0, S.TREATMENT]].reindex(ids)
    return pd.concat([head, out, y.rename(S.LABEL)], axis=1).reset_index()
