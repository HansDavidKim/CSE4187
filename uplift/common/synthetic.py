"""시뮬레이터가 나오기 전까지 쓰는 임시 데이터.

feature.md 3절의 입력 형식(실험 고객 목록, 고객별 주간 합계, w0 장바구니)을 그대로
흉내 낸다. 분포와 쿠폰 효과는 임의로 정한 값이며 시뮬레이터의 행동을 반영하지 않는다.
파이프라인 점검용으로만 쓰고, 여기서 나온 성능 수치는 결과로 인용하지 않는다.
"""

import numpy as np
import pandas as pd

from . import schema as S


def make_synthetic(n_customers: int = 5000, seed: int = 0, w0: str = "2026-06-01"):
    """(customers, weekly, cart, truth) 반환. truth는 고객별 실제 p0, p1, uplift."""
    rng = np.random.default_rng(seed)
    w0 = pd.Timestamp(w0)
    ids = np.arange(n_customers)
    offsets = np.arange(-S.HISTORY_WEEKS, S.OUTCOME_WEEKS)

    activity = rng.gamma(2.0, 0.6, n_customers)  # 주당 방문 강도
    buy_rate = rng.beta(2, 8, n_customers)  # 방문당 구매 확률
    trend = rng.normal(0, 0.03, n_customers)  # 주별 활동 변화
    tenure = rng.integers(2, 40, n_customers)  # 가입 후 경과 주

    lam = activity[:, None] * np.clip(1 + trend[:, None] * (offsets + S.HISTORY_WEEKS), 0.05, None)
    sessions = rng.poisson(lam).astype(float)
    pre = offsets < 0
    # 적격 고객: w0 이전 4주에 방문 1회 이상
    recent = (offsets >= -S.RECENT_WEEKS) & pre
    none = sessions[:, recent].sum(axis=1) == 0
    sessions[none, np.flatnonzero(recent)[-1]] = 1

    purchases = rng.binomial(sessions.astype(int), buy_rate[:, None]).astype(float)
    order_value = rng.lognormal(10.0, 0.5, n_customers)
    revenue = np.round(purchases * order_value[:, None])
    views = rng.poisson(sessions * 4.0).astype(float)
    minutes = np.round(sessions * rng.gamma(2.0, 4.0, n_customers)[:, None], 1)

    cart_sku = rng.poisson(0.6 * (activity > 1.0) + 0.2, n_customers)
    cart_qty = cart_sku + rng.poisson(0.3 * cart_sku)
    cart_value = np.round(cart_qty * rng.lognormal(9.5, 0.5, n_customers))

    # 결과 4주의 실제 유지 확률. 쿠폰 효과는 장바구니가 있고 활동이 줄어드는 고객에서 크다.
    fut = lam[:, ~pre]
    p0 = 1 - np.exp(-(fut * buy_rate[:, None]).sum(axis=1))
    effect = 0.15 * (cart_qty > 0) + 0.10 * (trend < 0) - 0.05 * (buy_rate > 0.35)
    p1 = np.clip(p0 + effect * (1 - p0), 0, 1)

    treatment = rng.integers(0, 2, n_customers)
    retained = rng.random(n_customers) < np.where(treatment == 1, p1, p0)
    # 결과 주의 구매를 유지 여부와 맞춘다: 유지면 1건 이상, 이탈이면 0건
    fut_p = purchases[:, ~pre]
    fut_p[~retained] = 0
    fix = retained & (fut_p.sum(axis=1) == 0)
    fut_p[fix, rng.integers(0, S.OUTCOME_WEEKS, fix.sum())] = 1
    purchases[:, ~pre] = fut_p
    sessions = np.maximum(sessions, purchases)
    revenue[:, ~pre] = np.round(fut_p * order_value[:, None])

    weekly = pd.DataFrame({
        S.CUSTOMER_ID: np.repeat(ids, len(offsets)),
        S.REFERENCE_WEEK: np.tile(w0 + pd.to_timedelta(offsets * 7, unit="D"), n_customers),
        "sessions": sessions.ravel(), "purchases": purchases.ravel(),
        "net_revenue": revenue.ravel(), "views": views.ravel(),
        "session_minutes": minutes.ravel(),
    })
    # 가입 전 주는 NULL
    before_signup = np.tile(offsets, n_customers) < -np.repeat(tenure, len(offsets))
    weekly.loc[before_signup, S.WEEKLY_COLUMNS] = np.nan

    customers = pd.DataFrame({S.CUSTOMER_ID: ids, S.TREATMENT: treatment, S.W0: w0})
    cart = pd.DataFrame({S.CUSTOMER_ID: ids, "cart_qty": cart_qty, "cart_sku": cart_sku,
                         "cart_gross_value": cart_value})
    truth = pd.DataFrame({S.CUSTOMER_ID: ids, "p0": p0, "p1": p1, "uplift": p1 - p0})
    return customers, weekly, cart, truth
