"""시뮬레이터 실행 결과가 없을 때 쓰는 임시 원시 로그.

load_raw와 같은 형태(RawData: 업무 테이블 7개 + 배정)를 만든다. 행동 분포와 쿠폰 효과는
임의로 정한 값이며 시뮬레이터의 행동을 반영하지 않는다. 파이프라인 점검용으로만 쓰고,
여기서 나온 성능 수치는 결과로 인용하지 않는다.
"""

import numpy as np
import pandas as pd

from . import schema as S
from .raw import RawData, normalize_tables

DAY = 86400.0


def make_synthetic(n_customers: int = 2000, seed: int = 0, t0: str = "2026-06-01 00:00+09:00"):
    """(raw, truth) 반환. truth는 고객별 실제 p0, p1, uplift (방문 일정이 주어졌을 때의 값)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(t0).tz_convert("UTC")
    end = OUTCOME = S.OUTCOME_DAYS * DAY  # 초 단위. t0 = 0
    prices = np.round(rng.lognormal(6.5, 0.6, 200)).astype(int) + 100
    products = pd.DataFrame({"product_id": [f"P{i:04d}" for i in range(len(prices))], "unit_price_minor": prices})

    customers, sessions, events, orders, items, coupons, assign, truth = [], [], [], [], [], [], [], []

    def emit(uid, sid, at, kind, **extra):
        events.append({"tmp": len(events), "customer_id": uid, "session_id": sid, "at": at, "event_type": kind, **extra})
        return len(events) - 1

    for u in range(n_customers):
        uid = f"U{u:07d}"
        rate, q0 = rng.gamma(2.0, 0.08), rng.beta(2, 10)  # 하루 방문 강도, 방문당 구매 확률
        signup = -rng.uniform(10, 240) * DAY
        treated = int(rng.integers(0, 2))
        starts = signup + np.cumsum(np.maximum(rng.exponential(1 / rate, 200), 0.05)) * DAY
        starts = np.sort(np.append(starts[starts < end - 0.05 * DAY], -rng.uniform(0.5, min(27.5, -signup / DAY)) * DAY))
        keep = np.append(True, np.diff(starts) > 0.05 * DAY) & ~((starts >= -0.05 * DAY) & (starts < 0))
        starts = np.append(signup, starts[keep & (starts > signup + 0.05 * DAY)])

        cart, cart_at_t0, q1, coupon_used = {}, None, q0, False
        for k, start in enumerate(starts):
            sid = f"{uid}-S{k:03d}"
            if start >= 0 and cart_at_t0 is None:
                cart_at_t0 = sum(cart.values())
                q1 = float(np.clip(q0 + (0.25 if cart_at_t0 else 0.08) - 0.10 * (q0 > 0.3), 0, 1))
            q = q1 if (start >= 0 and treated) else q0
            at, viewed = start, None
            for _ in range(1 + rng.poisson(2)):
                at += rng.uniform(20, 120)
                viewed = int(rng.integers(len(prices)))
                pid = products.product_id[viewed]
                emit(uid, sid, at, "view", product_id=pid, unit_price_minor=prices[viewed])
                if rng.random() < 0.3:
                    at += rng.uniform(5, 30)
                    cart[viewed] = cart.get(viewed, 0) + 1
                    emit(uid, sid, at, "cart", product_id=pid, quantity=1, unit_price_minor=prices[viewed])
                if cart and rng.random() < 0.1:
                    at += rng.uniform(5, 30)
                    gone = list(cart)[int(rng.integers(len(cart)))]
                    cart[gone] -= 1
                    if not cart[gone]:
                        del cart[gone]
                    emit(uid, sid, at, "remove_from_cart", product_id=products.product_id[gone], quantity=1,
                         unit_price_minor=prices[gone])
            if rng.random() < q:
                at += rng.uniform(20, 120)
                source, basket = ("cart", dict(cart)) if cart else ("buy_now", {viewed: 1})
                if source == "cart":
                    cart = {}
                oid = f"O{len(orders):08d}"
                purchase = emit(uid, sid, at, "purchase")
                gross = sum(prices[p] * n for p, n in basket.items())
                discount, coupon_id = 0, None
                if treated and start >= 0 and not coupon_used:
                    coupon_used, coupon_id = True, f"C{uid}"
                    discount = int(np.floor(max(prices[p] for p in basket) * 0.1 + 0.5))  # 최고 단가 1개의 10%
                    emit(uid, sid, at, "coupon_usage", related_purchase_event_id=purchase)
                orders.append({"order_id": oid, "purchase_event_id": purchase, "checkout_source": source,
                               "coupon_id": coupon_id, "discount_minor": discount, "paid_minor": gross - discount})
                reviewed = rng.random() < 0.5
                arrive = at + rng.uniform(1, 10) * DAY
                for line, (p, n) in enumerate(basket.items(), start=1):
                    items.append({"order_id": oid, "line_no": line, "product_id": products.product_id[p], "quantity": n})
                    if reviewed and arrive < end:
                        emit(uid, None, arrive, "review", product_id=products.product_id[p],
                             related_purchase_event_id=purchase, order_line_no=line,
                             rating=int(rng.choice([1, 2, 3, 4, 5], p=[.05, .05, .1, .2, .6])),
                             rating_kind=S.PRODUCT_REVIEW_KIND)
            sessions.append({"session_id": sid, "customer_id": uid, "started_at": start, "ended_at": at + 30})
        if cart_at_t0 is None:  # t0 이후 방문이 없는 고객
            cart_at_t0 = sum(cart.values())
            q1 = float(np.clip(q0 + (0.25 if cart_at_t0 else 0.08) - 0.10 * (q0 > 0.3), 0, 1))
        visits = int((starts >= 0).sum())
        p0, p1 = 1 - (1 - q0) ** visits, 1 - (1 - q1) ** visits
        customers.append({"customer_id": uid, "signed_up_at": signup})
        assign.append({S.CUSTOMER_ID: uid, S.T0: t0, S.TREATMENT: treated, "assignment_probability": 0.5})
        truth.append({S.CUSTOMER_ID: uid, "p0": p0, "p1": p1, "uplift": p1 - p0})
        if treated:
            coupons.append({"coupon_id": f"C{uid}", "customer_id": uid, "sent_at": 0.0, "expires_at": OUTCOME})

    clock = lambda seconds: t0 + pd.to_timedelta(np.asarray(seconds, dtype=float).round(), unit="s")
    columns = ["tmp", "at", *(c for c in S.RAW_COLUMNS["events"] if c not in ("event_id", "event_at"))]
    events = pd.DataFrame(events, columns=columns).sort_values(["at", "tmp"], kind="stable")
    events["event_id"] = np.arange(1, len(events) + 1)
    new_id = events.set_index("tmp")["event_id"]
    events["related_purchase_event_id"] = events["related_purchase_event_id"].map(new_id)
    events["event_at"] = clock(events["at"])
    orders = pd.DataFrame(orders)
    orders["purchase_event_id"] = orders["purchase_event_id"].map(new_id)
    frames = {"customers": pd.DataFrame(customers), "sessions": pd.DataFrame(sessions),
              "coupons": pd.DataFrame(coupons, columns=S.RAW_COLUMNS["coupons"])}
    for name, frame in frames.items():
        for column in S.TIME_COLUMNS.intersection(frame.columns):
            frame[column] = clock(frame[column])
    tables = {**frames, "products": products, "events": events, "orders": orders, "order_items": pd.DataFrame(items)}

    raw = RawData(tables=normalize_tables(tables), assignments=pd.DataFrame(assign),
                  observation_start=frames["customers"]["signed_up_at"].min(),
                  observation_end=t0 + pd.Timedelta(seconds=end), metadata={"dataset_id": f"synthetic-{seed}"})
    return raw, pd.DataFrame(truth)
