"""시뮬레이터 v5 스키마를 따르는 작은 합성 입력(테스트 배선용이며 성능 근거가 아니다).

구매는 BG/NBD와 비슷한 과정으로 만들어 확률 모델이 가정하는 구조를 갖게 했다.
실제 고객 행동을 추정한 값은 하나도 없다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

SEOUL = "Asia/Seoul"
WEEK = pd.Timedelta(weeks=1)
FIRST_WEEK = pd.Timestamp("2024-01-01 00:00", tz=SEOUL)
# 시뮬레이터 시계는 00:00 UTC(서울 09:00)에 시작하므로 첫 서울 주는 부분 관측 주다.
OBSERVATION_START = pd.Timestamp("2024-01-01T00:00:00Z")


def synthetic_outputs(n_customers: int = 240, n_weeks: int = 72, seed: int = 0, late_share: float = 0.15,
                      assignment_week: int = 40, currency: str = "SOURCE_UNSPECIFIED", amount_unit: str = "0.01"):
    """시뮬레이터 산출물과 같은 모양의 ``(weekly_panel, orders, metadata)``를 반환한다.

    ``late_share`` 비율의 고객은 관측 시작 뒤에 가입하고(가입 전 주는 미관측 주),
    인덱스가 홀수인 고객은 Treatment로 ``assignment_week``에 쿠폰을 받는다.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(FIRST_WEEK, periods=n_weeks, freq="7D")
    observation_end = (weeks[-1] + pd.Timedelta(days=3)).tz_convert("UTC")
    assigned_at = weeks[assignment_week]
    ids = [f"c{index:04d}" for index in range(n_customers)]
    signup = {cid: OBSERVATION_START for cid in ids}
    for cid in rng.choice(ids, int(late_share * n_customers), replace=False):
        signup[cid] = (weeks[int(rng.integers(3, 30))] + pd.Timedelta(days=2, hours=10)).tz_convert("UTC")
    treatment = {cid: index % 2 for index, cid in enumerate(ids)}
    orders, sessions, death = [], [], {}
    for cid in ids:
        rate = rng.gamma(0.9, 0.3)  # 생존 중 주당 구매율
        dropout = rng.beta(1.3, 9.0)
        now, alive, used = signup[cid], True, False
        while alive and rate > 0:
            now = now + pd.Timedelta(weeks=float(rng.exponential(1 / rate)))
            if now >= observation_end:
                break
            gross = int(round(rng.gamma(4.0, 400.0))) + 1
            discount = 0
            # Treatment 고객은 배정 후 28일 안의 첫 주문에서 10% 쿠폰을 한 번 쓴다.
            if treatment[cid] and not used and assigned_at <= now < assigned_at + pd.Timedelta(days=28):
                discount, used = int(round(gross * 0.1)), True
            orders.append({"order_id": f"o{len(orders):06d}", "customer_id": cid, "event_at": now,
                           "gross_minor": gross, "discount_minor": discount, "paid_minor": gross - discount,
                           "coupon_id": f"k-{cid}" if discount else None})
            sessions.append((cid, now))
            alive = rng.random() > dropout
        death[cid] = now if not alive else observation_end
    order_frame = pd.DataFrame(orders)
    order_frame["event_at"] = pd.to_datetime(order_frame["event_at"], utc=True)
    rows = []
    for cid in ids:
        mine = order_frame.loc[order_frame["customer_id"].eq(cid)]
        order_week = mine["event_at"].dt.tz_convert(SEOUL).dt.normalize()
        order_week = order_week - pd.to_timedelta(order_week.dt.weekday, unit="D")
        for position, week in enumerate(weeks):
            # 그 주 중 가입 후이면서 관측 기간 안인 시간(초). 0이면 미관측 주, 1주 미만이면 부분 관측 주.
            start = max(week.tz_convert("UTC"), signup[cid], OBSERVATION_START)
            end = min((week + WEEK).tz_convert("UTC"), observation_end)
            coverage = max(0.0, (end - start).total_seconds())
            observed = coverage > 0
            inside = mine.loc[order_week.eq(week)]
            browse = int(rng.poisson(0.4)) if observed and week.tz_convert("UTC") < death[cid] else 0
            sent = int(treatment[cid] == 1 and week == assigned_at)
            values = {"sessions": len(inside) + browse, "purchases": len(inside),
                      "gross_revenue_minor": int(inside["gross_minor"].sum()),
                      "redemption_discount_cost_minor": int(inside["discount_minor"].sum()),
                      "net_revenue_minor": int(inside["paid_minor"].sum()),
                      "coupons_sent": sent, "coupon_actions": sent, "marketing_send_cost_minor": 100 * sent,
                      "views": 3 * (len(inside) + browse)}
            # 미관측 주의 활동·금액은 0이 아니라 NA로 둔다.
            rows.append({"customer_id": cid, "reference_week": week, "observed_week": observed,
                         "partial_coverage": observed and coverage < WEEK.total_seconds(), "coverage_seconds": coverage,
                         **{key: (value if observed else pd.NA) for key, value in values.items()},
                         "group": "treatment" if treatment[cid] else "control", "treatment": treatment[cid],
                         "assigned_at": assigned_at, "as_of_at": assigned_at, "relative_week": position - assignment_week,
                         "dataset_id": f"synthetic-fixture-seed{seed}", "seed": seed,
                         "currency": currency, "amount_unit": amount_unit})
    panel = pd.DataFrame(rows)
    for column in ("sessions", "purchases", "gross_revenue_minor", "redemption_discount_cost_minor", "net_revenue_minor",
                   "coupons_sent", "coupon_actions", "marketing_send_cost_minor", "views"):
        panel[column] = panel[column].astype("Int64")
    panel["net_after_marketing_minor"] = panel["net_revenue_minor"] - panel["marketing_send_cost_minor"]
    metadata = {"dataset_id": f"synthetic-fixture-seed{seed}", "seed": seed, "currency": currency,
                "amount_unit": amount_unit, "observation_start_at": OBSERVATION_START.isoformat(),
                "observation_end_at": observation_end.isoformat(), "as_of_at": assigned_at.tz_convert("UTC").isoformat()}
    return panel, order_frame, metadata


def write_raw_sqlite(directory: Path, orders: pd.DataFrame, metadata: dict) -> Path:
    """시뮬레이터 파일 구성대로 최소한의 orders/events SQLite와 체크섬이 든 manifest를 쓴다."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "raw.sqlite"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE events(event_id INTEGER PRIMARY KEY, customer_id TEXT, session_id TEXT, "
                           "event_at TEXT, event_type TEXT)")
        connection.execute("CREATE TABLE orders(order_id TEXT PRIMARY KEY, purchase_event_id INTEGER, checkout_source TEXT, "
                           "coupon_id TEXT, gross_minor INTEGER, discount_minor INTEGER, paid_minor INTEGER)")
        for event_id, row in enumerate(orders.itertuples(index=False), start=1):
            connection.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                               (event_id, row.customer_id, f"s{event_id}", row.event_at.isoformat(timespec="microseconds"),
                                "purchase"))
            connection.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                               (row.order_id, event_id, "cart", row.coupon_id, int(row.gross_minor),
                                int(row.discount_minor), int(row.paid_minor)))
    connection.close()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(json.dumps(
        {"raw_schema_version": 5, "database_sha256": digest, "metadata": metadata}))
    return path
