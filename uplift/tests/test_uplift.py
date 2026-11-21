import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from uplift.common import schema as S
from uplift.common.evaluate import purchase_roc_auc, qini_area, qini_coefficient, qini_curve, qini_decile_curve
from uplift.common.features import _replay_cart, build_dataset, build_features, build_labels
from uplift.common.raw import RawData, load_raw, normalize_tables
from uplift.common.synthetic import make_synthetic
from uplift.Hlearner import HLearner
from uplift.Tlearner import TLearner
from uplift.Xlearner import XLearner

T0 = pd.Timestamp("2026-06-01 00:00", tz="Asia/Seoul").tz_convert("UTC")  # 월요일 00시
ago = lambda days, minutes=0: T0 - pd.Timedelta(days=days) + pd.Timedelta(minutes=minutes)


def test_purchase_roc_auc_uses_the_assigned_arm():
    pred = pd.DataFrame({"p0": [0.1, 0.9, 0.8, 0.2], "p1": [0.9, 0.1, 0.2, 0.8]})
    treatment = [0, 1, 0, 1]
    assert purchase_roc_auc(pred, treatment, [0, 0, 1, 1]) == 1.0
    assert np.isnan(purchase_roc_auc(pred, treatment, [0, 0, 0, 0]))


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(1500, seed=1)[0]


@pytest.fixture(scope="module")
def data(raw):
    return build_dataset(raw)


def hand_tables():
    """고객 1명. 세션 4개(100·50·10·2일 전)와 t0 3일 뒤 구매 1건."""
    event = lambda sid, at, kind, pid=None, qty=None, price=None, **extra: {
        "customer_id": "A", "session_id": sid, "event_at": at, "event_type": kind, "product_id": pid,
        "quantity": qty, "unit_price_minor": price, **extra}
    events = [
        event("S1", ago(100, 1), "view", "P1", price=1000),
        event("S1", ago(100, 2), "cart", "P1", 1, 1000),
        event("S1", ago(100, 3), "cart", "P2", 1, 500),
        event("S2", ago(50, 5), "view", "P2", price=500),
        event("S2", ago(50, 10), "purchase"),  # O1: 장바구니의 P1 구매
        event(None, ago(40), "review", "P1", related_purchase_event_id=5, order_line_no=1, rating=5,
              rating_kind=S.PRODUCT_REVIEW_KIND),
        event("S3", ago(10, 1), "view", "P3", price=300),
        event("S3", ago(10, 2), "view", "P1", price=1000),
        event("S3", ago(10, 3), "cart", "P3", 1, 300),
        event("S3", ago(10, 4), "cart", "P3", 1, 300),
        event("S3", ago(10, 5), "remove_from_cart", "P3", 1, 300),
        event("S3", ago(10, 20), "purchase"),  # O2: P1 2개 바로 구매
        event(None, ago(5), "review", "P1", related_purchase_event_id=12, order_line_no=1, rating=4,
              rating_kind=S.PRODUCT_REVIEW_KIND),
        event("S4", ago(2, 1), "view", "P2", price=500),
        event("S4", ago(2, 2), "cart", "P2", 1, 500),
        event("S5", ago(-3, 1), "purchase"),  # 결과 기간의 구매
    ]
    events = pd.DataFrame(events)
    events.insert(0, "event_id", range(1, len(events) + 1))
    session = lambda sid, start, minutes: {"session_id": sid, "customer_id": "A", "started_at": start,
                                           "ended_at": start + pd.Timedelta(minutes=minutes)}
    order = lambda oid, event_id, source, paid: {"order_id": oid, "purchase_event_id": event_id,
                                                  "checkout_source": source, "coupon_id": None,
                                                  "discount_minor": 0, "paid_minor": paid}
    item = lambda oid, pid, qty: {"order_id": oid, "line_no": 1, "product_id": pid, "quantity": qty}
    return normalize_tables({
        "customers": pd.DataFrame({"customer_id": ["A"], "signed_up_at": [ago(100)]}),
        "products": pd.DataFrame({"product_id": ["P1", "P2", "P3"], "unit_price_minor": [1000, 500, 300]}),
        "sessions": pd.DataFrame([session("S1", ago(100), 10), session("S2", ago(50), 20),
                                  session("S3", ago(10), 30), session("S4", ago(2), 5),
                                  session("S5", ago(-3), 5)]),
        "events": events,
        "orders": pd.DataFrame([order("O1", 5, "cart", 1000), order("O2", 12, "buy_now", 2000),
                                order("O3", 16, "buy_now", 300)]),
        "order_items": pd.DataFrame([item("O1", "P1", 1), item("O2", "P1", 2), item("O3", "P3", 1)]),
        "coupons": pd.DataFrame(columns=S.RAW_COLUMNS["coupons"]),
    })


SNAP = pd.DataFrame({S.CUSTOMER_ID: ["A"], S.T0: [T0]})


def test_dataset_shape(raw, data):
    assert list(data.columns) == [S.CUSTOMER_ID, S.T0, S.TREATMENT, *S.KIT39, S.LABEL]
    assert len(data) == 1500 and data[S.CUSTOMER_ID].is_unique
    assert data[S.LABEL].isin([0, 1]).all() and data[S.LABEL].nunique() == 2
    assert len(S.KIT39) == 39 and len(S.FEATURES) == 36
    assert (data["session_count_30d"] >= 1).all()  # 적격 고객은 직전 28일에 방문했다
    assert data[S.FEATURES].notna().any().all()  # 전부 NULL인 feature가 없다


def test_features_by_hand():
    row = build_features(hand_tables(), SNAP).iloc[0]
    gaps = np.array([50.0, 40.0, 8.0])
    expected = {
        "visit_recency_days": 2, "purchase_recency_days": 10 - 20 / 1440, "observed_tenure_days": 100,
        "session_count_30d": 2, "view_count_30d": 3, "purchase_count_30d": 1, "net_spend_30d": 2000,
        "mean_net_order_value_30d": 2000, "mean_session_duration_minutes_30d": 17.5,
        "views_per_session_30d": 1.5, "orders_per_session_30d": 0.5,
        "mean_purchase_gap_days": 40 + 10 / 1440,
        "intervisit_start_gap_mean_days_90d": gaps.mean(),
        "intervisit_start_gap_cv_90d": gaps.std(ddof=1) / gaps.mean(),
        "unique_purchase_products_30d": 1, "purchased_qty_30d": 2, "mean_basket_qty_30d": 2,
        "cart_qty_current": 3, "cart_sku_current": 2, "cart_gross_value_current": 2 * 500 + 300,
        "last_session_min_viewed_price": 500, "last_session_max_viewed_price": 500,
        "last_session_start_hour": 0, "last_session_start_weekday": 5,
        "coupon_uses_30d": 0, "coupon_discount_amount_30d": 0,
        "orders_delta_30d": 0, "net_spend_delta_30d": 1000, "mean_net_order_value_delta_30d": 1000,
        "unique_purchase_products_delta_30d": 0, "purchased_qty_delta_30d": 1,
        "review_count_30d": 1, "mean_review_rating_30d": 4,
        "cart_adds_per_nonpurchase_session_30d": 1, "cart_removes_per_nonpurchase_session_30d": 0,
        "review_last_order_mean_rating": 4, "review_recency_days": 5, "review_non5_share_30d": 1,
    }
    for name, value in expected.items():
        assert row[name] == pytest.approx(value), name
    assert np.isnan(row["coupon_redemption_recency_days"])
    assert set(expected) | {"coupon_redemption_recency_days"} == set(S.KIT39)
    assert build_labels(hand_tables(), SNAP, T0 + pd.Timedelta(days=28)).iloc[0] == 1


def test_unobserved_history_is_null():
    # 관측이 t0 45일 전에 시작: 최근 30일 창은 유지, 60일·90일 범위와 현재 장바구니는 NULL
    row = build_features(hand_tables(), SNAP, observation_start=ago(45)).iloc[0]
    assert row["session_count_30d"] == 2 and row["visit_recency_days"] == 2
    for name in ("orders_delta_30d", "intervisit_start_gap_mean_days_90d", "cart_qty_current"):
        assert np.isnan(row[name]), name
    # 가입 시각이 기준 시각이면 전부 NULL
    early = pd.DataFrame({S.CUSTOMER_ID: ["A"], S.T0: [ago(100)]})
    assert build_features(hand_tables(), early).iloc[0].isna().all()


def test_cart_replay_checkout_sources():
    at = lambda m: ago(1, m)
    move = lambda i, pid, kind="cart": {"customer_id": "A", "product_id": pid, "event_at": at(i), "event_id": i,
                                        "event_type": kind, "quantity": 1}
    events = pd.DataFrame([move(1, "A"), move(2, "B"), move(3, "C"), move(4, "C", "remove_from_cart")])
    item = lambda i, pid, qty, source: {"customer_id": "A", "product_id": pid, "event_at": at(i),
                                        "purchase_event_id": i, "quantity": qty, "checkout_source": source}
    # mixed: A 2개 구매 -> 보유한 1개만 차감. buy_now: B는 그대로 남는다.
    items = pd.DataFrame([item(5, "A", 2, "mixed"), item(6, "B", 1, "buy_now")])
    cart = _replay_cart(events, items)
    assert dict(zip(cart["product_id"], cart["qty"])) == {"B": 1}


def test_future_records_do_not_change_features(raw, data):
    cut = {name: frame.copy() for name, frame in raw.tables.items()}
    cut["sessions"] = cut["sessions"][cut["sessions"]["started_at"] < T0]
    cut["events"] = cut["events"][cut["events"]["event_at"] < T0]
    cut["orders"] = cut["orders"][cut["orders"]["purchase_event_id"].isin(cut["events"]["event_id"])]
    cut["order_items"] = cut["order_items"][cut["order_items"]["order_id"].isin(cut["orders"]["order_id"])]
    past_only = RawData(cut, raw.assignments, raw.observation_start, raw.observation_end, {})
    pd.testing.assert_frame_equal(build_dataset(past_only)[S.KIT39], data[S.KIT39])


def test_unobserved_outcome_is_null(raw):
    early = RawData(raw.tables, raw.assignments, raw.observation_start,
                    raw.observation_end - pd.Timedelta(days=1), {})
    assert build_dataset(early)[S.LABEL].isna().all()


def test_load_raw_reads_simulator_run(raw, data, tmp_path):
    database = tmp_path / S.RAW_DATABASE
    with sqlite3.connect(database) as connection:
        for name, frame in raw.tables.items():
            frame = frame.copy()
            for column in S.TIME_COLUMNS.intersection(frame.columns):
                frame[column] = frame[column].map(lambda v: v.isoformat(), na_action="ignore")
            frame.to_sql(name, connection, index=False)
    connection.close()
    manifest = {"database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                "metadata": {"observation_start_at": raw.observation_start.isoformat(),
                             "observation_end_at": raw.observation_end.isoformat()}}
    (tmp_path / S.RAW_MANIFEST).write_text(json.dumps(manifest))
    raw.assignments.rename(columns={S.T0: "as_of_at"}).to_parquet(tmp_path / S.ASSIGNMENT_FILE)

    loaded = load_raw(tmp_path)
    pd.testing.assert_frame_equal(build_dataset(loaded), data, check_dtype=False)
    (tmp_path / S.RAW_MANIFEST).write_text(json.dumps({**manifest, "database_sha256": "0"}))
    with pytest.raises(ValueError):
        load_raw(tmp_path)


@pytest.mark.parametrize("cls", [TLearner, XLearner, HLearner])
def test_learner_contract(cls, data, tmp_path):
    kwargs = {"n_rounds": 30, "lambdas": (0.0, 0.5, 1.0)} if cls is HLearner else {}
    X, t, y = data[S.FEATURES], data[S.TREATMENT], data[S.LABEL]
    model = cls(**kwargs).fit(X, t, y)
    pred = model.predict(X)
    assert list(pred.columns) == ["p0", "p1", "uplift_score"] and len(pred) == len(X)
    assert pred[["p0", "p1"]].stack().between(0, 1).all()
    assert np.allclose(pred["uplift_score"], pred["p1"] - pred["p0"])
    assert pred["uplift_score"].std() > 0

    model.save(tmp_path / "m.pkl")
    again = cls.load(tmp_path / "m.pkl").predict(X)
    pd.testing.assert_frame_equal(pred, again)


def test_hlearner_lambda_zero_fits_each_arm_only(data):
    # lam = 0이면 f1은 처치군 결과에만, f0은 대조군 결과에만 맞춰진다 (T-learner와 같은 목적).
    X = data[S.FEATURES].to_numpy(float)
    t, y = data[S.TREATMENT].to_numpy(float), data[S.LABEL].to_numpy(float)
    model = HLearner(lambdas=(0.0,), n_rounds=40).fit(X, t, y)
    f0, f1 = model.booster_.predict(X)
    assert np.mean((f1 - y)[t == 1] ** 2) < np.var(y[t == 1])
    assert np.mean((f0 - y)[t == 0] ** 2) < np.var(y[t == 0])
    flipped = HLearner(lambdas=(0.0,), n_rounds=40).fit(X, t, np.where(t == 0, 1 - y, y))
    assert np.allclose(flipped.booster_.predict(X)[1], f1)  # 대조군 결과를 바꿔도 f1은 그대로


def test_qini_perfect_beats_random():
    rng = np.random.default_rng(0)
    n = 4000
    t = rng.integers(0, 2, n)
    tau = np.where(rng.random(n) < 0.3, 0.5, 0.0)
    y = (rng.random(n) < 0.2 + tau * t).astype(float)
    assert qini_coefficient(tau, t, y) > 0.03
    assert abs(qini_coefficient(rng.random(n), t, y)) < 0.02
    curve = qini_curve(tau, t, y)
    assert curve["qini"].iloc[-1] == pytest.approx(y[t == 1].mean() - y[t == 0].mean() * 1.0, abs=1e-9)


def test_decile_qini_compares_equal_fractions_of_each_arm():
    score = np.arange(30, 0, -1)
    t = np.r_[np.ones(10), np.zeros(20)]
    y = np.zeros(30)
    y[[0, -1]] = 1
    curve = qini_decile_curve(score, t, y)
    assert curve["qini"].iloc[1] == pytest.approx(0.10)
    assert curve["qini"].iloc[-1] == pytest.approx(0.05)
    assert qini_area(curve) == pytest.approx(0.0675)

    tied = qini_decile_curve(np.ones(30), t, y)
    np.testing.assert_allclose(tied["qini"], tied["random"], atol=1e-15)


def test_simulator_sqlite_cohort(tmp_path):
    from uplift.common.simulator import load_feature_dataset

    keys = pd.DataFrame({"customer_id": list("ABCD"), "snapshot_id": list("abcd"),
                         "dataset_id": "run", "as_of_at": T0.isoformat()})
    features = keys.assign(**{name: [1, 2, None, 4] for name in S.KIT39})
    experiment = keys.assign(
        assigned_at=T0.isoformat(), assignment_probability=0.5, treatment=[0, 1, 0, 1],
        eligibility_definition="session start in [as_of_at-28d,as_of_at)",
        eligible_at_assignment=[1, 1, 1, 0], outcome_horizon_days=28,
        outcome_observed=[1, 1, 0, 1], factual_no_order=[0, 1, None, 1],
    )
    registry = pd.DataFrame({"ordinal": range(1, 40), "feature_name": S.KIT39,
                             "feature_set_version": "kit39-review-cart-v3"})
    with sqlite3.connect(tmp_path / "features.sqlite") as conn:
        features.iloc[::-1].to_sql("features", conn, index=False)
        registry.to_sql("feature_registry", conn, index=False)
        experiment.to_sql("experiment_log", conn, index=False)
    conn.close()

    data = load_feature_dataset(tmp_path).set_index("customer_id").sort_index()
    assert data.index.tolist() == list("ABC")
    assert data.y.iloc[:2].tolist() == [1, 0] and pd.isna(data.y.iloc[2])
    assert data[S.TREATMENT].tolist() == [0, 1, 0]
    assert data.loc["B", S.FEATURES].eq(2).all() and data.loc["C", S.FEATURES].isna().all()
