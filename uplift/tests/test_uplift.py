import numpy as np
import pandas as pd
import pytest

from uplift.common import schema as S
from uplift.common.evaluate import qini_coefficient, qini_curve
from uplift.common.features import build_dataset
from uplift.common.synthetic import make_synthetic
from uplift.Hlearner import HLearner
from uplift.Tlearner import TLearner
from uplift.Xlearner import XLearner


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(1500, seed=1)


@pytest.fixture(scope="module")
def data(raw):
    customers, weekly, cart, _ = raw
    return build_dataset(customers, weekly, cart)


def test_dataset_shape(data):
    assert list(data.columns) == [S.CUSTOMER_ID, S.W0, S.TREATMENT, *S.FEATURES, S.LABEL]
    assert len(data) == 1500 and data[S.CUSTOMER_ID].is_unique
    assert data[S.LABEL].isin([0, 1]).all()


def test_features_by_hand():
    w0 = pd.Timestamp("2026-06-01")
    week = lambda o: w0 + pd.Timedelta(weeks=o)
    rows = []
    for o in range(-12, 4):
        r = {S.CUSTOMER_ID: 7, S.REFERENCE_WEEK: week(o), "sessions": 0.0, "purchases": 0.0,
             "net_revenue": 0.0, "views": 0.0, "session_minutes": 0.0}
        if o < -9:  # 가입 전
            r.update(dict.fromkeys(S.WEEKLY_COLUMNS, np.nan))
        rows.append(r)
    weekly = pd.DataFrame(rows).set_index(S.REFERENCE_WEEK)
    weekly.loc[week(-6), ["sessions", "purchases", "net_revenue"]] = [2, 2, 3000]
    weekly.loc[week(-2), ["sessions", "purchases", "net_revenue", "views", "session_minutes"]] = [
        3, 1, 1000, 12, 30]
    weekly.loc[week(-1), ["sessions", "views"]] = [1, 4]
    weekly.loc[week(2), "purchases"] = 1
    customers = pd.DataFrame({S.CUSTOMER_ID: [7], S.TREATMENT: [1], S.W0: [w0]})
    cart = pd.DataFrame({S.CUSTOMER_ID: [7], "cart_qty": [2], "cart_sku": [1], "cart_gross_value": [500]})

    row = build_dataset(customers, weekly.reset_index(), cart).iloc[0]
    expected = {
        "weeks_since_last_purchase": 2, "weeks_since_last_session": 1, "purchase_count_4w": 1,
        "session_count_4w": 4, "observed_tenure_weeks": 9, "view_count_4w": 16,
        "net_spend_4w": 1000, "mean_net_order_value_4w": 1000, "mean_session_minutes_4w": 7.5,
        "views_per_session_4w": 4, "orders_per_session_4w": 0.25, "purchase_weeks_12w": 2,
        "session_weeks_12w": 3, "cart_qty_w0": 2, "cart_sku_w0": 1, "cart_gross_value_w0": 500,
        "orders_delta_4w": -1, "net_spend_delta_4w": -2000, S.LABEL: 1,
    }
    for name, value in expected.items():
        assert row[name] == pytest.approx(value), name


def test_future_weeks_do_not_change_features(raw, data):
    customers, weekly, cart, _ = raw
    w0 = customers[S.W0].iloc[0]
    changed = weekly.copy()
    future = changed[S.REFERENCE_WEEK] >= w0
    changed.loc[future, S.WEEKLY_COLUMNS] = 99.0
    pd.testing.assert_frame_equal(build_dataset(customers, changed, cart)[S.FEATURES], data[S.FEATURES])


def test_unobserved_outcome_is_null(raw):
    customers, weekly, cart, _ = raw
    w0 = customers[S.W0].iloc[0]
    cut = weekly[weekly[S.REFERENCE_WEEK] < w0 + pd.Timedelta(weeks=3)]  # 결과 주 1개 누락
    assert build_dataset(customers, cut, cart)[S.LABEL].isna().all()


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
