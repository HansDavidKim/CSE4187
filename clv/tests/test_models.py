"""WBS 4.4-4.5: 평균 Baseline, BG/NBD + Gamma-Gamma, XGBoost의 적합·예측·저장·재현을 검사한다."""
import numpy as np
import pandas as pd
import pytest

from clv.baseline import BaselineModel
from clv.dataset import build_dataset, find_eligible_cutoffs, labeled_rows, select_cutoff, split_customers
from clv.features import WEEK
from clv.probabilistic import BGNBD, GammaGamma, ProbabilisticCLV, rfm_summary
from clv.schema import PREDICTOR_FEATURES


def simulate_bgnbd(n=3000, r=0.8, alpha=5.0, a=1.5, b=6.0, seed=3):
    """BG/NBD 생성 과정에서 (x, t_x, T)를 뽑는다(시간 단위는 주)."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(30, 52, n)
    rate, dropout = rng.gamma(r, 1 / alpha, n), rng.beta(a, b, n)
    x, tx = np.zeros(n), np.zeros(n)
    for index in range(n):
        now = 0.0
        while True:
            now += rng.exponential(1 / rate[index])
            if now > T[index]:
                break
            x[index] += 1
            tx[index] = now
            if rng.random() < dropout[index]:
                break
    return x, tx, T


def simulate_gamma_gamma(n=2000, p=6.0, q=4.0, v=15.0, seed=4):
    """Gamma-Gamma 생성 과정에서 구매 수 x와 고객별 평균 결제액 m을 뽑는다."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 10, n).astype(float)
    nu = rng.gamma(q, 1 / v, n)
    m = np.array([rng.gamma(p, 1 / nu[i], int(x[i])).mean() for i in range(n)])
    return x, m


@pytest.fixture(scope="module")
def labelled(panel, orders):
    """기본 기준시점의 ``(t0, 정답 행)``. 정답 행에는 고객 단위 분할(seed 42)이 붙어 있다."""
    t0 = select_cutoff(find_eligible_cutoffs(panel))
    rows = labeled_rows(build_dataset(panel, t0, orders))
    rows["split"] = rows["customer_id"].map(split_customers(rows["customer_id"], seed=42))
    rows["actual_clv_12m"] = rows["actual_clv_12m"].astype(float)
    return t0, rows


def test_baseline_formula_and_save(tmp_path):
    """Baseline = net_spend_13w / 관측 주 수 × 52이고 관측 주가 0이면 0이다. 저장 후 다시 읽어도 같다."""
    frame = pd.DataFrame({"net_spend_13w": [1300.0, 0.0, 260.0], "history_weeks_13w": [13.0, 13.0, 0.0]})
    np.testing.assert_allclose(BaselineModel().predict(frame), [1300 / 13 * 52, 0.0, 0.0])
    BaselineModel().save(tmp_path)
    np.testing.assert_allclose(BaselineModel.load(tmp_path).predict(frame), BaselineModel().predict(frame))


def test_bgnbd_recovers_generating_parameters():
    x, tx, T = simulate_bgnbd()
    model = BGNBD()
    info = model.fit(x, tx, T)
    assert info["converged"]
    params = model.params
    assert params["r"] == pytest.approx(0.8, rel=0.25) and params["alpha"] == pytest.approx(5.0, rel=0.25), params
    # a, b는 각각은 식별이 약하지만 평균 이탈 확률 a/(a+b)는 잘 식별된다.
    assert params["a"] / (params["a"] + params["b"]) == pytest.approx(1.5 / 7.5, rel=0.15), params
    assert params["a"] == pytest.approx(1.5, rel=0.5) and params["b"] == pytest.approx(6.0, rel=0.5), params
    expected = model.expected_purchases(52, x, tx, T)
    assert np.isfinite(expected).all() and (expected >= 0).all()
    alive = model.probability_alive(x, tx, T)
    # 반복 구매가 없으면 BG/NBD 정의상 이탈 기회가 없어 생존 확률이 1이다.
    assert ((alive >= 0) & (alive <= 1)).all() and (alive[x == 0] == 1).all()


def test_gamma_gamma_recovers_generating_parameters():
    x, m = simulate_gamma_gamma()
    model = GammaGamma()
    assert model.fit(x, m)["converged"]
    assert model.params["p"] == pytest.approx(6.0, rel=0.25)
    assert model.population_mean() == pytest.approx(6.0 * 15.0 / 3.0, rel=0.2)  # p·v/(q−1)
    # 반복 구매 0회 고객은 모집단 평균을 받는다.
    assert model.expected_value([0.0], [np.nan])[0] == pytest.approx(model.population_mean())


def test_matches_lifetimes_reference_when_installed():
    """Lifetimes가 설치돼 있으면 모수, 52주 기대 구매수, 기대 결제액이 2% 이내로 일치한다."""
    lifetimes = pytest.importorskip("lifetimes")
    x, tx, T = simulate_bgnbd(n=1500, seed=11)
    ours = BGNBD()
    ours.fit(x, tx, T)
    try:
        reference = lifetimes.BetaGeoFitter(penalizer_coef=0.0).fit(x, tx, T)
    except Exception as error:  # Lifetimes는 유지보수가 끊겨 버전마다 수렴 여부가 다르다
        pytest.skip(f"lifetimes BG/NBD fit failed: {error}")
    for name in ("r", "alpha", "a", "b"):
        assert ours.params[name] == pytest.approx(reference.params_[name], rel=0.02), name
    expected = reference.conditional_expected_number_of_purchases_up_to_time(52, x, tx, T)
    np.testing.assert_allclose(ours.expected_purchases(52, x, tx, T), expected, rtol=0.02)
    fx, fm = simulate_gamma_gamma(n=1000, seed=12)
    gg = GammaGamma()
    gg.fit(fx, fm)
    try:
        ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.0).fit(fx, fm)
    except Exception as error:
        pytest.skip(f"lifetimes Gamma-Gamma fit failed: {error}")
    np.testing.assert_allclose(gg.expected_value(fx, fm),
                               ggf.conditional_expected_average_profit(pd.Series(fx), pd.Series(fm)), rtol=0.02)


def test_rfm_summary_uses_only_calibration_orders(orders, labelled):
    """RFM 요약은 [t0 - 13주, t0) 주문만 쓰고 T는 13주 이하로 잘린다(좌측 절단).
    t0 이후 주문을 바꿔도 결과가 같다(누수 없음)."""
    t0, _ = labelled
    summary = rfm_summary(orders, t0)
    window = orders.loc[orders["order_at"].ge(t0 - 13 * WEEK) & orders["order_at"].lt(t0)]
    assert set(summary.index) == set(window["customer_id"])
    assert (summary["T"] <= 13).all() and (summary["recency"] <= summary["T"]).all()
    assert (summary["frequency"] >= 0).all()
    assert summary["calibration_paid"].sum() == window["paid_amount"].sum()
    full = rfm_summary(orders, t0, calibration="full_history")
    assert len(full) >= len(summary) and full["T"].max() > 13
    future = orders.copy()
    future.loc[future["order_at"].ge(t0), "paid_amount"] *= 7
    pd.testing.assert_frame_equal(summary, rfm_summary(future, t0))


def test_probabilistic_fit_predict_status_save_and_reproduce(orders, labelled, tmp_path):
    """Train 고객으로만 적합한다. 보정 기간 구매가 없는 고객은 status로 표시하고 기본값 0을,
    ``train_mean``이면 같은 처지 Train 고객의 평균 정답을 받는다. 재적합·재로딩해도 결과가 같다."""
    t0, rows = labelled
    train = rows.loc[rows["split"].eq("train"), "customer_id"]
    model = ProbabilisticCLV().fit(orders, t0, train)
    checks = model.diagnostics["checks"]
    assert checks["bgnbd_converged"] and checks["gamma_gamma_converged"] and checks["q_gt_1"]
    assert model.diagnostics["customers_without_purchase"] > 0
    out = model.predict(orders, t0, rows["customer_id"])
    assert set(out["status"]) <= {"modeled", "no_purchase_in_calibration", "numerical_failure"}
    assert (out["predicted_clv_12m"] >= 0).all() and np.isfinite(out["predicted_clv_12m"]).all()
    assert (out.loc[out["status"].eq("no_purchase_in_calibration"), "predicted_clv_12m"] == 0).all()
    again = ProbabilisticCLV().fit(orders, t0, train)
    assert again.bgnbd.params == model.bgnbd.params and again.gamma_gamma.params == model.gamma_gamma.params
    model.save(tmp_path)
    reloaded = ProbabilisticCLV.load(tmp_path)
    pd.testing.assert_frame_equal(out, reloaded.predict(orders, t0, rows["customer_id"]))
    labels = rows.set_index("customer_id")["actual_clv_12m"]
    mean_model = ProbabilisticCLV(no_purchase_fallback="train_mean").fit(orders, t0, train, labels=labels)
    non_buyers = out.index[out["status"].eq("no_purchase_in_calibration")]
    expected = labels.reindex(train).loc[lambda s: s.index.isin(non_buyers)].mean()
    assert mean_model.fallback_value == pytest.approx(expected)


def test_probabilistic_requires_enough_buyers(orders, labelled):
    """최소 적합 인원(구매 고객 수)에 못 미치면 적합하지 않고 오류를 낸다."""
    t0, rows = labelled
    with pytest.raises(ValueError, match="at least"):
        ProbabilisticCLV().fit(orders.iloc[:0], t0, rows["customer_id"])


def test_xgboost_train_reload_reproduce_and_feature_contract(labelled, tmp_path):
    """seed와 n_jobs=1을 고정하면 같은 입력에 같은 예측이 나온다. 재로딩·열 순서 변경에도 같고,
    예측 변수가 빠지면 오류다."""
    pytest.importorskip("xgboost")
    from clv.xgboost_model import XGBoostCLV

    _, rows = labelled
    grid = {"objective": ["reg:squarederror", "reg:tweedie"], "max_depth": [2, 3], "learning_rate": [0.1]}
    fixed = {"n_estimators": 200, "early_stopping_rounds": 20, "subsample": 0.8, "tree_method": "hist", "n_jobs": 1}
    train, valid = rows.loc[rows["split"].eq("train")], rows.loc[rows["split"].eq("validation")]
    model = XGBoostCLV(seed=42, grid=grid, fixed=fixed).fit(train, train["actual_clv_12m"], valid, valid["actual_clv_12m"])
    assert len(model.grid_results) == 4 and model.best_params["n_estimators"] >= 1
    predicted = model.predict(rows)
    assert (predicted >= 0).all()
    same = XGBoostCLV(seed=42, grid=grid, fixed=fixed).fit(train, train["actual_clv_12m"], valid, valid["actual_clv_12m"])
    np.testing.assert_array_equal(predicted, same.predict(rows))
    model.save(tmp_path / "xgb")
    reloaded = XGBoostCLV.load(tmp_path / "xgb")
    np.testing.assert_array_equal(predicted, reloaded.predict(rows))
    # 열 순서가 달라도 메타데이터의 Feature 순서로 맞춰 예측한다.
    shuffled = rows[list(reversed(rows.columns))]
    np.testing.assert_array_equal(predicted, reloaded.predict(shuffled))
    assert reloaded.feature_names == list(PREDICTOR_FEATURES)
    with pytest.raises(ValueError, match="missing features"):
        reloaded.predict(rows.drop(columns=["net_spend_13w"]))


def test_xgboost_negative_predictions_are_clipped(labelled):
    pytest.importorskip("xgboost")
    from clv.xgboost_model import XGBoostCLV

    _, rows = labelled
    # 중앙값을 빼 목표의 절반을 음수로 만들어 음수 예측이 나오게 한다.
    target = rows["net_spend_13w"].to_numpy() - rows["net_spend_13w"].median()
    grid = {"objective": ["reg:squarederror"], "max_depth": [2], "learning_rate": [0.3]}
    fixed = {"n_estimators": 30, "early_stopping_rounds": 10, "n_jobs": 1}
    model = XGBoostCLV(grid=grid, fixed=fixed).fit(rows, target, rows, target)
    assert (model.predict(rows) >= 0).all() and model.last_clipped > 0


@pytest.mark.parametrize("params", [(0.8, 5.0, 0.6, 3.0), (0.8, 5.0, 1.5, 6.0)])
def test_bgnbd_52_week_expectation_matches_monte_carlo(params):
    """참 모수에서 52주 조건부 기대 구매수의 합이 Monte Carlo 실제 구매수와 맞는다.

    a < 1(생애 거래 수 평균이 발산하는 두꺼운 꼬리)과 a > 1 두 경우 모두 확인한다.
    """
    rng = np.random.default_rng(5)
    n, T, horizon = 10000, 13.0, 52.0
    r, alpha, a, b = params
    rate, dropout = rng.gamma(r, 1 / alpha, n), rng.beta(a, b, n)
    # 13주 보정 기간의 (x, t_x)와 그 뒤 52주 동안의 실제 구매수를 함께 기록한다.
    x, tx, future = np.zeros(n), np.zeros(n), np.zeros(n)
    for index in range(n):
        now = 0.0
        while True:
            now += rng.exponential(1 / rate[index])
            if now > T + horizon:
                break
            if now <= T:
                x[index], tx[index] = x[index] + 1, now
            else:
                future[index] += 1
            if rng.random() < dropout[index]:
                break
    expected = BGNBD(dict(zip(("r", "alpha", "a", "b"), params))).expected_purchases(horizon, x, tx, np.full(n, T))
    assert expected.sum() / future.sum() == pytest.approx(1.0, abs=0.05)


def test_applicability_separates_rejections_from_warnings():
    """q <= 1·미수렴은 모델 제외 사유이고, a + b 과대·상관 과대·반복 구매 고객 부족은 경고만 남긴다."""
    from clv.probabilistic import applicability

    sane = applicability({"r": 0.8, "alpha": 5.0, "a": 1.5, "b": 6.0}, {"p": 6.0, "q": 4.0, "v": 15.0},
                         True, True, 0.1, 200)
    assert sane["rejections"] == [] and sane["warnings"] == []
    assert sane["checks"]["dropout_heterogeneity_identified"] and sane["checks"]["q_gt_1"]
    bad = applicability({"r": 0.45, "alpha": 2.3, "a": 1.9e5, "b": 9.6e5}, {"p": 2.7, "q": 0.9, "v": 5000.0},
                        True, False, 0.5, 10)
    assert any("q = 0.9" in reason for reason in bad["rejections"])
    assert any("Gamma-Gamma optimisation" in reason for reason in bad["rejections"])
    assert len(bad["warnings"]) == 3 and "point mass p = 0.1652" in bad["warnings"][0]
    assert bad["checks"]["dropout_heterogeneity_identified"] is False and bad["checks"]["independence_ok"] is False


def test_probabilistic_diagnostics_record_dropout_concentration(orders, labelled):
    t0, rows = labelled
    model = ProbabilisticCLV().fit(orders, t0, rows.loc[rows["split"].eq("train"), "customer_id"])
    params = model.bgnbd.params
    assert model.diagnostics["dropout_concentration"] == pytest.approx(params["a"] + params["b"])
    assert model.diagnostics["mean_dropout_probability"] == pytest.approx(params["a"] / (params["a"] + params["b"]))
    assert {"checks", "rejections", "warnings"} <= set(model.diagnostics)
