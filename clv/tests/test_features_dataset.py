"""WBS 4.2-4.3: 13주 입력·52주 정답 구간, 정답, 누수 방지, 예측 변수, 고객 단위 분할을 검사한다."""
import numpy as np
import pandas as pd
import pytest

from clv.adapter import adapt_weekly
from clv.baseline import BaselineModel
from clv.dataset import (
    build_dataset, build_labels, complete_grid, customer_windows, find_eligible_cutoffs, labeled_rows,
    ledger_label_sums, select_cutoff, split_customers,
)
from clv.features import build_features, to_cutoff
from clv.schema import NO_EVENT_RECENCY, PREDICTOR_FEATURES, check_predictors, forbidden_predictors

from .conftest import clv_rows

START = pd.Timestamp("2024-03-04", tz="Asia/Seoul")
T0 = START + pd.Timedelta(weeks=13)
ZERO = (0, 0, 0, 0)


def hand_panel():
    """손으로 값을 계산할 수 있는 4명 패널(기준시점 T0 = START + 13주).

    a: 13주 완전 이력, 정답 구간은 4주마다 100. b: 전 구간 활동 없음.
    c: 입력 5주만 관측, 마지막 정답 주 미관측. d: t0에 가입해 이력 없음.
    """
    a_input = [(2, 1, 1000, 0), (1, 0, 0, 0), ZERO, (3, 2, 3000, 300), ZERO, ZERO, ZERO, ZERO, ZERO,
               (1, 0, 0, 0), (2, 1, 2000, 200), ZERO, ZERO]
    a_label = [(1, 1, 100, 0) if offset % 4 == 0 else ZERO for offset in range(52)]
    c_input = [None] * 8 + [(1, 1, 500, 0), ZERO, ZERO, ZERO, ZERO]
    weeks = {
        "a": a_input + a_label,
        "b": [ZERO] * 65,
        "c": c_input + [ZERO] * 51 + [None],          # 마지막 정답 주가 미관측 주
        "d": [None] * 13 + [(1, 1, 700, 0)] + [ZERO] * 51,  # t0에 가입: 이력 없음
    }
    return complete_grid(adapt_weekly(clv_rows(weeks, start=str(START.date()))))


@pytest.fixture(scope="module")
def hand():
    """모듈에서 함께 쓰는 ``hand_panel()``."""
    return hand_panel()


def test_hand_computed_features(hand):
    features = build_features(hand, T0)
    assert list(features.index) == ["a", "b", "c"]  # d는 t0 이전 관측 주가 없다
    assert list(features.columns[:len(PREDICTOR_FEATURES)]) == list(PREDICTOR_FEATURES)
    a = features.loc["a"]
    expected = {"purchase_count_4w": 1, "purchase_count_13w": 4, "session_count_4w": 3, "session_count_13w": 9,
                "net_spend_4w": 1800, "net_spend_13w": 5500, "weeks_since_last_purchase": 3,
                "purchase_active_weeks_13w": 3, "zero_purchase_streak": 2, "purchase_trend": 1 / 4 - 3 / 9,
                "spend_trend": 1800 / 4 - 3700 / 9, "purchase_per_session_13w": 4 / 9, "weeks_since_last_session": 3,
                "session_active_weeks_13w": 5, "session_trend": 3 / 4 - 6 / 9, "avg_paid_per_purchase_13w": 1375,
                "coupon_sent_13w": 0}
    for name, value in expected.items():
        assert a[name] == pytest.approx(value), name
    assert a["history_weeks_13w"] == 13 and a["feature_window_complete"]


def test_no_purchase_customer_policy(hand):
    """구매 없는 고객: 합계와 추세는 0, 최근성은 14, 분모가 0인 비율은 0이 아니라 NaN."""
    b = build_features(hand, T0).loc["b"]
    assert b["purchase_count_13w"] == 0 and b["net_spend_13w"] == 0
    assert b["weeks_since_last_purchase"] == NO_EVENT_RECENCY == 14
    assert b["weeks_since_last_session"] == NO_EVENT_RECENCY
    assert b["zero_purchase_streak"] == 13
    assert b["purchase_trend"] == 0 and b["spend_trend"] == 0
    assert np.isnan(b["purchase_per_session_13w"]) and np.isnan(b["avg_paid_per_purchase_13w"])


def test_partial_history_is_not_zero_filled(hand):
    """부분 이력 고객의 미관측 주는 0으로 채우지 않는다. 추세와 Baseline의 분모는 관측 주 수다."""
    c = build_features(hand, T0).loc["c"]
    assert c["history_weeks_13w"] == 5 and not c["feature_window_complete"]
    assert c["history_weeks_4w"] == 4 and c["history_weeks_prev9w"] == 1
    assert c["purchase_trend"] == pytest.approx(0 / 4 - 1 / 1)  # 9가 아니라 관측 주 수로 나눈다
    assert c["zero_purchase_streak"] == 4
    baseline = BaselineModel().predict(build_features(hand, T0).loc[["a", "c"]])
    assert baseline[0] == pytest.approx(5500 / 13 * 52)
    assert baseline[1] == pytest.approx(500 / 5 * 52)


def test_labels_cover_exactly_52_weeks_from_t0(hand):
    """정답은 [t0, t0 + 52주) 결제액 합이고, 정답 구간에 미관측 주가 하나라도 있으면 만들지 않는다(NA)."""
    labels = build_labels(hand, T0)
    assert labels.loc["a", "actual_clv_12m"] == 1300
    assert labels.loc["b", "actual_clv_12m"] == 0
    assert labels.loc["d", "actual_clv_12m"] == 700
    assert pd.isna(labels.loc["c", "actual_clv_12m"]) and not labels.loc["c", "label_window_complete"]
    assert labels.loc["c", "label_weeks_observed"] == 51


def test_65_week_windows_split_13_input_and_52_label(hand):
    windows = customer_windows(hand, T0)
    a = windows.loc[windows["customer_id"].eq("a")]
    assert len(a) == 65 and a["offset_week"].min() == -13 and a["offset_week"].max() == 51
    assert (a["window_part"] == "input").sum() == 13 and (a["window_part"] == "label").sum() == 52
    assert a.loc[a["window_part"].eq("input"), "week"].max() < T0 <= a.loc[a["window_part"].eq("label"), "week"].min()


def test_future_rows_do_not_change_features(hand):
    """누수 방지: t0 이후 행을 바꿔도 예측 변수가 그대로이고,
    ID·정답·실험 배정·gap·잠재 상태·사후 결과 열은 예측 변수로 쓸 수 없다."""
    before = build_features(hand, T0)
    future = hand.copy()
    future.attrs = dict(hand.attrs)
    after_t0 = future["week"].ge(T0) & future["observed_week"]
    future.loc[after_t0, ["purchases", "sessions", "coupons"]] = 99
    future.loc[after_t0, ["gross_revenue", "paid_amount"]] = 10 ** 6
    pd.testing.assert_frame_equal(before, build_features(future, T0))
    dataset = build_dataset(hand, T0)
    assert forbidden_predictors(PREDICTOR_FEATURES) == []
    for column in ("customer_id", "reference_week", "actual_clv_12m", "treatment", "gap_days", "future_revenue_12m",
                   "latent_trait", "posthoc_response_type", "week"):
        assert forbidden_predictors([column]) == [column]
    with pytest.raises(ValueError):
        check_predictors([*PREDICTOR_FEATURES, "actual_clv_12m"])
    assert not set(PREDICTOR_FEATURES) & {"reference_week", "customer_id", "actual_clv_12m"} and len(dataset) == 3


def test_cutoff_must_be_a_seoul_monday():
    assert to_cutoff("2024-06-03") == pd.Timestamp("2024-06-03", tz="Asia/Seoul")
    # UTC 일요일 15:00 = 서울 월요일 00:00
    assert to_cutoff(pd.Timestamp("2024-06-02T15:00:00Z")) == pd.Timestamp("2024-06-03", tz="Asia/Seoul")
    with pytest.raises(ValueError, match="Monday"):
        to_cutoff("2024-06-04")


def test_eligible_cutoffs_respect_partial_edge_weeks(panel):
    """기본 기준시점은 적격 고객이 가장 많은 t0이고, 적격 고객이 없는 t0는 지정해도 거부한다."""
    cutoffs = find_eligible_cutoffs(panel)
    weeks = pd.DatetimeIndex(panel["week"].unique()).sort_values()
    assert cutoffs["reference_week"].iloc[0] == weeks[13]
    assert cutoffs["eligible_customers"].iloc[0] == 0      # 첫 입력 주가 부분 관측 주
    assert cutoffs["eligible_customers"].iloc[-1] == 0     # 마지막 정답 주가 부분 관측 주
    assert cutoffs["eligible_customers"].max() > 0
    chosen = select_cutoff(cutoffs)
    assert cutoffs.set_index("reference_week").loc[chosen, "eligible_customers"] == cutoffs["eligible_customers"].max()
    with pytest.raises(ValueError, match="no customer"):
        select_cutoff(cutoffs, cutoffs["reference_week"].iloc[0])


def test_labels_match_ledger_and_mismatch_raises(panel, orders):
    """정답은 주문 원장 합계와 고객별로 같아야 하며 다르면 오류다.
    기본 정답 행(학습·평가 대상)은 13주 이력이 완전한 고객뿐이다."""
    t0 = select_cutoff(find_eligible_cutoffs(panel))
    dataset = build_dataset(panel, t0, orders)
    labelled = labeled_rows(dataset)
    assert labelled["feature_window_complete"].all() and labelled["label_window_complete"].all()
    ledger = ledger_label_sums(orders, t0).reindex(labelled["customer_id"]).fillna(0).to_numpy()
    np.testing.assert_array_equal(labelled["actual_clv_12m"].astype(float).to_numpy(), ledger)
    assert dataset["ledger_checked"].sum() == dataset["label_window_complete"].sum()
    tampered = orders.copy()
    tampered.attrs = dict(orders.attrs)
    # 정답 구간 안 주문 하나의 결제액만 바꾼다.
    inside = tampered.index[tampered["order_at"].ge(t0)][0]
    tampered.loc[inside, "paid_amount"] += 5
    with pytest.raises(ValueError, match="ledger"):
        build_dataset(panel, t0, tampered)
    # 늦게 가입해 입력 구간이 완전하지 않은 고객은 기본 정답 행에서 빠진다.
    late = dataset.loc[~dataset["feature_window_complete"]]
    assert len(late) and not labeled_rows(dataset)["customer_id"].isin(late["customer_id"]).any()
    assert len(labeled_rows(dataset, require_complete_history=False)) > len(labelled)


def test_treatment_is_context_not_a_predictor(panel):
    t0 = select_cutoff(find_eligible_cutoffs(panel))
    dataset = build_dataset(panel, t0)
    assert {"treatment", "treatment_in_label_window"} <= set(dataset)
    # 배정 주가 정답 구간 안에 있으므로 Treatment 고객은 모두 표시된다.
    assert dataset["treatment_in_label_window"].sum() == dataset["treatment"].eq(1).sum()
    assert "treatment" not in PREDICTOR_FEATURES


def test_customer_split_is_disjoint_complete_and_deterministic():
    ids = [f"c{index}" for index in range(101)]
    split = split_customers(ids, seed=7)
    assert set(split.index) == set(ids) and split.index.is_unique
    counts = split.value_counts()
    # 101명 × 0.6/0.2 반올림: train 61, validation 20, 나머지 20은 test
    assert counts["train"] == 61 and counts["validation"] == 20 and counts["test"] == 20
    # 입력 순서와 무관하고 seed가 바뀌면 분할도 바뀐다.
    pd.testing.assert_series_equal(split, split_customers(list(reversed(ids)), seed=7))
    assert not split.equals(split_customers(ids, seed=8))
    with pytest.raises(ValueError):  # 3명 미만이면 세 분할을 만들 수 없다
        split_customers(["a", "b"])
