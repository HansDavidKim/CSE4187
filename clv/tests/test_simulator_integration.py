"""시뮬레이터 동봉 population 설정으로 만든 데이터에 대한 계약 테스트(WBS 4.1~4.7 전 과정).

``population.yaml``(동봉 집계 모델, 외부 자산 불필요) 60명 × 24개월로 스키마·금액·관측·원장 배선을
처음부터 끝까지 확인한다. 표본이 작아 모델 지표에는 의미가 없다.
"""
import json

import pandas as pd
import pytest

from clv.config import CLVConfig
from clv.dataset import build_dataset, find_eligible_cutoffs, labeled_rows
from clv.pipeline import predict, prepare_inputs, train, write_predictions
from clv.api import load_clv_predictions

from .simulator_fixture import generate, import_simulator

# 배선 확인용 최소 XGBoost 설정
FAST = {"xgb_grid": {"objective": ["reg:squarederror"], "max_depth": [2], "learning_rate": [0.1]},
        "xgb_fixed": {"n_estimators": 100, "early_stopping_rounds": 10, "n_jobs": 1}}


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """시뮬레이터로 60명·24개월 데이터를 만들어 ``(dataset, features)`` 경로를 반환한다(코드가 없으면 skip)."""
    try:
        import_simulator()
    except Exception as error:  # sklearn이나 시뮬레이터 코드가 없는 경우
        pytest.skip(f"simulator code path unavailable: {error}")
    folder = tmp_path_factory.mktemp("simulator-v5")
    return generate(folder, n_customers=60, months=24)


def test_simulator_outputs_pass_clv_validation_and_ledger_reconciliation(simulated):
    dataset, features = simulated
    panel, orders, reports = prepare_inputs(features, dataset)
    assert all(report.ok for report in reports), [report.errors for report in reports]
    by_name = {report.name: report for report in reports}
    assert by_name["ledger_vs_weekly"].stats["weekly_paid"] == by_name["ledger_vs_weekly"].stats["ledger_paid"]
    manifest = json.loads((dataset / "raw.manifest.json").read_text())["metadata"]
    assert panel.attrs["currency"] == manifest["currency"] == orders.attrs["currency"]
    assert panel.attrs["amount_unit"] == manifest["amount_unit"] == orders.attrs["amount_unit"]
    # 실제 시뮬레이터 출력에도 부분 관측 주와 미관측 주가 있다.
    assert panel["partial_coverage"].any() and (~panel["observed_week"]).any()


def test_simulator_weekly_panel_maps_coupon_columns(simulated):
    """coupons_sent·coupon_actions·net_revenue_minor가 합계 그대로 CLV 열(coupons·coupon_action·결제액)로 옮겨진다."""
    _, features = simulated
    raw = pd.read_parquet(features / "weekly_panel.parquet")
    panel, _, _ = prepare_inputs(features)
    assert int(panel["coupons"].sum()) == int(raw["coupons_sent"].sum())
    assert int(panel["coupon_action"].sum()) == int(raw["coupon_actions"].sum())
    assert int(panel["paid_amount"].sum()) == int(raw["net_revenue_minor"].sum())


def test_clv_labels_from_simulator_match_raw_orders(simulated):
    dataset, features = simulated
    panel, orders, _ = prepare_inputs(features, dataset)
    cutoffs = find_eligible_cutoffs(panel)
    t0 = cutoffs.loc[cutoffs["eligible_customers"].idxmax(), "reference_week"]
    rows = labeled_rows(build_dataset(panel, t0, orders))  # 정답 하나라도 원장과 다르면 여기서 오류
    assert len(rows) > 0


def test_end_to_end_on_simulator_outputs(simulated, tmp_path):
    pytest.importorskip("xgboost")
    dataset, features = simulated
    registry = train(features, dataset, tmp_path / "clv", CLVConfig.from_dict(FAST))
    assert registry["model_status"]["baseline"] == "trained"
    assert registry["model_status"]["xgboost"] == "trained"
    # 표본이 작아 BG/NBD + GG는 최소 적합 인원 미달로 failed일 수 있다.
    assert registry["model_status"]["bgnbd_gg"] == "trained" or registry["model_status"]["bgnbd_gg"].startswith("failed")
    predictions = predict(features, dataset, tmp_path / "clv")
    experiment = pd.read_parquet(features / "experiment_log.parquet")
    as_of = pd.Timestamp(experiment["as_of_at"].iloc[0]).tz_convert("Asia/Seoul").isoformat()
    assert predictions["reference_week"].eq(as_of).all()
    path = write_predictions(predictions, tmp_path / "results" / "clv_predictions.csv")
    loaded = load_clv_predictions(path)
    assert set(loaded["customer_id"]) <= set(experiment["customer_id"])
    assert loaded["is_high_value_customer"].sum() == int(-(-len(loaded) * 2 // 10))  # ceil(0.2·N)
