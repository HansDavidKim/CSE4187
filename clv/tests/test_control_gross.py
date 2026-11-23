"""Control-only 학습 모집단, 4주 조건부 총구매액 정답·모델·평가, 확장된 추론 API, 기존 번들 호환을 검사한다.

합성 Fixture는 배선 검증용이며 성능 근거가 아니다.
"""
import json

import numpy as np
import pandas as pd
import pytest

from clv.adapter import adapt_orders, adapt_weekly
from clv.api import (
    GROSS_STRATEGY_COLUMNS, HANDOFF_BUNDLE, STRATEGY_COLUMNS, load_model, load_strategy_predictions, predict_clv,
)
from clv.bundle import file_sha256
from clv.config import CLVConfig
from clv.dataset import build_dataset, build_gross_labels, complete_grid
from clv.pipeline import evaluate, train
from clv.population import policy_mask
from clv.xgboost_model import GrossXGBoost4w

from .conftest import clv_rows
from .fixtures import synthetic_outputs
from .test_inference import FAST

FAST = {**FAST, "gross_4w_grid": {"objective": ["reg:squarederror"], "max_depth": [3], "learning_rate": [0.1]}}
# 아래 Fixture의 실험 시점(배정 주 30 = 서울 2024-07-29 월요일 00:00). 이 t0에서 52주 정답이 관측된다.
AS_OF = pd.Timestamp("2024-07-28T15:00:00Z")
T0 = "2024-07-29"


@pytest.fixture(scope="module")
def experiment():
    """t0 = 실험 시점에서 52주 정답이 관측되는 합성 시뮬레이터 주간 패널(400명, 96주, 짝수 Control·홀수 Treatment)."""
    weekly, _, _ = synthetic_outputs(n_customers=400, n_weeks=96, assignment_week=30, seed=3)
    return weekly


@pytest.fixture(scope="module")
def bundle(experiment, tmp_path_factory):
    """기본 정책(control_no_coupon)으로 t0 = 실험 시점에 학습한 번들."""
    pytest.importorskip("xgboost")
    output = tmp_path_factory.mktemp("control") / "clv"
    registry = train(experiment, None, output, CLVConfig.from_dict(FAST), cutoff=T0)
    return output, registry


def _splits(output):
    return pd.read_csv(output / "splits.csv", dtype={"customer_id": str})


def test_control_only_split_leaves_treatment_unused(bundle, experiment):
    output, registry = bundle
    splits = _splits(output)
    assignment = experiment.drop_duplicates("customer_id").set_index("customer_id")["treatment"]
    splits["assigned"] = splits["customer_id"].map(assignment)
    # Treatment는 분할되지 않고 unused이며, Train/Validation/Test는 모두 Control이다.
    assert splits.loc[splits["assigned"].eq(1), "split"].eq("unused").all()
    assert splits.loc[splits["split"].isin(["train", "validation", "test"]), "assigned"].eq(0).all()
    assert registry["population"]["policy"] == "control_no_coupon"
    assert registry["population"]["eligible"] == registry["population"]["control"] == 200
    assert registry["population"]["treatment"] == 200 and registry["population"]["control_with_coupon_excluded"] == 0
    assert sum(registry["split_counts"].values()) == 200
    card = json.loads((output / "model_card.json").read_text())
    assert card["policy_condition"] == "control_no_coupon"


def test_treatment_customers_do_not_influence_any_model(bundle, experiment, tmp_path):
    """Treatment 고객의 금액을 모두 바꿔 다시 학습해도 52주 모델과 4주 모델 파일이 바이트 단위로 같다."""
    output, _ = bundle
    changed = experiment.copy()
    treated = changed["treatment"].eq(1) & changed["observed_week"]
    changed.loc[treated, "gross_revenue_minor"] = changed.loc[treated, "gross_revenue_minor"] * 3 + 7
    changed["net_revenue_minor"] = changed["gross_revenue_minor"] - changed["redemption_discount_cost_minor"]
    other = tmp_path / "changed"
    train(changed, None, other, CLVConfig.from_dict(FAST), cutoff=T0)
    for name in ("models/baseline.json", "models/xgboost/xgboost_model.json", "models/xgboost_two_part/classifier.json",
                 "models/xgboost_two_part/regressor.json", "models/xgboost_gross_4w/regressor.json"):
        assert file_sha256(output / name) == file_sha256(other / name), name
    pd.testing.assert_frame_equal(_splits(output), _splits(other))


def test_inference_scores_treatment_and_control(bundle, experiment):
    output, _ = bundle
    predictions = predict_clv(experiment, model_dir=output, reference_week=T0)
    assignment = experiment.drop_duplicates("customer_id").set_index("customer_id")["treatment"]
    groups = predictions["customer_id"].map(assignment)
    assert set(groups) == {0, 1} and groups.eq(1).sum() == 200
    for column in ("predicted_clv_12m", "predicted_gross_4w_if_purchase"):
        values = predictions[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and (values >= 0).all()
    assert predictions["gross_4w_status"].eq("trained").all()
    assert predictions["reference_week"].eq("2024-07-29T00:00:00+09:00").all()
    assert predictions["gross_4w_prediction_end_at"].eq("2024-08-26T00:00:00+09:00").all()
    assert predictions["prediction_end_at"].eq("2025-07-28T00:00:00+09:00").all()


def _four_week_panel():
    """입력 13주 + 4주 정답 + 그 뒤 1주. (sessions, purchases, gross, discount), None은 미관측 주."""
    quiet = [(1, 0, 0, 0)] * 13
    return clv_rows({
        # 반복 주문 3건(1주차 2건, 3주차 1건), 할인 포함: 총구매액은 할인 전 1000 + 500 = 1500
        "repeat": quiet + [(2, 2, 1000, 100), (0, 0, 0, 0), (1, 1, 500, 50), (0, 0, 0, 0), (0, 0, 0, 0)],
        # 4주 안 구매 없음(5주차 구매는 구간 밖): 구매 0회, 금액 정답 결측
        "none": quiet + [(0, 0, 0, 0)] * 4 + [(1, 1, 900, 0)],
        # 4주 중 1주 미관측: 관측 불완전, 두 값 모두 결측
        "gap": quiet + [(1, 1, 300, 0), None, (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
    })


def test_four_week_labels_sum_repeat_orders_before_discount():
    panel = complete_grid(adapt_weekly(_four_week_panel()))
    t0 = pd.Timestamp("2024-03-04", tz="Asia/Seoul") + pd.Timedelta(weeks=13)
    labels = build_gross_labels(panel, t0)
    assert labels.loc["repeat", "actual_purchases_4w"] == 3
    assert labels.loc["repeat", "actual_gross_4w_if_purchase"] == 1500  # 할인(150) 미차감
    assert labels.loc["none", "actual_purchases_4w"] == 0 and pd.isna(labels.loc["none", "actual_gross_4w_if_purchase"])
    assert not labels.loc["gap", "gross_4w_window_complete"]
    assert pd.isna(labels.loc["gap", "actual_purchases_4w"]) and pd.isna(labels.loc["gap", "actual_gross_4w_if_purchase"])
    assert labels.loc[["repeat", "none"], "gross_4w_window_complete"].all()
    dataset = build_dataset(panel, t0).set_index("customer_id")
    assert dataset.loc["repeat", "actual_gross_4w_if_purchase"] == 1500
    # 4주 정답 열은 예측 변수가 아니다(누수 방지 목록에 걸린다).
    from clv.schema import forbidden_predictors
    assert forbidden_predictors(["actual_gross_4w_if_purchase", "predicted_gross_4w_if_purchase",
                                 "gross_4w_window_complete"]) == ["actual_gross_4w_if_purchase",
                                                                  "predicted_gross_4w_if_purchase",
                                                                  "gross_4w_window_complete"]


def test_four_week_labels_are_checked_against_the_ledger():
    """원장 대조는 할인 전 금액(gross_amount)으로 한다. 원장이 결제액과 같으면(할인 차감) 거절한다."""
    panel = complete_grid(adapt_weekly(_four_week_panel()))
    t0 = pd.Timestamp("2024-03-04", tz="Asia/Seoul") + pd.Timedelta(weeks=13)
    at = [t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=2), t0 + pd.Timedelta(days=15),
          t0 + pd.Timedelta(days=29), t0 + pd.Timedelta(days=3)]
    orders = pd.DataFrame({"order_id": list("abcde"), "customer_id": ["repeat", "repeat", "repeat", "none", "gap"],
                           "order_at": at, "gross_amount": [600, 400, 500, 900, 300],
                           "discount_amount": [60, 40, 50, 0, 0], "paid_amount": [540, 360, 450, 900, 300]})
    ledger = adapt_orders(orders, currency="SOURCE_UNSPECIFIED", amount_unit="0.01")
    build_dataset(panel, t0, ledger)  # 일치하면 통과(52주 정답은 이 구간 밖이라 대조 대상이 없다)
    wrong = adapt_orders(orders.assign(gross_amount=orders["paid_amount"]), currency="SOURCE_UNSPECIFIED",
                         amount_unit="0.01")
    with pytest.raises(ValueError, match="4-week gross labels disagree"):
        build_dataset(panel, t0, wrong)


def test_four_week_model_trains_on_control_buyers_only(bundle):
    output, registry = bundle
    gross = pd.read_csv(output / "reports" / "gross_4w_labelled_predictions.csv", dtype={"customer_id": str})
    buyers = gross.loc[gross["actual_gross_4w_if_purchase"].notna() & gross["training_population"]]
    assert buyers["treatment"].eq(0).all()
    counts = buyers["split"].value_counts().to_dict()
    assert registry["gross_4w"]["status"] == "trained"
    assert registry["gross_4w"]["split_counts"] == {name: counts.get(name, 0) for name in ("train", "validation", "test")}
    model = GrossXGBoost4w.load(output / "models" / "xgboost_gross_4w")
    assert model.train_counts["train_buyers"] == counts["train"]
    assert model.train_counts["validation_buyers"] == counts["validation"]
    # 비구매 고객은 금액 정답이 없고(0으로 쓰지 않음), 예측은 4주 관측이 완전한 전원에게 있다.
    assert gross.loc[gross["actual_purchases_4w"].eq(0), "actual_gross_4w_if_purchase"].isna().all()
    assert gross["predicted_gross_4w_if_purchase"].notna().all()


def test_four_week_evaluation_and_assignment_diagnostic(bundle, experiment):
    output, registry = bundle
    evaluation = json.loads((output / "reports" / "evaluation.json").read_text())["gross_4w"]
    gross = pd.read_csv(output / "reports" / "gross_4w_labelled_predictions.csv", dtype={"customer_id": str})
    test = gross.loc[gross["split"].eq("test") & gross["actual_gross_4w_if_purchase"].notna()]
    metrics = evaluation["test"]
    assert metrics["n"] == len(test) == registry["gross_4w"]["test_metrics"]["n"]
    assert metrics["mean_actual"] == pytest.approx(test["actual_gross_4w_if_purchase"].mean())
    assert metrics["mean_predicted"] == pytest.approx(test["predicted_gross_4w_if_purchase"].mean())
    error = test["predicted_gross_4w_if_purchase"] - test["actual_gross_4w_if_purchase"]
    assert metrics["mae"] == pytest.approx(error.abs().mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt((error ** 2).mean()))
    diagnostic = evaluation["assignment_diagnostic"]
    for name, code in (("control", 0), ("treatment", 1)):
        part = gross.loc[gross["treatment"].eq(code) & gross["actual_gross_4w_if_purchase"].notna()]
        assert diagnostic["groups"][name]["buyers"] == len(part)
        assert diagnostic["groups"][name]["mean_gross_per_buyer"] == pytest.approx(
            part["actual_gross_4w_if_purchase"].mean())
    difference = diagnostic["treatment_minus_control"]
    assert difference["ci95"][0] < difference["difference"] < difference["ci95"][1]
    assert diagnostic["treatment_assigned_in_window_share"] == 1.0  # t0 = 실험 시점
    assert diagnostic["t0_equals_as_of_at"] and "comparison_suppressed" not in diagnostic
    # 예측 요약은 적합·선택에 쓰지 않은 구매 고객만: Control은 Test 구매 고객, Treatment는 전원이다.
    assert diagnostic["groups"]["control"]["predicted_buyers"] == len(test)
    assert diagnostic["groups"]["treatment"]["predicted_buyers"] == diagnostic["groups"]["treatment"]["buyers"]
    # 저장된 번들로 다시 평가해도 같은 4주 Test 지표가 나온다.
    report = evaluate(experiment, None, output)
    assert report["gross_4w"]["test"]["mae"] == pytest.approx(metrics["mae"])
    assert report["gross_4w"]["in_sample_buyers"] == 0  # Test 고객은 적합·선택에 쓰지 않았다
    everyone = evaluate(experiment, None, output, population="all")["gross_4w"]
    assert everyone["in_sample_buyers"] == registry["gross_4w"]["split_counts"]["train"] + \
        registry["gross_4w"]["split_counts"]["validation"]
    assert report["population_policy"] == "control_no_coupon"


def test_strategy_batch_adds_a_four_week_batch_with_the_same_keys(bundle, tmp_path):
    output, _ = bundle
    other, _, _ = synthetic_outputs(n_customers=60, n_weeks=96, assignment_week=30, seed=4)
    model = load_model(output)
    ids = model.predict(other, reference_week=T0)["customer_id"].tolist()
    population = pd.DataFrame({
        "customer_id": ids, "as_of_at": AS_OF, "snapshot_id": [f"{cid}:{AS_OF.isoformat()}" for cid in ids],
        "reference_week": T0, "dataset_id": "synthetic-fixture-seed4", "currency": "SOURCE_UNSPECIFIED",
        "amount_unit": "0.01", "raw_database_sha256": "0" * 64})
    result = model.predict_for_strategy(population, other)
    # Control-only 번들은 Uplift 결합용 검증(require_control_only)을 통과하고 결과도 같다.
    checked = model.predict_for_strategy(population, other, require_control_only=True)
    pd.testing.assert_frame_equal(checked.frame, result.frame)
    assert list(result.frame.columns) == list(STRATEGY_COLUMNS)  # 기존 11열 계약은 그대로
    assert result.gross_4w_status == "trained" and list(result.gross_4w.columns) == list(GROSS_STRATEGY_COLUMNS)
    assert result.gross_4w["customer_id"].equals(result.frame["customer_id"])
    assert result.gross_4w["input_cutoff_at"].equals(result.frame["input_cutoff_at"])
    assert (result.gross_4w["prediction_end_at"] - result.gross_4w["as_of_at"]).eq(pd.Timedelta(weeks=4)).all()
    assert result.gross_4w_metadata == {**result.gross_4w_metadata, "horizon_value": 4, "horizon_unit": "weeks",
                                        "target_definition_id": "gross_amount_4w_if_purchase_v1"}
    path = result.save(tmp_path / "batch.parquet")
    loaded = load_strategy_predictions(path)
    pd.testing.assert_frame_equal(loaded.gross_4w, result.gross_4w)
    assert loaded.gross_4w_status == "trained" and loaded.gross_4w_metadata == result.gross_4w_metadata


def test_existing_bundle_still_predicts_and_reports_no_four_week_model(outputs, raw_dir):
    """기존 전달 번들(배정군 혼합, 4주 모델 없음): 52주 예측 값이 변경 전 코드와 같고, 4주 값은 만들지 않는다."""
    pytest.importorskip("xgboost")
    weekly, _, _ = outputs
    # 변경 전 코드(clv 0.2.0)로 같은 입력에서 기록한 예측 합계와 customer_id 순 첫 세 고객의 순위
    expected = {"baseline": (2072164.2862, [76, 108, 105]), "bgnbd_gg": (684977.2282, [8, 108, 77]),
                "xgboost": (1171207.2465, [64, 169, 105]), "xgboost_two_part": (1458785.7341, [55, 130, 100])}
    for name, (total, ranks) in expected.items():
        model = load_model(HANDOFF_BUNDLE, name)
        assert model.policy_condition == "observed_experiment_mix" and not model.has_gross_4w
        assert model.gross_4w_status == "not_trained" and model.gross_4w_metadata is None
        assert model.config.population_policy == "observed_experiment_mix"
        predictions = model.predict(weekly, raw_dir / "dataset", reference_week="2024-10-07")
        ordered = predictions.sort_values("customer_id")
        assert np.round(ordered["predicted_clv_12m"].to_numpy(dtype=float), 4).sum() == pytest.approx(total, rel=1e-12)
        assert ordered["clv_rank"].iloc[:3].tolist() == ranks
        # 4주 값은 0이나 52주 값으로 채우지 않고 결측 + 상태로 표시한다.
        assert predictions["predicted_gross_4w_if_purchase"].isna().all()
        assert predictions["gross_4w_status"].eq("not_trained").all()
    with pytest.raises(ValueError, match="no trained xgboost_gross_4w"):
        load_model(HANDOFF_BUNDLE).predict(weekly, reference_week="2024-10-07", require_gross_4w=True)


def test_existing_bundle_strategy_batch_has_no_four_week_batch(outputs, tmp_path):
    weekly, _, _ = outputs
    model = load_model(HANDOFF_BUNDLE)
    ids = model.predict(weekly, reference_week="2024-10-07")["customer_id"].tolist()[:10]
    as_of = pd.Timestamp("2024-10-06T15:00:00Z")
    population = pd.DataFrame({
        "customer_id": ids, "as_of_at": as_of, "snapshot_id": ids, "reference_week": "2024-10-07",
        "dataset_id": "synthetic-fixture-seed0", "currency": "SOURCE_UNSPECIFIED", "amount_unit": "0.01",
        "raw_database_sha256": "0" * 64})
    result = model.predict_for_strategy(population, weekly)
    assert list(result.frame.columns) == list(STRATEGY_COLUMNS)
    assert result.gross_4w is None and result.gross_4w_metadata is None and result.gross_4w_status == "not_trained"
    assert any("xgboost_gross_4w" in warning for warning in result.producer_metadata["warnings"])
    assert result.producer_metadata["policy_condition"] == "observed_experiment_mix"
    # Uplift 결합용 검증을 켜면 혼합 번들은 거절한다(끄면 위처럼 그대로 추론한다).
    with pytest.raises(ValueError, match="require_control_only: .* not 'control_no_coupon'"):
        model.predict_for_strategy(population, weekly, require_control_only=True)
    # 같은 경로에 예전 4주 batch가 있으면 저장할 때 지운다.
    (tmp_path / "batch_gross_4w.parquet").write_bytes(b"stale")
    path = result.save(tmp_path / "batch.parquet")
    assert not (tmp_path / "batch_gross_4w.parquet").exists()
    assert load_strategy_predictions(path).gross_4w is None


def test_existing_bundle_directory_is_not_overwritten(bundle, experiment):
    output, _ = bundle
    before = file_sha256(output / "model_card.json")
    with pytest.raises(FileExistsError, match="already holds a CLV bundle"):
        train(experiment, None, output, CLVConfig.from_dict(FAST), cutoff=T0)
    assert file_sha256(output / "model_card.json") == before


def test_policy_settings_and_legacy_registry_defaults():
    assert CLVConfig().population_policy == "control_no_coupon"
    assert CLVConfig.from_registry({}).population_policy == "observed_experiment_mix"
    assert CLVConfig.from_registry({"population_policy": "control_no_coupon"}).population_policy == "control_no_coupon"
    with pytest.raises(ValueError, match="population_policy"):
        CLVConfig(population_policy="treatment_only")
    # 배정 열이 없으면 Control-only를 적용할 수 없어 멈춘다(혼합 정책은 가능).
    panel = complete_grid(adapt_weekly(_four_week_panel()))
    dataset = build_dataset(panel, pd.Timestamp("2024-06-03", tz="Asia/Seoul"))
    with pytest.raises(ValueError, match="needs the experiment assignment column"):
        policy_mask(dataset, "control_no_coupon")
    assert policy_mask(dataset, "observed_experiment_mix").all()


def test_control_customer_with_a_coupon_is_excluded(experiment, tmp_path):
    changed = experiment.copy()
    control = changed["customer_id"].eq("c0000") & changed["reference_week"].eq(pd.Timestamp(T0, tz="Asia/Seoul"))
    changed.loc[control, ["coupons_sent", "coupon_actions"]] = 1
    registry = train(changed, None, tmp_path / "clv", CLVConfig.from_dict(FAST))  # t0 기본값 = 실험 시각
    assert registry["reference_week"] == "2024-07-29T00:00:00+09:00"
    assert registry["population"]["control_with_coupon_excluded"] == 1
    splits = _splits(tmp_path / "clv").set_index("customer_id")
    assert splits.loc["c0000", "split"] == "unused" and not splits.loc["c0000", "policy_eligible"]


def test_assignment_must_agree_with_group_labels(experiment):
    broken = experiment.copy()
    broken.loc[broken["customer_id"].eq("c0001"), "group"] = "control"  # 홀수 = Treatment
    with pytest.raises(ValueError, match="must match treatment"):
        adapt_weekly(broken)


def test_four_week_model_keeps_zero_gross_buyers_and_rejects_invalid_labels():
    """금액 0인 구매 고객도 정답으로 쓰고(gamma 후보만 뺀다), 결측·음수 정답은 거절한다."""
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((40, 17)), columns=GrossXGBoost4w().feature_names)
    labels = rng.random(40) * 1000 + 1
    labels[0] = 0.0
    grid = {"objective": ["reg:squarederror", "reg:gamma"], "max_depth": [2], "learning_rate": [0.1]}
    fixed = {"n_estimators": 20, "early_stopping_rounds": 5, "n_jobs": 1}
    model = GrossXGBoost4w(grid=grid, fixed=fixed).fit(frame, labels, frame, labels)
    assert model.skipped_objectives == ["reg:gamma"] and model.train_counts["zero_gross_buyers"] == 2
    assert {row["objective"] for row in model.grid_results} == {"reg:squarederror"}
    for bad in (-1.0, np.nan):
        broken = labels.copy()
        broken[1] = bad
        with pytest.raises(ValueError, match="finite nonnegative gross amounts"):
            GrossXGBoost4w(grid=grid, fixed=fixed).fit(frame, broken, frame, labels)


def test_saved_bundle_evaluation_diagnostic_includes_treatment(bundle, experiment):
    """기본 Test 평가에서도 지표는 Control Test 구매 고객으로, 배정군 진단은 같은 t0의 전원으로 만든다."""
    output, registry = bundle
    report = evaluate(experiment, None, output)["gross_4w"]
    groups = report["assignment_diagnostic"]["groups"]
    assert groups["treatment"]["buyers"] > 0 and groups["control"]["buyers"] > report["test"]["n"]
    assert report["test"]["n"] == registry["gross_4w"]["test_metrics"]["n"]


def test_off_experiment_cutoff_suppresses_the_group_comparison(experiment, tmp_path):
    """t0가 실험 시각과 다르면 4주 구간에 쿠폰 발송이 없으므로 배정군 차이를 내지 않고 경고를 남긴다."""
    registry = train(experiment, None, tmp_path / "clv", CLVConfig.from_dict(FAST), cutoff="2024-08-05")
    diagnostic = json.loads((tmp_path / "clv" / "reports" / "evaluation.json").read_text())["gross_4w"][
        "assignment_diagnostic"]
    assert diagnostic["treatment_assigned_in_window_share"] == 0.0
    assert "treatment_minus_control" not in diagnostic and "comparison_suppressed" in diagnostic
    assert any("differs from the experiment as_of_at" in warning for warning in registry["warnings"])


def test_four_week_model_needs_control_only_policy_and_surfaces_errors(experiment, tmp_path, monkeypatch):
    mixed = train(experiment, None, tmp_path / "mix", CLVConfig.from_dict({**FAST, "population_policy":
                                                                              "observed_experiment_mix"}), cutoff=T0)
    assert mixed["gross_4w"]["status"].startswith("skipped: the 4-week model is defined for")
    assert not (tmp_path / "mix" / "models" / "xgboost_gross_4w").exists()
    card = json.loads((tmp_path / "mix" / "model_card.json").read_text())
    assert card["policy_condition"] == "observed_experiment_mix" and card["gross_4w"]["model_version"] is None
    # 구매 고객 부족만 failed로 남기고, 그 밖의 오류는 그대로 낸다. require_gross_4w면 부족해도 저장하지 않고 멈춘다.
    from clv import xgboost_model
    monkeypatch.setattr(xgboost_model, "MIN_GROSS_BUYERS", 10_000)
    few = train(experiment, None, tmp_path / "few", CLVConfig.from_dict(FAST), cutoff=T0)
    assert few["gross_4w"]["status"].startswith("failed: xgboost_gross_4w needs at least")
    with pytest.raises(ValueError, match="require_gross_4w"):
        train(experiment, None, tmp_path / "required", CLVConfig.from_dict(FAST), cutoff=T0, require_gross_4w=True)
    assert not (tmp_path / "required").exists()
    monkeypatch.setattr(GrossXGBoost4w, "fit", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        train(experiment, None, tmp_path / "broken", CLVConfig.from_dict(FAST), cutoff=T0)


def test_cutoff_before_the_experiment_suppresses_the_group_comparison(bundle, experiment, tmp_path):
    """회귀: t0가 실험 시각보다 2주 앞서면 배정 시각은 4주 구간 안(비율 1.0)이지만 t0 != as_of_at이므로
    학습 보고서와 저장 번들 재평가 모두 배정군 차이를 내지 않는다."""
    early = "2024-07-15"
    registry = train(experiment, None, tmp_path / "clv", CLVConfig.from_dict(FAST), cutoff=early)
    diagnostic = json.loads((tmp_path / "clv" / "reports" / "evaluation.json").read_text())["gross_4w"][
        "assignment_diagnostic"]
    assert registry["gross_4w"]["status"] == "trained"
    assert diagnostic["treatment_assigned_in_window_share"] == 1.0
    assert diagnostic["t0_equals_as_of_at"] is False
    assert "treatment_minus_control" not in diagnostic
    assert "is not the experiment as_of_at" in diagnostic["comparison_suppressed"]
    output, _ = bundle
    report = evaluate(experiment, None, output, cutoff=early)["gross_4w"]["assignment_diagnostic"]
    assert report["treatment_assigned_in_window_share"] == 1.0 and report["t0_equals_as_of_at"] is False
    assert "treatment_minus_control" not in report and "comparison_suppressed" in report
    aligned = evaluate(experiment, None, output)["gross_4w"]["assignment_diagnostic"]
    assert aligned["t0_equals_as_of_at"] and "treatment_minus_control" in aligned
