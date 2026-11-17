"""WBS 4.6-4.7: 평가 지표, 상위 20%, 저장된 학습 결과, 예측 CSV 계약, API와 CLI를 검사한다."""
import json

import numpy as np
import pandas as pd
import pytest

from clv.api import load_clv_predictions, predict_clv, run_clv
from clv.cli import main
from clv.config import CLVConfig
from clv.evaluation import evaluate_predictions, regression_metrics, top_count, top_flags, top_overlap
from clv.pipeline import evaluate, predict, train, write_predictions
from clv.schema import OUTPUT_COLUMNS

# 테스트 시간을 줄이는 작은 XGBoost 그리드
FAST = {"xgb_grid": {"objective": ["reg:squarederror", "reg:tweedie"], "max_depth": [3], "learning_rate": [0.1]},
        "xgb_fixed": {"n_estimators": 200, "early_stopping_rounds": 20, "subsample": 0.8, "tree_method": "hist", "n_jobs": 1}}


def test_mae_rmse_on_known_values():
    metrics = regression_metrics([0, 10, 20], [0, 13, 16])
    assert metrics["mae"] == pytest.approx(7 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(25 / 3))
    assert metrics["bias"] == pytest.approx(-1 / 3)
    with pytest.raises(ValueError):  # NaN이 섞인 입력은 거부한다
        regression_metrics([1.0, np.nan], [1.0, 2.0])


def test_top_20_percent_count_ties_and_overlap():
    """상위 20%는 ceil(0.2·N)명이고 동점은 customer_id 오름차순으로 정한다.
    일치율은 같은 k에서 |예측 ∩ 실제| / k다."""
    assert top_count(10, 0.2) == 2 and top_count(11, 0.2) == 3 and top_count(3, 0.2) == 1
    ids = ["e", "d", "c", "b", "a"]
    flags = top_flags([5, 9, 9, 1, 0], ids, 0.2)
    assert flags.tolist() == [False, False, True, False, False]  # 9/9 동점 -> customer_id가 작은 "c"
    actual = [100, 80, 0, 0, 0, 0, 0, 0, 0, 0]
    predicted = [90, 0, 70, 0, 0, 0, 0, 0, 0, 0]
    result = top_overlap(actual, predicted, [f"c{i}" for i in range(10)], 0.2)
    assert result["k"] == 2 and result["overlap"] == 1 and result["overlap_rate"] == 0.5
    assert result["captured_actual_value_share"] == pytest.approx(100 / 180)


def test_models_are_scored_on_the_same_rows():
    """모든 모델을 같은 고객 행으로 평가한다. 한 모델이라도 예측이 빠진 고객이 있으면 오류다."""
    frame = pd.DataFrame({"customer_id": list("abcd"), "actual_clv_12m": [0.0, 10, 20, 30],
                          "pred_baseline": [1.0, 9, 25, 30], "pred_xgboost": [0.0, 10, 20, 31]})
    report = evaluate_predictions(frame, ["baseline", "xgboost"])
    assert report["customers"] == 4
    assert {name: value["n"] for name, value in report["models"].items()} == {"baseline": 4, "xgboost": 4}
    with pytest.raises(ValueError, match="every evaluated customer"):
        evaluate_predictions(frame.assign(pred_xgboost=[np.nan, 1, 2, 3]), ["baseline", "xgboost"])


@pytest.fixture(scope="module")
def trained(raw_dir, tmp_path_factory):
    """작은 그리드로 한 번 학습한 ``(출력 디렉터리, registry)``. 모듈 안에서 함께 쓴다."""
    pytest.importorskip("xgboost")
    output = tmp_path_factory.mktemp("clv-run") / "clv"
    registry = train(raw_dir / "features", raw_dir / "dataset", output, CLVConfig.from_dict(FAST))
    return output, registry


def test_training_registry_and_reports(trained):
    output, registry = trained
    assert registry["model_status"] == {"baseline": "trained", "bgnbd_gg": "trained", "xgboost": "trained"}
    assert registry["final_model"] in registry["trained_models"]
    assert registry["provenance"]["currency"] == "SOURCE_UNSPECIFIED"
    assert sum(registry["split_counts"].values()) == registry["dataset"]["labelled_customers"]
    for name in ("registry.json", "splits.csv", "models/baseline.json", "models/bgnbd_gg.json",
                 "models/xgboost/xgboost_model.json", "models/xgboost/xgboost_meta.json", "reports/evaluation.json",
                 "reports/validation.json", "reports/eligible_cutoffs.csv", "reports/deciles_test.csv",
                 "reports/top20_test.csv", "reports/labelled_predictions.csv"):
        assert (output / name).is_file(), name
    evaluation = json.loads((output / "reports" / "evaluation.json").read_text())
    test = evaluation["test"]
    assert set(test["models"]) == {"baseline", "bgnbd_gg", "xgboost"}
    # 모든 모델의 평가 인원이 같고 Test 고객 수와도 같다.
    assert len({value["n"] for value in test["models"].values()}) == 1 == len({test["customers"]})
    assert "by_assignment_group" in test
    splits = pd.read_csv(output / "splits.csv")
    assert splits["customer_id"].is_unique


def test_evaluate_reproduces_training_test_metrics(trained, raw_dir):
    output, registry = trained
    report = evaluate(raw_dir / "features", raw_dir / "dataset", output)
    for name, metrics in registry["test_metrics"].items():
        assert report["models"][name]["mae"] == pytest.approx(metrics["mae"])
        assert report["models"][name]["rmse"] == pytest.approx(metrics["rmse"])


def test_prediction_csv_contract_without_labels(trained, raw_dir, tmp_path):
    """정답이 없는 기준시점도 예측한다. 앞 4개 열은 계약 열이고 상위 20%는 ceil(0.2·N)명이며,
    CSV를 다시 읽어도 자료형·시간대가 유지된다."""
    output, registry = trained
    late = "2025-05-12"  # 미래 52주가 관측되지 않아 정답이 없다
    predictions = predict(raw_dir / "features", raw_dir / "dataset", output, reference_week=late)
    assert predictions["reference_week"].eq("2025-05-12T00:00:00+09:00").all()
    assert (predictions["predicted_clv_12m"] >= 0).all()
    k = predictions["is_high_value_customer"].sum()
    assert k == int(np.ceil(0.2 * len(predictions)))
    top = predictions.loc[predictions["is_high_value_customer"], "predicted_clv_12m"].min()
    assert top >= predictions.loc[~predictions["is_high_value_customer"], "predicted_clv_12m"].max()
    path = write_predictions(predictions, tmp_path / "results" / "clv_predictions.csv")
    raw = pd.read_csv(path)
    assert list(raw.columns[:4]) == list(OUTPUT_COLUMNS)
    loaded = load_clv_predictions(path)
    assert loaded["customer_id"].map(type).eq(str).all()
    assert str(loaded["reference_week"].dt.tz) == "Asia/Seoul" and loaded["is_high_value_customer"].dtype == bool
    np.testing.assert_allclose(loaded["predicted_clv_12m"], predictions["predicted_clv_12m"], rtol=1e-6)
    default = predict(raw_dir / "features", raw_dir / "dataset", output)
    assert default["reference_week"].iloc[0] == "2024-10-07T00:00:00+09:00"  # 기본값 = 실험 as_of_at
    baseline = predict(raw_dir / "features", raw_dir / "dataset", output, reference_week=late, model="baseline")
    assert baseline["model_name"].eq("baseline").all()


def test_model_without_ledger_is_skipped_not_faked(raw_dir, tmp_path):
    """주문 원장이 없으면 BG/NBD + GG는 주간 집계로 주문을 복원하지 않고 skipped로 남기며 평가에서 빠진다."""
    registry = train(raw_dir / "features", None, tmp_path / "clv", CLVConfig.from_dict(FAST))
    assert registry["model_status"]["bgnbd_gg"].startswith("skipped")
    evaluation = json.loads((tmp_path / "clv" / "reports" / "evaluation.json").read_text())
    assert "bgnbd_gg" not in evaluation["test"]["models"]


def test_unit_mismatch_between_model_and_input_is_refused(trained, outputs):
    """학습 때와 통화가 다른 입력으로는 예측하지 않는다."""
    output, _ = trained
    weekly, _, _ = outputs
    with pytest.raises(ValueError, match="currency"):
        predict(weekly.assign(currency="KRW"), None, output, model="baseline")


def test_api_run_and_cli(raw_dir, tmp_path, capsys):
    pytest.importorskip("xgboost")
    config = tmp_path / "clv.json"
    config.write_text(json.dumps(FAST))
    result = run_clv(raw_dir / "features", raw_dir / "dataset", tmp_path / "api", tmp_path / "api.csv", config=config)
    assert result["predicted_customers"] == len(load_clv_predictions(tmp_path / "api.csv"))
    frame = predict_clv(raw_dir / "features", raw_dir / "dataset", tmp_path / "api", output=tmp_path / "again.csv")
    pd.testing.assert_frame_equal(load_clv_predictions(tmp_path / "again.csv"), load_clv_predictions(tmp_path / "api.csv"))
    assert len(frame) == result["predicted_customers"]
    summary = main(["validate", "--weekly", str(raw_dir / "features"), "--ledger", str(raw_dir / "dataset")])
    assert summary["ok"]
    summary = main(["cutoffs", "--weekly", str(raw_dir / "features")])
    assert summary["cutoffs"]
    summary = main(["predict", "--weekly", str(raw_dir / "features"), "--ledger", str(raw_dir / "dataset"),
                    "--model-dir", str(tmp_path / "api"), "--output", str(tmp_path / "cli.csv")])
    assert summary["customers"] == result["predicted_customers"]
    capsys.readouterr()


def test_history_segments_use_the_same_customer_split(trained, raw_dir):
    """이력 완전·미완전 고객은 하나의 분할을 나눠 쓴다. 완전 구간 지표는 기본 Test 지표와 같다."""
    output, registry = trained
    evaluation = json.loads((output / "reports" / "evaluation.json").read_text())
    segments = evaluation["by_history"]["test"]
    assert segments["complete"]["customers"] + segments["incomplete"]["customers"] == \
        registry["split_counts_all_labelled"]["test"]
    assert segments["incomplete"]["customers"] > 1 and registry["split_counts"]["test"] == segments["complete"]["customers"]
    for name in registry["trained_models"]:
        assert segments["complete"]["models"][name]["mae"] == pytest.approx(evaluation["test"]["models"][name]["mae"])
        assert registry["test_metrics_by_history"]["incomplete"][name]["n"] == segments["incomplete"]["customers"]
    splits = pd.read_csv(output / "splits.csv", dtype={"customer_id": str})
    assert {"history_complete", "training_population"} <= set(splits.columns)
    assert len(splits) == sum(registry["split_counts_all_labelled"].values())
    report = evaluate(raw_dir / "features", raw_dir / "dataset", output)
    assert report["by_history"]["incomplete"]["customers"] == segments["incomplete"]["customers"]


def test_all_history_training_keeps_the_same_split(trained, raw_dir, tmp_path):
    """require_complete_history=False여도 고객 분할은 같고, 학습 대상만 정답 고객 전체로 넓어진다."""
    output, registry = trained
    wide = train(raw_dir / "features", raw_dir / "dataset", tmp_path / "wide",
                 CLVConfig.from_dict({**FAST, "require_complete_history": False}))
    narrow_splits = pd.read_csv(output / "splits.csv", dtype={"customer_id": str})
    wide_splits = pd.read_csv(tmp_path / "wide" / "splits.csv", dtype={"customer_id": str})
    pd.testing.assert_series_equal(narrow_splits["split"], wide_splits["split"])
    assert wide_splits["training_population"].all()
    assert wide["split_counts"] == wide["split_counts_all_labelled"] == registry["split_counts_all_labelled"]


def test_baseline_ratios_and_bgnbd_sensitivity_report(trained):
    output, registry = trained
    evaluation = json.loads((output / "reports" / "evaluation.json").read_text())
    models = evaluation["test"]["models"]
    assert models["baseline"]["vs_baseline"] == {"mae_ratio": 1.0, "rmse_ratio": 1.0}
    assert models["xgboost"]["vs_baseline"]["mae_ratio"] == pytest.approx(models["xgboost"]["mae"] / models["baseline"]["mae"])
    table = pd.read_csv(output / "reports" / "bgnbd_sensitivity.csv")
    assert table["bgnbd_penalizer"].tolist() == list(CLVConfig().bgnbd_sensitivity_penalizers)
    configured = table.loc[table["configured"]].iloc[0]
    params = registry["model_details"]["bgnbd_gg"]["bgnbd"]["params"]
    for name in ("r", "alpha", "a", "b"):
        assert configured[name] == pytest.approx(params[name])
    assert configured["spearman_vs_configured"] == pytest.approx(1.0)


def test_bgnbd_violating_a_required_condition_is_rejected(raw_dir, tmp_path, monkeypatch):
    """Gamma-Gamma q <= 1이면 BG/NBD + GG를 rejected로 기록하고 저장·평가·최종 선택에서 뺀다."""
    from clv.probabilistic import GammaGamma

    original = GammaGamma.fit

    def broken_fit(self, frequency, monetary_value, penalizer=0.0):
        info = original(self, frequency, monetary_value, penalizer)
        self.params = {**self.params, "q": 0.9}
        return info

    monkeypatch.setattr(GammaGamma, "fit", broken_fit)
    registry = train(raw_dir / "features", raw_dir / "dataset", tmp_path / "clv",
                     CLVConfig.from_dict({**FAST, "bgnbd_sensitivity_penalizers": []}))
    assert registry["model_status"]["bgnbd_gg"].startswith("rejected: Gamma-Gamma q = 0.9")
    assert "bgnbd_gg" not in registry["trained_models"] and registry["final_model"] != "bgnbd_gg"
    assert registry["model_details"]["bgnbd_gg"]["rejections"]
    assert not (tmp_path / "clv" / "models" / "bgnbd_gg.json").exists()
    evaluation = json.loads((tmp_path / "clv" / "reports" / "evaluation.json").read_text())
    assert "bgnbd_gg" not in evaluation["test"]["models"]
    assert not (tmp_path / "clv" / "reports" / "bgnbd_sensitivity.csv").exists()
