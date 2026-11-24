"""번들 모델 카드·체크섬, ``CLVModel`` 추론, 전략 모듈 입력 batch(열·기간·키·출처·학습 고객 중복)를 검사한다."""
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from clv.api import load_model, load_strategy_predictions, model_info, predict_clv, predict_for_strategy
from clv.bundle import CARD_NAME, MODEL_FILES, TARGET_DEFINITION_ID, file_sha256
from clv.cli import main
from clv.config import CLVConfig
from clv.inference import STRATEGY_COLUMNS, CLVModel
from clv.pipeline import prepare_inputs, train

from .fixtures import synthetic_outputs, write_raw_sqlite

FAST = {"xgb_grid": {"objective": ["reg:squarederror"], "max_depth": [3], "learning_rate": [0.1]},
        "xgb_fixed": {"n_estimators": 100, "early_stopping_rounds": 10, "n_jobs": 1},
        "two_part_classifier_grid": {"max_depth": [3], "learning_rate": [0.1]},
        "two_part_regressor_grid": {"objective": ["reg:squarederror"], "max_depth": [3], "learning_rate": [0.1]}}
# 합성 Fixture의 실험 시점(서울 2024-10-07 월요일 00:00)
AS_OF = pd.Timestamp("2024-10-06T15:00:00Z")


@pytest.fixture(scope="module")
def bundle(raw_dir, tmp_path_factory):
    """합성 데이터(seed 0)로 한 번 학습한 번들 디렉터리."""
    pytest.importorskip("xgboost")
    output = tmp_path_factory.mktemp("bundle") / "clv"
    train(raw_dir / "features", raw_dir / "dataset", output, CLVConfig.from_dict(FAST))
    return output


def _raw_sha(directory):
    return json.loads((directory / "dataset" / "raw.manifest.json").read_text())["database_sha256"]


def _population(directory, ids, dataset_id="synthetic-fixture-seed0", as_of=AS_OF, week="2024-10-07"):
    """시뮬레이터 Feature Store 키 열과 같은 모양의 예측 대상 표."""
    return pd.DataFrame({
        "customer_id": list(ids), "as_of_at": as_of, "snapshot_id": [f"{cid}:{as_of.isoformat()}" for cid in ids],
        "reference_week": week, "dataset_id": dataset_id, "currency": "SOURCE_UNSPECIFIED", "amount_unit": "0.01",
        "raw_database_sha256": _raw_sha(directory),
    })


@pytest.fixture(scope="module")
def other_dataset(tmp_path_factory):
    """학습에 쓰지 않은 별도 합성 데이터셋(seed 1). 전략 평가용 고객을 흉내 낸다."""
    weekly, ledger, metadata = synthetic_outputs(seed=1)
    directory = tmp_path_factory.mktemp("other-dataset")
    write_raw_sqlite(directory / "dataset", ledger, metadata)
    (directory / "features").mkdir()
    weekly.to_parquet(directory / "features" / "weekly_panel.parquet", index=False)
    return directory


def _customers_with_history(directory, bundle):
    """기준 시각 직전 13주 이력이 있는 고객(예측 가능한 고객)."""
    return load_model(bundle, "baseline").predict(directory / "features", reference_week=AS_OF)["customer_id"].tolist()


def test_model_card_records_versions_and_checksums(bundle):
    card = json.loads((bundle / CARD_NAME).read_text())
    registry = json.loads((bundle / "registry.json").read_text())
    assert card["target"]["horizon_value"] == 52 and card["target"]["horizon_unit"] == "weeks"
    assert card["target"]["definition_id"] == TARGET_DEFINITION_ID
    # 4주 모델도 같은 방식으로 파일 체크섬과 버전을 남긴다.
    assert registry["gross_4w"]["status"] == "trained"
    versioned = [*registry["trained_models"], "xgboost_gross_4w"]
    expected = {"registry.json", "splits.csv", *(path for name in versioned for path in MODEL_FILES[name])}
    assert set(card["files_sha256"]) == expected
    assert all(file_sha256(bundle / path) == digest for path, digest in card["files_sha256"].items())
    assert set(card["model_versions"]) == set(versioned)
    assert re.fullmatch(r"clv-\d+\.\d+\.\d+\+xgboost\.[0-9a-f]{12}", card["model_versions"]["xgboost"])
    assert card["gross_4w"]["model_version"] == card["model_versions"]["xgboost_gross_4w"]
    assert card["bundle_schema_version"] == 2 and card["policy_condition"] == "control_no_coupon"
    assert card["gross_4w"]["target"]["horizon_value"] == 4
    assert card["training_data"]["dataset_id"] == "synthetic-fixture-seed0"
    assert card["training_data"]["simulator"]["seed"] == 0  # raw manifest에서 옮겨 적은 출처
    assert card["environment"]["xgboost"] and card["code"]["clv_version"]
    info = model_info(bundle)
    assert info["model_version"] == card["model_versions"][registry["final_model"]]


def test_changed_or_incomplete_bundle_is_refused(bundle, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    CLVModel(copy)  # 그대로 복사한 번들은 통과
    with (copy / "models" / "baseline.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed after training"):
        CLVModel(copy)
    (copy / CARD_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="model_card"):
        CLVModel(copy)


def test_predict_selected_customers(bundle, raw_dir):
    model = load_model(bundle)
    everyone = model.predict(raw_dir / "features", raw_dir / "dataset", reference_week=AS_OF)
    assert everyone["model_version"].eq(model.model_version).all()
    assert everyone["prediction_end_at"].eq("2025-10-06T00:00:00+09:00").all()  # t0 + 364일
    chosen = everyone["customer_id"].iloc[[7, 3, 11, 0, 5]].tolist()
    subset = predict_clv(raw_dir / "features", raw_dir / "dataset", bundle, reference_week=AS_OF, customers=chosen)
    assert sorted(subset["customer_id"]) == sorted(chosen)
    assert subset["is_high_value_customer"].sum() == 1  # ceil(0.2 * 5), 고른 고객 안에서 다시 정한다
    merged = subset.merge(everyone, on="customer_id", suffixes=("", "_all"))
    np.testing.assert_allclose(merged["predicted_clv_12m"], merged["predicted_clv_12m_all"])
    with pytest.raises(ValueError, match="no observed week"):
        model.predict(raw_dir / "features", reference_week=AS_OF, customers=[*chosen, "nobody"])
    with pytest.raises(ValueError, match="duplicate"):
        model.predict(raw_dir / "features", reference_week=AS_OF, customers=[chosen[0], chosen[0]])


def test_strategy_batch_follows_the_final_strategy_contract(bundle, other_dataset):
    """다른 데이터셋 고객의 batch: 열 11개(순서 고정), UTC 시각, 종료 = as_of_at + 52주, 0 이상 금액, 키·순서 보존."""
    ids = _customers_with_history(other_dataset, bundle)[::-1]
    population = _population(other_dataset, ids, dataset_id="synthetic-fixture-seed1")
    result = predict_for_strategy(population, other_dataset / "features", other_dataset / "dataset", bundle)
    frame = result.frame
    assert tuple(frame.columns) == STRATEGY_COLUMNS
    assert result.metadata == {"model_version": load_model(bundle).model_version,
                               "target_definition_id": TARGET_DEFINITION_ID, "horizon_value": 52,
                               "horizon_unit": "weeks", "value_semantics": "nonnegative_amount"}
    for column in ("as_of_at", "input_cutoff_at", "prediction_end_at"):
        assert str(frame[column].dtype) == "datetime64[ns, UTC]"
    assert frame["input_cutoff_at"].le(frame["as_of_at"]).all()
    assert frame["prediction_end_at"].eq(frame["as_of_at"] + pd.Timedelta(weeks=52)).all()
    assert frame["predicted_clv_12m"].dtype == "float64" and np.isfinite(frame["predicted_clv_12m"]).all()
    assert frame["predicted_clv_12m"].ge(0).all()
    keys = ["dataset_id", "customer_id", "as_of_at", "reference_week", "snapshot_id", "currency", "amount_unit",
            "raw_database_sha256"]
    pd.testing.assert_frame_equal(frame[keys], population[keys].assign(
        as_of_at=population["as_of_at"].astype("datetime64[ns, UTC]")))
    overlap = result.producer_metadata["training_overlap"]
    assert overlap["same_dataset"] is False and overlap["roles"]["unused"] == len(ids)
    assert result.producer_metadata["inputs"]["raw_database_sha256_verified_by"] == ["ledger_manifest"]
    assert result.audit["clv_training_role"].eq("unused").all()
    # 같은 고객의 일반 예측과 값이 같다.
    plain = load_model(bundle).predict(other_dataset / "features", reference_week=AS_OF).set_index("customer_id")
    np.testing.assert_allclose(frame["predicted_clv_12m"], plain.loc[ids, "predicted_clv_12m"])


def test_strategy_refuses_customers_the_model_was_trained_on(bundle, raw_dir):
    ids = _customers_with_history(raw_dir, bundle)
    population = _population(raw_dir, ids)
    with pytest.raises(ValueError, match="never saw"):
        predict_for_strategy(population, raw_dir / "features", raw_dir / "dataset", bundle)
    result = predict_for_strategy(population, raw_dir / "features", raw_dir / "dataset", bundle,
                                  allow_training_customers=True)
    splits = pd.read_csv(bundle / "splits.csv", dtype={"customer_id": str}).set_index("customer_id")["split"]
    expected = pd.Series(ids).map(splits).fillna("unused")
    assert result.audit["clv_training_role"].tolist() == expected.tolist()
    roles = result.producer_metadata["training_overlap"]["roles"]
    assert roles["train"] == int(expected.eq("train").sum()) and roles["train"] > 0
    assert result.producer_metadata["warnings"] and result.producer_metadata["training_overlap"]["same_dataset"]
    # Test 고객만 남기면 학습에 쓰지 않은 고객이므로 기본 설정으로도 통과한다.
    test_only = population.loc[expected.eq("test").to_numpy()]
    assert len(predict_for_strategy(test_only, raw_dir / "features", raw_dir / "dataset", bundle).frame) == len(test_only)


def test_strategy_refuses_mismatched_inputs(bundle, raw_dir, other_dataset):
    ids = _customers_with_history(other_dataset, bundle)
    good = _population(other_dataset, ids, dataset_id="synthetic-fixture-seed1")
    args = (other_dataset / "features", other_dataset / "dataset", bundle)
    with pytest.raises(ValueError, match="raw_database_sha256"):
        predict_for_strategy(good.assign(raw_database_sha256="0" * 64), *args)
    with pytest.raises(ValueError, match="dataset_id"):
        predict_for_strategy(good.assign(dataset_id="another"), *args)
    with pytest.raises(ValueError, match="currency"):
        predict_for_strategy(good.assign(currency="KRW"), *args)
    with pytest.raises(ValueError, match="Monday"):
        predict_for_strategy(good.assign(as_of_at=AS_OF + pd.Timedelta(hours=3)), *args)
    with pytest.raises(ValueError, match="no observed week"):
        predict_for_strategy(_population(other_dataset, [*ids, "nobody"], dataset_id="synthetic-fixture-seed1"), *args)
    with pytest.raises(ValueError, match="duplicate"):
        predict_for_strategy(pd.concat([good, good.head(1)]), *args)
    with pytest.raises(ValueError, match="timezone"):
        predict_for_strategy(good.assign(as_of_at=AS_OF.tz_localize(None)), *args)


def test_strategy_files_round_trip_and_detect_changes(bundle, other_dataset, tmp_path):
    ids = _customers_with_history(other_dataset, bundle)
    population = _population(other_dataset, ids, dataset_id="synthetic-fixture-seed1")
    path = tmp_path / "out" / "clv_strategy.parquet"
    result = predict_for_strategy(population, other_dataset / "features", other_dataset / "dataset", bundle,
                                  output=path)
    loaded = load_strategy_predictions(path)
    pd.testing.assert_frame_equal(loaded.frame, result.frame)
    pd.testing.assert_frame_equal(loaded.audit, result.audit, check_dtype=False)
    assert loaded.metadata == result.metadata
    assert (tmp_path / "out" / "clv_strategy_metadata.json").is_file() and (tmp_path / "out" / "clv_strategy_audit.csv").is_file()
    result.frame.assign(predicted_clv_12m=0.0).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="changed"):
        load_strategy_predictions(path)


def test_weekly_manifest_checksum_and_raw_database_agreement(outputs, raw_dir, tmp_path):
    """features manifest가 있으면 주간 파일 체크섬을 대조하고, 원장과 다른 raw SQLite면 섞어 쓰지 않는다."""
    weekly, _, _ = outputs
    features = tmp_path / "features"
    features.mkdir()
    weekly.to_parquet(features / "weekly_panel.parquet", index=False)
    digest = hashlib.sha256((features / "weekly_panel.parquet").read_bytes()).hexdigest()
    manifest = {"raw_database_sha256": _raw_sha(raw_dir), "file_sha256": {"weekly_panel.parquet": digest}}
    (features / "manifest.json").write_text(json.dumps(manifest))
    panel, _, _ = prepare_inputs(features, raw_dir / "dataset")
    assert panel.attrs["raw_database_sha256"] == _raw_sha(raw_dir) and panel.attrs["weekly_sha256"] == digest
    (features / "manifest.json").write_text(json.dumps({**manifest, "raw_database_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="different raw databases"):
        prepare_inputs(features, raw_dir / "dataset")
    (features / "manifest.json").write_text(json.dumps({**manifest, "file_sha256": {"weekly_panel.parquet": "0" * 64}}))
    with pytest.raises(ValueError, match="checksum"):
        prepare_inputs(features)
    prepare_inputs(features, verify_checksum=False)  # 명시적으로 끄면 읽는다


def test_cli_info_and_strategy(bundle, other_dataset, tmp_path, capsys):
    summary = main(["info", "--model-dir", str(bundle)])
    assert summary["model_version"] == load_model(bundle).model_version
    ids = _customers_with_history(other_dataset, bundle)
    store = tmp_path / "store" / "feature_store.parquet"
    store.parent.mkdir()
    _population(other_dataset, ids, dataset_id="synthetic-fixture-seed1").to_parquet(store, index=False)
    summary = main(["strategy", "--features", str(store), "--weekly", str(other_dataset / "features"),
                    "--ledger", str(other_dataset / "dataset"), "--model-dir", str(bundle),
                    "--output", str(tmp_path / "clv_strategy.parquet")])
    assert summary["customers"] == len(ids) and summary["metadata"]["horizon_value"] == 52
    assert len(load_strategy_predictions(tmp_path / "clv_strategy.parquet").frame) == len(ids)
    # Uplift 결합용 검증 옵션(이 Fixture 번들은 기본 정책 control_no_coupon으로 학습했다)
    summary = main(["strategy", "--features", str(store), "--weekly", str(other_dataset / "features"),
                    "--model-dir", str(bundle), "--require-control-only",
                    "--output", str(tmp_path / "clv_strategy_checked.parquet")])
    assert summary["customers"] == len(ids)
    capsys.readouterr()


def test_strategy_records_unverified_provenance_and_week_mismatch(bundle, other_dataset):
    """체크섬을 건너뛰면 검증된 출처로 기록하지 않고, as_of_at과 다른 reference_week는 거절한다."""
    ids = _customers_with_history(other_dataset, bundle)
    population = _population(other_dataset, ids, dataset_id="synthetic-fixture-seed1")
    args = (other_dataset / "features", other_dataset / "dataset", bundle)
    result = predict_for_strategy(population, *args, verify_checksum=False)
    inputs = result.producer_metadata["inputs"]
    assert inputs["checksums_skipped"] and inputs["raw_database_sha256_verified_by"] == []
    assert inputs["raw_database_sha256_matched"] == ["ledger_manifest"]
    assert any("skipped" in warning for warning in result.producer_metadata["warnings"])
    assert result.producer_metadata["snapshots"][0]["input_cutoff_at"] == AS_OF.isoformat()  # UTC로 통일
    assert result.producer_metadata["inference_code"]["clv_version"]
    with pytest.raises(ValueError, match="Seoul Monday of as_of_at"):
        predict_for_strategy(population.assign(reference_week="2024-01-01"), *args)
    as_stamp = predict_for_strategy(population.assign(reference_week=pd.Timestamp("2024-10-07", tz="Asia/Seoul")), *args)
    assert as_stamp.frame["reference_week"].eq("2024-10-07").all()  # 시각으로 받아도 날짜 문자열로 통일


def test_same_seed_in_another_dataset_is_flagged(bundle, other_dataset):
    """학습 데이터셋과 seed가 같은 다른 데이터셋이면 학습 고객을 다시 만든 것일 수 있어 경고한다."""
    weekly = pd.read_parquet(other_dataset / "features" / "weekly_panel.parquet").assign(seed=0)
    ids = _customers_with_history(other_dataset, bundle)
    population = _population(other_dataset, ids, dataset_id="synthetic-fixture-seed1")
    result = predict_for_strategy(population, weekly, None, bundle)
    assert any("same simulator seed" in warning for warning in result.producer_metadata["warnings"])
