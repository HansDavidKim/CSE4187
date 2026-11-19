"""Consumer contracts exercised with dummy predictions, without trained models."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import (
    BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION,
)
from simulate.v4.storage import FEATURE_KEYS

from final_strategy.inputs import PredictionBatch, integrate_inputs, load_feature_store
from final_strategy.validation import (
    CONTEXT_COLUMNS, PREDICTION_COLUMNS, InputValidationError, PredictionMetadata,
    validate_feature_frame, validate_prediction_frame,
)


T0 = pd.Timestamp("2024-03-03T15:00:00Z")  # Seoul Monday midnight


@pytest.fixture
def features():
    rows = []
    for customer in ("001", "002", "003"):
        row = dict.fromkeys(EXPANDED_FEATURES, 0.0)
        row.update(customer_id=customer, dataset_id="dummy", as_of_at=T0,
                   reference_week="2024-03-04", snapshot_id=f"opaque:{customer}",
                   currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
                   raw_database_sha256="a" * 64, source_model_sha256="b" * 64,
                   feature_set_version="kit32-v1",
                   behavior_feature_set_version=BEHAVIOR_FEATURE_VERSION,
                   expanded_feature_set_version=EXPANDED_FEATURE_VERSION)
        row["mean_net_order_value_30d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def batches_for(features):
    declarations = {
        "churn": PredictionMetadata("dummy-churn", "dummy_no_session_28d", 28, "days", "probability"),
        "uplift": PredictionMetadata("dummy-uplift", "dummy_purchase_retention_4w_coupon", 4, "weeks", "probability_difference"),
        "clv": PredictionMetadata("dummy-clv", "dummy_paid_revenue_no_new_coupon", 12, "calendar_months", "nonnegative_amount"),
    }
    batches = {}
    for model, metadata in declarations.items():
        frame = features[list(CONTEXT_COLUMNS)].copy()
        frame["input_cutoff_at"] = frame.as_of_at
        offset = (pd.DateOffset(months=12) if model == "clv" else
                  pd.Timedelta(days=28) if model == "churn" else pd.Timedelta(weeks=4))
        frame["prediction_end_at"] = frame.as_of_at.map(lambda value: value + offset)
        frame[PREDICTION_COLUMNS[model]] = {"churn": .25, "uplift": -.2, "clv": 123.45}[model]
        batches[model] = PredictionBatch(frame, metadata)
    return batches


@pytest.fixture
def predictions(features):
    return batches_for(features)


def test_three_customers_join_by_keys_not_row_position(features, predictions):
    predictions["churn"].frame["churn_probability"] = [.1, .5, .9]
    predictions["churn"] = replace(predictions["churn"], frame=predictions["churn"].frame.iloc[::-1])
    original = features.copy(deep=True)
    result = integrate_inputs(features, predictions)
    assert result.data.customer_id.tolist() == ["001", "002", "003"]
    assert result.data.churn_probability.tolist() == [.1, .5, .9]
    assert result.data.uplift_score.tolist() == [-.2, -.2, -.2]
    assert result.data.predicted_clv_12m.tolist() == [123.45] * 3
    assert result.data.mean_net_order_value_30d.isna().all()
    assert set(result.data) == {*FEATURE_KEYS, *EXPANDED_FEATURES, *PREDICTION_COLUMNS.values()}
    pd.testing.assert_frame_equal(features, original)


def test_multiple_snapshots_in_same_week_are_preserved(features):
    second = features.copy()
    second["as_of_at"] += pd.Timedelta(days=1)
    second["snapshot_id"] += ":second"
    combined = pd.concat([features, second], ignore_index=True)
    result = integrate_inputs(combined, batches_for(combined))
    assert len(result.data) == 6
    assert result.data.as_of_at.nunique() == 2
    assert result.data.reference_week.nunique() == 1


@pytest.mark.parametrize("model", ["features", "churn", "uplift", "clv"])
def test_duplicate_snapshot_is_rejected(features, predictions, model):
    if model == "features":
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    else:
        batch = predictions[model]
        predictions[model] = replace(batch, frame=pd.concat([batch.frame, batch.frame.iloc[[0]]]))
    with pytest.raises(InputValidationError, match="duplicate customer/snapshot"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("column", ["customer_id", "as_of_at", "raw_database_sha256", "churn_probability"])
def test_required_prediction_columns(features, predictions, column):
    predictions["churn"].frame.drop(columns=column, inplace=True)
    with pytest.raises(InputValidationError, match="missing columns"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("column,value", [("customer_id", None), ("customer_id", ""), ("as_of_at", None),
                                          ("as_of_at", "2024-03-04"), ("raw_database_sha256", "unknown")])
def test_missing_keys_naive_clocks_and_unknown_provenance(features, predictions, column, value):
    features[column] = value
    with pytest.raises(InputValidationError):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("column,value,match", [
    ("dataset_id", "other", "coverage mismatch"),
    ("raw_database_sha256", "c" * 64, "raw_database_sha256 disagrees"),
    ("snapshot_id", "another-opaque-id", "snapshot_id disagrees"),
    ("currency", "KRW", "currency disagrees"),
    ("amount_unit", "1", "amount_unit disagrees"),
])
def test_model_context_must_match_store(features, predictions, column, value, match):
    predictions["clv"].frame[column] = value
    with pytest.raises(InputValidationError, match=match):
        integrate_inputs(features, predictions)


def test_mixed_datasets_are_rejected(features, predictions):
    features.loc[0, "dataset_id"] = "other"
    with pytest.raises(InputValidationError, match="mixed dataset_id"):
        integrate_inputs(features, predictions)


def test_equivalent_utc_and_seoul_clocks_and_decimal_units(features, predictions):
    for batch in predictions.values():
        for column in ("as_of_at", "input_cutoff_at", "prediction_end_at"):
            batch.frame[column] = batch.frame[column].dt.tz_convert("Asia/Seoul")
        batch.frame["reference_week"] = pd.Timestamp("2024-03-04T00:00:00+09:00")
        batch.frame["amount_unit"] = "0.0100"
    result = integrate_inputs(features, predictions)
    assert str(result.data.as_of_at.dt.tz) == "UTC"
    assert result.data.amount_unit.eq("0.01").all()


@pytest.mark.parametrize("week", ["2024-03-11", "2024-03-05", "2024-03-04T01:00:00+09:00"])
def test_reference_week_must_match_local_clock(features, predictions, week):
    predictions["uplift"].frame["reference_week"] = week
    with pytest.raises(InputValidationError, match="reference_week"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("case", ["missing", "extra", "wrong_time"])
def test_no_silent_customer_or_snapshot_loss(features, predictions, case):
    batch = predictions["churn"]
    frame = batch.frame.copy()
    if case == "missing":
        frame = frame.iloc[:-1]
    elif case == "extra":
        extra = frame.iloc[[0]].copy()
        extra["customer_id"] = "unexpected"
        frame = pd.concat([frame, extra])
    else:
        frame["as_of_at"] += pd.Timedelta(hours=1)
        frame["prediction_end_at"] += pd.Timedelta(hours=1)
    predictions["churn"] = replace(batch, frame=frame)
    with pytest.raises(InputValidationError, match="coverage mismatch"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("value", [-.01, 1.01, np.nan, np.inf, -np.inf, True, "invalid", 1+2j])
def test_invalid_churn_probability(features, predictions, value):
    predictions["churn"].frame["churn_probability"] = value
    with pytest.raises(InputValidationError):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("model,value", [("uplift", -1.01), ("uplift", 1.01), ("uplift", np.nan),
                                        ("clv", -1), ("clv", np.inf), ("clv", np.nan)])
def test_invalid_uplift_and_clv_values(features, predictions, model, value):
    predictions[model].frame[PREDICTION_COLUMNS[model]] = value
    with pytest.raises(InputValidationError):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("value", [-123.45, 0.0, 123.45])
def test_finite_clv_amount_does_not_assume_a_sign_contract(features, predictions, value):
    batch = predictions["clv"]
    batch.frame["predicted_clv_12m"] = value
    predictions["clv"] = replace(batch, metadata=replace(batch.metadata, value_semantics="amount"))
    result = integrate_inputs(features, predictions)
    assert result.data.predicted_clv_12m.eq(value).all()
    assert result.prediction_metadata["clv"].value_semantics == "amount"


@pytest.mark.parametrize("model", ["features", "churn", "clv"])
def test_temporal_values_are_not_numeric_inputs(features, predictions, model):
    if model == "features":
        features["net_spend_30d"] = T0
    else:
        predictions[model].frame[PREDICTION_COLUMNS[model]] = T0
    with pytest.raises(InputValidationError, match="real numeric"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("source", ["features", "prediction"])
def test_mixed_raw_sha_is_rejected(features, predictions, source):
    frame = features if source == "features" else predictions["churn"].frame
    frame.loc[0, "raw_database_sha256"] = "c" * 64
    with pytest.raises(InputValidationError, match="mixed raw_database_sha256"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", True])
def test_invalid_amount_units(features, predictions, value):
    predictions["clv"].frame["amount_unit"] = value
    with pytest.raises(InputValidationError, match="amount unit"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("column", ["churn", "future_revenue_12m", "treatment", "actual_sent", "redeemed",
                                    "canonical_type", "evaluation_only", "future_coupon_used"])
@pytest.mark.parametrize("source", ["features", "prediction"])
def test_unregistered_outcome_and_experiment_columns_fail_closed(features, predictions, column, source):
    frame = features if source == "features" else predictions["uplift"].frame
    frame[column] = 1
    with pytest.raises(InputValidationError, match="unexpected columns"):
        integrate_inputs(features, predictions)


def test_declared_future_input_cutoff_is_rejected(features, predictions):
    predictions["uplift"].frame["input_cutoff_at"] += pd.Timedelta(seconds=1)
    with pytest.raises(InputValidationError, match="future inputs"):
        integrate_inputs(features, predictions)


def test_model_definitions_and_horizon_units_remain_distinct(features, predictions):
    result = integrate_inputs(features, predictions)
    assert result.prediction_metadata["churn"].horizon_unit == "days"
    assert result.prediction_metadata["uplift"].horizon_unit == "weeks"
    assert result.prediction_metadata["clv"].horizon_unit == "calendar_months"
    assert len({item.target_definition_id for item in result.prediction_metadata.values()}) == 3
    assert "prediction_end_at" not in result.data
    assert result.prediction_context["clv"].prediction_end_at.eq(T0 + pd.DateOffset(months=12)).all()


def test_30_days_is_not_silently_accepted_as_28_days(features, predictions):
    predictions["churn"].frame["prediction_end_at"] = T0 + pd.Timedelta(days=30)
    with pytest.raises(InputValidationError, match="declared horizon"):
        integrate_inputs(features, predictions)


@pytest.mark.parametrize("start,week,end,wrong_days", [
    ("2024-02-29T00:00:00Z", "2024-02-26", "2025-02-28T00:00:00Z", 360),
    ("2023-03-31T00:00:00Z", "2023-03-27", "2024-03-31T00:00:00Z", 365),
])
def test_calendar_month_horizon_handles_leap_year_and_month_end(features, start, week, end, wrong_days):
    features["as_of_at"] = pd.Timestamp(start)
    features["reference_week"] = week
    batch = batches_for(features)["clv"]
    assert batch.frame.prediction_end_at.eq(pd.Timestamp(end)).all()
    validate_prediction_frame(batch.frame, batch.metadata, "clv")
    batch.frame["prediction_end_at"] = features.as_of_at + pd.Timedelta(days=wrong_days)
    with pytest.raises(InputValidationError, match="declared horizon"):
        validate_prediction_frame(batch.frame, batch.metadata, "clv")


def test_explicit_30_day_churn_contract_is_preserved(features, predictions):
    batch = predictions["churn"]
    metadata = replace(batch.metadata, target_definition_id="dummy_no_session_30d", horizon_value=30)
    batch.frame["prediction_end_at"] = T0 + pd.Timedelta(days=30)
    predictions["churn"] = replace(batch, metadata=metadata)
    result = integrate_inputs(features, predictions)
    assert result.prediction_metadata["churn"].horizon_value == 30
    assert result.prediction_metadata["uplift"].horizon_value == 4


def test_empty_or_duplicate_column_inputs_are_rejected(features):
    with pytest.raises(InputValidationError, match="cannot be empty"):
        validate_feature_frame(features.iloc[:0])
    duplicate = pd.concat([features, features[["customer_id"]]], axis=1)
    with pytest.raises(InputValidationError, match="duplicate column"):
        validate_feature_frame(duplicate)


@pytest.mark.parametrize("changes", [dict(horizon_value=0), dict(horizon_value=True), dict(horizon_unit="months"),
                                     dict(target_definition_id=""), dict(model_version=""),
                                     dict(value_semantics="unknown")])
def test_explicit_valid_metadata_is_required(predictions, changes):
    batch = predictions["churn"]
    with pytest.raises(InputValidationError):
        validate_prediction_frame(batch.frame, replace(batch.metadata, **changes), "churn")


def test_unrepresentable_horizon_has_a_field_specific_error(predictions):
    batch = predictions["churn"]
    with pytest.raises(InputValidationError, match="horizon.*timestamp range"):
        validate_prediction_frame(batch.frame, replace(batch.metadata, horizon_value=10**100), "churn")


def test_clv_12m_does_not_accept_day_horizon(predictions):
    batch = predictions["clv"]
    with pytest.raises(InputValidationError, match="twelve calendar months"):
        validate_prediction_frame(batch.frame, replace(batch.metadata, horizon_unit="days", horizon_value=365), "clv")


def test_all_three_batches_and_current_feature_version_required(features, predictions):
    with pytest.raises(InputValidationError, match="exactly churn"):
        integrate_inputs(features, {"churn": predictions["churn"]})
    features["expanded_feature_set_version"] = "old"
    with pytest.raises(InputValidationError, match="unsupported version"):
        validate_feature_frame(features)


@pytest.fixture
def published_store(features, tmp_path, monkeypatch):
    from simulate.v4 import storage

    metadata = dict(dataset_id="dummy", raw_database_sha256="a" * 64, model_sha256="b" * 64)
    # Exercise the real publisher/manifest, without raw generation or KIT39 recomputation.
    monkeypatch.setattr(storage, "load_tables", lambda path: {"_metadata": metadata})
    monkeypatch.setattr(storage, "build_features", lambda tables, as_of_at=None: features.copy())

    output = tmp_path / "store"
    storage.materialize_features("unused", output)
    return output


def test_loader_reads_actual_simulator_publication(published_store):
    output = published_store
    loaded = load_feature_store(output)
    assert loaded.customer_id.tolist() == ["001", "002", "003"]
    assert loaded.mean_net_order_value_30d.isna().all()
    assert len(integrate_inputs(loaded, batches_for(loaded)).data) == 3
    pd.testing.assert_frame_equal(loaded, load_feature_store(output / "feature_store.parquet"))


def test_loader_detects_parquet_tampering(published_store):
    path = published_store / "feature_store.parquet"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(InputValidationError, match="checksum mismatch"):
        load_feature_store(published_store)


@pytest.mark.parametrize("field,value", [("dataset_id", "wrong"), ("raw_database_sha256", "c" * 64),
                                        ("rows", {"features": 999}), ("snapshot_times_utc", []),
                                        ("predictors", ["future_revenue_12m"])])
def test_loader_checks_manifest_contract(published_store, field, value):
    path = published_store / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InputValidationError, match="manifest"):
        load_feature_store(published_store)


def test_loader_rejects_outcome_file_or_evaluation_directory(tmp_path):
    with pytest.raises(InputValidationError, match="expected feature_store"):
        load_feature_store(tmp_path / "labels.parquet")
    with pytest.raises(InputValidationError, match="evaluation_only"):
        load_feature_store(tmp_path / "evaluation_only" / "feature_store.parquet")


def test_loader_requires_manifest(tmp_path):
    with pytest.raises(InputValidationError, match="manifest.json"):
        load_feature_store(tmp_path / "feature_store.parquet")
