"""Independent producer-shaped fixtures; no Retention model or bundle is loaded."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION
from simulate.v4.storage import FEATURE_KEYS
from final_strategy.inputs import PredictionBatch, integrate_inputs
from final_strategy.retention_prediction_adapter import (
    DEFINITION, RETENTION_COLUMNS, adapt_retention_predictions,
)
from final_strategy.validation import CONTEXT_COLUMNS, InputValidationError, PredictionMetadata


T0 = pd.Timestamp("2024-03-03T15:00:00Z")


@pytest.fixture
def features():
    rows = []
    for customer in ("001", "NA", "003"):
        row = dict.fromkeys(EXPANDED_FEATURES, 0.)
        row.update(customer_id=customer, dataset_id="adapter-fixture", as_of_at=T0,
                   reference_week="2024-03-04", snapshot_id=f"opaque::{customer}",
                   currency="SOURCE_UNSPECIFIED", amount_unit="0.01", raw_database_sha256="a" * 64,
                   source_model_sha256="b" * 64, feature_set_version="kit32-v1",
                   behavior_feature_set_version=BEHAVIOR_FEATURE_VERSION,
                   expanded_feature_set_version=EXPANDED_FEATURE_VERSION)
        row["mean_net_order_value_30d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def predictions(features):
    # Mirrors pipeline.infer's simulator-mode DataFrame, including optional diagnostics.
    result = features[["customer_id", "dataset_id", "snapshot_id", "as_of_at", "raw_database_sha256",
                       "currency", "amount_unit", "expanded_feature_set_version"]].copy()
    result = result.assign(churn_probability=[.1, .8, .4], status="scored", model_version="fixture-model-1",
                           definition_id=DEFINITION, target="visit", horizon_days=28,
                           policy_condition="observed_experiment_mix", model="fixture-logistic",
                           threshold=.5, snapshot_scope="other_snapshot")
    result["as_of"] = result.as_of_at
    result["predicted_churn"] = pd.Series([0, 1, 0], dtype="Int64")
    result.attrs["runtime_profile"] = {"fixture_only": True}
    return result


def exclude(frame, index, status="no_recent_session"):
    frame.loc[index, "status"] = status
    frame.loc[index, "churn_probability"] = np.nan
    frame.loc[index, "predicted_churn"] = pd.NA


def dummy_batches(features, churn):
    batches = {"churn": churn}
    for model, column, metadata, value, end in (
        ("uplift", "uplift_score", PredictionMetadata("dummy-uplift", "dummy_purchase_retention_coupon",
                                                     4, "weeks", "probability_difference"), -.2,
         T0 + pd.Timedelta(days=28)),
        ("clv", "predicted_clv_12m", PredictionMetadata("dummy-clv", "dummy_paid_revenue",
                                                       52, "weeks", "nonnegative_amount"), 100.,
         T0 + pd.Timedelta(weeks=52)),
    ):
        frame = features[list(CONTEXT_COLUMNS)].copy()
        frame["input_cutoff_at"] = frame.as_of_at
        frame["prediction_end_at"] = end
        frame[column] = value
        batches[model] = PredictionBatch(frame, metadata)
    return batches


def test_scored_contract_and_original_ids(predictions, features):
    result = adapt_retention_predictions(predictions, features)
    assert isinstance(result.batch, PredictionBatch)
    assert result.batch.frame.customer_id.tolist() == ["001", "NA", "003"]
    assert result.batch.frame.churn_probability.tolist() == [.1, .8, .4]
    assert result.batch.metadata == PredictionMetadata("fixture-model-1", DEFINITION, 28, "days", "probability")
    assert result.batch.frame.input_cutoff_at.eq(T0).all()
    assert result.batch.frame.prediction_end_at.eq(T0 + pd.Timedelta(days=28)).all()
    assert result.batch.frame.reference_week.eq("2024-03-04").all()
    assert result.batch.frame.snapshot_id.tolist() == features.snapshot_id.tolist()
    assert result.features.mean_net_order_value_30d.isna().all()
    assert result.excluded.empty
    assert result.input_customer_count == result.scored_customer_count == 3
    assert result.unscored_customer_count == 0


def test_unscored_audit_and_population_reconciliation(predictions, features):
    exclude(predictions, 1, "not_signed_up_at_cutoff")
    exclude(predictions, 2)
    result = adapt_retention_predictions(predictions, features)
    assert result.features.customer_id.tolist() == ["001"]
    assert result.batch.frame.customer_id.tolist() == ["001"]
    assert result.excluded.customer_id.tolist() == ["NA", "003"]
    assert result.excluded.status.tolist() == ["not_signed_up_at_cutoff", "no_recent_session"]
    assert result.excluded.exclusion_reason.equals(result.excluded.status)
    assert result.excluded.exclusion_stage.eq("retention_prediction").all()
    assert result.excluded.churn_probability.isna().all()
    assert result.excluded.snapshot_id.tolist() == features.snapshot_id.iloc[1:].tolist()
    assert result.excluded.raw_database_sha256.eq("a" * 64).all()
    assert result.excluded.as_of_at.eq(T0).all()
    assert result.input_customer_count == result.scored_customer_count + result.unscored_customer_count == 3


def test_all_unscored_preserves_audit_without_invalid_empty_batch(predictions, features):
    for i in predictions.index:
        exclude(predictions, i)
    result = adapt_retention_predictions(predictions, features)
    assert result.batch is None and result.features.empty
    assert list(result.features) == [*FEATURE_KEYS, *EXPANDED_FEATURES]
    assert result.input_customer_count == result.unscored_customer_count == 3
    assert result.scored_customer_count == 0
    assert result.producer_metadata["policy_condition"] == "observed_experiment_mix"


@pytest.mark.parametrize("value", [-.01, 1.01, np.nan, np.inf, -np.inf, True, "invalid", 1j])
def test_bad_scored_probability(predictions, features, value):
    predictions["churn_probability"] = predictions.churn_probability.astype(object)
    predictions.loc[0, "churn_probability"] = value
    with pytest.raises(InputValidationError, match="churn_probability"):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("value", [0., 1.])
def test_probability_boundaries(predictions, features, value):
    predictions["churn_probability"] = value
    assert adapt_retention_predictions(predictions, features).batch.frame.churn_probability.eq(value).all()


@pytest.mark.parametrize("value", [0., .5, np.inf, "NaN"])
def test_unscored_requires_actual_null(predictions, features, value):
    predictions["churn_probability"] = predictions.churn_probability.astype(object)
    predictions.loc[1, "status"] = "no_recent_session"
    predictions.loc[1, "churn_probability"] = value
    with pytest.raises(InputValidationError, match="unscored customers must have NULL"):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("column", RETENTION_COLUMNS)
def test_missing_producer_field(predictions, features, column):
    with pytest.raises(InputValidationError, match="missing columns"):
        adapt_retention_predictions(predictions.drop(columns=column), features)


@pytest.mark.parametrize("column,value", [
    ("dataset_id", "another"), ("snapshot_id", "reconstructed"), ("raw_database_sha256", "c" * 64),
    ("currency", "KRW"), ("amount_unit", "1"), ("amount_unit", "NaN"),
    ("expanded_feature_set_version", "kit39-review-cart-v2"),
    ("definition_id", "future_purchase"), ("target", "purchase"), ("horizon_days", 30),
    ("horizon_days", 28.5), ("horizon_days", True), ("horizon_days", "28"),
    ("policy_condition", "no_new_coupon"), ("status", "unknown"),
    ("model_version", ""), ("model_version", None), ("customer_id", 1),
    ("customer_id", ""), ("snapshot_id", None),
])
def test_invalid_or_inconsistent_declarations(predictions, features, column, value):
    predictions[column] = predictions[column].astype(object)
    predictions.loc[0, column] = value
    with pytest.raises(InputValidationError, match=column):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("value", ["2024-03-03T15:00:00", None, "2024-03-04T15:00:00Z"])
def test_bad_or_wrong_cutoff(predictions, features, value):
    predictions["as_of_at"] = predictions.as_of_at.astype(object)
    predictions.loc[0, "as_of_at"] = value
    with pytest.raises(InputValidationError, match="as_of_at"):
        adapt_retention_predictions(predictions, features)


def test_equivalent_seoul_clock_and_decimal_unit(predictions, features):
    predictions["as_of_at"] = predictions.as_of_at.dt.tz_convert("Asia/Seoul")
    predictions["as_of"] = predictions.as_of_at
    predictions["amount_unit"] = "0.0100"
    result = adapt_retention_predictions(predictions, features)
    assert str(result.batch.frame.as_of_at.dt.tz) == "UTC"
    assert result.batch.frame.amount_unit.eq("0.01").all()


def test_optional_cutoff_alias_must_agree(predictions, features):
    predictions["as_of"] += pd.Timedelta(days=1)
    with pytest.raises(InputValidationError, match="retention.as_of:"):
        adapt_retention_predictions(predictions, features)


def test_duplicate_customer(predictions, features):
    predictions.loc[1, "customer_id"] = "001"
    with pytest.raises(InputValidationError, match="duplicate customer"):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_exact_population_including_unscored(predictions, features, case):
    exclude(predictions, 1)
    if case == "missing":
        predictions = predictions.drop(index=1)
    else:
        extra = predictions.iloc[[1]].assign(customer_id="unexpected")
        predictions = pd.concat([predictions, extra], ignore_index=True)
    with pytest.raises(InputValidationError, match="customer coverage mismatch"):
        adapt_retention_predictions(predictions, features)


def test_unscored_provenance_is_not_ignored(predictions, features):
    exclude(predictions, 1)
    predictions.loc[1, "snapshot_id"] = "wrong"
    with pytest.raises(InputValidationError, match="snapshot_id"):
        adapt_retention_predictions(predictions, features)


def test_mixed_model_versions(predictions, features):
    predictions.loc[1, "model_version"] = "other-model"
    with pytest.raises(InputValidationError, match="mixed model versions"):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("column,value", [("reference_week", "2024-02-26"),
                                          ("expanded_feature_set_version", "kit39-review-cart-v2")])
def test_feature_contract_is_reused(predictions, features, column, value):
    features[column] = value
    with pytest.raises(InputValidationError, match=column):
        adapt_retention_predictions(predictions, features)


def test_one_cutoff_required(predictions, features):
    features.loc[1, "as_of_at"] += pd.Timedelta(days=1)
    with pytest.raises(InputValidationError, match="exactly one cutoff"):
        adapt_retention_predictions(predictions, features)


def test_policy_and_target_preserved_without_relabeling_definition(predictions, features):
    result = adapt_retention_predictions(predictions, features)
    assert result.producer_metadata == dict(model_version="fixture-model-1", definition_id=DEFINITION,
        target="visit", horizon_days=28, policy_condition="observed_experiment_mix",
        expanded_feature_set_version="kit39-review-cart-v3")
    assert result.batch.metadata.target_definition_id == predictions.definition_id.iloc[0]
    predictions["policy_condition"] = "no_new_coupon"
    other = adapt_retention_predictions(predictions, features, expected_policy_condition="no_new_coupon")
    assert other.producer_metadata["policy_condition"] == "no_new_coupon"
    assert other.batch.metadata.target_definition_id == DEFINITION
    with pytest.raises(InputValidationError, match="unsupported expected policy"):
        adapt_retention_predictions(predictions, features, expected_policy_condition="invented")


def test_output_projection_drops_model_diagnostics(predictions, features):
    result = adapt_retention_predictions(predictions, features)
    assert set(result.batch.frame) == {*CONTEXT_COLUMNS, "churn_probability", "input_cutoff_at", "prediction_end_at"}
    assert not {"predicted_churn", "threshold", "model", "status"} & set(result.features)


@pytest.mark.parametrize("column", ["future_churn", "future_revenue", "treatment", "source_model_sha256"])
def test_unknown_or_future_columns_fail_closed(predictions, features, column):
    predictions[column] = 1
    with pytest.raises(InputValidationError, match="unexpected columns"):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("which", ["predictions", "features"])
def test_evaluation_only_rejected(predictions, features, which):
    {"predictions": predictions, "features": features}[which].attrs["evaluation_only"] = True
    with pytest.raises(InputValidationError, match="evaluation_only"):
        adapt_retention_predictions(predictions, features)


def test_duplicate_column_names_rejected(predictions, features):
    predictions = pd.concat([predictions, predictions[["status"]]], axis=1)
    with pytest.raises(InputValidationError, match="duplicate column"):
        adapt_retention_predictions(predictions, features)


@pytest.mark.parametrize("which", ["predictions", "features"])
def test_empty_input_is_not_an_all_unscored_cohort(predictions, features, which):
    if which == "predictions":
        predictions = predictions.iloc[:0]
    else:
        features = features.iloc[:0]
    with pytest.raises(InputValidationError, match="cannot be empty"):
        adapt_retention_predictions(predictions, features)


def test_only_required_producer_fields_are_sufficient(predictions, features):
    result = adapt_retention_predictions(predictions[list(RETENTION_COLUMNS)], features)
    assert result.scored_customer_count == 3


def test_invalid_policy_parameter_is_a_validation_error(predictions, features):
    with pytest.raises(InputValidationError, match="policy_condition"):
        adapt_retention_predictions(predictions, features, expected_policy_condition=np.array(["x", "y"]))


def test_timestamp_overflow_is_a_validation_error(predictions, features):
    point = pd.Timestamp("2262-04-01T00:00:00Z")
    features["as_of_at"] = point
    features["reference_week"] = "2262-03-31"
    predictions["as_of_at"] = point
    predictions["as_of"] = point
    with pytest.raises(InputValidationError, match="prediction_end_at"):
        adapt_retention_predictions(predictions, features)


def test_all_unscored_still_requires_valid_semantics(predictions, features):
    for i in predictions.index:
        exclude(predictions, i)
    predictions["target"] = "purchase"
    with pytest.raises(InputValidationError, match="target"):
        adapt_retention_predictions(predictions, features)


def test_integration_with_dummy_uplift_clv_uses_exact_scored_population(predictions, features):
    exclude(predictions, 2)
    result = adapt_retention_predictions(predictions, features)
    batches = dummy_batches(result.features, result.batch)
    integrated = integrate_inputs(result.features, batches)
    assert integrated.data.customer_id.tolist() == ["001", "NA"]
    assert integrated.data.churn_probability.tolist() == [.1, .8]
    assert integrated.data.uplift_score.eq(-.2).all()
    assert integrated.data.predicted_clv_12m.eq(100.).all()
    assert integrated.prediction_metadata["churn"] == result.batch.metadata
    assert integrated.prediction_context["churn"].prediction_end_at.eq(T0 + pd.Timedelta(days=28)).all()
    assert result.excluded.customer_id.tolist() == ["003"]
    # The original full population is deliberately not a compatible scored-only input.
    with pytest.raises(InputValidationError, match="coverage mismatch"):
        integrate_inputs(features, batches)
    bad = dummy_batches(features, result.batch)
    with pytest.raises(InputValidationError, match="coverage mismatch"):
        integrate_inputs(result.features, bad)


def test_nonmutation_and_row_alignment(predictions, features):
    exclude(predictions, 1)
    features.attrs["note"] = {"nested": [1, 2]}
    before_p, before_f = predictions.copy(deep=True), features.copy(deep=True)
    attrs_p, attrs_f = deepcopy(predictions.attrs), deepcopy(features.attrs)
    expected = adapt_retention_predictions(predictions, features)
    result = adapt_retention_predictions(predictions.iloc[::-1], features)
    pd.testing.assert_frame_equal(expected.batch.frame, result.batch.frame)
    pd.testing.assert_frame_equal(expected.excluded, result.excluded)
    reordered = adapt_retention_predictions(predictions, features.iloc[::-1])
    pd.testing.assert_frame_equal(result.batch.frame.sort_values("customer_id").reset_index(drop=True),
                                  reordered.batch.frame.sort_values("customer_id").reset_index(drop=True))
    result.batch.frame.loc[0, "churn_probability"] = .99
    result.features.loc[0, "customer_id"] = "changed"
    result.excluded.loc[0, "status"] = "changed"
    result.producer_metadata["policy_condition"] = "changed"
    pd.testing.assert_frame_equal(predictions, before_p)
    pd.testing.assert_frame_equal(features, before_f)
    assert predictions.attrs == attrs_p and features.attrs == attrs_f


def test_import_does_not_load_retention_or_training_dependencies():
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(root / "simulator/src"), str(root)])}
    code = ("import sys; import final_strategy.retention_prediction_adapter; "
            "assert not any(k.split('.')[0] in {'retention','optuna','torch','lightgbm','catboost','xgboost'} "
            "for k in sys.modules)")
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
