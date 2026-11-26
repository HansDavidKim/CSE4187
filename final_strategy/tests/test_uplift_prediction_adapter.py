"""Injected mock models only: no fitted Uplift bundle or producer runtime."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import EXPANDED_FEATURES
from final_strategy.inputs import integrate_inputs
from final_strategy.uplift_prediction_adapter import (
    COUPON_HISTORY_FEATURES, DIFFERENCE_ATOL, UPLIFT_FEATURES,
    UpliftModelMetadata, adapt_uplift_predictions,
)
from final_strategy.validation import CONTEXT_COLUMNS, InputValidationError
from final_strategy.tests.test_inputs import T0, batches_for, features


class MockModel:
    def __init__(self, transform=None):
        self.transform = transform
        self.calls = 0
        self.seen = None

    def predict(self, x):
        self.calls += 1
        self.seen = x.copy(deep=True)
        p0 = pd.Series(.5, index=x.index)
        p1 = x.visit_recency_days / 10
        result = pd.DataFrame({"p0": p0, "p1": p1, "uplift_score": p1 - p0}, index=x.index)
        return self.transform(result, x) if self.transform else result


@pytest.fixture
def source(features):
    frame = features.copy(deep=True)
    frame["customer_id"] = ["001", "NA", "003"]
    frame["visit_recency_days"] = [0., 5., 10.]
    return frame


@pytest.fixture
def declaration():
    return UpliftModelMetadata(
        model_version="mock-uplift-test-1", feature_names=UPLIFT_FEATURES,
        target_definition_id="mock_purchase_28d_coupon_assignment_v1",
        target="purchase_occurrence", horizon_days=28,
        treatment="coupon_assignment", control="no_coupon_assignment",
        policy_condition="mock_10pct_coupon_assignment_vs_no_assignment",
        training_source_version="mock-training-source-v1", feature_set_version="kit39-review-cart-v3",
        currency="SOURCE_UNSPECIFIED", amount_unit="0.01", model_artifact_sha256="c" * 64,
    )


def run(source, declaration, model=None, **kwargs):
    return adapt_uplift_predictions(model or MockModel(), source, declaration,
                                    input_cutoff_at=kwargs.pop("input_cutoff_at", T0), **kwargs)


def eligibility_for(source):
    frame = source[list(CONTEXT_COLUMNS)].copy()
    frame["eligible"] = [True, False, True]
    frame["exclusion_reason"] = [None, "caller_declared_insufficient_history", None]
    return frame


def test_exact_ordered_features_no_identifiers_and_null_preservation(source, declaration):
    model = MockModel()
    result = run(source, declaration, model)
    assert len(UPLIFT_FEATURES) == 36
    assert COUPON_HISTORY_FEATURES == tuple(EXPANDED_FEATURES[24:27])
    assert UPLIFT_FEATURES == (*EXPANDED_FEATURES[:24], *EXPANDED_FEATURES[27:])
    assert model.seen.columns.tolist() == list(UPLIFT_FEATURES)
    assert not set(CONTEXT_COLUMNS).intersection(model.seen.columns)
    assert not {"treatment", "y", *COUPON_HISTORY_FEATURES}.intersection(model.seen.columns)
    assert model.seen.index.is_unique and model.seen.index.name == "uplift_row"
    assert model.seen.mean_net_order_value_30d.isna().all()
    assert result.features.mean_net_order_value_30d.isna().all()
    assert result.batch.frame.customer_id.tolist() == ["001", "NA", "003"]
    assert result.batch.frame.uplift_score.tolist() == [-.5, 0., .5]
    assert result.scored_customer_count == result.input_customer_count == 3
    assert result.unscored_customer_count == 0


def test_metadata_context_and_horizon_are_preserved(source, declaration):
    result = run(source, declaration)
    assert result.batch.metadata.model_version == declaration.model_version
    assert result.batch.metadata.target_definition_id == declaration.target_definition_id
    assert result.batch.metadata.horizon_value == 28
    assert result.batch.metadata.horizon_unit == "days"
    assert result.batch.metadata.value_semantics == "probability_difference"
    for column in CONTEXT_COLUMNS:
        pd.testing.assert_series_equal(result.batch.frame[column], source[column])
    assert result.batch.frame.input_cutoff_at.eq(T0).all()
    assert result.batch.frame.prediction_end_at.eq(T0 + pd.Timedelta(days=28)).all()
    for key, value in asdict(declaration).items():
        assert result.producer_metadata[key] == value
    assert result.producer_metadata["input_cutoff_at"] == T0.isoformat()


def test_shuffled_source_restores_by_keys_not_customer_position(source, declaration):
    first = run(source, declaration)
    shuffled = source.iloc[[2, 0, 1]].copy()
    shuffled.index = [9, 9, 9]  # Source pandas index is not a customer identifier.
    second = run(shuffled, declaration)
    pd.testing.assert_frame_equal(first.batch.frame.sort_values("customer_id").reset_index(drop=True),
                                  second.batch.frame.sort_values("customer_id").reset_index(drop=True))


@pytest.mark.parametrize("mode", ["reset", "reorder", "duplicate", "drop", "extra", "rename", "wrong_token"])
def test_bad_output_index_is_not_repaired(source, declaration, mode):
    def change(out, x):
        if mode == "reset":
            return out.reset_index(drop=True)
        if mode == "reorder":
            return out.iloc[::-1]
        if mode == "duplicate":
            out.index = [out.index[0]] * len(out)
        if mode == "drop":
            return out.iloc[:-1]
        if mode == "extra":
            return pd.concat([out, out.iloc[[0]]])
        if mode == "rename":
            out.index = out.index.rename("wrong")
        if mode == "wrong_token":
            out.index = pd.Index(["alien", *out.index[1:]], name=out.index.name)
        return out
    with pytest.raises(InputValidationError, match="index"):
        run(source, declaration, MockModel(change))


@pytest.mark.parametrize("mode", ["array", "missing_column", "extra_column", "duplicate_column", "column_order", "evaluation"])
def test_bad_output_shape_or_evaluation_data_rejected(source, declaration, mode):
    def change(out, x):
        if mode == "array":
            return out.to_numpy()
        if mode == "missing_column":
            return out.drop(columns="p0")
        if mode == "extra_column":
            return out.assign(y=1)
        if mode == "duplicate_column":
            out.columns = ["p0", "p0", "uplift_score"]
        if mode == "column_order":
            return out[["p1", "p0", "uplift_score"]]
        if mode == "evaluation":
            out.attrs["evaluation_only"] = True
        return out
    with pytest.raises(InputValidationError):
        run(source, declaration, MockModel(change))


@pytest.mark.parametrize("column,value", [
    ("p0", -.01), ("p0", 1.01), ("p1", -.01), ("p1", 1.01),
    ("uplift_score", -1.01), ("uplift_score", 1.01),
    *[(c, v) for c in ("p0", "p1", "uplift_score") for v in (np.nan, np.inf, -np.inf, True)],
])
def test_invalid_numbers_rejected_without_clipping(source, declaration, column, value):
    def change(out, x):
        out[column] = out[column].astype(object)
        out.loc[out.index[0], column] = value
        return out
    with pytest.raises(InputValidationError, match=column):
        run(source, declaration, MockModel(change))


def test_probability_boundaries_and_negative_one_preserved(source, declaration):
    def change(out, x):
        out["p0"] = [1., 0., 0.]
        out["p1"] = [0., 1., 0.]
        out["uplift_score"] = [-1., 1., 0.]
        return out
    assert run(source, declaration, MockModel(change)).batch.frame.uplift_score.tolist() == [-1., 1., 0.]


@pytest.mark.parametrize("delta,accepted", [(np.finfo(float).eps, True), (1e-8, False)])
def test_difference_roundoff_policy(source, declaration, delta, accepted):
    def change(out, x):
        out.loc[out.index[1], "uplift_score"] = delta
        return out
    if accepted:
        result = run(source, declaration, MockModel(change))
        assert result.batch.frame.uplift_score.iloc[1] == delta  # Not recomputed.
        assert delta < DIFFERENCE_ATOL
    else:
        with pytest.raises(InputValidationError, match="p1 - p0"):
            run(source, declaration, MockModel(change))


@pytest.mark.parametrize("field,value", [
    ("model_version", ""), ("model_version", "T-learner"), ("model_version", "Xlearner"),
    ("model_version", "H_learner"), ("target_definition_id", " "),
    ("training_source_version", None), ("target", "visit"), ("horizon_days", 30),
    ("horizon_days", 27), ("horizon_days", True), ("horizon_days", 28.0),
    ("treatment", "coupon_redemption"), ("control", "coupon_assignment"),
    ("policy_condition", ""), ("feature_set_version", "kit39-review-cart-v2"),
    ("currency", "USD"), ("amount_unit", "1"), ("amount_unit", "0"),
    ("model_artifact_sha256", "bad"), ("model_artifact_sha256", "A" * 64),
    ("feature_names", UPLIFT_FEATURES[::-1]), ("feature_names", UPLIFT_FEATURES[:-1]),
    ("feature_names", (*UPLIFT_FEATURES, "y")), ("feature_names", (UPLIFT_FEATURES[0],) * 36),
    ("feature_names", "not a sequence of names"),
])
def test_bad_model_declaration_rejected_before_predict(source, declaration, field, value):
    model = MockModel()
    with pytest.raises(InputValidationError, match=field):
        run(source, replace(declaration, **{field: value}), model)
    assert model.calls == 0


def test_optional_artifact_sha_is_not_fabricated(source, declaration):
    result = run(source, replace(declaration, model_artifact_sha256=None))
    assert result.producer_metadata["model_artifact_sha256"] is None


@pytest.mark.parametrize("mode", ["duplicate", "cutoff", "dataset", "sha", "currency", "unit", "version", "future_column", "empty"])
def test_invalid_source_rejected_before_predict(source, declaration, mode):
    if mode == "duplicate":
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    elif mode == "cutoff":
        source.loc[0, "as_of_at"] += pd.Timedelta(days=1)
    elif mode == "dataset":
        source.loc[0, "dataset_id"] = "other"
    elif mode == "sha":
        source.loc[0, "raw_database_sha256"] = "f" * 64
    elif mode == "currency":
        source.loc[0, "currency"] = "USD"
    elif mode == "unit":
        source.loc[0, "amount_unit"] = "1"
    elif mode == "version":
        source["expanded_feature_set_version"] = "kit39-review-cart-v2"
    elif mode == "future_column":
        source["y"] = 1
    elif mode == "empty":
        source = source.iloc[:0]
    model = MockModel()
    with pytest.raises(InputValidationError):
        run(source, declaration, model)
    assert model.calls == 0


@pytest.mark.parametrize("cutoff", [None, "2024-03-03 15:00", T0 + pd.Timedelta(seconds=1), T0 - pd.Timedelta(days=1)])
def test_unknown_or_inconsistent_cutoff_rejected(source, declaration, cutoff):
    with pytest.raises(InputValidationError, match="input_cutoff_at"):
        run(source, declaration, input_cutoff_at=cutoff)


def test_seoul_cutoff_normalizes_to_utc(source, declaration):
    source["as_of_at"] = source.as_of_at.dt.tz_convert("Asia/Seoul")
    result = run(source, declaration, input_cutoff_at=T0.tz_convert("Asia/Seoul"))
    assert str(result.batch.frame.as_of_at.dtype) == "datetime64[ns, UTC]"
    assert result.batch.frame.reference_week.eq("2024-03-04").all()


def test_explicit_exclusions_preserved_with_exact_coverage(source, declaration):
    eligible = eligibility_for(source).iloc[::-1]
    result = run(source, declaration, eligibility=eligible)
    assert result.scored_customer_count == 2
    assert result.unscored_customer_count == 1
    assert result.scored_customer_count + result.unscored_customer_count == result.input_customer_count
    assert result.excluded.customer_id.tolist() == ["NA"]
    assert result.excluded.exclusion_reason.tolist() == ["caller_declared_insufficient_history"]
    assert result.excluded.exclusion_stage.eq("uplift_prediction").all()
    assert not result.excluded.eligible.any()
    assert result.batch.frame.customer_id.tolist() == ["001", "003"]


def test_all_excluded_skips_model_and_has_no_batch(source, declaration):
    eligible = eligibility_for(source)
    eligible["eligible"] = False
    eligible["exclusion_reason"] = "caller_declared_not_eligible"
    model = MockModel()
    result = run(source, declaration, model, eligibility=eligible)
    assert model.calls == 0 and result.batch is None and result.features.empty
    assert result.unscored_customer_count == result.input_customer_count == 3


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "reason", "eligible_reason", "boolean", "snapshot", "sha", "unit", "cutoff"])
def test_invalid_eligibility_does_not_hide_missing_predictions(source, declaration, mode):
    eligible = eligibility_for(source)
    if mode == "missing":
        eligible = eligible.iloc[:-1]
    elif mode == "extra":
        extra = eligible.iloc[[0]].assign(customer_id="alien")
        eligible = pd.concat([eligible, extra], ignore_index=True)
    elif mode == "duplicate":
        eligible = pd.concat([eligible, eligible.iloc[[0]]], ignore_index=True)
    elif mode == "reason":
        eligible["exclusion_reason"] = None
    elif mode == "eligible_reason":
        eligible.loc[0, "exclusion_reason"] = "contradiction"
    elif mode == "boolean":
        eligible["eligible"] = [1, 0, 1]
    elif mode == "snapshot":
        eligible.loc[0, "snapshot_id"] = "wrong"
    elif mode == "sha":
        eligible["raw_database_sha256"] = "d" * 64
    elif mode == "unit":
        eligible["amount_unit"] = "1"
    elif mode == "cutoff":
        eligible["as_of_at"] += pd.Timedelta(days=1)
    model = MockModel()
    with pytest.raises(InputValidationError):
        run(source, declaration, model, eligibility=eligible)
    assert model.calls == 0


@pytest.mark.parametrize("mode", ["value", "index", "index_values", "columns", "column_values", "attrs"])
def test_model_mutation_detected_original_preserved(source, declaration, mode):
    original = source.copy(deep=True)
    def mutate(out, x):
        if mode == "value":
            x.iloc[0, 0] = 999
        elif mode == "index":
            x.index = x.index[::-1]
        elif mode == "index_values":
            x.index.values[0] = "changed-token"
        elif mode == "columns":
            x["y"] = 1
        elif mode == "column_values":
            x.columns.values[0] = "changed-column"
        else:
            x.attrs["evaluation_only"] = True
        return out
    with pytest.raises(InputValidationError, match="mutated"):
        run(source, declaration, MockModel(mutate))
    pd.testing.assert_frame_equal(source, original)


def test_model_failure_never_returns_default_predictions(source, declaration):
    def fail(out, x):
        raise RuntimeError("mock model not fitted")
    with pytest.raises(InputValidationError, match="model call failed") as error:
        run(source, declaration, MockModel(fail))
    assert isinstance(error.value.__cause__, RuntimeError)


def test_requires_declared_metadata_and_callable_model(source, declaration):
    with pytest.raises(InputValidationError, match="UpliftModelMetadata"):
        run(source, asdict(declaration))
    with pytest.raises(InputValidationError, match="callable predict"):
        adapt_uplift_predictions(object(), source, declaration, input_cutoff_at=T0)


def test_inputs_metadata_and_output_are_independent(source, declaration):
    eligible = eligibility_for(source)
    before = source.copy(deep=True)
    before_eligibility = eligible.copy(deep=True)
    before_metadata = deepcopy(asdict(declaration))
    result = run(source, declaration, eligibility=eligible)
    result.features.iloc[0, 0] = "changed-result"
    result.producer_metadata["policy_condition"] = "changed-result"
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_frame_equal(eligible, before_eligibility)
    assert asdict(declaration) == before_metadata


@pytest.mark.parametrize("with_exclusions", [False, True])
def test_existing_integrate_inputs_with_dummy_churn_and_52_week_clv(source, declaration, with_exclusions):
    result = run(source, declaration, eligibility=eligibility_for(source) if with_exclusions else None)
    batches = batches_for(result.features)
    batches["uplift"] = result.batch
    integrated = integrate_inputs(result.features, batches)
    assert integrated.data.customer_id.tolist() == result.features.customer_id.tolist()
    assert integrated.data.uplift_score.tolist() == result.batch.frame.uplift_score.tolist()
    assert integrated.prediction_metadata["clv"].horizon_value == 52
    assert integrated.prediction_metadata["clv"].horizon_unit == "weeks"
    assert integrated.prediction_metadata["uplift"].horizon_value == 28
    assert integrated.prediction_metadata["churn"].target_definition_id != result.batch.metadata.target_definition_id
    assert "p0" not in integrated.data and "policy_condition" not in integrated.data
    assert "target_definition_id" in result.producer_metadata
