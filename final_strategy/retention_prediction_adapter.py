"""Adapt declared v5 Retention results without importing a model runtime."""
from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np
import pandas as pd

from .inputs import PredictionBatch
from .validation import (
    CONTEXT_COLUMNS, JOIN_KEYS, InputValidationError, PredictionMetadata,
    _frame, _strings, _times, _units,
    validate_feature_frame, validate_prediction_frame,
)


RETENTION_COLUMNS = (
    "customer_id", "dataset_id", "snapshot_id", "as_of_at", "raw_database_sha256",
    "currency", "amount_unit", "expanded_feature_set_version", "churn_probability",
    "status", "model_version", "definition_id", "target", "horizon_days", "policy_condition",
)
OPTIONAL_COLUMNS = ("as_of", "predicted_churn", "model", "threshold", "snapshot_scope")
STATUSES = ("scored", "not_signed_up_at_cutoff", "no_recent_session")
DEFINITION = "future_no_session_28d_active28d"
FEATURE_VERSION = "kit39-review-cart-v3"


@dataclass(frozen=True)
class RetentionPredictionResult:
    """Scored-only inputs plus excluded rows and unchanged producer semantics.

    Keep producer_metadata and excluded alongside downstream results: the existing
    PredictionMetadata/IntegratedInputs APIs have no policy_condition field.
    An all-unscored cohort has batch=None; it cannot be passed to integrate_inputs.
    """

    features: pd.DataFrame
    batch: PredictionBatch | None
    excluded: pd.DataFrame
    producer_metadata: dict
    input_customer_count: int

    @property
    def scored_customer_count(self):
        return len(self.features)

    @property
    def unscored_customer_count(self):
        return len(self.excluded)


def adapt_retention_predictions(predictions: pd.DataFrame, features: pd.DataFrame, *,
                               expected_policy_condition="observed_experiment_mix") -> RetentionPredictionResult:
    """Validate one complete dataset/cutoff cohort, then partition by producer status.

    Requires exact coverage, including unscored rows, against the supplied Feature
    Store frame. It does not silently select a snapshot or invent missing customers.
    The caller must explicitly request no_new_coupon to consume that producer policy.

    Cutoff/end derive from the inspected v5 prediction_inputs contract: observed
    records precede as_of_at, and the outcome is [as_of_at, as_of_at + 28 days).
    This declaration does not prove that an arbitrary DataFrame came from that API
    or that a model was trained without leakage.
    """
    name = "retention"
    if (not isinstance(expected_policy_condition, str)
            or expected_policy_condition not in ("observed_experiment_mix", "no_new_coupon")):
        raise InputValidationError("retention.policy_condition: unsupported expected policy")
    source = validate_feature_frame(features)
    if source.as_of_at.nunique() != 1:
        raise InputValidationError("retention.features.as_of_at: select exactly one cutoff cohort")
    if not source.expanded_feature_set_version.eq(FEATURE_VERSION).all():
        raise InputValidationError("retention.features.expanded_feature_set_version: requires v3")
    frame = _frame(predictions, RETENTION_COLUMNS, (*RETENTION_COLUMNS, *OPTIONAL_COLUMNS), name)
    _strings(frame, [c for c in RETENTION_COLUMNS
                     if c not in {"as_of_at", "amount_unit", "churn_probability", "horizon_days"}], name)
    frame["as_of_at"] = _times(frame.as_of_at, "retention.as_of_at")
    frame["amount_unit"] = _units(frame.amount_unit, "retention.amount_unit")
    if frame.customer_id.duplicated().any():
        raise InputValidationError("retention.customer_id: duplicate customer in one cutoff cohort")
    for column in ("dataset_id", "as_of_at"):
        if not frame[column].eq(source[column].iloc[0]).all():
            raise InputValidationError(f"retention.{column}: disagrees with Feature Store cohort")
    if "as_of" in frame and not _times(frame.as_of, "retention.as_of").eq(frame.as_of_at).all():
        raise InputValidationError("retention.as_of: disagrees with as_of_at")
    if not frame.status.isin(STATUSES).all():
        raise InputValidationError("retention.status: unsupported producer status")
    declarations = {"definition_id": DEFINITION, "target": "visit",
                    "policy_condition": expected_policy_condition,
                    "expanded_feature_set_version": FEATURE_VERSION}
    for column, expected in declarations.items():
        if not frame[column].eq(expected).all():
            raise InputValidationError(f"retention.{column}: expected {expected}")
    if not frame.horizon_days.map(lambda value: isinstance(value, numbers.Integral)
                                 and not isinstance(value, (bool, np.bool_)) and value == 28).all():
        raise InputValidationError("retention.horizon_days: requires the declared integer 28 days")
    if frame.model_version.nunique() != 1:
        raise InputValidationError("retention.model_version: mixed model versions")

    index = pd.MultiIndex.from_frame(source[list(JOIN_KEYS)])
    actual = pd.MultiIndex.from_frame(frame[list(JOIN_KEYS)])
    missing, extra = index.difference(actual), actual.difference(index)
    if len(missing) or len(extra):
        raise InputValidationError(f"retention: customer coverage mismatch; missing={len(missing)}, unexpected={len(extra)}")
    frame = frame.set_index(list(JOIN_KEYS)).reindex(index).reset_index()
    for column in ("snapshot_id", "raw_database_sha256", "currency", "amount_unit",
                   "expanded_feature_set_version"):
        if not frame[column].eq(source[column]).all():
            raise InputValidationError(f"retention.{column}: disagrees with Feature Store")

    scored = frame.status.eq("scored")
    if frame.loc[~scored, "churn_probability"].notna().any():
        raise InputValidationError("retention.churn_probability: unscored customers must have NULL probability")
    metadata = PredictionMetadata(frame.model_version.iloc[0], DEFINITION, 28, "days", "probability")
    producer_metadata = {column: frame[column].iloc[0] for column in
                         ("model_version", "definition_id", "target", "policy_condition", "expanded_feature_set_version")}
    producer_metadata["horizon_days"] = 28
    scored_features = source.loc[scored].copy().reset_index(drop=True)
    batch = None
    if scored.any():
        values = source.loc[scored, list(CONTEXT_COLUMNS)].copy().reset_index(drop=True)
        values["churn_probability"] = frame.loc[scored, "churn_probability"].to_numpy()
        values["input_cutoff_at"] = values.as_of_at
        try:
            values["prediction_end_at"] = values.as_of_at + pd.Timedelta(days=28)
        except (ValueError, OverflowError) as exc:
            raise InputValidationError("retention.prediction_end_at: 28-day horizon exceeds timestamp range") from exc
        batch = PredictionBatch(validate_prediction_frame(values, metadata, "churn"), metadata)
    # No model extras, future outcomes, or synthesized provenance enter predictors.
    excluded = frame.loc[~scored, list(RETENTION_COLUMNS)].copy()
    excluded["reference_week"] = source.loc[~scored, "reference_week"]
    excluded["exclusion_stage"] = "retention_prediction"
    excluded["exclusion_reason"] = excluded.status
    excluded = excluded.reset_index(drop=True)
    return RetentionPredictionResult(scored_features, batch, excluded, producer_metadata, len(source))
