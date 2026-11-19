"""Adapt an injected, trusted Uplift model without loading a model runtime."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import numbers
import re

import numpy as np
import pandas as pd

from simulate.v4.features import EXPANDED_FEATURES

from .inputs import PredictionBatch
from .validation import (
    CONTEXT_COLUMNS, JOIN_KEYS, InputValidationError, PredictionMetadata,
    _context, _frame, _numeric, _times, _units,
    validate_feature_frame, validate_prediction_frame,
)


FEATURE_VERSION = "kit39-review-cart-v3"
COUPON_HISTORY_FEATURES = (
    "coupon_uses_30d", "coupon_redemption_recency_days", "coupon_discount_amount_30d",
)
UPLIFT_FEATURES = tuple(name for name in EXPANDED_FEATURES if name not in COUPON_HISTORY_FEATURES)
PREDICTION_COLUMNS = ("p0", "p1", "uplift_score")
# Only subtraction roundoff is allowed; probabilities themselves are never clipped.
DIFFERENCE_ATOL = 8 * np.finfo(float).eps


@dataclass(frozen=True)
class UpliftModelMetadata:
    """Caller-verified training declaration, not a team-approved model registry.

    target/treatment/control are internal semantic tokens. target_definition_id
    and policy_condition must identify the producer's actual outcome and policy.
    A declaration alone cannot prove model provenance or feature-value parity.
    """

    model_version: str
    feature_names: tuple[str, ...]
    target_definition_id: str
    target: str
    horizon_days: int
    treatment: str
    control: str
    policy_condition: str
    training_source_version: str
    feature_set_version: str
    currency: str
    amount_unit: str
    model_artifact_sha256: str | None = None


@dataclass(frozen=True)
class UpliftPredictionResult:
    """Scored-only batch plus caller-declared exclusions and producer semantics.

    Preserve excluded and producer_metadata alongside downstream results. Existing
    PredictionMetadata does not contain policy/training declarations. If all rows
    are explicitly excluded, features is empty and batch is None.
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


def _model_metadata(metadata, source):
    if not isinstance(metadata, UpliftModelMetadata):
        raise InputValidationError("uplift.metadata: requires UpliftModelMetadata")
    declaration = asdict(metadata)
    for field in ("model_version", "target_definition_id", "target", "treatment", "control",
                  "policy_condition", "training_source_version", "feature_set_version", "currency"):
        if not isinstance(declaration[field], str) or not declaration[field].strip():
            raise InputValidationError(f"uplift.{field}: requires a nonempty producer declaration")
    if re.sub(r"[\s_-]", "", metadata.model_version).lower() in {"tlearner", "xlearner", "hlearner"}:
        raise InputValidationError("uplift.model_version: learner family alone is not a unique model version")
    names = metadata.feature_names
    if (not isinstance(names, (tuple, list))
            or any(not isinstance(name, str) for name in names)
            or tuple(names) != UPLIFT_FEATURES or len(UPLIFT_FEATURES) != 36):
        raise InputValidationError("uplift.feature_names: requires the exact ordered 36-feature contract")
    expected = {"target": "purchase_occurrence", "treatment": "coupon_assignment",
                "control": "no_coupon_assignment", "feature_set_version": FEATURE_VERSION}
    for field, value in expected.items():
        if declaration[field] != value:
            raise InputValidationError(f"uplift.{field}: expected {value}")
    if (isinstance(metadata.horizon_days, (bool, np.bool_))
            or not isinstance(metadata.horizon_days, numbers.Integral) or metadata.horizon_days != 28):
        raise InputValidationError("uplift.horizon_days: requires the declared integer 28 days")
    digest = metadata.model_artifact_sha256
    if digest is not None and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise InputValidationError("uplift.model_artifact_sha256: requires a lowercase SHA256 digest")
    unit = _units(pd.Series([metadata.amount_unit]), "uplift.amount_unit")[0]
    for field, value in (("currency", metadata.currency), ("amount_unit", unit)):
        if not source[field].eq(value).all():
            raise InputValidationError(f"uplift.{field}: disagrees with Feature Store")
    return declaration


def _eligibility(source, eligibility):
    if eligibility is None:
        return pd.Series(True, index=source.index), pd.Series(None, index=source.index, dtype=object)
    columns = (*CONTEXT_COLUMNS, "eligible", "exclusion_reason")
    frame = _context(_frame(eligibility, columns, columns, "uplift.eligibility"), "uplift.eligibility")
    expected = pd.MultiIndex.from_frame(source[list(JOIN_KEYS)])
    actual = pd.MultiIndex.from_frame(frame[list(JOIN_KEYS)])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise InputValidationError("uplift.eligibility: customer coverage mismatch")
    frame = frame.set_index(list(JOIN_KEYS)).reindex(expected).reset_index()
    for column in CONTEXT_COLUMNS:
        if not frame[column].eq(source[column]).all():
            raise InputValidationError(f"uplift.eligibility.{column}: disagrees with Feature Store")
    if not frame.eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise InputValidationError("uplift.eligibility.eligible: requires explicit booleans")
    eligible = frame.eligible.astype(bool)
    if not frame.loc[~eligible, "exclusion_reason"].map(
            lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise InputValidationError("uplift.eligibility.exclusion_reason: excluded rows require a declared reason")
    if frame.loc[eligible, "exclusion_reason"].notna().any():
        raise InputValidationError("uplift.eligibility.exclusion_reason: eligible rows must have NULL reason")
    return eligible, frame.exclusion_reason


def adapt_uplift_predictions(model, features: pd.DataFrame, metadata: UpliftModelMetadata, *,
                             input_cutoff_at, eligibility: pd.DataFrame | None = None) -> UpliftPredictionResult:
    """Call trusted model.predict(X) for one validated v3 dataset/cutoff cohort.

    The caller must verify the injected object and its training declaration. No
    pickle or Uplift package is loaded here. input_cutoff_at explicitly declares
    the exclusive historical boundary of the supplied v3 features and must equal
    their as_of_at. It is not an inferred training timestamp.

    Optional eligibility requires CONTEXT_COLUMNS, eligible (bool), and
    exclusion_reason for the complete cohort. No eligibility rules are inferred.
    Empty source frames are rejected by the existing Feature Store validator.
    """
    source = validate_feature_frame(features)
    if source.as_of_at.nunique() != 1:
        raise InputValidationError("uplift.features.as_of_at: select exactly one cutoff cohort")
    if not source.expanded_feature_set_version.eq(FEATURE_VERSION).all():
        raise InputValidationError("uplift.features.expanded_feature_set_version: requires v3")
    producer_metadata = _model_metadata(metadata, source)
    cutoff = _times(pd.Series([input_cutoff_at]), "uplift.input_cutoff_at").iloc[0]
    if cutoff != source.as_of_at.iloc[0]:
        raise InputValidationError("uplift.input_cutoff_at: must match the supplied Feature Store boundary")
    if not callable(getattr(model, "predict", None)):
        raise InputValidationError("uplift.model: requires a trusted object with callable predict(X)")
    eligible, reasons = _eligibility(source, eligibility)
    scored_features = source.loc[eligible].copy().reset_index(drop=True)
    excluded = source.loc[~eligible, list(CONTEXT_COLUMNS)].copy()
    excluded["eligible"] = False
    excluded["exclusion_stage"] = "uplift_prediction"
    excluded["exclusion_reason"] = reasons.loc[~eligible]
    producer_metadata["input_cutoff_at"] = cutoff.isoformat()
    batch = None
    if len(scored_features):
        # Opaque tokens keep identifiers out of X. The separate mapping binds each
        # token to validated full keys and per-customer snapshot/source metadata.
        tokens = pd.Index([f"uplift-row-{i}" for i in range(len(scored_features))], name="uplift_row")
        mapping = scored_features.loc[:, list(CONTEXT_COLUMNS)].copy()
        mapping.index = tokens.copy(deep=True)
        x = scored_features.loc[:, list(UPLIFT_FEATURES)].copy(deep=True)
        x.index = tokens.copy(deep=True)
        before = x.copy(deep=True)
        before.index = x.index.copy(deep=True)
        before.columns = x.columns.copy(deep=True)
        before.attrs = deepcopy(x.attrs)
        try:
            output = model.predict(x)
        except Exception as exc:
            raise InputValidationError("uplift.predict: model call failed") from exc
        try:
            pd.testing.assert_frame_equal(x, before, check_exact=True)
            if x.attrs != before.attrs:
                raise AssertionError("changed attrs")
        except (AssertionError, ValueError) as exc:
            raise InputValidationError("uplift.predict: model mutated its feature input") from exc
        if not isinstance(output, pd.DataFrame) or list(output.columns) != list(PREDICTION_COLUMNS):
            raise InputValidationError("uplift.predict: expected DataFrame columns p0, p1, uplift_score in order")
        if len(output) != len(mapping) or not output.index.is_unique:
            raise InputValidationError("uplift.predict: output row count or duplicate index mismatch")
        try:
            pd.testing.assert_index_equal(output.index, tokens, exact=True)
        except AssertionError as exc:
            raise InputValidationError("uplift.predict: output index must preserve every token and its order") from exc
        # _frame checks evaluation_only; preserve the verified index separately.
        values = _frame(output, PREDICTION_COLUMNS, PREDICTION_COLUMNS, "uplift.predict")
        values.index = tokens.copy(deep=True)
        for column in PREDICTION_COLUMNS:
            values[column] = _numeric(values[column], f"uplift.{column}")
            lower = -1 if column == "uplift_score" else 0
            if not values[column].between(lower, 1).all():
                raise InputValidationError(f"uplift.{column}: outside declared value range")
        if not np.isclose(values.uplift_score.to_numpy(dtype=float),
                          values.p1.to_numpy(dtype=float) - values.p0.to_numpy(dtype=float),
                          rtol=0, atol=DIFFERENCE_ATOL).all():
            raise InputValidationError("uplift.uplift_score: disagrees with p1 - p0")
        # Join by validated opaque keys, never attach IDs to unchecked row positions.
        prediction = mapping.join(values[["uplift_score"]], validate="one_to_one").reset_index(drop=True)
        prediction["input_cutoff_at"] = cutoff
        try:
            prediction["prediction_end_at"] = prediction.as_of_at + pd.Timedelta(days=28)
        except (ValueError, OverflowError) as exc:
            raise InputValidationError("uplift.prediction_end_at: 28-day horizon exceeds timestamp range") from exc
        declaration = PredictionMetadata(producer_metadata["model_version"],
                                         producer_metadata["target_definition_id"],
                                         28, "days", "probability_difference")
        batch = PredictionBatch(validate_prediction_frame(prediction, declaration, "uplift"), declaration)
    return UpliftPredictionResult(scored_features, batch, excluded.reset_index(drop=True),
                                  producer_metadata, len(source))
