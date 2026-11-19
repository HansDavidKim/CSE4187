"""Validate consumer inputs without changing the simulator or model APIs."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import numbers
import re

import numpy as np
import pandas as pd

from simulate.v4.features import (
    BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION,
)
from simulate.v4.storage import FEATURE_KEYS


JOIN_KEYS = ("dataset_id", "customer_id", "as_of_at")
CONTEXT_COLUMNS = (
    *JOIN_KEYS, "reference_week", "snapshot_id", "currency", "amount_unit",
    "raw_database_sha256",
)
PREDICTION_COLUMNS = {
    "churn": "churn_probability",
    "uplift": "uplift_score",
    "clv": "predicted_clv_12m",
}
VALUE_SEMANTICS = {
    "churn": "probability",
    "uplift": "probability_difference",
    "clv": "nonnegative_amount",
}


class InputValidationError(ValueError):
    """An input violates the internal strategy consumer contract."""


@dataclass(frozen=True)
class PredictionMetadata:
    """Internal adapter declaration, not an agreed external model interface.

    target_definition_id must name the actual outcome/policy definition. Horizons
    keep their declared unit; calendar months are never replaced by fixed days.
    """

    model_version: str
    target_definition_id: str
    horizon_value: int
    horizon_unit: str
    value_semantics: str


def _frame(frame, required, allowed, name):
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError(f"{name}: expected a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise InputValidationError(f"{name}: duplicate column names")
    if any(not isinstance(column, str) for column in frame.columns):
        raise InputValidationError(f"{name}: column names must be strings")
    missing, extra = set(required) - set(frame), set(frame) - set(allowed)
    if missing or extra:
        raise InputValidationError(
            f"{name}: missing columns={sorted(missing)}, unexpected columns={sorted(extra)}"
        )
    if frame.empty:
        raise InputValidationError(f"{name}: input cannot be empty")
    if frame.attrs.get("evaluation_only") is not None:
        raise InputValidationError(f"{name}: evaluation_only inputs are not accepted")
    result = frame.copy(deep=True).reset_index(drop=True)
    result.attrs = {}
    return result


def _strings(frame, columns, name):
    for column in columns:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise InputValidationError(f"{name}.{column}: requires nonempty strings")


def _times(values, name):
    points = []
    for value in values:
        try:
            point = pd.Timestamp(value)
            if pd.isna(point) or point.tzinfo is None:
                raise ValueError("missing timezone")
            points.append(point.tz_convert("UTC"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(f"{name}: requires known timezone-aware timestamps") from exc
    return pd.Series(points, index=values.index, dtype="datetime64[ns, UTC]")


def _units(values, name):
    result = []
    for value in values:
        try:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("boolean unit")
            unit = Decimal(str(value))
            if not unit.is_finite() or unit <= 0:
                raise ValueError("nonpositive unit")
            result.append(format(unit.normalize(), "f"))
        except (InvalidOperation, ValueError) as exc:
            raise InputValidationError(f"{name}: requires a positive finite amount unit") from exc
    return result


def _context(frame, name):
    _strings(frame, [c for c in CONTEXT_COLUMNS if c not in {"as_of_at", "reference_week", "amount_unit"}], name)
    if frame.dataset_id.nunique() != 1:
        raise InputValidationError(f"{name}: mixed dataset_id values")
    for column in ("raw_database_sha256",):
        if not frame[column].map(lambda value: re.fullmatch(r"[0-9a-f]{64}", value) is not None).all():
            raise InputValidationError(f"{name}.{column}: requires a lowercase SHA256 digest")
    for column in ("raw_database_sha256", "currency"):
        if frame[column].nunique() != 1:
            raise InputValidationError(f"{name}: mixed {column} values")
    frame["amount_unit"] = _units(frame.amount_unit, f"{name}.amount_unit")
    if frame.amount_unit.nunique() != 1:
        raise InputValidationError(f"{name}: mixed amount_unit values")
    frame["as_of_at"] = _times(frame.as_of_at, f"{name}.as_of_at")
    local = frame.as_of_at.dt.tz_convert("Asia/Seoul")
    expected = (local.dt.normalize() - pd.to_timedelta(local.dt.weekday, unit="D")).dt.date.astype(str)
    weeks = []
    for value in frame.reference_week:
        try:
            if isinstance(value, numbers.Number):
                raise ValueError("numeric week")
            week = pd.Timestamp(value)
            if week.tzinfo is not None:
                week = week.tz_convert("Asia/Seoul")
            if pd.isna(week) or week != week.normalize() or week.weekday() != 0:
                raise ValueError("not Monday midnight")
            weeks.append(week.date().isoformat())
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(f"{name}.reference_week: requires a Seoul Monday date/midnight") from exc
    frame["reference_week"] = weeks
    if not frame.reference_week.eq(expected).all():
        raise InputValidationError(f"{name}: reference_week disagrees with as_of_at in Asia/Seoul")
    if frame.duplicated(list(JOIN_KEYS)).any():
        raise InputValidationError(f"{name}: duplicate customer/snapshot keys")
    return frame


def _numeric(values, name, allow_null=False):
    if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise InputValidationError(f"{name}: boolean values are not numeric predictions/features")
    try:
        if (pd.api.types.is_datetime64_any_dtype(values.dtype)
                or pd.api.types.is_timedelta64_dtype(values.dtype)):
            raise ValueError("timestamps and durations are not numeric values")
        result = pd.to_numeric(values, errors="raise")
        if pd.api.types.is_complex_dtype(result.dtype):
            raise ValueError("complex values")
        finite = np.isfinite(result.dropna().to_numpy(dtype=float))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{name}: requires real numeric values") from exc
    if not finite.all() or (not allow_null and result.isna().any()):
        raise InputValidationError(f"{name}: non-finite or missing values")
    return result


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy/normalize the current KIT39 store; preserve legitimate feature NULLs."""
    columns = (*FEATURE_KEYS, *EXPANDED_FEATURES)
    result = _context(_frame(frame, columns, columns, "features"), "features")
    provenance = set(FEATURE_KEYS) - set(CONTEXT_COLUMNS)
    _strings(result, provenance, "features")
    versions = {
        "feature_set_version": "kit32-v1",
        "behavior_feature_set_version": BEHAVIOR_FEATURE_VERSION,
        "expanded_feature_set_version": EXPANDED_FEATURE_VERSION,
    }
    for column, expected in versions.items():
        if not result[column].eq(expected).all():
            raise InputValidationError(f"features.{column}: unsupported version, expected {expected}")
    if not result.source_model_sha256.map(lambda value: re.fullmatch(r"[0-9a-f]{64}", value) is not None).all():
        raise InputValidationError("features.source_model_sha256: requires a lowercase SHA256 digest")
    if result.source_model_sha256.nunique() != 1:
        raise InputValidationError("features: mixed source_model_sha256 values")
    for column in EXPANDED_FEATURES:
        result[column] = _numeric(result[column], f"features.{column}", allow_null=True)
    return result.loc[:, list(columns)]


def validate_prediction_frame(frame: pd.DataFrame, metadata: PredictionMetadata, model: str) -> pd.DataFrame:
    """Validate an internal model batch, including declared cutoff and horizon.

    input_cutoff_at is an exclusive historical-input boundary, not proof that a
    producer trained without leakage. prediction_end_at verifies the declared
    horizon for every snapshot. Adapters must obtain both from the producer.
    """
    if model not in PREDICTION_COLUMNS:
        raise InputValidationError(f"unknown model: {model}")
    if not isinstance(metadata, PredictionMetadata):
        raise InputValidationError(f"{model}: requires PredictionMetadata")
    for field in ("model_version", "target_definition_id"):
        value = getattr(metadata, field)
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(f"{model}.{field}: requires a nonempty declaration")
    if (isinstance(metadata.horizon_value, (bool, np.bool_))
            or not isinstance(metadata.horizon_value, numbers.Integral) or metadata.horizon_value <= 0):
        raise InputValidationError(f"{model}: horizon_value must be a positive integer")
    if metadata.horizon_unit not in ("days", "weeks", "calendar_months"):
        raise InputValidationError(f"{model}: horizon_unit must be days, weeks or calendar_months")
    semantics = ("amount", "nonnegative_amount") if model == "clv" else (VALUE_SEMANTICS[model],)
    if metadata.value_semantics not in semantics:
        raise InputValidationError(f"{model}: expected value_semantics in {semantics}")
    if model == "clv" and (metadata.horizon_value != 52 or metadata.horizon_unit != "weeks"):
        raise InputValidationError("clv: predicted_clv_12m requires exactly 52 weeks")
    value_column = PREDICTION_COLUMNS[model]
    columns = (*CONTEXT_COLUMNS, "input_cutoff_at", "prediction_end_at", value_column)
    result = _context(_frame(frame, columns, columns, model), model)
    for column in ("input_cutoff_at", "prediction_end_at"):
        result[column] = _times(result[column], f"{model}.{column}")
    if result.input_cutoff_at.gt(result.as_of_at).any():
        raise InputValidationError(f"{model}: input_cutoff_at is after as_of_at (future inputs)")
    horizon = int(metadata.horizon_value)
    try:
        expected_end = result.as_of_at.map(
            lambda point: point + pd.DateOffset(months=horizon)
            if metadata.horizon_unit == "calendar_months"
            else point + pd.Timedelta(**{metadata.horizon_unit: horizon})
        )
    except (ValueError, OverflowError) as exc:
        raise InputValidationError(f"{model}: declared horizon exceeds supported timestamp range") from exc
    if not result.prediction_end_at.eq(expected_end).all():
        raise InputValidationError(f"{model}: prediction_end_at disagrees with declared horizon")
    values = _numeric(result[value_column], f"{model}.{value_column}")
    lower, upper = {"churn": (0, 1), "uplift": (-1, 1), "clv": (None, None)}[model]
    if metadata.value_semantics == "nonnegative_amount":
        lower = 0
    if ((lower is not None and values.lt(lower).any())
            or (upper is not None and values.gt(upper).any())):
        raise InputValidationError(f"{model}.{value_column}: outside declared value range")
    result[value_column] = values
    return result.loc[:, list(columns)]
