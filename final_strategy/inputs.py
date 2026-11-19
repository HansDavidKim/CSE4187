"""Read an existing KIT39 Feature Store and integrate declared model batches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from simulate.v4.features import (
    BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION,
)
from simulate.v4.storage import FEATURE_KEYS

from .validation import (
    CONTEXT_COLUMNS, JOIN_KEYS, PREDICTION_COLUMNS, InputValidationError,
    PredictionMetadata, validate_feature_frame, validate_prediction_frame,
)


@dataclass(frozen=True)
class PredictionBatch:
    """A normalized internal DataFrame and its producer-declared semantics."""

    frame: pd.DataFrame
    metadata: PredictionMetadata


@dataclass(frozen=True)
class IntegratedInputs:
    """Safe input columns plus per-model context kept outside the predictor table."""

    data: pd.DataFrame
    prediction_metadata: dict[str, PredictionMetadata]
    prediction_context: dict[str, pd.DataFrame]


def load_feature_store(path: str | Path) -> pd.DataFrame:
    """Read feature_store.parquet only, verifying its adjacent simulator manifest.

    This consumer never regenerates raw data, features or experiment outputs.
    Other files listed in the manifest are not loaded or required here.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "feature_store.parquet"
    if path.name != "feature_store.parquet":
        raise InputValidationError("expected feature_store.parquet or its directory")
    if any(part.lower() == "evaluation_only" for part in path.resolve().parts):
        raise InputValidationError("evaluation_only paths are not strategy inputs")
    manifest_path = path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputValidationError("cannot read Feature Store manifest.json") from exc
    if not isinstance(manifest, dict):
        raise InputValidationError("Feature Store manifest must be an object")
    expected = {
        "storage_schema_version": 1,
        "feature_set_version": EXPANDED_FEATURE_VERSION,
        "behavior_feature_set_version": BEHAVIOR_FEATURE_VERSION,
        "predictors": list(EXPANDED_FEATURES),
        "keys_and_provenance": list(FEATURE_KEYS),
        "week_timezone": "Asia/Seoul",
        "sqlite_timestamp_timezone": "UTC",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise InputValidationError(f"Feature Store manifest: unsupported {key}")
    bindings = manifest.get("file_sha256")
    if not isinstance(bindings, dict) or not isinstance(bindings.get(path.name), str):
        raise InputValidationError("Feature Store manifest: missing Parquet checksum")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != bindings[path.name]:
        raise InputValidationError("Feature Store Parquet checksum mismatch")
    frame = validate_feature_frame(pd.read_parquet(path))
    for column in ("dataset_id", "raw_database_sha256", "source_model_sha256"):
        if not frame[column].eq(manifest.get(column)).all():
            raise InputValidationError(f"Feature Store manifest: {column} mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, dict) or rows.get("features") != len(frame):
        raise InputValidationError("Feature Store manifest: feature row count mismatch")
    times = sorted(frame.as_of_at.map(lambda value: value.isoformat()).unique().tolist())
    if manifest.get("snapshot_times_utc") != times:
        raise InputValidationError("Feature Store manifest: snapshot_times_utc mismatch")
    return frame


def integrate_inputs(features: pd.DataFrame, predictions: Mapping[str, PredictionBatch]) -> IntegratedInputs:
    """Join complete churn/uplift/clv coverage by dataset/customer/exact UTC time.

    Unknown columns fail closed. No customers are discarded, no NULL features
    are filled, and no segmentation, policy or score is calculated.
    """
    if not isinstance(predictions, Mapping) or set(predictions) != set(PREDICTION_COLUMNS):
        raise InputValidationError("supply exactly churn, uplift and clv prediction batches")
    result = validate_feature_frame(features)
    feature_index = pd.MultiIndex.from_frame(result[list(JOIN_KEYS)])
    feature_context = result.set_index(list(JOIN_KEYS))
    metadata, contexts = {}, {}
    for model, value_column in PREDICTION_COLUMNS.items():
        batch = predictions[model]
        if not isinstance(batch, PredictionBatch):
            raise InputValidationError(f"{model}: expected PredictionBatch")
        frame = validate_prediction_frame(batch.frame, batch.metadata, model)
        prediction_index = pd.MultiIndex.from_frame(frame[list(JOIN_KEYS)])
        missing = feature_index.difference(prediction_index)
        extra = prediction_index.difference(feature_index)
        if len(missing) or len(extra):
            raise InputValidationError(
                f"{model}: customer/snapshot coverage mismatch; missing={len(missing)}, unexpected={len(extra)}"
            )
        aligned = frame.set_index(list(JOIN_KEYS)).reindex(feature_index)
        for column in CONTEXT_COLUMNS:
            if column not in JOIN_KEYS and not aligned[column].eq(feature_context[column]).all():
                raise InputValidationError(f"{model}: {column} disagrees with Feature Store")
        result[value_column] = aligned[value_column].to_numpy()
        metadata[model] = batch.metadata
        contexts[model] = frame[[*JOIN_KEYS, "input_cutoff_at", "prediction_end_at"]].copy()
    return IntegratedInputs(result, metadata, contexts)
