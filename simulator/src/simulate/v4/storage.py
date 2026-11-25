"""Materialize public KIT39 features and factual experiment tables beside raw SQLite."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd

from .data.loader import load_tables
from .data.weekly import build_experiment_log, build_weekly_panel
from .features import (
    BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION,
    KIT_FEATURES, build_features,
)
from .time_spec import WEEKLY_FEATURE_VERSION, WEEKLY_TIME_SPEC, is_weekly, weekly_metadata


FEATURE_KEYS = (
    "customer_id", "as_of_at", "snapshot_id", "reference_week", "dataset_id",
    "currency", "amount_unit", "raw_database_sha256", "source_model_sha256",
    "feature_set_version", "behavior_feature_set_version", "expanded_feature_set_version",
)


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_frame(frame):
    result = frame.copy()
    for name in result:
        if isinstance(result[name].dtype, pd.DatetimeTZDtype):
            result[name] = result[name].dt.tz_convert("UTC").map(lambda value: None if pd.isna(value) else value.isoformat())
        elif pd.api.types.is_datetime64_dtype(result[name]):
            raise ValueError(f"SQLite export requires timezone-aware timestamps: {name}")
    return result


def materialize_features(dataset, output, as_of_at=None):
    """Atomically write features.sqlite, Parquet tables and an inspectable manifest.

    The SQL features table contains identifiers/provenance and exactly 39 allowed
    predictors. Labels, assignments and factual experiment results get separate
    tables. No simulator diagnostics or counterfactual response types are loaded.
    Without explicit assignment/label sidecars, their dependent exports are absent.
    """
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"feature output must be new or empty: {output}")
    tables = load_tables(dataset)
    metadata = tables.get("_metadata", {})
    feature_version = WEEKLY_FEATURE_VERSION if is_weekly(metadata) else EXPANDED_FEATURE_VERSION
    if as_of_at is None and "labels" not in tables:
        as_of_at = metadata.get("as_of_at")
    features = build_features(tables, as_of_at=as_of_at)
    features = features[[*FEATURE_KEYS, *EXPANDED_FEATURES]]
    if features.duplicated(["customer_id", "as_of_at"]).any():
        raise ValueError("feature snapshots must be unique per customer/time")
    if len(EXPANDED_FEATURES) != 39 or len(set(EXPANDED_FEATURES)) != 39:
        raise ValueError("the v4 contract requires exactly 39 unique predictors")
    frames = {"features": features}
    omitted = {}
    for name in ("labels", "treatments"):
        if name in tables:
            frames[name] = tables[name]
    if {"labels", "treatments"} <= set(tables):
        frames["experiment_log"] = build_experiment_log(tables, metadata)
        frames["weekly_panel"] = build_weekly_panel(tables, metadata)
        if is_weekly(metadata):
            t0 = pd.Timestamp(metadata["as_of_at"])
            panel = frames["weekly_panel"]
            columns = ["customer_id", "as_of_at", "snapshot_id", "reference_week", "dataset_id", "currency", "amount_unit",
                       "sessions", "purchases", "net_revenue_minor", "views", "session_seconds"]
            frames["clv_weekly_inputs"] = panel.loc[
                panel.reference_week.ge(t0 - pd.Timedelta(days=WEEKLY_TIME_SPEC["clv_input_days"]))
                & panel.reference_week.lt(t0), columns].copy()
            from .session_observations import session_availability
            sessions = session_availability(tables)
            unknown = sessions.loc[sessions.started_at.lt(t0) & ~sessions.completion_available_at.lt(t0)]
            unknown_weeks = []
            for row in unknown.itertuples(index=False):
                local = row.started_at.tz_convert("Asia/Seoul")
                first = local.normalize() - pd.Timedelta(days=local.weekday())
                first = max(first, t0 - pd.Timedelta(days=WEEKLY_TIME_SPEC["clv_input_days"])).tz_convert(t0.tz)
                unknown_weeks.extend((row.customer_id, week) for week in pd.date_range(first, t0, freq="7D", inclusive="left"))
            inputs = frames["clv_weekly_inputs"]
            if unknown_weeks:
                inputs.loc[pd.MultiIndex.from_frame(inputs[["customer_id", "reference_week"]]).isin(unknown_weeks), "session_seconds"] = float("nan")
            frames["clv_labels"] = tables["labels"][["customer_id", "as_of_at", "snapshot_id", "future_revenue_12m", "clv_outcome_end_at", "observation_end_at"]].copy()
    else:
        reason = "requires explicit labels and fixed assignment sidecars; raw events do not identify policy decisions"
        omitted = dict.fromkeys(("experiment_log", "weekly_panel"), reason)
    registry = pd.DataFrame({
        "ordinal": range(1, 40), "feature_name": EXPANDED_FEATURES,
        "feature_family": ["KIT32" if name in KIT_FEATURES else "review_cart7" for name in EXPANDED_FEATURES],
        "feature_set_version": feature_version,
    })
    manifest = {
        "storage_schema_version": 1, "feature_set_version": feature_version,
        "behavior_feature_set_version": BEHAVIOR_FEATURE_VERSION,
        "predictors": list(EXPANDED_FEATURES), "keys_and_provenance": list(FEATURE_KEYS),
        "dataset_id": metadata.get("dataset_id"), "raw_database_sha256": metadata.get("raw_database_sha256"),
        "source_model_sha256": metadata.get("model_sha256"),
        "snapshot_times_utc": sorted(features.as_of_at.map(lambda value: value.isoformat()).unique().tolist()),
        "week_timezone": "Asia/Seoul", "sqlite_timestamp_timezone": "UTC",
        "rows": {name: len(frame) for name, frame in frames.items()}, "omitted": omitted,
        "predictor_exclusions": "labels, treatments, future outcomes, latent traits, lifecycle diagnostics and response types",
    }
    if is_weekly(metadata):
        manifest.update(weekly_metadata(), clv_target_alias="future_revenue_12m is exact52weeks/364days; controls only")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        with closing(sqlite3.connect(staging / "features.sqlite")) as connection, connection:
            for name, frame in frames.items():
                _sql_frame(frame).to_sql(name, connection, index=False, if_exists="fail")
                frame.to_parquet(staging / ("feature_store.parquet" if name == "features" else f"{name}.parquet"), index=False)
            registry.to_sql("feature_registry", connection, index=False, if_exists="fail")
            connection.execute("CREATE UNIQUE INDEX feature_snapshot ON features(customer_id, as_of_at)")
            connection.execute("CREATE UNIQUE INDEX feature_ordinal ON feature_registry(ordinal)")
            pd.DataFrame({"key": list(manifest), "value_json": [json.dumps(value, ensure_ascii=False, allow_nan=False) for value in manifest.values()]}).to_sql(
                "metadata", connection, index=False, if_exists="fail")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("feature SQLite integrity check failed")
        manifest["file_sha256"] = {path.name: _checksum(path) for path in sorted(staging.iterdir())}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
