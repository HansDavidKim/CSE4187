"""Load canonical records and verify their explicit outcome sidecars."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _utc(value):
    return pd.to_datetime(value, utc=True, errors="raise")


def _outcome_days(frame, metadata=None):
    """Read the outcome window separately from the fixed 30-day feature windows."""
    metadata = metadata or {}
    explicit = frame["outcome_window_days"].dropna().unique() if "outcome_window_days" in frame else []
    if len(explicit) > 1:
        raise ValueError("One analysis run must use one outcome window")
    if len(explicit):
        value = explicit[0]
    elif {"as_of_at", "outcome_end_at"} <= set(frame):
        durations = ((_utc(frame.outcome_end_at) - _utc(frame.as_of_at)).dt.total_seconds() / 86400).dropna().unique()
        if len(durations) != 1:
            raise ValueError("Outcome end times must define one common observation window")
        value = durations[0]
    else:
        value = metadata.get("outcome_window_days", 30)
    if isinstance(value, (bool, np.bool_)) or not float(value).is_integer() or value <= 0:
        raise ValueError("Outcome window must be a positive integer number of days")
    if "outcome_window_days" in metadata and int(metadata["outcome_window_days"]) != int(value):
        raise ValueError("Label and dataset outcome windows disagree")
    return int(value)


def _validate_sidecar(name, frame, tables, metadata):
    keys = ["customer_id", "as_of_at"]
    if not set(keys) <= set(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name} sidecar requires known, unique customer/snapshot keys")
    known = set(tables["customers"].customer_id)
    if not set(frame.customer_id) <= known or (name == "treatments" and set(frame.customer_id) != known):
        raise ValueError(f"{name} sidecar customers disagree with raw SQLite")
    if any(pd.Timestamp(value).tzinfo is None for value in frame.as_of_at):
        raise ValueError(f"{name} sidecar requires timezone-aware snapshot clocks")
    points = _utc(frame.as_of_at)
    if metadata.get("as_of_at") is not None and not points.eq(_utc(metadata["as_of_at"])).all():
        raise ValueError(f"{name} sidecar snapshot disagrees with raw metadata")
    for column in ("dataset_id", "seed", "raw_database_sha256", "definition_id", "outcome_window_days"):
        if column in frame and column in metadata and not frame[column].eq(metadata[column]).all():
            raise ValueError(f"{name} sidecar {column} disagrees with raw metadata")
    if name == "labels":
        days = _outcome_days(frame, metadata)
        if "outcome_end_at" in frame and not _utc(frame.outcome_end_at).eq(points + pd.Timedelta(days=days)).all():
            raise ValueError("labels sidecar outcome horizon disagrees with raw metadata")
        if "observation_end_at" in frame and "observation_end_at" in metadata:
            if not _utc(frame.observation_end_at).eq(_utc(metadata["observation_end_at"])).all():
                raise ValueError("labels sidecar observation end disagrees with raw metadata")


def check_generation_sidecars(database, metadata):
    """Verify every sidecar checksum declared by the generation manifest."""
    database = Path(database)
    generation_path = database.parent / "manifest.json"
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    if generation.get("dataset_id", metadata.get("dataset_id")) != metadata.get("dataset_id"):
        raise ValueError("generation manifest disagrees with raw SQLite dataset")
    bindings = generation.get("sidecar_sha256", {})
    if not isinstance(bindings, dict) or any(not isinstance(name, str) or not name.isidentifier() for name in bindings):
        raise ValueError("generation manifest has invalid sidecar names")
    for name, expected in bindings.items():
        sidecar = database.parent / f"{name}.parquet"
        if not sidecar.is_file() or hashlib.sha256(sidecar.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} sidecar checksum disagrees with generation manifest")


def load_tables(directory: str | Path = "data/processed") -> dict[str, Any]:
    """Load canonical raw SQLite plus explicit outcome sidecars, or legacy Parquet."""
    path = Path(directory)
    database = path if path.is_file() else path / "raw.sqlite"
    if database.is_file():
        from simulate.v4.simulator.raw_store import load_raw_dataset
        tables, metadata = load_raw_dataset(database)
        raw_manifest = json.loads(database.with_suffix(".manifest.json").read_text())
        tables["_metadata"] = {**metadata, "raw_database_sha256": raw_manifest["database_sha256"]}
        check_generation_sidecars(database, metadata)
        for name in ("labels", "treatments"):
            sidecar = database.parent / f"{name}.parquet"
            if sidecar.exists():
                frame = pd.read_parquet(sidecar)
                _validate_sidecar(name, frame, tables, tables["_metadata"])
                tables[name] = frame
        return tables
    names = ("customers", "events", "sessions", "orders", "treatments", "labels")
    tables = {name: pd.read_parquet(path / f"{name}.parquet") for name in names}
    manifest = path / "manifest.json"
    tables["_metadata"] = json.loads(manifest.read_text()) if manifest.exists() else {}
    tables["_format"] = "legacy_parquet"
    return tables
