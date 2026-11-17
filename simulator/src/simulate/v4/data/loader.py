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


def _read_materialized_parquet(path, manifest, name, columns, *, filters=None):
    file = path / name
    checksum = manifest.get("file_sha256", {}).get(name)
    if checksum is None or not file.is_file():
        raise ValueError(f"Feature input {name} is missing from the materialized output")
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Feature input {name} differs from its manifest")
        stream.seek(0)
        return pd.read_parquet(stream, columns=columns, filters=filters)


def load_retention_inputs(
    feature_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read snapshot keys, the ordered KIT39 matrix, and mature 28-day no-visit labels."""
    from simulate.v4.features import EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION

    path = Path(feature_directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("feature_set_version")
    if (version not in ("kit39-review-cart-v2", EXPANDED_FEATURE_VERSION)
            or manifest.get("predictors") != list(EXPANDED_FEATURES)):
        raise ValueError("Retention requires the registered KIT39 predictors in canonical order")
    keys = ["customer_id", "as_of_at", "snapshot_id"]
    identity = ["dataset_id", "raw_database_sha256"]
    units = ["currency", "amount_unit"]
    label_fields = ["definition_id", "outcome_window_days", "eligibility_window_days",
                    "outcome_end_at", "observation_end_at", "is_censored", "policy_id", "churn"]
    feature_fields = [*keys, *identity, *units, "expanded_feature_set_version", *EXPANDED_FEATURES]
    frames = []
    for name, columns in (("feature_store.parquet", feature_fields), ("labels.parquet", [*keys, *label_fields])):
        frame = _read_materialized_parquet(path, manifest, name, columns)
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"Retention input {name} requires unique, known snapshot keys")
        if not isinstance(frame.as_of_at.dtype, pd.DatetimeTZDtype):
            raise ValueError("Retention snapshots require a timezone-aware as_of_at")
        frames.append(frame)
    features, labels = frames
    for name in identity:
        declared = manifest.get(name)
        if not isinstance(declared, str) or not declared or not features[name].eq(declared).fillna(False).all():
            raise ValueError(f"Retention feature {name} differs from its manifest")
    if not features.expanded_feature_set_version.eq(version).fillna(False).all():
        raise ValueError("Retention feature rows require the registered KIT39 version")
    if features[units].isna().any().any() or len(features[units].drop_duplicates()) != 1:
        raise ValueError("Retention feature rows require a common currency and amount unit")
    rows = features.merge(labels, on=keys, how="left", validate="one_to_one", indicator=True)
    if rows.empty or not rows["_merge"].eq("both").all():
        raise ValueError("Every Retention feature row requires a label from the same snapshot")
    definition = "future_no_session_28d_active28d"
    if (not rows.definition_id.eq(definition).fillna(False).all()
            or not rows.outcome_window_days.eq(28).fillna(False).all()
            or not rows.eligibility_window_days.eq(28).fillna(False).all()
            or not rows.outcome_end_at.eq(rows.as_of_at + pd.Timedelta(days=28)).all()):
        raise ValueError("Retention requires a 28-day no-visit outcome and 28-day eligibility")
    if (not pd.api.types.is_bool_dtype(rows.is_censored) or rows.is_censored.isna().any()
            or not isinstance(rows.observation_end_at.dtype, pd.DatetimeTZDtype)
            or rows.observation_end_at.isna().any()):
        raise ValueError("Retention labels require an observed cutoff and an explicit censoring flag")
    mature = rows.loc[~rows.is_censored & rows.outcome_end_at.le(rows.observation_end_at)].reset_index(drop=True)
    if mature.empty or not mature.churn.isin([0, 1]).all():
        raise ValueError("Retention training requires mature binary no-visit labels")
    snapshots = mature[[*keys, *identity, *units, "expanded_feature_set_version",
                        "outcome_end_at", "definition_id", "policy_id"]]
    return snapshots, mature[list(EXPANDED_FEATURES)], mature.churn.astype("int8")


def load_uplift_inputs(feature_directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the customer, weekly and cart inputs for the four-week purchase learner.

    Monetary values keep the materialized minor units. Currency, amount unit,
    dataset identity and experiment seed travel in each frame's attrs.
    """
    path = Path(feature_directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    provenance_columns = ["dataset_id", "currency", "amount_unit"]
    keys = ["customer_id", "as_of_at", "snapshot_id", *provenance_columns]
    assignments = _read_materialized_parquet(
        path, manifest, "experiment_log.parquet",
        [*keys, "seed", "assigned_at", "treatment", "outcome_end_at", "outcome_observed"],
    )
    if assignments.empty or assignments.customer_id.duplicated().any():
        raise ValueError("Uplift input requires one assignment per customer")
    w0 = pd.Timestamp(assignments.as_of_at.iloc[0])
    if pd.isna(w0) or w0.tzinfo is None:
        raise ValueError("Uplift w0 must be a timezone-aware Monday midnight in Asia/Seoul")
    w0 = w0.tz_convert("Asia/Seoul")
    if w0.weekday() != 0 or w0 != w0.normalize():
        raise ValueError("Uplift w0 must be a timezone-aware Monday midnight in Asia/Seoul")
    start, end = w0 - pd.Timedelta(weeks=12), w0 + pd.Timedelta(weeks=4)
    if not assignments.assigned_at.eq(w0).all() or not assignments.outcome_end_at.eq(end).all():
        raise ValueError("Uplift requires assignment at w0 and a four-week outcome")
    weekly_columns = ["sessions", "purchases", "net_revenue_minor", "views", "session_seconds"]
    weekly = _read_materialized_parquet(
        path, manifest, "weekly_panel.parquet", [*keys, "seed", "reference_week", *weekly_columns],
        filters=[("reference_week", ">=", start), ("reference_week", "<", end)],
    )
    cart_columns = {f"{name}_current": name for name in ("cart_qty", "cart_sku", "cart_gross_value")}
    cart = _read_materialized_parquet(path, manifest, "feature_store.parquet", [*keys, *cart_columns])
    provenance = assignments.iloc[0][provenance_columns].to_dict()
    by_customer = assignments.set_index("customer_id")
    snapshots = by_customer.snapshot_id
    for frame in (assignments, weekly, cart):
        if (frame[keys].isna().any().any() or not frame.as_of_at.eq(w0).all()
                or set(frame.customer_id) != set(snapshots.index)
                or not frame.customer_id.map(snapshots).eq(frame.snapshot_id).all()
                or any(not frame[name].eq(value).all() for name, value in provenance.items())):
            raise ValueError("Uplift inputs must share one dataset and the assignment snapshots")
    if (cart.customer_id.duplicated().any() or weekly.duplicated(["customer_id", "reference_week"]).any()
            or not weekly.reference_week.isin(pd.date_range(start, end, freq="7D", inclusive="left")).all()):
        raise ValueError("Uplift inputs require unique customers and unique Monday weeks")
    if (not assignments.treatment.isin([0, 1]).all()
            or not pd.api.types.is_bool_dtype(assignments.outcome_observed)
            or assignments.outcome_observed.isna().any()):
        raise ValueError("Uplift assignments require binary treatment and an explicit outcome observation flag")
    if not pd.api.types.is_integer_dtype(assignments.seed) or assignments.seed.isna().any():
        raise ValueError("Uplift inputs require one recorded integer experiment seed")
    seed = int(assignments.seed.iloc[0])
    if not assignments.seed.eq(seed).all() or not weekly.seed.eq(seed).fillna(False).all():
        raise ValueError("Uplift inputs must share the assignment experiment seed")
    outcome_observed = weekly.customer_id.map(by_customer.outcome_observed)
    # pandas nullable integers produce pd.NA in NumPy comparisons in build_dataset.
    weekly[["sessions", "purchases"]] = weekly[["sessions", "purchases"]].astype(float)
    weekly.loc[weekly.reference_week.ge(w0) & ~outcome_observed.astype(bool), "purchases"] = np.nan
    weekly["session_seconds"] = weekly.session_seconds.astype(float) / 60
    customers = assignments[["customer_id", "treatment"]].assign(w0=w0)
    weekly = weekly[["customer_id", "reference_week", *weekly_columns]].rename(
        columns={"net_revenue_minor": "net_revenue", "session_seconds": "session_minutes"},
    )
    cart = cart[["customer_id", *cart_columns]].rename(columns=cart_columns)
    for frame in (customers, weekly, cart):
        frame.attrs = {**provenance, "seed": seed}
    return customers, weekly, cart
