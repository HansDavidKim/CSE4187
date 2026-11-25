"""Read the maintained simulator's weekly Feature Store without rebuilding predictors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, WEEKLY_TIME_SPEC, is_weekly
from . import schema as S

KEYS = ["dataset_id", "customer_id", "as_of_at"]
SNAPSHOT = ["customer_id", "as_of_at", "snapshot_id"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


def manifest(path):
    file = directory(path) / "manifest.json"
    return json.loads(file.read_text()) if file.is_file() else {}


def read_table(path, metadata, name):
    file = directory(path) / name
    expected = metadata.get("file_sha256", {}).get(name)
    if expected is None or sha256(file) != expected:
        raise ValueError(f"Missing or changed Feature Store checksum: {name}")
    frame = pd.read_parquet(file)
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in {name}")
    return frame


def load_features(path, *, ledger=None):
    """Return all decision customers, provenance and KIT39; do not read future labels."""
    meta = manifest(path)
    if (not is_weekly(meta) or meta.get("time_windows_days") != WEEKLY_TIME_SPEC
            or meta.get("predictors") != S.KIT39):
        raise ValueError("Uplift requires the canonical KIT39 and explicit weekly 4/8/13-week windows")
    base = directory(path)
    source = {}
    explicit_ledger = None
    if ledger is not None:
        candidate = Path(ledger)
        explicit_ledger = candidate / "raw.sqlite" if candidate.is_dir() else candidate
        if not explicit_ledger.is_file():
            raise ValueError("Explicit Uplift source ledger does not exist")
        sources = (explicit_ledger.with_suffix(".manifest.json"), explicit_ledger.parent / "manifest.json")
    else:
        sources = (base.parent / "manifest.json", base.parent / "raw.manifest.json",
                   base.parent / "dataset" / "raw.manifest.json")
    declared_sha = None
    for file in sources:
        if file.is_file():
            document = json.loads(file.read_text())
            source = document.get("metadata", document)
            declared_sha = document.get("database_sha256")
            break
    if explicit_ledger is not None and not source:
        raise ValueError("Explicit Uplift ledger requires its source manifest")
    features = read_table(path, meta, "feature_store.parquet")
    if "dataset_id" not in features:
        features["dataset_id"] = source.get("dataset_id", meta.get("dataset_id"))
    required = [*KEYS, "snapshot_id", *S.KIT39]
    if not set(required) <= set(features) or features[KEYS + ["snapshot_id"]].isna().any().any():
        raise ValueError("Uplift requires complete customer/snapshot identity and KIT39")
    if (features.empty or features.duplicated(KEYS).any() or features.customer_id.duplicated().any()
            or features.dataset_id.nunique() != 1 or features.as_of_at.nunique() != 1):
        raise ValueError("Expected one nonempty dataset/cutoff and unique customer snapshots")
    if source.get("dataset_id") is not None and not features.dataset_id.eq(source["dataset_id"]).all():
        raise ValueError("Feature Store and source manifest dataset identity differ")
    if not features.as_of_at.map(lambda x: pd.Timestamp(x).tzinfo is not None).all():
        raise ValueError("Uplift cutoff must be timezone-aware")
    features["as_of_at"] = pd.to_datetime(features.as_of_at, utc=True)
    local = features.as_of_at.dt.tz_convert("Asia/Seoul")
    if not (local.dt.weekday.eq(0) & local.eq(local.dt.normalize())).all():
        raise ValueError("Uplift cutoff must be Monday midnight Asia/Seoul")
    ledger = explicit_ledger or next((q for q in (base.parent / "raw.sqlite", base.parent / "dataset" / "raw.sqlite")
                                      if q.is_file()), None)
    recorded = meta.get("raw_database_sha256", source.get("raw_database_sha256"))
    if ledger is not None:
        current = sha256(ledger)
        if ((recorded is not None and current != recorded)
                or (declared_sha is not None and current != declared_sha)):
            raise ValueError("Feature Store and source ledger checksum differ")
        recorded = current
    if recorded is None and "raw_database_sha256" in features:
        if features.raw_database_sha256.nunique() == 1:
            recorded = features.raw_database_sha256.iloc[0]
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("Uplift Feature Store requires a recorded source ledger SHA256")
    if "raw_database_sha256" in features and not features.raw_database_sha256.eq(recorded).all():
        raise ValueError("Feature rows and source ledger checksum differ")
    features["raw_database_sha256"] = recorded
    for name in ("currency", "amount_unit"):
        if name not in features and name in source:
            features[name] = source[name]
    if "expanded_feature_set_version" in features and not features.expanded_feature_set_version.eq(WEEKLY_FEATURE_VERSION).all():
        raise ValueError("Feature rows disagree with the weekly feature version")
    features.attrs = {"input_contract": {"time_spec": meta["time_spec"],
        "feature_version": WEEKLY_FEATURE_VERSION, "time_windows_days": meta["time_windows_days"]},
        "feature_manifest": meta, "source_metadata": source,
        "raw_database_sha256_verified": ledger is not None,
        "feature_directory": str(base.resolve())}
    return features


def assignment_probability(assignments, source):
    """Fitting requires a declared randomized design, both arms and constant overlap."""
    if source.get("randomized_assignment") is not True:
        raise ValueError("Uplift fitting requires an explicitly randomized assignment design")
    if set(assignments.treatment) != {0, 1}:
        raise ValueError("Uplift fitting requires both treatment and control customers")
    p = pd.to_numeric(assignments.assignment_probability, errors="raise")
    if p.isna().any() or not np.isfinite(p).all() or p.nunique() != 1 or not p.between(0, 1, inclusive="neither").all():
        raise ValueError("Uplift requires one known randomized assignment probability in (0,1)")
    return float(p.iloc[0])


def load_dataset(path, *, for_training=True, ledger=None):
    features = load_features(path, ledger=ledger)
    attrs = dict(features.attrs)
    meta, source = attrs["feature_manifest"], attrs["source_metadata"]
    labels = read_table(path, meta, "labels.parquet")
    assignments = read_table(path, meta, "treatments.parquet")
    if for_training:
        propensity = assignment_probability(assignments, source)
    else:
        p = pd.to_numeric(assignments.assignment_probability, errors="raise")
        if p.isna().any() or not p.between(0, 1).all():
            raise ValueError("Invalid assignment probabilities")
        propensity = float(p.iloc[0]) if p.nunique() == 1 else None
    required = {*SNAPSHOT, "outcome_end_at", "observation_end_at", "outcome_window_days",
                "eligibility_window_days", "is_censored"}
    if not required <= set(labels) or labels.duplicated(SNAPSHOT).any():
        raise ValueError("Uplift labels require unique snapshots and explicit outcome availability")
    for name in ("as_of_at", "outcome_end_at", "observation_end_at"):
        labels[name] = pd.to_datetime(labels[name], utc=True)
    if (not labels.outcome_window_days.eq(28).all() or not labels.eligibility_window_days.eq(28).all()
            or not labels.outcome_end_at.eq(labels.as_of_at + pd.Timedelta(days=28)).all()
            or not labels.is_censored.isin([False, True]).all()):
        raise ValueError("Uplift requires complete declared 4-week eligibility/outcome windows")
    assignments = assignments.copy()
    assignments["as_of_at"] = pd.to_datetime(assignments.as_of_at, utc=True)
    if (assignments.customer_id.duplicated().any() or not assignments.treatment.isin([0, 1]).all()
            or not pd.to_datetime(assignments.assigned_at, utc=True).eq(assignments.as_of_at).all()):
        raise ValueError("Expected one binary assignment at the customer cutoff")
    if "snapshot_id" not in assignments:
        assignments = assignments.merge(features[SNAPSHOT], on=["customer_id", "as_of_at"],
                                         how="left", validate="one_to_one")
    rows = features.merge(labels, on=SNAPSHOT, validate="one_to_one", how="left", indicator=True,
                          suffixes=("", "_label"))
    if not rows._merge.eq("both").all() or len(rows) != len(labels):
        raise ValueError("Uplift labels must cover the exact Feature Store cohort")
    rows = rows.drop(columns="_merge").merge(assignments[[*SNAPSHOT, "treatment"]], on=SNAPSHOT,
        validate="one_to_one", how="left", indicator=True)
    if not rows._merge.eq("both").all() or len(rows) != len(assignments):
        raise ValueError("Uplift assignments must cover the exact Feature Store cohort")
    mature = ~rows.is_censored.astype(bool) & rows.outcome_end_at.le(rows.observation_end_at)
    if "purchased_28d" in rows:
        target = pd.to_numeric(rows.purchased_28d, errors="raise")
        eligible = (rows.visit_recency_days.ge(0) & rows.visit_recency_days.le(28)
                    & rows.observed_tenure_days.ge(WEEKLY_TIME_SPEC["min_history_days"]))
    else:
        experiment = read_table(path, meta, "experiment_log.parquet")
        experiment["as_of_at"] = pd.to_datetime(experiment.as_of_at, utc=True)
        evidence = rows[SNAPSHOT].merge(experiment[[*SNAPSHOT, "eligible_at_assignment", "factual_no_order"]],
                                      on=SNAPSHOT, how="left", validate="one_to_one")
        if not evidence.eligible_at_assignment.isin([0, 1]).all():
            raise ValueError("Missing factual eligibility evidence")
        eligible = evidence.eligible_at_assignment.eq(1)
        target = 1 - pd.to_numeric(evidence.factual_no_order, errors="raise")
    if not target.loc[mature & eligible].isin([0, 1]).all():
        raise ValueError("Mature purchase outcomes must be binary; unavailable outcomes stay missing")
    rows[S.LABEL] = target.where(mature)
    rows[S.T0] = rows.as_of_at
    rows = rows.loc[eligible, [*features.columns, S.T0, S.TREATMENT, S.LABEL]].copy()
    rows.attrs = {**attrs, "dataset_id": features.dataset_id.iloc[0], "assignment_probability": propensity}
    return rows
