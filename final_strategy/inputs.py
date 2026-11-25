"""Join existing retention/CLV handoffs and externally produced purchase probabilities."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd
from clv.bundle import file_sha256
from simulate.v4.time_spec import WEEKLY_TIME_SPEC, is_weekly

from .validation import KEYS, PROVENANCE, PredictionMetadata, keys, numeric, require, utc


@dataclass
class PredictionBatch:
    frame: pd.DataFrame
    metadata: PredictionMetadata
    producer_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.metadata, dict):
            self.metadata = PredictionMetadata(**self.metadata)


def load_feature_store(path):
    """Read the existing feature artifact without adding treatment or future outcomes."""
    path = Path(path)
    path = path / "feature_store.parquet" if path.is_dir() else path
    manifest = json.loads((path.parent / "manifest.json").read_text())
    require(is_weekly(manifest) and manifest.get("time_windows_days") == WEEKLY_TIME_SPEC,
            "Feature Store requires the shared weekly104w 4/8/13-week time contract")
    expected = manifest.get("file_sha256", {}).get(path.name)
    require(expected is not None and file_sha256(path) == expected,
            "Feature Store checksum is absent or changed")
    frame = pd.read_parquet(path)
    parent = path.parent.parent / "manifest.json"
    metadata = json.loads(parent.read_text()) if parent.is_file() else manifest
    if "dataset_id" not in frame and metadata.get("dataset_id"):
        frame["dataset_id"] = metadata["dataset_id"]
    if "raw_database_sha256" not in frame:
        checksum = manifest.get("raw_database_sha256") or metadata.get("raw_database_sha256")
        ledger = path.parent.parent / "raw.sqlite"
        if checksum is None and ledger.is_file():
            checksum = file_sha256(ledger)
        require(checksum is not None, "Feature Store lacks its raw ledger identity")
        frame["raw_database_sha256"] = checksum
    result = keys(frame)
    ledger = next((candidate for candidate in (path.parent.parent/"raw.sqlite", path.parent.parent/"dataset/raw.sqlite")
                   if candidate.is_file()), None)
    verified = False
    if ledger is not None:
        current = file_sha256(ledger)
        require(result.raw_database_sha256.eq(current).all(), "Feature Store and current raw ledger checksum differ")
        verified = True
    result.attrs = {"feature_manifest": manifest, "dataset_metadata": metadata, "current_ledger_verified": verified}
    return result


def _alias(frame, names, output, *, probability=False):
    found = [name for name in names if name in frame]
    require(bool(found), f"Prediction batch lacks {output}: expected one of {names}")
    result = numeric(frame[found[0]], output, probability=probability)
    for name in found[1:]:
        other = numeric(frame[name], name, probability=probability)
        require(np.allclose(result, other, equal_nan=True, rtol=0, atol=1e-12), f"Conflicting aliases for {output}")
    return result


def _reference_week(values, cutoff):
    def local(value):
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("Asia/Seoul") if stamp.tzinfo is None else stamp.tz_convert("Asia/Seoul")
    parsed = values.map(local)
    require(parsed.map(lambda x:x.weekday()==0 and x==x.normalize()).all(), "Reference week must be Seoul Monday")
    return pd.to_datetime(parsed, utc=True).eq(cutoff.reset_index(drop=True))


def integrate_inputs(features, predictions):
    """Require exact cohort coverage and endpoints; preserve unavailable values as missing."""
    base = keys(features).reset_index(drop=True)
    require(set(predictions) <= {"churn", "uplift", "clv", "gross_4w"}, "Unknown prediction role")
    out = base[KEYS + [name for name in (*PROVENANCE, "observed_tenure_days") if name in base]].copy()
    provenance = {}
    for role in ("churn", "uplift", "clv", "gross_4w"):
        batch = predictions.get(role)
        if batch is None:
            continue
        require(isinstance(batch, PredictionBatch), f"{role} requires a PredictionBatch")
        batch.metadata.validate(role)
        frame = keys(batch.frame)
        required_provenance = [name for name in ("snapshot_id", "raw_database_sha256") if name in base]
        if role in ("clv", "gross_4w"):
            required_provenance += [name for name in ("currency", "amount_unit") if name in base]
        require(set(required_provenance) <= set(frame), f"{role} lacks required provenance: {required_provenance}")
        require(len(frame) == len(base) and set(map(tuple, frame[KEYS].to_numpy())) == set(map(tuple, base[KEYS].to_numpy())),
                f"{role} prediction batch does not cover the exact decision cohort")
        require({"input_cutoff_at", "prediction_end_at"} <= set(frame), f"{role} requires input/prediction endpoints")
        require(utc(frame.input_cutoff_at, "input_cutoff_at").eq(frame.as_of_at).all(),
                f"{role} input cutoff must equal as_of_at")
        weeks = 52 if role == "clv" else 4
        require(utc(frame.prediction_end_at, "prediction_end_at").eq(frame.as_of_at + pd.Timedelta(weeks=weeks)).all(),
                f"{role} prediction endpoint must be exactly {weeks} weeks")
        ordered = base[KEYS].merge(frame, on=KEYS, validate="one_to_one", sort=False)
        for name in PROVENANCE:
            if name in base and name in ordered:
                if name == "reference_week":
                    require(_reference_week(base[name].reset_index(drop=True), base.as_of_at).all()
                            and _reference_week(ordered[name], base.as_of_at).all(), f"{role} reference week differs")
                else:
                    require(base[name].reset_index(drop=True).fillna("<unavailable>").astype(str).eq(
                        ordered[name].fillna("<unavailable>").astype(str)).all(), f"{role} {name} differs from Feature Store")
        if role == "churn":
            out["churn_prob"] = _alias(ordered, ["churn_prob", "churn_probability"], "churn_prob", probability=True)
        elif role == "uplift":
            out["p0"] = _alias(ordered, ["p0", "purchase_probability_no_coupon"], "p0", probability=True)
            out["p1"] = _alias(ordered, ["p1", "purchase_probability_coupon"], "p1", probability=True)
            out["uplift_score"] = out.p1 - out.p0
            for name in ("uplift_score", "uplift_4w"):
                if name in ordered:
                    supplied = pd.to_numeric(ordered[name], errors="raise")
                    require(np.allclose(supplied, out.uplift_score, equal_nan=True, rtol=0, atol=1e-12),
                            "uplift_score must equal purchase p1-p0")
        elif role == "clv":
            policy = batch.producer_metadata.get("policy_condition")
            require(policy is None or policy == "control_no_coupon",
                    "CLV combined with uplift must use a control_no_coupon bundle")
            out["clv"] = _alias(ordered, ["clv", "predicted_clv_12m", "clv_12m"], "clv")
        else:
            out["predicted_gross_4w_if_purchase"] = _alias(ordered, ["predicted_gross_4w_if_purchase"], "gross_4w")
        provenance[role] = {"metadata": vars(batch.metadata), "producer_metadata": batch.producer_metadata}
    for name in ("churn_prob", "p0", "p1", "uplift_score", "clv", "predicted_gross_4w_if_purchase"):
        if name not in out:
            out[name] = np.nan
    out.attrs["prediction_provenance"] = provenance
    out.attrs["source_metadata"] = features.attrs.get("dataset_metadata", {})
    out.attrs["source_monetary_values_observed"] = features.attrs.get("feature_manifest", {}).get(
        "source_monetary_values_observed", True)
    return out
