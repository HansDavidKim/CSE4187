import numpy as np
import pandas as pd
import pytest

from simulate.v4 import storage
from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, weekly_calendar, weekly_metadata


@pytest.mark.parametrize("timezone", ["UTC", "Asia/Seoul"])
def test_open_session_clv_export_masks_future_duration_in_matching_timezone(tmp_path, monkeypatch, timezone):
    first, t0, end = weekly_calendar("2024-01-01T00:00:00+09:00")
    metadata = {**weekly_metadata(), "as_of_at": t0.tz_convert(timezone).isoformat(),
                "dataset_id": "fixture", "observation_start_at": first.isoformat()}
    weeks = pd.date_range(t0 - pd.Timedelta(days=91), t0, freq="7D", inclusive="left")
    panel = pd.DataFrame([{"customer_id": "u", "as_of_at": t0, "snapshot_id": "u:t0",
        "reference_week": week, "dataset_id": "fixture", "currency": "source", "amount_unit": ".01",
        "sessions": 1, "purchases": 0, "net_revenue_minor": 0, "views": 1, "session_seconds": 60.}
        for week in weeks])
    feature_row = dict.fromkeys(storage.FEATURE_KEYS, "fixture")
    feature_row.update(customer_id="u", as_of_at=t0, reference_week=t0,
                       expanded_feature_set_version=WEEKLY_FEATURE_VERSION)
    feature_row.update(dict.fromkeys(EXPANDED_FEATURES, 1.))
    features = pd.DataFrame([feature_row])
    tables = {"_metadata": metadata, "labels": pd.DataFrame([{"customer_id": "u", "as_of_at": t0,
        "snapshot_id": "u:t0", "future_revenue_12m": 7., "clv_outcome_end_at": end, "observation_end_at": end}]),
        "treatments": pd.DataFrame({"customer_id": ["u"], "treatment": [0]}),
        "sessions": pd.DataFrame([{"customer_id": "u", "started_at": t0 - pd.Timedelta(hours=1),
                                     "ended_at": t0 + pd.Timedelta(hours=1)}])}
    monkeypatch.setattr(storage, "load_tables", lambda _: tables)
    monkeypatch.setattr(storage, "build_features", lambda *args, **kwargs: features.copy())
    monkeypatch.setattr(storage, "build_experiment_log", lambda *args: pd.DataFrame({"customer_id": ["u"]}))
    monkeypatch.setattr(storage, "build_weekly_panel", lambda *args: panel.copy())
    output = tmp_path / "features"
    storage.materialize_features("fixture", output)
    exported = pd.read_parquet(output / "clv_weekly_inputs.parquet")
    assert len(exported) == 13
    assert exported.reference_week.tolist() == weeks.tolist()
    np.testing.assert_array_equal(exported.session_seconds.iloc[:-1], np.full(12, 60.))
    assert pd.isna(exported.session_seconds.iloc[-1])
    pd.testing.assert_frame_equal(pd.read_parquet(output / "feature_store.parquet"), features)


import hashlib
import json
from simulate.v4.data.loader import load_clv_inputs


def test_clv_loader_requires_thirteen_pre_t0_weeks_and_a_364_day_target(tmp_path):
    _, t0, end = weekly_calendar("2024-01-01T00:00:00+09:00")
    weeks = pd.date_range(t0 - pd.Timedelta(days=91), t0, freq="7D", inclusive="left")
    weekly = pd.DataFrame([{"customer_id": uid, "as_of_at": t0, "snapshot_id": f"{uid}:t0", "reference_week": week,
        "sessions": 1, "purchases": 0, "net_revenue_minor": 0, "views": 2, "session_seconds": 60}
        for uid in ("u", "v") for week in weeks])
    labels = pd.DataFrame([{"customer_id": uid, "as_of_at": t0, "snapshot_id": f"{uid}:t0",
        "future_revenue_12m": 123 if uid == "u" else None, "clv_outcome_end_at": end,
        "observation_end_at": end} for uid in ("u", "v")])
    manifest = {**weekly_metadata(), "file_sha256": {}}

    def save():
        for name, frame in (("clv_weekly_inputs.parquet", weekly), ("clv_labels.parquet", labels)):
            frame.to_parquet(tmp_path / name, index=False)
            manifest["file_sha256"][name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    save()
    snapshots, inputs, targets = load_clv_inputs(tmp_path)
    assert snapshots.customer_id.tolist() == ["u"] and len(inputs) == 13 and targets.tolist() == [123]
    assert not {"treatment", "future_revenue_12m", "canonical_type"} & set(inputs)
    weekly.loc[0, "reference_week"] = t0
    save()
    with pytest.raises(ValueError, match="thirteen strict-past"):
        load_clv_inputs(tmp_path)
    weekly.loc[0, "reference_week"] = weeks[0]
    labels["clv_outcome_end_at"] = t0 + pd.Timedelta(days=365)
    save()
    with pytest.raises(ValueError, match="exactly52weeks"):
        load_clv_inputs(tmp_path)
