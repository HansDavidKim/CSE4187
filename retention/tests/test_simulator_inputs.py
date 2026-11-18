from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retention.dataset import experiment_splits
from retention.simulator_input import (
    CONTROL_POLICY, KEYS, prediction_inputs, simulator_imports, training_inputs,
)


@pytest.fixture(scope="module")
def simulator_release(tmp_path_factory):
    source = Path(__file__).resolve().parents[2] / "simulator" / "src"
    if not source.is_dir():
        pytest.skip("Sibling simulator source is unavailable")
    simulator_imports()
    from simulate.v4.configuration import POPULATION_CONFIG
    from simulate.v4.simulator.engine import EcommerceSimulator
    from simulate.v4.storage import materialize_features

    folder = tmp_path_factory.mktemp("simulator_retention")
    dataset, features = folder / "dataset", folder / "features"
    EcommerceSimulator(POPULATION_CONFIG).generate(n_customers=24, months=6, seed=17).save(dataset)
    materialize_features(dataset, features)
    return dataset, features


def test_official_training_and_raw_only_prediction_match(simulator_release):
    dataset, features = simulator_release
    config = {"input_dir": dataset, "feature_dir": features}
    meta, x, seqs, _, _ = training_inputs(config)
    # Deliberately pass only raw SQLite and its manifest to inference.
    raw_only = dataset.parent / "raw_only"
    raw_only.mkdir()
    from shutil import copyfile
    for name in ("raw.sqlite", "raw.manifest.json"):
        copyfile(dataset / name, raw_only / name)
    predicted, features_at_t0, inference_seqs, excluded = prediction_inputs(raw_only, meta.as_of_at.iloc[0], config)
    assert excluded.empty
    order = predicted.reset_index().merge(meta[KEYS], on=KEYS, validate="one_to_one").set_index("customer_id")
    indices = order.loc[meta.customer_id, "index"].to_numpy()
    pd.testing.assert_frame_equal(x, features_at_t0.iloc[indices].reset_index(drop=True))
    assert x.shape == (24, 49)
    for i, seq in zip(indices, seqs):
        np.testing.assert_array_equal(inference_seqs[i], seq)
    assert meta.label_end.eq(meta.as_of_at + pd.Timedelta(days=28)).all()
    control, _, _, _, _ = training_inputs({**config, "policy_condition": CONTROL_POLICY})
    assert control.policy_id.eq(CONTROL_POLICY).all()
    assert len(control) == 12


def test_fixed_snapshot_holdouts_are_disjoint_and_keep_both_classes():
    meta = pd.DataFrame({"customer_id": [f"u{i}" for i in range(120)],
                         "as_of_at": pd.Timestamp("2024-04-08", tz="Asia/Seoul"),
                         "y": [int(i % 4 == 0) for i in range(120)]})
    folds, dev, test, fit, selection, inner_fit, inner_stop = experiment_splits(meta, 42)
    assert set(fit) | set(selection) == set(dev)
    assert set(fit).isdisjoint(selection) and set(dev).isdisjoint(test)
    assert set(inner_fit) | set(inner_stop) == set(fit)
    assert set(inner_fit).isdisjoint(inner_stop)
    assert len(folds) == 5
    for train, valid in folds:
        assert set(train).isdisjoint(valid) and set(train) | set(valid) == set(fit)
        assert meta.loc[valid, "y"].nunique() == 2


def test_native_future_removal_and_exact_boundary_do_not_change_features(simulator_release, monkeypatch):
    from retention import simulator_input
    dataset, _ = simulator_release
    tables, metadata = simulator_input.read_raw(dataset)
    point = simulator_input.aware_utc(metadata["as_of_at"])
    before, x, seqs, _ = prediction_inputs(dataset, point, {})
    past = dict(tables)
    for name, clock in (("sessions", "started_at"), ("events", "event_at"), ("coupons", "sent_at")):
        past[name] = tables[name].loc[tables[name][clock].lt(point)].copy()
    # A session exactly at t0 belongs to the outcome, never to the past predictors.
    boundary = tables["sessions"].iloc[[0]].copy()
    boundary["session_id"] = "boundary-session"
    boundary["started_at"] = point
    boundary["ended_at"] = point + pd.Timedelta(minutes=1)
    past["sessions"] = pd.concat([past["sessions"], boundary], ignore_index=True)
    monkeypatch.setattr(simulator_input, "read_prediction_raw", lambda *_: (past, metadata))
    after, other, other_seqs, _ = prediction_inputs(dataset, point, {})
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(x, other)
    for left, right in zip(seqs, other_seqs):
        np.testing.assert_array_equal(left, right)


def test_native_empty_eligibility_and_source_mismatch(simulator_release, monkeypatch):
    from retention import simulator_input
    dataset, features = simulator_release
    tables, metadata = simulator_input.read_raw(dataset)
    point = simulator_input.aware_utc(metadata["observation_start_at"])
    meta, x, seqs, excluded = prediction_inputs(dataset, point, {})
    assert meta.empty and x.empty and not seqs and len(excluded) == 24
    monkeypatch.setattr(simulator_input, "read_training_raw", lambda *_: (tables, {**metadata, "dataset_id": "different-source"}))
    with pytest.raises(ValueError, match="dataset_id"):
        training_inputs({"input_dir": dataset, "feature_dir": features})


def test_projected_training_records_match_full_validated_reader(simulator_release, monkeypatch):
    from retention import simulator_input
    dataset, features = simulator_release
    config = {"input_dir": dataset, "feature_dir": features}
    projected, x, seqs, _, _ = training_inputs(config)
    monkeypatch.setattr(simulator_input, "read_training_raw", lambda path, _: simulator_input.read_raw(path))
    full, other, other_seqs, _, _ = training_inputs(config)
    pd.testing.assert_frame_equal(projected, full)
    pd.testing.assert_frame_equal(x, other)
    for left, right in zip(seqs, other_seqs):
        np.testing.assert_array_equal(left, right)


def test_projected_prediction_prefix_matches_full_validated_reader(simulator_release, monkeypatch):
    from retention import simulator_input
    dataset, _ = simulator_release
    _, metadata = simulator_input.read_raw(dataset)
    point = simulator_input.aware_utc(metadata["as_of_at"])
    projected, x, seqs, _ = prediction_inputs(dataset, point, {})
    monkeypatch.setattr(simulator_input, "read_prediction_raw", lambda path, _: simulator_input.read_raw(path))
    full, other, other_seqs, _ = prediction_inputs(dataset, point, {})
    pd.testing.assert_frame_equal(projected, full)
    pd.testing.assert_frame_equal(x, other)
    for left, right in zip(seqs, other_seqs):
        np.testing.assert_array_equal(left, right)
