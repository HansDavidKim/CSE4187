"""Maintained peer learners consume the public weekly export, never reconstructed futures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulate.v4.time_spec import weekly_metadata
from uplift.common import schema as S
from uplift.common.weekly import load_features, sha256
from uplift.common.simulator import load_feature_dataset
from uplift.common.base import make_classifier, make_regressor
from uplift.Tlearner import TLearner
from uplift.Xlearner import XLearner
from uplift.Hlearner import HLearner
from uplift.api import bind_training_contract, predict_for_strategy


def make_export(root, *, dataset="train", probability=.25, randomized=True, daily=False, incomplete=False):
    root.mkdir()
    folder = root / "features"
    folder.mkdir()
    at = pd.Timestamp("2025-01-06T00:00:00+09:00")
    ids = [f"u{i:02}" for i in range(16)]
    features = pd.DataFrame({"customer_id": ids, "dataset_id": dataset, "as_of_at": at,
        "snapshot_id": [f"{uid}:{at.isoformat()}" for uid in ids], "reference_week": at,
        "currency": "SOURCE_UNSPECIFIED", "amount_unit": "0.01"})
    for name in S.KIT39:
        features[name] = np.arange(16, dtype=float) / 16
    features["visit_recency_days"] = 2.
    features["observed_tenure_days"] = 100.
    if daily:
        features[["net_spend_30d", "cart_qty_current", "mean_net_order_value_30d"]] = np.nan
    ledger = root / "raw.sqlite"
    ledger.write_bytes(("engineering fixture " + dataset).encode())
    checksum = sha256(ledger)
    features["raw_database_sha256"] = checksum
    features.to_parquet(folder / "feature_store.parquet", index=False)
    keys = features[["customer_id", "as_of_at", "snapshot_id"]]
    labels = keys.assign(outcome_end_at=at + pd.Timedelta(days=28),
        observation_end_at=at + pd.Timedelta(days=364), outcome_window_days=28,
        eligibility_window_days=28, is_censored=False, purchased_28d=[0, 1] * 8)
    if incomplete:
        labels.loc[0, "is_censored"] = True
        labels.loc[0, "observation_end_at"] = at + pd.Timedelta(days=27)
    labels.to_parquet(folder / "labels.parquet", index=False)
    # Campaign exports need not repeat snapshot_id in their assignment sidecar.
    assignments = features[["customer_id", "as_of_at"]].assign(assigned_at=at,
        treatment=[int(i < round(16 * probability)) for i in range(16)],
        assignment_probability=probability)
    assignments.to_parquet(folder / "treatments.parquet", index=False)
    meta = {**weekly_metadata(), "predictors": S.KIT39, "raw_database_sha256": checksum,
            "file_sha256": {p.name: sha256(p) for p in folder.iterdir()}}
    (folder / "manifest.json").write_text(json.dumps(meta))
    source = {**weekly_metadata(), "dataset_id": dataset, "randomized_assignment": randomized}
    (root / "raw.manifest.json").write_text(json.dumps({"metadata": source, "database_sha256": checksum}))
    return folder, ledger


def classifier():
    return make_classifier(max_iter=3, min_samples_leaf=2, early_stopping=False)


def regressor():
    return make_regressor(max_iter=3, min_samples_leaf=2, early_stopping=False)


def test_weekly_quarter_assignment_retains_missing_predictors_and_outcomes(tmp_path):
    path, ledger = make_export(tmp_path / "data", daily=True, incomplete=True)
    data = load_feature_dataset(path, ledger=ledger)
    assert len(data) == 16 and data.attrs["assignment_probability"] == .25
    assert data.attrs["dataset_id"] == "train"
    assert data.net_spend_30d.isna().all() and pd.isna(data.y.iloc[0])
    assert data.y.iloc[1:].tolist() == [1., 0.] * 7 + [1.]
    assert list(data[S.KIT39]) == S.KIT39


def test_none_predicts_but_cannot_fit(tmp_path):
    path, _ = make_export(tmp_path / "none", probability=0, randomized=False)
    with pytest.raises(ValueError, match="randomized"):
        load_feature_dataset(path)
    assert load_feature_dataset(path, for_training=False).treatment.eq(0).all()


def test_feature_clock_and_ledger_tampering_are_rejected(tmp_path):
    path, ledger = make_export(tmp_path / "data")
    ledger.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum"):
        load_feature_dataset(path, ledger=ledger)
    meta = json.loads((path / "manifest.json").read_text())
    meta["time_windows_days"]["recent_days"] = 30
    (path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="4/8/13"):
        load_features(path)


@pytest.mark.parametrize("learner", [
    lambda: TLearner(make_model=classifier),
    lambda: XLearner(make_model=classifier, make_effect_model=regressor),
    lambda: HLearner(make_model=classifier, n_rounds=3, lambdas=(.5,), min_samples_leaf=2),
])
def test_actual_learners_save_reload_and_strategy_handoff_without_future_files(tmp_path, learner):
    path, _ = make_export(tmp_path / "train", daily=True)
    data = load_feature_dataset(path)
    model = learner().fit(data[S.FEATURES], data.treatment, data.y, .25)
    bind_training_contract(model, data)
    file = tmp_path / "model.pkl"
    model.save(file)
    with pytest.raises(ValueError, match="training/selection"):
        predict_for_strategy(file, path)
    target, _ = make_export(tmp_path / "decision", dataset="unseen", probability=0, randomized=False, daily=True)
    (target / "labels.parquet").unlink()
    (target / "treatments.parquet").unlink()
    before = predict_for_strategy(model, target)
    after = predict_for_strategy(file, target)
    pd.testing.assert_frame_equal(before.frame, after.frame)
    assert before.frame.customer_id.tolist() == [f"u{i:02}" for i in range(16)]
    assert (before.frame.prediction_end_at-before.frame.input_cutoff_at).eq(pd.Timedelta(weeks=4)).all()
    from final_strategy.inputs import PredictionBatch, integrate_inputs
    joined = integrate_inputs(load_features(target), {"uplift": PredictionBatch(
        before.frame, before.metadata, before.producer_metadata)})
    np.testing.assert_allclose(joined.p1-joined.p0, joined.uplift_score)
    # A legacy fit and an unbound model cannot be silently used on weekly predictors.
    model.uplift_input_contract_["time_spec"] = "legacy_calendar"
    with pytest.raises(ValueError, match="input contract"):
        predict_for_strategy(model, target)
    del model.uplift_input_contract_
    with pytest.raises(ValueError, match="input contract"):
        predict_for_strategy(model, target)


def test_explicit_assignment_can_be_read_without_causal_training(tmp_path):
    path, _ = make_export(tmp_path / "explicit", randomized=False)
    with pytest.raises(ValueError, match="randomized"):
        load_feature_dataset(path)
    assert len(load_feature_dataset(path, for_training=False)) == 16


def test_retention_peer_caller_uses_maintained_t_x_h_models(tmp_path, monkeypatch):
    from retention.experiments import train_peer_models, predict_peer_models
    import uplift.Tlearner as t
    import uplift.Xlearner as x
    import uplift.Hlearner as h
    import clv.api as clv_api
    import retention.experiments as experiments
    path, ledger = make_export(tmp_path / "data")
    def local_output(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)
    monkeypatch.setattr(experiments, "local_output", local_output)
    monkeypatch.setattr(t, "TLearner", lambda: TLearner(make_model=classifier))
    monkeypatch.setattr(x, "XLearner", lambda: XLearner(make_model=classifier, make_effect_model=regressor))
    monkeypatch.setattr(h, "HLearner", lambda: HLearner(make_model=classifier, n_rounds=3,
                                                        lambdas=(.5,), min_samples_leaf=2))
    monkeypatch.setattr(clv_api, "train_clv", lambda *a, **kw: {"provenance": {"dataset_id": "train"}})
    peers = train_peer_models(path, ledger, tmp_path / "peers")
    assert isinstance(peers["uplift"], (TLearner, XLearner, HLearner))
    selection = json.loads((tmp_path / "peers/uplift_selection.json").read_text())
    assert selection["assignment_probability"] == .25
    assert set(selection["selection_qini"]) == {"tlearner", "xlearner", "hlearner"}
    target, ledger = make_export(tmp_path / "target", dataset="new", probability=0, randomized=False)
    features = load_features(target)
    retained = features[["customer_id", "dataset_id", "as_of_at", "currency", "amount_unit"]].assign(
        churn_probability=.2)
    clv_frame = retained.drop(columns="churn_probability").assign(reference_week=features.as_of_at,
                                                                  predicted_clv_12m=100.)
    monkeypatch.setattr(clv_api, "predict_clv", lambda *a, **kw: clv_frame.copy())
    monkeypatch.setattr(clv_api, "evaluate_clv",
                        lambda *a, **kw: {"scope": "unit fixture; no CLV model fitted"})
    result = predict_peer_models(peers, target, ledger, retained, tmp_path / "peer_predictions")
    assert len(result) == 16
    np.testing.assert_allclose(result.purchase_probability_coupon-result.purchase_probability_no_coupon,
                               result.uplift_score)
