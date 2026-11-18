import numpy as np
import pandas as pd
import pytest
import json
import joblib
from pathlib import Path

from retention.api import ab_analysis
from retention.evaluate import metrics, threshold_on_validation
from retention.models import QuantileClipper
from retention.monitoring import build_reference, compare
from retention.pipeline import score


def test_clipping_limits_are_learned_from_fit_only():
    fit = pd.DataFrame({"amount": [0., 1., 2., 3.], "missing": [np.nan]*4})
    transform = QuantileClipper(lower=0, upper=1).fit(fit)
    heldout = transform.transform(pd.DataFrame({"amount": [1e12, -1e12, np.inf], "missing": [5., np.nan, 1.]}))
    assert heldout.amount.iloc[:2].tolist() == [3., 0.]
    assert np.isnan(heldout.amount.iloc[2])
    assert transform.upper_["amount"] == 3.


def test_monitoring_detects_value_and_missingness_shift():
    x = pd.DataFrame({"value": np.linspace(0, 1, 100), "missing": [np.nan]*100})
    reference = build_reference(x, {"roc_auc": .8, "precision": .6, "recall": .7})
    unchanged = compare(reference, x)
    assert all(row["psi"] == 0 for row in unchanged["features"].values())
    shifted = pd.DataFrame({"value": np.linspace(5, 6, 100), "missing": [1.]*100})
    report = compare(reference, shifted, {"roc_auc": .6, "precision": .4, "recall": .3})
    assert report["features"]["value"]["psi_exceeded"]
    assert report["features"]["missing"]["psi_exceeded"]
    assert any(alert["kind"] == "performance_drop" for alert in report["alerts"])


def test_operating_point_and_single_class_reports():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .3, .7, .9])
    threshold = threshold_on_validation(y, p)
    report = metrics(y, p, threshold)
    assert report["precision"] == report["recall"] == 1
    assert report["flagged"] == 2
    assert metrics([1, 1], [.7, .9], .5)["roc_auc"] is None


def test_selected_tree_prediction_does_not_execute_sequence_models():
    class Estimator:
        def predict_proba(self, x):
            return np.tile([.8, .2], (len(x), 1))
    class Unused:
        def predict_proba(self, x):
            raise AssertionError("Unused sequence model was executed")
    bundle = {"models": {"tree": Estimator()}, "columns": {"tree": ["x"]},
              "sequence_models": {"lstm": Unused()}, "best_tree": "tree"}
    result = score(bundle, pd.DataFrame({"x": [1, 2]}), [], names=["tree"])
    np.testing.assert_allclose(result["tree"], [.2, .2])


def test_ab_analysis_uses_observed_groups_and_records_design():
    meta = pd.DataFrame({"policy_id": ["no_new_coupon"]*100 + ["coupon"]*100,
                         "y": [1]*30+[0]*70 + [1]*10+[0]*90})
    report = ab_analysis(meta)
    assert report["treatment_minus_control"] < 0
    assert report["p_value"] < .05
    assert report["design"]["required_per_arm"] > 0
    assert report["design"]["target_power"] == .8


def test_atomic_training_failure_can_be_retried(tmp_path, monkeypatch):
    from retention import pipeline
    def temporary_output(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    monkeypatch.setattr(pipeline, "local_output", temporary_output)
    output = tmp_path / "model"
    config = {"input_mode": "simulator", "input_dir": "unused", "feature_dir": "unused", "output_dir": str(output)}
    def fail(config):
        joblib.dump({"partial": True}, Path(config["output_dir"]) / "bundle.joblib")
        raise RuntimeError("report failed")
    monkeypatch.setattr(pipeline, "_train", fail)
    with pytest.raises(RuntimeError, match="report failed"):
        pipeline.train(config)
    assert not (output / "bundle.joblib").exists()
    assert not (tmp_path / "model.lock").exists()
    def finish(config):
        joblib.dump({"model_version": "complete", "config": config}, Path(config["output_dir"]) / "bundle.joblib")
    monkeypatch.setattr(pipeline, "_train", finish)
    pipeline.train(config)
    assert json.loads((output / "run_status.json").read_text())["status"] == "complete"
    assert joblib.load(output / "bundle.joblib")["config"]["output_dir"] == str(output)


def test_direct_training_validates_the_same_28_day_contract():
    from retention.pipeline import train
    with pytest.raises(ValueError, match="28-day"):
        train({"input_mode": "simulator", "input_dir": "unused", "feature_dir": "unused",
               "output_dir": "unused", "horizon_days": 30})


def test_empty_monitoring_has_no_drift_alarm():
    x = pd.DataFrame({"value": [1., 2.]})
    report = compare(build_reference(x, {}), x.iloc[:0])
    assert report["status"] == "no_eligible_customers"
    assert report["alerts"] == []
