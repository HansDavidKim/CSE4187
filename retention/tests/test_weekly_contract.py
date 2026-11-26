"""Weekly model handoffs keep explicit horizons and compatibility column names."""
import numpy as np
import pandas as pd
import pytest
from contextlib import contextmanager
import sqlite3
import hashlib
import json
from pathlib import Path

from retention import api
from retention.feature_dictionary import write_dictionary
from retention import simulator_input


def test_weekly_dictionary_uses_declared_windows_without_renaming_predictors(tmp_path):
    write_dictionary(["session_count_30d", "intervisit_start_gap_mean_days_90d",
                      "orders_relative_change_30d", "history_session_rate_30d"], tmp_path,
                     time_windows_days={"recent_days": 28, "gap_days": 91})
    text = (tmp_path / "feature_dictionary.md").read_text()
    assert "`session_count_30d`" in text and "recent 28d" in text
    assert "91d plus preceding start" in text and "two adjacent 28d windows" in text
    assert "times 28 /" in text and "compatibility aliases" in text
    assert "`session_count_28d`" not in text


def test_strategy_handoff_preserves_exclusions_and_declares_four_week_target(monkeypatch):
    at = pd.Timestamp("2025-01-06T00:00:00+09:00").tz_convert("UTC")
    frame = pd.DataFrame({"dataset_id": ["d", "d"], "customer_id": ["u", "v"],
        "as_of_at": [at, at], "snapshot_id": ["u:t0", "v:t0"],
        "raw_database_sha256": ["a" * 64] * 2, "currency": ["source"] * 2,
        "amount_unit": [".01"] * 2, "expanded_feature_set_version": ["weekly"] * 2,
        "churn_probability": [.25, np.nan], "status": ["scored", "no_recent_session"]})
    frame.attrs["prediction_metadata"] = {"model_version": "fixture", "target_definition_id": api.DEFINITION,
        "horizon_value": 4, "horizon_unit": "weeks", "value_semantics": "probability"}
    frame.attrs["producer_metadata"] = {"policy_condition": "no_new_coupon", "excluded_rows": 1}
    monkeypatch.setattr(api, "infer", lambda *args: frame)
    batch = api.predict_for_strategy("fixture", "fixture", at)
    assert batch.frame.customer_id.tolist() == ["u", "v"]
    assert batch.frame.input_cutoff_at.eq(at).all()
    assert batch.frame.prediction_end_at.eq(at + pd.Timedelta(weeks=4)).all()
    assert batch.frame.churn_prob.iloc[0] == .25 and pd.isna(batch.frame.churn_prob.iloc[1])
    assert batch.frame.status.tolist() == ["scored", "no_recent_session"]
    assert batch.metadata == frame.attrs["prediction_metadata"]
    assert batch.producer_metadata["excluded_rows"] == 1


def test_strategy_handoff_rejects_undeclared_legacy_target(monkeypatch):
    monkeypatch.setattr(api, "infer", lambda *args: pd.DataFrame())
    with pytest.raises(ValueError, match="28-day"):
        api.predict_for_strategy("fixture", "fixture", "2025-01-06T00:00:00+09:00")


@pytest.mark.parametrize("clv_labels_available", [True, False])
def test_peer_predictions_join_purchase_probabilities_by_customer_and_t0(tmp_path, monkeypatch, clv_labels_available):
    from retention import experiments
    from uplift import api as uplift_api
    from clv import api as clv_api
    at = pd.Timestamp("2025-01-06T00:00:00+09:00")
    units = {"dataset_id": "d", "currency": "source", "amount_unit": ".01"}
    purchase = pd.DataFrame({"customer_id": ["v", "u"], "as_of_at": [at, at],
        "p0": [.7, .1], "p1": [.8, .3], "uplift_score": [.1, .2], **units})
    def handoff(model, path):
        assert model == "bound-purchase-model" and path == "features"
        return uplift_api.StrategyPredictions(purchase, {}, {})
    monkeypatch.setattr(uplift_api, "predict_for_strategy", handoff)
    clv = pd.DataFrame({"customer_id": ["v", "u"], "reference_week": [at, at],
        "predicted_clv_12m": [10., 20.], **units})
    def predict(*args, reference_week, **kwargs):
        assert reference_week == at
        return clv.copy()
    monkeypatch.setattr(clv_api, "predict_clv", predict)
    def evaluate(*args, **kwargs):
        if not clv_labels_available:
            raise ValueError("no labelled customers to evaluate at this cutoff")
        return {"fixture_only": True}
    monkeypatch.setattr(clv_api, "evaluate_clv", evaluate)
    monkeypatch.setattr(experiments, "local_output", lambda path: tmp_path)
    retained = pd.DataFrame({"customer_id": ["u", "v"], "as_of_at": [at.tz_convert("UTC")]*2,
        "churn_probability": [.4, .6], **units})
    result = experiments.predict_peer_models({"uplift": "bound-purchase-model", "clv_dir": "models"},
        "features", "raw", retained, tmp_path).set_index("customer_id")
    assert result.loc["u", "purchase_probability_no_coupon"] == .1
    assert result.loc["u", "purchase_probability_coupon"] == .3
    assert result.loc["u", "predicted_clv_12m"] == 20.
    assert result.loc["u", "churn_probability"] == .4
    assert result.loc["v", "purchase_probability_no_coupon"] == .7
    assert result.loc["v", "predicted_clv_12m"] == 10.
    assert (tmp_path / "combined_predictions.parquet").exists()
    report = json.loads((tmp_path / "clv_independent_evaluation.json").read_text())
    assert report.get("status") == (None if clv_labels_available else "unavailable")


def test_peer_strategy_integration_rejects_legacy_calendar_before_generation(tmp_path):
    from retention.experiments import run_experiment
    with pytest.raises(ValueError, match="requires weekly104w"):
        run_experiment({}, tmp_path, months=24, integrate=True)


@pytest.mark.parametrize("mismatch", ["none", "uplift_raw", "clv_raw", "valid"])
def test_cached_peer_cannot_bypass_the_official_model_contract(tmp_path, monkeypatch, mismatch):
    from retention import experiments
    from uplift import api as uplift_api
    from uplift.common import simulator as uplift_input
    from uplift.common.base import BaseUpliftLearner
    data = pd.DataFrame({"treatment": [0, 1], "y": [0, 1],
        "raw_database_sha256": [("b" if mismatch == "uplift_raw" else "a")*64]*2})
    data.attrs = {"dataset_id": "d", "assignment_probability": .5}
    monkeypatch.setattr(uplift_input, "load_feature_dataset", lambda path, ledger: data)
    monkeypatch.setattr(experiments, "local_output", lambda path: tmp_path)
    (tmp_path / "uplift_selection.json").write_text(json.dumps({"winner": "tlearner", "dataset_id": "d",
        "raw_database_sha256": "a"*64}))
    (tmp_path / "clv").mkdir()
    (tmp_path / "clv/registry.json").write_text(json.dumps({"provenance": {"dataset_id": "d",
        "raw_database_sha256": ("b" if mismatch == "clv_raw" else "a")*64}}))
    cached = object()
    monkeypatch.setattr(BaseUpliftLearner, "load", lambda path: cached)
    def verify(model, path, *, allow_training_customers=False):
        assert model is cached and path == "features" and allow_training_customers
        if mismatch != "valid":
            raise ValueError("Model input contract differs from the weekly KIT36")
    monkeypatch.setattr(uplift_api, "predict_for_strategy", verify)
    if mismatch == "valid":
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        result = experiments.train_peer_models("features", "raw", tmp_path)
        assert result["uplift"] is cached and result["clv_dir"] == tmp_path / "clv"
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
        return
    with pytest.raises(ValueError, match="input contract differs" if mismatch == "none" else "raw dataset changed"):
        experiments.train_peer_models("features", "raw", tmp_path)


@pytest.mark.parametrize("cache", ["different_dataset", "partial"])
def test_peer_training_preserves_an_existing_nonreusable_output(tmp_path, monkeypatch, cache):
    from retention import experiments
    from uplift.common import simulator as uplift_input
    data = pd.DataFrame({"treatment": [0, 1], "y": [0, 1], "raw_database_sha256": ["b"*64]*2})
    data.attrs = {"dataset_id": "new", "assignment_probability": .5}
    monkeypatch.setattr(uplift_input, "load_feature_dataset", lambda path, ledger: data)
    monkeypatch.setattr(experiments, "local_output", lambda path: tmp_path)
    monkeypatch.setattr(experiments, "train_test_split", lambda *args, **kwargs: pytest.fail("Existing models must be preserved before fitting"))
    (tmp_path / "tlearner.pkl").write_bytes(b"existing-model")
    (tmp_path / "uplift_selection.json").write_text(json.dumps({"winner": "tlearner", "dataset_id": "old"}))
    if cache == "different_dataset":
        (tmp_path / "clv").mkdir()
        (tmp_path / "clv/registry.json").write_text(json.dumps({"provenance": {"dataset_id": "old"}}))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Existing peer output cannot be reused"):
        experiments.train_peer_models("features", "raw", tmp_path)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_weekly_sequences_exclude_the_legacy_day29_event():
    simulator_input.simulator_imports()
    from simulate.v4.features import EXPANDED_FEATURES
    from simulate.v4.time_spec import weekly_metadata
    at = pd.Timestamp("2025-01-06T00:00:00Z")
    tables = {"_metadata": {**weekly_metadata(), "observation_start_at": (at-pd.Timedelta(days=400)).isoformat()},
        "customers": pd.DataFrame({"customer_id": ["u"], "signed_up_at": [at-pd.Timedelta(days=365)]}),
        "events": pd.DataFrame({"customer_id": ["u", "u"], "session_id": ["s1", "s2"],
            "event_at": [at-pd.Timedelta(days=29), at-pd.Timedelta(days=27)],
            "event_type": ["view", "purchase"]})}
    meta = pd.DataFrame({"customer_id": ["u"], "as_of_at": [at], "snapshot_id": ["u:t0"]})
    x = pd.DataFrame([{name: 1. for name in EXPANDED_FEATURES}])
    _, _, weekly = simulator_input.enrich(tables, meta, x, {})
    assert len(weekly[0]) == 1
    tables["_metadata"].pop("time_spec")
    tables["_metadata"].pop("feature_version")
    _, _, legacy = simulator_input.enrich(tables, meta, x, {})
    assert len(legacy[0]) == 2


def test_weekly_history_queries_include_day905_and_use_a_28day_rate(monkeypatch):
    simulator_input.simulator_imports()
    from simulate.v4.time_spec import weekly_metadata
    at = pd.Timestamp("2025-01-06T00:00:00Z")
    metadata = {**weekly_metadata(), "observation_start_at": (at-pd.Timedelta(days=400)).isoformat()}
    with sqlite3.connect(":memory:") as connection:
        pd.DataFrame({"customer_id": ["u"], "signed_up_at": [(at-pd.Timedelta(days=365)).isoformat()]}).to_sql("customers", connection)
        pd.DataFrame({"customer_id": ["u"]*3, "started_at": [(at-pd.Timedelta(days=d)).isoformat(timespec="nanoseconds")
                     for d in (90.5, 29, 27)]}).to_sql("sessions", connection)
        pd.DataFrame(columns=["customer_id", "event_id", "event_at", "event_type"]).to_sql("events", connection)
        pd.DataFrame(columns=["order_id", "purchase_event_id"]).to_sql("orders", connection)
        @contextmanager
        def database(_):
            yield connection, metadata
        monkeypatch.setattr(simulator_input, "_verified_database", database)
        weekly = simulator_input.read_history_features("fixture", at).iloc[0]
        assert weekly.session_count_90d == 3
        assert weekly.history_session_rate_30d == pytest.approx(3*28/365)
        metadata.pop("time_spec")
        metadata.pop("feature_version")
        legacy = simulator_input.read_history_features("fixture", at).iloc[0]
        assert legacy.session_count_90d == 2
        assert legacy.history_session_rate_30d == pytest.approx(3*30/365)


def test_experiment_cli_forwards_weekly_clock_and_coupon_assignment(monkeypatch):
    from retention import __main__ as cli, experiments
    called = {}
    monkeypatch.setattr(cli, "read_config", lambda path: {})
    def run(*args, **kwargs):
        called.update(args=args, kwargs=kwargs)
        return "fixture"
    monkeypatch.setattr(experiments, "run_experiment", run)
    cli.main(["experiment", "--config", "fixture", "--output", "fixture", "--customers", "8",
              "--weeks", "104", "--coupon-mode", "assign", "--coupon-ratio", ".25",
              "--coupon-assignment-seed", "7"])
    assert called["args"][3] is None and called["kwargs"]["weeks"] == 104
    assert called["kwargs"]["coupon_assignment"] == {"mode": "assign", "ratio": .25, "seed": 7}


@pytest.mark.parametrize("probability", [.5, .25, 3/8])
def test_uplift_uses_the_recorded_cohort_assignment_probability(probability):
    from retention.experiments import _uplift_propensity
    assignments = pd.DataFrame({"treatment": [0, 0, 1, 0], "assignment_probability": [probability]*4})
    assert _uplift_propensity(assignments) == probability


@pytest.mark.parametrize("treatment,probability", [([0, 0], [0, 0]), ([1, 1], [1, 1]),
                                                  ([0, 1], [0, 1]), ([0, 1], [.5, np.nan])])
def test_uplift_rejects_nonidentifiable_coupon_designs(treatment, probability):
    from retention.experiments import _uplift_propensity
    with pytest.raises(ValueError, match="positive overlap"):
        _uplift_propensity(pd.DataFrame({"treatment": treatment, "assignment_probability": probability}))


def test_experiment_rolls_cutoffs_by_thirteen_weeks_before_any_model_fit(tmp_path, monkeypatch):
    from retention import experiments
    calls = []
    monkeypatch.setattr(experiments, "local_output", lambda path: tmp_path)
    def generate(output, customers, months, seed, start_at, **kwargs):
        calls.append((pd.Timestamp(start_at), months, kwargs))
        if len(calls) == 3:
            raise RuntimeError("stop before fitting")
        return Path(output)
    monkeypatch.setattr(experiments, "generate_dataset", generate)
    with pytest.raises(RuntimeError, match="before fitting"):
        experiments.run_experiment({}, tmp_path, customers=8, coupon_assignment={"mode": "none"})
    assert all(start.tzinfo is not None and start.weekday() == 0 and start == start.normalize()
               for start, _, _ in calls)
    assert calls[1][0]-calls[0][0] == calls[2][0]-calls[1][0] == pd.Timedelta(weeks=13)
    assert all(months is None and settings == {"weeks": 104, "coupon_assignment": {"mode": "none"}}
               for _, months, settings in calls)


def test_small_weekly_generation_reuses_only_the_same_coupon_design(tmp_path, monkeypatch):
    from retention import experiments
    def local(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    monkeypatch.setattr(experiments, "local_output", local)
    output = experiments.generate_dataset(tmp_path / "weekly", customers=8, seed=17,
                                          coupon_assignment={"mode": "none"})
    metadata = json.loads((output / "dataset/raw.manifest.json").read_text())["metadata"]
    t0 = pd.Timestamp(metadata["as_of_at"])
    assert t0-pd.Timestamp(metadata["observation_start_at"]) == pd.Timedelta(weeks=52)
    assert pd.Timestamp(metadata["observation_end_at"])-t0 == pd.Timedelta(weeks=52)
    assert metadata["weeks"] == 104 and metadata["months"] is None
    assert metadata["coupon_assignment"]["mode"] == "none"
    path = output / "dataset/raw.sqlite"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert experiments.generate_dataset(output, customers=8, seed=17, coupon_assignment={"mode": "none"}) == output
    with pytest.raises(ValueError, match="coupon assignment"):
        experiments.generate_dataset(output, customers=8, seed=17,
                                     coupon_assignment={"mode": "assign", "count": 1})
    from simulate.v4.simulator import engine
    original_config = engine.load_config
    original_bytes = Path.read_bytes
    for fault in ("configuration", "model", "runtime"):
        with monkeypatch.context() as patch:
            if fault == "configuration":
                def changed_config(source):
                    config = original_config(source)
                    config["source_simulation"]["coupon_send_cost_minor"] += 1
                    return config
                patch.setattr(engine, "load_config", changed_config)
            else:
                changed_path = Path(metadata["model_path"] if fault == "model" else engine.__file__)
                patch.setattr(Path, "read_bytes", lambda file: original_bytes(file) + (b"\n" if file == changed_path else b""))
            with pytest.raises(ValueError, match="generator inputs changed"):
                experiments.generate_dataset(output, customers=8, seed=17, coupon_assignment={"mode": "none"})
    manifest_path = output / "dataset/raw.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["simulator_code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generator inputs changed"):
        experiments.generate_dataset(output, customers=8, seed=17, coupon_assignment={"mode": "none"})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
