"""Weekly handoff contract tests with synthetic predictions and official publication."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from simulate.v4 import storage
from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, weekly_metadata
from final_strategy.inputs import load_feature_store
from final_strategy.segmentation import segment_customers
from final_strategy.validation import CONTEXT_COLUMNS, JOIN_KEYS
from final_strategy.weekly_prediction_adapter import adapt_weekly_predictions
from final_strategy.tests.test_inputs import features
from simulator.tests.test_v4_features_sqlite import dataset, T0


@pytest.fixture
def published(features, tmp_path, monkeypatch):
    raw = tmp_path / "raw.sqlite"
    raw.write_bytes(b"source checksum fixture")
    features["raw_database_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    features["expanded_feature_set_version"] = WEEKLY_FEATURE_VERSION
    metadata = dict(dataset_id="dummy", raw_database_sha256=features.raw_database_sha256.iloc[0],
                    model_sha256="b" * 64, **weekly_metadata())
    monkeypatch.setattr(storage, "load_tables", lambda path: {"_metadata": metadata})
    monkeypatch.setattr(storage, "build_features", lambda *args, **kwargs: features.copy())
    store = tmp_path / "features"
    storage.materialize_features("fixture", store)
    return store


@pytest.fixture
def cases(published):
    features = load_feature_store(published)
    keys = features[list(CONTEXT_COLUMNS)].copy()
    def batch(target, semantics, weeks, producer, **values):
        frame = keys.assign(input_cutoff_at=keys.as_of_at,
                            prediction_end_at=keys.as_of_at + pd.Timedelta(weeks=weeks), **values)
        return SimpleNamespace(frame=frame.iloc[::-1].copy(), producer_metadata=producer,
                               metadata=dict(model_version="v1", target_definition_id=target, horizon_value=weeks,
                                             horizon_unit="weeks", value_semantics=semantics))
    rp = dict(component="retention", policy_condition="observed_experiment_mix",
              status_counts={"scored": 1, "not_signed_up_at_cutoff": 1, "no_recent_session": 1},
              excluded_rows=2, **weekly_metadata())
    retention = batch("future_no_session_28d_active28d", "probability", 4, rp,
                      expanded_feature_set_version=WEEKLY_FEATURE_VERSION,
                      status=["scored", "not_signed_up_at_cutoff", "no_recent_session"],
                      churn_prob=[.75, np.nan, np.nan])
    up = dict(component="uplift", input_contract=weekly_metadata(), development_overlap_allowed=False,
              training_overlap_customers=[], training={"dataset_id": "other", "training_customers": ["old"],
              "selection_customers": []})
    uplift = batch("purchase_28d_randomized_coupon_v1", "probability_pair", 4, up,
                   p0=[.2, .8, 0], p1=[.5, .5, 0], uplift_score=[.3, -.3, 0])
    cp = dict(producer="clv", model_version="v1", policy_condition="control_no_coupon",
              training_overlap={"allowed": False, "roles": dict(train=0, validation=0, unused=3, test=0)},
              inputs=dict(dataset_id="dummy", raw_database_sha256=features.raw_database_sha256.iloc[0]),
              gross_4w={"status":"not_trained","model_version":None})
    clv = batch("clv_paid_amount_52w_v1", "nonnegative_amount", 52, cp, predicted_clv_12m=[120, 90, 0])
    clv.audit = keys[list(JOIN_KEYS)].assign(clv_training_role="unused").iloc[::-1]
    clv.gross_4w = clv.gross_4w_metadata = None
    clv.gross_4w_status = "not_trained"
    return features, dict(retention=retention, uplift=uplift, clv=clv)


def test_projection_and_exclusion_use_keys_and_preserve_inputs(cases):
    features, batches = cases
    before = deepcopy(batches)
    result = adapt_weekly_predictions(features, **batches)
    assert result.roster.customer_id.tolist() == ["001", "002", "003"]
    assert result.roster.uplift_score.tolist() == [.3, -.3, 0]
    assert result.roster.predicted_gross_4w_if_purchase.isna().all()
    assert result.integrated.data.predicted_clv_12m.tolist() == [120]
    assert result.integrated.prediction_metadata["uplift"].value_semantics == "probability_difference"
    assert result.prediction_metadata["uplift"].value_semantics == "probability_pair"
    assert result.excluded.exclusion_reason.tolist() == ["not_signed_up_at_cutoff", "no_recent_session"]
    assert result.evidence["feature_store"]["raw_file_verified"]
    assert segment_customers(result.integrated).customers.priority_score.tolist() == [36]
    assert features.mean_net_order_value_30d.isna().all()
    result.producer_metadata["uplift"]["training"]["training_customers"].clear()
    result.roster.loc[0, "p0"] = 0
    for role, obj in batches.items():
        assert obj.producer_metadata == before[role].producer_metadata
        pd.testing.assert_frame_equal(obj.frame, before[role].frame)


def test_optional_gross_uses_its_own_model_and_four_week_endpoint(cases):
    features, batches = cases
    clv = batches["clv"]
    clv.gross_4w = clv.frame.drop(columns="predicted_clv_12m").assign(predicted_gross_4w_if_purchase=13)
    clv.gross_4w["prediction_end_at"] = clv.gross_4w.as_of_at + pd.Timedelta(weeks=4)
    clv.gross_4w_metadata = dict(clv.metadata, target_definition_id="gross_amount_4w_if_purchase_v1",
                                 model_version="gross-v1", horizon_value=4)
    clv.gross_4w_status = "trained"
    clv.producer_metadata["gross_4w"] = dict(status="trained", model_version="gross-v1")
    result = adapt_weekly_predictions(features, **batches)
    assert result.roster.predicted_gross_4w_if_purchase.tolist() == [13]*3
    assert result.integrated.data.predicted_clv_12m.tolist() == [120]
    clv.gross_4w["prediction_end_at"] = clv.frame.prediction_end_at
    with pytest.raises(ValueError, match="prediction_end_at"):
        adapt_weekly_predictions(features, **batches)


@pytest.mark.parametrize("role,field,value", [
    ("retention", "snapshot_id", "other"), ("retention", "churn_prob", 0),
    ("retention", "status", "unknown"), ("uplift", "p0", np.inf), ("uplift", "p1", True),
    ("uplift", "p1", 1.1), ("uplift", "uplift_score", .1), ("uplift", "amount_unit", "1"),
    ("clv", "dataset_id", "other"), ("clv", "raw_database_sha256", "e"*64),
    ("clv", "currency", "other"), ("clv", "predicted_clv_12m", np.nan),
    ("clv", "input_cutoff_at", pd.Timestamp("2020-01-01", tz="UTC")),
])
def test_bad_predictions_fail_even_on_excluded_customers(cases, role, field, value):
    features, batches = cases
    batches[role].frame[field] = value
    with pytest.raises(ValueError):
        adapt_weekly_predictions(features, **batches)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_clock", "overlap", "selection",
                                    "bad_retention_clock", "clv_overlap", "empty_model_version"])
def test_coverage_contracts_and_development_evidence(cases, fault):
    features, batches = cases
    u, r, c = batches["uplift"], batches["retention"], batches["clv"]
    if fault == "missing":
        u.frame = u.frame.iloc[:-1]
    elif fault == "duplicate":
        u.frame = pd.concat([u.frame, u.frame.iloc[:1]])
    elif fault == "wrong_clock":
        u.producer_metadata["input_contract"]["time_windows_days"]["recent_days"] = 30
    elif fault == "overlap":
        u.producer_metadata["development_overlap_allowed"] = True
    elif fault == "selection":
        u.producer_metadata["training"].update(dataset_id="dummy", selection_customers=["001"])
    elif fault == "bad_retention_clock":
        r.producer_metadata["time_spec"] = "legacy"
    elif fault == "clv_overlap":
        c.audit["clv_training_role"] = "validation"
    else:
        c.metadata["model_version"] = " "
    with pytest.raises(ValueError):
        adapt_weekly_predictions(features, **batches)


def test_policy_and_full_exclusion(cases):
    features, batches = cases
    c, r = batches["clv"], batches["retention"]
    c.producer_metadata["policy_condition"] = "observed_experiment_mix"
    with pytest.raises(ValueError, match="CLV producer or policy"):
        adapt_weekly_predictions(features, **batches)
    r.frame["status"], r.frame["churn_prob"] = "no_recent_session", np.nan
    r.producer_metadata.update(excluded_rows=3, status_counts={"no_recent_session": 3},
                               policy_condition="no_new_coupon")
    with pytest.raises(ValueError, match="Retention policy"):
        adapt_weekly_predictions(features, **batches, allow_mixed_clv_for_development=True)
    result = adapt_weekly_predictions(features, **batches, allow_mixed_clv_for_development=True,
                                      expected_retention_policy="no_new_coupon")
    assert result.integrated is None and len(result.excluded) == len(features)


@pytest.mark.parametrize("fault", ["mixed", "clock", "window", "order", "raw_sha", "rename", "missing_raw"])
def test_weekly_publication_evidence(published, fault):
    path = published / "feature_store.parquet"
    mp = published / "manifest.json"
    manifest = json.loads(mp.read_text())
    if fault in ("mixed", "clock", "order"):
        frame = pd.read_parquet(path)
        if fault == "mixed":
            frame.loc[0, "expanded_feature_set_version"] = "kit39-review-cart-v3"
        elif fault == "clock":
            frame["as_of_at"] += pd.Timedelta(hours=1)
        else:
            frame = frame[frame.columns[::-1]]
        frame.to_parquet(path, index=False)
        manifest["file_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "window":
        manifest["time_windows_days"]["gap_days"] = 90
    elif fault == "rename":
        manifest["feature_set_version"] = "kit39-review-cart-v3"
    elif fault == "raw_sha":
        (published.parent / "raw.sqlite").write_bytes(b"changed")
    else:
        (published.parent / "raw.sqlite").unlink()
        assert load_feature_store(published).attrs["feature_store_evidence"]["raw_file_verified"] is False
        return
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_feature_store(published)


def test_public_producer_calls_use_one_cutoff_and_no_overlap_override(cases, published, monkeypatch):
    from final_strategy.weekly_prediction_adapter import predict_weekly_inputs
    retention_api = pytest.importorskip("retention.api")
    from uplift import api as uplift_api
    from clv import api as clv_api

    features, batches = cases
    calls = []
    def retention(bundle, dataset, at):
        calls.append(("retention", bundle, dataset, at))
        return retention_api.StrategyPredictions(**vars(batches["retention"]))
    def uplift(model, directory, *, allow_training_customers):
        calls.append(("uplift", model, directory, allow_training_customers))
        return uplift_api.StrategyPredictions(**vars(batches["uplift"]))
    def clv(directory, *, model_dir, ledger, allow_training_customers, require_control_only, require_gross_4w):
        calls.append(("clv", directory, model_dir, ledger, allow_training_customers, require_control_only, require_gross_4w))
        return clv_api.StrategyPredictions(**vars(batches["clv"]))
    for module, call in ((retention_api, retention), (uplift_api, uplift), (clv_api, clv)):
        monkeypatch.setattr(module, "predict_for_strategy", call)
    result = predict_weekly_inputs(published, retention_bundle="r", raw_dataset="raw", uplift_model="u",
                                   clv_model_dir="c")
    assert calls == [("retention", "r", "raw", features.as_of_at.iloc[0]),
                     ("uplift", "u", published, False), ("clv", published, "c", "raw", False, True, False)]
    assert result.integrated.data.customer_id.tolist() == ["001"]


@pytest.mark.parametrize("weekly", [False, True])
def test_real_raw_sqlite_to_published_store(dataset, tmp_path, weekly):
    from simulate.v4.simulator.raw_store import load_raw_dataset, save_raw_dataset

    tables, metadata = load_raw_dataset(dataset / "raw.sqlite")
    root = tmp_path / "publication"
    root.mkdir()
    raw = root / "raw.sqlite"
    metadata["model_sha256"] = "b" * 64
    if weekly:
        metadata.update(weekly_metadata())
    save_raw_dataset(raw, tables, metadata)
    directory = root / "features"
    storage.materialize_features(raw, directory, as_of_at=T0)
    result = load_feature_store(directory)
    assert result.expanded_feature_set_version.eq(
        WEEKLY_FEATURE_VERSION if weekly else "kit39-review-cart-v3").all()
    assert result.attrs["feature_store_evidence"]["raw_file_verified"] is True
    assert result.net_spend_30d.sum() == 300
    assert result.loc[result.customer_id.eq("v"), "mean_net_order_value_30d"].isna().all()


@pytest.mark.parametrize("age,recent,delta", [
    (28, (1, 1), (1, 1)), (30, (1, 0), (1, -1)),
    (56, (0, 0), (-1, -1)), (60, (0, 0), (-1, 0)),
])
def test_observed_purchase_on_exact_lower_boundaries(dataset, age, recent, delta):
    from simulate.v4.features import build_features, load_tables

    tables = load_tables(dataset / "raw.sqlite")
    event = tables["events"]
    buy = event.session_id.eq("buy")
    shift = T0 - pd.Timedelta(days=age) - pd.Timestamp(event.loc[event.event_type.eq("purchase"), "event_at"].iloc[0])
    event.loc[buy, "event_at"] = pd.to_datetime(event.loc[buy, "event_at"], utc=True) + shift
    session = tables["sessions"]
    for field in ("started_at", "ended_at"):
        session.loc[session.session_id.eq("buy"), field] += shift
    values = []
    for clock in ({}, weekly_metadata()):
        tables["_metadata"].update(clock)
        row = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
        values.append(row)
    assert tuple(r.purchase_count_30d for r in values) == recent
    assert tuple(r.orders_delta_30d for r in values) == delta
    assert values[0].purchase_recency_days == values[1].purchase_recency_days == age


def test_gap_ending_on_day91_is_weekly_only(dataset):
    from simulate.v4.features import build_features, load_tables

    tables = load_tables(dataset / "raw.sqlite")
    extra = tables["sessions"].iloc[[0, 0, 0]].copy()
    extra["session_id"] = ["day95", "day91", "day90"]
    extra["started_at"] = [T0-pd.Timedelta(days=age) for age in (95, 91, 90)]
    extra["ended_at"] = extra.started_at + pd.Timedelta(minutes=1)
    tables["sessions"] = pd.concat([extra, tables["sessions"]], ignore_index=True)
    expected = tables["sessions"].loc[lambda s: s.customer_id.eq("u"), "started_at"].sort_values()
    gaps = expected.diff().dt.total_seconds()/86400
    for days, clock in ((90, {}), (91, weekly_metadata())):
        tables["_metadata"].update(clock)
        row = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
        assert row.intervisit_start_gap_mean_days_90d == pytest.approx(gaps[expected.ge(T0-pd.Timedelta(days=days))].mean())


def test_engine_rejects_legacy_predictions_on_weekly_features(cases):
    from final_strategy.tests.test_inputs import batches_for
    from final_strategy.inputs import integrate_inputs

    weekly, _ = cases
    with pytest.raises(ValueError, match="prediction feature version"):
        integrate_inputs(weekly, batches_for(weekly))
