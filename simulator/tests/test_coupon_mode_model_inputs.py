"""Small accounting fixtures exercise the actual CLV and Retention consumer APIs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

from clv import pipeline as clv_pipeline
from clv.api import predict_for_strategy as clv_for_strategy
from clv.config import CLVConfig
from clv.dataset import build_dataset
from clv.population import policy_mask
from retention.simulator_input import training_inputs, prediction_inputs
from retention.api import predict_for_strategy as retention_for_strategy
from final_strategy import PredictionBatch, load_feature_store, integrate_inputs
from simulate.v4.data.campaign_export import export_campaign, load_campaign_inputs
from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.simulator import main_campaign
from simulate.v4.simulator.canonical_campaign import CanonicalMainCampaign
from simulate.v4.simulator.canonical_response import CanonicalResponseAdapter
from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION
from test_v4_main_campaign import factory
from test_v4_canonical_response import fixture_model


@pytest.fixture
def retail_campaign(factory, monkeypatch):
    native = factory(choice=True, gmm={}, warmup_days=364, horizon_days=364,
        edit=lambda model: model["latent_shopping"].update(persona_labels=["0", "1"]))
    def purchase_each_session(model, context, component, probabilities=None):
        action = ("view", "purchase", "end")[min(context["session_decision_count"], 2)]
        return {name: float(name == action) for name in main_campaign.shopping.ACTIONS}
    monkeypatch.setattr(main_campaign.shopping, "policy", purchase_each_session)
    model, bank, _ = fixture_model("retail_choice_logit", native.model_shas)
    model.update(new_world_calibrated=True, engineering_fixture_only="consumer accounting fixture")
    return CanonicalMainCampaign(native.config, canonical_adapter=CanonicalResponseAdapter(model, bank),
                                 warmup_days=364, min_history_days=0, horizon_days=364)


@pytest.mark.parametrize("assignment,treated", [({"mode": "auto"}, 4), ({"mode": "none"}, 0),
    ({"mode": "assign", "count": 3, "seed": 59}, 3), ({"mode": "assign", "ratio": 1.}, 8)])
def test_campaign_modes_feed_weekly_clv_and_retention(retail_campaign, tmp_path, monkeypatch, assignment, treated):
    output = tmp_path / "campaign"
    result = export_campaign(retail_campaign, output, domain="cosmetics", customers=8, seed=43,
                             coupon_assignment=assignment)
    assert not result["production_export"]
    features = output / "features"
    supplied = pd.read_parquet(features / "feature_store.parquet")
    labels = pd.read_parquet(features / "labels.parquet")
    assignments = pd.read_parquet(features / "treatments.parquet")
    assert int(assignments.treatment.sum()) == treated
    assert labels.clv_outcome_end_at.eq(labels.as_of_at + pd.Timedelta(weeks=52)).all()
    assert labels.future_revenue_12m.notna().sum() == 8-treated
    assert supplied.expanded_feature_set_version.eq(WEEKLY_FEATURE_VERSION).all()
    assert not {"treatment", "assignment_probability", "future_revenue_12m"} & set(EXPANDED_FEATURES)

    panel, orders, reports = clv_pipeline.prepare_inputs(features, output)
    assert all(report.ok for report in reports)
    t0 = supplied.as_of_at.iloc[0]
    dataset = build_dataset(panel, t0, orders)
    assert dataset.label_window_complete.all()
    assert int(policy_mask(dataset, "control_no_coupon").sum()) == 8-treated
    # The source-compatible CLV loader has only mature factual control outcomes.
    if treated == 8:
        with pytest.raises(ValueError, match="targets|control"):
            load_campaign_inputs(output, "clv")
    else:
        keys, x, target = load_campaign_inputs(output, "clv")
        assert len(keys) == len(target) == 8-treated and x.shape[1] == 39
        # This test exercises persistence and prediction without launching model searches.
        monkeypatch.setattr(clv_pipeline, "xgboost_status", lambda: (False, "baseline-only wiring fixture"))
        bundle = tmp_path / "clv"
        registry = clv_pipeline.train(features, output, bundle,
            CLVConfig(final_model="baseline", bgnbd_sensitivity_penalizers=()), cutoff=t0)
        assert registry["model_status"]["baseline"] == "trained"
        predictions = clv_pipeline.predict(features, output, bundle, reference_week=t0, model="baseline")
        assert len(predictions) == 8 and predictions.predicted_clv_12m.notna().all()
        clv_batch = clv_for_strategy(features / "feature_store.parquet", weekly=features, ledger=output,
            model_dir=bundle, model="baseline", allow_training_customers=True, require_control_only=True)
        assert len(clv_batch.frame) == 8 and clv_batch.metadata["horizon_value"] == 52
        assert clv_batch.frame.input_cutoff_at.eq(t0).all()
        assert clv_batch.frame.prediction_end_at.eq(t0 + pd.Timedelta(weeks=52)).all()
        assert set(clv_batch.frame.snapshot_id) == set(supplied.snapshot_id)
        assert clv_batch.gross_4w is None

    config = {"input_dir": output, "feature_dir": features}
    meta, x, seqs, excluded, provenance = training_inputs(config)
    inferred, other, other_seqs, rejected = prediction_inputs(output, t0, config)
    assert len(meta) == 8 and excluded.empty and rejected.empty
    assert provenance["feature_set_version"] == WEEKLY_FEATURE_VERSION
    assert provenance["time_windows_days"]["recent_days"] == 28
    indices = inferred.reset_index().merge(meta[["customer_id", "as_of_at", "snapshot_id"]],
        on=["customer_id", "as_of_at", "snapshot_id"], validate="one_to_one").set_index("customer_id").loc[meta.customer_id, "index"]
    pd.testing.assert_frame_equal(x, other.iloc[indices].reset_index(drop=True))
    for index, sequence in zip(indices, seqs):
        np.testing.assert_array_equal(sequence, other_seqs[index])
    assert meta.policy_id.eq("no_new_coupon").sum() == 8-treated
    assert not {"treatment", "assignment_probability", "future_revenue_12m", "canonical_type"} & set(x)
    # Fit a tiny binary-outcome model to these actual fixture labels, then reload it through the public API.
    imputer = SimpleImputer(strategy="constant", fill_value=0., keep_empty_features=True).fit(x)
    learner = BernoulliNB().partial_fit(imputer.transform(x), meta.y.to_numpy(int), classes=[0, 1])
    model = Pipeline([("imputer", imputer), ("classifier", learner)])
    model_path = tmp_path / "retention.joblib"
    joblib.dump({"schema_version": 2, "model_version": "accounting-fixture", "config": {**config,
        "input_mode": "simulator", "target": "visit", "horizon_days": 28}, "winner": "fixture",
        "threshold": .5, "models": {"fixture": model}, "columns": {"fixture": list(x.columns)},
        "input_contract": {name: meta[name].iloc[0] for name in
            ("definition_id", "expanded_feature_set_version", "currency", "amount_unit")},
        "training_provenance": provenance}, model_path)
    batch = retention_for_strategy(model_path, output, t0)
    assert len(batch.frame) == 8 and batch.frame.churn_prob.between(0, 1).all()
    assert batch.frame.input_cutoff_at.eq(t0).all()
    assert batch.frame.prediction_end_at.eq(t0 + pd.Timedelta(weeks=4)).all()
    assert batch.metadata["horizon_value"] == 4 and batch.metadata["horizon_unit"] == "weeks"
    assert set(batch.frame.snapshot_id) == set(supplied.snapshot_id)
    handoffs = {"churn": PredictionBatch(batch.frame, batch.metadata, batch.producer_metadata)}
    if treated < 8:
        handoffs["clv"] = PredictionBatch(clv_batch.frame, clv_batch.metadata, clv_batch.producer_metadata)
    merged = integrate_inputs(load_feature_store(features), handoffs)
    assert len(merged) == 8 and merged.churn_prob.notna().all()
    assert merged[["p0", "p1", "uplift_score"]].isna().all().all()
    assert bool(merged.clv.notna().all()) == (treated < 8)
