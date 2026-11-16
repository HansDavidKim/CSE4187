import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import KIT_FEATURES, build_features, load_tables
from simulate.v4.simulator.paired_evaluation import paired_retention_rollouts
from simulate.v4.simulator.raw_store import RAW_COLUMNS, save_raw_dataset


@pytest.fixture
def inputs():
    diagnostics = pd.DataFrame({
        "customer_id": ["positive", "negative", "neutral", "always", "never"],
        "no_coupon_return_probability_30d": [0.2, 0.8, 0.4, 0.0, 0.0],
        "coupon_return_probability_30d": [0.75, 0.3, 0.4, 1.0, 0.0],
        "response_type": ["P", "SD", "ST", "P", "LC"],
    })
    metadata = dict(schema_version=7, simulator_version="source-shopping-v2", dataset_id="paired-test",
                    seed=42, as_of_at="2024-01-07T15:00:00Z", definition_id="future_no_session_30d_active30d",
                    model_sha256="a"*64, return_model_sha256="b"*64,
                    observation_start_at="2023-12-01T00:00:00Z", observation_end_at="2024-03-01T00:00:00Z",
                    currency="FIXTURE", amount_unit="0.01")
    return diagnostics, metadata


def test_both_conditions_share_each_draw_and_analytic_truth_is_separate(inputs):
    diagnostics, metadata = inputs
    pairs, truth = paired_retention_rollouts(diagnostics, metadata)
    assert len(pairs) == 5*32 and len(truth) == 5
    assert not pairs.duplicated(["customer_id", "replicate"]).any()
    assert pairs.shared_uniform.ge(0).all() and pairs.shared_uniform.lt(1).all()
    expected = truth.set_index("customer_id")
    assert pairs.control_retained.eq((pairs.shared_uniform < pairs.customer_id.map(expected.no_coupon_return_probability_30d)).astype(int)).all()
    assert pairs.coupon_retained.eq((pairs.shared_uniform < pairs.customer_id.map(expected.coupon_return_probability_30d)).astype(int)).all()
    assert pairs.realized_retention_difference.eq(pairs.coupon_retained-pairs.control_retained).all()
    assert pairs.loc[pairs.customer_id.eq("always"), "realized_retention_difference"].eq(1).all()
    assert pairs.loc[pairs.customer_id.eq("never"), "realized_retention_difference"].eq(0).all()
    assert pairs.loc[pairs.customer_id.eq("neutral"), "realized_retention_difference"].eq(0).all()
    assert pairs.loc[pairs.customer_id.eq("positive"), "realized_retention_difference"].ge(0).all()
    assert pairs.loc[pairs.customer_id.eq("negative"), "realized_retention_difference"].le(0).all()
    assert expected.loc["positive", "true_retention_uplift"] == pytest.approx(0.55)
    assert expected.loc["negative", "true_retention_uplift"] == pytest.approx(-0.5)
    pd.testing.assert_series_equal(expected.mean_paired_retention_difference.sort_index(),
                                  pairs.groupby("customer_id").realized_retention_difference.mean().sort_index(), check_names=False)
    assert expected.loc["positive", "mean_paired_retention_difference"] != expected.loc["positive", "true_retention_uplift"]
    assert "latent_response_type" not in pairs
    assert pairs.paired_outcome_type.eq([{(1,1):"ST",(0,1):"P",(1,0):"SD",(0,0):"LC"}[row.control_retained,row.coupon_retained]
                                        for row in pairs.itertuples(index=False)]).all()
    assert expected.loc["neutral","latent_response_type"] == "ST"
    assert set(pairs.loc[pairs.customer_id.eq("neutral"),"paired_outcome_type"]) == {"ST","LC"}
    assert not pairs.is_factual_replay.any() and not truth.is_factual_replay.any()
    assert pairs.evaluation_only.all() and truth.evaluation_only.all()
    assert pairs.actual_experiment_seed.eq(42).all()
    assert pairs.outcome_scope.eq("first_30_day_activity_equation").all()


def test_repeatability_is_independent_of_row_order_batching_and_numpy_rng(inputs):
    diagnostics, metadata = inputs
    original = diagnostics.copy(deep=True)
    pairs, truth = paired_retention_rollouts(diagnostics, metadata)
    np.random.seed(810)
    np.random.random(1000)
    again, again_truth = paired_retention_rollouts(diagnostics.sample(frac=1, random_state=7), metadata)
    pd.testing.assert_frame_equal(pairs, again)
    pd.testing.assert_frame_equal(truth, again_truth)
    batch, _ = paired_retention_rollouts(diagnostics.iloc[:1], metadata)
    pd.testing.assert_frame_equal(pairs.loc[pairs.customer_id.eq("positive")].reset_index(drop=True), batch)
    prefix, _ = paired_retention_rollouts(diagnostics, metadata, repetitions=8)
    pd.testing.assert_frame_equal(pairs.loc[pairs.replicate.lt(8)].reset_index(drop=True), prefix)
    pd.testing.assert_frame_equal(diagnostics, original)


def test_new_noise_is_not_the_factual_global_rng_and_seed_changes_it(inputs):
    diagnostics, metadata = inputs
    first, _ = paired_retention_rollouts(diagnostics, metadata)
    diagnostics["actual_factual_churn"] = 1
    same, _ = paired_retention_rollouts(diagnostics, metadata)
    pd.testing.assert_frame_equal(first, same)
    changed, _ = paired_retention_rollouts(diagnostics, {**metadata, "seed":43})
    assert not first.shared_uniform.equals(changed.shared_uniform)
    assert changed.actual_experiment_seed.eq(43).all()
    assert changed.paired_derived_seed.ne(first.paired_derived_seed).all()


@pytest.mark.parametrize("mutation", ["duplicate", "empty_id", "numeric_id", "nan", "infinite", "negative",
                                       "too_large", "boolean", "string_probability", "reaction", "empty"])
def test_invalid_diagnostics_are_rejected(inputs, mutation):
    diagnostics, metadata = inputs
    if mutation=="duplicate": diagnostics.loc[1,"customer_id"]="positive"
    elif mutation=="empty_id": diagnostics.loc[0,"customer_id"]=" "
    elif mutation=="numeric_id": diagnostics.loc[0,"customer_id"]=12
    elif mutation=="reaction": diagnostics.loc[0,"response_type"]="guessed"
    elif mutation=="empty": diagnostics=diagnostics.iloc[:0]
    else:
        value={"nan":np.nan,"infinite":np.inf,"negative":-0.1,"too_large":1.1,
               "boolean":True,"string_probability":"0.5"}[mutation]
        diagnostics["coupon_return_probability_30d"] = diagnostics.coupon_return_probability_30d.astype(object)
        diagnostics.loc[0,"coupon_return_probability_30d"]=value
    with pytest.raises(ValueError):
        paired_retention_rollouts(diagnostics, metadata)


@pytest.mark.parametrize("key,value", [("seed",-1),("seed",True),("seed",1.5),("schema_version",7.0),
    ("schema_version",8),("simulator_version","unverified-v3"),("definition_id","past_inactivity"),
    ("dataset_id",""),("as_of_at","2024-01-01"),("model_sha256","wrong")])
def test_seed_version_and_history_contract_are_explicit(inputs,key,value):
    diagnostics,metadata=inputs
    with pytest.raises(ValueError):
        paired_retention_rollouts(diagnostics,{**metadata,key:value})


@pytest.mark.parametrize("repetitions", [0,-1,True,1.5])
def test_repetitions_must_be_a_positive_integer(inputs,repetitions):
    with pytest.raises(ValueError):
        paired_retention_rollouts(*inputs,repetitions=repetitions)


def test_private_evaluation_exports_cannot_enter_raw_loader_or_kit_whitelist(inputs,tmp_path):
    diagnostics,metadata=inputs
    pairs,truth=paired_retention_rollouts(diagnostics,metadata)
    tables={name:pd.DataFrame(columns=columns) for name,columns in RAW_COLUMNS.items()}
    tables["customers"]=pd.DataFrame({"customer_id":diagnostics.customer_id,
                                      "signed_up_at":pd.Timestamp("2023-12-01",tz="UTC")})
    save_raw_dataset(tmp_path/"raw.sqlite",tables,metadata)
    before=build_features(load_tables(tmp_path),as_of_at=metadata["as_of_at"])
    private=tmp_path/"evaluation_only"
    private.mkdir()
    pairs.to_parquet(private/"paired_rollouts.parquet",index=False)
    truth.to_parquet(private/"latent_truth.parquet",index=False)
    loaded=load_tables(tmp_path)
    assert "latent_truth" not in loaded and "paired_rollouts" not in loaded and "diagnostics" not in loaded
    loaded.update(diagnostics=diagnostics,paired_evaluation=truth)
    after=build_features(loaded,as_of_at=metadata["as_of_at"])
    pd.testing.assert_frame_equal(before[list(KIT_FEATURES)],after[list(KIT_FEATURES)])
    assert set(truth)-{"customer_id"} and not set(KIT_FEATURES)&set(truth)
    assert truth.attrs["recommended_directory"]=="evaluation_only"


def test_28_day_view_selects_new_marginals_without_changing_generation_or_history(inputs):
    diagnostics,metadata=inputs
    original=diagnostics.copy(deep=True)
    diagnostics["no_coupon_return_probability_28d"]=diagnostics.no_coupon_return_probability_30d*0.95
    diagnostics["coupon_return_probability_28d"]=diagnostics.coupon_return_probability_30d*0.95
    view={**metadata,"definition_id":"future_no_session_28d_active30d","outcome_window_days":28,
          "generation_window_days":30,"eligibility_window_days":30,"generation_definition_id":metadata["definition_id"],
          "parent_dataset_id":metadata["dataset_id"],"dataset_id":metadata["dataset_id"]+"-28d",
          "source_raw_database_sha256":"c"*64}
    pairs,truth=paired_retention_rollouts(diagnostics,view)
    expected=truth.set_index("customer_id")
    assert pairs.outcome_window_days.eq(28).all() and pairs.generation_window_days.eq(30).all()
    assert pairs.eligibility_window_days.eq(30).all()
    assert pairs.outcome_scope.eq("first_28_day_return_marginal_of_30d_activity_window").all()
    assert pairs.control_retained.eq((pairs.shared_uniform < pairs.customer_id.map(expected.no_coupon_return_probability_28d)).astype(int)).all()
    assert expected.loc["positive","true_retention_uplift"]==pytest.approx(0.55*0.95)
    assert truth.source_raw_database_sha256.eq("c"*64).all()
    pd.testing.assert_frame_equal(diagnostics[original.columns],original)
    for key in ("outcome_window_days","generation_window_days","eligibility_window_days","generation_definition_id","parent_dataset_id"):
        missing=view.copy()
        missing.pop(key)
        with pytest.raises(ValueError):
            paired_retention_rollouts(diagnostics,missing)
    with pytest.raises(ValueError,match="both return probabilities"):
        paired_retention_rollouts(original,view)
