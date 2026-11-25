"""Ported human profile declarations retain strict gates and fixed cohort ranks."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from final_strategy import build_decisions, build_strategy_profiles, write_outputs
from final_strategy.pipeline import CUSTOMER_COLUMNS, SEGMENTS
from final_strategy.profile_adapter import profile_gate
from final_strategy.profile_contract import SegmentationConfig
from final_strategy.tests.test_contract import fixture


CUTS = dict(version="fixture-v1", threshold_source="synthetic", common=".2", risk_high=".1", risk_low=".3",
            high_value=".1", standard_value=".3", h4_r=dict(a=0, b=".1", c=".3", d=".4"))


def test_native_declarations_aliases_and_immutable_evidence():
    profiles = build_strategy_profiles(**CUTS)
    by_id = {profile.policy_id: profile for profile in profiles.policies}
    assert set(by_id) == {"P0", "H0", "H1", "H2", "H3", "H4-R", "H4-V"}
    assert by_id["P0"].configuration_fingerprint() == by_id["H0"].configuration_fingerprint()
    assert all(len(profile.policies) == 12 for profile in profiles.policies)
    original = profiles.to_dict()
    exposed = profiles.report
    exposed["segmentation"].clear()
    assert profiles.to_dict() == original
    assert not original["policy_effect_verified"] and not original["dispatch_authorized"]
    json.dumps(original, allow_nan=False)


@pytest.mark.parametrize("patch", [
    {"common": True}, {"common": np.nan}, {"common": 0}, {"common": 1.01},
    {"risk_high": .3}, {"standard_value": .1}, {"threshold_source": " "},
    {"h4_r": dict(a=0,b=.3,c=.3,d=1)}, {"segmentation": SegmentationConfig(churn_threshold=.6)},
])
def test_native_invalid_profile_declarations_fail(patch):
    with pytest.raises(ValueError):
        build_strategy_profiles(**(CUTS | patch))


def test_native_cohort_examples_match_each_strict_profile_gate():
    # The maintained team's example: CLV ties use customer ID and the top ceil(20% n).
    frame = pd.DataFrame([
        ("001",.5,.2,100),("002",.2,.2,100),("003",.8,.2,100),("004",.2,.4,50),
        ("005",.8,-.1,50),("006",.2,0,50),("007",.8,.0001,50),("008",.2,.5,0),
        ("009",.8,.3,50),("010",.2,.31,50),
    ], columns=["customer_id","churn_prob","uplift_score","clv"])
    expected = {
        "P0": {"001","002","003","004","007","008","009","010"},
        "H0": {"001","002","003","004","007","008","009","010"},
        "H1": {"004","008","009","010"}, "H2": {"001","003","004","008","009","010"},
        "H3": {"001","002","004","008","010"}, "H4-R": {"001","003","008","009"},
        "H4-V": {"001","002","008"},
    }
    for profile in build_strategy_profiles(**CUTS).policies:
        for supplied in (frame, frame.iloc[::-1]):
            mask, audit, metadata = profile_gate(supplied, profile)
            assert set(supplied.loc[mask, "customer_id"]) == expected[profile.policy_id]
            assert set(audit.loc[audit.value_band.eq("high_value"), "customer_id"]) == {"001","002"}
            assert not metadata["policy_effect_verified"]


def test_profile_uses_existing_predictions_budget_and_exact_ten_field_output(fixture,tmp_path):
    features,predictions,config,_=fixture
    profiles={p.policy_id:p for p in build_strategy_profiles(**CUTS).policies}
    base=build_decisions(features,predictions,config)
    h0=build_decisions(features,predictions,config,profile=profiles["H0"])
    pd.testing.assert_frame_equal(base.customers,h0.customers)
    selected=build_decisions(features,predictions,config,profile=profiles["H1"])
    assert selected.customers.columns.tolist()==CUSTOMER_COLUMNS
    assert set(selected.customers.segment)==set(SEGMENTS)
    assert selected.budget_summary["expected_cost"]<=config.budget
    assert selected.what_if.expected_cost.le(selected.what_if.total_budget).all()
    eligible=set(selected.graph_data["human_profile"].query("profile_candidate").customer_id)
    assert set(selected.customers.query("send==1").customer_id)<=eligible
    assert selected.metadata["human_strategy_profile"]["policy_id"]=="H1"
    write_outputs(selected,tmp_path/"profile")
    recorded=json.loads((tmp_path/"profile/metadata.json").read_text())
    assert recorded["human_strategy_profile"]["configuration_fingerprint"]==profiles["H1"].configuration_fingerprint()
    assert len(list((tmp_path/"profile/graphs").glob("*.png")))==6
    bad=dict(predictions)
    bad["uplift"]=replace(predictions["uplift"],frame=predictions["uplift"].frame.drop(columns=["p0","p1"]))
    with pytest.raises(ValueError,match="p0"):
        build_decisions(features,bad,config,profile=profiles["H1"])


def test_missing_prediction_is_not_assigned_a_risk_or_value_band():
    frame=pd.DataFrame({"customer_id":["001","002"],"churn_prob":[np.nan,.8],
                        "uplift_score":[.4,.3],"clv":[100.,50.]})
    profile=build_strategy_profiles(**CUTS).policies[0]
    mask,audit,_=profile_gate(frame,profile)
    assert not mask.iloc[0]
    assert audit.loc[0,["risk_band","value_band","minimum_uplift"]].isna().all()
    delayed=replace(profile,policies=tuple(replace(rule,send_delay="1 day") for rule in profile.policies))
    with pytest.raises(ValueError,match="single-t0"):
        profile_gate(frame,delayed)


def test_csv_handoff_preserves_numeric_string_ids_and_provenance(tmp_path):
    from final_strategy.__main__ import read
    path=tmp_path/"predictions.csv"
    pd.DataFrame({"dataset_id":["001"],"customer_id":["0007"],"snapshot_id":["0008"],
                  "raw_database_sha256":["1"*64],"amount_unit":["0.01"]}).to_csv(path,index=False)
    observed=read(path).iloc[0]
    assert observed.customer_id=="0007" and observed.dataset_id=="001" and observed.snapshot_id=="0008"
    assert observed.raw_database_sha256=="1"*64 and observed.amount_unit=="0.01"
