"""Small checks for fidelity benchmark units, boundaries, and comparison claims."""
import copy
import hashlib
import json

import pandas as pd
import pytest

from simulate.v4 import quality as quality


@pytest.fixture
def sample():
    t0 = pd.Timestamp("2019-12-01", tz="UTC")
    sessions = pd.DataFrame([
        ("u", "s", t0, t0+pd.Timedelta(seconds=2)),
        ("v", "empty", t0, t0),
        ("u", "crosses", t0-pd.Timedelta(seconds=1), t0),
        ("u", "ends_on_boundary", t0, t0+pd.Timedelta(days=1)),
    ], columns=["customer_id","session_id","started_at","ended_at"])
    rows = pd.DataFrame([
        ("u","s",t0,1,"cart","a",1,100,"c"),
        ("u","s",t0+pd.Timedelta(seconds=1),2,"purchase","a",2,100,"c"),
        ("u","crosses",t0,3,"view","a",1,100,"c"),
    ], columns=["customer_id","session_id","event_at","event_id","event_type","product_id","quantity","unit_price_minor","category_id"])
    rows.index = [0,0,1]  # Input index labels never multiply the purchased units.
    return t0, rows, sessions


def test_transitions_expand_purchase_units_and_keep_empty_and_boundary_sessions(sample):
    t0, rows, sessions = sample
    result = quality.transition_matrix(rows,sessions,t0,t0+pd.Timedelta(days=1))
    assert result["complete_sessions"] == 2
    assert result["item_units"] == 3 and result["edges"] == 5
    assert result["empty_sessions"] == 1
    assert result["counts"]["start"]["end"] == 1
    assert result["counts"]["start"]["cart"] == 1
    assert result["counts"]["cart"]["purchase"] == 1
    assert result["counts"]["purchase"]["purchase"] == 1
    assert result["counts"]["purchase"]["end"] == 1
    assert result["row_probabilities"]["purchase"]["end"] == .5
    assert quality.transition_distance(result,result)["source_event_weighted_row_L1"] == 0


@pytest.mark.parametrize("quantity", [0,-1,1.5,None])
def test_invalid_item_weights_fail(sample,quantity):
    t0, rows, sessions = sample
    rows=rows.reset_index(drop=True)
    rows["quantity"]=rows.quantity.astype(object)
    rows.loc[0,"quantity"]=quantity
    with pytest.raises(ValueError,match="positive integers"):
        quality.transition_matrix(rows,sessions,t0,t0+pd.Timedelta(days=1))


def test_empty_transition_rows_are_missing_support_not_zero_distance(sample):
    t0, rows, sessions = sample
    result = quality.transition_matrix(rows.iloc[:0],sessions.iloc[:0],t0,t0+pd.Timedelta(days=1))
    assert result["complete_sessions"] == result["edges"] == 0
    assert quality.transition_distance(result,result)["source_event_weighted_row_L1"] is None


def test_selected_customers_with_no_events_stay_in_activity_denominator(sample):
    t0, rows, sessions = sample
    arrays,_,summary=quality.scope_metrics(rows.reset_index(drop=True),sessions,{"u","v"},{"u":"buyer","v":"browser"},t0,t0+pd.Timedelta(days=1),t0)
    assert arrays["purchase_sessions_per_selected_customer"].tolist() == [1,0]
    assert summary["selected_customers_including_zero_activity"] == 2
    assert summary["home_group_counts"] == {"buyer":1,"browser":1}
    json.dumps(summary,allow_nan=False)


def test_report_comparison_marks_protocol_mismatch_and_direction():
    protocol = dict(source_cache_fingerprint="x",customers=200,seeds=[17,29],months=5,clock_start="a",clock_end="b",as_of_at="c")
    scope = dict(action_share_L1=.4,transition_distance={"source_event_weighted_row_L1":.3},
                 numeric_distances={"idle_gap_seconds":{"ks_statistic_unweighted_records":.2,"wasserstein":10}},
                 generated={"home_group_counts":{"browser":100}})
    for domain in ("source","generated"):
        scope.setdefault(domain,{})["repeat_product_across_sessions"] = {"share":.1}
        scope[domain]["remove_then_later"] = {"horizons":{day:{"cart_share":.2,"purchase_share":None} for day in ("7","30")}}
    before=dict(protocol=protocol,profile_artifact_sha256="m",runs=[dict(seed=17,scopes={"five":scope})])
    after=copy.deepcopy(before)
    after["runs"][0]["scopes"]["five"]["action_share_L1"] = .2
    result=quality.compare_reports(before,after)
    assert result["distribution_comparison_eligible"]
    assert result["changes"][0]["distance_deltas_candidate_minus_baseline"]["action_share_L1"] == pytest.approx(-.2)
    after["protocol"]["months"] = 24
    assert not quality.compare_reports(before,after)["distribution_comparison_eligible"]


def test_comparison_skips_history_scopes_with_different_eligibility_windows():
    baseline = {"protocol": {}, "profile_artifact_sha256": "model",
                "runs": [{"scopes": {"pre_asof_30d": {}}}]}
    candidate = {"protocol": {"eligibility_window_days": 28}, "profile_artifact_sha256": "model",
                 "runs": [{"scopes": {"pre_asof_28d": {}}}]}
    result = quality.compare_reports(baseline, candidate)
    assert not result["protocol_matches"]["eligibility_window_days"]
    assert not result["distribution_comparison_eligible"]
    assert result["changes"] == []


@pytest.mark.parametrize("field", ["source_cache_sha256", "configuration_except_profile_sha256"])
def test_comparison_rejects_different_source_content_or_other_settings(field):
    protocol = dict(source_cache_fingerprint="same-stats", customers=200, seeds=[71], months=5,
                    clock_start="a", clock_end="b", as_of_at="c", **{field: "old"})
    baseline = dict(protocol=protocol, profile_artifact_sha256="base", runs=[])
    candidate = copy.deepcopy(baseline)
    candidate["protocol"][field] = "changed"
    result = quality.compare_reports(baseline, candidate)
    assert not result["protocol_matches"][field]
    assert not result["distribution_comparison_eligible"]


def test_source_groups_join_hashed_profile_ids_without_losing_raw_customer_ids():
    profile=hashlib.sha256(b"cosmetics:100").hexdigest()[:16]
    model={"profiles":[{"profile_id":profile,"home_group":"buyer"}]}
    source=pd.DataFrame({"user_id":["100","100","200"]})
    assert quality.source_home_groups(model,source) == {"100":"buyer","200":"unmapped"}
