import pandas as pd
import numpy as np
import pytest

from preprocessing.amexpert import SEQUENCE_FEATURES, build_customer_week_panel, save_amexpert_panel
from preprocessing.nts_adapter import load_sequences, make_lag_windows
from models.nts_training import adjacency_to_edges, default_prior_knowledge, instantaneous_is_dag, fit_nts_notears
from analysis.eda import extract_markov_boundary, save_feature_sets


def write_fixture(root):
    pd.DataFrame([[1, "X", "02/01/12", "21/01/12"], [2, "X", "01/01/14", "31/01/14"]], columns=["campaign_id", "campaign_type", "start_date", "end_date"]).to_csv(root / "campaign_data.csv", index=False)
    pd.DataFrame([[10, 100]], columns=["coupon_id", "item_id"]).to_csv(root / "coupon_item_mapping.csv", index=False)
    pd.DataFrame([[1, "18-25", "Single", 0, "1", None, 2]], columns=["customer_id", "age_range", "marital_status", "rented", "family_size", "no_of_children", "income_bracket"]).to_csv(root / "customer_demographics.csv", index=False)
    pd.DataFrame([
        ["2012-01-02", 1, 100, 1, 20.0, -2.0, 0.0],
        ["2012-01-09", 1, 100, 2, 35.0, 0.0, -5.0],
        ["2012-01-02", 2, 100, 1, 10.0, 0.0, 0.0],
        ["2012-01-29", 1, 100, 1, 10.0, 0.0, 0.0],
    ], columns=["date", "customer_id", "item_id", "quantity", "selling_price", "other_discount", "coupon_discount"]).to_csv(root / "customer_transaction_data.csv", index=False)
    pd.DataFrame([[100, 1, "Established", "Grocery"]], columns=["item_id", "brand", "brand_type", "category"]).to_csv(root / "item_data.csv", index=False)
    pd.DataFrame([[1, 1, 10, 1, 1]], columns=["id", "campaign_id", "coupon_id", "customer_id", "redemption_status"]).to_csv(root / "train.csv", index=False)


def test_build_boundary_safe_panel(tmp_path):
    write_fixture(tmp_path)
    panel, _, report = build_customer_week_panel(tmp_path)
    assert panel.customer_id.nunique() == 2
    assert panel.groupby("customer_id").size().nunique() == 1
    assert panel[SEQUENCE_FEATURES].isna().sum().sum() == 0
    user = panel[panel.customer_id.eq(1)]
    assert user.offers_started.sum() == 1
    assert user.active_offers.sum() == 3
    assert user.redemptions_reported.sum() == 1
    save_amexpert_panel(tmp_path, tmp_path / "out")
    assert (tmp_path / "out" / "customer_sequences.npz").exists()
    assert report["panel"]["customers"] == 2
    assert report["panel"]["first_week"] == "2012-01-02"
    assert report["panel"]["last_week"] == "2012-01-23"
    assert report["panel"]["weeks"] == 4

    sequences, features = load_sequences(tmp_path / "out" / "customer_sequences.npz")
    windows, sequence_ids = make_lag_windows(sequences, max_lag=2)
    assert features == SEQUENCE_FEATURES
    assert windows.shape[1:] == (len(SEQUENCE_FEATURES), 3)
    assert set(sequence_ids) == {0, 1}


def test_lag_windows_do_not_cross_sequence_boundaries():
    first = np.arange(4, dtype=float).reshape(-1, 1)
    second = np.arange(100, 104, dtype=float).reshape(-1, 1)
    windows, sequence_ids = make_lag_windows([first, second], max_lag=2)
    assert windows[:, 0, :].tolist() == [[0, 1, 2], [1, 2, 3], [100, 101, 102], [101, 102, 103]]
    assert sequence_ids.tolist() == [0, 0, 1, 1]


def test_nts_prior_and_edge_labels():
    features = ["spend", "offers_started"]
    prior = default_prior_knowledge(features)
    assert prior == [{
        "from_node": "spend", "from_lag": 0, "to_node": "offers_started",
        "lower_bound": 0, "upper_bound": 0,
    }]
    adjacency = np.zeros((6, 6))
    adjacency[0, 5] = 0.7  # spend(t-2) -> offers_started(t)
    adjacency[4, 5] = 0.4  # spend(t) -> offers_started(t)
    edges = adjacency_to_edges(adjacency, features, max_lag=2)
    assert [(edge["source"], edge["target"], edge["lag"]) for edge in edges] == [
        ("spend", "offers_started", 2),
        ("spend", "offers_started", 0),
    ]
    assert instantaneous_is_dag(adjacency, 2)
    adjacency[5, 4] = 0.2
    assert not instantaneous_is_dag(adjacency, 2)


def test_temporal_markov_boundary_separates_state_from_post_outcome_nodes():
    edges = [
        {"source": "offers_started", "target": "spend", "lag": 0, "weight": 0.5},
        {"source": "transactions", "target": "spend", "lag": 1, "weight": 0.8},
        {"source": "spend", "target": "redemptions_reported", "lag": 0, "weight": 0.4},
        {"source": "active_offers", "target": "redemptions_reported", "lag": 0, "weight": 0.3},
    ]
    result = extract_markov_boundary(edges, "spend", known_at_decision={"offers_started"})
    assert result["state_candidates"] == [
        {"feature": "offers_started", "lag": 0},
        {"feature": "transactions", "lag": 1},
    ]
    assert {item["feature"] for item in result["spouses"]} == {"active_offers"}
    assert "redemptions_reported" in result["children"]


def test_feature_extraction_rejects_unconverged_graph(tmp_path):
    dag = tmp_path / "dag.json"
    dag.write_text('{"converged": false, "instantaneous_is_dag": false, "edges": []}')
    with pytest.raises(ValueError, match="must converge and be acyclic"):
        save_feature_sets(dag, tmp_path / "features.json", ["spend"])


def test_nts_feature_subset_rejects_unknown_names(tmp_path):
    path = tmp_path / "sequences.npz"
    np.savez(path, values=np.ones((4, 2)), offsets=np.array([0, 4]), features=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="unknown features"):
        fit_nts_notears(path, tmp_path / "out", tmp_path, drop_features=["missing"])

