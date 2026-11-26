import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import fit_return as returning
from simulate.v4.irl.fit_return import build_panel


def test_return_cli_preserves_existing_model_before_fitting(tmp_path, monkeypatch):
    from simulate.v4.irl import fit_return as fitting
    output = tmp_path / "return.json"
    output.write_text("preserved original")
    monkeypatch.setattr("sys.argv", ["fit_return", "--output", str(output)])
    def unexpected_read(*args):
        raise AssertionError("an existing candidate must be rejected before source reading")
    monkeypatch.setattr(fitting, "load_source", unexpected_read)
    with pytest.raises(FileExistsError, match="new artifact path"):
        fitting.main()
    assert output.read_text() == "preserved original"


ORIGIN = pd.Timestamp("2020-01-01", tz="UTC")
SEED = 42


def _ids(train, count):
    return [str(uid) for uid in range(100)
            if (hashlib.sha256(f"{SEED}:{uid}".encode()).digest()[0] < 205) == train][:count]


@pytest.fixture
def first_return_source():
    returned, inactive, late, boundary, immature = _ids(True, 5)
    heldout, anchor = _ids(False, 2)
    histories = {returned: [0, 10], inactive: [1, 60], late: [100, 110],
                 boundary: [90, 91], immature: [145, 146], heldout: [0, 10], anchor: [0, 149]}
    rows = pd.DataFrame([{"user_id": uid, "session": f"{uid}:{index}",
                          "at": ORIGIN + pd.Timedelta(days=day),
                          "event_type": "view", "price_minor": 100}
                         for uid, days in histories.items() for index, day in enumerate(days)])
    return rows, {"returned": returned, "inactive": inactive, "late": late,
                  "boundary": boundary, "immature": immature, "heldout": heldout}


@pytest.mark.parametrize("customer,new_return_day", [("heldout", 40), ("late", 140)])
def test_cold_start_ignores_heldout_users_and_postcutoff_first_return_outcomes(
        first_return_source, customer, new_return_day):
    rows, ids = first_return_source
    original = build_panel(rows, SEED)[2]
    changed = rows.copy()
    changed.loc[changed.session.eq(f"{ids[customer]}:1"), "at"] = ORIGIN + pd.Timedelta(days=new_return_day)
    candidate = build_panel(changed, SEED)[2]

    assert original["cold_start_probability"] == candidate["cold_start_probability"] == 0.5
    assert original["cold_start_fit"] == candidate["cold_start_fit"]


def test_cold_start_requires_full_window_strictly_before_actual_fit_cutoff(first_return_source):
    rows, _ = first_return_source
    observation = build_panel(rows, SEED)[2]
    audit = observation["cold_start_fit"]

    assert audit["cutoff_at"] == observation["snapshots"][-1] == (ORIGIN + pd.Timedelta(days=120)).isoformat()
    assert observation["cold_start_eligible"] == audit["training_windows"] == 2
    assert observation["cold_start_returned"] == audit["returned"] == 1
    assert audit["source_users"] == 7 and audit["training_users"] == 5
    assert audit["heldout_users_excluded"] == 2
    assert audit["training_windows_not_available_before_cutoff"] == 3
    assert audit["training_windows_right_censored_at_observation_end"] == 1
    assert audit["max_outcome_end_at"] == (ORIGIN + pd.Timedelta(days=31)).isoformat()


def test_cold_start_keeps_return_window_end_exclusive(first_return_source):
    rows, ids = first_return_source
    for customer, day in [("returned", 30), ("inactive", 31)]:
        rows.loc[rows.session.eq(f"{ids[customer]}:1"), "at"] = ORIGIN + pd.Timedelta(days=day)

    assert build_panel(rows, SEED)[2]["cold_start_probability"] == 0


def test_cold_start_without_train_mature_support_fails_explicitly(first_return_source):
    rows, ids = first_return_source
    rows = rows.loc[~rows.user_id.isin([ids["returned"], ids["inactive"]])]

    with pytest.raises(ValueError, match="cold-start.*train users.*strictly before cutoff"):
        build_panel(rows, SEED)


@pytest.fixture
def return_fit_events():
    rows = []
    for uid in range(30):
        days = ([0, 10, 40, 70, 100, 140], [1, 60, 120], [2, 149])[uid % 3]
        for session, day in enumerate(days):
            actions = ["view", "cart", "purchase"] if session % 2 else ["view", "cart"]
            for step, action in enumerate(actions):
                rows.append({"user_id": str(uid), "user_session": f"{uid}:{session}",
                    "event_time": (ORIGIN + pd.Timedelta(days=day, seconds=step)).isoformat(),
                    "event_type": action, "product_id": "A", "category_id": "1",
                    "category_code": "", "brand": "", "price": "1.00"})
    return pd.DataFrame(rows)


def test_all_action_return_support_preserves_default_fitted_model(return_fit_events):
    default = returning.fit_return(return_fit_events)
    explicit = returning.fit_return(return_fit_events, available_actions=["view", "cart", "remove", "purchase"])

    assert default == explicit
    assert "available_actions" not in default and "feature_support" not in default


def test_electronics_return_fits_only_observed_coordinates_and_ignores_placeholders(
        return_fit_events, monkeypatch):
    actions = ["view", "cart", "purchase"]
    model = returning.fit_return(return_fit_events, available_actions=actions)
    index = returning.FEATURE_NAMES.index("log_removes_30d")
    assert model["available_actions"] == actions
    assert len(model["feature_support"]) == len(returning.FEATURE_NAMES) == 13
    assert model["feature_support"][index] is False
    assert model["coef"][index] == model["scaler_mean"][index] == model["feature_min"][index] == model["feature_max"][index] == 0
    assert model["scaler_scale"][index] == 1
    panel = returning.build_panel

    def perturbed(*args, **kwargs):
        features, labels, observation = panel(*args, **kwargs)
        features[:, index] = np.nan
        return features, labels, observation

    monkeypatch.setattr(returning, "build_panel", perturbed)
    assert returning.fit_return(return_fit_events, available_actions=actions) == model
    context = dict.fromkeys(returning.CONTEXT_NAMES, 1.)
    context["has_observed_session"] = True
    expected = returning.return_probability(model, context)
    for placeholder in (np.nan, np.inf, -1e100, None):
        context["remove_count_30d"] = placeholder
        assert returning.return_probability(model, context) == expected
    model["coef"][index] = 1
    with pytest.raises(ValueError, match="unused reference"):
        returning.return_probability(model, context)


def test_electronics_return_rejects_contradictory_demonstration(return_fit_events):
    return_fit_events.loc[0, "event_type"] = "remove_from_cart"
    with pytest.raises(ValueError, match="unsupported demonstrated action"):
        returning.fit_return(return_fit_events, available_actions=["view", "cart", "purchase"])


def test_explicit_return_cutoff_uses_strict_same_clock_for_model_and_coldstart(return_fit_events):
    cutoff = ORIGIN + pd.Timedelta(days=90)
    rows, _, _ = returning.prepare_source(return_fit_events)
    _, labels, observation = returning.build_panel(rows, cutoff_at=cutoff)
    model = returning.fit_return(return_fit_events, cutoff_at=cutoff)
    expected = labels.train_user & labels.outcome_end_at.lt(cutoff)

    assert model["validation"]["temporal_cutoff"] == model["cold_start_fit"]["cutoff_at"] == cutoff.isoformat()
    assert model["validation"]["fit_label_available_rule"] == "outcome_end_at < cutoff"
    assert model["validation"]["splits"]["train"]["records"] == int(expected.sum())
    assert (labels.train_user & labels.outcome_end_at.eq(cutoff)).any()
    assert observation["snapshots"][-1] != cutoff.isoformat()


@pytest.mark.parametrize("cutoff", [ORIGIN, ORIGIN + pd.Timedelta(days=150), "2020-03-01"])
def test_return_cutoff_requires_timezone_and_source_bounds(first_return_source, cutoff):
    with pytest.raises(ValueError, match="timezone-aware|observation bounds"):
        build_panel(first_return_source[0], cutoff_at=cutoff)


def test_prepared_return_cli_preserves_verified_rows_and_contract(return_fit_events, tmp_path, monkeypatch):
    from simulate.v4.irl import source_contract
    rows, _, quality = returning.prepare_source(return_fit_events)
    cutoff = (ORIGIN + pd.Timedelta(days=90)).isoformat()
    contract = {"domain": "electronics", "observed_actions": ["view", "cart", "purchase"]}

    def loaded(path, **kwargs):
        assert path == tmp_path / "source.parquet"
        assert kwargs == {"domain": "electronics", "cutoff_at": cutoff}
        return rows, {"source_provenance": {"dataset": "prepared fixture"},
                      "source_contract": contract, "source_quality": quality}

    def must_not_reprepare(*args, **kwargs):
        raise AssertionError("verified prepared rows must not be rebuilt")

    monkeypatch.setattr(source_contract, "load_prepared_source", loaded)
    monkeypatch.setattr(returning, "prepare_source", must_not_reprepare)
    output = tmp_path / "return.json"
    monkeypatch.setattr(sys, "argv", ["fit_return", "--prepared-source", str(tmp_path / "source.parquet"),
        "--domain", "electronics", "--cutoff-at", cutoff, "--output", str(output)])
    returning.main()
    fitted = json.loads(output.read_text())

    assert fitted["source_contract"] == contract
    assert fitted["available_actions"] == contract["observed_actions"]
    assert fitted["cold_start_fit"]["cutoff_at"] == fitted["validation"]["temporal_cutoff"] == cutoff
