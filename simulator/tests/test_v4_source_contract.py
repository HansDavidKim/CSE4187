from copy import deepcopy
import hashlib
import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import fit_shopping as shopping, fit_donor_prior as prior
from simulate.v4.irl.source_contract import (canonical_available_actions, file_sha, load_prepared_source,
                                           source_profile_id, validate_prepared_rows)


@pytest.fixture
def prepared_cache(tmp_path):
    def create(domain="cosmetics"):
        raw_path = tmp_path / f"{domain}-original.csv"
        raw_path.write_text("unchanged original source\n")
        rows = []
        kinds = ("view", "purchase") if domain == "tianchi" else (
            ("view", "cart", "purchase") if domain == "electronics" else ("view", "cart", "remove_from_cart", "purchase"))
        for user in ("100", "200"):
            for date in ("2024-01-01", "2024-01-20", "2024-02-01", "2024-03-01"):
                for step, kind in enumerate(kinds):
                    at = pd.Timestamp(date, tz="UTC") + (pd.Timedelta(0) if domain == "tianchi" else pd.Timedelta(seconds=step))
                    rows.append(dict(event_time=at.isoformat(), event_type=kind, product_id="p", category_id="c",
                        category_code="", brand="", price="1.05", user_id=user, user_session=f"{user}:{date}",
                        source_event_id=f"source:{len(rows):09d}"))
        frame, _, _ = shopping.prepare_source(pd.DataFrame(rows))
        if domain == "tianchi":
            frame = frame.drop(columns=["event_time", "event_type", "price", "category_code", "brand"])
            frame["price_minor"] = np.nan
        path = tmp_path / f"{domain}.parquet"
        frame.to_parquet(path, index=False)
        stat = raw_path.stat()
        metadata = {"spec": {"domain": domain, "files": [{"path": str(raw_path), "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns}], "sample_modulus": 20 if domain == "electronics" else 100},
            "prepared_sha256": file_sha(path), "rows": len(frame), "users": 2, "sessions": 8,
            "observation_start_at": "2024-01-01T00:00:00Z", "observation_end_at": "2024-03-02T00:00:00Z",
            "inner_cutoff_at": "2024-01-15T00:00:00Z", "outer_cutoff_at": "2024-02-01T00:00:00Z",
            "observed_actions": list(canonical_available_actions(frame.action.unique())),
            "observed_prices": domain != "tianchi", "clock_resolution": "day" if domain == "tianchi" else "second"}
        path.with_suffix(".json").write_text(json.dumps(metadata))
        return path, frame, metadata, raw_path
    return create


@pytest.mark.parametrize("value", [[], "view", ["view", "view"], ["remove_from_cart"], ["end"], 1])
def test_action_support_rejects_ambiguous_or_unknown_values(value):
    with pytest.raises(ValueError, match="available_actions"):
        canonical_available_actions(value)


def test_action_support_canonical_order_and_legacy_fingerprint():
    assert canonical_available_actions(["purchase", "view", "cart"]) == ("view", "cart", "purchase")
    assert canonical_available_actions() == ("view", "cart", "remove", "purchase")
    assert hashlib.sha256(inspect.getsource(shopping.prepare_source).encode()).hexdigest() == "90fdbb4fc7b8e7e85d996e4da8bb31650f89b17048ac60be6dff4e1fd14d2b8f"


@pytest.mark.parametrize("domain", ["cosmetics", "electronics"])
def test_prepared_loader_preserves_text_order_ids_and_contract_without_writing(prepared_cache, domain):
    path, frame, metadata, _ = prepared_cache(domain)
    before = path.read_bytes()
    loaded, descriptor = load_prepared_source(path, domain=domain, cutoff_at=metadata["outer_cutoff_at"])
    pd.testing.assert_frame_equal(loaded, frame.reset_index(drop=True))
    assert loaded.price.eq("1.05").all()
    assert descriptor["source_contract"]["domain"] == domain
    assert descriptor["source_contract"]["rows_at_or_after_outer_cutoff"] > 0
    assert path.read_bytes() == before
    base = {"source": descriptor["source_provenance"], "source_contract": descriptor["source_contract"]}
    again, _ = prior.load_prior_source(base, path)
    pd.testing.assert_frame_equal(loaded, again)
    changed = deepcopy(base)
    changed["source"]["prepared_source_cache"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fitted source contract"):
        prior.load_prior_source(changed, path)


def test_prepared_loader_rejects_source_hash_raw_stat_domain_and_cutoff_changes(prepared_cache):
    path, _, metadata, raw = prepared_cache()
    with pytest.raises(ValueError, match="domain or SHA256"):
        load_prepared_source(path, domain="electronics", cutoff_at=metadata["outer_cutoff_at"])
    with pytest.raises(ValueError, match="cutoffs"):
        load_prepared_source(path, domain="cosmetics", cutoff_at="2024-02-02T00:00:00Z")
    raw.write_text("changed raw size\n")
    with pytest.raises(ValueError, match="original source file changed"):
        load_prepared_source(path, domain="cosmetics", cutoff_at=metadata["outer_cutoff_at"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_prepared_source(path, domain="cosmetics", cutoff_at=metadata["outer_cutoff_at"])


def test_inner_fitting_boundary_preserves_registered_inner_outer_source_split(prepared_cache):
    path, _, metadata, _ = prepared_cache("electronics")
    rows, descriptor = load_prepared_source(path, domain="electronics", cutoff_at=metadata["inner_cutoff_at"])
    contract = descriptor["source_contract"]
    assert pd.Timestamp(contract["outer_cutoff_at"]) == pd.Timestamp(metadata["outer_cutoff_at"])
    assert pd.Timestamp(contract["inner_cutoff_at"]) == pd.Timestamp(metadata["inner_cutoff_at"])
    assert contract["training_cutoff_at"] == descriptor["source_provenance"]["training_cutoff_at"] == contract["inner_cutoff_at"]
    assert contract["rows_before_training_cutoff"] < contract["rows_before_outer_cutoff"]
    again, inherited = prior.load_prior_source({"source": descriptor["source_provenance"], "source_contract": contract}, path)
    pd.testing.assert_frame_equal(rows, again)
    assert inherited["source_contract"]["training_cutoff_at"] == contract["inner_cutoff_at"]
    with pytest.raises(ValueError, match="cutoffs"):
        load_prepared_source(path, domain="electronics", cutoff_at="2024-01-16T00:00:00Z")


def test_daily_loader_delivers_weekly_window_contract_to_actual_panel_and_model(prepared_cache):
    from simulate.v4.irl import tianchi_daily as daily
    path, _, metadata, _ = prepared_cache("tianchi")
    metadata['feature_windows_days'] = [28,56,91]
    path.with_suffix('.json').write_text(json.dumps(metadata))
    rows, descriptor = load_prepared_source(path,domain='tianchi',cutoff_at=metadata['outer_cutoff_at'],require_second_clock=False)
    contract = descriptor['source_contract']
    assert contract['feature_windows_days'] == [28,56,91]
    coverage = {k:contract[k] for k in ('prepared_sha256','observation_start_at','observation_end_at')}
    coverage['assume_complete_behavior_capture_within_declared_interval'] = True
    panel = daily.build_daily_panel(rows,contract,observation_contract=coverage,through_at=metadata['outer_cutoff_at'])
    assert panel['audit']['feature_windows_days'] == [28,56,91]
    fitted = daily.fit_daily_model(panel,max_states=2,max_iter=500)
    assert fitted['feature_windows_days'] == fitted['source_contract']['feature_windows_days'] == [28,56,91]


@pytest.mark.parametrize("mutation", ["order", "sessions", "price", "float_uid", "unsupported"])
def test_prepared_row_validation_rejects_reinterpretation(prepared_cache, mutation):
    _, frame, _, _ = prepared_cache("electronics")
    if mutation == "order":
        frame = frame.iloc[::-1]
    elif mutation == "sessions":
        frame["session"] += 100
    elif mutation == "price":
        frame.loc[frame.index[0], "price_minor"] = 999
    elif mutation == "float_uid":
        frame["user_id"] = frame.user_id.astype(float)
    else:
        frame.loc[frame.index[0], ["action", "event_type"]] = ["remove", "remove_from_cart"]
    with pytest.raises(ValueError):
        validate_prepared_rows(frame, domain="electronics", clock_resolution="second")


def test_daily_source_is_rejected_by_second_clock_fitting_but_keeps_daily_contract(prepared_cache):
    path, frame, metadata, _ = prepared_cache("tianchi")
    with pytest.raises(ValueError, match="daily adapter"):
        load_prepared_source(path, domain="tianchi", cutoff_at=metadata["outer_cutoff_at"])
    loaded, descriptor = load_prepared_source(path, domain="tianchi", cutoff_at=metadata["outer_cutoff_at"], require_second_clock=False)
    pd.testing.assert_frame_equal(loaded, frame.reset_index(drop=True))
    assert descriptor["source_contract"]["observed_prices"] is False


def test_unsupported_choice_is_exactly_zero_even_if_its_unused_logit_is_large():
    size = len(shopping.FEATURE_NAMES)
    block = {"actions": list(shopping.ACTIONS), "feature_names": shopping.FEATURE_NAMES,
             "scaler_mean": [0.] * size, "scaler_scale": [1.] * size,
             "coef": np.zeros((5, size)).tolist(), "intercept": [0., 0., 1000., 0., 0.],
             "available_actions": ["view", "cart", "purchase"]}
    p = shopping.choice_probabilities_batch(block, np.zeros((1, size)))[0]
    np.testing.assert_allclose(p, [.25, .25, 0, .25, .25], rtol=0, atol=1e-15)


def test_train_only_runtime_catalog_and_profiles_exclude_future_and_heldout_summaries(monkeypatch):
    users = [str(value) for value in range(100, 10000, 100)]
    train = next(uid for uid in users if hashlib.sha256(f"42:{uid}".encode()).digest()[0] < 205)
    holdout = next(uid for uid in users if hashlib.sha256(f"42:{uid}".encode()).digest()[0] >= 205)
    rows = []
    for uid in (train, holdout):
        for date in ("2024-01-01", "2024-01-20", "2024-03-01"):
            for second, action in enumerate(("view", "cart", "purchase")):
                future = date == "2024-03-01"
                rows.append(dict(user_id=uid, product_id="future" if future else ("heldout" if uid == holdout else "past"),
                    category_id="c", category_code="", brand="", price="999.00" if future else "1.05",
                    event_time=f"{date}T00:00:0{second}Z", event_type=action, user_session=f"{uid}:{date}"))
    monkeypatch.setattr(shopping, "_fit_choice", lambda features, labels, cutoff, seed, available: ({}, {"cutoff_at_utc": cutoff.isoformat()}))
    model = shopping.fit_shopping(pd.DataFrame(rows), domain="electronics", cutoff_at="2024-02-01T00:00:00Z",
                                  summary_scope="train_only", checkout_episodes=True)
    assert {item["product_id"] for item in model["catalog"]} == {"past"}
    assert {profile["profile_id"] for profile in model["profiles"]} == {source_profile_id("electronics", train)}
    assert model["profiles"][0]["last_observed_at"].startswith("2024-01-20")
    assert model["source_summary"]["source_rows"] == 6
    assert model["validation"]["full_observation_diagnostics"]["source_rows"] == 18
    assert model["available_actions"] == ["view", "cart", "purchase"]


def test_shopping_candidate_publication_never_overwrites_a_racing_writer(tmp_path, monkeypatch):
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(sys, "argv", ["fit", "--output", str(output)])
    monkeypatch.setattr(shopping, "load_source", lambda *args: (pd.DataFrame(), {}))
    def fit(*args, **kwargs):
        output.write_text("other writer's model")
        return {"source": {}}
    monkeypatch.setattr(shopping, "fit_shopping", fit)
    with pytest.raises(FileExistsError):
        shopping.main()
    assert output.read_text() == "other writer's model"
    assert not list(tmp_path.glob(".candidate.json-*"))
