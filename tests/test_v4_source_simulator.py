"""Behavioral contracts on a fitted test fixture; no external raw files required."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from simulate.v4.cli import load_config

from simulate.v4.irl.fit_shopping import fit_shopping
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, discount_minor, load_raw_dataset, validate_raw_dataset


@pytest.fixture(scope="module")
def source_config(tmp_path_factory):
    # Synthetic test observations verify wiring, never estimate production behavior.
    sequence = [("view", "A"), ("cart", "A"), ("cart", "A"), ("remove_from_cart", "A"),
                ("view", "B"), ("cart", "B"), ("purchase", "B"), ("purchase", "A"), ("view", "A")]
    rows = []
    for user in range(40):
        for session, day in enumerate(("2020-01-01", "2020-01-03", "2020-02-15")):
            for step, (kind, product) in enumerate(sequence):
                at = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=user % 20, seconds=step * 5)
                rows.append(dict(event_time=at.isoformat(), event_type=kind, product_id=product,
                                 category_id=f"category-{product}", category_code="", brand="",
                                 price="1.05" if product == "A" else "2.50", user_id=str(user),
                                 user_session=f"{user}:{session}"))
    model = fit_shopping(pd.DataFrame(rows), {"dataset": "deterministic generator test fixture"})
    folder = tmp_path_factory.mktemp("source-simulator")
    model_path = folder / "profiles.json"
    model_path.write_text(json.dumps(model, allow_nan=False))
    config = load_config()
    config["churn"]["target_rate"] = .2
    config["source_simulation"]["model_path"] = str(model_path)
    config["source_simulation"].update(retention_mode="target", signup_mode="legacy_bounded_uniform")
    config["source_simulation"]["return_model_path"] = None
    config["source_simulation"]["coupon_response_model_path"] = None
    config["source_simulation"]["visit_timing_model_path"] = None
    for name in ("search", "review", "cs"):
        config["source_simulation"][f"{name}_model_path"] = None
    return config


@pytest.fixture(scope="module")
def generated(source_config):
    return EcommerceSimulator(copy.deepcopy(source_config)).generate(n_customers=40, months=3, seed=17)


def test_source_seed_is_reproducible_and_artifact_hash_is_exact(source_config, generated):
    again = EcommerceSimulator(copy.deepcopy(source_config)).generate(n_customers=40, months=3, seed=17)
    assert generated.metadata == again.metadata
    for name in generated.tables:
        pd.testing.assert_frame_equal(generated.tables[name], again.tables[name])
    path = Path(source_config["source_simulation"]["model_path"])
    assert generated.metadata["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_units_utc_snapshot_eligibility_and_balanced_assignment(generated):
    tables, metadata = generated.tables, generated.metadata
    validate_raw_dataset({name: tables[name] for name in RAW_COLUMNS}, metadata)
    assert set(tables["customers"]) == {"customer_id", "signed_up_at"}
    assert metadata["currency"] == "SOURCE_UNSPECIFIED" and metadata["amount_unit"] == "0.01"
    t0 = pd.Timestamp(metadata["as_of_at"])
    local = t0.tz_convert("Asia/Seoul")
    assert local.weekday() == 0 and local.hour == 0 and local.minute == 0
    assert str(tables["events"].event_at.dt.tz) == "UTC"
    assert tables["events"].event_id.is_unique
    assert tables["events"].sort_values("event_id").event_at.is_monotonic_increasing
    eligible = tables["sessions"].loc[tables["sessions"].started_at.ge(t0-pd.Timedelta(days=30)) &
                                      tables["sessions"].started_at.lt(t0), "customer_id"]
    assert set(eligible) == set(tables["customers"].customer_id)
    assert tables["treatments"].treatment.value_counts().to_dict() == {0: 20, 1: 20}
    assert tables["treatments"].assignment_probability.eq(0.5).all()
    assert tables["coupons"].customer_id.nunique() == 20
    assert tables["coupons"].sent_at.eq(t0).all()


def test_future_labels_are_recomputed_from_observed_sessions_and_paid_orders(generated):
    tables = generated.tables
    labels = tables["labels"]
    t0, upper = labels.as_of_at.iloc[0], labels.outcome_end_at.iloc[0]
    assert upper - t0 == pd.Timedelta(weeks=4)
    assert labels.definition_id.eq("future_no_session_28d_active30d").all()
    assert generated.metadata["outcome_window_days"] == 28
    assert generated.metadata["generation_window_days"] == 30
    assert tables["coupons"].expires_at.eq(t0 + pd.Timedelta(days=30)).all()
    future = tables["sessions"].loc[tables["sessions"].started_at.ge(t0) & tables["sessions"].started_at.lt(upper)]
    active = set(future.customer_id)
    assert labels.set_index("customer_id").churn.to_dict() == {
        uid: int(uid not in active) for uid in tables["customers"].customer_id}
    events = tables["events"].set_index("event_id")
    purchases = tables["orders"].join(events[["customer_id", "event_at"]], on="purchase_event_id")
    observed = purchases.loc[purchases.event_at.ge(t0) & purchases.event_at.lt(upper)].groupby("customer_id").paid_minor.sum()
    assert labels.set_index("customer_id").future_revenue_28d.to_dict() == {
        uid: int(observed.get(uid, 0)) for uid in tables["customers"].customer_id}
    pd.testing.assert_series_equal(labels.future_revenue, labels.future_revenue_28d, check_names=False)
    observed30 = purchases.loc[purchases.event_at.ge(t0) & purchases.event_at.lt(t0 + pd.Timedelta(days=30))].groupby("customer_id").paid_minor.sum()
    assert labels.set_index("customer_id").future_revenue_30d.to_dict() == {
        uid: int(observed30.get(uid, 0)) for uid in tables["customers"].customer_id}
    assert generated.metadata["observed_churn_rate"] == labels.churn.mean()
    assert labels.future_revenue_12m.isna().all()


def test_cart_prefixes_persist_across_sessions_and_whole_checkout_reconciles(generated):
    tables = generated.tables
    order_lookup = tables["orders"].set_index("purchase_event_id")
    items = {key: group for key, group in tables["order_items"].groupby("order_id")}
    carried = False
    for customer_id, events in tables["events"].groupby("customer_id", sort=False):
        cart = Counter()
        previous_session = None
        for event in events.sort_values(["event_at", "event_id"]).itertuples(index=False):
            if event.session_id != previous_session and +cart:
                carried = True
                assert cart_at(tables, customer_id, event.event_at, event.event_id) == dict(+cart)
            previous_session = event.session_id
            if event.event_type == "cart": cart[event.product_id] += int(event.quantity)
            elif event.event_type == "remove_from_cart": cart[event.product_id] -= int(event.quantity)
            elif event.event_type == "purchase":
                order = order_lookup.loc[event.event_id]
                basket = Counter()
                for item in items[order.order_id].itertuples(index=False):
                    basket[item.product_id] += int(item.quantity)
                if order.checkout_source == "cart":
                    assert basket == +cart
                    cart.clear()
                else:
                    assert not +cart and sum(basket.values()) == 1
            assert all(quantity >= 0 for quantity in cart.values())
        assert cart_at(tables, customer_id, generated.metadata["end_at"]) == dict(+cart)
    assert carried, "fixture should exercise a cart carried into another session"


def test_catalog_prices_reach_the_choice_policy_and_accounting(source_config, generated):
    catalog = generated.tables["products"].copy()
    catalog["unit_price_minor"] *= 100_000
    changed = EcommerceSimulator(copy.deepcopy(source_config)).generate(
        n_customers=40, months=3, seed=17, products=catalog)
    validate_raw_dataset({name: changed.tables[name] for name in RAW_COLUMNS}, changed.metadata)
    observed = changed.tables["events"].dropna(subset=["product_id"])
    prices = catalog.set_index("product_id").unit_price_minor
    assert observed.unit_price_minor.eq(observed.product_id.map(prices)).all()
    assert generated.tables["events"].event_type.tolist() != changed.tables["events"].event_type.tolist()


def test_raw_save_roundtrip_coupon_accounting_and_no_overwrite(generated, tmp_path):
    output = tmp_path / "dataset"
    generated.save(output)
    loaded, metadata = load_raw_dataset(output / "raw.sqlite")
    assert metadata["model_sha256"] == generated.metadata["model_sha256"]
    coupons = loaded["coupons"].set_index("coupon_id")
    highest_unit = loaded["order_items"].groupby("order_id").unit_price_minor.max()
    for order in loaded["orders"].itertuples(index=False):
        expected_discount = (discount_minor(highest_unit[order.order_id], coupons.loc[order.coupon_id, "discount_rate"])
                             if pd.notna(order.coupon_id) else 0)
        assert order.discount_minor == expected_discount
        assert order.paid_minor == order.gross_minor - expected_discount
    before = (output / "raw.sqlite").read_bytes()
    with pytest.raises(FileExistsError):
        generated.save(output)
    assert (output / "raw.sqlite").read_bytes() == before


def test_full_horizon_matures_control_clv_and_logs_lifecycle_exit(source_config):
    result = EcommerceSimulator(copy.deepcopy(source_config)).generate(n_customers=20, months=24, seed=17)
    validate_raw_dataset({name: result.tables[name] for name in RAW_COLUMNS}, result.metadata)
    labels = result.tables["labels"].merge(result.tables["treatments"][["customer_id", "treatment"]], on="customer_id")
    assert labels.observation_end_at.ge(labels.clv_outcome_end_at).all()
    assert labels.loc[labels.treatment.eq(0), "future_revenue_12m"].notna().all()
    assert labels.loc[labels.treatment.eq(1), "future_revenue_12m"].isna().all()
    lifecycle = result.tables["lifecycle"]
    assert {"ENTRY", "HOME", "EXIT"} == set(lifecycle.life_cycle)
    exits = lifecycle.loc[lifecycle.life_cycle.eq("EXIT")]
    assert exits.session_id.isna().all()
    assert (exits.as_of_at - exits.last_session_started_at).eq(pd.Timedelta(days=28)).all()
    for customer_id, history in lifecycle.groupby("customer_id"):
        session_states = history.loc[history.session_id.notna()].sort_values("as_of_at").life_cycle.tolist()
        assert session_states[0] == "ENTRY"
        assert all(state == "HOME" for state in session_states[1:])
    assert result.metadata["lifecycle_contract"]["inactivity_days"] == 28
    diagnostics = result.tables["session_diagnostics"].merge(
        result.tables["sessions"][["session_id", "ended_at"]], on="session_id", validate="one_to_one")
    assert diagnostics.as_of_at.eq(diagnostics.ended_at).all()
    end_states = lifecycle.groupby("customer_id").life_cycle.last()
    assert result.tables["diagnostics"].set_index("customer_id").life_cycle_end.to_dict() == end_states.to_dict()


@pytest.mark.parametrize("arguments,match", [
    ({"distribution": "toy"}, "source distribution"),
    ({"mode": "unsupported"}, "small/full"),
    ({"n_customers": 3}, "even customer"),
    ({"months": 2}, "three calendar months"),
])
def test_invalid_generation_contract_is_rejected(source_config, arguments, match):
    kwargs = dict(n_customers=40, months=3, seed=17)
    kwargs.update(arguments)
    with pytest.raises(ValueError, match=match):
        EcommerceSimulator(copy.deepcopy(source_config)).generate(**kwargs)


def test_coupon_experiment_rejects_a_different_discount(source_config):
    config = copy.deepcopy(source_config)
    config["source_simulation"]["coupon_discount"] = 0.2
    with pytest.raises(ValueError, match="10%"):
        EcommerceSimulator(config).generate(n_customers=40, months=3, seed=17)


def test_choice_context_category_follows_last_actual_item_including_remove(source_config, monkeypatch):
    import simulate.v4.irl.fit_shopping as shopping

    calls = []
    def policy(model, context):
        calls.append(context.copy())
        action = {1: "purchase", 2: "cart", 3: "view", 4: "cart", 5: "remove"}.get(
            context["session_event_count"], "end")
        return {name: float(name == action) for name in shopping.ACTIONS}
    monkeypatch.setattr(shopping, "choice_probabilities", policy)
    customers = pd.DataFrame({"customer_id": [f"fixture-{i}" for i in range(4)],
                              "signed_up_at": pd.Timestamp("2024-01-10", tz="UTC")})
    result = EcommerceSimulator(copy.deepcopy(source_config)).generate(months=3, seed=17, customers=customers)
    tables, t0 = result.tables, pd.Timestamp(result.metadata["as_of_at"])
    sessions = tables["sessions"].assign(future=lambda frame: frame.started_at.ge(t0),
                                         ordinal=lambda frame: frame.session_id.str.rsplit("S", n=1).str[-1].astype(int))
    # Generation visits every customer's history first, then each customer's future.
    sessions = sessions.sort_values(["future", "customer_id", "ordinal"])
    categories = tables["products"].set_index("product_id").category_id.to_dict()
    orders = tables["orders"].set_index("purchase_event_id")
    items = {key: group.sort_values("line_no") for key, group in tables["order_items"].groupby("order_id")}
    purchased, totals, cursor, other_category_removals = {}, Counter(), 0, 0
    for session in sessions.itertuples(index=False):
        counts = purchased.setdefault(session.customer_id, Counter())
        offer_category = None
        events = tables["events"].loc[tables["events"].session_id.eq(session.session_id) &
                                      tables["events"].event_type.ne("coupon_usage")].sort_values("event_id")
        for depth, event in enumerate(events.itertuples(index=False), 1):
            category = categories.get(event.product_id)
            if event.event_type == "purchase":
                for line in items[orders.loc[event.event_id, "order_id"]].itertuples(index=False):
                    category = categories[line.product_id]
                    counts[category] += line.quantity
                    totals[session.customer_id] += line.quantity
            elif event.event_type == "view":
                offer_category = category
            elif event.event_type == "remove_from_cart" and category != offer_category:
                other_category_removals += 1
            context = calls[cursor]
            cursor += 1
            assert context["session_event_count"] == depth
            assert context["category_purchase_share"] == pytest.approx(
                counts[category] / max(1, totals[session.customer_id]))
    assert cursor == len(calls)
    assert other_category_removals, "exercise an actual removed category different from the current offer"


def test_donor_length_stopping_is_reported_separately_from_conditional_mnl(source_config, monkeypatch):
    import simulate.v4.irl.fit_shopping as shopping

    calls = []
    def stop(model, context):
        calls.append(context)
        return {action: {"end": .99, "view": .01}.get(action, 0) for action in shopping.ACTIONS}
    monkeypatch.setattr(shopping, "choice_probabilities", stop)
    result = EcommerceSimulator(copy.deepcopy(source_config)).generate(n_customers=8, months=3, seed=17)
    assert result.tables["events"].groupby("session_id").size().eq(9).all()
    assert result.tables["events"].event_type.eq("view").all()
    sessions = result.tables["session_diagnostics"]
    assert sessions.termination_reason.eq("donor_length").all()
    assert not sessions.truncated_session.any()
    assert len(calls) >= len(sessions)


def test_event_cap_marks_truncated_donor_length(source_config, monkeypatch):
    import simulate.v4.irl.fit_shopping as shopping

    monkeypatch.setattr(shopping, "choice_probabilities", lambda model, context:
                        {action: float(action == "view") for action in shopping.ACTIONS})
    config = copy.deepcopy(source_config)
    config["source_simulation"]["max_events_per_session"] = 2
    result = EcommerceSimulator(config).generate(n_customers=8, months=3, seed=17)
    assert result.tables["events"].groupby("session_id").size().eq(2).all()
    sessions = result.tables["session_diagnostics"]
    assert sessions.termination_reason.eq("max_events").all()
    assert sessions.truncated_session.all()


def test_window_time_cap_is_censoring_not_a_learned_end(source_config, monkeypatch, tmp_path):
    import simulate.v4.irl.fit_shopping as shopping

    monkeypatch.setattr(shopping, "choice_probabilities", lambda model, context:
                        {action: float(action == "view") for action in shopping.ACTIONS})
    config = copy.deepcopy(source_config)
    artifact = json.loads(Path(config["source_simulation"]["model_path"]).read_text())
    for profile in artifact["profiles"]:
        profile["event_gap_seconds"] = [31 * 86400]
    path = tmp_path / "time-cap-profiles.json"
    path.write_text(json.dumps(artifact))
    config["source_simulation"]["model_path"] = str(path)
    result = EcommerceSimulator(config).generate(n_customers=8, months=3, seed=17)
    assert result.tables["events"].groupby("session_id").size().eq(1).all()
    sessions = result.tables["session_diagnostics"]
    assert sessions.termination_reason.eq("window_end").all()
    assert sessions.truncated_session.all()


def test_unknown_initial_cart_never_fabricates_an_add_or_natural_end(source_config, tmp_path):
    config = copy.deepcopy(source_config)
    artifact = json.loads(Path(config["source_simulation"]["model_path"]).read_text())
    for profile in artifact["profiles"]:
        profile["session_first_action_counts"] = {"remove": 1}
    path = tmp_path / "left-censored-start-profiles.json"
    path.write_text(json.dumps(artifact))
    config["source_simulation"]["model_path"] = str(path)
    result = EcommerceSimulator(config).generate(n_customers=8, months=3, seed=17)
    assert result.tables["events"].empty
    assert result.tables["orders"].empty
    sessions = result.tables["session_diagnostics"]
    assert sessions.termination_reason.eq("infeasible_initial_action").all()
    assert sessions.truncated_session.all()


def test_saved_sidecars_are_bound_without_mutating_generation_metadata(generated, tmp_path):
    before = copy.deepcopy(generated.metadata)
    output = tmp_path / "bound-dataset"
    generated.save(output)
    manifest = json.loads((output / "manifest.json").read_text())
    expected = {name for name in generated.tables if name not in RAW_COLUMNS}
    assert set(manifest["sidecar_sha256"]) == expected
    for name, digest in manifest["sidecar_sha256"].items():
        assert hashlib.sha256((output / f"{name}.parquet").read_bytes()).hexdigest() == digest
    assert generated.metadata == before
    _, metadata = load_raw_dataset(output / "raw.sqlite")
    assert "sidecar_sha256" not in metadata
