"""New generation has six event types; legacy search/CS settings are ignored."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from test_v4_checkout_simulator import checkout_fixture, _customers
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS, load_raw_dataset, validate_raw_dataset
from simulate.v4.simulator.review_events import ORDER_FEATURE_NAMES, REVIEW_FEATURE_NAMES, review_context


SHOPPING = {"view", "cart", "remove_from_cart", "purchase"}
CORE = SHOPPING | {"coupon_usage"}
CUSTOMER_COUNT = 20


def _parameters(names, classes, intercept):
    width = len(names)
    return dict(feature_names=list(names), classes=classes,
                coef=[[0.] * width for _ in intercept], intercept=intercept,
                scaler_mean=[0.] * width, scaler_scale=[1.] * width,
                feature_min=[-100.] * width, feature_max=[100.] * width)


@pytest.fixture(scope="module")
def event_config(checkout_fixture, tmp_path_factory):
    config, shopping = checkout_fixture
    config = copy.deepcopy(config)
    folder = tmp_path_factory.mktemp("auxiliary-event-models")
    review = dict(schema_version=1, type="source_conditional_review_events",
                  product_stars=dict(model=_parameters(REVIEW_FEATURE_NAMES, [1, 2, 3, 4, 5], [0.] * 5)),
                  order_survey=dict(model=_parameters(ORDER_FEATURE_NAMES, [0, 1], [700.]), window_days=28,
                                    delay_score_banks=dict(pooled=[[60., 5]],
                                                          first_observed_order=[[60., 5]],
                                                          repeat_observed_order=[[60., 5]])))
    path = folder / "review.json"
    path.write_text(json.dumps(review, allow_nan=False))
    config["source_simulation"]["review_model_path"] = str(path)
    for name in ("search", "cs"):
        config["source_simulation"][f"{name}_model_path"] = str(folder / f"ignored-{name}.json")
    return config, shopping


def _generate(config):
    return EcommerceSimulator(copy.deepcopy(config)).generate(months=3, seed=17, customers=_customers(CUSTOMER_COUNT))


@pytest.fixture(scope="module")
def event_runs(event_config):
    enabled_config, _ = event_config
    disabled_config = copy.deepcopy(enabled_config)
    disabled_config["source_simulation"]["review_model_path"] = None
    return _generate(disabled_config), _generate(enabled_config)


def _core_tables(result):
    events = result.tables["events"].loc[result.tables["events"].event_type.isin(CORE)].copy()
    mapping = dict(zip(events.event_id, range(1, len(events) + 1)))
    events["event_id"] = events.event_id.map(mapping)
    events["related_purchase_event_id"] = events.related_purchase_event_id.map(mapping)
    orders = result.tables["orders"].copy()
    orders["purchase_event_id"] = orders.purchase_event_id.map(mapping)
    frames = {**{name: result.tables[name] for name in ("customers", "products", "sessions", "order_items", "coupons", "labels", "treatments", "lifecycle")},
              "events": events.reset_index(drop=True), "orders": orders}
    return {name: frame.astype(object).where(frame.notna(), None) for name, frame in frames.items()}


def test_reviews_preserve_core_visits_choices_money_and_labels(event_runs):
    disabled, enabled = event_runs
    expected, actual = _core_tables(disabled), _core_tables(enabled)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name], check_dtype=False, obj=name)
    assert set(disabled.tables["events"].event_type) == CORE
    assert set(enabled.tables["events"].event_type) == CORE | {"review"}
    assert set(enabled.metadata["event_models"]) == {"review"}
    assert enabled.metadata["event_scope_version"] == "shopping_coupon_review_v1"
    assert not {"search_generation", "cs_generation"} & set(enabled.metadata)
    assert enabled.metadata["currency"] == disabled.metadata["currency"]
    assert enabled.metadata["amount_unit"] == disabled.metadata["amount_unit"]


@pytest.mark.parametrize("existing", [False, True])
def test_removed_search_and_cs_custom_artifacts_are_not_read(event_config, tmp_path, monkeypatch, existing):
    config, _ = event_config
    config = copy.deepcopy(config)
    ignored = set()
    for name in ("search", "cs"):
        path = tmp_path / f"custom-{name}.json"
        if existing:
            path.write_text("not a model or valid JSON")
        ignored.add(path.resolve())
        config["source_simulation"][f"{name}_model_path"] = str(path)
    original_bytes, original_text = Path.read_bytes, Path.read_text
    def read_bytes(path, *args, **kwargs):
        assert path.resolve() not in ignored, "removed model path must never be read"
        return original_bytes(path, *args, **kwargs)
    def read_text(path, *args, **kwargs):
        assert path.resolve() not in ignored, "removed model path must never be read"
        return original_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    result = _generate(config)
    assert set(result.tables["events"].event_type) == CORE | {"review"}
    assert set(result.metadata["event_models"]) == {"review"}
    assert "cs_diagnostics" not in result.tables


def test_removed_search_and_cs_generators_are_never_called(event_config, monkeypatch):
    import simulate.v4.simulator.engine as engine
    real_import = engine.importlib.import_module
    def guarded_import(name, *args, **kwargs):
        assert name not in {"simulate.v4.simulator.search_events", "simulate.v4.simulator.cs_events"}
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(engine.importlib, "import_module", guarded_import)
    result = _generate(event_config[0])
    assert not result.tables["events"].event_type.isin(["search", "cs_contact"]).any()
    assert "cs_diagnostics" not in result.tables


def test_async_reviews_link_actual_purchased_products_without_inventing_sessions(event_runs):
    _, result = event_runs
    events = result.tables["events"].set_index("event_id")
    reviews = events.loc[events.event_type.eq("review")]
    assert len(reviews) > 0 and not reviews.duplicated(["related_purchase_event_id", "product_id"]).any()
    purchases = events.loc[reviews.related_purchase_event_id.astype(int)]
    assert purchases.event_type.eq("purchase").all()
    assert purchases.customer_id.tolist() == reviews.customer_id.tolist()
    assert (reviews.event_at.to_numpy() - purchases.event_at.to_numpy() == pd.Timedelta(seconds=60)).all()
    assert reviews.rating.between(1, 5).all() and reviews.rating_kind.eq("product_star_scenario").all()
    assert reviews[["session_id", "quantity", "unit_price_minor", "coupon_id", "discount_minor", "contact_id"]].isna().all().all()
    items = result.tables["order_items"].merge(result.tables["orders"][["order_id", "purchase_event_id"]], on="order_id")
    linked = reviews.reset_index().merge(items, left_on=["related_purchase_event_id", "order_line_no"],
                                        right_on=["purchase_event_id", "line_no"], suffixes=("_review", "_item"), validate="one_to_one")
    assert len(linked) == len(reviews) and linked.product_id_review.eq(linked.product_id_item).all()
    assert reviews.event_at.lt(pd.Timestamp(result.metadata["observation_end_at"])).all()


def test_review_scope_is_distinct_purchased_products_and_survey_context_counts_orders(event_config, tmp_path, monkeypatch):
    import simulate.v4.irl.fit_shopping as shopping
    import simulate.v4.simulator.review_events as module
    from simulate.v4.irl.fit_shopping import ACTIONS
    config, model = copy.deepcopy(event_config)
    for holder in [model["distributions"], *model["home_groups"].values(), *model["profiles"]]:
        for template in holder.get("checkout_templates", []):
            template["items"] = [dict(product_id="A", quantity=3), dict(product_id="B", quantity=1)]
    path = tmp_path / "multi-unit-checkout.json"
    path.write_text(json.dumps(model))
    config["source_simulation"]["model_path"] = str(path)
    monkeypatch.setattr(shopping, "choice_probabilities", lambda model, context:
                        {action: float(action == "purchase") for action in ACTIONS})
    original, calls = module.order_review_context, []
    def context(orders, reviews, at):
        assert len(reviews) == len({row["purchase_event_id"] for row in reviews})
        result = original(orders, reviews, at)
        calls.append(result)
        assert result["prior_review_count"] <= result["prior_order_count"]
        return result
    monkeypatch.setattr(module, "order_review_context", context)
    result = _generate(config)
    reviews = result.tables["events"].loc[lambda f: f.event_type.eq("review")]
    details = result.tables["review_diagnostics"]
    assert len(calls) == len(result.tables["orders"]) and any(c["prior_review_count"] > 0 for c in calls)
    assert len(details) == 2 * len(result.tables["orders"])
    assert not details.duplicated(["purchase_event_id", "product_id"]).any()
    assert details.groupby("purchase_event_id").scheduled.nunique().eq(1).all()
    assert details.groupby("purchase_event_id").scheduled_at.nunique(dropna=False).eq(1).all()
    assert not reviews.empty and reviews.groupby("related_purchase_event_id").size().eq(2).all()
    assert reviews.groupby("related_purchase_event_id").product_id.agg(set).map(lambda s: s == {"A", "B"}).all()
    assert reviews.quantity.isna().all() and reviews.order_line_no.isin([1, 2]).all()
    assert reviews.duplicated(["customer_id", "product_id"]).any()  # A later purchase may review the same SKU again.
    validate_raw_dataset({name: result.tables[name] for name in RAW_COLUMNS}, result.metadata)


def test_rating_context_uses_reviews_already_arrived_not_pending_scores(event_config, monkeypatch):
    import simulate.v4.simulator.review_events as module
    contexts = []
    def rating(model, context, rng):
        contexts.append(context.copy())
        return 1 + context["prior_review_count"] % 5
    monkeypatch.setattr(module, "sample_rating", rating)
    config, _ = event_config
    config = copy.deepcopy(config)
    for name in ("search", "cs"):
        config["source_simulation"][f"{name}_model_path"] = None
    result = _generate(config)
    reviews = result.tables["events"].loc[result.tables["events"].event_type.eq("review")]
    expected = []
    for _, local in reviews.groupby("customer_id", sort=True):
        history = []
        for event in local.sort_values(["event_at", "event_id"]).itertuples(index=False):
            context = review_context(history, event.event_at)
            expected.append(context)
            assert event.rating == 1 + context["prior_review_count"] % 5
            history.append(dict(event_at=event.event_at, rating=int(event.rating)))
    assert any(context["prior_review_count"] > 0 for context in expected)
    # Enrollment histories are generated for all shoppers before their futures.
    assert sorted(json.dumps(item, sort_keys=True) for item in contexts) == sorted(
        json.dumps(item, sort_keys=True) for item in expected)


def test_reviews_at_observation_end_remain_censored(event_config, monkeypatch):
    import simulate.v4.simulator.review_events as module
    config, _ = event_config
    config = copy.deepcopy(config)
    config["source_simulation"]["search_model_path"] = None
    config["source_simulation"]["cs_model_path"] = None
    end, latest = pd.Timestamp("2024-04-01T00:00:00Z"), {}
    original = module.order_review_context
    def context(orders, reviews, at):
        latest["at"] = at
        return original(orders, reviews, at)
    def schedule(model, context, rng):
        # Integer-second delays avoid rounding a boundary just inside the window.
        delay = ((end - latest["at"]).value + 999_999_999) // 1_000_000_000
        selected = 0 <= delay < 28 * 86400
        return dict(scheduled=selected, delay_seconds=delay if selected else None,
                    source_order_score=5 if selected else None, response_probability=1.,
                    source_cohort="first_observed_order" if context["prior_order_count"] == 0 else "repeat_observed_order")
    monkeypatch.setattr(module, "order_review_context", context)
    monkeypatch.setattr(module, "sample_review_schedule", schedule)
    result = _generate(config)
    pending = result.tables["review_diagnostics"].loc[lambda f: f.scheduled]
    assert not pending.empty and pending.scheduled_at.ge(end).all()
    assert pending.scheduled_at.eq(end).any()
    assert pending.status.eq("censored").all() and not pending.emitted.any()
    assert pending.review_event_id.isna().all() and pending.rating.isna().all()
    assert not result.tables["events"].event_type.eq("review").any()


def test_no_purchases_keep_review_diagnostics_empty_and_emit_no_cs(event_config, monkeypatch):
    import simulate.v4.irl.fit_shopping as shopping
    from simulate.v4.irl.fit_shopping import ACTIONS
    config, _ = event_config
    monkeypatch.setattr(shopping, "choice_probabilities", lambda model, context: {action: float(action == "view") for action in ACTIONS})
    result = _generate(config)
    assert result.tables["orders"].empty
    assert result.tables["review_diagnostics"].empty and {
        "customer_id", "order_id", "purchase_event_id", "scheduled", "scheduled_at", "emitted", "rating", "status"
    } <= set(result.tables["review_diagnostics"])
    assert not result.tables["events"].event_type.isin(["review", "search", "cs_contact"]).any()
    assert "cs_diagnostics" not in result.tables


def test_raw5_six_event_types_roundtrip_and_reproducibility(event_config, event_runs, tmp_path):
    _, result = event_runs
    again = _generate(event_config[0])
    for name in result.tables:
        pd.testing.assert_frame_equal(result.tables[name], again.tables[name], obj=name)
    assert len(RAW_COLUMNS["events"]) == 17
    assert result.metadata["raw_schema_version"] == 5
    assert result.metadata["simulator_version"] == "source-shopping-v7"
    assert set(result.tables["events"].event_type) == set(result.metadata["event_types"])
    assert set(result.metadata["event_types"]) == CORE | {"review"}
    validate_raw_dataset({name: result.tables[name] for name in RAW_COLUMNS}, result.metadata)
    result.save(tmp_path / "dataset")
    loaded, metadata = load_raw_dataset(tmp_path / "dataset" / "raw.sqlite")
    assert metadata["raw_schema_version"] == 5
    assert len(loaded["events"].columns) == 17
    for name in RAW_COLUMNS:
        pd.testing.assert_frame_equal(loaded[name], result.tables[name], check_dtype=False, obj=name)
    from simulate.v4.simulator.paired_evaluation import paired_retention_rollouts
    pairs, _ = paired_retention_rollouts(result.tables["diagnostics"], metadata, repetitions=1)
    assert pairs.simulator_version.eq("source-shopping-v7").all()


def test_opt_out_needs_no_auxiliary_artifacts(checkout_fixture, tmp_path):
    config, _ = checkout_fixture
    config = copy.deepcopy(config)
    config["source_simulation"]["review_model_path"] = None
    for name in ("search", "cs"):
        path = tmp_path / f"missing-{name}.json"
        assert not path.exists()
        config["source_simulation"][f"{name}_model_path"] = str(path)
    result = _generate(config)
    assert result.metadata["event_models"] == {}
    assert not ({"search", "review", "cs_contact"} & set(result.tables["events"].event_type))
    assert "review_diagnostics" not in result.tables and "cs_diagnostics" not in result.tables
