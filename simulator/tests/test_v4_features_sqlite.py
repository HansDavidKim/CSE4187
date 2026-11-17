"""Canonical SQLite, past-only KIT39 and isolated factual export contracts."""
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from simulate.v4.features import EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION, build_features, load_tables
from simulate.v4.simulator.raw_store import RAW_COLUMNS, save_raw_dataset
from simulate.v4.storage import FEATURE_KEYS, materialize_features


T0 = pd.Timestamp("2024-03-03T15:00:00Z")


@pytest.fixture
def dataset(tmp_path):
    def at(value):
        return pd.Timestamp(value, tz="UTC")

    def event(identifier, session, when, kind, product=None, quantity=None, offer=None, **extra):
        return dict(event_id=identifier, customer_id="u", session_id=session, event_at=at(when), event_type=kind,
                    product_id=product, quantity=quantity, unit_price_minor={"A": 100, "B": 200}.get(product) if kind != "review" else None,
                    offer_product_id=offer, offer_unit_price_minor={"A": 100, "B": 200}.get(offer), **extra)

    records = {
        "customers": [dict(customer_id=uid, signed_up_at=at("2024-01-01")) for uid in ("u", "v")],
        "products": [dict(product_id=pid, category_id="beauty", unit_price_minor=price) for pid, price in (("A", 100), ("B", 200))],
        "sessions": [dict(session_id=sid, customer_id="u", started_at=at(f"2024-02-{day} 10:00"),
                          ended_at=at(f"2024-02-{day} 10:10")) for sid, day in (("buy", "10"), ("edit", "20"))],
        "events": [event(1, "buy", "2024-02-10 10:00", "view", "A"),
                   event(2, "buy", "2024-02-10 10:01", "cart", "A", 1),
                   event(3, "buy", "2024-02-10 10:02", "view", "B"),
                   event(4, "buy", "2024-02-10 10:03", "cart", "B", 1),
                   event(5, "buy", "2024-02-10 10:04", "purchase", offer="B"),
                   event(6, None, "2024-02-11 10:00", "review", "A", related_purchase_event_id=5, order_line_no=1,
                         rating=1, rating_kind="product_star_scenario"),
                   event(7, "edit", "2024-02-20 10:00", "cart", "A", 1),
                   event(8, None, "2024-03-03 15:00", "review", "B", related_purchase_event_id=5, order_line_no=2,
                         rating=5, rating_kind="product_star_scenario")],
        "orders": [dict(order_id="o", purchase_event_id=5, checkout_source="cart", coupon_id=None,
                        gross_minor=300, discount_minor=0, paid_minor=300)],
        "order_items": [dict(order_id="o", line_no=line, product_id=pid, quantity=1, unit_price_minor=price)
                        for line, pid, price in ((1, "A", 100), (2, "B", 200))],
        "coupons": [dict(coupon_id="coupon", customer_id="u", sent_at=T0, expires_at=T0 + pd.Timedelta(days=30),
                         discount_rate=.1, policy_id="once", send_cost_minor=7)],
    }
    tables = {name: pd.DataFrame(records[name], columns=columns) for name, columns in RAW_COLUMNS.items()}
    metadata = dict(dataset_id="v4-contract", seed=3, currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
                    raw_schema_version=5, observation_start_at=at("2024-01-01").isoformat(),
                    observation_end_at=at("2024-04-15").isoformat(), as_of_at=T0.isoformat(), outcome_window_days=28)
    directory = tmp_path / "dataset"
    directory.mkdir()
    save_raw_dataset(directory / "raw.sqlite", tables, metadata)
    return directory


def add_sidecars(directory):
    pd.DataFrame([dict(customer_id=uid, as_of_at=T0, assigned_at=T0, treatment=value, assignment_probability=.5,
                       experiment_id="fixed", coupon_id="coupon" if value else None, actual_sent=bool(value),
                       send_cost_minor=7 if value else 0) for uid, value in (("u", 1), ("v", 0))]).to_parquet(directory / "treatments.parquet", index=False)
    pd.DataFrame([dict(customer_id=uid, as_of_at=T0, snapshot_id=f"{uid}|w0", outcome_end_at=T0 + pd.Timedelta(days=28),
                       churn=1, definition_id="future_no_session_28d_active30d", is_censored=False) for uid in ("u", "v")]).to_parquet(directory / "labels.parquet", index=False)


def test_registered_features_use_strict_event_prefix_and_product_reviews(dataset):
    tables = load_tables(dataset / "raw.sqlite")
    row = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
    assert len(EXPANDED_FEATURES) == 39 and len(set(EXPANDED_FEATURES)) == 39
    assert row.expanded_feature_set_version == EXPANDED_FEATURE_VERSION == "kit39-review-cart-v2"
    assert row.review_count_30d == 1 and row.mean_review_rating_30d == 1
    assert row.review_last_order_mean_rating == 1 and row.review_non5_share_30d == 1
    assert row.cart_adds_per_nonpurchase_session_30d == 1 and row.cart_removes_per_nonpurchase_session_30d == 0
    assert row.cart_qty_current == 1 and row.cart_gross_value_current == 100
    assert row.purchase_count_30d == 1 and row.net_spend_30d == 300
    later = build_features(tables, as_of_at=T0 + pd.Timedelta(nanoseconds=1)).set_index("customer_id").loc["u"]
    assert later.review_count_30d == 2 and later.review_last_order_mean_rating == 3
    assert later.review_non5_share_30d == .5


def test_bare_sqlite_materializes_only_known_facts_without_inventing_assignments(dataset, tmp_path):
    before = hashlib.sha256((dataset / "raw.sqlite").read_bytes()).hexdigest()
    output = tmp_path / "features"
    manifest = materialize_features(dataset / "raw.sqlite", output)
    assert manifest["rows"] == {"features": 2}
    assert set(manifest["omitted"]) == {"weekly_panel", "experiment_log"}
    with sqlite3.connect(output / "features.sqlite") as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(features)")]
        assert columns == [*FEATURE_KEYS, *EXPANDED_FEATURES]
        assert not {"treatment", "churn", "persona", "latent_component", "response_type"} & set(columns)
        assert connection.execute("SELECT count(*) FROM feature_registry").fetchone()[0] == 39
        assert connection.execute("SELECT cart_qty_current,net_spend_30d FROM features WHERE customer_id='u'").fetchone() == (1, 300)
        assert connection.execute("SELECT mean_review_rating_30d FROM features WHERE customer_id='v'").fetchone()[0] is None
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert before == hashlib.sha256((dataset / "raw.sqlite").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="new or empty"):
        materialize_features(dataset, output)
    assert before == hashlib.sha256((dataset / "raw.sqlite").read_bytes()).hexdigest()


def test_sidecar_experiment_coverage_and_explicit_new_snapshot(dataset, tmp_path):
    add_sidecars(dataset)
    output = tmp_path / "materialized"
    materialize_features(dataset, output, as_of_at=T0 + pd.Timedelta(days=1))
    features = pd.read_parquet(output / "feature_store.parquet")
    assert features.as_of_at.eq(T0 + pd.Timedelta(days=1)).all()
    assert features.set_index("customer_id").loc["u", "review_count_30d"] == 2
    assert not {"churn", "treatment"} & set(features)
    with sqlite3.connect(output / "features.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM labels").fetchone()[0] == 2
        assert connection.execute("SELECT actual_sent,redeemed FROM experiment_log WHERE customer_id='u'").fetchone() == (1, 0)
        assert connection.execute("SELECT actual_sent,redeemed FROM experiment_log WHERE customer_id='v'").fetchone() == (0, 0)
    weekly = pd.read_parquet(output / "weekly_panel.parquet")
    assert not {"searches", "cs_contacts"} & set(weekly)
    assert str(weekly.reference_week.dt.tz) == "Asia/Seoul"
    assert weekly.loc[weekly.customer_id.eq("v"), "sessions"].eq(0).all()
    assert weekly.purchases.sum() == 1 and weekly.coupons_sent.sum() == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["omitted"] == {} and set(manifest["file_sha256"]) >= {"features.sqlite", "feature_store.parquet"}


def test_experiment_exports_use_declared_eligibility_and_coupon_validity(dataset):
    from simulate.v4.data.experiment_exports import build_no_purchase_labels
    from simulate.v4.data.weekly import build_experiment_log

    add_sidecars(dataset)
    tables = load_tables(dataset)
    metadata = {**tables["_metadata"], "definition_id": "future_no_session_28d_active30d"}
    start = T0 - pd.Timedelta(days=29)
    tables["sessions"].loc[len(tables["sessions"])] = dict(
        session_id="old-only", customer_id="v", started_at=start, ended_at=start + pd.Timedelta(minutes=1),
    )
    legacy = build_experiment_log(tables, metadata)
    assert legacy.eligible_at_assignment.all()
    assert build_no_purchase_labels(tables, metadata, legacy).definition_id.eq("future_no_order_28d_active30d").all()
    metadata.update(eligibility_window_days=28, coupon_validity_days=28, definition_id="future_no_session_28d_active28d")
    tables["labels"]["definition_id"] = metadata["definition_id"]
    with pytest.raises(ValueError, match="exactly 28 days"):
        build_experiment_log(tables, metadata)
    tables["coupons"]["expires_at"] = T0 + pd.Timedelta(days=28)
    current = build_experiment_log(tables, metadata)
    assert current.set_index("customer_id").eligible_at_assignment.to_dict() == {"u": True, "v": False}
    labels = build_no_purchase_labels(tables, metadata, current)
    assert labels.definition_id.eq("future_no_order_28d_active28d").all()
    assert labels.eligibility_window_days.eq(28).all()
    assert labels.no_purchase_churn.eq(1).all()


def test_retention_reads_kit39_and_mature_labels_at_exact_snapshot(dataset, tmp_path):
    from simulate.v4.data.loader import load_retention_inputs

    add_sidecars(dataset)
    (dataset / "treatments.parquet").unlink()
    label_path = dataset / "labels.parquet"
    labels = pd.read_parquet(label_path)
    labels["definition_id"] = "future_no_session_28d_active28d"
    labels["outcome_window_days"] = 28
    labels["eligibility_window_days"] = 28
    labels["observation_end_at"] = pd.Timestamp("2024-04-15", tz="UTC")
    labels["policy_id"] = ["one_coupon_10pct_highest_unit_28d", "no_new_coupon"]
    labels.loc[labels.customer_id.eq("v"), "is_censored"] = True
    labels.loc[labels.customer_id.eq("v"), "churn"] = pd.NA
    labels.to_parquet(label_path, index=False)

    output = tmp_path / "retention"
    materialize_features(dataset, output)
    snapshots, x, y = load_retention_inputs(output)
    assert snapshots.customer_id.tolist() == ["u"] and snapshots.as_of_at.eq(T0).all()
    assert y.tolist() == [1]
    assert tuple(x.columns) == EXPANDED_FEATURES
    expected = pd.read_parquet(output / "feature_store.parquet", columns=["customer_id", *EXPANDED_FEATURES])
    pd.testing.assert_frame_equal(x, expected.loc[expected.customer_id.eq("u"), list(EXPANDED_FEATURES)].reset_index(drop=True))

    later = tmp_path / "later"
    materialize_features(dataset, later, as_of_at=T0 + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="same snapshot"):
        load_retention_inputs(later)
    # The rows are structurally valid but no longer the manifest-bound label file.
    labels.assign(churn=0).to_parquet(output / "labels.parquet", index=False)
    with pytest.raises(ValueError, match="differs from its manifest"):
        load_retention_inputs(output)


def test_uplift_reader_aligns_assignment_inputs_and_preserves_censoring(dataset, tmp_path):
    from simulate.v4.data.loader import load_uplift_inputs

    add_sidecars(dataset)
    features = tmp_path / "features"
    materialize_features(dataset, features)
    experiment_path = features / "experiment_log.parquet"
    experiment = pd.read_parquet(experiment_path)
    experiment.loc[experiment.customer_id.eq("v"), "outcome_observed"] = False
    experiment.to_parquet(experiment_path, index=False)

    customers, weekly, cart = load_uplift_inputs(features)
    assert set(customers) == {"customer_id", "w0", "treatment"}
    assert customers.w0.eq(T0.tz_convert("Asia/Seoul")).all()
    assert set(weekly) == {"customer_id", "reference_week", "sessions", "purchases", "net_revenue", "views", "session_minutes"}
    assert weekly.reference_week.ge(T0 - pd.Timedelta(weeks=12)).all()
    assert weekly.reference_week.lt(T0 + pd.Timedelta(weeks=4)).all()
    assert weekly.net_revenue.sum() == 300 and weekly.session_minutes.sum() == 20
    assert pd.api.types.is_integer_dtype(weekly.net_revenue)
    assert weekly.sessions.dtype == float and weekly.purchases.dtype == float
    future = weekly.loc[weekly.reference_week.ge(T0)]
    assert future.loc[future.customer_id.eq("u"), "purchases"].eq(0).all()
    assert future.loc[future.customer_id.eq("v"), "purchases"].isna().all()
    assert set(cart) == {"customer_id", "cart_qty", "cart_sku", "cart_gross_value"}
    assert cart.set_index("customer_id").loc["u", "cart_gross_value"] == 100
    assert customers.attrs == weekly.attrs == cart.attrs == {
        "dataset_id": "v4-contract", "currency": "SOURCE_UNSPECIFIED", "amount_unit": "0.01", "seed": 3,
    }
    cart_path = features / "feature_store.parquet"
    featured = pd.read_parquet(cart_path)
    featured.loc[featured.customer_id.eq("u"), "snapshot_id"] = "wrong snapshot"
    featured.to_parquet(cart_path, index=False)
    with pytest.raises(ValueError, match="share one dataset and the assignment snapshots"):
        load_uplift_inputs(features)


def test_feature_database_is_closed_before_publication(dataset, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from simulate.v4 import storage

    connections = []
    publish = storage.os.rename

    def connect(path):
        connection = sqlite3.connect(path)
        connections.append(connection)
        return connection

    def rename(staging, output):
        (connection,) = connections
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        publish(staging, output)

    monkeypatch.setattr(storage, "sqlite3", SimpleNamespace(connect=connect))
    monkeypatch.setattr(storage, "os", SimpleNamespace(rename=rename))
    output = tmp_path / "features"
    materialize_features(dataset, output)
    assert (output / "features.sqlite").is_file()


def test_failed_publication_leaves_no_partial_feature_database(dataset, tmp_path, monkeypatch):
    from simulate.v4 import storage
    output = tmp_path / "failed"
    monkeypatch.setattr(storage.pd.DataFrame, "to_parquet", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        materialize_features(dataset, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-*"))


@pytest.mark.parametrize("case", ["customer", "snapshot", "horizon"])
def test_sidecars_cannot_silently_mix_another_customer_or_snapshot(dataset, case):
    add_sidecars(dataset)
    labels = pd.read_parquet(dataset / "labels.parquet")
    if case == "customer":
        labels.loc[0, "customer_id"] = "absent"
    elif case == "snapshot":
        labels["as_of_at"] += pd.Timedelta(days=1)
    else:
        labels["outcome_end_at"] += pd.Timedelta(days=1)
    labels.to_parquet(dataset / "labels.parquet", index=False)
    with pytest.raises(ValueError, match="sidecar|outcome windows"):
        load_tables(dataset)


@pytest.mark.parametrize("case", ["changed", "missing"])
def test_declared_sidecar_checksums_reject_tampering_and_missing_files(dataset, case):
    add_sidecars(dataset)
    metadata = json.loads((dataset / "raw.manifest.json").read_text())["metadata"]
    metadata["sidecar_sha256"] = {name: hashlib.sha256((dataset / f"{name}.parquet").read_bytes()).hexdigest()
                                 for name in ("labels", "treatments")}
    (dataset / "manifest.json").write_text(json.dumps(metadata))
    assert len(load_tables(dataset)["labels"]) == 2
    if case == "changed":
        labels = pd.read_parquet(dataset / "labels.parquet")
        labels["churn"] = 0
        labels.to_parquet(dataset / "labels.parquet", index=False)
    else:
        (dataset / "labels.parquet").unlink()
    with pytest.raises(ValueError, match="sidecar checksum"):
        load_tables(dataset)


@pytest.mark.parametrize("case", ["changed", "missing"])
@pytest.mark.parametrize("raw_path", [False, True])
def test_cli_validate_checks_declared_diagnostics_before_reporting_valid(dataset, capsys, case, raw_path):
    from types import SimpleNamespace
    from simulate.v4.cli import run

    diagnostics = dataset / "diagnostics.parquet"
    pd.DataFrame({"customer_id": ["u"], "latent_score": [0.25]}).to_parquet(diagnostics, index=False)
    metadata = json.loads((dataset / "raw.manifest.json").read_text())["metadata"]
    metadata["sidecar_sha256"] = {"diagnostics": hashlib.sha256(diagnostics.read_bytes()).hexdigest()}
    (dataset / "manifest.json").write_text(json.dumps(metadata))
    args = SimpleNamespace(v4_command="validate", data=dataset / "raw.sqlite" if raw_path else dataset)
    run(args)
    assert json.loads(capsys.readouterr().out)["valid"]
    if case == "changed":
        pd.DataFrame({"customer_id": ["u"], "latent_score": [0.75]}).to_parquet(diagnostics, index=False)
    else:
        diagnostics.unlink()
    with pytest.raises(ValueError, match="diagnostics sidecar checksum"):
        run(args)
    assert capsys.readouterr().out == ""


def test_user_snapshot_requires_explicit_timezone(dataset, tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        materialize_features(dataset, tmp_path / "naive", as_of_at="2024-03-04")
    assert not (tmp_path / "naive").exists()
