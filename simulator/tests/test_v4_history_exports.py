"""Minimum history eligibility follows observed shopping time through factual exports."""
import pandas as pd
import pytest

from simulate.v4.data import experiment_exports as exports
from simulate.v4.data.weekly import build_experiment_log, build_weekly_panel
from test_v4_source_validation import records


@pytest.fixture
def history_data(records):
    tables, metadata, labels, treatments = records
    tables.update(labels=labels, treatments=treatments, diagnostics=pd.DataFrame({
        "customer_id": ["u1", "u2"], "response_type": ["P", "ST"],
        "no_coupon_return_probability_30d": [.4, .5], "coupon_return_probability_30d": [.6, .5],
    }))
    metadata.update(schema_version=7, simulator_version="source-shopping-v2", outcome_window_days=30,
                    definition_id="future_no_session_30d_active30d")
    return tables, metadata


@pytest.mark.parametrize("seconds_late, eligible", [(0, True), (1, False)])
def test_history_boundary_is_elapsed_time_before_assignment(history_data, seconds_late, eligible):
    tables, metadata = history_data
    t0 = pd.Timestamp(metadata["as_of_at"])
    when = t0 - pd.Timedelta(days=8) + pd.Timedelta(seconds=seconds_late)
    first = tables["events"].iloc[0].copy()
    first.event_id, first.session_id, first.event_at = 11, "early", when
    tables["events"].loc[len(tables["events"])] = first
    tables["sessions"].loc[len(tables["sessions"])] = dict(
        session_id="early", customer_id="u1", started_at=when, ended_at=when + pd.Timedelta(minutes=1))
    metadata["min_customer_history_days"] = 8
    log = build_experiment_log(tables, metadata).set_index("customer_id")
    assert bool(log.loc["u1", "eligible_at_assignment"]) == eligible
    assert not log.loc["u2", "eligible_at_assignment"]
    assert log.eligibility_definition.str.contains("as_of_at-8d", regex=False).all()
    panel = build_weekly_panel(tables, metadata).set_index("customer_id")
    assert panel.loc["u1", "eligible_at_assignment"].eq(eligible).all()


def test_auxiliary_and_future_events_do_not_establish_history(history_data):
    tables, metadata = history_data
    metadata["min_customer_history_days"] = 1
    event = tables["events"].iloc[0].copy()
    event.customer_id, event.session_id = "u2", None
    for identifier, kind, when in ((11, "review", pd.Timestamp(metadata["as_of_at"]) - pd.Timedelta(days=8)),
                                   (12, "coupon_usage", pd.Timestamp(metadata["as_of_at"]) - pd.Timedelta(days=7)),
                                   (13, "view", pd.Timestamp(metadata["as_of_at"]))):
        event.event_id, event.event_type, event.event_at = identifier, kind, when
        tables["events"].loc[len(tables["events"])] = event
    log = build_experiment_log(tables, metadata).set_index("customer_id")
    assert not log.loc["u2", "eligible_at_assignment"]
    assert log.loc["u2", "treatment"] == 0


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "30", None, 61, 10**100])
def test_invalid_history_metadata_is_rejected(history_data, value):
    tables, metadata = history_data
    with pytest.raises(ValueError, match="min_customer_history_days"):
        build_experiment_log(tables, {**metadata, "min_customer_history_days": value})


def test_zero_history_preserves_legacy_export_and_cache(history_data, tmp_path):
    tables, metadata = history_data
    previous = build_experiment_log(tables, metadata)
    pd.testing.assert_frame_equal(previous, build_experiment_log(tables, {**metadata, "min_customer_history_days": 0}))
    output, private = tmp_path / "facts", tmp_path / "evaluation_only" / "facts"
    exports.export_tables(tables, metadata, output, private, repetitions=2)
    assert exports.export_tables(tables, {**metadata, "min_customer_history_days": 0}, output, private, repetitions=2)["reused"]
    with pytest.raises(FileExistsError, match="another dataset"):
        exports.export_tables(tables, {**metadata, "min_customer_history_days": 1}, output, private, repetitions=2)
    with pytest.raises(ValueError, match="min_customer_history_days"):
        exports.export_tables(tables, {**metadata, "min_customer_history_days": False}, output, private, repetitions=2)


def test_exports_record_history_and_share_factual_eligibility(history_data, tmp_path):
    tables, metadata = history_data
    metadata["min_customer_history_days"] = 1
    event = tables["events"].iloc[0].copy()
    event.event_id, event.event_at = 11, pd.Timestamp(metadata["as_of_at"]) - pd.Timedelta(days=1)
    tables["events"].loc[len(tables["events"])] = event
    output, private = tmp_path / "facts", tmp_path / "evaluation_only" / "facts"
    report = exports.export_tables(tables, metadata, output, private, repetitions=2)
    assert report["fingerprint"]["min_customer_history_days"] == 1
    assert report["analysis_metadata"]["min_customer_history_days"] == 1
    log = pd.read_parquet(output / "experiment_log.parquet").set_index("customer_id")
    labels = pd.read_parquet(output / "no_purchase_labels.parquet").set_index("customer_id")
    assert log.loc["u1", "eligible_at_assignment"] and not log.loc["u2", "eligible_at_assignment"]
    assert labels.eligible_at_assignment.equals(log.eligible_at_assignment)
