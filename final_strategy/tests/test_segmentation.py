"""In-memory segmentation tests; no raw SQLite or publisher fixtures."""
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import (
    BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION,
)
from final_strategy.inputs import IntegratedInputs, PredictionBatch, integrate_inputs
from final_strategy.validation import CONTEXT_COLUMNS, JOIN_KEYS, PREDICTION_COLUMNS, InputValidationError, PredictionMetadata
from final_strategy.segmentation import (
    CHURN_BANDS, UPLIFT_BANDS, CLV_BANDS, MEAN_COLUMNS, SegmentationConfig,
    export_segments, plot_segments, segment_customers,
)


T0 = pd.Timestamp("2024-03-03T15:00:00Z")


def make_inputs(records):
    """Records are customer/churn/uplift/CLV with an optional exact snapshot."""
    feature_rows, values = [], []
    for record in records:
        customer, churn, uplift, clv = record[:4]
        clock = pd.Timestamp(record[4]) if len(record) == 5 else T0
        local = clock.tz_convert("Asia/Seoul")
        row = dict.fromkeys(EXPANDED_FEATURES, 0.0)
        row.update(customer_id=customer, dataset_id="dummy-segmentation", as_of_at=clock,
                   reference_week=(local - pd.Timedelta(days=local.weekday())).date().isoformat(),
                   snapshot_id=f"opaque:{customer}:{clock.isoformat()}", currency="SOURCE_UNSPECIFIED",
                   amount_unit="0.01", raw_database_sha256="a" * 64, source_model_sha256="b" * 64,
                   feature_set_version="kit32-v1", behavior_feature_set_version=BEHAVIOR_FEATURE_VERSION,
                   expanded_feature_set_version=EXPANDED_FEATURE_VERSION)
        row["mean_net_order_value_30d"] = np.nan
        feature_rows.append(row)
        values.append((churn, uplift, clv))
    features = pd.DataFrame(feature_rows)
    declarations = {
        "churn": PredictionMetadata("dummy-churn", "dummy_no_visit_28d", 28, "days", "probability"),
        "uplift": PredictionMetadata("dummy-uplift", "dummy_purchase_retention_4w", 4, "weeks", "probability_difference"),
        "clv": PredictionMetadata("dummy-clv", "dummy_future_paid_revenue_52w", 52, "weeks", "amount"),
    }
    batches = {}
    for index, (model, column) in enumerate(PREDICTION_COLUMNS.items()):
        frame = features[list(CONTEXT_COLUMNS)].copy()
        frame["input_cutoff_at"] = frame.as_of_at
        offset = pd.Timedelta(weeks=52) if model == "clv" else pd.Timedelta(days=28)
        frame["prediction_end_at"] = frame.as_of_at.map(lambda point: point + offset)
        frame[column] = [row[index] for row in values]
        batches[model] = PredictionBatch(frame, declarations[model])
    return integrate_inputs(features, batches)


@pytest.fixture
def inputs():
    return make_inputs([("003", .7, -.2, 100), ("001", .5, .2, 100), ("002", .1, 0, 0)])


def eligibility_for(inputs, excluded):
    table = inputs.data[list(JOIN_KEYS)].copy()
    table["eligible"] = ~table.customer_id.isin(excluded)
    table["exclusion_reason"] = table.customer_id.map(excluded)
    return table


@pytest.fixture
def twelve_inputs():
    rows = []
    for risk, uplift in product((.2, .8), (-.1, 0, .1)):
        for index in range(10):
            rows.append((f"{len(rows):03}", risk, uplift, 1000 if index < 2 else 100))
    return make_inputs(rows)


def test_all_twelve_combinations_and_exactly_one_assignment(twelve_inputs):
    result = segment_customers(twelve_inputs)
    combinations = set(result.customers[["churn_band", "uplift_band", "clv_band"]].itertuples(index=False, name=None))
    assert combinations == set(product(CHURN_BANDS, UPLIFT_BANDS, CLV_BANDS))
    assert len(result.customers) == 60 and result.customers.segment_id.nunique() == 12
    assert not result.customers.duplicated(list(JOIN_KEYS)).any()
    assert result.summary.customer_count.sum() == 60
    assert result.summary.customer_ratio.sum() == pytest.approx(1)
    assert len(result.summary) == 12
    assert result.customers.segment_name.notna().all()


@pytest.mark.parametrize("value,band", [(.499, "low"), (.5, "high"), (.501, "high")])
def test_churn_threshold_boundary(value, band):
    row = segment_customers(make_inputs([("u", value, .1, 10)])).customers.iloc[0]
    assert row.churn_band == band


@pytest.mark.parametrize("value,band", [(-.001, "negative"), (0, "neutral_low"),
                                        (.049, "neutral_low"), (.05, "neutral_low"), (.051, "positive")])
def test_uplift_threshold_boundary(value, band):
    result = segment_customers(make_inputs([("u", .5, value, 10)]), SegmentationConfig(uplift_threshold=.05))
    assert result.customers.uplift_band.iloc[0] == band


def test_default_neutral_is_exact_zero():
    result = segment_customers(make_inputs([("u", .5, 1e-12, 10)]))
    assert result.customers.uplift_band.iloc[0] == "positive"


@pytest.mark.parametrize("field,value", list(product(
    ("churn_threshold", "uplift_threshold", "high_clv_fraction"),
    (-.1, 1.1, np.nan, np.inf, True, "0.5"))))
def test_invalid_thresholds(field, value):
    with pytest.raises(InputValidationError, match=field):
        SegmentationConfig(**{field: value})


def test_config_version_required():
    with pytest.raises(InputValidationError, match="segmentation_version"):
        SegmentationConfig(segmentation_version="")


def test_top_twenty_percent_and_id_tie_break():
    records = [(f"{index:03}", .5, .1, 100 if index < 3 else 1) for index in range(10)]
    result = segment_customers(make_inputs(records[::-1]))
    assert result.customers.loc[result.customers.is_high_clv, "customer_id"].tolist() == ["000", "001"]
    assert result.metadata["cohorts"][0]["high_clv_count"] == 2
    assert result.metadata["cohorts"][0]["high_clv_ratio"] == .2


def test_all_equal_clv_uses_string_ids():
    result = segment_customers(make_inputs([(key, .2, .1, 100) for key in ("3", "2", "10", "01", "0")]))
    assert result.customers.loc[result.customers.is_high_clv, "customer_id"].tolist() == ["0"]
    assert result.customers.customer_id.tolist() == ["0", "01", "10", "2", "3"]


@pytest.mark.parametrize("size,expected", [(1, 1), (3, 1), (6, 2)])
def test_small_population_ceiling(size, expected):
    result = segment_customers(make_inputs([(str(i), .5, .1, i) for i in range(size)]))
    assert result.customers.is_high_clv.sum() == expected


@pytest.mark.parametrize("fraction,expected", [(0, 0), (1, 3)])
def test_fraction_endpoints(inputs, fraction, expected):
    result = segment_customers(inputs, SegmentationConfig(high_clv_fraction=fraction))
    assert result.customers.is_high_clv.sum() == expected


def test_clv_negative_is_domain_error_without_clipping():
    source = make_inputs([("u", .5, -.1, -100)])  # Generic amount input permits this.
    with pytest.raises(InputValidationError, match="CLV must be nonnegative"):
        segment_customers(source)
    assert source.data.predicted_clv_12m.iloc[0] == -100


def test_priority_signed_zero_and_no_rounding():
    source = make_inputs([("neg", .2, -.1, 100), ("zero_u", .2, 0, 100),
                          ("zero_clv", .2, -.1, 0), ("p1", .2, .100001, 100), ("p2", .2, .100002, 100)])
    rows = segment_customers(source).customers.set_index("customer_id")
    assert rows.loc["neg", "priority_score"] == -10
    assert rows.loc["zero_u", "priority_score"] == 0
    assert rows.loc["zero_clv", "priority_score"] == 0
    assert rows.loc["p2", "priority_rank"] < rows.loc["p1", "priority_rank"]
    assert rows.priority_rank.sort_values().tolist() == [1, 2, 3, 4, 5]


def test_equal_scores_have_sequential_id_order():
    result = segment_customers(make_inputs([("b", .8, .2, 50), ("a", .1, .1, 100)]))
    assert result.customers.customer_id.tolist() == ["a", "b"]
    assert result.customers.priority_rank.tolist() == [1, 2]


def test_snapshots_in_same_week_are_independent():
    source = make_inputs([("a", .8, .1, 10), ("b", .8, .1, 20),
                          ("a", .8, .1, 40, T0 + pd.Timedelta(days=1)),
                          ("b", .8, .1, 10, T0 + pd.Timedelta(days=1))])
    result = segment_customers(source)
    assert len(result.summary) == 24
    high = result.customers.loc[result.customers.is_high_clv].sort_values("as_of_at")
    assert high.customer_id.tolist() == ["b", "a"]
    for _, group in result.customers.groupby("as_of_at"):
        assert group.priority_rank.tolist() == [1, 2]
    assert len(result.metadata["input_snapshots"]) == 4


def test_eligibility_changes_population_and_preserves_reasons(inputs):
    decisions = eligibility_for(inputs, {"001": "provided exclusion"}).iloc[::-1]
    original = decisions.copy(deep=True)
    result = segment_customers(inputs, eligibility=decisions)
    assert set(result.customers.customer_id) == {"002", "003"}
    assert result.excluded.customer_id.tolist() == ["001"]
    assert result.excluded.exclusion_reason.tolist() == ["provided exclusion"]
    assert result.metadata["population_scope"] == "explicit_eligibility"
    assert result.metadata["analyzed_count"] == 2 and result.metadata["excluded_count"] == 1
    assert result.customers.loc[result.customers.is_high_clv, "customer_id"].tolist() == ["003"]
    assert result.summary.customer_count.sum() == 2
    pd.testing.assert_frame_equal(decisions, original)


def test_all_excluded_is_a_normal_empty_cohort(inputs):
    result = segment_customers(inputs, eligibility=eligibility_for(inputs, {key: "excluded" for key in inputs.data.customer_id}))
    assert result.customers.empty and len(result.excluded) == 3
    assert len(result.summary) == 12
    assert result.summary.customer_count.eq(0).all() and result.summary.customer_ratio.eq(0).all()
    assert result.summary[[f"mean_{column}" for column in MEAN_COLUMNS]].isna().all().all()
    assert result.metadata["cohorts"][0]["high_clv_count"] == 0


def test_empty_integrated_slice_returns_empty_results(inputs):
    source = IntegratedInputs(inputs.data.iloc[:0].copy(), deepcopy(inputs.prediction_metadata),
                              {name: frame.iloc[:0].copy() for name, frame in inputs.prediction_context.items()})
    result = segment_customers(source)
    assert result.customers.empty and result.excluded.empty and result.summary.empty
    assert result.metadata["input_count"] == 0 and result.metadata["cohorts"] == []


@pytest.mark.parametrize("value,unit", [(12, "calendar_months"), (51, "weeks"),
                                       (53, "weeks"), (364, "days")])
def test_empty_segmentation_requires_52_week_clv(inputs, value, unit):
    source = IntegratedInputs(inputs.data.iloc[:0].copy(), deepcopy(inputs.prediction_metadata),
                              {name: frame.iloc[:0].copy() for name, frame in inputs.prediction_context.items()})
    source.prediction_metadata["clv"] = replace(source.prediction_metadata["clv"],
                                                horizon_value=value, horizon_unit=unit)
    with pytest.raises(InputValidationError, match="exactly 52 weeks"):
        segment_customers(source)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "non_bool", "missing_reason", "reason_for_eligible"])
def test_invalid_eligibility_is_not_silently_interpreted(inputs, case):
    decisions = eligibility_for(inputs, {"003": "excluded"})
    if case == "missing":
        decisions = decisions.iloc[:-1]
    elif case == "extra":
        decisions.loc[0, "customer_id"] = "unknown"
    elif case == "duplicate":
        decisions = pd.concat([decisions, decisions.iloc[[0]]])
    elif case == "non_bool":
        decisions["eligible"] = 1
    elif case == "missing_reason":
        decisions["exclusion_reason"] = pd.NA
    else:
        decisions["exclusion_reason"] = "reason"
    with pytest.raises(InputValidationError, match="eligibility"):
        segment_customers(inputs, eligibility=decisions)


def test_summary_values_empty_bins_and_reconciliation():
    source = make_inputs([("a", .8, .2, 10), ("b", .6, .4, 20)])
    result = segment_customers(source, SegmentationConfig(high_clv_fraction=1))
    row = result.summary.loc[result.summary.customer_count.gt(0)].iloc[0]
    assert row.customer_count == 2 and row.customer_ratio == 1
    assert row.mean_churn_probability == pytest.approx(.7)
    assert row.mean_uplift_score == pytest.approx(.3)
    assert row.mean_predicted_clv_12m == 15
    assert row.mean_priority_score == 5 and row.priority_score_sum == 10
    empty = result.summary.loc[result.summary.customer_count.eq(0)]
    assert len(empty) == 11 and empty.priority_score_sum.eq(0).all()
    assert empty.mean_priority_score.isna().all()


def test_nulls_inputs_context_and_metadata_are_not_mutated(inputs):
    before = deepcopy(inputs)
    result = segment_customers(inputs)
    assert result.customers.mean_net_order_value_30d.isna().all()
    assert result.metadata["coupon_eligibility_inferred"] is False
    pd.testing.assert_frame_equal(inputs.data, before.data)
    for model in PREDICTION_COLUMNS:
        pd.testing.assert_frame_equal(inputs.prediction_context[model], before.prediction_context[model])
        assert result.metadata["models"][model] == asdict(before.prediction_metadata[model])
    result.metadata["models"]["uplift"]["target_definition_id"] = "modified result"
    assert inputs.prediction_metadata["uplift"].target_definition_id == "dummy_purchase_retention_4w"


def test_row_order_does_not_change_results(twelve_inputs):
    shuffled = IntegratedInputs(twelve_inputs.data.sample(frac=1, random_state=7), deepcopy(twelve_inputs.prediction_metadata),
                                {name: frame.sample(frac=1, random_state=3) for name, frame in twelve_inputs.prediction_context.items()})
    first, second = segment_customers(twelve_inputs), segment_customers(shuffled)
    pd.testing.assert_frame_equal(first.customers, second.customers)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    assert first.metadata == second.metadata


@pytest.mark.parametrize("column,value", [("churn_probability", 1.1), ("uplift_score", np.nan),
                                          ("snapshot_id", ""), ("reference_week", "2024-03-11")])
def test_mutable_integrated_data_is_revalidated(inputs, column, value):
    inputs.data[column] = value
    with pytest.raises(InputValidationError):
        segment_customers(inputs)


@pytest.mark.parametrize("case", ["missing_context", "wrong_customer", "future_cutoff", "wrong_period", "wrong_semantics", "future_column"])
def test_inconsistent_context_and_metadata_are_rejected(inputs, case):
    if case == "missing_context":
        inputs.prediction_context.pop("churn")
    elif case == "wrong_customer":
        inputs.prediction_context["churn"].loc[0, "customer_id"] = "unknown"
    elif case == "future_cutoff":
        inputs.prediction_context["churn"]["input_cutoff_at"] += pd.Timedelta(seconds=1)
    elif case == "wrong_period":
        inputs.prediction_metadata["churn"] = replace(inputs.prediction_metadata["churn"], horizon_value=30)
    elif case == "wrong_semantics":
        inputs.prediction_metadata["uplift"] = replace(inputs.prediction_metadata["uplift"], value_semantics="revenue")
    else:
        inputs.data["treatment"] = 1
    with pytest.raises(InputValidationError):
        segment_customers(inputs)


def test_csv_roundtrip_summary_metadata_and_no_overwrite(inputs, tmp_path):
    result = segment_customers(inputs, eligibility=eligibility_for(inputs, {"003": "reason preserved"}))
    paths = export_segments(result, tmp_path / "segments_6plus.csv")
    rows = pd.read_csv(paths["customers"], dtype={"customer_id": str, "dataset_id": str, "amount_unit": str}, parse_dates=["as_of_at"])
    assert rows.customer_id.tolist() == result.customers.customer_id.tolist()
    for column in ("priority_score", "priority_rank", "churn_probability", "uplift_score", "predicted_clv_12m"):
        np.testing.assert_allclose(rows[column], result.customers[column].to_numpy(dtype=float))
    assert rows.snapshot_id.tolist() == result.customers.snapshot_id.tolist()
    assert rows.segment_id.tolist() == result.customers.segment_id.tolist()
    assert rows.is_high_clv.tolist() == result.customers.is_high_clv.tolist()
    assert rows.as_of_at.tolist() == result.customers.as_of_at.tolist()
    assert rows.amount_unit.eq("0.01").all()
    assert pd.read_csv(paths["summary"]).customer_count.sum() == 2
    assert pd.read_csv(paths["excluded"], dtype={"customer_id": str}).exclusion_reason.tolist() == ["reason preserved"]
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["models"] == result.metadata["models"]
    assert metadata["models"]["churn"]["horizon_unit"] == "days"
    assert metadata["models"]["uplift"]["horizon_unit"] == "weeks"
    assert metadata["models"]["clv"]["horizon_unit"] == "weeks"
    assert metadata["models"]["clv"]["horizon_value"] == 52
    with pytest.raises(FileExistsError):
        export_segments(result, paths["customers"])


def test_empty_csv_outputs_keep_headers(inputs, tmp_path):
    result = segment_customers(inputs, eligibility=eligibility_for(inputs, {key: "excluded" for key in inputs.data.customer_id}))
    paths = export_segments(result, tmp_path / "segments_6plus.csv")
    assert pd.read_csv(paths["customers"]).empty
    assert len(pd.read_csv(paths["summary"])) == 12


def test_plot_generation_is_cohort_specific_and_preserves_empty_means(twelve_inputs, tmp_path):
    result = segment_customers(twelve_inputs)
    figure = plot_segments(result, "dummy-segmentation", T0)
    assert len(figure.axes) == 4
    assert figure.axes[2].get_title() == "Mean predicted 52-week CLV"
    path = tmp_path / "segments.png"
    figure.savefig(path, dpi=80)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sum(patch.get_height() for patch in figure.axes[0].patches) == 60
    with pytest.raises(InputValidationError, match="cohort does not exist"):
        plot_segments(result, "dummy-segmentation", T0 + pd.Timedelta(days=1))


def test_all_excluded_plot_does_not_turn_mean_nulls_into_zero(inputs):
    result = segment_customers(inputs, eligibility=eligibility_for(inputs, {key: "excluded" for key in inputs.data.customer_id}))
    figure = plot_segments(result, "dummy-segmentation", T0)
    assert len(figure.axes[2].patches) == 0 and len(figure.axes[3].patches) == 0


def test_plots_never_combine_two_valid_snapshot_cohorts():
    inputs = make_inputs([("a", .8, .1, 10),
                          ("a", .8, .1, 100, T0 + pd.Timedelta(days=1)),
                          ("b", .8, .1, 200, T0 + pd.Timedelta(days=1))])
    result = segment_customers(inputs)
    first = plot_segments(result, "dummy-segmentation", T0)
    second = plot_segments(result, "dummy-segmentation", T0 + pd.Timedelta(days=1))
    assert sum(patch.get_height() for patch in first.axes[0].patches) == 1
    assert sum(patch.get_height() for patch in second.axes[0].patches) == 2
    assert "N=1" in first._suptitle.get_text()
    assert "N=2" in second._suptitle.get_text()


def test_normalized_context_clocks_preserve_periods(inputs):
    for context in inputs.prediction_context.values():
        for column in ("as_of_at", "input_cutoff_at", "prediction_end_at"):
            context[column] = context[column].dt.tz_convert("Asia/Seoul").map(lambda point: point.isoformat())
    result = segment_customers(inputs)
    assert all(row["as_of_at"] == T0.isoformat() for rows in result.metadata["prediction_context"].values() for row in rows)


def test_decimal_fraction_does_not_add_an_extra_high_value_customer():
    inputs = make_inputs([(f"{i:03}", .5, .1, i) for i in range(50)])
    result = segment_customers(inputs, SegmentationConfig(high_clv_fraction=.14))
    assert result.customers.is_high_clv.sum() == 7
