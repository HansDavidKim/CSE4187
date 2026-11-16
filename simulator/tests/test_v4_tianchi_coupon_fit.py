"""Receipt-grain calendar fixtures; no production coupon data is read."""
import copy
import json
import sys

import pandas as pd
import pytest

from simulate.v4.irl.fit_tianchi_coupon import COLUMNS, fit_tianchi_coupon


def row(user, received="null", purchased="null", discount="null", coupon="null"):
    return dict(zip(COLUMNS, [user, "merchant", coupon, discount, "0", received, purchased]))


def receipt(user, received, purchased="null", discount="0.9"):
    return row(user, received, purchased, discount, "coupon")


def anchors():
    return [row("observation-start", purchased="20160101"), row("observation-end", purchased="20160630")]


def test_strict_past_cohorts_include_ordinary_and_coupon_purchases_but_not_same_day():
    source = pd.DataFrame(anchors() + [
        receipt("no-history", "20160301", "20160301"),
        row("recent", purchased="20160202"), receipt("recent", "20160301", "20160330"),
        row("lapsed", purchased="20160201"), receipt("lapsed", "20160301", "20160331"),
        row("same-day", purchased="20160301"), receipt("same-day", "20160301"),
        receipt("coupon-history", "20160201", "20160205", "0.95"), receipt("coupon-history", "20160301"),
    ])
    before = source.copy(deep=True)
    model = fit_tianchi_coupon(source)
    pd.testing.assert_frame_equal(source, before)
    groups = model["groups"]
    assert groups["no_observed_purchase"] == {"receipts": 2, "redeemed": 1, "response_probability": .5, "redemption_delay_days": [0]}
    assert groups["purchased_within_28d"] == {"receipts": 2, "redeemed": 1, "response_probability": .5, "redemption_delay_days": [29]}
    assert groups["lapsed_purchase"] == {"receipts": 1, "redeemed": 0, "response_probability": 0., "redemption_delay_days": []}
    assert groups["pooled"]["receipts"] == 5
    assert groups["pooled"]["redeemed"] == 2  # Day 30 is outside [0,30).
    from simulate.v4.simulator.coupon_response import validate_response_model
    validate_response_model(model)


def test_history_and_followup_calendar_edges_and_direct_discount_only():
    source = pd.DataFrame(anchors() + [
        receipt("too-early", "20160128"), receipt("history-boundary", "20160129"),
        receipt("followup-boundary", "20160531"), receipt("too-late", "20160601"),
        receipt("threshold-is-not-direct", "20160301", discount="100:10"),
        receipt("other-direct", "20160301", discount="0.8"),
        receipt("direct-format", "20160301", "20160302", discount=" .900 "),
    ])
    model = fit_tianchi_coupon(source)
    assert model["eligible_receipt_start_inclusive"] == "2016-01-29"
    assert model["eligible_receipt_end_inclusive"] == "2016-05-31"
    assert model["groups"]["pooled"]["receipts"] == 3
    assert model["groups"]["pooled"]["redemption_delay_days"] == [1]
    quality = model["quality"]
    assert quality["excluded_insufficient_global_history_receipts"] == 1
    assert quality["excluded_incomplete_30d_followup_receipts"] == 1
    assert quality["excluded_nonmatching_discount_receipts"] == 2


def test_duplicate_receipts_are_counted_and_zero_support_requires_pooled_fallback():
    repeated = receipt("no-history", "20160301", "20160303")
    source = pd.DataFrame(anchors() + [repeated, copy.deepcopy(repeated)])
    model = fit_tianchi_coupon(source)
    assert model["groups"]["pooled"] == {"receipts": 2, "redeemed": 2, "response_probability": 1., "redemption_delay_days": [2, 2]}
    assert model["groups"]["lapsed_purchase"] == {"receipts": 0, "redeemed": 0, "response_probability": 0., "redemption_delay_days": []}
    assert "only when" in model["cohort_fallback"]
    assert model["quality"]["exact_duplicate_extra_rows"] == 1
    assert model["quality"]["eligible_exact_duplicate_extra_rows"] == 1
    assert model["quality"]["duplicates_dropped"] == 0


def test_future_and_same_day_purchase_cannot_change_receipt_history_cohort():
    source = pd.DataFrame(anchors() + [receipt("person", "20160301", "20160302")])
    baseline = fit_tianchi_coupon(source)
    changed = pd.concat([source, pd.DataFrame([row("person", purchased="20160301"),
                                             row("person", purchased="20160601")])], ignore_index=True)
    assert fit_tianchi_coupon(changed)["groups"] == baseline["groups"]


@pytest.mark.parametrize("sentinel", [None, "", "null", "NULL", "nan", float("nan")])
def test_missing_date_sentinels_are_not_fabricated_redemptions(sentinel):
    source = pd.DataFrame(anchors() + [receipt("person", "20160301", sentinel)])
    assert fit_tianchi_coupon(source)["groups"]["pooled"]["redeemed"] == 0


@pytest.mark.parametrize("invalid", ["20160230", "2016-02-01", "20161301", "yesterday"])
def test_invalid_dates_fail_closed(invalid):
    source = pd.DataFrame(anchors() + [receipt("person", invalid)])
    with pytest.raises(ValueError, match="invalid Date_received"):
        fit_tianchi_coupon(source)


def test_purchase_before_receipt_and_absent_supported_receipts_fail():
    with pytest.raises(ValueError, match="before Date_received"):
        fit_tianchi_coupon(pd.DataFrame(anchors() + [receipt("person", "20160302", "20160301")]))
    with pytest.raises(ValueError, match="no fully observed"):
        fit_tianchi_coupon(pd.DataFrame(anchors() + [receipt("person", "20160301", discount="100:10")]))


def test_cli_records_source_hash_and_refuses_overwrite(tmp_path, monkeypatch):
    import hashlib
    import simulate.v4.irl.fit_tianchi_coupon as coupon

    source = tmp_path / "source.csv"
    pd.DataFrame(anchors() + [receipt("person", "20160301", "20160302")]).to_csv(source, index=False)
    output = tmp_path / "model.json"
    monkeypatch.setattr(sys, "argv", ["fit_tianchi_coupon", "--source", str(source), "--output", str(output)])
    coupon.main()
    model = json.loads(output.read_text())
    assert model["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert model["source"]["path"] == str(source.resolve())
    original = output.read_bytes()
    source.unlink()  # Existing output must reject before touching the source.
    with pytest.raises(FileExistsError, match="must be new"):
        coupon.main()
    assert output.read_bytes() == original
