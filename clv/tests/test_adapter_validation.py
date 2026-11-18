"""WBS 4.1: Adapter 열 매핑, 통화·금액 단위, 결제액 금액식, 키 중복, 관측 범위, 주문 원장↔주간 대조를 검사한다."""
import numpy as np
import pandas as pd
import pytest

from clv.adapter import adapt_orders, adapt_weekly, load_orders, load_weekly
from clv.dataset import complete_grid
from clv.schema import SIMULATOR_TO_CLV, UNSPECIFIED_CURRENCY
from clv.validation import reconcile_orders_with_weekly, validate_orders, validate_weekly

from .conftest import clv_rows


def test_simulator_columns_map_to_agreed_clv_names(outputs):
    weekly, _, _ = outputs
    adapted = adapt_weekly(weekly)
    assert adapted.attrs["source_format"] == "simulator_v5_weekly_panel"
    for source, target in SIMULATOR_TO_CLV.items():
        assert target in adapted, target
    observed = weekly["observed_week"].to_numpy()
    original = weekly.sort_values(["customer_id", "reference_week"]).reset_index(drop=True)
    for source, target in SIMULATOR_TO_CLV.items():
        if source in ("customer_id", "reference_week"):
            continue
        left = adapted[target].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        right = original[source].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        np.testing.assert_array_equal(left, right)
    assert str(adapted["week"].dt.tz) == "Asia/Seoul"
    assert adapted["week"].equals(original["reference_week"].dt.tz_convert("Asia/Seoul"))
    assert observed.sum() == adapted["observed_week"].sum()


def test_paid_amount_is_gross_minus_discount_and_send_cost_is_not_deducted(outputs):
    """결제액 = gross - discount이며 시뮬레이터 net_revenue_minor와 같다. 쿠폰 발송비는 차감하지도 읽지도 않는다."""
    weekly, _, _ = outputs
    adapted = adapt_weekly(weekly)
    seen = adapted["observed_week"]
    expected = adapted.loc[seen, "gross_revenue"] - adapted.loc[seen, "redemption_discount_cost"]
    assert adapted.loc[seen, "paid_amount"].equals(expected)
    assert adapted.loc[seen, "paid_amount"].equals(adapted.loc[seen, "source_net_revenue"])
    assert "marketing_send_cost_minor" not in adapted and "net_after_marketing_minor" not in adapted
    # Fixture에 쿠폰 발송과 발송비가 실제로 있어야 위 검사가 의미가 있다.
    sent = adapted.loc[seen & adapted["coupons"].gt(0)]
    assert len(sent) and (weekly["marketing_send_cost_minor"].fillna(0) > 0).any()


def test_units_are_carried_unchanged_and_never_defaulted(outputs):
    """통화·금액 단위는 원천 값 그대로 전달한다. SOURCE_UNSPECIFIED를 KRW로 바꾸지 않고 경고만 남기며,
    값이 없거나 섞여 있거나 지정한 통화와 다르면 오류다."""
    weekly, _, _ = outputs
    adapted = adapt_weekly(weekly)
    assert adapted.attrs["currency"] == UNSPECIFIED_CURRENCY and adapted.attrs["amount_unit"] == "0.01"
    report = validate_weekly(complete_grid(adapted))
    assert report.ok and any("not KRW" in warning for warning in report.warnings)
    with pytest.raises(ValueError, match="currency and amount_unit"):
        adapt_weekly(weekly.drop(columns=["currency", "amount_unit"]))
    with pytest.raises(ValueError, match="disagrees"):
        adapt_weekly(weekly, currency="KRW")
    with pytest.raises(ValueError, match="one currency"):
        adapt_weekly(weekly.assign(currency=np.where(np.arange(len(weekly)) % 2, "KRW", UNSPECIFIED_CURRENCY)))


def test_unobserved_weeks_stay_missing_not_zero(outputs):
    """미관측 주(가입 전·관측 밖)의 활동·금액은 0으로 채우지 않고 NA로 남긴다. 부분 관측 주도 있어야 한다."""
    weekly, _, _ = outputs
    adapted = adapt_weekly(weekly)
    unobserved = ~adapted["observed_week"]
    assert unobserved.any()
    assert adapted.loc[unobserved, ["sessions", "purchases", "gross_revenue", "paid_amount"]].isna().all().all()
    assert adapted["partial_coverage"].any()


def test_clv_schema_csv_round_trip_and_absent_rows_are_unobserved(tmp_path):
    """관측 열이 없는 CLV 스키마 CSV는 행이 있는 주를 완전 관측, 없는 주를 미관측 주로 보고 그 가정을 기록한다."""
    rows = clv_rows({"a": [(1, 1, 100, 10), None, (2, 0, 0, 0)], "b": [(0, 0, 0, 0)] * 3})
    rows["week"] = rows["week"].dt.strftime("%Y-%m-%d")  # 시간대 없는 날짜 = 서울 현지 월요일
    path = tmp_path / "weekly.csv"
    rows.to_csv(path, index=False)
    adapted = load_weekly(path)
    assert adapted.attrs["source_format"] == "clv_schema"
    assert adapted.attrs["coverage_source"] == "assumed_from_rows"
    assert any("naive" in note for note in adapted.attrs["notes"])
    grid = complete_grid(adapted)
    assert len(grid) == 6
    gap = grid.loc[grid["customer_id"].eq("a") & ~grid["observed_week"]]
    assert len(gap) == 1 and gap["paid_amount"].isna().all()
    report = validate_weekly(grid)
    assert report.ok, report.errors
    assert any("absent" in note for note in report.warnings)
    assert grid.loc[grid["customer_id"].eq("a"), "paid_amount"].dropna().tolist() == [90, 0]


def test_duplicate_customer_week_is_an_error():
    rows = clv_rows({"a": [(1, 1, 100, 0), (1, 0, 0, 0)]})
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    report = validate_weekly(adapt_weekly(rows))
    assert any("duplicate" in error for error in report.errors)


def test_negative_money_discount_above_gross_and_net_mismatch_are_errors(outputs):
    weekly, _, _ = outputs
    broken = weekly.copy()
    first = broken.index[broken["observed_week"] & broken["purchases"].gt(0)][0]
    # 할인을 gross보다 크게 만들면 net_revenue_minor와의 일치도 함께 깨진다.
    broken.loc[first, "redemption_discount_cost_minor"] = broken.loc[first, "gross_revenue_minor"] + 1
    report = validate_weekly(complete_grid(adapt_weekly(broken)))
    assert any("discount more than gross" in error for error in report.errors)
    assert any("net_revenue_minor" in error for error in report.errors)
    negative = clv_rows({"a": [(1, 1, -5, 0)]})
    assert any("negative gross_revenue" in error for error in validate_weekly(adapt_weekly(negative)).errors)


def test_fractional_amounts_and_non_monday_weeks_are_rejected():
    rows = clv_rows({"a": [(1, 1, 100.5, 0)]})
    with pytest.raises(ValueError, match="whole numbers"):
        adapt_weekly(rows)
    shifted = clv_rows({"a": [(1, 1, 100, 0)]})
    shifted["week"] = shifted["week"] + pd.Timedelta(days=1)
    assert any("Monday" in error for error in validate_weekly(adapt_weekly(shifted)).errors)


def test_order_ledger_from_raw_sqlite_matches_weekly(raw_dir, panel):
    ledger = load_orders(raw_dir / "dataset")
    assert ledger.attrs["currency"] == UNSPECIFIED_CURRENCY and ledger.attrs["raw_database_sha256"]
    assert str(ledger["order_at"].dt.tz) == "UTC"
    assert validate_orders(ledger).ok
    report = reconcile_orders_with_weekly(panel, ledger)
    assert report.ok, report.errors
    assert report.stats["weekly_paid"] == report.stats["ledger_paid"] == int(ledger["paid_amount"].sum())


def test_raw_sqlite_checksum_is_enforced(raw_dir, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(raw_dir / "dataset", copy)
    # 파일 끝에 1바이트를 붙여 manifest 체크섬과 어긋나게 한다.
    with open(copy / "raw.sqlite", "ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ValueError, match="checksum"):
        load_orders(copy)
    assert len(load_orders(copy, verify_checksum=False)) > 0


def test_ledger_errors_are_reported(orders, panel):
    """원장 금액식 위반, 주간 결제액과의 불일치, 미관측 주의 주문, 통화 불일치를 각각 오류로 보고한다."""
    bad = orders.copy()
    bad.attrs = dict(orders.attrs)
    bad.loc[bad.index[0], "paid_amount"] += 1
    assert any("paid != gross - discount" in error for error in validate_orders(bad).errors)
    assert any("weekly paid_amount" in error for error in reconcile_orders_with_weekly(panel, bad).errors)
    late = adapt_orders(pd.DataFrame({"customer_id": ["c0000"], "order_at": ["2030-01-01T00:00:00Z"],
                                      "gross_amount": [10], "discount_amount": [0], "paid_amount": [10]}),
                        currency=UNSPECIFIED_CURRENCY, amount_unit="0.01")
    assert any("unobserved" in error for error in reconcile_orders_with_weekly(panel, late).errors)
    other = orders.copy()
    other.attrs = {**orders.attrs, "currency": "KRW"}
    assert any("currency differs" in error for error in reconcile_orders_with_weekly(panel, other).errors)


def test_simulator_weekly_panel_exported_to_csv_reads_like_parquet(outputs, tmp_path):
    weekly, _, _ = outputs
    path = tmp_path / "weekly_panel.csv"
    weekly.to_csv(path, index=False)
    from_csv, from_frame = load_weekly(path), adapt_weekly(weekly)
    columns = ["customer_id", "sessions", "purchases", "paid_amount", "observed_week", "partial_coverage", "coverage_seconds"]
    pd.testing.assert_frame_equal(from_csv[columns], from_frame[columns])
    assert from_csv["week"].equals(from_frame["week"])


def test_missing_coupon_action_stays_missing():
    """보조 열 coupon_action이 없으면 0으로 채우지 않고 NA로 두며, 검증은 통과한다."""
    adapted = adapt_weekly(clv_rows({"a": [(1, 1, 100, 0)]}).drop(columns=["coupon_action"]))
    assert adapted["coupon_action"].isna().all() and validate_weekly(adapted).ok
