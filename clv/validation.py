"""WBS 4.1 입력 검증: 스키마·타입·키 중복·금액식·통화·단위·관측 정보, 주문 원장과 주간 패널 대조."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import (
    AMOUNT_COLUMNS, COUNT_COLUMNS, REQUIRED_WEEKLY, SEOUL, UNSPECIFIED_CURRENCY, WEEK_SECONDS,
)


@dataclass
class ValidationReport:
    """검증 결과. ``errors``가 있으면 실행을 멈추고, ``warnings``는 산출물에 함께 기록한다."""

    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> "ValidationReport":
        """오류가 하나라도 있으면 모두 이어 붙여 ``ValueError``를 내고, 없으면 자신을 돌려준다."""
        if self.errors:
            raise ValueError(f"{self.name} validation failed: " + "; ".join(self.errors))
        return self

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "stats": self.stats}


def week_start(values: pd.Series) -> pd.Series:
    """각 시각이 속한 주의 시작(서울 월요일 00:00, 시뮬레이터 주 경계)을 돌려준다."""
    local = values.dt.tz_convert(SEOUL)
    return local.dt.normalize() - pd.to_timedelta(local.dt.weekday, unit="D")


def _units(report: ValidationReport, frame: pd.DataFrame) -> None:
    """``attrs``의 통화·단위를 검사한다. 없으면 오류, `SOURCE_UNSPECIFIED`면 KRW가 아니라는 경고."""
    currency, unit = frame.attrs.get("currency"), frame.attrs.get("amount_unit")
    report.stats.update(currency=currency, amount_unit=unit)
    if not currency or not unit:
        report.errors.append("currency and amount_unit are required")
    elif currency == UNSPECIFIED_CURRENCY:
        report.warnings.append(f"currency is {UNSPECIFIED_CURRENCY}: amounts are source minor units "
                               f"(amount_unit={unit}), not KRW")


def validate_weekly(panel: pd.DataFrame) -> ValidationReport:
    """``adapter.adapt_weekly``로 변환한 주간 프레임을 검사해 ``ValidationReport``를 돌려준다.

    컬럼·타입·중복·주 경계·음수·결제액 식·관측 정보 일관성을 보고, 빠진 고객-주는 미관측으로 경고한다.
    """
    report = ValidationReport("weekly")
    missing = [name for name in (*REQUIRED_WEEKLY, "paid_amount", "observed_week", "partial_coverage",
                                 "coverage_seconds") if name not in panel]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report
    if not isinstance(panel["week"].dtype, pd.DatetimeTZDtype):
        report.errors.append("week must be timezone-aware")
        return report
    for name in (*COUNT_COLUMNS, *AMOUNT_COLUMNS, "paid_amount"):
        if name in panel and not pd.api.types.is_integer_dtype(panel[name]):
            report.errors.append(f"{name} must be an integer column")
    if not panel["customer_id"].map(lambda value: isinstance(value, str)).all():
        report.errors.append("customer_id must be a string")
    if panel["week"].isna().any():
        report.errors.append("week cannot be missing")
        return report
    duplicated = panel.duplicated(["customer_id", "week"])
    if duplicated.any():
        report.errors.append(f"{int(duplicated.sum())} duplicate customer_id/week rows")
    if not week_start(panel["week"]).eq(panel["week"].dt.tz_convert(SEOUL)).all():
        report.errors.append("week must be an Asia/Seoul Monday 00:00 boundary")
    # 관측 주에는 값이 모두 있어야 한다(미관측 주만 결측 허용).
    observed = panel["observed_week"].astype(bool)
    values = [*REQUIRED_WEEKLY[2:], "paid_amount"]
    incomplete = observed & panel[values].isna().any(axis=1)
    if incomplete.any():
        report.errors.append(f"{int(incomplete.sum())} observed weeks have missing values")
    for name in (*COUNT_COLUMNS, *AMOUNT_COLUMNS):
        negative = panel[name].fillna(0).lt(0)
        if negative.any():
            report.errors.append(f"{int(negative.sum())} negative {name} values")
    over = (panel["redemption_discount_cost"] > panel["gross_revenue"]).fillna(False)
    if over.any():
        report.errors.append(f"{int(over.sum())} weeks discount more than gross revenue")
    # 시뮬레이터 `net_revenue_minor`가 결제액(총매출 - 할인액)과 주마다 같은지 대조한다.
    if "source_net_revenue" in panel:
        differs = (panel["source_net_revenue"] != panel["paid_amount"]).fillna(False)
        if differs.any():
            report.errors.append(f"{int(differs.sum())} weeks: net_revenue_minor != gross - discount")
        report.stats["net_revenue_checked_weeks"] = int(panel["source_net_revenue"].notna().sum())
    coverage = panel["coverage_seconds"]
    if coverage.lt(0).any() or coverage.gt(WEEK_SECONDS + 1e-6).any():
        report.errors.append("coverage_seconds must be within one week")
    if not observed.eq(coverage.gt(0)).all():
        report.errors.append("observed_week disagrees with coverage_seconds")
    # 부분 관측 주 = 관측은 됐지만 관측 초가 한 주보다 짧은 주.
    partial = panel["partial_coverage"].astype(bool)
    if not partial.eq(observed & coverage.lt(WEEK_SECONDS - 1e-6)).all():
        report.errors.append("partial_coverage disagrees with coverage_seconds")
    # 전체 고객 x 달력 주 격자와 비교해 행이 없는 고객-주를 센다. 이 주들은 0이 아니라 미관측이다.
    weeks = panel["week"].drop_duplicates().sort_values()
    grid = pd.date_range(weeks.iloc[0], weeks.iloc[-1], freq="7D") if len(weeks) else weeks
    expected = panel["customer_id"].nunique() * len(grid)
    gaps = expected - len(panel.drop_duplicates(["customer_id", "week"]))
    if gaps:
        report.warnings.append(f"{gaps} customer-week rows are absent; they are treated as unobserved, not zero")
    _units(report, panel)
    report.warnings += list(panel.attrs.get("notes", []))
    report.stats.update(
        customers=int(panel["customer_id"].nunique()), rows=int(len(panel)),
        first_week=weeks.iloc[0].isoformat() if len(weeks) else None,
        last_week=weeks.iloc[-1].isoformat() if len(weeks) else None, calendar_weeks=int(len(grid)),
        observed_weeks=int(observed.sum()), partial_weeks=int(partial.sum()),
        unobserved_weeks=int((~observed).sum()) + int(gaps),
        gross_revenue=int(panel["gross_revenue"].sum()), discount=int(panel["redemption_discount_cost"].sum()),
        paid_amount=int(panel["paid_amount"].sum()), coverage_source=panel.attrs.get("coverage_source"))
    return report


def validate_orders(orders: pd.DataFrame) -> ValidationReport:
    """주문 원장의 `order_id` 중복, 시간대, 금액 결측·음수, ``paid = gross - discount``를 검사한다."""
    report = ValidationReport("orders")
    if orders["order_id"].duplicated().any():
        report.errors.append(f"{int(orders['order_id'].duplicated().sum())} duplicate order_id")
    if not isinstance(orders["order_at"].dtype, pd.DatetimeTZDtype):
        report.errors.append("order_at must be timezone-aware")
    for name in ("gross_amount", "discount_amount", "paid_amount"):
        if orders[name].isna().any():
            report.errors.append(f"{name} cannot be missing")
        if orders[name].fillna(0).lt(0).any():
            report.errors.append(f"negative {name}")
    # 결측 때문에 비교할 수 없는 주문도 불일치로 센다(fillna(True)).
    mismatch = (orders["paid_amount"] != orders["gross_amount"] - orders["discount_amount"]).fillna(True)
    if mismatch.any():
        report.errors.append(f"{int(mismatch.sum())} orders: paid != gross - discount")
    over = (orders["discount_amount"] > orders["gross_amount"]).fillna(False)
    if over.any():
        report.errors.append(f"{int(over.sum())} orders discount more than gross")
    _units(report, orders)
    report.warnings += list(orders.attrs.get("notes", []))
    report.stats.update(orders=int(len(orders)), customers=int(orders["customer_id"].nunique()),
                        paid_amount=int(orders["paid_amount"].sum()) if len(orders) else 0,
                        first_order_at=orders["order_at"].min().isoformat() if len(orders) else None,
                        last_order_at=orders["order_at"].max().isoformat() if len(orders) else None)
    return report


def reconcile_orders_with_weekly(panel: pd.DataFrame, orders: pd.DataFrame, max_examples: int = 5) -> ValidationReport:
    """주문 원장을 고객·서울 주 단위로 합산해 주간 패널의 구매 수·총매출·할인액·결제액과 대조한다.

    미관측 주나 행이 없는 주에 주문이 있으면 오류다. 예시는 최대 ``max_examples``개만 남긴다.
    """
    report = ValidationReport("ledger_vs_weekly")
    for name in ("currency", "amount_unit"):
        if panel.attrs.get(name) != orders.attrs.get(name):
            report.errors.append(f"{name} differs: weekly={panel.attrs.get(name)!r}, orders={orders.attrs.get(name)!r}")
    ledger = orders.assign(week=week_start(orders["order_at"])).groupby(["customer_id", "week"]).agg(
        ledger_purchases=("order_id", "size"), ledger_gross=("gross_amount", "sum"),
        ledger_discount=("discount_amount", "sum"), ledger_paid=("paid_amount", "sum"))
    weekly = panel.set_index(["customer_id", "week"])[["observed_week", "purchases", "gross_revenue",
                                                       "redemption_discount_cost", "paid_amount"]]
    # outer join이라 패널에 행이 없는 주문 주도 남는다. 그 주의 `observed_week`는 NA -> 미관측으로 본다.
    joined = weekly.join(ledger, how="outer")
    outside = ~joined["observed_week"].astype("boolean").fillna(False).astype(bool)
    orphan = outside & joined["ledger_purchases"].notna()
    if orphan.any():
        examples = [f"{c}@{w.isoformat()}" for c, w in joined.index[orphan][:max_examples]]
        report.errors.append(f"{int(orphan.sum())} ledger customer-weeks fall in unobserved or absent weeks: {examples}")
    inside = joined.loc[~outside].copy()
    pairs = {"purchases": "ledger_purchases", "gross_revenue": "ledger_gross",
             "redemption_discount_cost": "ledger_discount", "paid_amount": "ledger_paid"}
    for weekly_name, ledger_name in pairs.items():
        # 관측 주에 주문이 없으면 원장 합계는 0이다(관측된 무구매 주).
        expected = inside[ledger_name].fillna(0).astype("int64")
        actual = inside[weekly_name].astype("int64")
        differs = actual.ne(expected)
        if differs.any():
            examples = [f"{c}@{w.isoformat()}" for c, w in inside.index[differs][:max_examples]]
            report.errors.append(f"{int(differs.sum())} weeks: weekly {weekly_name} != ledger sum {examples}")
    report.stats.update(compared_weeks=int(len(inside)), ledger_orders=int(len(orders)),
                        weekly_paid=int(inside["paid_amount"].sum()),
                        ledger_paid=int(inside["ledger_paid"].fillna(0).sum()))
    return report


def summarize_reports(reports) -> dict:
    """여러 검증 결과를 이름별 dict로 묶어 JSON에 쓸 수 있게 돌려준다."""
    return {report.name: report.to_dict() for report in reports}
