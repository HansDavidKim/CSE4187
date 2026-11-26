"""Request-based estimates for one declared historical cohort and one decision cohort.

The exposures and order lines are an internal, normalized evidence contract, not
a simulator export. Keep every assignment, including failed requests and controls.
Historical KIT39 snapshots use an exclusive feature cutoff. The caller supplies
the original cohort size and manifest digest; this in-memory API does not verify
files, causal independence, or the completeness of a producer's records.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from numbers import Integral
import re

import numpy as np
import pandas as pd

from . import budget
from .budget import CostMetadata
from .retention_strategy import COST_INPUT_COLUMNS, audit_observed_coupon_order
from .validation import CONTEXT_COLUMNS, InputValidationError, validate_feature_frame


EXPOSURE_COLUMNS = (*CONTEXT_COLUMNS, "policy_id", "discount_rate", "assigned_at", "treatment",
                    "coupon_action", "request_id", "actual_sent", "execution_status", "failure_reason",
                    "coupon_id", "sent_at", "expires_at", "outcome_end_at", "outcome_observed",
                    "is_censored", "known_at", "redeemed_in_outcome", "redemption_order_id",
                    "redemption_at", "charged_minor", "outcome_redemption_discount_cost_minor")
ORDER_COLUMNS = ("order_id", "customer_id", "coupon_id", "purchased_at", "paid_minor",
                 "line_no", "quantity", "unit_price_minor")
COMPONENTS = ("send_cost", "redemption_probability", "expected_discount_if_redeemed")
WINDOW = pd.Timedelta(days=28)


@dataclass(frozen=True)
class HistoricalCohort:
    dataset_id: str
    raw_database_sha256: str
    manifest_sha256: str
    policy_id: str
    billing_contract: str
    failure_charge_allowed: bool
    assignment_count: int
    observation_start_at: object
    observation_end_at: object


@dataclass(frozen=True)
class EstimationConfig:
    version: str
    min_requests: int
    min_redemptions: int
    send_places: int
    probability_places: int
    discount_places: int
    decimal_places: int
    # None uses population means only; count >= threshold is the upper group.
    purchase_count_30d_threshold: int | None = None


@dataclass(frozen=True)
class CostEstimate:
    status: str
    cost_metadata: CostMetadata
    _rows: tuple
    _audit_json: str

    @property
    def cost_inputs(self):
        _require(self.status == "complete", "cost estimates withheld; inspect audit")
        return pd.DataFrame.from_records(self._rows, columns=COST_INPUT_COLUMNS)

    @property
    def audit(self):
        return json.loads(self._audit_json)


def _require(ok, message):
    if pd.isna(ok) or not ok:
        raise InputValidationError("coupon costs: " + message)


def _integer(value, name, minimum=0):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
             and value >= minimum, f"invalid {name}")
    return int(value)


def _flag(value, name):
    _require(isinstance(value, (bool, np.bool_)), f"invalid {name}")
    return bool(value)


def _frame(frame, columns):
    _require(isinstance(frame, pd.DataFrame) and not frame.columns.duplicated().any()
             and set(frame) == set(columns), "unexpected ledger columns")
    return frame.loc[:, list(columns)].copy()


def _group(count, threshold):
    if threshold is None:
        return "population"
    if pd.isna(count):
        return "missing_purchase_count"
    _require(count >= 0 and float(count).is_integer(), "invalid purchase_count_30d")
    return "at_or_above" if count >= threshold else "below"


@dataclass
class _Counts:
    requests: int = 0
    sent: int = 0
    used: int = 0
    charges: int = 0
    discounts: int = 0

    def record(self, sent, used, charge, discount):
        self.requests += 1
        self.sent += sent
        self.used += used
        self.charges += charge
        self.discounts += discount

    def ratios(self):
        return (Fraction(self.charges, self.requests) if self.requests else None,
                Fraction(self.used, self.requests) if self.requests else None,
                Fraction(self.discounts, self.used) if self.used else None)

    def audit(self):
        return dict(**asdict(self), charge_samples=self.requests, discount_samples=self.used,
                    exact={k: str(v) if v is not None else None
                           for k, v in zip(COMPONENTS, self.ratios(), strict=True)})


def _exposure(row, orders, cohort, start, end, freeze):
    assigned = budget._clock(row.assigned_at, "assigned_at")
    outcome_end = budget._clock(row.outcome_end_at, "outcome_end_at")
    known = budget._clock(row.known_at, "known_at")
    _require(start <= assigned and row.as_of_at <= assigned and assigned + WINDOW == outcome_end
             and outcome_end <= end and outcome_end <= known <= freeze, "incomplete or future observation")
    _require(_flag(row.outcome_observed, "outcome_observed") and not _flag(row.is_censored, "is_censored"),
             "censored cohort")
    _require(row.policy_id == cohort.policy_id, "coupon policy mismatch")
    action = _integer(row.coupon_action, "coupon_action")
    _require(action in (0, 1) and _integer(row.treatment, "treatment") == action, "assignment/action mismatch")
    sent = _flag(row.actual_sent, "actual_sent")
    if sent or not pd.isna(row.discount_rate):
        _require(budget._decimal(row.discount_rate, "rate") == Decimal(".1"), "coupon specification mismatch")
    used = _flag(row.redeemed_in_outcome, "redeemed_in_outcome")
    charge = budget._ticks(budget._decimal(row.charged_minor, "charged_minor"), 0, "charged_minor")
    discount = budget._ticks(budget._decimal(row.outcome_redemption_discount_cost_minor, "discount_minor"),
                             0, "discount_minor")
    if action:
        budget._text(row.request_id, "request_id")
    else:
        _require(pd.isna(row.request_id) and not sent and charge == 0, "invalid non-request")
    if sent:
        budget._text(row.coupon_id, "coupon_id")
        _require(row.execution_status == "sent" and pd.isna(row.failure_reason)
                 and budget._clock(row.sent_at, "sent_at") == assigned
                 and budget._clock(row.expires_at, "expires_at") == outcome_end, "invalid coupon execution")
    else:
        _require(all(pd.isna(v) for v in (row.coupon_id, row.sent_at, row.expires_at)) and not used,
                 "unexecuted coupon has send/use evidence")
        if action:
            budget._text(row.failure_reason, "failure_reason")
            _require(row.execution_status in ("failed", "not_sent") and (cohort.failure_charge_allowed or charge == 0),
                     "failed request/billing contract mismatch")
        else:
            _require(row.execution_status == "not_requested" and pd.isna(row.failure_reason), "invalid control")
    if used:
        order_id = budget._text(row.redemption_order_id, "redemption_order_id")
        items = orders.pop(order_id, None)
        used_at = budget._clock(row.redemption_at, "redemption_at")
        _require(items is not None and assigned <= used_at < outcome_end, "invalid redemption order/time")
        _require(items.customer_id.eq(row.customer_id).all() and items.coupon_id.eq(row.coupon_id).all()
                 and all(budget._clock(t, "purchased_at") == used_at for t in items.purchased_at)
                 and items.paid_minor.nunique(dropna=False) == 1, "order linkage mismatch")
        audit_observed_coupon_order(items[["line_no", "quantity", "unit_price_minor"]], purchase_at=used_at,
                                    as_of_at=outcome_end, actual_discount_minor=discount,
                                    actual_paid_minor=items.paid_minor.iloc[0])
    else:
        _require(pd.isna(row.redemption_at) and pd.isna(row.redemption_order_id) and discount == 0,
                 "unexpected redemption evidence")
    return action, sent, used, charge, discount


def _ceil(value, places):
    ticks = -(-value.numerator * 10**places // value.denominator)
    return Decimal(f"{ticks}e-{places}")


def _estimate_group(group, counts, population, config):
    values, details = [], {}
    for index, (key, places) in enumerate(zip(COMPONENTS, (config.send_places, config.probability_places,
                                                           config.discount_places), strict=True)):
        threshold = config.min_redemptions if index == 2 else config.min_requests
        size = "used" if index == 2 else "requests"
        source, level = counts, group
        reason = None
        if getattr(counts, size) < threshold:
            source, level = population, "population"
            reason = f"insufficient {size} in {group}"
        exact = source.ratios()[index]
        if getattr(source, size) < threshold:
            values.append(None)
            details[key] = dict(level="withheld", reason=f"insufficient population {size}",
                                exact=None if exact is None else str(exact))
        else:
            value = _ceil(exact, places)
            error = Fraction(value) - exact
            _require(error >= 0, "component rounding underestimated " + key)
            values.append(value)
            details[key] = dict(level=level, reason=reason, exact=str(exact), output=str(value),
                                error=str(error))
    if None not in values:
        raw = [Fraction(details[k]["exact"]) for k in COMPONENTS]
        exact = raw[0] + raw[1] * raw[2]
        rounded = Fraction(values[0]) + Fraction(values[1]) * Fraction(values[2])
        total = _ceil(rounded, config.decimal_places)
        _require(Fraction(total) == rounded and rounded >= exact, "invalid combined cost rounding")
        budget._ticks(budget._decimal(total, "expected_cost"), config.decimal_places, "expected_cost")
        details["expected_cost"] = dict(exact=str(exact), output=str(total), error=str(rounded - exact),
                                        mixed_levels=len({details[k]["level"] for k in COMPONENTS}) > 1)
    return values, details


def estimate_coupon_costs(exposures, order_lines, historical_features, target_features, *,
                          cohort: HistoricalCohort, config: EstimationConfig, frozen_at,
                          development_features=None):
    """Produce a complete common cost table or withhold it for the entire target roster."""
    freeze = budget._clock(frozen_at, "frozen_at")
    budget._text(config.version, "version")
    budget._text(cohort.billing_contract, "billing_contract")
    budget._text(cohort.policy_id, "policy_id")
    _flag(cohort.failure_charge_allowed, "failure_charge_allowed")
    _require(isinstance(cohort.manifest_sha256, str) and re.fullmatch("[a-f0-9]{64}", cohort.manifest_sha256) is not None,
             "invalid manifest SHA")
    for name in ("min_requests", "min_redemptions"):
        _integer(getattr(config, name), name, 1)
    places = {name: _integer(getattr(config, name), name)
              for name in ("send_places", "probability_places", "discount_places", "decimal_places")}
    _require(max(places.values()) <= 9, "unsupported precision")
    config = replace(config, **places)
    _require(max(config.send_places, config.probability_places + config.discount_places) <= config.decimal_places,
             "precision cannot guarantee exact budget ticks")
    threshold = config.purchase_count_30d_threshold
    if threshold is not None:
        _integer(threshold, "purchase_count_30d_threshold")
    start = budget._clock(cohort.observation_start_at, "observation_start_at")
    end = budget._clock(cohort.observation_end_at, "observation_end_at")
    _require(start < end <= freeze, "historical observation boundary after freeze")
    ledger = _frame(exposures, EXPOSURE_COLUMNS)
    units = set()
    for frame in (historical_features, ledger, target_features):
        _require(isinstance(frame, pd.DataFrame) and list(frame.columns).count("amount_unit") == 1,
                 "missing or duplicate amount_unit")
        units.update(budget._decimal(v, "amount_unit", positive=True) for v in frame.amount_unit)
    _require(len(units) == 1, "inconsistent original amount_unit")
    # Budget accepts up to 64 significant digits. Preserve them through Feature normalization.
    with localcontext() as context:
        context.prec = 64
        history = validate_feature_frame(historical_features)
        targets = validate_feature_frame(target_features)
    _require(targets.as_of_at.nunique() == 1 and freeze <= targets.as_of_at.iloc[0], "invalid decision cutoff")
    _require(history.dataset_id.iloc[0] == cohort.dataset_id
             and history.raw_database_sha256.iloc[0] == cohort.raw_database_sha256, "training provenance mismatch")
    _require(not set(history.customer_id) & set(targets.customer_id)
             and cohort.raw_database_sha256 != targets.raw_database_sha256.iloc[0], "training/evaluation overlap")
    development = None
    if development_features is not None:
        dev = validate_feature_frame(development_features)
        _require(not set(dev.customer_id).intersection(set(history.customer_id) | set(targets.customer_id))
                 and dev.raw_database_sha256.iloc[0] not in (cohort.raw_database_sha256, targets.raw_database_sha256.iloc[0]),
                 "development cohort overlap")
        _require(dev.as_of_at.le(freeze).all(), "future development snapshot")
        development = dict(dataset_id=dev.dataset_id.iloc[0], raw_database_sha256=dev.raw_database_sha256.iloc[0],
                           customers=dev.customer_id.nunique())
    _require(history.currency.iloc[0] == targets.currency.iloc[0]
             and history.amount_unit.iloc[0] == targets.amount_unit.iloc[0], "target money mismatch")
    optional = {"request_id", "failure_reason", "coupon_id", "sent_at", "expires_at", "discount_rate",
                "redemption_order_id", "redemption_at"}
    _require(not ledger.drop(columns=list(optional)).isna().any().any(), "missing assignment or cost evidence")
    _require(len(ledger) == _integer(cohort.assignment_count, "assignment_count")
             and len(ledger) == len(history), "incomplete assignment/snapshot roster")
    ledger["as_of_at"] = [budget._clock(v, "as_of_at") for v in ledger.as_of_at]
    ledger["amount_unit"] = history.amount_unit.iloc[0]
    for col in ("request_id", "coupon_id", "redemption_order_id"):
        _require(not ledger[col].dropna().duplicated().any(), f"duplicate {col}")
    _require(not ledger.duplicated(["customer_id", "as_of_at"]).any(), "duplicate assignment snapshot")
    ledger = ledger.merge(history[[*CONTEXT_COLUMNS, "purchase_count_30d"]], on=list(CONTEXT_COLUMNS),
                          how="left", validate="one_to_one", indicator=True)
    _require(ledger._merge.eq("both").all(), "historical snapshot/provenance mismatch")
    assigned = [budget._clock(v, "assigned_at") for v in ledger.assigned_at]
    _require(not ledger.assign(assigned_at=assigned).duplicated(["customer_id", "assigned_at"]).any(),
             "duplicate customer assignment")
    lines = _frame(order_lines, ORDER_COLUMNS)
    _require(not lines.isna().any().any(), "missing order evidence")
    lines["paid_minor"] = [budget._ticks(budget._decimal(v, "paid_minor"), 0, "paid_minor")
                           for v in lines.paid_minor]
    orders = {order_id: items for order_id, items in lines.groupby("order_id", sort=False)}
    groups, population = defaultdict(_Counts), _Counts()
    for row in ledger.itertuples(index=False):
        action, *observed = _exposure(row, orders, cohort, start, end, freeze)
        if action:
            population.record(*observed)
            groups[_group(row.purchase_count_30d, threshold)].record(*observed)
    _require(not orders, "unlinked redemption order")
    rows, customers, estimates = [], [], {}
    for row in targets.itertuples(index=False):
        group = _group(row.purchase_count_30d, threshold)
        if group not in estimates:
            estimates[group] = _estimate_group(group, groups[group], population, config)
        values, _ = estimates[group]
        customers.append(dict(customer_id=row.customer_id, snapshot_id=row.snapshot_id, group=group,
                              status="withheld" if None in values else "complete"))
        if None not in values:
            rows.append((*[getattr(row, k) for k in CONTEXT_COLUMNS], *values, cohort.manifest_sha256,
                         config.version, freeze))
    status = "complete" if len(rows) == len(targets) else "withheld"
    metadata = CostMetadata("request_expected_coupon_cost", config.version, int(config.decimal_places), freeze)
    audit = dict(method="completed_request_means", version=config.version, population_definition="all coupon requests including failures",
                 status=status, frozen_at=freeze.isoformat(), development=development, training=dict(dataset_id=cohort.dataset_id,
                 raw_database_sha256=cohort.raw_database_sha256, manifest_sha256=cohort.manifest_sha256,
                 start=start.isoformat(), end=end.isoformat(), assignments=len(history), policy_id=cohort.policy_id,
                 billing_contract=cohort.billing_contract, failure_charge_allowed=bool(cohort.failure_charge_allowed)),
                 target=dict(dataset_id=targets.dataset_id.iloc[0], as_of_at=targets.as_of_at.iloc[0].isoformat(),
                 raw_database_sha256=targets.raw_database_sha256.iloc[0], currency=targets.currency.iloc[0],
                 amount_unit=targets.amount_unit.iloc[0]), config={k: int(v) if isinstance(v, Integral) else v
                 for k, v in asdict(config).items()}, rounding="ceiling_each_component", population=population.audit(),
                 groups={g: c.audit() for g, c in groups.items()}, applied={g: e[1] for g, e in estimates.items()},
                 customers=customers, independence_verified=False, predictive_performance_verified=False,
                 dispatch_authorized=False)
    return CostEstimate(status, metadata, tuple(rows) if status == "complete" else (),
                        json.dumps(audit, allow_nan=False))
