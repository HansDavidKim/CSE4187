"""The same customer, source ledger and pre-assignment weekly cutoff for every model."""
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

KEYS = ["dataset_id", "customer_id", "as_of_at"]
PROVENANCE = ["snapshot_id", "reference_week", "currency", "amount_unit", "raw_database_sha256"]


class InputValidationError(ValueError):
    """A model or profile declaration violates the strategy input contract."""


def require(condition, message):
    if not condition:
        raise InputValidationError(message)


def finite_number(value, name, *, minimum=0.):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
            and np.isfinite(value) and value >= minimum, f"{name} must be finite and >= {minimum}")
    return float(value)


def utc(values, name):
    require(not values.isna().any() and values.map(lambda x: pd.Timestamp(x).tzinfo is not None).all(),
            f"{name} must contain timezone-aware timestamps")
    return pd.to_datetime(values, utc=True, errors="raise")


def keys(frame):
    require(not frame.columns.duplicated().any(), "Duplicate input columns")
    require(set(KEYS) <= set(frame), f"Input requires {KEYS}")
    result = frame.copy()
    for name in ("dataset_id", "customer_id"):
        require(result[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                f"{name} must be nonempty strings")
    result["as_of_at"] = utc(result.as_of_at, "as_of_at")
    require(not result.duplicated(KEYS).any(), "Duplicate customer/cutoff keys")
    require(len(result) > 0 and result.dataset_id.nunique() == result.as_of_at.nunique() == 1,
            "One nonempty dataset and cutoff are required per decision budget")
    local = result.as_of_at.dt.tz_convert("Asia/Seoul")
    require((local.dt.weekday.eq(0) & local.eq(local.dt.normalize())).all(),
            "as_of_at must be Monday 00:00 Asia/Seoul")
    return result


@dataclass(frozen=True)
class PredictionMetadata:
    model_version: str
    target_definition_id: str
    horizon_value: int
    horizon_unit: str
    value_semantics: str

    def validate(self, role):
        require(bool(self.model_version) and bool(self.target_definition_id), "Model/target version is required")
        expected = 52 if role == "clv" else 4
        require(type(self.horizon_value) is int and self.horizon_value == expected and self.horizon_unit == "weeks",
                f"{role} requires exactly {expected} weeks")
        if role == "churn":
            require(any(term in self.target_definition_id for term in ("no_session", "no_visit", "churn", "no_active_date")),
                    "churn target must explicitly describe inactivity risk")
            require(self.value_semantics == "probability", "churn values must be probabilities")
        elif role == "uplift":
            require("purchase" in self.target_definition_id and self.value_semantics in ("probability", "probability_pair"),
                    "p0/p1 must predict 4-week purchase, not churn or latent response types")
        else:
            target = "clv_paid_amount_52w_v1" if role == "clv" else "gross_amount_4w_if_purchase_v1"
            require(self.target_definition_id == target and self.value_semantics == "nonnegative_amount",
                    f"{role} requires {target} nonnegative amounts")


def numeric(values, name, *, probability=False):
    result = pd.to_numeric(values, errors="raise").astype(float)
    observed = result.notna()
    require(np.isfinite(result[observed]).all(), f"{name} contains infinite values")
    require(result[observed].ge(0).all() and (not probability or result[observed].le(1).all()),
            f"{name} must be {'a probability in [0,1]' if probability else 'nonnegative'} or unavailable")
    return result
