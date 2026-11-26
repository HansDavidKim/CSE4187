"""Team declaration types from final_strategy 11a1fcd; no search or solver runtime.

Only the native profile dependencies are retained. The planning adapter uses
our validated purchase-probability pipeline, not the remote solver interfaces.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from itertools import product
import json
import math
import numbers
import numpy as np
import pandas as pd
from .validation import InputValidationError

MAX_AMOUNT = Decimal("1e18")
CHURN_BANDS = ("high", "low")
UPLIFT_BANDS = ("negative", "neutral_low", "positive")
CLV_BANDS = ("high_value", "standard_value")
SEGMENT_IDS = tuple("__".join(x) for x in product(CHURN_BANDS, UPLIFT_BANDS, CLV_BANDS))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"budget.{name}: requires nonempty string")
    return value


def _decimal(value, name, positive=False):
    try:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, Decimal, numbers.Real)):
            raise ValueError("unsupported decimal value")
        # For floats this preserves str(float), not an unknown original decimal.
        result = value if isinstance(value, Decimal) else Decimal(str(value))
        if not result.is_finite() or result < 0 or (positive and result == 0):
            raise ValueError("invalid sign or non-finite")
        if result > MAX_AMOUNT or len(result.as_tuple().digits) > 64:
            raise ValueError("unsupported decimal magnitude/length")
        return result
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise InputValidationError(f"budget.{name}: requires finite nonnegative decimal <= 1e18 (at most 64 digits)") from exc


@dataclass(frozen=True)
class SegmentationConfig:
    """Development defaults, not validated marketing policy thresholds."""

    churn_threshold: float = 0.5
    uplift_threshold: float = 0.0
    high_clv_fraction: float = 0.2
    segmentation_version: str = "segmentation-12-v1"

    def __post_init__(self):
        for name in ("churn_threshold", "uplift_threshold", "high_clv_fraction"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise InputValidationError(f"segmentation.{name}: must be finite and in [0,1]")
        if not isinstance(self.segmentation_version, str) or not self.segmentation_version.strip():
            raise InputValidationError("segmentation.segmentation_version: requires a nonempty string")


@dataclass(frozen=True)
class SegmentPolicy:
    segment_id: str
    policy_id: str = "development-positive-uplift"
    version: str = "retention-dev-v1"
    allow_candidate: bool = True
    min_uplift: object = Decimal(0)
    exclusion_reason: str = "policy_disabled"
    send_delay: object = pd.Timedelta(0)


def default_segment_policies():
    """All twelve labels supported; strict positive uplift, not business validation."""
    return tuple(SegmentPolicy(identifier) for identifier in SEGMENT_IDS)


def _fail(message):
    raise InputValidationError(f"retention: {message}")


def _duration(value, field):
    if isinstance(value, (bool, int, float, np.number)):
        _fail(f"{field} requires an explicit timedelta or duration string")
    try:
        result = pd.Timedelta(value)
        if pd.isna(result) or result < pd.Timedelta(0):
            raise ValueError()
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputValidationError(f"retention: invalid {field}") from exc


def _policies(policies):
    policies = default_segment_policies() if policies is None else tuple(policies)
    if any(not isinstance(p, SegmentPolicy) for p in policies):
        _fail("SegmentPolicy declarations required")
    identifiers = [p.segment_id for p in policies]
    if len(identifiers) != 12 or set(identifiers) != set(SEGMENT_IDS):
        _fail("exactly one policy for each of twelve segments required")
    for p in policies:
        for key in ("policy_id", "version", "exclusion_reason"):
            _text(getattr(p, key), f"policy.{key}")
        if not isinstance(p.allow_candidate, bool) or _decimal(p.min_uplift, "min_uplift") > 1:
            _fail("invalid candidate flag or uplift threshold")
        _duration(p.send_delay, "send_delay")
    return {p.segment_id: p for p in policies}


def _decimal_key(value):
    if value.is_zero():
        return ("0", 0)
    digits = "".join(map(str, value.as_tuple().digits)).rstrip("0")
    return (digits, value.adjusted())


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, np.floating):
        return _json_value(float(value))
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else (0.0 if value == 0 else value)
    return value


def _fingerprint(value):
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyCandidate:
    """One complete policy. Use per-segment rules to express differential targeting."""

    policy_id: str
    version: str
    description: str
    source: str
    segmentation: SegmentationConfig = SegmentationConfig()
    policies: tuple[SegmentPolicy, ...] = default_segment_policies()

    def __post_init__(self):
        for field in ("policy_id", "version", "description", "source"):
            _text(getattr(self, field), f"policy_candidate.{field}")
        if not isinstance(self.segmentation, SegmentationConfig):
            raise InputValidationError("policy_candidate: requires SegmentationConfig")
        if self.policies is None:
            raise InputValidationError("policy_candidate: declare all twelve segment policies")
        try:
            rules = _policies(self.policies)
        except TypeError as exc:
            raise InputValidationError("policy_candidate: policies must be iterable") from exc
        object.__setattr__(self, "policies", tuple(rules[key] for key in sorted(rules)))

    def configuration_fingerprint(self) -> str:
        """Hash behavior and exclusion rules, independent of declaration IDs and order."""
        config = asdict(self.segmentation)
        for key in ("churn_threshold", "uplift_threshold"):
            config[key] = float(config[key])
        config["high_clv_fraction"] = _decimal_key(Decimal(str(config["high_clv_fraction"])))
        rules = []
        for rule in self.policies:
            rules.append([
                rule.segment_id, rule.allow_candidate,
                _decimal_key(_decimal(rule.min_uplift, "min_uplift")),
                rule.exclusion_reason, _duration(rule.send_delay, "send_delay").isoformat(),
            ])
        return _fingerprint([config, rules])


@dataclass(frozen=True)
class CandidateSet:
    policies: tuple[PolicyCandidate, ...]
    report: dict

    def to_dict(self):
        return deepcopy(self.report)
