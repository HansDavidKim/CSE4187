"""Declare human-designed retention hypotheses for the existing policy search APIs."""
from collections.abc import Mapping
from dataclasses import asdict, replace
from decimal import Decimal
import json

from .budget import _decimal, _text
from .policy_candidates import CandidateSet
from .policy_search import PolicyCandidate
from .retention_strategy import default_segment_policies
from .segmentation import SegmentationConfig
from .validation import InputValidationError


_SEGMENTATION = SegmentationConfig(churn_threshold=0.5, uplift_threshold=0.0, high_clv_fraction=0.2)
_GROUPS = (("high", "high_value"), ("high", "standard_value"),
           ("low", "high_value"), ("low", "standard_value"))


class _StrategyProfiles(CandidateSet):
    """Keep the declaration as immutable JSON; each report access gets its own tree."""

    def __init__(self, policies, report):
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "_report_json", json.dumps(report, allow_nan=False))

    @property
    def report(self):
        return json.loads(self._report_json)

    def to_dict(self):
        return self.report


def _threshold(value, field):
    cut = _decimal(value, field)
    if cut > 1:
        raise InputValidationError(f"{field} must be in [0,1]")
    return cut


def build_strategy_profiles(*, version: str, threshold_source: str, common, risk_high, risk_low,
                            high_value, standard_value, h4_r: Mapping,
                            segmentation: SegmentationConfig = _SEGMENTATION) -> CandidateSet:
    """Build P0 and H0–H4-V for one comparison, with no fitted or default cutoffs.

    Supply H4-R as {a, b, c, d} in high-risk/high-value, high/standard,
    low/high, low/standard order. H4-V exchanges b and c so only the
    preference between the two middle groups changes. All gates use strict >.
    """
    _text(version, "profile version")
    _text(threshold_source, "threshold source")
    if not isinstance(segmentation, SegmentationConfig) or segmentation != _SEGMENTATION:
        raise InputValidationError("strategy profiles require the common development SegmentationConfig")
    common = _threshold(common, "common")
    risk_high, risk_low = _threshold(risk_high, "risk_high"), _threshold(risk_low, "risk_low")
    high_value = _threshold(high_value, "high_value")
    standard_value = _threshold(standard_value, "standard_value")
    if common == 0:
        raise InputValidationError("H1 requires common > 0")
    if risk_high >= risk_low:
        raise InputValidationError("H2 requires risk_high < risk_low")
    if high_value >= standard_value:
        raise InputValidationError("H3 requires high_value < standard_value")
    if not isinstance(h4_r, Mapping) or set(h4_r) != {"a", "b", "c", "d"}:
        raise InputValidationError("H4-R requires exactly a, b, c, d")
    a, b, c, d = (_threshold(h4_r[key], f"H4-R.{key}") for key in "abcd")
    if not a <= b < c <= d:
        raise InputValidationError("H4-R requires a <= b < c <= d")

    baseline = PolicyCandidate("P0", version, "Development positive-uplift baseline",
                               "default_segment_policies", _SEGMENTATION, default_segment_policies())
    zero = (Decimal(0),) * 4
    declarations = (
        ("P0", baseline.description, "Existing development reference", "none", zero),
        ("H0", "Consider all eligible positive uplift", "Reproduce P0 as the human reference", "none", zero),
        ("H1", "Require a common minimum purchase uplift", "Screen out small purchase effects", "none",
         (common,) * 4),
        ("H2", "Lower the entry threshold for high churn risk", "Test a risk-specific retention gate", "risk",
         (risk_high, risk_high, risk_low, risk_low)),
        ("H3", "Lower the entry threshold for high CLV", "Test a value gate in addition to continuous CLV",
         "value", (high_value, standard_value, high_value, standard_value)),
        ("H4-R", "Favor risk within mixed risk/value groups", "Admit high-risk standard-value customers earlier",
         "risk_first", (a, b, c, d)),
        ("H4-V", "Favor value within mixed risk/value groups", "Admit low-risk high-value customers earlier",
         "value_first", (a, c, b, d)),
    )
    candidates, entries = [], {}
    for policy_id, hypothesis, rationale, preference, cuts in declarations:
        thresholds = dict(zip(_GROUPS, cuts, strict=True))
        source = baseline.source if policy_id in ("P0", "H0") else threshold_source
        rules = baseline.policies
        if policy_id not in ("P0", "H0"):
            rules = []
            for rule in baseline.policies:
                risk, _, value = rule.segment_id.split("__")
                rules.append(replace(rule, policy_id=policy_id, version=version, min_uplift=thresholds[risk, value]))
        candidate = replace(baseline, policy_id=policy_id, description=hypothesis, source=source, policies=tuple(rules))
        candidates.append(candidate)
        fingerprint = candidate.configuration_fingerprint()
        entries.setdefault(fingerprint, {})[policy_id] = dict(
            version=version, hypothesis=hypothesis, rationale=rationale, preference=preference,
            threshold_source=source, thresholds={"__".join(group): str(t) for group, t in thresholds.items()},
        )
    return _StrategyProfiles(tuple(candidates), dict(policies=entries, segmentation=asdict(_SEGMENTATION),
                                                     policy_effect_verified=False, dispatch_authorized=False))
