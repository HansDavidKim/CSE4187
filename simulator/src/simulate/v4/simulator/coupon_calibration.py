"""Calibrate campaign effects to aggregate experimental contrasts."""
from math import isfinite

import numpy as np
from scipy.special import expit


def validate_effect_calibration(calibration, coupon_days):
    if not isinstance(calibration, dict):
        raise ValueError("coupon effect calibration must be a mapping")
    for name in ("contact_days", "price_days"):
        days = calibration.get(name)
        if type(days) is not int or not 1 <= days <= coupon_days:
            raise ValueError(f"{name} must be within the coupon window")
    for name in ("contact_visit_lift", "induced_purchase_log_odds", "price_purchase_log_odds",
                 "redemption_odds_multiplier"):
        value = calibration.get(name)
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not -1 < calibration["contact_visit_lift"] < 1:
        raise ValueError("contact visit lift must be a risk difference")
    if calibration["redemption_odds_multiplier"] < 0:
        raise ValueError("redemption odds multiplier must be nonnegative")


def calibrate_contact_support(logits, timing_shares, support, weights, target):
    """Match the population ITT visit risk difference at the source horizon."""
    if target == 0:
        return np.zeros_like(support)
    baseline = expit(logits)
    lower, upper = -80.0, 80.0

    def risk_difference(shift):
        exposed = expit(logits[:, None] + support + shift) @ weights
        return float(np.mean(timing_shares * (exposed - baseline)))

    if not risk_difference(lower) <= target <= risk_difference(upper):
        raise ValueError("contact visit lift is outside the fitted visit clock's support")
    for _ in range(60):
        middle = (lower + upper) / 2
        if risk_difference(middle) < target:
            lower = middle
        else:
            upper = middle
    return support + (lower + upper) / 2
