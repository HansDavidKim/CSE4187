"""A descriptive O2O response scenario, separate from identified causal effects."""
import hashlib
import json
import math

import numpy as np
import pandas as pd


def validate_response_model(model):
    if (not isinstance(model, dict) or type(model.get("schema_version")) is not int
            or model.get("schema_version") != 1 or model.get("type") != "tianchi_o2o_descriptive_response"
            or model.get("discount_rate") != .1 or model.get("window_days") != 30
            or model.get("inactivity_days") != 28):
        raise ValueError("incompatible Tianchi coupon response artifact")
    if (not isinstance(model.get("groups"), dict) or
            not {"pooled", "no_observed_purchase", "purchased_within_28d", "lapsed_purchase"} <= model["groups"].keys()):
        raise ValueError("Tianchi response artifact is missing purchase-history cohorts")
    for group in model["groups"].values():
        if not isinstance(group, dict):
            raise ValueError("Tianchi response cohorts must be objects")
        receipts, redeemed = group.get("receipts"), group.get("redeemed")
        probability = group.get("response_probability")
        delays = group.get("redemption_delay_days")
        histogram = group.get("redemption_delay_day_counts")
        if (type(receipts) is not int or type(redeemed) is not int or not 0 <= redeemed <= receipts
                or not isinstance(probability, (int, float)) or isinstance(probability, bool)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("invalid observed Tianchi coupon counts, probability or delays")
        if histogram is None:
            if (not isinstance(delays, list) or len(delays) != redeemed
                    or any(type(day) is not int or not 0 <= day < 30 for day in delays)):
                raise ValueError("invalid observed Tianchi coupon delays")
        elif (delays is not None or not isinstance(histogram, list) or len(histogram) != 30
              or any(type(count) is not int or count < 0 for count in histogram)
              or sum(histogram) != redeemed):
            raise ValueError("coupon redemption day counts must account for all redemptions")
        if receipts and not math.isclose(probability, redeemed / receipts, abs_tol=1e-12):
            raise ValueError("Tianchi response probability must equal observed redeemed/receipts")
    if not model["groups"]["pooled"]["receipts"]:
        raise ValueError("Tianchi response artifact has no supported receipts")


def sample_redemption_delay(group, rng):
    if histogram := group.get("redemption_delay_day_counts"):
        index = int(rng.integers(group["redeemed"]))
        return int(np.searchsorted(np.cumsum(histogram), index, side="right"))
    return int(rng.choice(group["redemption_delay_days"]))


def sample_coupon_response(model, seed, customer_id, as_of, last_purchase):
    """Use history at sending and keyed noise; never resample after a rejection."""
    cohort = ("no_observed_purchase" if last_purchase is None else
              "purchased_within_28d" if as_of - last_purchase <= pd.Timedelta(days=28) else "lapsed_purchase")
    scope = cohort if model["groups"][cohort]["receipts"] else "pooled"
    group = model["groups"][scope]
    key = json.dumps(["tianchi-coupon-response-v1", int(seed), customer_id, as_of.isoformat()]).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    uniform = float(rng.random())
    selected = uniform < group["response_probability"]
    delay = sample_redemption_delay(group, rng) if selected else None
    return {"customer_id": customer_id, "as_of_at": as_of, "history_cohort": cohort,
            "source_cohort": scope, "response_probability": group["response_probability"],
            "response_uniform": uniform, "response_selected": selected,
            "source_delay_days": delay,
            "response_ready_at": as_of + pd.Timedelta(days=delay) if selected else pd.NaT,
            "opportunity_at": pd.NaT, "baseline_action": None, "purchase_event_id": None,
            "promoted_purchase": False, "coupon_used": False}
