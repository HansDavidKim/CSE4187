"""Checkout integration uses fitted synthetic observations, never production artifacts."""
from collections import Counter
import copy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from simulate.v4.irl.fit_shopping import ACTIONS, fit_shopping
from simulate.v4.irl.fit_return import FEATURE_NAMES as RETURN_FEATURE_NAMES
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, discount_minor, load_raw_dataset, validate_raw_dataset


@pytest.fixture(scope="module")
def checkout_fixture(tmp_path_factory):
    rows = []
    sequence = [("view", "A"), ("cart", "A"), ("cart", "A"), ("remove_from_cart", "A"),
                ("view", "B"), ("cart", "B"), ("purchase", "B"), ("purchase", "A"), ("view", "A")]
    for user in range(40):
        for session, day in enumerate(("2020-01-01", "2020-01-03", "2020-02-15")):
            for step, (action, product) in enumerate(sequence):
                at = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=user % 20, seconds=(6 if step == 7 else step) * 5)
                rows.append(dict(event_time=at.isoformat(), event_type=action, product_id=product,
                                 category_id=f"category-{product}", category_code="", brand="",
                                 price="1.05" if product == "A" else "2.50", user_id=str(user),
                                 user_session=f"{user}:{session}"))
    model = fit_shopping(pd.DataFrame(rows), {"dataset": "synthetic checkout wiring fixture"}, checkout_episodes=True)
    folder = tmp_path_factory.mktemp("checkout-simulator")
    def write(name, value):
        path = folder / name
        path.write_text(json.dumps(value, allow_nan=False))
        return str(path)
    features = len(RETURN_FEATURE_NAMES)
    return_model = {"schema_version": 1, "feature_names": RETURN_FEATURE_NAMES,
                    "scaler_mean": [0.] * features, "scaler_scale": [1.] * features,
                    "feature_min": [0.] * features, "feature_max": [100.] * features,
                    "coef": [0.] * features, "intercept": math.log(4),
                    "cold_start_probability": .8, "calibration_reference_probabilities": [.8]}
    config = {"simulation": {"seed": 17, "start_at": "2024-01-01T00:00:00+00:00"},
              "churn": {"target_rate": .2},
              "source_simulation": {"model_path": write("shopping.json", model),
                                    "return_model_path": write("return.json", return_model),
                                    "max_events_per_session": 64, "outcome_days": 28,
                                    "coupon_discount": .1, "coupon_send_cost_minor": 100},
              "toy": {"message_model_path": write("message.json", {
                  "response_lift": {"positive": .1, "neutral": 0., "negative": -.1},
                  "visit_lift_scale": 1., "observed": {"pooled": {"visit_logit_shift": 0.}}}),
                      "coupon_model_path": write("coupon.json", {
                          "groups": {"baseline": {"coupon_use_probability": 1.}}})}}
    return config, model


def _customers(count=8):
    # Every supplied signup lies in the eligible history window; no rejected donor traces.
    return pd.DataFrame({"customer_id": [f"fixture-{i:02}" for i in range(count)],
                         "signed_up_at": pd.Timestamp("2024-01-10", tz="UTC")})


def _basket(items):
    result = Counter()
    for item in items.itertuples(index=False):
        result[item.product_id] += int(item.quantity)
    return result
