import pandas as pd
import pytest
from retention.data import Records, normalize, validate, SCHEMA


@pytest.fixture
def records():
    t = {name: pd.DataFrame(columns=cols) for name, cols in SCHEMA.items()}
    t["customers"] = pd.DataFrame([{"customer_id": "u", "signed_up_at": "2024-01-01"}])
    t["sessions"] = pd.DataFrame([{"customer_id": "u", "session_id": "s", "started_at": "2024-02-01", "ended_at": "2024-02-01 00:02:00"}])
    t["events"] = pd.DataFrame([{"customer_id": "u", "session_id": "s", "event_id": str(i), "event_at": "2024-02-01 00:01:00", "event_type": a} for i, a in enumerate(["cart", "purchase"])])
    t["orders"] = pd.DataFrame([{"order_id": "o", "customer_id": "u", "session_id": "s", "purchased_at": "2024-02-01 00:01:00"}])
    t["items"] = pd.DataFrame([{"order_id": "o", "product_id": "p1", "unit_price": 10., "quantity": 2},
                               {"order_id": "o", "product_id": "p2", "unit_price": 5., "quantity": 1}])
    t["coupons"] = pd.DataFrame([{"coupon_id": "c", "customer_id": "u", "policy": "ten"}])
    t["redemptions"] = pd.DataFrame([{"coupon_id": "c", "order_id": "o"}])
    r = Records({k: normalize(v) for k, v in t.items()}, {"mode": "canonical"})
    validate(r)
    return r

