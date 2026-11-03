"""Seven-table synthetic contract fixture, NOT a calibrated user simulator."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .pipeline import local_output


def generate(directory, customers=120, seed=42):
    directory = local_output(directory)
    if any(directory.glob("*.parquet")) or any(directory.glob("*.csv")):
        raise ValueError("Fixture output must be an empty data directory; refusing to overwrite records")
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-10-01")
    rows = {n: [] for n in ("customers", "sessions", "events", "orders", "items", "coupons", "redemptions")}
    for user in range(customers):
        uid = f"u{user:04d}"
        rows["customers"].append({"customer_id": uid, "signed_up_at": start})
        probability = float(rng.uniform(.008, .025) if user % 3 == 0 else rng.uniform(.06, .25))
        unused = []
        for day in range((end - start).days):
            when = start + pd.Timedelta(days=day, hours=int(rng.integers(8, 22)))
            if day % 7 == 0 and rng.random() < .3:
                cid = f"c{user}_{day}"
                rows["coupons"].append({"coupon_id": cid, "customer_id": uid, "policy": "fixed_10_percent",
                    "sent_at": when - pd.Timedelta(hours=1), "expires_at": when - pd.Timedelta(hours=1) + pd.Timedelta(days=42)})
                unused.append((cid, when - pd.Timedelta(hours=1) + pd.Timedelta(days=42)))
            if rng.random() >= probability:
                continue
            sid = f"s{user}_{day}"
            purchase = rng.random() < .4
            actions = ["view", "cart"] + (["purchase"] if purchase else ["remove_from_cart"])
            for i, action in enumerate(actions):
                rows["events"].append({"event_id": f"e{user}_{day}_{i}", "customer_id": uid,
                    "session_id": sid, "event_at": when + pd.Timedelta(seconds=i * 30), "event_type": action})
            rows["sessions"].append({"session_id": sid, "customer_id": uid,
                "started_at": when, "ended_at": when + pd.Timedelta(seconds=90)})
            if purchase:
                oid = f"o{user}_{day}"
                rows["orders"].append({"order_id": oid, "customer_id": uid, "session_id": sid,
                    "purchased_at": when + pd.Timedelta(seconds=60)})
                for product in range(int(rng.integers(1, 4))):
                    rows["items"].append({"order_id": oid, "product_id": f"p{product}",
                        "unit_price": float(rng.integers(100, 3000)) / 100, "quantity": int(rng.integers(1, 3))})
                unused = [(cid, expiry) for cid, expiry in unused if expiry > when + pd.Timedelta(seconds=60)]
                if unused:
                    cid, _ = unused.pop(0)
                    rows["redemptions"].append({"coupon_id": cid, "order_id": oid})
    for name, data in rows.items():
        pd.DataFrame(data).to_parquet(directory / f"{name}.parquet", index=False)
    config = {"input_dir": str(directory), "output_dir": str(directory.parent / "smoke_run"),
        "first_as_of": "2024-03-01", "last_as_of": "2024-08-30", "observation_end": "2024-10-01",
        "test_start": "2024-08-15", "observation_assumption": "Contract fixture explicitly generated complete records through 2024-10-01; no missing collection intervals.",
        "data_role": "synthetic_contract_smoke_only", "target": "visit", "horizon_days": 30, "lookback_days": 30,
        "stride_days": 7, "minimum_age_days": 30, "seed": seed, "trials": 2, "iterations": 60,
        "epochs": 8, "hidden": 24, "batch_size": 128, "max_length": 128, "bootstrap_repeats": 30}
    path = directory / "smoke_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
