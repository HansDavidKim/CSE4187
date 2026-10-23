"""Explore logged activity after a quiet period; not a coupon causal effect."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/coupon-reactivation"
OUT.mkdir(parents=True, exist_ok=True)


def interval(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, mid-half), min(1, mid+half)]


def cohort_from_counts(data):
    prior = data["prior_0"].gt(0) | data["prior_1"].gt(0)
    quiet = data["quiet_0"].eq(0) & data["quiet_1"].eq(0)
    cohort = data.loc[prior & quiet].copy()
    clicks, buys = cohort.prior_0.gt(0), cohort.prior_1.gt(0)
    cohort["group"] = np.select([clicks & ~buys, clicks & buys], ["click_only", "click_purchase"], default="purchase_only")
    cohort["treated"] = cohort.anchor_2.gt(0).astype(int)
    return cohort


def run():
    chunks = []
    source = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    for c in pd.read_csv(source, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        receipt = c.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(c.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        diff = (day - pd.Timestamp("2016-05-15")).dt.days
        period = np.select([diff.between(-44, -15), diff.between(-14, -1), diff.eq(0),
                            diff.between(1, 7), diff.between(8, 14), diff.between(15, 21), diff.between(22, 28)],
                           ["prior", "quiet", "anchor", "w1", "w2", "w3", "w4"], default="outside")
        mask = (period != "outside") & c.Action.isin([0, 1, 2])
        data = c.loc[mask, ["User_id", "Action"]].copy()
        data["period"] = period[mask]
        chunks.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(chunks).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{p}_{a}" for p, a in wide.columns]
    for p in ["prior", "quiet", "anchor", "w1", "w2", "w3", "w4"]:
        for action in [0, 1, 2]:
            name = f"{p}_{action}"
            if name not in wide:
                wide[name] = 0
    cohort = cohort_from_counts(wide)
    assert cohort.index.is_unique
    result = []
    for (group, treated), part in cohort.groupby(["group", "treated"]):
        n = len(part)
        click = part[[f"w{w}_0" for w in range(1, 5)]].to_numpy() > 0
        buy = part[[f"w{w}_1" for w in range(1, 5)]].to_numpy() > 0
        any_action = click | buy
        first = np.where(any_action.any(axis=1), np.argmax(any_action, axis=1) + 1, 0)
        active28, purchased28 = int(any_action.any(axis=1).sum()), int(buy.any(axis=1).sum())
        first7 = int(any_action[:, 0].sum())
        later_receipt = part[[f"w{w}_2" for w in range(1, 5)]].gt(0).any(axis=1)
        result.append({"group": group, "anchor_receipt": int(treated), "n": n,
                       "return_7d": first7, "return_7d_ci": interval(first7, n),
                       "return_28d": active28, "return_28d_ci": interval(active28, n),
                       "purchase_28d": purchased28, "first_return_week_0_none": np.bincount(first, minlength=5).tolist(),
                       "prior_receipt_quiet": int(part.quiet_2.gt(0).sum()),
                       "later_receipt_28d": int(later_receipt.sum()),
                       "prior_click_events_mean": float(part.prior_0.mean()), "prior_purchase_events_mean": float(part.prior_1.mean())})
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
