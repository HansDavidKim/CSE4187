"""Observed 4-week outcomes by receipt, not a causal intervention estimate."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/uplift-sequence-pilot"
OUT.mkdir(parents=True, exist_ok=True)


def run():
    cohort = pd.read_csv(ROOT / "artifacts/canonical-type-bounds/tianchi_cohort.csv")
    assert cohort.User_id.is_unique
    users = pd.Index(cohort.User_id)
    # One row per user; action and week counts include later coupon receipts.
    records = []
    source = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    for c in pd.read_csv(source, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        c = c[c.User_id.isin(users)].copy()
        date = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        receipt = c.Action.eq(2)
        date.loc[receipt] = pd.to_datetime(c.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (date - pd.Timestamp("2016-05-15")).dt.days
        mask = delta.between(1, 28) & c.Action.isin([0, 1, 2])
        c = c[mask].copy()
        c["week"] = ((delta[mask] - 1) // 7 + 1).astype(int)
        records.append(c.groupby(["User_id", "week", "Action"]).size())
    wide = pd.concat(records).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"w{week}_a{action}" for week, action in wide.columns]
    frame = cohort.join(wide, on="User_id").fillna(0)
    result = []
    for (group, treated), part in frame.groupby(["group", "treated"]):
        item = {"group": group, "receipt_on_anchor": int(treated), "n": len(part), "weeks": []}
        for week in range(1, 5):
            item["weeks"].append({"week": week, "click_any": float(part[f"w{week}_a0"].gt(0).mean()),
                                   "purchase_any": float(part[f"w{week}_a1"].gt(0).mean()),
                                   "another_receipt_any": float(part[f"w{week}_a2"].gt(0).mean())})
        result.append(item)
    (OUT / "long_term_observed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
