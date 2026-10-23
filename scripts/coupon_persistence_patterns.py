"""Six-week observed event patterns, not causal effect durations."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/coupon-persistence"
OUT.mkdir(parents=True, exist_ok=True)


def summarize(bits):
    assert bits.ndim == 2 and bits.shape[1] == 6
    n = len(bits)
    count = bits.sum(axis=1)
    first = np.where(count > 0, np.argmax(bits, axis=1) + 1, 0)
    last = np.where(count > 0, 6 - np.argmax(bits[:, ::-1], axis=1), 0)
    return {"n": n, "weekly_n": bits.sum(axis=0).astype(int).tolist(),
            "active_weeks_distribution": np.bincount(count, minlength=7).astype(int).tolist(),
            "first_recorded_week_distribution_0_none": np.bincount(first, minlength=7).astype(int).tolist(),
            "last_recorded_week_distribution_0_none": np.bincount(last, minlength=7).astype(int).tolist(),
            "week1_users": int(bits[:, 0].sum()),
            "week1_and_week6_users": int((bits[:, 0] & bits[:, 5]).sum()),
            "week1_and_any_week2to6_users": int((bits[:, 0] & bits[:, 1:].any(axis=1)).sum()),
            "active_week6_right_boundary": int(bits[:, 5].sum())}


def run():
    cohort = pd.read_csv(ROOT / "artifacts/canonical-type-bounds/tianchi_cohort.csv")
    assert cohort.User_id.is_unique
    users = pd.Index(cohort.User_id)
    chunks = []
    source = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    for c in pd.read_csv(source, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        c = c[c.User_id.isin(users)].copy()
        day = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        receipt = c.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(c.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - pd.Timestamp("2016-05-15")).dt.days
        mask = delta.between(1, 42) & c.Action.isin([0, 1, 2])
        c = c[mask].copy()
        c["week"] = ((delta[mask] - 1) // 7 + 1).astype(int)
        chunks.append(c.groupby(["User_id", "week", "Action"]).size())
    totals = pd.concat(chunks).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    totals.columns = [f"w{w}_a{a}" for w, a in totals.columns]
    data = cohort.join(totals, on="User_id").fillna(0)
    result = []
    for (group, treated), part in data.groupby(["group", "treated"]):
        row = {"group": group, "receipt_on_anchor": int(treated), "n": len(part)}
        for action, label in [(0, "click"), (1, "purchase")]:
            bits = part[[f"w{w}_a{action}" for w in range(1, 7)]].to_numpy() > 0
            row[label] = summarize(bits)
            assert sum(row[label]["active_weeks_distribution"]) == len(part)
        # Receipt dates stop on June 15: week 5 is partial and week 6 unknown.
        row["later_receipt_week1to4_n"] = [int(part[f"w{w}_a2"].gt(0).sum()) for w in range(1, 5)]
        row["later_receipt_week5_partial_n"] = int(part["w5_a2"].gt(0).sum())
        row["later_receipt_week6_n"] = None
        result.append(row)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
