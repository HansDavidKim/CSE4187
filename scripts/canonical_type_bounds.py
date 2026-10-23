"""Exploratory risk differences and assumption-dependent joint potential-outcome bounds."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/canonical-type-bounds"
OUT.mkdir(parents=True, exist_ok=True)


def bounds(p0, p1):
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1):
        raise ValueError("Invalid marginal probability")
    low, high = max(0, p0 + p1 - 1), min(p0, p1)
    return {"sure_thing": [low, high], "persuadable": [p1 - high, p1 - low],
            "sleeping_dog": [p0 - high, p0 - low], "lost_cause": [1 - p0 - p1 + low, 1 - p0 - p1 + high]}


def report(df, dataset):
    results = []
    for group, part in df.groupby("group", observed=True):
        arms = part.groupby("treated").outcome.agg(["count", "sum", "mean"])
        if set(arms.index) != {0, 1}:
            continue
        p0, p1 = arms.loc[0, "mean"], arms.loc[1, "mean"]
        results.append({"group": str(group), "n0": int(arms.loc[0, "count"]), "y0": int(arms.loc[0, "sum"]),
                        "n1": int(arms.loc[1, "count"]), "y1": int(arms.loc[1, "sum"]),
                        "p0": p0, "p1": p1, "observed_difference": p1 - p0,
                        "conditional_bounds": bounds(p0, p1)})
    df.to_csv(OUT / f"{dataset}_cohort.csv", index=False)
    return results


def starbucks():
    raw = ROOT / "data/raw/starbucks"
    events = pd.read_json(raw / "transcript.json", lines=True)
    offers = events[events.event.eq("offer received")].copy()
    portfolio = pd.read_json(raw / "portfolio.json", lines=True).set_index("id").offer_type.to_dict()
    offers["offer_type"] = offers.value.map(lambda v: portfolio[v.get("offer id", v.get("offer_id"))])
    at = offers[offers.time.eq(168)].set_index("person").offer_type
    assert not at.index.duplicated().any()
    tx = events[events.event.eq("transaction")]
    before = tx[tx.time.gt(0) & tx.time.lt(168)].groupby("person").size()
    after = tx[tx.time.gt(168) & tx.time.lt(336)].groupby("person").size()
    cohort = pd.DataFrame({"user": pd.read_json(raw / "profile.json", lines=True).id})
    cohort["offer"] = cohort.user.map(at).fillna("none")
    cohort = cohort[cohort.offer.isin(["none", "discount", "bogo"])].copy()
    cohort["treated"] = cohort.offer.ne("none").astype(int)
    cohort["pre"] = cohort.user.map(before).fillna(0).astype(int)
    cohort["outcome"] = cohort.user.map(after).fillna(0).gt(0).astype(int)
    cohort["group"] = pd.cut(cohort.pre, [-1, 0, 1, float("inf")], labels=["0", "1", "2+"])
    return report(cohort, "starbucks")


def tianchi():
    file = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    counts = []
    for c in pd.read_csv(file, usecols=["User_id", "Action", "Date", "Date_received"],
                         dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        c["day"] = pd.to_datetime(c.Date, format="%Y%m%d", errors="coerce")
        is_offer = c.Action.eq(2)
        c.loc[is_offer, "day"] = pd.to_datetime(c.loc[is_offer, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (c.day - pd.Timestamp("2016-05-15")).dt.days
        c["period"] = np.select([delta.between(-14, -1), delta.eq(0), delta.between(1, 14)],
                                 ["pre", "anchor", "post"], default="outside")
        c = c[c.period.ne("outside") & c.Action.isin([0, 1, 2])]
        counts.append(c.groupby(["User_id", "period", "Action"]).size())
    combined = pd.concat(counts).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    combined.columns = [f"{a}_{b}" for a, b in combined.columns]
    for name in ["pre_0", "pre_1", "anchor_2", "post_1"]:
        if name not in combined:
            combined[name] = 0
    # Cohort eligibility uses pre-anchor clicks/purchases only, never future outcomes.
    combined = combined[(combined.pre_0 > 0) | (combined.pre_1 > 0)].copy()
    combined["group"] = np.select([(combined.pre_0 > 0) & (combined.pre_1 == 0),
                                   (combined.pre_0 > 0) & (combined.pre_1 > 0)],
                                  ["click_only", "click_purchase"], default="purchase_only")
    combined["treated"] = combined.anchor_2.gt(0).astype(int)
    combined["outcome"] = combined.post_1.gt(0).astype(int)
    return report(combined.reset_index(), "tianchi")


if __name__ == "__main__":
    result = {"starbucks": starbucks(), "tianchi": tianchi()}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
