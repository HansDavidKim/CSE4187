"""Compare observed pre/post activity, without assigning cross-domain GRU clusters."""
import json

import numpy as np
import pandas as pd
from ecommerce_sim.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "artifacts/coupon_persona_patterns"
OUT.mkdir(parents=True, exist_ok=True)
ONLINE = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"


def online():
    # Pick a receipt in May; both 14-day windows then lie within the logged Jan–June period.
    receipts, activities = [], []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date_received"], dtype={"Date_received": "string"}, chunksize=1_000_000):
        dates = pd.to_datetime(chunk.Date_received, format="%Y%m%d", errors="coerce")
        chosen = chunk.Action.eq(2) & dates.between("2016-05-01", "2016-05-31")
        receipts.append(pd.DataFrame({"user": chunk.loc[chosen, "User_id"], "anchor": dates[chosen]}))
    candidates = pd.concat(receipts, ignore_index=True).sort_values(["user", "anchor"])
    anchors = candidates.drop_duplicates("user").reset_index(drop=True)
    assert len(anchors) > 0, "No eligible receipts: check dates and schema"
    users = set(anchors.user)
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"], dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        chunk = chunk[chunk.User_id.isin(users)].copy()
        chunk["day"] = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        is_receipt = chunk.Action.eq(2)
        chunk.loc[is_receipt, "day"] = pd.to_datetime(chunk.loc[is_receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        chunk = chunk.merge(anchors, left_on="User_id", right_on="user")
        delta = (chunk.day - chunk.anchor).dt.days
        chunk["period"] = np.select([delta.between(-14, -1), delta.between(1, 14)], ["pre", "post"], default="outside")
        chunk = chunk[chunk.period.ne("outside") & chunk.Action.isin([0, 1, 2])]
        activities.append(chunk.groupby(["user", "period", "Action"]).size())
    counts = pd.concat(activities).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    counts.columns = [f'{period}_{action}' for period, action in counts.columns]
    pairs = anchors.join(counts, on="user").fillna(0)
    for period in ["pre", "post"]:
        for action in [0, 1, 2]:
            col = f"{period}_{action}"
            if col not in pairs:
                pairs[col] = 0
            pairs[col] = pairs[col].astype(int)
    pre_click, pre_buy = pairs.pre_0.gt(0), pairs.pre_1.gt(0)
    pairs["group"] = np.select([pre_click & ~pre_buy, pre_click & pre_buy, ~pre_click & pre_buy],
        ["click_only", "click_and_purchase", "purchase_only"], default="neither_recorded")
    post_click, post_buy = pairs.post_0.gt(0), pairs.post_1.gt(0)
    pairs["post_group"] = np.select([post_click & ~post_buy, post_click & post_buy, ~post_click & post_buy],
        ["click_only", "click_and_purchase", "purchase_only"], default="neither_recorded")
    pairs.to_csv(OUT / "tianchi_user_pairs.csv", index=False)
    assert len(pairs) == pairs.user.nunique()
    report = []
    for name, g in pairs.groupby("group"):
        report.append({"group": name, "n": len(g), "pre_click_mean": g.pre_0.mean(), "post_click_mean": g.post_0.mean(),
                       "pre_purchase_mean": g.pre_1.mean(), "post_purchase_mean": g.post_1.mean(),
                       "post_purchase_users": int(g.post_1.gt(0).sum()), "post_click_users": int(g.post_0.gt(0).sum()),
                       "post_more_clicks": int(g.post_0.gt(g.pre_0).sum()),
                       "post_fewer_clicks": int(g.post_0.lt(g.pre_0).sum()),
                       "another_post_receipt": int(g.post_2.gt(0).sum()),
                       "post_groups": {str(k): int(v) for k, v in g.post_group.value_counts().items()},
                       "no_later_receipt_n": int(g.post_2.eq(0).sum()),
                       "no_later_receipt_post_purchase_users": int((g.post_2.eq(0) & g.post_1.gt(0)).sum())})
    return {"users": len(pairs), "groups": report, "other_post_receipt_users": int(pairs.post_2.gt(0).sum())}


def purchase_history(dataset):
    p = pd.read_csv(ROOT / f"artifacts/coupon_before_after/{dataset}_pairs.csv")
    p["prior_purchase_count"] = pd.cut(p.pre, [-1, 0, 1, float("inf")], labels=["0", "1", "2+"])
    result = []
    for name, g in p.groupby("prior_purchase_count", observed=True):
        result.append({"prior_count": str(name), "n": len(g), "post_any": int(g.post.gt(0).sum()),
                       "post_more": int(g.post.gt(g.pre).sum()), "post_fewer": int(g.post.lt(g.pre).sum()),
                       "pre_mean": g.pre.mean(), "post_mean": g.post.mean()})
    return result


if __name__ == "__main__":
    result = {"tianchi": online(), "starbucks": purchase_history("starbucks"),
              "complete_journey": purchase_history("complete_journey")}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
