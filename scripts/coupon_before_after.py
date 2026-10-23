"""Descriptive paired windows; intervention and redemption are not equivalent."""
import json

import pandas as pd
from ecommerce_sim.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "artifacts" / "coupon_before_after"
OUT.mkdir(parents=True, exist_ok=True)


def summarize(name, pairs):
    delta = pairs.post - pairs.pre
    result = {
        "dataset": name, "n": len(pairs),
        "pre_mean": pairs.pre.mean(), "post_mean": pairs.post.mean(),
        "decrease": int((delta < 0).sum()), "same": int((delta == 0).sum()),
        "increase": int((delta > 0).sum()),
        "zero_to_positive": int(((pairs.pre == 0) & (pairs.post > 0)).sum()),
        "positive_to_zero": int(((pairs.pre > 0) & (pairs.post == 0)).sum()),
        "no_purchase_both": int(((pairs.pre == 0) & (pairs.post == 0)).sum()),
    }
    assert result["decrease"] + result["same"] + result["increase"] == result["n"]
    pairs.to_csv(OUT / f"{name}_pairs.csv", index=False)
    return result


def starbucks():
    raw = ROOT / "data/raw/starbucks"
    events = pd.read_json(raw / "transcript.json", lines=True)
    portfolio = pd.read_json(raw / "portfolio.json", lines=True)
    offer_type = portfolio.set_index("id").offer_type.to_dict()
    offers = events[events.event.eq("offer received")].copy()
    offers["type"] = offers.value.map(lambda x: offer_type[x.get("offer id", x.get("offer_id"))])
    # t=168 provides a fully observed 7-day baseline and follow-up for every user.
    anchors = offers[(offers.time == 168) & offers.type.isin(["discount", "bogo"])].copy()
    assert not anchors.person.duplicated().any()
    tx = events[events.event.eq("transaction")]
    # Exclude timestamp 168: the data cannot order events with equal timestamps.
    pre = tx[(tx.time > 0) & (tx.time < 168)].groupby("person").size()
    post = tx[(tx.time > 168) & (tx.time < 336)].groupby("person").size()
    pairs = anchors[["person", "type"]].copy()
    pairs["pre"] = pairs.person.map(pre).fillna(0).astype(int)
    pairs["post"] = pairs.person.map(post).fillna(0).astype(int)
    previous = set(offers.loc[offers.time < 168, "person"])
    pairs["prior_offer"] = pairs.person.isin(previous)
    result = summarize("starbucks", pairs)
    result["prior_offer_n"] = int(pairs.prior_offer.sum())
    result["at_anchor_transaction_excluded"] = int(tx[tx.time.eq(168) & tx.person.isin(pairs.person)].shape[0])
    result["by_type"] = {k: {"n": len(v), "pre": v.pre.mean(), "post": v.post.mean()} for k, v in pairs.groupby("type")}
    users = pd.read_json(raw / "profile.json", lines=True).id
    received_at_anchor = set(offers.loc[offers.time == 168, "person"])
    controls = pd.DataFrame({"person": users[~users.isin(received_at_anchor)]})
    controls["pre"] = controls.person.map(pre).fillna(0).astype(int)
    controls["post"] = controls.person.map(post).fillna(0).astype(int)
    controls["prior_offer"] = controls.person.isin(previous)
    result["no_offer_at_anchor"] = {
        "n": len(controls), "pre_mean": controls.pre.mean(), "post_mean": controls.post.mean(),
        "prior_offer_n": int(controls.prior_offer.sum()),
    }
    return result


def journey():
    days = pd.read_csv(OUT / "journey_daily.csv", dtype={"household_id": str}, parse_dates=["day"])
    redeems = pd.read_csv(OUT / "journey_redemptions.csv", dtype={"household_id": str}, parse_dates=["redemption_date"])
    lo, hi = days.day.min(), days.day.max()
    eligible = redeems[(redeems.redemption_date >= lo + pd.Timedelta(days=14)) &
                       (redeems.redemption_date <= hi - pd.Timedelta(days=14))]
    anchors = eligible.sort_values("redemption_date").drop_duplicates("household_id")
    joined = days.merge(anchors[["household_id", "redemption_date"]], on="household_id", how="inner")
    joined["diff"] = (joined.day - joined.redemption_date).dt.days
    pre = joined[joined['diff'].between(-14, -1)].groupby("household_id").baskets.sum()
    post = joined[joined['diff'].between(1, 14)].groupby("household_id").baskets.sum()
    pairs = anchors[["household_id", "redemption_date"]].copy()
    pairs["pre"] = pairs.household_id.map(pre).fillna(0).astype(int)
    pairs["post"] = pairs.household_id.map(post).fillna(0).astype(int)
    result = summarize("complete_journey", pairs)
    result.update({"first_data_day": str(lo.date()), "last_data_day": str(hi.date()),
                   "redemption_households_total": int(redeems.household_id.nunique()),
                   "anchor_day_excluded": True})
    return result


if __name__ == "__main__":
    output = [starbucks(), journey()]
    (OUT / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
