"""Descriptive coupon-receipt and redemption analyses; not causal effects."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from utils.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT

REACT_OUT = ROOT / "artifacts/coupon-reactivation"


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


def run_reactivation():
    REACT_OUT.mkdir(parents=True, exist_ok=True)
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
    (REACT_OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


PERSIST_OUT = ROOT / "artifacts/coupon-persistence"


def summarize_persistence(bits):
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


def run_persistence():
    PERSIST_OUT.mkdir(parents=True, exist_ok=True)
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
            row[label] = summarize_persistence(bits)
            assert sum(row[label]["active_weeks_distribution"]) == len(part)
        # Receipt dates stop on June 15: week 5 is partial and week 6 unknown.
        row["later_receipt_week1to4_n"] = [int(part[f"w{w}_a2"].gt(0).sum()) for w in range(1, 5)]
        row["later_receipt_week5_partial_n"] = int(part["w5_a2"].gt(0).sum())
        row["later_receipt_week6_n"] = None
        result.append(row)
    (PERSIST_OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


LONG_OUT = ROOT / "artifacts/uplift-sequence-pilot"


def run_long_term():
    LONG_OUT.mkdir(parents=True, exist_ok=True)
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
    (LONG_OUT / "long_term_observed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


PERSONA_OUT = ROOT / "artifacts/coupon_persona_patterns"
ONLINE = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
TIANCHI_PROXY_OUT = ROOT / "artifacts/tianchi-persona-proxies"
TIANCHI_UPLIFT_OUT = ROOT / "artifacts/tianchi-persona-uplift"
TIANCHI_SINGLE_RECEIPT_OUT = ROOT / "artifacts/tianchi-single-receipt-duration"
TIANCHI_OVERLAP_OUT = ROOT / "artifacts/tianchi-coupon-overlap"
TIANCHI_CUMULATIVE_OUT = ROOT / "artifacts/tianchi-cumulative-coupon-history"


def coupon_overlap_band(distinct_coupon_count):
    """Map distinct receipts in an exposure window to 0, 1, or 2+ dose."""
    return np.minimum(np.asarray(distinct_coupon_count, dtype=int), 2)


def _history_bins(frame):
    """Small, pre-exposure-only strata used for descriptive standardization."""
    click = np.select([frame.pre_click.eq(0), frame.pre_click.eq(1)], ["0", "1"], default="2+")
    purchase = np.where(frame.pre_purchase.gt(0), "1+", "0")
    receipt = np.where(frame.pre_receipt.gt(0), "1+", "0")
    return (pd.Series(click, index=frame.index, dtype="string") + " click / " +
            pd.Series(purchase, index=frame.index, dtype="string") + " purchase / " +
            pd.Series(receipt, index=frame.index, dtype="string") + " prior receipt")


def run_tianchi_coupon_overlap():
    """Describe the association of one-week multiple-coupon receipt with next-week purchase.

    Tianchi records coupon receipt, not coupon expiry or redeemed inventory.  Therefore
    ``2+`` means two or more *distinct coupon IDs received in one calendar week*, not
    verified simultaneously redeemable coupons.
    """
    import matplotlib.pyplot as plt

    TIANCHI_OVERLAP_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    periods = ["pre", "exposure", "outcome"]
    counts, exposure_ids = [], []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Coupon_id", "Date", "Date_received"],
                             dtype={"Coupon_id": "string", "Date": "string", "Date_received": "string"},
                             chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.select([delta.between(-14, -1), delta.between(1, 7), delta.between(8, 14)], periods, default="outside")
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        counts.append(data.groupby(["User_id", "period", "Action"]).size())
        overlap = chunk.loc[receipt & delta.between(1, 7), ["User_id", "Coupon_id"]].dropna()
        if not overlap.empty:
            exposure_ids.append(overlap)

    wide = pd.concat(counts).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in periods:
        for action in (0, 1, 2):
            name = f"{period}_{action}"
            if name not in wide:
                wide[name] = 0
    ids = pd.concat(exposure_ids, ignore_index=True).drop_duplicates(["User_id", "Coupon_id"])
    distinct = ids.groupby("User_id").size().rename("distinct_coupon_ids")
    frame = wide.join(distinct, how="outer").fillna(0).reset_index()
    for name in wide.columns:
        frame[name] = frame[name].astype(int)
    frame["distinct_coupon_ids"] = frame.distinct_coupon_ids.astype(int)
    frame["pre_click"], frame["pre_purchase"], frame["pre_receipt"] = frame.pre_0, frame.pre_1, frame.pre_2
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["receipt_events_exposure"] = frame.exposure_2
    frame["overlap_band"] = coupon_overlap_band(frame.distinct_coupon_ids)
    frame["overlap_label"] = pd.Categorical(np.where(frame.overlap_band.eq(2), "2+", frame.overlap_band.astype(str)),
                                             categories=["0", "1", "2+"], ordered=True)
    frame["outcome_purchase_any"] = frame.outcome_1.gt(0).astype(int)
    frame["outcome_any"] = (frame.outcome_0 + frame.outcome_1).gt(0).astype(int)
    frame["outcome_receipt_any"] = frame.outcome_2.gt(0).astype(int)
    frame = frame.loc[frame.pre_total.gt(0)].copy()
    frame["stratum"] = _history_bins(frame)

    common = [name for name, part in frame.groupby("stratum", observed=True)
              if set(part.overlap_band.unique()) == {0, 1, 2}]
    frame = frame.loc[frame.stratum.isin(common)].copy()
    reference_counts = frame.loc[frame.overlap_band.eq(1), "stratum"].value_counts()
    weights = reference_counts / reference_counts.sum()
    results = []
    for band, label in [(0, "0"), (1, "1"), (2, "2+")]:
        part = frame.loc[frame.overlap_band.eq(band)]
        table = part.groupby("stratum").agg(purchase=("outcome_purchase_any", "mean"),
                                             purchase_n=("outcome_purchase_any", "count"),
                                             any_action=("outcome_any", "mean"),
                                             receipt=("outcome_receipt_any", "mean"))
        table = table.loc[weights.index]
        purchase = float((table.purchase * weights).sum())
        purchase_se = float(np.sqrt((weights ** 2 * table.purchase * (1 - table.purchase) / table.purchase_n).sum()))
        results.append({"overlap_band": label, "n": int(len(part)),
                        "distinct_coupon_ids_mean": float(part.distinct_coupon_ids.mean()),
                        "receipt_events_exposure_mean": float(part.receipt_events_exposure.mean()),
                        "standardized_purchase_8to14": purchase,
                        "standardized_purchase_ci95": [purchase - 1.96 * purchase_se, purchase + 1.96 * purchase_se],
                        "standardized_any_action_8to14": float((table.any_action * weights).sum()),
                        "outcome_receipt_rate_8to14": float((table.receipt * weights).sum())})
    baseline, one, multiple = results
    for item, reference, name in [(one, baseline, "difference_vs_0"), (multiple, one, "increment_2plus_vs_1")]:
        difference = item["standardized_purchase_8to14"] - reference["standardized_purchase_8to14"]
        item[name] = difference
        item[f"{name}_ci95"] = [
            difference - 1.96 * np.sqrt(((item["standardized_purchase_ci95"][1] - item["standardized_purchase_ci95"][0]) / 3.92) ** 2 +
                                         ((reference["standardized_purchase_ci95"][1] - reference["standardized_purchase_ci95"][0]) / 3.92) ** 2),
            difference + 1.96 * np.sqrt(((item["standardized_purchase_ci95"][1] - item["standardized_purchase_ci95"][0]) / 3.92) ** 2 +
                                         ((reference["standardized_purchase_ci95"][1] - reference["standardized_purchase_ci95"][0]) / 3.92) ** 2),
        ]
    keep = ["User_id", "stratum", "pre_click", "pre_purchase", "pre_receipt", "pre_total", "distinct_coupon_ids",
            "receipt_events_exposure", "overlap_label", "outcome_purchase_any", "outcome_any", "outcome_receipt_any"]
    frame[keep].to_parquet(TIANCHI_OVERLAP_OUT / "cohort.parquet", index=False)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "exposure_window": "days 1-7: number of distinct Coupon_id values with Action=2",
              "outcome_window": "days 8-14: any purchase record (Action=1)",
              "overlap_definition": "0, 1, or 2+ distinct coupon IDs received in the exposure window; active/expiry state is unavailable",
              "eligibility": "at least one pre-anchor click or purchase; common support standardized over pre click, purchase, and receipt-history strata",
              "n_common_support": int(len(frame)), "stratum_weights_from_one_coupon_arm": {str(k): float(v) for k, v in weights.items()},
              "results": results,
              "caveat": "This is an observational dose-response description. Higher multiple-receipt rates and outcome-window receipt rates can reflect continuing targeting, user demand, or unobserved offer rules; it is not a causal coupon-stacking estimate."}
    (TIANCHI_OVERLAP_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["overlap_band"] for row in results]
    x = np.arange(len(results))
    rate = np.array([row["standardized_purchase_8to14"] for row in results]) * 100
    ci = np.array([row["standardized_purchase_ci95"] for row in results]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x, rate, color=["#4e79a7", "#f28e2b", "#59a14f"], width=.62)
    axes[0].errorbar(x, rate, yerr=np.vstack([rate - ci[:, 0], ci[:, 1] - rate]), fmt="none", color="black", capsize=4)
    axes[0].set(xticks=x, xticklabels=labels, xlabel="Distinct coupon IDs received in days 1–7", ylabel="Purchase record in days 8–14 (%)",
                title="Standardized next-week purchase rate")
    later = [row["outcome_receipt_rate_8to14"] * 100 for row in results]
    axes[1].plot(x, later, marker="o", linewidth=2, color="#b07aa1")
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Distinct coupon IDs received in days 1–7", ylabel="Received another coupon in days 8–14 (%)",
                title="Continued coupon targeting diagnostic")
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi multiple-coupon receipt: descriptive overlap proxy", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_coupon_overlap.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_cumulative_coupon_history():
    """Describe next-week purchase by coupon receipts accumulated before an anchor."""
    import matplotlib.pyplot as plt

    TIANCHI_CUMULATIVE_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.where(delta.lt(0), "history", np.where(delta.between(1, 14), "outcome", "outside"))
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        pieces.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in ("history", "outcome"):
        for action in (0, 1, 2):
            name = f"{period}_{action}"
            if name not in wide:
                wide[name] = 0
    frame = wide.reset_index()
    frame["pre_click"], frame["pre_purchase"], frame["cumulative_receipts"] = frame.history_0, frame.history_1, frame.history_2
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame = frame.loc[frame.pre_total.gt(0)].copy()
    frame["coupon_history_band"] = pd.cut(frame.cumulative_receipts, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3–4", "5+"])
    frame["outcome_purchase_any"] = frame.outcome_1.gt(0).astype(int)
    frame["outcome_receipt_any"] = frame.outcome_2.gt(0).astype(int)
    click_bin = np.select([frame.pre_click.eq(0), frame.pre_click.eq(1)], ["0", "1"], default="2+")
    purchase_bin = np.where(frame.pre_purchase.gt(0), "1+", "0")
    frame["stratum"] = pd.Series(click_bin, index=frame.index, dtype="string") + " click / " + pd.Series(purchase_bin, index=frame.index, dtype="string") + " purchase"
    bands = ["0", "1", "2", "3–4", "5+"]
    common = [key for key, part in frame.groupby("stratum", observed=True) if set(part.coupon_history_band.dropna()) == set(bands)]
    frame = frame.loc[frame.stratum.isin(common)].copy()
    weights = frame.loc[frame.coupon_history_band.eq("1"), "stratum"].value_counts()
    weights = weights / weights.sum()
    results = []
    for band in bands:
        part = frame.loc[frame.coupon_history_band.eq(band)]
        table = part.groupby("stratum").agg(purchase=("outcome_purchase_any", "mean"), n=("outcome_purchase_any", "count"),
                                             later_receipt=("outcome_receipt_any", "mean")).loc[weights.index]
        rate = float((table.purchase * weights).sum())
        se = float(np.sqrt((weights ** 2 * table.purchase * (1 - table.purchase) / table.n).sum()))
        results.append({"cumulative_receipt_band": band, "n": int(len(part)), "mean_cumulative_receipts": float(part.cumulative_receipts.mean()),
                        "standardized_purchase_1to14": rate, "standardized_purchase_ci95": [rate - 1.96 * se, rate + 1.96 * se],
                        "outcome_receipt_rate_1to14": float((table.later_receipt * weights).sum())})
    frame[["User_id", "stratum", "pre_click", "pre_purchase", "cumulative_receipts", "coupon_history_band", "outcome_purchase_any", "outcome_receipt_any"]].to_parquet(TIANCHI_CUMULATIVE_OUT / "cohort.parquet", index=False)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "exposure": "all Action=2 coupon-receipt events observed before the anchor, from 2016-01-01 onward",
              "outcome": "any purchase record in days 1-14 after anchor", "eligibility": "at least one pre-anchor click or purchase; standardized over pre click and purchase strata with common support",
              "interpretation": "Cumulative receipts measure historical targeting/exposure, not an active coupon inventory because expiry and redemption balance are unobserved.",
              "results": results,
              "caveat": "Observational association only: accumulated receipt count is confounded by user activity, eligibility, and targeting."}
    (TIANCHI_CUMULATIVE_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures = ROOT / "img/uplift"; figures.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(results)); rates = np.array([row["standardized_purchase_1to14"] for row in results]) * 100
    ci = np.array([row["standardized_purchase_ci95"] for row in results]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(x, rates, marker="o", linewidth=2, color="#4e79a7")
    axes[0].fill_between(x, ci[:, 0], ci[:, 1], color="#4e79a7", alpha=.18)
    axes[0].set(xticks=x, xticklabels=bands, xlabel="Cumulative coupon receipts before anchor", ylabel="Purchase record, days 1–14 (%)", title="Standardized purchase association")
    axes[1].plot(x, [row["outcome_receipt_rate_1to14"] * 100 for row in results], marker="o", linewidth=2, color="#b07aa1")
    axes[1].set(xticks=x, xticklabels=bands, xlabel="Cumulative coupon receipts before anchor", ylabel="Another receipt, days 1–14 (%)", title="Continued targeting diagnostic")
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi cumulative coupon-receipt history: descriptive association", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_cumulative_coupon_history.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_uplift_audit():
    """Multi-anchor balance, placebo, and temporal-stability audit for receipt uplift."""
    import matplotlib.pyplot as plt

    out = ROOT / "artifacts/tianchi-uplift-audit"; out.mkdir(parents=True, exist_ok=True)
    anchors = pd.to_datetime(["2016-05-01", "2016-05-05", "2016-05-09", "2016-05-13", "2016-05-17"])
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        for anchor in anchors:
            delta = (day - anchor).dt.days
            period = np.select([delta.between(-28, -15), delta.between(-14, -1), delta.eq(0), delta.between(1, 14)],
                               ["placebo", "pre", "anchor", "outcome"], default="outside")
            keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
            data = chunk.loc[keep, ["User_id", "Action"]].copy()
            data["period"], data["anchor_day"] = period[keep], anchor.strftime("%Y-%m-%d")
            pieces.append(data.groupby(["anchor_day", "User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2, 3]).sum().unstack([2, 3], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in ("placebo", "pre", "anchor", "outcome"):
        for action in (0, 1, 2):
            if f"{period}_{action}" not in wide: wide[f"{period}_{action}"] = 0
    rows = []
    for anchor_day, part in wide.groupby(level=0):
        f = part.droplevel(0).reset_index()
        f["treated"] = f.anchor_2.gt(0).astype(int)
        f["pre_click"], f["pre_purchase"] = f.pre_0, f.pre_1
        click = np.select([f.pre_click.eq(0), f.pre_click.eq(1)], ["0", "1"], default="2+")
        purchase = np.where(f.pre_purchase.gt(0), "1+", "0")
        f["stratum"] = pd.Series(click, index=f.index, dtype="string") + "_click__" + pd.Series(purchase, index=f.index, dtype="string")
        f = f.loc[(f.pre_click + f.pre_purchase).gt(0)].copy()
        common = [s for s, g in f.groupby("stratum") if g.treated.nunique() == 2]
        f = f[f.stratum.isin(common)]
        weights = f[f.treated.eq(1)].stratum.value_counts(); weights = weights / weights.sum()
        def standardized(column):
            tab = f.groupby(["stratum", "treated"])[column].mean().unstack().loc[weights.index]
            return float((tab[1] * weights).sum() - (tab[0] * weights).sum())
        f["outcome_purchase"] = f.outcome_1.gt(0).astype(int)
        f["placebo_purchase"] = f.placebo_1.gt(0).astype(int)
        f["later_receipt"] = f.outcome_2.gt(0).astype(int)
        pooled_sd = np.sqrt((f.loc[f.treated.eq(1), "pre_purchase"].var() + f.loc[f.treated.eq(0), "pre_purchase"].var()) / 2)
        rows.append({"anchor_day": str(anchor_day), "treated_n": int(f.treated.sum()), "control_n": int((1-f.treated).sum()),
                     "purchase_difference": standardized("outcome_purchase"), "placebo_purchase_difference": standardized("placebo_purchase"),
                     "later_receipt_difference": standardized("later_receipt"),
                     "pre_purchase_smd": None if not pooled_sd else float((f.loc[f.treated.eq(1), "pre_purchase"].mean()-f.loc[f.treated.eq(0), "pre_purchase"].mean())/pooled_sd)})
    report = {"dataset": "Tianchi O2O online", "anchors": rows,
              "design": "five fixed anchors; 14-day pre-period stratification, 14-day post purchase outcome, and earlier 14-day placebo outcome",
              "interpretation": "Nonzero placebo or baseline imbalance indicates receipt is not conditionally randomized under these observed covariates."}
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = pd.DataFrame(rows); table.to_csv(out / "anchor_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table)); ax.plot(x, table.purchase_difference*100, marker="o", label="post purchase difference", color="#4e79a7")
    ax.plot(x, table.placebo_purchase_difference*100, marker="o", label="pre-treatment placebo difference", color="#e15759")
    ax.axhline(0, color="black", linewidth=.8); ax.set(xticks=x, xticklabels=table.anchor_day.str[5:], ylabel="Treated − control (pp)", xlabel="Anchor date (2016)", title="Receipt uplift audit: temporal stability and placebo")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); (ROOT / "img/uplift").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / "img/uplift/tianchi_uplift_audit.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def tianchi_persona_proxy_labels(frame):
    """Return disjoint observable proxies; Tianchi does not observe carts or removals."""
    active = frame.pre_total.gt(0)
    q25, q75, q95 = frame.loc[active, "pre_total"].quantile([.25, .75, .95]).astype(int)
    labels = np.full(len(frame), "unassigned", dtype=object)
    labels[(frame.pre_purchase.eq(0) & frame.pre_total.between(1, q25)).to_numpy()] = "P1_observation_proxy"
    labels[(frame.pre_purchase.gt(0) & frame.pre_total.ge(q95)).to_numpy()] = "P2_deep_consideration_proxy"
    labels[(frame.pre_purchase.eq(0) & frame.pre_total.ge(q75)).to_numpy()] = "P3_cart_revision_proxy"
    labels[(frame.pre_purchase.gt(0) & frame.pre_total.lt(q95)).to_numpy()] = "P4_purchase_path_proxy"
    return labels, {"pre_total_q25": int(q25), "pre_total_q75": int(q75), "pre_total_q95": int(q95)}


def run_tianchi_persona_proxies():
    """Find Tianchi coupon-anchor users similar on the actions shared with Cosmetics."""
    TIANCHI_PROXY_OUT.mkdir(parents=True, exist_ok=True)
    anchors = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date_received"],
                             dtype={"Date_received": "string"}, chunksize=1_000_000):
        received = pd.to_datetime(chunk.Date_received, format="%Y%m%d", errors="coerce")
        mask = chunk.Action.eq(2) & received.between("2016-05-01", "2016-05-31")
        anchors.append(pd.DataFrame({"User_id": chunk.loc[mask, "User_id"], "anchor": received[mask]}))
    anchors = (pd.concat(anchors, ignore_index=True).sort_values(["User_id", "anchor"])
               .drop_duplicates("User_id").set_index("User_id")["anchor"])
    counts, days = [], []
    users = set(anchors.index)
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date"], dtype={"Date": "string"},
                             chunksize=1_000_000):
        chunk = chunk.loc[chunk.User_id.isin(users) & chunk.Action.isin([0, 1])].copy()
        if chunk.empty:
            continue
        date = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        delta = (date - chunk.User_id.map(anchors)).dt.days
        period = np.select([delta.between(-14, -1), delta.between(1, 14)], ["pre", "post"], default="outside")
        chunk = chunk.loc[period != "outside", ["User_id", "Action"]].copy()
        chunk["period"] = period[period != "outside"]
        chunk["date"] = date[period != "outside"].to_numpy()
        counts.append(chunk.groupby(["User_id", "period", "Action"]).size())
        days.append(chunk[["User_id", "period", "date"]].drop_duplicates())
    wide = pd.concat(counts).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    active_days = pd.concat(days).drop_duplicates().groupby(["User_id", "period"]).size().unstack(fill_value=0)
    frame = anchors.rename("anchor").to_frame().join(wide, how="left").join(active_days.add_suffix("_active_days"), how="left").fillna(0)
    for period in ("pre", "post"):
        for action in (0, 1):
            if f"{period}_{action}" not in frame:
                frame[f"{period}_{action}"] = 0
        if f"{period}_active_days" not in frame:
            frame[f"{period}_active_days"] = 0
    frame = frame.reset_index()
    frame["pre_click"], frame["pre_purchase"] = frame.pop("pre_0").astype(int), frame.pop("pre_1").astype(int)
    frame["post_click"], frame["post_purchase"] = frame.pop("post_0").astype(int), frame.pop("post_1").astype(int)
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["pre_purchase_share"] = np.divide(frame.pre_purchase, frame.pre_total,
                                               out=np.zeros(len(frame)), where=frame.pre_total.gt(0))
    frame["proxy_persona"], quantiles = tianchi_persona_proxy_labels(frame)
    frame["post_any"] = (frame.post_click + frame.post_purchase).gt(0).astype(int)
    frame["post_purchase_any"] = frame.post_purchase.gt(0).astype(int)
    keep = ["User_id", "anchor", "proxy_persona", "pre_click", "pre_purchase", "pre_total",
            "pre_active_days", "pre_purchase_share", "post_click", "post_purchase", "post_active_days",
            "post_any", "post_purchase_any"]
    frame[keep].to_parquet(TIANCHI_PROXY_OUT / "tianchi_coupon_anchor_persona_proxies.parquet", index=False)
    summary = []
    for persona, part in frame.groupby("proxy_persona", sort=True):
        summary.append({"proxy_persona": persona, "users": int(len(part)),
                        "pre_click_median": float(part.pre_click.median()),
                        "pre_purchase_median": float(part.pre_purchase.median()),
                        "pre_active_days_median": float(part.pre_active_days.median()),
                        "post_any_14d_rate": float(part.post_any.mean()),
                        "post_purchase_14d_rate": float(part.post_purchase_any.mean()),
                        "sample_user_ids": part.User_id.sort_values().head(10).astype(int).tolist()})
    result = {"dataset": "Tianchi O2O online", "anchor": "first coupon receipt in May 2016",
              "matching_basis": "pre-anchor click, purchase, total activity, and active days only",
              "missing_from_tianchi": ["cart", "remove_from_cart", "session boundaries"],
              "quantiles": quantiles, "proxies": summary,
              "causal_caveat": "receipt is observational; post rates are not coupon uplift estimates"}
    (TIANCHI_PROXY_OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_tianchi_persona_uplift():
    """Describe post-receipt trajectories and canonical-type bounds by Tianchi proxy."""
    from analysis.uplift.canonical import bounds
    import matplotlib.pyplot as plt

    TIANCHI_UPLIFT_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    periods = ["pre", "anchor", "w1", "w2", "w3", "w4", "w5", "w6"]
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.select([delta.between(-14, -1), delta.eq(0), delta.between(1, 7), delta.between(8, 14),
                            delta.between(15, 21), delta.between(22, 28), delta.between(29, 35), delta.between(36, 42)],
                           periods, default="outside")
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        pieces.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in periods:
        for action in (0, 1, 2):
            if f"{period}_{action}" not in wide:
                wide[f"{period}_{action}"] = 0
    frame = wide.reset_index()
    frame["pre_click"], frame["pre_purchase"] = frame["pre_0"], frame["pre_1"]
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["proxy_persona"], quantiles = tianchi_persona_proxy_labels(frame)
    frame["treated"] = frame["anchor_2"].gt(0).astype(int)
    for week in range(1, 7):
        frame[f"w{week}_any"] = (frame[f"w{week}_0"] + frame[f"w{week}_1"]).gt(0).astype(int)
        frame[f"w{week}_purchase_any"] = frame[f"w{week}_1"].gt(0).astype(int)
        frame[f"w{week}_later_receipt"] = frame[f"w{week}_2"].gt(0).astype(int)
    frame["purchase_14d"] = frame[["w1_purchase_any", "w2_purchase_any"]].any(axis=1).astype(int)
    cohort = frame.loc[frame.proxy_persona.ne("unassigned")].copy()
    cohort.to_parquet(TIANCHI_UPLIFT_OUT / "tianchi_persona_uplift_cohort.parquet", index=False)
    result = []
    for persona, part in cohort.groupby("proxy_persona", sort=True):
        arms = part.groupby("treated").purchase_14d.agg(["count", "sum", "mean"])
        item = {"proxy_persona": persona, "n": int(len(part)), "quantiles": quantiles}
        if set(arms.index) == {0, 1}:
            p0, p1 = float(arms.loc[0, "mean"]), float(arms.loc[1, "mean"])
            item.update({"n_control": int(arms.loc[0, "count"]), "n_treated": int(arms.loc[1, "count"]),
                         "purchase_14d_control": p0, "purchase_14d_treated": p1,
                         "observed_difference": p1 - p0, "canonical_type_bounds": bounds(p0, p1)})
        item["weekly"] = []
        for week in range(1, 7):
            arms_week = part.groupby("treated")[[f"w{week}_any", f"w{week}_purchase_any", f"w{week}_later_receipt"]].mean()
            if set(arms_week.index) != {0, 1}:
                continue
            item["weekly"].append({"week": week,
                                   "any_control": float(arms_week.loc[0, f"w{week}_any"]),
                                   "any_treated": float(arms_week.loc[1, f"w{week}_any"]),
                                   "any_difference": float(arms_week.loc[1, f"w{week}_any"] - arms_week.loc[0, f"w{week}_any"]),
                                   "purchase_control": float(arms_week.loc[0, f"w{week}_purchase_any"]),
                                   "purchase_treated": float(arms_week.loc[1, f"w{week}_purchase_any"]),
                                   "purchase_difference": float(arms_week.loc[1, f"w{week}_purchase_any"] - arms_week.loc[0, f"w{week}_purchase_any"]),
                                   "later_receipt_treated": float(arms_week.loc[1, f"w{week}_later_receipt"]),
                                   "later_receipt_control": float(arms_week.loc[0, f"w{week}_later_receipt"])})
        result.append(item)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "treatment": "coupon receipt record on anchor day", "outcome": "any purchase record in days 1-14",
              "proxy_rule": "pre-anchor click/purchase depth only; carts, removal, sessions are unavailable",
              "canonical_interpretation": "Fréchet bounds under the additional assumption that arm-specific observed rates equal comparable potential-outcome marginals; no individual type labels",
              "duration_interpretation": "weekly treated-control differences are observational and can be confounded by later coupon receipts",
              "results": result}
    (TIANCHI_UPLIFT_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [item["proxy_persona"].replace("_proxy", "") for item in result]
    control = [item["purchase_14d_control"] * 100 for item in result]
    treated = [item["purchase_14d_treated"] * 100 for item in result]
    x = np.arange(len(result))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - .18, control, .36, label="no anchor-day receipt", color="#4e79a7")
    ax.bar(x + .18, treated, .36, label="anchor-day receipt", color="#f28e2b")
    for index, item in enumerate(result):
        ax.text(index, max(control[index], treated[index]) + 1, f"Δ {item['observed_difference'] * 100:+.1f}pp",
                ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="Purchase record in days 1–14 (%)",
           title="Tianchi proxy cohorts: observed 14-day purchase contrast", ylim=(0, max(treated) + 12))
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(figures / "tianchi_proxy_14d_contrast.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    types = ["sure_thing", "persuadable", "sleeping_dog", "lost_cause"]
    type_labels = ["sure thing", "persuadable", "sleeping dog", "lost cause"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, item in zip(axes.flat, result):
        intervals = item["canonical_type_bounds"]
        for index, key in enumerate(types):
            low, high = np.array(intervals[key]) * 100
            ax.hlines(index, low, high, color="#59a14f" if key == "persuadable" else "#4e79a7", linewidth=5)
            ax.plot([low, high], [index, index], "o", color="black", ms=3)
            ax.text(high + 1, index, f"{low:.1f}–{high:.1f}%", va="center", fontsize=8)
        ax.set(title=item["proxy_persona"].replace("_proxy", ""), yticks=range(4), yticklabels=type_labels,
               xlim=(0, 100), xlabel="Possible share (%)")
        ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Canonical-type Fréchet bounds under the comparability assumption", y=1.02)
    fig.tight_layout(); fig.savefig(figures / "tianchi_proxy_canonical_bounds.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    weeks = np.arange(1, 7)
    colors = ["#4e79a7", "#76b7b2", "#f28e2b", "#59a14f"]
    for item, color, label in zip(result, colors, labels):
        weekly = item["weekly"]
        axes[0].plot(weeks, [row["purchase_difference"] * 100 for row in weekly], marker="o", color=color, label=label)
        # Receipt dates are right-censored after mid-June; only weeks 1–5 are comparable.
        axes[1].plot(weeks[:5], [(row["later_receipt_treated"] - row["later_receipt_control"]) * 100 for row in weekly[:5]],
                     marker="o", color=color, label=label)
    axes[0].axhline(0, color="black", linewidth=.8); axes[1].axhline(0, color="black", linewidth=.8)
    axes[0].set(title="Observed weekly purchase-record difference", xlabel="Week after anchor", ylabel="Treated − control (percentage points)", xticks=weeks)
    axes[1].set(title="Difference in subsequent coupon receipt", xlabel="Week after anchor", ylabel="Treated − control (percentage points)", xticks=weeks[:5])
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Duration is confounded by later coupon receipt; week 6 receipt data are censored", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_proxy_duration.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_single_receipt_duration():
    """Descriptive duration analysis with no other recorded coupon receipt in the window."""
    import matplotlib.pyplot as plt

    TIANCHI_SINGLE_RECEIPT_OUT.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2016-05-15")
    periods = ["pre", "anchor", "w1", "w2", "w3", "w4", "w5", "w6"]
    pieces = []
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        day.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        delta = (day - anchor).dt.days
        period = np.select([delta.between(-14, -1), delta.eq(0), delta.between(1, 7), delta.between(8, 14),
                            delta.between(15, 21), delta.between(22, 28), delta.between(29, 35), delta.between(36, 42)],
                           periods, default="outside")
        keep = (period != "outside") & chunk.Action.isin([0, 1, 2])
        data = chunk.loc[keep, ["User_id", "Action"]].copy()
        data["period"] = period[keep]
        pieces.append(data.groupby(["User_id", "period", "Action"]).size())
    wide = pd.concat(pieces).groupby(level=[0, 1, 2]).sum().unstack([1, 2], fill_value=0)
    wide.columns = [f"{period}_{action}" for period, action in wide.columns]
    for period in periods:
        for action in (0, 1, 2):
            if f"{period}_{action}" not in wide:
                wide[f"{period}_{action}"] = 0
    frame = wide.reset_index()
    frame["pre_click"], frame["pre_purchase"] = frame["pre_0"], frame["pre_1"]
    frame["pre_total"] = frame.pre_click + frame.pre_purchase
    frame["receipt_total"] = frame[[f"{period}_2" for period in periods]].sum(axis=1)
    frame["single_anchor_receipt"] = frame.anchor_2.eq(1) & frame.receipt_total.eq(1)
    frame["never_receipt_window"] = frame.receipt_total.eq(0)
    frame["treated"] = frame.single_anchor_receipt.astype(int)
    frame["arm"] = np.where(frame.single_anchor_receipt, "single_anchor_receipt",
                            np.where(frame.never_receipt_window, "never_receipt", "excluded_other_receipt"))
    frame = frame.loc[frame.pre_total.gt(0) & frame.arm.ne("excluded_other_receipt")].copy()
    frame["pre_click_bin"] = np.select([frame.pre_click.eq(0), frame.pre_click.eq(1)], ["0", "1"], default="2+")
    frame["pre_purchase_bin"] = np.where(frame.pre_purchase.gt(0), "1+", "0")
    frame["stratum"] = frame.pre_click_bin + "_click__" + frame.pre_purchase_bin + "_purchase"
    for week in range(1, 7):
        frame[f"w{week}_purchase_any"] = frame[f"w{week}_1"].gt(0).astype(int)
        frame[f"w{week}_any"] = (frame[f"w{week}_0"] + frame[f"w{week}_1"]).gt(0).astype(int)
    eligible_strata = []
    for stratum, part in frame.groupby("stratum"):
        if part.treated.nunique() == 2:
            eligible_strata.append(stratum)
    frame = frame[frame.stratum.isin(eligible_strata)].copy()
    treated_counts = frame[frame.treated.eq(1)].stratum.value_counts()
    weights = treated_counts / treated_counts.sum()
    weekly = []
    for week in range(1, 7):
        values = {}
        for outcome in ("purchase_any", "any"):
            table = frame.groupby(["stratum", "treated"])[f"w{week}_{outcome}"].agg(["mean", "count"]).unstack()
            control = float((table[("mean", 0)] * weights.loc[table.index]).sum())
            treated = float((table[("mean", 1)] * weights.loc[table.index]).sum())
            variance = float((weights.loc[table.index] ** 2 * (
                table[("mean", 0)] * (1 - table[("mean", 0)]) / table[("count", 0)] +
                table[("mean", 1)] * (1 - table[("mean", 1)]) / table[("count", 1)])).sum())
            values[f"{outcome}_control"] = control
            values[f"{outcome}_treated"] = treated
            values[f"{outcome}_difference"] = treated - control
            values[f"{outcome}_difference_ci95"] = [treated - control - 1.96 * np.sqrt(variance),
                                                       treated - control + 1.96 * np.sqrt(variance)]
        weekly.append({"week": week, **values})
    keep = ["User_id", "arm", "treated", "stratum", "pre_click", "pre_purchase", "pre_total", "receipt_total"] + [
        f"w{week}_{kind}" for week in range(1, 7) for kind in ("purchase_any", "any")]
    frame[keep].to_parquet(TIANCHI_SINGLE_RECEIPT_OUT / "cohort.parquet", index=False)
    report = {"dataset": "Tianchi O2O online", "anchor_day": "2016-05-15",
              "treated_definition": "exactly one receipt event, on anchor day, and no other receipt in days -14 through 42",
              "control_definition": "no receipt event in days -14 through 42",
              "eligibility": "at least one pre-anchor click or purchase; standardized within click-count and purchase-history strata",
              "n_treated": int(frame.treated.sum()), "n_control": int((1 - frame.treated).sum()),
              "stratum_weights": {str(key): float(value) for key, value in weights.items()},
              "weekly_standardized": weekly,
              "caveat": "conditioning on no future receipt is post-treatment selection; results describe a single-receipt trajectory, not an intervention effect"}
    (TIANCHI_SINGLE_RECEIPT_OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    weeks = np.arange(1, 7)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    axes[0].plot(weeks, [row["purchase_any_control"] * 100 for row in weekly], marker="o", color="#4e79a7", label="never receipt")
    axes[0].plot(weeks, [row["purchase_any_treated"] * 100 for row in weekly], marker="o", color="#f28e2b", label="single anchor receipt")
    axes[0].set(title="Standardized weekly purchase-record rate", xlabel="Week after anchor", ylabel="Purchase record (%)", xticks=weeks)
    axes[0].legend(frameon=False)
    axes[1].plot(weeks, [row["purchase_any_difference"] * 100 for row in weekly], marker="o", color="#59a14f", label="purchase")
    purchase_ci = np.array([row["purchase_any_difference_ci95"] for row in weekly]) * 100
    axes[1].fill_between(weeks, purchase_ci[:, 0], purchase_ci[:, 1], color="#59a14f", alpha=.18, label="purchase 95% normal CI")
    axes[1].plot(weeks, [row["any_difference"] * 100 for row in weekly], marker="o", color="#b07aa1", label="any click/purchase")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(title="Standardized treated − control difference", xlabel="Week after anchor", ylabel="Percentage points", xticks=weeks)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi single-receipt descriptive trajectory (no later coupon receipt)", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_single_receipt_duration.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def run_tianchi_coupon_redemption_ordering():
    """Audit whether Tianchi can support an oldest-received-coupon choice rule.

    A coupon purchase has a ``Date_received`` field, but this need not have a
    matching Action=2 receipt record.  We consequently report a deliberately
    strict, directly linkable subset rather than projecting its pattern to all
    coupon purchases.
    """
    import matplotlib.pyplot as plt

    out = ROOT / "artifacts/tianchi-coupon-redemption-ordering"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    action_counts = {0: 0, 1: 0, 2: 0}
    coupon_purchase_count = 0
    for chunk in pd.read_csv(ONLINE, usecols=["User_id", "Merchant_id", "Action", "Coupon_id", "Date_received", "Date"],
                             dtype={"Coupon_id": "string", "Date_received": "string", "Date": "string"}, chunksize=1_000_000):
        for action, n in chunk.Action.value_counts().items():
            action_counts[int(action)] += int(n)
        coupon_purchase_count += int((chunk.Action.eq(1) & chunk.Coupon_id.notna()).sum())
        records.append(chunk.loc[chunk.Action.eq(2) | (chunk.Action.eq(1) & chunk.Coupon_id.notna())])
    events = pd.concat(records, ignore_index=True)
    events["received"] = pd.to_datetime(events.Date_received, format="%Y%m%d", errors="coerce")
    events["event_date"] = pd.to_datetime(events.Date, format="%Y%m%d", errors="coerce")
    keys = ["User_id", "Merchant_id", "Coupon_id", "received"]
    receipts = events.loc[events.Action.eq(2), keys].copy()
    receipt_counts = receipts.groupby(keys, dropna=False).size().rename("receipt_record_count")
    issued = receipts.drop_duplicates(keys)
    redemptions = events.loc[events.Action.eq(1) & events.Coupon_id.notna()].copy()
    redemptions = redemptions.merge(receipt_counts.reset_index(), on=keys, how="left")
    direct_unique = redemptions.loc[redemptions.receipt_record_count.eq(1)].copy()
    direct_unique["token"] = direct_unique[keys].astype(str).agg("|".join, axis=1)
    # Multiple purchase records against one token are not an orderable single-coupon use.
    first_claim = direct_unique.groupby("token").event_date.transform("min")
    direct_unique = direct_unique.loc[direct_unique.event_date.eq(first_claim)].drop_duplicates(["token", "event_date"])

    issues_by_user = {user: group.sort_values("received") for user, group in issued.groupby("User_id", sort=False)}
    choice_rows = []
    for user, claims in direct_unique.groupby("User_id", sort=False):
        user_issues = issues_by_user.get(user)
        if user_issues is None:
            continue
        issue_rows = [(row.received, (row.Merchant_id, row.Coupon_id, row.received))
                      for row in user_issues.itertuples(index=False)]
        used = set()
        # Within-day ordering is unavailable: retain only a single, uniquely linked claim per user-day.
        for purchase_day, claims_on_day in claims.groupby("event_date", sort=True):
            if len(claims_on_day) != 1:
                continue
            claim = claims_on_day.iloc[0]
            token = (claim.Merchant_id, claim.Coupon_id, claim.received)
            candidates = [(received, issue) for received, issue in issue_rows if received < purchase_day and issue not in used]
            if token not in {issue for _, issue in candidates}:
                continue
            receipt_days = np.array([received for received, _ in candidates])
            oldest_day = receipt_days.min()
            older_count = int((receipt_days < claim.received).sum())
            candidate_count = len(candidates)
            choice_rows.append({"User_id": user, "purchase_day": purchase_day, "candidate_count": candidate_count,
                                "used_coupon_age_days": int((purchase_day - claim.received).days),
                                "selected_oldest": int(older_count == 0),
                                "random_oldest_probability": float((receipt_days == oldest_day).mean()),
                                "relative_age_rank_0_oldest_1_youngest": None if candidate_count == 1 else older_count / (candidate_count - 1)})
            used.add(token)
    choices = pd.DataFrame(choice_rows)
    choices.to_parquet(out / "orderable_coupon_choices.parquet", index=False)
    multi = choices.loc[choices.candidate_count.ge(2)].copy()
    multi["choice_set_bin"] = pd.cut(multi.candidate_count, [1, 2, 3, 5, 10, np.inf], labels=["2", "3", "4–5", "6–10", "11+"])
    by_size = []
    for label, part in multi.groupby("choice_set_bin", observed=True):
        by_size.append({"candidate_count": str(label), "n": int(len(part)),
                        "selected_oldest_rate": float(part.selected_oldest.mean()),
                        "random_oldest_rate": float(part.random_oldest_probability.mean()),
                        "mean_relative_age_rank": float(part.relative_age_rank_0_oldest_1_youngest.mean())})
    report = {
        "dataset": "Tianchi O2O online",
        "action_counts": action_counts,
        "coupon_purchase_records": coupon_purchase_count,
        "coupon_purchase_has_direct_unique_receipt": int(len(direct_unique)),
        "direct_unique_receipt_coverage": float(len(direct_unique) / coupon_purchase_count),
        "same_day_receipt_and_purchase_among_coupon_purchases": int(redemptions.event_date.eq(redemptions.received).sum()),
        "same_day_receipt_and_purchase_rate": float(redemptions.event_date.eq(redemptions.received).mean()),
        "strictly_orderable_choice_sets": int(len(choices)),
        "strictly_orderable_with_2plus_candidates": int(len(multi)),
        "multi_candidate_summary": {
            "selected_oldest_rate": float(multi.selected_oldest.mean()),
            "random_oldest_rate": float(multi.random_oldest_probability.mean()),
            "mean_relative_age_rank_0_oldest_1_youngest": float(multi.relative_age_rank_0_oldest_1_youngest.mean()),
            "median_used_coupon_age_days": float(multi.used_coupon_age_days.median())},
        "by_choice_set_size": by_size,
        "interpretation": "The strict subset is not representative of all coupon purchases. It is an orderability audit, not a population FIFO estimate.",
        "modeling_implication": "Do not impose FIFO from Tianchi. If coupon-level inventory is simulated, estimate/sweep an age coefficient in a conditional coupon-choice model and retain a no-age-effect case."}
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    figures = ROOT / "img/uplift"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    age_bins = [-.5, .5, 1.5, 3.5, 7.5, 14.5, 30.5, choices.used_coupon_age_days.max() + 1]
    axes[0].hist(choices.used_coupon_age_days, bins=age_bins, color="#4e79a7", edgecolor="white")
    axes[0].set(title="Used coupon age in strict orderable subset", xlabel="Days from receipt to purchase", ylabel="Coupon purchases")
    labels = [row["candidate_count"] for row in by_size]
    x = np.arange(len(labels))
    axes[1].plot(x, [row["selected_oldest_rate"] * 100 for row in by_size], marker="o", linewidth=2, color="#f28e2b", label="observed oldest selected")
    axes[1].plot(x, [row["random_oldest_rate"] * 100 for row in by_size], marker="o", linewidth=2, color="#7f7f7f", linestyle="--", label="uniform-choice benchmark")
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Observed available coupons in choice set", ylabel="Oldest coupon selected (%)", title="FIFO diagnostic; strict subset only")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tianchi coupon ordering audit", y=1.03)
    fig.tight_layout(); fig.savefig(figures / "tianchi_coupon_ordering_audit.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return report


def online():
    PERSONA_OUT.mkdir(parents=True, exist_ok=True)
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
    pairs.to_csv(PERSONA_OUT / "tianchi_user_pairs.csv", index=False)
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


PAIRS_OUT = ROOT / "artifacts" / "coupon_before_after"


def summarize_pairs(name, pairs):
    PAIRS_OUT.mkdir(parents=True, exist_ok=True)
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
    pairs.to_csv(PAIRS_OUT / f"{name}_pairs.csv", index=False)
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
    result = summarize_pairs("starbucks", pairs)
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
    days = pd.read_csv(PAIRS_OUT / "journey_daily.csv", dtype={"household_id": str}, parse_dates=["day"])
    redeems = pd.read_csv(PAIRS_OUT / "journey_redemptions.csv", dtype={"household_id": str}, parse_dates=["redemption_date"])
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
    result = summarize_pairs("complete_journey", pairs)
    result.update({"first_data_day": str(lo.date()), "last_data_day": str(hi.date()),
                   "redemption_households_total": int(redeems.household_id.nunique()),
                   "anchor_day_excluded": True})
    return result


def run_persona_patterns():
    result = {"tianchi": online(), "starbucks": purchase_history("starbucks"),
              "complete_journey": purchase_history("complete_journey")}
    (PERSONA_OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_paired_windows():
    output = [starbucks(), journey()]
    (PAIRS_OUT / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("reactivation", "persistence", "long-term", "persona", "persona-proxies", "persona-uplift", "single-receipt-duration", "coupon-overlap", "cumulative-coupons", "uplift-audit", "paired"))
    args = parser.parse_args()
    {"reactivation": run_reactivation, "persistence": run_persistence,
     "long-term": run_long_term, "persona": run_persona_patterns, "persona-proxies": run_tianchi_persona_proxies,
     "persona-uplift": run_tianchi_persona_uplift,
     "single-receipt-duration": run_tianchi_single_receipt_duration,
     "coupon-overlap": run_tianchi_coupon_overlap,
     "cumulative-coupons": run_tianchi_cumulative_coupon_history,
     "uplift-audit": run_tianchi_uplift_audit,
     "paired": run_paired_windows}[args.task]()


if __name__ == "__main__":
    main()
