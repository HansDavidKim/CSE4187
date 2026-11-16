"""Fit descriptive 30-day use of directly 10%-off Tianchi O2O coupons."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


COLUMNS = ("User_id", "Merchant_id", "Coupon_id", "Discount_rate", "Distance", "Date_received", "Date")
COHORTS = ("no_observed_purchase", "purchased_within_28d", "lapsed_purchase")
NULLS = {"", "null", "none", "nan", "nat"}


def _text(values):
    text = values.astype("string").str.strip()
    return text.mask(text.isna() | text.str.lower().isin(NULLS))


def _dates(values, name):
    text = _text(values).str.replace(r"^(\d{8})\.0+$", r"\1", regex=True)
    malformed = text.notna() & ~text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if malformed.any() or (text.notna() & parsed.isna()).any():
        raise ValueError(f"invalid {name}: expected a valid YYYYMMDD date or a null sentinel")
    return parsed


def _direct_ninety_percent(discounts):
    accepted = []
    for value in discounts.dropna().unique():
        try:
            if ":" not in value and Decimal(value) == Decimal("0.9"):
                accepted.append(value)
        except InvalidOperation:
            pass
    return discounts.isin(accepted)


def _group(rows):
    redeemed = rows.loc[rows.delay_days.ge(0) & rows.delay_days.lt(30), "delay_days"]
    receipts = len(rows)
    return {"receipts": receipts, "redeemed": len(redeemed),
            "response_probability": len(redeemed) / receipts if receipts else 0.,
            "redemption_delay_days": sorted(redeemed.astype(int).tolist())}


def fit_tianchi_coupon(frame: pd.DataFrame, source: dict | None = None) -> dict:
    """One observation per receipt row; only strictly prior dates determine its cohort."""
    if not set(COLUMNS) <= set(frame) or frame.empty:
        raise ValueError("Tianchi coupon fitting requires nonempty offline training columns")
    users = _text(frame.User_id)
    if users.isna().any():
        raise ValueError("Tianchi user IDs must be observed")
    received, purchased = _dates(frame.Date_received, "Date_received"), _dates(frame.Date, "Date")
    if (purchased.notna() & received.notna() & purchased.lt(received)).any():
        raise ValueError("purchase Date before Date_received is invalid")
    if purchased.notna().sum() == 0:
        raise ValueError("observed purchase Dates are required to infer the observation end")
    observation_start = pd.concat([received, purchased]).min()
    observation_end = purchased.max()
    first_eligible, last_eligible = observation_start + pd.Timedelta(days=28), observation_end - pd.Timedelta(days=30)
    coupons = _text(frame.Coupon_id).notna()
    receipt = received.notna() & coupons
    direct = receipt & _direct_ninety_percent(_text(frame.Discount_rate))
    early = direct & received.lt(first_eligible)
    right_censored = direct & ~early & received.gt(last_eligible)
    eligible = direct & ~early & ~right_censored
    if not eligible.any():
        raise ValueError("no fully observed direct 0.9-paid-rate coupon receipts")
    receipts = pd.DataFrame({"user_id": users[eligible], "received_at": received[eligible],
                             "delay_days": (purchased[eligible] - received[eligible]).dt.days})
    history = pd.DataFrame({"user_id": users[purchased.notna()], "prior_purchase_at": purchased[purchased.notna()]})
    receipts = pd.merge_asof(receipts.sort_values(["received_at", "user_id"]),
                             history.sort_values(["prior_purchase_at", "user_id"]),
                             left_on="received_at", right_on="prior_purchase_at", by="user_id",
                             direction="backward", allow_exact_matches=False)
    receipts["cohort"] = "no_observed_purchase"
    prior = receipts.prior_purchase_at.notna()
    receipts.loc[prior, "cohort"] = "lapsed_purchase"
    recent = prior & (receipts.received_at - receipts.prior_purchase_at).le(pd.Timedelta(days=28))
    receipts.loc[recent, "cohort"] = "purchased_within_28d"
    groups = {"pooled": _group(receipts), **{cohort: _group(receipts.loc[receipts.cohort.eq(cohort)]) for cohort in COHORTS}}
    quality = {
        "rows": len(frame), "users": int(users.nunique()), "coupon_receipt_rows": int(receipt.sum()),
        "ordinary_purchase_rows": int((purchased.notna() & ~coupons).sum()),
        "coupon_purchase_rows": int((purchased.notna() & coupons).sum()),
        "missing_purchase_date_rows": int(purchased.isna().sum()),
        "receipt_date_without_coupon_id_rows": int((received.notna() & ~coupons).sum()),
        "exact_duplicate_extra_rows": int(frame.loc[:, list(COLUMNS)].duplicated().sum()),
        "exact_duplicate_participating_rows": int(frame.loc[:, list(COLUMNS)].duplicated(keep=False).sum()),
        "repeated_receipt_key_extra_rows": int(frame.loc[receipt, ["User_id", "Merchant_id", "Coupon_id", "Date_received"]].duplicated().sum()),
        "eligible_exact_duplicate_extra_rows": int(frame.loc[eligible, list(COLUMNS)].duplicated().sum()),
        "invalid_dates": 0, "purchase_before_receipt_rows": 0,
        "direct_10pct_receipts_before_window_exclusions": int(direct.sum()),
        "excluded_nonmatching_discount_receipts": int((receipt & ~direct).sum()),
        "excluded_insufficient_global_history_receipts": int(early.sum()),
        "excluded_incomplete_30d_followup_receipts": int(right_censored.sum()),
        "eligible_receipts": int(eligible.sum()),
        "eligible_observed_purchase_outside_30d": int((receipts.delay_days >= 30).sum()),
        "duplicates_dropped": 0,
    }
    return {"schema_version": 1, "type": "tianchi_o2o_descriptive_response", "discount_rate": .1,
            "source_paid_rate": .9, "window_days": 30, "inactivity_days": 28,
            "groups": groups, "source": source or {"kind": "explicit dataframe fixture"},
            "observation_start_date": observation_start.date().isoformat(),
            "observation_end_date": observation_end.date().isoformat(),
            "eligible_receipt_start_inclusive": first_eligible.date().isoformat(),
            "eligible_receipt_end_inclusive": last_eligible.date().isoformat(), "quality": quality,
            "cohort_fallback": "Use pooled only when the requested cohort has zero receipts; zero-support cohort encodes counts/probability zero and an empty delay list.",
            "assumptions": [
                "Receipt-row descriptive redemption rates, not randomized treatment effects, CATE, or additional visits caused by a coupon.",
                "Only direct paid-rate 0.9 coupons match a 10% discount; threshold reductions are excluded regardless of nominal percentage.",
                "Redemption uses integer Date minus Date_received in [0,30); this analytic window is not an asserted 15-day or 30-day source expiry.",
                "Ordinary purchases and all coupon purchase Dates contribute history; only purchase dates strictly before receipt are visible. Same-day purchases are excluded because intraday ordering is unknown.",
                "purchased_within_28d means a prior purchase in [receipt-28d,receipt); lapsed_purchase has an older observed purchase. no_observed_purchase does not mean a verified never-buyer.",
                "Earliest observed receipt/purchase date bounds history; latest observed purchase Date proxies the observation end. Source coverage before/after these bounds is unknown.",
                "Exact duplicate rows may represent repeated receipts or extraction duplicates. No unique receipt ID resolves the grain; all rows are retained and duplicate counts are audited.",
                "Tianchi offline O2O rates transfer to Cosmetics as a scenario assumption. Source customer and merchant IDs are not joined across datasets.",
                "In generation an observed redemption delay may be used as an availability floor for the first actual eligible action; it adds no visits, and unavailable opportunities can lower realized use below the descriptive source probability."]}


def _sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"coupon response output must be new: {args.output}")
    before, checksum = args.source.stat(), _sha256(args.source)
    provenance_path = args.source.with_suffix(".acquisition.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {"kind": "user-supplied local CSV; no acquisition sidecar"}
    if provenance.get("sha256", checksum) != checksum:
        raise ValueError("coupon source checksum differs from its acquisition record")
    frame = pd.read_csv(args.source, dtype=str, keep_default_na=False, na_filter=False)
    after = args.source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("coupon source changed during fitting")
    source = {"dataset": "Tianchi CCF2016 O2O offline stage1 training", "path": str(args.source.resolve()),
              "bytes": before.st_size, "sha256": checksum, "provenance": provenance,
              "fitter_sha256": _sha256(Path(__file__))}
    artifact = fit_tianchi_coupon(frame, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}-", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.link(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output),
                      "groups": {name: {key: value for key, value in group.items() if key != "redemption_delay_days"}
                                 for name, group in artifact["groups"].items()}, "quality": artifact["quality"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
