from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMAS = {
    "campaign_data.csv": ["campaign_id", "campaign_type", "start_date", "end_date"],
    "coupon_item_mapping.csv": ["coupon_id", "item_id"],
    "customer_demographics.csv": [
        "customer_id", "age_range", "marital_status", "rented", "family_size",
        "no_of_children", "income_bracket",
    ],
    "customer_transaction_data.csv": [
        "date", "customer_id", "item_id", "quantity", "selling_price",
        "other_discount", "coupon_discount",
    ],
    "item_data.csv": ["item_id", "brand", "brand_type", "category"],
    "train.csv": ["id", "campaign_id", "coupon_id", "customer_id", "redemption_status"],
}

SEQUENCE_FEATURES = [
    "transactions",
    "quantity",
    "spend",
    "other_discount",
    "coupon_discount",
    "unique_items",
    "unique_categories",
    "offers_started",
    "unique_coupons_started",
    "active_offers",
    "redemptions_reported",
]


def validate_amexpert(root: str | Path) -> dict:
    """Validate a user-supplied AmExpert archive without redistributing it."""
    root = Path(root)
    report = {"root": str(root), "files": {}}
    for name, columns in SCHEMAS.items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing AmExpert file: {path}")
        frame = pd.read_csv(path)
        if list(frame.columns) != columns:
            raise ValueError(f"unexpected schema in {name}: {list(frame.columns)}")
        report["files"][name] = {"rows": len(frame), "columns": columns}
    campaigns = pd.read_csv(root / "campaign_data.csv")
    transactions = pd.read_csv(root / "customer_transaction_data.csv")
    assignments = pd.read_csv(root / "train.csv")
    report.update(
        customers=int(transactions.customer_id.nunique()),
        transaction_start=str(pd.to_datetime(transactions.date).min().date()),
        transaction_end=str(pd.to_datetime(transactions.date).max().date()),
        campaigns=int(campaigns.campaign_id.nunique()),
        assignment_rows=len(assignments),
        redemption_rate=float(assignments.redemption_status.mean()),
        license_status="Competition-data reuse/redistribution license not verified; keep raw files local.",
    )
    return report


def _week(values: pd.Series, format: str) -> pd.Series:
    return pd.to_datetime(values, format=format).dt.to_period("W-SUN").dt.start_time


def build_customer_week_panel(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build dense independent customer sequences; never concatenate user boundaries."""
    root = Path(root)
    report = validate_amexpert(root)
    transactions = pd.read_csv(root / "customer_transaction_data.csv")
    items = pd.read_csv(root / "item_data.csv")
    campaigns = pd.read_csv(root / "campaign_data.csv")
    assignments = pd.read_csv(root / "train.csv")
    demographics = pd.read_csv(root / "customer_demographics.csv")

    transactions["week"] = _week(transactions.date, "%Y-%m-%d")
    transactions = transactions.merge(items, on="item_id", how="left", validate="many_to_one")
    weekly = transactions.groupby(["customer_id", "week"], observed=True).agg(
        transactions=("item_id", "size"),
        quantity=("quantity", "sum"),
        spend=("selling_price", "sum"),
        other_discount=("other_discount", lambda x: -float(x.sum())),
        coupon_discount=("coupon_discount", lambda x: -float(x.sum())),
        unique_items=("item_id", "nunique"),
        unique_categories=("category", "nunique"),
    ).reset_index()

    campaigns["start_week"] = _week(campaigns.start_date, "%d/%m/%y")
    campaigns["end_week"] = _week(campaigns.end_date, "%d/%m/%y")
    exposure = assignments.merge(
        campaigns[["campaign_id", "campaign_type", "start_week", "end_week"]],
        on="campaign_id",
        how="left",
        validate="many_to_one",
    )
    started = exposure.groupby(["customer_id", "start_week"], observed=True).agg(
        offers_started=("id", "size"),
        unique_coupons_started=("coupon_id", "nunique"),
    ).reset_index().rename(columns={"start_week": "week"})
    ended = exposure.groupby(["customer_id", "end_week"], observed=True).agg(
        redemptions_reported=("redemption_status", "sum"),
    ).reset_index().rename(columns={"end_week": "week"})

    active_parts = []
    for row in exposure[["customer_id", "start_week", "end_week"]].itertuples(index=False):
        active_parts.append(pd.DataFrame({
            "customer_id": row.customer_id,
            "week": pd.date_range(row.start_week, row.end_week, freq="7D"),
        }))
    active = pd.concat(active_parts, ignore_index=True).groupby(
        ["customer_id", "week"], observed=True
    ).size().rename("active_offers").reset_index()

    customers = np.union1d(transactions.customer_id.unique(), assignments.customer_id.unique())
    first_day = pd.to_datetime(transactions.date, format="%Y-%m-%d").min().normalize()
    last_day = pd.to_datetime(transactions.date, format="%Y-%m-%d").max().normalize()
    start = first_day + pd.Timedelta(days=(-first_day.weekday()) % 7)
    last_sunday = last_day - pd.Timedelta(days=(last_day.weekday() + 1) % 7)
    end = last_sunday - pd.Timedelta(days=6)
    if end < start:
        raise ValueError("transaction observation period contains no complete week")
    index = pd.MultiIndex.from_product(
        [customers, pd.date_range(start, end, freq="7D")], names=["customer_id", "week"]
    )
    panel = pd.DataFrame(index=index).reset_index()
    for frame in (weekly, started, active, ended):
        panel = panel.merge(frame, on=["customer_id", "week"], how="left", validate="one_to_one")
    panel[SEQUENCE_FEATURES] = panel[SEQUENCE_FEATURES].fillna(0.0)
    panel = panel.sort_values(["customer_id", "week"]).reset_index(drop=True)

    lengths = panel.groupby("customer_id").size()
    report["panel"] = {
        "rows": len(panel),
        "customers": int(panel.customer_id.nunique()),
        "weeks": int(panel.week.nunique()),
        "first_week": str(start.date()),
        "last_week": str(end.date()),
        "excluded_transaction_rows_in_partial_boundary_weeks": int(
            ((transactions.week < start) | (transactions.week > end)).sum()
        ),
        "excluded_reported_redemptions_outside_complete_weeks": int(
            exposure.loc[(exposure.end_week < start) | (exposure.end_week > end), "redemption_status"].sum()
        ),
        "min_sequence_length": int(lengths.min()),
        "max_sequence_length": int(lengths.max()),
        "features": SEQUENCE_FEATURES,
        "assumptions": [
            "Coupon assignment time is represented by campaign start week.",
            "Reported redemption is placed at campaign end week because exact redemption time is absent.",
            "Overlapping campaigns are retained as active_offers counts.",
            "Customer sequences remain separate when constructing lag windows.",
            "Only complete weeks inside the transaction observation period are retained; unobserved future weeks are not zero-filled.",
        ],
    }
    return panel, demographics, report


def save_amexpert_panel(root: str | Path, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panel, demographics, report = build_customer_week_panel(root)
    panel.to_csv(output / "customer_week_panel.csv.gz", index=False)
    demographics.to_csv(output / "customer_demographics.csv", index=False)
    arrays = [group[SEQUENCE_FEATURES].to_numpy(dtype=np.float64) for _, group in panel.groupby("customer_id")]
    np.savez_compressed(
        output / "customer_sequences.npz",
        values=np.concatenate(arrays),
        offsets=np.cumsum([0] + [len(x) for x in arrays]),
        features=np.asarray(SEQUENCE_FEATURES),
    )
    (output / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")

