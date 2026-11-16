"""Fit observed Olist survey scheduling and a separate verified Amazon star model."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from simulate.v4.simulator.review_events import ORDER_FEATURE_NAMES, REVIEW_FEATURE_NAMES, _probabilities, validate_review_model


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_amazon(path):
    records, quality = [], Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            quality["raw_reviews"] += 1
            if row.get("verified_purchase") is not True:
                quality["excluded_not_verified_purchase"] += 1
                continue
            rating, user, when, item = row.get("rating"), row.get("user_id"), row.get("timestamp"), row.get("asin")
            if (isinstance(rating, bool) or rating not in (1, 2, 3, 4, 5) or not isinstance(user, str)
                    or not user or not isinstance(item, str) or not item or isinstance(when, bool)
                    or not isinstance(when, (int, float)) or not np.isfinite(when)):
                quality["excluded_invalid_metadata"] += 1
                continue
            records.append((user, item, when, int(rating)))
    frame = pd.DataFrame(records, columns=["user", "item", "at", "rating"])
    frame["at"] = pd.to_datetime(frame["at"], unit="ms", utc=True, errors="coerce")
    quality["excluded_invalid_timestamp"] = int(frame["at"].isna().sum())
    frame = frame.dropna(subset=["at"])
    duplicate = frame.duplicated(["user", "item", "at", "rating"])
    quality["excluded_duplicate_user_item_time_rating"] = int(duplicate.sum())
    frame = frame.loc[~duplicate].reset_index(drop=True)
    quality["included_verified_reviews"] = len(frame)
    quality["users"] = int(frame.user.nunique())
    return frame, dict(quality)


def review_history_contexts(queries, history):
    """As-of join excludes every same-time review, including ambiguous source ties."""
    states = history.groupby(["user", "at"], as_index=False).rating.agg(["sum", "count"]).reset_index()
    by_user = states.groupby("user", sort=False)
    states["prior_review_count"] = by_user["count"].cumsum()
    states["prior_mean_rating"] = by_user["sum"].cumsum() / states.prior_review_count
    states["last_rating"] = states["sum"] / states["count"]
    states["last_review_at"] = states["at"]
    states = states[["user", "at", "prior_review_count", "prior_mean_rating", "last_rating", "last_review_at"]]
    result = pd.merge_asof(queries.sort_values("at"), states.sort_values("at"), on="at", by="user", allow_exact_matches=False)
    result["prior_review_count"] = result.prior_review_count.fillna(0).astype(int)
    result["days_since_review"] = (result["at"] - result.last_review_at).dt.total_seconds() / 86400
    return result


def _features(frame, kind):
    count = frame.prior_review_count.to_numpy(dtype=float)
    reviews = [np.log1p(count), (count > 0).astype(float),
               frame.prior_mean_rating.fillna(3).to_numpy() - 3,
               frame.last_rating.fillna(3).to_numpy() - 3,
               np.log1p(frame.days_since_review.fillna(0).to_numpy())]
    if kind == "product_stars":
        return np.column_stack(reviews)
    orders = frame.prior_order_count.to_numpy(dtype=float)
    return np.column_stack([np.log1p(orders), (orders > 0).astype(float),
                            np.log1p(frame.days_since_order.fillna(0).to_numpy()), reviews[0], reviews[1], reviews[4]])


def _masks(frame, outcome_days=0):
    heldout = frame.user.map(lambda user: int.from_bytes(hashlib.sha256(user.encode()).digest()[:4], "big") % 5 == 0).to_numpy()
    cutoff = frame.loc[~heldout, "at"].quantile(.8)
    early = (frame["at"].add(pd.Timedelta(days=outcome_days)).le(cutoff) if outcome_days
             else frame["at"].lt(cutoff)).to_numpy()
    future = frame["at"].ge(cutoff).to_numpy()
    return {"train": ~heldout & early, "user_holdout": heldout & early,
            "time_holdout": ~heldout & future, "user_and_time_holdout": heldout & future}, cutoff


def _fit_parameters(features, labels, rows, regularization):
    mean, scale = features[rows].mean(axis=0), features[rows].std(axis=0)
    scale[scale == 0] = 1
    estimator = LogisticRegression(C=regularization, solver="lbfgs", max_iter=250, random_state=7)
    estimator.fit((features[rows] - mean) / scale, labels[rows])
    return {"classes": estimator.classes_.tolist(), "coef": estimator.coef_.tolist(), "intercept": estimator.intercept_.tolist(),
            "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
            "feature_min": features[rows].min(axis=0).tolist(), "feature_max": features[rows].max(axis=0).tolist(),
            "regularization_C": regularization, "n_iter": estimator.n_iter_.tolist()}


def _select_star_training(frame, features, labels, train, cutoff):
    """Select recency and shrinkage using only a disjoint inner future-user panel."""
    outer_rows = np.flatnonzero(train)
    local = frame.iloc[outer_rows]
    heldout = local.user.map(lambda user: int.from_bytes(hashlib.sha256(("review-inner:" + user).encode()).digest()[:4], "big") % 5 == 0).to_numpy()
    inner_cutoff = local.loc[~heldout, "at"].quantile(.8)
    inner_train = outer_rows[~heldout & local["at"].lt(inner_cutoff).to_numpy()]
    inner_validation = outer_rows[heldout & local["at"].ge(inner_cutoff).to_numpy()]
    results = []
    for days in (None, 365, 1095, 1825):
        rows = inner_train if days is None else inner_train[frame.iloc[inner_train]["at"].ge(inner_cutoff - pd.Timedelta(days=days)).to_numpy()]
        if not len(inner_validation) or sorted(np.unique(labels[rows]).tolist()) != [1, 2, 3, 4, 5]:
            continue
        for regularization in (.01, 1.):
            parameters = _fit_parameters(features, labels, rows, regularization)
            probabilities = _probabilities({"model": parameters}, features[inner_validation])
            selected = probabilities[np.arange(len(inner_validation)), labels[inner_validation] - 1]
            results.append({"lookback_days": days, "regularization_C": regularization, "fit_records": len(rows),
                            "nll": float(-np.log(np.maximum(selected, 1e-15)).mean())})
    chosen = min(results, key=lambda row: row["nll"]) if results else {"lookback_days": None, "regularization_C": 1.}
    days = chosen["lookback_days"]
    fit_rows = outer_rows if days is None else outer_rows[frame.iloc[outer_rows]["at"].ge(cutoff - pd.Timedelta(days=days)).to_numpy()]
    selection = {"method": "inner user-and-time NLL; candidate selection never reads outer holdout labels",
                 "inner_user_rule": "sha256('review-inner:' + source user) modulo 5 equals 0",
                 "inner_time_cutoff": inner_cutoff.isoformat(), "inner_train_records": len(inner_train),
                 "inner_validation_records": len(inner_validation), "candidates": results, "selected": chosen,
                 "fit_records": len(fit_rows), "fit_users": int(frame.iloc[fit_rows].user.nunique()),
                 "fit_start_at": frame.iloc[fit_rows]["at"].min().isoformat(),
                 "deployment_scope": "forward-time source transfer; older historical user-only fit may worsen"}
    if not results:
        selection["fallback_reason"] = "insufficient inner class support or future-user validation records; use full outer train and C=1"
    return fit_rows, chosen["regularization_C"], selection


def _fit_component(frame, kind):
    features = _features(frame, kind)
    labels = frame.rating.to_numpy(dtype=int)
    masks, cutoff = _masks(frame, outcome_days=28 if kind == "order_survey" else 0)
    train = masks["train"]
    fit_rows, regularization, selection = (np.flatnonzero(train), 1., None)
    if kind == "product_stars":
        fit_rows, regularization, selection = _select_star_training(frame, features, labels, train, cutoff)
    parameters = _fit_parameters(features, labels, fit_rows, regularization)
    expected = [1, 2, 3, 4, 5] if kind == "product_stars" else [0, 1]
    if parameters["classes"] != expected:
        raise ValueError(f"insufficient source labels for {kind}")
    component = {"model": {**parameters, "type": "regularized_conditional_maximum_entropy", "feature_names": list(REVIEW_FEATURE_NAMES if kind == "product_stars" else ORDER_FEATURE_NAMES),
                    "bounds_policy": "clip transformed features to training-only extrema for source-extrapolation control"},
                 "validation": {"time_cutoff": cutoff.isoformat(), "user_holdout_rule": "sha256(source user) modulo 5 equals 0; source IDs never cross datasets",
                                "label_window_purge_days": 28 if kind == "order_survey" else 0,
                                "purged_between_training_and_time_holdout": int((~np.logical_or.reduce(list(masks.values()))).sum())}}
    probabilities = _probabilities(component, features)
    if selection is not None:
        component["training_selection"] = selection
    frequencies = np.array([(labels[fit_rows] == value).mean() for value in expected])
    indices = np.searchsorted(expected, labels)
    for name, mask in masks.items():
        selected = probabilities[mask, indices[mask]]
        metrics = {"records": int(mask.sum()), "nll": float(-np.log(np.maximum(selected, 1e-15)).mean()) if mask.any() else None,
                   "frequency_baseline_nll": float(-np.log(frequencies[indices[mask]]).mean()) if mask.any() else None,
                   "observed_class_share": [(labels[mask] == value).mean().item() for value in expected] if mask.any() else [],
                   "predicted_class_share": probabilities[mask].mean(axis=0).tolist() if mask.any() else []}
        if kind == "order_survey" and mask.any():
            metrics["brier"] = float(np.mean((probabilities[mask, 1] - labels[mask]) ** 2))
        component["validation"][name] = metrics
    component["training_frequency_baseline"] = dict(zip(map(str, expected), frequencies.tolist()))
    return component, masks


def load_olist(folder):
    folder = Path(folder)
    orders = pd.read_csv(folder / "olist_orders_dataset.csv", usecols=["order_id", "customer_id", "order_purchase_timestamp"])
    people = pd.read_csv(folder / "olist_customers_dataset.csv", usecols=["customer_id", "customer_unique_id"])
    reviews = pd.read_csv(folder / "olist_order_reviews_dataset.csv", usecols=["review_id", "order_id", "review_score", "review_creation_date", "review_answer_timestamp"])
    orders = orders.merge(people, on="customer_id", validate="one_to_one").rename(columns={"customer_unique_id": "user", "order_purchase_timestamp": "at"})
    orders["at"] = pd.to_datetime(orders["at"], errors="coerce")
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews.review_answer_timestamp, errors="coerce")
    reviews["review_creation_date"] = pd.to_datetime(reviews.review_creation_date, errors="coerce")
    joined = reviews.merge(orders[["order_id", "user", "at"]], on="order_id", validate="many_to_one", how="left")
    joined["delay_seconds"] = (joined.review_answer_timestamp - joined["at"]).dt.total_seconds()
    valid = joined.delay_seconds.ge(0) & joined.review_score.isin([1, 2, 3, 4, 5])
    first = joined.loc[valid].sort_values("review_answer_timestamp").drop_duplicates("order_id")
    invalid_orders = set(joined.order_id) - set(first.order_id)
    capture_end = reviews.review_answer_timestamp.max()
    mature = orders["at"].notna() & orders["at"].add(pd.Timedelta(days=28)).le(capture_end) & ~orders.order_id.isin(invalid_orders)
    first28 = first.loc[first.delay_seconds.lt(28 * 86400)]
    history = first28[["user", "review_answer_timestamp", "review_score"]].rename(columns={"review_answer_timestamp": "at", "review_score": "rating"})
    order_blocks = orders.groupby(["user", "at"]).size().rename("size").reset_index().sort_values("at")
    order_blocks["prior_order_count"] = order_blocks.groupby("user")["size"].cumsum() - order_blocks["size"]
    order_blocks["last_order_at"] = order_blocks.groupby("user")["at"].shift()
    frame = orders.loc[mature].merge(order_blocks[["user", "at", "prior_order_count", "last_order_at"]], on=["user", "at"], validate="many_to_one")
    frame["days_since_order"] = (frame["at"] - frame.last_order_at).dt.total_seconds() / 86400
    frame = review_history_contexts(frame, history)
    targets = first28.set_index("order_id")
    frame["rating"] = frame.order_id.isin(targets.index).astype(int)
    frame["delay_seconds"] = frame.order_id.map(targets.delay_seconds)
    frame["source_order_score"] = frame.order_id.map(targets.review_score)
    quality = {"orders": len(orders), "review_rows": len(reviews), "reviewed_orders_any_observed_delay": int(reviews.order_id.nunique()),
               "answer_before_purchase_rows": int(joined.delay_seconds.lt(0).sum()), "excluded_orders_without_valid_review_time": len(invalid_orders),
               "excluded_not_mature_28d_or_bad_purchase_clock": int((~(orders["at"].notna() & orders["at"].add(pd.Timedelta(days=28)).le(capture_end))).sum()),
               "included_orders": len(frame), "included_review_within28d": int(frame.rating.sum()),
               "responses_outside28d_not_lifetime_nonresponse": int(first.delay_seconds.ge(28 * 86400).sum()),
               "capture_end_proxy_naive_source_clock": capture_end.isoformat(),
               "last_observed_survey_creation_naive_source_clock": reviews.review_creation_date.max().isoformat(),
               "capture_end_is_official_extract_time": False,
               "clock_policy": "Source clock has no timezone. Only same-source elapsed times and order are used; no UTC interpretation is asserted.",
               "projection": "Earliest valid answer per order; later observed answers do not create multiple synthetic reviews."}
    return frame, quality


def fit_review_events(amazon_path, olist_folder, seed=7):
    amazon, amazon_quality = load_amazon(amazon_path)
    stars_frame = review_history_contexts(amazon, amazon)
    stars, _ = _fit_component(stars_frame, "product_stars")
    del stars_frame, amazon
    survey_frame, survey_quality = load_olist(olist_folder)
    survey, masks = _fit_component(survey_frame, "order_survey")
    banks, cohort_counts = {}, {}
    for name, mask in {"pooled": np.ones(len(survey_frame), dtype=bool),
                       "first_observed_order": survey_frame.prior_order_count.eq(0).to_numpy(),
                       "repeat_observed_order": survey_frame.prior_order_count.gt(0).to_numpy()}.items():
        scope = survey_frame.loc[mask & masks["train"]]
        positive = scope.loc[scope.rating.eq(1), ["delay_seconds", "source_order_score"]]
        banks[name] = [[float(delay), int(score)] for delay, score in positive.itertuples(index=False, name=None)]
        cohort_counts[name] = {"train_orders": len(scope), "train_reviewed_within28d": int(scope.rating.sum()),
                              "observed_panel_q": float(scope.rating.mean()) if len(scope) else None, "stored_observed_delay_score_pairs": len(positive)}
    survey.update(window_days=28, delay_score_banks=banks, cohort_counts=cohort_counts,
                  delay_bank_policy="all observed positive outer-training delay/score pairs; no random compression",
                  target="observed survey answer within [purchase,purchase+28d) in provided Olist panel, not natural voluntary-review propensity")
    source_paths = [Path(amazon_path), *(Path(olist_folder) / f"olist_{name}_dataset.csv" for name in ("orders", "customers", "order_reviews"))]
    sources = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in source_paths]
    artifact = {"schema_version": 1, "type": "source_conditional_review_events", "seed": seed,
                "product_stars": stars, "order_survey": survey, "source_quality": {"amazon": amazon_quality, "olist": survey_quality},
                "sources": sources, "fitter_sha256": _sha(__file__),
                "runtime_sha256": _sha(Path(__file__).parents[1] / "simulator/review_events.py"),
                "assumptions": ["Olist email satisfaction-survey scheduling is transported to Cosmetics as an explicit opportunity policy, not voluntary review propensity.",
                    "Amazon verified-purchase product stars and Olist order-experience scores are different constructs; source_order_score is a diagnostic only.",
                    "Product stars are sampled on arrival from strict prior review history. Combining Amazon stars with Olist scheduling is cross-source factorization, not an observed joint distribution.",
                    "No price sensitivity, SKU rating effect, true signup, unreviewed Amazon purchase denominator or individual causal review effect is inferred.",
                    "Olist followup maturity uses the maximum observed answer timestamp as an administrative-end proxy; true extraction time and invitation denominator are not supplied.",
                    "First observed order/review is left-truncated history, not lifetime first. Same-time observations are excluded from all prior context.",
                    "Delay banks retain all observed outer-training pairs without replacing extreme values; the explicit 28-day target excludes later responses, not their lifetime existence.",
                    "No review session, retention visit, review text, or cross-dataset customer/product identity is fabricated."],
                "runtime_event_contract": {"raw_schema_version": 5, "rating_kind": "product_star_scenario", "session_id": None,
                                           "product_id": "actual purchased product reference", "order_line_no": "matching purchased item line",
                                           "related_purchase_event_id": "actual generated purchase reference", "rating": "integer 1..5",
                                           "uniqueness": "purchase_event_id/product_id; quantity does not multiply reviews",
                                           "schedule": "one order response/delay draw transported to its distinct purchased products"}}
    validate_review_model(artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon", type=Path, default=Path("data/raw/amazon_reviews_2023/All_Beauty.jsonl.gz"))
    parser.add_argument("--olist", type=Path, default=Path("data/raw/kaggle/olist"))
    parser.add_argument("--output", type=Path, default=Path("data/models/review_events.json"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"review artifact already exists: {args.output}")
    model = fit_review_events(args.amazon, args.olist, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
        temporary = Path(stream.name)
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink()
    print(json.dumps({"output": str(args.output), "sha256": _sha(args.output), "source_quality": model["source_quality"],
                      "validation": {name: model[name]["validation"] for name in ("product_stars", "order_survey")}}, indent=2))


if __name__ == "__main__":
    main()
