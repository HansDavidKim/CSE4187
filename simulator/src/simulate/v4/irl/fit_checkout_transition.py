"""Fit checkout-price transitions without inventing baskets, quantities or orders."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from simulate.v4.irl.fit_donor_prior import load_prior_source


TYPE = "source_laplace_checkout_log_unitmean_price_transition"


def validate_checkout_kernel(kernel):
    if (not isinstance(kernel, dict) or type(kernel.get("schema_version")) is not int
            or kernel["schema_version"] != 1 or kernel.get("type") != TYPE):
        raise ValueError("incompatible checkout transition kernel")
    for name in ("loc", "scale"):
        value = kernel.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError("checkout Laplace location/scale must be finite numbers")
    if kernel["scale"] < 0 or type(kernel.get("training_transitions")) is not int or kernel["training_transitions"] < 1:
        raise ValueError("checkout fitting requires positive transition support and a nonnegative scale")


def checkout_pool_weights(kernel, sourceweights, unitmeanprices, last_meanprice):
    """Normalize source frequency times the learned price likelihood, with no smoothing."""
    validate_checkout_kernel(kernel)
    weights, prices = np.asarray(sourceweights), np.asarray(unitmeanprices)
    if (weights.ndim != 1 or not len(weights) or prices.shape != weights.shape
            or weights.dtype.kind not in "iuf" or prices.dtype.kind not in "iuf"):
        raise ValueError("checkout pool needs matching numeric frequency and unit-mean price vectors")
    weights, prices = weights.astype(float), prices.astype(float)
    if (not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any()
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError("checkout frequencies must be nonnegative with support and prices positive/finite")
    # Normalize before summation, including cold starts, to avoid overflowing source counts.
    weights /= weights.max()
    if last_meanprice is None:
        return weights / weights.sum()
    if (isinstance(last_meanprice, bool) or not isinstance(last_meanprice, numbers.Real)
            or not math.isfinite(last_meanprice) or last_meanprice <= 0):
        raise ValueError("last checkout unit-mean price must be positive and finite")
    changes = np.log(prices) - np.log(float(last_meanprice))
    if kernel["scale"] == 0:
        weights = np.where(changes == kernel["loc"], weights, 0.)
        if not weights.any():
            raise ValueError("checkout point-mass transition has no supported candidate")
        return weights / weights.sum()
    distances = np.abs(changes - kernel["loc"])
    with np.errstate(divide="ignore", over="ignore"):
        logits = np.log(weights) - (distances - distances[weights > 0].min()) / kernel["scale"]
    probabilities = np.exp(logits - np.max(logits))
    return probabilities / probabilities.sum()


def build_checkout_episodes(rows, catalog):
    """Only contiguous purchase rows sharing user/session/time form one unit-proxy basket."""
    if not {"user_id", "session", "at", "action", "product_id"} <= set(rows) or rows.empty:
        raise ValueError("checkout fitting needs the complete prepared action stream")
    if not isinstance(rows["at"].dtype, pd.DatetimeTZDtype) or rows["at"].isna().any():
        raise ValueError("checkout source timestamps must be known and timezone-aware")
    if not rows.action.isin(["view", "cart", "remove", "purchase"]).all():
        raise ValueError("checkout source has an unknown action")
    prices = {str(item["product_id"]): item["unit_price_minor"] for item in catalog}
    if (len(prices) != len(catalog) or any(isinstance(price, bool) or not isinstance(price, numbers.Integral)
                                         or price <= 0 for price in prices.values())):
        raise ValueError("checkout fixed catalog needs unique products and positive integer prices")
    order = ["user_id", "at"] + (["source_event_id"] if "source_event_id" in rows else [])
    source = rows.sort_values(order, kind="stable").copy()
    previous = source.shift()
    continuing = (source.action.eq("purchase") & previous.action.eq("purchase")
                  & source.user_id.eq(previous.user_id) & source.session.eq(previous.session)
                  & source["at"].eq(previous["at"]))
    source["episode_id"] = (source.action.eq("purchase") & ~continuing).cumsum()
    result = []
    for identity, units in source.loc[source.action.eq("purchase")].groupby("episode_id", sort=False):
        basket = Counter(units.product_id.astype(str))
        supported = set(basket) <= prices.keys()
        mean = sum(prices[pid] * count for pid, count in basket.items()) / len(units) if supported else None
        result.append({"episode_id": int(identity), "user_id": str(units.user_id.iloc[0]),
                       "session": units.session.iloc[0], "at": units["at"].iloc[0],
                       "unit_count": len(units), "unit_mean_price_minor": mean,
                       "catalog_supported": supported})
    return pd.DataFrame(result, columns=["episode_id", "user_id", "session", "at", "unit_count",
                                         "unit_mean_price_minor", "catalog_supported"])


def _price_transitions(episodes):
    records = []
    for uid, history in episodes.sort_values(["user_id", "at", "episode_id"], kind="stable").groupby("user_id", sort=False):
        previous = None
        for at, tied in history.groupby("at", sort=True):
            if previous is not None:
                for row in tied.itertuples(index=False):
                    if row.catalog_supported and previous.catalog_supported:
                        price, anchor = float(row.unit_mean_price_minor), float(previous.unit_mean_price_minor)
                        records.append({"user_id": uid, "at": at, "previous_at": previous.at,
                                        "log_price": float(np.log(price)),
                                        "log_price_change": float(np.log(price) - np.log(anchor))})
            # The latest episode is usable only by a strictly later target timestamp.
            previous = next(tied.iloc[-1:].itertuples(index=False))
    return pd.DataFrame(records, columns=["user_id", "at", "previous_at", "log_price", "log_price_change"])


def _laplace_fit(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("checkout Laplace fit needs actual finite training observations")
    loc = float(np.median(values))
    return loc, float(np.mean(np.abs(values - loc)))


def _nll(values, loc, scale):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, 0
    if scale == 0:
        unsupported = int(np.count_nonzero(values != loc))
        return (None if unsupported else 0.), unsupported
    loss = math.log(2 * scale) + np.abs(values - loc) / scale
    return float(np.mean(loss)), 0


def fit_checkout_kernel(episodes, cutoff, currency, amount_unit, seed=42, source=None):
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("checkout training cutoff must be timezone-aware")
    transitions = _price_transitions(episodes)
    training_user = lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
    selected = transitions.user_id.map(training_user)
    before = transitions["at"].lt(cutoff)
    train = selected & before
    loc, scale = _laplace_fit(transitions.loc[train, "log_price_change"])
    initial = episodes.loc[episodes.catalog_supported & episodes.user_id.map(training_user) & episodes["at"].lt(cutoff)]
    baseline_loc, baseline_scale = _laplace_fit(np.log(initial.unit_mean_price_minor.to_numpy(dtype=float)))
    masks = {"train": train, "user_holdout": ~selected & before,
             "time_holdout": selected & ~before, "user_and_time_holdout": ~selected & ~before}
    validation = {}
    for name, mask in masks.items():
        part = transitions.loc[mask]
        conditional, unsupported = _nll(part.log_price_change, loc, scale)
        independent, baseline_unsupported = _nll(part.log_price, baseline_loc, baseline_scale)
        validation[name] = {"records": len(part), "users": int(part.user_id.nunique()),
                            "transition_nll": conditional, "independent_log_price_nll": independent,
                            "zero_support_transitions": unsupported, "zero_support_independent": baseline_unsupported,
                            "nll_gain": independent - conditional if len(part) and scale > 0 and baseline_scale > 0 else None}
    kernel = {"schema_version": 1, "type": TYPE, "loc": loc, "scale": scale,
              "training_transitions": int(train.sum()), "training_users": int(transitions.loc[train, "user_id"].nunique()),
              "currency": currency, "amount_unit": amount_unit, "seed": seed,
              "price_definition": "sum(fixed catalog unit price * observed unit-proxy count) / total unit-proxy count",
              "split": {"cutoff_at_utc": cutoff.isoformat(), "user_rule": "SHA256(seed:user_id) first byte <205",
                        "training_target": "train users and target_at strictly before cutoff",
                        "history": "last actual same-user episode with previous_at strictly less than target_at"},
              "source": source or {"kind": "explicit checkout fixture"},
              "source_counts": {"episodes": len(episodes), "catalog_supported_episodes": int(episodes.catalog_supported.sum()),
                                "unit_rows": int(episodes.unit_count.sum()), "price_transitions": len(transitions)},
              "independent_price_baseline": {"loc": baseline_loc, "scale": baseline_scale, "training_episodes": len(initial)},
              "validation": validation,
              "template_policy": {"profile": "observed_count tickets without replacement",
                                  "after_exhaustion": "observed home-group or global templates with profile-observed unit counts",
                                  "pool_price_weight": "source frequency times Laplace(log next unitmean - log last unitmean; loc, scale)",
                                  "no_profile_templates": "existing source pool behavior; no fabricated personal size law",
                                  "cold_price": "source frequencies unchanged", "point_mass_out_of_support": "raise; no epsilon or unreported smoothing"},
              "limitations": ["Finite personal template tickets are an empirical extrapolation policy, not measured future SKU probabilities.",
                              "Price NLL is a working density in log catalog unit-mean price, not basket/SKU likelihood or causal price elasticity.",
                              "The unchanged fixed catalog comes from the base source artifact; it is not re-fitted within these transition splits.",
                              "Original purchase rows are quantity proxies and contiguous episode groups are not verified order IDs.",
                              "Full-history donor templates/pools remain latent source descriptors; only these transition parameters use training targets."]}
    validate_checkout_kernel(kernel)
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/models/shopping_profiles_alignment_timing_candidate.json"))
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("checkout candidate and report must be new")
    content = args.base.read_bytes()
    base = json.loads(content)
    if base.get("choice_unit") != "source_checkout_episode" or "checkout_kernel" in base:
        raise ValueError("augment a source checkout artifact without an existing checkout kernel")
    rows, descriptor = load_prior_source(base, args.source_cache)
    episodes = build_checkout_episodes(rows, base["catalog"])
    expected = base["source_summary"]["checkout_episodes"]
    if len(episodes) != expected:
        raise ValueError("contiguous source checkout count differs from the base artifact")
    source = {"base_artifact_path": str(args.base), "base_artifact_sha256": hashlib.sha256(content).hexdigest(),
              "prepared_source_cache": descriptor, "fitter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    kernel = fit_checkout_kernel(episodes, base["validation"]["cutoff_at_utc"], base["currency"], base["minor_unit"], base["seed"], source)
    candidate = copy.deepcopy(base)
    candidate["checkout_kernel"] = kernel
    assert {key: value for key, value in candidate.items() if key != "checkout_kernel"} == base
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(candidate, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report = {"artifact_path": str(args.output), "artifact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "base_fields_unchanged": True, "kernel": kernel}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "loc": kernel["loc"], "scale": kernel["scale"],
                      "training_transitions": kernel["training_transitions"], "validation": kernel["validation"]}))


if __name__ == "__main__":
    main()
