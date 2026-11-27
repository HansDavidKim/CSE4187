"""Fit persistent checkout size/price preferences from repeated source purchases."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TYPE = "source_joint_gaussian_log_checkout_random_intercept"


def validate_spending_model(model):
    if (not isinstance(model, dict) or type(model.get("schema_version")) is not int or model["schema_version"] != 1
            or model.get("type") != TYPE):
        raise ValueError("incompatible checkout spending model")
    mean = np.asarray(model.get("mean"), dtype=float)
    if mean.shape != (2,) or not np.isfinite(mean).all():
        raise ValueError("checkout spending mean must contain two finite log amounts")
    for name, positive in (("between_covariance", False), ("within_covariance", True)):
        covariance = np.asarray(model.get(name), dtype=float)
        if (covariance.shape != (2, 2) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)):
            raise ValueError("checkout spending covariances must be finite symmetric matrices")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.min() < -1e-12 * max(1, abs(covariance).max()) or (positive and eigenvalues.min() <= 0):
            raise ValueError("checkout spending within covariance must be positive definite; between must be nonnegative")
    copula = model.get("assortment_price_copula")
    if copula is not None:
        rho = copula.get("rho") if isinstance(copula, dict) else None
        if (not isinstance(copula, dict) or copula.get("type") != "source_gaussian_rank_preferred_checkout_price"
                or isinstance(rho, bool) or not isinstance(rho, (int, float)) or not np.isfinite(rho) or abs(rho) > 1
                or model["between_covariance"][1][1] <= 0):
            raise ValueError("assortment price copula needs a finite correlation and varying checkout price preference")
    fulfillment = model.get("cart_fulfillment")
    if fulfillment is not None:
        fractions = fulfillment.get("fraction_counts") if isinstance(fulfillment, dict) else None
        if (not isinstance(fulfillment, dict) or fulfillment.get("type") != "source_observed_checkout_cart_fraction"
                or not isinstance(fractions, list) or not fractions):
            raise ValueError("cart fulfillment needs observed capacity fractions")
        seen = set()
        for row in fractions:
            if (not isinstance(row, list) or len(row) != 3 or any(type(value) is not int for value in row)
                    or not 0 <= row[0] <= row[1] or row[1] <= 0 or row[2] <= 0):
                raise ValueError("cart fulfillment fractions need integer 0<=numerator<=denominator and positive counts")
            fraction = Fraction(row[0], row[1])
            if fraction in seen:
                raise ValueError("cart fulfillment fractions must be unique")
            seen.add(fraction)


def spending_pool_weights(model, sourceweights, units, unitprices, preference):
    """Reweight the supported joint pool by conditional/marginal density, once."""
    validate_spending_model(model)
    weights, sizes, prices = (np.asarray(value, dtype=float) for value in (sourceweights, units, unitprices))
    preference = np.asarray(preference, dtype=float)
    if (weights.ndim != 1 or not len(weights) or sizes.shape != weights.shape or prices.shape != weights.shape
            or not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any()
            or not np.isfinite(sizes).all() or (sizes <= 0).any()
            or not np.isfinite(prices).all() or (prices <= 0).any()
            or preference.shape != (2,) or not np.isfinite(preference).all()):
        raise ValueError("checkout spending needs positive finite sizes/prices and a finite private preference")
    values = np.log(np.column_stack([sizes, prices]))
    within = np.asarray(model["within_covariance"])
    marginal = within + np.asarray(model["between_covariance"])
    def quadratic(center, covariance):
        differences = values - center
        return np.einsum("ij,ji->i", differences, np.linalg.solve(covariance, differences.T))
    # Determinants cancel during normalization; source frequencies already supply the marginal.
    ratio = -.5 * (quadratic(preference, within) - quadratic(model["mean"], marginal))
    if not np.isfinite(ratio).all():
        raise ValueError("checkout spending density ratio must be finite on the supported pool")
    with np.errstate(divide="ignore"):
        logits = np.log(weights / weights.max()) + ratio
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum()


def _log_density(value, mean, covariance):
    difference = value - mean
    return -.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
                   + difference @ np.linalg.solve(covariance, difference))


def fit_spending_model(episodes, cutoff, seed=42, *, purchase_rows=None):
    """Method-of-moments random intercept; source users and future targets stay held out."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("checkout spending cutoff must be timezone-aware")
    supported = episodes.loc[episodes.catalog_supported].copy()
    values = supported[["unit_count", "unit_mean_price_minor"]].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("checkout spending fitting needs positive finite source checkouts")
    supported[["log_units", "log_price"]] = np.log(values)
    selected = supported.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
    train = supported.loc[selected & supported["at"].lt(cutoff)]
    columns = ["log_units", "log_price"]
    means = train.groupby("user_id")[columns].mean()
    counts = train.groupby("user_id").size()
    degrees = len(train) - len(means)
    if degrees < 2 or len(means) < 3:
        raise ValueError("checkout spending fitting requires repeated purchases by multiple source customers")
    residuals = train[columns].to_numpy() - means.loc[train.user_id].to_numpy()
    within = residuals.T @ residuals / degrees
    between = means.cov().to_numpy() - within * np.mean(1 / counts)
    eigenvalues, eigenvectors = np.linalg.eigh(between)
    # Negative variance estimates are projected onto the nonnegative covariance cone.
    between = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    model = {"schema_version": 1, "type": TYPE, "variables": ["log_unit_count", "log_unit_mean_price_minor"],
             "mean": means.mean().tolist(), "between_covariance": between.tolist(), "within_covariance": within.tolist(),
             "fit": {"cutoff_at": cutoff.isoformat(), "seed": seed, "training_episodes": len(train),
                     "training_users": len(means), "repeat_training_users": int((counts > 1).sum()),
                     "within_degrees_of_freedom": degrees, "between_covariance_projected": bool((eigenvalues < 0).any()),
                     "user_rule": "SHA256(seed:user_id) first byte <205", "targets": "strictly before cutoff"},
             "limitations": ["Gaussian log-size/price random effects are a working observational model, not causal effects.",
                             "Five source months do not identify persistence across 104 weeks.",
                             "Purchase rows are unit proxies; contiguous equal-timestamp episodes are not verified order IDs."]}
    validate_spending_model(model)
    prior = np.asarray(model["mean"])
    marginal = within + between
    validation = {name: [] for name in ("train", "user_holdout", "time_holdout", "user_and_time_holdout")}
    for uid, history in supported.sort_values(["user_id", "at", "episode_id"], kind="stable").groupby("user_id"):
        mean, covariance = prior.copy(), between.copy()
        selected_user = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
        for at, tied in history.groupby("at", sort=True):
            current = tied[columns].to_numpy()
            if len(history.loc[history["at"].lt(at)]):
                name = ("train" if selected_user else "user_holdout") if at < cutoff else (
                    "time_holdout" if selected_user else "user_and_time_holdout")
                for value in current:
                    validation[name].append((-_log_density(value, mean, within + covariance),
                                              -_log_density(value, prior, marginal)))
            # A tied timestamp cannot be its own predictor; condition only after all tied targets.
            for value in current:
                gain = np.linalg.solve((covariance + within).T, covariance.T).T
                mean = mean + gain @ (value - mean)
                covariance = covariance - gain @ covariance
    model["validation"] = {name: {"records": len(losses),
        "persistent_nll": float(np.mean(losses, axis=0)[0]) if losses else None,
        "independent_nll": float(np.mean(losses, axis=0)[1]) if losses else None}
        for name, losses in validation.items()}
    if purchase_rows is not None:
        model["assortment_price_copula"] = fit_assortment_price_copula(episodes, purchase_rows, cutoff, seed)
        fulfillment = fit_cart_fulfillment(purchase_rows, cutoff, seed)
        if fulfillment is not None:
            model["cart_fulfillment"] = fulfillment
        validate_spending_model(model)
    return model


def fit_assortment_price_copula(episodes, rows, cutoff, seed=42):
    """Couple existing price marginals using training-user purchase ranks only."""
    cutoff = pd.Timestamp(cutoff)
    training_user = lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
    purchases = rows.loc[rows.action.eq("purchase") & rows["at"].lt(cutoff)
        & rows.user_id.map(training_user) & rows.price_minor.gt(0)]
    preferred = purchases.groupby("user_id").price_minor.quantile(.5, interpolation="lower").rename("preferred")
    checkouts = episodes.loc[episodes.catalog_supported & episodes["at"].lt(cutoff)
        & episodes.user_id.map(training_user)]
    average = checkouts.groupby("user_id").unit_mean_price_minor.mean().rename("checkout")
    paired = pd.concat([preferred, average], axis=1, join="inner")
    correlation = paired.preferred.corr(paired.checkout, method="spearman")
    if len(paired) < 3 or not np.isfinite(correlation):
        raise ValueError("assortment price coupling requires varying purchase prices from multiple training users")
    return {"type": "source_gaussian_rank_preferred_checkout_price", "rho": float(2 * np.sin(np.pi * correlation / 6)),
            "source_spearman": float(correlation), "training_users": len(paired), "training_purchase_rows": len(purchases),
            "training_checkout_episodes": len(checkouts),
            "preferred_price": "lower median observed purchase unit price per customer",
            "checkout_price": "mean fixed-catalog checkout unit-mean price per customer",
            "marginals": "existing home-group preferred-price histogram and persistent checkout Gaussian unchanged",
            "limitation": "observed customer-average price ranks approximate latent budget ranks; not an identified causal link"}


def fit_cart_fulfillment(rows, cutoff, seed=42):
    """Replay observed SKU units; empty initial source inventory is only a lower bound."""
    cutoff = pd.Timestamp(cutoff)
    selected = rows.loc[rows["at"].lt(cutoff) & rows.user_id.map(
        lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)]
    order = ["user_id", "at"] + (["source_event_id"] if "source_event_id" in selected else [])
    fractions, audit = Counter(), Counter()
    for _, history in selected.sort_values(order, kind="stable").groupby("user_id", sort=False):
        cart = Counter()
        active, before, units, matched = None, 0, 0, 0

        def finish():
            if units:
                audit["training_checkouts"] += 1
                audit["purchase_unit_proxies"] += units
                audit["matched_cart_unit_proxies"] += matched
                capacity = min(before, units)
                if capacity:
                    fractions[Fraction(matched, capacity)] += 1
                    audit["checkouts_with_known_cart_capacity"] += 1
                    audit["known_cart_capacity_unit_proxies"] += capacity

        for row in history.itertuples(index=False):
            key = (row.session, row.at) if row.action == "purchase" else None
            if key is None or key != active:
                finish()
                active, before, units, matched = key, sum(cart.values()), 0, 0
            product = str(row.product_id)
            if row.action == "cart":
                cart[product] += 1
            elif row.action == "remove":
                if not cart[product]:
                    audit["remove_rows_without_known_unit"] += 1
                cart[product] -= min(cart[product], 1)
            elif row.action == "purchase":
                units += 1
                held = min(cart[product], 1)
                matched += held
                cart[product] -= held
        finish()
    if not fractions:
        return None
    return {"type": "source_observed_checkout_cart_fraction",
            "fraction_counts": [[value.numerator, value.denominator, count] for value, count in sorted(fractions.items())],
            "fit": dict(audit), "fraction": "matched observed SKU units / min(known cart units before checkout, checkout units)",
            "source_units": "one action row is a unit proxy; contiguous same-user/session/timestamp purchases form one checkout",
            "inventory": "unknown initial cart; replay observed adds and clip unmatched removals without fabricating units",
            "empty_cart": "no fraction observation when cart capacity is zero",
            "limitations": "Observed source cart is a lower bound; no cart lifetime, TTL, or causal abandonment rate is identified."}


if __name__ == "__main__":
    from .fit_checkout_transition import build_checkout_episodes
    from .fit_donor_prior import load_prior_source

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    content = args.source_model.read_bytes()
    source = json.loads(content)
    rows, descriptor = load_prior_source(source, args.source_cache)
    cutoff = source.get("source_contract", {}).get("training_cutoff_at",
        source.get("source", {}).get("training_cutoff_at", source.get("validation", {}).get("cutoff_at_utc")))
    result = fit_spending_model(build_checkout_episodes(rows, source["catalog"]),
        cutoff, source["seed"], purchase_rows=rows)
    result["source"] = {"model_sha256": hashlib.sha256(content).hexdigest(), "prepared_cache": descriptor}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
