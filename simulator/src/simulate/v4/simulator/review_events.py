"""Source-trained survey scheduling and separate conditional product-star sampling."""
from __future__ import annotations

import numpy as np
import pandas as pd


REVIEW_FEATURE_NAMES = ("log_prior_review_count", "has_prior_review", "prior_mean_centered",
                        "last_rating_centered", "log_days_since_review")
ORDER_FEATURE_NAMES = ("log_prior_order_count", "has_prior_order", "log_days_since_order",
                       "log_prior_review_count", "has_prior_review", "log_days_since_review")


def _time(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("review runtime timestamps must be timezone-aware")
    return value.tz_convert("UTC")


def review_context(prior_reviews, as_of_at, product_id=None):
    """Use reviews strictly before this event; identities are never linked across sources."""
    at = _time(as_of_at)
    history = []
    for row in prior_reviews:
        when = _time(row["event_at"])
        if when >= at:
            continue
        rating = row["rating"]
        if isinstance(rating, bool) or rating not in (1, 2, 3, 4, 5):
            raise ValueError("prior review ratings must be integers from 1 to 5")
        history.append((when, float(rating)))
    if not history:
        return {"prior_review_count": 0, "prior_mean_rating": None, "last_rating": None, "days_since_review": None}
    latest = max(when for when, _ in history)
    return {"prior_review_count": len(history), "prior_mean_rating": float(np.mean([score for _, score in history])),
            "last_rating": float(np.mean([score for when, score in history if when == latest])),
            "days_since_review": (at - latest).total_seconds() / 86400}


def order_review_context(prior_orders, prior_reviews, purchased_at):
    at = _time(purchased_at)
    orders = [when for value in prior_orders if (when := _time(value)) < at]
    return {**review_context(prior_reviews, at), "prior_order_count": len(orders),
            "days_since_order": (at - max(orders)).total_seconds() / 86400 if orders else None}


def context_features(context, kind="product_stars"):
    def log(value):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("review counts and elapsed history must be nonnegative and finite")
        return float(np.log1p(value)) if value is not None else 0.0
    count = context["prior_review_count"]
    features = [log(count), float(count > 0),
                float(context["prior_mean_rating"] - 3) if count else 0.0,
                float(context["last_rating"] - 3) if count else 0.0,
                log(context["days_since_review"])]
    if kind == "order_survey":
        orders = context["prior_order_count"]
        features = [log(orders), float(orders > 0), log(context["days_since_order"]), features[0], features[1], features[4]]
    elif kind != "product_stars":
        raise ValueError("unknown review component")
    values = np.asarray(features, dtype=float)
    if not np.isfinite(values).all() or count < 0:
        raise ValueError("review context must contain finite prior history")
    return values


def _probabilities(component, features):
    parameters = component["model"]
    values = np.asarray(features, dtype=float)
    values = np.clip(values, parameters["feature_min"], parameters["feature_max"])
    values = (values - parameters["scaler_mean"]) / parameters["scaler_scale"]
    logits = values @ np.asarray(parameters["coef"], dtype=float).T + parameters["intercept"]
    if len(parameters["classes"]) == 2:
        p = 1.0 / (1.0 + np.exp(-np.clip(logits[..., 0], -700, 700)))
        return np.stack([1 - p, p], axis=-1)
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=-1, keepdims=True)


def rating_probabilities(model, context):
    """Return probabilities for stars 1..5, conditional on an observed review arriving."""
    component = model["product_stars"]
    if component["model"]["classes"] != [1, 2, 3, 4, 5]:
        raise ValueError("product-star model must expose all five rating classes")
    return _probabilities(component, context_features(context)).tolist()


def sample_rating(model, context, rng):
    return int(rng.choice(np.arange(1, 6), p=rating_probabilities(model, context)))


def sample_review_schedule(model, context, rng):
    """A transported Olist 28-day survey-record opportunity, not a voluntary-review rate."""
    component = model["order_survey"]
    probability = float(_probabilities(component, context_features(context, "order_survey"))[1])
    cohort = "first_observed_order" if context["prior_order_count"] == 0 else "repeat_observed_order"
    if rng.random() >= probability:
        return {"scheduled": False, "delay_seconds": None, "source_order_score": None,
                "response_probability": probability, "source_cohort": cohort}
    bank = component["delay_score_banks"].get(cohort) or component["delay_score_banks"]["pooled"]
    delay, order_score = bank[int(rng.integers(len(bank)))]
    return {"scheduled": True, "delay_seconds": float(delay), "source_order_score": int(order_score),
            "response_probability": probability, "source_cohort": cohort}


def validate_review_model(model):
    if model.get("schema_version") != 1 or model.get("type") != "source_conditional_review_events":
        raise ValueError("unsupported review-event artifact")
    for kind, names, classes in (("product_stars", REVIEW_FEATURE_NAMES, [1, 2, 3, 4, 5]),
                                  ("order_survey", ORDER_FEATURE_NAMES, [0, 1])):
        parameters = model[kind]["model"]
        if parameters["feature_names"] != list(names) or parameters["classes"] != classes:
            raise ValueError("review coefficient/context order differs")
        numeric = [parameters[key] for key in ("scaler_mean", "scaler_scale", "feature_min", "feature_max", "coef", "intercept")]
        if not all(np.isfinite(np.asarray(value)).all() for value in numeric):
            raise ValueError("review parameters must be finite")
        if np.any(np.asarray(parameters["scaler_scale"]) <= 0):
            raise ValueError("review scaling must be positive")
        rows, width = (1 if len(classes) == 2 else len(classes)), len(names)
        if (np.shape(parameters["coef"]) != (rows, width) or np.shape(parameters["intercept"]) != (rows,)
                or any(np.shape(parameters[key]) != (width,) for key in ("scaler_mean", "scaler_scale", "feature_min", "feature_max"))
                or np.any(np.asarray(parameters["feature_min"]) > parameters["feature_max"])):
            raise ValueError("review parameter shapes or bounds differ")
    if model["order_survey"]["window_days"] != 28:
        raise ValueError("review survey observation window must be explicit")
    for rows in model["order_survey"]["delay_score_banks"].values():
        if any(not np.isfinite(delay) or not 0 <= delay < 28 * 86400 or score not in (1, 2, 3, 4, 5)
               for delay, score in rows):
            raise ValueError("invalid observed survey delay/score")
    if not model["order_survey"]["delay_score_banks"]["pooled"]:
        raise ValueError("review scheduling requires observed source delays")
