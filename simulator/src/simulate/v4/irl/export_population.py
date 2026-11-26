"""Export population calibration for generating new profiles without source records."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from simulate.v4.simulator.population_profiles import aggregate_population
from simulate.v4.simulator.population_timing import aggregate_review_model, aggregate_visit_timing
from .source_contract import canonical_available_actions, utc_cutoff


def _select(model, keys):
    return {key: model[key] for key in keys.split()}


def export_population(models: Path, output: Path, reward_gmm: Path | None = None, *,
                      return_state: Path | None = None, checkout_spending: Path | None = None) -> None:
    """Convert fitted v10 assets into model parameters and aggregate calibration."""
    if output.exists():
        raise FileExistsError(output)

    def read(name):
        return json.loads((models / name).read_bytes())

    original_shopping_bytes = (models / "shopping_joint_latent_source_traits.json").read_bytes()
    shopping = json.loads(original_shopping_bytes)
    original_return_bytes = (models / "return_model.json").read_bytes()
    returning = json.loads(original_return_bytes)
    coupon = read("tianchi_coupon_response.json")
    review = read("review_events_six_v4.json")
    timing = read("visit_timing_v4.json")
    activity = shopping["latent_shopping"]["return_activity"]
    if activity["base_model_sha256"] != hashlib.sha256(original_return_bytes).hexdigest():
        raise ValueError("population export requires the shopping policy's fitted return model")
    source_sample = _select(shopping["source"], "files sample_modulus")
    if source_sample != _select(timing["source"], "files sample_modulus"):
        raise ValueError("population export requires shopping and visit timing from the same source sample")
    if "domain" in shopping["source"]:
        if shopping["source"]["domain"] != timing["source"].get("domain"):
            raise ValueError("population export requires the same source domain and UID namespace")
        source_sample["domain"] = shopping["source"]["domain"]
    if "prepared_source_cache" in shopping["source"] and "source_contract" not in shopping:
        raise ValueError("prepared population export requires its explicit source contract")
    if "source_contract" in shopping:
        if shopping.get("summary_scope", {}).get("mode") != "train_only":
            raise ValueError("prepared population export requires train-only runtime summaries")
        if shopping["source"].get("prepared_source_cache") != timing["source"].get("prepared_source_cache"):
            raise ValueError("population export requires the same immutable prepared observations")
        training_at = utc_cutoff(shopping["source_contract"].get("training_cutoff_at", shopping["source"].get("training_cutoff_at")))
        if (utc_cutoff(shopping["summary_scope"].get("cutoff_at")) != training_at
                or utc_cutoff(shopping["source"].get("training_cutoff_at")) != training_at):
            raise ValueError("population runtime summaries differ from the source fitting cutoff")
        for fitted in (returning, timing):
            source = fitted.get("source", {})
            if (source.get("domain") != shopping["source"]["domain"]
                    or source.get("prepared_source_cache") != shopping["source"].get("prepared_source_cache")
                    or utc_cutoff(source.get("training_cutoff_at")) != training_at):
                raise ValueError("population components differ in source, UID namespace or fitting cutoff")
        if canonical_available_actions(returning.get("available_actions")) != canonical_available_actions(shopping.get("available_actions")):
            raise ValueError("population shopping and return observation supports differ")
    sample_sha256 = hashlib.sha256(json.dumps(source_sample, sort_keys=True).encode()).hexdigest()

    ref = _select(returning, "schema_version model_type cold_start_probability feature_names coef intercept "
                            "scaler_mean scaler_scale feature_min feature_max")
    for name in ("cold_start_fit", "available_actions", "feature_support"):
        if name in returning:
            ref[name] = returning[name]
    # The population run uses the reference mean, without a target intercept shift.
    ref["calibration_reference_probabilities"] = [fmean(returning["calibration_reference_probabilities"])]
    return_bytes = json.dumps(ref, separators=(",", ":"), allow_nan=False).encode()

    latent = _select(shopping["latent_shopping"],
                     "schema_version type actions discount regularization state_count feature_names "
                     "encoder action_mask transition_csr component_count mixture_weights common_reward "
                     "component_offsets component_policies component_values")
    for name in ("component_reward_deviations", "persona_labels", "linear_reward", "source_cutoff_at", "source_domain"):
        if name in shopping["latent_shopping"]:
            latent[name] = shopping["latent_shopping"][name]
    latent["profile_posteriors"] = {}
    latent["return_activity"] = _select(
        activity,
        "schema_version type horizon_days component_logit_offsets cutoff_at fit_label_available_rule posterior_available_rule",
    )
    latent["return_activity"]["base_model_sha256"] = hashlib.sha256(return_bytes).hexdigest()

    checkout = _select(shopping, "schema_version currency minor_unit reference_price_minor choice_unit checkout_episode_definition")
    checkout["source"] = {"dataset": shopping["source"]["dataset"], "sample_sha256": sample_sha256}
    for name in ("source_contract", "summary_scope", "available_actions"):
        if name in shopping:
            checkout[name] = shopping[name]
    if "domain" in shopping["source"]:
        checkout["source"]["domain"] = shopping["source"]["domain"]
    checkout["population"] = aggregate_population(shopping)
    if checkout_spending is not None:
        from .fit_checkout_spending import validate_spending_model
        spending = json.loads(checkout_spending.read_bytes())
        validate_spending_model(spending)
        if spending.get("source", {}).get("model_sha256") != hashlib.sha256(original_shopping_bytes).hexdigest():
            raise ValueError("population checkout spending must bind its exact source shopping model")
        cutoff = shopping.get("source_contract", {}).get("training_cutoff_at",
            shopping.get("source", {}).get("training_cutoff_at", shopping.get("validation", {}).get("cutoff_at_utc")))
        fit = spending.get("fit", {})
        if utc_cutoff(fit.get("cutoff_at")) != utc_cutoff(cutoff) or fit.get("seed") != shopping.get("seed"):
            raise ValueError("population checkout spending must share the source training cutoff and seed")
        checkout["population"]["checkout_spending"] = {key: value for key, value in spending.items() if key != "source"}
    checkout["latent_shopping"] = latent
    checkout["checkout_kernel"] = _select(shopping["checkout_kernel"], "schema_version type loc scale training_transitions")
    coupon_response = _select(coupon, "schema_version type discount_rate window_days inactivity_days")
    coupon_response["source"] = {"dataset": "Tianchi CCF2016 O2O", "scope": "cohort counts and day-of-redemption frequencies"}
    coupon_response["groups"] = {}
    for cohort, stats in coupon["groups"].items():
        delays = Counter(stats["redemption_delay_days"])
        coupon_response["groups"][cohort] = {**_select(stats, "receipts redeemed response_probability"),
                                             "redemption_delay_day_counts": [delays[day] for day in range(30)]}

    assets = {
        "population_coupon.json": coupon_response,
        "population_review.json": aggregate_review_model(review),
        "population_visit_timing.json": {
            **aggregate_visit_timing(timing, profiles=shopping["profiles"]), "source_sample_sha256": sample_sha256,
        },
    }
    shopping_bytes = json.dumps(checkout, separators=(",", ":"), allow_nan=False).encode()
    if reward_gmm is not None:
        from simulate.v4.irl.reward_gmm import validate_gmm
        from simulate.v4.irl.choice_gmm import TYPE as CHOICE_GMM_TYPE, validate_choice_gmm

        gmm = json.loads(reward_gmm.read_bytes())
        validator = validate_choice_gmm if gmm.get("type") == CHOICE_GMM_TYPE else validate_gmm
        validator(gmm, hashlib.sha256(original_shopping_bytes).hexdigest(),
                  hashlib.sha256(original_return_bytes).hexdigest())
        projected = _select(gmm, "schema_version type purpose parameter_names component_count mixture_weights means covariances")
        for name in ("available_actions", "fitted_parameter_names", "component_model_indices", "choice_temperatures", "dispersion_scales",
                     "shopping_dispersion_scales", "return_dispersion_scales",
                     "condition_on_profile_components", "persona_labels", "component_fitted_parameter_names",
                     "backend", "parameter_units", "shopping_dispersion_scales", "return_dispersion_scales"):
            if name in gmm:
                projected[name] = gmm[name]
        projected["base_model_sha256"] = hashlib.sha256(shopping_bytes).hexdigest()
        projected["return_model_sha256"] = latent["return_activity"]["base_model_sha256"]
        assets["population_gmm.json"] = projected

    if return_state is not None:
        from . import return_state as state_model

        dynamic = json.loads(return_state.read_bytes())
        state_model.validate_model(dynamic)
        fitted_prior = dynamic.get("fit", {}).get("static_return_prior") or {}
        if fitted_prior:
            if reward_gmm is None or fitted_prior.get("population_model_fingerprint") != hashlib.sha256(
                    json.dumps(gmm, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest():
                raise ValueError("population return state requires its exact frozen static reward GMM")
        projected_state = _select(dynamic, "schema_version type rho stationary_std quadrature_order horizon_days")
        for name in ("private_state", "transition_unit", "initial_distribution", "no_session_transition", "emission",
                     "rho_identified", "rho_grid_estimated", "fit", "limitations"):
            if name in dynamic:
                projected_state[name] = dynamic[name]
        if reward_gmm is not None:
            projected_state["runtime_gmm_fingerprint"] = hashlib.sha256(json.dumps(
                projected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assets["population_return_state.json"] = projected_state

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "models"
        staging.mkdir()
        for name, model in assets.items():
            (staging / name).write_text(json.dumps(model, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        (staging / "population_shopping.json").write_bytes(shopping_bytes)
        (staging / "population_return.json").write_bytes(return_bytes)
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="directory containing the fitted v10 models")
    parser.add_argument("--output", type=Path, required=True, help="new directory for population calibration")
    parser.add_argument("--reward-gmm", type=Path, help="fitted reward GMM to project onto the population assets")
    parser.add_argument("--return-state", type=Path, help="optional private retail return-state sidecar")
    parser.add_argument("--checkout-spending", type=Path, help="fitted persistent checkout size/price preferences")
    args = parser.parse_args()
    export_population(args.models, args.output, args.reward_gmm, return_state=args.return_state,
                      checkout_spending=args.checkout_spending)
