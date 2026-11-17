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


def _select(model, keys):
    return {key: model[key] for key in keys.split()}


def export_population(models: Path, output: Path, reward_gmm: Path | None = None) -> None:
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
    sample_sha256 = hashlib.sha256(json.dumps(source_sample, sort_keys=True).encode()).hexdigest()

    ref = _select(returning, "schema_version model_type cold_start_probability feature_names coef intercept "
                            "scaler_mean scaler_scale feature_min feature_max")
    # The population run uses the reference mean, without a target intercept shift.
    ref["calibration_reference_probabilities"] = [fmean(returning["calibration_reference_probabilities"])]
    return_bytes = json.dumps(ref, separators=(",", ":"), allow_nan=False).encode()

    latent = _select(shopping["latent_shopping"],
                     "schema_version type actions discount regularization state_count feature_names "
                     "encoder action_mask transition_csr component_count mixture_weights common_reward "
                     "component_offsets component_policies component_values")
    latent["profile_posteriors"] = {}
    latent["return_activity"] = _select(
        activity,
        "schema_version type horizon_days component_logit_offsets cutoff_at fit_label_available_rule posterior_available_rule",
    )
    latent["return_activity"]["base_model_sha256"] = hashlib.sha256(return_bytes).hexdigest()

    checkout = _select(shopping, "schema_version currency minor_unit reference_price_minor choice_unit checkout_episode_definition")
    checkout["source"] = {"dataset": shopping["source"]["dataset"], "sample_sha256": sample_sha256}
    checkout["population"] = aggregate_population(shopping)
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

        gmm = json.loads(reward_gmm.read_bytes())
        validate_gmm(gmm, hashlib.sha256(original_shopping_bytes).hexdigest(),
                     hashlib.sha256(original_return_bytes).hexdigest())
        projected = _select(gmm, "schema_version type purpose parameter_names component_count mixture_weights means covariances")
        projected["base_model_sha256"] = hashlib.sha256(shopping_bytes).hexdigest()
        projected["return_model_sha256"] = latent["return_activity"]["base_model_sha256"]
        assets["population_gmm.json"] = projected

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
    args = parser.parse_args()
    export_population(args.models, args.output, args.reward_gmm)
