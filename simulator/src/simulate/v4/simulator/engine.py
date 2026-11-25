"""Source-conditioned shopping, persistent carts, and immutable KIT raw records."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import importlib
import json
import logging
import math
import numbers
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from ..configuration import DEFAULT_CONFIG, load_config
from .raw_store import RAW_COLUMNS, discount_minor, save_raw_dataset
from .outcomes import experiment_window_days, redefine_outcomes
from .lifecycle import build_observed_lifecycle
from .item_choice import item_probabilities
from .coupon_response import sample_coupon_response, validate_response_model
from .coupon_assignment import resolve_coupon_assignment, assign_coupons
from ..time_spec import WEEKLY_TIME_SPEC, weekly_calendar, weekly_metadata

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    tables: dict[str, pd.DataFrame]
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] | None = None

    def save(self, directory: str | Path) -> None:
        """Publish a complete dataset without replacing a previous run."""
        output = Path(directory)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"dataset output must be new or empty: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            save_raw_dataset(temporary / "raw.sqlite", {key: self.tables[key] for key in RAW_COLUMNS}, self.metadata or {})
            for name, frame in self.tables.items():
                frame.to_parquet(temporary / f"{name}.parquet", index=False)
            manifest = dict(self.metadata or {})
            manifest["sidecar_sha256"] = {name: hashlib.sha256((temporary / f"{name}.parquet").read_bytes()).hexdigest()
                                          for name in self.tables if name not in RAW_COLUMNS}
            (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
            os.rename(temporary, output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def _logit(probability):
    probability = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(probability / (1 - probability))


def _sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def _calibration(logits, target):
    low, high = -30.0, 30.0
    for _ in range(60):
        middle = (low + high) / 2
        if np.mean(1 - _sigmoid(logits + middle)) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


class EcommerceSimulator:
    """Generate observed records; latent profiles and life-cycle remain diagnostics."""

    def __init__(self, config: str | Path | dict | None = None):
        if config is None:
            config = DEFAULT_CONFIG
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        settings = self.config.get("source_simulation", {})
        self.model_path = Path(settings.get("model_path", "data/models/shopping_profiles.json"))
        self.settings = settings
        resolve_coupon_assignment(settings.get("coupon_assignment"))
        mode = settings.get("retention_mode", "target")
        if mode not in {"source", "target"}:
            raise ValueError("retention_mode must be source or target")
        if settings.get("signup_mode", "legacy_bounded_uniform") not in {"full_history_uniform", "legacy_bounded_uniform"}:
            raise ValueError("signup_mode must be full_history_uniform or legacy_bounded_uniform")
        history_days = settings.get("min_customer_history_days", 0)
        if isinstance(history_days, bool) or not isinstance(history_days, numbers.Integral) or history_days < 0:
            raise ValueError("min_customer_history_days must be a nonnegative integer")
        target = self.config["churn"].get("target_rate")
        if mode == "target" and not (isinstance(target, (int, float)) and not isinstance(target, bool) and .15 <= target <= .25):
            raise ValueError("source simulation target_rate must be between 0.15 and 0.25")

    @staticmethod
    def _frame(value):
        if value is None or isinstance(value, pd.DataFrame):
            return value
        value = Path(value)
        if value.suffix == ".parquet":
            return pd.read_parquet(value)
        if value.suffix == ".csv":
            return pd.read_csv(value, dtype={"customer_id": str, "product_id": str})
        raise ValueError("input must be a DataFrame, CSV or Parquet")

    def generate(self, n_customers=None, months=None, seed=None, mode="full", customers=None,
                 products=None, distribution=None, coupon_assignment=None, weeks=None) -> SimulationResult:
        from simulate.v4.irl.fit_shopping import choice_probabilities

        if mode not in {"small", "full"} or distribution not in {None, "source"}:
            raise ValueError("use small/full with the source distribution; historical toy is in legacy_engine")
        assignment_spec = resolve_coupon_assignment(self.settings.get("coupon_assignment")
            if coupon_assignment is None else coupon_assignment)
        weekly = months is None or weeks is not None
        if weeks is not None and (type(weeks) is not int or weeks != 104 or months is not None):
            raise ValueError("weekly generation requires weeks=104 and no calendar months")
        if not self.model_path.exists():
            raise FileNotFoundError(f"fit observed shopping profiles first: python -m simulate.v4.irl.fit_shopping --output {self.model_path}")
        started = time.perf_counter()
        simulator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        runtime_paths = {"engine": Path(__file__),
                         **{name: Path(__file__).with_name(f"{name}.py")
                            for name in ("lifecycle", "item_choice", "outcomes", "raw_store", "coupon_response", "coupon_assignment")},
                         **{name: Path(__file__).parents[1] / "irl" / f"{name}.py"
                            for name in ("fit_shopping", "fit_return")}}
        runtime_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in runtime_paths.items()}
        model_bytes = self.model_path.read_bytes()
        model = json.loads(model_bytes)
        population = model.get("population")
        if model.get("schema_version") != 1 or (population is None and (not model.get("profiles") or not model.get("catalog"))):
            raise ValueError("incompatible or empty shopping profiles artifact")
        causal_block = model.get("causal_shopping")
        latent_block = model.get("latent_shopping")
        if population is not None and (not latent_block or not latent_block.get("return_activity")
                                      or not self.settings.get("visit_timing_model_path")):
            raise ValueError("population synthesis requires the joint latent policy and population visit calibration")
        if population is not None and (model.get("profiles") or model.get("catalog") or latent_block.get("profile_posteriors")):
            raise ValueError("population calibration cannot contain source profiles, catalogs or individual posteriors")
        if causal_block is not None and latent_block is not None:
            raise ValueError("shopping artifact must select one learned planning policy")
        causal_policy = None
        latent_sampler = None
        if latent_block is not None:
            from simulate.v4.irl.fit_latent_shopping import policy, sample_component, validate_model
            from simulate.v4.irl.shopping_history import ShoppingHistory
            validate_model(model)
            causal_policy, latent_sampler = policy, sample_component
            for name in ("fit_latent_shopping", "shopping_history", "fit_causal_shopping"):
                path = Path(__file__).parents[1] / "irl" / f"{name}.py"
                runtime_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif causal_block is not None:
            from simulate.v4.irl.fit_causal_shopping import policy, validate_model
            validate_model(model)
            causal_policy = policy
            causal_path = Path(__file__).parents[1] / "irl" / "fit_causal_shopping.py"
            runtime_hashes["fit_causal_shopping"] = hashlib.sha256(causal_path.read_bytes()).hexdigest()
        checkout_episodes = model.get("choice_unit") == "source_checkout_episode"
        if causal_policy is not None and not checkout_episodes:
            raise ValueError("causal shopping requires checkout-decision units")
        item_kernel = model.get("item_choice")
        if item_kernel is not None and (not checkout_episodes or not isinstance(item_kernel, dict) or
                                        item_kernel.get("schema_version") != 1):
            raise ValueError("item-choice generation requires a compatible checkout-episode artifact")
        sim = self.config["simulation"]
        seed = int(sim["seed"] if seed is None else seed)
        rng = np.random.default_rng(seed)
        months = None if weekly else int(months)
        n_customers = int(n_customers if n_customers is not None else (5000 if mode == "small" else 20000))
        if n_customers < 1 or (assignment_spec["mode"] == "auto" and (n_customers < 2 or n_customers % 2)):
            raise ValueError("auto requires an even customer count >=2; none/assign require a positive customer count")
        if months is not None and months < 3:
            raise ValueError("legacy generation requires at least three calendar months")
        start = pd.Timestamp(sim["start_at"])
        if weekly:
            # Legacy config files used a naive clock. The weekly clock declares
            # that value in Seoul; explicit offsets must already match Monday.
            if start.tzinfo is None:start = start.tz_localize("Asia/Seoul")
            start,t0,end = weekly_calendar(start)
        elif start.tzinfo is None:
            # This is the explicit simulation clock, not a conversion of source timestamps.
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if not weekly:
            end = start + pd.DateOffset(months=months)
            t0 = start + pd.DateOffset(months=12 if months >= 24 else max(1, months // 2))
            local = t0.tz_convert("Asia/Seoul")
            decision = local.normalize() + pd.Timedelta(days=(-local.weekday()) % 7)
            if decision < local:
                decision += pd.Timedelta(days=7)
            t0 = decision.tz_convert("UTC")
        history_days = self.settings.get("min_customer_history_days", 0)
        if weekly:history_days = max(history_days, WEEKLY_TIME_SPEC["min_history_days"])
        if history_days > (t0 - start).total_seconds() / 86400:
            raise ValueError("min_customer_history_days exceeds the available pre-coupon observation period")
        history_cutoff = t0 - pd.Timedelta(days=int(history_days))
        eligibility_days = experiment_window_days(self.settings, "eligibility_window_days")
        coupon_validity_days = experiment_window_days(self.settings, "coupon_validity_days")
        if weekly and (eligibility_days != 28 or coupon_validity_days != 28 or self.settings.get("outcome_days",28) != 28):
            raise ValueError("weekly generation requires four-week eligibility, coupon validity and outcome")
        eligibility_start = t0 - pd.Timedelta(days=eligibility_days)
        coupon_expires_at = t0 + pd.Timedelta(days=coupon_validity_days)
        experiment_policy = f"coupon_once_highest_unit_{coupon_validity_days}d"
        coupon_policy = f"one_coupon_10pct_highest_unit_{coupon_validity_days}d"
        generation_definition = f"future_no_session_30d_active{eligibility_days}d"
        if months is not None and months >= 24:
            # Explicit legacy containers retain their raw calendar history;
            # their CLV label below still ends at exactly52weeks.
            end = max(end, t0 + pd.DateOffset(months=12))
        outcome_end = t0 + pd.Timedelta(days=30)
        if outcome_end >= end:
            raise ValueError("simulation must observe the full 30-day outcome")
        product_input = self._frame(products)
        if population is not None:
            from . import population_profiles, population_timing

            population_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
            for module in (population_profiles, population_timing):
                runtime_hashes[module.__name__.rsplit(".", 1)[-1]] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if product_input is not None:
            catalog = product_input.to_dict("records")
        elif population is not None:
            catalog = population_profiles.make_catalog(population, population_rng)
        else:
            catalog = model["catalog"]
        required = {"product_id", "category_id", "unit_price_minor"}
        if not catalog or any(not required <= set(row) for row in catalog):
            raise ValueError("product input requires product_id/category_id/unit_price_minor in the model's declared currency")
        products_by_id = {str(row["product_id"]): row for row in catalog}
        if len(products_by_id) != len(catalog):
            raise ValueError("duplicate product_id")
        catalog_ids = np.array(list(products_by_id), dtype=object)
        product_rows = [{key: (str(row[key]) if key != "unit_price_minor" and row[key] is not None else row[key])
                         for key in RAW_COLUMNS["products"]} for row in catalog]
        for row in product_rows:
            if isinstance(row["unit_price_minor"], bool) or not isinstance(row["unit_price_minor"], (int, np.integer)) or row["unit_price_minor"] <= 0:
                raise ValueError("catalog prices must be positive integer minor amounts")
        products_by_id = {row["product_id"]: row for row in product_rows}
        profiles = model.get("profiles", [])
        if population is not None:
            if any(len(group["component_probabilities"]) != len(latent_block["mixture_weights"])
                   for group in population["home_groups"].values()):
                raise ValueError("population group probabilities must match the fitted reward components")
            model["home_groups"], draw_profile = population_profiles.prepare_population(population, product_rows, population_rng)
        checkout_kernel = model.get("checkout_kernel")
        if checkout_kernel is not None:
            if not checkout_episodes:
                raise ValueError("checkout transition kernels require checkout episodes")
            from simulate.v4.irl.fit_checkout_transition import checkout_pool_weights, validate_checkout_kernel
            validate_checkout_kernel(checkout_kernel)
            kernel_path = Path(__file__).parents[1] / "irl" / "fit_checkout_transition.py"
            runtime_hashes["fit_checkout_transition"] = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
        donor_prior = model.get("donor_prior")
        donor_weights = {}
        donor_purchase_timing = None
        if donor_prior is not None and causal_policy is None:
            if not checkout_episodes:
                raise ValueError("learned donor priors require checkout-decision count units")
            from simulate.v4.irl.fit_donor_prior import donor_action_weights, validate_donor_prior
            validate_donor_prior(donor_prior)
            calibration = donor_prior.get("policy_calibration") or {}
            if calibration.get("purchase_timing") is not None:
                from simulate.v4.irl.fit_donor_prior import donor_purchase_time_multiplier
                donor_purchase_timing = donor_purchase_time_multiplier
            for profile in profiles:
                if profile["profile_id"] in donor_weights:
                    raise ValueError("donor profile identities must be unique")
                donor_weights[profile["profile_id"]] = donor_action_weights(
                    donor_prior, profile.get("decision_action_counts"), profile.get("observed_sessions"))
            prior_path = Path(__file__).parents[1] / "irl" / "fit_donor_prior.py"
            runtime_hashes["fit_donor_prior"] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        logits = _logit(np.array([profile["visit_probability_30d"] for profile in profiles]))
        return_model_path = self.settings.get("return_model_path")
        return_model = None
        return_model_bytes = None
        latent_return_offset = None
        if return_model_path:
            from simulate.v4.irl.fit_return import return_context, return_probability
            return_model_bytes = Path(return_model_path).read_bytes()
            return_model = json.loads(return_model_bytes)
        if latent_block is not None and latent_block.get("return_activity") is not None:
            from simulate.v4.irl.fit_joint_latent_shopping import return_logit_offset, validate_model
            if return_model_bytes is None:
                raise ValueError("joint latent activity requires its fitted base return model")
            validate_model(model, return_model_sha256=hashlib.sha256(return_model_bytes).hexdigest())
            latent_return_offset = return_logit_offset
            path = Path(__file__).parents[1] / "irl" / "fit_joint_latent_shopping.py"
            runtime_hashes["fit_joint_latent_shopping"] = hashlib.sha256(path.read_bytes()).hexdigest()
        gaussian_path = self.settings.get("gaussian_reward_model_path")
        choice_gmm_path = self.settings.get("choice_gmm_model_path")
        reward_gmm_path = self.settings.get("reward_gmm_model_path")
        if sum(bool(path) for path in (gaussian_path, choice_gmm_path, reward_gmm_path)) > 1:
            raise ValueError("configure one continuous reward or choice model at a time")
        gmm_path = choice_gmm_path or reward_gmm_path
        gmm_return_only = self.settings.get("reward_gmm_return_only", False)
        if choice_gmm_path and gmm_return_only:
            raise ValueError("choice-logit GMM requires its full joint choice/return path")
        if type(gmm_return_only) is not bool or (gmm_return_only and not gmm_path):
            raise ValueError("reward_gmm_return_only requires a fitted GMM")
        if gaussian_path and gmm_path:
            raise ValueError("configure one continuous reward model at a time")
        gaussian_block = gaussian_bytes = gaussian_matrices = None
        if reward_path := gmm_path or gaussian_path:
            if latent_block is None or latent_return_offset is None or return_model_bytes is None:
                raise ValueError("continuous reward synthesis requires a joint latent shopping and return model")
            from simulate.v4.irl.fit_latent_shopping import unpack_matrices
            gaussian_bytes = Path(reward_path).read_bytes()
            gaussian_block = json.loads(gaussian_bytes)
            bindings = (hashlib.sha256(model_bytes).hexdigest(), hashlib.sha256(return_model_bytes).hexdigest())
            if gmm_path:
                if choice_gmm_path:
                    from simulate.v4.irl.choice_gmm import prepare_choice_policy as prepare_bellman, sample_choice as sample_gmm_reward, validate_choice_gmm as validate_gmm
                else:
                    from simulate.v4.irl.reward_gmm import prepare_bellman, sample_reward as sample_gmm_reward, validate_gmm
                if population is not None and gaussian_block.get("profile_posteriors"):
                    raise ValueError("population GMM calibration cannot contain individual posteriors")
                factors = validate_gmm(gaussian_block, *bindings,
                                       available_actions=latent_block["encoder"].get("available_actions"))
                if gaussian_block.get("component_model_indices") is not None and gmm_return_only:
                    raise ValueError("persona-specific Gaussian rewards require the full joint sampling path")
                if gaussian_block.get("condition_on_profile_components") and (
                        gaussian_block["component_count"] != len(latent_block["mixture_weights"])):
                    raise ValueError("profile-conditioned Gaussian and actor groups must match")
                module = "choice_gmm" if choice_gmm_path else "reward_gmm"
            else:
                from simulate.v4.irl.fit_gaussian_rewards import sample_reward_policy, validate_gaussian
                validate_gaussian(gaussian_block, *bindings,
                                  available_actions=latent_block["encoder"].get("available_actions"))
                if (gaussian_block["component_count"] != len(latent_block["mixture_weights"]) or
                        gaussian_block["mixture_weights"] != latent_block["mixture_weights"]):
                    raise ValueError("Gaussian reward components differ from the frozen donor mixture")
                module = "fit_gaussian_rewards"
            if gmm_return_only:
                from simulate.v4.irl import conditional_return

                draw_return_offset = conditional_return.prepare_conditional_return(
                    gaussian_block, latent_block["component_offsets"], latent_block["return_activity"]["component_logit_offsets"],
                    available_actions=latent_block["encoder"].get("available_actions"))
                runtime_hashes["conditional_return"] = hashlib.sha256(Path(conditional_return.__file__).read_bytes()).hexdigest()
            else:
                gaussian_matrices = unpack_matrices(latent_block)
                if gmm_path:
                    gmm_policy = (prepare_bellman(latent_block) if choice_gmm_path
                                  else prepare_bellman(latent_block, gaussian_matrices))
            path = Path(__file__).parents[1] / "irl" / f"{module}.py"
            runtime_hashes[module] = hashlib.sha256(path.read_bytes()).hexdigest()
        continuous_shopping = gaussian_block is not None and not gmm_return_only
        return_state_path = self.settings.get("return_state_model_path")
        return_state_model = return_state_bytes = None
        if return_state_path:
            from simulate.v4.irl import return_state as private_state_model
            from .private_return import PrivateReturnState
            if return_model is None or model.get("source_contract", {}).get("clock_resolution", "second") != "second":
                raise ValueError("private retail return state requires a second-clock base return GLM")
            return_state_bytes = Path(return_state_path).read_bytes()
            return_state_model = json.loads(return_state_bytes)
            private_state_model.validate_model(return_state_model)
            if return_state_model.get("fit", {}).get("static_return_prior") is not None and not gmm_path:
                raise ValueError("action-only static-prior return state requires its persistent reward GMM")
            prior_fit = return_state_model.get("fit", {}).get("static_return_prior") or {}
            binding = return_state_model.get("runtime_gmm_fingerprint", prior_fit.get("population_model_fingerprint"))
            if binding is not None and binding != hashlib.sha256(json.dumps(
                    gaussian_block, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest():
                raise ValueError("private return state requires its exact frozen static reward GMM")
            for name, path in (("return_state", Path(private_state_model.__file__)),
                               ("private_return", Path(__file__).with_name("private_return.py"))):
                runtime_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        retention_mode = self.settings.get("retention_mode", "target")
        signup_mode = self.settings.get("signup_mode", "legacy_bounded_uniform")
        target = float(self.config["churn"]["target_rate"]) if retention_mode == "target" else None
        history_reference = _logit(np.asarray(return_model["calibration_reference_probabilities"])) if return_model else logits
        history_shift = _calibration(history_reference, target) if target is not None else 0.
        source_reference_activity = float(np.mean(_sigmoid(history_reference)))
        settings = self.settings
        source_observation_mode = settings.get("source_observation_mode", "synthetic_signup")
        if source_observation_mode not in {"synthetic_signup", "first_observed_entry_conditioned"}:
            raise ValueError("unknown source_observation_mode")
        max_events = int(settings.get("max_events_per_session", 64))
        if max_events < 2:
            raise ValueError("max_events_per_session must be at least two")
        fail_on_session_cap = settings.get("fail_on_session_cap", False)
        if type(fail_on_session_cap) is not bool:
            raise ValueError("fail_on_session_cap must be boolean")
        termination_mode = settings.get("session_termination_mode", "policy_end" if causal_policy is not None else "source_length")
        if termination_mode not in {"policy_end", "source_length"} or (termination_mode == "policy_end" and causal_policy is None):
            raise ValueError("session termination requires source_length or a learned policy_end actor")
        visit_timing_model = None
        visit_timing_path = settings.get("visit_timing_model_path")
        visit_timing_bytes = None
        timing_scopes = Counter()
        if visit_timing_path:
            from simulate.v4.irl.fit_visit_timing import sample_session_offsets, validate_model
            visit_timing_bytes = Path(visit_timing_path).read_bytes()
            visit_timing_model = json.loads(visit_timing_bytes)
            if population is not None:
                if visit_timing_model.get("source_sample_sha256") != model["source"]["sample_sha256"]:
                    raise ValueError("population shopping and visit timing must share the same source sample")
                visit_timing_model = population_timing.materialize_visit_timing(visit_timing_model, model["source"])
            validate_model(visit_timing_model)
            if any(visit_timing_model["source"].get(k) != model["source"].get(k) for k in ("files", "sample_modulus")):
                raise ValueError("visit timing must use the shopping model's exact native source sample")
            path = Path(__file__).parents[1] / "irl" / "fit_visit_timing.py"
            runtime_hashes["fit_visit_timing"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_observation_mode == "first_observed_entry_conditioned" and visit_timing_model is None:
            raise ValueError("conditioned first-observed entry requires its fitted source visit timing")
        message_path = Path(self.config["toy"]["message_model_path"])
        message = json.loads(message_path.read_text())
        response_path = settings.get("coupon_response_model_path")
        purchase_response = None
        response_bytes = None
        if response_path:
            response_bytes = Path(response_path).read_bytes()
            purchase_response = json.loads(response_bytes)
            validate_response_model(purchase_response)
            fitter_path = Path(__file__).parents[1] / "irl" / "fit_tianchi_coupon.py"
            runtime_hashes["fit_tianchi_coupon"] = hashlib.sha256(fitter_path.read_bytes()).hexdigest()
        coupon_path = Path(self.config["toy"]["coupon_model_path"]) if purchase_response is None else None
        coupon_model = json.loads(coupon_path.read_text()) if coupon_path else None
        event_models, event_modules, event_provenance = {}, {}, {}
        for name in ("review",):
            path = settings.get(f"{name}_model_path")
            if not path:
                continue
            content = Path(path).read_bytes()
            artifact = json.loads(content)
            if population is not None:
                artifact = population_timing.materialize_review_model(artifact)
            module = importlib.import_module(f"simulate.v4.simulator.{name}_events")
            getattr(module, f"validate_{name}_model")(artifact)
            event_models[name], event_modules[name] = artifact, module
            event_provenance[name] = {"model_path": str(path), "model_sha256": hashlib.sha256(content).hexdigest(),
                                      "type": artifact.get("type")}
            for relative, root in ((f"{name}_events", Path(__file__).parent),
                                   (f"fit_{name}_events", Path(__file__).parents[1] / "irl")):
                runtime_hashes[relative] = hashlib.sha256((root / f"{relative}.py").read_bytes()).hexdigest()
        # A documented external email proxy supplies the mean, not identified coupon CATE.
        support = np.array([message["response_lift"][key] for key in ("positive", "neutral", "negative")])
        support *= float(message["visit_lift_scale"])
        proxy_mean = float(message["observed"]["pooled"]["visit_logit_shift"])
        low, high = -100.0, 100.0
        for _ in range(60):
            coefficient = (low + high) / 2
            weights = np.exp(coefficient * support - max(coefficient * support))
            weights /= weights.sum()
            if weights @ support < proxy_mean:
                low = coefficient
            else:
                high = coefficient
        response_weights = weights
        coupon_effects = settings.get("coupon_effect_calibration")
        if coupon_effects is not None:
            from . import coupon_calibration
            from simulate.v4.irl.fit_visit_timing import first_session_share

            if purchase_response is None or visit_timing_model is None:
                raise ValueError("coupon effect calibration requires coupon redemption and visit timing models")
            coupon_calibration.validate_effect_calibration(coupon_effects, coupon_validity_days)
            runtime_hashes["coupon_calibration"] = hashlib.sha256(Path(coupon_calibration.__file__).read_bytes()).hexdigest()
            contact_until = t0 + pd.Timedelta(days=coupon_effects["contact_days"])
            price_until = t0 + pd.Timedelta(days=coupon_effects["price_days"])
            induced_purchase_weight = math.exp(np.clip(coupon_effects["induced_purchase_log_odds"], -40, 40))
            price_purchase_weight = math.exp(np.clip(coupon_effects["price_purchase_log_odds"], -40, 40))
        coupon_use = float(coupon_model["groups"]["baseline"]["coupon_use_probability"]) if coupon_model else 0.0
        send_cost = int(settings.get("coupon_send_cost_minor", 100))
        if send_cost < 0:
            raise ValueError("coupon_send_cost_minor must be nonnegative")
        discount = float(settings.get("coupon_discount", 0.10))
        if discount != 0.10:
            raise ValueError("the declared experiment fixes the coupon discount at 10%")
        discount_minor(100, discount)
        frames = {name: [] for name in RAW_COLUMNS}
        frames["products"] = product_rows
        session_diagnostic_rows, diagnostic_rows, treatment_rows, coupon_response_rows = [], [], [], []
        review_diagnostic_rows = []
        max_candidates = max(100, n_customers * (100 if retention_mode == "source" else 10))
        supplied = self._frame(customers)
        history_schedule_path = settings.get("source_history_schedule_path")
        history_bank = None
        if history_schedule_path:
            from .source_history import load_history_schedule, draw_history_schedule, history_provenance
            if supplied is not None:
                raise ValueError("source history conditioning cannot also supply customer signup clocks")
            history_bank = load_history_schedule(history_schedule_path, model,
                history_days=history_days, eligibility_days=eligibility_days)
            runtime_hashes["source_history"] = hashlib.sha256(Path(__file__).with_name("source_history.py").read_bytes()).hexdigest()
            history_profile_indices = {row["profile_id"]: index for index, row in enumerate(profiles)}
        if supplied is not None:
            if (not {"customer_id", "signed_up_at"} <= set(supplied) or supplied.customer_id.isna().any()
                    or supplied.customer_id.astype(str).duplicated().any()):
                raise ValueError("customers require unique customer_id and signed_up_at; persona is not a fixed customer field")
            n_customers = len(supplied)
            if n_customers < 1 or (assignment_spec["mode"] == "auto" and (n_customers < 2 or n_customers % 2)):
                raise ValueError("supplied customer count must be even and at least two")
            max_candidates = n_customers
        if assignment_spec["mode"] != "auto":
            assignment_ids = (supplied.customer_id.astype(str).tolist() if supplied is not None
                else [f"U{i+1:07d}" for i in range(n_customers)])
            assign_coupons(assignment_ids, assignment_spec, seed=seed)

        def sample(profile, key, fallback, positive=False):
            group = model["home_groups"][profile["home_group"]]
            values = profile.get(key) or group.get(key) or model.get("distributions", {}).get(key) or fallback
            if positive:
                values = [value for value in values if value > 0] or fallback
            return values[int(rng.integers(len(values)))]

        def offered(profile):
            weights = {str(key): value for key, value in profile.get("product_weights", {}).items() if str(key) in products_by_id}
            if not weights:
                return str(rng.choice(catalog_ids))
            keys = list(weights)
            p = np.asarray(list(weights.values()), dtype=float)
            return keys[int(rng.choice(len(keys), p=p / p.sum()))]

        available_products = set(products_by_id)
        item_choice_scopes, item_choice_fallbacks = Counter(), Counter()

        def selected_item(profile, action, previous_action, previous_product, held=None):
            probabilities, detail = item_probabilities(
                item_kernel, profile["profile_id"], profile["home_group"], action,
                previous_action, previous_product, available_products, held)
            if not probabilities:
                raise ValueError(f"no source-supported {action} product in the available catalog")
            item_choice_scopes[f"{action}:{detail['preference_scope']}"] += 1
            if detail["fallback_reason"]:
                item_choice_fallbacks[f"{action}:{detail['fallback_reason']}"] += 1
            products = list(probabilities)
            p = np.asarray(list(probabilities.values()), dtype=float)
            return products[int(rng.choice(len(products), p=p / p.sum()))]

        checkout_banks, checkout_scopes, checkout_support = {}, Counter(), {}

        def checkout_basket(profile, state):
            group = model["home_groups"][profile["home_group"]]
            target_units = None
            for scope, key, templates in (
                ("profile", profile["profile_id"], profile.get("checkout_templates", [])),
                ("home_group", profile["home_group"], group.get("checkout_templates", [])),
                ("source_pool", "all", model.get("distributions", {}).get("checkout_templates", [])),
            ):
                cache_key = (scope, str(key))
                if cache_key not in checkout_banks:
                    baskets, weights = [], []
                    for template in templates:
                        basket = {}
                        for item in template["items"]:
                            quantity = item["quantity"]
                            if isinstance(quantity, bool) or not isinstance(quantity, (int, np.integer)) or quantity < 1:
                                raise ValueError("checkout template quantities must be positive integer unit proxies")
                            pid = str(item["product_id"])
                            basket[pid] = basket.get(pid, 0) + int(quantity)
                        count = template["observed_count"]
                        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
                            raise ValueError("checkout template frequencies must be observed positive integers")
                        if basket and set(basket) <= products_by_id.keys():
                            baskets.append(basket)
                            weights.append(count)
                    probabilities = np.asarray(weights, dtype=float)
                    if len(probabilities):
                        probabilities /= probabilities.sum()
                    checkout_banks[cache_key] = (baskets, probabilities)
                    if checkout_kernel is not None:
                        units = np.asarray([sum(basket.values()) for basket in baskets], dtype=int)
                        means = np.asarray([sum(products_by_id[pid]["unit_price_minor"] * quantity
                                                for pid, quantity in basket.items()) / sum(basket.values())
                                            for basket in baskets], dtype=float)
                        checkout_support[cache_key] = (np.asarray(weights, dtype=int), units, means)
                baskets, probabilities = checkout_banks[cache_key]
                if baskets:
                    if checkout_kernel is not None:
                        frequencies, units, means = checkout_support[cache_key]
                        if scope == "profile":
                            remaining = state.setdefault("checkout_remaining", {}).setdefault(cache_key, frequencies.copy())
                            if remaining.sum():
                                index = int(rng.choice(len(baskets), p=remaining / remaining.sum()))
                                remaining[index] -= 1
                                checkout_scopes["profile_finite_observed_bank"] += 1
                                return baskets[index].copy()
                            target_units = int(units[int(rng.choice(len(baskets), p=probabilities))])
                            continue
                        if target_units is None and state.get("last_checkout_unit_mean") is not None:
                            target_units = int(units[int(rng.choice(len(baskets), p=probabilities))])
                        indices = np.flatnonzero(units == target_units) if target_units is not None else np.arange(len(baskets))
                        if not len(indices):
                            continue
                        weighted = checkout_pool_weights(checkout_kernel, probabilities[indices], means[indices],
                                                         state.get("last_checkout_unit_mean"))
                        index = int(indices[int(rng.choice(len(indices), p=weighted))])
                        checkout_scopes[f"{scope}_source_size_price_pool"] += 1
                        return baskets[index].copy()
                    checkout_scopes[scope] += 1
                    return baskets[int(rng.choice(len(baskets), p=probabilities))].copy()
            raise ValueError("catalog has no supported observed checkout template")

        def source_return_logit(local, profile_index, as_of, state):
            if return_model is None:
                return float(logits[profile_index])
            item_units = Counter()
            for item in local["order_items"]:
                item_units[item["order_id"]] += int(item["quantity"])
            quantities = {order["purchase_event_id"]: item_units[order["order_id"]] for order in local["orders"]}
            context = return_context([session["started_at"] for session in local["sessions"]], local["events"], as_of,
                                     purchase_item_units=quantities if checkout_episodes else None)
            value = float(_logit(return_probability(return_model, context)))
            if gmm_return_only:
                value += state["conditional_return_offset"]
            elif continuous_shopping:
                value += state["gaussian_reward"]["return_offset"]
            elif latent_return_offset is not None:
                value += latent_return_offset(model, state["latent_reward_component"])
            if return_state_model is not None:
                value += state["return_dynamics"].value_at(as_of)
            return value

        def auxiliary_rng(uid, kind):
            key = json.dumps(["source-event-stream-v1", seed, uid, kind]).encode()
            return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

        def flush_reviews(uid, state, local, until):
            if "review" not in event_models:
                return
            module, artifact = event_modules["review"], event_models["review"]
            while state["review_queue"] and state["review_queue"][0][0] < until:
                at, _, detail = heapq.heappop(state["review_queue"])
                if at >= end:
                    continue
                context = module.review_context(state["reviews"], at)
                rating = module.sample_rating(artifact, context, state["review_rng"])
                token = f"{uid}:E{len(local['events']) + 1}"
                row = {key: None for key in RAW_COLUMNS["events"]}
                row.update(event_id=token, customer_id=uid, event_at=at, event_type="review",
                           related_purchase_event_id=detail["purchase_event_id"], product_id=detail["product_id"],
                           order_line_no=detail["order_line_no"], rating=rating,
                           rating_kind="product_star_scenario")
                local["events"].append(row)
                state["reviews"].append({"event_at": at, "rating": rating, "product_id": detail["product_id"],
                                         "purchase_event_id": detail["purchase_event_id"]})
                detail.update(emitted=True, review_event_id=token, rating=rating, status="observed")

        def schedule_review(uid, state, purchased_at, purchase_token, order_id, basket):
            if "review" not in event_models:
                return
            module = event_modules["review"]
            # Olist counts responded orders; Amazon counts individual product reviews.
            responded_orders = list({row["purchase_event_id"]: row for row in state["reviews"]}.values())
            context = module.order_review_context(state["prior_orders"], responded_orders, purchased_at)
            selected = module.sample_review_schedule(event_models["review"], context, state["review_rng"])
            at = purchased_at + pd.Timedelta(seconds=selected["delay_seconds"]) if selected["scheduled"] else pd.NaT
            for line, product_id in enumerate(basket, 1):
                detail = {"customer_id": uid, "order_id": order_id, "purchase_event_id": purchase_token,
                          "product_id": product_id, "order_line_no": line,
                          "purchased_at": purchased_at, "scheduled_at": at, **selected,
                          "emitted": False, "review_event_id": None, "rating": None,
                          "status": "censored" if pd.notna(at) and at >= end else "scheduled" if selected["scheduled"] else "not_scheduled"}
                state["review_diagnostics"].append(detail)
                if selected["scheduled"]:
                    heapq.heappush(state["review_queue"], (at, len(state["review_diagnostics"]), detail))
            state["prior_orders"].append(purchased_at)

        def user_segment(uid, profile, signed, lower, upper, probability, state, local, treatment=False,
                         baseline_probability=None, history_starts=None):
            """Thirty-day activity with observed conditional session timing and choice."""
            flush_reviews(uid, state, local, lower)
            if history_starts is not None and (upper > t0 or treatment or baseline_probability is not None):
                raise ValueError("conditioned source history clocks are strictly pre-intervention only")
            activity_draw = rng.random()
            active = bool(history_starts) if history_starts is not None else activity_draw < probability
            induced_visit = baseline_probability is not None and activity_draw >= baseline_probability
            if lower <= signed < upper:
                active = True
            baseline_active = baseline_probability is not None and activity_draw < baseline_probability
            if not (active or baseline_active):
                flush_reviews(uid, state, local, upper)
                return
            scheduled = None
            if history_starts is not None:
                scheduled = list(history_starts)
                if not scheduled or any(not max(lower, signed) <= at < upper for at in scheduled):
                    raise ValueError("conditioned source history START lies outside its strict past window")
                count, when = len(scheduled), scheduled[0]
            elif visit_timing_model is not None:
                entry = lower <= signed < upper
                prior_active = any(lower - pd.Timedelta(days=30) <= row["started_at"] < lower for row in local["sessions"])
                timing_key = profile["home_group"] if population is not None else profile["profile_id"]
                offsets, scope = sample_session_offsets(visit_timing_model, timing_key, entry, prior_active, rng)
                timing_scopes[scope] += 1
                if entry and source_observation_mode == "first_observed_entry_conditioned":
                    if not offsets:
                        raise ValueError("conditioned first-observed entry requires a nonempty source schedule")
                    offsets = [value - min(offsets) for value in offsets]
                    anchor = signed
                else:
                    anchor = signed.normalize() if entry else lower
                scheduled = [anchor + pd.Timedelta(seconds=value) for value in offsets]
                scheduled = [at for at in scheduled if max(lower, signed) <= at < upper]
                count = len(scheduled)
                # The contact changes the activity gate only for a potential first visit during the trial horizon.
                if not count or (baseline_probability is not None and
                                 not (active if scheduled[0] < contact_until else baseline_active)):
                    flush_reviews(uid, state, local, upper)
                    return
                when = scheduled[0]
            else:
                count = int(sample(profile, "session_counts_active_30d", [1], positive=True))
                # shortcut: independent bounded timing can clip planned visits; native joint templates are used when configured.
                available = (upper - max(lower, signed)).total_seconds()
                if available <= 1:
                    return
                first_gap = float(sample(profile, "idle_gap_seconds", [1.0], positive=True))
                when = max(lower, signed) + pd.Timedelta(seconds=min(first_gap, available * rng.random()))
                if state["last_end"] is not None:
                    when = max(when, state["last_end"] + pd.Timedelta(seconds=1))
            for session_index in range(count):
                session_upper = upper
                if scheduled is not None:
                    when = scheduled[session_index]
                    if session_index + 1 < count:
                        session_upper = scheduled[session_index + 1]
                    if state["last_end"] is not None and when < state["last_end"]:
                        raise ValueError("scheduled source sessions must not overlap a generated session")
                if when >= (upper if scheduled is not None else upper - pd.Timedelta(seconds=1)):
                    break
                sid = f"{uid}:S{state['sessions'] + 1}"
                session_start = when
                state["sessions"] += 1
                if return_state_model is not None:
                    return_eta_at_start = state["return_dynamics"].begin_session(when)
                    return_eta_updates_at_start = state["return_dynamics"].updates
                previous, in_session, prior_gap = "start", 0, 0.0
                previous_run_length = 0
                shopping_history = state.get("shopping_history")
                if shopping_history is not None:
                    shopping_history.begin_session(sid, when)
                offer_seen = None
                previous_category = None
                previous_product = None
                product = None if item_kernel else offered(profile)
                if product is None:
                    price = None
                else:
                    price = products_by_id[product]["unit_price_minor"]
                def event(kind, pid=None, quantity=None, offer=None):
                    token = f"{uid}:E{len(local['events']) + 1}"
                    local["events"].append({"event_id": token, "customer_id": uid, "session_id": sid,
                        "event_at": when, "event_type": kind, "product_id": pid,
                        "quantity": quantity, "unit_price_minor": products_by_id[pid]["unit_price_minor"] if pid is not None else None,
                        "offer_product_id": offer, "offer_unit_price_minor": products_by_id[offer]["unit_price_minor"] if offer is not None else None,
                        "related_purchase_event_id": None, "coupon_id": None, "order_line_no": None, "discount_minor": None,
                        "rating": None, "rating_kind": None, "contact_id": None})
                    if kind == "coupon_usage":
                        return token
                    if state["first_event_at"] is None:
                        state["first_event_at"] = when
                    state["reference_price"] = (state["price_sum"] / state["price_count"] if state["price_count"]
                                                else model.get("reference_price_minor", profile["preferred_price_minor"]))
                    observed_product = pid if pid is not None else offer
                    if observed_product is not None and kind != "purchase":
                        state["price_sum"] += products_by_id[observed_product]["unit_price_minor"]
                        state["price_count"] += 1
                    return token
                length_key = "session_choice_lengths" if checkout_episodes else "session_lengths"
                if causal_policy is not None and termination_mode == "policy_end":
                    decision_limit, limited, termination_reason = max_events, True, "max_events"
                else:
                    observed_length = int(sample(profile, length_key, [1], positive=True))
                    limited = observed_length > max_events
                    termination_reason = "max_events" if limited else "donor_length"
                    decision_limit = min(observed_length, max_events)
                for decision in range(decision_limit):
                    flush_reviews(uid, state, local, when)
                    category = previous_category or (products_by_id[product]["category_id"] if product is not None else None)
                    context = {"previous_action": previous, "unit_price_minor": price if offer_seen is not None else None,
                        "reference_price_minor": state["reference_price"], "cart_quantity": state["cart"][product] if offer_seen is not None else 0,
                        "cart_total_quantity": sum(state["cart"].values()),
                        "days_since_first_observed": (when - state["first_event_at"]).total_seconds() / 86400 if state["first_event_at"] is not None else 0.0,
                        "days_since_purchase": (when - state["last_purchase"]).total_seconds() / 86400 if state["last_purchase"] is not None else None,
                        "purchase_count": state["purchased_units"], "product_purchase_count": state["purchased_products"][product] if offer_seen is not None else 0,
                        "purchase_in_current_session": state.get("last_purchase_session") == sid,
                        "category_purchase_share": state["purchased_categories"][category] / max(1, state["purchased_units"]),
                        "session_event_count": in_session, "session_decision_count": decision,
                        "observed_session_number": state["sessions"],
                        "prior_gap_seconds": prior_gap}
                    if shopping_history is not None:
                        context.update(shopping_history.snapshot(when))
                        context["previous_action_run_length"] = previous_run_length
                    if latent_block is not None:
                        probabilities = (causal_policy(model, context, None if gmm_path else state["latent_reward_component"],
                                         state["gaussian_reward"]["policy"]) if continuous_shopping else
                                         causal_policy(model, context, state["latent_reward_component"]))
                    elif causal_policy is not None:
                        probabilities = causal_policy(model, context)
                    elif decision == 0:
                        first_counts = profile.get("session_first_action_counts", {"view": profile["observed_sessions"]})
                        probabilities = {key: float(first_counts.get(key, 0)) for key in ("view", "cart", "remove", "purchase", "end")}
                    else:
                        probabilities = choice_probabilities(model, context)
                        baseline = model["validation"]["frequency_baseline_probabilities"]
                        counts = profile["decision_action_counts"] if checkout_episodes else profile["action_counts"]
                        total = sum(counts.values()) + profile["observed_sessions"]
                        for action in probabilities:
                            if donor_prior is not None:
                                probabilities[action] *= donor_weights[profile["profile_id"]][action]
                            else:
                                count_a = profile["observed_sessions"] if action == "end" else counts.get(action, 0)
                                probabilities[action] *= ((count_a + 0.5) / (total + 2.5)) / max(1e-6, baseline[action])
                        if donor_purchase_timing is not None:
                            probabilities["purchase"] *= donor_purchase_timing(donor_prior, context)
                    if not any(state["cart"].values()):
                        probabilities["remove"] = 0.0
                    if causal_policy is None or termination_mode == "source_length":
                        probabilities["end"] = 0.0
                    actions = list(probabilities)
                    p = np.array(list(probabilities.values()), dtype=float)
                    if not p.sum():
                        # An observed remove-only start with unknown initial cart cannot be fabricated in a complete empty-cart history.
                        limited = True
                        termination_reason = "infeasible_initial_action" if decision == 0 else "no_nonterminal_support"
                        break
                    gap_key = "choice_gap_seconds" if checkout_episodes else "event_gap_seconds"
                    next_time, gap_rng_state = None, None
                    if (coupon_effects is not None and treatment and t0 <= when < coupon_expires_at
                            and (when < price_until or (induced_visit and when < contact_until))
                            and (state["last_purchase"] is None or state["last_purchase"] < t0)):
                        # Keep the seeded action draw before the gap while gating the odds at the emitted time.
                        action_draw = rng.random()
                        gap_rng_state = rng.bit_generator.state if decision and probabilities.get("end", 0.) else None
                        gap = float(sample(profile, gap_key, [0.0])) if decision else 0.0
                        next_time = when + pd.Timedelta(seconds=gap)
                        purchase_index = actions.index("purchase")
                        if induced_visit and next_time < contact_until:
                            p[purchase_index] *= induced_purchase_weight
                        if next_time < price_until:
                            p[purchase_index] *= price_purchase_weight
                        cdf = (p / p.sum()).cumsum()
                        cdf /= cdf[-1]
                        action = actions[cdf.searchsorted(action_draw, side="right")]
                    else:
                        action = actions[int(rng.choice(len(actions), p=p / p.sum()))]
                    if action == "end":
                        if gap_rng_state is not None:
                            rng.bit_generator.state = gap_rng_state
                        limited, termination_reason = False, "causal_policy_end"
                        break
                    if next_time is None:
                        gap = float(sample(profile, gap_key, [0.0])) if decision else 0.0
                        next_time = when + pd.Timedelta(seconds=gap)
                    if next_time >= upper or (scheduled is not None and next_time > session_upper):
                        limited, termination_reason = True, "scheduled_next_start" if session_upper < upper else "window_end"
                        break
                    flush_reviews(uid, state, local, next_time)
                    when, prior_gap = next_time, gap
                    decision_offer = offer_seen
                    interaction_units = 1
                    if action == "view":
                        if item_kernel:
                            product = selected_item(profile, "view", previous, previous_product)
                        elif decision:
                            product = offered(profile)
                        price = products_by_id[product]["unit_price_minor"]
                        event("view", product, offer=decision_offer)
                        offer_seen = product
                        previous_product = product
                        previous_category = products_by_id[product]["category_id"]
                    elif action == "cart":
                        added = selected_item(profile, "cart", previous, previous_product) if item_kernel else product
                        state["cart"][added] += 1
                        event("cart", added, 1, decision_offer)
                        previous_product = added
                        previous_category = products_by_id[added]["category_id"]
                    elif action == "remove":
                        held = [pid for pid, quantity in state["cart"].items() if quantity > 0]
                        removed = (selected_item(profile, "remove", previous, previous_product, state["cart"])
                                   if item_kernel else str(rng.choice(held)))
                        state["cart"][removed] -= 1
                        event("remove_from_cart", removed, 1, decision_offer)
                        previous_product = removed
                        previous_category = products_by_id[removed]["category_id"]
                    elif action == "purchase":
                        if checkout_episodes:
                            basket = checkout_basket(profile, state)
                            if population is not None:
                                basket = population_profiles.align_checkout_basket(basket, profile, products_by_id, rng)
                            held_units = sum(min(state["cart"][pid], quantity) for pid, quantity in basket.items())
                            interaction_units = sum(basket.values())
                            source = "cart" if held_units == interaction_units else "mixed" if held_units else "buy_now"
                        else:
                            basket = {pid: quantity for pid, quantity in state["cart"].items() if quantity > 0}
                            source = "cart" if basket else "buy_now"
                            basket = basket or {product: 1}
                        token = event("purchase", offer=decision_offer)
                        order_id = f"{uid}:O{state['purchases'] + 1}"
                        gross = sum(quantity * products_by_id[pid]["unit_price_minor"] for pid, quantity in basket.items())
                        response_record = state.get("coupon_response")
                        use_coupon = (treatment and not state["coupon_used"]
                                      and (response_record["response_selected"] and response_record["response_ready_at"] <= when
                                           if purchase_response is not None else state["will_use_coupon"])
                                      and t0 <= when < coupon_expires_at)
                        target_line, target_product = max(enumerate(basket, 1),
                            key=lambda item: (products_by_id[item[1]]["unit_price_minor"], -item[0]))
                        reduction = discount_minor(products_by_id[target_product]["unit_price_minor"], discount) if use_coupon else 0
                        local["orders"].append({"order_id": order_id, "purchase_event_id": token, "checkout_source": source,
                            "coupon_id": f"{uid}:C1" if use_coupon else None,
                            "gross_minor": gross, "discount_minor": reduction, "paid_minor": gross - reduction})
                        for line, (pid, quantity) in enumerate(basket.items(), 1):
                            local["order_items"].append({"order_id": order_id, "line_no": line, "product_id": pid,
                                "quantity": quantity, "unit_price_minor": products_by_id[pid]["unit_price_minor"]})
                            state["purchased_products"][pid] += quantity
                            state["purchased_categories"][products_by_id[pid]["category_id"]] += quantity
                            state["purchased_units"] += quantity
                            # Source uses the last item interaction; aggregate checkout retains that item-level proxy.
                            previous_category = products_by_id[pid]["category_id"]
                            previous_product = pid
                            unit = products_by_id[pid]["unit_price_minor"]
                            prior_count = state["price_count"] + quantity - 1
                            state["reference_price"] = ((state["price_sum"] + (quantity - 1) * unit) / prior_count
                                                        if prior_count else model.get("reference_price_minor", profile["preferred_price_minor"]))
                            state["price_sum"] += quantity * unit
                            state["price_count"] += quantity
                            if source in {"cart", "mixed"}:
                                state["cart"][pid] -= min(state["cart"][pid], quantity)
                        if use_coupon:
                            event("coupon_usage", target_product, 1)
                            local["events"][-1].update(related_purchase_event_id=token, coupon_id=f"{uid}:C1",
                                                       order_line_no=target_line, discount_minor=reduction)
                        if checkout_episodes and interaction_units > 1:
                            prior_gap = 0.0
                        state["purchases"] += 1
                        state["last_purchase"] = when
                        state["last_purchase_session"] = sid
                        if checkout_kernel is not None:
                            state["last_checkout_unit_mean"] = gross / interaction_units
                        state["coupon_used"] |= use_coupon
                        schedule_review(uid, state, when, token, order_id, basket)
                        if use_coupon and purchase_response is not None:
                            response_record.update(opportunity_at=when, baseline_action="purchase",
                                                   coupon_used=True, purchase_event_id=token)
                    if shopping_history is not None:
                        shopping_history.record_action(sid, when, action, quantity=interaction_units,
                            view_price_minor=price if action == "view" else None)
                    previous_run_length = previous_run_length + 1 if action == previous else 1
                    previous = action
                    in_session += interaction_units
                if shopping_history is not None:
                    shopping_history.end_session(sid, when, completed=termination_reason in {"causal_policy_end", "donor_length"})
                local["sessions"].append({"session_id": sid, "customer_id": uid, "started_at": session_start, "ended_at": when})
                session_diagnostic_rows.append({"customer_id": uid, "session_id": sid, "as_of_at": when,
                    "session_started_at": session_start, "cart_quantity_after": sum(state["cart"].values()),
                    "truncated_session": limited, "termination_reason": termination_reason})
                if fail_on_session_cap and termination_reason == "max_events":
                    raise ValueError(f"Main session cap reached ({max_events}); customer={uid}, session={sid}; candidate is unscorable")
                if return_state_model is not None:
                    dynamics = state["return_dynamics"]
                    dynamics.complete_session(when, termination_reason, bool(limited))
                    session_diagnostic_rows[-1].update(return_eta_at_session_start=float(return_eta_at_start),
                        return_eta_known_updates_at_start=return_eta_updates_at_start,
                        return_eta_pending_completions=len(dynamics.pending))
                state["last_end"] = when
                if scheduled is None:
                    idle = float(sample(profile, "idle_gap_seconds", [1.0], positive=True))
                    when += pd.Timedelta(seconds=max(1.0, idle))
            flush_reviews(uid, state, local, upper)

        donors = []
        candidates = 0
        while len(donors) < n_customers and candidates < max_candidates:
            history_starts = None
            if history_bank is not None:
                profile_id, history_starts = draw_history_schedule(history_bank, rng, t0, earliest=start)
                index = history_profile_indices[profile_id]
                profile = profiles[index]
            elif population is not None:
                index, profile = None, draw_profile(candidates)
            else:
                index = int(rng.integers(len(profiles)))
                profile = profiles[index]
            uid = str(supplied.iloc[candidates].customer_id) if supplied is not None else f"U{len(donors) + 1:07d}"
            if history_starts is not None:
                signed = history_starts[0]
            elif supplied is not None:
                signed = pd.Timestamp(supplied.iloc[candidates].signed_up_at)
                if signed.tzinfo is None:
                    raise ValueError("signed_up_at must include timezone")
                signed = signed.tz_convert("UTC")
            else:
                span = (max(1, int((t0 - start).days)) if signup_mode == "full_history_uniform"
                        else max(1, min(150, int((t0 - start).days) - 31)))
                signed = start + pd.Timedelta(days=int(rng.integers(span)))
            if not start <= signed < t0:
                raise ValueError("signup must precede the experimental snapshot within observation")
            state = {"sessions": 0, "purchases": 0, "purchased_units": 0, "cart": Counter(),
                     "purchased_products": Counter(), "purchased_categories": Counter(), "last_purchase": None,
                     "last_end": None, "coupon_used": False,
                     "price_sum": 0, "price_count": 0, "reference_price": None, "first_event_at": None,
                     "will_use_coupon": rng.random() < coupon_use}
            if latent_sampler is not None:
                if gmm_path and continuous_shopping:
                    component, scope, state["gaussian_reward"] = sample_gmm_reward(
                        gaussian_block, factors, gmm_policy, profile["profile_id"], rng,
                        **({"profile_components": profile.get("component_probabilities",
                            latent_block.get("profile_posteriors", {}).get(profile["profile_id"], latent_block["mixture_weights"]))}
                           if gaussian_block.get("condition_on_profile_components") else {}))
                else:
                    if population is not None:
                        probabilities = profile["component_probabilities"]
                        component = int(rng.choice(len(probabilities), p=probabilities))
                        scope = "synthetic_population_group"
                    else:
                        component, scope = latent_sampler(model, profile["profile_id"], rng)
                    if gmm_return_only:
                        state["conditional_return_offset"] = draw_return_offset(
                            component, auxiliary_rng(f"{uid}:{candidates}", "conditional_return"))
                    elif continuous_shopping:
                        state["gaussian_reward"] = sample_reward_policy(model, gaussian_block, component, rng,
                                                                        matrices=gaussian_matrices)
                state.update(latent_reward_component=component, latent_assignment_scope=scope,
                             shopping_history=ShoppingHistory(start))
            state.update(review_rng=auxiliary_rng(uid, "review"),
                         reviews=[], prior_orders=[], review_queue=[], review_diagnostics=[])
            if return_state_model is not None:
                state["return_dynamics"] = PrivateReturnState(
                    return_state_model, auxiliary_rng(f"{uid}:{candidates}", "return_state"))
            local = {key: [] for key in ("sessions", "events", "orders", "order_items")}
            old_diagnostics_size = len(session_diagnostic_rows)
            if history_starts is not None:
                user_segment(uid, profile, signed, signed, t0, 1., state, local, history_starts=history_starts)
            else:
                window = t0
                windows = []
                while window > signed:
                    windows.append((max(window - pd.Timedelta(days=30), signed), window))
                    window -= pd.Timedelta(days=30)
                for lower, upper in reversed(windows):
                    probability = float(_sigmoid(source_return_logit(local, index, lower, state) + history_shift))
                    user_segment(uid, profile, signed, lower, upper, probability, state, local)
            eligible = any(eligibility_start <= row["started_at"] < t0 for row in local["sessions"])
            if history_days:
                eligible = eligible and state["first_event_at"] is not None and state["first_event_at"] <= history_cutoff
            candidates += 1
            if not eligible:
                del session_diagnostic_rows[old_diagnostics_size:]
                if supplied is not None:
                    raise ValueError("supplied customers must be eligible for the declared active-customer experiment and meet min_customer_history_days")
                continue
            donors.append((uid, index, profile, signed, state, local))
            if len(donors) % 1000 == 0:
                LOGGER.info("Eligible history: %s/%s customers (%s candidates)", len(donors), n_customers, candidates)
        if len(donors) != n_customers:
            raise ValueError("unable to populate the eligible active-customer panel; check min_customer_history_days and observation period")
        snapshot_logits = np.array([source_return_logit(local, index, t0, state) for _, index, _, _, state, local in donors])
        if return_state_model is not None:
            for _, _, _, _, state, _ in donors:
                state["return_eta_at_snapshot"] = state["return_dynamics"].value_at(t0)
        outcome_shift = _calibration(snapshot_logits, target) if target is not None else 0.
        if coupon_effects is not None:
            contact_keys = [profile["home_group"] if population is not None else profile["profile_id"]
                            for _, _, profile, _, _, _ in donors]
            contact_shares = {key: first_session_share(visit_timing_model, key, True, coupon_effects["contact_days"])
                              for key in sorted(set(contact_keys))}
            support = coupon_calibration.calibrate_contact_support(
                snapshot_logits + outcome_shift, np.array([contact_shares[key] for key in contact_keys]),
                support, response_weights, coupon_effects["contact_visit_lift"])
        legacy_assignments = None
        if assignment_spec["mode"] == "auto":
            legacy_assignments = np.array([1] * (n_customers // 2) + [0] * (n_customers - n_customers // 2))
            rng.shuffle(legacy_assignments)
        assignments, assignment_probabilities, assignment_metadata = assign_coupons(
            [row[0] for row in donors], assignment_spec, seed=seed, legacy=legacy_assignments)
        for treatment, propensity, (uid, index, profile, signed, state, local), snapshot_logit in zip(
                assignments, assignment_probabilities, donors, snapshot_logits):
            if coupon_effects is not None:
                rng = auxiliary_rng(uid, "coupon-calibrated-outcome")
            if purchase_response is not None:
                if coupon_effects is None:
                    record = sample_coupon_response(purchase_response, seed, uid, t0, state["last_purchase"])
                else:
                    record = sample_coupon_response(purchase_response, seed, uid, t0, state["last_purchase"],
                                                    odds_multiplier=coupon_effects["redemption_odds_multiplier"])
                record.update(treatment=int(treatment), coupon_id=f"{uid}:C1" if treatment else None)
                state["coupon_response"] = record
                coupon_response_rows.append(record)
            response = int(rng.choice(3, p=response_weights))
            shift = float(support[response])
            coupon_id = f"{uid}:C1" if treatment else None
            if treatment:
                frames["coupons"].append({"coupon_id": coupon_id, "customer_id": uid, "sent_at": t0,
                    "expires_at": coupon_expires_at, "discount_rate": discount, "policy_id": coupon_policy,
                    "send_cost_minor": send_cost})
            treatment_rows.append({"customer_id": uid, "as_of_at": t0, "assigned_at": t0,
                "treatment": int(treatment), "assignment_probability": float(propensity), "experiment_id": experiment_policy,
                "coupon_id": coupon_id, "actual_sent": bool(treatment), "sent": bool(treatment),
                "send_cost_minor": send_cost if treatment else 0,
                "intervention_spec": f"one 10% coupon at t0 on the highest-unit-price purchased unit, valid {coupon_validity_days}d, no later sending"})
            window = t0
            while window < end:
                upper = min(window + pd.Timedelta(days=30), end)
                baseline_logit = snapshot_logit if window == t0 else source_return_logit(local, index, window, state)
                probability = float(_sigmoid(baseline_logit + outcome_shift + (shift if treatment and window == t0 else 0)))
                baseline_probability = (float(_sigmoid(baseline_logit + outcome_shift))
                                        if coupon_effects is not None and treatment and window == t0 else None)
                user_segment(uid, profile, signed, window, upper, probability, state, local, bool(treatment),
                             baseline_probability=baseline_probability)
                window = upper
            flush_reviews(uid, state, local, end)
            frames["customers"].append({"customer_id": uid, "signed_up_at": signed})
            for key in local:
                frames[key].extend(local[key])
            review_diagnostic_rows.extend(state["review_diagnostics"])
            response_type = response if coupon_effects is None else (0 if shift > 0 else 2 if shift < 0 else 1)
            baseline_return = float(_sigmoid(snapshot_logit + outcome_shift))
            coupon_return = float(_sigmoid(snapshot_logit + outcome_shift + shift))
            if coupon_effects is not None:
                key = profile["home_group"] if population is not None else profile["profile_id"]
                coupon_return = baseline_return + (coupon_return - baseline_return) * contact_shares[key]
            diagnostic_rows.append({"customer_id": uid, "source_profile_id": profile["profile_id"],
                "home_persona": profile["home_group"],
                "response_type": ("P" if response_type == 0 else "SD" if response_type == 2 else "ST" if snapshot_logit > np.median(snapshot_logits) else "LC"),
                "no_coupon_return_probability_30d": baseline_return,
                "coupon_return_probability_30d": coupon_return,
                "intervention_logit_shift": shift})
            if return_state_model is not None:
                dynamics = state["return_dynamics"]
                diagnostic_rows[-1].update(return_eta_at_observation_end=float(dynamics.value_at(end)),
                    return_eta_at_snapshot=float(state["return_eta_at_snapshot"]),
                    return_eta_known_completed_sessions=dynamics.updates,
                    return_eta_right_censored_completions=len(dynamics.pending))
            if latent_sampler is not None:
                diagnostic_rows[-1].update(latent_reward_component=state["latent_reward_component"],
                                          latent_assignment_scope=state["latent_assignment_scope"])
            if gmm_return_only:
                diagnostic_rows[-1]["conditional_return_logit_offset"] = state["conditional_return_offset"]
            if continuous_shopping:
                draw = state.pop("gaussian_reward")
                offset_prefix = "gaussian_choice_logit_offset" if choice_gmm_path else "gaussian_reward_offset"
                diagnostic_rows[-1].update({f"{offset_prefix}_{action}": float(value)
                                           for action, value in zip(("view", "cart", "remove", "purchase"), draw["action_offsets"])})
                diagnostic_rows[-1]["gaussian_return_logit_offset"] = float(draw["return_offset"])
            if len(diagnostic_rows) % 1000 == 0:
                LOGGER.info("Completed outcomes: %s/%s customers", len(diagnostic_rows), n_customers)
        frames["events"].sort(key=lambda row: row["event_at"])
        identifiers = {row["event_id"]: index for index, row in enumerate(frames["events"], 1)}
        for row in frames["events"]:
            row["event_id"] = identifiers[row["event_id"]]
            if row.get("related_purchase_event_id") is not None:
                row["related_purchase_event_id"] = identifiers[row["related_purchase_event_id"]]
        for row in frames["orders"]:
            row["purchase_event_id"] = identifiers[row["purchase_event_id"]]
        for row in coupon_response_rows:
            if row["purchase_event_id"] is not None:
                row["purchase_event_id"] = identifiers[row["purchase_event_id"]]
            row["response_status"] = ("control" if not row["treatment"] else "redeemed" if row["coupon_used"]
                                      else "no_eligible_opportunity" if row["response_selected"] else "no_response")
        for row in review_diagnostic_rows:
            row["purchase_event_id"] = identifiers[row["purchase_event_id"]]
            if row["review_event_id"] is not None:
                row["review_event_id"] = identifiers[row["review_event_id"]]
        tables = {name: pd.DataFrame(rows, columns=RAW_COLUMNS[name]) for name, rows in frames.items()}
        for frame in tables.values():
            for column in ("signed_up_at", "started_at", "ended_at", "event_at", "sent_at", "expires_at"):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], utc=True)
        purchase_lookup = tables["events"].set_index("event_id")[["customer_id", "event_at"]]
        order_details = tables["orders"].join(purchase_lookup, on="purchase_event_id")
        future_sessions = tables["sessions"].loc[(tables["sessions"].started_at >= t0) & (tables["sessions"].started_at < outcome_end)]
        active = set(future_sessions.customer_id)
        revenue30 = order_details.loc[(order_details.event_at >= t0) & (order_details.event_at < outcome_end)].groupby("customer_id").paid_minor.sum()
        clv_end = t0 + pd.Timedelta(days=WEEKLY_TIME_SPEC["clv_label_days"])
        revenue12 = order_details.loc[(order_details.event_at >= t0) & (order_details.event_at < clv_end)].groupby("customer_id").paid_minor.sum()
        labels = []
        for row in treatment_rows:
            uid = row["customer_id"]
            labels.append({"customer_id": uid, "as_of_at": t0, "snapshot_at": t0, "snapshot_id": f"{uid}:{t0.isoformat()}",
                "outcome_end_at": outcome_end, "observation_end_at": end, "churn": int(uid not in active), "retained": int(uid in active),
                "future_retained": int(uid in active), "future_revenue_30d": int(revenue30.get(uid, 0)),
                "future_revenue": int(revenue30.get(uid, 0)),
                "future_revenue_12m": int(revenue12.get(uid, 0)) if end >= clv_end and not row["treatment"] else None,
                "clv_outcome_end_at": clv_end, "definition_id": generation_definition,
                "policy_id": coupon_policy if row["treatment"] else "no_new_coupon",
                "is_censored": False, "observation_status": "complete"})
        tables.update(labels=pd.DataFrame(labels), treatments=pd.DataFrame(treatment_rows),
                      diagnostics=pd.DataFrame(diagnostic_rows), session_diagnostics=pd.DataFrame(session_diagnostic_rows))
        if purchase_response is not None:
            tables["coupon_response_diagnostics"] = pd.DataFrame(coupon_response_rows)
        if "review" in event_models:
            tables["review_diagnostics"] = pd.DataFrame(review_diagnostic_rows, columns=[
                "customer_id", "order_id", "purchase_event_id", "product_id", "order_line_no", "purchased_at", "scheduled_at",
                "scheduled", "delay_seconds", "source_order_score", "response_probability", "source_cohort",
                "emitted", "review_event_id", "rating", "status"])
        observed = float(tables["labels"].churn.mean())
        unit_tag = "checkout-items" if item_kernel else "checkout" if checkout_episodes else "unit"
        if purchase_response is not None:
            unit_tag += "-tianchi"
        if causal_policy is not None and termination_mode == "source_length":
            unit_tag += "-source-length"
        if visit_timing_model is not None:
            unit_tag += "-joint-timing"
        if population is not None:
            unit_tag += "-population"
        if gmm_return_only:
            unit_tag += "-gmm-return"
        elif choice_gmm_path:
            unit_tag += "-choice-gmm"
        elif gmm_path:
            unit_tag += "-gmm"
        elif continuous_shopping:
            unit_tag += "-gaussian"
        version = 13 if choice_gmm_path else 12 if gmm_path and continuous_shopping else 11 if continuous_shopping else 10 if latent_return_offset is not None else 9 if latent_block is not None else 8 if causal_policy is not None else 7 if "review" in event_models else 6 if donor_prior is not None or checkout_kernel is not None or retention_mode == "source" else 5
        metadata = {"schema_version": 7, "raw_schema_version": 5 if "review" in event_models else 4,
            "dataset_id": f"cosmetics-source-v{version}-six-event-{unit_tag}-{mode}-{'104w' if weekly else str(months)+'m'}-seed{seed}",
            "seed": seed, "months": months, "clv_label_days": WEEKLY_TIME_SPEC["clv_label_days"],
            "currency": model["currency"], "amount_unit": model["minor_unit"],
            "distribution": ("source_conditioned_with_explicit_retention_calibration" if target is not None
                             else "source_conditioned_without_retention_intercept_shift"),
            "retention_mode": retention_mode, "source_reference_activity_probability_30d": source_reference_activity,
            "signup_clock": "supplied" if supplied is not None else signup_mode,
            "configured_target_churn_rate": self.config["churn"].get("target_rate"),
            "start_at": start.isoformat(), "end_at": end.isoformat(),
            "observation_start_at": start.isoformat(), "observation_end_at": end.isoformat(), "as_of_at": t0.isoformat(),
            "definition_id": generation_definition, "policy_id": experiment_policy,
            "eligibility_window_days": eligibility_days, "coupon_validity_days": coupon_validity_days,
            "min_customer_history_days": int(history_days), "coupon_assignment": assignment_metadata,
            "randomized_assignment": assignment_metadata["randomized"],
            "assignment_design": assignment_metadata["design"],
            "target_churn_rate": target, "expected_control_churn_rate": float(np.mean(1 - _sigmoid(snapshot_logits + outcome_shift))),
            "observed_churn_rate": observed, "history_logit_shift": history_shift, "outcome_logit_shift": outcome_shift,
            "candidate_customers": candidates, "eligible_customers": n_customers,
            "eligibility": f"session in [t0-{eligibility_days}d,t0), selection uses history only",
            "model_path": str(self.model_path), "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "return_model_path": return_model_path,
            "return_model_sha256": hashlib.sha256(return_model_bytes).hexdigest() if return_model_bytes else None,
            "return_model_type": "source_point_in_time_return_maxent" if return_model else "observed_profile_activity_baseline",
            "feature_set_version": "kit32-v1", "simulator_version": f"source-shopping-v{version}",
            "donor_personalization": ({"type": "persistent_latent_reward_policy", "posthoc_donor_multipliers": False}
                if latent_block is not None else {"type": "causal_reward_policy", "posthoc_donor_multipliers": False}
                if causal_policy is not None else {"type": donor_prior["type"], "strength": donor_prior["strength"],
                "base_probabilities": donor_prior["base_probabilities"], "alpha": donor_prior["alpha"],
                "policy_calibration": donor_prior.get("policy_calibration"),
                "count_units": donor_prior.get("count_units"),
                "meaning": "source-fitted posterior/prior multiplier; full-history donor counts are latent scenario traits"}
                if donor_prior is not None else {"type": "legacy_uniform_half_pseudocount_divided_by_global_frequency"}),
            "causal_shopping": ({"type": causal_block["type"], "discount": causal_block["discount"],
                "regularization": causal_block["regularization"], "state_count": len(causal_block["encoder"]["state_keys"]),
                "reward_basis_type": causal_block.get("reward_basis_type", "continuous"),
                "support_rule": causal_block.get("support_rule", "matching-sku"),
                "fit_target": causal_block.get("fit_target", "customer-macro"),
                "reward_features": causal_block.get("reward_feature_names", ["intercept", *causal_block["encoder"]["feature_names"]]),
                "termination": ("native source stopping-length; local non-END policy, not an exact fixed-length bridge"
                                if termination_mode == "source_length" else "learned END; max_events is a safety cap"),
                "scope": "source-observed within-session shopping MDP; auxiliary-event purchase effects are not jointly identified"}
                if causal_block is not None else None),
            "latent_shopping": ({"type": latent_block["type"], "discount": latent_block["discount"],
                "regularization": latent_block["regularization"], "state_count": latent_block["state_count"],
                "mixture_weights": latent_block["mixture_weights"], "feature_names": latent_block["feature_names"],
                "assignment": ("discrete shopping component with a GMM return draw conditioned on its action offsets"
                               if gmm_return_only else
                               "frozen representative IRL policy; individual choice-logit and return tastes use choice_gmm"
                               if choice_gmm_path else
                               "frozen parent dynamics; active reward mixture is specified by reward_gmm"
                               if gmm_path else
                               "one persistent component drawn from the synthetic profile group's aggregate mixture"
                               if population is not None else
                               "one persistent private component per synthetic customer; full observed source donor trait reconstruction, not heldout forecasting"
                               if latent_block.get("profile_posterior_assignment_scope") == "source_observation_trait_posterior"
                               else "one persistent private component per customer; train-prefix profile posterior or source population prior"),
                "posterior_bank_scope": latent_block.get("posterior_bank_scope"),
                "trait_reconstruction": latent_block.get("trait_reconstruction"),
                "observed_history": "shopping_observed_history_v1; completed last-action clock +30min strictly before decision",
                "termination": ("native source stopping-length; local non-END policy, not an exact fixed-length bridge"
                                if termination_mode == "source_length" else "learned END; max_events is a safety cap"),
                "visibility": "component and assignment scope occur only in private diagnostics, never business tables or predictor features",
                "scope": "source-observed shopping behavior with latent reward heterogeneity; no identified psychological or auxiliary-event causal effect"}
                if latent_block is not None else None),
            "latent_return_activity": (latent_block["return_activity"] if latent_return_offset is not None else None),
            "visit_timing": ({"model_path": str(visit_timing_path), "model_sha256": hashlib.sha256(visit_timing_bytes).hexdigest(),
                "type": visit_timing_model["type"], "scope_counts_all_attempts": dict(timing_scopes),
                "clock": ("synthetic 30d schedules generated from population histograms; partial windows clipped without resampling"
                          if population is not None else "native joint 30d start offsets; entry first-observed-day UTC anchor; regular windows KST midnight; partial windows clipped without resampling"),
                "collision": "ties retained; later scheduled start right-censors earlier shopping choices; no time compression"}
                if visit_timing_model is not None else None),
            "shopping_session_termination": {"mode": termination_mode,
                "decision_cap": max_events,
                "meaning": ("native observed stopping-length draw; policy conditioned on non-END actions; capped/window-clipped sessions are incomplete"
                            if termination_mode == "source_length" else "learned END; observed donor length is not sampled")},
            "event_scope_version": "shopping_coupon_review_v1",
            "event_types": ["view", "cart", "remove_from_cart", "purchase", "coupon_usage", "review"],
            "shopping_event_types": ["view", "cart", "remove_from_cart", "purchase"],
            "activity_clock": "shopping sessions only; asynchronous reviews do not reset nonvisit churn or lifecycle",
            "event_models": event_provenance,
            "review_generation": {"clock": "asynchronous, purchased-product-linked, no shopping session", "rating_kind": "product_star_scenario",
                "uniqueness": "one review per purchase_event_id/product_id; purchased quantity does not multiply reviews",
                "schedule_unit": ("one response and binned delay draw per order; responded order expands to its purchased products"
                                  if population is not None else "one Olist response/delay draw per order; responded order expands to its purchased products"),
                "assumption": "Olist survey scheduling transported to all purchased products and Amazon conditional stars are separate components; per-product response propensity and their joint distribution are not observed"},
            "coupon_application": {
                "scope": "one purchased unit", "selection": "highest unit_price_minor; ties choose lowest line_no",
                "discount_basis": "selected unit price, not basket gross or full line quantity",
                "source": "explicit user-requested simulator policy; not inferred from Tianchi or Cosmetics",
                "usage_event": "one coupon_usage per redeemed coupon, same time/session/customer as its related purchase",
                "shopping_context": "usage is an accounting event; excluded from MNL/return behavior counts, price reference and cart updates",
            },
            "simulator_code_sha256": simulator_sha,
            "runtime_code_sha256": runtime_hashes,
            "configuration_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
            "marketing_effect_source": "Hillstrom randomized 14d email mean transferred to the 30d visit window; latent signs are MaxEnt decomposition, not identified individual effects",
            "marketing_proxy_sha256": hashlib.sha256(message_path.read_bytes()).hexdigest(),
            "coupon_takeup_source": "dunnhumby targeted descriptive baseline; cross-domain transfer assumption",
            "coupon_model_sha256": hashlib.sha256(response_bytes if response_bytes else coupon_path.read_bytes()).hexdigest(),
            "coupon_purchase_response": ({
                "model_path": str(response_path), "model_sha256": hashlib.sha256(response_bytes).hexdigest(),
                "source": purchase_response.get("source"),
                "type": "observational_response_scenario", "window_days": 30,
                "cohort_clock": "purchase history strictly before sending; no observed purchase is not never purchased",
                "mechanism": "one keyed response draw per coupon, applied only at an eligible purchase sampled from the shopping policy",
                "requires_policy_purchase": True,
                "probability_scope": "source receipt rates and response timing transferred to already sampled eligible purchases",
                "replaces": "dunnhumby will_use_coupon gate; no multiplication of response rates",
                "causal_effect_identified": False, "full_purchase_counterfactual_replay": False,
                "response_type_scope": "ST/P/SD/LC remain Hillstrom visit-proxy diagnostics, not purchase effect types",
            } if purchase_response is not None else None),
            "shopping_choice_unit": model.get("choice_unit", "source_item_interaction"),
            "checkout_template_scope_counts": dict(checkout_scopes),
            "checkout_generation": ({"type": checkout_kernel["type"], "loc": checkout_kernel["loc"],
                "scale": checkout_kernel["scale"], "training_transitions": checkout_kernel["training_transitions"],
                "profile_bank": ("synthetic checkout templates" if population is not None
                                 else "observed template frequencies without replacement; source pool after exhaustion"),
                "pool": ("generated group baskets conditioned on calibrated size and fitted log unit-mean price transitions"
                         if population is not None else "source templates conditioned on observed size and fitted log unit-mean price transitions"),
                "meaning": "finite evidence bank, not product stock or a lifetime personal assortment"}
                if checkout_kernel is not None else {"type": "iid_observed_donor_templates"}),
            "sampling_diagnostic_scope": "all generation attempts, including rejected enrollment histories; not factual event counts",
            "item_choice_scope_counts": dict(item_choice_scopes),
            "item_choice_fallback_counts": dict(item_choice_fallbacks),
            "item_choice_type": ("synthetic customer SKU preferences" if population is not None else
                                 item_kernel.get("type") if item_kernel else "fixed donor all-action top16 product frequencies"),
            "purchase_proxy": model.get("checkout_episode_definition") if checkout_episodes else "whole cart or one direct item; source has no verified orders",
            "assumptions": ["native Cosmetics currency not documented; no KRW conversion",
                ("prices are fixed by the supplied product catalog" if product_input is not None
                 else "synthetic SKU prices are sampled from aggregate price bins" if population is not None
                 else "fixed product price is positive observed source mode"),
                ("initial cart empty; generated checkout baskets consume matching held units and preserve the rest"
                 if population is not None else
                 "initial cart empty; observed checkout templates consume matching held units and preserve the rest; rows are unit proxies, not verified source quantities"
                 if checkout_episodes else "initial cart empty; additions one unit; cart purchases checkout the whole basket"), "first observed event is not an actual source signup",
                "signup is supplied or structural uniform sampling, not verified source signup; lifecycle ENTRY/HOME/EXIT uses observed sessions and elapsed 28-day inactivity only",
                ("30d activity windows and retention calibration alter absolute source visit rates" if target is not None
                 else "fitted return probabilities have no target intercept shift; active-window scheduling and enrollment remain structural"),
                ("native source stopping length with a locally non-END learned policy; no donor action multipliers; not an exact fixed-length bridge"
                 if causal_policy is not None and termination_mode == "source_length" else
                 "first actions and END use the source-fitted causal reward policy; no posthoc donor weighting or forced donor length"
                 if causal_policy is not None else "first actions and session termination sample observed donor frequencies/lengths; MNL models subsequent nonterminal choices"),
                "the offer context is a last-view proxy; remove-only starts with unknown initial cart end without fabricated actions",
                ("active windows generate fresh timing patterns from fitted population summaries"
                 if population is not None else
                 "active windows use source joint count/time templates; five-month patterns extrapolate to longer runs"
                 if visit_timing_model is not None else "active-window timing is bounded observed idle gaps; five-month patterns extrapolate to longer runs"),
                ("experimental contact and price contrasts are transferred scenarios; direct effects end at the respective study horizons"
                 if coupon_effects is not None else "email-to-coupon, 14d-to-30d and individual response decomposition are scenarios"),
                "send cost is declared policy cost, not a measured source cost"],
            "table_rows": {name: len(frame) for name, frame in tables.items()}}
        if history_days:
            metadata["eligibility"] += f"; first observed shopping event at or before t0-{history_days}d"
        if source_observation_mode == "first_observed_entry_conditioned":
            metadata["source_observation_mode"] = source_observation_mode
            metadata["assumptions"].append("customer origin conditions its first observed entry; source entry offsets are relative to their first START, not verified account signup; later returns use the fitted activity law")
        if history_bank is not None:
            metadata["source_history_conditioning"] = history_provenance(history_schedule_path, history_bank)
            metadata["assumptions"].append("past session START clocks are bootstrapped from eligible outer-training source histories; past choices are generated by IRL; future activity is unconditioned on outcomes")
        if population is not None:
            metadata["profile_generation"] = "synthetic_population"
        if coupon_effects is not None:
            metadata["coupon_effect_calibration"] = {**coupon_effects, "visit_logit_shifts": support.tolist()}
            metadata["marketing_effect_source"] = "Hillstrom randomized contact risk difference, transferred to the configured contact horizon"
            metadata["coupon_purchase_response"].update(
                mechanism="calibrated shopping purchase odds, with separate Tianchi redemption draws on generated purchases",
                probability_scope="Tianchi purchase-history odds and delay shape, scaled to trial redemption among buyers",
                response_type_scope="visit shifts remain a latent scenario; no identified purchase effect types",
            )
        if purchase_response is not None:
            metadata["coupon_takeup_source"] = (
                "Tianchi timing and recency odds scaled to a trial redemption share among purchasers"
                if coupon_effects is not None else
                "Tianchi O2O observed 30d redemption by purchase history; scenario transfer, not causal uplift"
            )
            metadata["coupon_model_sha256"] = hashlib.sha256(response_bytes).hexdigest()
        if gmm_path:
            metadata["choice_gmm" if choice_gmm_path else "reward_gmm"] = {
                "model_path": str(gmm_path), "model_sha256": hashlib.sha256(gaussian_bytes).hexdigest(),
                "type": gaussian_block["type"], "means": gaussian_block["means"], "covariances": gaussian_block["covariances"],
                "parameter_names": gaussian_block["parameter_names"], "mixture_weights": gaussian_block["mixture_weights"],
                "assignment": ("one persistent GMM return draw conditional on the discrete shopping reward"
                               if gmm_return_only else
                               "one draw per customer using learned GMM source-donor posterior or population prior"
                               if gaussian_block.get("profile_posteriors") else
                               "one draw per customer using learned GMM population prior"),
                "visibility": "individual offsets occur only in private diagnostics"}
            if choice_gmm_path:
                metadata["choice_gmm"].update(parameter_units=gaussian_block["parameter_units"],
                    choice_temperatures=gaussian_block.get("choice_temperatures", [1.] * gaussian_block["component_count"]),
                    dispersion_scales=gaussian_block.get("dispersion_scales", [1.] * gaussian_block["component_count"]),
                    choice_rule="softmax((log representative IRL policy + personal choice-logit taste) / temperature)",
                    individual_bellman_replanning=False)
        elif gaussian_block is not None:
            metadata["gaussian_reward"] = {
                "model_path": str(gaussian_path), "model_sha256": hashlib.sha256(gaussian_bytes).hexdigest(),
                "type": gaussian_block["type"], "means": gaussian_block["means"], "variances": gaussian_block["variances"],
                "parameter_names": gaussian_block["parameter_names"],
                "assignment": "one persistent Gaussian reward draw per customer conditional on frozen discrete donor component gating",
                "scope": "two-stage source-MAP Gaussian synthesis extension; parent discrete forecast scores do not validate the Gaussian model",
                "forecast_validation_uses_gaussian": False,
                "visibility": "individual offsets occur only in private diagnostics, never business tables or predictor features"}
        if return_state_model is not None:
            metadata["private_return_state"] = {
                "model_path": str(return_state_path), "model_sha256": hashlib.sha256(return_state_bytes).hexdigest(),
                "type": return_state_model["type"], "rho": return_state_model["rho"],
                "stationary_std": return_state_model["stationary_std"],
                "initialization": "stationary draw at first observed START; independent private RNG stream",
                "update": "natural untruncated END +30min strictly before current clock; no EXIT/reentry reset",
                "emission": "global GLM + persistent static return offset + private eta, each added once",
                "visibility": "private diagnostics only; no public predictor added"}
        if weekly:metadata.update(weekly_metadata(), weeks=104,
            clock_resolution="weekly clock in Asia/Seoul; naive legacy start_at is interpreted in Asia/Seoul")
        identity = {key: metadata.get(key) for key in ("configuration_sha256", "model_sha256", "return_model_sha256",
            "coupon_model_sha256", "marketing_proxy_sha256", "runtime_code_sha256", "event_models", "visit_timing",
            "seed", "months", "start_at", "end_at")}
        if gmm_path:
            key = "choice_gmm" if choice_gmm_path else "reward_gmm"
            identity[key + "_model_sha256"] = metadata[key]["model_sha256"]
        elif gaussian_block is not None:
            identity["gaussian_reward_model_sha256"] = metadata["gaussian_reward"]["model_sha256"]
        if return_state_model is not None:
            identity["return_state_model_sha256"] = metadata["private_return_state"]["model_sha256"]
        if history_bank is not None:
            identity["source_history_schedule_sha256"] = metadata["source_history_conditioning"]["sha256"]
        identity["n_customers"] = n_customers
        identity["coupon_assignment"] = assignment_metadata
        identity["time_spec"] = metadata.get("time_spec")
        identity["products_input_sha256"] = hashlib.sha256(pd.util.hash_pandas_object(
            pd.DataFrame(product_rows), index=False).to_numpy().tobytes()).hexdigest()
        identity["supplied_customers_sha256"] = (hashlib.sha256(pd.util.hash_pandas_object(
            supplied[["customer_id", "signed_up_at"]], index=False).to_numpy().tobytes()).hexdigest() if supplied is not None else None)
        # Attempt counts are outcomes of the seeded run, not inputs to its identity.
        if identity["visit_timing"] is not None:
            identity["visit_timing"] = {k: v for k, v in identity["visit_timing"].items() if k != "scope_counts_all_attempts"}
        metadata["generation_fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        metadata["dataset_id"] += f"-n{n_customers}-{metadata['generation_fingerprint'][:16]}"
        personas = tables["diagnostics"].set_index("customer_id").home_persona.to_dict()
        tables["lifecycle"] = build_observed_lifecycle(tables["sessions"], metadata, personas)
        final_states = tables["lifecycle"].groupby("customer_id", sort=False).life_cycle.last()
        tables["diagnostics"]["life_cycle_end"] = tables["diagnostics"].customer_id.map(final_states)
        metadata["lifecycle_contract"] = tables["lifecycle"].attrs
        metadata["session_diagnostic_clock"] = "completed session ended_at; cart stock is not known at session start"
        metadata["table_rows"] = {name: len(frame) for name, frame in tables.items()}
        tables, metadata = redefine_outcomes(
            tables, metadata, outcome_days=self.settings.get("outcome_days", 28), shopping_model=model,
            eligibility_days=eligibility_days,
            **({"visit_timing_model": visit_timing_model} if visit_timing_model is not None else {}))
        return SimulationResult(tables, time.perf_counter() - started, metadata)
