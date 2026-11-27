"""Stateful Main retail runtime on the explicit 104-week research clock.

Source shopping/return/timing assets remain domain-specific and frozen. Coupon
callbacks are scenario assumptions, separate from the no-coupon source process.
The raw checkout/FIFO implementation follows v6 weekly_campaign; no legacy
persona/cross-domain adapter or policy backend is loaded.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import heapq
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from ..configuration import DEFAULT_CONFIG, load_config
from ..irl import fit_latent_shopping as shopping
from ..irl import reward_gmm
from ..irl.fit_joint_latent_shopping import return_logit_offset, validate_model as validate_joint
from ..irl.fit_return import return_context, return_probability
from ..irl.fit_visit_timing import sample_session_offsets, validate_model as validate_timing
from ..irl.shopping_history import ShoppingHistory
from ..irl.source_contract import canonical_available_actions
from ..time_spec import WEEKLY_TIME_SPEC, weekly_calendar, weekly_metadata
from . import review_events
from .item_choice import item_probabilities
from .raw_store import RAW_COLUMNS, discount_minor
from .private_return import PrivateReturnState


RUNTIME_LIFECYCLE_COLUMNS = ("customer_id", "as_of_at", "phase", "reason", "observed_session_number")
DIAGNOSTIC_COLUMNS = ("customer_id", "session_id", "as_of_at", "session_started_at",
                      "cart_quantity_after", "truncated_session", "termination_reason")


def _logit(value):
    value = min(1 - 1e-5, max(1e-5, float(value)))
    return math.log(value / (1 - value))


def _sigmoid(value):
    return 1 / (1 + math.exp(-max(-40., min(40., value))))


class MainCampaignEnv:
    """A persistent source-profile customer with raw39 observations.

    Defaults are 52-week warmup and 52-week decisions. Shorter explicit windows
    are useful for tests. Initial eligibility uses the generated past only;
    synthetic signup/warmup is not a reconstruction of the donor's real history.
    ``copy.copy`` and ``copy.deepcopy`` isolate all runtime state, including RNGs.

    ``response_sampler(env, sent_at)`` must return response_selected (bool) and
    response_ready_at (aware timestamp, or None when not selected). Without it,
    sending a coupon is rejected. ``runtime_reward_offsets(env, decision_at)``
    returns purchase_reward_shift (reward GMM) or purchase_choice_logit_shift
    (choice GMM), plus return_logit_shift; default both zero.
    A pending action is fixed at its original decision time. Receipt readiness
    may affect fulfillment/discounts when that already chosen purchase is emitted.
    """

    @staticmethod
    def customer_id_for_seed(seed):
        return f"main-weekly-{seed}"

    def __init__(self, config=None, *, assets_root=None, cost_model="S",
                 warmup_days=364, horizon_days=364, min_history_days=91,
                 coupon_valid_days=28, feature_builder=None, profile_ids=None,
                 response_sampler=None, runtime_reward_offsets=None, component_conditioning=None,
                 score_conversion_value=1000., score_coupon_cost=100.):
        if cost_model not in {"S", "D"}:
            raise ValueError("cost_model must be S or D")
        for name, value, positive in (("score_conversion_value",score_conversion_value,True),
                                     ("score_coupon_cost",score_coupon_cost,False)):
            if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Real)
                    or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
                raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} real value")
        self.score_conversion_value,self.score_coupon_cost = float(score_conversion_value),float(score_coupon_cost)
        for name, value, minimum in (("warmup_days", warmup_days, 0),
                ("horizon_days", horizon_days, 1), ("min_history_days", min_history_days, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if min_history_days > warmup_days or coupon_valid_days != 28 or type(coupon_valid_days) is not int:
            raise ValueError("history must fit warmup; weekly coupons require 28 days")
        self.config = copy.deepcopy(load_config(config or DEFAULT_CONFIG, assets_root=assets_root)
            if config is None or isinstance(config, (str, Path)) else config)
        self.settings = self.config["source_simulation"]
        if self.settings.get("backend") not in (None, "main", "main_campaign"):
            raise ValueError("MainCampaignEnv does not load legacy/daily policy backends")
        self.origin_clock = weekly_calendar(self.config["simulation"]["start_at"])[0]
        self.cost_model, self.warmup_days, self.horizon_days = cost_model, warmup_days, horizon_days
        self.min_history_days, self.coupon_valid_days = min_history_days, coupon_valid_days
        self.feature_builder = feature_builder
        self.response_sampler, self.runtime_reward_offsets = response_sampler, runtime_reward_offsets
        for callback in (feature_builder, response_sampler, runtime_reward_offsets):
            if callback is not None and not callable(callback):
                raise ValueError("runtime hooks must be callable")
        self.model_shas = {}

        def read(key, required=True):
            path = self.settings.get(key)
            if not path:
                if required:
                    raise ValueError(f"Main campaigns require {key}")
                return None
            content = Path(path).read_bytes()
            self.model_shas[key] = hashlib.sha256(content).hexdigest()
            return json.loads(content)

        self.model = read("model_path")
        self.return_model = read("return_model_path")
        self.timing_model = read("visit_timing_model_path")
        self.review_model = read("review_model_path", False)
        self.return_state_model = read("return_state_model_path", False)
        if self.return_state_model is not None:
            from ..irl.return_state import validate_model
            validate_model(self.return_state_model)
        self.latent = self.model.get("latent_shopping")
        self.population = self.model.get("population")
        if not self.latent or (self.population is None and (not self.model.get("profiles") or not self.model.get("catalog"))):
            raise ValueError("Main campaigns require source profiles/catalog or calibrated population assets")
        if self.population is not None:
            from . import population_profiles, population_timing
            if self.model.get("profiles") or self.model.get("catalog") or self.latent.get("profile_posteriors"):
                raise ValueError("population assets cannot contain source profiles/catalog or individual posteriors")
            if self.timing_model.get("source_sample_sha256") != self.model["source"]["sample_sha256"]:
                raise ValueError("population shopping and timing must share the exact source sample")
            self.timing_model = population_timing.materialize_visit_timing(self.timing_model, self.model["source"])
            if self.review_model is not None:
                self.review_model = population_timing.materialize_review_model(self.review_model)
            if any(len(group["component_probabilities"]) != len(self.latent["mixture_weights"])
                   for group in self.population["home_groups"].values()):
                raise ValueError("population group probabilities must match the fitted IRL actors")
            if profile_ids is not None:
                raise ValueError("source profile IDs cannot select aggregate population customers")
        self.component_conditioning = None
        if component_conditioning is not None:
            weights=np.asarray(component_conditioning,float)
            if (self.population is None or weights.shape!=(len(self.latent['mixture_weights']),)
                    or not np.isfinite(weights).all() or (weights<0).any() or (weights>1).any() or not weights.any()):
                raise ValueError('private calibration conditioning requires population actor likelihoods in[0,1]')
            self.component_conditioning=weights.copy()
        self.source_observation_mode = self.settings.get("source_observation_mode", "synthetic_signup")
        if self.source_observation_mode not in {"synthetic_signup", "first_observed_entry_conditioned"}:
            raise ValueError("unknown source_observation_mode")
        domains = {v for v in (self.model.get("source_contract", {}).get("domain"),
            self.latent.get("source_domain"), self.model.get("source", {}).get("domain"),
            self.settings.get("domain")) if v is not None}
        if len(domains) != 1 or not domains <= {"cosmetics", "electronics"}:
            raise ValueError("retail assets must declare one matching cosmetics/electronics domain")
        self.domain = next(iter(domains))
        validate_joint(self.model, return_model_sha256=self.model_shas["return_model_path"])
        validate_timing(self.timing_model)
        timing_domain = self.timing_model.get("source_domain", self.timing_model.get("source", {}).get("domain", "cosmetics"))
        if timing_domain != self.domain:
            raise ValueError("visit-timing domain differs from its shopping source")
        contract = self.model.get("source_contract", {})
        if contract.get("prepared_sha256"):
            returned = self.return_model.get("source_contract", {})
            for name in ("domain", "prepared_sha256", "clock_resolution", "training_cutoff_at"):
                if contract.get(name) != returned.get(name):
                    raise ValueError(f"return source binding differs at {name}")
            if self.timing_model.get("provenance", {}).get("source_cache_sha256") != contract["prepared_sha256"]:
                raise ValueError("visit timing must bind the exact prepared source")
            if pd.Timestamp(self.timing_model.get("cutoff_at")) != pd.Timestamp(contract["training_cutoff_at"]):
                raise ValueError("visit-timing training cutoff differs from its actor")
        self.activity_days = self.return_model.get("horizon_days", 30)
        if (self.activity_days != 30 or self.timing_model.get("horizon_days", 30) != 30
                or self.latent.get("return_activity", {}).get("horizon_days", 30) != 30):
            raise ValueError("Main requires the fitted native30d return/timing heads")
        if self.review_model is not None:
            review_events.validate_review_model(self.review_model)
        available = canonical_available_actions(self.latent["encoder"].get("available_actions"))
        self.observable_actions = set(available)
        allowed = None if profile_ids is None else set(profile_ids)
        self.profiles = tuple(p for p in self.model.get("profiles", []) if allowed is None or p["profile_id"] in allowed)
        if self.population is None and (not self.profiles or len({p["profile_id"] for p in self.profiles}) != len(self.profiles)):
            raise ValueError("source profiles must be nonempty and unique")
        self.profile_probabilities = np.full(len(self.profiles), 1. / len(self.profiles)) if self.profiles else np.empty(0)
        if self.population is not None:
            self.population_catalog_seed = self.settings.get("population_catalog_seed", self.config["simulation"].get("seed", 42))
            if type(self.population_catalog_seed) is not int or self.population_catalog_seed < 0:
                raise ValueError("population_catalog_seed must be a nonnegative integer")
            catalog_rng = np.random.default_rng(np.random.SeedSequence(self.population_catalog_seed, spawn_key=(1,)))
            catalog = population_profiles.make_catalog(self.population, catalog_rng)
        else:
            catalog = self.model["catalog"]
        self.home_groups = self.model.get("home_groups", {})
        self.product_rows = tuple({key: str(row[key]) if key != "unit_price_minor" and row[key] is not None else row[key]
            for key in RAW_COLUMNS["products"]} for row in catalog)
        self.products = {str(row["product_id"]): row for row in self.product_rows}
        if len(self.products) != len(self.product_rows) or any(
                isinstance(row["unit_price_minor"], bool) or not isinstance(row["unit_price_minor"], int)
                or row["unit_price_minor"] <= 0 for row in self.product_rows):
            raise ValueError("source catalog requires unique products and positive integer minor prices")
        self.population_catalog_index = None
        if self.population is not None:
            from .population_profiles import compile_population_catalog
            self.population_catalog_index = compile_population_catalog(self.product_rows)
        self.history_bank = None
        if self.settings.get('source_history_schedule_path'):
            from .source_history import load_history_schedule
            read('source_history_schedule_path')
            self.history_bank=load_history_schedule(self.settings['source_history_schedule_path'],self.model,
                history_days=self.min_history_days,eligibility_days=28)
            self._history_profiles={p['profile_id']:p for p in self.profiles}
            if any(row['profile_id'] not in self._history_profiles for row in self.history_bank['records']):
                raise ValueError('selected source profiles do not contain every conditioned history donor')
        self.product_prices = {key: int(row["unit_price_minor"]) for key, row in self.products.items()}
        self.available_products = set(self.products)
        self.max_events = self.settings.get("max_events_per_session", 782)
        if type(self.max_events) is not int or self.max_events < 2:
            raise ValueError("max_events_per_session must be an integer >=2")
        if self.settings.get("session_termination_mode", "policy_end") != "policy_end":
            raise ValueError("Main campaigns require natural policy END")
        self.send_cost = self.settings.get("coupon_send_cost_minor", 100)
        if type(self.send_cost) is not int or self.send_cost < 0 or self.settings.get("coupon_discount", .1) != .1:
            raise ValueError("Main campaigns fix 10% coupons and nonnegative integer sending costs")
        continuous = [key for key in ("gaussian_reward_model_path", "reward_gmm_model_path", "choice_gmm_model_path")
                      if self.settings.get(key)]
        if len(continuous) > 1:
            raise ValueError("configure one continuous reward/choice sidecar at a time")
        if self.settings.get("gaussian_reward_model_path"):
            raise ValueError("use a joint reward/choice GMM sidecar for Main campaigns")
        self.fixed_irl_critic = self.settings.get("choice_model") == "fixed_irl_critic"
        if self.settings.get("choice_model") not in (None, "fixed_irl_critic"):
            raise ValueError("unknown Main choice_model")
        if self.fixed_irl_critic and self.settings.get("choice_gmm_model_path"):
            raise ValueError("fixed IRL critic uses the original discrete policy, without a choice GMM")
        self.choice_space = bool(self.settings.get("choice_gmm_model_path")) or self.fixed_irl_critic
        self.purchase_shift_name = "purchase_choice_logit_shift" if self.choice_space else "purchase_reward_shift"
        self.gmm = read("choice_gmm_model_path" if self.settings.get("choice_gmm_model_path") else "reward_gmm_model_path", False)
        if self.return_state_model is not None:
            prior = self.return_state_model.get("fit", {}).get("static_return_prior")
            if prior is not None and self.gmm is None:
                raise ValueError("action-only static-prior return state requires its persistent reward GMM")
            binding = self.return_state_model.get("runtime_gmm_fingerprint", (prior or {}).get("population_model_fingerprint"))
            if binding is not None and binding != hashlib.sha256(json.dumps(
                    self.gmm, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest():
                raise ValueError("private return state requires its exact frozen static reward GMM")
        self.gmm_factors = self.solve_policy = self.draw_return_offset = None
        self._sample_continuous = reward_gmm.sample_reward
        self.return_only = self.settings.get("reward_gmm_return_only", False)
        if type(self.return_only) is not bool or self.return_only and (self.gmm is None or self.settings.get("choice_gmm_model_path")):
            raise ValueError("reward_gmm_return_only requires a reward GMM, not a choice-logit GMM")
        if self.gmm is not None:
            if self.settings.get("choice_gmm_model_path"):
                from ..irl import choice_gmm
                self.gmm_factors = choice_gmm.validate_choice_gmm(self.gmm, self.model_shas["model_path"],
                    self.model_shas["return_model_path"], available_actions=available)
                self.solve_policy = choice_gmm.prepare_choice_policy(self.latent)
                self._sample_continuous = choice_gmm.sample_choice
            else:
                self.gmm_factors = reward_gmm.validate_gmm(self.gmm, self.model_shas["model_path"],
                    self.model_shas["return_model_path"], available_actions=available)
            indices = self.gmm.get("component_model_indices", [])
            if any(index >= len(self.latent["mixture_weights"]) for index in indices):
                raise ValueError("GMM actor bindings exceed the source actor support")
            if self.return_only:
                if indices:
                    raise ValueError("persona GMM requires full joint reward sampling")
                from ..irl.conditional_return import prepare_conditional_return
                self.draw_return_offset = prepare_conditional_return(self.gmm,
                    self.latent["component_offsets"], self.latent["return_activity"]["component_logit_offsets"],
                    available_actions=available)
        if self.fixed_irl_critic and self.gmm is not None and not self.return_only:
            raise ValueError("fixed IRL critic requires the old reward GMM to be return-only")
        if not self.choice_space and (self.gmm is not None and not self.return_only or runtime_reward_offsets is not None):
            self.solve_policy = reward_gmm.prepare_bellman(self.latent, shopping.unpack_matrices(self.latent))
        self._basket_banks = {}
        self.assumptions = [
            "Source-specific catalog, prices, checkout and observational Main models; no cross-domain transport.",
            "Synthetic signup and warmup, not reconstruction of a source customer's missing past.",
            "Initial active-past28d and history eligibility uses only the predecision generated prefix.",
            "Native30d gate uniform and complete potential-offset bank persist across weekly boundaries.",
            "Each customer's reward5 and lifetime session phase persist through warmup, inactivity and decisions.",
            "END is immediate; nonterminal choice is frozen before latency and later coupon arrivals.",
            "Coupon callbacks, if supplied, are explicit scenarios rather than identified causal effects.",
            "Single-use ready receipts redeem FIFO; 10% discount applies to one highest-priced unit.",
            "S/D scores are separate from recorded actual payments, discounts and sending costs."]

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        # Compiled function closures capture read-only fitted assets. Everything
        # mutable, including active-row aliases, is copied together with one memo.
        for name in ("model", "latent", "return_model", "timing_model", "review_model", "return_state_model",
                     "gmm", "gmm_factors", "profiles", "profile", "history_bank", "_history_profiles", "product_rows", "products", "product_prices", "available_products",
                     "population_catalog_index"):
            if name in self.__dict__:
                asset = self.__dict__[name]
                memo[id(asset)] = asset
        result.__dict__ = copy.deepcopy(self.__dict__, memo)
        return result

    @property
    def tables(self):
        result = {name: pd.DataFrame(self.records[name], columns=columns) for name, columns in RAW_COLUMNS.items()}
        result["runtime_lifecycle"] = pd.DataFrame(self.records["runtime_lifecycle"], columns=RUNTIME_LIFECYCLE_COLUMNS)
        result["session_diagnostics"] = pd.DataFrame(self.records["session_diagnostics"], columns=DIAGNOSTIC_COLUMNS)
        for frame in result.values():
            for column in ("signed_up_at", "started_at", "ended_at", "event_at", "sent_at", "expires_at", "as_of_at", "session_started_at"):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], utc=True)
        result["_metadata"] = copy.deepcopy(self.records["_metadata"])
        if self.now > self.origin:
            from .lifecycle import build_observed_lifecycle
            observed = {**result["_metadata"], "observation_end_at": self.now.isoformat()}
            result["lifecycle"] = build_observed_lifecycle(result["sessions"], observed)
        return result

    def _features(self):
        if self.feature_builder is not None:
            values = np.asarray(self.feature_builder(self), dtype=float)
        else:
            from .online_features import observed_snapshot
            values = observed_snapshot(self.records, self.now, product_prices=self.product_prices)
        if values.shape != (39,) or np.isinf(values).any():
            raise ValueError("Main observations require raw39; unobserved values may be NaN")
        return values

    def reset(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("episode seed must be a nonnegative integer")
        for attempt in range(getattr(self, "INITIAL_CANDIDATES", 1000)):
            self._reset_candidate(seed, attempt)
            lower = self.decision_start - pd.Timedelta(days=28)
            first = self.state["first_event_at"]
            mature = not self.min_history_days or first is not None and first <= self.decision_start - pd.Timedelta(days=self.min_history_days)
            if mature and any(lower <= row["started_at"] < self.decision_start for row in self.records["sessions"]):
                self.records["_metadata"].update(initial_candidate_attempts=attempt + 1,
                    initial_eligibility=f"past28d START and observed history >= {self.min_history_days}d; future outcomes excluded")
                return self._features()
        raise ValueError("no active-past28d initial customer found in the bounded past-only candidate bank")

    def _reset_candidate(self, seed, attempt):
        self.seed = seed
        streams = np.random.SeedSequence([seed, attempt]).spawn(4)
        self.rng, self.visit_rng, self.review_rng, self.response_rng = [np.random.default_rng(s) for s in streams]
        history_starts=None
        if self.history_bank is not None:
            from .source_history import draw_history_schedule
            profile_id,history_starts=draw_history_schedule(self.history_bank,self.rng,
                self.origin_clock+pd.Timedelta(days=self.warmup_days),earliest=self.origin_clock)
            self.profile=self._history_profiles[profile_id]
        elif self.population is not None:
            from .population_profiles import prepare_population
            self.profile_rng = self._population_profile_rng(seed, attempt)
            self.home_groups, draw_profile = prepare_population(self.population, self.product_rows, self.profile_rng,
                catalog_index=self.population_catalog_index)
            self.profile = draw_profile(attempt)
            if self.component_conditioning is not None:
                bound=max(float(np.asarray(g['component_probabilities'])@self.component_conditioning)
                    for g in self.population['home_groups'].values())
                if bound<=0:raise ValueError('requested response role has no original population support')
                for _ in range(100000):
                    original=np.asarray(self.profile['component_probabilities'],float)
                    mass=float(original@self.component_conditioning)
                    if mass>0 and self.profile_rng.random()<mass/bound:
                        self.profile={**self.profile,'component_probabilities':(original*self.component_conditioning/mass).tolist()}
                        break
                    self.profile=draw_profile(attempt)
                else:raise RuntimeError('private role conditioning exceeded its bounded prior rejection bank')
            self._basket_banks = {}
        else:
            self.profile = self.profiles[int(self.rng.integers(len(self.profiles)))]
        self.source_profile_id = self.profile["profile_id"]
        self.uid = self.customer_id_for_seed(seed)
        self._candidate_attempt = attempt
        self.gmm_component = None
        self.temperature = 1.
        if self.gmm is not None and not self.return_only:
            kwargs = {}
            if self.gmm.get("condition_on_profile_components"):
                kwargs["profile_components"] = self.profile.get("component_probabilities",
                    self.latent["profile_posteriors"].get(self.source_profile_id, self.latent["mixture_weights"]))
            self.gmm_component, self.component_scope, reward = self._sample_continuous(
                self.gmm, self.gmm_factors, self.solve_policy, self.source_profile_id, self.rng, **kwargs)
            self.component = self.gmm.get("component_model_indices", [None] * self.gmm["component_count"])[self.gmm_component]
            self.temperature = self.gmm.get("choice_temperatures", [1.] * self.gmm["component_count"])[self.gmm_component]
            self.theta = np.r_[reward["action_offsets"], reward["return_offset"]]
            self._policy = reward["policy"]
        else:
            if self.population is not None:
                probabilities = self.profile["component_probabilities"]
                self.component = int(self.rng.choice(len(probabilities), p=probabilities))
                self.component_scope = "synthetic_population_group"
            else:
                self.component, self.component_scope = shopping.sample_component(self.model, self.source_profile_id, self.rng)
            offset = self.draw_return_offset(self.component, self._auxiliary_rng("conditional_return")) if self.return_only else return_logit_offset(self.model, self.component)
            self.theta = np.r_[np.zeros(4) if self.fixed_irl_critic else self.latent["component_offsets"][self.component], offset]
            self._policy = np.asarray(self.latent["component_policies"][self.component], dtype=float)
        self._shifted_policies = {0.: self._policy}
        self.origin = self.now = self.origin_clock if history_starts is None else history_starts[0]
        self.decision_start = self.origin_clock + pd.Timedelta(days=self.warmup_days)
        self.episode_end = self.decision_start + pd.Timedelta(days=self.horizon_days)
        self.records = {name: [] for name in RAW_COLUMNS}
        self.records["products"] = [dict(row) for row in self.product_rows]
        self.records["customers"] = [{"customer_id": self.uid, "signed_up_at": self.origin}]
        self.records["session_diagnostics"] = []
        self.records["runtime_lifecycle"] = []
        self.records["_metadata"] = {"raw_schema_version": 5 if self.review_model else 4,
            "dataset_id": f"main-weekly-{self.domain}-{seed}", "seed": seed,
            "currency": self.model["currency"], "amount_unit": self.model["minor_unit"],
            "observation_start_at": self.origin.isoformat(), "observation_end_at": self.episode_end.isoformat(),
            "as_of_at": self.decision_start.isoformat(), "domain": self.domain, "cost_model": self.cost_model,
            "score_conversion_value": self.score_conversion_value, "score_coupon_cost": self.score_coupon_cost,
            "score_objective_units": "normalized simulated value; independent of raw payment, discount and sending amounts",
            "coupon_validity_days": self.coupon_valid_days, "model_sha256": dict(self.model_shas),
            "warmup_days": self.warmup_days, "decision_days": self.horizon_days,
            "assumptions": list(self.assumptions), **weekly_metadata(),
            "runtime_lifecycle_contract": "private generator NEW(first two STARTs)/HOME; natural END of purchase session PURCHASE_EXIT; failed cached30d gate CHURNING; elapsed last START+28d EXIT; actual START reentry with same persona/trait; not source labels or predictors",
            "shopping_offset_units": "choice_logit_shift" if self.choice_space else "MDP_reward_shift",
            "choice_model": "fixed_irl_critic" if self.fixed_irl_critic else "choice_gmm" if self.choice_space else "reward",
            "profile_generation": "synthetic_population" if self.population is not None else "source_profiles",
            "checkout_spending_type": (self.population.get("checkout_spending") or {}).get("type") if self.population else None,
            "source_observation_mode": self.source_observation_mode,
            "calibration_component_conditioning": None if self.component_conditioning is None else self.component_conditioning.tolist()}
        self.state = {"cart": Counter(), "purchased_products": Counter(), "purchased_categories": Counter(),
            "purchased_units": 0, "purchases": 0, "last_purchase": None, "last_purchase_session": None,
            "first_event_at": None, "price_sum": 0, "price_count": 0, "observed_session_number": 0,
            "reference_price": self.model.get("reference_price_minor", self.profile["preferred_price_minor"]),
            "shopping_history": ShoppingHistory(self.origin), "checkout_remaining": {}}
        if self.return_state_model is not None:
            self.state["return_state"] = PrivateReturnState(self.return_state_model, self._auxiliary_rng("return_state"))
        self._receipts, self._reviews, self._prior_orders, self._review_queue = [], [], [], []
        self._purchase_units = {}
        self._window = self._active = None
        self._next_window = self.origin
        self._totals = Counter()
        self._failed = False
        self._lifecycle_phase, self._last_start = None, None
        self._set_runtime_phase("NEW", self.origin, "synthetic_entry_before_first_observed_START")
        if history_starts is not None:
            from .source_history import history_provenance
            self.records['_metadata']['source_history_conditioning']=history_provenance(
                self.settings['source_history_schedule_path'],self.history_bank)
            self.records['_metadata']['calendar_warmup_start_at']=self.origin_clock.isoformat()
            self.records['_metadata']['entry']='first observed source-history START; account signup is unobserved'
            self._window={'lower':self.origin,'upper':self.decision_start,'potential':history_starts,
                'index':0,'logit':0.,'draw':0.,'entry':True,'scope':'conditioned_observed_training_prefix_STARTs'}
            self._next_window=self.decision_start
        self._run_until(self.decision_start)
        if history_starts is not None:
            if self._active is not None:self._close_session('window_end')
            self._window=None
        self._totals.clear()

    @staticmethod
    def _population_profile_rng(seed, attempt):
        # Separate from choice/visit/response streams; no customer closure survives reset.
        return np.random.default_rng(np.random.SeedSequence([seed, attempt], spawn_key=(5,)))

    def _auxiliary_rng(self, kind):
        key = json.dumps(["source-event-stream-v1", self.seed, f"{self.uid}:{self._candidate_attempt}", kind]).encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def _set_runtime_phase(self, phase, at, reason):
        if phase != self._lifecycle_phase:
            self._lifecycle_phase = phase
            self.records["runtime_lifecycle"].append({"customer_id": self.uid, "as_of_at": at,
                "phase": phase, "reason": reason, "observed_session_number": self.state["observed_session_number"]})

    def _sample(self, key, fallback):
        group = self.home_groups[self.profile["home_group"]]
        values = self.profile.get(key) or group.get(key) or self.model.get("distributions", {}).get(key) or fallback
        return values[int(self.rng.integers(len(values)))]

    def _offsets(self, at):
        values = {self.purchase_shift_name: 0., "return_logit_shift": 0.}
        if self.runtime_reward_offsets is not None:
            supplied = self.runtime_reward_offsets(self, at)
            if not isinstance(supplied, dict) or not set(supplied) <= set(values):
                raise ValueError(f"offset hook must return {self.purchase_shift_name}/return_logit_shift in the selected backend units")
            for key, value in supplied.items():
                if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
                    raise ValueError("runtime reward shifts must be finite real values")
                values[key] = float(value)
        return values

    def _new_window(self):
        lower = self._next_window
        self.now = lower
        starts = [r["started_at"] for r in self.records["sessions"]]
        context = return_context(starts, self.records["events"], lower, purchase_item_units=self._purchase_units)
        eta = self.state["return_state"].value_at(lower) if self.return_state_model is not None else 0.
        logit = _logit(return_probability(self.return_model, context)) + self.theta[4] + eta
        prior = any(lower - pd.Timedelta(days=30) <= at < lower for at in starts)
        entry = lower == self.origin
        timing_key = self.profile["home_group"] if self.population is not None else self.source_profile_id
        offsets, scope = sample_session_offsets(self.timing_model, timing_key, entry, prior, self.visit_rng)
        if entry and self.source_observation_mode == "first_observed_entry_conditioned":
            if not offsets:
                raise ValueError("conditioned first-observed entry requires a nonempty schedule")
            offsets = [value - min(offsets) for value in offsets]
            anchor = self.origin
        else:
            anchor = self.origin.normalize() if entry else lower
        upper = lower + pd.Timedelta(days=30)
        potential = sorted(anchor + pd.Timedelta(seconds=float(v)) for v in offsets)
        potential = [at for at in potential if max(lower, self.origin) <= at < upper]
        self._window = {"lower": lower, "upper": upper, "potential": potential, "index": 0,
            "logit": logit, "draw": float(self.visit_rng.random()), "entry": entry, "scope": scope}
        self._next_window = upper
        if not entry:
            self._gate(lower)

    def _gate(self, at):
        active = self._window["entry"] or self._window["draw"] < _sigmoid(
            self._window["logit"] + self._offsets(at)["return_logit_shift"])
        if not active and self._lifecycle_phase != "EXIT":
            self._set_runtime_phase("CHURNING", at, "native30d_cached_gate_failed")
        return active

    def _context(self, session):
        state, at = self.state, session["last_at"]
        product = session.get("product")
        category = session["previous_category"] or (self.products[product]["category_id"] if product is not None else None)
        seen = session["offer_seen"] is not None
        context = {"previous_action": session["previous"], "unit_price_minor": self.product_prices.get(product) if seen else None,
            "reference_price_minor": state["reference_price"], "cart_quantity": state["cart"][product] if seen else 0,
            "cart_total_quantity": sum(state["cart"].values()),
            "days_since_first_observed": (at-state["first_event_at"]).total_seconds()/86400 if state["first_event_at"] is not None else 0.,
            "days_since_purchase": (at-state["last_purchase"]).total_seconds()/86400 if state["last_purchase"] is not None else None,
            "purchase_count": state["purchased_units"], "product_purchase_count": state["purchased_products"][product] if seen else 0,
            "purchase_in_current_session": state["last_purchase_session"] == session["id"],
            "category_purchase_share": state["purchased_categories"][category]/max(1,state["purchased_units"]),
            "session_event_count": session["units"], "session_decision_count": session["decisions"],
            "observed_session_number": session["number"],
            "prior_gap_seconds": session["prior_gap"], "previous_action_run_length": session["run"]}
        context.update(state["shopping_history"].snapshot(at))
        return context

    def _plan(self):
        session = self._active
        if session["decisions"] >= self.max_events:
            self._close_session("max_events")
            self._failed = True
            raise RuntimeError(f"Main session cap reached ({self.max_events}); episode is unscorable")
        shift = self._offsets(session["last_at"])[self.purchase_shift_name]
        if shift not in self._shifted_policies:
            if self.fixed_irl_critic:
                base = np.asarray(self._policy, dtype=float)
                logits = np.full(base.shape, -np.inf)
                np.log(base, out=logits, where=base > 0)
                logits[:, shopping.ACTIONS.index("purchase")] += shift
                logits -= logits.max(axis=1, keepdims=True)
                shifted = np.exp(logits)
                self._shifted_policies[shift] = shifted / shifted.sum(axis=1, keepdims=True)
            else:
                if self.solve_policy is None:
                    self.solve_policy = reward_gmm.prepare_bellman(self.latent, shopping.unpack_matrices(self.latent))
                offsets = self.theta[:4].copy()
                offsets[3] += shift
                self._shifted_policies[shift] = self.solve_policy(offsets, component_id=self.component, temperature=self.temperature)
        probabilities = shopping.policy(self.model, self._context(session), self.component,
                                        probabilities=self._shifted_policies[shift])
        values = np.asarray([probabilities[a] for a in shopping.ACTIONS], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("Main policy has no finite supported action distribution")
        action = shopping.ACTIONS[int(self.rng.choice(len(values), p=values/values.sum()))]
        if action == "end":
            self._close_session("causal_policy_end")
            return
        gap = float(self._sample("choice_gap_seconds", [0.])) if session["decisions"] else 0.
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("choice latency must be finite and nonnegative")
        at = session["last_at"] + pd.Timedelta(seconds=gap)
        if (at >= session["window_upper"] or at >= self.episode_end
                or session["next_start"] is not None and at > session["next_start"]):
            reason = "scheduled_next_start" if (session["next_start"] is not None
                and session["next_start"] < min(session["window_upper"], self.episode_end)) else "window_end"
            self._close_session(reason)
        else:
            session["planned"] = (at, action, gap)

    def _close_session(self, reason):
        session = self._active
        completed = reason == "causal_policy_end"
        self.state["shopping_history"].end_session(session["id"], session["last_at"], completed=completed)
        session["row"]["ended_at"] = session["last_at"]
        session["diagnostic"].update(as_of_at=session["last_at"], cart_quantity_after=sum(self.state["cart"].values()),
            truncated_session=not completed, termination_reason=reason)
        if self.return_state_model is not None:
            self.state["return_state"].complete_session(session["last_at"], reason, not completed)
        if completed and session["converted"]:
            self._set_runtime_phase("PURCHASE_EXIT", session["last_at"], "natural_END_of_purchase_session")
        self._active = None

    def _event(self, kind, at, session_id=None, **fields):
        row = dict.fromkeys(RAW_COLUMNS["events"])
        row.update(event_id=len(self.records["events"])+1, customer_id=self.uid, session_id=session_id,
                   event_at=at, event_type=kind, **fields)
        self.records["events"].append(row)
        return row

    def _offered(self):
        weights = {str(pid): value for pid, value in self.profile.get("product_weights", {}).items()
                   if str(pid) in self.available_products}
        ids = list(weights) or list(self.products)
        p = np.asarray(list(weights.values()), float) if weights else np.ones(len(ids))
        return ids[int(self.rng.choice(len(ids), p=p/p.sum()))]

    def _item(self, action, session, held=None):
        if self.model.get("item_choice") is None:
            if action == "remove":
                return str(self.rng.choice([pid for pid, qty in held.items() if qty > 0]))
            if action == "view" and session["decisions"]:
                return self._offered()
            return session["product"]
        probabilities, _ = item_probabilities(self.model["item_choice"], self.source_profile_id,
            self.profile["home_group"], action, session["previous"], session["previous_product"], self.available_products, held)
        if not probabilities:
            raise ValueError(f"no fitted item support for {action}")
        ids = list(probabilities)
        p = np.asarray(list(probabilities.values()),float)
        return ids[int(self.rng.choice(len(ids),p=p/p.sum()))]

    def _basket(self):
        from ..irl.fit_checkout_transition import checkout_pool_weights
        target_units = None
        for scope,key,templates in (("profile",self.source_profile_id,self.profile.get("checkout_templates", [])),
                ("home_group",self.profile["home_group"],self.home_groups[self.profile["home_group"]].get("checkout_templates", [])),
                ("source_pool","all",self.model.get("distributions",{}).get("checkout_templates", []))):
            cache_key = (scope,str(key))
            if cache_key not in self._basket_banks:
                baskets,counts=[],[]
                for template in templates:
                    basket={str(i["product_id"]):int(i["quantity"]) for i in template["items"]}
                    if basket and set(basket)<=self.available_products:
                        baskets.append(basket);counts.append(template["observed_count"])
                counts=np.asarray(counts,int);units=np.asarray([sum(b.values()) for b in baskets],int)
                means=np.asarray([sum(self.product_prices[p]*q for p,q in b.items())/sum(b.values()) for b in baskets])
                self._basket_banks[cache_key]=(baskets,counts,units,means)
            baskets,counts,units,means=self._basket_banks[cache_key]
            if not baskets: continue
            weights=counts/counts.sum()
            spending = self.population.get("checkout_spending") if self.population is not None else None
            if spending is not None:
                from ..irl.fit_checkout_spending import spending_pool_weights
                personal_weights = self.state.setdefault("checkout_spending_weights", {})
                if cache_key not in personal_weights:
                    personal_weights[cache_key] = spending_pool_weights(spending, weights, units, means,
                        self.profile["checkout_spending_preference"])
                p = personal_weights[cache_key]
                return baskets[int(self.rng.choice(len(baskets), p=p))].copy()
            if self.model.get("checkout_kernel"):
                if scope=="profile":
                    remaining=self.state["checkout_remaining"].setdefault(cache_key,counts.copy())
                    if remaining.sum():
                        index=int(self.rng.choice(len(baskets),p=remaining/remaining.sum()));remaining[index]-=1
                        return baskets[index].copy()
                    target_units=int(self.rng.choice(units,p=weights));continue
                if target_units is None and self.state.get("last_checkout_unit_mean") is not None:
                    target_units=int(self.rng.choice(units,p=weights))
                indices=np.flatnonzero(units==target_units) if target_units is not None else np.arange(len(baskets))
                if not len(indices):continue
                p=checkout_pool_weights(self.model["checkout_kernel"],weights[indices],means[indices],self.state.get("last_checkout_unit_mean"))
                return baskets[int(self.rng.choice(indices,p=p))].copy()
            return baskets[int(self.rng.choice(len(baskets),p=weights))].copy()
        raise ValueError("no supported observed checkout template")

    def _flush_reviews(self, until):
        while self._review_queue and self._review_queue[0][0] < until:
            at,_,purchase,product,line=heapq.heappop(self._review_queue)
            rating=review_events.sample_rating(self.review_model,review_events.review_context(self._reviews,at),self.review_rng)
            self._event("review",at,related_purchase_event_id=purchase,product_id=product,order_line_no=line,
                        rating=rating,rating_kind="product_star_scenario")
            self._reviews.append({"event_at":at,"rating":rating,"product_id":product,"purchase_event_id":purchase})

    def _purchase(self, at, session):
        basket=self._basket()
        from .population_profiles import align_checkout_basket
        spending=(self.population or {}).get("checkout_spending") or {}
        basket=align_checkout_basket(basket,self.profile,self.products,self.rng,cart=self.state["cart"],
            cart_fulfillment=spending.get("cart_fulfillment"))
        units=sum(basket.values());state=self.state
        held=sum(min(state["cart"][pid],qty) for pid,qty in basket.items())
        source="cart" if held==units else "mixed" if held else "buy_now"
        offer = session["offer_seen"]
        purchase=self._event("purchase",at,session["id"],offer_product_id=offer,
            offer_unit_price_minor=self.product_prices.get(offer));token=purchase["event_id"]
        order_id=f"{self.uid}:O{len(self.records['orders'])+1}"
        gross=sum(self.product_prices[pid]*qty for pid,qty in basket.items())
        eligible=[r for r in self._receipts if not r["used"] and r["sent_at"]<=at<r["expires_at"]
                  and r["response"]["response_selected"] and r["response"]["response_ready_at"]<=at]
        receipt=eligible[0] if eligible else None
        line,target=max(enumerate(basket,1),key=lambda pair:(self.product_prices[pair[1]],-pair[0]))
        reduction=discount_minor(self.product_prices[target],.1) if receipt else 0
        self.records["orders"].append({"order_id":order_id,"purchase_event_id":token,"checkout_source":source,
            "coupon_id":receipt["coupon_id"] if receipt else None,"gross_minor":gross,"discount_minor":reduction,"paid_minor":gross-reduction})
        for number,(pid,qty) in enumerate(basket.items(),1):
            self.records["order_items"].append({"order_id":order_id,"line_no":number,"product_id":pid,"quantity":qty,"unit_price_minor":self.product_prices[pid]})
            state["cart"][pid]-=min(state["cart"][pid],qty)
            state["purchased_products"][pid]+=qty;state["purchased_categories"][self.products[pid]["category_id"]]+=qty
            prior_count = state["price_count"] + qty - 1
            state["reference_price"] = ((state["price_sum"] + (qty-1)*self.product_prices[pid])/prior_count
                if prior_count else self.model.get("reference_price_minor", self.profile["preferred_price_minor"]))
            state["price_sum"]+=qty*self.product_prices[pid];state["price_count"]+=qty
            session["previous_product"] = pid
            session["previous_category"] = self.products[pid]["category_id"]
        if receipt:
            receipt["used"]=True
            self._event("coupon_usage",at,session["id"],product_id=target,quantity=1,unit_price_minor=self.product_prices[target],
                        related_purchase_event_id=token,coupon_id=receipt["coupon_id"],order_line_no=line,discount_minor=reduction)
        if self.review_model is not None:
            responded=list({r["purchase_event_id"]:r for r in self._reviews}.values())
            schedule=review_events.sample_review_schedule(self.review_model,review_events.order_review_context(self._prior_orders,responded,at),self.review_rng)
            if schedule["scheduled"]:
                due=at+pd.Timedelta(seconds=schedule["delay_seconds"])
                for number,pid in enumerate(basket,1):
                    heapq.heappush(self._review_queue,(due,token*100000+number,token,pid,number))
        self._prior_orders.append(at);self._purchase_units[token]=units
        state["last_purchase"]=at;state["last_purchase_session"]=session["id"];state["purchases"]+=1
        state["purchased_units"]+=units;state["last_checkout_unit_mean"]=gross/units
        self._totals.update(orders=1,paid_minor=gross-reduction,discount_minor=reduction,redemptions=int(receipt is not None))
        if not session["converted"]:
            self._totals["converted_sessions"]+=1;session["converted"]=True
        return units

    def _emit_action(self):
        session = self._active
        at, action, gap = session["planned"]
        self._flush_reviews(at)
        if self.state["first_event_at"] is None:
            self.state["first_event_at"] = at
        units, view_price = 1, None
        self.state["reference_price"] = (self.state["price_sum"] / self.state["price_count"]
            if self.state["price_count"] else self.model.get("reference_price_minor", self.profile["preferred_price_minor"]))
        if action == "purchase":
            units = self._purchase(at, session)
        else:
            pid = self._item(action, session, self.state["cart"] if action == "remove" else None)
            self.state["cart"][pid] += int(action == "cart") - int(action == "remove")
            self._event("remove_from_cart" if action == "remove" else action, at, session["id"], product_id=pid,
                        quantity=None if action == "view" else 1, unit_price_minor=self.product_prices[pid],
                        offer_product_id=session["offer_seen"], offer_unit_price_minor=self.product_prices.get(session["offer_seen"]))
            if action == "view":
                session["product"] = session["offer_seen"] = pid
            session["previous_product"] = pid
            session["previous_category"] = self.products[pid]["category_id"]
            view_price = self.product_prices[pid] if action == "view" else None
            self.state["price_sum"] += self.product_prices[pid]
            self.state["price_count"] += 1
        self.state["shopping_history"].record_action(session["id"], at, action, quantity=units, view_price_minor=view_price)
        session["last_at"] = at
        session["units"] += units
        session["decisions"] += 1
        session["prior_gap"] = 0. if units > 1 else gap
        session["run"] = session["run"] + 1 if session["previous"] == action else 1
        session["previous"] = action
        self._plan()

    def _start_session(self, at, next_start):
        sid = f"{self.uid}:S{len(self.records['sessions'])+1}"
        row = {"session_id": sid, "customer_id": self.uid, "started_at": at, "ended_at": None}
        diagnostic = {"customer_id": self.uid, "session_id": sid, "as_of_at": at, "session_started_at": at,
                      "cart_quantity_after": None, "truncated_session": True, "termination_reason": "ongoing"}
        self.records["sessions"].append(row)
        self.records["session_diagnostics"].append(diagnostic)
        self.state["shopping_history"].begin_session(sid, at)
        if self.return_state_model is not None:
            self.state["return_state"].begin_session(at)
        self.state["observed_session_number"] += 1
        old_phase = self._lifecycle_phase
        self._last_start = at
        self._set_runtime_phase("NEW" if self.state["observed_session_number"] <= 2 else "HOME", at,
            f"actual_START_return_after_{old_phase}" if old_phase in {"EXIT", "CHURNING", "PURCHASE_EXIT"} else "actual_START")
        self._active = {"id": sid, "row": row, "diagnostic": diagnostic, "last_at": at,
            "window_upper": self._window["upper"], "next_start": next_start,
            "number": self.state["observed_session_number"], "previous": "start", "units": 0,
            "decisions": 0, "prior_gap": 0., "run": 0,
            "product": self._offered() if self.model.get("item_choice") is None else None,
            "offer_seen": None, "previous_product": None, "previous_category": None, "converted": False}
        self._plan()

    def _run_until(self, until):
        if self._failed:
            raise RuntimeError("episode hit a session cap; reset before advancing")
        while True:
            if self._window is None:
                self._new_window()
            window = self._window
            potential = window["potential"]
            next_visit = potential[window["index"]] if window["index"] < len(potential) else window["upper"]
            next_action = self._active["planned"][0] if self._active is not None else self.episode_end + pd.Timedelta(days=1)
            deadline = (self._last_start + pd.Timedelta(days=28) if self._last_start is not None
                and self._lifecycle_phase != "EXIT" else self.episode_end + pd.Timedelta(days=1))
            at = min(next_visit, next_action, deadline)
            if at >= until:
                break
            self.now = at
            if deadline <= min(next_visit, next_action):
                self._set_runtime_phase("EXIT", deadline, "elapsed_28d_without_observed_START")
                continue
            # Native Main permits an action exactly at the next scheduled START.
            # Process the previous session first; tied STARTs remain separate rows.
            if next_action <= next_visit:
                self._emit_action()
                continue
            if window["index"] >= len(potential):
                if self._active is not None:
                    self._close_session("window_end")
                self._flush_reviews(window["upper"])
                self._window = None
                continue
            self._flush_reviews(at)
            window["index"] += 1
            if self._gate(at):
                if self._active is not None:
                    self._close_session("scheduled_next_start")
                next_start = potential[window["index"]] if window["index"] < len(potential) else None
                self._start_session(at, next_start)
        self._flush_reviews(until)
        self.now = until
        if self.return_state_model is not None:
            self.state["return_state"].value_at(until)
        self.records["_metadata"]["recorded_through_at"] = until.isoformat()

    def advance(self, action, days=7):
        if type(action) is not int or action not in (0, 1):
            raise ValueError("coupon action must be integer zero or one")
        if type(days) is not int or days <= 0:
            raise ValueError("days must be a positive integer")
        if self._failed:
            raise RuntimeError("episode hit a session cap; reset before advancing")
        if self.now >= self.episode_end:
            raise ValueError("episode finished; reset before advancing")
        before, lower = self._totals.copy(), self.now
        if action:
            if self.response_sampler is None:
                raise ValueError("coupon sending requires an explicit response_sampler scenario")
            response = copy.deepcopy(self.response_sampler(self, lower))
            if not isinstance(response, dict) or type(response.get("response_selected")) is not bool:
                raise ValueError("response_sampler must return response_selected bool and response_ready_at")
            if response["response_selected"]:
                ready = pd.Timestamp(response.get("response_ready_at"))
                if pd.isna(ready) or ready.tzinfo is None or ready < lower:
                    raise ValueError("selected receipt readiness must be aware and not precede sending")
                response["response_ready_at"] = ready.tz_convert("UTC")
            else:
                response["response_ready_at"] = None
            coupon_id = f"{self.uid}:C{len(self.records['coupons'])+1}"
            row = {"coupon_id": coupon_id, "customer_id": self.uid, "sent_at": lower,
                "expires_at": lower + pd.Timedelta(days=self.coupon_valid_days), "discount_rate": .1,
                "policy_id": "main_weekly_fifo_10pct_28d", "send_cost_minor": self.send_cost}
            self.records["coupons"].append(row)
            self._receipts.append({**row, "used": False, "response": response})
            self._totals.update(sends=1, send_cost_minor=self.send_cost)
        self._run_until(min(lower + pd.Timedelta(days=days), self.episode_end))
        fields = ("converted_sessions", "orders", "sends", "redemptions", "paid_minor", "discount_minor", "send_cost_minor")
        info = {field: int(self._totals[field] - before[field]) for field in fields}
        cost = info["sends"] if self.cost_model == "S" else info["redemptions"]
        reward = self.score_conversion_value * info["converted_sessions"] - self.score_coupon_cost * cost
        done = self.now >= self.episode_end
        info.update(cost_model=self.cost_model, domain=self.domain, at=self.now.isoformat(), score=reward,
            score_cost=self.score_coupon_cost * cost, score_conversion_value=self.score_conversion_value,
            score_coupon_cost=self.score_coupon_cost, actual_payment_net_send_cost_minor=info["paid_minor"] - info["send_cost_minor"])
        return self._features(), float(reward), done, info
