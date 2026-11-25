"""Weekly decisions over Tianchi's four unordered daily presence outcomes.

Date labels encoded at UTC00 are coordinates, mapped explicitly to Asia/Seoul
business dates. A sampled day remains private until the next business midnight.
There are no invented retail sessions, ENDs, SKUs, carts, quantities or payments.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from ..configuration import load_config
from ..features import EXPANDED_FEATURES
from ..irl import tianchi_daily as daily
from ..time_spec import weekly_calendar, weekly_metadata


ACTIVE = [0, 3, 4]
RAW_DAILY_COLUMNS = ("customer_id", "model_day", "day_start", "day_end", "known_at", "view", "purchase", "outcome", "conditional_entry")
LIFECYCLE_COLUMNS = ("customer_id", "as_of_at", "phase", "reason", "observed_active_days")


def model_date(at):
    """Map a KST business midnight to its UTC00 model date coordinate."""
    at = pd.Timestamp(at)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError("daily campaign clocks must be timezone-aware business midnights")
    local = at.tz_convert("Asia/Seoul")
    if local != local.normalize():
        raise ValueError("daily campaign clocks must be Asia/Seoul business midnights")
    return pd.Timestamp(local.date(), tz="UTC")


class TianchiDailyCampaignEnv:
    """Persistent daily customer with the Main reset/advance raw39 interface.

    ``response_sampler(env, sent_at)`` supplies a separately calibrated/scenario
    receipt: response_selected bool and response_ready_at business midnight.
    ``runtime_reward_offsets(env, decision_at)`` supplies purchase_reward_shift
    and return_reward_shift (daily activity reward, never a30d logit).
    Neither hook is used by default. No retail return/AR/coupon model is loaded.
    """

    @staticmethod
    def customer_id_for_seed(seed):
        return str(seed)

    def __init__(self, config, *, assets_root=None, cost_model="S", warmup_days=364,
                 horizon_days=364, min_history_days=91, coupon_valid_days=28,
                 response_sampler=None, runtime_reward_offsets=None, feature_builder=None,
                 score_conversion_value=1000., score_coupon_cost=100.):
        if cost_model not in {"S", "D"}:
            raise ValueError("cost_model must be S or D")
        for name, value, positive in (("score_conversion_value",score_conversion_value,True),
                                     ("score_coupon_cost",score_coupon_cost,False)):
            if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Real)
                    or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
                raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} real value")
        self.score_conversion_value,self.score_coupon_cost = float(score_conversion_value),float(score_coupon_cost)
        for name, value, minimum in (("warmup_days",warmup_days,1),("horizon_days",horizon_days,1),("min_history_days",min_history_days,0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if min_history_days > warmup_days or type(coupon_valid_days) is not int or coupon_valid_days != 28:
            raise ValueError("history must fit warmup and daily campaign coupons last28 days")
        self.config = copy.deepcopy(load_config(config,assets_root=assets_root) if isinstance(config,(str,Path)) else config)
        self.settings = self.config["source_simulation"]
        if self.settings.get("backend") != "tianchi_daily_campaign":
            raise ValueError("Tianchi daily campaigns require their separate daily backend")
        forbidden = ("model_path", "return_model_path", "return_state_model_path", "visit_timing_model_path",
                     "coupon_response_model_path", "gaussian_reward_model_path")
        if any(self.settings.get(key) for key in forbidden):
            raise ValueError("retail actors,30d return/AR,timing and coupon-response assets cannot enter the daily backend")
        self.origin_clock = weekly_calendar(self.config["simulation"]["start_at"])[0]
        self.cost_model,self.warmup_days,self.horizon_days = cost_model,warmup_days,horizon_days
        self.min_history_days,self.coupon_valid_days = min_history_days,coupon_valid_days
        self.response_sampler,self.runtime_reward_offsets,self.feature_builder = response_sampler,runtime_reward_offsets,feature_builder
        if any(callback is not None and not callable(callback) for callback in (response_sampler,runtime_reward_offsets,feature_builder)):
            raise ValueError("daily runtime hooks must be callable")
        self.model_shas = {}
        root = Path(assets_root or (Path(config).resolve().parent if isinstance(config,(str,Path)) else Path.cwd()))

        def read(path, name):
            path = Path(path)
            path = path if path.is_absolute() else root/path
            content = path.read_bytes()
            self.model_shas[name] = hashlib.sha256(content).hexdigest()
            return json.loads(content)

        paths = self.settings.get("daily_actor_model_paths")
        if not isinstance(paths,dict) or not paths:
            raise ValueError("frozen daily group actor paths are required")
        self.models = {int(label):read(path,f"daily_actor:{label}") for label,path in paths.items()}
        if any(str(int(label)) != str(label) for label in paths) or len(self.models) != len(paths):
            raise ValueError("daily actor IDs must be distinct integer text")
        self.actor_sha = {str(index):self.model_shas[f"daily_actor:{index}"] for index in self.models}
        self.verification = read(self.settings["daily_actor_verification_path"],"daily_actor_verification_path")
        self.entry = read(self.settings["initial_entry_model_path"],"initial_entry_model_path")
        self.gmm = read(self.settings["reward_gmm_model_path"],"reward_gmm_model_path") if self.settings.get("reward_gmm_model_path") else None
        # This also validates every actor and the common frozen state/transition basis.
        reference = self.models[min(self.models)]
        self.source_contract = reference["source_contract"]
        if (self.source_contract.get("feature_windows_days") != [28,56,91]
                or reference.get("lifecycle_memory") != daily.LIFECYCLE_MEMORY):
            raise ValueError("weekly daily actors require the frozen4/8/13w basis and first-two-active-days memory")
        self._validate_entry(reference)
        self.gmm_factors = self._validate_gmm(reference) if self.gmm is not None else None
        self.labels = list(self.entry["persona_labels"])
        self.profile_probabilities = np.asarray(self.entry["persona_probabilities"],float)
        self.profiles = tuple({"profile_id":label,"home_group":label} for label in self.labels)
        self.temperatures = {index:1. for index in self.models}
        if self.gmm is not None:
            self.temperatures.update({actor:float(self.gmm.get("choice_temperatures",[1.]*len(self.labels))[component])
                for component,actor in enumerate(self.gmm["component_model_indices"])})
        scaled = {index:{**model,"reward_weights":(np.asarray(model["reward_weights"])/self.temperatures[index]).tolist()}
                  for index,model in self.models.items()}
        self._policy_provider = daily.daily_policy_log_provider(scaled)
        self.send_cost = self.settings.get("coupon_send_cost_minor",100)
        if type(self.send_cost) is not int or self.send_cost < 0:
            raise ValueError("sending cost must be a nonnegative integer scenario amount")

    def _validate_entry(self, reference):
        entry = self.entry
        if (entry.get("schema_version") != 1 or entry.get("type") != "tianchi_daily_initial_entry_priors"
                or entry.get("base_model_sha256") != self.model_shas["daily_actor_verification_path"]
                or entry.get("daily_actor_model_sha256") != self.actor_sha
                or entry.get("source_contract") != reference["source_contract"]
                or entry.get("observation_contract") != reference["observation_contract"]
                or entry.get("entry_outcome_order") != list(daily.OUTCOMES[1:])):
            raise ValueError("daily entry priors must bind the exact frozen source and actor models")
        daily._coverage(entry["source_contract"],entry["observation_contract"])
        train = entry.get("training",{})
        cutoff = pd.Timestamp(reference["training"]["cutoff_at"])
        known = pd.Timestamp(train.get("max_entry_known_at"))
        if (train.get("training_users_only") is not True or train.get("first_active_day_is_conditioned") is not True
                or pd.Timestamp(train.get("cutoff_at")) != cutoff or pd.isna(known) or known.tzinfo is None or not known < cutoff):
            raise ValueError("entry outcomes require strictly mature source training prefixes")
        labels = entry.get("persona_labels",[])
        mass = np.asarray(entry.get("persona_probabilities"),float)
        if (len(labels)!=len(self.models) or len(set(labels))!=len(labels) or set(labels)!={str(i) for i in self.models}
                or mass.shape!=(len(labels),) or not np.isfinite(mass).all() or (mass<0).any() or not np.isclose(mass.sum(),1.,atol=1e-10,rtol=0)):
            raise ValueError("daily entry persona prior differs from its actor support")
        outcomes = entry.get("entry_outcome_probabilities",{})
        if set(outcomes)!=set(labels):
            raise ValueError("every daily entry persona needs its conditional active-day outcomes")
        for values in outcomes.values():
            p=np.asarray(values,float)
            if p.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1.,atol=1e-10,rtol=0):
                raise ValueError("entry distributions contain only the three active combinations")

    def _validate_gmm(self, reference):
        block = self.gmm
        count = len(self.models)
        if (block.get("schema_version")!=1 or block.get("type")!="source_persona_daily_reward_gmm"
                or block.get("backend")!="tianchi_daily" or block.get("purpose")!="synthesis_only"
                or block.get("parameter_names")!=list(daily.TRAIT_NAMES) or block.get("component_count")!=count
                or block.get("base_model_sha256")!=self.model_shas["daily_actor_verification_path"]
                or block.get("daily_actor_model_sha256")!=self.actor_sha
                or block.get("source_contract")!=reference["source_contract"]
                or block.get("observation_contract")!=reference["observation_contract"]
                or block.get("available_actions")!=["view","purchase"]
                or block.get("fitted_parameter_names")!=["view","purchase","return"]):
            raise ValueError("daily GMM requires exact actor/source bindings and daily reward units")
        labels=block.get("persona_labels",[])
        indices=block.get("component_model_indices",[])
        if (labels!=self.entry["persona_labels"] or indices!=[int(label) for label in labels]
                or block.get("condition_on_profile_components") is not True
                or not np.allclose(block.get("mixture_weights"),self.entry["persona_probabilities"],atol=1e-10,rtol=0)):
            raise ValueError("daily GMM and initial persona probabilities must share their frozen source prior")
        means,covariance=(np.asarray(block.get(name),float) for name in ("means","covariances"))
        if means.shape!=(count,5) or covariance.shape!=(count,5,5) or not np.isfinite(means).all() or not np.isfinite(covariance).all():
            raise ValueError("daily GMM requires finite canonical5 arrays")
        marginal=covariance[:,ACTIVE][:,:,ACTIVE]
        if not np.allclose(marginal,marginal.swapaxes(1,2),atol=1e-12,rtol=1e-10):
            raise ValueError("daily GMM active covariance must be symmetric")
        for name,positive in (("choice_temperatures",True),("dispersion_scales",False)):
            values=np.asarray(block.get(name,[1.]*count),float)
            if values.shape!=(count,) or not np.isfinite(values).all() or (values<0).any() or positive and (values<=0).any():
                raise ValueError("daily Gaussian temperature/dispersion is invalid")
        return np.linalg.cholesky(marginal)

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self,memo):
        result=type(self).__new__(type(self));memo[id(self)]=result
        for name in ("models","verification","entry","gmm","gmm_factors","source_contract","profiles"):
            if name in self.__dict__:memo[id(self.__dict__[name])]=self.__dict__[name]
        if "_daily" in self.__dict__:memo[id(self._daily.model)]=self._daily.model
        result.__dict__=copy.deepcopy(self.__dict__,memo)
        return result

    def reset(self,seed):
        if type(seed) is not int or seed<0:
            raise ValueError("episode seed must be a nonnegative integer")
        for attempt in range(getattr(self,"INITIAL_CANDIDATES",1000)):
            self._reset_candidate(seed,attempt)
            days=self._daily.history
            lower=self._daily.at-pd.Timedelta(days=28)
            if ((days.day_start.ge(lower)&(days.view|days.purchase)).any()
                    and (self.now-self.origin).days>=self.min_history_days):
                self.records["_metadata"].update(initial_candidate_attempts=attempt+1,
                    initial_eligibility="generated known active-day in past28d; no future-dependent selection")
                return self._features()
        raise ValueError("no eligible daily initial customer in the bounded past-only bank")

    def _reset_candidate(self,seed,attempt):
        self.seed,self._candidate_attempt=seed,attempt
        streams=np.random.SeedSequence([seed,attempt]).spawn(4)
        self.rng,self.day_rng,self.entry_rng,self.response_rng=[np.random.default_rng(s) for s in streams]
        self.component=int(self.rng.choice(len(self.labels),p=self.profile_probabilities))
        label=self.labels[self.component]
        self.actor=int(label);self.source_profile_id=label
        self.profile=self.profiles[self.component]
        self.theta=np.zeros(5)
        self.temperature=self.temperatures[self.actor]
        if self.gmm is not None:
            self.theta[ACTIVE]=np.asarray(self.gmm["means"])[self.component,ACTIVE]
            dispersion=self.gmm.get("dispersion_scales",[1.]*len(self.labels))[self.component]
            if dispersion:
                self.theta[ACTIVE]+=dispersion*(self.gmm_factors[self.component]@self.rng.standard_normal(3))
        self.uid=self.customer_id_for_seed(seed)
        self.origin=self.now=self.origin_clock
        self.decision_start=self.origin+pd.Timedelta(days=self.warmup_days)
        self.episode_end=self.decision_start+pd.Timedelta(days=self.horizon_days)
        entry_outcome=1+int(self.entry_rng.choice(3,p=self.entry["entry_outcome_probabilities"][label]))
        view,purchase=daily.FLAGS[entry_outcome]
        first=model_date(self.origin)
        prefix=pd.DataFrame(dict(user_id=[self.uid],day_start=[first],view=[view],purchase=[purchase]))
        self._daily=daily.TianchiDailyEnv(self.models[self.actor],prefix,as_of_at=first+daily.DAY,seed=0)
        self._daily.rng=self.day_rng
        self._policy_cache={}
        self._receipts=[];self._totals=Counter()
        self.state={"observed_active_days":0,"inactive_days":0}
        self._lifecycle_phase=None;self._last_active=None;self._inactive_days=0
        self.records={"customers":[{"customer_id":self.uid,"first_observed_day":self.origin}],
            "raw_daily":[],"coupons":[],"coupon_redemptions":[],"runtime_lifecycle":[],"daily_diagnostics":[],
            "_metadata":{**weekly_metadata(),"domain":"tianchi","backend":"tianchi_daily_campaign","cost_model":self.cost_model,
                "score_conversion_value":self.score_conversion_value,"score_coupon_cost":self.score_coupon_cost,
                "score_objective_units":"normalized simulated value; independent of raw sending amounts; cash remains unobserved",
                "model_sha256":dict(self.model_shas),"observation_start_at":self.origin.isoformat(),
                "observation_end_at":self.episode_end.isoformat(),"as_of_at":self.decision_start.isoformat(),
                "warmup_days":self.warmup_days,"decision_days":self.horizon_days,
                "clock_mapping":"source UTC00 date indices are coordinates; map date to Asia/Seoul business midnight; previous day released before next day's choice",
                "entry":"first observed active date is conditioned on training persona priors, not signup or a predicted arrival",
                "observable_unit":"unordered daily click/purchase presence; no fabricated sessions,END,SKU,price,quantity or payments",
                "feature_roles":dict(self.models[self.actor]["public_feature_roles"]),
                "score_unit":"converted_sessions is a compatibility alias for purchase-days; orders and actual cash are unobserved",
                "lifecycle":"private waiting diagnostic CHURNING after sampled no-activity day;28 elapsed observed-inactive days EXIT; actual active day NEW/HOME by strict-past count",
                "capture_assumption":copy.deepcopy(self.models[self.actor]["observation_contract"])}}
        self._pending={"model_day":first,"day_start":self.origin,"day_end":self.origin+daily.DAY,
            "known_at":self.origin+daily.DAY,"view":view,"purchase":purchase,"outcome":daily.OUTCOMES[entry_outcome],"conditional_entry":True}
        self._set_phase("NEW",self.origin,"conditioned_entry_pending")
        self._run_until(self.decision_start)
        self._totals.clear()

    def _set_phase(self,phase,at,reason):
        if phase!=self._lifecycle_phase:
            self._lifecycle_phase=phase
            self.records["runtime_lifecycle"].append(dict(customer_id=self.uid,as_of_at=at,phase=phase,
                reason=reason,observed_active_days=self.state["observed_active_days"]))

    def _offsets(self,at):
        offsets={"purchase_reward_shift":0.,"return_reward_shift":0.}
        if self.runtime_reward_offsets is not None:
            supplied=self.runtime_reward_offsets(self,at)
            if not isinstance(supplied,dict) or not set(supplied)<=set(offsets):
                raise ValueError("daily shifts are purchase_reward_shift/return_reward_shift; no30d logit")
            for name,value in supplied.items():
                if isinstance(value,(bool,np.bool_)) or not np.isfinite(float(value)):
                    raise ValueError("daily reward shifts must be finite real values")
                offsets[name]=float(value)
        return offsets

    def _plan_day(self):
        if self._pending is not None:
            return
        at=self.now
        if at>=self.episode_end:
            return
        offsets=self._offsets(at)
        key=(offsets["purchase_reward_shift"],offsets["return_reward_shift"])
        if key not in self._policy_cache:
            theta=self.theta.copy();theta[3]+=key[0];theta[4]+=key[1]
            logs=self._policy_provider(self.actor,(theta/self.temperature)[None,:])
            self._policy_cache[key]=np.exp(logs.reshape(-1,5)[:,:4])
        self._daily._policy=self._policy_cache[key]
        before=len(self._daily._active)
        phase="NEW" if before<2 else "HOME"
        row=self._daily.step()
        self._pending=dict(model_day=row["day_start"],day_start=at,day_end=at+daily.DAY,known_at=at+daily.DAY,
            view=row["view"],purchase=row["purchase"],outcome=row["outcome"],conditional_entry=False)
        self.records["daily_diagnostics"].append(dict(customer_id=self.uid,decision_at=at,known_at=at+daily.DAY,
            actor_phase_before=phase,observed_active_days_before=before))
        # This is a private sampled waiting state, never an observed source label.
        if not(row["view"] or row["purchase"]) and self._lifecycle_phase!="EXIT":
            self._set_phase("CHURNING",at,"sampled_next_day_no_activity_pending")

    def _release_day(self):
        row=self._pending;at=row["known_at"]
        self.records["raw_daily"].append({"customer_id":self.uid,**row})
        if row["view"] or row["purchase"]:
            self._last_active=row["day_start"]
            self._inactive_days=0
            self.state["observed_active_days"]+=1
            count=self.state["observed_active_days"]
            self._set_phase("NEW" if count<2 else "HOME",at,"known_active_date_reentry")
        else:
            self._inactive_days+=1
            if self._inactive_days>=28:
                self._set_phase("EXIT",at,"28_fully_observed_consecutive_no_activity_days")
        if row["purchase"]:
            eligible=[r for r in self._receipts if not r["used"] and r["sent_at"]<=row["day_start"]<r["expires_at"]
                and r["response"]["response_selected"] and r["response"]["response_ready_at"]<=row["day_start"]]
            if eligible:
                receipt=eligible[0];receipt["used"]=True
                self.records["coupon_redemptions"].append(dict(customer_id=self.uid,coupon_id=receipt["coupon_id"],
                    purchase_day=row["day_start"],known_at=at,discount_minor=None,paid_minor=None))
                self._totals["redemptions"]+=1
            self._totals["converted_sessions"]+=1
        self.state["inactive_days"]=self._inactive_days
        self._released_observation=self._daily.observation()
        self._pending=None

    def _run_until(self,until):
        while self.now<until:
            self._plan_day()
            if self._pending["known_at"]>until:
                break
            self.now=self._pending["known_at"]
            self._release_day()
        self.now=until
        self.records["_metadata"]["recorded_through_at"]=until.isoformat()

    def _features(self):
        if self.feature_builder is not None:
            values=np.asarray(self.feature_builder(self),float)
        else:
            # The incremental engine may already hold a private pending day.
            # Cache its observation only when the corresponding label is released.
            observation=self._released_observation
            values=np.asarray([np.nan if observation[name] is None else observation[name] for name in EXPANDED_FEATURES])
        if values.shape!=(39,) or np.isinf(values).any():
            raise ValueError("daily public observation must preserve nullable raw39")
        return values

    @property
    def tables(self):
        result={name:pd.DataFrame(rows) for name,rows in self.records.items() if name!="_metadata"}
        result["raw_daily"]=pd.DataFrame(self.records["raw_daily"],columns=RAW_DAILY_COLUMNS)
        result["runtime_lifecycle"]=pd.DataFrame(self.records["runtime_lifecycle"],columns=LIFECYCLE_COLUMNS)
        result["_metadata"]=copy.deepcopy(self.records["_metadata"])
        return result

    def advance(self,action,days=7):
        if type(action) is not int or action not in (0,1):
            raise ValueError("coupon action must be integer zero or one")
        if type(days) is not int or days<=0:
            raise ValueError("days must be a positive integer")
        if self.now>=self.episode_end:
            raise ValueError("episode finished; reset before advancing")
        before=self._totals.copy();lower=self.now
        if action:
            if self.response_sampler is None:
                raise ValueError("daily coupon sending requires an explicit separately calibrated/scenario response_sampler")
            response=copy.deepcopy(self.response_sampler(self,lower))
            if not isinstance(response,dict) or type(response.get("response_selected")) is not bool:
                raise ValueError("response_sampler must return selected bool and ready business date")
            if response["response_selected"]:
                ready=pd.Timestamp(response.get("response_ready_at"));model_date(ready)
                if ready<lower:raise ValueError("receipt readiness cannot precede sending")
                response["response_ready_at"]=ready.tz_convert("UTC")
            else:response["response_ready_at"]=None
            row=dict(customer_id=self.uid,coupon_id=f"{self.uid}:C{len(self._receipts)+1}",sent_at=lower,
                expires_at=lower+pd.Timedelta(days=28),send_cost_minor=self.send_cost,discount_minor=None)
            self.records["coupons"].append(row)
            self._receipts.append({**row,"used":False,"response":response})
            self._totals.update(sends=1,send_cost_minor=self.send_cost)
        self._run_until(min(lower+pd.Timedelta(days=days),self.episode_end))
        info={key:int(self._totals[key]-before[key]) for key in ("converted_sessions","sends","redemptions","send_cost_minor")}
        cost=info["sends"] if self.cost_model=="S" else info["redemptions"]
        reward=self.score_conversion_value*info["converted_sessions"]-self.score_coupon_cost*cost
        info.update(orders=None,paid_minor=None,discount_minor=None,actual_payment_net_send_cost_minor=None,
            converted_purchase_days=info["converted_sessions"],score=reward,score_cost=self.score_coupon_cost*cost,
            score_conversion_value=self.score_conversion_value,score_coupon_cost=self.score_coupon_cost,
            cost_model=self.cost_model,domain="tianchi",at=self.now.isoformat(),unit="daily purchase presence; no observed order or cash amounts")
        return self._features(),float(reward),self.now>=self.episode_end,info
