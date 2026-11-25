"""Tianchi date observations and a linear daily soft-Bellman choice model.

The four outcomes are no activity and three unordered click/purchase-presence
combinations. They occupy numerical slots, not the retail action semantics.
There is no within-day sequence, session END, price, SKU, cart or item quantity.
Absence is a supplied-corpus measurement only under an explicit capture contract.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_left
from copy import deepcopy
import json
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from simulate.v4.features import EXPANDED_FEATURES
from . import feature_reward, fit_latent_shopping as shopping
from .source_contract import load_prepared_source, utc_cutoff


TYPE = "tianchi_observed_daily_linear_soft_bellman"
LIFECYCLE_MEMORY = "first_two_observed_active_days"
OUTCOMES = ("no_activity", "click_only_day", "purchase_only_day", "click_purchase_day")
FLAGS = ((False, False), (True, False), (False, True), (True, True))
DAY = pd.Timedelta(days=1)
CORE_NAMES = tuple(shopping.FEATURE_NAMES)
CORE_SUPPORTED = ("log_days_since_first_observed", "log_days_since_purchase", "has_purchase_history")
CORE_INDEX = tuple(CORE_NAMES.index(name) for name in CORE_SUPPORTED)
TRAIT_NAMES = ("view", "cart", "remove", "purchase", "return")
CANONICAL_PARAMETER_NAMES = ("view", "cart", "remove", "purchase", "return_logit")
TRAIT_INCIDENCE = np.asarray([[0, 0, 0, 0, 0], [1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=float)
PUBLIC_ROLES = {
    "visit_recency_days": "daily_proxy:last observed active date recency",
    "purchase_recency_days": "daily_proxy:last observed purchase date recency",
    "observed_tenure_days": "daily_proxy:first observed active date, not signup",
    "session_count_30d": "daily_proxy:active dates, not second-resolution sessions",
    "purchase_count_30d": "daily_proxy:purchase dates, not orders or item quantity",
    "mean_purchase_gap_days": "daily_proxy:gaps between observed purchase dates",
    "intervisit_start_gap_mean_days_90d": "daily_proxy:active-date gaps with later date in90d",
    "intervisit_start_gap_cv_90d": "daily_proxy:sample CV of active-date gaps",
    "orders_delta_30d": "daily_proxy:purchase-date count difference between adjacent30d windows",
}
STATE_NAMES = (*EXPANDED_FEATURES, *CORE_NAMES)
REWARD_FEATURE_NAMES = ("intercept", *(f"public:{name}" for name in EXPANDED_FEATURES),
                        *(f"internal:{name}" for name in CORE_NAMES))
STATE_INDEX = tuple(index for index, name in enumerate(EXPANDED_FEATURES) if name in PUBLIC_ROLES) + tuple(
    len(EXPANDED_FEATURES) + index for index in CORE_INDEX)


def _day(value):
    value = utc_cutoff(value)
    if value != value.floor("D"):
        raise ValueError("daily decision and fitting cutoffs must be UTC date boundaries")
    return value


def _user_ids(series):
    if not series.map(lambda value: isinstance(value, str) and value.isdigit()).all():
        raise ValueError("Tianchi IDs must retain original numeric text; no floating or joint IDs")


def _coverage(source, observation):
    if source.get("domain") != "tianchi" or source.get("clock_resolution") != "day":
        raise ValueError("a daily Tianchi source contract is required")
    if (source.get("observed_prices") is not False or not source.get("observed_actions")
            or not set(source["observed_actions"]) <= {"view", "purchase"}):
        raise ValueError("Tianchi observation support cannot include retail prices or other actions")
    if not observation or observation.get("assume_complete_behavior_capture_within_declared_interval") is not True:
        raise ValueError("explicit complete supplied-corpus capture assumption required; sparse absence is not observed inactivity")
    for name in ("observation_start_at", "observation_end_at"):
        if utc_cutoff(observation[name]) != utc_cutoff(source[name]):
            raise ValueError("daily capture boundaries differ from the bound source contract")
    if not source.get("prepared_sha256") or observation.get("prepared_sha256") != source["prepared_sha256"]:
        raise ValueError("daily capture assumption must bind the prepared source SHA256")
    start, end = (utc_cutoff(source[name]) for name in ("observation_start_at", "observation_end_at"))
    if end <= start:
        raise ValueError("daily capture must have positive observation duration")
    return start, end


def daily_presence(rows):
    """Collapse only unordered user/date action presence; source rows stay unchanged."""
    if rows.empty or not {"user_id", "at", "action"} <= set(rows):
        raise ValueError("nonempty prepared Tianchi user/date/action rows required")
    _user_ids(rows.user_id)
    if (not isinstance(rows["at"].dtype, pd.DatetimeTZDtype) or rows["at"].isna().any()
            or not rows["at"].eq(rows["at"].dt.floor("D")).all()
            or not rows.action.isin(["view", "purchase"]).all()
            or "price_minor" in rows and rows.price_minor.notna().any()):
        raise ValueError("Tianchi has only dated click/purchase presence and unobserved prices")
    flags = (rows.assign(view=rows.action.eq("view"), purchase=rows.action.eq("purchase"))
             .groupby(["user_id", "at"], sort=True)[["view", "purchase"]].max().reset_index())
    return flags.rename(columns={"at": "day_start"})


def _logs(rows, source, observation, through):
    start, end = _coverage(source, observation)
    through = min(_day(through), end.floor("D"))
    presence = daily_presence(rows)
    if presence.day_start.lt(start.floor("D")).any() or presence.day_start.ge(end).any():
        raise ValueError("Tianchi dates fall outside the explicit capture interval")
    presence = presence.loc[presence.day_start.lt(through)]
    for uid, user in presence.groupby("user_id", sort=True):
        days = pd.date_range(user.day_start.min(), through, freq="D", inclusive="left")
        user = user.set_index("day_start")[["view", "purchase"]].reindex(days, fill_value=False)
        user.index.name = "day_start"
        user = user.reset_index()
        user.insert(0, "user_id", uid)
        user["day_end"] = user.day_start + DAY
        user["outcome"] = (user.view.astype(int) + 2 * user.purchase.astype(int)).map(dict(enumerate(OUTCOMES)))
        yield user


def daily_history(rows, source_contract, *, observation_contract, as_of_at):
    """Known date prefix including explicit covered no-activity days, never pre-entry zeros."""
    parts = list(_logs(rows, source_contract, observation_contract, as_of_at))
    if not parts:
        raise ValueError("no observed active-date prefix is available")
    return pd.concat(parts, ignore_index=True)


def _vectors(log):
    n = len(log)
    index = np.arange(n)
    purchases = np.maximum.accumulate(np.where(log.purchase.to_numpy(), index, -1))
    previous = np.r_[-1, purchases[:-1]]
    before, after = np.zeros((n, len(CORE_NAMES))), np.zeros((n, len(CORE_NAMES)))
    before[:, CORE_INDEX[0]], after[:, CORE_INDEX[0]] = np.log1p(index), np.log1p(index + 1)
    for target, age, last in ((before, index, previous), (after, index + 1, purchases)):
        observed = last >= 0
        target[observed, CORE_INDEX[1]] = np.log1p(age[observed] - last[observed])
        target[:, CORE_INDEX[2]] = observed
    return before, after


def _feature_windows(source):
    windows = tuple(source.get("feature_windows_days", (30, 60, 90)))
    if windows not in ((30, 60, 90), (28, 56, 91)):
        raise ValueError("daily feature windows must preserve the calendar or weekly39 contract")
    return windows


def _public_vectors(log, windows=(30, 60, 90)):
    """Existing39 values at every date boundary; undefined observations remain NaN."""
    n = len(log)
    day = np.arange(n)
    clock = np.arange(n + 1)
    active = (log.view | log.purchase).to_numpy()
    bought = log.purchase.to_numpy()
    last_active = np.r_[-1, np.maximum.accumulate(np.where(active, day, -1))]
    last_buy = np.r_[-1, np.maximum.accumulate(np.where(bought, day, -1))]
    active_count, buy_count = (np.r_[0, np.cumsum(values)] for values in (active, bought))
    recent_days, delta_days, gap_days = windows
    lower30, lower60 = np.maximum(0, clock - recent_days), np.maximum(0, clock - delta_days)
    result = np.full((n + 1, len(EXPANDED_FEATURES)), np.nan)

    def put(name, values):
        result[:, EXPANDED_FEATURES.index(name)] = values

    put("visit_recency_days", np.where(last_active >= 0, clock - last_active, np.nan))
    put("purchase_recency_days", np.where(last_buy >= 0, clock - last_buy, np.nan))
    put("observed_tenure_days", np.where(clock > 0, clock, np.nan))
    put("session_count_30d", active_count - active_count[lower30])
    recent_buy = buy_count - buy_count[lower30]
    put("purchase_count_30d", recent_buy)
    put("orders_delta_30d", recent_buy - (buy_count[lower30] - buy_count[lower60]))
    gap_mean = np.full(n + 1, np.nan)
    purchase_dates = day[bought]
    if len(purchase_dates):
        known = buy_count >= 2
        gap_mean[known] = (last_buy[known] - purchase_dates[0]) / (buy_count[known] - 1)
    put("mean_purchase_gap_days", gap_mean)
    active_dates = day[active]
    later, gaps = active_dates[1:], np.diff(active_dates).astype(float)
    left, right = np.searchsorted(later, clock - gap_days), np.searchsorted(later, clock)
    count = right - left
    sums, squares = np.r_[0., np.cumsum(gaps)], np.r_[0., np.cumsum(gaps ** 2)]
    total, total2 = sums[right] - sums[left], squares[right] - squares[left]
    mean, cv = np.full(n + 1, np.nan), np.full(n + 1, np.nan)
    known = count > 0
    mean[known] = total[known] / count[known]
    known = count >= 2
    variance = np.maximum(0., (total2[known] - total[known] ** 2 / count[known]) / (count[known] - 1))
    cv[known] = np.sqrt(variance) / mean[known]
    put("intervisit_start_gap_mean_days_90d", mean)
    put("intervisit_start_gap_cv_90d", cv)
    return result


def _state_vectors(log, windows=(30, 60, 90)):
    before, after = _vectors(log)
    public = np.nan_to_num(_public_vectors(log, windows), nan=0.)
    return np.column_stack([public[:-1], before]), np.column_stack([public[1:], after])


def build_daily_panel(rows, source_contract, *, observation_contract=None, seed=42, user_ids=None, through_at=None):
    """Build date likelihood rows with strict outcome availability and unchanged user splits.

    A customer's first observed date conditions entry and is history only. Future
    outcome availability is next date; fitting requires it strictly before cutoff.
    The capture assumption is conditional on the supplied corpus, not service-wide
    complete observation. Session proxies in the prepared cache are never read.
    """
    start, end = _coverage(source_contract, observation_contract)
    through = end.floor("D") if through_at is None else _day(through_at)
    if not start < through <= end.floor("D"):
        raise ValueError("daily panel endpoint must lie inside declared capture")
    windows = _feature_windows(source_contract)
    cutoff = _day(source_contract["outer_cutoff_at"])
    inner = _day(source_contract["inner_cutoff_at"])
    if not start < inner < cutoff < end:
        raise ValueError("daily source split boundaries must lie inside capture")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("a nonnegative integer source split seed required")
    if user_ids is not None:
        selected = pd.Series(list(user_ids), dtype=object)
        _user_ids(selected)
        if selected.duplicated().any() or not set(selected) <= set(rows.user_id):
            raise ValueError("daily cohort must name distinct existing source users")
        rows = rows.loc[rows.user_id.isin(selected)]
    records, features, next_features = [], [], []
    conditioned = 0
    for log in _logs(rows, source_contract, observation_contract, through):
        conditioned += 1
        if len(log) == 1:
            continue
        before, after = _state_vectors(log, windows)
        part = log.iloc[1:].copy()  # The first observed activity is left-censored entry, not a sampled arrival.
        uid = part.user_id.iloc[0]
        part["at"] = part.day_start
        part["label_available_at"] = part.day_end
        part["action"] = part.outcome.map({name: index for index, name in enumerate(OUTCOMES)})
        active = (log.view | log.purchase).astype(int).to_numpy()
        part["past_active_days"] = np.cumsum(active)[:-1]
        part["next_past_active_days"] = np.cumsum(active)[1:]
        part["train_user"] = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
        part["inner_user"] = hashlib.sha256(f"{seed}:inner:{uid}".encode()).digest()[0] < 205
        part["train_eligible"] = part.train_user & part.label_available_at.lt(cutoff)
        records.append(part)
        features.append(sparse.csr_matrix(before[1:]))
        next_features.append(sparse.csr_matrix(after[1:]))
    if not records:
        raise ValueError("no fully observed post-entry daily likelihood rows")
    return {"records": pd.concat(records, ignore_index=True), "features": sparse.vstack(features, format="csr"),
            "next_features": sparse.vstack(next_features, format="csr"), "source_contract": dict(source_contract),
            "observation_contract": dict(observation_contract), "seed": int(seed),
            "audit": {"conditioned_first_observed_dates": conditioned, "pre_entry_inactivity_rows": 0,
                      "partial_last_day_excluded": end != end.floor("D"), "source_time_unit": "date",
                      "panel_through_at": through.isoformat(), "feature_windows_days": list(windows),
                      "likelihood_unit": "one fully captured user-date including no activity"}}


def load_daily_panel(cache_path, *, cutoff_at, observation_contract, inner_cutoff_at=None, seed=42, user_ids=None, through_at=None):
    rows, descriptor = load_prepared_source(cache_path, domain="tianchi", cutoff_at=cutoff_at,
                                           inner_cutoff_at=inner_cutoff_at, require_second_clock=False)
    panel = build_daily_panel(rows, descriptor["source_contract"], observation_contract=observation_contract,
                              seed=seed, user_ids=user_ids, through_at=through_at)
    panel["source_provenance"] = descriptor["source_provenance"]
    return panel


def _prefix(history, as_of_at):
    at = _day(as_of_at)
    if not {"user_id", "day_start", "view", "purchase"} <= set(history):
        raise ValueError("daily history requires user/date and boolean presence")
    _user_ids(history.user_id)
    if (not isinstance(history.day_start.dtype, pd.DatetimeTZDtype) or history.day_start.isna().any()
            or not history.day_start.eq(history.day_start.dt.floor("D")).all()
            or any(not pd.api.types.is_bool_dtype(history[name]) or history[name].isna().any() for name in ("view", "purchase"))):
        raise ValueError("daily history requires known date boundaries and boolean presence")
    past = history.loc[history.day_start.lt(at)].sort_values("day_start", kind="stable")
    if (past.empty or past.user_id.nunique() != 1 or past.day_start.duplicated().any()
            or not bool(past.iloc[0].view or past.iloc[0].purchase)
            or not pd.DatetimeIndex(past.day_start).equals(pd.date_range(past.day_start.iloc[0], at, freq="D", inclusive="left"))):
        raise ValueError("a contiguous known prefix from an observed active date is required; empty/unknown history is not zero")
    if "day_end" in past and not past.day_end.eq(past.day_start + DAY).all():
        raise ValueError("daily history completion boundaries changed")
    return past, at


def core_vector(history, as_of_at):
    past, _ = _prefix(history, as_of_at)
    return _vectors(past)[1][-1]


def public_observation(history, as_of_at, *, source_contract=None):
    """Return exactly the existing39 names; unknown monetary/SKU/session details stay null."""
    past, at = _prefix(history, as_of_at)
    values = _public_vectors(past, _feature_windows(source_contract or {}))[-1]
    return {name: None if np.isnan(value) else float(value) for name, value in zip(EXPANDED_FEATURES, values)}


def state_vector(history, as_of_at, *, source_contract=None):
    """The same existing39+40 state used by panel construction and runtime."""
    past, _ = _prefix(history, as_of_at)
    return _state_vectors(past, _feature_windows(source_contract or {}))[1][-1]


def _supported_values(x):
    if x.shape[1] == len(STATE_INDEX) and not sparse.issparse(x):
        return x
    if x.shape[1] != len(STATE_NAMES):
        raise ValueError("daily panel state width differs from existing39+40")
    unsupported = [i for i in range(len(STATE_NAMES)) if i not in STATE_INDEX]
    if sparse.issparse(x):
        if not np.isfinite(x.data).all() or x[:, unsupported].count_nonzero():
            raise ValueError("daily state contains unobserved or nonfinite coordinates")
        return x[:, STATE_INDEX].toarray()
    if not np.isfinite(x).all() or x[:, unsupported].any():
        raise ValueError("daily state contains unobserved or nonfinite coordinates")
    return x[:, STATE_INDEX]


def _state_indices(x, prototypes, center, scale, *, phases=None, state_phases=None):
    x = (_supported_values(x) - center) / scale
    p = (prototypes[:, STATE_INDEX] - center) / scale
    result = np.empty(len(x), dtype=int)
    def assign(selected, candidates):
        for offset in range(0, len(selected), 2048):
            indices = selected[offset:offset + 2048]
            distances = np.square(p[candidates][None, :, :] - x[indices, None, :]).sum(axis=2)
            result[indices] = candidates[distances.argmin(axis=1)]
    if phases is None:
        assign(np.arange(len(x)), np.arange(len(p)))
        return result
    phases, state_phases = np.asarray(phases), np.asarray(state_phases)
    if phases.shape != (len(x),) or not np.isin(phases, [0, 1]).all():
        raise ValueError("private daily lifecycle phases must align with strict-past states")
    for phase in (0, 1):
        candidates = np.flatnonzero(state_phases == phase)
        if not len(candidates):
            raise ValueError("daily private actor phase has no training state support")
        selected = np.flatnonzero(phases == phase)
        assign(selected, candidates)
    return result


def _row_phases(model, records, *, after=False):
    if model.get("lifecycle_memory") is None:
        return None
    name = "next_past_active_days" if after else "past_active_days"
    return records[name].ge(2).to_numpy(dtype=int)


def _training_users(records, panel, cutoff):
    seed = panel["seed"]
    selected = records.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
    if cutoff == _day(panel["source_contract"]["inner_cutoff_at"]):
        selected &= records.user_id.map(lambda uid: hashlib.sha256(f"{seed}:inner:{uid}".encode()).digest()[0] < 205)
    return selected


def fit_daily_model(panel, *, cutoff_at=None, assignments=None, persona_cluster=None,
                    persona_usage="training_prefix", lifecycle_memory=None, max_states=32,
                    regularization=.001, max_iter=1000, discount=.95, partition_max_rows=None):
    """Fit linear rewards and one-date transitions using only mature training users.

    Prefix groups may condition retrospective training likelihoods descriptively.
    With persona_usage='forecast', targets must follow assignment availability.
    Daily outcomes never inherit the retail semantics of their numeric slots.
    """
    cutoff = _day(cutoff_at or panel["source_contract"]["outer_cutoff_at"])
    _coverage(panel["source_contract"], panel["observation_contract"])
    if cutoff not in {_day(panel["source_contract"][name]) for name in ("inner_cutoff_at", "outer_cutoff_at")}:
        raise ValueError("daily fitting cutoff must be a declared inner/outer source cutoff")
    records = panel["records"]
    expected_train = records.user_id.map(lambda uid: hashlib.sha256(f"{panel['seed']}:{uid}".encode()).digest()[0] < 205)
    expected_inner = records.user_id.map(lambda uid: hashlib.sha256(f"{panel['seed']}:inner:{uid}".encode()).digest()[0] < 205)
    if (not records.train_user.eq(expected_train).all()
            or not records.inner_user.eq(expected_inner).all()
            or not records.label_available_at.eq(records["at"] + DAY).all()
            or not records.action.isin(range(4)).all()
            or panel["features"].shape[0] != len(records) or panel["next_features"].shape[0] != len(records)):
        raise ValueError("daily source user split, label availability or aligned outcome rows changed")
    selected = _training_users(records, panel, cutoff) & records.label_available_at.lt(cutoff)
    persona = None
    if persona_usage not in {"training_prefix", "forecast"}:
        raise ValueError("persona usage must distinguish training-prefix strata and forecast availability")
    if assignments is not None:
        from .source_personas import attach_frozen_assignments
        persona = assignments.attrs.get("frozen_persona_contract")
        if (not persona or persona["domain"] != "tianchi"
                or persona["prepared_source_sha256"] != panel["source_contract"]["prepared_sha256"]):
            raise ValueError("frozen daily-prefix personas must bind this Tianchi source")
        if utc_cutoff(persona["cutoff_at"]) > cutoff:
            raise ValueError("persona prefix cannot include outcomes after the daily fitting cutoff")
        if persona_usage == "forecast":
            selected &= records["at"].ge(utc_cutoff(persona["cutoff_at"]))
        attached = attach_frozen_assignments(records.loc[selected], assignments, forecast=persona_usage == "forecast")
        if persona_cluster is not None:
            selected.loc[attached.index] &= attached.persona_cluster.astype(str).eq(str(persona_cluster))
    elif persona_cluster is not None:
        raise ValueError("persona fitting requires bound available daily-prefix assignments")
    if not selected.any():
        raise ValueError("no mature training daily likelihood rows")
    if isinstance(max_states, bool) or not isinstance(max_states, Integral) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if partition_max_rows is not None and (isinstance(partition_max_rows, bool)
            or not isinstance(partition_max_rows, Integral) or partition_max_rows < max_states):
        raise ValueError("partition row budget must be an integer at least max_states")
    if lifecycle_memory not in (None, LIFECYCLE_MEMORY):
        raise ValueError("daily lifecycle memory must be first_two_observed_active_days or None")
    if (not pd.api.types.is_integer_dtype(records.past_active_days) or records.past_active_days.lt(1).any()
            or not records.next_past_active_days.eq(records.past_active_days + records.action.ne(0).astype(int)).all()):
        raise ValueError("daily actor phase requires lifetime active counts from the pre-label prefix")
    x = _supported_values(panel["features"][selected.to_numpy()])
    after = _supported_values(panel["next_features"][selected.to_numpy()])
    if after.shape != x.shape:
        raise ValueError("daily before/after states must align")
    train = records.loc[selected]
    counts_per_user = train.user_id.value_counts()
    weights = train.user_id.map(1 / counts_per_user).to_numpy(dtype=float)
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average(np.square(x - center), axis=0, weights=weights))
    scale[scale == 0] = 1
    phases = train.past_active_days.ge(2).to_numpy(dtype=int) if lifecycle_memory else None
    next_phases = train.next_past_active_days.ge(2).to_numpy(dtype=int) if lifecycle_memory else None
    groups = (0, 1) if lifecycle_memory else (0,)
    representative_parts, phase_parts, partition_rows = [], [], []
    for phase in groups:
        selected_phase = np.ones(len(x), dtype=bool) if phases is None else phases == phase
        if not selected_phase.any():
            raise ValueError("daily lifecycle fitting requires mature NEW and HOME training support")
        partition_indices = np.flatnonzero(selected_phase)
        if partition_max_rows is not None and len(partition_indices) > partition_max_rows:
            partition_indices = np.sort(np.random.default_rng(panel["seed"] + phase).choice(
                partition_indices, size=partition_max_rows, replace=False))
        partition_rows.append(len(partition_indices))
        values = x[partition_indices]
        unique = np.unique(values, axis=0)
        if len(unique) <= max_states:
            part = unique
        else:
            scaled = (values - center) / scale
            clustering = KMeans(n_clusters=int(max_states), n_init=10, random_state=panel["seed"]).fit(
                scaled, sample_weight=weights[partition_indices])
            medoids = [np.square(scaled - point).sum(axis=1).argmin() for point in clustering.cluster_centers_]
            part = np.unique(values[medoids], axis=0)
        representative_parts.append(part)
        phase_parts.extend([phase] * len(part))
    representatives = np.vstack(representative_parts)
    state_phases = np.asarray(phase_parts, dtype=int)
    prototypes = np.zeros((len(representatives), len(STATE_NAMES)))
    prototypes[:, STATE_INDEX] = representatives
    states = _state_indices(x, prototypes, center, scale, phases=phases, state_phases=state_phases)
    targets = _state_indices(after, prototypes, center, scale, phases=next_phases, state_phases=state_phases)
    action = train.action.to_numpy(dtype=int)
    n = len(prototypes)
    state_means = np.zeros_like(prototypes)
    state_means[:, STATE_INDEX] = np.asarray([np.average(x[states == state], axis=0, weights=weights[states == state]) for state in range(n)])
    counts = np.zeros((n, 5))
    np.add.at(counts, (states, action), weights)
    pooled = np.bincount(targets, weights=weights, minlength=n)
    matrices, backoffs = [], []
    for a in range(4):
        observed = action == a
        matrix = sparse.coo_matrix((weights[observed], (states[observed], targets[observed])), shape=(n, n)).toarray()
        mass = matrix.sum(axis=1)
        fallback = matrix.sum(axis=0)
        if not fallback.sum():
            fallback = pooled
        for state in np.flatnonzero(mass == 0):
            allowed = np.ones(n, dtype=bool) if not lifecycle_memory else state_phases == (0 if state_phases[state] == 0 and a == 0 else 1)
            row = fallback * allowed
            matrix[state] = row if row.sum() else allowed.astype(float)
        matrix /= matrix.sum(axis=1, keepdims=True)
        matrices.append(sparse.csr_matrix(matrix))
        backoffs.append(int((mass == 0).sum()))
    matrices.append(sparse.csr_matrix((n, n)))
    mask = np.column_stack([np.ones((n, 4), dtype=bool), np.zeros(n, dtype=bool)])
    basis = np.column_stack([np.ones(n), np.zeros((n, len(STATE_NAMES)))])
    basis[:, np.asarray(STATE_INDEX) + 1] = (state_means[:, STATE_INDEX] - center) / scale
    names = list(REWARD_FEATURE_NAMES)
    fit_basis = basis if not lifecycle_memory else np.column_stack([basis * (state_phases == phase)[:, None] for phase in (0, 1)])
    fit_names = names if not lifecycle_memory else [f"{phase}::{name}" for phase in ("NEW", "HOME") for name in names]
    fitted = feature_reward.fit_feature_reward(fit_basis, counts, matrices, mask,
        feature_names=fit_names, regularization=regularization, max_iter=max_iter, discount=discount)
    if lifecycle_memory:
        fitted["reward_weights"] = np.asarray(fitted["reward_weights"]).reshape(4, 2, len(names)).transpose(1, 0, 2).tolist()
        fitted["feature_names"] = names
    fitted["fit"].update(reward_features_per_actor=len(names), private_actor_banks=2 if lifecycle_memory else 1)
    return {"schema_version": 1, "type": TYPE, "backend": "tianchi_daily", "outcomes": list(OUTCOMES), "clock_resolution": "day",
            "feature_windows_days": list(_feature_windows(panel["source_contract"])),
            "day_transition_semantics": "one fully observed date advances regardless of activity; no END decision",
            "core_feature_names": list(CORE_NAMES), "core_supported": list(CORE_SUPPORTED),
            "state_feature_names": list(STATE_NAMES),
            "state_supported": [STATE_NAMES[index] for index in STATE_INDEX],
            "missing_value_rule": "Unsupported columns are zero; undefined supported nullable values use zero imputation before train-only scaling; no new missing flags.",
            "public_transform": "reported39 values, zero imputation, source-training standardization; no extra predictors",
            "public_feature_names": list(EXPANDED_FEATURES),
            "public_feature_roles": {name: PUBLIC_ROLES.get(name, "unobserved") for name in EXPANDED_FEATURES},
            "state_vectors": prototypes.tolist(), "scaler_center": center.tolist(), "scaler_scale": scale.tolist(),
            "lifecycle_memory": lifecycle_memory, "state_phases": state_phases.tolist(),
            "phase_names": ["NEW", "HOME"] if lifecycle_memory else ["COMMON"],
            "phase_rule": "private key before the target date: lifetime observed active dates<2 NEW,otherwise HOME; not a predictor",
            "transition_fallback": "training pooled destinations restricted to the correct next phase; uniform training support if absent",
            "state_feature_means": state_means.tolist(),
            "state_basis": "frozen training customer-weighted39+40 means within nearest-prototype partitions",
            "transition_csr": [{"shape": list(m.shape), "data": m.data.tolist(), "indices": m.indices.tolist(), "indptr": m.indptr.tolist()} for m in matrices],
            "mask": mask.tolist(), **fitted,
            "trait_names": list(TRAIT_NAMES), "active_trait_coordinates": ["view", "purchase", "return"],
            "canonical_parameter_names": list(CANONICAL_PARAMETER_NAMES), "active_parameter_indices": [0, 3, 4],
            "parameter_units": ["daily_presence_reward", None, None, "daily_presence_reward", "daily_activity_reward"],
            "return_horizon_days": 1, "return_backend": "joint_daily_outcomes", "coupon_response_backend": None,
            "trait_incidence": TRAIT_INCIDENCE.tolist(), "return_trait_unit": "daily activity reward offset, not a30d return logit",
            "source_contract": {**panel["source_contract"], "training_cutoff_at": cutoff.isoformat()},
            "observation_contract": dict(panel["observation_contract"]),
            "training": {"cutoff_at": cutoff.isoformat(), "max_label_available_at": train.label_available_at.max().isoformat(),
                "min_decision_at": train["at"].min().isoformat(),
                "user_rule": f"sha256({panel['seed']}:raw_user_id).digest()[0] <205", "users": int(train.user_id.nunique()),
                "inner_user_hash_required": cutoff == _day(panel["source_contract"]["inner_cutoff_at"]),
                "daily_rows": len(train), "weighting": "equal customer mass; equal dates within each customer",
                "partition_max_rows_per_phase": partition_max_rows, "partition_rows_by_phase": partition_rows,
                "partition_sampling": "fixed-seed training rows only; every training user-date remains in likelihood and transitions",
                "transition_backoff_rows_by_outcome": backoffs, "persona": persona,
                "persona_usage": persona_usage,
                "persona_cluster": None if persona_cluster is None else str(persona_cluster)},
            "limitations": ["No-activity means absence in the explicitly assumed complete supplied corpus, not guaranteed service-wide inactivity.",
                "First observed date is conditioned entry, not observed signup or an unconditional cold-start distribution.",
                "The state abstraction and unsupported state-action dynamics use training-only nearest prototypes and pooled transition backoff.",
                "Daily co-occurrence identifies no within-day ordering, session duration, cart, SKU, price, quantity or END.",
                "Retrospective likelihood conditional on training-prefix personas is descriptive group fitting, not a forecast score.",
                "Daily source associations and pooled dynamics do not identify causal coupon or activity effects.",
                "Temporal extrapolation and generated104-week histories are synthesis assumptions, not104-week source validation."]}


def fit_daily_group_model(state_model, panel, assignments, persona_cluster, *, regularization=.01, max_iter=1000):
    """Fit persona coefficients on one frozen global state/dynamics/phase basis.

    Group likelihood conditions descriptively on a training-prefix label. An
    unsupported group-state coefficient shrinks to the global linear actor;
    absent evidence is not a fabricated zero individual trait.
    """
    _bound_panel(state_model, panel)
    persona = assignments.attrs.get("frozen_persona_contract")
    cutoff = _day(state_model["training"]["cutoff_at"])
    if (not persona or persona["domain"] != "tianchi"
            or persona["prepared_source_sha256"] != state_model["source_contract"]["prepared_sha256"]
            or utc_cutoff(persona["cutoff_at"]) > cutoff):
        raise ValueError("daily group fitting requires bound available training-prefix assignments")
    group_model = deepcopy(state_model)
    group_model["training"].update(persona=persona, persona_cluster=str(persona_cluster), persona_usage="training_prefix")
    records = panel["records"]
    selected = _training_users(records, panel, cutoff) & records.label_available_at.lt(cutoff)
    from .source_personas import attach_frozen_assignments
    attached = attach_frozen_assignments(records.loc[selected], assignments, forecast=False)
    selected.loc[attached.index] &= attached.persona_cluster.astype(str).eq(str(persona_cluster))
    if not selected.any():
        raise ValueError("daily persona group has no mature training observations")
    actions = daily_action_records(group_model, panel, selected.to_numpy(), assignments=assignments)
    _, _, _, matrices, basis, _ = _model_arrays(state_model)
    counts = np.zeros((len(basis), 5))
    per_user = actions.user_id.value_counts()
    np.add.at(counts, (actions.state.to_numpy(), actions.action.to_numpy()), actions.user_id.map(1 / per_user).to_numpy(float))
    memory = state_model.get("lifecycle_memory")
    phases = np.asarray(state_model["state_phases"])
    fit_basis = basis if not memory else np.column_stack([basis * (phases == phase)[:, None] for phase in (0, 1)])
    names = list(REWARD_FEATURE_NAMES)
    fit_names = names if not memory else [f"{phase}::{name}" for phase in ("NEW", "HOME") for name in names]
    prior = np.asarray(state_model["reward_weights"])
    if memory:
        prior = prior.transpose(1, 0, 2).reshape(4, -1)
    fitted = feature_reward.fit_feature_reward(fit_basis, counts, matrices, np.asarray(state_model["mask"], dtype=bool),
        feature_names=fit_names, discount=state_model["discount"], regularization=regularization,
        max_iter=max_iter, initial=prior, prior=prior)
    if memory:
        fitted["reward_weights"] = np.asarray(fitted["reward_weights"]).reshape(4, 2, 80).transpose(1, 0, 2).tolist()
    fitted["feature_names"] = names
    fitted["fit"].update(reward_features_per_actor=80, private_actor_banks=2 if memory else 1,
                         frozen_global_state_and_dynamics=True)
    group_model.update(fitted)
    train = records.loc[selected]
    group_model["training"].update(users=int(train.user_id.nunique()), daily_rows=len(train),
        min_decision_at=train["at"].min().isoformat(), max_label_available_at=train.label_available_at.max().isoformat(),
        frozen_state_model_sha256=hashlib.sha256(json.dumps(state_model, sort_keys=True, allow_nan=False).encode()).hexdigest())
    return group_model


def _model_arrays(model):
    if model.get("feature_windows_days", list(_feature_windows(model["source_contract"]))) != list(_feature_windows(model["source_contract"])):
        raise ValueError("daily model feature windows differ from its source contract")
    if (model.get("schema_version") != 1 or model.get("type") != TYPE or model.get("backend") != "tianchi_daily"
            or model.get("outcomes") != list(OUTCOMES)
            or model.get("core_feature_names") != list(CORE_NAMES)
            or model.get("state_feature_names") != list(STATE_NAMES)):
        raise ValueError("incompatible Tianchi daily model")
    if (model.get("return_backend") != "joint_daily_outcomes" or model.get("coupon_response_backend") is not None
            or model.get("canonical_parameter_names") != list(CANONICAL_PARAMETER_NAMES)
            or model.get("active_parameter_indices") != [0, 3, 4] or model.get("return_horizon_days") != 1
            or model.get("parameter_units") != ["daily_presence_reward", None, None, "daily_presence_reward", "daily_activity_reward"]
            or any(model.get(name) is not None for name in ("retail_policy_logs", "return_model_path", "return_model",
                                                           "coupon_response_model_path", "coupon_response_model"))):
        raise ValueError("daily backend cannot reuse retail policy_logs,30d return GLM or coupon-response models")
    prototypes = np.asarray(model["state_vectors"], dtype=float)
    center, scale = (np.asarray(model[name], dtype=float) for name in ("scaler_center", "scaler_scale"))
    weights = np.asarray(model["reward_weights"], dtype=float)
    means = np.asarray(model["state_feature_means"], dtype=float)
    unsupported = [i for i in range(len(STATE_NAMES)) if i not in STATE_INDEX]
    memory = model.get("lifecycle_memory")
    if memory not in (None, LIFECYCLE_MEMORY):
        raise ValueError("incompatible daily lifecycle memory")
    phases = np.asarray(model["state_phases"])
    expected_weights = (4, len(STATE_NAMES) + 1) if memory is None else (2, 4, len(STATE_NAMES) + 1)
    if (prototypes.ndim != 2 or prototypes.shape[1] != len(STATE_NAMES) or not len(prototypes)
            or center.shape != (len(STATE_INDEX),) or scale.shape != (len(STATE_INDEX),) or (scale <= 0).any()
            or weights.shape != expected_weights
            or phases.shape != (len(prototypes),) or not np.isin(phases, [0, 1] if memory else [0]).all()
            or memory is not None and set(phases) != {0, 1}
            or means.shape != prototypes.shape
            or not all(np.isfinite(v).all() for v in (prototypes, center, scale, weights, means))
            or prototypes[:, unsupported].any() or means[:, unsupported].any() or weights[..., np.asarray(unsupported) + 1].any()
            or model.get("core_supported") != list(CORE_SUPPORTED)
            or not np.isfinite(model["discount"]) or not 0 <= model["discount"] < 1):
        raise ValueError("invalid daily frozen state basis or reward coefficients")
    matrices = [sparse.csr_matrix((b["data"], b["indices"], b["indptr"]), shape=b["shape"]) for b in model["transition_csr"]]
    expected_mask = np.column_stack([np.ones((len(prototypes), 4), dtype=bool), np.zeros(len(prototypes), dtype=bool)])
    if len(matrices) != 5 or not np.array_equal(model["mask"], expected_mask):
        raise ValueError("daily model must support four continuing outcomes and no END")
    for action, matrix in enumerate(matrices):
        if (matrix.shape != (len(prototypes), len(prototypes)) or not np.isfinite(matrix.data).all()
                or (matrix.data < 0).any() or (matrix.nnz if action == 4 else
                    not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1, atol=1e-10, rtol=0))):
            raise ValueError("daily continuation matrices must be stochastic; END continuation must be zero")
        if memory and action < 4:
            coo = matrix.tocoo()
            destination = np.where((phases == 0) & (action == 0), 0, 1)
            if np.any(phases[coo.col] != destination[coo.row]):
                raise ValueError("daily transitions cannot reset HOME or peek at current-day activity for NEW")
    basis = np.column_stack([np.ones(len(prototypes)), np.zeros_like(prototypes)])
    basis[:, np.asarray(STATE_INDEX) + 1] = (means[:, STATE_INDEX] - center) / scale
    reward = basis @ weights.T if not memory else np.einsum("sf,saf->sa", basis, weights[phases])
    return prototypes, center, scale, matrices, basis, reward


def policy_for_trait(model, trait=None):
    """Canonical5 coefficients act through daily incidence; cart/remove are unsupported."""
    _, _, _, matrices, basis, base = _model_arrays(model)
    trait = np.zeros(5) if trait is None else np.asarray(trait, dtype=float)
    if trait.shape != (5,) or not np.isfinite(trait).all() or trait[[1, 2]].any():
        raise ValueError("daily private traits support only view/purchase/daily-return coordinates")
    reward = np.column_stack([base + TRAIT_INCIDENCE @ trait, np.zeros(len(basis))])
    probability, _, _ = shopping.soft_policy(reward, matrices, np.asarray(model["mask"], dtype=bool), model["discount"])
    return probability[:, :4]


def daily_policy_log_provider(models):
    """Prepare the daily-only QMC callback: (actor_index,theta5_batch)->samples×(states*5)."""
    if not models:
        raise ValueError("at least one daily actor is required")
    prepared, reference = {}, None
    for index, model in models.items():
        prototypes, center, scale, matrices, _, base = _model_arrays(model)
        shared = {name: model[name] for name in ("source_contract", "state_vectors", "scaler_center", "scaler_scale",
                                                "state_feature_means", "state_phases", "transition_csr", "lifecycle_memory", "discount")}
        if reference is None:
            reference = shared
        elif shared != reference:
            raise ValueError("daily group callbacks require the same frozen global state space and dynamics")
        mask = np.asarray(model["mask"], dtype=bool)
        _, value, _ = shopping.soft_policy(np.column_stack([base, np.zeros(len(base))]), matrices, mask, model["discount"])
        solve = None
        if model["discount"] == shopping.DISCOUNT:
            from .reward_gmm import prepare_bellman
            # Numerical Bellman solver only: offset slots below are daily incidence, not retail action labels.
            solve = prepare_bellman({"common_reward": base, "action_mask": mask,
                                     "component_offsets": np.zeros((1, 4)), "component_values": value[None, :]}, matrices)
        prepared[index] = model, base, matrices, mask, solve

    def provider(actor_index, theta5_batch):
        if actor_index not in prepared:
            raise ValueError("unknown frozen daily actor index")
        model, base, matrices, mask, solve = prepared[actor_index]
        samples = np.asarray(theta5_batch, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 5 or not np.isfinite(samples).all() or samples[:, [1, 2]].any():
            raise ValueError("daily QMC samples can use only view/purchase/daily-return coordinates")
        logs = np.zeros((len(samples), len(base), 5))
        for row, theta in enumerate(samples):
            offset = TRAIT_INCIDENCE @ theta
            if solve is None:
                probability = shopping.soft_policy(np.column_stack([base + offset, np.zeros(len(base))]),
                                                   matrices, mask, model["discount"])[0]
            else:
                probability = solve(offset)
            logs[row, :, :4] = np.log(np.maximum(probability[:, :4], np.finfo(float).tiny))
        return logs.reshape(len(samples), -1)

    return provider


def _bound_panel(model, panel):
    _coverage(panel["source_contract"], panel["observation_contract"])
    _coverage(model["source_contract"], model["observation_contract"])
    keys = ("domain", "clock_resolution", "prepared_sha256", "observation_start_at", "observation_end_at",
            "inner_cutoff_at", "outer_cutoff_at", "feature_windows_days")
    if any(model["source_contract"].get(key) != panel["source_contract"].get(key) for key in keys):
        raise ValueError("daily likelihood/trait panel differs from its frozen source binding")


def _bound_group(model, records, assignments, *, forecast):
    persona = model["training"].get("persona")
    if persona:
        from .source_personas import attach_frozen_assignments
        if assignments is None or assignments.attrs.get("frozen_persona_contract") != persona:
            raise ValueError("persona-conditioned daily models require their same SHA-bound frozen assignments")
        attached = attach_frozen_assignments(records, assignments, forecast=forecast)
        group = model["training"].get("persona_cluster")
        if group is not None and not attached.persona_cluster.astype(str).eq(group).all():
            raise ValueError("daily scoring/trait users do not belong to the model's frozen persona group")


def daily_action_records(model, panel, mask, *, assignments=None, retrospective_training=True):
    """Freeze user/state/date counts once for a daily-backend persona-GMM callback.

    All four joint outcomes, including no activity, are policy observations. No
    extra30-day activity/GLM likelihood should be added to the same observations.
    """
    _bound_panel(model, panel)
    selected = np.asarray(mask)
    if selected.shape != (len(panel["records"]),) or selected.dtype.kind != "b":
        raise ValueError("daily action-record mask must align with the panel")
    records = panel["records"].loc[selected]
    if retrospective_training:
        cutoff = _day(model["training"]["cutoff_at"])
        if not _training_users(records, panel, cutoff).all() or not records.label_available_at.lt(cutoff).all():
            raise ValueError("daily training action records cannot include heldout or unavailable outcomes")
    _bound_group(model, records, assignments, forecast=not retrospective_training)
    prototypes, center, scale, _, _, _ = _model_arrays(model)
    state = _state_indices(panel["features"][selected], prototypes, center, scale,
                           phases=_row_phases(model, records), state_phases=model["state_phases"])
    result = records[["user_id", "at", "action", "label_available_at"]].copy().rename(columns={"label_available_at": "known_at"})
    result["state"], result["count"] = state, 1
    result.attrs = {"backend": "tianchi_daily", "outcomes": list(OUTCOMES),
                    "prepared_source_sha256": model["source_contract"]["prepared_sha256"],
                    "return_horizon_days": 1, "active_parameter_indices": [0, 3, 4],
                    "observation_unit": "joint fully observed supplied-corpus user-date"}
    return result.reset_index(drop=True)


def daily_log_likelihood(model, panel, mask=None, *, trait=None, assignments=None, retrospective_training=False):
    """One joint date likelihood including covered no-activity dates; no step/run proxy."""
    prototypes, center, scale, _, _, _ = _model_arrays(model)
    _bound_panel(model, panel)
    selected = np.ones(len(panel["records"]), dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != (len(panel["records"]),) or selected.dtype.kind != "b":
        raise ValueError("daily likelihood mask must align with panel records")
    records = panel["records"].loc[selected]
    if retrospective_training:
        cutoff = _day(model["training"]["cutoff_at"])
        if not _training_users(records, panel, cutoff).all() or not records.label_available_at.lt(cutoff).all():
            raise ValueError("retrospective training diagnostic cannot score heldout or unavailable outcomes")
    _bound_group(model, records, assignments, forecast=not retrospective_training)
    x = panel["features"][selected]
    action = panel["records"].loc[selected, "action"].to_numpy(dtype=int)
    state = _state_indices(x, prototypes, center, scale, phases=_row_phases(model, records), state_phases=model["state_phases"])
    probability = policy_for_trait(model, trait)[state, action]
    log_probability = np.log(probability)
    return {"daily_rows": len(action), "log_likelihood": float(log_probability.sum()),
            "mean_nll": float(-log_probability.mean()) if len(action) else None,
            "unit": "joint fully observed date outcome including no activity",
            "interpretation": "descriptive training likelihood, not forecast" if retrospective_training else
                "joint date likelihood; source/group availability checked and caller mask determines data split"}


def fit_daily_trait(model, panel, mask, *, assignments=None, regularization=.1, max_iter=250):
    """Fit only canonical(view,purchase,return) offsets through daily Bellman likelihood.

    This is a retrospective training trait estimate for later active-coordinate
    GMM fitting. It cannot use heldout users or immature outcome dates. The return
    coordinate is a daily activity reward, never a monthly return-logit estimate.
    """
    _bound_panel(model, panel)
    selected = np.asarray(mask)
    records = panel["records"]
    if selected.shape != (len(records),) or selected.dtype.kind != "b" or not selected.any():
        raise ValueError("nonempty aligned daily training mask required")
    train = records.loc[selected]
    cutoff = _day(model["training"]["cutoff_at"])
    expected = _training_users(train, panel, cutoff)
    if not expected.all() or not train.label_available_at.lt(cutoff).all():
        raise ValueError("daily trait fitting requires mature training-user outcomes")
    _bound_group(model, train, assignments, forecast=False)
    if not np.isfinite(regularization) or regularization < 0 or isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("finite nonnegative trait penalty and positive iteration budget required")
    prototypes, center, scale, matrices, basis, base = _model_arrays(model)
    state = _state_indices(panel["features"][selected], prototypes, center, scale,
                           phases=_row_phases(model, train), state_phases=model["state_phases"])
    counts = np.zeros((len(prototypes), 5))
    count_user = train.user_id.value_counts()
    weights = train.user_id.map(1 / count_user).to_numpy(float)
    np.add.at(counts, (state, train.action.to_numpy(int)), weights)
    total = counts.sum()
    incidence = TRAIT_INCIDENCE[:, [0, 3, 4]]

    def objective(value):
        reward = np.column_stack([base + incidence @ value, np.zeros(len(base))])
        ll, gradient, _ = shopping.policy_likelihood_gradient(reward, counts, matrices,
                                                             np.asarray(model["mask"], dtype=bool), model["discount"])
        derivative = -(gradient[:, :4].sum(axis=0) @ incidence) / total + regularization * value
        return -ll / total + regularization * (value @ value) / 2, derivative

    initial = float(objective(np.zeros(3))[0])
    optimum = minimize(objective, np.zeros(3), method="L-BFGS-B", jac=True,
                       options={"maxiter": int(max_iter), "gtol": 1e-6, "ftol": 1e-11})
    canonical = np.zeros(5)
    canonical[[0, 3, 4]] = optimum.x
    return {"trait_names": list(TRAIT_NAMES), "canonical_theta": canonical.tolist(),
            "canonical_parameter_names": list(CANONICAL_PARAMETER_NAMES), "active_parameter_indices": [0, 3, 4],
            "parameter_units": model["parameter_units"], "return_horizon_days": 1,
            "active_coordinates": ["view", "purchase", "return"], "return_unit": model["return_trait_unit"],
            "training_users": int(train.user_id.nunique()), "daily_rows": len(train),
            "max_label_available_at": train.label_available_at.max().isoformat(),
            "fit": {"converged": bool(optimum.success), "message": str(optimum.message),
                    "iterations": int(optimum.nit), "objective_initial": initial, "objective": float(optimum.fun)},
            "interpretation": "retrospective training trait estimate; not a heldout forecast score"}


class TianchiDailyEnv:
    """Continue a known daily prefix; one sampled joint outcome advances one date."""

    def __init__(self, model, history, *, as_of_at, seed=42, trait=None):
        past, self.at = _prefix(history, as_of_at)
        self._prefix_history = past[["user_id", "day_start", "view", "purchase"]].copy()
        self._generated = []
        self._n = len(past)
        self._active = np.flatnonzero((past.view | past.purchase).to_numpy()).tolist()
        self._purchased = np.flatnonzero(past.purchase.to_numpy()).tolist()
        gaps = np.diff(self._active).astype(float)
        self._gap_sum, self._gap_squared_sum = np.r_[0., np.cumsum(gaps)].tolist(), np.r_[0., np.cumsum(gaps ** 2)].tolist()
        self.model, self.rng = model, np.random.default_rng(seed)
        self._windows = _feature_windows(model["source_contract"])
        self._prototypes, self._center, self._scale, _, _, _ = _model_arrays(model)
        self.set_trait(trait)

    def set_trait(self, trait=None):
        self._policy = policy_for_trait(self.model, trait)

    @property
    def history(self):
        if not self._generated:
            return self._prefix_history.copy()
        return pd.concat([self._prefix_history, pd.DataFrame(self._generated)[self._prefix_history.columns]], ignore_index=True)

    def _observed_values(self):
        """Cache existing history summaries; no new predictor or latent process."""
        n, active, bought = self._n, self._active, self._purchased
        recent_days, delta_days, gap_days = self._windows
        left = max(1, bisect_left(active, n - gap_days)) - 1
        gap_count = len(active) - 1 - left
        total = self._gap_sum[-1] - self._gap_sum[left]
        total2 = self._gap_squared_sum[-1] - self._gap_squared_sum[left]
        mean = total / gap_count if gap_count else None
        cv = np.sqrt(max(0., (total2 - total ** 2 / gap_count) / (gap_count - 1))) / mean if gap_count >= 2 else None
        recent_buy = len(bought) - bisect_left(bought, n - recent_days)
        previous_buy = bisect_left(bought, n - recent_days) - bisect_left(bought, n - delta_days)
        return {"visit_recency_days": n - active[-1], "purchase_recency_days": n - bought[-1] if bought else None,
                  "observed_tenure_days": n, "session_count_30d": len(active) - bisect_left(active, n - recent_days),
                  "purchase_count_30d": recent_buy, "orders_delta_30d": recent_buy - previous_buy,
                  "mean_purchase_gap_days": (bought[-1] - bought[0]) / (len(bought) - 1) if len(bought) >= 2 else None,
                  "intervisit_start_gap_mean_days_90d": mean, "intervisit_start_gap_cv_90d": cv}

    def observation(self):
        """Existing39 nullable observations without replaying/materializing all daily logs."""
        values = self._observed_values()
        return {name: values.get(name) for name in EXPANDED_FEATURES}

    def _state(self):
        n, bought = self._n, self._purchased
        vector = np.zeros(len(STATE_NAMES))
        for name, value in self._observed_values().items():
            vector[EXPANDED_FEATURES.index(name)] = 0. if value is None else value
        vector[len(EXPANDED_FEATURES) + CORE_INDEX[0]] = np.log1p(n)
        if bought:
            vector[len(EXPANDED_FEATURES) + CORE_INDEX[1]] = np.log1p(n - bought[-1])
            vector[len(EXPANDED_FEATURES) + CORE_INDEX[2]] = 1.
        return vector

    def probabilities(self):
        x = self._state()[None, :]
        phase = None if self.model.get("lifecycle_memory") is None else [int(len(self._active) >= 2)]
        state = _state_indices(x, self._prototypes, self._center, self._scale,
                               phases=phase, state_phases=self.model["state_phases"])[0]
        values = self._policy[state]
        active = float(values[1:].sum())
        return {"joint": dict(zip(OUTCOMES, map(float, values))), "activity_probability": active,
                "actor_phase": None if phase is None else ("HOME" if phase[0] else "NEW"),
                "active_combinations": dict(zip(OUTCOMES[1:], map(float, values[1:] / active))) if active else None}

    def step(self):
        values = np.asarray(list(self.probabilities()["joint"].values()))
        outcome = int(self.rng.choice(4, p=values / values.sum()))
        view, purchase = FLAGS[outcome]
        row = {"user_id": self._prefix_history.user_id.iloc[0], "day_start": self.at,
               "day_end": self.at + DAY, "view": view, "purchase": purchase, "outcome": OUTCOMES[outcome]}
        self._generated.append(row)
        if view or purchase:
            gap = self._n - self._active[-1]
            self._gap_sum.append(self._gap_sum[-1] + gap)
            self._gap_squared_sum.append(self._gap_squared_sum[-1] + gap ** 2)
            self._active.append(self._n)
        if purchase:
            self._purchased.append(self._n)
        self._n += 1
        self.at += DAY
        return row
