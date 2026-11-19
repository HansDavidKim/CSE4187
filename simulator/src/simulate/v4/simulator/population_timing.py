"""Generate visit windows and review delays from population histograms."""
from collections import Counter, defaultdict
from math import lcm

import numpy as np

from simulate.v4.irl.fit_visit_timing import HORIZON_SECONDS, _pack, validate_model as validate_visit_timing
from simulate.v4.simulator.review_events import validate_review_model


_HOUR = 3600
_DAY = 24 * _HOUR
_VISIT_SPAN_EDGES = np.asarray([0, 1, 60, 300, 900, 1800, _HOUR, 2 * _HOUR, 6 * _HOUR, 12 * _HOUR,
                               *range(_DAY, HORIZON_SECONDS + 1, _DAY)])
_SPAN_BINS = len(_VISIT_SPAN_EDGES) - 1
_GAP_EDGES = np.asarray([0., .001, 1., 10., 30., 60., 120., 300., 600., 1800., 3600.,
                         10800., 21600., 43200., 86400., 2 * _DAY, 4 * _DAY,
                         8 * _DAY, 16 * _DAY, HORIZON_SECONDS])
_GAP_SPAN_SPLITS = (300, _HOUR, _DAY)
_INTERIOR_BINS = 24
_REVIEW_BIN_SECONDS = 6 * _HOUR
_BANK_SIZE = 2048
_VISIT_FIELDS = ("schema_version", "type", "horizon_days", "cutoff_at", "fit_available_rule",
                 "regular_anchor_hour_utc", "observation_start_at", "observation_end_at", "seed")
_REVIEW_FIELDS = ("classes", "coef", "intercept", "scaler_mean", "scaler_scale", "feature_min",
                  "feature_max", "feature_names")


def _visit_histograms(rows):
    groups = defaultdict(lambda: {key: Counter() for key in ("count", "clocks", "gap")})
    for offsets, weight in rows:
        count = len(offsets)
        first = offsets[0]
        span = offsets[-1] - first
        group = groups[count]
        group["count"][count] += weight
        span_index = int(np.searchsorted(_VISIT_SPAN_EDGES, span, side="right")) - 1
        group["clocks"][int(first // _HOUR) * _SPAN_BINS + span_index] += weight
        gap_scope = int(np.searchsorted(_GAP_SPAN_SPLITS, span, side="right"))
        for prior, offset in zip(offsets, offsets[1:]):
            index = int(np.searchsorted(_GAP_EDGES, offset - prior, side="right")) - 1
            group["gap"][gap_scope * len(_GAP_EDGES) + index] += weight
    return [{key: sorted(histogram.items()) for key, histogram in cohort.items()}
            for _, cohort in sorted(groups.items())]


def _mix_visit_banks(profile_banks, fallback, profile_count):
    weighted = [(bank, 1) for bank in profile_banks]
    missing = profile_count - len(weighted)
    if missing:
        weighted.append((fallback, missing))
    denominator = lcm(*(bank["observed_windows"] for bank, _ in weighted))
    return _visit_histograms(
        (offsets, weight * multiplier * (denominator // bank["observed_windows"]))
        for bank, multiplier in weighted
        for offsets, weight in zip(bank["offsets_seconds"], bank["frequencies"])
    )


def aggregate_visit_timing(model, profiles=None):
    """Fit aggregate schedules by entry and prior activity, with no per-person lists."""
    validate_visit_timing(model)
    entry = model["entry_pool"]
    regular = model["regular_pool"]
    strata = model["regular_stratum_banks"]
    calibration = {**{key: model[key] for key in _VISIT_FIELDS},
                   "population_schema_version": 2,
                   "training": {"max_window_end_at": model["training"]["max_window_end_at"]},
                   "entry": _mix_visit_banks([], entry, 1),
                   "regular": _mix_visit_banks([], regular, 1),
                   "prior_activity": {key: _mix_visit_banks([], bank, 1) for key, bank in strata.items()}}
    if profiles:
        ids_by_group = defaultdict(list)
        for profile in profiles:
            ids_by_group[profile["home_group"]].append(profile["profile_id"])
        calibration["groups"] = {}
        for group, ids in ids_by_group.items():
            entry_banks = [model["entry_profile_banks"][pid] for pid in ids if pid in model["entry_profile_banks"]]
            regular_banks = [model["regular_profile_banks"].get(pid, {}) for pid in ids]
            calibration["groups"][group] = {
                "entry": _mix_visit_banks(entry_banks, entry, len(ids)),
                "prior_activity": {
                    flag: _mix_visit_banks([bank[flag] for bank in regular_banks if flag in bank],
                                           strata.get(flag, regular), len(ids))
                    for flag in ("0", "1")
                },
            }
    return calibration


def _histogram(rows, lower, upper):
    if not rows or any(type(index) is not int or not lower <= index < upper or type(weight) is not int or weight <= 0
                       for index, weight in rows):
        raise ValueError("invalid population timing histogram")
    indices, weights = zip(*rows)
    probabilities = np.asarray(weights, dtype=float)
    return np.asarray(indices), probabilities / probabilities.sum()


def _bank_draws(groups, rng, version):
    total = sum(weight for group in groups for _, weight in group["count"])
    if total <= 0:
        raise ValueError("population timing requires nonempty count groups")
    templates = Counter()
    for group in groups:
        _histogram(group["count"], 1, HORIZON_SECONDS + 1)
        if version == 1:
            first_bins, first_p = _histogram(group["first"], 0, HORIZON_SECONDS // _HOUR)
            span_bins, span_p = _histogram(group["span"], 0, _SPAN_BINS)
            interior = _histogram(group["interior"], 0, _INTERIOR_BINS) if group["interior"] else None
        else:
            clocks, clock_p = _histogram(group["clocks"], 0, (HORIZON_SECONDS // _HOUR) * _SPAN_BINS)
            scopes = defaultdict(list)
            for cell, weight in group["gap"]:
                scope, index = divmod(cell, len(_GAP_EDGES))
                scopes[scope].append((index, weight))
            gaps_by_scope = {scope: _histogram(rows, 0, len(_GAP_EDGES) - 1) for scope, rows in scopes.items()}
        for count, weight in group["count"]:
            size = max(1, round(_BANK_SIZE * weight / total))
            frequency, remainder = divmod(weight * _BANK_SIZE, size)
            if version == 1:
                firsts = rng.choice(first_bins, size=size, p=first_p) * _HOUR + rng.integers(_HOUR, size=size)
                spans = rng.choice(span_bins, size=size, p=span_p)
            else:
                draws = rng.choice(clocks, size=size, p=clock_p)
                firsts = draws // _SPAN_BINS * _HOUR + rng.integers(_HOUR, size=size)
                spans = draws % _SPAN_BINS
            for index, (first, span) in enumerate(zip(firsts.tolist(), spans.tolist())):
                offsets = [first]
                if count > 1:
                    duration = int(rng.integers(_VISIT_SPAN_EDGES[span], _VISIT_SPAN_EDGES[span + 1]))
                    duration = min(duration, HORIZON_SECONDS - 1 - first)
                    if count > 2 and duration:
                        if version == 1:
                            if interior is None:
                                raise ValueError("population timing needs interior positions for three or more starts")
                            n = count - 2
                            phases = (rng.choice(interior[0], size=n, p=interior[1]) + rng.random(n)) / _INTERIOR_BINS
                            offsets.extend(np.sort(first + (phases * duration).astype(int)).tolist())
                        else:
                            scope = int(np.searchsorted(_GAP_SPAN_SPLITS, _VISIT_SPAN_EDGES[span], side="right"))
                            if scope not in gaps_by_scope:
                                raise ValueError("population timing needs gaps for the selected span")
                            gap_bins, gap_p = gaps_by_scope[scope]
                            bins = rng.choice(gap_bins, size=count - 1, p=gap_p)
                            lower, upper = _GAP_EDGES[bins], _GAP_EDGES[bins + 1]
                            gaps = np.where(bins == 0, 0, rng.uniform(lower, upper))
                            if not gaps.any():
                                gaps[-1] = 1
                            offsets.extend(np.minimum(first + np.cumsum(gaps[:-1]) * duration / gaps.sum(),
                                                      first + duration).tolist())
                    elif count > 2:
                        offsets.extend([first] * (count - 2))
                    offsets.append(first + duration)
                templates[tuple(offsets)] += frequency + int(index < remainder)
    return _pack(templates)


def materialize_visit_timing(calibration, source):
    """Draw synthetic schedules once and reuse the native sampler and CDF."""
    version = calibration.get("population_schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported population visit-timing calibration")
    rng = np.random.default_rng(calibration["seed"])
    model = {**{key: calibration[key] for key in _VISIT_FIELDS},
             "population_schema_version": version, "source": source, "training": calibration["training"],
             "entry_profile_banks": {}, "regular_profile_banks": {},
             "entry_pool": _bank_draws(calibration["entry"], rng, version),
             "regular_pool": _bank_draws(calibration["regular"], rng, version),
             "regular_stratum_banks": {key: _bank_draws(bank, rng, version)
                                      for key, bank in sorted(calibration["prior_activity"].items())},
             "contract": "Synthetic schedules drawn from cohort count, timing and elapsed-position histograms."}
    if calibration.get("groups"):
        model["profile_key_scope"] = "synthetic_home_group"
        for group, cohort in sorted(calibration["groups"].items()):
            model["entry_profile_banks"][group] = _bank_draws(cohort["entry"], rng, version)
            model["regular_profile_banks"][group] = {flag: _bank_draws(bank, rng, version)
                                                     for flag, bank in sorted(cohort["prior_activity"].items())}
    validate_visit_timing(model)
    return model


def aggregate_review_model(model):
    """Keep fitted response heads and only joint delay/score bin counts."""
    validate_review_model(model)
    survey = model["order_survey"]
    return {"schema_version": model["schema_version"], "type": model["type"], "population_schema_version": 1,
            "product_stars": {"model": {key: model["product_stars"]["model"][key] for key in _REVIEW_FIELDS}},
            "order_survey": {"model": {key: survey["model"][key] for key in _REVIEW_FIELDS},
                             "window_days": survey["window_days"],
                             "delay_score_histograms": {
                                 name: [[hour, score, n] for (hour, score), n in sorted(Counter(
                                     (int(delay // _REVIEW_BIN_SECONDS), int(score)) for delay, score in rows).items())]
                                 for name, rows in survey["delay_score_banks"].items()}}}


def materialize_review_model(calibration):
    """Build a bounded stratified draw of synthetic bin-midpoint delays."""
    if calibration.get("population_schema_version") != 1:
        raise ValueError("unsupported population review calibration")
    survey = calibration["order_survey"]
    banks = {}
    for name, rows in survey["delay_score_histograms"].items():
        banks[name] = []
        if not rows:
            continue
        if any(type(hour) is not int or not 0 <= hour < 28 * _DAY // _REVIEW_BIN_SECONDS
               or type(score) is not int or not 1 <= score <= 5 or type(count) is not int or count <= 0
               for hour, score, count in rows):
            raise ValueError("invalid population review delay/score counts")
        cumulative = np.cumsum([count for _, _, count in rows])
        indices = np.searchsorted(cumulative, cumulative[-1] * (np.arange(_BANK_SIZE) + 0.5) / _BANK_SIZE)
        banks[name] = [[(rows[index][0] + 0.5) * _REVIEW_BIN_SECONDS, rows[index][1]] for index in indices]
    model = {"schema_version": calibration["schema_version"], "type": calibration["type"], "population_schema_version": 1,
             "product_stars": calibration["product_stars"],
             "order_survey": {"model": survey["model"], "window_days": survey["window_days"], "delay_score_banks": banks,
                              "delay_bank_policy": "synthetic midpoint delays from six-hour cohort histograms"}}
    validate_review_model(model)
    return model
