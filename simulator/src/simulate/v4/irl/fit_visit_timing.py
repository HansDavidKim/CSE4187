"""Joint native session-count/start-time banks on strictly mature source windows."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import accumulate
import hashlib
import json
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

TYPE = "source_native_joint_visit_timing"
HORIZON_SECONDS = 30 * 86400
CUTOFF = "2020-01-29T00:00:00+00:00"


def _utc(value):
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError("visit timing clocks must be timezone-aware")
    return at.tz_convert("UTC")


def _pack(counter):
    templates = sorted(counter)
    frequencies = [int(counter[value]) for value in templates]
    return {"offsets_seconds": [list(value) for value in templates], "frequencies": frequencies,
            "cumulative_frequencies": list(accumulate(frequencies)), "observed_windows": sum(frequencies)}


def fit_visit_timing(rows, observation_start, observation_end, seed=42, cutoff=CUTOFF, *,
                     domain="cosmetics"):
    """Fit empirical conditional timing only; the existing activity model supplies p30."""
    start, end, cutoff = map(_utc, (observation_start, observation_end, cutoff))
    if domain not in {"cosmetics", "electronics"}:
        raise ValueError("retail visit timing requires an identified source domain")
    if not start < cutoff <= end or not {"user_id", "session", "at"} <= set(rows) or rows.empty:
        raise ValueError("visit timing needs prepared source and ordered observation bounds")
    if (rows[["user_id", "session", "at"]].isna().any().any()
            or not isinstance(rows['at'].dtype, pd.DatetimeTZDtype)
            or rows['at'].lt(start).any() or rows['at'].ge(end).any()):
        raise ValueError("visit timing source IDs/clocks must be known inside observation")
    sessions = (rows.groupby(["user_id", "session"], sort=False)['at'].min().reset_index()
                .sort_values(["user_id", "at", "session"], kind="stable"))
    anchors = [at for at in pd.date_range(start.floor('D') + pd.Timedelta(days=30, hours=15),
                end - pd.Timedelta(days=30), freq="30D")
               if at + pd.Timedelta(days=30) < cutoff]
    entry_profiles, regular_profiles = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    entry_pool, regular_pool, strata = Counter(), Counter(), defaultdict(Counter)
    counts = Counter(); active_counts = {"entry": [], "regular": []}; firsts = {"entry": [], "regular": []}
    availability, source_ids = [], set()
    for uid, user in sessions.groupby("user_id", sort=True):
        uid = str(uid)
        if hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] >= 205:
            continue
        source_ids.add(uid)
        times = user['at'].tolist(); first = times[0]
        profile_id = hashlib.sha256(f"{domain}:{uid}".encode()).hexdigest()[:16]
        windows = [(first.floor('D'), True), *((at, False) for at in anchors if first < at)]
        for at, entry in windows:
            until = at + pd.Timedelta(days=30)
            if at < start or until > end or not until < cutoff:
                continue
            kind = "entry" if entry else "regular"
            counts[f"{kind}_complete_windows"] += 1
            offsets = tuple((time - at).total_seconds() for time in times if at <= time < until)
            if not offsets:
                continue
            previous = any(at - pd.Timedelta(days=30) <= time < at for time in times)
            counts[f"{kind}_active_windows"] += 1
            counts[f"{kind}_session_starts"] += len(offsets)
            counts[f"{kind}_duplicate_start_clocks"] += len(offsets) - len(set(offsets))
            counts[f"{kind}_templates_with_ties"] += int(len(offsets) != len(set(offsets)))
            active_counts[kind].append(len(offsets)); firsts[kind].append(offsets[0]); availability.append(until)
            if entry:
                entry_profiles[profile_id][offsets] += 1; entry_pool[offsets] += 1
            else:
                flag = str(int(previous))
                regular_profiles[profile_id][flag][offsets] += 1
                regular_pool[offsets] += 1; strata[flag][offsets] += 1
                counts[f"regular_active_previous30_{flag}_windows"] += 1
    if not entry_pool or not regular_pool:
        raise ValueError("visit timing requires observed complete entry and regular active windows")
    model = {"schema_version": 1, "type": TYPE, "horizon_days": 30, "cutoff_at": cutoff.isoformat(),
        "fit_available_rule": "window_end < cutoff", "seed": int(seed),
        "observation_start_at": start.isoformat(), "observation_end_at": end.isoformat(),
        "entry_anchor": "first-observed UTC day; source time-of-day retained; not actual signup",
        "regular_anchor_hour_utc": 15, "regular_anchors": [at.isoformat() for at in anchors],
        "entry_profile_banks": {pid: _pack(bank) for pid, bank in entry_profiles.items()},
        "entry_pool": _pack(entry_pool),
        "regular_profile_banks": {pid: {flag: _pack(bank) for flag, bank in groups.items()} for pid, groups in regular_profiles.items()},
        "regular_stratum_banks": {flag: _pack(bank) for flag, bank in strata.items()},
        "regular_pool": _pack(regular_pool),
        "training": {"user_rule": "sha256(seed:user_id).digest()[0] < 205", "source_train_partition_users": len(source_ids),
            "source_training_users": len(set(entry_profiles) | set(regular_profiles)),
            **dict(counts), "max_window_end_at": max(availability).isoformat(),
            "entry_profile_coverage": len(entry_profiles), "regular_profile_coverage": len(regular_profiles),
            "active_window_count_summary": {kind: {"mean": float(np.mean(values)), "max": max(values),
                "p99": float(np.quantile(values, .99))} for kind, values in active_counts.items()},
            "pooled_first_offset_lt_28d_share": {kind: float(np.mean(np.asarray(values) < 28 * 86400)) for kind, values in firsts.items()}},
        "contract": "Draw a complete active30 template once; retain offsets inside actual exposure without time compression, jitter, retry or count forcing.",
        "limitations": ["Conditional empirical timing, not newly identified activity hazard or IRL reward.",
            "First-observed day is left-truncated observation entry, not verified registration.",
            "Only source training users and strict mature windows; missing profile support falls back to observed train pools.",
            "Equal-clock starts are retained; engine must separately enforce nonoverlapping sessions and report actor right censoring.",
            "Does not correct signup/enrollment donor-composition bias or prove long-term extrapolation."]}
    validate_model(model)
    if domain != "cosmetics":
        model["source_domain"] = domain
    return model


def validate_model(model):
    if (type(model.get("schema_version")) is not int or model.get("schema_version") != 1
            or model.get("type") != TYPE or type(model.get("horizon_days")) is not int or model.get("horizon_days") != 30
            or model.get("fit_available_rule") != "window_end < cutoff" or model.get("regular_anchor_hour_utc") != 15):
        raise ValueError("incompatible native visit-timing model")
    cutoff = _utc(model.get("cutoff_at"))
    if not _utc(model["observation_start_at"]) < cutoff <= _utc(model["observation_end_at"]):
        raise ValueError("invalid visit-timing source observation bounds")
    if not _utc(model['training']['max_window_end_at']) < cutoff:
        raise ValueError("visit-timing windows must be strictly mature before cutoff")
    blocks = [model['entry_pool'], model['regular_pool'], *model['entry_profile_banks'].values(),
              *model['regular_stratum_banks'].values(),
              *(block for groups in model['regular_profile_banks'].values() for block in groups.values())]
    if set(model['regular_stratum_banks']) - {'0', '1'} or any(set(groups) - {'0', '1'} for groups in model['regular_profile_banks'].values()):
        raise ValueError("invalid prior-activity visit-timing stratum")
    for block in blocks:
        offsets, frequencies = block['offsets_seconds'], block['frequencies']
        if not offsets or len(offsets) != len(frequencies):
            raise ValueError("visit timing bank needs matched nonempty templates/frequencies")
        if any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n <= 0 for n in frequencies):
            raise ValueError("visit timing frequencies must be positive integers")
        if (type(block['observed_windows']) is not int or block['cumulative_frequencies'] != list(accumulate(frequencies))
                or block['observed_windows'] != sum(frequencies)):
            raise ValueError("visit timing empirical frequencies disagree")
        for values in offsets:
            if (not values or any(isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(t)
                    or not 0 <= t < HORIZON_SECONDS for t in values) or values != sorted(values)):
                raise ValueError("visit timing offsets must be ordered finite source starts in [0,30d)")


def _bank(model, profile_id, entry, active_previous30):
    if not isinstance(entry, (bool, np.bool_)) or not isinstance(active_previous30, (bool, np.bool_)):
        raise ValueError("visit timing requires boolean entry and past-activity flags")
    if not isinstance(profile_id, str):
        raise ValueError("visit timing profile identity must be a string")
    if entry:
        block = model['entry_profile_banks'].get(profile_id)
        return (block, "entry_profile") if block else (model['entry_pool'], "entry_pool")
    flag = str(int(active_previous30))
    block = model['regular_profile_banks'].get(profile_id, {}).get(flag)
    if block:
        return block, f"regular_profile_active30_{flag}"
    if flag in model['regular_stratum_banks']:
        return model['regular_stratum_banks'][flag], f"regular_active30_{flag}_pool"
    return model['regular_pool'], "regular_pool"


def sample_session_offsets(model, profile_id, entry, active_previous30, rng):
    """Return one joint source schedule; caller intersects its actual visible exposure."""
    bank, scope = _bank(model, profile_id, entry, active_previous30)
    index = bisect_right(bank['cumulative_frequencies'], int(rng.integers(bank['observed_windows'])))
    return list(bank['offsets_seconds'][index]), scope


def first_session_share(model, profile_id, active_previous30, outcome_days):
    """Exact conditional CDF of the same regular bank used by the runtime sampler."""
    if isinstance(outcome_days, bool) or not isinstance(outcome_days, numbers.Real) or not math.isfinite(outcome_days) or not 0 <= outcome_days <= 30:
        raise ValueError("visit timing outcome_days must be finite in [0,30]")
    bank, _ = _bank(model, profile_id, False, active_previous30)
    return sum(n for offsets, n in zip(bank['offsets_seconds'], bank['frequencies'])
               if offsets[0] < outcome_days * 86400) / bank['observed_windows']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('data/models/shopping_profiles_aligned_v6.json'))
    parser.add_argument('--source-cache', type=Path, default=Path('artifacts/v4/source_sample.parquet'))
    parser.add_argument('--cutoff-at')
    parser.add_argument('--domain', choices=('cosmetics', 'electronics'))
    parser.add_argument('--output', type=Path, default=Path('data/models/visit_timing_v4.json'))
    parser.add_argument('--report', type=Path, default=Path('docs/audit/simulator/visit_timing_training_v4.json'))
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve() or any(path.exists() for path in (args.output, args.report)):
        raise FileExistsError("visit timing output/report must be separate unused paths")
    from simulate.v4.irl.fit_donor_prior import load_prior_source
    base = json.loads(args.base.read_text())
    rows, cache = load_prior_source(base, args.source_cache)
    contract = base.get('source_contract', {})
    declared_cutoff = contract.get('training_cutoff_at', contract.get('outer_cutoff_at', CUTOFF))
    cutoff = args.cutoff_at or declared_cutoff
    domain = args.domain or contract.get('domain', base.get('source', {}).get('domain', 'cosmetics'))
    if ((args.cutoff_at and contract
            and _utc(args.cutoff_at) != _utc(declared_cutoff))
            or (args.domain and contract.get('domain') and args.domain != contract['domain'])
            or contract.get('clock_resolution', 'second') != 'second'):
        raise ValueError("retail timing arguments differ from the prepared source contract")
    model = fit_visit_timing(rows, base['observation_start'], base['observation_end'],
                             int(base.get('seed', 42)), cutoff, domain=domain)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    model.update(source=base['source'], provenance={'base_path': str(args.base), 'base_sha256': digest(args.base),
        'source_cache': cache, 'source_cache_sha256': digest(args.source_cache), 'fitter_sha256': digest(__file__)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(model, stream, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    report = {"passed": True, "model_path": str(args.output), "model_sha256": digest(args.output),
        "type": model['type'], "training": model['training'], "provenance": model['provenance'],
        "regular_anchors": model['regular_anchors'], "entry_anchor": model['entry_anchor'],
        "regular_anchor_hour_utc": 15, "limitations": model['limitations'],
        "adoption": "candidate; requires generated timing/fidelity and raw-integrity checks"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('model_path', 'model_sha256', 'training')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
