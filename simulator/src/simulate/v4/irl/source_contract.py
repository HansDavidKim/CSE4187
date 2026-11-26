"""Validate immutable prepared observations without changing IDs, clocks or sessions."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ("cosmetics", "electronics", "tianchi")
ACTIONS = ("view", "cart", "remove", "purchase")
DOMAIN_ACTIONS = {"cosmetics": ACTIONS, "electronics": ("view", "cart", "purchase"),
                  "tianchi": ("view", "purchase")}


def canonical_available_actions(value=None):
    if value is None:
        return ACTIONS
    if isinstance(value, (str, bytes)):
        raise ValueError("available_actions must be a nonempty subset of shopping actions")
    try:
        names = tuple(value)
        if not names or len(set(names)) != len(names) or not set(names) <= set(ACTIONS):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("available_actions must be a nonempty, unique subset of shopping actions") from exc
    return tuple(action for action in ACTIONS if action in names)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_cutoff(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("source cutoffs must be explicit timezone-aware timestamps")
    return result.tz_convert("UTC")


def source_profile_id(domain, user_id):
    if domain not in DOMAINS or not isinstance(user_id, str) or not user_id or user_id.strip() != user_id:
        raise ValueError("profile IDs require a domain and unchanged original user ID text")
    return hashlib.sha256(f"{domain}:{user_id}".encode()).hexdigest()[:16]


def validate_prepared_rows(rows, *, domain, clock_resolution):
    from .fit_shopping import ACTION_MAP, SOURCE_COLUMNS, price_minor
    required = {"user_id", "product_id", "category_id", "user_session", "source_event_id", "at", "action", "price_minor", "session"}
    if domain not in DOMAINS or rows.empty or not required <= set(rows):
        raise ValueError("prepared source needs a supported domain and observed row schema")
    if not isinstance(rows["at"].dtype, pd.DatetimeTZDtype) or rows["at"].isna().any():
        raise ValueError("prepared source clocks must remain timezone-aware and known")
    for name in ("user_id", "product_id", "category_id", "user_session", "source_event_id"):
        if not rows[name].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all():
            raise ValueError(f"prepared {name} must retain original nonempty text")
    if not rows.user_id.str.fullmatch(r"\d+").all() or rows.source_event_id.duplicated().any():
        raise ValueError("source user ID namespace or event identity changed")
    available = canonical_available_actions(rows.action.unique())
    if not set(available) <= set(DOMAIN_ACTIONS[domain]):
        raise ValueError("prepared actions exceed the source observation support")
    order = pd.MultiIndex.from_frame(rows[["user_id", "at", "source_event_id"]])
    if not order.is_monotonic_increasing:
        raise ValueError("prepared source row/session order changed")
    gap = rows.groupby("user_id", sort=False)["at"].diff()
    previous = rows.groupby("user_id", sort=False).user_session.shift()
    sessions = (gap.isna() | gap.gt(pd.Timedelta(minutes=30)) | rows.user_session.ne(previous)).cumsum()
    if not pd.api.types.is_integer_dtype(rows.session) or not np.array_equal(rows.session.to_numpy(), sessions.to_numpy()):
        raise ValueError("prepared source session boundaries or IDs changed")
    if domain == "tianchi":
        if clock_resolution != "day" or not rows["at"].eq(rows["at"].dt.floor("D")).all() or rows.price_minor.notna().any():
            raise ValueError("Tianchi must retain daily clocks and unobserved monetary values")
    else:
        if clock_resolution != "second" or not set(SOURCE_COLUMNS) <= set(rows):
            raise ValueError("retail source requires original text and second-resolution observations")
        if not rows.event_time.map(lambda value: isinstance(value, str)).all() or not rows.price.map(lambda value: isinstance(value, str)).all():
            raise ValueError("retail timestamp and price text must be preserved")
        if (not pd.to_datetime(rows.event_time, utc=True, errors="raise").eq(rows["at"]).all()
                or not rows["at"].eq(rows["at"].dt.floor("s")).all()
                or not rows.event_type.map(ACTION_MAP).eq(rows.action).all()):
            raise ValueError("prepared retail clock/action differs from original text")
        expected = pd.to_numeric(rows.price.map(price_minor), errors="coerce")
        actual = pd.to_numeric(rows.price_minor, errors="raise")
        if not (expected.eq(actual) | (expected.isna() & actual.isna())).all():
            raise ValueError("prepared retail monetary conversion differs from original price text")
    return available


def load_prepared_source(cache_path, *, domain, cutoff_at, inner_cutoff_at=None, require_second_clock=True):
    """Read v7 JSON-bound caches or the original embedded Main Cosmetics cache."""
    if domain not in DOMAINS:
        raise ValueError("an explicit source domain is required")
    path = Path(cache_path).resolve()
    manifest = path.with_suffix(".json")
    cache_sha = file_sha(path)
    requested = utc_cutoff(cutoff_at)
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["spec"]["domain"] != domain or saved["prepared_sha256"] != cache_sha:
            raise ValueError("prepared source domain or SHA256 differs from its manifest")
        manifest_sha = file_sha(manifest)
        rows = pd.read_parquet(path)
    else:
        import pyarrow.parquet as pq
        from .fit_shopping import prepare_source
        metadata = pq.read_schema(path).metadata or {}
        embedded = json.loads(metadata.get(b"quality_source_cache", b"{}"))
        if domain != "cosmetics" or not embedded.get("spec"):
            raise ValueError("prepared source requires its JSON manifest or original Main Cosmetics binding")
        spec = embedded["spec"]
        if (spec["prepare_source_sha256"] != hashlib.sha256(inspect.getsource(prepare_source).encode()).hexdigest()
                or embedded["fingerprint"] != hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()):
            raise ValueError("original Main prepared-source fingerprint changed")
        rows = pd.read_parquet(path)
        legacy_inner = (requested - pd.DateOffset(months=1) if inner_cutoff_at is None else utc_cutoff(inner_cutoff_at))
        legacy_outer = legacy_inner + pd.DateOffset(months=1)
        saved = {"spec": {**spec, "domain": domain}, "prepared_sha256": cache_sha,
                 "rows": embedded["rows"], "users": int(rows.user_id.nunique()), "sessions": int(rows.session.nunique()),
                 "observation_start_at": rows["at"].min().floor("D").isoformat(),
                 "observation_end_at": (rows["at"].max().floor("D") + pd.Timedelta(days=1)).isoformat(),
                 "outer_cutoff_at": legacy_outer.isoformat(),
                 "inner_cutoff_at": legacy_inner.isoformat(),
                 "observed_actions": list(canonical_available_actions(rows.action.unique())),
                 "observed_prices": bool(rows.price_minor.notna().any()), "clock_resolution": "second",
                 "inherited_cache": embedded}
        manifest_sha = None
    if require_second_clock and saved["clock_resolution"] != "second":
        raise ValueError("daily Tianchi observations cannot enter the second-clock shopping/return fitter; use the daily adapter")
    inner = utc_cutoff(saved["inner_cutoff_at"] if inner_cutoff_at is None else inner_cutoff_at)
    outer = utc_cutoff(saved["outer_cutoff_at"])
    start, end = utc_cutoff(saved["observation_start_at"]), utc_cutoff(saved["observation_end_at"])
    if (requested not in (inner, outer) or inner != utc_cutoff(saved["inner_cutoff_at"])
            or not start < inner < outer < end):
        raise ValueError("explicit source cutoffs differ from the prepared observation split")
    if not saved["spec"].get("files"):
        raise ValueError("prepared source must bind the original source files")
    for raw in saved["spec"]["files"]:
        stat = Path(raw["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (raw["bytes"], raw["mtime_ns"]):
            raise ValueError("original source file changed after cache preparation")
    actions = validate_prepared_rows(rows, domain=domain, clock_resolution=saved["clock_resolution"])
    if (len(rows) != saved["rows"] or rows.user_id.nunique() != saved["users"] or rows.session.nunique() != saved["sessions"]
            or set(actions) != set(saved["observed_actions"])
            or bool(rows.price_minor.notna().any()) != saved["observed_prices"]
            or rows["at"].min().floor("D") != start or rows["at"].max().floor("D") + pd.Timedelta(days=1) != end):
        raise ValueError("prepared source counts, observation support or clock bounds changed")
    modulus = saved["spec"]["sample_modulus"]
    if isinstance(modulus, bool) or not isinstance(modulus, int) or modulus < 20:
        raise ValueError("prepared source sampling modulus must be an integer of at least 20")
    if not pd.to_numeric(rows.user_id, errors="raise").mod(modulus).eq(0).all():
        raise ValueError("prepared source user sample/namespace changed")
    contract = {"domain": domain, "clock_resolution": saved["clock_resolution"], "observed_actions": list(actions),
                "observed_prices": saved["observed_prices"], "observation_start_at": start.isoformat(),
                "observation_end_at": end.isoformat(), "outer_cutoff_at": outer.isoformat(), "inner_cutoff_at": inner.isoformat(),
                "training_cutoff_at": requested.isoformat(),
                "user_id_rule": "original source user ID text; never joint namespace",
                "profile_id_rule": "sha256(domain:raw_user_id)[:16]", "prepared_sha256": cache_sha,
                "manifest_sha256": manifest_sha, "rows_before_outer_cutoff": int(rows["at"].lt(outer).sum()),
                "rows_at_or_after_outer_cutoff": int(rows["at"].ge(outer).sum()), "coverage": None,
                "rows_before_training_cutoff": int(rows["at"].lt(requested).sum()),
                "observation_bounds_rule": "first/last observed dated events; not evidence of complete user-day capture"}
    if "feature_windows_days" in saved:
        windows = saved["feature_windows_days"]
        if windows not in ([28, 56, 91], [30, 60, 90]):
            raise ValueError("prepared source feature windows violate the weekly or calendar39 contract")
        contract["feature_windows_days"] = list(windows)
    provenance = {"dataset": {"cosmetics": "Cosmetics Shop events", "electronics": "Electronics events", "tianchi": "Tianchi online events"}[domain],
                  "domain": domain, "files": saved["spec"]["files"], "sample_modulus": modulus,
                  "training_cutoff_at": requested.isoformat(), "prepared_source_cache": {
                      "path": str(path), "sha256": cache_sha, "manifest_path": str(manifest) if manifest.exists() else None,
                      "manifest_sha256": manifest_sha}}
    if file_sha(path) != cache_sha or (manifest.exists() and file_sha(manifest) != manifest_sha):
        raise ValueError("prepared source changed during validation")
    for raw in saved["spec"]["files"]:
        stat = Path(raw["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (raw["bytes"], raw["mtime_ns"]):
            raise ValueError("original source file changed during cache validation")
    return rows, {**saved, "path": str(path), "cache_hit": True, "source_contract": contract,
                  "source_provenance": provenance, "fingerprint": (saved.get("inherited_cache") or {}).get("fingerprint", cache_sha)}
