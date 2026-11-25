"""Read the simulator's public inputs and assemble aligned Retention samples."""

import json
from pathlib import Path
import sys
import hashlib
import sqlite3
import time
from contextlib import contextmanager, closing

import numpy as np
import pandas as pd

from .features import ACTIONS, sequences


DEFINITION = "future_no_session_28d_active28d"
DAYS = 28
CONTROL_POLICY = "no_new_coupon"
KEYS = ["customer_id", "as_of_at", "snapshot_id"]
PROVENANCE = ["dataset_id", "raw_database_sha256", "currency", "amount_unit",
              "expanded_feature_set_version"]
EXTRAS = {
    "purchase_cycle_anomaly": "Purchase recency divided by mean past purchase gap.",
    "journey_stage": "Past order count capped at 3: none, first, repeat, three or more.",
    "stage_tenure_days": "Days since signup or the order that began the current stage.",
    "action_group_transitions": "Adjacent timestamps with differing action presence.",
    "last_action_group_run": "Trailing timestamp groups with the same action presence.",
}
RELATIVE_CHANGES = {
    "orders": ("purchase_count_30d", "orders_delta_30d"),
    "net_spend": ("net_spend_30d", "net_spend_delta_30d"),
    "mean_net_order_value": ("mean_net_order_value_30d", "mean_net_order_value_delta_30d"),
    "unique_purchase_products": ("unique_purchase_products_30d", "unique_purchase_products_delta_30d"),
    "purchased_qty": ("purchased_qty_30d", "purchased_qty_delta_30d"),
}
HISTORY_FEATURES = (
    "history_session_count", "history_active_days", "history_session_rate_30d", "session_count_90d",
    "history_order_count", "history_view_share", "history_cart_share", "history_remove_share", "history_purchase_share",
)
EXTRAS.update(dict.fromkeys(HISTORY_FEATURES, "Observed historical summary; see the feature dictionary."))


def simulator_imports():
    # Support the sibling simulator source checkout as well as an installed package.
    try:
        import simulate.v4
    except ModuleNotFoundError as exc:
        if exc.name not in ("simulate", "simulate.v4"):
            raise
        source = Path(__file__).resolve().parents[1] / "simulator" / "src"
        if not source.is_dir():
            raise ValueError("Install the project's simulator or make simulator/src importable") from exc
        sys.path.insert(0, str(source))
    from simulate.v4.data.loader import load_retention_inputs
    from simulate.v4.features import build_features, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION
    from simulate.v4.simulator.raw_store import load_raw_dataset
    return load_retention_inputs, build_features, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION, load_raw_dataset


def aware_utc(value):
    point = pd.Timestamp(value)
    if pd.isna(point) or point.tzinfo is None:
        raise ValueError("Simulator cutoff requires a timezone, for example 2024-04-08T00:00:00+09:00")
    return point.tz_convert("UTC")


def _feature_contract(metadata):
    _, _, _, legacy, _ = simulator_imports()
    from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, feature_window_days, is_weekly
    return (WEEKLY_FEATURE_VERSION if is_weekly(metadata) else legacy), feature_window_days(metadata)


def read_raw(path):
    _, _, _, _, load_raw = simulator_imports()
    path = Path(path)
    if path.is_dir():
        path = path / "raw.sqlite"
    tables, metadata = load_raw(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    metadata = {**metadata, "raw_database_sha256": manifest["database_sha256"]}
    tables["_metadata"] = metadata
    return tables, metadata


def eligible_customers(tables, point):
    sessions = tables["sessions"]
    start = point - pd.Timedelta(days=DAYS)
    # Eligibility requires positive evidence of a recent session.
    observed = aware_utc(tables["_metadata"]["observation_start_at"])
    end = aware_utc(tables["_metadata"]["observation_end_at"])
    if not observed <= point <= end:
        raise ValueError("Cutoff falls outside the raw observation period")
    customers = tables["customers"]
    eligible = set(sessions.loc[sessions.started_at.ge(start) & sessions.started_at.lt(point), "customer_id"])
    return customers.customer_id.isin(eligible) & customers.signed_up_at.lt(point)


@contextmanager
def _verified_database(path):
    """Read-only canonical SQLite access after authenticating the complete file."""
    path = Path(path)
    if path.is_dir():
        path = path / "raw.sqlite"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    metadata = dict(manifest["metadata"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["database_sha256"]:
        raise ValueError("raw database checksum mismatch")
    metadata["raw_database_sha256"] = digest.hexdigest()
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != 5 or metadata.get("raw_schema_version") != version or manifest.get("raw_schema_version") != version:
            raise ValueError("Projected input requires canonical raw schema 5")
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if names != {"customers", "products", "sessions", "events", "orders", "order_items", "coupons"}:
            raise ValueError("Raw database requires the seven canonical business tables")
        yield connection, metadata


def _projectable(path):
    path = Path(path)
    if path.is_dir():
        path = path / "raw.sqlite"
    return json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))["raw_schema_version"] == 5


def read_training_raw(path, point):
    """Project records needed beyond the already materialized official KIT39."""
    if not _projectable(path):
        return read_raw(path)
    recent = (point-pd.Timedelta(days=DAYS)).isoformat(timespec="nanoseconds")
    cutoff = point.isoformat(timespec="nanoseconds")
    end = (point+pd.Timedelta(days=DAYS)).isoformat(timespec="nanoseconds")
    with _verified_database(path) as (connection, metadata):
        _, (recent_days, _, _) = _feature_contract(metadata)
        lower = (point-pd.Timedelta(days=recent_days)).isoformat(timespec="nanoseconds")
        tables = {
            "customers": pd.read_sql_query("SELECT customer_id, signed_up_at FROM customers", connection),
            "sessions": pd.read_sql_query("SELECT customer_id, started_at FROM sessions WHERE started_at >= ? AND started_at < ?", connection, params=(recent, end)),
            "events": pd.read_sql_query("SELECT customer_id, session_id, event_at, event_type FROM events WHERE event_at < ? AND (event_type = 'purchase' OR event_at >= ?)", connection, params=(cutoff, lower)),
            "coupons": pd.read_sql_query("SELECT customer_id, sent_at, policy_id FROM coupons WHERE sent_at >= ? AND sent_at < ?", connection, params=(cutoff, end)),
        }
    for table, clock in (("customers", "signed_up_at"), ("sessions", "started_at"), ("events", "event_at"), ("coupons", "sent_at")):
        tables[table][clock] = pd.to_datetime(tables[table][clock], utc=True)
    tables["_metadata"] = metadata
    return tables, metadata


def read_prediction_raw(path, point):
    """Keep the full observed prefix needed by official cart/review feature replay."""
    if not _projectable(path):
        return read_raw(path)
    cutoff = point.isoformat(timespec="nanoseconds")
    queries = {
        "customers": ("SELECT * FROM customers", ()),
        "products": ("SELECT * FROM products", ()),
        "sessions": ("SELECT * FROM sessions WHERE started_at < ?", (cutoff,)),
        "events": ("SELECT * FROM events WHERE event_at < ?", (cutoff,)),
        "coupons": ("SELECT * FROM coupons WHERE sent_at < ?", (cutoff,)),
        "orders": ("SELECT o.* FROM orders o JOIN events e ON o.purchase_event_id=e.event_id WHERE e.event_at < ?", (cutoff,)),
        "order_items": ("SELECT i.* FROM order_items i JOIN orders o ON i.order_id=o.order_id JOIN events e ON o.purchase_event_id=e.event_id WHERE e.event_at < ?", (cutoff,)),
    }
    with _verified_database(path) as (connection, metadata):
        tables = {}
        for name, (query, params) in queries.items():
            integer_types = {row[1]: "Int64" for row in connection.execute(f'PRAGMA table_info("{name}")') if row[2] == "INTEGER"}
            tables[name] = pd.read_sql_query(query, connection, params=params, dtype=integer_types)
    clocks = {"customers": ("signed_up_at",), "sessions": ("started_at", "ended_at"),
              "events": ("event_at",), "coupons": ("sent_at", "expires_at")}
    for name, columns in clocks.items():
        for column in columns:
            tables[name][column] = pd.to_datetime(tables[name][column], utc=True)
    tables["_metadata"] = metadata
    return tables, metadata


def read_history_features(path, point):
    """Aggregate observed prefixes without exposing generator parameters or assignments."""
    cutoff = point.isoformat(timespec="nanoseconds")
    with _verified_database(path) as (connection, metadata):
        _, (recent_days, _, gap_days) = _feature_contract(metadata)
        low90 = (point-pd.Timedelta(days=gap_days)).isoformat(timespec="nanoseconds")
        customers = pd.read_sql_query("SELECT customer_id, signed_up_at FROM customers", connection).set_index("customer_id")
        sessions = pd.read_sql_query(
            "SELECT customer_id, COUNT(*) AS history_session_count, COUNT(DISTINCT substr(started_at,1,10)) AS history_active_days, SUM(started_at >= ?) AS session_count_90d FROM sessions WHERE started_at < ? GROUP BY customer_id",
            connection, params=(low90, cutoff)).set_index("customer_id")
        actions = pd.read_sql_query(
            "SELECT customer_id, COUNT(*) AS actions, SUM(event_type='view') AS views, SUM(event_type='cart') AS carts, SUM(event_type='remove_from_cart') AS removes, SUM(event_type='purchase') AS purchases FROM events WHERE event_at < ? AND event_type IN ('view','cart','remove_from_cart','purchase') GROUP BY customer_id",
            connection, params=(cutoff,)).set_index("customer_id")
        orders = pd.read_sql_query(
            "SELECT e.customer_id, COUNT(*) AS history_order_count FROM orders o JOIN events e ON o.purchase_event_id=e.event_id WHERE e.event_at < ? GROUP BY e.customer_id",
            connection, params=(cutoff,)).set_index("customer_id")
    result = customers.join(sessions).join(orders)
    counts = ["history_session_count", "history_active_days", "session_count_90d", "history_order_count"]
    result[counts] = result[counts].fillna(0)
    signed = pd.to_datetime(result.signed_up_at, utc=True)
    observed = aware_utc(metadata["observation_start_at"])
    age = (point-signed.clip(lower=observed)).dt.total_seconds()/86400
    result["history_session_rate_30d"] = result.history_session_count*recent_days/age.where(age.gt(0))
    covered90 = observed <= signed.where(signed.ge(point-pd.Timedelta(days=gap_days)), point-pd.Timedelta(days=gap_days))
    result["session_count_90d"] = result.session_count_90d.where(covered90)
    for kind, column in (("view", "views"), ("cart", "carts"), ("remove", "removes"), ("purchase", "purchases")):
        result[f"history_{kind}_share"] = (actions[column]/actions.actions.where(actions.actions.gt(0))).reindex(result.index)
    return result[list(HISTORY_FEATURES)].astype(float).reset_index()


def enrich(tables, meta, x, config):
    """Join only strict-past behavior to the shared predictor matrix."""
    events = tables["events"]
    _, (recent_days, _, _) = _feature_contract(tables["_metadata"])
    earliest, latest = meta.as_of_at.min() - pd.Timedelta(days=recent_days), meta.as_of_at.max()
    behavior = events.loc[events.event_type.isin(ACTIONS) & events.customer_id.isin(meta.customer_id)
                          & events.event_at.ge(earliest) & events.event_at.lt(latest),
                          ["customer_id", "session_id", "event_at", "event_type"]]
    event_groups = dict(tuple(behavior.sort_values("event_at").groupby("customer_id", sort=False)))
    purchases = events.loc[events.event_type.eq("purchase") & events.event_at.lt(latest)
                           & events.customer_id.isin(meta.customer_id), ["customer_id", "event_at"]]
    order_times = {u: g.event_at.sort_values() for u, g in purchases.groupby("customer_id", sort=False)}
    signup = tables["customers"].set_index("customer_id").signed_up_at
    observed_from = aware_utc(tables["_metadata"]["observation_start_at"])
    seqs, truncated, additions = [], [], []
    for row in meta.itertuples(index=False):
        point, user = row.as_of_at, row.customer_id
        ev = event_groups.get(user, behavior.iloc[:0])
        recent = ev.loc[ev.event_at.lt(point) & ev.event_at.ge(point - pd.Timedelta(days=recent_days))]
        cut = 0
        if config.get("_include_sequences", True):
            seq, cut = sequences(recent, point, config.get("max_length", 128))
            seqs.append(seq)
        truncated.append(cut)
        if not config.get("extra_features", True):
            continue
        times = order_times.get(user, purchases.event_at.iloc[:0])
        past = times[times.lt(point)]
        stage = min(len(past), 3)
        transition = past.iloc[stage - 1] if stage else signup.loc[user]
        # Sequence summary covers the same window as the encoder, before token truncation.
        signatures = recent.groupby("event_at").event_type.agg(frozenset).tolist()
        run = 0
        for signature in reversed(signatures):
            if signature != signatures[-1]:
                break
            run += 1
        history_known = signup.loc[user] >= observed_from
        window_known = observed_from <= max(signup.loc[user], point - pd.Timedelta(days=recent_days))
        additions.append({
            "journey_stage": stage if history_known else np.nan,
            "stage_tenure_days": (point - transition).total_seconds() / 86400 if history_known else np.nan,
            "action_group_transitions": sum(a != b for a, b in zip(signatures, signatures[1:])) if window_known else np.nan,
            "last_action_group_run": run if window_known else np.nan,
        })
    x = x.reset_index(drop=True).astype(float)
    if additions:
        x = pd.concat([x, pd.DataFrame(additions)], axis=1)
        x["purchase_cycle_anomaly"] = x.purchase_recency_days / x.mean_purchase_gap_days.where(x.mean_purchase_gap_days.gt(0))
        for name, (current, delta) in RELATIVE_CHANGES.items():
            previous = x[current] - x[delta]
            x[f"{name}_relative_change_30d"] = x[delta] / previous.where(previous.gt(0))
    if config.get("history_features", False):
        history = tables["_history"].set_index("customer_id").reindex(meta.customer_id).reset_index(drop=True)
        x = pd.concat([x, history[list(HISTORY_FEATURES)]], axis=1)
    meta = meta.assign(as_of=meta.as_of_at, truncated_groups=truncated)
    meta["history_group"] = pd.cut(x.observed_tenure_days, [-np.inf, 30, 90, np.inf],
                                    labels=["under_30d", "30_to_90d", "over_90d"]).astype(str)
    return meta, x.replace([np.inf, -np.inf], np.nan), seqs


def training_inputs(config):
    started = time.perf_counter()
    load_features, _, _, current_version, _ = simulator_imports()
    meta, x, y = load_features(config["feature_dir"])
    features_loaded = time.perf_counter()
    tables, metadata = read_training_raw(config["input_dir"], aware_utc(meta.as_of_at.iloc[0]))
    current_version, _ = _feature_contract(metadata)
    raw_loaded = time.perf_counter()
    for key in ("dataset_id", "raw_database_sha256", "currency", "amount_unit"):
        if not meta[key].eq(str(metadata[key])).all():
            raise ValueError(f"Feature and raw inputs disagree on {key}")
    if not meta.expanded_feature_set_version.eq(current_version).all():
        raise ValueError(f"Re-materialize features with the installed simulator ({current_version}) before training")
    if meta.as_of_at.nunique() != 1 or meta.customer_id.duplicated().any():
        raise ValueError("Simulator experiment mode requires one snapshot and one row per customer")
    cutoff = aware_utc(metadata["as_of_at"])
    if not meta.as_of_at.eq(cutoff).all():
        raise ValueError("Feature snapshot does not match the raw experiment cutoff")
    customers = tables["customers"]
    eligible = set(customers.loc[eligible_customers(tables, cutoff), "customer_id"])
    if not set(meta.customer_id) <= eligible:
        raise ValueError("Retention labels contain customers outside the 28-day eligibility window")
    end = cutoff + pd.Timedelta(days=DAYS)
    if end > aware_utc(metadata["observation_end_at"]):
        raise ValueError("Experiment outcome exceeds the raw observation end")
    sessions = tables["sessions"]
    returned = set(sessions.loc[sessions.started_at.ge(cutoff) & sessions.started_at.lt(end), "customer_id"])
    if not np.array_equal((~meta.customer_id.isin(returned)).to_numpy(), y.to_numpy()):
        raise ValueError("Materialized Retention labels differ from observed raw sessions")
    coupons = tables["coupons"]
    outcome_sends = coupons.loc[coupons.sent_at.ge(cutoff) & coupons.sent_at.lt(end)]
    sent_customers = set(outcome_sends.customer_id)
    if meta.policy_id.isna().any() or (meta.policy_id.eq(CONTROL_POLICY) & meta.customer_id.isin(sent_customers)).any():
        raise ValueError("Experiment policies disagree with raw coupon sends")
    expected_policy = outcome_sends.loc[outcome_sends.sent_at.eq(cutoff)].set_index("customer_id").policy_id
    treated = meta.loc[meta.policy_id.ne(CONTROL_POLICY)]
    if (outcome_sends.customer_id.duplicated().any() or
            not treated.customer_id.map(expected_policy).eq(treated.policy_id).fillna(False).all()):
        raise ValueError("Treatment labels require one matching coupon at the experiment cutoff")
    meta = meta.assign(y=y.to_numpy(), label_end=meta.outcome_end_at)
    if config.get("policy_condition", "observed_experiment_mix") == CONTROL_POLICY:
        keep = meta.policy_id.eq(CONTROL_POLICY)
        meta, x = meta.loc[keep].reset_index(drop=True), x.loc[keep].reset_index(drop=True)
        if meta.empty:
            raise ValueError("No mature control observations")
    if config.get("history_features", False):
        tables["_history"] = read_history_features(config["input_dir"], cutoff)
    meta, x, seqs = enrich(tables, meta, x, config)
    validation = {"dataset_id": metadata["dataset_id"], "raw_database_sha256": metadata["raw_database_sha256"],
                  "currency": metadata["currency"], "amount_unit": str(metadata["amount_unit"]),
                  "feature_set_version": current_version, "observed_policy_counts": meta.policy_id.value_counts().to_dict(),
                  "observation_start_at": metadata["observation_start_at"], "observation_end_at": metadata["observation_end_at"]}
    validation.update({key: metadata.get(key) for key in (
        "seed", "as_of_at", "definition_id", "outcome_window_days", "eligibility_window_days",
        "policy_id", "retention_mode", "configured_target_churn_rate", "model_sha256", "time_spec", "time_windows_days")})
    validation["stage_seconds"] = {"official_features": features_loaded-started,
                                    "verified_raw_projection": raw_loaded-features_loaded,
                                    "contracts_extra_features_sequences": time.perf_counter()-raw_loaded}
    return meta, x, seqs, pd.DataFrame(columns=["customer_id", "as_of", "reason"]), validation


def prediction_inputs(path, as_of, config, profile=None):
    started = time.perf_counter()
    profile = {} if profile is None else profile
    _, build_features, columns, version, _ = simulator_imports()
    point = aware_utc(as_of)
    tables, metadata = read_prediction_raw(path, point)
    version, _ = _feature_contract(metadata)
    raw_loaded = time.perf_counter()
    profile["verified_raw_prefix_seconds"] = raw_loaded-started
    customers = tables["customers"]
    eligible = eligible_customers(tables, point)
    excluded = customers.loc[~eligible, ["customer_id"]].assign(
        as_of_at=point, reason=np.where(customers.loc[~eligible, "signed_up_at"].ge(point),
                                       "not_signed_up_at_cutoff", "no_recent_session"))
    experiment_cutoff = point == aware_utc(metadata.get("as_of_at", point))
    separator = ":" if experiment_cutoff else "|"
    excluded["snapshot_id"] = excluded.customer_id + separator + point.isoformat()
    excluded["expanded_feature_set_version"] = version
    for key in PROVENANCE[:-1]:
        excluded[key] = str(metadata[key])
    if not eligible.any():
        profile["official_features_seconds"] = 0.
        profile["extra_features_sequences_seconds"] = 0.
        frame = pd.DataFrame(columns=KEYS + PROVENANCE)
        return frame, pd.DataFrame(), [], excluded
    # build_features uses only past records with an explicit cutoff, without outcome sidecars.
    frame = build_features(tables, as_of_at=point)
    features_built = time.perf_counter()
    profile["official_features_seconds"] = features_built-raw_loaded
    frame = frame.loc[frame.customer_id.isin(customers.loc[eligible, "customer_id"])].reset_index(drop=True)
    if experiment_cutoff:
        # Match the source experiment's stable snapshot ID convention.
        frame["snapshot_id"] = frame.customer_id + ":" + point.isoformat()
    if not frame.expanded_feature_set_version.eq(version).all():
        raise ValueError("Unexpected simulator feature version")
    if config.get("history_features", False):
        tables["_history"] = read_history_features(path, point)
    meta, x, seqs = enrich(tables, frame[KEYS + PROVENANCE], frame[list(columns)], config)
    profile["extra_features_sequences_seconds"] = time.perf_counter()-features_built
    return meta, x, seqs, excluded
