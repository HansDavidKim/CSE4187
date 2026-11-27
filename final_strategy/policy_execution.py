"""Validate a fixed weekly policy in two simulator runs. No external dispatch.

run_policy_replay reserves a new output directory. prepare_inputs receives only
the baseline KIT39 and thirteen past weeks and returns (prediction batches,
build_strategy_decisions keyword arguments). Its predictions, cost declarations
and policy parameters must be fixed without consulting post-cutoff outcomes.
"""
from copy import deepcopy
from decimal import Context, localcontext
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as parquet
from clv.bundle import file_sha256
from simulate.v4.configuration import POPULATION_CONFIG
from simulate.v4.data.loader import load_tables
from simulate.v4.data.weekly import build_experiment_log
from simulate.v4.session_observations import completion_exclusions, session_availability
from simulate.v4.simulator.coupon_assignment import assign_coupons
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.storage import FEATURE_KEYS, materialize_features
from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, weekly_metadata
from simulate.v4.validation import audit_coupon_usage, audit_sidecars

from .inputs import load_feature_store
from .budget import _decimal
from .pipeline import write_outputs
from .policy_search import _fingerprint
from .weekly_planning import build_strategy_decisions, validate_planning_output
from .weekly_prediction_adapter import adapt_weekly_predictions
from .weekly_validation import require


CLV_INPUT_COLUMNS = ("customer_id", "as_of_at", "snapshot_id", "reference_week", "dataset_id",
                     "currency", "amount_unit", "sessions", "purchases", "net_revenue_minor",
                     "views", "session_seconds")
OUTCOME_FIELDS = ("dataset_id", "snapshot_id", "as_of_at", "assigned_at", "treatment", "assignment_probability",
                  "coupon_action", "decision_action", "requested_coupon_id", "actual_sent", "coupon_id",
                  "sent_at", "expires_at", "execution_status", "failure_reason", "redeemed_in_outcome",
                  "outcome_redemption_discount_cost_minor", "send_cost_minor", "factual_no_order",
                  "factual_no_session", "factual_outcome_net_revenue_minor", "outcome_end_at",
                  "outcome_observed", "is_censored", "label_known_at", "currency", "amount_unit")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _publish(path, payload):
    temporary = path.with_suffix(".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.rename(path)


def _plan_manifest(directory, expected):
    path = directory / "manifest.json"
    require(file_sha256(path) == expected, "changed plan manifest")
    manifest = _json(path)
    files = manifest["file_sha256"]
    required = {"decision_audit.csv", "customer_identity.parquet", "availability.csv",
                "budget_summary.json", "metadata.json", "customers.csv"}
    require(manifest["status"] == "complete" and required <= files.keys(), "incomplete plan manifest")
    for name, digest in files.items():
        target = (directory / name).resolve()
        require(target.is_relative_to(directory.resolve()) and file_sha256(target) == digest,
                f"plan checksum differs: {name}")


def _same(name, left, right):
    """Compare strictly; retain the keyed row and column in failure evidence."""
    if not left.index.identical(right.index):
        missing = left.index.difference(right.index).tolist()[:3]
        extra = right.index.difference(left.index).tolist()[:3]
        raise ValueError(f"{name}: row keys/order differ; missing={missing}, extra={extra}")
    require(left.columns.equals(right.columns), f"{name}: columns differ")
    for column in left:
        a, b = left[column], right[column]
        require(a.dtype == b.dtype, f"{name}.{column}: dtype {a.dtype} != {b.dtype}")
        if not a.equals(b):
            equal = a.eq(b).fillna(False) | (a.isna() & b.isna())
            keys = a.index[~equal].tolist()[:3]
            # A Parquet round trip can change the representation of an object
            # null (for example, pd.NA to None) without changing missingness.
            if keys:
                key = keys[0]
                raise ValueError(f"{name}.{column}: values differ at {keys}; expected {a.loc[key]!r}, got {b.loc[key]!r}")


def _frozen_plan(plan, features, options):
    summary, audit = plan.budget_summary, plan.decision_audit
    with localcontext(Context(prec=200)):
        validate_planning_output(plan)
    idx = "customer_id"
    require(not audit[idx].duplicated().any(), "duplicate customer in plan")
    _same("availability", audit[list(plan.availability)].set_index(idx).sort_index(),
          plan.availability.set_index(idx).sort_index())
    _same("plan identity", features[list(FEATURE_KEYS)].set_index(idx).sort_index(),
          plan.identity.set_index(idx).sort_index())
    profile = options["profile"]
    fingerprint = profile.configuration_fingerprint()
    scenario = plan.metadata["scenarios"]["budget_100"]
    require(plan.metadata["engine"] == "final_strategy" and summary["has_valid_solution"] is True
            and summary["accepted_for_planning"] is True, "invalid or unapproved budget_100 plan")
    require(plan.metadata["dispatch_authorized"] is False and summary["dispatch_authorized"] is False
            and audit.dispatch_authorized.isin([False]).all(), "external dispatch is forbidden")
    for field, value in dict(configuration_fingerprint=fingerprint, policy_id=profile.policy_id,
                             policy_version=profile.version, algorithm=options["algorithm"],
                             decision_fingerprint=summary["decision_fingerprint"]).items():
        require(value is not None and scenario[field] == summary[field] == value
                and audit[field].notna().all() and audit[field].eq(value).all(), f"plan {field} mismatch")
    require(summary["total_expected_cost"] <= summary["budget"] == _decimal(options["budget"].amount, "budget"),
            "plan budget mismatch")
    scored = audit.status.eq("scored")
    chosen = audit.send.eq(1).fillna(False)
    require(audit.row_state.eq("selected").equals(chosen.astype(bool))
            and audit.plan_state.eq("valid_approved_for_planning").all()
            and audit.has_valid_solution.isin([True]).all() and audit.accepted_for_planning.isin([True]).all()
            and scenario["accepted_for_planning"] is True and scenario["has_valid_solution"] is True
            and scenario["total_expected_cost"] == summary["total_expected_cost"]
            and scenario["budget"] == summary["budget"], "inconsistent canonical allocation")
    require(audit.loc[scored, "selected"].isin([0, 1]).all()
            and audit.loc[~scored, "selected"].isna().all()
            and audit.row_state.eq("model_unavailable").equals(~scored), "invalid model or decision status")
    require(audit.loc[chosen, "planned_sent_at"].notna().all()
            and audit.loc[chosen, "planned_sent_at"].eq(audit.loc[chosen, "as_of_at"]).all()
            and audit.loc[~chosen, "planned_sent_at"].isna().all(), "replay requires t0-only sends")
    return sorted(audit.loc[chosen, idx].tolist())


def _past(tables, t0):
    """Normalize global event numbers while preserving purchase and review FKs."""
    events = tables["events"]
    events = events.loc[events.event_at.lt(t0)].sort_values(["customer_id", "event_at", "event_id"]).copy()
    # Sequence within customer is stable under a global re-numbering. The FK also
    # needs the customer namespace, so use a globally unique canonical ordinal.
    ordinals = pd.Series(range(len(events)), index=events.event_id, dtype="Int64")
    refs = events.related_purchase_event_id
    require(refs.dropna().isin(ordinals.index).all(), "past event references an unavailable purchase")
    events["related_purchase_event_id"] = refs.map(ordinals).astype("Int64")
    orders = tables["orders"].loc[tables["orders"].purchase_event_id.isin(ordinals.index)].copy()
    orders["purchase_event_id"] = orders.purchase_event_id.map(ordinals).astype("Int64")
    events["event_id"] = events.event_id.map(ordinals).astype("Int64")
    items = tables["order_items"]
    sessions = session_availability(tables)
    sessions = sessions.loc[sessions.started_at.lt(t0)].copy()
    pending = ~sessions.completion_available_at.lt(t0)
    sessions.loc[pending, ["ended_at", "completion_available_at"]] = pd.NaT
    coupons = tables["coupons"].loc[lambda frame: frame.sent_at.lt(t0)]
    if coupons.empty:
        # Empty SQLite TEXT columns have no values for pandas to infer from.
        coupons = coupons.astype({k: "string" for k in ("coupon_id", "customer_id", "policy_id")})
    return {
        "customers": tables["customers"].set_index("customer_id"),
        "products": tables["products"].set_index("product_id").sort_index(),
        "sessions": sessions.set_index(["customer_id", "session_id"]).sort_index(),
        "events": events.set_index(["customer_id", "event_id"]),
        "orders": orders.set_index("order_id").sort_index(),
        "order_items": items.loc[items.order_id.isin(orders.order_id)].set_index(["order_id", "line_no"]).sort_index(),
        "coupons": coupons.set_index("coupon_id").sort_index(),
    }


def _generation_inputs(meta):
    keys = ("simulator_version", "raw_schema_version", "feature_version", "retention_mode", "weeks",
            "simulator_code_sha256", "runtime_code_sha256", "configuration_sha256",
            "model_sha256", "seed", "time_spec", "time_windows_days", "as_of_at", "observation_start_at",
            "observation_end_at", "currency", "amount_unit", "min_customer_history_days",
            "eligibility_window_days", "coupon_validity_days", "outcome_window_days")
    evidence = {key: meta[key] for key in keys}
    # Capture optional artifacts, including nested fitted models, without comparing
    # post-assignment diagnostics such as visitation counts or fitted outcomes.
    def hashes(mapping, prefix):
        for key, value in mapping.items():
            if isinstance(value, dict):
                yield from hashes(value, prefix + key + ".")
            elif key.endswith("sha256") and key not in ("assignment_sha256", "raw_database_sha256"):
                yield prefix + key, value
    evidence["artifact_hashes"] = dict(hashes(meta, ""))
    return evidence


def _input_provenance(features, history, tables):
    meta = tables["_metadata"]
    require(tuple(history.columns) == CLV_INPUT_COLUMNS,
            f"CLV history schema mismatch: expected {CLV_INPUT_COLUMNS}, got {tuple(history.columns)}")
    point = pd.Timestamp(meta["as_of_at"])
    for name, frame in (("Feature Store", features), ("CLV history", history)):
        for field in ("dataset_id", "as_of_at", "currency", "amount_unit"):
            value = point if field == "as_of_at" else meta[field]
            column = frame[field]
            matches = column.astype(str).eq(str(value)) if field == "amount_unit" else column.eq(value)
            require(column.notna().all() and matches.all(), f"{name} {field} mismatch")
    for column, key in (("raw_database_sha256", "raw_database_sha256"), ("source_model_sha256", "model_sha256"),
                        ("expanded_feature_set_version", "feature_version")):
        require(features[column].isin([meta[key]]).all(), f"Feature Store {column} mismatch")
    ids = features.set_index("customer_id", verify_integrity=True).snapshot_id
    require(set(ids.index) == set(tables["customers"].customer_id), "Feature Store roster differs from raw ledger")
    labels = tables["labels"].set_index("customer_id", verify_integrity=True)
    require(ids.eq(labels.snapshot_id.reindex(ids.index)).fillna(False).all(), "Feature Store snapshot_id mismatch")
    weeks = pd.date_range(end=point-pd.Timedelta(weeks=1), periods=13, freq="7D")
    expected_snapshot = history.customer_id.map(ids)
    require(expected_snapshot.notna().all() and history.snapshot_id.notna().all()
            and expected_snapshot.eq(history.snapshot_id).all(), "CLV history snapshot_id mismatch")
    require(not history.duplicated(["customer_id", "reference_week"]).any()
            and history.reference_week.isin(weeks).all()
            and len(history) == len(ids)*13, "incomplete strict-past CLV history")


def _generate(root, config, count, seed, assignment):
    result = EcommerceSimulator(deepcopy(config)).generate(n_customers=count, weeks=104, seed=seed,
                                                          coupon_assignment=assignment)
    result.save(root / "dataset")
    del result
    directory = root / "features"
    materialize_features(root / "dataset", directory)
    features = load_feature_store(directory)
    manifest = _json(directory / "manifest.json")
    require(features.attrs["feature_store_evidence"]["raw_file_verified"]
            and manifest["feature_set_version"] == WEEKLY_FEATURE_VERSION
            and all(manifest[k] == v for k, v in weekly_metadata().items()), "unverified weekly store")
    path = directory / "clv_weekly_inputs.parquet"
    require(file_sha256(path) == manifest["file_sha256"][path.name], "CLV history checksum mismatch")
    history = pd.read_parquet(path)
    tables = load_tables(root / "dataset")
    meta = tables["_metadata"]
    point = pd.Timestamp(meta["as_of_at"])
    require(meta["seed"] == seed and len(features) == count
            and point - pd.Timestamp(meta["observation_start_at"]) == pd.Timedelta(weeks=52)
            and pd.Timestamp(meta["observation_end_at"]) - point == pd.Timedelta(weeks=52),
            "weekly generation inputs differ from request")
    _input_provenance(features, history, tables)
    generation_manifest = root / "dataset/manifest.json"
    generation = _json(generation_manifest)
    for field in ("dataset_id", "generation_fingerprint", "coupon_assignment", "session_completion"):
        require(generation[field] == meta[field], f"generation manifest {field} mismatch")
    _, _, requested_assignment = assign_coupons(tables["customers"].customer_id, assignment, seed=seed)
    require(meta["coupon_assignment"] == requested_assignment, "raw ledger assignment differs from requested execution")
    require(meta.get("session_completion") is not None, "missing producer session completion")
    completion_exclusions(meta, tables["sessions"].session_id)
    evidence = _generation_inputs(meta)
    evidence["feature_schema"] = str(parquet.read_schema(directory / "feature_store.parquet").remove_metadata())
    past = _past(tables, point)
    evidence["catalog_sha256"] = _fingerprint(past["products"].reset_index().to_dict("records"))
    evidence["ordered_roster_sha256"] = _fingerprint(past["customers"].reset_index().to_dict("records"))
    for name, report in (("coupon accounting", audit_coupon_usage(tables, meta)),
                         ("sidecars", audit_sidecars(tables, meta, tables["labels"], tables["treatments"]))):
        require(report["passed"], f"{name}: {report['mismatch_counts']}")
    # Rebuild from the audited ledger and sidecars independently of the published Parquet.
    source_log = build_experiment_log(tables, meta)
    path = directory / "experiment_log.parquet"
    require(file_sha256(path) == manifest["file_sha256"][path.name], "experiment log checksum mismatch")
    log = pd.read_parquet(path)
    origin = dict(dataset_id=meta["dataset_id"], raw_database_sha256=meta["raw_database_sha256"],
                  assignment=meta["coupon_assignment"], table_rows=meta["table_rows"],
                  dataset_manifest_sha256=file_sha256(generation_manifest),
                  raw_manifest_sha256=file_sha256(root / "dataset/raw.manifest.json"),
                  feature_manifest_sha256=file_sha256(directory / "manifest.json"),
                  experiment_log_sha256=manifest["file_sha256"][path.name])
    return past, features, history, evidence, origin, log, source_log


def _verify_origins(baseline, replay, selected):
    for field in ("dataset_id", "dataset_manifest_sha256", "raw_manifest_sha256"):
        require(baseline[field] != replay[field], f"replay reused baseline {field}")
    require(not selected or baseline["raw_database_sha256"] != replay["raw_database_sha256"],
            "nonempty replay reused baseline raw_database_sha256")
    before, after = baseline["assignment"], replay["assignment"]
    require(before["design"] == "no_new_coupon" and before["spec"] == {"mode": "none"}
            and before["randomized"] is False and before["treated_count"] == 0
            and after["design"] == "specified_customers" and after["spec"]["mode"] == "assign"
            and after["randomized"] is False and after["spec"]["customer_ids"] == selected
            and after["treated_count"] == len(selected), "replay assignment differs from execution origins")


def _outcomes(log, audit, origin, selected, source_log):
    key = "customer_id"
    require(not log[key].duplicated().any() and set(log[key]) == set(audit[key]), "outcome roster mismatch")
    log = log.set_index(key).sort_index()
    plan = audit.set_index(key).sort_index()
    requested = pd.Series(log.index.isin(selected), index=log.index)
    design = origin["assignment"]
    require(selected == sorted(plan.index[plan.send.eq(1).fillna(False)])
            and len(selected) == len(set(selected)), "request differs from frozen plan")
    require(design["design"] == "specified_customers" and design["randomized"] is False
            and sorted(design["spec"]["customer_ids"]) == selected, "unexpected assignment design")
    for field in ("treatment", "coupon_action", "assignment_probability"):
        require(log[field].notna().all() and log[field].eq(requested.astype(int)).all(), f"{field} mismatch")
    require(log.dataset_id.eq(origin["dataset_id"]).all(), "outcome dataset mismatch")
    for field in ("snapshot_id", "as_of_at", "currency", "amount_unit"):
        require(log[field].astype(str).equals(plan[field].astype(str)), f"outcome {field} mismatch")
    require(log.assigned_at.eq(log.as_of_at).all() and log.outcome_end_at.eq(log.as_of_at + pd.Timedelta(weeks=4)).all(),
            "invalid assignment or outcome window")
    sent = log.actual_sent
    require(sent.isin([True, False]).all() and not (sent & ~requested).any()
            and log.coupon_id.notna().equals(sent) and log.sent_at.notna().equals(sent)
            and log.loc[sent, "sent_at"].eq(log.loc[sent, "as_of_at"]).all()
            and log.loc[sent, "expires_at"].eq(log.loc[sent, "sent_at"] + pd.Timedelta(days=28)).all(),
            "coupon execution mismatch")
    observed = log.outcome_observed
    require(log.execution_status.isin(["sent", "not_sent", "failed", "not_requested"]).all()
            and log.execution_status.eq("sent").eq(sent).all()
            and log.execution_status.eq("not_requested").eq(~requested).all()
            and log.send_cost_minor.notna().all() and log.send_cost_minor.ge(0).all()
            and log.loc[~sent, "send_cost_minor"].eq(0).all(), "invalid factual execution or cost")
    facts = log[["factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"]]
    require(observed.isin([True, False]).all() and log.is_censored.notna().all()
            and log.is_censored.eq(~observed).all()
            and facts.loc[observed].notna().all().all() and facts.loc[~observed].isna().all().all()
            and log.label_known_at.notna().eq(observed).all()
            and log.loc[observed, "label_known_at"].eq(log.loc[observed, "outcome_end_at"]).all(),
            "invalid censored or mature outcomes")
    has_discount = observed | log.redeemed_in_outcome.fillna(False)
    require(log.redeemed_in_outcome.loc[observed].notna().all()
            and log.redeemed_in_outcome.notna().eq(has_discount).all()
            and log.outcome_redemption_discount_cost_minor.notna().eq(has_discount).all()
            and log.outcome_redemption_discount_cost_minor.dropna().ge(0).all(), "invalid realized discount")
    fields = list(OUTCOME_FIELDS)
    source = source_log.set_index(key, verify_integrity=True).sort_index()
    _same("experiment log versus raw ledger and sidecars", source[fields], log[fields])
    return log[fields].join(plan[["selected", "row_state", "model_exclusion_reason", "planned_sent_at"]]).reset_index()


def _verify_artifacts(root, receipt, outcome):
    _plan_manifest(root / "plan", receipt["plan_manifest_sha256"])
    for name in ("baseline", "replay"):
        origin = receipt[name]
        run = root / name
        for relative, key in (("dataset/raw.sqlite", "raw_database_sha256"),
                              ("dataset/raw.manifest.json", "raw_manifest_sha256"),
                              ("dataset/manifest.json", "dataset_manifest_sha256"),
                              ("features/manifest.json", "feature_manifest_sha256")):
            require(file_sha256(run / relative) == origin[key], f"changed {name}/{relative}")
        raw = _json(run / "dataset/raw.manifest.json")
        generation = _json(run / "dataset/manifest.json")
        feature = _json(run / "features/manifest.json")
        require(raw["database_sha256"] == feature["raw_database_sha256"] == origin["raw_database_sha256"]
                and raw["metadata"]["dataset_id"] == generation["dataset_id"] == feature["dataset_id"]
                == origin["dataset_id"] and raw["metadata"]["coupon_assignment"] == generation["coupon_assignment"]
                == origin["assignment"], f"{name} manifests disagree with execution origin")
        checksums = feature["file_sha256"]
        require(checksums["experiment_log.parquet"] == origin["experiment_log_sha256"],
                f"{name} experiment log manifest mismatch")
        for file in ("feature_store.parquet", "clv_weekly_inputs.parquet", "experiment_log.parquet"):
            require(file_sha256(run / "features" / file) == checksums[file], f"changed {name}/features/{file}")
        for source in ("labels", "treatments"):
            file = run / "dataset" / f"{source}.parquet"
            require(file_sha256(file) == generation["sidecar_sha256"][source], f"changed {name}/dataset/{file.name}")
    _verify_origins(receipt["baseline"], receipt["replay"], receipt["selected_ids"])
    require(_fingerprint(receipt["selected_ids"]) == receipt["request_sha256"], "changed request roster")
    require(file_sha256(outcome) == receipt["outcomes_sha256"], "changed outcomes.parquet")


def verify_replay_receipt(output):
    """Return a completed receipt after verifying its artifacts. Ignore partial runs."""
    root = Path(output).resolve()
    path = root / "receipt.json"
    require(path.is_file() and not (root / "failure.json").exists(), "no successful replay receipt")
    receipt = _json(path)
    require(receipt["status"] == "public_replay_verified" and receipt["outcomes_file"] == "outcomes.parquet",
            "no successful replay receipt")
    _verify_artifacts(root, receipt, root / "outcomes.parquet")
    return receipt


def run_policy_replay(output, *, prepare_inputs, seed, n_customers, mode, config=POPULATION_CONFIG):
    """Run one simulator-only verification. A used output path is never reused.

    prepare_inputs(features, clv_history) returns (retention/uplift/clv batch
    mapping, planning keyword arguments). Only the official engine selects IDs.
    Consumers should call verify_replay_receipt before using outcomes.parquet.
    An interrupted run can leave partial files without a success receipt.
    """
    require(mode == "simulation_verification", "explicit simulator verification mode required")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    stage = "baseline"
    receipt = dict(execution_id=uuid4().hex, mode=mode, dispatch_authorized=False,
                   internal_state_status="producer_verification_unavailable")
    try:
        settings = deepcopy(EcommerceSimulator(config).config)
        past, features, history, inputs, baseline, baseline_log, _ = _generate(
            root / "baseline", settings, n_customers, seed, {"mode": "none"})
        require(baseline["assignment"]["design"] == "no_new_coupon" and baseline_log.treatment.eq(0).all(),
                "baseline must be untreated")
        del baseline_log, _
        receipt.update(baseline=baseline, generation_inputs=inputs, customer_input="simulator_population",
                       n_customers=n_customers, weeks=104)
        stage = "planning"
        batches, options = prepare_inputs(features.copy(deep=True), history.copy(deep=True))
        handoff = adapt_weekly_predictions(features, **batches)
        with localcontext(Context(prec=200)):
            plan = build_strategy_decisions(handoff, **options)
            selected = _frozen_plan(plan, features, options)
            directory = write_outputs(plan, root / "plan")
        digest = file_sha256(directory / "manifest.json")
        _plan_manifest(directory, digest)
        summary = plan.budget_summary
        receipt.update(plan_manifest_sha256=digest, policy_id=summary["policy_id"],
                       policy_version=summary["policy_version"], configuration_fingerprint=summary["configuration_fingerprint"],
                       decision_fingerprint=summary["decision_fingerprint"], algorithm=summary["algorithm"],
                       accepted_for_planning=summary["accepted_for_planning"], budget=str(summary["budget"]),
                       expected_cost=str(summary["total_expected_cost"]), selected_ids=selected,
                       request_sha256=_fingerprint(selected), roster_sha256=_fingerprint(
                           plan.identity.sort_values("customer_id").to_dict("records")))
        stage = "replay"
        other, after, after_history, replay_inputs, replay, log, source_log = _generate(
            root / "replay", settings, n_customers, seed, {"mode": "assign", "customer_ids": selected})
        receipt["replay"] = replay
        stage = "public_pre_state"
        _verify_origins(baseline, replay, selected)
        for field in inputs:
            require(inputs[field] == replay_inputs[field], f"generation input differs: {field}")
        for name, frame in past.items():
            _same(name, frame, other[name])
        for name, first, second, keys in (
            ("features", features, after, ["customer_id"]),
            ("clv_history", history, after_history, ["customer_id", "reference_week"]),
        ):
            _same(name, first.drop(columns=["dataset_id", "raw_database_sha256"], errors="ignore").set_index(keys).sort_index(),
                  second.drop(columns=["dataset_id", "raw_database_sha256"], errors="ignore").set_index(keys).sort_index())
        receipt.update(public_pre_state="verified", pre_cutoff_rows={k: len(v) for k, v in past.items()},
                       feature_rows=len(features), clv_history_rows=len(history))
        stage = "outcomes"
        outcomes = _outcomes(log, plan.decision_audit, replay, selected, source_log)
        path = root / "outcomes.parquet"
        pending = root / "outcomes.pending.parquet"
        outcomes.to_parquet(pending, index=False)
        stage = "publication"
        completed = dict(receipt, status="public_replay_verified", outcomes_file=path.name,
                         outcomes_sha256=file_sha256(pending), customers=len(outcomes),
                         requested=len(selected), sent=int(outcomes.actual_sent.sum()),
                         observed=int(outcomes.outcome_observed.sum()))
        _verify_artifacts(root, completed, pending)
        pending.rename(path)
        completed["elapsed_seconds"] = perf_counter()-started
        _publish(root / "receipt.json", completed)
        return completed
    except Exception as exc:
        if stage == "public_pre_state":
            receipt["public_pre_state"] = "mismatch"
        _publish(root / "failure.json", dict(receipt, status="failed", stage=stage, reason=str(exc)))
        raise

