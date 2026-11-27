"""Simulator contract checks. Prediction values and declared costs are fixtures."""
import json
from shutil import copytree
from types import SimpleNamespace

import pandas as pd
import pytest
from simulate.v4.time_spec import WEEKLY_FEATURE_VERSION, weekly_metadata

from final_strategy.budget import Budget, CostMetadata
from final_strategy import policy_execution
from final_strategy.policy_execution import (
    _frozen_plan, _input_provenance, _outcomes, _past, _plan_manifest, _same, _verify_origins,
    run_policy_replay, verify_replay_receipt,
)
from final_strategy.policy_search import _decision_fingerprint
from simulate.v4.data.loader import load_tables
from simulate.v4.data.weekly import build_experiment_log
from simulate.v4.simulator.coupon_assignment import assign_coupons
from final_strategy.retention_strategy import COUPON_COLUMNS, REDEMPTION_COLUMNS, HistoryContext, HistoryPolicy
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.tests.test_weekly_planning import DECLARATIONS, REPORT, options, weekly
from final_strategy.tests.test_weekly_prediction_adapter import cases, published, features
from final_strategy.validation import CONTEXT_COLUMNS, JOIN_KEYS
from final_strategy.weekly_planning import build_strategy_decisions


def fixture_inputs(frame, clv_history):
    keys = frame[list(CONTEXT_COLUMNS)]
    first = keys.iloc[0]
    n = len(keys)
    assert len(clv_history) == 13*n and clv_history.reference_week.lt(first.as_of_at).all()
    assert not {"treatment", "future_revenue_12m", "churn"} & set(clv_history)
    def batch(target, unit, weeks, producer, **values):
        return SimpleNamespace(frame=keys.assign(input_cutoff_at=keys.as_of_at,
            prediction_end_at=keys.as_of_at + pd.Timedelta(weeks=weeks), **values),
            metadata=dict(model_version="fixture", target_definition_id=target, horizon_value=weeks,
                          horizon_unit="weeks", value_semantics=unit), producer_metadata=producer)
    # Explicit synthetic producer exclusion (not a factual simulator status).
    status = ["scored"]*(n-1) + ["no_recent_session"]
    retention = batch("future_no_session_28d_active28d", "probability", 4,
        dict(component="retention", policy_condition="observed_experiment_mix",
             status_counts={"scored":n-1, "no_recent_session":1}, excluded_rows=1, **weekly_metadata()),
        churn_prob=[.6]*(n-1)+[None], status=status, expanded_feature_set_version=WEEKLY_FEATURE_VERSION)
    uplift = batch("purchase_28d_randomized_coupon_v1", "probability_pair", 4,
        dict(component="uplift", input_contract=weekly_metadata(), development_overlap_allowed=False,
             training_overlap_customers=[], training={"dataset_id":"fixture-train","training_customers":[],
                                                      "selection_customers":[]}), p0=.2, p1=.5, uplift_score=.3)
    clv = batch("clv_paid_amount_52w_v1", "nonnegative_amount", 52,
        dict(producer="clv", model_version="fixture", policy_condition="control_no_coupon",
             inputs=dict(dataset_id=first.dataset_id, raw_database_sha256=first.raw_database_sha256),
             training_overlap={"allowed":False, "roles":{"train":0,"validation":0,"test":0,"unused":n}},
             gross_4w={"status":"not_trained","model_version":None}), predicted_clv_12m=list(range(101,101+n)))
    clv.audit = keys[list(JOIN_KEYS)].assign(clv_training_role="unused")
    clv.gross_4w = clv.gross_4w_metadata = None
    clv.gross_4w_status = "not_trained"
    kwargs = dict(profile=build_strategy_profiles(**DECLARATIONS).policies[1], reporting=REPORT,
        budget=Budget("4", first.currency, first.amount_unit), algorithm="greedy",
        eligibility=keys.assign(business_eligible=True, business_reason=None, history_complete=True),
        coupons=pd.DataFrame(columns=COUPON_COLUMNS), redemptions=pd.DataFrame(columns=REDEMPTION_COLUMNS),
        history_context=HistoryContext(first.dataset_id, first.raw_database_sha256, first.as_of_at,
                                      "fixture", "1", first.as_of_at), history_policy=HistoryPolicy(False, False, "0d"),
        cost_inputs=keys.assign(send_cost="1", redemption_probability=".5", expected_discount_if_redeemed="2",
                                source_id="fixture", source_version="1", source_as_of_at=keys.as_of_at),
        cost_metadata=CostMetadata("fixture-28d-request","1",2,first.as_of_at))
    return dict(retention=retention,uplift=uplift,clv=clv), kwargs


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    root = tmp_path_factory.mktemp("verification") / "six"
    receipt = run_policy_replay(root, prepare_inputs=fixture_inputs, seed=17, n_customers=6,
                                mode="simulation_verification")
    return root, receipt


def test_six_customer_weekly_replay(simulated):
    root, receipt = simulated
    assert json.loads((root/"receipt.json").read_text()) == receipt
    assert verify_replay_receipt(root) == receipt
    assert receipt["baseline"]["raw_database_sha256"] != receipt["replay"]["raw_database_sha256"]
    assert receipt["baseline"]["dataset_id"] != receipt["replay"]["dataset_id"]
    assert receipt["requested"] == receipt["sent"] == 2
    assert receipt["customers"] == receipt["observed"] == 6
    assert receipt["public_pre_state"] == "verified" and not receipt["dispatch_authorized"]
    assert receipt["internal_state_status"] == "producer_verification_unavailable"
    outcomes = pd.read_parquet(root/"outcomes.parquet")
    assert set(outcomes.loc[outcomes.actual_sent,"customer_id"]) == set(receipt["selected_ids"])
    assert outcomes.row_state.eq("model_unavailable").sum() == 1
    assert outcomes.factual_outcome_net_revenue_minor.notna().all()
    assert json.loads((root/"plan/budget_summary.json").read_text())["expected_roi"] is None
    with pytest.raises(FileExistsError):
        run_policy_replay(root, prepare_inputs=fixture_inputs, seed=17, n_customers=6, mode="simulation_verification")


def test_operational_dispatch_unavailable(tmp_path):
    with pytest.raises(ValueError, match="verification mode"):
        run_policy_replay(tmp_path/"unused", prepare_inputs=fixture_inputs, seed=17, n_customers=6, mode="dispatch")
    assert not (tmp_path/"unused").exists()


@pytest.mark.parametrize("field", ["configuration_fingerprint", "decision_fingerprint", "algorithm",
                                   "accepted_for_planning", "budget", "roster", "repeated", "delayed"])
def test_reject_invalid_execution_plan(weekly, field):
    args = options(weekly)
    plan = build_strategy_decisions(weekly, **args)
    identity = plan.identity.copy()
    assert _frozen_plan(plan, identity, args) == ["001"]
    if field in ("configuration_fingerprint", "decision_fingerprint", "algorithm"):
        plan.metadata["scenarios"]["budget_100"][field] = "altered"
    elif field == "accepted_for_planning":
        plan.budget_summary[field] = False
    elif field == "budget":
        args["budget"] = Budget("1", args["budget"].currency, args["budget"].amount_unit)
    elif field == "roster":
        identity = identity.iloc[:-1]
    elif field == "repeated":
        identity = pd.concat([identity, identity.iloc[:1]])
    else:
        audit = plan.decision_audit
        audit.loc[audit.send.eq(1), "planned_sent_at"] += pd.Timedelta(hours=1)
        digest = _decision_fingerprint(audit[audit.status.eq("scored")])
        plan.budget_summary["decision_fingerprint"] = digest
        plan.metadata["scenarios"]["budget_100"]["decision_fingerprint"] = digest
        audit["decision_fingerprint"] = digest
    with pytest.raises(ValueError, match="t0-only sends" if field == "delayed" else None):
        _frozen_plan(plan, identity, args)


def test_withheld_cost_and_unapproved_hybrid_plans(weekly):
    for override in (dict(cost_inputs=None, cost_metadata=None), dict(algorithm="hybrid")):
        args = options(weekly) | override
        plan = build_strategy_decisions(weekly, **args)
        with pytest.raises(ValueError, match="invalid or unapproved"):
            _frozen_plan(plan, plan.identity, args)


def test_manifest_and_artifact_changes(tmp_path, simulated):
    root, receipt = simulated
    directory = tmp_path / "copy"
    copytree(root / "plan", directory)
    digest = receipt["plan_manifest_sha256"]
    _plan_manifest(directory, digest)
    with pytest.raises(ValueError, match="changed plan manifest"):
        _plan_manifest(directory, "0"*64)
    with (directory / "decision_audit.csv").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="decision_audit.csv"):
        _plan_manifest(directory, digest)


@pytest.mark.parametrize("field", ["dataset_id", "raw_database_sha256", "dataset_manifest_sha256", "raw_manifest_sha256"])
def test_reused_run_identity(simulated, field):
    _, receipt = simulated
    replay = dict(receipt["replay"])
    replay[field] = receipt["baseline"][field]
    with pytest.raises(ValueError, match=field):
        _verify_origins(receipt["baseline"], replay, receipt["selected_ids"])


def test_public_values_nulls_dtypes_and_session_contract(simulated):
    root, _ = simulated
    tables = load_tables(root / "baseline/dataset")
    t0 = pd.Timestamp(tables["_metadata"]["as_of_at"])
    past = _past(tables, t0)
    events = past["events"].copy()
    events.iloc[0, events.columns.get_loc("event_type")] = None
    with pytest.raises(ValueError, match="events.event_type"):
        _same("events", past["events"], events)
    feature = pd.DataFrame({"value": pd.array([0, None], dtype="Int64")}, index=["001", "002"])
    with pytest.raises(ValueError, match="002"):
        _same("features", feature, feature.fillna(0))
    with pytest.raises(ValueError, match="dtype"):
        _same("features", feature, feature.astype("Float64"))
    tables["_metadata"]["session_completion"]["clock"] = "unverified"
    with pytest.raises(ValueError, match="completion contract"):
        _past(tables, t0)


@pytest.fixture
def execution_rows(simulated):
    root, receipt = simulated
    log = pd.read_parquet(root / "replay/features/experiment_log.parquet")
    audit = pd.read_parquet(root / "outcomes.parquet")
    audit["send"] = audit.selected.fillna(0)
    return log, audit, receipt["replay"], receipt["selected_ids"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown_request", "propensity", "coupon", "zero_censored"])
def test_invalid_outcome_evidence(execution_rows, defect):
    log, audit, origin, ids = execution_rows
    source = log.copy()
    sent = log.index[log.actual_sent][0]
    if defect == "missing":
        log = log.iloc[:-1]
    elif defect == "duplicate":
        log = pd.concat([log, log.iloc[:1]])
    elif defect == "unknown_request":
        ids = [*ids, "unknown"]
    elif defect == "propensity":
        log["assignment_probability"] = .5
    elif defect == "coupon":
        log.loc[sent, "coupon_id"] = None
    else:
        log.loc[sent, ["outcome_observed", "is_censored"]] = [False, True]
        log.loc[sent, "label_known_at"] = pd.NaT
        log.loc[sent, "factual_outcome_net_revenue_minor"] = 0
    with pytest.raises(ValueError):
        _outcomes(log, audit, origin, ids, source)


def test_failed_and_censored_fixture_keeps_assignment_and_unknowns(execution_rows):
    log, audit, origin, ids = execution_rows
    i = log.index[log.actual_sent][0]
    log.loc[i, ["sent_at", "expires_at", "label_known_at"]] = pd.NaT
    log.loc[i, ["coupon_id", "redeemed_in_outcome", "outcome_redemption_discount_cost_minor",
                "factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"]] = pd.NA
    log.loc[i, ["execution_status", "failure_reason"]] = ["failed", "synthetic rejection"]
    log.loc[i, ["actual_sent", "outcome_observed", "is_censored", "send_cost_minor"]] = [False, False, True, 0]
    # This fixture exercises the final projection. Native generation always
    # sends the assigned coupon and supplies complete follow-up.
    outcome = _outcomes(log, audit, origin, ids, log.copy())
    row = outcome.set_index("customer_id").loc[log.loc[i, "customer_id"]]
    assert len(outcome) == len(audit) and row.coupon_action == row.selected == 1
    assert not row.actual_sent and row.is_censored and row.failure_reason == "synthetic rejection"
    assert pd.isna(row.factual_outcome_net_revenue_minor) and pd.isna(row.outcome_redemption_discount_cost_minor)


def test_pre_state_mismatch_publishes_only_failure(tmp_path, monkeypatch):
    generate = policy_execution._generate

    def changed_feature(root, config, count, seed, assignment):
        result = generate(root, config, count, seed, assignment)
        if assignment["mode"] == "assign":
            frame = result[1]
            feature = next(column for column in frame if column not in policy_execution.FEATURE_KEYS)
            frame.loc[frame.index[0], feature] = 123456
        return result

    monkeypatch.setattr(policy_execution, "_generate", changed_feature)
    root = tmp_path / "changed"
    with pytest.raises(ValueError, match=r"features\..*U0000001"):
        run_policy_replay(root, prepare_inputs=fixture_inputs, seed=17, n_customers=6, mode="simulation_verification")
    assert not (root / "receipt.json").exists() and not (root / "outcomes.parquet").exists()
    failure = json.loads((root / "failure.json").read_text())
    assert (failure["stage"], failure["public_pre_state"], failure["status"]) == ("public_pre_state", "mismatch", "failed")


@pytest.mark.parametrize(("field", "value", "sent"), [
    ("factual_outcome_net_revenue_minor", 73, False),
    ("outcome_redemption_discount_cost_minor", 23, False),
    ("send_cost_minor", 15, True),
    ("decision_action", "send_coupon", False),
])
def test_outcome_disagrees_with_source(execution_rows, field, value, sent):
    log, audit, origin, ids = execution_rows
    changed = log.copy()
    i = log.index[log.actual_sent.eq(sent)][0]
    uid = log.loc[i, "customer_id"]
    changed.loc[i, field] = value
    with pytest.raises(ValueError, match=rf"{field}.*{uid}.*expected.*got"):
        _outcomes(changed, audit, origin, ids, log)


def test_zero_budget_executes_real_empty_assignment(tmp_path):
    def no_budget(frame, history):
        batches, options = fixture_inputs(frame, history)
        previous = options["budget"]
        options["budget"] = Budget("0", previous.currency, previous.amount_unit)
        return batches, options

    root = tmp_path / "zero"
    receipt = run_policy_replay(root, prepare_inputs=no_budget, seed=17, n_customers=6,
                                mode="simulation_verification")
    assert receipt["selected_ids"] == []
    assert receipt["baseline"]["assignment"]["spec"] == {"mode": "none"}
    assert receipt["replay"]["assignment"]["spec"]["customer_ids"] == []
    assert receipt["baseline"]["dataset_id"] != receipt["replay"]["dataset_id"]
    assert receipt["baseline"]["raw_database_sha256"] == receipt["replay"]["raw_database_sha256"]
    assert receipt["requested"] == receipt["sent"] == 0 and receipt["observed"] == 6
    outcome = pd.read_parquet(root / "outcomes.parquet")
    assert len(outcome) == 6 and not outcome.actual_sent.any()
    assert outcome.assignment_probability.eq(0).all()
    assert json.loads((root/"plan/budget_summary.json").read_text())["expected_roi"] is None
    assert verify_replay_receipt(root) == receipt


@pytest.mark.parametrize(("table", "field", "value", "error"), [
    ("features", "dataset_id", "other", "Feature Store dataset_id"),
    ("features", "raw_database_sha256", "other", "Feature Store raw_database_sha256"),
    ("features", "currency", "other", "Feature Store currency"),
    ("features", "amount_unit", "1", "Feature Store amount_unit"),
    ("features", "snapshot_id", "other", "Feature Store snapshot_id"),
    ("history", "dataset_id", "other", "CLV history dataset_id"),
    ("history", "snapshot_id", "other", "CLV history snapshot_id"),
    ("history", "as_of_at", pd.Timestamp("2025-01-01", tz="UTC"), "CLV history as_of_at"),
    ("history", "future_revenue_12m", 1, "CLV history schema"),
    ("history", "unknown", 1, "CLV history schema"),
    ("history", "sessions", None, "CLV history schema"),
])
def test_published_inputs_bind_to_raw_ledger(source_inputs, table, field, value, error):
    original, weeks, tables = source_inputs
    features, history = original.copy(), weeks.copy()
    _input_provenance(features, history, tables)
    frame = features if table == "features" else history
    if value is None:
        frame.drop(columns=field, inplace=True)
    else:
        frame[field] = value
    with pytest.raises(ValueError, match=error):
        _input_provenance(features, history, tables)


@pytest.fixture(scope="module")
def source_inputs(simulated):
    root, _ = simulated
    baseline = root / "baseline"
    tables = load_tables(baseline / "dataset")
    features = policy_execution.load_feature_store(baseline / "features")
    history = pd.read_parquet(baseline / "features/clv_weekly_inputs.parquet")
    return features, history, tables


def test_official_positive_purchase_discount_and_send_cost():
    t0 = pd.Timestamp("2024-01-07T15:00:00Z")
    purchase_at = t0 + pd.Timedelta(seconds=1)
    tables = {
        "customers": pd.DataFrame([dict(customer_id="u", signed_up_at=t0-pd.Timedelta(days=30))]),
        "sessions": pd.DataFrame([dict(session_id="s", customer_id="u", started_at=t0)]),
        "events": pd.DataFrame([dict(event_id=1, customer_id="u", session_id="s", event_at=purchase_at,
                                     event_type="purchase")]),
        "orders": pd.DataFrame([dict(order_id="o", purchase_event_id=1, coupon_id="coupon",
                                     gross_minor=1065, discount_minor=25, paid_minor=1040)]),
        "coupons": pd.DataFrame([dict(coupon_id="coupon", customer_id="u", sent_at=t0,
                                      expires_at=t0 + pd.Timedelta(days=28), discount_rate=.1, send_cost_minor=1)]),
    }
    _, _, design = assign_coupons(["u"], {"mode": "assign", "customer_ids": ["u"]})
    metadata = dict(dataset_id="positive-amount-fixture", seed=42, currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
                    as_of_at=t0.isoformat(), observation_start_at=(t0-pd.Timedelta(days=40)).isoformat(),
                    observation_end_at=(t0+pd.Timedelta(days=60)).isoformat(),
                    coupon_assignment=design, outcome_window_days=28, coupon_validity_days=28, eligibility_window_days=28)
    tables["treatments"] = pd.DataFrame([dict(customer_id="u", as_of_at=t0, assigned_at=t0, treatment=1,
                                             assignment_probability=1., coupon_id="coupon")])
    tables["labels"] = pd.DataFrame([dict(customer_id="u", as_of_at=t0, snapshot_id="fixture",
                                          outcome_end_at=t0 + pd.Timedelta(days=28), is_censored=False)])
    source = build_experiment_log(tables, metadata)
    audit = source.assign(selected=1, send=1, row_state="selected", model_exclusion_reason=None,
                          planned_sent_at=t0)
    origin = dict(dataset_id=metadata["dataset_id"], assignment=design)
    outcome = _outcomes(source.copy(), audit, origin, ["u"], source).iloc[0]
    assert outcome.redeemed_in_outcome and not outcome.factual_no_order
    for field, value in (("factual_outcome_net_revenue_minor", 1040),
                         ("outcome_redemption_discount_cost_minor", 25), ("send_cost_minor", 1)):
        assert outcome[field] == value
        with pytest.raises(ValueError, match=rf"{field}.*expected .*{value}.*got .*{value+1}"):
            _outcomes(source.assign(**{field: pd.array([value+1], dtype="Int64")}), audit, origin, ["u"], source)


@pytest.mark.parametrize("relative", ["outcomes.parquet", "baseline/features/feature_store.parquet",
                                     "replay/features/clv_weekly_inputs.parquet", "replay/dataset/raw.sqlite",
                                     "plan/decision_audit.csv"])
def test_receipt_consumer_rejects_mutated_bytes(simulated, tmp_path, relative):
    root, _ = simulated
    copy = tmp_path / "copy"
    copytree(root, copy)
    with (copy / relative).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match=relative.rsplit("/", 1)[-1].replace(".", r"\.")):
        verify_replay_receipt(copy)


def test_valid_feature_store_from_other_run_is_rejected(simulated, source_inputs):
    root, _ = simulated
    features = policy_execution.load_feature_store(root / "replay/features")
    _, history, tables = source_inputs
    with pytest.raises(ValueError, match="Feature Store dataset_id"):
        _input_provenance(features, history, tables)


@pytest.fixture
def use_saved_runs(simulated, monkeypatch):
    saved, _ = simulated
    class SavedRun:
        def save(self, directory):
            copytree(saved / directory.parent.name / "dataset", directory)
    monkeypatch.setattr(policy_execution.EcommerceSimulator, "generate", lambda *a, **kw: SavedRun())

    def publish(dataset, output):
        copytree(saved / dataset.parent.name / "features", output)
    monkeypatch.setattr(policy_execution, "materialize_features", publish)


@pytest.mark.parametrize(("file", "column", "stage", "message"), [
    ("clv_weekly_inputs.parquet", "future_revenue_12m", "baseline", "CLV history schema"),
    ("experiment_log.parquet", "factual_outcome_net_revenue_minor", "outcomes", "raw ledger.*U0000001"),
])
def test_public_api_rejects_bound_but_incorrect_artifact(use_saved_runs, tmp_path, monkeypatch,
                                                        file, column, stage, message):
    publish = policy_execution.materialize_features
    def change_export(dataset, output):
        publish(dataset, output)
        if dataset.parent.name == ("baseline" if stage == "baseline" else "replay"):
            path = output / file
            frame = pd.read_parquet(path)
            frame[column] = 73 if stage == "baseline" else frame[column].mask(frame.customer_id.eq("U0000001"), 73)
            frame.to_parquet(path, index=False)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["file_sha256"][file] = policy_execution.file_sha256(path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(policy_execution, "materialize_features", change_export)
    calls = []
    def prepare(frame, history):
        calls.append(True)
        return fixture_inputs(frame, history)

    root = tmp_path / "malformed"
    with pytest.raises(ValueError, match=message):
        run_policy_replay(root, prepare_inputs=prepare, seed=17, n_customers=6, mode="simulation_verification")
    assert len(calls) == (0 if stage == "baseline" else 1)
    assert json.loads((root / "failure.json").read_text())["stage"] == stage
    assert not (root / "receipt.json").exists() and not (root / "outcomes.parquet").exists()


def test_publication_rejects_changed_clv_bytes(use_saved_runs, tmp_path, monkeypatch):
    verify = policy_execution._verify_artifacts
    def corrupt_clv(root, receipt, path):
        assert path.is_file()
        with (root / "baseline/features/clv_weekly_inputs.parquet").open("ab") as stream:
            stream.write(b"changed")
        verify(root, receipt, path)

    monkeypatch.setattr(policy_execution, "_verify_artifacts", corrupt_clv)
    root = tmp_path / "partial"
    with pytest.raises(ValueError, match="changed baseline/features/clv_weekly_inputs.parquet"):
        run_policy_replay(root, prepare_inputs=fixture_inputs, seed=17, n_customers=6, mode="simulation_verification")
    assert (root / "outcomes.pending.parquet").is_file()
    assert not (root / "outcomes.parquet").exists() and not (root / "receipt.json").exists()
    assert json.loads((root / "failure.json").read_text())["stage"] == "publication"
    with pytest.raises(ValueError, match="no successful replay receipt"):
        verify_replay_receipt(root)

