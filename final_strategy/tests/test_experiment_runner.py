"""Five-arm Simulator contract checks with explicitly synthetic predictions."""
import json
from decimal import Decimal
from hashlib import sha256
from shutil import copytree

import pandas as pd
import pytest

from final_strategy import experiment_runner as runner
from final_strategy.experiment_runner import (
    _approved, _assign, _ledgers, _population_ledgers, _random_selection, _selection_union,
    run_experiment, verify_experiment_receipt,
)
from final_strategy.experiment_spec import ExperimentSpec
from final_strategy.tests.test_policy_execution import fixture_inputs
from final_strategy.experiment_validation import CONTEXT, LEDGERS, validate_experiment_run

SPEC = ExperimentSpec("fixture-five-arm", "1", (1, 1, 1, 1, 1))


def predictions(features, history):
    batches, options = fixture_inputs(features, history)
    # Exercise signed and zero scores under an independently valid Uplift batch.
    uplift = batches["uplift"].frame
    for ids, p1 in ((["U0000002", "U0000009"], .1), (["U0000003", "U0000010"], .2)):
        mask = uplift.customer_id.isin(ids)
        uplift.loc[mask, "p1"] = p1
        uplift.loc[mask, "uplift_score"] = p1 - .2
    costs = options["cost_inputs"]
    costs.loc[costs.index[::2], "send_cost"] = "3"
    return batches, options


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("five") / "run"
    receipt = run_experiment(root, spec=SPEC, prepare_inputs=predictions, n_customers=20,
                             simulator_seed=17, assignment_seed=21, random_seed=29,
                             budget_per_customer="1.5", mode="simulation_verification")
    return root, receipt


def test_real_twenty_customer_experiment(experiment):
    root, receipt = experiment
    assert verify_experiment_receipt(root) == receipt
    assert receipt["public_pre_state"] == "verified"
    assert receipt["internal_state_status"] == "producer_verification_unavailable"
    assert not receipt["confirmatory_approved"] and not receipt["dispatch_authorized"]
    assert receipt["analysis_status"] == "withheld"
    assert {arm: values["itt"] for arm, values in receipt["by_arm"].items()} == dict.fromkeys(("A","B","C","D1","D2"), 4)
    assert receipt["by_arm"]["A"]["sent"] == 0
    assignments = pd.read_parquet(root / "ledgers/assignments.parquet")
    factual = pd.read_parquet(root / "execution_outcomes.parquet")
    assert assignments.assignment_probability.eq(.2).all()
    assert factual.assignment_probability.isin([0, 1]).all()
    assert set(factual.loc[factual.actual_sent, "customer_id"]) == set(receipt["selected_ids"])
    assert len(assignments) == len(factual) == 20
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["model_exclusions"]) == 1
    assert manifest["arms"]["D1"]["classification_reference_sha256"] == manifest["arms"]["D2"]["classification_reference_sha256"]
    assert all(arm["budget"] in ("0", "6.0") for arm in manifest["arms"].values())
    for arm in ("D1", "D2"):
        official = json.loads((root / "plans" / arm / "budget_summary.json").read_text())
        assert official["decision_fingerprint"] == manifest["arms"][arm]["decision_fingerprint"]
        assert official["algorithm"] == manifest["arms"][arm]["solver_kind"]
    c_plan = pd.read_parquet(root / "plans/C/decisions.parquet")
    assert c_plan.loc[c_plan.uplift_score.le(0), "candidate"].eq(False).all()
    assert manifest["arms"]["C"]["solver_summary"]["algorithm"] == "milp"
    for name, columns in LEDGERS.items():
        frame = pd.read_parquet(root / "ledgers" / f"{name}.parquet")
        assert list(frame) == [*CONTEXT, *columns]
        expected = manifest["result_raw_sha256" if name in ("executions", "outcomes") else "feature_raw_sha256"]
        assert frame.raw_database_sha256.eq(expected).all()
    executions = pd.read_parquet(root / "ledgers/executions.parquet")
    assert executions.loc[executions.actual_sent, "discount_rate"].eq(.1).all()
    assert executions.loc[~executions.actual_sent, "discount_rate"].isna().all()
    analysis = json.loads((root / "analysis.json").read_text())
    assert analysis["causal_interpretation"] == "unverified"
    assert analysis["validation"]["design_status"] == "planning_unapproved"


def test_assignment_seed_quotas_and_stable_order():
    ids = [f"u{i:02}" for i in range(12)]
    assigned = _assign(ids, SPEC, 21)
    assert assigned == _assign(ids[::-1], SPEC, 21)
    assert assigned != _assign(ids, SPEC, 22)
    for arm, count in zip(("A", "B", "C", "D1", "D2"), (3, 3, 2, 2, 2)):
        assert [p for a, p in assigned.values() if a == arm] == [count / len(ids)] * count
    with pytest.raises(ValueError, match="duplicate eligible"):
        _assign(ids + [ids[0]], SPEC, 21)


def test_random_budget_scan_skips_expensive_customer():
    candidates = pd.DataFrame({"customer_id": ["001", "002", "003"]})
    _, order = _random_selection(candidates.assign(expected_cost=0), Decimal(4), 0, 11)
    costs = dict(zip(order, [6, 2, 2]))
    selected, second_order = _random_selection(candidates.assign(expected_cost=candidates.customer_id.map(costs)),
                                               Decimal(4), 0, 11)
    assert second_order == order and selected == set(order[1:])


@pytest.mark.parametrize("error", ["duplicate", "missing", "move", "A"])
def test_selected_roster_rejects_invalid_arm_evidence(error):
    assignment = {"u": ("A", .5), "v": ("B", .5)}
    decision = pd.DataFrame({"customer_id": ["u", "v"], "arm_id": ["A", "B"], "send": [False, True]})
    assert _selection_union(decision, assignment) == ["v"]
    assert _selection_union(decision.assign(send=False), assignment) == []
    if error == "duplicate":
        decision = pd.concat([decision, decision.iloc[1:]])
    elif error == "missing":
        decision = decision.iloc[1:]
    elif error == "move":
        decision.loc[1, "arm_id"] = "C"
    else:
        decision.loc[0, "send"] = True
    with pytest.raises(ValueError, match="duplicate or missing|moved customer|No Action"):
        _selection_union(decision, assignment)


@pytest.mark.parametrize("field", ["has_valid_solution", "accepted_for_planning", "budget_compliant"])
def test_milp_plan_must_be_approved(field):
    summary = dict(algorithm="milp", status="fixture", has_valid_solution=True, accepted_for_planning=True,
                   budget_compliant=True, optimality="numeric_optimal")
    _approved(summary, "D2")
    summary[field] = False
    with pytest.raises(ValueError, match="D2: invalid or unapproved"):
        _approved(summary, "D2")


def test_consumer_rejects_partial_and_tampered_ledgers(experiment, tmp_path):
    root, _ = experiment
    with pytest.raises(ValueError, match="no successful experiment receipt"):
        verify_experiment_receipt(tmp_path)
    duplicate = tmp_path / "copy"
    copytree(root, duplicate)
    with (duplicate / "ledgers/assignments.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed experiment artifact: ledgers/assignments.parquet"):
        verify_experiment_receipt(duplicate)


def test_censored_contract_fixture_keeps_assignment_and_null(experiment):
    root, receipt = experiment
    manifest = json.loads((root / "manifest.json").read_text())
    load = lambda name: pd.read_parquet(root / "ledgers" / f"{name}.parquet")
    population, assigned, decisions = (load(name) for name in ("population", "assignments", "decisions"))
    decisions = decisions.rename(columns={"evaluation_dataset_id": "dataset_id", "send_requested": "send"})
    factual = pd.read_parquet(root / "execution_outcomes.parquet")
    index = factual.index[~factual.actual_sent][0]
    factual["censoring_reason"] = None
    factual.loc[index, "censoring_reason"] = "fixture interrupted observation"
    factual.loc[index, "outcome_observed"] = False
    factual.loc[index, ["factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor",
                        "outcome_redemption_discount_cost_minor"]] = pd.NA
    ledgers = _ledgers(population, assigned, decisions, factual, receipt["baseline"], receipt["replay"])
    observed = ledgers["outcomes"].set_index("customer_id").loc[factual.loc[index, "customer_id"]]
    assert not observed.outcome_observed
    assert observed[["purchase_occurred", "session_occurred", "paid_minor", "coupon_discount_minor"]].isna().all()
    validated = validate_experiment_run(SPEC, manifest, **ledgers)
    assert validated.by_arm[observed.arm_id]["censored"] == 1
    assert validated.by_arm[observed.arm_id]["itt"] == 4


def test_population_records_preassignment_exclusion(experiment):
    root, _ = experiment
    full = pd.read_parquet(root / "ledgers/population.parquet")
    features = full.rename(columns={"evaluation_dataset_id": "dataset_id"})
    excluded = full.customer_id.iloc[0]
    eligibility = full[["customer_id"]].assign(business_reason=None)
    eligibility.loc[eligibility.customer_id.eq(excluded), "business_reason"] = "preassignment fixture"
    assignments = _assign(full.loc[full.customer_id.ne(excluded), "customer_id"], SPEC, 21)
    population, assignment = _population_ledgers(features, eligibility, assignments)
    row = population.set_index("customer_id").loc[excluded]
    assert len(population) == 20 and len(assignment) == 19
    assert not row.eligible and row.exclusion_reason == "preassignment fixture"
    assert excluded not in set(assignment.customer_id)


def test_real_zero_selection_reuses_raw_bytes_but_keeps_run_identity(tmp_path):
    root = tmp_path / "zero"
    receipt = run_experiment(root, spec=SPEC, prepare_inputs=fixture_inputs, n_customers=6,
                             simulator_seed=17, assignment_seed=21, random_seed=29,
                             budget_per_customer=0, mode="simulation_verification")
    assert verify_experiment_receipt(root) == receipt
    assert receipt["selected_ids"] == []
    assert receipt["baseline"]["dataset_id"] != receipt["replay"]["dataset_id"]
    assert receipt["baseline"]["raw_database_sha256"] == receipt["replay"]["raw_database_sha256"]
    assert sum(arm["itt"] for arm in receipt["by_arm"].values()) == 6
    assert all(arm["sent"] == 0 for arm in receipt["by_arm"].values())


def test_unequal_assignment_weights_preserve_exact_quotas():
    spec = ExperimentSpec("weighted", "1", (1, 2, 2, 3, 4))
    ids = [f"u{i}" for i in range(17)]
    result = _assign(ids, spec, 42)
    assert result == _assign(reversed(ids), spec, 42)
    for arm, count in zip(("A", "B", "C", "D1", "D2"), (1, 3, 3, 4, 6)):
        assert [p for group, p in result.values() if group == arm] == [count / 17] * count


def _update_bundle(root, name, *, frame=None, document=None):
    path = root / name
    if frame is not None:
        frame.to_parquet(path, index=False)
    else:
        path.write_text(json.dumps(document), encoding="utf-8")
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"][name] = sha256(path.read_bytes()).hexdigest()
    if name.startswith("ledgers/"):
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["ledger_sha256"][path.stem] = receipt["files"][name]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        receipt["files"]["manifest.json"] = sha256(manifest_path.read_bytes()).hexdigest()
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest.mark.parametrize(("table", "column", "uid", "value"), [
    ("outcomes", "paid_minor", "U0000018", 1959),
    ("outcomes", "coupon_discount_minor", "U0000018", 23),
    ("outcomes", "purchase_occurred", "U0000018", False),
    ("executions", "send_cost_minor", "U0000018", 101),
    ("executions", "actual_sent", "U0000018", False),
    ("assignments", "arm_id", "U0000018", "D2"),
    ("decisions", "send_requested", "U0000018", False),
    ("decisions", "selected", "U0000018", False),
    ("population", "eligible", "U0000018", False),
])
def test_rehashed_ledger_must_match_original_sources(experiment, tmp_path, table, column, uid, value):
    saved, _ = experiment
    root = tmp_path / "changed"
    copytree(saved, root)
    name = f"ledgers/{table}.parquet"
    frame = pd.read_parquet(root / name)
    frame.loc[frame.customer_id.eq(uid), column] = value
    _update_bundle(root, name, frame=frame)
    with pytest.raises(ValueError, match=rf"{table}.*{column}.*{uid}"):
        verify_experiment_receipt(root)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("public_pre_state", "mismatch", "public pre-state"),
    ("experiment_id", "wrong", "inconsistent experiment identity"),
    ("experiment_version", "wrong", "inconsistent experiment identity"),
    ("spec_fingerprint", "wrong", "manifest, specification"),
    ("seeds", {"simulator": 17, "strategy_assignment": 22, "random_policy": 29}, "inconsistent experiment identity"),
    ("selected_ids", [], "inconsistent experiment identity"),
])
def test_inconsistent_receipt_state_is_rejected(experiment, tmp_path, field, value, message):
    saved, _ = experiment
    root = tmp_path / "changed"
    copytree(saved, root)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt[field] = value
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_experiment_receipt(root)


@pytest.mark.parametrize("case", ["missing_reference", "reference_class", "plan_class", "plan_selection",
                                 "reference_config", "source_raw", "analysis"])
def test_rehashed_artifact_cross_checks(experiment, tmp_path, case):
    saved, _ = experiment
    root = tmp_path / "changed"
    copytree(saved, root)
    reference_name = "segmentation_reference.parquet"
    manifest = json.loads((root / "manifest.json").read_text())
    if case in ("missing_reference", "reference_class"):
        reference = pd.read_parquet(root / reference_name)
        if case == "missing_reference":
            reference = reference.iloc[1:]
            message = "common Segmentation roster"
        else:
            reference.loc[reference.index[0], "clv_band"] = "altered"
            message = "official common Segmentation reference differs"
        _update_bundle(root, reference_name, frame=reference)
        manifest["segmentation_reference"]["sha256"] = sha256((root / reference_name).read_bytes()).hexdigest()
        _update_bundle(root, "manifest.json", document=manifest)
    elif case in ("plan_class", "plan_selection"):
        name = "plans/D1/decisions.parquet"
        plan = pd.read_parquet(root / name)
        if case == "plan_class":
            plan.loc[plan.index[0], "segment_id"] = "altered"
            message = "official decision_audit.csv.*segment_id"
        else:
            plan.loc[plan.index[0], "selected"] = 0
            message = "decision fingerprint differs"
        _update_bundle(root, name, frame=plan)
    elif case == "analysis":
        analysis = json.loads((root / "analysis.json").read_text())
        analysis["primary_comparisons"][0]["absolute_difference"] = .99
        _update_bundle(root, "analysis.json", document=analysis)
        message = "analysis differs from revalidated ledgers"
    else:
        if case == "reference_config":
            manifest["segmentation_reference"]["config"]["high_clv_fraction"] = .7
            message = "official common Segmentation reference differs"
        else:
            manifest["result_raw_sha256"] = "a" * 64
            message = "generation or ledger provenance differs"
        _update_bundle(root, "manifest.json", document=manifest)
    with pytest.raises(ValueError, match=message):
        verify_experiment_receipt(root)


@pytest.fixture
def saved_simulator(experiment, monkeypatch):
    saved, _ = experiment

    def generate(target, config, count, seed, assignment):
        copytree(saved / target.name, target)
        return runner._read_generation(target, count, seed, assignment)

    monkeypatch.setattr(runner, "_generate", generate)


@pytest.mark.parametrize(("failure", "stage", "message"), [
    ("C", "planning", "C: invalid or unapproved plan"),
    ("D2", "planning", "D2: invalid or unapproved plan"),
    ("pre_state", "public_pre_state", "features.*values differ"),
    ("source", "outcomes", "factual_outcome_net_revenue_minor.*U0000018"),
    ("publication", "publication", "analysis differs from revalidated ledgers"),
])
def test_public_execution_does_not_publish_incomplete_result(saved_simulator, tmp_path, monkeypatch,
                                                             failure, stage, message):
    if failure in ("C", "D2"):
        approved = runner._approved
        def fail_approval(summary, arm):
            approved(summary | {"accepted_for_planning": False} if arm == failure else summary, arm)
        monkeypatch.setattr(runner, "_approved", fail_approval)
    elif failure in ("pre_state", "source"):
        generate = runner._generate
        def alter_replay(target, config, count, seed, assignment):
            parts = generate(target, config, count, seed, assignment)
            if target.name == "replay":
                if failure == "pre_state":
                    parts[1].loc[parts[1].index[0], "visit_recency_days"] = 777
                else:
                    parts[5].loc[parts[5].customer_id.eq("U0000018"), "factual_outcome_net_revenue_minor"] = 1959
            return parts
        monkeypatch.setattr(runner, "_generate", alter_replay)
    else:
        publish = runner._publish
        def alter_analysis(path, payload):
            if path.name == "analysis.json":
                payload["primary_comparisons"][0]["absolute_difference"] = .99
            publish(path, payload)
        monkeypatch.setattr(runner, "_publish", alter_analysis)
    root = tmp_path / "failed"
    with pytest.raises(ValueError, match=message):
        run_experiment(root, spec=SPEC, prepare_inputs=predictions, n_customers=20, simulator_seed=17,
                       assignment_seed=21, random_seed=29, budget_per_customer="1.5", mode="simulation_verification")
    assert json.loads((root / "failure.json").read_text())["stage"] == stage
    assert not (root / "receipt.json").exists()
    if failure in ("C", "D2"):
        assert not (root / "replay").exists()
    with pytest.raises(ValueError, match="no successful experiment receipt"):
        verify_experiment_receipt(root)
