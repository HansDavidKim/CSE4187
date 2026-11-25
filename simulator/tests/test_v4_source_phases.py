from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import fit_latent_shopping as shopping, source_phases as phases

MEMORY = phases.LIFECYCLE_MEMORY
AVAILABLE = ("view", "cart")


def original_rows():
    rows = []
    for sid, day, actions in ((1, 0, ("remove",)), (2, 35, ("view",)),
                              (3, 40, ("view", "cart")), (4, 41, ("view", "cart"))):
        for step, action in enumerate(actions):
            rows.append(dict(user_id="100", session=sid, action=action, product_id="p", price_minor=100,
                             at=pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(days=day, seconds=step),
                             source_event_id=f"source:{len(rows):09d}"))
    return pd.DataFrame(rows)


def phase_panel():
    vectors, records = [], []
    for number in (1, 3):
        for step, action in enumerate((0, 1, 4)):
            context = dict(previous_action=("start", "view", "cart")[step], session_decision_count=step,
                           previous_action_run_length=int(step > 0), cart_total_quantity=int(step == 2))
            vectors.append(shopping.context_vector(context))
            records.append(dict(user_id="100", session=number, step=step, action=action,
                                at=pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(days=number, seconds=step),
                                observed_session_number=number))
    return {"features": np.asarray(vectors), "records": pd.DataFrame(records),
            "dynamics_usable": np.asarray([True, True, False] * 2),
            "starts": np.asarray([True, False, False] * 2), "weights": np.ones(6), "encoder_weights": np.ones(6)}


def phase_model():
    panel = phase_panel()
    encoder = shopping.fit_encoder(panel, minimum_leaf=1, available_actions=AVAILABLE, lifecycle_memory=MEMORY)
    matrices, mask, _states, audit = shopping.fit_dynamics(encoder, panel)
    keys = np.asarray(encoder["state_keys"])
    n = len(keys)
    base = np.zeros((n, 4))
    base[keys[:, 2] == 0, 0] = .6
    base[keys[:, 2] == 1, 1] = .6
    offsets = np.asarray([[.1, -.1, 0., 0.], [-.1, .1, 0., 0.]])
    policies, values = [], []
    for offset in offsets:
        probability, value, _ = shopping.soft_policy(shopping._reward(base + offset), matrices, mask)
        policies.append(probability.tolist())
        values.append(value.tolist())
    model = dict(schema_version=1, type=shopping.TYPE, actions=list(shopping.ACTIONS), feature_names=shopping.FEATURE_NAMES,
                 state_count=n, discount=shopping.DISCOUNT, regularization=.01, encoder=encoder,
                 transition_csr=[shopping.pack_matrix(p) for p in matrices], action_mask=mask.tolist(),
                 mixture_weights=[.5, .5], component_count=2, common_reward=base.tolist(),
                 component_offsets=offsets.tolist(), component_policies=policies, component_values=values, profile_posteriors={})
    return model, panel, matrices, audit


def test_original_prefix_ordinals_survive_invalid_session_filtering_and_long_inactivity():
    rows = original_rows()
    default = shopping.build_source(rows, "2024-04-01T00:00:00Z")
    memory = shopping.build_source(rows, "2024-04-01T00:00:00Z", lifecycle_memory=MEMORY)
    np.testing.assert_array_equal(default["features"], memory["features"])
    assert memory["features"].shape[1] == len(shopping.FEATURE_NAMES) == 40
    assert "observed_session_number" not in shopping.FEATURE_NAMES
    pd.testing.assert_frame_equal(default["records"], memory["records"].drop(columns="observed_session_number"))
    pd.testing.assert_frame_equal(default["sessions"], memory["sessions"].drop(columns="observed_session_number"))
    assert memory["sessions"].observed_session_number.tolist() == [1, 2, 3, 4]
    assert memory["sessions"].invalid_total_cart.iloc[0]
    assert phases.phase_ids(memory["records"].loc[memory["records"].session.eq(2)]).tolist() == [0, 0]
    panel = shopping.select_panel(memory, memory["sessions"].session.ge(3))
    assert panel["records"].observed_session_number.min() == 3
    assert phases.phase_ids(panel["records"]).tolist() == [1] * len(panel["records"])
    future = rows.iloc[-1:].copy()
    future["session"] = 5
    future["at"] += pd.Timedelta(days=100)
    future["source_event_id"] = "source:999999999"
    extended = shopping.build_source(pd.concat([rows, future]), "2025-01-01T00:00:00Z", lifecycle_memory=MEMORY)
    np.testing.assert_array_equal(memory["records"].observed_session_number,
                                  extended["records"].loc[extended["records"].session.le(4), "observed_session_number"])


@pytest.mark.parametrize("numbers", [[0], [-1], [1.], [True], [None]])
def test_private_ordinals_cannot_be_missing_rebased_or_inexact(numbers):
    with pytest.raises(ValueError, match="positive integer"):
        phases.phase_ids(pd.DataFrame({"observed_session_number": numbers}))
    with pytest.raises(ValueError):
        phases.phase_from_session_number(numbers[0])


def test_encoder_adds_private_phase_state_without_expanding_or_changing_features():
    panel = phase_panel()
    legacy = shopping.fit_encoder(panel, minimum_leaf=1, available_actions=AVAILABLE)
    clean = {**panel, "records": panel["records"].drop(columns="observed_session_number")}
    assert legacy == shopping.fit_encoder(clean, minimum_leaf=1, available_actions=AVAILABLE, lifecycle_memory=None)
    encoder = shopping.fit_encoder(panel, minimum_leaf=1, available_actions=AVAILABLE, lifecycle_memory=MEMORY)
    assert encoder["tree"] == legacy["tree"]
    assert encoder["feature_names"] == legacy["feature_names"]
    assert len(encoder["feature_scale"]) == len(shopping.FEATURE_NAMES) == 40
    assert np.asarray(encoder["state_keys"]).shape[1] == 3
    assert encoder["maximum_states"] == 2 * legacy["maximum_states"]
    with pytest.raises(ValueError, match="phase_ids"):
        shopping.encode(encoder, panel["features"])
    with pytest.raises(ValueError, match="both phases"):
        shopping.fit_encoder({**panel, "records": panel["records"].assign(observed_session_number=1)},
                             minimum_leaf=1, available_actions=AVAILABLE, lifecycle_memory=MEMORY)


def test_encoding_and_dynamics_backoffs_never_borrow_another_phase():
    model, panel, matrices, audit = phase_model()
    encoder = deepcopy(model["encoder"])
    keys = np.asarray(encoder["state_keys"])
    assert any(item["backoff_rows"].get("pooled_identified_action", 0) for item in audit.values())
    for matrix in matrices:
        coo = matrix.tocoo()
        assert not (keys[coo.row, 2] != keys[coo.col, 2]).any()
    encoder["state_keys"] = (keys + np.asarray([100, 0, 0])).tolist()
    ids = phases.phase_ids(panel["records"])
    states, fallback = shopping.encode(encoder, panel["features"], phase_ids=ids)
    assert fallback.all()
    np.testing.assert_array_equal(np.asarray(encoder["state_keys"])[states, 2], ids)
    keep = ~((keys[:, 1] == 1) & (keys[:, 2] == 1))
    encoder["state_keys"] = keys[keep].tolist()
    encoder["context_means"] = np.asarray(encoder["context_means"])[keep].tolist()
    with pytest.raises(ValueError, match="requested phase"):
        shopping.encode(encoder, panel["features"][[3]], phase_ids=np.asarray([1]))
    crossed = {**panel, "records": panel["records"].copy()}
    crossed["records"].loc[2, "observed_session_number"] = 3
    with pytest.raises(ValueError, match="cannot cross"):
        shopping.fit_dynamics(model["encoder"], crossed)


def test_runtime_phase_changes_real_policy_and_roundtrips_with_the_same_customer_offset():
    model, panel, _matrices, _audit = phase_model()
    model = json.loads(json.dumps(model))
    shopping.validate_model(model)
    context = dict(previous_action="start", session_decision_count=0, previous_action_run_length=0, cart_total_quantity=0)
    new = shopping.policy(model, {**context, "observed_session_number": 1}, 0)
    second = shopping.policy(model, {**context, "observed_session_number": 2}, 0)
    home = shopping.policy(model, {**context, "observed_session_number": 3}, 0)
    later = shopping.policy(model, {**context, "observed_session_number": 300}, 0)
    assert new == second and home == later and new != home
    np.testing.assert_array_equal(shopping.context_vector({**context, "observed_session_number": 1}),
                                  shopping.context_vector({**context, "observed_session_number": 300}))
    assert all(values["end"] == values["remove"] == values["purchase"] == 0 for values in (new, home))
    with pytest.raises(ValueError, match="observed_session_number"):
        shopping.policy(model, context, 0)
    probabilities = np.asarray(model["component_policies"])[0]
    assert shopping.policy(model, {**context, "observed_session_number": 3}, None, probabilities) == home


def test_rare_quantity_bucket_fallback_keeps_phase_start_cart_presence_and_action_mask():
    def vector(quantity, started=True):
        return shopping.context_vector(dict(previous_action='start' if started else 'view',
            session_decision_count=0 if started else 1,previous_action_run_length=0 if started else 1,
            cart_total_quantity=quantity))
    panel={'features':np.asarray([vector(3),vector(7)]),
           'records':pd.DataFrame({'action':[0,0],'observed_session_number':[1,3]})}
    encoder=shopping.fit_encoder(panel,minimum_leaf=1,lifecycle_memory=MEMORY)
    states,fallback=shopping.encode(encoder,np.asarray([vector(7)]),phase_ids=np.asarray([0]))
    assert fallback.tolist()==[True]
    assert np.asarray(encoder['state_keys'])[states[0],1:].tolist()==[5,0]
    expected=shopping.action_mask({**encoder,'state_keys':[[0,7,0]]})[0]
    np.testing.assert_array_equal(shopping.action_mask(encoder)[states[0]],expected)
    for unsupported in (vector(0),vector(7,False)):
        with pytest.raises(ValueError,match='requested phase'):
            shopping.encode(encoder,np.asarray([unsupported]),phase_ids=np.asarray([0]))
    legacy=shopping.fit_encoder(panel,minimum_leaf=1)
    with pytest.raises(ValueError,match='requested phase'):
        shopping.encode(legacy,np.asarray([vector(1)]))


def test_one_continuous_customer_draw_drives_both_phase_policies():
    from simulate.v4.irl import fit_gaussian_rewards as gaussian
    model, _panel, matrices, _audit = phase_model()
    block = dict(schema_version=1, type=gaussian.TYPE, purpose="synthesis_only",
        parameter_names=gaussian.PARAMETERS.copy(), component_count=2, mixture_weights=[.5, .5],
        means=[[*offset, .2] for offset in model["component_offsets"]], variances=[[0.] * 5 for _ in range(2)],
        available_actions=list(AVAILABLE), fitted_parameter_names=["view", "cart", "return_logit"])
    draw = gaussian.sample_reward_policy(model, block, 0, np.random.default_rng(17), matrices)
    before = draw["action_offsets"].copy()
    context = dict(previous_action="start", session_decision_count=0, previous_action_run_length=0, cart_total_quantity=0)
    actual = [shopping.policy(model, {**context, "observed_session_number": number}, None, draw["policy"])
              for number in (1, 3)]
    assert actual[0] != actual[1]
    for number, probability in zip((1, 3), actual):
        assert probability == shopping.policy(model, {**context, "observed_session_number": number}, 0)
    np.testing.assert_array_equal(before, draw["action_offsets"])
    assert draw["return_offset"] == .2


def test_serialized_models_reject_cross_phase_transitions_and_incorrect_deviation_policies():
    model, _panel, matrices, _audit = phase_model()
    keys = np.asarray(model["encoder"]["state_keys"])
    corrupted = deepcopy(model)
    matrix = matrices[0].tolil()
    row = int(np.flatnonzero((keys[:, 1] == 1) & (keys[:, 2] == 0))[0])
    other = int(np.flatnonzero((keys[:, 1] == 0) & (keys[:, 2] == 1))[0])
    matrix[row, :] = 0
    matrix[row, other] = 1
    corrupted["transition_csr"][0] = shopping.pack_matrix(matrix)
    with pytest.raises(ValueError, match="cross private lifecycle"):
        shopping.validate_model(corrupted)
    deviations = np.zeros((2, len(keys), 4))
    deviations[0, keys[:, 2] == 0, 1] = .3
    model["component_reward_deviations"] = deviations.tolist()
    with pytest.raises(ValueError, match="does not reproduce"):
        shopping.validate_model(model)
    for index in range(2):
        reward = np.asarray(model["common_reward"]) + deviations[index] + np.asarray(model["component_offsets"])[index]
        probability, value, _ = shopping.soft_policy(shopping._reward(reward), matrices, np.asarray(model["action_mask"]))
        model["component_policies"][index], model["component_values"][index] = probability.tolist(), value.tolist()
    shopping.validate_model(model)
    invalid = deepcopy(model)
    invalid["component_reward_deviations"][0][0][2] = 1
    with pytest.raises(ValueError, match="unobserved component reward"):
        shopping.validate_model(invalid)
