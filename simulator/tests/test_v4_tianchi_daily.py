from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.irl import tianchi_daily as daily


ORIGIN = pd.Timestamp("2016-01-01", tz="UTC")


def _ids(train, number):
    return [str(uid) for uid in range(100)
            if (hashlib.sha256(f"42:{uid}".encode()).digest()[0] < 205) == train][:number]


@pytest.fixture
def source():
    train, heldout = _ids(True, 2), _ids(False, 1)
    train.sort(key=lambda uid: hashlib.sha256(f"42:inner:{uid}".encode()).digest()[0] >= 205)
    patterns = {train[0]: [(0, "view"), (0, "purchase"), (2, "view"), (4, "purchase"),
                           (6, "view"), (6, "purchase"), (9, "view"), (10, "purchase"), (14, "view")],
                train[1]: [(0, "view"), (3, "view"), (5, "purchase"), (8, "view"), (15, "purchase")],
                heldout[0]: [(0, "view"), (3, "purchase"), (6, "view"), (6, "purchase"), (14, "view")]}
    rows = pd.DataFrame([{"user_id": uid, "at": ORIGIN + day * daily.DAY, "action": action, "price_minor": np.nan}
                         for uid, values in patterns.items() for day, action in values])
    contract = {"domain": "tianchi", "clock_resolution": "day", "observed_actions": ["view", "purchase"],
                "observed_prices": False, "observation_start_at": ORIGIN.isoformat(),
                "observation_end_at": (ORIGIN + 18 * daily.DAY).isoformat(),
                "inner_cutoff_at": (ORIGIN + 7 * daily.DAY).isoformat(),
                "outer_cutoff_at": (ORIGIN + 12 * daily.DAY).isoformat(), "prepared_sha256": "fixture"}
    coverage = {key: contract[key] for key in ("observation_start_at", "observation_end_at", "prepared_sha256")}
    coverage["assume_complete_behavior_capture_within_declared_interval"] = True
    return rows, contract, coverage, train, heldout


def _panel(source, rows=None):
    original, contract, coverage, _, _ = source
    return daily.build_daily_panel(original if rows is None else rows, contract, observation_contract=coverage)


@pytest.fixture
def model(source):
    return daily.fit_daily_model(_panel(source), max_states=4, max_iter=250)


def _single_state(model):
    result = deepcopy(model)
    result["state_vectors"] = result["state_vectors"][:1]
    result["state_feature_means"] = result["state_feature_means"][:1]
    result["state_phases"] = result["state_phases"][:1]
    result["reward_weights"] = np.zeros((4, len(daily.STATE_NAMES) + 1)).tolist()
    result["mask"] = [[True, True, True, True, False]]
    result["transition_csr"] = [{"shape": [1, 1], "data": [1.], "indices": [0], "indptr": [0, 1]} for _ in range(4)]
    result["transition_csr"].append({"shape": [1, 1], "data": [], "indices": [], "indptr": [0, 0]})
    return result


def test_chunked_state_assignment_preserves_nearest_prototype_ties_and_private_phases():
    rng = np.random.default_rng(872)
    values = rng.normal(size=(4203, len(daily.STATE_INDEX)))
    prototypes = np.zeros((8, len(daily.STATE_NAMES)))
    prototypes[:, daily.STATE_INDEX] = rng.normal(size=(8, len(daily.STATE_INDEX)))
    prototypes[1] = prototypes[0]
    values[0] = prototypes[0, daily.STATE_INDEX]
    phase = np.arange(len(values)) % 2
    state_phase = np.arange(8) % 2
    center, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])
    expected = []
    for row, group in zip(values, phase):
        candidates = np.flatnonzero(state_phase == group)
        expected.append(candidates[np.square(prototypes[candidates][:, daily.STATE_INDEX] - row).sum(1).argmin()])
    actual = daily._state_indices(values, prototypes, center, scale, phases=phase, state_phases=state_phase)
    np.testing.assert_array_equal(actual, expected)
    assert daily._state_indices(values[:1], prototypes, center, scale).tolist() == [0]


def test_panel_endpoint_keeps_exact_prefix_and_original_capture_binding(source):
    rows, contract, coverage, _, _ = source
    full = _panel(source)
    through = pd.Timestamp(contract["outer_cutoff_at"])
    short = daily.build_daily_panel(rows, contract, observation_contract=coverage, through_at=through)
    mask = full["records"].label_available_at.le(through)
    pd.testing.assert_frame_equal(short["records"].reset_index(drop=True), full["records"].loc[mask].reset_index(drop=True))
    np.testing.assert_array_equal(short["features"].toarray(), full["features"][mask.to_numpy()].toarray())
    assert short["source_contract"] == full["source_contract"]
    assert short["observation_contract"] == coverage


def test_partition_sample_limits_only_prototype_search_and_retains_all_training_dates(source):
    panel = _panel(source)
    fitted = daily.fit_daily_model(panel,max_states=2,partition_max_rows=3,max_iter=500)
    training = panel['records'].train_eligible
    assert fitted['training']['daily_rows'] == int(training.sum())
    assert fitted['training']['users'] == panel['records'].loc[training,'user_id'].nunique()
    assert fitted['training']['partition_rows_by_phase'] == [3]
    again = daily.fit_daily_model(panel,max_states=2,partition_max_rows=3,max_iter=500)
    assert fitted['state_vectors'] == again['state_vectors']
    assert fitted['common_reward'] == again['common_reward']


def test_weekly_daily_counts_expire_at28days_in_both_raw_and_incremental_paths(source, model):
    uid = source[3][0]
    dates = pd.date_range(ORIGIN, periods=31)
    history = pd.DataFrame({"user_id": uid, "day_start": dates,
                            "view": np.arange(31) == 2, "purchase": np.arange(31) == 0})
    weekly = deepcopy(model)
    weekly["source_contract"]["feature_windows_days"] = [28, 56, 91]
    weekly["feature_windows_days"] = [28, 56, 91]
    at = ORIGIN + 31 * daily.DAY
    raw = daily.public_observation(history, at, source_contract=weekly["source_contract"])
    env = daily.TianchiDailyEnv(weekly, history, as_of_at=at)
    assert env.observation() == raw
    assert raw["session_count_30d"] == 0 and raw["purchase_count_30d"] == 0
    assert raw["orders_delta_30d"] == -1
    assert daily.public_observation(history, at)["session_count_30d"] == 1


def test_unordered_day_presence_is_invariant_to_input_order_duplicates_and_merchant_proxy(source):
    rows, _, _, _, _ = source
    expected = daily.daily_presence(rows)
    shuffled = pd.concat([rows.iloc[::-1], rows.iloc[:2]], ignore_index=True).assign(product_id="merchant:unobserved-SKU")
    pd.testing.assert_frame_equal(expected, daily.daily_presence(shuffled))
    assert expected.loc[expected.user_id.eq(source[3][0])].iloc[0][["view", "purchase"]].tolist() == [True, True]
    assert "product_id" not in expected and "session" not in expected


def test_no_activity_requires_bound_explicit_capture_and_never_precedes_first_observation(source):
    rows, contract, coverage, train, _ = source
    with pytest.raises(ValueError, match="explicit complete"):
        daily.build_daily_panel(rows, contract)
    changed = {**coverage, "prepared_sha256": "different"}
    with pytest.raises(ValueError, match="SHA256"):
        daily.build_daily_panel(rows, contract, observation_contract=changed)
    late = rows.loc[rows.user_id.eq(train[1]) & rows["at"].ge(ORIGIN + 3 * daily.DAY)]
    panel = daily.build_daily_panel(late, contract, observation_contract=coverage)
    assert panel["records"]["at"].min() == ORIGIN + 4 * daily.DAY
    assert panel["audit"]["pre_entry_inactivity_rows"] == 0
    assert panel["records"].outcome.eq("no_activity").any()


def test_daily_training_labels_mature_strictly_before_fixed_cutoff(source):
    panel = _panel(source)
    records = panel["records"]
    cutoff = pd.Timestamp(source[1]["outer_cutoff_at"])
    assert records.loc[records.train_eligible, "label_available_at"].max() == cutoff - daily.DAY
    assert not records.loc[records.label_available_at.eq(cutoff), "train_eligible"].any()
    assert records.loc[records.train_eligible].groupby("user_id").size().tolist() == [10, 10]
    unsupported = [i for i in range(79) if i not in daily.STATE_INDEX]
    assert len(daily.CORE_NAMES) == 40
    assert not panel["features"][:, unsupported].nnz
    assert not panel["next_features"][:, unsupported].nnz


def test_partial_final_date_is_not_a_complete_no_activity_label(source):
    rows, contract, coverage, train, _ = source
    end = ORIGIN + 18 * daily.DAY + pd.Timedelta(hours=12)
    contract = {**contract, "observation_end_at": end.isoformat()}
    coverage = {**coverage, "observation_end_at": end.isoformat()}
    partial = pd.DataFrame([dict(user_id=train[0], at=end.floor("D"), action="purchase", price_minor=np.nan)])
    panel = daily.build_daily_panel(pd.concat([rows, partial]), contract, observation_contract=coverage)
    assert panel["audit"]["partial_last_day_excluded"]
    assert panel["records"].day_start.max() == end.floor("D") - daily.DAY


@pytest.mark.parametrize("kind", ["price", "receipt", "intraday"])
def test_unobserved_retail_fields_and_subday_time_cannot_enter(source, kind):
    rows = source[0].copy()
    if kind == "price":
        rows.loc[0, "price_minor"] = 1
    elif kind == "receipt":
        rows.loc[0, "action"] = "coupon_receive"
    else:
        rows.loc[0, "at"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Tianchi"):
        daily.daily_presence(rows)


def test_training_basis_reward_and_dynamics_do_not_read_heldout_or_future_outcomes(source, model):
    rows, contract, _, train, heldout = source
    changed = rows.copy()
    selected = changed.user_id.isin(heldout) | changed["at"].ge(pd.Timestamp(contract["outer_cutoff_at"]))
    changed.loc[selected, "action"] = changed.loc[selected, "action"].map({"view": "purchase", "purchase": "view"})
    other = daily.fit_daily_model(_panel(source, changed), max_states=4, max_iter=250)
    for name in ("state_vectors", "state_feature_means", "scaler_center", "scaler_scale", "reward_weights", "common_reward", "policy", "transition_csr"):
        assert other[name] == model[name]
    assert model["training"]["users"] == len(train)
    assert model["fit"]["trace"][-1]["objective"] < model["fit"]["trace"][0]["objective"]
    assert model["feature_names"] == list(daily.REWARD_FEATURE_NAMES)
    assert len(model["feature_names"]) == 80
    basis = np.column_stack([np.ones(len(model["state_vectors"])), np.zeros((len(model["state_vectors"]), 79))])
    basis[:, np.asarray(daily.STATE_INDEX) + 1] = (np.asarray(model["state_feature_means"])[:, daily.STATE_INDEX]
                                                  - model["scaler_center"]) / model["scaler_scale"]
    np.testing.assert_allclose(model["common_reward"], basis @ np.asarray(model["reward_weights"]).T, atol=1e-12, rtol=0)
    assert model["core_supported"] == list(daily.CORE_SUPPORTED)
    json.dumps(model, allow_nan=False)


def test_daily_joint_likelihood_includes_observed_inactive_dates(source, model):
    panel = _panel(source)
    mask = panel["records"].train_eligible.to_numpy()
    result = daily.daily_log_likelihood(model, panel, mask)
    active = mask & panel["records"].action.ne(0).to_numpy()
    assert result["daily_rows"] == 20
    assert result["daily_rows"] > daily.daily_log_likelihood(model, panel, active)["daily_rows"]
    assert np.isfinite(result["mean_nll"]) and result["mean_nll"] > 0


def test_canonical_trait_incidence_keeps_both_as_view_plus_purchase_and_return_as_daily_activity(model):
    one = _single_state(model)
    trait = [np.log(2), 0, 0, np.log(3), np.log(5)]
    np.testing.assert_allclose(daily.policy_for_trait(one, trait)[0], np.asarray([1, 10, 15, 30]) / 56,
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="only view/purchase"):
        daily.policy_for_trait(one, [0, 1, 0, 0, 0])


def test_unsupported_coordinate_or_artificial_end_rejected(model):
    changed = deepcopy(model)
    changed["reward_weights"][0][1 + 39] = 1  # Existing40 coordinate0 is unobserved cart quantity.
    with pytest.raises(ValueError, match="coefficients"):
        daily.policy_for_trait(changed)
    changed = deepcopy(model)
    changed["mask"][0][4] = True
    with pytest.raises(ValueError, match="no END"):
        daily.policy_for_trait(changed)


def test_known_prefix_environment_advances_dates_through_inactivity_without_new_signup_or_fake_raw_fields(source, model):
    rows, contract, coverage, train, _ = source
    start = ORIGIN + 5 * daily.DAY
    prefix = daily.daily_history(rows.loc[rows.user_id.eq(train[0])], contract, observation_contract=coverage, as_of_at=start)
    env_model = _single_state(model)
    env_model["reward_weights"][0][0] = 30
    env = daily.TianchiDailyEnv(env_model, prefix, as_of_at=start, seed=1)
    first = env.history.day_start.min()
    for _ in range(35):
        row = env.step()
        assert row["outcome"] == "no_activity"
        assert set(row) == {"user_id", "day_start", "day_end", "view", "purchase", "outcome"}
    observed = daily.public_observation(env.history, env.at)
    assert observed["session_count_30d"] == 0 and observed["observed_tenure_days"] == (env.at-first).days
    env.set_trait([100, 0, 0, -100, 0])
    assert env.step()["outcome"] == "click_only_day"
    assert env.history.day_start.min() == first
    assert daily.public_observation(env.history, env.at)["session_count_30d"] == 1
    assert set(observed) == set(EXPANDED_FEATURES) and len(observed) == 39
    for name in ("view_count_30d", "net_spend_30d", "cart_qty_current", "purchased_qty_30d",
                 "mean_session_duration_minutes_30d", "last_session_start_hour", "last_session_min_viewed_price"):
        assert observed[name] is None
    json.dumps(observed, allow_nan=False)


def test_empty_or_gapped_history_is_not_silently_zero(source, model):
    rows, contract, coverage, train, _ = source
    start = ORIGIN + 5 * daily.DAY
    history = daily.daily_history(rows.loc[rows.user_id.eq(train[0])], contract, observation_contract=coverage, as_of_at=start)
    for changed in (history.iloc[:0], history.iloc[[0, 2, 3, 4]]):
        with pytest.raises(ValueError, match="contiguous known prefix"):
            daily.TianchiDailyEnv(model, changed, as_of_at=start)
    future = pd.DataFrame([dict(user_id=train[0], day_start=start + daily.DAY, view=True, purchase=True)])
    np.testing.assert_array_equal(daily.core_vector(history, start), daily.core_vector(pd.concat([history, future]), start))


def test_frozen_daily_prefix_persona_is_available_only_on_forecast_dates(source):
    panel = _panel(source)
    _, contract, _, train, _ = source
    cutoff = pd.Timestamp(contract["inner_cutoff_at"])
    assignments = pd.DataFrame({"user_id": train, "cluster": ["0", "1"], "persona": ["daily0", "daily1"],
                                "assignment_cutoff": cutoff, "representation_end_at": cutoff-daily.DAY})
    assignments.attrs["frozen_persona_contract"] = {
        "domain": "tianchi", "cutoff_at": cutoff.isoformat(), "prepared_source_sha256": contract["prepared_sha256"]}
    result = daily.fit_daily_model(panel, assignments=assignments, persona_cluster="0", persona_usage="forecast", max_states=2, max_iter=200)
    assert result["training"]["users"] == 1 and result["training"]["daily_rows"] == 4
    assert pd.Timestamp(result["training"]["min_decision_at"]) == cutoff
    changed = deepcopy(panel)
    changed["records"].loc[changed["records"].user_id.eq(train[0]), "train_user"] = False
    with pytest.raises(ValueError, match="user split"):
        daily.fit_daily_model(changed, max_states=1)


def test_existing39_nullable_values_and_existing40_are_the_same_fit_and_runtime_state(source):
    rows, contract, coverage, train, _ = source
    panel = _panel(source)
    record = panel["records"].loc[lambda frame: frame.user_id.eq(train[0])].iloc[7]
    history = daily.daily_history(rows.loc[rows.user_id.eq(train[0])], contract,
                                  observation_contract=coverage, as_of_at=record.day_start)
    np.testing.assert_array_equal(daily.state_vector(history, record.day_start), panel["features"][record.name].toarray().ravel())
    observed = daily.public_observation(history, record.day_start)
    expected = [0. if observed[name] is None else observed[name] for name in EXPANDED_FEATURES]
    np.testing.assert_array_equal(panel["features"][record.name, :39].toarray().ravel(), expected)


def test_prefix_persona_training_strata_are_descriptive_and_not_early_forecast_scores(source):
    panel = _panel(source)
    _, contract, _, train, _ = source
    cutoff = pd.Timestamp(contract["inner_cutoff_at"])
    assignments = pd.DataFrame({"user_id": train, "cluster": ["0", "1"], "persona": ["daily0", "daily1"],
                                "assignment_cutoff": cutoff, "representation_end_at": cutoff-daily.DAY})
    assignments.attrs["frozen_persona_contract"] = {
        "domain": "tianchi", "cutoff_at": cutoff.isoformat(), "prepared_source_sha256": contract["prepared_sha256"]}
    model = daily.fit_daily_model(panel, cutoff_at=cutoff, assignments=assignments, persona_cluster="0",
                                  max_states=1, max_iter=100)
    mask = panel["records"].user_id.eq(train[0]) & panel["records"].label_available_at.lt(cutoff)
    with pytest.raises(ValueError, match="unavailable before"):
        daily.daily_log_likelihood(model, panel, mask, assignments=assignments)
    assert "descriptive" in daily.daily_log_likelihood(model, panel, mask, assignments=assignments,
                                                       retrospective_training=True)["interpretation"]
    with pytest.raises(ValueError, match="same SHA-bound"):
        daily.daily_log_likelihood(model, panel, mask, retrospective_training=True)
    with pytest.raises(ValueError, match="frozen persona group"):
        daily.fit_daily_trait(model, panel, mask, assignments=assignments.assign(cluster="wrong"))


def test_daily_trait_fits_only_three_supported_canonical_coordinates_from_mature_training(source, model):
    panel = _panel(source)
    train = source[3][0]
    mask = panel["records"].user_id.eq(train) & panel["records"].train_eligible
    result = daily.fit_daily_trait(model, panel, mask, max_iter=100)
    assert len(result["canonical_theta"]) == 5 and result["canonical_theta"][1:3] == [0, 0]
    assert result["fit"]["objective"] <= result["fit"]["objective_initial"]
    assert result["training_users"] == 1
    with pytest.raises(ValueError, match="mature training-user"):
        daily.fit_daily_trait(model, panel, panel["records"].user_id.isin(source[4]))
    with pytest.raises(ValueError, match="mature training-user"):
        daily.fit_daily_trait(model, panel, panel["records"].user_id.eq(train))


def test_daily_scoring_and_trait_fit_require_same_bound_source(source, model):
    panel = deepcopy(_panel(source))
    panel["source_contract"]["prepared_sha256"] = "different"
    panel["observation_contract"]["prepared_sha256"] = "different"
    with pytest.raises(ValueError, match="frozen source binding"):
        daily.daily_log_likelihood(model, panel)
    with pytest.raises(ValueError, match="frozen source binding"):
        daily.fit_daily_trait(model, panel, panel["records"].train_eligible)


def test_incremental_runtime_matches_raw_daily_recalculation_across_returns_and_window_expiry(source, model):
    rows, contract, coverage, train, _ = source
    start = ORIGIN + 5 * daily.DAY
    history = daily.daily_history(rows.loc[rows.user_id.eq(train[0])], contract,
                                  observation_contract=coverage, as_of_at=start)
    env = daily.TianchiDailyEnv(_single_state(model), history, as_of_at=start, seed=24)
    for step in range(110):
        if step % 11 == 0:
            np.testing.assert_allclose(env._state(), daily.state_vector(env.history, env.at), rtol=0, atol=1e-12)
            assert env.observation() == daily.public_observation(env.history, env.at)
        env.step()
    np.testing.assert_allclose(env._state(), daily.state_vector(env.history, env.at), rtol=0, atol=1e-12)
    assert env.observation() == daily.public_observation(env.history, env.at)


def test_inner_fit_does_not_use_inner_holdout_users(source):
    rows, contract, _, train, _ = source
    cutoff = pd.Timestamp(contract["inner_cutoff_at"])
    panel = _panel(source)
    model = daily.fit_daily_model(panel, cutoff_at=cutoff, max_states=2, max_iter=150)
    assert model["training"]["inner_user_hash_required"] and model["training"]["users"] == 1
    changed = rows.copy()
    held = changed.user_id.eq(train[1])
    changed.loc[held, "action"] = changed.loc[held, "action"].map({"view": "purchase", "purchase": "view"})
    other = daily.fit_daily_model(_panel(source, changed), cutoff_at=cutoff, max_states=2, max_iter=150)
    for name in ("state_vectors", "state_feature_means", "scaler_center", "scaler_scale", "reward_weights", "transition_csr"):
        assert model[name] == other[name]


def test_daily_backend_metadata_keeps_canonical_names_but_rejects_retail_return_and_coupon(model):
    from simulate.v4.irl.fit_gaussian_rewards import PARAMETERS
    assert model["canonical_parameter_names"] == PARAMETERS
    assert model["trait_names"][-1] == "return" and model["return_horizon_days"] == 1
    assert model["active_parameter_indices"] == [0, 3, 4]
    assert model["parameter_units"][-1] == "daily_activity_reward"
    for name in ("return_model_path", "coupon_response_model_path", "retail_policy_logs"):
        changed = deepcopy(model)
        changed[name] = "legacy-retail.json"
        with pytest.raises(ValueError, match="daily backend cannot reuse retail"):
            daily.policy_for_trait(changed)
    changed = deepcopy(model)
    changed["return_horizon_days"] = 30
    with pytest.raises(ValueError, match="daily backend cannot reuse retail"):
        daily.policy_for_trait(changed)


def test_daily_private_phase_is_past_only_and_separate_from_the_same80_reward_features(source):
    panel = _panel(source)
    records = panel["records"]
    first = records.loc[records.user_id.eq(source[3][0])].iloc[:3]
    assert first.past_active_days.tolist() == [1, 1, 2]
    assert first.action.tolist() == [0, 1, 0]
    model = daily.fit_daily_model(panel, lifecycle_memory=daily.LIFECYCLE_MEMORY, max_states=2, max_iter=250)
    assert model["feature_names"] == list(daily.REWARD_FEATURE_NAMES) and len(model["feature_names"]) == 80
    assert np.asarray(model["reward_weights"]).shape == (2, 4, 80)
    assert set(model["state_phases"]) == {0, 1} and model["fit"]["private_actor_banks"] == 2
    means = np.asarray(model["state_feature_means"])
    basis = np.column_stack([np.ones(len(means)), np.zeros_like(means)])
    basis[:, np.asarray(daily.STATE_INDEX) + 1] = (means[:, daily.STATE_INDEX] - model["scaler_center"]) / model["scaler_scale"]
    theta = np.asarray(model["reward_weights"])[model["state_phases"]]
    np.testing.assert_allclose(model["common_reward"], np.einsum("sf,saf->sa", basis, theta), atol=1e-12, rtol=0)
    phase = np.asarray(model["state_phases"])
    for outcome, block in enumerate(model["transition_csr"][:4]):
        from scipy import sparse
        matrix = sparse.csr_matrix((block["data"], block["indices"], block["indptr"]), shape=block["shape"]).tocoo()
        expected = np.where((phase == 0) & (outcome == 0), 0, 1)
        assert np.array_equal(phase[matrix.col], expected[matrix.row])
    scored = daily.daily_log_likelihood(model, panel, records.train_eligible)
    assert scored["daily_rows"] == 20 and np.isfinite(scored["mean_nll"])


def test_new_day_actor_changes_only_after_second_observed_active_date_and_never_resets_home(source):
    panel = _panel(source)
    model = daily.fit_daily_model(panel, lifecycle_memory=daily.LIFECYCLE_MEMORY, max_states=1, max_iter=200)
    rows, contract, coverage, train, _ = source
    start = ORIGIN + daily.DAY
    history = daily.daily_history(rows.loc[rows.user_id.eq(train[0])], contract,
                                  observation_contract=coverage, as_of_at=start)
    env = daily.TianchiDailyEnv(model, history, as_of_at=start, seed=42)
    env._policy[:] = [1., 0., 0., 0.]
    for _ in range(35):
        assert env.probabilities()["actor_phase"] == "NEW"
        assert env.step()["outcome"] == "no_activity"
    env._policy[:] = [0., 0., 1., 0.]
    assert env.probabilities()["actor_phase"] == "NEW"  # Today's purchase label has not been seen.
    assert env.step()["outcome"] == "purchase_only_day"
    assert env.probabilities()["actor_phase"] == "HOME"
    env._policy[:] = [1., 0., 0., 0.]
    for _ in range(40):
        env.step()
        assert env.probabilities()["actor_phase"] == "HOME"
    assert env.observation()["session_count_30d"] == 0
    env._policy[:] = [0., 1., 0., 0.]
    env.step()
    assert env.probabilities()["actor_phase"] == "HOME"
    np.testing.assert_allclose(env._state(), daily.state_vector(env.history, env.at), atol=1e-12, rtol=0)


def test_phase_dynamics_cannot_fallback_from_home_into_new(source):
    model = daily.fit_daily_model(_panel(source), lifecycle_memory=daily.LIFECYCLE_MEMORY, max_states=1, max_iter=150)
    phase = np.asarray(model["state_phases"])
    home, new = np.flatnonzero(phase == 1)[0], np.flatnonzero(phase == 0)[0]
    from scipy import sparse
    block = model["transition_csr"][0]
    matrix = sparse.csr_matrix((block["data"], block["indices"], block["indptr"]), shape=block["shape"]).tolil()
    matrix[home, :] = 0
    matrix[home, new] = 1
    matrix = matrix.tocsr()
    model["transition_csr"][0] = {"shape": list(matrix.shape), "data": matrix.data.tolist(),
                                  "indices": matrix.indices.tolist(), "indptr": matrix.indptr.tolist()}
    with pytest.raises(ValueError, match="cannot reset HOME"):
        daily.policy_for_trait(model)


def test_daily_gmm_action_counts_are_bound_joint_date_counts_including_inactivity(source, model):
    panel = _panel(source)
    mask = panel["records"].train_eligible.to_numpy()
    actions = daily.daily_action_records(model, panel, mask)
    assert len(actions) == 20 and actions.action.eq(0).any() and actions["count"].eq(1).all()
    assert actions.attrs["backend"] == "tianchi_daily" and actions.attrs["active_parameter_indices"] == [0, 3, 4]
    assert actions.known_at.eq(actions["at"] + daily.DAY).all()
    probability = daily.policy_for_trait(model)[actions.state, actions.action]
    np.testing.assert_allclose(np.log(probability).sum(), daily.daily_log_likelihood(model, panel, mask)["log_likelihood"],
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="heldout or unavailable"):
        daily.daily_action_records(model, panel, np.ones(len(panel["records"]), dtype=bool))


def test_group_actors_share_global_states_and_daily_qmc_callback_preserves_likelihood(source):
    panel = _panel(source)
    global_model = daily.fit_daily_model(panel, lifecycle_memory=daily.LIFECYCLE_MEMORY, max_states=2, max_iter=200)
    _, contract, _, train, _ = source
    cutoff = pd.Timestamp(contract["inner_cutoff_at"])
    assignments = pd.DataFrame({"user_id": train, "cluster": ["0", "1"], "persona": ["daily0", "daily1"],
                                "assignment_cutoff": cutoff, "representation_end_at": cutoff-daily.DAY})
    assignments.attrs["frozen_persona_contract"] = {
        "domain": "tianchi", "cutoff_at": cutoff.isoformat(), "prepared_source_sha256": contract["prepared_sha256"]}
    grouped = daily.fit_daily_group_model(global_model, panel, assignments, "0", max_iter=200)
    for name in ("state_vectors", "state_feature_means", "scaler_center", "scaler_scale", "transition_csr", "state_phases"):
        assert grouped[name] == global_model[name]
    assert grouped["training"]["users"] == 1 and grouped["fit"]["frozen_global_state_and_dynamics"]
    from simulate.v4.irl.linear_actor import FEATURES
    assert grouped["feature_names"] == list(FEATURES)
    callback = daily.daily_policy_log_provider({0: grouped, 1: global_model})
    sample = np.asarray([[0., 0., 0., 0., 0.], [.3, 0., 0., -.2, .1]])
    result = callback(0, sample).reshape(2, len(grouped["state_vectors"]), 5)
    for index, theta in enumerate(sample):
        np.testing.assert_allclose(np.exp(result[index, :, :4]), daily.policy_for_trait(grouped, theta), atol=1e-9, rtol=0)
    assert not result[:, :, 4].any()
    wrong = deepcopy(global_model)
    wrong["state_feature_means"][0][daily.STATE_INDEX[0]] += 1
    with pytest.raises(ValueError, match="same frozen global state"):
        daily.daily_policy_log_provider({0: global_model, 1: wrong})
