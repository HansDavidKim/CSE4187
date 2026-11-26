import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import return_state as state


ORIGIN = pd.Timestamp("2020-01-01", tz="UTC")


def _uid_role(uid):
    return (hashlib.sha256(f"42:{uid}".encode()).digest()[0] < 205,
            hashlib.sha256(f"42:inner:{uid}".encode()).digest()[0] < 205)


def _observations():
    users = []
    for role in ((True, True), (True, False), (False, True), (False, False)):
        users.extend([str(uid) for uid in range(1000) if _uid_role(str(uid)) == role][:2])
    activity, sessions = [], []
    for index, uid in enumerate(users):
        label = index % 2
        sessions.append(dict(user_id=uid, session_id=f"{uid}:entry", known_at=ORIGIN))
        for day in range(30, 241, 30):
            at = ORIGIN + pd.Timedelta(days=day)
            activity.append(dict(user_id=uid, at=at, target_at=at + pd.Timedelta(days=30),
                                 base_logit=6. if label else -6., label=label,
                                 train_user=_uid_role(uid)[0]))
            if label:
                sessions.append(dict(user_id=uid, session_id=f"{uid}:{day}", known_at=at + pd.Timedelta(days=1)))
    return pd.DataFrame(activity), pd.DataFrame(sessions)


def _fit(activity, sessions):
    return state.fit_return_state(activity, sessions,
        inner_cutoff_at=ORIGIN + pd.Timedelta(days=120), cutoff_at=ORIGIN + pd.Timedelta(days=210),
        observation_end_at=ORIGIN + pd.Timedelta(days=300), rhos=(0., .7), stationary_stds=(0., .7),
        quadrature_order=9)


def test_session_transition_has_gaussian_moments_and_no_session_identity():
    model = state.make_model(.5, .7, 21)
    points, prior = state.quadrature_grid(model)
    transition = state.transition_matrix(model, 2)
    np.testing.assert_allclose(transition.sum(axis=1), 1., atol=1e-14)
    np.testing.assert_array_equal(state.transition_matrix(model, 0), np.eye(21))
    central = np.abs(points) < 1.5
    mean = transition @ points
    variance = transition @ np.square(points) - np.square(mean)
    np.testing.assert_allclose(mean[central], .5 ** 2 * points[central], atol=1e-7)
    np.testing.assert_allclose(variance[central], .7 ** 2 * (1 - .5 ** 4), atol=1e-7)
    np.testing.assert_allclose(prior @ transition, prior, atol=1e-7)


def test_live_update_uses_stationary_std_and_never_draws_without_sessions():
    class RecordedNormal:
        def normal(self, mean, scale):
            assert mean == pytest.approx(.8 ** 3 * 2.)
            assert scale == pytest.approx(.6 * np.sqrt(1 - .8 ** 6))
            return mean + scale
    model = state.make_model(.8, .6)
    assert state.advance_state(model, 2., None, 0) == 2.
    assert state.advance_state(model, 2., RecordedNormal(), 3) > .8 ** 3 * 2.
    collapsed = state.make_model(0., 0.)
    assert state.sample_initial_state(collapsed, None) == 0
    assert state.advance_state(collapsed, 0., None) == 0


def test_completed_counts_use_only_strictly_past_known_sessions_and_keep_ties():
    activity, _ = _observations()
    activity = activity.loc[activity.user_id.eq(activity.user_id.iloc[0])].iloc[:3].copy()
    uid = activity.user_id.iloc[0]
    sessions = pd.DataFrame([dict(user_id=uid, session_id=str(i), known_at=ORIGIN + pd.Timedelta(days=day))
                            for i, day in enumerate((5, 30, 50, 50, 60, 200))])
    prepared = state.prepare_activity(activity, sessions)
    assert prepared.completed_sessions_before_at.tolist() == [1, 4, 5]
    more_future = pd.concat([sessions, pd.DataFrame([dict(user_id=uid, session_id="future",
                                             known_at=ORIGIN + pd.Timedelta(days=100))])])
    pd.testing.assert_frame_equal(prepared, state.prepare_activity(activity, more_future))
    activity["at"] = activity["at"].dt.as_unit("us")
    activity["target_at"] = activity["target_at"].dt.as_unit("us")
    pd.testing.assert_frame_equal(prepared, state.prepare_activity(activity, sessions))


def test_zero_variance_is_exact_fixed_logit_likelihood_with_explicit_persistent_offset():
    activity, sessions = _observations()
    offsets = {uid: .3 for uid in activity.user_id.unique()}
    prepared = state.prepare_activity(activity, sessions, persistent_offsets=offsets)
    cutoff = ORIGIN + pd.Timedelta(days=210)
    score = state.sequence_log_likelihood(prepared, state.make_model(.8, 0.), label_cutoff_at=cutoff)
    observed = prepared.loc[prepared.known_at.lt(cutoff)]
    expected = (observed.label * observed.base_logit - np.logaddexp(0., observed.base_logit)).sum()
    assert score["log_likelihood"] == pytest.approx(expected, abs=1e-12)
    assert score["records"] == len(observed)
    assert not observed.target_at.eq(cutoff).any()


def test_grid_keeps_zero_variance_and_outer_outcomes_cannot_select_parameters(tmp_path):
    activity, sessions = _observations()
    model = _fit(activity, sessions)
    assert model["stationary_std"] == model["rho"] == 0.
    assert model["rho_identified"] is False
    changed = activity.copy()
    external = ~changed.train_user | changed["at"].ge(ORIGIN + pd.Timedelta(days=210))
    changed.loc[external, "label"] = 1 - changed.loc[external, "label"]
    candidate = _fit(changed, sessions)
    assert model["rho"] == candidate["rho"] and model["stationary_std"] == candidate["stationary_std"]
    assert model["fit"]["trials"] == candidate["fit"]["trials"]
    path = tmp_path / "return_state.json"
    state.save_model(path, model)
    assert json.loads(path.read_text()) == model
    with pytest.raises(FileExistsError):
        state.save_model(path, model)


def test_users_filter_independently_and_observations_must_have_mature_support():
    activity, sessions = _observations()
    prepared = state.prepare_activity(activity, sessions)
    model = state.make_model(.5, .7, 11)
    cutoff = ORIGIN + pd.Timedelta(days=210)
    all_users = state.sequence_log_likelihood(prepared, model, label_cutoff_at=cutoff)
    separated = sum(state.sequence_log_likelihood(group, model, label_cutoff_at=cutoff)["log_likelihood"]
                    for _uid, group in prepared.groupby("user_id"))
    assert all_users["log_likelihood"] == pytest.approx(separated, abs=1e-12)
    with pytest.raises(ValueError, match="zero-variance"):
        state.fit_return_state(activity, sessions, inner_cutoff_at=cutoff - pd.Timedelta(days=90),
            cutoff_at=cutoff, observation_end_at=cutoff + pd.Timedelta(days=90), stationary_stds=(.5,))
    with pytest.raises(ValueError, match="support"):
        _fit(activity.loc[~activity.train_user], sessions)


@pytest.mark.parametrize("mutation,error", [("missing_label", "binary outcomes"),
    ("early_known", "binary outcomes"), ("overlap", "nonoverlapping"), ("forged_split", "hash rule")])
def test_observation_contract_rejects_unknown_or_leaking_rows(mutation, error):
    activity, sessions = _observations()
    if mutation == "missing_label":
        activity.loc[0, "label"] = np.nan
    elif mutation == "early_known":
        activity["known_at"] = activity.target_at
        activity.loc[0, "known_at"] = activity.loc[0, "target_at"] - pd.Timedelta(seconds=1)
    elif mutation == "overlap":
        activity.loc[1, "at"] = activity.loc[0, "at"] + pd.Timedelta(days=15)
        activity.loc[1, "target_at"] = activity.loc[1, "at"] + pd.Timedelta(days=30)
    else:
        activity.loc[0, "train_user"] = not activity.loc[0, "train_user"]
    with pytest.raises(ValueError, match=error):
        state.prepare_activity(activity, sessions)


def _static_prior(users):
    return {uid: {"kind": "action_only_static_return_prior", "conditioning": "source_actions_only",
        "posterior_uses_activity_labels": False, "population_model_fingerprint": "a" * 64,
        "conditioning_cutoff_at": (ORIGIN + pd.Timedelta(days=210)).isoformat(),
        "max_action_known_at": None, "offsets": np.array([-.7, .8]),
        "log_weights": np.log([.3, .7])} for uid in users}


def test_zero_dynamic_variance_equals_persistent_static_mixture_not_window_redraw():
    from scipy.special import logsumexp
    activity, sessions = _observations()
    activity = activity.loc[activity.user_id.eq(activity.user_id.iloc[0])].iloc[:2].copy()
    activity["base_logit"], activity["label"] = [.2, -.3], [0, 1]
    prepared = state.prepare_activity(activity, sessions)
    priors = _static_prior(activity.user_id.unique())
    score = state.sequence_log_likelihood(prepared, state.make_model(.7, 0.),
        label_cutoff_at=ORIGIN + pd.Timedelta(days=210), static_return_prior=priors)
    eta = prepared.base_logit.to_numpy()[:, None] + np.array([-.7, .8])[None]
    terms = prepared.label.to_numpy()[:, None] * eta - np.logaddexp(0., eta)
    expected = logsumexp(np.log([.3, .7]) + terms.sum(axis=0))
    wrong_redraw = logsumexp(np.log([.3, .7])[None] + terms, axis=1).sum()
    assert score["log_likelihood"] == pytest.approx(expected, abs=1e-12)
    assert abs(expected - wrong_redraw) > .01


def test_static_mixture_is_frozen_during_dynamic_grid_and_forbids_activity_posteriors():
    from copy import deepcopy
    activity, sessions = _observations()
    priors = _static_prior(activity.user_id.unique())
    before = {uid: (p["offsets"].copy(), p["log_weights"].copy()) for uid, p in priors.items()}
    arguments = dict(inner_cutoff_at=ORIGIN + pd.Timedelta(days=120),
        cutoff_at=ORIGIN + pd.Timedelta(days=210), observation_end_at=ORIGIN + pd.Timedelta(days=300),
        rhos=(0., .7), stationary_stds=(0., .7), quadrature_order=9, static_return_prior=priors)
    model = state.fit_return_state(activity, sessions, **arguments)
    assert model["fit"]["static_return_prior"]["population_model_fingerprint"] == "a" * 64
    for uid, (offsets, weights) in before.items():
        np.testing.assert_array_equal(priors[uid]["offsets"], offsets)
        np.testing.assert_array_equal(priors[uid]["log_weights"], weights)
    bad = deepcopy(priors)
    bad[next(iter(bad))]["posterior_uses_activity_labels"] = True
    with pytest.raises(ValueError, match="activity-conditioned"):
        state.fit_return_state(activity, sessions, **{**arguments, "static_return_prior": bad})
    with pytest.raises(ValueError, match="added twice"):
        state.fit_return_state(activity, sessions, **arguments,
            persistent_offsets={uid: 0. for uid in activity.user_id.unique()})
