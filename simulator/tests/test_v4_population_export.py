import hashlib
import json
from pathlib import Path

import pytest

from simulate.v4.irl import export_population as exporting


@pytest.mark.parametrize("with_cold_start_audit", [False, True])
@pytest.mark.parametrize("with_observation_support", [False, True])
@pytest.mark.parametrize("prepared_bundle", [None, "aligned", "wrong_return_cutoff"])
@pytest.mark.parametrize("with_compiled_metadata", [False, True])
def test_population_export_preserves_optional_cold_start_fit_and_return_binding(
        tmp_path, monkeypatch, with_cold_start_audit, with_observation_support, prepared_bundle, with_compiled_metadata):
    _check_population_export(tmp_path,monkeypatch,with_cold_start_audit,with_observation_support,
                             prepared_bundle,with_compiled_metadata)


def _check_population_export(tmp_path,monkeypatch,with_cold_start_audit,with_observation_support,
                             prepared_bundle,with_compiled_metadata,*,choice=False):
    config = Path(exporting.__file__).parents[1] / "config"
    shopping = json.loads((config / "population_shopping.json").read_text())
    returning = json.loads((config / "population_return.json").read_text())
    optional_actor = {}
    if with_compiled_metadata:
        block = shopping["latent_shopping"]
        optional_actor = {
            "component_reward_deviations": [[[0.] * 4 for _ in range(block["state_count"])] for _ in range(block["component_count"])],
            "persona_labels": [f"P{index}" for index in range(block["component_count"])],
            "linear_reward": {"type": "aggregate_export_fixture", "parameters_per_action": 80},
            "source_cutoff_at": "2020-01-29T00:00:00Z", "source_domain": "cosmetics",
        }
        block.update(optional_actor)
    if with_observation_support:
        returning["available_actions"] = ["view", "cart", "purchase"]
        returning["feature_support"] = {name: index != 5 for index, name in enumerate(returning["feature_names"])}
        shopping["available_actions"] = returning["available_actions"]
    if with_cold_start_audit:
        returning["cold_start_probability"] = 12 / 40
        returning["cold_start_fit"] = {
            "cutoff_at": "2020-01-29T00:00:00+00:00", "horizon_days": 30,
            "label_available_rule": "first_observed_session_start + 30d < cutoff",
            "training_windows": 40, "returned": 12, "heldout_users_excluded": 10,
        }
    models = tmp_path / "models"
    models.mkdir()
    source = {"dataset": "export wiring fixture", "files": [], "sample_modulus": 100}
    if prepared_bundle:
        source.update(domain="cosmetics", training_cutoff_at="2020-01-29T00:00:00Z",
                      prepared_source_cache={"sha256": "a" * 64})
        shopping["source_contract"] = {"domain": "cosmetics", "training_cutoff_at": source["training_cutoff_at"]}
        shopping["summary_scope"] = {"mode": "train_only", "cutoff_at": source["training_cutoff_at"]}
        returning["source"] = {**source, "training_cutoff_at": "2019-12-29T00:00:00Z" if prepared_bundle == "wrong_return_cutoff" else source["training_cutoff_at"]}
    return_path = models / "return_model.json"
    return_path.write_text(json.dumps(returning))
    shopping["source"], shopping["profiles"] = source, []
    shopping["latent_shopping"]["return_activity"]["base_model_sha256"] = hashlib.sha256(return_path.read_bytes()).hexdigest()
    (models / "shopping_joint_latent_source_traits.json").write_text(json.dumps(shopping))
    reward_path = state_path = None
    if with_compiled_metadata:
        count = shopping["latent_shopping"]["component_count"]
        names = ["view", "cart", "remove", "purchase", "return_logit"]
        gmm = dict(schema_version=1, type="source_persona_joint_reward_gmm", purpose="synthesis_only",
            parameter_names=names, component_count=count, mixture_weights=[1 / count] * count,
            means=[[0.] * 5 for _ in range(count)],
            covariances=[[[float(i == j) for j in range(5)] for i in range(5)] for _ in range(count)],
            base_model_sha256=hashlib.sha256((models / "shopping_joint_latent_source_traits.json").read_bytes()).hexdigest(),
            return_model_sha256=hashlib.sha256(return_path.read_bytes()).hexdigest(),
            component_model_indices=list(range(count)), choice_temperatures=[.75] * count,
            dispersion_scales=[.5] * count, condition_on_profile_components=True,
            persona_labels=optional_actor["persona_labels"], component_fitted_parameter_names=[names.copy() for _ in range(count)])
        if with_observation_support:
            gmm.update(available_actions=returning["available_actions"],
                       fitted_parameter_names=[name for name in names if name != "remove"])
            gmm["component_fitted_parameter_names"] = [gmm["fitted_parameter_names"].copy() for _ in range(count)]
        if choice:
            from simulate.v4.irl import choice_gmm
            active=[0,1,3,4] if with_observation_support else list(range(5))
            gmm.update(type=choice_gmm.TYPE,backend=choice_gmm.BACKEND,
                       shopping_dispersion_scales=[0.] * count,return_dispersion_scales=[1.] * count,
                       parameter_names=choice_gmm.PARAMETERS.copy(),parameter_units=choice_gmm.UNITS.copy(),
                       fitted_parameter_names=[choice_gmm.PARAMETERS[i] for i in active],
                       component_fitted_parameter_names=[[choice_gmm.PARAMETERS[i] for i in active] for _ in range(count)])
        reward_path = models / "reward_gmm.json"
        reward_path.write_text(json.dumps(gmm))
        from simulate.v4.irl import return_state
        dynamic = return_state.make_model(.5, .25)
        dynamic["fit"] = {"static_return_prior": {"population_model_fingerprint": hashlib.sha256(json.dumps(
            gmm, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}}
        dynamic["profile_posteriors"] = {"private-source-user": [1.]}
        state_path = models / "return_state.json"
        state_path.write_text(json.dumps(dynamic))
    (models / "tianchi_coupon_response.json").write_text(json.dumps({
        "schema_version": 1, "type": "fixture", "discount_rate": 0.1,
        "window_days": 30, "inactivity_days": 28, "groups": {},
    }))
    (models / "visit_timing_v4.json").write_text(json.dumps({"source": source}))
    (models / "review_events_six_v4.json").write_text("{}")
    monkeypatch.setattr(exporting, "aggregate_population", lambda model: model["population"])
    monkeypatch.setattr(exporting, "aggregate_visit_timing", lambda model, profiles: {})
    monkeypatch.setattr(exporting, "aggregate_review_model", lambda model: {})

    output = tmp_path / "population"
    if prepared_bundle == "wrong_return_cutoff":
        with pytest.raises(ValueError, match="fitting cutoff"):
            exporting.export_population(models, output, reward_path, return_state=state_path)
        assert not output.exists()
        return
    exporting.export_population(models, output, reward_path, return_state=state_path)

    exported = json.loads((output / "population_return.json").read_text())
    assert exported == {name: value for name, value in returning.items() if name != "source"}
    bound = json.loads((output / "population_shopping.json").read_text())
    if with_compiled_metadata:
        for name, value in optional_actor.items():
            assert bound["latent_shopping"][name] == value
        projected = json.loads((output / "population_gmm.json").read_text())
        for name in ("component_model_indices", "choice_temperatures", "dispersion_scales", "condition_on_profile_components",
                     "persona_labels", "component_fitted_parameter_names"):
            assert projected[name] == gmm[name]
        if choice:
            from simulate.v4.irl import choice_gmm
            for name in ('type','backend','parameter_names','parameter_units','fitted_parameter_names',
                         'shopping_dispersion_scales','return_dispersion_scales'):
                assert projected[name]==gmm[name]
            choice_gmm.validate_choice_gmm(projected,hashlib.sha256((output/'population_shopping.json').read_bytes()).hexdigest(),
                hashlib.sha256((output/'population_return.json').read_bytes()).hexdigest())
        exported_state = json.loads((output / "population_return_state.json").read_text())
        assert exported_state["rho"] == .5 and exported_state["stationary_std"] == .25
        assert "profile_posteriors" not in exported_state
        assert exported_state["runtime_gmm_fingerprint"] == hashlib.sha256(json.dumps(
            projected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if with_observation_support:
        assert bound["available_actions"] == returning["available_actions"]
    assert bound["latent_shopping"]["return_activity"]["base_model_sha256"] == hashlib.sha256(
        (output / "population_return.json").read_bytes()).hexdigest()


@pytest.mark.parametrize('with_observation_support',[False,True])
def test_choice_gmm_export_preserves_separate_units_personas_and_exact_binding(tmp_path,monkeypatch,with_observation_support):
    _check_population_export(tmp_path,monkeypatch,False,with_observation_support,'aligned',True,choice=True)
