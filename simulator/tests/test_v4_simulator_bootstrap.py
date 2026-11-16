from types import SimpleNamespace
import json

import pytest

from simulate.v4.data import make_dataset


def test_missing_simulator_model_uses_same_checkout_item_recipe_and_preserves_paths(tmp_path, monkeypatch):
    simulator = SimpleNamespace(model_path=tmp_path / "a model.json", settings={
        "fit_checkout_episodes": True, "fit_item_choice": True, "return_model_path": None})
    calls = []
    monkeypatch.setattr(make_dataset.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1
    command, arguments = calls[0]
    assert command[1:3] == ["-m", "simulate.v4.irl.fit_shopping"]
    assert command[command.index("--output") + 1] == str(simulator.model_path)
    assert command[-2:] == ["--checkout-episodes", "--item-choice"]
    assert arguments == {"check": True}


def test_existing_source_artifacts_are_not_refit(tmp_path, monkeypatch):
    model, returning = tmp_path / "shopping.json", tmp_path / "return.json"
    model.write_text("immutable shopping fixture")
    returning.write_text("immutable return fixture")
    simulator = SimpleNamespace(model_path=model, settings={"return_model_path": str(returning)})
    monkeypatch.setattr(make_dataset.subprocess, "run", lambda *a, **k: pytest.fail("existing source artifact was refit"))
    make_dataset.ensure_source_models(simulator)
    assert model.read_text() == "immutable shopping fixture"


def test_invalid_item_recipe_fails_before_fitting(tmp_path, monkeypatch):
    simulator = SimpleNamespace(model_path=tmp_path / "model.json", settings={"fit_item_choice": True})
    monkeypatch.setattr(make_dataset.subprocess, "run", lambda *a, **k: pytest.fail("invalid recipe reached fitting"))
    with pytest.raises(ValueError, match="checkout"):
        make_dataset.ensure_source_models(simulator)


def test_missing_coupon_component_uses_configured_raw_source(tmp_path, monkeypatch):
    model, source = tmp_path / "shopping.json", tmp_path / "raw source.csv"
    model.touch()
    source.touch()
    response = tmp_path / "coupon model.json"
    simulator = SimpleNamespace(model_path=model, settings={
        "coupon_response_model_path": str(response), "coupon_response_source": str(source)})
    calls = []
    monkeypatch.setattr(make_dataset.subprocess, "run", lambda command, **kwargs: calls.append(command))
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1
    assert calls[0][1:] == ["-m", "simulate.v4.irl.fit_tianchi_coupon", "--source", str(source), "--output", str(response)]


def test_missing_coupon_raw_data_fails_instead_of_inventing_response(tmp_path):
    model = tmp_path / "shopping.json"
    model.touch()
    simulator = SimpleNamespace(model_path=model, settings={
        "coupon_response_model_path": str(tmp_path / "response.json"),
        "coupon_response_source": str(tmp_path / "absent.csv")})
    with pytest.raises(FileNotFoundError, match="no invented"):
        make_dataset.ensure_source_models(simulator)


@pytest.mark.parametrize("name", ["review"])
def test_missing_auxiliary_artifact_fits_configured_real_source_once(tmp_path, monkeypatch, name):
    shopping = tmp_path / "shopping.json"
    shopping.touch()
    source = tmp_path / "raw source"
    source.touch()
    model = tmp_path / f"{name} model.json"
    settings = {f"{name}_model_path": str(model)}
    if name == "review":
        settings.update(review_amazon_source=str(source), review_olist_source=str(source))
    else:
        settings[f"{name}_source"] = str(source)
    calls = []
    monkeypatch.setattr(make_dataset.subprocess, "run", lambda command, **kwargs: calls.append(command))
    simulator = SimpleNamespace(model_path=shopping, settings=settings)
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1 and calls[0][1:3] == ["-m", f"simulate.v4.irl.fit_{name}_events"]
    assert calls[0][-2:] == ["--output", str(model)]
    assert str(source) in calls[0]
    model.touch()
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1


def test_retired_search_and_cs_paths_never_trigger_fitting(tmp_path, monkeypatch):
    shopping = tmp_path / "shopping.json"
    shopping.touch()
    settings = {f"{name}_model_path": str(tmp_path/f"missing_{name}.json") for name in ("search", "cs")}
    def forbidden(*args, **kwargs):
        raise AssertionError("Removed event model training was invoked")
    monkeypatch.setattr(make_dataset.subprocess, "run", forbidden)
    make_dataset.ensure_source_models(SimpleNamespace(model_path=shopping, settings=settings))


@pytest.mark.parametrize("base_exists", [False, True])
def test_joint_policy_bootstrap_uses_source_fitted_recipe_not_plain_shopping(tmp_path, monkeypatch, base_exists):
    base, target = tmp_path / "base.json", tmp_path / "joint.json"
    if base_exists:
        base.write_text("original base")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        output = command[command.index("--output") + 1]
        from pathlib import Path
        Path(output).write_text(json.dumps({"donor_prior": {"policy_calibration": {"fixture": True}}}))
    monkeypatch.setattr(make_dataset.subprocess, "run", run)
    simulator = SimpleNamespace(model_path=target, settings={"donor_prior_base_model_path": str(base)})
    make_dataset.ensure_source_models(simulator)
    assert calls[-1][1:] == ["-m", "simulate.v4.irl.fit_donor_prior", "--base", str(base),
                            "--joint-policy", "--output", str(target)]
    if base_exists:
        assert len(calls) == 1 and base.read_text() == "original base"
    else:
        assert len(calls) == 2 and "--checkout-episodes" in calls[0]
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == (1 if base_exists else 2)


def test_existing_joint_artifact_needs_no_refit_or_base_copy(tmp_path, monkeypatch):
    model = tmp_path / "joint.json"
    model.write_text(json.dumps({"donor_prior": {"policy_calibration": {"fixture": True}}}))
    simulator = SimpleNamespace(model_path=model, settings={"donor_prior_base_model_path": str(tmp_path / "absent.json")})
    monkeypatch.setattr(make_dataset.subprocess, "run", lambda *a, **k: pytest.fail("existing joint artifact was refit"))
    make_dataset.ensure_source_models(simulator)


def test_joint_recipe_rejects_plain_or_overlapping_artifact(tmp_path):
    model = tmp_path / "plain.json"
    model.write_text("{}")
    simulator = SimpleNamespace(model_path=model, settings={"donor_prior_base_model_path": str(tmp_path / "base.json")})
    with pytest.raises(ValueError, match="joint-policy artifact"):
        make_dataset.ensure_source_models(simulator)
    simulator.settings["donor_prior_base_model_path"] = str(model)
    with pytest.raises(ValueError, match="separate artifact"):
        make_dataset.ensure_source_models(simulator)


def test_checkout_recipe_builds_each_source_fitted_stage_and_then_reuses(tmp_path, monkeypatch):
    base, policy, final = (tmp_path / name for name in ("base.json", "policy.json", "final.json"))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        from pathlib import Path
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "donor_prior": {"policy_calibration": {}}, "checkout_kernel": {"fixture": True}}))
    monkeypatch.setattr(make_dataset.subprocess, "run", run)
    simulator = SimpleNamespace(model_path=final, settings={"donor_prior_base_model_path": str(base),
        "checkout_transition_base_model_path": str(policy)})
    make_dataset.ensure_source_models(simulator)
    assert [command[2] for command in calls] == ["simulate.v4.irl.fit_shopping", "simulate.v4.irl.fit_donor_prior", "simulate.v4.irl.fit_checkout_transition"]
    assert calls[-1][3:] == ["--base", str(policy), "--output", str(final), "--report", str(final.with_suffix(".training.json"))]
    base.unlink()
    policy.unlink()
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 3


def test_checkout_recipe_rejects_a_plain_policy_in_the_final_path(tmp_path):
    final = tmp_path / "final.json"
    final.write_text("{}")
    simulator = SimpleNamespace(model_path=final, settings={"checkout_transition_base_model_path": str(tmp_path / "policy.json")})
    with pytest.raises(ValueError, match="fitted kernel"):
        make_dataset.ensure_source_models(simulator)


@pytest.mark.parametrize('basis', ['continuous', 'tabular'])
@pytest.mark.parametrize('support', ['matching-sku', 'known-total'])
@pytest.mark.parametrize('target', ['customer-macro', 'pooled-session'])
def test_causal_policy_bootstrap_wraps_existing_base_and_refuses_silent_plain_fallback(tmp_path, monkeypatch, basis, support, target):
    from simulate.v4.irl import fit_causal_shopping as causal
    base, final = tmp_path/'checkout.json', tmp_path/'causal.json'
    base.write_text('immutable base')
    calls = []
    def fit(command, **kwargs):
        calls.append(command)
        final.write_text(json.dumps({'causal_shopping': {'fixture': True, 'reward_basis_type': basis,
            'support_rule': support, 'fit_target': target}}))
    def validate(model):
        if not model.get('causal_shopping'):
            raise ValueError('incompatible causal shopping model')
    monkeypatch.setattr(make_dataset.subprocess, 'run', fit)
    monkeypatch.setattr(causal, 'validate_model', validate)
    simulator = SimpleNamespace(model_path=final, settings={'causal_policy_base_model_path': str(base),
        'causal_reward_basis': basis, 'causal_cart_support': support, 'causal_fit_target': target})
    make_dataset.ensure_source_models(simulator)
    assert len(calls)==1 and calls[0][1:]==['-m','simulate.v4.irl.fit_causal_shopping','--base',str(base),
        '--output',str(final),'--report',str(final.with_suffix('.training.json')), '--reward-basis', basis,
        '--cart-support', support, '--fit-target', target]
    assert base.read_text()=='immutable base'
    make_dataset.ensure_source_models(simulator)
    assert len(calls)==1
    final.write_text(json.dumps({'causal_shopping': {'reward_basis_type': 'tabular' if basis=='continuous' else 'continuous'}}))
    with pytest.raises(ValueError, match='reward basis disagrees'):
        make_dataset.ensure_source_models(simulator)
    final.write_text(json.dumps({'causal_shopping': {'reward_basis_type': basis,
        'support_rule': 'known-total' if support=='matching-sku' else 'matching-sku'}}))
    with pytest.raises(ValueError, match='cart support disagrees'):
        make_dataset.ensure_source_models(simulator)
    final.write_text(json.dumps({'causal_shopping': {'reward_basis_type': basis, 'support_rule': support,
        'fit_target': 'pooled-session' if target=='customer-macro' else 'customer-macro'}}))
    with pytest.raises(ValueError, match='fit target disagrees'):
        make_dataset.ensure_source_models(simulator)
    final.write_text('{}')
    with pytest.raises(ValueError,match='causal shopping'):
        make_dataset.ensure_source_models(simulator)
    simulator.settings['causal_policy_base_model_path']=str(final)
    with pytest.raises(ValueError,match='separate artifact'):
        make_dataset.ensure_source_models(simulator)


def test_causal_basis_requires_a_valid_recipe(tmp_path):
    for settings in ({'causal_reward_basis': 'tabular'},
                     {'causal_reward_basis': 'invented', 'causal_policy_base_model_path': str(tmp_path/'base.json')}):
        with pytest.raises(ValueError, match='configured causal-policy recipe'):
            make_dataset.ensure_source_models(SimpleNamespace(model_path=tmp_path/'final.json', settings=settings))

    for settings in ({'causal_cart_support': 'known-total'},
                     {'causal_cart_support': 'invented', 'causal_policy_base_model_path': str(tmp_path/'base.json')}):
        with pytest.raises(ValueError, match='configured causal-policy recipe'):
            make_dataset.ensure_source_models(SimpleNamespace(model_path=tmp_path/'final.json', settings=settings))
    for settings in ({'causal_fit_target': 'pooled-session'},
                     {'causal_fit_target': 'invented', 'causal_policy_base_model_path': str(tmp_path/'base.json')}):
        with pytest.raises(ValueError, match='configured causal-policy recipe'):
            make_dataset.ensure_source_models(SimpleNamespace(model_path=tmp_path/'final.json', settings=settings))


@pytest.mark.parametrize('candidates', [[2], [3, 4]])
def test_latent_recipe_wraps_existing_base_and_rejects_plain_artifacts(tmp_path, monkeypatch, candidates):
    from simulate.v4.irl import fit_latent_shopping as latent
    base, final = tmp_path/'checkout.json', tmp_path/'latent.json'
    base.write_text('immutable source base')
    calls = []
    def fit(command, **kwargs):
        calls.append(command)
        final.write_text(json.dumps({'latent_shopping': {'fixture': True, 'training': {'component_candidates': candidates}}}))
    def validate(model):
        if not model.get('latent_shopping'):
            raise ValueError('incompatible latent shopping model')
    monkeypatch.setattr(make_dataset.subprocess, 'run', fit)
    monkeypatch.setattr(latent, 'validate_model', validate)
    simulator = SimpleNamespace(model_path=final, settings={'latent_policy_base_model_path': str(base),
                                                        'latent_component_candidates': candidates})
    make_dataset.ensure_source_models(simulator)
    assert calls[0][1:] == ['-m', 'simulate.v4.irl.fit_latent_shopping', '--base', str(base), '--output', str(final),
                           '--report', str(final.with_suffix('.training.json')), '--component-candidates', *map(str, candidates)]
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1
    assert base.read_text() == 'immutable source base'
    final.write_text(json.dumps({'latent_shopping': {'training': {'component_candidates': [4]}}}))
    with pytest.raises(ValueError, match='component candidates disagree'):
        make_dataset.ensure_source_models(simulator)
    final.write_text('{}')
    with pytest.raises(ValueError, match='latent shopping'):
        make_dataset.ensure_source_models(simulator)
    simulator.settings['causal_policy_base_model_path'] = str(tmp_path/'another.json')
    with pytest.raises(ValueError, match='one causal or latent'):
        make_dataset.ensure_source_models(simulator)


@pytest.mark.parametrize('candidates', [[], [2, 2], [1], [True], '2', [2.5], None])
def test_invalid_latent_component_recipe_is_rejected_before_training(tmp_path, candidates):
    settings = {'latent_policy_base_model_path': str(tmp_path/'base.json'), 'latent_component_candidates': candidates}
    with pytest.raises(ValueError, match='valid latent-policy recipe'):
        make_dataset.ensure_source_models(SimpleNamespace(model_path=tmp_path/'final.json', settings=settings))


def test_joint_latent_recipe_fits_and_checks_the_actual_return_base(tmp_path, monkeypatch):
    import hashlib
    from simulate.v4.irl import fit_joint_latent_shopping as joint
    base, returns, final = (tmp_path/name for name in ('base.json', 'returns.json', 'joint.json'))
    base.write_text('immutable shopping base')
    returns.write_text('immutable return base')
    calls, hashes = [], []
    def fit(command, **kwargs):
        calls.append(command)
        final.write_text(json.dumps({'latent_shopping': {'return_activity': {}, 'training': {'component_candidates': [4]}}}))
    def validate(model, return_model_sha256=None):
        hashes.append(return_model_sha256)
    monkeypatch.setattr(make_dataset.subprocess, 'run', fit)
    monkeypatch.setattr(joint, 'validate_model', validate)
    settings = dict(latent_policy_base_model_path=str(base), latent_component_candidates=[4],
                    latent_joint_activity=True, return_model_path=str(returns))
    simulator = SimpleNamespace(model_path=final, settings=settings)
    make_dataset.ensure_source_models(simulator)
    assert calls[0][1:] == ['-m', 'simulate.v4.irl.fit_joint_latent_shopping', '--base', str(base), '--output', str(final),
                           '--report', str(final.with_suffix('.training.json')), '--return-model', str(returns)]
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1
    assert hashes == [hashlib.sha256(returns.read_bytes()).hexdigest()] * 2
    final.write_text(json.dumps({'latent_shopping': {'training': {'component_candidates': [4]}}}))
    with pytest.raises(ValueError, match='requires its fitted activity'):
        make_dataset.ensure_source_models(simulator)


@pytest.mark.parametrize('settings', [
    {'latent_joint_activity': True},
    {'latent_joint_activity': 'yes'},
    {'latent_joint_activity': True, 'latent_policy_base_model_path': 'base', 'latent_component_candidates': [4]},
    {'latent_joint_activity': True, 'latent_policy_base_model_path': 'base', 'return_model_path': 'returns'},
])
def test_joint_activity_requires_its_complete_source_recipe(tmp_path, settings):
    with pytest.raises(ValueError, match='joint latent activity requires'):
        make_dataset.ensure_source_models(SimpleNamespace(model_path=tmp_path/'unused.json', settings=settings))


def test_native_timing_missing_artifact_fits_once_from_the_actual_shopping_source(tmp_path, monkeypatch):
    shopping, timing = tmp_path/'shopping.json', tmp_path/'timing.json'
    shopping.touch()
    calls = []
    def fit(command, **kwargs):
        calls.append(command)
        timing.write_text('{}')
    monkeypatch.setattr(make_dataset.subprocess, 'run', fit)
    sim = SimpleNamespace(model_path=shopping, settings={'visit_timing_model_path': str(timing)})
    make_dataset.ensure_source_models(sim)
    make_dataset.ensure_source_models(sim)
    assert len(calls) == 1 and calls[0][1:] == ['-m','simulate.v4.irl.fit_visit_timing','--base',str(shopping),
        '--output',str(timing),'--report',str(timing.with_suffix('.training.json'))]


def test_source_trait_recipe_reconstructs_and_guards_the_exact_parent(tmp_path, monkeypatch):
    import hashlib
    from simulate.v4.irl import fit_latent_shopping as latent
    parent, final = tmp_path/'parent.json', tmp_path/'traits.json'
    parent.write_text(json.dumps({'latent_shopping': {}}))
    calls = []
    def fit(command, **kwargs):
        calls.append(command)
        final.write_text(json.dumps({'latent_shopping': {'profile_posterior_assignment_scope':'source_observation_trait_posterior',
            'trait_reconstruction': {'parent_model_sha256': hashlib.sha256(parent.read_bytes()).hexdigest()}}}))
    monkeypatch.setattr(make_dataset.subprocess, 'run', fit)
    monkeypatch.setattr(latent, 'validate_model', lambda m: None)
    sim = SimpleNamespace(model_path=final, settings={'latent_trait_base_model_path': str(parent)})
    make_dataset.ensure_source_models(sim)
    make_dataset.ensure_source_models(sim)
    assert len(calls) == 1 and calls[0][1:] == ['-m','simulate.v4.irl.reconstruct_latent_traits','--base',str(parent),
        '--output',str(final),'--report',str(final.with_suffix('.traits.json'))]
    original = final.read_text()
    changed = json.loads(original)
    changed['latent_shopping']['changed_reward'] = True
    final.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed fitted parent parameters'):
        make_dataset.ensure_source_models(sim)
    final.write_text(original)
    parent.write_text('changed parent')
    with pytest.raises(ValueError, match='exact fitted parent'):
        make_dataset.ensure_source_models(sim)
    with pytest.raises(ValueError, match='separate parent'):
        make_dataset.ensure_source_models(SimpleNamespace(model_path=parent, settings=sim.settings))

@pytest.mark.parametrize(("name", "option", "path"), [
    ("fit_shopping", "--cosmetics", "raw cosmetics"),
    ("fit_return", "--cosmetics", "raw cosmetics"),
    ("fit_joint_latent_shopping", "--source-cache", "prepared cache.parquet"),
    ("fit_visit_timing", "--source-cache", "prepared cache.parquet"),
    ("reconstruct_latent_traits", "--source-cache", "prepared cache.parquet"),
    ("fit_gaussian_rewards", "--source-cache", "prepared cache.parquet"),
])
def test_portable_source_paths_reach_fitter_without_shell_parsing(name, option, path):
    simulator = SimpleNamespace(settings={"cosmetics_source": "raw cosmetics",
                                          "prepared_source_cache": "prepared cache.parquet"})
    command = make_dataset._fit_command(simulator, name)
    assert command[1:3] == ["-m", f"simulate.v4.irl.{name}"]
    assert command[3:] == [option, path]


def test_checkout_bootstrap_supplies_the_required_training_report(tmp_path, monkeypatch):
    base, output = tmp_path / "policy.json", tmp_path / "checkout.json"
    base.touch()
    simulator = SimpleNamespace(model_path=output, settings={"checkout_transition_base_model_path": str(base)})
    calls = []
    class FittingStarted(Exception):
        pass
    def capture(command, **kwargs):
        calls.append(command)
        raise FittingStarted
    monkeypatch.setattr(make_dataset.subprocess, "run", capture)
    with pytest.raises(FittingStarted):
        make_dataset.ensure_source_models(simulator)
    command, = calls
    assert command[2] == "simulate.v4.irl.fit_checkout_transition"
    assert command[command.index("--report") + 1] == str(output.with_suffix(".training.json"))


def test_gaussian_rewards_fit_once_after_source_traits_without_refitting_parent(tmp_path, monkeypatch):
    import hashlib
    from simulate.v4.irl import fit_latent_shopping as latent

    parent, traits, returning, gaussian = [tmp_path / name for name in (
        "parent.json", "traits.json", "return.json", "gaussian.json")]
    parent.write_text(json.dumps({"latent_shopping": {}}))
    returning.write_text("frozen return")
    traits.write_text(json.dumps({"latent_shopping": {
        "profile_posterior_assignment_scope": "source_observation_trait_posterior",
        "trait_reconstruction": {"parent_model_sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}}}))
    monkeypatch.setattr(latent, "validate_model", lambda model: None)
    calls = []

    def fit(command, **kwargs):
        calls.append(command)
        gaussian.write_text("fitted Gaussian")

    monkeypatch.setattr(make_dataset.subprocess, "run", fit)
    simulator = SimpleNamespace(model_path=traits, settings={
        "latent_trait_base_model_path": str(parent), "return_model_path": str(returning),
        "gaussian_reward_model_path": str(gaussian), "prepared_source_cache": "source cache.parquet"})
    make_dataset.ensure_source_models(simulator)
    make_dataset.ensure_source_models(simulator)
    assert len(calls) == 1
    assert calls[0][1:] == ["-m", "simulate.v4.irl.fit_gaussian_rewards",
        "--source-cache", "source cache.parquet", "--base", str(traits), "--return-model", str(returning),
        "--output", str(gaussian), "--report", str(gaussian.with_suffix(".training.json"))]


def test_gaussian_fitting_requires_the_return_model(tmp_path):
    model = tmp_path / "shopping.json"
    model.touch()
    with pytest.raises(ValueError, match="requires a fitted return model"):
        make_dataset.ensure_source_models(SimpleNamespace(model_path=model, settings={
            "gaussian_reward_model_path": str(tmp_path / "gaussian.json")}))
