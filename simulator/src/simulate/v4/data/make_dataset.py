"""Generate immutable raw SQLite records and explicit outcome sidecars."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from simulate.v4.simulator import EcommerceSimulator


REWARD_FITTERS = {
    "gaussian_reward_model_path": "fit_gaussian_rewards",
    "reward_gmm_model_path": "fit_joint_reward_gmm",
}


def _fit_command(simulator, name):
    command = [sys.executable, "-m", f"simulate.v4.irl.{name}"]
    settings = simulator.settings
    if name in {"fit_shopping", "fit_return"}:
        if settings.get("cosmetics_source"):
            command += ["--cosmetics", str(settings["cosmetics_source"])]
    elif name in {"fit_donor_prior", "fit_checkout_transition", "fit_causal_shopping", "fit_latent_shopping",
                  "fit_joint_latent_shopping", "fit_visit_timing", "reconstruct_latent_traits", *REWARD_FITTERS.values()}:
        if settings.get("prepared_source_cache"):
            command += ["--source-cache", str(settings["prepared_source_cache"])]
    return command


def _ensure_visit_timing(simulator):
    path = simulator.settings.get("visit_timing_model_path")
    if path and not Path(path).exists():
        subprocess.run(_fit_command(simulator, "fit_visit_timing") + [ "--base", str(simulator.model_path),
                        "--output", str(path), "--report", str(Path(path).with_suffix(".training.json"))], check=True)


def _ensure_reward_model(simulator):
    for key, module in REWARD_FITTERS.items():
        path = simulator.settings.get(key)
        if path and not Path(path).exists():
            returning = simulator.settings.get("return_model_path")
            if not returning:
                raise ValueError("continuous reward fitting requires a fitted return model")
            subprocess.run(_fit_command(simulator, module) + [
                "--base", str(simulator.model_path), "--return-model", str(returning),
                "--output", str(path), "--report", str(Path(path).with_suffix(".training.json"))], check=True)


def ensure_source_models(simulator):
    """Fit only missing simulator components with the configured source recipe."""
    settings = simulator.settings
    if all(settings.get(key) for key in REWARD_FITTERS):
        raise ValueError("configure one continuous reward model at a time")
    trait_base = settings.get("latent_trait_base_model_path")
    if trait_base:
        parent = Path(trait_base)
        recipe_paths = [parent, *[Path(settings[k]) for k in ("latent_policy_base_model_path", "causal_policy_base_model_path",
                        "checkout_transition_base_model_path", "donor_prior_base_model_path") if settings.get(k)]]
        if simulator.model_path.resolve() in {p.resolve() for p in recipe_paths}:
            raise ValueError("latent trait reconstruction requires a separate parent artifact")
        parent_settings = {k: v for k, v in settings.items() if k not in {
            "latent_trait_base_model_path", "visit_timing_model_path", *REWARD_FITTERS}}
        parent_settings["model_path"] = str(parent)
        ensure_source_models(SimpleNamespace(model_path=parent, settings=parent_settings))
        if not simulator.model_path.exists():
            subprocess.run(_fit_command(simulator, "reconstruct_latent_traits") + [ "--base", str(parent),
                            "--output", str(simulator.model_path),
                            "--report", str(simulator.model_path.with_suffix(".traits.json"))], check=True)
        from simulate.v4.irl.fit_latent_shopping import validate_model, trait_invariant_sha256
        artifact = json.loads(simulator.model_path.read_text())
        validate_model(artifact)
        block = artifact.get("latent_shopping", {})
        if (block.get("profile_posterior_assignment_scope") != "source_observation_trait_posterior" or
                block.get("trait_reconstruction", {}).get("parent_model_sha256") != hashlib.sha256(parent.read_bytes()).hexdigest()):
            raise ValueError("latent trait reconstruction requires its exact fitted parent")
        parent_block = json.loads(parent.read_text()).get("latent_shopping", {})
        if trait_invariant_sha256(block) != trait_invariant_sha256(parent_block):
            raise ValueError("latent trait reconstruction changed fitted parent parameters")
        _ensure_visit_timing(simulator)
        _ensure_reward_model(simulator)
        return
    if settings.get("fit_item_choice") and not settings.get("fit_checkout_episodes"):
        raise ValueError("item-choice fitting requires checkout episodes")
    prior_base = settings.get("donor_prior_base_model_path")
    checkout_base = settings.get("checkout_transition_base_model_path")
    causal_base = settings.get("causal_policy_base_model_path")
    latent_base = settings.get("latent_policy_base_model_path")
    if causal_base and latent_base:
        raise ValueError("source recipe must select one causal or latent planning policy")
    latent_candidates = settings.get("latent_component_candidates", [2])
    if (not isinstance(latent_candidates, (list, tuple)) or not latent_candidates
            or any(isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 4 for k in latent_candidates)
            or len(set(latent_candidates)) != len(latent_candidates)
            or ("latent_component_candidates" in settings and not latent_base)):
        raise ValueError("latent component candidates require a valid latent-policy recipe")
    joint_activity = settings.get("latent_joint_activity", False)
    return_model = settings.get("return_model_path")
    if (not isinstance(joint_activity, bool) or (joint_activity and
            (not latent_base or list(latent_candidates) != [4] or not return_model))):
        raise ValueError("joint latent activity requires a K4 latent-policy recipe and base return model")
    causal_basis = settings.get("causal_reward_basis", "continuous")
    if causal_basis not in {"continuous", "tabular"} or ("causal_reward_basis" in settings and not causal_base):
        raise ValueError("causal reward basis requires its configured causal-policy recipe")
    causal_support = settings.get("causal_cart_support", "matching-sku")
    if causal_support not in {"matching-sku", "known-total"} or ("causal_cart_support" in settings and not causal_base):
        raise ValueError("causal cart support requires its configured causal-policy recipe")
    causal_target = settings.get("causal_fit_target", "customer-macro")
    if causal_target not in {"customer-macro", "pooled-session"} or ("causal_fit_target" in settings and not causal_base):
        raise ValueError("causal fit target requires its configured causal-policy recipe")
    planner_base = latent_base or causal_base
    checkout_path = Path(planner_base) if planner_base else simulator.model_path
    policy_path = Path(checkout_base) if checkout_base else checkout_path
    stages = [simulator.model_path, *([Path(planner_base)] if planner_base else []),
              *([Path(checkout_base)] if checkout_base else []), *([Path(prior_base)] if prior_base else [])]
    if len({path.resolve() for path in stages}) != len(stages):
        raise ValueError("source fitting stages must use separate artifact paths")
    if return_model and not Path(return_model).exists():
        subprocess.run(_fit_command(simulator, "fit_return") + [ "--output", return_model], check=True)
    if not simulator.model_path.exists():
        if not checkout_path.exists():
            if not policy_path.exists():
                shopping_path = Path(prior_base) if prior_base else policy_path
                if not shopping_path.exists():
                    command = _fit_command(simulator, "fit_shopping") + [ "--output", str(shopping_path)]
                    if settings.get("fit_checkout_episodes") or prior_base or checkout_base or planner_base:
                        command.append("--checkout-episodes")
                    if settings.get("fit_item_choice"):
                        command.append("--item-choice")
                    subprocess.run(command, check=True)
                if prior_base:
                    subprocess.run(_fit_command(simulator, "fit_donor_prior") + [ "--base", str(shopping_path),
                                    "--joint-policy", "--output", str(policy_path)], check=True)
            if checkout_base:
                subprocess.run(_fit_command(simulator, "fit_checkout_transition") + [ "--base", str(policy_path),
                                "--output", str(checkout_path),
                                "--report", str(checkout_path.with_suffix(".training.json"))], check=True)
        if latent_base:
            command = _fit_command(simulator, "fit_joint_latent_shopping" if joint_activity else "fit_latent_shopping") + [
                       "--base", str(checkout_path), "--output", str(simulator.model_path),
                       "--report", str(simulator.model_path.with_suffix(".training.json"))]
            command += (["--return-model", return_model] if joint_activity else
                        ["--component-candidates", *map(str, latent_candidates)])
            subprocess.run(command, check=True)
        elif causal_base:
            subprocess.run(_fit_command(simulator, "fit_causal_shopping") + [ "--base", str(checkout_path),
                            "--output", str(simulator.model_path),
                            "--report", str(simulator.model_path.with_suffix(".training.json")),
                            "--reward-basis", causal_basis, "--cart-support", causal_support,
                            "--fit-target", causal_target], check=True)
    artifact = json.loads(simulator.model_path.read_text()) if prior_base or checkout_base or planner_base else None
    if prior_base:
        block = artifact.get("donor_prior")
        if not isinstance(block, dict) or not isinstance(block.get("policy_calibration"), dict):
            raise ValueError("configured donor-prior recipe requires its fitted joint-policy artifact")
    if checkout_base and not isinstance(artifact.get("checkout_kernel"), dict):
        raise ValueError("configured checkout-transition recipe requires its fitted kernel")
    if causal_base:
        from simulate.v4.irl.fit_causal_shopping import validate_model
        validate_model(artifact)
        if artifact["causal_shopping"].get("reward_basis_type", "continuous") != causal_basis:
            raise ValueError("configured causal reward basis disagrees with the fitted artifact")
        if artifact["causal_shopping"].get("support_rule", "matching-sku") != causal_support:
            raise ValueError("configured causal cart support disagrees with the fitted artifact")
        if artifact["causal_shopping"].get("fit_target", "customer-macro") != causal_target:
            raise ValueError("configured causal fit target disagrees with the fitted artifact")
    if latent_base:
        if joint_activity:
            from simulate.v4.irl.fit_joint_latent_shopping import validate_model
            if artifact.get("latent_shopping", {}).get("return_activity") is None:
                raise ValueError("joint latent activity recipe requires its fitted activity component")
            validate_model(artifact, return_model_sha256=hashlib.sha256(Path(return_model).read_bytes()).hexdigest())
        else:
            from simulate.v4.irl.fit_latent_shopping import validate_model
            validate_model(artifact)
            if artifact["latent_shopping"].get("return_activity") is not None:
                raise ValueError("joint latent activity artifact requires its configured recipe")
        if artifact["latent_shopping"].get("training", {}).get("component_candidates", [2]) != list(latent_candidates):
            raise ValueError("configured latent component candidates disagree with the fitted artifact")
    coupon_model = settings.get("coupon_response_model_path")
    if coupon_model and not Path(coupon_model).exists():
        source = settings.get("coupon_response_source")
        if not source or not Path(source).is_file():
            raise FileNotFoundError("configured Tianchi O2O source CSV is required; no invented coupon response fallback")
        subprocess.run(_fit_command(simulator, "fit_tianchi_coupon") + [ "--source", source,
                        "--output", coupon_model], check=True)
    _ensure_visit_timing(simulator)
    recipes = {
        "review": ["--amazon", settings.get("review_amazon_source", "data/raw/amazon_reviews_2023/All_Beauty.jsonl.gz"),
                   "--olist", settings.get("review_olist_source", "data/raw/kaggle/olist")],
    }
    for name, arguments in recipes.items():
        path = settings.get(f"{name}_model_path")
        if path and not Path(path).exists():
            for source in arguments[1::2]:
                if not Path(source).exists():
                    raise FileNotFoundError(f"{name} event model needs its configured real source: {source}")
            subprocess.run([sys.executable, "-m", f"simulate.v4.irl.fit_{name}_events", *arguments,
                            "--output", path], check=True)
    _ensure_reward_model(simulator)
