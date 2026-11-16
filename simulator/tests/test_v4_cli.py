from pathlib import Path
from types import SimpleNamespace

import pytest

from simulate.v4.cli import DEFAULT_CONFIG, load_config, run


def test_bundled_config_resolves_models_and_calibrations_separately(tmp_path):
    config = load_config(assets_root=tmp_path)
    assert config["source_simulation"]["model_path"] == str(tmp_path / "data/models/shopping_joint_latent_source_traits.json")
    assert config["source_simulation"]["gaussian_reward_model_path"] == str(tmp_path / "data/models/shopping_gaussian_rewards.json")
    assert config["source_simulation"]["outcome_days"] == 28
    assert config["source_simulation"]["max_events_per_session"] == 782
    assert Path(config["toy"]["message_model_path"]).is_file()
    assert "search_model_path" not in config["source_simulation"]
    assert "cs_model_path" not in config["source_simulation"]


@pytest.mark.parametrize("override_assets", [False, True])
def test_custom_config_resolves_only_asset_and_calibration_paths(tmp_path, monkeypatch, override_assets):
    from simulate.v4 import configuration

    config_path = tmp_path / "settings" / "custom.yaml"
    config_path.parent.mkdir()
    absolute_model = tmp_path / "absolute-model.json"
    config_path.write_text(f"""source_simulation:
  model_path: models/shopping.json
  return_model_path: {absolute_model}
  shopping_source: data/raw
  prepared_source_cache: data/prepared.parquet
  optional_model_path: null
  retention_mode: source
toy:
  message_model_path: message.json
  coupon_model_path: {absolute_model}
experiment:
  model_path: untouched.json
""")
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "assets" if override_assets else tmp_path
    config = load_config(config_path, assets if override_assets else None)
    assert load_config is configuration.load_config
    assert DEFAULT_CONFIG == configuration.DEFAULT_CONFIG
    assert config["source_simulation"] == {
        "model_path": str(assets / "models/shopping.json"),
        "return_model_path": str(absolute_model), "shopping_source": str(assets / "data/raw"),
        "prepared_source_cache": str(assets / "data/prepared.parquet"),
        "optional_model_path": None, "retention_mode": "source",
    }
    assert config["toy"] == {"message_model_path": str(config_path.parent / "message.json"),
                             "coupon_model_path": str(absolute_model)}
    assert config["experiment"]["model_path"] == "untouched.json"


def test_missing_trained_assets_never_start_implicit_fitting(tmp_path, monkeypatch):
    from simulate.v4.data import make_dataset
    monkeypatch.setattr(make_dataset, "ensure_source_models", lambda _: pytest.fail("unexpected training"))
    args = SimpleNamespace(v4_command="generate", config=DEFAULT_CONFIG, assets_root=tmp_path,
                           output=tmp_path / "output", fit_missing=False)
    with pytest.raises(FileNotFoundError, match="--assets-root"):
        run(args)
    assert not args.output.exists()


def test_existing_output_rejected_before_fitting(tmp_path, monkeypatch):
    from simulate.v4.data import make_dataset
    monkeypatch.setattr(make_dataset, "ensure_source_models", lambda _: pytest.fail("unexpected training"))
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("existing")
    args = SimpleNamespace(v4_command="generate", config=DEFAULT_CONFIG, assets_root=tmp_path,
                           output=output, fit_missing=True)
    with pytest.raises(FileExistsError):
        run(args)
    assert sentinel.read_text() == "existing"
