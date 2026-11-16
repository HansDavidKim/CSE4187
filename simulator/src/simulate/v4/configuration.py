"""Resolve simulator configuration and its source/model asset paths."""
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config") / "simulator.yaml"


def load_config(path=DEFAULT_CONFIG, assets_root=None):
    """Resolve source/model paths independently from bundled calibration files."""
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("source_simulation"), dict):
        raise ValueError("v4 config requires a source_simulation mapping")
    root = Path(assets_root or Path.cwd()).expanduser().resolve()
    for key, value in config["source_simulation"].items():
        if value and (key.endswith(("_path", "_source")) or key == "prepared_source_cache"):
            configured = Path(value).expanduser()
            config["source_simulation"][key] = str(configured if configured.is_absolute() else root / configured)
    for key, value in config.get("toy", {}).items():
        if key.endswith("_path"):
            configured = Path(value).expanduser()
            config["toy"][key] = str(configured if configured.is_absolute() else path.parent / configured)
    return config
