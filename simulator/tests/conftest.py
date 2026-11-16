from pathlib import Path

import pytest


@pytest.fixture
def legacy_session_artifacts():
    root = Path(__file__).parents[1] / "artifacts"
    paths = (
        "cosmetics-shop-persona-maxent-irl/coefficient_gmm.json",
        "cosmetics-shop-persona-quadratic-irl/lifecycle_irl_model.json",
        "cosmetics-shop-persona-quadratic-irl/gap_token_irl_model.json",
        "cross-domain-lifecycle/lifecycle_model.json",
        "tianchi-uplift-model/model.json",
        "domain-simulator-profiles/model.json",
    )
    missing = [str(root / path) for path in paths if not (root / path).is_file()]
    if missing:
        pytest.skip("requires local trained legacy artifacts: " + ", ".join(missing))
