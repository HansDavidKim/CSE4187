import pytest


from analysis.uplift import canonical as module


def test_bounds_against_feasible_joint_endpoints():
    result = module.bounds(0.4, 0.6)
    assert result["sure_thing"] == pytest.approx([0, 0.4])
    assert result["persuadable"] == pytest.approx([0.2, 0.6])
    assert result["sleeping_dog"] == pytest.approx([0, 0.4])
    assert result["lost_cause"] == pytest.approx([0, 0.4])
    for index in [0, 1]:
        # Endpoints cannot all be combined: latent joint has one free parameter.
        q11 = result["sure_thing"][index]
        assert q11 + (0.6 - q11) + (0.4 - q11) + (1 - 0.4 - 0.6 + q11) == pytest.approx(1)


def test_rejects_invalid_marginals():
    with pytest.raises(ValueError):
        module.bounds(-0.1, 0.5)
