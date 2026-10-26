"""Reconstruct every new martingale endpoint before accepting saved tables."""
import numpy as np
import pytest

from rvlab.experiments import _pricing_rough_policy_validation_stopping as validation


@pytest.fixture
def workspace():
    rng = np.random.default_rng(42)
    n, t = 4, 65
    payoff = np.maximum(3. + rng.normal(size=(n, t)), 0.)
    baseline = np.cumsum(rng.normal(0, .01, size=(n, t)), axis=1)
    baseline -= baseline[:, :1]
    correction = np.cumsum(rng.normal(0, .02, size=(n, t, 3)), axis=1)
    correction -= correction[:, :1]
    stops = np.array([12, 23, 43, 64])
    raw = payoff[np.arange(n), stops]
    controls = {"stable baseline": {"bins": 1, "weights": [0., 0., 0.]},
        "local": {"bins": 4, "weights": [.1, 0., 0., .2, 0., 0., .3, 0., 0., .4, 0., 0.]}}
    arrays = {"raw": raw, "stops": stops, "payoff": payoff,
        "baseline_martingale": baseline, "local_corrections": correction}
    # Direct interval summation is independent of the collector's basis tensor.
    martingale = baseline.copy()
    for coefficient, left, right in zip((.1, .2, .3, .4), (0, 16, 32, 48), (16, 32, 48, 64)):
        for j in range(left+1, t):
            martingale[:, j] += coefficient*(correction[:, min(j, right), 0]-correction[:, left, 0])
    for i, m in enumerate((baseline, martingale)):
        arrays[f"control{i}_lower"] = raw - m[np.arange(n), stops]
        arrays[f"control{i}_upper"] = np.max(payoff-m, axis=1)
    return arrays, controls


def test_endpoint_reconstruction_retains_same_exercise_rule_and_pathwise_ordering(workspace):
    arrays, controls = workspace
    result = validation._check_samples(arrays, controls, 4)
    for i, name in enumerate(controls):
        lower, upper = result[name]
        np.testing.assert_allclose(lower, arrays[f"control{i}_lower"], rtol=0., atol=2e-14)
        np.testing.assert_allclose(upper, arrays[f"control{i}_upper"], rtol=0., atol=2e-14)
        assert np.all(upper >= lower)


@pytest.mark.parametrize("change", ["missing", "nonfinite", "lower", "upper", "stops", "origin", "raw", "payoff", "coefficient", "fallback"])
def test_changed_endpoint_inputs_or_coefficient_constraints_are_rejected(workspace, change):
    arrays, controls = workspace
    if change == "missing": arrays.pop("control0_lower")
    elif change == "nonfinite": arrays["local_corrections"][0, 2, 0] = np.inf
    elif change == "lower": arrays["control1_lower"][0] += 1e-5
    elif change == "upper": arrays["control1_upper"][0] -= 1e-5
    elif change == "stops": arrays["stops"][0] = 65
    elif change == "origin": arrays["baseline_martingale"][0, 0] = 1e-8
    elif change == "raw": arrays["raw"][0] += 1e-8
    elif change == "payoff": arrays["payoff"][0, 0] = -1e-8
    elif change == "coefficient": controls["local"]["weights"][:3] = [.5, .5, .5]
    else: controls["stable baseline"]["weights"][0] = .001
    with pytest.raises(ValueError):
        validation._check_samples(arrays, controls, 4)


def test_stratified_uncertainty_preserves_independent_streams():
    a, b = np.array([1., 2., 3.]), np.array([4., 6., 8.])
    mean, se, low, high = validation._summary([a, b])
    assert mean == pytest.approx(4.)
    assert se == pytest.approx(np.sqrt(3*1+3*4)/6)
    assert low == pytest.approx(mean-1.96*se)
    assert high == pytest.approx(mean+1.96*se)
