"""Tampering with decisions, Greek endpoints or centering must remain visible."""
from types import SimpleNamespace
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_bellman_delta_samples import RELATIVE, validate_delta_samples


@pytest.fixture
def saved_delta():
    times = np.linspace(0., 60/365, 65)
    curve = SimpleNamespace(times=np.array([0., 1.]), discount=lambda t: np.exp(-.04*np.asarray(t)))
    market = SimpleNamespace(spot=100., discount_curve=curve, dividends=SimpleNamespace(events=()))
    proxy = SimpleNamespace(market=market, strike=100., is_call=False, config=SimpleNamespace(steps=128),
        params=object(), exercise_times=times)
    model = SimpleNamespace(value=lambda s, u, k: np.ones(s.shape))
    selected = SimpleNamespace(continuations=[model]*65, premium_cap=1.)
    components = ("european", "premium", "micro_boundary")
    fit = SimpleNamespace(config=SimpleNamespace(predictable_features=(), components=components),
        weights=np.ones((4, 3)), time_edges=times[[0, 16, 32, 48, 64]])
    bundle = {"proxy": proxy, "selected_policy": selected, "weights": fit}
    stocks = np.broadcast_to(100*(1+RELATIVE), (2, 65, 5)).copy()
    stocks[0, 1:] *= .95
    stocks[1, 1:] *= 1.04
    payoff = np.maximum(100-stocks, 0.)*curve.discount(times)[None, :, None]
    a = {"relative_spots": RELATIVE, "times": times, "discounts": curve.discount(times),
        "eligible": np.ones(65, bool), "initial_spots": 100*(1+RELATIVE),
        "actual_time_grid": np.linspace(0., 60/365, 129), "operator_time_grid": np.linspace(0., 60/365, 129),
        "strike": np.array(100.), "is_call": np.array(False), "stable_weights": fit.weights,
        "stable_time_edges": fit.time_edges, "stable_components": np.array(components),
        "stocks": stocks, "factors": np.zeros((2, 65, 4)), "payoff": payoff,
        "baseline_continuation": np.ones((2, 65, 5))}
    for field in ("european_proxy", "baseline_european_proxy", "european_martingale", "premium_martingale",
            "micro_boundary_martingale", "micro_poststop_martingale", "intrinsic_martingale", "stable_martingale"):
        a[field] = np.zeros((2, 65, 5))
    mask = np.zeros((2, 65, 5), bool)
    mask[0, 1] = True
    c = np.full((2, 65, 5), np.nan)
    c[mask] = 1.
    a.update(bellman_computed_mask=mask, bellman_continuation=c)
    stops = np.array([[1]*5, [64]*5])
    prices = payoff[np.arange(2)[:, None], stops, np.arange(5)[None, :]]
    deltas = np.column_stack((prices[:, 4]-prices[:, 0], (prices[:, 3]-prices[:, 1])/.5))
    for policy in ("baseline", "bellman"):
        a[policy+"_stops"] = stops.copy()
        for estimator in ("raw", "controlled"):
            a[policy+"_"+estimator] = prices.copy()
            a[policy+"_"+estimator+"_deltas"] = deltas.copy()
    return a, bundle


def test_complete_paths_and_declared_bumps_are_reconstructed(saved_delta):
    a, bundle = saved_delta
    result = validate_delta_samples(a, bundle, paths=2, steps=128, seed=1)
    assert result["bellman"]["raw"].shape == (2, 5)
    assert np.array_equal(result["bellman"]["stops"], a["bellman_stops"])


@pytest.mark.parametrize("tamper", ["endpoint", "delta", "centering", "mask", "unused", "decision", "operator", "premium"])
def test_inconsistent_saved_paths_are_rejected(saved_delta, tamper):
    a, bundle = saved_delta
    if tamper == "endpoint": a["bellman_controlled"][0, 0] += .01
    if tamper == "delta": a["bellman_raw_deltas"][0, 0] += .01
    if tamper == "centering": a["stable_martingale"][0, 1, 0] = .01
    if tamper == "mask": a["bellman_computed_mask"][0, 1, 0] = False
    if tamper == "unused": a["bellman_continuation"][1, 5, 0] = 0.
    if tamper == "decision": a["bellman_stops"][0, 0] = 64
    if tamper == "operator": a["operator_time_grid"] = np.linspace(0., 60/365, 257)
    if tamper == "premium": a["baseline_continuation"][0, 1, 0] += .01
    with pytest.raises(ValueError):
        validate_delta_samples(a, bundle, paths=2, steps=128, seed=1)
