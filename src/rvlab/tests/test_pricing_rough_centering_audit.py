"""Independent scalar integration of every exact-centered control component."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_centering_audit import audit_conditional_centers
from rvlab.pricing import (DiscountCurve, DividendSchedule, RoughHestonParams,
    SpotMarket, approximate_fractional_kernel)
from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy, value_fourier_stopping_policy


@pytest.fixture(scope="module", params=[(False, .04), (True, -.02)])
def independent_audit(request):
    call, rate = request.param
    params = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
    market = SpotMarket(100., DiscountCurve.flat(rate, 1.), DividendSchedule((), 1.))
    kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003)
    numerical = RoughAmericanConfig(training_paths=64, valuation_paths=64,
        training_seed=910_000_041, valuation_seed=910_000_042, steps=8,
        exercise_steps=4, batch_size=32, stock_knots=(.85, .95, 1., 1.05, 1.15))
    original = train_fourier_stopping_policy(params, market, kernel, strike=100.,
        maturity=60/365, is_call=call, config=numerical, premium_cap=.7)
    proxy = build_rough_obstacle_proxy(original, stock_knots=(.75, .9, .96, 1., 1.04, 1.2))
    trace = value_fourier_stopping_policy(proxy, trace=True)["trace"]
    result = audit_conditional_centers(proxy, trace, state_indices=((0, 0), (1, 1), (2, 3)))
    return proxy, trace, result


def test_all_component_local_means_agree_with_independent_scalar_gaussian_integrals(independent_audit):
    _, _, result = independent_audit
    branches, components = result["tables"]["branches"], result["tables"]["components"]
    assert len(branches) == 3*6*3
    assert len(components) == 3*5
    assert set(components.component) == {"european", "premium", "intrinsic", "micro_boundary", "micro_poststop"}
    assert np.max(np.abs(branches.difference)) < 2e-10
    assert np.max(np.abs(components.conditional_increment_mean)) < 2e-10
    assert result["manifest"]["source_unchanged"]
    assert not result["manifest"]["tolerances_changed"]
    assert not result["manifest"]["american_pricing_ready"]


def test_conditional_audit_preserves_original_trace_and_separates_decision_control_knots(independent_audit):
    policy, trace, result = independent_audit
    assert policy.config.stock_knots != policy.continuations[0].knots
    assert not trace["stocks"].flags.writeable
    assert result["manifest"]["state_trace_seed"] != policy.config.training_seed
    stopped = result["tables"]["components"].query("component == 'micro_poststop' and not original_policy_stopped")
    assert np.all(stopped.conditional_increment_mean == 0.)


@pytest.mark.parametrize("indices", [[], [(0, 0), (0, 0)], [(0, 4)], [(-1, 1)], [(True, 1)]])
def test_conditional_audit_rejects_invalid_or_terminal_states(independent_audit, indices):
    policy, trace, _ = independent_audit
    with pytest.raises(ValueError):
        audit_conditional_centers(policy, trace, state_indices=indices)


def test_conditional_audit_rejects_mismatched_or_training_stream_trace(independent_audit):
    policy, trace, _ = independent_audit
    for changed in ({"policy_fingerprint":"incorrect"}, {"seed":policy.config.training_seed}):
        with pytest.raises(ValueError, match="independent"):
            audit_conditional_centers(policy, {**trace, **changed}, state_indices=((0, 0),))


def test_conditional_audit_combines_selected_calendar_bin_weights(independent_audit):
    from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    policy, trace, _ = independent_audit
    config = RoughDualConfig(components=("european", "premium", "micro_boundary"))
    fit = RoughDualWeights(np.array([0., policy.exercise_times[2], policy.maturity]),
        np.array([[1., .8, .2], [1., 1.1, .3]]), fourier_policy_fingerprint(policy),
        policy.config.training_seed, 910_000_040, policy.config.valuation_seed, config, {})
    audit = audit_conditional_centers(policy, trace, state_indices=((0, 2),), martingale_weights=fit)
    means = audit["tables"]["components"].set_index("component").conditional_increment_mean
    assert means.selected_martingale == pytest.approx(means.european+1.1*means.premium+.3*means.micro_boundary, abs=1e-25)
    assert audit["manifest"]["selected_martingale_checked"]
    assert np.max(np.abs(means)) < 2e-10


def test_predictable_calendar_features_preserve_exact_optional_stopping_on_a_finite_tree():
    """Enumerate every history, independently of Monte Carlo uncertainty.

    Each component increment is predictable times an independent symmetric
    innovation. All eight equally likely three-step histories are present.
    This catches a future-state multiplier or calendar-bin off-by-one error.
    """
    from itertools import product
    from scipy.special import expit
    from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights, _weighted_martingale
    innovations = np.array(list(product((-1., 1.), repeat=3)))
    state = np.column_stack((np.zeros(8), np.cumsum(innovations, axis=1)))
    euro = state
    premium = np.column_stack((np.zeros(8), np.cumsum((.3+state[:, :-1]**2)*innovations, axis=1)))
    boundary = np.column_stack((np.zeros(8), np.cumsum(expit(state[:, :-1])*innovations, axis=1)))
    stops = np.full(8, 3, dtype=int)
    for date in (1, 2):
        stops[(stops == 3) & (state[:, date] <= -1)] = date
    stopped_left = np.arange(3)[None, :] >= stops[:, None]
    poststop = np.column_stack((np.zeros(8), np.cumsum(stopped_left*innovations, axis=1)))
    payoff = np.maximum(-2*state, 0.)[:, :, None]
    trace = {"times":np.arange(4.), "stocks":(100+2*state)[:, :, None], "strike":100.,
        "discounts":np.ones(4), "discounted_payoffs":payoff,
        "continuation_proxy":payoff+.2-.4*state[:, :, None],
        "european_martingale":euro[:, :, None], "premium_martingale":premium[:, :, None],
        "micro_boundary_martingale":boundary[:, :, None], "micro_poststop_martingale":poststop[:, :, None],
        "stop_indices":stops[:, None]}
    config = RoughDualConfig(components=("european", "premium", "micro_boundary", "micro_poststop"),
        predictable_features=("moneyness", "exercise_margin", "policy_stopped"))
    coefficients = np.arange(32.).reshape(2, 16)/17-.8
    fit = RoughDualWeights(np.array([0., 1., 3.]), coefficients, "toy", 1, 2, 3, config, {})
    martingale = _weighted_martingale(trace, fit)[:, :, 0]
    np.testing.assert_allclose(martingale.mean(axis=0), 0., atol=3e-15, rtol=0.)
    # A second policy stops in the opposite region. Both are adapted and may
    # use the original policy's stopping-history-dependent martingale.
    opposite = np.full(8, 3, dtype=int)
    for date in (1, 2):
        opposite[(opposite == 3) & (state[:, date] >= 1)] = date
    a, b = martingale[np.arange(8), stops], martingale[np.arange(8), opposite]
    assert np.mean(a) == pytest.approx(0., abs=3e-15)
    assert np.mean(b) == pytest.approx(0., abs=3e-15)
    assert np.mean(b-a) == pytest.approx(0., abs=3e-15)
    # The analogous future-state multiplication has a detectable nonzero
    # mean: this check would not pass a right-state centering error.
    assert np.mean(np.sum(state[:, 1:]*innovations, axis=1)) == pytest.approx(3.)

