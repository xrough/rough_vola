"""Independent Bellman integration, units and approximation counterexamples."""
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.special import ndtr

from rvlab.experiments._pricing_rough_bellman_audit import adaptive_block_expectation, clipped_premium_breakpoints


def mixture():
    return {"new_factors": np.array([[[.01, .02], [.02, .03]], [[.01, .02], [.02, .03]]]),
        "probabilities": np.array([[.3, .7], [.3, .7]]),
        "log_stock_shifts": np.array([[[.01, .02], [-.01, -.02]]]*2),
        "orthogonal": np.array([[[.002, .003], [.004, .005]]]*2), "endpoint_probability": .5}


def exact_put(mean, variance, strike):
    sigma = np.sqrt(variance)
    d2 = (mean-np.log(strike))/sigma
    return strike*ndtr(-d2)-np.exp(mean+.5*variance)*ndtr(-d2-sigma)


def test_terminal_put_matches_independent_exact_mixture_payoff():
    law = mixture(); stocks = np.array([99., 101.])
    value = lambda s, u: np.maximum(100.-s, 0.)
    result = adaptive_block_expectation(law, stocks, value, stock_breakpoints=(100.,),
        envelope_constant=100., envelope_stock=0.)
    mean = np.log(stocks)[:, None, None]+law["log_stock_shifts"]
    reference = np.sum(exact_put(mean, law["orthogonal"], 100.).mean(axis=-1)*law["probabilities"], axis=-1)
    np.testing.assert_allclose(result["expectation"], reference, rtol=0., atol=2e-12)
    assert not result["discounting_applied"] and not result["changes_martingale_centering"]
    assert not result["policy_improvement_guaranteed"] and not result["quadrature_error_is_rigorous_bound"]


def test_snell_max_crossing_is_integrated_before_branch_average():
    law = mixture(); stocks = np.array([99., 101.]); strike = 100.
    continuation = lambda s, u: .5+.01*s+u[:, 0]
    intrinsic = lambda s: np.maximum(strike-s, 0.)
    value = lambda s, u: np.maximum(intrinsic(s), continuation(s, u))
    kink = lambda s, u: intrinsic(s)-continuation(s, u)
    result = adaptive_block_expectation(law, stocks, value, kink_function=kink,
        stock_breakpoints=(strike,), envelope_constant=100.6, envelope_stock=.01)
    reference = np.zeros(2)
    for path in range(2):
        for b in range(2):
            intercept = .5+law["new_factors"][path, b, 0]
            crossing = (strike-intercept)/1.01
            for o in range(2):
                mean = np.log(stocks[path])+law["log_stock_shifts"][path, b, o]
                variance = law["orthogonal"][path, b, o]; sigma = np.sqrt(variance)
                z = (np.log(crossing)-mean)/sigma
                # Continuation plus a put-like exercise correction at crossing.
                correction = (strike-intercept)*ndtr(z)-1.01*np.exp(mean+.5*variance)*ndtr(z-sigma)
                expected = intercept+.01*np.exp(mean+.5*variance)+correction
                reference[path] += law["probabilities"][path, b]*.5*expected
    np.testing.assert_allclose(result["expectation"], reference, rtol=0., atol=2e-12)
    assert np.all(result["detected_snell_crossings"] >= 4)


def test_constant_endpoint_and_calendar_discount_units_are_explicit():
    law = mixture()
    result = adaptive_block_expectation(law, [100., 100.], lambda s, u: np.full_like(s, 7.))
    np.testing.assert_allclose(result["expectation"], 7., atol=2e-13, rtol=0.)
    # A left-date continuation uses D(right)/D(left), not D(right) alone.
    left_discount, right_discount = .97, .94
    np.testing.assert_allclose(result["expectation"]*right_discount/left_discount, 7.*.94/.97)
    assert result["tail_value_bound"] is None
    assert not result["tail_envelope_declared_by_caller"]


def test_deterministic_clocks_are_exact_without_normal_quadrature():
    law = mixture(); law["orthogonal"][:] = 0.
    stocks = np.array([100., 102.])
    result = adaptive_block_expectation(law, stocks, lambda s, u: s+u[:, 0])
    reference = ((stocks[:, None, None]*np.exp(law["log_stock_shifts"])+law["new_factors"][:, :, :1])
        .mean(axis=-1)*law["probabilities"]).sum(axis=-1)
    np.testing.assert_allclose(result["expectation"], reference, rtol=0., atol=2e-13)
    np.testing.assert_array_equal(result["function_evaluations"], 4)
    np.testing.assert_array_equal(result["quadrature_error_estimate"], 0.)


def test_tail_probability_does_not_bound_unbounded_stock_value():
    law = {"new_factors": np.zeros((1, 1, 1)), "probabilities": np.ones((1, 1)),
        "log_stock_shifts": np.zeros((1, 1, 1)), "orthogonal": np.full((1, 1, 1), 4.), "endpoint_probability": 1.}
    result = adaptive_block_expectation(law, [100.], lambda s, u: s, normal_limit=8.,
        envelope_constant=0., envelope_stock=1.)
    exact = 100.*np.exp(2.)
    omitted = exact-result["expectation"][0]
    assert omitted > 1e6*result["normal_tail_probability"]
    assert omitted <= result["tail_value_bound"][0]+2e-12
    assert not result["gaussian_tail_renormalized"]


def test_identical_left_laws_and_state_functions_ignore_external_future_outcomes():
    law = mixture(); stocks = np.array([100., 100.])
    # The two records may have opposite realized future paths, but these
    # outcomes are deliberately absent from the state-function API.
    external_future_rewards = np.array([1., 20.])
    value = lambda s, u: np.maximum(100.-s, .3+u[:, 0])
    result = adaptive_block_expectation(law, stocks, value, stock_breakpoints=(100.,))
    assert external_future_rewards[0] != external_future_rewards[1]
    np.testing.assert_array_equal(result["expectation"], np.repeat(result["expectation"][0], 2))


def test_approximate_snell_lookahead_has_no_policy_improvement_guarantee():
    # A deterministic finite stopping problem with early reward and zero
    # terminal payoff has exact European value0. Its frozen q is bounded by
    # the same .7 cap. The approximate Snell function is formed exactly as
    # max(g,European+q), but q is not the exact value of the frozen strategy.
    rewards = np.array([0., .05, 0., .1, 0.])
    european = np.zeros(5)
    premium = np.array([0., .06, 0., 0., 0.])
    assert np.all((premium >= 0) & (premium <= .7))
    proxy = np.maximum(rewards, european+premium)
    baseline_stop = next(j for j in range(1, 4) if rewards[j] > 0 and rewards[j] >= european[j]+premium[j])
    new_stop = next(j for j in range(1, 4) if rewards[j] > 0 and rewards[j] >= proxy[j+1])
    assert np.all(proxy >= rewards) and proxy[-1] == rewards[-1]
    assert rewards[new_stop] == .05 < rewards[baseline_stop] == .1
    # Neither exact transition enumeration nor exact quadrature repairs this
    # value-function approximation error. Fresh paired valuation is necessary.


def test_premium_breakpoints_include_strike_knots_floor_and_cap_crossings():
    class Quadratic:
        knots = (.75, 1.25)
        def coefficients(self, factors): return np.tile([1., -2., 1.], (len(factors), 3, 1))
    points = clipped_premium_breakpoints(Quadratic(), np.zeros(2), 100., .25)
    np.testing.assert_allclose(points, [50., 75., 100., 125., 150.], atol=1e-12)


def test_premium_breakpoint_roots_are_restricted_to_their_stock_interval():
    class Linear:
        knots = (1.,)
        def coefficients(self, factors): return np.tile([[0., 1., 0.], [-2., 1., 0.]], (len(factors), 1, 1))
    points = clipped_premium_breakpoints(Linear(), np.zeros(2), 100., .5)
    np.testing.assert_allclose(points, [50., 100., 200., 250.], atol=1e-12)


@pytest.mark.parametrize("tamper", ["probability", "endpoint_probability", "clock", "shape", "nonfinite"])
def test_transition_law_tampering_is_rejected(tamper):
    law = mixture()
    if tamper == "probability": law["probabilities"][0, 0] = .4
    if tamper == "endpoint_probability": law["endpoint_probability"] = .4
    if tamper == "clock": law["orthogonal"][0, 0, 0] = -.1
    if tamper == "shape": law["new_factors"] = law["new_factors"][:1]
    if tamper == "nonfinite": law["log_stock_shifts"][0, 0, 0] = np.nan
    with pytest.raises(ValueError): adaptive_block_expectation(law, [100., 100.], lambda s, u: s)


def test_function_breakpoints_and_envelope_inputs_are_validated():
    law = mixture()
    with pytest.raises(ValueError, match="Endpoint audit"):
        adaptive_block_expectation(law, [100., 100.], lambda s, u: np.array([np.nan]))
    with pytest.raises(ValueError, match="breakpoints"):
        adaptive_block_expectation(law, [100., 100.], lambda s, u: s, stock_breakpoints=[-1.])
    with pytest.raises(ValueError, match="both terms"):
        adaptive_block_expectation(law, [100., 100.], lambda s, u: s, envelope_stock=1.)
    with pytest.raises(ValueError, match="nonnegative"):
        adaptive_block_expectation(law, [100., 100.], lambda s, u: s, envelope_constant=-1., envelope_stock=0.)


@dataclass(frozen=True)
class AuditConstantPremium:
    amount: float
    knots: tuple = ()
    def value(self, stocks, factors, strike): return np.full_like(stocks, self.amount)
    def coefficients(self, factors):
        coefficients = np.zeros(factors.shape[:-1]+(3,)); coefficients[..., 0] = self.amount
        return coefficients


@pytest.fixture(scope="module")
def small_bellman_evaluator():
    # No training or simulation: fully specified constant premium functions.
    from rvlab.experiments._pricing_rough_bellman_policy import BellmanEvaluator
    from rvlab.pricing import DiscountCurve, DividendSchedule, ExponentialKernel, RoughHestonParams, SpotMarket
    from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy
    params = RoughHestonParams(.04, .04, 1.5, .3, -.6, .15)
    market = SpotMarket(100., DiscountCurve.flat(1., 1.), DividendSchedule((), 1.))
    kernel = ExponentialKernel((.5, 2.), (.4, .6), .15, 1.)
    times = np.array([0., .1, .2]); models = tuple(AuditConstantPremium(0.) for _ in times)
    config = RoughAmericanConfig(training_paths=8, valuation_paths=4, training_seed=10,
        valuation_seed=11, steps=4, exercise_steps=2, batch_size=2)
    proxy = FourierStoppingPolicy(params, market, kernel, 100., .2, False, "american", times, times,
        models, models, config, RoughEuropeanControlConfig(), False, {}, .7)
    selected = ControlledExercisePolicy(proxy, times, models, False, (10, 12), 0., .7, {})
    return BellmanEvaluator(proxy, selected, batch_size=2)


def test_actual_bellman_endpoint_caps_q_only_not_exercise_envelope(small_bellman_evaluator):
    evaluator = small_bellman_evaluator
    result = evaluator.endpoint_components(1, np.array([10.]), np.zeros((1, 2)))
    np.testing.assert_array_equal(result["envelope"], [90.])
    assert result["payoff"][0]-result["european"][0] > evaluator.selected.premium_cap
    assert result["envelope"][0] > result["european"][0]+evaluator.selected.premium_cap


def test_actual_bellman_terminal_step_matches_adaptive_mixture_without_future_states(small_bellman_evaluator):
    from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
    evaluator = small_bellman_evaluator; stocks = np.array([99., 99.]); factors = np.zeros((2, 2))
    result = evaluator.conditional_expectations(1, stocks, factors)
    a, b = np.searchsorted(evaluator.grid, evaluator.times[1:3])
    law = exercise_block_law(evaluator.proxy, factors, times=evaluator.grid[a:b+1])
    reference = adaptive_block_expectation(law, stocks, lambda s, u: np.maximum(100.-s, 0.),
        stock_breakpoints=(100.,), envelope_constant=100., envelope_stock=0.)
    expected = reference["expectation"]*evaluator.discounts[2]/evaluator.discounts[1]
    np.testing.assert_allclose(result["continuation"], np.broadcast_to(expected[:, None], (2, 3)), atol=2e-12, rtol=0.)
    np.testing.assert_array_equal(result["continuation"], result["direct_continuation"])
    np.testing.assert_array_equal(result["continuation"][0], result["continuation"][1])
    assert result["diagnostics"]["physical_increments"] == 2


def test_adaptive_reference_exposes_coarse_quadrature_failure_without_changing_tolerance(small_bellman_evaluator):
    from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
    evaluator = small_bellman_evaluator; stocks = np.array([99.]); factors = np.zeros((1, 2))
    result = evaluator.conditional_expectations(0, stocks, factors)
    a, b = np.searchsorted(evaluator.grid, evaluator.times[:2])
    law = exercise_block_law(evaluator.proxy, factors, times=evaluator.grid[a:b+1])
    reference = adaptive_block_expectation(law, stocks,
        lambda s, u: evaluator.endpoint_values(1, s, u), stock_breakpoints=(100.,),
        kink_function=lambda s, u: evaluator.endpoint_components(1, s, u)["kink"])
    expected = reference["expectation"][0]*evaluator.discounts[1]/evaluator.discounts[0]
    # This deliberately coarse high-rate/two-date case fails GH32. Preserve
    # that counterexample: increasing the quadrature budget, rather than the
    # acceptance tolerance, is necessary. GH refinement need not be monotone.
    allowance = .001+1e-4*abs(expected)
    assert abs(result["continuation"][0, -1]-expected) > allowance
    refined = type(evaluator)(evaluator.proxy, evaluator.selected, orders=(128,), batch_size=2)
    fine = refined.conditional_expectations(0, stocks, factors)
    assert abs(fine["continuation"][0, 0]-expected) <= allowance
