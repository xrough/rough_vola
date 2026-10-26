"""Independent scalar integration checks for exact stopping martingales."""
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from rvlab.pricing.rough_stopping_controls import (
    _decimal_polynomial_integral, _polynomial_integral,
    expected_stopping_value, stopping_value,
)
from rvlab.pricing.rough_simulation import RoughSimulationError


def _reference(mean, variance, coefficients, strike, call, cash=0., side="pre", knots=(), exercise=True, value_cap=None):
    """Independent scalar lognormal density integration, with all kink roots."""
    coefficients = np.asarray(coefficients, float)
    if not len(knots):
        coefficients = coefficients[None, :]
    # Split at cash, exercise, spline knots, and independently computed roots
    # of continuation minus the relevant linear intrinsic on each segment.
    boundaries = sorted(set([0., cash, strike if side == "pre" or (side == "both" and call)
        else cash+strike]+[cash+strike*k for k in knots]))
    roots = []
    for lo, hi in zip(boundaries, boundaries[1:]+[math.inf]):
        point = .5*(lo+hi) if math.isfinite(hi) else lo+strike
        segment = int(np.searchsorted(knots, max(point-cash, 0.)/strike, side="right"))
        a, b, c = coefficients[segment]
        p = np.poly1d([c/strike**2, b/strike, a])
        p = p(np.poly1d([1., -cash])) if point > cash else np.poly1d([a])
        if value_cap is not None:
            cap_poly = np.poly1d([1., 0.]) if isinstance(value_cap, str) else np.poly1d([value_cap])
            for root in np.roots(p-cap_poly):
                if abs(root.imag) < 1e-8 and lo < root.real < hi:
                    roots.append(float(root.real))
        pre = side == "pre" or (side == "both" and call)
        intrinsic_stock = point if pre else max(point-cash, 0.)
        sign = 1 if call else -1
        if exercise and sign*(intrinsic_stock-strike) > 0:
            payoff = np.poly1d([sign, -sign*(strike+(0 if pre else cash))])
            if not pre and point < cash:
                payoff = np.poly1d([strike if not call else 0.])
            p = p-payoff
        for root in np.roots(p):
            if abs(root.imag) < 1e-8 and lo < root.real < hi:
                roots.append(float(root.real))
    points = sorted(set((math.log(v)-mean)/math.sqrt(variance)
        for v in boundaries+roots if v > 0))
    points = [-12.]+[v for v in points if -12 < v < 12]+[12.]

    def evaluate(z):
        stock = math.exp(mean+math.sqrt(variance)*z)
        post = max(stock-cash, 0.)
        segment = int(np.searchsorted(knots, post/strike, side="right"))
        a, b, c = coefficients[segment]
        continuation = a+b*post/strike+c*(post/strike)**2
        pre = side == "pre" or (side == "both" and call)
        payoff = max((1 if call else -1)*((stock if pre else post)-strike), 0.) if exercise else 0.
        value = max(payoff, continuation)
        if value_cap is not None:
            value = min(value, stock if isinstance(value_cap, str) else value_cap)
        return value*math.exp(-z*z/2)/math.sqrt(2*math.pi)

    return sum(quad(evaluate, a, b, epsabs=1e-10, epsrel=1e-11, limit=100)[0]
        for a, b in zip(points[:-1], points[1:]))


@pytest.mark.parametrize("call", [False, True])
def test_payoff_exact_lognormal_control(call):
    mean, variance, strike = math.log(103)-.5*.09, .09, 100.
    forward = math.exp(mean+.5*variance)
    d2, d1 = (mean-math.log(strike))/math.sqrt(variance), (mean-math.log(strike)+variance)/math.sqrt(variance)
    expected = forward*ndtr(d1)-strike*ndtr(d2) if call else strike*ndtr(-d2)-forward*ndtr(-d1)
    assert expected_stopping_value(mean, variance, [0., 0., 0.],
        strike=strike, is_call=call) == pytest.approx(expected, abs=3e-13)


@pytest.mark.parametrize("side", ["pre", "post", "both"])
@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("cash", [0., 17., 130.])
def test_quadratic_and_cash_against_independent_integral(side, call, cash):
    mean, variance = math.log(94), .16
    coefficients = np.array([38., -53., 21.])
    actual = expected_stopping_value(mean, variance, coefficients, strike=100.,
        is_call=call, cash_dividend=cash, payoff_side=side)
    expected = _reference(mean, variance, coefficients, 100., call, cash, side)
    assert actual == pytest.approx(expected, abs=4e-11)


@pytest.mark.parametrize("coefficients", [
    [10., 0., 0.], [-1., 2., 0.], [2., -1., 0.],
    [4., -4., 1.], [-4., 4., -1.], [2., -3., 1.],
    [-2., 3., -1.], [10., -2., 1e-12], [1., -2e-7, 1e-14],
])
def test_constant_linear_repeated_and_near_linear_roots(coefficients):
    actual = expected_stopping_value(math.log(100), .04, coefficients,
        strike=100., is_call=True)
    expected = _reference(math.log(100), .04, coefficients, 100., True)
    assert actual == pytest.approx(expected, abs=4e-11)


def test_piecewise_spline_against_independent_integral():
    knots = (.8, 1., 1.2)
    coefficients = np.array([[20., -15., 5.], [52., -95., 55.],
        [-8., 25., -5.], [64., -95., 45.]])
    for side in ("pre", "post", "both"):
        for call in (False, True):
            value = expected_stopping_value(math.log(102), .09, coefficients,
                strike=100., is_call=call, cash_dividend=12., payoff_side=side,
                stock_knots=knots)
            ref = _reference(math.log(102), .09, coefficients, 100., call, 12., side, knots)
            assert value == pytest.approx(ref, abs=7e-11)


def test_broadcasting_mixed_deterministic_and_stochastic():
    means = np.log([[80.], [100.], [125.]])
    variance = np.array([0., .04])
    coeff = np.array([8., -9., 3.])
    value = expected_stopping_value(means, variance, coeff, strike=100., is_call=False,
        cash_dividend=2.)
    assert value.shape == (3, 2)
    np.testing.assert_allclose(value[:, 0], stopping_value(np.exp(means[:, 0]), coeff,
        strike=100., is_call=False, cash_dividend=2.), atol=2e-13)
    for i in range(3):
        assert value[i, 1] == pytest.approx(_reference(means[i, 0], .04, coeff,
            100., False, 2.), abs=4e-11)


def test_piecewise_deterministic_cash_atom_selects_correct_segment():
    coeff = np.array([[20., 1., 0.], [50., -2., 0.]])
    spots = np.array([0., 10., 110., 140.])
    values = stopping_value(spots, coeff, strike=100., is_call=False,
        cash_dividend=10., payoff_side="both", stock_knots=(1.,))
    actual = expected_stopping_value(np.log(np.maximum(spots, np.finfo(float).tiny)), 0.,
        coeff, strike=100., is_call=False, cash_dividend=10., payoff_side="both",
        stock_knots=(1.,))
    np.testing.assert_allclose(actual, values, atol=1e-12)


def test_absorbing_zero_stock_atom_after_cash_cap():
    coefficients = np.array([[20., 1., 0.], [50., -2., 0.]])
    actual = expected_stopping_value(np.array([-np.inf, -np.inf]), np.array([0., .2]),
        coefficients, strike=100., is_call=False, cash_dividend=5.,
        payoff_side="both", stock_knots=(1.,))
    np.testing.assert_array_equal(actual, [100., 100.])


@pytest.mark.parametrize("cash", [0., 20., 130.])
def test_nonexercise_dates_integrate_positive_continuation_only(cash):
    knots = (.8, 1., 1.2)
    coefficients = np.array([[20., -15., 5.], [-22., 45., -19.],
        [-8., 25., -5.], [64., -95., 45.]])
    actual = expected_stopping_value(math.log(105), .09, coefficients,
        strike=100., is_call=False, cash_dividend=cash, stock_knots=knots,
        exercise=False)
    expected = _reference(math.log(105), .09, coefficients, 100., False,
        cash, "pre", knots, exercise=False)
    assert actual == pytest.approx(expected, abs=7e-11)
    # Exercise eligibility affects the intrinsic lower bound, not the target
    # Gaussian law or a caller's right to declare a nonexercise microstep.
    assert stopping_value(0., [0., -1., -1.], strike=100., is_call=False,
        exercise=False) == 0.
    assert expected_stopping_value(-np.inf, .2, [-5., 0., 0.],
        strike=100., is_call=False, exercise=False) == 0.


@pytest.mark.parametrize("cash", [0., 20., 130.])
@pytest.mark.parametrize("exercise", [False, True])
@pytest.mark.parametrize("call", [False, True])
def test_bounded_splines_against_independent_quad(cash, exercise, call):
    knots = (.8, 1., 1.2)
    coefficients = np.array([[120., -150., 55.], [-22., 45., 79.],
        [-8., 125., -5.], [64., -95., 145.]])
    # The put cap includes the larger discount factor in a negative-rate
    # horizon; the local expectation itself remains undiscounted.
    cap = "stock" if call else 100.*math.exp(.1*.5)
    for side in ("pre", "post", "both"):
        actual = expected_stopping_value(math.log(105), .09, coefficients,
            strike=100., is_call=call, cash_dividend=cash, stock_knots=knots,
            exercise=exercise, payoff_side=side, value_cap=cap)
        reference = _reference(math.log(105), .09, coefficients, 100., call,
            cash, side, knots, exercise=exercise, value_cap=cap)
        assert actual == pytest.approx(reference, abs=1e-10)


@pytest.mark.parametrize("level", [1e12, 1e16, 1e30])
def test_large_extrapolation_is_integrated_directly_without_cap_cancellation(level):
    put = expected_stopping_value(math.log(100), .04, [level, 0., 0.],
        strike=100., is_call=False, value_cap=110.)
    call = expected_stopping_value(math.log(100), .04, [level, 0., 0.],
        strike=100., is_call=True, value_cap="stock")
    assert put == pytest.approx(110., abs=3e-12)
    assert call == pytest.approx(100*math.exp(.02), abs=3e-12)


def test_repeated_root_at_interval_midpoint_is_not_a_constant_zero_region():
    scale, cap = 1e4, 100.
    width = math.sqrt(cap/scale)
    # Integrate the small capped deficit after rescaling z=1+width*y;
    # this avoids the very moment cancellation being tested.
    deficit = cap*width*quad(lambda y: (1-y*y)
        *math.exp(-math.log1p(width*y)**2/(2*.04))
        /((1+width*y)*math.sqrt(2*math.pi*.04)), -1., 1.,
        epsabs=1e-12, epsrel=1e-12)[0]
    actual = expected_stopping_value(math.log(100), .04,
        [scale, -2*scale, scale], strike=100., is_call=False,
        exercise=False, value_cap=cap)
    assert actual == pytest.approx(cap-deficit, abs=1e-10)


@pytest.mark.parametrize("scale", [1e8, 1e12, 1e16])
def test_high_precision_fallback_recovers_ill_conditioned_cap_bands(scale):
    width = math.sqrt(100/scale)
    deficit = 100*width*quad(lambda y: (1-y*y)
        *math.exp(-math.log1p(width*y)**2/(2*.04))
        /((1+width*y)*math.sqrt(2*math.pi*.04)), -1., 1.,
        epsabs=1e-12, epsrel=1e-12)[0]
    actual = expected_stopping_value(math.log(100), .04,
        [scale, -2*scale, scale], strike=100., is_call=False,
        exercise=False, value_cap=100.)
    assert actual == pytest.approx(100-deficit, abs=1e-10)


def test_truly_unresolved_cap_band_still_raises():
    with pytest.raises(RoughSimulationError, match="precision"):
        expected_stopping_value(math.log(100), .04,
            [1e20, -2e20, 1e20], strike=100., is_call=False,
            exercise=False, value_cap=100.)


def test_captured_cancellation_rows_recover_independent_density_integrals():
    # The fixture contains 19 real guard-triggering rows, with reference
    # Gaussian-density integrals computed independently using a Decimal
    # polynomial evaluation. It is independent of notebook output files.
    fixture = Path(__file__).with_name("fixtures")/"rough_stopping_cancellation.npz"
    with np.load(fixture) as data:
        arguments = [data[name] for name in ("coefficients", "mean", "variance", "lower", "upper")]
        expected = data["independent_density_integral"]
    actual = _polynomial_integral(*arguments)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)
    for row in zip(*arguments):
        sixty, bound60, precision60 = _decimal_polynomial_integral(*row, minimum_precision=60)
        eighty, bound80, precision80 = _decimal_polynomial_integral(*row, minimum_precision=80)
        assert precision60 >= 60 and precision80 >= 80
        assert abs(sixty-eighty) <= bound60+bound80
        assert bound60 <= 1e-9*(1+abs(sixty))


def test_thin_cap_bands_share_a_stable_width_across_moment_powers():
    # A real residual-policy probe generated cap crossings only ~1e-11
    # apart with coefficients ~1e12. Standardizing both endpoints and then
    # subtracting separately for each power formerly produced -4e-5 for a
    # positive integral of 1.57e-10. References integrate the Gaussian
    # density on a rescaled [0,1] interval using Decimal polynomial values.
    fixture = Path(__file__).with_name("fixtures")/"rough_stopping_thin_cap_bands.npz"
    with np.load(fixture) as data:
        arguments = [data[name] for name in ("coefficients", "mean", "variance", "lower", "upper")]
        expected = data["independent_density_integral"]
    actual = _polynomial_integral(*arguments)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)
    for row, reference in zip(zip(*arguments), expected):
        decimal, error, _ = _decimal_polynomial_integral(*row)
        assert abs(decimal-reference) <= error+1e-18


def test_decimal_tail_certificate_and_explicit_precision_budget():
    value, error, _ = _decimal_polynomial_integral([1., 0., 0.], 0., 1., math.exp(12), np.inf)
    assert abs(value-ndtr(-12.)) <= error
    with pytest.raises(RoughSimulationError, match="100 decimal digits"):
        _decimal_polynomial_integral([1e100, 0., 0.], 0., 1., 0., np.inf)


def test_uncapped_repeated_positive_quadratic_integrates_full_polynomial():
    variance = .04
    actual = expected_stopping_value(math.log(100), variance,
        [1., -2., 1.], strike=100., is_call=False, exercise=False)
    expected = 1-2*math.exp(.5*variance)+math.exp(2*variance)
    assert actual == pytest.approx(expected, abs=3e-14)


def test_cap_broadcast_and_absorbing_zero_atom():
    caps = np.array([100., 120.])
    actual = expected_stopping_value(np.array([-np.inf, math.log(100)]), np.array([.2, 0.]),
        [200., 0., 0.], strike=100., is_call=False, value_cap=caps)
    np.testing.assert_array_equal(actual, [100., 120.])
    assert expected_stopping_value(-np.inf, .2, [1e16, 0., 0.],
        strike=100., is_call=True, value_cap="stock") == 0.


@pytest.mark.parametrize("cap", [99., 99.99999999999999, -1., np.inf, np.nan, "stock"])
def test_put_caps_must_dominate_the_intrinsic_exactly(cap):
    with pytest.raises(ValueError, match="cap"):
        expected_stopping_value(math.log(100), .04, [100., 0., 0.],
            strike=100., is_call=False, value_cap=cap)


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("cash", [0., 20., 130.])
@pytest.mark.parametrize("cap", [0., .005, .7, 99.])
def test_residual_proxy_caps_below_strike_agree_with_independent_integration(call, cash, cap):
    knots = (.8, 1., 1.2)
    coefficients = np.array([[12., -15., 5.5], [-2.2, 4.5, 7.9],
        [-.8, 12.5, -.5], [6.4, -9.5, 14.5]])
    actual = expected_stopping_value(math.log(105), .09, coefficients,
        strike=100., is_call=call, cash_dividend=cash, stock_knots=knots,
        exercise=False, value_cap=cap)
    reference = _reference(math.log(105), .09, coefficients, 100., call,
        cash, "pre", knots, exercise=False, value_cap=cap)
    assert actual == pytest.approx(reference, abs=1e-10)
    values = stopping_value(np.array([0., 50., 100., 180.]), coefficients,
        strike=100., is_call=call, cash_dividend=cash, stock_knots=knots,
        exercise=False, value_cap=cap)
    assert np.all((values >= 0.) & (values <= cap))


def test_small_proxy_caps_cover_deterministic_and_zero_stock_atoms():
    actual = expected_stopping_value(np.array([-np.inf, math.log(100)]), [.2, 0.],
        [200., 0., 0.], strike=100., is_call=np.array([True, False]),
        exercise=False, value_cap=np.array([.1, .7]))
    np.testing.assert_array_equal(actual, [.1, .7])


@pytest.mark.parametrize("cap", [-1., np.nan, np.inf, True])
def test_nonexercise_proxy_caps_reject_invalid_values(cap):
    with pytest.raises(ValueError, match="cap"):
        expected_stopping_value(math.log(100), .04, [1., 0., 0.],
            strike=100., is_call=True, exercise=False, value_cap=cap)


def test_discounting_is_external_including_negative_rates():
    value = expected_stopping_value(math.log(100), .04, [0., 0., 0.],
        strike=100., is_call=False)
    for rate in (-.1, .04):
        discount = math.exp(-rate*.5)
        assert discount*value == pytest.approx(discount*_reference(math.log(100), .04,
            [0., 0., 0.], 100., False), abs=4e-12)


@pytest.mark.parametrize("kwargs", [
    {"variance": -.1}, {"variance": True}, {"log_mean": 1j},
    {"log_mean": np.inf}, {"log_mean": np.nan},
    {"coefficients": [1., 2.]}, {"coefficients": [1j, 2., 3.]},
    {"strike": 0.}, {"strike": True}, {"is_call": 1},
    {"cash_dividend": -1.}, {"payoff_side": "unknown"},
    {"exercise": 1},
    {"value_cap": 100.}, {"value_cap": "invalid"}, {"value_cap": True},
    {"stock_knots": (1., .8)}, {"stock_knots": (0.,)},
    {"stock_knots": (.8, 1.)},
])
def test_invalid_inputs(kwargs):
    arguments = dict(log_mean=math.log(100), variance=.04,
        coefficients=[0., 0., 0.], strike=100., is_call=True)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        expected_stopping_value(**arguments)


def test_negative_actual_spot_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        stopping_value(-1., [0., 0., 0.], strike=100., is_call=True)
