"""Limits, temporal covariance and pairing of the selected synthetic simulator."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from roughvol.models.rough_bergomi_model import RoughBergomiModel
from roughvol.sim.volterra import _near_conditional_factor, simulate_blp, simulate_exact
from roughvol.types import MarketData, SimConfig


def test_near_field_residuals_have_the_joint_conditional_covariance():
    H, dt, count = .12, .03, 4
    beta, factor = _near_conditional_factor(H, dt, count)
    recovered = factor@factor.T+np.outer(beta, beta)*dt
    expected = np.empty_like(recovered)
    for i in range(count):
        for j in range(count):
            if i == j:
                expected[i, j] = dt**(2*H)*((i+1)**(2*H)-i**(2*H))/(2*H)
            else:
                expected[i, j] = dt**(2*H)*quad(
                    lambda u: (i+1-u)**(H-.5)*(j+1-u)**(H-.5), 0., 1., epsabs=1e-12)[0]
    np.testing.assert_allclose(recovered, expected, rtol=2e-9, atol=1e-12)
    assert abs((factor@factor.T)[0, 1]) > 1e-4


def test_full_near_field_matches_riemann_liouville_moments():
    H, count, samples = .15, 5, 30_000
    t = np.linspace(0., .25, count+1)
    rng = np.random.default_rng(42)
    dW = rng.normal(size=(samples, count))*np.sqrt(t[1])
    Y = simulate_blp(dW, t, H, kappa=count, rng=rng)
    for index in (1, 3, 5):
        variance = t[index]**(2*H)
        # Five sampling standard deviations for an independent Gaussian moment.
        assert abs(np.mean(Y[:, index]**2)-variance) < 5*variance*np.sqrt(2/samples)
    expected = 2*H*quad(lambda s: (t[3]-s)**(H-.5)*(t[5]-s)**(H-.5), 0., t[3])[0]
    product = Y[:, 3]*Y[:, 5]
    assert abs(product.mean()-expected) < 5*product.std(ddof=1)/np.sqrt(samples)


def test_antithetic_driver_is_paired_including_extra_gaussian_noise():
    rng = np.random.default_rng(19)
    t = np.linspace(0., .2, 9)
    base = rng.normal(size=(3, 8))*np.sqrt(t[1])
    increments = np.concatenate([base, -base])
    Y = simulate_blp(increments, t, .1, rng=rng, antithetic=True)
    np.testing.assert_array_equal(Y[3:], -Y[:3])
    exact, brownian = simulate_exact(t, .1, 6, rng=np.random.default_rng(2), antithetic=True)
    np.testing.assert_array_equal(exact[3:], -exact[:3])
    np.testing.assert_array_equal(brownian[3:], -brownian[:3])


def test_hybrid_rejects_nonuniform_grids_and_unpaired_inputs():
    with pytest.raises(ValueError, match="uniform"):
        simulate_blp(np.zeros((4, 2)), [0., .1, .3], .1)
    with pytest.raises(ValueError, match="paired"):
        simulate_blp(np.ones((4, 2)), [0., .1, .2], .1, antithetic=True)


@pytest.mark.parametrize("scheme", ["volterra-midpoint", "blp-hybrid", "exact-gaussian"])
def test_zero_eta_bs_limit_and_variance_alias_for_rvlab(scheme):
    market = MarketData(100., rate=.01)
    simulation = SimConfig(20_000, .25, n_steps=16, antithetic=True)
    paths = RoughBergomiModel(.1, 0., -.7, .04, scheme=scheme).simulate_paths(
        market=market, sim=simulation, rng=np.random.default_rng(42))
    assert np.isfinite(paths.spot).all()
    np.testing.assert_array_equal(paths.state["variance"], paths.state["var"])
    np.testing.assert_array_equal(paths.state["var"], np.full(paths.spot.shape, .04))
    # Full Brownian pairing holds for the orthogonal exact-scheme innovation too.
    increments = paths.extras["dW_s"]
    np.testing.assert_array_equal(increments[10_000:], -increments[:10_000])
    discounted = np.exp(-.01*.25)*np.maximum(paths.spot_T-100., 0.)
    paired = (discounted[:10_000]+discounted[10_000:])/2
    d1 = (.01+.04/2)*.25/(.2*np.sqrt(.25))
    expected = 100.*ndtr(d1)-100.*np.exp(-.01*.25)*ndtr(d1-.2*np.sqrt(.25))
    assert abs(paired.mean()-expected) < 5*paired.std(ddof=1)/np.sqrt(len(paired))


def test_invalid_forward_variance_curve_is_identifiable():
    market = MarketData(100., forward_variance_curve=lambda t: np.full(t.shape, np.nan))
    with pytest.raises(ValueError, match="finite"):
        RoughBergomiModel(.1, .2, -.7, .04).simulate_paths(
            market=market, sim=SimConfig(4, .1, n_steps=4), rng=np.random.default_rng(42))


def test_expiration_respects_declared_initial_forward_variance():
    market = MarketData(100., forward_variance_curve=lambda t: np.full(t.shape, .09))
    paths = RoughBergomiModel(.1, .2, -.7, .04).simulate_paths(
        market=market, sim=SimConfig(4, 0., n_steps=4), rng=np.random.default_rng(42))
    np.testing.assert_array_equal(paths.spot, np.full((4, 1), 100.))
    np.testing.assert_array_equal(paths.state["variance"], np.full((4, 1), .09))


@pytest.mark.parametrize("times", [[0., .1, .1], [0., np.nan, .2], [.1, .2, .3]])
def test_gaussian_reference_rejects_invalid_time_grids(times):
    with pytest.raises(ValueError, match="grid"):
        simulate_exact(times, .1, 4, rng=np.random.default_rng(42))
