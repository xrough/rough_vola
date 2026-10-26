"""European-only discrete-law martingale controls for the rough-Heston lift.

The control sums ``g(next) - E_discrete[g(next) | current]`` with a fixed
coefficient of one. Its conditional expectation is obtained by enumerating
the *same* three variance outcomes and two stock-splitting orders used by
``rough_simulation``. No continuous-model stock/option expectation centers
the control: time-discretization bias remains in the reported estimate.

An approximate conditional affine-Fourier value supplies g. Its quadrature
and Riccati errors affect variance reduction only, not the discrete target.
The independent Gaussian stock driver is integrated analytically. These
European controls do not implement or validate American optimal stopping,
conditional historical initialization, stock deltas, or cash-dividend prices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.special import ndtr

from .fourier import _quadrature
from .rough_lift import ExponentialKernel, _curve, _implicit_riccati_step
from .rough_simulation import (
    RoughSimulationConfig, RoughSimulationError, _MIN_POSITIVE_VOL_OF_VOL,
    _check_state, _cone_translation, _drift_flow_with_integral,
    _nonnegative_clock, _params, _theta, _time_grid, _weak_transition_branches,
    european_payoff_samples, simulate_rough_heston,
)
from .vanilla_types import SpotMarket, UnsupportedModelError


@dataclass(frozen=True)
class RoughEuropeanControlConfig:
    """Deterministic surrogate budgets; these are not price tolerances."""
    quadrature_nodes: int = 24
    fourier_cutoff: float = 100.
    riccati_steps: int = 512

    def __post_init__(self):
        for name in ("quadrature_nodes", "riccati_steps"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if (isinstance(self.fourier_cutoff, (bool, np.bool_))
                or not isinstance(self.fourier_cutoff, Real)
                or not math.isfinite(self.fourier_cutoff) or self.fourier_cutoff <= 0):
            raise ValueError("fourier_cutoff must be finite and positive")


@dataclass(frozen=True)
class RoughEuropeanEstimate:
    """IID path estimates in broadcast contract order.

    ``sample_covariances[...,i,j]`` uses the variable order conditional
    payoff, control, controlled payoff. Its diagonal divided by paths gives
    the squared reported standard errors. It allows independent replications
    to be pooled without treating repeated/overlapping paths as independent.
    Arrays are read-only. Coefficients are fixed; none are estimated on paths.
    """
    prices: np.ndarray
    standard_errors: np.ndarray
    conditional_prices: np.ndarray
    conditional_standard_errors: np.ndarray
    control_means: np.ndarray
    control_standard_errors: np.ndarray
    sample_covariances: np.ndarray
    config: RoughSimulationConfig
    control_config: RoughEuropeanControlConfig
    diagnostics: object

    def __post_init__(self):
        for name in ("prices", "standard_errors", "conditional_prices",
                "conditional_standard_errors", "control_means", "control_standard_errors",
                "sample_covariances"):
            values = np.asarray(getattr(self, name), float).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _contracts(strike, maturity, is_call):
    try:
        k, t, calls = np.broadcast_arrays(np.asarray(strike), np.asarray(maturity), np.asarray(is_call))
    except (TypeError, ValueError) as exc:
        raise ValueError("strike, maturity and is_call must be broadcastable") from exc
    if (k.dtype.kind not in "iuf" or t.dtype.kind not in "iuf"
            or calls.dtype.kind != "b" or not np.isfinite(k).all()
            or not np.isfinite(t).all() or np.any(k <= 0) or np.any(t < 0) or k.size == 0):
        raise ValueError("strikes must be finite positive reals, maturities finite nonnegative reals, and is_call boolean")
    return k.astype(float), t.astype(float), calls


def _conditional_payoffs(log_mean, variance, strikes, calls):
    """Undiscounted conditional Gaussian values; arbitrary leading axes."""
    std = np.sqrt(variance)
    forward = np.exp(log_mean+.5*variance)
    shape = np.broadcast_shapes(log_mean.shape, np.shape(strikes))
    d2 = np.divide(log_mean-np.log(strikes), std, out=np.zeros(shape), where=std > 0)
    d1 = d2+std
    values = np.where(calls, forward*ndtr(d1)-strikes*ndtr(d2),
        strikes*ndtr(-d2)-forward*ndtr(-d1))
    intrinsic = np.maximum(np.where(calls, forward-strikes, strikes-forward), 0.)
    return np.where(variance == 0, intrinsic, values)


def _centered_increment(values, probabilities, choices, endpoints):
    """Fixed-coefficient zero-mean control under an enumerable discrete law."""
    expectation = np.sum(np.mean(values, axis=2)*probabilities[:, :, None], axis=1)
    return values[np.arange(len(values)), choices, endpoints, :]-expectation


class _AffineSurrogate:
    """Precomputed affine Fourier coefficients and exact future first moments."""
    def __init__(self, params, market, kernel, strikes, maturities, calls, grid, config):
        self.params, self.market = params, market
        self.strikes, self.maturities, self.calls = strikes, maturities, calls
        self.nodes, self.weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
        self.matrix = -np.diag(self.nodes)-params.kappa*np.ones((len(self.nodes), 1))*self.weights
        self.frequencies, self.quadrature = _quadrature(config.quadrature_nodes, config.fourier_cutoff)
        self.contour = self.frequencies-.5j
        self.coefficients, self.moments = {}, {}
        self.groups = {float(t): np.flatnonzero(maturities == t) for t in np.unique(maturities)}
        for tenor in self.groups:
            self._coefficients(tenor, grid[grid <= tenor], config.riccati_steps)

    def _coefficients(self, tenor, times, steps):
        z, x, weights = self.contour, self.nodes, self.weights
        curve = _curve(self.params)
        remaining = tenor-times
        # Distinct small calendar times can share the rounded value tenor-t.
        # They may use the same surrogate coefficient, but every requested
        # time still needs an entry. Surrogate rounding cannot bias a control.
        targets = {}
        for tau, time in zip(remaining, times):
            targets.setdefault(float(tau), []).append(float(time))
        grid = np.unique(np.r_[np.linspace(0., tenor, steps+1), remaining,
            [tenor-k for k in curve.knot_times if 0 < k < tenor]])
        values = np.zeros((len(x), len(z)), complex)
        psi, exponent = np.zeros(len(z), complex), np.zeros(len(z), complex)
        forcing = -.5*(z*z+1j*z)
        linear = 1j*z*self.params.rho*self.params.vol_of_vol-self.params.kappa
        quadratic = .5*self.params.vol_of_vol**2
        previous_f = forcing.copy()
        self.coefficients[(tenor, tenor)] = (exponent.copy(), values.T*weights)
        for start, end in zip(grid[:-1], grid[1:]):
            dt = end-start
            decay = np.exp(-x*dt)
            phi = np.ones_like(x)
            positive = x > 0
            phi[positive] = -np.expm1(-x[positive]*dt)/(x[positive]*dt)
            endpoint = np.full_like(x, .5*dt)
            small = x*dt < 1e-3
            q = x[small]*dt
            endpoint[small] = dt*(.5+q*(-1/6+q*(1/24-q/120)))
            endpoint[~small] = (1-phi[~small])/x[~small]
            initial = dt*phi-endpoint
            history = (weights*decay)@values+(weights@initial)*previous_f
            next_psi = _implicit_riccati_step(history, forcing, linear, quadratic, float(weights@endpoint))
            next_f = forcing+linear*next_psi+quadratic*next_psi*next_psi
            values = decay[:, None]*values+initial[:, None]*previous_f+endpoint[:, None]*next_f
            theta = curve(tenor-.5*(start+end))
            exponent += .5*dt*(self.params.v0*(previous_f+next_f)
                +self.params.kappa*theta*(psi+next_psi))
            if not np.isfinite(values).all() or not np.isfinite(exponent).all():
                raise RoughSimulationError("conditional Fourier control Riccati calculation diverged")
            if float(end) in targets:
                for time in targets[float(end)]:
                    self.coefficients[(tenor, time)] = (exponent.copy(), values.T*weights)
            psi, previous_f = next_psi, next_f

    def _remaining_integral(self, time, tenor):
        key = (time, tenor)
        if key in self.moments:
            return self.moments[key]
        count = len(self.nodes)
        flow, shift = np.eye(count), np.zeros(count)
        coefficient, constant = np.zeros(count), 0.
        knots = _curve(self.params).knot_times
        grid = [time]+[k for k in knots if time < k < tenor]+[tenor]
        for start, end in zip(grid[:-1], grid[1:]):
            theta = _theta(self.params, .5*(start+end))
            forcing = np.full(count, self.params.kappa*(theta-self.params.v0))
            exact = _drift_flow_with_integral(self.matrix, forcing, self.weights,
                self.params.v0, end-start)
            coefficient += exact[count, :count]@flow
            constant += exact[count, :count]@shift+exact[count, -1]
            flow, shift = exact[:count, :count]@flow, exact[:count, :count]@shift+exact[:count, -1]
        self.moments[key] = coefficient, constant
        return coefficient, constant

    def values(self, time, factors, log_mean, variance, active):
        """Values on (path, diffusion branch, endpoint, active contract)."""
        result = np.empty(log_mean.shape+(len(active),))
        frequency_square = self.frequencies**2+.25
        for tenor, all_columns in self.groups.items():
            columns = all_columns[np.isin(all_columns, active)]
            if not len(columns):
                continue
            positions = np.searchsorted(active, columns)
            a, b = self.coefficients[(tenor, time)]
            coefficient, constant = self._remaining_integral(time, tenor)
            remaining_variance = _nonnegative_clock(factors@coefficient+constant)
            total_variance = variance+remaining_variance[:, :, None]
            growth = math.log(self.market.discount_curve.discount(time)/self.market.discount_curve.discount(tenor))
            future_mean = log_mean+growth-.5*remaining_variance[:, :, None]
            forward = np.exp(future_mean+.5*total_variance)
            with np.errstate(over="ignore", invalid="ignore"):
                cf = np.exp(a+factors@b.T)[:, :, None, :]*np.exp(-.5*variance[:, :, :, None]*frequency_square)
                reference = np.exp(-.5*total_variance[:, :, :, None]*frequency_square)
                integrand = (cf-reference)*(self.quadrature/frequency_square)
                for column, position in zip(columns, positions):
                    strike = self.strikes[column]
                    baseline = _conditional_payoffs(future_mean, total_variance, strike, self.calls[column])
                    phase = np.exp(1j*np.log(forward/strike)[:, :, :, None]*self.frequencies)
                    correction = np.real(np.sum(phase*integrand, axis=-1))*np.sqrt(forward*strike)/np.pi
                    result[..., position] = self.market.discount_curve.discount(tenor)*(baseline-correction)
        if not np.isfinite(result).all():
            raise RoughSimulationError("conditional Fourier control produced nonfinite values")
        return result


def _estimate_result(payoffs, controls, shape, config, control_config, diagnostics):
    if (payoffs.ndim != 2 or payoffs.shape != controls.shape
            or payoffs.shape[0] != config.paths or not np.isfinite(payoffs).all()
            or not np.isfinite(controls).all()):
        raise RoughSimulationError("European payoff/control samples are nonfinite or malformed")
    with np.errstate(over="ignore", invalid="ignore"):
        controlled = payoffs-controls
        variables = np.stack((payoffs, controls, controlled), axis=-1)
        means = variables.mean(axis=0)
        centered = variables-means
        covariances = np.einsum("nci,ncj->cij", centered, centered)/(config.paths-1)
        errors = np.sqrt(np.maximum(np.diagonal(covariances, axis1=-2, axis2=-1), 0.)/config.paths)
    if not all(np.isfinite(array).all() for array in (controlled, means, covariances, errors)):
        raise RoughSimulationError("European sample means, covariances or standard errors overflowed")
    return RoughEuropeanEstimate(means[:, 2].reshape(shape), errors[:, 2].reshape(shape),
        means[:, 0].reshape(shape), errors[:, 0].reshape(shape), means[:, 1].reshape(shape),
        errors[:, 1].reshape(shape), covariances.reshape(shape+(3, 3)), config, control_config, diagnostics)


def estimate_rough_european(params, market, kernel, *, strike, maturity, is_call=True,
                            config=None, control_config=None):
    """Estimate inception European prices with IID discrete-law controls.

    All contracts share paths, preserving their supplied broadcast order.
    Standard errors describe sampling uncertainty in the selected *discrete*
    scheme, not its time bias, finite-kernel bias, or reference-price errors.
    Changing control budgets cannot change the expectation of that scheme.
    No reference price is accepted or used to center a sample.
    """
    started = perf_counter()
    _params(params)
    if not isinstance(market, SpotMarket):
        raise TypeError("market must be SpotMarket with explicit dividends")
    if not isinstance(kernel, ExponentialKernel):
        raise TypeError("kernel must be ExponentialKernel")
    config = RoughSimulationConfig() if config is None else config
    control_config = RoughEuropeanControlConfig() if control_config is None else control_config
    if not isinstance(config, RoughSimulationConfig):
        raise TypeError("config must be RoughSimulationConfig")
    if not isinstance(control_config, RoughEuropeanControlConfig):
        raise TypeError("control_config must be RoughEuropeanControlConfig")
    if config.paths < 2:
        raise ValueError("European uncertainty estimates require at least two independent paths")
    strikes, maturities, calls = _contracts(strike, maturity, is_call)
    shape = strikes.shape
    strikes, maturities, calls = (a.reshape(-1) for a in (strikes, maturities, calls))
    times = np.unique(maturities)
    horizon = float(times[-1])
    if horizon > market.dividends.coverage+1e-12:
        raise ValueError("dividend expectations do not cover all maturities")
    if any(e.amount > 0 and e.time <= horizon for e in market.dividends.events):
        raise UnsupportedModelError("conditional rough European controls do not support positive cash dividends")
    market.discount_curve.discount(times)
    if kernel.horizon is not None and horizon > kernel.horizon+1e-12:
        raise ValueError("maturities exceed the kernel approximation horizon")
    if kernel.hurst is not None and abs(kernel.hurst-params.hurst) > 1e-14:
        raise ValueError("kernel Hurst parameter does not match model")
    if horizon > 0 and 0 < params.vol_of_vol < _MIN_POSITIVE_VOL_OF_VOL:
        raise RoughSimulationError("positive vol_of_vol below 1e-8 is unsupported by the correlated-stock representation")
    n, count, contracts = config.paths, len(kernel.nodes), len(strikes)
    grid = _time_grid(params, market, times, config.steps)
    estimated_memory = 8*(n*(count+6+12*contracts)
        +min(n, config.batch_size)*(100*control_config.quadrature_nodes+12*count+20*contracts+96)
        +len(grid)*len(times)*(count+1)*control_config.quadrature_nodes*2)
    if estimated_memory > config.memory_budget_mb*1024**2:
        raise MemoryError(f"European controls require approximately {estimated_memory/1024**2:.2f} MiB, exceeding memory_budget_mb={config.memory_budget_mb}")
    diagnostics = {"backend": "rough_european_discrete_martingale_control", "exercise": "european",
        "model_clock": "inception", "paths": n, "actual_time_steps": len(grid)-1,
        "stock_integral": config.stock_integral, "estimated_memory_bytes": estimated_memory,
        "kernel_fingerprint": kernel.fingerprint, "control_coefficient": 1.,
        "control_centering": "exact shared six-branch discrete transition law",
        "standard_error": "IID sample standard deviation divided by sqrt(paths)",
        "continuous_reference_used_to_center": False,
        "american_pricing_ready": False, "stock_delta_ready": False,
        "surrogate_settings": asdict(control_config)}
    if horizon == 0 or params.vol_of_vol == 0:
        simulation = simulate_rough_heston(params, market, kernel, observation_times=times, config=config)
        payoffs = np.column_stack([european_payoff_samples(simulation, strike=k,
            maturity=t, is_call=bool(call)) for k, t, call in zip(strikes, maturities, calls)])
        diagnostics["deterministic_conditional_estimator"] = True
        diagnostics["elapsed_seconds"] = perf_counter()-started
        return _estimate_result(payoffs, np.zeros_like(payoffs), shape, config, control_config, diagnostics)
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    translation, cone = _cone_translation(params.v0, nodes, weights)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    drift_cache = {}
    surrogate = _AffineSurrogate(params, market, kernel, strikes, maturities, calls, grid, control_config)
    factors, log_mean, variance = np.zeros((n, count)), np.full(n, math.log(market.spot)), np.zeros(n)
    controls, payoffs = np.zeros((n, contracts)), np.zeros((n, contracts))
    zero_columns = np.flatnonzero(maturities == 0)
    if len(zero_columns):
        payoffs[:, zero_columns] = np.maximum(np.where(calls[zero_columns],
            market.spot-strikes[zero_columns], strikes[zero_columns]-market.spot), 0.)
    state_diagnostics = {"minimum_raw_variance": params.v0,
        "minimum_cone_coordinate": float(np.min(translation@cone.T)), "aggregate_roundoff_count": 0}
    gains = []
    rng = np.random.default_rng(config.seed)
    for left, right in zip(grid[:-1], grid[1:]):
        dt, theta = float(right-left), _theta(params, float((left+right)/2))
        key = (dt, theta)
        if key not in drift_cache:
            forcing = np.full(count, params.kappa*(theta-params.v0))
            exact = _drift_flow_with_integral(matrix, forcing, weights, params.v0, dt/2)
            drift_cache[key] = exact
        exact = drift_cache[key]
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        active = np.flatnonzero(maturities >= right)
        for begin in range(0, n, config.batch_size):
            end = min(n, begin+config.batch_size)
            old = factors[begin:end]
            _check_state(old, params.v0, weights, translation, cone, state_diagnostics, stage="step start", time=left)
            branches = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
                flow=exact[:count, :count], shift=exact[:count, -1], integrated_flow=exact,
                dt=dt, theta=theta, growth=growth, stock_integral=config.stock_integral)
            # The third uniform is deliberately consumed to match the path
            # simulator's time/path stream although its Gaussian is integrated.
            random = rng.random((end-begin, 3))
            probability = branches["probabilities"]
            choice = (random[:, 0] > probability[:, 0]).astype(int)+(random[:, 0] > probability[:, :2].sum(axis=1)).astype(int)
            endpoint = (random[:, 1] >= .5).astype(int)
            mean = log_mean[begin:end, None, None]+branches["correlated"][:, :, None]-.5*branches["orthogonal"]
            clock = variance[begin:end, None, None]+branches["orthogonal"]
            values = surrogate.values(float(right), branches["new_factors"], mean, clock, active)
            controls[begin:end, active] += _centered_increment(values, probability, choice, endpoint)
            selected = np.arange(end-begin)
            factors[begin:end] = branches["new_factors"][selected, choice]
            log_mean[begin:end] = mean[selected, choice, endpoint]
            variance[begin:end] = clock[selected, choice, endpoint]
            _check_state(branches["half_factors"], params.v0, weights, translation, cone,
                state_diagnostics, stage="first drift", time=left+dt/2)
            _check_state(factors[begin:end], params.v0, weights, translation, cone,
                state_diagnostics, stage="second drift", time=right)
        columns = np.flatnonzero(maturities == right)
        if len(columns):
            payoffs[:, columns] = market.discount_curve.discount(right)*_conditional_payoffs(
                log_mean[:, None], variance[:, None], strikes[columns], calls[columns])
            discounted_stock = market.discount_curve.discount(right)*np.exp(log_mean+.5*variance)
            gains.append({"time": float(right), "mean": float(discounted_stock.mean()),
                "standard_error": float(discounted_stock.std(ddof=1)/np.sqrt(n))})
    diagnostics.update(state_diagnostics)
    diagnostics["conditional_discounted_stock"] = gains
    diagnostics["elapsed_seconds"] = perf_counter()-started
    diagnostics["time_grid"] = grid.tolist()
    return _estimate_result(payoffs, controls, shape, config, control_config, diagnostics)


__all__ = ["RoughEuropeanControlConfig", "RoughEuropeanEstimate", "estimate_rough_european"]
