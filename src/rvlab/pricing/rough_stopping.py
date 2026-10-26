"""Independent-policy primal/dual stopping for the finite rough-Heston lift.

The requested American contract is approximated on an explicitly refined
exercise grid.  A frozen least-squares policy is evaluated on an independent
random stream.  At *every* integration step a piecewise quadratic stopping
surrogate is centered with the exact six-branch/Gaussian transition law.
The resulting martingale supplies both a stopped primal control and a dual
upper estimate (Andersen--Broadie, 2004).  The bounds concern this discrete
model and exercise grid; they do not certify kernel or continuous-time error.

All factors enter the policy, including their cross products.  Only model
inception (zero initial factor memory) is supported.  Stock bumps keep model
parameters, factor initialization, rates and declared cash dividends fixed.
No fitted European price or continuous-model martingale centers a control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.special import ndtri

from .rough_estimators import _contracts, RoughEuropeanControlConfig
from .rough_dual import RoughDualConfig
from .rough_lift import ExponentialKernel
from .rough_simulation import (
    RoughSimulationConfig, RoughSimulationError, _drift_flow_with_integral,
    _nonnegative_clock, _params, _theta, _time_grid, _weak_transition_branches,
    simulate_rough_heston,
)
from .rough_stopping_controls import expected_stopping_value, stopping_value
from .vanilla_types import SpotMarket, UnsupportedModelError


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _real(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}real number")


@dataclass(frozen=True)
class RoughAmericanConfig:
    """Simulation/regression budgets, never replacements for acceptance gates.

    ``steps`` and ``exercise_steps`` are independent uniform subdivisions;
    rate/mean-reversion knots and cash events are added exactly.  Distinct
    training and valuation seeds are mandatory.  ``retrain_bumps`` adds an
    independent-policy construction sensitivity check using the same training
    random numbers at bumped stocks, never valuation paths.

    In the public pricing/training entry points, ``dual_proxy='obstacle'``
    replaces only the value-control interpolation after the exercise policy
    has been trained. ``stock_knots`` still specifies that policy's regression
    basis; ``dual_proxy_knots`` separately specifies the control stock grid.
    """
    training_paths: int = 8192
    valuation_paths: int = 8192
    steps: int = 512
    exercise_steps: int = 64
    training_seed: int = 42
    valuation_seed: int = 43
    batch_size: int = 1024
    memory_budget_mb: float = 1024.
    ridge: float = 1e-8
    stock_knots: tuple[float, ...] = (.7, .85, .95, 1., 1.05, 1.15, 1.3)
    bump_relative: float = .005
    retrain_bumps: bool = False
    stock_integral: str = "exact_drift"
    training_method: str = "least_squares"
    bounded_surrogate: bool = True
    surrogate_interpolation: str = "coefficients"
    premium_cap: float = .7
    premium_control: RoughEuropeanControlConfig = field(default_factory=lambda: RoughEuropeanControlConfig(24, 100., 512))
    dual_config: RoughDualConfig | None = None
    dual_proxy: str = "regression"
    dual_proxy_knots: tuple[float, ...] | None = None

    def __post_init__(self):
        for name in ("training_paths", "valuation_paths"):
            _positive_integer(getattr(self, name), name, 2)
        for name in ("steps", "exercise_steps", "batch_size"):
            _positive_integer(getattr(self, name), name)
        for name in ("training_seed", "valuation_seed"):
            _positive_integer(getattr(self, name), name, 0)
        if self.training_seed == self.valuation_seed:
            raise ValueError("training_seed and valuation_seed must differ: policy valuation paths must be independent")
        for name in ("memory_budget_mb", "bump_relative"):
            _real(getattr(self, name), name, positive=True)
        _real(self.ridge, "ridge")
        if self.ridge < 0 or self.bump_relative >= .25:
            raise ValueError("ridge must be nonnegative and bump_relative must be less than .25")
        if not isinstance(self.retrain_bumps, (bool, np.bool_)):
            raise ValueError("retrain_bumps must be boolean")
        knots = tuple(self.stock_knots)
        for knot in knots:
            _real(knot, "stock knot", positive=True)
        if any(b <= a for a, b in zip(knots, knots[1:])):
            raise ValueError("stock_knots must be strictly increasing positive normalized stocks")
        object.__setattr__(self, "stock_knots", tuple(float(k) for k in knots))
        if self.stock_integral not in ("exact_drift", "trapezoidal"):
            raise ValueError("stock_integral must be 'exact_drift' or 'trapezoidal'")
        if self.training_method not in ("least_squares", "conditional", "fourier_premium", "fourier_bellman"):
            raise ValueError("unknown rough stopping training_method")
        _real(self.premium_cap, "premium_cap", positive=True)
        if not isinstance(self.premium_control, RoughEuropeanControlConfig):
            raise TypeError("premium_control must be RoughEuropeanControlConfig")
        if self.dual_config is not None:
            if not isinstance(self.dual_config, RoughDualConfig):
                raise TypeError("dual_config must be RoughDualConfig or None")
            if self.training_method not in ("fourier_premium", "fourier_bellman"):
                raise ValueError("dual_config requires a Fourier-premium stopping backend")
            if self.dual_config.training_seed in (self.training_seed, self.valuation_seed):
                raise ValueError("dual-training seed must differ from policy-training and valuation seeds")
        if self.dual_proxy not in ("regression", "obstacle"):
            raise ValueError("dual_proxy must be 'regression' or 'obstacle'")
        if self.dual_proxy == "obstacle" and self.training_method not in ("fourier_premium", "fourier_bellman"):
            raise ValueError("obstacle dual_proxy requires a Fourier-premium stopping backend")
        if self.dual_proxy_knots is not None:
            if self.dual_proxy != "obstacle":
                raise ValueError("dual_proxy_knots requires the obstacle dual_proxy")
            proxy_knots = tuple(self.dual_proxy_knots)
            if not proxy_knots:
                raise ValueError("dual_proxy_knots must be nonempty")
            for knot in proxy_knots:
                _real(knot, "dual proxy stock knot", positive=True)
            if any(b <= a for a,b in zip(proxy_knots,proxy_knots[1:])):
                raise ValueError("dual_proxy_knots must be strictly increasing")
            object.__setattr__(self,"dual_proxy_knots",tuple(float(k) for k in proxy_knots))
        if self.surrogate_interpolation not in ("coefficients", "value_bridge"):
            raise ValueError("surrogate_interpolation must be 'coefficients' or 'value_bridge'")
        if not isinstance(self.bounded_surrogate, (bool, np.bool_)):
            raise ValueError("bounded_surrogate must be boolean")

    def refined(self, axis="joint"):
        """Declare independent numerical refinements without changing tolerances."""
        if axis not in ("time", "exercise", "sampling", "training", "bump", "joint"):
            raise ValueError("unknown rough-American refinement axis")
        changes = {}
        if axis in ("time", "joint"):
            changes["steps"] = 2*self.steps
        if axis in ("exercise", "joint"):
            changes["exercise_steps"] = 2*self.exercise_steps
        if axis in ("sampling", "joint"):
            changes["valuation_paths"] = 2*self.valuation_paths
        if axis in ("training", "joint"):
            changes["training_paths"] = 2*self.training_paths
        if axis == "bump":
            changes["bump_relative"] = self.bump_relative/2
        return replace(self, **changes)


@dataclass(frozen=True)
class _Continuation:
    center: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    knots: tuple
    rank: int
    condition: float

    def __post_init__(self):
        for name in ("center", "scale", "beta"):
            array = np.asarray(getattr(self, name), float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def coefficients(self, factors):
        """Piecewise coefficients in z=S/K, retaining every factor coordinate."""
        r = (factors-self.center)/self.scale
        n, h = r.shape[-1], len(self.knots)
        a = np.full(r.shape[:-1], self.beta[0])
        b = np.full(r.shape[:-1], self.beta[1])
        c = np.full(r.shape[:-1], self.beta[2])
        offset = 3+h
        a += r@self.beta[offset:offset+n]
        b += r@self.beta[offset+n:offset+2*n]
        offset += 2*n
        i, j = np.triu_indices(n)
        a += (r[..., i]*r[..., j])@self.beta[offset:]
        result = np.stack((a, b, c), axis=-1)[..., None, :]
        result = np.broadcast_to(result, r.shape[:-1]+(h+1, 3)).copy()
        for k, knot in enumerate(self.knots):
            gamma = self.beta[3+k]
            result[..., k+1:, 0] += gamma*knot*knot
            result[..., k+1:, 1] -= 2*gamma*knot
            result[..., k+1:, 2] += gamma
        return result if h else result[..., 0, :]

    def value(self, spots, factors, strike):
        if spots.ndim > factors.ndim-1:
            extra = spots.ndim-(factors.ndim-1)
            factors = factors.reshape(factors.shape[:-1]+(1,)*extra+(factors.shape[-1],))
            factors = np.broadcast_to(factors, spots.shape+(factors.shape[-1],))
        z = spots/strike
        coefficients = self.coefficients(factors)
        if self.knots:
            index = np.searchsorted(self.knots, z, side="right")
            coefficients = np.take_along_axis(coefficients, index[..., None, None], axis=-2)[..., 0, :]
        return coefficients[..., 0]+z*(coefficients[..., 1]+z*coefficients[..., 2])


@dataclass(frozen=True)
class RoughStoppingPolicy:
    """Frozen inception policy and training-only normalization/provenance.

    ``continuations`` align with ``continuation_times``; eligible exercise
    dates are stored separately in ``exercise_times``. European policies
    contain zero and expiry in that array, with exercise allowed only at expiry.
    """
    params: object
    market: SpotMarket
    kernel: ExponentialKernel
    strike: float
    maturity: float
    is_call: bool
    exercise_style: str
    exercise_times: np.ndarray
    continuation_times: np.ndarray
    continuations: tuple
    exercise_at_inception: bool
    config: RoughAmericanConfig
    diagnostics: object

    def __post_init__(self):
        for name in ("exercise_times", "continuation_times"):
            times = np.asarray(getattr(self, name), float).copy()
            times.setflags(write=False)
            object.__setattr__(self, name, times)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class RoughAmericanEstimate:
    """Independent frozen-policy estimates in broadcast contract order.

    ``prices`` are controlled primal estimates.  Conditional on the trained
    policy their expectations are lower bounds for the *discrete* problem;
    ``dual_prices`` are upper expectations for that same problem.  Finite
    The paired sampled gap is nonnegative pathwise; neither endpoint is clipped.
    Stock delta differentiates this frozen policy with paired random numbers.
    Price bounds do not imply delta bounds.  All readiness flags start false.
    """
    prices: np.ndarray
    standard_errors: np.ndarray
    dual_prices: np.ndarray
    dual_standard_errors: np.ndarray
    dual_gaps: np.ndarray
    dual_gap_standard_errors: np.ndarray
    deltas: np.ndarray
    delta_standard_errors: np.ndarray
    one_sided_deltas: np.ndarray
    one_sided_delta_standard_errors: np.ndarray
    bump_deltas: np.ndarray
    bump_delta_standard_errors: np.ndarray
    raw_prices: np.ndarray
    raw_standard_errors: np.ndarray
    exercise_probabilities: np.ndarray
    retrained_deltas: np.ndarray
    retrained_delta_standard_errors: np.ndarray
    config: RoughAmericanConfig
    policies: tuple
    diagnostics: object

    def __post_init__(self):
        for name in ("prices", "standard_errors", "dual_prices", "dual_standard_errors",
                "dual_gaps", "dual_gap_standard_errors", "deltas", "delta_standard_errors",
                "one_sided_deltas", "one_sided_delta_standard_errors", "bump_deltas",
                "bump_delta_standard_errors", "raw_prices", "raw_standard_errors",
                "exercise_probabilities", "retrained_deltas", "retrained_delta_standard_errors"):
            array = np.asarray(getattr(self, name), float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def settings(self):
        return {"backend": "rough_heston_lifted_primal_dual", **asdict(self.config)}

    @property
    def primal_standard_errors(self):
        return self.standard_errors

    @property
    def paired_gaps(self):
        return self.dual_gaps

    @property
    def paired_gap_standard_errors(self):
        return self.dual_gap_standard_errors

    @property
    def delta_left(self):
        return self.one_sided_deltas[..., 0]

    @property
    def delta_right(self):
        return self.one_sided_deltas[..., 1]

    @property
    def exercise_style(self):
        return self.diagnostics["exercise_style"]

    @property
    def price_stable(self):
        return self.diagnostics.get("price_stable")

    @property
    def delta_stable(self):
        return self.diagnostics.get("delta_stable")

    @property
    def grid(self):
        return None

    @property
    def exercise_boundaries(self):
        return ()


def _payoff(spots, strike, calls):
    return np.maximum(np.where(calls, spots-strike, strike-spots), 0.)


def _fit_continuation(spots, factors, targets, strike, config):
    center = factors.mean(axis=0)
    scale = factors.std(axis=0)
    # A deterministic coordinate is retained as zero, not replaced by an
    # aggregate-variance-only policy or a fictitious stochastic factor.
    scale[scale < 128*np.finfo(float).eps*np.maximum(1., np.abs(center))] = 1.
    r, z = (factors-center)/scale, spots/strike
    i, j = np.triu_indices(factors.shape[-1])
    columns = [np.ones((len(z), 1)), z[:, None], (z*z)[:, None]]
    columns += [np.maximum(z[:, None]-np.asarray(config.stock_knots), 0.)**2,
        r, z[:, None]*r, r[:, i]*r[:, j]]
    design = np.column_stack(columns)
    column_scale = np.sqrt(np.mean(design*design, axis=0))
    column_scale[column_scale < np.finfo(float).eps] = 1.
    x = design/column_scale
    gram, rhs = x.T@x/len(z), x.T@targets/len(z)
    eigenvalues, vectors = np.linalg.eigh(gram)
    threshold = max(float(eigenvalues[-1]), 1.)*len(eigenvalues)*np.finfo(float).eps
    active = eigenvalues > threshold
    denominator = np.maximum(eigenvalues, 0.)+config.ridge
    inverse = np.divide(1., denominator, out=np.zeros_like(denominator),
        where=(denominator > 0) & (active | (config.ridge > 0)))
    beta = (vectors@(inverse*(vectors.T@rhs)))/column_scale
    if not np.isfinite(beta).all():
        raise RoughSimulationError("stopping regression produced nonfinite coefficients")
    condition = float(eigenvalues[-1]/eigenvalues[active][0]) if np.any(active) else math.inf
    return _Continuation(center, scale, beta, config.stock_knots, int(np.sum(active)), condition)


def _zero_continuation(factors, config):
    n = factors.shape[-1]
    count = 3+len(config.stock_knots)+2*n+n*(n+1)//2
    return _Continuation(np.zeros(n), np.ones(n), np.zeros(count), config.stock_knots, 0, 1.)


def _surrogate_cap(market, *, strike, maturity, time, is_call, enabled):
    """A numerical proxy cap, not a price bound for the weak stock law."""
    if not enabled:
        return None
    if is_call:
        return "stock"
    times = [time, maturity]+[t for t in market.discount_curve.times if time < t < maturity]
    return max(float(strike), float(strike*np.max(market.discount_curve.discount(times))
        /market.discount_curve.discount(time)))


def _decision_continuation(values, spots, market, *, strike, maturity, time, is_call, enabled):
    if not enabled:
        return values
    cap = _surrogate_cap(market, strike=strike, maturity=maturity, time=time,
        is_call=is_call, enabled=True)
    return np.minimum(np.maximum(values, 0.), spots if cap == "stock" else cap)


def _conditional_training_labels(params, market, kernel, spots, factors, *, left, right,
                                  continuation, config, strike, is_call, exercise,
                                  payoff_side, maturity=None):
    """Noise-free one-step Bellman labels under the actual discrete law."""
    count = len(kernel.nodes)
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    dt, theta = float(right-left), _theta(params, .5*(left+right))
    duration = dt if params.vol_of_vol == 0 else dt/2
    exact = _drift_flow_with_integral(matrix, np.full(count, params.kappa*(theta-params.v0)),
        weights, params.v0, duration)
    growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
    cash = sum(event.amount for event in market.dividends.events if event.time == right)
    value_cap = _surrogate_cap(market, strike=strike,
        maturity=right if maturity is None else maturity, time=right, is_call=is_call,
        enabled=config.bounded_surrogate)
    labels = np.empty(len(spots))
    for begin in range(0, len(spots), config.batch_size):
        end = min(len(spots), begin+config.batch_size)
        old, stock = factors[begin:end], spots[begin:end]
        with np.errstate(divide="ignore"):
            log_stock = np.log(stock)
        if params.vol_of_vol == 0:
            new = old@exact[:count, :count].T+exact[:count, -1]
            clock = _nonnegative_clock(old@exact[count, :count]+exact[count, -1])
            branch_factors = new[:, None, :]
            means = log_stock[:, None, None]+growth-.5*clock[:, None, None]
            clocks = clock[:, None, None]
            probabilities = np.ones((end-begin, 1))
        else:
            law = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
                flow=exact[:count, :count], shift=exact[:count, -1], integrated_flow=exact,
                dt=dt, theta=theta, growth=growth, stock_integral=config.stock_integral)
            branch_factors, probabilities = law["new_factors"], law["probabilities"]
            clocks = law["orthogonal"]
            means = log_stock[:, None, None]+law["correlated"][:, :, None]-.5*clocks
        coefficients = continuation.coefficients(branch_factors)[:, :, None, ...]
        values = expected_stopping_value(means, clocks, coefficients, strike=strike,
            is_call=is_call, cash_dividend=cash, payoff_side=payoff_side,
            stock_knots=config.stock_knots, exercise=exercise, value_cap=value_cap)
        labels[begin:end] = (market.discount_curve.discount(right)/market.discount_curve.discount(left)
            *np.sum(probabilities*values.mean(axis=2), axis=1))
    if not np.isfinite(labels).all():
        raise RoughSimulationError("conditional Bellman training produced nonfinite labels")
    return labels


def _inputs(params, market, kernel, strike, maturity, is_call, config, exercise_style,
            conditional_state):
    _params(params)
    if not isinstance(market, SpotMarket) or not isinstance(kernel, ExponentialKernel):
        raise TypeError("market must be SpotMarket and kernel must be ExponentialKernel")
    if not isinstance(config, RoughAmericanConfig):
        raise TypeError("config must be RoughAmericanConfig")
    if exercise_style not in ("american", "european"):
        raise ValueError("exercise_style must be 'american' or 'european'")
    if conditional_state is not None:
        raise UnsupportedModelError("rough stopping supports inception only; historical factor memory cannot be reset")
    k, t, calls = _contracts(strike, maturity, is_call)
    market.discount_curve.discount(t)
    if np.max(t) > market.dividends.coverage+1e-12:
        raise ValueError("dividend expectations do not cover all contracts")
    if kernel.horizon is not None and np.max(t) > kernel.horizon+1e-12:
        raise ValueError("maturities exceed the kernel approximation horizon")
    if kernel.hurst is not None and abs(kernel.hurst-params.hurst) > 1e-14:
        raise ValueError("kernel Hurst parameter does not match model")
    return k, t, calls


def train_rough_stopping_policy(params, market, kernel, *, strike, maturity,
                                 is_call=True, config=None, exercise_style="american",
                                 conditional_state=None):
    """Train one scalar contract on a stream never used for reported valuation."""
    started = perf_counter()
    config = RoughAmericanConfig() if config is None else config
    k, t, calls = _inputs(params, market, kernel, strike, maturity, is_call, config,
        exercise_style, conditional_state)
    if k.ndim:
        raise ValueError("policy training accepts one scalar contract")
    k, t, calls = float(k), float(t), bool(calls)
    if t > 0 and config.training_method in ("fourier_premium", "fourier_bellman"):
        from .rough_stopping_fourier import train_fourier_stopping_policy, train_fourier_bellman_policy
        train = (train_fourier_stopping_policy if config.training_method == "fourier_premium"
            else train_fourier_bellman_policy)
        policy = train(params, market, kernel, strike=k, maturity=t, is_call=calls,
            config=config, control_config=config.premium_control,
            exercise_style=exercise_style, premium_cap=config.premium_cap)
        if config.dual_proxy == "obstacle":
            from .rough_obstacle_premium import build_rough_obstacle_proxy
            options = {} if config.dual_proxy_knots is None else {"stock_knots":config.dual_proxy_knots}
            policy = build_rough_obstacle_proxy(policy, **options)
        return policy
    positive_events = {e.time: e.amount for e in market.dividends.events if e.time <= t and e.amount > 0}
    if t == 0:
        training_dates = np.array([0.])
    else:
        # European controls also learn terminal-payoff continuation surfaces
        # at these dates.  Their policy still prohibits all early exercise.
        training_dates = np.unique(np.r_[np.linspace(0., t, config.exercise_steps+1), list(positive_events)])
    exercise_times = training_dates if exercise_style == "american" else np.unique([0., t])
    times = (_time_grid(params, market, training_dates, config.steps)
        if config.training_method == "conditional" else training_dates)
    count = len(kernel.nodes)
    features = 3+len(config.stock_knots)+2*count+count*(count+1)//2
    estimated_memory = 8*(config.training_paths*len(times)*(count+5)
        +config.training_paths*(2*features+count+16)
        +min(config.training_paths, config.batch_size)
            *(256*(len(config.stock_knots)+1)+48*count))
    if estimated_memory > config.memory_budget_mb*1024**2:
        raise MemoryError(f"rough stopping training requires approximately {estimated_memory/1024**2:.2f} MiB, exceeding memory_budget_mb={config.memory_budget_mb}")
    simulation_config = RoughSimulationConfig(config.training_paths, config.steps,
        config.training_seed, config.batch_size, config.memory_budget_mb, config.stock_integral)
    simulation = simulate_rough_heston(params, market, kernel, observation_times=times,
        config=simulation_config)
    spots, factors = simulation.spots, simulation.factors
    continuations = [None]*len(times)
    continuations[-1] = _zero_continuation(factors[:, -1], config)
    if config.training_method == "conditional" and len(times) > 1:
        for index in range(len(times)-2, -1, -1):
            labels = _conditional_training_labels(params, market, kernel,
                spots[:, index], factors[:, index], left=float(times[index]), right=float(times[index+1]),
                continuation=continuations[index+1], config=config, strike=k, is_call=calls,
                exercise=(exercise_style == "american" and times[index+1] in exercise_times)
                    or times[index+1] == t,
                payoff_side="both" if exercise_style == "american" else "post", maturity=t)
            if index:
                continuations[index] = _fit_continuation(spots[:, index], factors[:, index], labels, k, config)
            else:
                zero = _zero_continuation(factors[:, index], config)
                beta = zero.beta.copy()
                beta[0] = float(labels.mean())
                continuations[index] = replace(zero, beta=beta, rank=1)
        target = np.full(len(spots), labels.mean())
    else:
        cash = simulation.discounted_gains-simulation.discounts[None, :]*spots
        cash_payouts = np.column_stack((np.zeros(len(spots)), np.diff(cash, axis=1)))
        pre_spots = spots+cash_payouts/simulation.discounts[None, :]
        target = simulation.discounts[-1]*_payoff(spots[:, -1], k, calls)
        if exercise_style == "american" and times[-1] in positive_events:
            target = np.maximum(target, simulation.discounts[-1]*_payoff(pre_spots[:, -1], k, calls))
        for index in range(len(times)-2, 0, -1):
            continuation = _fit_continuation(spots[:, index], factors[:, index],
                target/simulation.discounts[index], k, config)
            continuations[index] = continuation
            if exercise_style == "european":
                continue
            expected = continuation.value(spots[:, index], factors[:, index], k)
            expected = _decision_continuation(expected, spots[:, index], market, strike=k,
                maturity=t, time=float(times[index]), is_call=calls, enabled=config.bounded_surrogate)
            post = _payoff(spots[:, index], k, calls)
            pre = _payoff(pre_spots[:, index], k, calls)
            post_stop = (post > 0) & (post >= expected)
            pre_stop = (pre > 0) & (pre >= np.maximum(post, expected)) & (times[index] in positive_events)
            target = np.where(pre_stop, simulation.discounts[index]*pre,
                np.where(post_stop, simulation.discounts[index]*post, target))
        if len(times) > 1:
            continuations[0] = _zero_continuation(factors[:, 0], config)
    initial = float(_payoff(market.spot, k, calls))
    initial_stop = exercise_style == "american" and initial > 0 and initial >= float(target.mean())
    diagnostics = {"training_paths": config.training_paths, "training_seed": config.training_seed,
        "valuation_seed": config.valuation_seed, "independent_training_valuation": True,
        "factor_state_dimension": len(kernel.nodes), "all_factor_coordinates_in_policy": True,
        "regression": "ridge quadratic stock spline with all factor linear/cross/quadratic terms",
        "normalization_source": "training paths only", "exercise_style": exercise_style,
        "training_method": config.training_method,
        "training_labels": ("exact one-step discrete Bellman conditional expectations"
            if config.training_method == "conditional" else "discounted future realized policy cashflows"),
        "bounded_surrogate": bool(config.bounded_surrogate),
        "surrogate_cap_scope": "numerical function stabilization; not a certified weak-law price bound",
        "exercise_dates": exercise_times.tolist() if exercise_style == "american" else [t],
        "training_state_dates": times.tolist(), "initial_training_continuation": float(target.mean()),
        "estimated_memory_bytes": estimated_memory,
        "regression_ranks": [c.rank for c in continuations],
        "regression_conditions": [c.condition for c in continuations],
        "elapsed_seconds": perf_counter()-started,
        "simulation": dict(simulation.diagnostics)}
    return RoughStoppingPolicy(params, market, kernel, k, t, calls, exercise_style,
        exercise_times, times, tuple(continuations), initial_stop, config, diagnostics)


def _coefficients_at(policy, time, factors):
    """Interpolate training-only functions at every integration microstep."""
    times = policy.continuation_times
    if len(times) == 1:
        return policy.continuations[0].coefficients(factors)
    right = min(int(np.searchsorted(times, time, side="left")), len(times)-1)
    if right == 0:
        return policy.continuations[0].coefficients(factors)
    left = right-1
    # The initial regression is not identifiable because the inception state
    # is common. Use the first fitted function on the initial interval.
    if left == 0:
        return policy.continuations[right].coefficients(factors)
    fraction = (time-times[left])/(times[right]-times[left])
    return ((1-fraction)*policy.continuations[left].coefficients(factors)
        +fraction*policy.continuations[right].coefficients(factors))


def _sample_summary(samples):
    if not np.isfinite(samples).all():
        raise RoughSimulationError("rough stopping generated nonfinite payoff/control samples")
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)/math.sqrt(len(samples))


def _surrogate_terms(policy, time, factors, config, eligible):
    """Bridge continuation after the left date to value at the right date.

    Interpolating a continuation coefficient to the terminal zero placeholder
    destroys the payoff anchor. A mixture of two separately integrated
    functions retains that anchor and still has an exact local expectation.
    """
    if config.surrogate_interpolation == "coefficients":
        return [(1., _coefficients_at(policy, time, factors), eligible)]
    times = policy.continuation_times
    right = min(int(np.searchsorted(times, time, side="left")), len(times)-1)
    if right == 0 or time == times[right]:
        return [(1., policy.continuations[right].coefficients(factors), eligible)]
    left = right-1
    fraction = float((time-times[left])/(times[right]-times[left]))
    # The initial constant regression carries no state dependence. Reuse the
    # first identified continuation, as in the historical interpolator.
    previous = policy.continuations[right if left == 0 else left]
    next_exercise = times[right] == policy.maturity or (
        policy.exercise_style == "american" and times[right] in policy.exercise_times)
    return [(1-fraction, previous.coefficients(factors), False),
        (fraction, policy.continuations[right].coefficients(factors), next_exercise)]


def value_rough_stopping_policy(policy, *, config=None, spots=None):
    """Value a frozen policy on independent paths and optionally paired spots.

    This low-level result retains per-path primal/dual samples for paired
    sensitivities and independent replication.  ``spots`` changes inception
    stock only; factor memory, cash amounts, curves and policy stay fixed.
    """
    if not isinstance(policy, RoughStoppingPolicy):
        from .rough_stopping_fourier import FourierStoppingPolicy, value_fourier_stopping_policy
        if isinstance(policy, FourierStoppingPolicy):
            return value_fourier_stopping_policy(policy, config=config, spots=spots)
        raise TypeError("policy must be RoughStoppingPolicy")
    config = policy.config if config is None else config
    if not isinstance(config, RoughAmericanConfig):
        raise TypeError("config must be RoughAmericanConfig")
    if config.valuation_seed == policy.config.training_seed:
        raise ValueError("valuation seed overlaps the policy training stream")
    if config.stock_integral != policy.config.stock_integral:
        raise ValueError("valuation stock-integral scheme must match the trained policy")
    if config.stock_knots != policy.config.stock_knots:
        raise ValueError("valuation spline knots must match the trained policy")
    spots = np.asarray([policy.market.spot] if spots is None else spots)
    if spots.dtype.kind not in "iuf" or spots.ndim != 1 or len(spots) == 0 or not np.isfinite(spots).all() or np.any(spots <= 0):
        raise ValueError("spots must be a finite positive real vector")
    spots = spots.astype(float)
    params, market, kernel = policy.params, policy.market, policy.kernel
    n, count, bumped = config.valuation_paths, len(kernel.nodes), len(spots)
    simulation_config = RoughSimulationConfig(n, config.steps, config.valuation_seed,
        config.batch_size, config.memory_budget_mb, config.stock_integral)
    # Training and valuation share the declared integration/event law even
    # when uniform integration/exercise subdivisions are not nested.  Extra
    # valuation refinements can add dates, but cannot silently drop a date on
    # which the conditional Bellman policy was trained.
    grid = _time_grid(params, market, policy.continuation_times, config.steps)
    # Persistent factors and primal/dual sample arrays plus conservative
    # piecewise-root work arrays in the largest valuation batch.
    memory = 8*(n*(count+18*bumped)+min(n, config.batch_size)
        *(48*count+256*bumped*(len(config.stock_knots)+1)))
    if memory > config.memory_budget_mb*1024**2:
        raise MemoryError(f"rough stopping valuation requires approximately {memory/1024**2:.2f} MiB, exceeding memory_budget_mb={config.memory_budget_mb}")
    initial = _payoff(spots, policy.strike, policy.is_call)
    stock = np.broadcast_to(spots, (n, bumped)).copy()
    factors = np.zeros((n, count))
    martingale = np.zeros_like(stock)
    primal, raw = np.zeros_like(stock), np.zeros_like(stock)
    upper = np.broadcast_to(initial, stock.shape).copy() if policy.exercise_style == "american" else np.full_like(stock, -np.inf)
    alive = np.ones_like(stock, bool)
    stop_time = np.full_like(stock, policy.maturity)
    if policy.maturity == 0 or policy.exercise_at_inception:
        raw[:] = initial
        primal[:] = initial
        alive[:] = False
        stop_time[:] = 0.
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    drift_cache, rng = {}, np.random.default_rng(config.valuation_seed)
    event_map = {e.time:e.amount for e in market.dividends.events if e.time <= policy.maturity and e.amount > 0}
    exercise_set = set(policy.exercise_times[1:])
    for left, right in zip(grid[:-1], grid[1:]):
        dt, theta = float(right-left), _theta(params, float((left+right)/2))
        key = (dt, theta)
        if key not in drift_cache:
            forcing = np.full(count, params.kappa*(theta-params.v0))
            duration = dt if params.vol_of_vol == 0 else dt/2
            drift_cache[key] = _drift_flow_with_integral(matrix, forcing, weights, params.v0, duration)
        exact = drift_cache[key]
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        cash = event_map.get(float(right), 0.)
        discount = market.discount_curve.discount(right)
        eligible_exercise = float(right) in exercise_set and (
            policy.exercise_style == "american" or right == policy.maturity)
        value_cap = _surrogate_cap(market, strike=policy.strike, maturity=policy.maturity,
            time=float(right), is_call=policy.is_call, enabled=config.bounded_surrogate)
        for begin in range(0, n, config.batch_size):
            end = min(n, begin+config.batch_size)
            old = factors[begin:end]
            random = rng.random((end-begin, 3))
            normal = ndtri(np.clip(random[:, 2], np.finfo(float).tiny, 1-np.finfo(float).eps))
            if params.vol_of_vol == 0:
                new = old@exact[:count, :count].T+exact[:count, -1]
                clock = _nonnegative_clock(old@exact[count, :count]+exact[count, -1])
                branch_factors = new[:, None, :]
                means = np.log(np.where(stock[begin:end] > 0, stock[begin:end], 1.))[:, None, None, :]+growth-.5*clock[:, None, None, None]
                clocks = clock[:, None, None, None]
                probability = np.ones((end-begin, 1))
                choice, endpoint = np.zeros(end-begin, int), np.zeros(end-begin, int)
            else:
                branches = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
                    flow=exact[:count, :count], shift=exact[:count, -1], integrated_flow=exact,
                    dt=dt, theta=theta, growth=growth, stock_integral=config.stock_integral)
                branch_factors, probability = branches["new_factors"], branches["probabilities"]
                choice = ((random[:, 0] > probability[:, 0]).astype(int)
                    +(random[:, 0] > probability[:, :2].sum(axis=1)).astype(int))
                endpoint = (random[:, 1] >= .5).astype(int)
                clocks = branches["orthogonal"][..., None]
                with np.errstate(divide="ignore"):
                    means = (np.log(np.where(stock[begin:end] > 0, stock[begin:end], 1.))[:, None, None, :]
                        +branches["correlated"][:, :, None, None]-.5*clocks)
            zero_stock = stock[begin:end] == 0
            branch_values = np.zeros(means.shape)
            for weight, coefficients, exercise in _surrogate_terms(
                    policy, float(right), branch_factors, config, eligible_exercise):
                coefficients = coefficients[:, :, None, None, ...]
                expectation = expected_stopping_value(means, clocks, coefficients,
                    strike=policy.strike, is_call=policy.is_call, cash_dividend=cash,
                    payoff_side="both" if policy.exercise_style == "american" else "post",
                    stock_knots=config.stock_knots, exercise=exercise, value_cap=value_cap)
                if np.any(zero_stock):
                    zero_values = stopping_value(np.zeros_like(means), coefficients,
                        strike=policy.strike, is_call=policy.is_call, cash_dividend=cash,
                        payoff_side="both" if policy.exercise_style == "american" else "post",
                        stock_knots=config.stock_knots, exercise=exercise, value_cap=value_cap)
                    expectation = np.where(zero_stock[:, None, None, :], zero_values, expectation)
                branch_values += weight*expectation
            conditional = np.sum(branch_values.mean(axis=2)*probability[:, :, None], axis=1)
            index = np.arange(end-begin)
            next_mean = means[index, choice, endpoint]
            next_clock = np.broadcast_to(clocks, means.shape)[index, choice, endpoint]
            with np.errstate(over="ignore", invalid="ignore"):
                pre_stock = np.exp(next_mean+np.sqrt(next_clock)*normal[:, None])
                pre_stock = np.where(zero_stock, 0., pre_stock)
            if not np.isfinite(pre_stock).all():
                raise RoughSimulationError("rough stopping stock transition overflowed")
            new = branch_factors[index, choice]
            actual = np.zeros_like(pre_stock)
            for weight, coefficients, exercise in _surrogate_terms(
                    policy, float(right), new, config, eligible_exercise):
                actual += weight*stopping_value(pre_stock, coefficients[:, None, ...],
                    strike=policy.strike, is_call=policy.is_call, cash_dividend=cash,
                    payoff_side="both" if policy.exercise_style == "american" else "post",
                    stock_knots=config.stock_knots, exercise=exercise, value_cap=value_cap)
            martingale[begin:end] += discount*(actual-conditional)
            factors[begin:end] = new
            stock[begin:end] = np.maximum(pre_stock-cash, 0.)
            if float(right) in exercise_set:
                terminal = right == policy.maturity
                eligible = policy.exercise_style == "american" or terminal
                if eligible:
                    post = _payoff(stock[begin:end], policy.strike, policy.is_call)
                    pre = _payoff(pre_stock, policy.strike, policy.is_call)
                    intrinsic = np.maximum(pre, post) if policy.exercise_style == "american" else post
                    upper[begin:end] = np.maximum(upper[begin:end], discount*intrinsic-martingale[begin:end])
                    continuation = policy.continuations[int(np.searchsorted(policy.continuation_times, right))]
                    future = continuation.value(stock[begin:end], new, policy.strike)
                    future = _decision_continuation(future, stock[begin:end], market,
                        strike=policy.strike, maturity=policy.maturity, time=float(right),
                        is_call=policy.is_call, enabled=config.bounded_surrogate)
                    pre_stop = ((pre > 0) & (pre >= np.maximum(post, future)) & (cash > 0)
                        & (policy.exercise_style == "american"))
                    post_stop = (post > 0) & (post >= future)
                    stop = alive[begin:end] & (pre_stop | post_stop | terminal)
                    realized = np.where(pre_stop, pre, post)
                    raw[begin:end] = np.where(stop, discount*realized, raw[begin:end])
                    primal[begin:end] = np.where(stop, discount*realized-martingale[begin:end], primal[begin:end])
                    stop_time[begin:end] = np.where(stop, right, stop_time[begin:end])
                    alive[begin:end] &= ~stop
    if policy.maturity == 0:
        upper[:] = initial
    if np.any(alive):
        raise RoughSimulationError("rough stopping policy left contracts alive after maturity")
    diagnostics = {"backend": "rough_heston_lifted_primal_dual", "exercise_style": policy.exercise_style,
        "bound_scope": "finite kernel, discrete integration law, declared exercise dates only",
        "martingale_centering": "exact six-branch and piecewise-quadratic truncated-lognormal expectation",
        "martingale_every_integration_step": True, "actual_time_steps": len(grid)-1,
        "training_seed": policy.config.training_seed, "valuation_seed": config.valuation_seed,
        "independent_training_valuation": True, "frozen_policy": True,
        "estimated_memory_bytes": memory, "time_grid": grid.tolist(),
        "exercise_dates": policy.exercise_times.tolist() if policy.exercise_style == "american" else [policy.maturity],
        "training_state_dates": policy.continuation_times.tolist(), "spots": spots.tolist(),
        "training_method": policy.config.training_method,
        "surrogate_interpolation": config.surrogate_interpolation,
        "bounded_surrogate": bool(config.bounded_surrogate),
        "american_pricing_ready": False, "stock_delta_ready": False,
        "readiness_reason": "time, exercise, kernel, training and delta refinements require independent checks"}
    for array in (primal, raw, upper, stop_time):
        array.setflags(write=False)
    return {"primal_samples": primal, "raw_samples": raw, "dual_samples": upper,
        "stop_times": stop_time, "diagnostics": diagnostics}


def estimate_rough_american(params, market, kernel, *, strike, maturity, is_call=True,
                             config=None, exercise_style="american", conditional_state=None):
    """Price and paired stock deltas without reusing policy training paths.

    ``exercise_style='european'`` is a terminal-only stopping control for
    numerical validation, including declared cash dividends. Public Bermudan
    contracts and historical conditional rough states are not implemented.
    Contracts retain supplied broadcasting/order; seeds are recorded rather
    than silently changed when processing another contract.
    """
    started = perf_counter()
    config = RoughAmericanConfig() if config is None else config
    strikes, maturities, calls = _inputs(params, market, kernel, strike, maturity,
        is_call, config, exercise_style, conditional_state)
    shape = strikes.shape
    fields = {name: [] for name in ("prices", "standard_errors", "dual_prices", "dual_standard_errors",
        "dual_gaps", "dual_gap_standard_errors", "deltas", "delta_standard_errors",
        "one_sided_deltas", "one_sided_delta_standard_errors", "bump_deltas",
        "bump_delta_standard_errors", "raw_prices", "raw_standard_errors", "exercise_probabilities",
        "retrained_deltas", "retrained_delta_standard_errors")}
    policies, contract_diagnostics = [], []
    dual_training_evaluations = 0
    bump = market.spot*config.bump_relative
    spots = market.spot+np.array([-bump, -bump/2, 0., bump/2, bump])
    for k, t, call in zip(strikes.flat, maturities.flat, calls.flat):
        policy = train_rough_stopping_policy(params, market, kernel, strike=float(k),
            maturity=float(t), is_call=bool(call), config=config, exercise_style=exercise_style)
        dual_fit = None
        if config.dual_config is not None and t > 0:
            from .rough_dual import train_rough_dual_weights, value_rough_dual_policy
            dual_fit = train_rough_dual_weights(policy, config=config.dual_config)
            dual_training_evaluations += 1
            value = value_rough_dual_policy(policy, dual_fit, spots=spots)
        else:
            value = value_rough_stopping_policy(policy, spots=spots)
        primal, dual, raw = (value[name] for name in ("primal_samples", "dual_samples", "raw_samples"))
        fields["prices"].append(_sample_summary(primal[:, 2])[0])
        fields["standard_errors"].append(_sample_summary(primal[:, 2])[1])
        fields["dual_prices"].append(_sample_summary(dual[:, 2])[0])
        fields["dual_standard_errors"].append(_sample_summary(dual[:, 2])[1])
        gap = dual[:, 2]-primal[:, 2]
        fields["dual_gaps"].append(_sample_summary(gap)[0])
        fields["dual_gap_standard_errors"].append(_sample_summary(gap)[1])
        delta = np.column_stack(((primal[:, 4]-primal[:, 0])/(2*bump),
            (primal[:, 3]-primal[:, 1])/bump))
        means, errors = _sample_summary(delta)
        fields["bump_deltas"].append(means)
        fields["bump_delta_standard_errors"].append(errors)
        fields["deltas"].append(means[1])
        fields["delta_standard_errors"].append(errors[1])
        sided = np.column_stack(((primal[:, 2]-primal[:, 1])/(bump/2),
            (primal[:, 3]-primal[:, 2])/(bump/2)))
        means, errors = _sample_summary(sided)
        fields["one_sided_deltas"].append(means)
        fields["one_sided_delta_standard_errors"].append(errors)
        fields["raw_prices"].append(_sample_summary(raw[:, 2])[0])
        fields["raw_standard_errors"].append(_sample_summary(raw[:, 2])[1])
        fields["exercise_probabilities"].append(float(np.mean(value["stop_times"][:, 2] < t)))
        retrained_mean, retrained_error = math.nan, math.nan
        retrained_dual_diagnostics = []
        if config.retrain_bumps:
            samples = []
            for spot in (spots[1], spots[3]):
                bumped_market = replace(market, spot=float(spot))
                bumped_policy = train_rough_stopping_policy(params, bumped_market, kernel,
                    strike=float(k), maturity=float(t), is_call=bool(call), config=config,
                    exercise_style=exercise_style)
                if config.dual_config is not None and t > 0:
                    bumped_fit = train_rough_dual_weights(bumped_policy, config=config.dual_config)
                    dual_training_evaluations += 1
                    bumped_value = value_rough_dual_policy(bumped_policy, bumped_fit)
                    retrained_dual_diagnostics.append({"spot": float(spot),
                        "dual_training": dict(bumped_fit.diagnostics)})
                else:
                    bumped_value = value_rough_stopping_policy(bumped_policy)
                samples.append(bumped_value["primal_samples"][:, 0])
            retrained_mean, retrained_error = _sample_summary((samples[1]-samples[0])/bump)
        fields["retrained_deltas"].append(retrained_mean)
        fields["retrained_delta_standard_errors"].append(retrained_error)
        policies.append(policy)
        contract_diagnostics.append({"strike": float(k), "maturity": float(t), "is_call": bool(call),
            "training": dict(policy.diagnostics), "valuation": value["diagnostics"],
            "dual_training": None if dual_fit is None else dict(dual_fit.diagnostics),
            "retrained_bump_dual_training": retrained_dual_diagnostics})
    values = {}
    for name, data in fields.items():
        trailing = (2,) if name in ("one_sided_deltas", "one_sided_delta_standard_errors",
            "bump_deltas", "bump_delta_standard_errors") else ()
        values[name] = np.asarray(data).reshape(shape+trailing)
    diagnostics = {"backend": "rough_heston_lifted_primal_dual", "exercise_style": exercise_style,
        "model_clock": "inception", "kernel_fingerprint": kernel.fingerprint,
        "independent_training_valuation": True, "factor_state_dimension": len(kernel.nodes),
        "training_seed": config.training_seed, "valuation_seed": config.valuation_seed,
        "delta_method": "paired central bumps of frozen policy, parameters/initial factors/curves/cash fixed",
        "bump_sizes": [bump, bump/2], "one_sided_order": ["left", "right"],
        "retrained_bump_check_executed": bool(config.retrain_bumps),
        "dual_training_evaluations": dual_training_evaluations,
        "martingale_selection_source": "independent dual-training paths only" if config.dual_config else "unit weights",
        "bound_scope": "finite kernel and discrete transition/exercise law; not continuous American bounds",
        "price_absolute_tolerance": .001, "price_relative_tolerance": 1e-4,
        "delta_absolute_tolerance": .002, "delta_relative_tolerance": .005,
        "spread_refinement_fraction": .1, "american_pricing_ready": False,
        "stock_delta_ready": False, "contracts": contract_diagnostics,
        "elapsed_seconds": perf_counter()-started}
    return RoughAmericanEstimate(**values, config=config, policies=tuple(policies), diagnostics=diagnostics)


__all__ = ["RoughAmericanConfig", "RoughAmericanEstimate", "RoughStoppingPolicy",
    "train_rough_stopping_policy", "value_rough_stopping_policy", "estimate_rough_american"]
