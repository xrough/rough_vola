"""Weak finite-factor rough-Heston simulation from model inception.

This is a dynamics foundation, not an American pricing backend.  We implement
the exact-drift / three-point diffusion Strang split and randomized stock
leapfrog in Bayer--Breneis, arXiv:2310.04146, Section 2. The default correlated
stock update also splits its generator: exact deterministic variance integrals
in the drift half-steps and the common diffusion-factor jump. This avoids
reconstructing a Brownian integral by subtracting a trapezoidal approximation
to stiff factor drift. The published first-factor/trapezoidal update remains
an explicit comparator. Second-order convergence for the complete scheme is
not asserted here; independent time refinement is required.

Factors use U(0)=0 and V=v0+sum(w*U).  Individual U components can be negative.
The affine state cone described in arXiv:2412.17526 is checked rather than
clipping factors.  Only floating point roundoff in a nonnegative aggregate is
rounded to zero, counted explicitly; material violations raise an error.

The correlated-stock representation divides a difference of factor states by
vol_of_vol. At positive horizons we support exact zero or vol_of_vol >= 1e-8;
smaller positive values raise rather than lose noise increments to cancellation
or silently substitute the deterministic model. This numerical restriction is
separate from model admissibility and from the unchanged price tolerances.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr, ndtri

from .rough_lift import ExponentialKernel
from .types import PricingError, RoughHestonParams
from .variance_curve import RoughHestonCurveParams
from .vanilla_types import SpotMarket, UnsupportedModelError


_MIN_POSITIVE_VOL_OF_VOL = 1e-8


class RoughSimulationError(PricingError):
    """A weak simulation step violated its numerical/admissibility contract."""


@dataclass(frozen=True)
class RoughSimulationConfig:
    """Explicit budgets; ``steps`` counts uniform steps before adding events.

    Memory is conservatively estimated before allocating saved paths. The
    random stream is indexed in time/path order, independent of batch size.
    Changing path count, steps or observation dates changes that stream's
    interpretation. Path prefixes are not invariant to path count changes.
    No antithetic or quasi-Monte-Carlo independence assumptions are hidden.
    """
    paths: int = 32768
    steps: int = 256
    seed: int = 42
    batch_size: int = 4096
    memory_budget_mb: float = 512.
    stock_integral: str = "exact_drift"

    def __post_init__(self):
        for name in ("paths", "steps", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (isinstance(self.memory_budget_mb, (bool, np.bool_))
                or not isinstance(self.memory_budget_mb, Real)
                or not math.isfinite(self.memory_budget_mb) or self.memory_budget_mb <= 0):
            raise ValueError("memory_budget_mb must be finite and positive")
        if not isinstance(self.stock_integral, str) or self.stock_integral not in ("exact_drift", "trapezoidal"):
            raise ValueError("stock_integral must be 'exact_drift' or 'trapezoidal'")


@dataclass(frozen=True)
class RoughSimulationResult:
    """Saved post-event states; axes are (path, observation[, factor]).

    ``discounted_gains`` includes the actual limited-liability cash payout.
    Conditional log-stock means/variances integrate out the independent stock
    Brownian motion only when no positive cash dividend occurs. They are not
    available for cash-jump paths, where that simplification is invalid.
    """
    times: np.ndarray
    spots: np.ndarray
    variances: np.ndarray
    factors: np.ndarray
    discounted_gains: np.ndarray
    discounts: np.ndarray
    conditional_log_spots: np.ndarray | None
    conditional_variances: np.ndarray | None
    config: RoughSimulationConfig
    diagnostics: object

    def __post_init__(self):
        for name in ("times", "spots", "variances", "factors", "discounted_gains", "discounts",
                     "conditional_log_spots", "conditional_variances"):
            array = getattr(self, name)
            if array is not None:
                array.setflags(write=False)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def settings(self):
        return {"backend": "finite_factor_weak_simulation", "exercise": "none",
                **asdict(self.config)}


def _params(params):
    if not isinstance(params, (RoughHestonParams, RoughHestonCurveParams)):
        raise TypeError("params must be RoughHestonParams or RoughHestonCurveParams")


def _theta(params, time):
    return float(params.curve(time)) if isinstance(params, RoughHestonCurveParams) else params.theta


def _cone_translation(v0, nodes, weights):
    """Reference initial vector with diag(x)*reference a nonnegative constant.

    A zero first node is handled by its continuous limiting allocation. The
    cone rows are partial weighted sums minus the following factor, followed
    by the total weighted sum; see Theorem 2.2 / Corollary 2.7 of 2412.17526.
    """
    if nodes[0] == 0:
        initial = np.zeros_like(nodes)
        initial[0] = v0 / weights[0]
    else:
        initial = (v0 / np.sum(weights / nodes)) / nodes
    q = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)-1):
        q[i, :i+1] = weights[:i+1]
        q[i, i+1] = -np.sum(weights[:i+1])
    q[-1] = weights
    return initial, q


def _affine_flow(matrix, forcing, duration):
    n = len(forcing)
    augmented = np.zeros((n+1, n+1))
    augmented[:n, :n] = matrix
    augmented[:n, n] = forcing
    flow = expm(augmented * duration)
    if not np.isfinite(flow).all():
        raise RoughSimulationError("exact drift matrix exponential produced nonfinite values")
    return flow[:n, :n], flow[:n, n]


def _drift_flow_with_integral(matrix, forcing, weights, v0, duration):
    """Exact affine factor drift and its integrated aggregate variance.

    The extra row accumulates ``integral(v0 + weights.U)``. Keeping a constant
    coordinate avoids inversion of a possibly singular drift matrix.
    """
    count = len(forcing)
    augmented = np.zeros((count+2, count+2))
    augmented[:count, :count] = matrix
    augmented[:count, -1] = forcing
    augmented[count, :count], augmented[count, -1] = weights, v0
    result = expm(augmented*duration)
    if not np.isfinite(result).all():
        raise RoughSimulationError("exact drift/integral matrix exponential produced nonfinite values")
    return result


def _nonnegative_clock(values):
    values = np.asarray(values, float)
    tolerance = 256*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
    if not np.isfinite(values).all() or np.min(values) < -tolerance:
        raise RoughSimulationError("exact drift integrated variance is materially negative or nonfinite")
    return np.maximum(values, 0.)


def _weak_transition_branches(old, params, *, nodes, weights, flow, shift,
                              integrated_flow, dt, theta, growth, stock_integral):
    """Enumerate the exact discrete law used by one stochastic simulation step.

    The three diffusion outcomes have ``probabilities``; each has two equally
    likely independent-stock endpoint choices. ``correlated`` includes rate
    growth and correlated log-stock drift. ``orthogonal`` is the nonnegative
    variance of the analytically integrable independent Gaussian stock driver.
    Arrays have leading ``(path, diffusion_branch[, endpoint])`` axes. This
    shared law also permits controls centered by the *discrete* transition,
    without assuming that discretized stock is an exact martingale.

    For ``exact_drift``, split A=(affine factor drift, -.5*rho**2*V log drift)
    and D=(eta*sqrt(V)*1, rho*sqrt(V) log noise). A integrates V exactly; D has
    log increment rho/eta times its common factor jump. Their Ito generators,
    together with the independent B stock generator, sum to the target model.
    """
    if stock_integral not in ("exact_drift", "trapezoidal"):
        raise ValueError("unknown stock integral construction")
    if params.vol_of_vol <= 0:
        raise ValueError("weak transition branches require positive vol_of_vol")
    old = np.asarray(old, float)
    count = len(weights)
    total_weight = float(np.sum(weights))
    translation, cone = _cone_translation(params.v0, nodes, weights)
    branch_diagnostics = {"minimum_cone_coordinate":np.inf,
        "minimum_raw_variance":np.inf, "aggregate_roundoff_count":0}
    before = _check_state(old, params.v0, weights, translation, cone, branch_diagnostics,
        stage="branch step start", time=0.)
    half = old@flow.T+shift
    mid = _check_state(half, params.v0, weights, translation, cone, branch_diagnostics,
        stage="branch first drift", time=dt/2)
    support, probabilities = _moment_support(mid, params.vol_of_vol**2*total_weight**2*dt)
    diffusion_jump = (support-mid[:, None])/total_weight
    diffused = half[:, None, :]+diffusion_jump[:, :, None]
    new = diffused@flow.T+shift
    # Every branch belongs to the executable transition law, including the
    # branches not selected by a particular uniform draw. Exact-law controls
    # must never center over an invalid but merely unselected outcome.
    after = _check_state(new.reshape(-1, count), params.v0, weights, translation, cone,
        branch_diagnostics, stage="enumerated branch second drift", time=dt).reshape(len(old), 3)
    if stock_integral == "exact_drift":
        first = _nonnegative_clock(old@integrated_flow[count, :count]+integrated_flow[count, -1])
        second = _nonnegative_clock(diffused@integrated_flow[count, :count]+integrated_flow[count, -1])
        clock = first[:, None]+second
        correlated = growth+params.rho/params.vol_of_vol*diffusion_jump-.5*params.rho**2*clock
    else:
        clock = .5*(before[:, None]+after)*dt
        first_integral = .5*(old[:, None, 0]+new[:, :, 0])*dt
        correlated = growth+params.rho/params.vol_of_vol*(-params.kappa*theta*dt
            +nodes[0]*first_integral+(params.kappa-.5*params.rho*params.vol_of_vol)*clock
            +(new[:, :, 0]-old[:, None, 0]))
    orthogonal = ((1-params.rho**2)*dt
        *np.stack((np.broadcast_to(before[:, None], after.shape), after), axis=-1))
    if not np.isfinite(new).all() or not np.isfinite(correlated).all() or not np.isfinite(orthogonal).all():
        raise RoughSimulationError("weak transition branches contain nonfinite values")
    return {"new_factors":new, "probabilities":probabilities,
        "correlated":correlated, "orthogonal":orthogonal,
        "before":before, "mid":mid, "after":after, "half_factors":half,
        "correlated_clock":clock, "branch_state_diagnostics":branch_diagnostics}


def _moment_support(x, z):
    """Published three-point square-root weak transition rule.

    Scaling and rationalization retain the small first support point near
    zero. This implements the LM21 rule reproduced in 2310.04146, not a
    Gaussian Euler update. The first four exact moments are reproduced; the
    fifth-moment discrepancy starts at O(h**3). Although Section 2 of the
    cited paper describes five exact moments, direct substitution of its
    stated support/probabilities gives the latter result (tested explicitly).
    The absorbing x=0 case is explicit.
    """
    x = np.asarray(x, float)
    if np.any(x < 0) or not np.isfinite(x).all() or not np.isfinite(z) or z < 0:
        raise RoughSimulationError("moment matching requires finite nonnegative variance and scale")
    if z == 0:
        return np.stack((x, x, x), axis=-1), np.stack((np.ones_like(x), np.zeros_like(x), np.zeros_like(x)), axis=-1)
    a = (6. + math.sqrt(3.)) / 4.
    scale = np.maximum(x, z)
    y, zz = x/scale, z/scale
    root = np.sqrt((3*y + a*a*zz)*zz)
    s1 = (y*y + (2*a-3)*y*zz) / (y+a*zz+root)
    s2, s3 = y+(a-.75)*zz, y+a*zz+root
    support = np.stack((s1, s2, s3), axis=-1)
    deltas = np.stack((a*zz-root,(a-.75)*zz,a*zz+root),axis=-1)
    probabilities = np.zeros_like(support)
    active = x > 0
    for j, (k, ell) in enumerate(((1,2), (0,2), (0,1))):
        dj, dk, dl = deltas[active,j],deltas[active,k],deltas[active,ell]
        # The published raw-moment numerator subtracts O(x**3) terms to
        # obtain O(x**2*z), losing precision as z/x -> 0. Centering around x
        # gives the identical polynomial without that cancellation.
        ratio = ((y+a*zz+root)/(y+(2*a-3)*zz))[active] if j == 0 else y[active]/support[active,j]
        probabilities[active,j] = ratio*(dk*dl+y[active]*zz[active]
            +1.5*zz[active]**2-zz[active]*(dk+dl))/((dj-dk)*(dj-dl))
    probabilities[~active, 0] = 1.
    # Only roundoff is allowed; probabilities outside this envelope are an
    # identifiable failure rather than silently altered transition laws.
    if (not np.isfinite(probabilities).all() or np.min(probabilities) < -2e-10
            or np.max(probabilities) > 1+2e-10
            or np.max(np.abs(np.sum(probabilities, axis=-1)-1)) > 2e-9):
        raise RoughSimulationError("three-point moment probabilities are numerically invalid")
    probabilities = np.clip(probabilities, 0., 1.)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return support*scale[:, None], probabilities


def _check_state(u, v0, weights, translation, cone, diagnostics, *, stage, time):
    values = u + translation
    coordinates = values @ cone.T
    scale = max(1., float(np.max(np.abs(values))) * float(np.sum(weights)))
    tolerance = 256*np.finfo(float).eps*scale
    minimum = float(np.min(coordinates))
    diagnostics["minimum_cone_coordinate"] = min(diagnostics["minimum_cone_coordinate"], minimum)
    if not np.isfinite(coordinates).all() or minimum < -tolerance:
        raise RoughSimulationError(f"admissible-cone violation during {stage} at t={time:.17g}: minimum={minimum:.6g}")
    variance = v0 + np.sum(u*weights, axis=-1)
    minimum = float(np.min(variance))
    diagnostics["minimum_raw_variance"] = min(diagnostics["minimum_raw_variance"], minimum)
    if not np.isfinite(variance).all() or minimum < -tolerance:
        raise RoughSimulationError(f"negative aggregate variance during {stage} at t={time:.17g}: minimum={minimum:.6g}")
    diagnostics["aggregate_roundoff_count"] += int(np.count_nonzero(variance < 0))
    return np.maximum(variance, 0.)


def _time_grid(params, market, times, steps):
    horizon = float(times[-1])
    if horizon == 0:
        return np.array([0.])
    extra = list(times) + [t for t in market.discount_curve.times if 0 < t < horizon]
    extra += [event.time for event in market.dividends.events if event.time <= horizon]
    if isinstance(params, RoughHestonCurveParams):
        extra += [t for t in params.curve.knot_times if t < horizon]
    return np.unique(np.r_[np.linspace(0., horizon, steps+1), extra])


def simulate_rough_heston(params, market, kernel, *, observation_times, config=None,
                          conditional_state=None):
    """Simulate the finite-lift process at explicitly requested dates.

    Only inception is supported. Passing a historical/conditional state raises
    instead of silently resetting volatility memory. Dividend observation
    states are post-jump; volatility factors are continuous through the jump.
    """
    started = perf_counter()
    _params(params)
    if not isinstance(market, SpotMarket):
        raise TypeError("market must be SpotMarket with explicit dividends")
    if not isinstance(kernel, ExponentialKernel):
        raise TypeError("kernel must be ExponentialKernel")
    if conditional_state is not None:
        raise UnsupportedModelError("conditional rough-Heston factor-state simulation is not implemented; inception cannot be reset implicitly")
    config = RoughSimulationConfig() if config is None else config
    if not isinstance(config, RoughSimulationConfig):
        raise TypeError("config must be RoughSimulationConfig")
    try:
        times = np.asarray(observation_times)
    except (TypeError,ValueError) as exc:
        raise ValueError("observation_times must be a nonempty strictly increasing finite nonnegative real sequence") from exc
    if times.dtype.kind not in "iuf" or times.ndim != 1 or not len(times):
        raise ValueError("observation_times must be a nonempty strictly increasing finite nonnegative real sequence")
    times = times.astype(float)
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("observation_times must be a nonempty strictly increasing finite nonnegative real sequence")
    horizon = float(times[-1])
    if horizon > 0 and 0 < params.vol_of_vol < _MIN_POSITIVE_VOL_OF_VOL:
        raise RoughSimulationError(
            "positive vol_of_vol below 1e-8 is unsupported by the correlated-stock "
            "representation: factor-increment cancellation is amplified by division; "
            "use exact zero only when the model is deterministic")
    if horizon > market.dividends.coverage + 1e-12:
        raise ValueError("dividend expectations do not cover all observation times")
    market.discount_curve.discount(times)
    if kernel.horizon is not None and horizon > kernel.horizon + 1e-12:
        raise ValueError("observation times exceed the kernel approximation horizon")
    if kernel.hurst is not None and abs(kernel.hurst-params.hurst) > 1e-14:
        raise ValueError("kernel Hurst parameter does not match model")
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    n, count = config.paths, len(nodes)
    positive_events = [e for e in market.dividends.events if e.time <= horizon and e.amount > 0]
    conditional = not positive_events
    # Saved: factors + S,V,gain + two conditional arrays; work: factors and a
    # conservative 48 vectors in the largest batch, plus full path vectors.
    grid = _time_grid(params, market, times, config.steps)
    estimated = 8*(n*len(times)*(count+3+2*conditional)+n*(count+6)
        +min(n,config.batch_size)*(24*count+128)+len(grid)*(count+2)**2*3)
    if estimated > config.memory_budget_mb*1024**2:
        raise MemoryError(f"simulation requires approximately {estimated/1024**2:.2f} MiB, exceeding memory_budget_mb={config.memory_budget_mb}")
    output = {name: np.empty((n, len(times))) for name in ("spots", "variances", "discounted_gains")}
    saved_factors = np.empty((n, len(times), count))
    cm = np.empty((n,len(times))) if conditional else None
    cq = np.empty((n,len(times))) if conditional else None
    u = np.zeros((n,count))
    stock = np.full(n, market.spot)
    gain_cash = np.zeros(n)
    log_cond = np.full(n, math.log(market.spot))
    cond_var = np.zeros(n)
    translation, cone = _cone_translation(params.v0,nodes,weights)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count,1))*weights
    drift_cache = {}
    diagnostics = {"backend": "finite_factor_weak_simulation", "exercise": "none",
                   "model_clock": "inception", "kernel_fingerprint": kernel.fingerprint,
                   "factor_count": count, "actual_time_steps": len(grid)-1,
                   "time_grid": grid.tolist(), "saved_time_count": len(times),
                   "estimated_memory_bytes": estimated, "minimum_raw_variance": params.v0,
                   "minimum_cone_coordinate": float(np.min(translation@cone.T)),
                   "aggregate_roundoff_count": 0, "individual_factor_clipping": False,
                   "conditional_independent_stock_integration": conditional,
                   "batch_size_invariant_random_stream": True,
                   "minimum_supported_positive_vol_of_vol": _MIN_POSITIVE_VOL_OF_VOL,
                   "stock_integral": config.stock_integral,
                   "discrete_transition_branches": 6,
                   "weak_order_claim": "not certified; refine time independently",
                   "source": "https://arxiv.org/abs/2310.04146", "dividend_events": []}
    rng = np.random.default_rng(config.seed)
    obs_index = 0
    def save(index):
        output["spots"][:,index] = stock
        output["variances"][:,index] = np.maximum(params.v0+np.sum(u*weights,axis=-1),0.)
        saved_factors[:,index] = u
        output["discounted_gains"][:,index] = market.discount_curve.discount(times[index])*stock+gain_cash
        if conditional:
            cm[:,index], cq[:,index] = log_cond,cond_var
    if times[0] == 0:
        save(0)
        obs_index += 1
    event_map = {e.time:e for e in positive_events}
    for left,right in zip(grid[:-1],grid[1:]):
        dt = float(right-left)
        theta = _theta(params, float((left+right)/2))
        key = (dt,theta)
        if key not in drift_cache:
            forcing = np.full(count,params.kappa*(theta-params.v0))
            half_integral_flow = _drift_flow_with_integral(matrix,forcing,weights,params.v0,dt/2)
            flow,shift = half_integral_flow[:count,:count],half_integral_flow[:count,-1]
            if params.vol_of_vol == 0:
                # Augment by exact integral of v0+w.U for deterministic
                # variance; an averaged-volatility American substitution is
                # intentionally not exposed by this dynamics function.
                full = _drift_flow_with_integral(matrix,forcing,weights,params.v0,dt)
                drift_cache[key] = (flow,shift,full)
            else:
                drift_cache[key] = (flow,shift,half_integral_flow)
        flow,shift,deterministic_flow = drift_cache[key]
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        for begin in range(0,n,config.batch_size):
            end = min(n,begin+config.batch_size)
            old = u[begin:end].copy()
            # Fixed three uniforms per time/path also in deterministic limits
            # preserve the same time/path random stream across batch sizes.
            random = rng.random((end-begin,3))
            gaussian = ndtri(np.clip(random[:,2],np.finfo(float).tiny,1-np.finfo(float).eps))
            before = _check_state(old,params.v0,weights,translation,cone,diagnostics,stage="step start",time=left)
            if params.vol_of_vol == 0:
                new = old@deterministic_flow[:count,:count].T+deterministic_flow[:count,-1]
                integrated = old@deterministic_flow[count,:count]+deterministic_flow[count,-1]
                if np.min(integrated) < -1e-12 or not np.isfinite(integrated).all():
                    raise RoughSimulationError("deterministic integrated variance is invalid")
                integrated = np.maximum(integrated,0.)
                _check_state(new,params.v0,weights,translation,cone,diagnostics,stage="deterministic drift",time=right)
                log_step = growth-.5*integrated+np.sqrt(integrated)*gaussian
                correlated = growth
                orthogonal = integrated
            else:
                branches = _weak_transition_branches(old,params,nodes=nodes,weights=weights,
                    flow=flow,shift=shift,integrated_flow=deterministic_flow,dt=dt,theta=theta,
                    growth=growth,stock_integral=config.stock_integral)
                _check_state(branches["half_factors"],params.v0,weights,translation,cone,diagnostics,
                    stage="first drift",time=left+dt/2)
                probabilities = branches["probabilities"]
                choice = (random[:,0]>probabilities[:,0]).astype(int)+(random[:,0]>(probabilities[:,0]+probabilities[:,1])).astype(int)
                batch_index = np.arange(end-begin)
                new = branches["new_factors"][batch_index,choice]
                _check_state(new,params.v0,weights,translation,cone,diagnostics,stage="second drift",time=right)
                correlated = branches["correlated"][batch_index,choice]
                endpoint = (random[:,1]>=.5).astype(int)
                orthogonal = branches["orthogonal"][batch_index,choice,endpoint]
                log_step = correlated-.5*orthogonal+np.sqrt(orthogonal)*gaussian
            u[begin:end] = new
            with np.errstate(over="ignore",invalid="ignore"):
                stock[begin:end] *= np.exp(log_step)
            if not np.isfinite(stock[begin:end]).all():
                raise RoughSimulationError(f"nonfinite stock simulation at t={right:.17g}")
            log_cond[begin:end] += correlated-.5*orthogonal
            cond_var[begin:end] += orthogonal
        if right in event_map:
            event = event_map[right]
            payout = np.minimum(event.amount,stock)
            before_mean = float(np.mean(stock))
            gain_cash += market.discount_curve.discount(right)*payout
            stock -= payout
            diagnostics["dividend_events"].append({"time":right,"declared_amount":event.amount,
                "mean_pre_stock":before_mean,"mean_payout":float(np.mean(payout)),
                "zero_stock_count":int(np.count_nonzero(stock==0))})
        if obs_index < len(times) and right == times[obs_index]:
            save(obs_index)
            obs_index += 1
    diagnostics["elapsed_seconds"] = perf_counter()-started
    diagnostics["minimum_saved_factor"] = float(np.min(saved_factors))
    diagnostics["discounted_gain_means"] = np.mean(output["discounted_gains"],axis=0).tolist()
    diagnostics["discounted_gain_standard_errors"] = (np.std(output["discounted_gains"],axis=0,ddof=1)/math.sqrt(n)).tolist() if n>1 else [None]*len(times)
    return RoughSimulationResult(times.copy(),output["spots"],output["variances"],saved_factors,
        output["discounted_gains"],np.asarray(market.discount_curve.discount(times)),cm,cq,config,diagnostics)


def european_payoff_samples(result, *, strike, maturity, is_call=True, conditional=True):
    """Discounted European payoff samples, retaining the path axis first.

    Strikes/option types broadcast. Maturity must be one saved observation.
    Conditional samples exactly integrate the independent Gaussian stock
    driver for a no-cash path; this removes sampling noise, not time/kernel
    error. With cash dividends request raw samples explicitly.
    """
    if not isinstance(result,RoughSimulationResult):
        raise TypeError("result must be RoughSimulationResult")
    if not isinstance(conditional,(bool,np.bool_)):
        raise ValueError("conditional must be boolean")
    if isinstance(maturity,(bool,np.bool_)) or not isinstance(maturity,Real) or not math.isfinite(maturity):
        raise ValueError("maturity must be finite and equal a saved observation")
    index = np.flatnonzero(result.times == maturity)
    if len(index) != 1:
        raise ValueError("maturity must equal a saved observation time")
    try:
        k,call = np.broadcast_arrays(np.asarray(strike),np.asarray(is_call))
    except (TypeError,ValueError) as exc:
        raise ValueError("strikes and is_call must be broadcastable numeric/boolean inputs") from exc
    if k.dtype.kind not in "iuf" or not np.isfinite(k).all() or np.any(k<=0) or call.dtype.kind != "b":
        raise ValueError("strikes must be positive and finite; is_call must contain booleans")
    k = k.astype(float)
    index = int(index[0])
    expand = (slice(None),)+(None,)*k.ndim
    if maturity == 0:
        stock = result.spots[:,index][expand]
        return np.maximum(np.where(call,stock-k,k-stock),0.)
    if conditional:
        if result.conditional_log_spots is None:
            raise ValueError("conditional European estimator is unavailable with positive cash dividends")
        mean = result.conditional_log_spots[:,index][expand]
        variance = result.conditional_variances[:,index][expand]
        std = np.sqrt(variance)
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            forward = np.exp(mean+.5*variance)
            d2 = np.divide(mean-np.log(k),std,out=np.zeros(np.broadcast_shapes(mean.shape,k.shape)),where=std>0)
            d1 = d2+std
            calls = forward*ndtr(d1)-k*ndtr(d2)
            puts = k*ndtr(-d2)-forward*ndtr(-d1)
            values = np.where(call,calls,puts)
            intrinsic = np.maximum(np.where(call,forward-k,k-forward),0.)
            values = np.where(variance==0,intrinsic,values)
    else:
        stock = result.spots[:,index][expand]
        values = np.maximum(np.where(call,stock-k,k-stock),0.)
    if not np.isfinite(values).all():
        raise RoughSimulationError("European payoff samples are nonfinite")
    return values*result.discounts[index]


__all__ = ["RoughSimulationConfig","RoughSimulationResult","RoughSimulationError",
           "simulate_rough_heston","european_payoff_samples"]
