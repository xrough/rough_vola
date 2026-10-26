"""Experimental European-baseline controls for American finite-lift stopping.

The affine Fourier function is a variance-reduction proxy, never a price used to
center a control. Every increment is centered under the actual enumerable weak
transition and its Gaussian stock law. Training and valuation streams are
independent. Cash dividends are deliberately unsupported in this version.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import hashlib
from numbers import Real
from types import MappingProxyType
import math
from time import perf_counter
import numpy as np
from scipy.special import ndtri, expit
from .rough_estimators import _AffineSurrogate, RoughEuropeanControlConfig, _conditional_payoffs
from .rough_simulation import (_time_grid, _theta, _drift_flow_with_integral,
    _weak_transition_branches, _nonnegative_clock, RoughSimulationError, _MIN_POSITIVE_VOL_OF_VOL)
from .rough_stopping import (_inputs, RoughAmericanConfig, _payoff,
    _fit_continuation, _zero_continuation, _sample_summary)
from .rough_stopping_controls import stopping_value, expected_stopping_value
from .vanilla_types import UnsupportedModelError


@dataclass(frozen=True)
class FourierStoppingPolicy:
    params: object
    market: object
    kernel: object
    strike: float
    maturity: float
    is_call: bool
    exercise_style: str
    exercise_times: np.ndarray
    continuation_times: np.ndarray
    continuations: tuple
    value_premiums: tuple
    config: RoughAmericanConfig
    control_config: RoughEuropeanControlConfig
    exercise_at_inception: bool
    diagnostics: dict
    premium_cap: float | None = None

    def __post_init__(self):
        for name in ("exercise_times", "continuation_times"):
            array=np.asarray(getattr(self,name),float).copy()
            array.setflags(write=False)
            object.__setattr__(self,name,array)
        object.__setattr__(self,"diagnostics",MappingProxyType(dict(self.diagnostics)))


def fourier_policy_fingerprint(policy):
    """Identity of the frozen decision/control functions, excluding path traces."""
    digest=hashlib.sha256()
    for value in (policy.params,policy.market,policy.kernel,policy.config,
            policy.control_config,policy.exercise_at_inception,policy.premium_cap):
        digest.update(repr(value).encode())
    for times in (policy.exercise_times,policy.continuation_times):
        digest.update(np.asarray(times,float).tobytes())
    for continuation in policy.continuations+policy.value_premiums:
        digest.update(type(continuation).__qualname__.encode())
        for field in fields(continuation):
            value=getattr(continuation,field.name)
            digest.update(field.name.encode())
            if isinstance(value,np.ndarray):
                digest.update(str(value.shape).encode());digest.update(value.dtype.str.encode())
                digest.update(value.tobytes())
            else:digest.update(repr(value).encode())
    return digest.hexdigest()


def _validate_cap(cap):
    if cap is not None and (isinstance(cap,(bool,np.bool_)) or not isinstance(cap,Real)
            or not np.isfinite(cap) or cap<=0):
        raise ValueError("premium_cap must be None or a finite positive real")
    return None if cap is None else float(cap)


def _clip_premium(values, cap):
    return np.maximum(values,0.) if cap is None else np.clip(values,0.,cap)


def _premium_values(spots, coefficients, *, strike, stock_knots, cap):
    return stopping_value(spots,coefficients,strike=strike,is_call=False,
        exercise=False,stock_knots=stock_knots,value_cap=cap)


def _expected_premium_values(mean, clock, coefficients, *, strike, stock_knots, cap):
    return expected_stopping_value(mean,clock,coefficients,strike=strike,is_call=False,
        exercise=False,stock_knots=stock_knots,value_cap=cap)


def _affine_values(proxy, time, factors, mean, clock):
    """Discounted European proxy, preserving a final paired-stock axis."""
    mean, clock = np.broadcast_arrays(mean, clock)
    n, branches, endpoints, bumps = mean.shape
    expanded = np.broadcast_to(factors[:, :, None, :],
        (n, branches, bumps, factors.shape[-1])).transpose(0, 2, 1, 3).reshape(n*bumps, branches, -1)
    means = mean.transpose(0, 3, 1, 2).reshape(n*bumps, branches, endpoints)
    clocks = clock.transpose(0, 3, 1, 2).reshape(n*bumps, branches, endpoints)
    values = proxy.values(time, expanded, means, clocks, np.array([0]))[..., 0]
    return values.reshape(n, bumps, branches, endpoints).transpose(0, 2, 3, 1)


def _interpolate(policy, time, factors):
    times = policy.continuation_times
    right = min(int(np.searchsorted(times, time)), len(times)-1)
    left = max(0, right-1)
    if left == 0:
        return policy.value_premiums[right].coefficients(factors)
    weight = (time-times[left])/(times[right]-times[left])
    return ((1-weight)*policy.value_premiums[left].coefficients(factors)
        +weight*policy.value_premiums[right].coefficients(factors))


def _continuation_coefficients(policy, time, factors):
    """Frozen continuation proxy at a LEFT microstep, in calendar time."""
    times = policy.continuation_times
    right = min(int(np.searchsorted(times, time)), len(times)-1)
    left = max(0, right-1)
    if right == left:
        return policy.continuations[right].coefficients(factors)
    weight = (time-times[left])/(times[right]-times[left])
    return ((1-weight)*policy.continuations[left].coefficients(factors)
        +weight*policy.continuations[right].coefficients(factors))


def _stream(params, market, kernel, strike, maturity, calls, times, config, control_config,
            *, training, policy=None, spots=None, trace=False):
    """Shared transition law; optional training snapshots or independent policy cashflows."""
    if not isinstance(trace,(bool,np.bool_)):
        raise TypeError("trace must be boolean")
    if trace and training:
        raise ValueError("traces require a frozen policy on an independent stream")
    n = config.training_paths if training else config.valuation_paths
    if not isinstance(control_config, RoughEuropeanControlConfig):
        raise TypeError("control_config must be RoughEuropeanControlConfig")
    seed = config.training_seed if training else config.valuation_seed
    spots = np.asarray([market.spot] if spots is None else spots)
    if (spots.dtype.kind not in "iuf" or spots.ndim != 1 or len(spots) == 0
            or not np.isfinite(spots).all() or np.any(spots <= 0)):
        raise ValueError("spots must be a finite positive real vector")
    spots = spots.astype(float)
    if training and len(spots) != 1:
        raise ValueError("training accepts one inception stock")
    grid = _time_grid(params, market, times, config.steps)
    if maturity>0 and 0<params.vol_of_vol<_MIN_POSITIVE_VOL_OF_VOL:
        raise RoughSimulationError("positive vol_of_vol below 1e-8 is unsupported by the correlated-stock representation")
    count, bumps = len(kernel.nodes), len(spots)
    estimated_memory = 8*(n*(count+20*bumps)
        + (n*len(times)*(count+3) if training else 0)
        + min(n, config.batch_size)*bumps*(128*control_config.quadrature_nodes
            +256*(len(config.stock_knots)+1)+48*count)
        + len(grid)*(count+1)*control_config.quadrature_nodes*4)
    if trace:
        estimated_memory += 8*n*len(policy.exercise_times)*(9*bumps+count)+32*n*bumps
    if estimated_memory > config.memory_budget_mb*1024**2:
        raise MemoryError(f"Fourier stopping needs approximately {estimated_memory/1024**2:.2f} MiB, exceeding memory_budget_mb={config.memory_budget_mb}")
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    proxy = _AffineSurrogate(params, market, kernel, np.array([strike]),
        np.array([maturity]), np.array([calls]), grid, control_config)
    stock = np.broadcast_to(spots, (n, bumps)).copy()
    factors, euro_m = np.zeros((n, count)), np.zeros((n, bumps))
    martingale = np.zeros_like(euro_m)
    intrinsic_m=np.zeros_like(euro_m) if trace else None
    micro_boundary_m=np.zeros_like(euro_m) if trace else None
    micro_poststop_m=np.zeros_like(euro_m) if trace else None
    initial = _payoff(spots, strike, calls)
    primal, raw = np.zeros_like(stock), np.zeros_like(stock)
    upper = np.broadcast_to(initial, stock.shape).copy() if policy is None or policy.exercise_style == "american" else np.full_like(stock, -np.inf)
    alive = np.ones_like(stock, bool)
    stop_time = np.full_like(stock, maturity)
    if policy is not None and (policy.exercise_at_inception or maturity == 0):
        primal[:] = initial
        raw[:] = initial
        alive[:] = False
        stop_time[:] = 0
    snapshot = {}
    path_trace=None
    trace_indices={}
    if trace:
        trace_times=np.asarray(policy.exercise_times,float)
        trace_indices={float(t):i for i,t in enumerate(trace_times)}
        shape=(n,len(trace_times),bumps)
        path_trace={"times":trace_times.copy(),
            "eligible":np.asarray((trace_times==maturity) if policy.exercise_style=="european"
                else np.ones(len(trace_times),bool)),
            "discounted_payoffs":np.empty(shape),
            "european_martingale":np.empty(shape),"premium_martingale":np.empty(shape),"intrinsic_martingale":np.empty(shape),
            "micro_boundary_martingale":np.empty(shape),
            "micro_poststop_martingale":np.empty(shape),
            "stocks":np.empty(shape),"european_proxy":np.empty(shape),
            "continuation_proxy":np.empty(shape),"discounts":market.discount_curve.discount(trace_times),
            "strike":float(strike),
            "factors":np.empty((n,len(trace_times),count)),
            "stop_indices":np.full((n,bumps),len(trace_times)-1,dtype=np.int64),
            "seed":int(seed),"policy_training_seed":int(policy.config.training_seed),
            "exercise_style":policy.exercise_style,"kernel_fingerprint":kernel.fingerprint,
            "micro_boundary_width":.05,
            "micro_boundary_feature_source":"left integration-step state; no new exercise dates",
            "initial_spots":spots.copy(),"actual_time_grid":grid.copy(),
            "policy_fingerprint":fourier_policy_fingerprint(policy),
            "additional_control_training_seeds":tuple(policy.diagnostics.get("additional_control_training_seeds",()))}
        zero_proxy=_affine_values(proxy,0.,np.zeros((1,1,count)),
            np.log(spots)[None,None,None,:],np.zeros((1,1,1,bumps)))[0,0,0]
        path_trace["discounted_payoffs"][:,0]=initial
        path_trace["stocks"][:,0]=spots
        path_trace["factors"][:,0]=0.
        path_trace["european_martingale"][:,0]=0.
        path_trace["premium_martingale"][:,0]=0.
        path_trace["intrinsic_martingale"][:,0]=0.
        path_trace["micro_boundary_martingale"][:,0]=0.
        path_trace["micro_poststop_martingale"][:,0]=0.
        path_trace["european_proxy"][:,0]=zero_proxy
        initial_q=_clip_premium(policy.continuations[0].value(stock,factors,strike),policy.premium_cap)
        path_trace["continuation_proxy"][:,0]=zero_proxy+initial_q
        left_european = np.broadcast_to(zero_proxy, stock.shape).copy()
        if policy.exercise_at_inception or maturity==0:
            path_trace["stop_indices"][:]=0
    if training:
        zero_euro = _affine_values(proxy, 0., factors[:, None],
            np.log(stock)[:, None, None], np.zeros((n, 1, 1, bumps)))[:, 0, 0]
        snapshot[0.] = (stock[:, 0].copy(), factors.copy(), euro_m[:, 0].copy(), zero_euro[:, 0])
    drift_cache, rng = {}, np.random.default_rng(seed)
    dates = set(times if training else policy.exercise_times)
    for left, right in zip(grid[:-1], grid[1:]):
        dt, theta = float(right-left), _theta(params, .5*(left+right))
        key = (dt, theta)
        if key not in drift_cache:
            drift_cache[key] = _drift_flow_with_integral(matrix,
                np.full(count, params.kappa*(theta-params.v0)), weights, params.v0,
                dt if params.vol_of_vol == 0 else dt/2)
        exact = drift_cache[key]
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        discount = market.discount_curve.discount(right)
        for begin in range(0, n, config.batch_size):
            end = min(n, begin+config.batch_size)
            old = factors[begin:end]
            random = rng.random((end-begin, 3))
            normal = ndtri(np.clip(random[:, 2], np.finfo(float).tiny, 1-np.finfo(float).eps))
            if params.vol_of_vol == 0:
                new = old@exact[:count, :count].T+exact[:count, -1]
                clock = _nonnegative_clock(old@exact[count, :count]+exact[count, -1])
                branches, probabilities = new[:, None], np.ones((end-begin, 1))
                means = np.log(stock[begin:end])[:, None, None]+growth-.5*clock[:, None, None, None]
                clocks = clock[:, None, None, None]
                choice, endpoint = np.zeros(end-begin, int), np.zeros(end-begin, int)
            else:
                law = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
                    flow=exact[:count, :count], shift=exact[:count, -1], integrated_flow=exact,
                    dt=dt, theta=theta, growth=growth, stock_integral=config.stock_integral)
                branches, probabilities = law["new_factors"], law["probabilities"]
                choice = ((random[:, 0] > probabilities[:, 0]).astype(int)
                    +(random[:, 0] > probabilities[:, :2].sum(axis=1)).astype(int))
                endpoint = (random[:, 1] >= .5).astype(int)
                clocks = law["orthogonal"][..., None]
                means = np.log(stock[begin:end])[:, None, None]+law["correlated"][:, :, None, None]-.5*clocks
            conditional_values = _affine_values(proxy, float(right), branches, means, clocks)
            expected = np.sum(conditional_values.mean(axis=2)*probabilities[:, :, None], axis=1)
            index = np.arange(end-begin)
            next_mean = means[index, choice, endpoint]
            next_clock = np.broadcast_to(clocks, means.shape)[index, choice, endpoint]
            with np.errstate(over="ignore",invalid="ignore",under="ignore"):
                next_stock = np.exp(next_mean+np.sqrt(next_clock)*normal[:, None])
            if not np.isfinite(next_stock).all() or np.any(next_stock<=0):
                raise RoughSimulationError("Fourier stopping stock transition overflowed or underflowed")
            new = branches[index, choice]
            actual = _affine_values(proxy, float(right), new[:, None],
                np.log(next_stock)[:, None, None], np.zeros((end-begin, 1, 1, bumps)))[:, 0, 0]
            increment = actual-expected
            euro_m[begin:end] += increment
            if trace:
                conditional_intrinsic=_conditional_payoffs(means,clocks,strike,calls)
                expected_intrinsic=np.sum(conditional_intrinsic.mean(axis=2)*probabilities[:,:,None],axis=1)
                intrinsic_increment=discount*(_payoff(next_stock,strike,calls)-expected_intrinsic)
                intrinsic_m[begin:end]+=intrinsic_increment
                # The multiplier uses only the state BEFORE this transition.
                # Its microstep updates do not add exercise opportunities.
                left_discount=market.discount_curve.discount(left)
                qleft=_premium_values(stock[begin:end],
                    _continuation_coefficients(policy,float(left),old)[:,None],
                    strike=strike,stock_knots=policy.continuations[0].knots,
                    cap=policy.premium_cap)
                margin=(left_discount*(_payoff(stock[begin:end],strike,calls)-qleft)
                    -left_european[begin:end])
                indicator=expit(margin/(left_discount*.05))
                micro_boundary_m[begin:end]+=indicator*(intrinsic_increment-increment)
                left_european[begin:end]=actual
            if policy is not None:
                coeff = _interpolate(policy, float(right), branches)[:, :, None, None]
                prem_expected = _expected_premium_values(means, clocks, coeff,
                    strike=strike, stock_knots=config.stock_knots, cap=policy.premium_cap)
                prem_expected = np.sum(prem_expected.mean(axis=2)*probabilities[:, :, None], axis=1)
                actual_coeff = _interpolate(policy, float(right), new)[:, None]
                prem_actual = _premium_values(next_stock, actual_coeff, strike=strike,
                    stock_knots=config.stock_knots, cap=policy.premium_cap)
                total_increment=increment+discount*(prem_actual-prem_expected)
                if trace:
                    # Exercise is known only at LEFT dates. This adjustment
                    # starts strictly after the policy's stopping increment,
                    # leaving its stopped primal samples exactly unchanged.
                    micro_poststop_m[begin:end] += (indicator*~alive[begin:end]
                        *(intrinsic_increment-total_increment))
                martingale[begin:end] += total_increment
            factors[begin:end], stock[begin:end] = new, next_stock
            if policy is not None and float(right) in dates:
                terminal = right == maturity
                eligible = policy.exercise_style == "american" or terminal
                if eligible:
                    payoff = _payoff(next_stock, strike, calls)
                    upper[begin:end] = np.maximum(upper[begin:end], discount*payoff-martingale[begin:end])
                    continuation = policy.continuations[int(np.searchsorted(times, right))]
                    premium = _clip_premium(continuation.value(next_stock, new, strike), policy.premium_cap)
                    future = actual/discount+premium
                    stop = alive[begin:end] & (((payoff > 0) & (payoff >= future)) | terminal)
                    raw[begin:end] = np.where(stop, discount*payoff, raw[begin:end])
                    primal[begin:end] = np.where(stop, discount*payoff-martingale[begin:end], primal[begin:end])
                    stop_time[begin:end] = np.where(stop, right, stop_time[begin:end])
                    alive[begin:end] &= ~stop
                    if trace:
                        path_trace["stop_indices"][begin:end]=np.where(stop,
                            trace_indices[float(right)],path_trace["stop_indices"][begin:end])
            if trace and float(right) in trace_indices:
                ti=trace_indices[float(right)]
                path_trace["discounted_payoffs"][begin:end,ti]=discount*_payoff(next_stock,strike,calls)
                path_trace["stocks"][begin:end,ti]=next_stock
                path_trace["factors"][begin:end,ti]=new
                path_trace["european_martingale"][begin:end,ti]=euro_m[begin:end]
                path_trace["premium_martingale"][begin:end,ti]=martingale[begin:end]-euro_m[begin:end]
                path_trace["intrinsic_martingale"][begin:end,ti]=intrinsic_m[begin:end]
                path_trace["micro_boundary_martingale"][begin:end,ti]=micro_boundary_m[begin:end]
                path_trace["micro_poststop_martingale"][begin:end,ti]=micro_poststop_m[begin:end]
                path_trace["european_proxy"][begin:end,ti]=actual
                continuation=policy.continuations[int(np.searchsorted(times,right))]
                q=_clip_premium(continuation.value(next_stock,new,strike),policy.premium_cap)
                path_trace["continuation_proxy"][begin:end,ti]=actual+discount*q
        if training and float(right) in dates:
            euro = _affine_values(proxy, float(right), factors[:, None],
                np.log(stock)[:, None, None], np.zeros((n, 1, 1, bumps)))[:, 0, 0]
            snapshot[float(right)] = (stock[:, 0].copy(), factors.copy(), euro_m[:, 0].copy(), euro[:, 0])
    if training:
        return snapshot
    if maturity==0:
        upper[:]=initial
    if not all(np.isfinite(a).all() for a in (primal,raw,upper,stop_time)):
        raise RoughSimulationError("Fourier stopping generated nonfinite cashflow/control samples")
    if np.any(alive):
        raise RoughSimulationError("Fourier stopping left live contracts")
    if path_trace is not None:
        for value in path_trace.values():
            if isinstance(value,np.ndarray):value.setflags(write=False)
        path_trace=MappingProxyType(path_trace)
    return {"trace":path_trace,"primal_samples": primal, "dual_samples": upper, "raw_samples": raw,
        "stop_times": stop_time, "european_control_samples": euro_m,
        "diagnostics": {"martingale_centering": "exact discrete six-branch/Gaussian Fourier plus positive spline premium",
            "training_seed": policy.config.training_seed, "valuation_seed": seed,
            "actual_time_steps": len(grid)-1, "independent_training_valuation": True,
            "american_pricing_ready": False, "stock_delta_ready": False,
            "frozen_policy": True,"premium_cap":policy.premium_cap,
            "time_grid":grid.tolist(),"exercise_dates":policy.exercise_times.tolist(),
            "factor_state_dimension":count}}


def train_fourier_stopping_policy(params, market, kernel, *, strike, maturity,
        is_call=False, config=None, control_config=None, exercise_style="american",
        fit_continuation=None, premium_cap=None):
    started = perf_counter()
    premium_cap = _validate_cap(premium_cap)
    config = RoughAmericanConfig() if config is None else config
    control_config = RoughEuropeanControlConfig(24, 100., 512) if control_config is None else control_config
    k, t, calls = _inputs(params, market, kernel, strike, maturity, is_call, config, exercise_style, None)
    if k.ndim:
        raise ValueError("Fourier policy training accepts one scalar contract")
    strike, maturity, is_call = float(k), float(t), bool(calls)
    if not isinstance(control_config, RoughEuropeanControlConfig):
        raise TypeError("control_config must be RoughEuropeanControlConfig")
    if any(e.amount > 0 and e.time <= maturity for e in market.dividends.events):
        raise UnsupportedModelError("Fourier stopping premium controls currently require an explicitly empty cash schedule")
    fit = _fit_continuation if fit_continuation is None else fit_continuation
    times = np.unique(np.linspace(0., maturity, config.exercise_steps+1))
    snapshots = _stream(params, market, kernel, strike, maturity, is_call,
        times, config, control_config, training=True)
    continuations, value_premiums = [None]*len(times), [None]*len(times)
    stock, factors, martingale, euro = snapshots[float(maturity)]
    target = market.discount_curve.discount(maturity)*_payoff(stock, strike, is_call)-martingale
    continuations[-1] = value_premiums[-1] = _zero_continuation(factors, config)
    for j in range(len(times)-2, 0, -1):
        time = float(times[j])
        stock, factors, martingale, euro = snapshots[time]
        discount = market.discount_curve.discount(time)
        residual = (target+martingale-euro)/discount
        continuation = fit(stock, factors, residual, strike, config)
        continuations[j] = continuation
        premium = _clip_premium(continuation.value(stock, factors, strike), premium_cap)
        intrinsic = _payoff(stock, strike, is_call)
        exercise = exercise_style == "american"
        snell_premium = np.maximum(intrinsic-euro/discount, premium) if exercise else premium
        value_premiums[j] = fit(stock, factors, snell_premium, strike, config)
        if exercise:
            stop = (intrinsic > 0) & (intrinsic >= euro/discount+premium)
            target = np.where(stop, discount*intrinsic-martingale, target)
    zero = _zero_continuation(snapshots[0.][1], config)
    continuations[0] = value_premiums[0] = zero
    initial = float(_payoff(market.spot, strike, is_call))
    initial_stop = exercise_style == "american" and initial > 0 and initial >= target.mean()
    return FourierStoppingPolicy(params, market, kernel, float(strike), float(maturity),
        bool(is_call), exercise_style, times, times, tuple(continuations), tuple(value_premiums),
        config, control_config, initial_stop,
        {"elapsed_seconds": perf_counter()-started, "training_paths": config.training_paths,
            "european_baseline": "affine-Fourier proxy with exact discrete centering",
            "normalization_source": "training only", "regression_ranks":[c.rank for c in continuations],
            "regression_conditions":[c.condition for c in continuations], "regression_target": "stopped European-martingale-controlled residual premium",
            "independent_training_valuation": True, "cash_dividends_supported": False,
            "premium_cap": premium_cap}, premium_cap)


def value_fourier_stopping_policy(policy, *, config=None, spots=None, trace=False):
    if not isinstance(policy, FourierStoppingPolicy):
        raise TypeError("policy must be FourierStoppingPolicy")
    config = policy.config if config is None else config
    if not isinstance(config, RoughAmericanConfig):
        raise TypeError("config must be RoughAmericanConfig")
    if config.valuation_seed == policy.config.training_seed:
        raise ValueError("valuation stream must be independent from training")
    if config.valuation_seed in policy.diagnostics.get("additional_control_training_seeds",()):
        raise ValueError("valuation stream must be independent from additional control training")
    if config.stock_integral != policy.config.stock_integral or config.stock_knots != policy.config.stock_knots:
        raise ValueError("valuation transition scheme and knots must match training")
    return _stream(policy.params, policy.market, policy.kernel, policy.strike,
        policy.maturity, policy.is_call, policy.continuation_times, config,
        policy.control_config, training=False, policy=policy, spots=spots,trace=trace)


def _bellman_expectation(params, market, kernel, proxy, stock, factors, *, left,
        right, premium, strike, is_call, config, premium_cap=None):
    """Exact expectation of the separate European and positive premium proxies."""
    count=len(kernel.nodes)
    nodes,weights=np.asarray(kernel.nodes),np.asarray(kernel.weights)
    matrix=-np.diag(nodes)-params.kappa*np.ones((count,1))*weights
    dt=right-left
    theta=_theta(params,.5*(left+right))
    exact=_drift_flow_with_integral(matrix,np.full(count,params.kappa*(theta-params.v0)),
        weights,params.v0,dt if params.vol_of_vol==0 else dt/2)
    growth=math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
    result=np.empty(len(stock))
    for begin in range(0,len(stock),config.batch_size):
        end=min(len(stock),begin+config.batch_size)
        old=factors[begin:end]
        if params.vol_of_vol==0:
            new=old@exact[:count,:count].T+exact[:count,-1]
            clock=_nonnegative_clock(old@exact[count,:count]+exact[count,-1])
            branches,probability=new[:,None],np.ones((end-begin,1))
            means=np.log(stock[begin:end])[:,None,None,None]+growth-.5*clock[:,None,None,None]
            clocks=clock[:,None,None,None]
        else:
            law=_weak_transition_branches(old,params,nodes=nodes,weights=weights,
                flow=exact[:count,:count],shift=exact[:count,-1],integrated_flow=exact,
                dt=dt,theta=theta,growth=growth,stock_integral=config.stock_integral)
            branches,probability=law['new_factors'],law['probabilities']
            clocks=law['orthogonal'][...,None]
            means=np.log(stock[begin:end])[:,None,None,None]+law['correlated'][:,:,None,None]-.5*clocks
        euro=_affine_values(proxy,right,branches,means,clocks)[...,0]
        coeff=premium.coefficients(branches)[:,:,None,...]
        prem=_expected_premium_values(means[...,0],clocks[...,0],coeff,
            strike=strike,stock_knots=config.stock_knots,cap=premium_cap)
        result[begin:end]=np.sum(probability*np.mean(euro+market.discount_curve.discount(right)*prem,axis=2),axis=1)
    return result


def train_fourier_bellman_policy(params, market, kernel, *, strike, maturity,
        is_call=False, config=None, control_config=None, exercise_style='american',
        fit_continuation=None, premium_cap=None):
    """Controlled fitted value iteration, with exact-centered training labels.

    Every integration date is a regression date. At nonexercise dates the
    European plus positive continuation-premium proxy is integrated exactly.
    At exercise dates the sampled Bellman value is controlled with the same
    European plus fitted value-premium proxy. Valuation remains independent.
    """
    started=perf_counter()
    premium_cap=_validate_cap(premium_cap)
    config=RoughAmericanConfig() if config is None else config
    control_config=RoughEuropeanControlConfig(24,100.,512) if control_config is None else control_config
    k,t,calls=_inputs(params,market,kernel,strike,maturity,is_call,config,exercise_style,None)
    if k.ndim: raise ValueError('Fourier policy training accepts one scalar contract')
    strike,maturity,is_call=float(k),float(t),bool(calls)
    if any(e.amount>0 and e.time<=maturity for e in market.dividends.events):
        raise UnsupportedModelError('Fourier stopping premium controls currently require an explicitly empty cash schedule')
    fit=_fit_continuation if fit_continuation is None else fit_continuation
    exercise_times=np.unique(np.linspace(0.,maturity,config.exercise_steps+1))
    times=_time_grid(params,market,exercise_times,config.steps)
    snapshots=_stream(params,market,kernel,strike,maturity,is_call,times,config,control_config,training=True)
    proxy=_AffineSurrogate(params,market,kernel,np.array([strike]),np.array([maturity]),
        np.array([is_call]),times,control_config)
    continuations,value_premiums=[None]*len(times),[None]*len(times)
    zero=_zero_continuation(snapshots[float(maturity)][1],config)
    continuations[-1]=value_premiums[-1]=zero
    exercise_set=set(exercise_times) if exercise_style=='american' else {maturity}
    initial_value=float(_payoff(market.spot,strike,is_call))
    for j in range(len(times)-2,-1,-1):
        time,right=float(times[j]),float(times[j+1])
        stock,factors,_,euro=snapshots[time]
        next_stock,next_factors,_,next_euro=snapshots[right]
        next_discount=market.discount_curve.discount(right)
        expected=_bellman_expectation(params,market,kernel,proxy,stock,factors,left=time,
            right=right,premium=value_premiums[j+1],strike=strike,is_call=is_call,config=config,premium_cap=premium_cap)
        # The only noisy correction is the discrepancy between a future
        # exercise-date Snell value and its fitted value-premium function.
        if right in exercise_set and right!=maturity:
            next_premium=_clip_premium(continuations[j+1].value(next_stock,next_factors,strike),premium_cap)
            next_value=np.maximum(_payoff(next_stock,strike,is_call),next_euro/next_discount+next_premium)
            fitted=_clip_premium(value_premiums[j+1].value(next_stock,next_factors,strike),premium_cap)
            expected+=next_discount*next_value-next_euro-next_discount*fitted
        discount=market.discount_curve.discount(time)
        if j==0:
            initial_value=float(np.mean(expected))
            continuations[j]=value_premiums[j]=_zero_continuation(factors,config)
            continue
        residual=(expected-euro)/discount
        continuation=fit(stock,factors,residual,strike,config)
        continuations[j]=continuation
        premium=_clip_premium(continuation.value(stock,factors,strike),premium_cap)
        if time in exercise_set:
            snell=np.maximum(_payoff(stock,strike,is_call)-euro/discount,premium)
            value_premiums[j]=fit(stock,factors,snell,strike,config)
        else:
            value_premiums[j]=continuation
    initial=float(_payoff(market.spot,strike,is_call))
    initial_stop=exercise_style=='american' and initial>0 and initial>=initial_value
    return FourierStoppingPolicy(params,market,kernel,strike,maturity,is_call,exercise_style,
        exercise_times,times,tuple(continuations),tuple(value_premiums),config,control_config,
        initial_stop,{'elapsed_seconds':perf_counter()-started,'training_paths':config.training_paths,
            'training_algorithm':'controlled fitted value iteration on every integration date',
            'regression_ranks':[c.rank for c in continuations],
            'regression_conditions':[c.condition for c in continuations],
            'independent_training_valuation':True,'initial_training_continuation':initial_value,
            'cash_dividends_supported':False,'premium_cap':premium_cap},premium_cap)
