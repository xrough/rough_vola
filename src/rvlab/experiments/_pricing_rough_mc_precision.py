"""Equivalent selected-estimator execution for a fixed rough-Heston MC study.

This successor preserves the sealed transition, functions and coefficients. It
shares the two-step law and Gaussian interval moments between controls, and
omits unselected endpoint tables. No old numerical source is modified.
"""
from dataclasses import replace
from numbers import Integral

import numpy as np
from scipy.special import ndtr

from ._pricing_rough_bellman_control_aligned import aligned_node_proxy, target_nodal_premiums
from ._pricing_rough_exercise_block import exercise_block_law
from ._pricing_rough_local_control import _correction_coefficients, _linear_coefficients
from ._pricing_rough_node_refinement_fast import refined_stock_knots, snell_refinement_coefficients
from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_simulation import _time_grid
from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint


def _zeros(workspace, name, shape, dtype=float):
    if workspace is None:return np.zeros(shape,dtype=dtype)
    from pathlib import Path
    folder=Path(workspace);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.npy'
    if path.exists():raise FileExistsError('Never overwrite numerical scratch arrays')
    value=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
    # A newly extended local file has zero-filled bytes; avoid touching every
    # page before numerical work (important on the declared8GiB machine).
    return value


def normal_interval(left, right):
    """Same stable subtraction as the reference, evaluating only its chosen side."""
    left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    positive = left > 0
    result = np.empty(left.shape)
    result[positive] = ndtr(-left[positive])-ndtr(-right[positive])
    result[~positive] = ndtr(right[~positive])-ndtr(left[~positive])
    return result


def gaussian_moments(mean, variance, *, strike, knots):
    """Reference interval probability/first moment; shared across spline families."""
    mean, variance = np.broadcast_arrays(np.asarray(mean, float), np.asarray(variance, float))
    if np.any(variance < 0) or not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError("Finite Gaussian means and nonnegative clocks required")
    logmean = mean-np.log(strike); root = np.sqrt(np.maximum(variance, np.finfo(float).tiny))
    bounds = np.r_[-np.inf, np.log(knots), np.inf]
    left = (bounds[:-1]-logmean[..., None])/root[..., None]
    right = (bounds[1:]-logmean[..., None])/root[..., None]
    probability = normal_interval(left, right)
    first = np.exp(logmean+.5*variance)[..., None]*normal_interval(left-root[..., None], right-root[..., None])
    return probability, first


def shared_expectations(mean, variance, coefficients, *, strike, knots, moments=None):
    """Reference arithmetic, including the exact zero-clock interval convention."""
    mean, variance = np.broadcast_arrays(np.asarray(mean, float), np.asarray(variance, float))
    coefficient = np.asarray(coefficients, float)
    if (coefficient.shape[-1] != 3 or np.any(coefficient[..., 2] != 0)
            or not np.isfinite(coefficient).all()):
        raise ValueError("Finite linear spline coefficients required")
    probability, first = gaussian_moments(mean, variance, strike=strike, knots=knots) if moments is None else moments
    value = np.sum(coefficient[..., None, :, 0]*probability[..., None, :, :]
        +coefficient[..., None, :, 1]*first[..., None, :, :], axis=-1)
    zero = variance == 0
    if np.any(zero):
        z = np.exp(mean-np.log(strike)); index = np.searchsorted(knots, z, side="right")
        c = np.broadcast_to(coefficient[..., None, :, :], coefficient.shape[:-2]+(mean.shape[-1],)+coefficient.shape[-2:])
        indices = np.broadcast_to(index[..., None, :, None, None], c.shape[:-2]+(1, 1))
        actual = np.take_along_axis(c, indices, axis=-2)[..., 0, :]
        value = np.where(zero[..., None, :], actual[..., 0]+actual[..., 1]*z[..., None, :], value)
    if not np.isfinite(value).all():
        raise FloatingPointError("Shared exact Gaussian expectation is nonfinite")
    return value


def _actual(coefficient, stock, strike, knots):
    z = stock/strike; index = np.searchsorted(knots, z, side="right")
    indices = np.broadcast_to(index[:, None, None, None], coefficient.shape[:2]+(1, 1))
    c = np.take_along_axis(coefficient, indices, axis=-2)[..., 0, :]
    return c[..., 0]+c[..., 1]*z[:, None]


def combined_martingales(proxy, selected, models, trace, *, batch_size=64, require_independent=True, forbidden_seeds=(), progress=None):
    """Unchanged local/grid/target processes sharing laws and fine-grid moments.

    The65-point and129-point European coefficient sets are deliberately evaluated
    separately. All three reference right-state function builders are called
    unchanged; this optimization changes neither approximation functions nor
    actual-minus-conditional-expectation arithmetic.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("Positive integer numerical work batch required")
    times = np.asarray(trace['times']); grid = np.asarray(trace['actual_time_grid'])
    stock, factors, euro = (np.asarray(trace[k]) for k in ('stocks', 'factors', 'european_proxy'))
    seed = trace.get('seed')
    if (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
            or (require_independent and seed in set(selected.training_seeds)|set(forbidden_seeds))
            or fourier_policy_fingerprint(selected.base_policy) != fourier_policy_fingerprint(proxy)
            or trace.get('policy_fingerprint') != fourier_policy_fingerprint(proxy)
            or trace.get('kernel_fingerprint') != proxy.kernel.fingerprint or trace.get('strike') != proxy.strike
            or not np.array_equal(trace.get('initial_spots'), [proxy.market.spot])
            or not np.array_equal(trace.get('discounts'), proxy.market.discount_curve.discount(times))
            or not np.array_equal(times, proxy.continuation_times) or not np.array_equal(times, selected.times)
            or not np.array_equal(grid, _time_grid(proxy.params, proxy.market, times, proxy.config.steps))
            or proxy.market.dividends.events or stock.ndim != 3 or stock.shape[-1] != 1
            or stock.shape[1] != len(times) or euro.shape != stock.shape
            or factors.shape != (len(stock), len(times), len(proxy.kernel.nodes))
            or not all(np.isfinite(v).all() for v in (stock, factors, euro)) or np.any(stock <= 0)):
        raise ValueError("Matching frozen full-state trace and independent original law required")
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]; knots = nodes[1:]
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([proxy.strike]),
        np.array([proxy.maturity]), np.array([proxy.is_call]), grid, proxy.control_config)
    aligned = aligned_node_proxy(proxy, affine)
    n, dates = stock.shape[:2]
    local = np.zeros((n, dates, 3)); refined = np.zeros((n, dates, 6)); targets = np.zeros((n, dates, 4))
    expected_euro = np.zeros((n, dates)); actual_premium = np.zeros_like(targets); expected_premium = np.zeros_like(targets)
    local_actual = np.zeros((n, dates)); local_expected = np.zeros((n, dates))
    grid_actual = np.zeros((n, dates, 2)); grid_expected = np.zeros_like(grid_actual)
    maxmass = 0.
    from rvlab.pricing.rough_stopping_fourier import _affine_values
    for j, (left, right) in enumerate(zip(times[:-1], times[1:])):
        if progress is not None and j % 8 == 0: progress('combined_interval', j, len(times)-1)
        a, b = np.searchsorted(grid, [left, right])
        if grid[a] != left or grid[b] != right or b-a != 2:
            raise ValueError("Exactly two original physical increments required")
        for begin in range(0, n, batch_size):
            end = min(begin+batch_size, n); old = factors[begin:end, j]; s = stock[begin:end, j, 0]
            law = exercise_block_law(proxy, old, times=grid[a:b+1]); new = law['new_factors']; p = law['probabilities']
            maxmass = max(maxmass, float(np.max(abs(p.sum(axis=1)-1))))
            mean = np.log(s)[:, None, None]+law['log_stock_shifts']; clock = law['orthogonal']
            moments = gaussian_moments(mean, clock, strike=proxy.strike, knots=knots)
            # Fine-grid signed corrections and targets use identical Gaussian moments.
            gc, _ = snell_refinement_coefficients(proxy, selected, j+1, new, fine_nodes=nodes)
            tc = _linear_coefficients(nodes, target_nodal_premiums(proxy, selected, models, j+1, new, fine_nodes=nodes, node_proxy=aligned))
            gp = shared_expectations(mean, clock, gc, strike=proxy.strike, knots=knots, moments=moments)
            tp = shared_expectations(mean, clock, tc, strike=proxy.strike, knots=knots, moments=moments)
            ge = np.sum(gp.mean(axis=-1)*p[..., None], axis=1)
            te = np.sum(tp.mean(axis=-1)*p[..., None], axis=1)
            # Local signed coarse spline keeps its own exact interval partition.
            lc = _correction_coefficients(proxy, selected, j+1, new, 3.)
            lp = shared_expectations(mean, clock, lc[:, :, None], strike=proxy.strike, knots=proxy.config.stock_knots)[:, :, 0]
            le = np.sum(lp.mean(axis=-1)*p, axis=1)
            ee = np.sum(_affine_values(affine, float(right), new, mean[..., None], clock[..., None])[..., 0].mean(axis=2)*p, axis=1)
            actual_factors = factors[begin:end, j+1]; actual_stock = stock[begin:end, j+1, 0]
            agc, _ = snell_refinement_coefficients(proxy, selected, j+1, actual_factors, fine_nodes=nodes)
            atc = _linear_coefficients(nodes, target_nodal_premiums(proxy, selected, models, j+1, actual_factors, fine_nodes=nodes, node_proxy=aligned))
            ga = _actual(agc, actual_stock, proxy.strike, knots); ta = _actual(atc, actual_stock, proxy.strike, knots)
            alc = _correction_coefficients(proxy, selected, j+1, actual_factors, 3.)
            z = actual_stock/proxy.strike; ix = np.searchsorted(proxy.config.stock_knots, z, side='right')
            c = np.take_along_axis(alc, ix[:, None, None], axis=-2)[:, 0]; la = c[:, 0]+c[:, 1]*z
            d = float(proxy.market.discount_curve.discount(right)); ld = float(proxy.market.discount_curve.discount(left))
            q = selected.continuations[j].value(s, old, proxy.strike)
            q = np.maximum(q, 0.) if selected.premium_cap is None else np.clip(q, 0., selected.premium_cap)
            margin = ld*(np.maximum((1 if proxy.is_call else -1)*(s-proxy.strike), 0.)-q)-euro[begin:end, j, 0]
            gates = np.column_stack((np.ones_like(margin), np.exp(-.5*(margin/.1)**2), np.exp(-.5*(margin/.3)**2)))
            local[begin:end, j+1] = local[begin:end, j]+d*(la-le)[:, None]*gates
            refined[begin:end, j+1] = refined[begin:end, j]+(d*(ga-ge)[..., None]*gates[:, None, :]).reshape(end-begin, -1)
            targets[begin:end, j+1] = targets[begin:end, j]+((euro[begin:end, j+1, 0]-ee)[:, None]+d*(ta-te))
            expected_euro[begin:end, j+1] = ee; actual_premium[begin:end, j+1] = ta; expected_premium[begin:end, j+1] = te
            local_actual[begin:end, j+1] = la; local_expected[begin:end, j+1] = le
            grid_actual[begin:end, j+1] = ga; grid_expected[begin:end, j+1] = ge
    arrays = dict(prior_local_corrections=local, grid_corrections=refined, target_martingales=targets,
        target_expected_european=expected_euro, target_actual_premiums=actual_premium, target_expected_premiums=expected_premium,
        local_actual=local_actual, local_expected=local_expected, grid_actual=grid_actual, grid_expected=grid_expected)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Combined selected-control processes are nonfinite")
    return arrays, dict(max_probability_error=maxmass, work_batch_size=int(batch_size), physical_random_batch_size=proxy.config.batch_size,
        exact_reference_functions=True, european65_and129_kept_separate=True, shared_gaussian_moment_tables=True,
        selected_estimator_only=True, all_paths_retained=True)


__all__ = ['normal_interval', 'gaussian_moments', 'shared_expectations', 'combined_martingales']


def selected_physical_trace(proxy, *, seed, paths, indices=None, progress=None, workspace=None):
    """Original512-batch RNG law with a vectorized linear-premium expectation.

    The frozen obstacle premium is continuous piecewise linear, bounded at every
    node and constant outside the upper node. Its clipping is redundant apart
    from floating-point rounding. The preflight compares every saved component
    against the preserved generic clipped-polynomial implementation.
    Optional indices restrict work, never full-stream RNG/state propagation.
    """
    import math
    from scipy.special import expit, ndtri
    from rvlab.pricing.rough_estimators import _conditional_payoffs
    from rvlab.pricing.rough_simulation import _theta, _drift_flow_with_integral, _weak_transition_branches
    from rvlab.pricing.rough_stopping import _payoff
    from rvlab.pricing.rough_stopping_fourier import _affine_values, _interpolate, _premium_values, _continuation_coefficients
    if (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
            or isinstance(paths, bool) or not isinstance(paths, Integral) or paths < 2
            or proxy.config.steps != 128 or proxy.config.batch_size != 512
            or proxy.params.vol_of_vol <= 0 or proxy.market.dividends.events
            or proxy.exercise_style != 'american'):
        raise ValueError('Frozen stochastic no-cash128/512 law and explicit stream required')
    wanted = np.arange(paths) if indices is None else np.asarray(indices)
    if (wanted.ndim != 1 or wanted.dtype.kind not in 'iu' or len(wanted) < 1
            or np.any(wanted < 0) or np.any(wanted >= paths) or np.any(np.diff(wanted) <= 0)):
        raise ValueError('Increasing absolute full-stream indices required')
    params, market, kernel = proxy.params, proxy.market, proxy.kernel
    count = len(kernel.nodes); nodes = np.asarray(kernel.nodes); weights = np.asarray(kernel.weights)
    times = np.asarray(proxy.exercise_times); grid = _time_grid(params, market, times, proxy.config.steps)
    affine = _AffineSurrogate(params, market, kernel, np.array([proxy.strike]), np.array([proxy.maturity]), np.array([proxy.is_call]), grid, proxy.control_config)
    fullstock = np.full((paths, 1), market.spot); fullfactors = np.zeros((paths, count))
    n = len(wanted); shape = (n, len(times), 1); physical = (n, len(grid))
    trace = {k:_zeros(workspace,'trace_'+k,shape) for k in ('stocks', 'european_proxy', 'discounted_payoffs', 'european_martingale', 'premium_martingale', 'intrinsic_martingale', 'micro_boundary_martingale')}
    trace.update(times=times, actual_time_grid=grid, factors=_zeros(workspace,'trace_factors',(n, len(times), count)), discounts=market.discount_curve.discount(times),
        eligible=np.ones(len(times), bool), seed=int(seed), policy_fingerprint=fourier_policy_fingerprint(proxy),
        kernel_fingerprint=kernel.fingerprint, strike=proxy.strike, initial_spots=np.array([market.spot]), exercise_style='american')
    trace['stocks'][:, 0, 0] = market.spot; trace['discounted_payoffs'][:, 0, 0] = _payoff(market.spot, proxy.strike, proxy.is_call)
    zero = _affine_values(affine, 0., np.zeros((1, 1, count)), np.log(np.array([market.spot]))[None, None, None, :], np.zeros((1, 1, 1, 1)))[0, 0, 0]
    trace['european_proxy'][:, 0] = zero
    em = np.zeros((n, 1)); total = np.zeros_like(em); im = np.zeros_like(em); micro = np.zeros_like(em)
    left_euro = np.broadcast_to(zero, (n, 1)).copy()
    mu = np.full(n, math.log(market.spot)); variance = np.zeros(n); cm = np.zeros(n)
    conditional = {k:_zeros(workspace,k,physical) for k in ('conditional_log_means','conditional_variances','conditional_expected_next','conditional_actual_next','replay_physical_stocks')}
    conditional['conditional_log_means'][:, 0] = mu; conditional['replay_physical_stocks'][:, 0] = market.spot
    conditional['replay_physical_factors'] = _zeros(workspace,'replay_physical_factors',physical+(count,))
    rng = np.random.default_rng(seed); cache = {}; dateindex = {float(t):j for j,t in enumerate(times)}
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    capround = 0.
    for k, (left, right) in enumerate(zip(grid[:-1], grid[1:]), 1):
        if progress is not None and (k-1) % 16 == 0: progress('physical_increment', k-1, len(grid)-1)
        dt = float(right-left); theta = _theta(params, .5*(left+right)); key = (dt, theta)
        if key not in cache:
            cache[key] = _drift_flow_with_integral(matrix, np.full(count, params.kappa*(theta-params.v0)), weights, params.v0, dt/2)
        flow = cache[key]; growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right)); discount = market.discount_curve.discount(right)
        for begin in range(0, paths, 512):
            end = min(begin+512, paths); old = fullfactors[begin:end]; random = rng.random((end-begin, 3))
            law = _weak_transition_branches(old, params, nodes=nodes, weights=weights, flow=flow[:count,:count], shift=flow[:count,-1], integrated_flow=flow,
                dt=dt, theta=theta, growth=growth, stock_integral=proxy.config.stock_integral)
            p = law['probabilities']; choice = (random[:,0]>p[:,0]).astype(int)+(random[:,0]>p[:,:2].sum(axis=1)).astype(int)
            endpoint = (random[:,1]>=.5).astype(int); row = np.arange(end-begin)
            clocks = law['orthogonal'][...,None]
            means = np.log(fullstock[begin:end])[:,None,None]+law['correlated'][:,:,None,None]-.5*clocks
            normal = ndtri(np.clip(random[:,2],np.finfo(float).tiny,1-np.finfo(float).eps))
            nextstock = np.exp(means[row,choice,endpoint]+np.sqrt(np.broadcast_to(clocks,means.shape)[row,choice,endpoint])*normal[:,None])
            new = law['new_factors'][row,choice]
            chosen = np.flatnonzero((wanted>=begin)&(wanted<end)); local = wanted[chosen]-begin
            if len(chosen):
                branch = law['new_factors'][local]; m = means[local]; clock = clocks[local]; probability = p[local]
                actual_stock = nextstock[local]; actual_factor = new[local]
                values = _affine_values(affine,float(right),branch,m,clock)
                expected = np.sum(values.mean(axis=2)*probability[:,:,None],axis=1)
                actual = _affine_values(affine,float(right),actual_factor[:,None],np.log(actual_stock)[:,None,None],np.zeros((len(local),1,1,1)))[:,0,0]
                increment = actual-expected; em[chosen] += increment
                intrinsic = _conditional_payoffs(m,clock,proxy.strike,proxy.is_call)
                expected_intrinsic = np.sum(intrinsic.mean(axis=2)*probability[:,:,None],axis=1)
                intrinsic_increment = discount*(_payoff(actual_stock,proxy.strike,proxy.is_call)-expected_intrinsic); im[chosen] += intrinsic_increment
                qleft = _premium_values(fullstock[wanted[chosen]],_continuation_coefficients(proxy,float(left),old[local])[:,None],strike=proxy.strike,stock_knots=proxy.continuations[0].knots,cap=proxy.premium_cap)
                margin = market.discount_curve.discount(left)*(_payoff(fullstock[wanted[chosen]],proxy.strike,proxy.is_call)-qleft)-left_euro[chosen]
                micro[chosen] += expit(margin/(market.discount_curve.discount(left)*.05))*(intrinsic_increment-increment); left_euro[chosen] = actual
                coefficient = _interpolate(proxy,float(right),branch)
                if np.any(coefficient[...,2] != 0) or np.any(coefficient[...,-1,1] != 0): raise ValueError('Vectorized premium requires exact linear coefficients')
                nodal = coefficient[...,0]+coefficient[...,1]*np.r_[0., proxy.config.stock_knots]
                violation = max(float(np.max(-nodal)),float(np.max(nodal-proxy.premium_cap)))
                capround = max(capround,violation)
                if violation > 2e-13: raise FloatingPointError('Obstacle spline is not bounded; clipping cannot be removed')
                premium = shared_expectations(m[...,0], np.broadcast_to(clock,m.shape)[...,0], coefficient[:,:,None],strike=proxy.strike,knots=proxy.config.stock_knots)[:,:,0]
                premium_expected = np.sum(premium.mean(axis=2)*probability,axis=1)[:,None]
                actual_coefficient = _interpolate(proxy,float(right),actual_factor)[:,None]
                premium_actual = _premium_values(actual_stock,actual_coefficient,strike=proxy.strike,stock_knots=proxy.config.stock_knots,cap=proxy.premium_cap)
                total[chosen] += increment+discount*(premium_actual-premium_expected)
                cmean = mu[chosen,None,None]+law['correlated'][local,:,None]-.5*law['orthogonal'][local]
                cclock = variance[chosen,None,None]+law['orthogonal'][local]
                cvalues = affine.values(float(right),branch,cmean,cclock,np.array([0]))[...,0]
                ce = np.sum(cvalues.mean(axis=2)*probability,axis=1)
                mu[chosen] = cmean[np.arange(len(chosen)),choice[local],endpoint[local]]
                variance[chosen] = cclock[np.arange(len(chosen)),choice[local],endpoint[local]]
                ca = affine.values(float(right),actual_factor[:,None],mu[chosen,None,None],variance[chosen,None,None],np.array([0]))[:,0,0,0]
                cm[chosen] += ca-ce
                conditional['conditional_expected_next'][chosen,k] = ce; conditional['conditional_actual_next'][chosen,k] = ca
                conditional['conditional_log_means'][chosen,k] = mu[chosen]; conditional['conditional_variances'][chosen,k] = variance[chosen]
                conditional['replay_physical_stocks'][chosen,k] = actual_stock[:,0]; conditional['replay_physical_factors'][chosen,k] = actual_factor
                if float(right) in dateindex:
                    j = dateindex[float(right)]
                    trace['stocks'][chosen,j] = actual_stock; trace['factors'][chosen,j] = actual_factor
                    trace['discounted_payoffs'][chosen,j] = discount*_payoff(actual_stock,proxy.strike,proxy.is_call)
                    trace['european_proxy'][chosen,j] = actual
                    for name,value in [('european_martingale',em),('premium_martingale',total-em),('intrinsic_martingale',im),('micro_boundary_martingale',micro)]:
                        trace[name][chosen,j] = value[chosen]
            fullstock[begin:end] = nextstock; fullfactors[begin:end] = new
    payoff = float(market.discount_curve.discount(proxy.maturity))*_conditional_payoffs(mu,variance,proxy.strike,proxy.is_call)
    conditional.update(absolute_path_indices=wanted,physical_time_grid=grid,conditional_european_payoff=payoff,conditional_european_martingale=cm,conditional_european_estimate=payoff-cm)
    return trace,conditional,dict(seed=int(seed),full_stream_paths=int(paths),evaluated_indices=wanted.tolist(),physical_steps=128,
        physical_random_batch_size=512,all_random_draws_consumed=True,max_nodal_cap_roundoff=capround,
        linear_premium_expectation=True,clipping_redundancy_checked=True,all_paths_retained=indices is None,
        no_continuous_price_centering=True,no_empirical_valuation_recentering=True)

__all__ += ['selected_physical_trace']


def _finish_selected(bundle, prior_control, grid_control, tail_control, target_control, models, coefficients, trace, conditional, *, forbidden_seeds=(), progress=None):
    """Generate only the already frozen estimator, with reconstructable primitives."""
    from ._pricing_rough_local_control import correction_basis
    from ._pricing_rough_tail_switch import tail_switch_corrections
    from ._pricing_rough_bellman_delta import bellman_policy_stops
    from ._pricing_rough_bellman_policy import BellmanEvaluator
    from ._pricing_rough_price_budget_controls import predictable_basis, common_shift
    from rvlab.pricing.rough_dual import _weighted_martingale
    proxy = bundle['proxy']; selected = bundle['selected_policy']
    if bundle['weights'].config.predictable_features:
        raise ValueError('This selected-only execution requires the sealed feature-free stable weights')
    from time import perf_counter
    phase = perf_counter(); phase_seconds = {}
    processes, process_diagnostic = combined_martingales(proxy, selected, models, trace, batch_size=64, forbidden_seeds=forbidden_seeds, progress=progress)
    phase_seconds['combined'] = perf_counter()-phase; phase = perf_counter()
    stable = _weighted_martingale(trace, bundle['weights'])[:,:,0]
    prior = stable+np.einsum('ntd,d->nt',correction_basis(processes['prior_local_corrections'],trace['times'],prior_control['bins']),prior_control['weights'])
    current = prior+np.einsum('ntd,d->nt',correction_basis(processes['grid_corrections'],trace['times'],grid_control['bins']),grid_control['weights'])
    tail = tail_switch_corrections(proxy,selected,trace,current,forbidden_seeds=forbidden_seeds)
    latest = current+np.einsum('ntd,d->nt',correction_basis(tail['corrections'],trace['times'],tail_control['bins']),tail_control['weights'])
    class ProgressEvaluator(BellmanEvaluator):
        def conditional_expectations(self, j, stocks, factors):
            if progress is not None and j % 8 == 0: progress('policy_date', j, 64)
            return super().conditional_expectations(j, stocks, factors)
    looked = bellman_policy_stops(trace,ProgressEvaluator(proxy,selected,orders=(16,)),outer_steps=128,forbidden_seeds=forbidden_seeds)
    phase_seconds['policy_and_algebra'] = perf_counter()-phase
    stops = looked['stops'][:,0]; rows = np.arange(len(stops)); payoff = trace['discounted_payoffs'][:,:,0]
    correction = processes['target_martingales']-latest[:,:,None]
    final_m = latest+np.einsum('ntd,d->nt',correction_basis(correction,trace['times'],target_control['bins']),target_control['weights'])
    raw = payoff[rows,stops]; lower = raw-final_m[rows,stops]; upper = np.max(payoff-final_m,axis=1)
    arrays = {k:trace[k] for k in ('times','discounts','eligible','actual_time_grid','stocks','factors','european_proxy')}
    arrays.update({k:trace[k][:,:,0] for k in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale')})
    arrays.update(processes,stable_old_martingale=stable,prior_martingale=prior,current_martingale=current,
        tail_corrections=tail['corrections'],tail_gates=tail['gates'],tail_margin=tail['margin'],latest_martingale=latest,
        gh16_stops=stops,gh16_raw=raw,gh16_continuation=looked['continuation'][:,:,0],gh16_computed_mask=looked['computed_mask'][:,:,0],
        payoff=payoff,candidate3_lower=lower,candidate3_upper=upper,unchanged_lower=lower,unchanged_upper=upper)
    # The replay indexes are absolute full-stream IDs; selected arrays are local
    # to these IDs during spent preflight and complete arange(paths) on finals.
    anchor = payoff[:,-1]-arrays['european_martingale'][:,-1]-conditional['conditional_european_estimate']
    basis,names = predictable_basis(arrays,bundle,include_anchor=True,anchor=anchor)
    shared_lower,shared_upper,shift = common_shift(lower,upper,basis,coefficients)
    arrays.update(basis=basis,shift=shift,shared_lower=shared_lower,shared_upper=shared_upper)
    conditional['shared_control'] = anchor
    return arrays,anchor,dict(combined=process_diagnostic,fixed_policy=looked['diagnostics'],basis_names=names,
        phase_seconds=phase_seconds,selected_dual_only=True,no_fitting_or_selection=True,optimal_value_ready=False)



# Forked workers inherit only immutable frozen models and spent/fresh simulated
# states. No child draws random numbers, fits a coefficient or selects a path.
_PURE_WORK = None


def _pure_chunk(bounds):
    import os
    from time import process_time
    arguments, trace, conditional, progress, forbidden_seeds = _PURE_WORK
    left, right = bounds
    path_fields = ('stocks','factors','european_proxy','discounted_payoffs','european_martingale',
        'premium_martingale','intrinsic_martingale','micro_boundary_martingale')
    part = {k:(v[left:right] if k in path_fields else v) for k,v in trace.items()}
    q = {'conditional_european_estimate':conditional['conditional_european_estimate'][left:right]}
    def callback(phase,done,total):
        if progress is not None: progress(f'chunk{left}:{right}:{phase}',done,total)
    before = process_time()
    arrays,anchor,diagnostics = _finish_selected(*arguments,part,q,forbidden_seeds=forbidden_seeds,progress=callback)
    diagnostics.update(path_range=[left,right],pid=os.getpid(),cpu_seconds=process_time()-before)
    return arrays,anchor,diagnostics


def selected_samples(bundle, prior_control, grid_control, tail_control, target_control, models, coefficients, *, seed, paths, indices=None, forbidden_seeds=(), progress=None, workers=2, workspace=None, pure_chunk_paths=4096):
    """Serial full RNG propagation, then at most two pure-state work processes.

    The standalone runner pins BLAS threads to one before importing NumPy. Fork
    avoids serializing immutable MappingProxy-backed model objects; child work
    has no random draws and uses contiguous fixed path-index partitions.
    """
    import os
    import multiprocessing as mp
    from time import perf_counter
    if isinstance(pure_chunk_paths,bool) or not isinstance(pure_chunk_paths,Integral) or pure_chunk_paths<64 or pure_chunk_paths%64:raise ValueError('Pure chunk paths must be a positive64multiple')
    if workers not in (1,2) or isinstance(workers,bool):raise ValueError('Exactly one or two pure-state workers supported')
    forbidden = set(forbidden_seeds)|set(bundle['manifest']['all_selection_training_seeds'])
    if seed in forbidden:raise ValueError('A valuation seed reuses previous training/selection/valuation')
    if workers==2 and (mp.current_process().name!='MainProcess' or 'fork' not in mp.get_all_start_methods()
            or any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))):
        raise ValueError('Pure-state fork workers require standalone main process and explicit single BLAS threads')
    before=perf_counter()
    trace,conditional,physical=selected_physical_trace(bundle['proxy'],seed=seed,paths=paths,indices=indices,progress=progress,workspace=workspace)
    physical_seconds=perf_counter()-before
    if workspace is not None:
        for value in list(trace.values())+list(conditional.values()):
            if isinstance(value,np.memmap):value.flush()
    arguments=(bundle,prior_control,grid_control,tail_control,target_control,models,coefficients)
    n=len(trace['stocks']);before=perf_counter()
    if workers==1:
        arrays,anchor,diagnostic=_finish_selected(*arguments,trace,conditional,forbidden_seeds=forbidden_seeds,progress=progress)
        chunks=[diagnostic];ranges=[[0,n]]
    else:
        global _PURE_WORK
        # Full streams use4096-path chunks, preserving each64-row work batch.
        # The small255-row spent panel deliberately checks a128/127 partition.
        if n<=pure_chunk_paths:
            split=(n+1)//2;ranges=[[0,split],[split,n]]
        else:ranges=[[left,min(left+pure_chunk_paths,n)] for left in range(0,n,pure_chunk_paths)]
        _PURE_WORK=(arguments,trace,conditional,progress,tuple(forbidden_seeds))
        common={'times','discounts','eligible','actual_time_grid'}
        arrays={};chunks=[];anchor=_zeros(workspace,'selected_anchor',(n,))
        shared_trace={k:trace[k] for k in ('stocks','factors','european_proxy')}
        shared_trace.update({k:trace[k][:,:,0] for k in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale')})
        shared_trace['payoff']=trace['discounted_payoffs'][:,:,0]
        try:
            with mp.get_context('fork').Pool(2) as pool:
                for offset in range(0,len(ranges),2):
                    results=pool.map(_pure_chunk,ranges[offset:offset+2])
                    for bounds,result in zip(ranges[offset:offset+2],results):
                        left,right=bounds;values,a,diagnostic=result
                        for k,v in values.items():
                            if k in common:
                                if k not in arrays:arrays[k]=v
                                elif not np.array_equal(arrays[k],v):raise ValueError('Parallel common calendar changed')
                            elif k in shared_trace:
                                if not np.array_equal(v,shared_trace[k][left:right]):raise ValueError('Pure work changed physical primitives')
                                arrays[k]=shared_trace[k]
                            else:
                                if k not in arrays:arrays[k]=_zeros(workspace,'selected_'+k,(n,)+v.shape[1:],v.dtype)
                                arrays[k][left:right]=v
                        anchor[left:right]=a;chunks.append(diagnostic)
                    del results,result,values,a
        finally:_PURE_WORK=None
        if any(c['basis_names']!=chunks[0]['basis_names'] for c in chunks):raise ValueError('Parallel feature order changed')
    conditional['shared_control']=anchor
    return arrays,conditional,dict(physical=physical,physical_seconds=physical_seconds,pure_work_wall_seconds=perf_counter()-before,
        pure_state_workers=workers,pure_chunk_paths=pure_chunk_paths,file_backed_workspace=workspace is not None,path_partitions=ranges,chunks=chunks,basis_names=chunks[0]['basis_names'],
        no_random_or_fitting_in_children=True,all_paths_retained=indices is None,optimal_value_ready=False)

__all__ += ['selected_samples']
