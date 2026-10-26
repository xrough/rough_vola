"""Stock-bump validation of frozen shared rough-Heston price controls.

This is a successor experiment module.  The preceding numerical sources and
their serialized experiments remain unchanged.
"""
from __future__ import annotations

import math
from numbers import Integral
import numpy as np
from scipy.special import expit, ndtri

from rvlab.pricing.rough_simulation import (
    _theta, _drift_flow_with_integral, _weak_transition_branches,
    _nonnegative_clock,
)


def composed_law(policy, factors, *, times):
    """Exact finite mixture of one through four physical weak increments.

    The factor branch and stock-order trees are retained, rather than replacing
    their distinct Gaussian clocks by an average.  A four-increment stochastic
    block has 81 factor branches and 16 equally weighted stock orders.
    """
    factors = np.asarray(factors, float)
    times = np.asarray(times, float)
    if (factors.ndim != 2 or len(factors) < 1
            or factors.shape[1] != len(policy.kernel.nodes)
            or not np.isfinite(factors).all() or times.ndim != 1
            or not 2 <= len(times) <= 5 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Finite full-factor states and one to four physical increments required")
    if policy.market.dividends.events:
        raise ValueError("Composed controls require an explicitly empty cash schedule")
    params = policy.params
    nodes = np.asarray(policy.kernel.nodes)
    weights = np.asarray(policy.kernel.weights)
    count = len(nodes)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    n = len(factors)
    branches = factors[:, None, :]
    probability = np.ones((n, 1))
    mean = np.zeros((n, 1, 1))
    variance = np.zeros_like(mean)
    for left, right in zip(times[:-1], times[1:]):
        b, orders = branches.shape[1], mean.shape[2]
        old = branches.reshape(n*b, count)
        dt = float(right-left)
        theta = _theta(params, float((left+right)/2))
        flow = _drift_flow_with_integral(
            matrix, np.full(count, params.kappa*(theta-params.v0)),
            weights, params.v0, dt if params.vol_of_vol == 0 else dt/2)
        growth = math.log(policy.market.discount_curve.discount(left)
                          / policy.market.discount_curve.discount(right))
        if params.vol_of_vol == 0:
            new = (old@flow[:count, :count].T+flow[:count, -1]).reshape(n, b, 1, count)
            clock = _nonnegative_clock(old@flow[count, :count]+flow[count, -1]).reshape(n, b, 1, 1)
            corr = np.full((n, b, 1), growth)
            p = np.ones((n, b, 1))
        else:
            law = _weak_transition_branches(
                old, params, nodes=nodes, weights=weights,
                flow=flow[:count, :count], shift=flow[:count, -1],
                integrated_flow=flow, dt=dt, theta=theta, growth=growth,
                stock_integral=policy.config.stock_integral)
            new = law['new_factors'].reshape(n, b, 3, count)
            clock = law['orthogonal'].reshape(n, b, 3, 2)
            corr = law['correlated'].reshape(n, b, 3)
            p = law['probabilities'].reshape(n, b, 3)
        nb, no = new.shape[2], clock.shape[-1]
        mean = (mean[:, :, None, :, None]+corr[:, :, :, None, None]
                -.5*clock[:, :, :, None, :]).reshape(n, b*nb, orders*no)
        variance = (variance[:, :, None, :, None]
                    +clock[:, :, :, None, :]).reshape(n, b*nb, orders*no)
        probability = (probability[:, :, None]*p).reshape(n, b*nb)
        branches = new.reshape(n, b*nb, count)
    return dict(new_factors=branches, probabilities=probability,
                log_stock_shifts=mean, orthogonal=variance,
                endpoint_probability=1./mean.shape[2],
                physical_steps=len(times)-1)


RELATIVE_SPOTS = np.array([-.005, -.0025, 0., .0025, .005])
PROCESS_NAMES = ('target0-minus-latest', 'target1-minus-latest',
                 'target2-minus-latest', 'target3-minus-latest',
                 'european', 'premium', 'intrinsic', 'micro_boundary')
PANEL_IDS = (0, 1, 511, 512)
PANEL_DATES = (16, 32, 48, 63)


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _actual_splines(coefficient, stocks, *, strike, knots):
    """Evaluate N×family linear splines at N×stock states."""
    z = np.asarray(stocks)/strike
    index = np.searchsorted(knots, z, side='right')[:, None, :, None]
    c0 = np.take_along_axis(coefficient[..., 0][:, :, None], index, axis=-1)[..., 0]
    c1 = np.take_along_axis(coefficient[..., 1][:, :, None], index, axis=-1)[..., 0]
    return (c0+c1*z[:, None]).transpose(0, 2, 1)


def _expected_splines(mean, clock, coefficient, *, strike, knots, moments=None):
    from ._pricing_rough_mc_precision import shared_expectations
    n, branches, orders, bumps = mean.shape
    m = mean.reshape(n, branches, orders*bumps)
    v = np.broadcast_to(clock, mean.shape).reshape(m.shape)
    values = shared_expectations(m, v, coefficient, strike=strike,
                                 knots=knots, moments=moments)
    return values.reshape(n, branches, coefficient.shape[2], orders, bumps)


def physical_trace(bundle, *, seed, paths, steps, indices=None, progress=None):
    """One full512-batch stream, five stock states and conditional histories.

    Branch/order/Gaussian uniforms are common to the five initial stocks. The
    complete full-stream RNG is consumed even for a spent selected-path probe.
    Actual-physical-grid affine functions center the physical martingales and
    the conditional European anchor. Calendar decision values are evaluated
    separately using the frozen original129-point function.
    """
    from rvlab.pricing.rough_estimators import _AffineSurrogate, _conditional_payoffs
    from rvlab.pricing.rough_simulation import _time_grid
    from rvlab.pricing.rough_stopping import _payoff
    from rvlab.pricing.rough_stopping_fourier import (
        _affine_values, _interpolate, _premium_values,
        _continuation_coefficients, fourier_policy_fingerprint)
    proxy = bundle['proxy']
    for value, name, minimum in ((seed, 'seed', 0), (paths, 'paths', 2)):
        _integer(value, name, minimum)
    if (steps not in (128, 256) or isinstance(steps, bool)
            or proxy.config.steps != 128 or proxy.config.batch_size != 512
            or len(proxy.exercise_times) != 65 or proxy.params.vol_of_vol <= 0
            or proxy.market.dividends.events):
        raise ValueError('Frozen stochastic no-cash128/256 outer law and64 exercise intervals required')
    wanted = np.arange(paths) if indices is None else np.asarray(indices)
    if (wanted.ndim != 1 or wanted.dtype.kind not in 'iu' or not len(wanted)
            or np.any(wanted < 0) or np.any(wanted >= paths)
            or np.any(np.diff(wanted) <= 0)):
        raise ValueError('Increasing absolute full-stream path indices required')
    params, market, kernel = proxy.params, proxy.market, proxy.kernel
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    count = len(nodes)
    times = np.asarray(proxy.exercise_times)
    grid = _time_grid(params, market, times, steps)
    original_grid = _time_grid(params, market, times, 128)
    args = (params, market, kernel, np.array([proxy.strike]),
            np.array([proxy.maturity]), np.array([proxy.is_call]))
    affine = _AffineSurrogate(*args, grid, proxy.control_config)
    calendar_affine = affine if steps == 128 else _AffineSurrogate(*args, original_grid, proxy.control_config)
    spots = market.spot*(1+RELATIVE_SPOTS)
    n, nt, nb, np_ = len(wanted), len(times), len(spots), len(grid)
    stocks = np.broadcast_to(spots, (paths, nb)).copy()
    factors = np.zeros((paths, count))
    trace = {k:np.zeros((n, nt, nb)) for k in (
        'stocks', 'european_proxy', 'discounted_payoffs', 'european_martingale',
        'premium_martingale', 'intrinsic_martingale', 'micro_boundary_martingale')}
    trace.update(times=times, factors=np.zeros((n, nt, count)),
        discounts=market.discount_curve.discount(times), eligible=np.ones(nt, bool),
        initial_spots=spots, actual_time_grid=grid, seed=int(seed),
        policy_fingerprint=fourier_policy_fingerprint(proxy),
        kernel_fingerprint=kernel.fingerprint, strike=proxy.strike,
        exercise_style='american')
    trace['stocks'][:, 0] = spots
    trace['discounted_payoffs'][:, 0] = _payoff(spots, proxy.strike, proxy.is_call)
    zero = _affine_values(affine, 0., np.zeros((1, 1, count)),
        np.log(spots)[None, None, None], np.zeros((1, 1, 1, nb)))[0, 0, 0]
    calendar_zero = zero if steps == 128 else _affine_values(
        calendar_affine, 0., np.zeros((1, 1, count)), np.log(spots)[None, None, None],
        np.zeros((1, 1, 1, nb)))[0, 0, 0]
    trace['european_proxy'][:, 0] = calendar_zero
    physical = dict(absolute_path_indices=wanted, physical_time_grid=grid,
        physical_stocks=np.zeros((n, np_, nb)),
        physical_factors=np.zeros((n, np_, count)),
        conditional_log_means=np.zeros((n, np_, nb)),
        conditional_variances=np.zeros((n, np_)),
        conditional_expected_next=np.zeros((n, np_, nb)),
        conditional_actual_next=np.zeros((n, np_, nb)))
    physical['physical_stocks'][:, 0] = spots
    mu = np.broadcast_to(np.log(spots), (n, nb)).copy()
    variance = np.zeros(n)
    cm = np.zeros((n, nb))
    physical['conditional_log_means'][:, 0] = mu
    em, total, im, micro = [np.zeros((n, nb)) for _ in range(4)]
    left_euro = np.broadcast_to(zero, (n, nb)).copy()
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    rng = np.random.default_rng(seed)
    cache, dateindex = {}, {float(t):j for j, t in enumerate(times)}
    for k, (left, right) in enumerate(zip(grid[:-1], grid[1:]), 1):
        if progress is not None and (k-1) % 16 == 0:
            progress('physical_increment', k-1, len(grid)-1)
        dt, theta = float(right-left), _theta(params, .5*(left+right))
        if (dt, theta) not in cache:
            cache[(dt, theta)] = _drift_flow_with_integral(matrix,
                np.full(count, params.kappa*(theta-params.v0)), weights, params.v0, dt/2)
        flow = cache[(dt, theta)]
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        discount = float(market.discount_curve.discount(right))
        for begin in range(0, paths, 512):
            end = min(begin+512, paths)
            old = factors[begin:end]
            random = rng.random((end-begin, 3))
            law = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
                flow=flow[:count, :count], shift=flow[:count, -1], integrated_flow=flow,
                dt=dt, theta=theta, growth=growth, stock_integral=proxy.config.stock_integral)
            p = law['probabilities']
            choice = (random[:, 0] > p[:, 0]).astype(int)+(random[:, 0] > p[:, :2].sum(axis=1)).astype(int)
            endpoint = (random[:, 1] >= .5).astype(int)
            row = np.arange(end-begin)
            clock = law['orthogonal'][..., None]
            mean = np.log(stocks[begin:end])[:, None, None]+law['correlated'][:, :, None, None]-.5*clock
            normal = ndtri(np.clip(random[:, 2], np.finfo(float).tiny, 1-np.finfo(float).eps))
            nextstock = np.exp(mean[row, choice, endpoint]
                +np.sqrt(clock[row, choice, endpoint])*normal[:, None])
            new = law['new_factors'][row, choice]
            chosen = np.flatnonzero((wanted >= begin)&(wanted < end))
            if len(chosen):
                local = wanted[chosen]-begin
                branch, m, v, probability = law['new_factors'][local], mean[local], clock[local], p[local]
                actualstock, actualfactor = nextstock[local], new[local]
                expected = np.sum(_affine_values(affine, float(right), branch, m, v)
                    .mean(axis=2)*probability[..., None], axis=1)
                actual = _affine_values(affine, float(right), actualfactor[:, None],
                    np.log(actualstock)[:, None, None], np.zeros((len(local), 1, 1, nb)))[:, 0, 0]
                ei = actual-expected
                em[chosen] += ei
                intrinsic = _conditional_payoffs(m, v, proxy.strike, proxy.is_call)
                expected_intrinsic = np.sum(intrinsic.mean(axis=2)*probability[..., None], axis=1)
                ii = discount*(_payoff(actualstock, proxy.strike, proxy.is_call)-expected_intrinsic)
                im[chosen] += ii
                q = _premium_values(stocks[wanted[chosen]],
                    _continuation_coefficients(proxy, float(left), old[local])[:, None],
                    strike=proxy.strike, stock_knots=proxy.continuations[0].knots, cap=proxy.premium_cap)
                margin = market.discount_curve.discount(left)*(_payoff(stocks[wanted[chosen]], proxy.strike, proxy.is_call)-q)-left_euro[chosen]
                micro[chosen] += expit(margin/(market.discount_curve.discount(left)*.05))*(ii-ei)
                left_euro[chosen] = actual
                coeff = _interpolate(proxy, float(right), branch)[:, :, None]
                nodal = coeff[..., 0]+coeff[..., 1]*np.r_[0., proxy.config.stock_knots]
                if (np.any(coeff[..., 2] != 0) or np.any(coeff[..., -1, 1] != 0)
                        or np.max(-nodal) > 2e-13
                        or np.max(nodal-proxy.premium_cap) > 2e-13):
                    raise ValueError('Physical premium must be bounded linear with constant tails')
                pv = _expected_splines(m, v, coeff, strike=proxy.strike, knots=proxy.config.stock_knots)[:, :, 0]
                pe = np.sum(pv.mean(axis=2)*probability[..., None], axis=1)
                pa = _premium_values(actualstock, _interpolate(proxy, float(right), actualfactor)[:, None],
                    strike=proxy.strike, stock_knots=proxy.config.stock_knots, cap=proxy.premium_cap)
                total[chosen] += ei+discount*(pa-pe)
                cmean = mu[chosen, None, None]+law['correlated'][local, :, None, None]-.5*v
                cclock = variance[chosen, None, None, None]+v
                ce = np.sum(_affine_values(affine, float(right), branch, cmean, cclock)
                    .mean(axis=2)*probability[..., None], axis=1)
                mu[chosen] = cmean[np.arange(len(local)), choice[local], endpoint[local]]
                variance[chosen] = cclock[np.arange(len(local)), choice[local], endpoint[local], 0]
                ca = _affine_values(affine, float(right), actualfactor[:, None],
                    mu[chosen, None, None], variance[chosen, None, None, None])[:, 0, 0]
                cm[chosen] += ca-ce
                physical['conditional_expected_next'][chosen, k] = ce
                physical['conditional_actual_next'][chosen, k] = ca
                physical['conditional_log_means'][chosen, k] = mu[chosen]
                physical['conditional_variances'][chosen, k] = variance[chosen]
                physical['physical_stocks'][chosen, k] = actualstock
                physical['physical_factors'][chosen, k] = actualfactor
                if float(right) in dateindex:
                    j = dateindex[float(right)]
                    trace['stocks'][chosen, j] = actualstock
                    trace['factors'][chosen, j] = actualfactor
                    trace['discounted_payoffs'][chosen, j] = discount*_payoff(actualstock, proxy.strike, proxy.is_call)
                    calendar_value = actual if steps == 128 else _affine_values(
                        calendar_affine, float(right), actualfactor[:, None],
                        np.log(actualstock)[:, None, None], np.zeros((len(local), 1, 1, nb)))[:, 0, 0]
                    trace['european_proxy'][chosen, j] = calendar_value
                    for name, value in (('european_martingale', em), ('premium_martingale', total-em),
                            ('intrinsic_martingale', im), ('micro_boundary_martingale', micro)):
                        trace[name][chosen, j] = value[chosen]
            stocks[begin:end], factors[begin:end] = nextstock, new
    cq = float(market.discount_curve.discount(proxy.maturity))*_conditional_payoffs(
        mu, variance[:, None], proxy.strike, proxy.is_call)
    physical.update(conditional_european_payoff=cq, conditional_european_martingale=cm,
                    conditional_european_estimate=cq-cm)
    return trace, physical


def right_function_primitives(proxy, selected, models, affine, node_proxy,
        j, law, leftstocks, rightstocks, rightfactors):
    """Actual/expected frozen calendar-right functions, N×5×8 columns.

    Columns are local, grid4, grid8, four targets, discounted European.
    Expectations use the supplied actual physical transition mixture.
    """
    from ._pricing_rough_mc_precision import gaussian_moments
    from ._pricing_rough_local_control import _correction_coefficients, _linear_coefficients
    from ._pricing_rough_node_refinement_fast import refined_stock_knots, snell_refinement_coefficients
    from ._pricing_rough_bellman_control_aligned import target_nodal_premiums
    from rvlab.pricing.rough_stopping_fourier import _affine_values
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    knots = nodes[1:]
    branch, p = law['new_factors'], law['probabilities']
    n, b, orders = law['orthogonal'].shape
    mean = np.log(leftstocks)[:, None, None]+law['log_stock_shifts'][..., None]
    clock = law['orthogonal'][..., None]
    mf = mean.reshape(n, b, orders*5)
    vf = np.broadcast_to(clock, mean.shape).reshape(mf.shape)
    moments = gaussian_moments(mf, vf, strike=proxy.strike, knots=knots)
    gc, _ = snell_refinement_coefficients(proxy, selected, j, branch, fine_nodes=nodes)
    tc = _linear_coefficients(nodes, target_nodal_premiums(proxy, selected, models,
        j, branch, fine_nodes=nodes, node_proxy=node_proxy))
    lc = _correction_coefficients(proxy, selected, j, branch, 3.)[:, :, None]
    def expected(coefficient, stockknots, shared=None):
        value = _expected_splines(mean, clock, coefficient, strike=proxy.strike,
                                 knots=stockknots, moments=shared)
        return np.sum(value.mean(axis=3)*p[..., None, None], axis=1).transpose(0, 2, 1)
    le = expected(lc, proxy.config.stock_knots)
    ge = expected(gc, knots, moments)
    te = expected(tc, knots, moments)
    ag, _ = snell_refinement_coefficients(proxy, selected, j, rightfactors, fine_nodes=nodes)
    at = _linear_coefficients(nodes, target_nodal_premiums(proxy, selected, models,
        j, rightfactors, fine_nodes=nodes, node_proxy=node_proxy))
    al = _correction_coefficients(proxy, selected, j, rightfactors, 3.)[:, None]
    la = _actual_splines(al, rightstocks, strike=proxy.strike, knots=proxy.config.stock_knots)
    ga = _actual_splines(ag, rightstocks, strike=proxy.strike, knots=knots)
    ta = _actual_splines(at, rightstocks, strike=proxy.strike, knots=knots)
    right = float(proxy.continuation_times[j])
    ee = np.sum(_affine_values(affine, right, branch, mean, clock)
        .mean(axis=2)*p[..., None], axis=1)
    ea = _affine_values(affine, right, rightfactors[:, None],
        np.log(rightstocks)[:, None, None], np.zeros((n, 1, 1, 5)))[:, 0, 0]
    actual = np.concatenate((la, ga, ta, ea[..., None]), axis=-1)
    expected_values = np.concatenate((le, ge, te, ee[..., None]), axis=-1)
    if not np.isfinite(actual).all() or not np.isfinite(expected_values).all():
        raise FloatingPointError('Frozen right-state functions are nonfinite')
    return actual, expected_values


def calendar_processes(bundle, models, trace, physical, *, representation, progress=None):
    """Original two-step block or distinct exact-centered microstep control.

    Microstep increments use the same calendar-right function throughout a
    block, and the gate known at its calendar LEFT state. Their sum is not the
    original block martingale and is never described as a telescoping bridge.
    """
    from ._pricing_rough_bellman_control_aligned import aligned_node_proxy
    from rvlab.pricing.rough_estimators import _AffineSurrogate
    from rvlab.pricing.rough_simulation import _time_grid
    proxy, selected = bundle['proxy'], bundle['selected_policy']
    times, grid = trace['times'], trace['actual_time_grid']
    if representation not in ('original', 'micro') or (representation == 'original' and len(grid) != 129):
        raise ValueError('Original128 or explicitly distinct microstep representation required')
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel,
        np.array([proxy.strike]), np.array([proxy.maturity]), np.array([proxy.is_call]),
        _time_grid(proxy.params, proxy.market, times, 128), proxy.control_config)
    node_proxy = aligned_node_proxy(proxy, affine)
    n = len(trace['stocks'])
    local = np.zeros((n, 65, 5, 3))
    refined = np.zeros((n, 65, 5, 6))
    targets = np.zeros((n, 65, 5, 4))
    panel_positions = np.flatnonzero(np.isin(physical['absolute_path_indices'], PANEL_IDS))
    panel = dict(indices=physical['absolute_path_indices'][panel_positions], records=[])
    for j, (left, right) in enumerate(zip(times[:-1], times[1:])):
        if progress is not None and j % 8 == 0:
            progress(representation+'_calendar_interval', j, 64)
        a, b = np.searchsorted(grid, [left, right])
        ranges = [(a, b)] if representation == 'original' else [(k, k+1) for k in range(a, b)]
        actual_sum = np.zeros((n, 5, 8))
        expected_sum = np.zeros_like(actual_sum)
        for start, end in ranges:
            law = composed_law(proxy, physical['physical_factors'][:, start], times=grid[start:end+1])
            actual, expected = right_function_primitives(proxy, selected, models, affine,
                node_proxy, j+1, law, physical['physical_stocks'][:, start],
                physical['physical_stocks'][:, end], physical['physical_factors'][:, end])
            actual_sum += actual
            expected_sum += expected
            if j in PANEL_DATES:
                panel['records'].append(dict(calendar_index=j, physical_left=int(start),
                    physical_right=int(end), actual=actual[panel_positions], expected=expected[panel_positions]))
        s, u = trace['stocks'][:, j], trace['factors'][:, j]
        q = selected.continuations[j].value(s, u, proxy.strike)
        q = np.maximum(q, 0.) if selected.premium_cap is None else np.clip(q, 0., selected.premium_cap)
        margin = trace['discounts'][j]*(np.maximum((1 if proxy.is_call else -1)*(s-proxy.strike), 0.)-q)-trace['european_proxy'][:, j]
        gates = np.stack((np.ones_like(margin), np.exp(-.5*(margin/.1)**2), np.exp(-.5*(margin/.3)**2)), axis=-1)
        delta = actual_sum-expected_sum
        d = float(trace['discounts'][j+1])
        local[:, j+1] = local[:, j]+d*delta[..., 0, None]*gates
        refined[:, j+1] = refined[:, j]+(d*delta[..., 1:3, None]*gates[..., None, :]).reshape(n, 5, 6)
        targets[:, j+1] = targets[:, j]+delta[..., 7, None]+d*delta[..., 3:7]
    return dict(local=local, grid=refined, targets=targets), panel


def _bin_weighted(process, times, control):
    from ._pricing_rough_local_control import correction_basis
    n, nt, nb, nf = process.shape
    flat = process.transpose(0, 2, 1, 3).reshape(n*nb, nt, nf)
    basis = correction_basis(flat, times, control['bins'])
    return np.einsum('ntd,d->nt', basis, control['weights']).reshape(n, nb, nt).transpose(0, 2, 1)


def finish_representation(bundle, controls, trace, physical, processes, stops, coefficients):
    """Fixed old control algebra and own-bump predictable stopped features."""
    from rvlab.pricing.rough_dual import _weighted_martingale
    from rvlab.pricing.rough_policy_diagnostics import _policy_stops
    from ._pricing_rough_price_budget_controls import predictable_basis
    proxy, selected = bundle['proxy'], bundle['selected_policy']
    prior_control, grid_control, tail_control, target_control = controls
    stable = _weighted_martingale(trace, bundle['weights'])
    local = _bin_weighted(processes['local'], trace['times'], prior_control)
    grid = _bin_weighted(processes['grid'], trace['times'], grid_control)
    current = stable+local+grid
    baseline_stops = _policy_stops(selected, trace['times'], trace['eligible'],
        trace['stocks'], trace['factors'], trace['european_proxy'])
    margin = np.empty_like(current)
    for j in range(65):
        q = selected.continuations[j].value(trace['stocks'][:, j], trace['factors'][:, j], proxy.strike)
        q = np.maximum(q, 0.) if selected.premium_cap is None else np.clip(q, 0., selected.premium_cap)
        margin[:, j] = (trace['discounted_payoffs'][:, j]-trace['european_proxy'][:, j])/trace['discounts'][j]-q
    left = margin[:, :-1]
    post = (np.arange(64)[None, :, None] >= baseline_stops[:, None]).astype(float)
    gates = np.stack((post, post*expit(left/.1), post*expit(left/.3),
        expit(left/.03), expit(left/.1), expit(left/.3),
        (trace['discounted_payoffs'][:, :-1] == 0).astype(float)), axis=-1)
    im = np.diff(trace['intrinsic_martingale']-current, axis=1)
    em = np.diff(trace['european_martingale']-current, axis=1)
    increments = gates*np.concatenate((np.repeat(im[..., None], 6, axis=-1), em[..., None]), axis=-1)
    tail_process = np.concatenate((np.zeros((len(current), 1, 5, 7)), np.cumsum(increments, axis=1)), axis=1)
    tail = _bin_weighted(tail_process, trace['times'], tail_control)
    latest = current+tail
    target_difference = processes['targets']-latest[..., None]
    selected_m = latest+_bin_weighted(target_difference, trace['times'], target_control)
    source = np.concatenate((target_difference,
        np.stack([trace[k] for k in ('european_martingale', 'premium_martingale',
            'intrinsic_martingale', 'micro_boundary_martingale')], axis=-1)), axis=-1)
    anchor = trace['discounted_payoffs'][:, -1]-trace['european_martingale'][:, -1]-physical['conditional_european_estimate']
    bases = []
    for bump in range(5):
        perstock = dict(stocks=trace['stocks'][:, :, bump:bump+1], factors=trace['factors'],
            target_martingales=processes['targets'][:, :, bump], latest_martingale=latest[:, :, bump],
            gh16_stops=stops[:, bump])
        perstock.update({k:trace[k][:, :, bump] for k in (
            'european_martingale', 'premium_martingale', 'intrinsic_martingale', 'micro_boundary_martingale')})
        basis, names = predictable_basis(perstock, bundle, include_anchor=True, anchor=anchor[:, bump])
        bases.append(basis)
    basis = np.stack(bases, axis=1)
    coefficient = np.asarray(coefficients, float)
    if coefficient.shape != (193,) or not np.isfinite(coefficient).all() or np.any(abs(coefficient) > 2):
        raise ValueError('Frozen193 bounded raw coefficients required')
    shift = basis@coefficient
    rows, bumps = np.arange(len(stops))[:, None], np.arange(5)[None]
    raw = trace['discounted_payoffs'][rows, stops, bumps]
    lower = raw-selected_m[rows, stops, bumps]
    upper = np.max(trace['discounted_payoffs']-selected_m, axis=1)
    return dict(processes=source, latest_martingale=latest, selected_martingale=selected_m,
        stage_corrections=np.stack((local, grid, tail), axis=-1),
        basis=basis, anchor=anchor, shift=shift, raw=raw,
        stable=raw-stable[rows, stops, bumps], unchanged_lower=lower,
        shared_lower=lower-shift, unchanged_upper=upper, shared_upper=upper-shift), names


def shared_delta_samples(bundle, prior_control, grid_control, tail_control,
        target_control, models, coefficients, *, seed, paths, steps,
        indices=None, forbidden_seeds=(), progress=None, work_batch=32):
    """Original128 plus micro128, or micro256; no fitting or path selection."""
    from ._pricing_rough_bellman_delta import bellman_policy_stops
    from ._pricing_rough_bellman_policy import BellmanEvaluator
    forbidden = set(forbidden_seeds)|set(bundle['manifest']['all_selection_training_seeds'])
    if seed in forbidden:
        raise ValueError('Independent seed must not reuse prior fitting/valuation streams')
    _integer(work_batch, 'work_batch', 1)
    trace, physical = physical_trace(bundle, seed=seed, paths=paths, steps=steps,
        indices=indices, progress=progress)
    evaluator = BellmanEvaluator(bundle['proxy'], bundle['selected_policy'], orders=(16,))
    looked = bellman_policy_stops(trace, evaluator, outer_steps=steps, forbidden_seeds=tuple(forbidden))
    n = len(trace['stocks'])
    arrays = {k:trace[k] for k in ('times', 'discounts', 'eligible', 'actual_time_grid',
        'stocks', 'factors')}
    arrays.update(relative_spots=RELATIVE_SPOTS, initial_spots=trace['initial_spots'],
        operator_time_grid=evaluator.grid, payoff=trace['discounted_payoffs'],
        calendar_european=trace['european_proxy'], gh16_stops=looked['stops'],
        gh16_continuation=looked['continuation'], gh16_computed_mask=looked['computed_mask'])
    arrays.update(physical)
    representations = ('original', 'micro') if steps == 128 else ('micro',)
    audits = {r:[] for r in representations}
    controls = (prior_control, grid_control, tail_control, target_control)
    pathfields = ('stocks', 'factors', 'european_proxy', 'discounted_payoffs',
        'european_martingale', 'premium_martingale', 'intrinsic_martingale', 'micro_boundary_martingale')
    physical_common = {'physical_time_grid'}
    for left in range(0, n, work_batch):
        right = min(left+work_batch, n)
        part = {k:(v[left:right] if k in pathfields else v) for k, v in trace.items()}
        q = {k:(v if k in physical_common else v[left:right]) for k, v in physical.items()}
        for representation in representations:
            process, panel = calendar_processes(bundle, models, part, q,
                representation=representation, progress=progress)
            result, names = finish_representation(bundle, controls, part, q,
                process, looked['stops'][left:right], coefficients)
            for key, value in result.items():
                dest = representation+'_'+key
                if dest not in arrays:
                    arrays[dest] = np.empty((n,)+value.shape[1:], dtype=value.dtype)
                arrays[dest][left:right] = value
            if len(panel['indices']):
                audits[representation].append(panel)
        if progress is not None:
            progress('completed_pure_paths', right, n)
    # The audit is a fixed primitive panel, not an extra Monte Carlo sample.
    diagnostic = dict(seed=int(seed), paths=int(paths), evaluated_paths=n,
        steps=int(steps), representations=list(representations), basis_names=names,
        process_names=list(PROCESS_NAMES), physical_random_batch_size=512,
        all_random_draws_consumed=True, all_paths_retained=indices is None,
        policy_operator_steps=128, exercise_intervals=64,
        all_bumps_recompute_continuations_and_stops=True,
        original_and_micro_are_distinct_centered_controls=True,
        fine_physical_centers_use_actual_law=True,
        calendar_european_function_grid_points=129,
        physical_european_function_grid_points=len(trace['actual_time_grid']),
        no_fitting_or_selection=True, no_empirical_valuation_recentering=True,
        optimal_delta_ready=False, tolerances_changed=False)
    return arrays, audits, diagnostic


__all__ = ['composed_law', 'RELATIVE_SPOTS', 'PROCESS_NAMES', 'PANEL_IDS',
           'PANEL_DATES', 'physical_trace', 'right_function_primitives',
           'calendar_processes', 'finish_representation', 'shared_delta_samples']
