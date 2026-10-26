"""Independent saved-path audit for the selected-only precision executor.

No omitted margin-policy or unselected price table is invented.  All retained
component arithmetic, own stopping decisions, common-control features and
conditional Gaussian random-stream primitives are checked on every path.
Conditional function values receive bounded explicit re-evaluation, rather
than claiming an exhaustive second pricing run.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import expit

from rvlab.pricing.rough_simulation import _time_grid
from ._pricing_rough_price_budget_basis import reconstruct_predictable_basis
from ._pricing_rough_price_budget_samples import validate_conditional_european


def _close(actual, expected, message, *, atol=2e-13):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, rtol=0., atol=atol):
        raise ValueError(message)


def _weighted(corrections, control):
    n, dates, features = corrections.shape
    bins = control.get('bins')
    weights = np.asarray(control.get('weights'), dtype=float)
    if (isinstance(bins, bool) or not isinstance(bins, Integral) or bins not in (1, 4)
            or dates != 65 or weights.shape != (bins*features,)
            or not np.isfinite(weights).all() or np.any(weights < -1e-12)
            or np.any(weights > 1.+1e-12)
            or np.any(weights.reshape(bins, features).sum(axis=1) > 1.+1e-12)):
        raise ValueError('A frozen control changed coefficient dimensions or its per-bin budget')
    rows = weights.reshape(bins, features)[np.arange(64)*bins//64]
    increment = np.sum(np.diff(corrections, axis=1)*rows, axis=-1)
    return np.column_stack((np.zeros(n), np.cumsum(increment, axis=1)))


def _linear_actual(coefficients, stocks, strike, knots):
    z = stocks/strike
    index = np.searchsorted(knots, z, side='right')
    if coefficients.ndim == 3:
        chosen = coefficients[np.arange(len(stocks)), index]
        return chosen[:, 0] + chosen[:, 1]*z
    chosen = coefficients[np.arange(len(stocks))[:, None], np.arange(coefficients.shape[1])[None, :], index[:, None]]
    return chosen[..., 0] + chosen[..., 1]*z[:, None]


def _bounded_probes(arrays, bundle, models):
    """Separate reference functions, not the new shared-moment executor."""
    from ._pricing_rough_bellman_control_aligned import aligned_node_proxy, target_nodal_premiums
    from ._pricing_rough_bellman_policy import BellmanEvaluator
    from ._pricing_rough_exercise_block import exercise_block_law
    from ._pricing_rough_local_control import _correction_coefficients, _linear_coefficients, signed_linear_expectation
    from ._pricing_rough_node_refinement_fast import refined_stock_knots, snell_refinement_coefficients, signed_shared_expectations
    from rvlab.pricing.rough_estimators import _AffineSurrogate
    from rvlab.pricing.rough_stopping_fourier import _affine_values
    proxy = bundle['proxy']; selected = bundle['selected_policy']
    times = arrays['times']; physical = arrays['actual_time_grid']
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([proxy.strike]),
        np.array([proxy.maturity]), np.array([proxy.is_call]), physical, proxy.control_config)
    aligned = aligned_node_proxy(proxy, affine)
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    evaluator = BellmanEvaluator(proxy, selected, orders=(16,))
    indices = np.arange(min(2, len(arrays['stocks'])))
    rows = []
    for j in (16, 32, 48, 63):
        a, b = np.searchsorted(physical, times[j:j+2])
        law = exercise_block_law(proxy, arrays['factors'][indices, j], times=physical[a:b+1])
        new = law['new_factors']; p = law['probabilities']
        mean = np.log(arrays['stocks'][indices, j, 0])[:, None, None] + law['log_stock_shifts']
        clock = law['orthogonal']
        local = _correction_coefficients(proxy, selected, j+1, new, 3.)
        lp = signed_linear_expectation(mean, clock, local[:, :, None], strike=proxy.strike, knots=proxy.config.stock_knots)
        local_expected = np.sum(lp.mean(axis=-1)*p, axis=1)
        gc, _ = snell_refinement_coefficients(proxy, selected, j+1, new, fine_nodes=nodes)
        gp = signed_shared_expectations(mean, clock, gc, strike=proxy.strike, knots=nodes[1:])
        grid_expected = np.sum(gp.mean(axis=-1)*p[..., None], axis=1)
        tc = _linear_coefficients(nodes, target_nodal_premiums(proxy, selected, models, j+1, new,
            fine_nodes=nodes, node_proxy=aligned))
        tp = signed_shared_expectations(mean, clock, tc, strike=proxy.strike, knots=nodes[1:])
        target_expected = np.sum(tp.mean(axis=-1)*p[..., None], axis=1)
        expected_euro = np.sum(_affine_values(affine, float(times[j+1]), new,
            mean[..., None], clock[..., None])[..., 0].mean(axis=2)*p, axis=1)
        factors = arrays['factors'][indices, j+1]; stocks = arrays['stocks'][indices, j+1, 0]
        local_actual = _linear_actual(_correction_coefficients(proxy, selected, j+1, factors, 3.),
                                     stocks, proxy.strike, proxy.config.stock_knots)
        gc, _ = snell_refinement_coefficients(proxy, selected, j+1, factors, fine_nodes=nodes)
        grid_actual = _linear_actual(gc, stocks, proxy.strike, nodes[1:])
        tc = _linear_coefficients(nodes, target_nodal_premiums(proxy, selected, models, j+1, factors,
            fine_nodes=nodes, node_proxy=aligned))
        target_actual = _linear_actual(tc, stocks, proxy.strike, nodes[1:])
        errors = {}
        for name, expected in (('local_actual', local_actual), ('local_expected', local_expected),
                ('grid_actual', grid_actual), ('grid_expected', grid_expected),
                ('target_actual_premiums', target_actual), ('target_expected_premiums', target_expected),
                ('target_expected_european', expected_euro)):
            error = float(np.max(abs(arrays[name][indices, j+1]-expected)))
            if error > 2e-11:
                raise ValueError(f'A retained {name} failed its bounded independent reference probe')
            errors[name+'_error'] = error
        active = np.flatnonzero(arrays['gh16_computed_mask'][:, j])[:2]
        policy_error = 0.
        if len(active):
            c = evaluator.conditional_expectations(j, arrays['stocks'][active, j, 0],
                arrays['factors'][active, j])['continuation'][:, 0]
            policy_error = float(np.max(abs(c-arrays['gh16_continuation'][active, j])))
            if policy_error > 2e-11:
                raise ValueError('The frozen GH16 rule failed a bounded conditional-continuation probe')
        rows.append({'kind': 'selected conditional functions', 'left_exercise_index': j,
            'path_indices': indices.tolist(), 'policy_probe_states': len(active),
            'gh16_continuation_error': policy_error, **errors,
            'all_path_accumulation_checked': True, 'exhaustive_conditional_repricing': False})
    return rows


def validate_selected_samples(arrays, bundle, prior, grid, tail_seal, target_seal, *,
                              paths, seed, coefficients, models):
    """Audit merged selected, conditional and common-shift archives.

    The caller separately verifies frozen source/input seals and chronology.
    This function independently checks all path algebra and replays all RNG
    states via the earlier independently implemented conditional validator.
    """
    for value, name, lower in ((paths, 'paths', 2), (seed, 'seed', 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    proxy = bundle['proxy']; selected = bundle['selected_policy']; fit = bundle['weights']
    if (len(proxy.kernel.nodes) != 4 or proxy.config.steps != 128 or proxy.config.batch_size != 512
            or proxy.market.dividends.events or len(models) != 65):
        raise ValueError('Retain the frozen four-factor,128-step,512-random-batch no-cash model')
    shapes = {'times': (65,), 'discounts': (65,), 'eligible': (65,), 'actual_time_grid': (129,),
        'stocks': (paths, 65, 1), 'factors': (paths, 65, 4), 'european_proxy': (paths, 65, 1),
        'payoff': (paths, 65), 'prior_local_corrections': (paths, 65, 3),
        'grid_corrections': (paths, 65, 6), 'tail_corrections': (paths, 65, 7),
        'tail_gates': (paths, 64, 7), 'tail_margin': (paths, 65),
        'local_actual': (paths, 65), 'local_expected': (paths, 65),
        'grid_actual': (paths, 65, 2), 'grid_expected': (paths, 65, 2),
        'target_martingales': (paths, 65, 4), 'target_expected_european': (paths, 65),
        'target_actual_premiums': (paths, 65, 4), 'target_expected_premiums': (paths, 65, 4),
        'gh16_stops': (paths,), 'gh16_raw': (paths,), 'gh16_computed_mask': (paths, 65),
        'candidate3_lower': (paths,), 'candidate3_upper': (paths,),
        'unchanged_lower': (paths,), 'unchanged_upper': (paths,),
        'shared_lower': (paths,), 'shared_upper': (paths,), 'basis': (paths, 193),
        'shift': (paths,), 'shared_control': (paths,)}
    martingales = ('european_martingale', 'intrinsic_martingale', 'premium_martingale',
        'micro_boundary_martingale', 'stable_old_martingale', 'prior_martingale',
        'current_martingale', 'latest_martingale')
    shapes.update({name: (paths, 65) for name in martingales})
    if any(name not in arrays or np.shape(arrays[name]) != shape
           or not np.isfinite(arrays[name]).all() for name, shape in shapes.items()):
        raise ValueError('A retained selected path, primitive, feature or endpoint is missing or malformed')
    times = arrays['times']; discounts = arrays['discounts']; eligible = arrays['eligible']
    if (not np.array_equal(times, proxy.exercise_times) or not np.array_equal(times, proxy.continuation_times)
            or not np.array_equal(times, selected.times)
            or not np.array_equal(discounts, proxy.market.discount_curve.discount(times))
            or eligible.dtype.kind != 'b' or not eligible.all()
            or not np.array_equal(arrays['actual_time_grid'], _time_grid(proxy.params, proxy.market, times, 128))):
        raise ValueError('Selected simulation, rate or exercise calendars changed')
    stock = arrays['stocks'][:, :, 0]; factors = arrays['factors']; euro = arrays['european_proxy'][:, :, 0]
    if (np.any(stock <= 0.) or not np.all(stock[:, 0] == proxy.market.spot)
            or np.any(factors[:, 0] != 0.)):
        raise ValueError('Selected paths changed inception stock or full factor memory')
    intrinsic = np.maximum((1. if proxy.is_call else -1.)*(stock-proxy.strike), 0.)
    if not np.array_equal(arrays['payoff'], intrinsic*discounts):
        raise ValueError('Every saved cashflow must equal its discounted vanilla payoff')
    _close(euro[:, -1], arrays['payoff'][:, -1], 'Terminal European proxy differs from its actual payoff')
    zero_names = martingales + ('prior_local_corrections', 'grid_corrections', 'tail_corrections',
        'local_actual', 'local_expected', 'grid_actual', 'grid_expected', 'target_martingales',
        'target_expected_european', 'target_actual_premiums', 'target_expected_premiums')
    if any(np.any(arrays[name][:, 0] != 0.) for name in zero_names):
        raise ValueError('A retained actual/conditional primitive or martingale has a nonzero origin')
    if (any(np.any(arrays[name][:, -1] != 0.) for name in
            ('local_actual', 'local_expected', 'grid_actual', 'grid_expected',
             'target_actual_premiums', 'target_expected_premiums'))
            or any(np.any(arrays[name] < -1e-12) for name in ('target_actual_premiums', 'target_expected_premiums'))):
        raise ValueError('A terminal correction or positive target Snell premium changed')
    q = np.column_stack([np.clip(model.value(stock[:, j], factors[:, j], proxy.strike),
        0., selected.premium_cap) for j, model in enumerate(selected.continuations)])
    local_margin = discounts*(intrinsic-q)-euro
    gates = np.stack((np.ones((paths, 64)), np.exp(-.5*(local_margin[:, :-1]/.1)**2),
        np.exp(-.5*(local_margin[:, :-1]/.3)**2)), axis=-1)
    local_increment = discounts[None, 1:]*(arrays['local_actual'][:, 1:]-arrays['local_expected'][:, 1:])
    local = np.concatenate((np.zeros((paths, 1, 3)), np.cumsum(local_increment[:, :, None]*gates, axis=1)), axis=1)
    grid_increment = discounts[None, 1:, None]*(arrays['grid_actual'][:, 1:]-arrays['grid_expected'][:, 1:])
    refined = np.concatenate((np.zeros((paths, 1, 6)),
        np.cumsum((grid_increment[:, :, :, None]*gates[:, :, None, :]).reshape(paths, 64, 6), axis=1)), axis=1)
    _close(arrays['prior_local_corrections'], local, 'Local controls changed right-state centering or predictable left gates')
    _close(arrays['grid_corrections'], refined, 'Grid controls changed their two-by-three feature ordering or left gates')
    if fit.config.predictable_features or tuple(fit.config.components) != ('european', 'premium', 'micro_boundary'):
        raise ValueError('The original stable component specification changed')
    components = np.stack([arrays[name] for name in
        ('european_martingale', 'premium_martingale', 'micro_boundary_martingale')], axis=-1)
    bins = np.searchsorted(fit.time_edges[1:], times[1:], side='left')
    stable = np.column_stack((np.zeros(paths), np.cumsum(np.sum(
        np.diff(components, axis=1)*fit.weights[bins], axis=-1), axis=1)))
    prior_m = stable+_weighted(arrays['prior_local_corrections'], prior)
    current = prior_m+_weighted(arrays['grid_corrections'], grid)
    for name, value in (('stable_old_martingale', stable), ('prior_martingale', prior_m), ('current_martingale', current)):
        _close(arrays[name], value, 'A selected stable/prior/grid martingale changed frozen arithmetic')
    margin = (arrays['payoff']-euro)/discounts-q
    _close(arrays['tail_margin'], margin, 'Tail margin changed the frozen q-rule or discount convention')
    baseline_stop = np.full(paths, 64, dtype=np.int64)
    for j in range(1, 64):
        action = (baseline_stop == 64)&(intrinsic[:, j] > 0.)&(intrinsic[:, j] >= euro[:, j]/discounts[j]+q[:, j])
        baseline_stop[action] = j
    post = (np.arange(64)[None, :] >= baseline_stop[:, None]).astype(float)
    left = margin[:, :-1]
    tail_gates = np.stack((post, post*expit(left/.1), post*expit(left/.3), expit(left/.03),
        expit(left/.1), expit(left/.3), (arrays['payoff'][:, :-1] == 0.).astype(float)), axis=-1)
    _close(arrays['tail_gates'], tail_gates, 'A tail gate used a future state or a different baseline stopping rule')
    # Preserve stored current arithmetic after checking it independently.
    di = np.diff(arrays['intrinsic_martingale']-arrays['current_martingale'], axis=1)
    de = np.diff(arrays['european_martingale']-arrays['current_martingale'], axis=1)
    tail_increment = np.stack([di]*6+[de], axis=-1)*tail_gates
    tail = np.concatenate((np.zeros((paths, 1, 7)), np.cumsum(tail_increment, axis=1)), axis=1)
    _close(arrays['tail_corrections'], tail, 'Tail controls changed their predictable centered increments')
    tail_control = tail_seal['controls'][tail_seal['selected_control']]
    latest = arrays['current_martingale']+_weighted(arrays['tail_corrections'], tail_control)
    _close(arrays['latest_martingale'], latest, 'The frozen selected tail-control weights changed')
    mask = arrays['gh16_computed_mask']; continuation = np.asarray(arrays.get('gh16_continuation'))
    if (mask.dtype.kind != 'b' or continuation.shape != (paths, 65)
            or not np.isfinite(continuation[mask]).all() or not np.isnan(continuation[~mask]).all()
            or np.any(continuation[mask] < 0.)):
        raise ValueError('GH16 continuations must be finite on their own alive-ITM mask and NaN elsewhere')
    stops = np.full(paths, 64, dtype=np.int64); expected_mask = np.zeros((paths, 65), dtype=bool)
    for j in range(1, 64):
        active = (stops == 64)&(intrinsic[:, j] > 0.)
        expected_mask[:, j] = active
        stops[active&(intrinsic[:, j] >= continuation[:, j])] = j
    if (arrays['gh16_stops'].dtype.kind not in 'iu' or not np.array_equal(mask, expected_mask)
            or not np.array_equal(arrays['gh16_stops'], stops)):
        raise ValueError('GH16 own stopping decisions changed or omitted an alive ITM state')
    path = np.arange(paths); raw = arrays['payoff'][path, stops]
    if not np.array_equal(arrays['gh16_raw'], raw):
        raise ValueError('A raw selected payoff used another stopping date')
    increment = (euro[:, 1:]-arrays['target_expected_european'][:, 1:])[:, :, None]
    increment = increment+discounts[None, 1:, None]*(arrays['target_actual_premiums'][:, 1:]-arrays['target_expected_premiums'][:, 1:])
    target = np.concatenate((np.zeros((paths, 1, 4)), np.cumsum(increment, axis=1)), axis=1)
    _close(arrays['target_martingales'], target, 'Target controls changed actual-minus-conditional accumulation')
    if target_seal.get('selected_control') != 'target 4 bins penalty0':
        raise ValueError('The fixed selected target control identity changed')
    control = target_seal['controls'][target_seal['selected_control']]
    selected_m = arrays['latest_martingale']+_weighted(
        arrays['target_martingales']-arrays['latest_martingale'][:, :, None], control)
    lower = raw-selected_m[path, stops]
    upper = np.max(arrays['payoff']-selected_m, axis=1)
    for name, expected in (('candidate3_lower', lower), ('candidate3_upper', upper),
            ('unchanged_lower', lower), ('unchanged_upper', upper)):
        _close(arrays[name], expected, 'The selected endpoint changed its own policy or frozen martingale')
    conditional = validate_conditional_european(arrays, bundle, seed=seed, paths=paths)
    anchor = conditional['shared_european_control']
    _close(arrays['shared_control'], anchor, 'The paired European anchor changed its exact centering identity', atol=2e-12)
    features, metadata = reconstruct_predictable_basis(arrays, bundle)
    basis = np.column_stack((features, anchor))
    _close(arrays['basis'], basis, 'A common-control feature changed its left state, own survival or order', atol=2e-12)
    coefficient = np.asarray(coefficients, dtype=float)
    if coefficient.shape != (193,) or not np.isfinite(coefficient).all() or np.any(abs(coefficient) > 2.):
        raise ValueError('The frozen193 raw common-control coefficients or bound changed')
    shift = basis@coefficient
    _close(arrays['shift'], shift, 'The common shift changed coefficients or empirically recentered a final stream', atol=2e-12)
    _close(arrays['shared_lower'], lower-shift, 'The lower price did not use the common frozen shift', atol=2e-12)
    _close(arrays['shared_upper'], upper-shift, 'The upper price did not use the common frozen shift', atol=2e-12)
    _close(arrays['shared_upper']-arrays['shared_lower'], upper-lower,
           'The shared price control changed a pathwise stopping gap')
    probes = _bounded_probes(arrays, bundle, models)
    probes.extend({'kind': 'conditional European', **p} for p in conditional['probes'])
    checks = {'paths': int(paths), 'seed': int(seed), 'basis_columns': 193,
        'all_selected_component_identities_checked': True, 'all_own_stops_and_masks_checked': True,
        'all_paths_retained': True, 'all_rng_states_checked': True,
        'max_stock_error': conditional['max_stock_error'],
        'max_factor_error': conditional['max_factor_error'],
        'max_log_mean_error': conditional['max_log_mean_error'],
        'max_clock_error': conditional['max_clock_error'],
        'max_probability_error': conditional['max_probability_error'],
        'max_gap_translation_error': float(np.max(abs((arrays['shared_upper']-arrays['shared_lower'])-(upper-lower)))),
        'bounded_probe_records': len(probes), 'all_path_conditional_means_repriced': False,
        'empirical_valuation_mean_subtracted': False, 'no_unselected_endpoints_invented': True,
        'feature_survival': metadata['survival']}
    return {'values': {'unchanged control': {'lower': lower, 'upper': upper},
                       'shared control': {'lower': arrays['shared_lower'], 'upper': arrays['shared_upper']}},
        'basis': basis, 'checks': checks, 'probes': probes, 'raw': raw, 'stops': stops}


__all__ = ['validate_selected_samples']
