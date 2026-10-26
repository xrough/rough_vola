"""Independent whole-path statistics for the fixed fresh precision campaign.

This module performs no pricing, simulation, fitting, or result selection and
imports no production-pricing or collector code.  The old 49,152 paths remain
historical evidence.  Only the complete new pool supplies the primary sampling
gate.  Seed and disjoint-quarter results are prespecified diagnostics, never
alternative acceptance pools.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


BASELINE = 'unchanged control'
PRIMARY = 'shared control'
ESTIMATORS = (BASELINE, PRIMARY)
PRIOR_SEEDS = (1106005042, 1107005042, 1108005042)
PRIOR_PATHS = (16384, 16384, 16384)
REPLICA_DECLARATION = {
    'replicates': 4, 'paths_per_seed_per_replicate': 16384,
    'paths_per_replicate': 49152,
    'index_rule': 'replicate k uses [k*16384:(k+1)*16384] from each fresh seed',
    'all_replicates_retained': True, 'acceptance_or_selection_role': False,
}


def _validate(streams, seeds, paths):
    if (len(seeds) != 3 or len(paths) != 3 or len(streams) != 3
            or len(set(seeds)) != 3
            or any(not isinstance(s, Integral) or isinstance(s, bool) for s in seeds)
            or any(not isinstance(n, Integral) or isinstance(n, bool) or n < 2 for n in paths)
            or tuple(s.get('seed') for s in streams) != tuple(seeds)):
        raise ValueError('All three declared independent seed streams and path counts are required')
    result = []
    for stream, n in zip(streams, paths):
        if set(stream.get('values', {})) != set(ESTIMATORS):
            raise ValueError('Both fixed estimators must be retained without replacement')
        values = {}
        for name in ESTIMATORS:
            endpoints = stream['values'][name]
            if set(endpoints) != {'lower', 'upper'}:
                raise ValueError('Complete lower and upper path arrays are required')
            lower = np.asarray(endpoints['lower'], dtype=float)
            upper = np.asarray(endpoints['upper'], dtype=float)
            if (lower.shape != (n,) or upper.shape != (n,)
                    or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                    or np.any(upper < lower - 2e-11)):
                raise ValueError('All declared finite whole-path endpoints must remain aligned')
            values[name] = {'lower': lower, 'upper': upper}
        gap0 = values[BASELINE]['upper'] - values[BASELINE]['lower']
        gap1 = values[PRIMARY]['upper'] - values[PRIMARY]['lower']
        if not np.allclose(gap0, gap1, rtol=0., atol=2e-13):
            raise ValueError('The common endpoint shift changed a pathwise stopping gap')
        result.append({'seed': int(stream['seed']), 'values': values})
    return result


def _moments(chunks):
    n = sum(map(len, chunks))
    mean = float(sum(np.sum(a) for a in chunks) / n)
    se = float(np.sqrt(sum(len(a) * np.var(a, ddof=1) for a in chunks)) / n)
    return {'estimate': mean, 'standard_error': se, 'half_width': 1.96 * se,
            'interval_low': mean - 1.96 * se, 'interval_high': mean + 1.96 * se}


def _chunks(streams, name, metric):
    if metric == 'gap':
        return [s['values'][name]['upper'] - s['values'][name]['lower'] for s in streams]
    return [s['values'][name][metric] for s in streams]


def _price_record(streams, name, *, role, primary_pool):
    p = _moments(_chunks(streams, name, 'lower'))
    u = _moments(_chunks(streams, name, 'upper'))
    g = _moments(_chunks(streams, name, 'gap'))
    allowance = .001 + 1e-4 * abs(p['estimate'])
    budget = g['estimate'] + p['half_width'] + g['half_width']
    return {'estimator': name, 'primary': name == PRIMARY,
        'paths': sum(len(s['values'][name]['lower']) for s in streams),
        'price': p['estimate'], 'price_se': p['standard_error'],
        'upper': u['estimate'], 'upper_se': u['standard_error'],
        'gap': g['estimate'], 'gap_se': g['standard_error'],
        'lower_sampling_term': p['half_width'], 'gap_sampling_term': g['half_width'],
        'total_sampling_term': p['half_width'] + g['half_width'],
        'price_error_budget': budget, 'price_allowance': allowance,
        'budget_minus_allowance': budget - allowance,
        'price_budget_pass': budget <= allowance,
        'role': role, 'primary_acceptance_pool': primary_pool,
        'used_for_selection': False, 'prior_paths_pooled_for_acceptance': False,
        'sampling_gate_only': True, 'american_pricing_ready': False,
        'stock_delta_ready': False, 'tolerances_changed': False}


def _gap_distribution(streams, cohort):
    rows = []
    for chosen, seed in [([s], s['seed']) for s in streams] + [(streams, None)]:
        gaps = _chunks(chosen, PRIMARY, 'gap')
        x = np.concatenate(gaps)
        n = len(x)
        top_count = max(1, int(np.ceil(.01 * n)))
        total = float(x.sum())
        stats = _moments(gaps)
        rows.append({'cohort': cohort, 'scope': 'pooled' if seed is None else 'one seed',
            'seed': seed, 'paths': n, 'gap': stats['estimate'],
            'gap_se': stats['standard_error'], 'p95': float(np.quantile(x, .95)),
            'p99': float(np.quantile(x, .99)), 'p999': float(np.quantile(x, .999)),
            'maximum': float(x.max()), 'positive_fraction': float(np.mean(x > 1e-12)),
            'positive_threshold': 1e-12, 'top1pct_paths': top_count,
            'top1pct_gap_share': float(np.sort(x)[-top_count:].sum()/total) if total > 0. else 0.,
            'count_above_001': int(np.sum(x > .01)),
            'count_above_003': int(np.sum(x > .03)),
            'count_above_010': int(np.sum(x > .1)),
            'all_paths_retained': True, 'used_for_selection': False,
            'diagnostic_only': True})
    return rows


def summarize_precision(streams, *, expected_seeds, prior_streams,
                        expected_paths=(65536, 65536, 65536)):
    """Summarize a fixed complete precision campaign without optional stopping.

    Canonical input streams have ``seed`` and ``values``; values maps the two
    fixed estimator names to one-dimensional ``lower``/``upper`` arrays.
    ``expected_seeds`` must come from the pre-valuation design, not observed
    outcomes.  Prior streams are strictly the old three 16,384-path streams.

    Standard errors use all path counts and within-seed variances.  The three
    seed estimates are not treated as three independent path observations.
    Normal 1.96 intervals are marginal sampling diagnostics; no multiplicity
    or readiness claim follows from the secondary results.
    """
    current = _validate(streams, expected_seeds, expected_paths)
    old = _validate(prior_streams, PRIOR_SEEDS, PRIOR_PATHS)
    if set(expected_seeds) & set(PRIOR_SEEDS):
        raise ValueError('Fresh primary streams must be independent of the historical pool')
    if tuple(expected_paths) != (65536, 65536, 65536):
        raise ValueError('This precision study requires the predeclared 196608 fresh paths')
    comparison, per_seed, paired, decomposition, consistency, replicas = [], [], [], [], [], []
    prior_rows = {}
    for name in ESTIMATORS:
        record = _price_record(current, name, role='complete independent fresh primary pool', primary_pool=True)
        comparison.append(record)
        prior_rows[name] = _price_record(old, name, role='historical planning evidence only', primary_pool=False)
        for component, key in (('Mean stopping gap', 'gap'),
                ('Lower-price sampling uncertainty', 'lower_sampling_term'),
                ('Gap sampling uncertainty', 'gap_sampling_term')):
            decomposition.append({'estimator': name, 'primary': name == PRIMARY,
                'component': component, 'contribution': record[key],
                'allowance': record['price_allowance'], 'budget': record['price_error_budget'],
                'price_budget_pass': record['price_budget_pass'], 'paths': record['paths']})
        for i, s in enumerate(current):
            row = _price_record([s], name, role='prespecified per-seed diagnostic only', primary_pool=False)
            row.update(seed=s['seed'], marginal_interval=True)
            per_seed.append(row)
            others = [other for j, other in enumerate(current) if i != j]
            for metric in ('lower', 'gap'):
                one = _moments(_chunks([s], name, metric))
                rest = _moments(_chunks(others, name, metric))
                delta = one['estimate'] - rest['estimate']
                se = float(np.hypot(one['standard_error'], rest['standard_error']))
                consistency.append({'estimator': name, 'metric': metric, 'seed': s['seed'],
                    'seed_minus_remaining': delta, 'standard_error': se,
                    'interval_low': delta - 1.96*se, 'interval_high': delta + 1.96*se,
                    'z': delta/se if se else np.nan, 'marginal_interval': True,
                    'contrasts_are_mutually_independent': False, 'used_for_selection': False,
                    'diagnostic_only': True})
        for metric in ('lower', 'upper', 'gap'):
            delta = [a-b for a, b in zip(_chunks(current, name, metric), _chunks(current, BASELINE, metric))]
            paired.append({'estimator': name, 'primary': name == PRIMARY,
                'metric': f'{metric} change', **_moments(delta), 'paths': record['paths'],
                'used_for_selection': False})
        for k in range(4):
            subset = [{'seed': s['seed'], 'values': {label: {
                endpoint: value[k*16384:(k+1)*16384]
                for endpoint, value in s['values'][label].items()} for label in ESTIMATORS}}
                for s in current]
            row = _price_record(subset, name,
                role='prespecified disjoint 49152-path replicate; diagnostic only', primary_pool=False)
            row.update(replicate=k, index_start=k*16384, index_stop=(k+1)*16384,
                       all_replicates_retained=True, stopping_or_selection_role=False)
            replicas.append(row)
    frame = pd.DataFrame(comparison)
    baseline_se = frame.loc[frame.estimator.eq(BASELINE), 'price_se'].iloc[0]
    frame['price_variance_ratio_vs_baseline'] = (frame.price_se/baseline_se)**2 if baseline_se else np.nan
    frame['gap_changed_by_common_control'] = False
    old_gap = _moments(_chunks(old, PRIMARY, 'gap'))
    new_gap = _moments(_chunks(current, PRIMARY, 'gap'))
    delta = new_gap['estimate'] - old_gap['estimate']
    delta_se = float(np.hypot(new_gap['standard_error'], old_gap['standard_error']))
    shift = {'old_paths': 49152, 'fresh_paths': 196608,
        'old_gap': old_gap['estimate'], 'fresh_gap': new_gap['estimate'],
        'old_gap_se': old_gap['standard_error'], 'fresh_gap_se': new_gap['standard_error'],
        'fresh_minus_old_gap': delta, 'difference_se': delta_se,
        'difference_low95': delta-1.96*delta_se, 'difference_high95': delta+1.96*delta_se,
        'difference_z': delta/delta_se if delta_se else np.nan,
        'difference_certified_as_noise_only': False, 'policy_and_dual_errors_identified': False,
        'historical_paths_used_for_acceptance': False, 'used_for_selection': False,
        'diagnostic_only': True}
    planning = []
    for record in comparison:
        past = prior_rows[record['estimator']]
        ratio = np.sqrt(past['paths']/record['paths'])
        projected_sampling = past['total_sampling_term']*ratio
        past_headroom = past['price_allowance']-past['gap']
        planning.append({'estimator': record['estimator'], 'historical_paths': past['paths'],
            'fresh_paths': record['paths'], 'historical_gap': past['gap'], 'fresh_gap': record['gap'],
            'projected_lower_se': past['price_se']*ratio, 'actual_lower_se': record['price_se'],
            'projected_gap_se': past['gap_se']*ratio, 'actual_gap_se': record['gap_se'],
            'projected_budget_historical_gap_fixed': past['gap']+projected_sampling,
            'projected_allowance': past['price_allowance'],
            'actual_fresh_budget': record['price_error_budget'],
            'actual_fresh_allowance': record['price_allowance'],
            'historical_projected_paths_to_gate': past['paths']*(past['total_sampling_term']/past_headroom)**2
                if past_headroom > 0. else np.inf,
            'projection_is_acceptance_result': False, 'used_for_selection': False,
            'historical_paths_used_for_acceptance': False, 'planning_only': True})
    return {'price_comparison': frame, 'per_seed': pd.DataFrame(per_seed),
        'paired_changes': pd.DataFrame(paired), 'budget_decomposition': pd.DataFrame(decomposition),
        'gap_distribution': pd.DataFrame(_gap_distribution(old, 'historical 49152-path pool')
                                        + _gap_distribution(current, 'fresh 196608-path pool')),
        'gap_shift': pd.DataFrame([shift]), 'planning_comparison': pd.DataFrame(planning),
        'seed_consistency': pd.DataFrame(consistency),
        'fixed_budget_replicates': pd.DataFrame(replicas)}


__all__ = ['summarize_precision', 'REPLICA_DECLARATION', 'BASELINE', 'PRIMARY']
