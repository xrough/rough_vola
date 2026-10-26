"""Post-valuation gap tails and precision planning; never new acceptance.

The former final paths are spent for this comparison.  Independent old/fresh
gap differences can be quantified, but cannot establish that a change is
entirely sampling noise or separate policy error from dual slack.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def _check(chunks, seeds):
    if (not chunks or len(chunks) != len(seeds) or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, Integral) or isinstance(s, bool) for s in seeds)):
        raise ValueError('Distinct labelled whole-path seed strata required')
    values = [np.asarray(a, dtype=float) for a in chunks]
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
           or np.min(a) < -2e-11 for a in values):
        raise ValueError('Finite pathwise nonnegative stopping gaps required')
    return values


def _moments(chunks):
    n = sum(map(len, chunks))
    mean = float(sum(a.sum() for a in chunks) / n)
    se = float(np.sqrt(sum(len(a) * np.var(a, ddof=1) for a in chunks)) / n)
    return n, mean, se


def _distribution(chunks, cohort, *, seed=None):
    n, mean, se = _moments(chunks)
    all_values = np.concatenate(chunks)
    top_count = max(1, int(np.ceil(.01 * n)))
    total = float(all_values.sum())
    top = np.sort(all_values)[-top_count:]
    return {'cohort': cohort, 'scope': 'pooled' if seed is None else 'one seed',
        'seed': seed, 'paths': n, 'gap': mean, 'gap_se': se,
        'p95': float(np.quantile(all_values, .95)),
        'p99': float(np.quantile(all_values, .99)),
        'p999': float(np.quantile(all_values, .999)),
        'maximum': float(np.max(all_values)),
        'positive_fraction': float(np.mean(all_values > 1e-12)),
        'positive_threshold': 1e-12, 'top1pct_paths': top_count,
        'top1pct_gap_share': float(top.sum() / total) if total > 0. else 0.,
        'count_above_001': int(np.sum(all_values > .01)),
        'count_above_003': int(np.sum(all_values > .03)),
        'count_above_010': int(np.sum(all_values > .1)),
        'all_paths_retained': True, 'used_for_selection': False,
        'new_acceptance_evidence': False}


def gap_diagnostics(old_chunks, fresh_chunks, old_seeds, fresh_seeds, price_row):
    """Describe all gap paths, independent change, and a fixed-variance plan.

    ``price_row`` is the already computed shared-control final price record.
    Its unchanged gate is G+1.96*(SE_lower+SE_gap) <= .001+.0001*abs(V).
    Projections hold the observed gap, allowance and per-path variances fixed;
    they are not promises that another sample will pass.  No path is removed,
    no seed is selected and no additional Monte Carlo simulation is run.
    """
    old = _check(old_chunks, old_seeds)
    fresh = _check(fresh_chunks, fresh_seeds)
    if set(old_seeds) & set(fresh_seeds):
        raise ValueError('Old and fresh diagnostic seeds must be independent')
    n_old, g_old, se_old = _moments(old)
    n, gap, gap_se = _moments(fresh)
    required = ('paths', 'gap', 'gap_se', 'price_se', 'price_allowance', 'price_error_budget')
    if (any(k not in price_row for k in required)
            or not all(np.isfinite(price_row[k]) for k in required)
            or price_row['paths'] != n or price_row['price_se'] < 0.
            or price_row['price_allowance'] <= 0.
            or not np.isclose(price_row['gap'], gap, rtol=1e-10, atol=2e-12)
            or not np.isclose(price_row['gap_se'], gap_se, rtol=1e-10, atol=2e-12)):
        raise ValueError('Price record must match all fresh gap paths and fixed allowance')
    price_se = float(price_row['price_se'])
    allowance = float(price_row['price_allowance'])
    sampling = 1.96 * (price_se + gap_se)
    budget = gap + sampling
    if (not np.isclose(price_row['price_error_budget'], budget, rtol=1e-10, atol=2e-12)
            or ('price' in price_row and not np.isclose(
                allowance, .001 + 1e-4 * abs(price_row['price']), rtol=0., atol=2e-12))):
        raise ValueError('The unchanged additive price gate was replaced')
    distribution = []
    for cohort, chunks, seeds in (('prior spent final', old, old_seeds),
                                 ('fresh independent final', fresh, fresh_seeds)):
        distribution.extend(_distribution([a], cohort, seed=int(s)) for a, s in zip(chunks, seeds))
        distribution.append(_distribution(chunks, cohort))
    difference = gap - g_old
    difference_se = float(np.hypot(gap_se, se_old))
    headroom = allowance - gap
    required_lower_se = headroom / 1.96 - gap_se
    projected_paths = n * (sampling / headroom) ** 2 if headroom > 0. else np.inf
    required_gap_decrease = max(0., budget - allowance)
    shift = {'old_paths': n_old, 'fresh_paths': n, 'old_gap': g_old,
        'fresh_gap': gap, 'old_gap_se': se_old, 'fresh_gap_se': gap_se,
        'fresh_minus_old_gap': difference, 'difference_se': difference_se,
        'difference_low95': difference - 1.96 * difference_se,
        'difference_high95': difference + 1.96 * difference_se,
        'difference_z': difference / difference_se if difference_se else np.nan,
        'independent_seeds': True, 'difference_is_noise_only_certified': False,
        'policy_and_dual_errors_separately_identified': False,
        'new_acceptance_evidence': False, 'used_for_selection': False}
    plan = {'paths': n, 'gap': gap, 'gap_se': gap_se, 'price_se': price_se,
        'price_allowance': allowance, 'price_error_budget': budget,
        'allowance_left_after_mean_gap': headroom, 'current_sampling_term': sampling,
        'required_lower_se_at_current_paths': required_lower_se,
        'projected_paths_with_unchanged_variance': projected_paths,
        'required_gap_decrease_at_current_sampling': required_gap_decrease,
        'required_gap_decrease_fraction': required_gap_decrease / gap if gap > 0. else np.nan,
        'gap_allowance_and_variances_held_fixed': True,
        'projection_is_acceptance_result': False, 'new_paths_run': 0,
        'new_acceptance_evidence': False, 'used_for_selection': False,
        'tolerances_changed': False,
        'role': 'planning only; observed tail mean and variances may change on another sample'}
    return {'gap_distribution': pd.DataFrame(distribution),
        'gap_shift': pd.DataFrame([shift]), 'precision_planning': pd.DataFrame([plan])}


__all__ = ['gap_diagnostics']
