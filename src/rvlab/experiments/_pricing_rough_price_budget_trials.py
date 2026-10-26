"""Spent-only European-control diagnostics, distinct from fresh valuation.

The paired variable X has exactly zero expectation under the discrete law.
The conditional controlled-European estimator Z has the unknown European
expectation.  Z-containing rows therefore show only the residual variance
one could obtain with a noiseless, known reference mean.  They are not
unbiased valuation results, nor rigorous bounds on attainable variance.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def _spent_variables(arrays, conditional, lower_key):
    lower = np.asarray(arrays[lower_key], dtype=float)
    payoff = np.asarray(arrays['payoff'], dtype=float)
    martingale = np.asarray(arrays['european_martingale'], dtype=float)
    n = len(lower) if lower.ndim else 0
    index = np.asarray(conditional['absolute_path_indices'])
    q = np.asarray(conditional['conditional_european_payoff'], dtype=float)
    c = np.asarray(conditional['conditional_european_martingale'], dtype=float)
    if (lower.shape != (n,) or n < 3 or payoff.ndim != 2 or payoff.shape[0] != n
            or payoff.shape[1] < 2 or martingale.shape != payoff.shape
            or index.dtype.kind not in 'iu' or not np.array_equal(index, np.arange(n))
            or q.shape != (n,) or c.shape != (n,)
            or not all(np.isfinite(v).all() for v in (lower, payoff, martingale, q, c))):
        raise ValueError('Full, aligned finite spent European samples are required')
    z = q - c
    if 'conditional_european_estimate' in conditional:
        saved = np.asarray(conditional['conditional_european_estimate'], dtype=float)
        if (saved.shape != (n,) or not np.isfinite(saved).all()
                or not np.allclose(saved, z, rtol=0., atol=2e-12)):
            raise ValueError('Conditional-European Z differs from payoff minus its martingale')
    x = (payoff[:, -1] - martingale[:, -1]) - z
    if 'shared_control' in conditional:
        saved = np.asarray(conditional['shared_control'], dtype=float)
        if (saved.shape != (n,) or not np.isfinite(saved).all()
                or not np.allclose(saved, x, rtol=0., atol=2e-12)):
            raise ValueError('Paired-European X differs from the exact two-estimator identity')
    return {'lower': lower, 'X': x, 'Z': z, 'EU_M_T': martingale[:, -1]}


def _correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0. else np.nan


def _fit_projection(features, lower):
    """Training-only unconstrained least squares for a variance diagnostic."""
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-12, 1., scale)
    design = (features - center) / scale
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, lower - lower.mean(), rcond=1e-12)
    return coefficients / scale, int(rank)


def naive_european_trials(old_arrays, conditional_replays, *,
                          seeds=(1094005042, 1095005042),
                          lower_key='candidate3_lower'):
    """Return six diagnostic rows from two full, already spent path streams.

    Each family (X, Z, X+Z) is fitted once on each stream and scored on the
    other stream.  Only held-out standard deviations are reported: unknown
    means of Z must never be replaced by a Fourier price or a held-out sample
    mean.  Any independent-reference estimator would additionally need to
    account for reference uncertainty, which these optimistic Z projections
    deliberately do not include.  No row is used for fresh acceptance or
    production control selection.

    X is ``(terminal payoff - European M_T) - Z``.  In particular, it is not
    the ``terminal`` family of the earlier spent ridge study, which is M_T.
    """
    if (len(old_arrays) != 2 or len(conditional_replays) != 2 or len(seeds) != 2
            or any(not isinstance(s, Integral) or isinstance(s, bool) for s in seeds)
            or seeds[0] == seeds[1]):
        raise ValueError('Exactly two distinct, labelled spent streams are required')
    data = [_spent_variables(a, c, lower_key)
            for a, c in zip(old_arrays, conditional_replays)]
    rows = []
    for family, columns in (('paired European X', ('X',)),
                            ('conditional European Z', ('Z',)),
                            ('paired X + conditional Z', ('X', 'Z'))):
        ideal_known_mean = 'Z' in columns
        for fold in (0, 1):
            train, heldout = data[fold], data[1 - fold]
            x = np.column_stack([train[c] for c in columns])
            h = np.column_stack([heldout[c] for c in columns])
            beta, rank = _fit_projection(x, train['lower'])
            diagnostic_sd = float(np.std(heldout['lower'] - h @ beta, ddof=1))
            baseline_sd = float(np.std(heldout['lower'], ddof=1))
            coefficients = dict(zip(columns, beta))
            rows.append({
                'family': family,
                'fit_seed': int(seeds[fold]),
                'diagnostic_seed': int(seeds[1 - fold]),
                'rows_fit': len(x),
                'rows_diagnostic': len(h),
                'features': len(columns),
                'fit_rank': rank,
                'coefficient_X': float(coefficients.get('X', 0.)),
                'coefficient_Z': float(coefficients.get('Z', 0.)),
                'training_sd': float(np.std(train['lower'] - x @ beta, ddof=1)),
                'baseline_diagnostic_sd': baseline_sd,
                'diagnostic_sd': diagnostic_sd,
                'diagnostic_variance_reduction': (1. - (diagnostic_sd / baseline_sd) ** 2
                                                  if baseline_sd > 0. else np.nan),
                'diagnostic_corr_lower_X': _correlation(heldout['lower'], heldout['X']),
                'diagnostic_corr_lower_Z': _correlation(heldout['lower'], heldout['Z']),
                'diagnostic_corr_lower_EU_M_T': _correlation(heldout['lower'], heldout['EU_M_T']),
                'diagnostic_corr_X_EU_M_T': _correlation(heldout['X'], heldout['EU_M_T']),
                'X_is_terminal_European_martingale': False,
                'X_exact_zero_mean_under_discrete_law': True,
                'Z_mean_known': False,
                'ideal_known_mean_projection_only': ideal_known_mean,
                'reference_mean_uncertainty_ignored_for_projection': ideal_known_mean,
                'projection_is_rigorous_variance_bound': False,
                'fit_standardization_training_only': True,
                'heldout_price_or_mean_reported': False,
                'continuous_price_used_to_center': False,
                'empirical_valuation_mean_subtracted': False,
                'actual_unbiased_valuation_reported': False,
                'used_for_control_selection': False,
                'new_acceptance_evidence': False,
                'used_for_fresh_acceptance': False,
                'role': 'spent crossfit diagnostic; excluded from fresh acceptance',
            })
    return pd.DataFrame(rows)


__all__ = ['naive_european_trials']
