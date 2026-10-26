"""Naive European projections must not become centered valuation results."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_price_budget_trials import naive_european_trials


def streams():
    x = .2 * np.tile([-1., 1.], 4)
    z = 3. + .1 * np.repeat([-1., 1.], 4)
    m = 20. * (z - 3.)
    result, conditional = [], []
    for order in (np.arange(8), np.arange(8)[::-1]):
        lower = 2.86 + .6 * x + .4 * (z - 3.)
        payoff = np.column_stack((np.zeros(8), m + z + x))
        martingale = np.column_stack((np.zeros(8), m))
        result.append({'candidate3_lower': lower[order], 'payoff': payoff[order],
                       'european_martingale': martingale[order]})
        conditional.append({'absolute_path_indices': np.arange(8),
            'conditional_european_payoff': (z + .3)[order],
            'conditional_european_martingale': np.full(8, .3),
            'conditional_european_estimate': z[order], 'shared_control': x[order]})
    return result, conditional


def test_paired_X_is_distinct_from_terminal_martingale_and_both_folds_are_heldout():
    old, conditional = streams()
    table = naive_european_trials(old, conditional, seeds=(11, 12))
    assert len(table) == 6
    assert set(zip(table.fit_seed, table.diagnostic_seed)) == {(11, 12), (12, 11)}
    x = table.loc[table.family.eq('paired European X')]
    np.testing.assert_allclose(x.coefficient_X, .6, rtol=0., atol=2e-14)
    np.testing.assert_allclose(x.diagnostic_corr_X_EU_M_T, 0., atol=2e-14)
    assert not x.X_is_terminal_European_martingale.any()
    combined = table.loc[table.family.eq('paired X + conditional Z')]
    np.testing.assert_allclose(combined.coefficient_X, .6, atol=2e-14)
    np.testing.assert_allclose(combined.coefficient_Z, .4, atol=2e-14)
    assert (combined.diagnostic_sd < 1e-14).all()


def test_conditional_mean_projection_has_explicit_roles_and_no_valuation_level():
    table = naive_european_trials(*streams())
    assert table.loc[table.family.str.contains('conditional'), 'ideal_known_mean_projection_only'].all()
    assert not table.loc[table.family.eq('paired European X'), 'ideal_known_mean_projection_only'].any()
    for name in ('Z_mean_known', 'actual_unbiased_valuation_reported', 'used_for_control_selection',
                 'new_acceptance_evidence', 'used_for_fresh_acceptance', 'heldout_price_or_mean_reported',
                 'continuous_price_used_to_center', 'empirical_valuation_mean_subtracted',
                 'projection_is_rigorous_variance_bound'):
        assert not table[name].any()
    assert not {'price', 'estimate', 'diagnostic_mean', 'diagnostic_mean_shift'} & set(table)
    assert table.role.eq('spent crossfit diagnostic; excluded from fresh acceptance').all()


def test_coefficient_training_does_not_use_heldout_target():
    old, conditional = streams()
    before = naive_european_trials(old, conditional, seeds=(11, 12))
    old[1]['candidate3_lower'] = 9. + np.arange(8) * .2
    after = naive_european_trials(old, conditional, seeds=(11, 12))
    for family in before.family.unique():
        a = before.query('fit_seed == 11 and family == @family').iloc[0]
        b = after.query('fit_seed == 11 and family == @family').iloc[0]
        assert (a.coefficient_X, a.coefficient_Z) == (b.coefficient_X, b.coefficient_Z)


@pytest.mark.parametrize('kind', ['Z_identity', 'X_identity', 'subset', 'nonfinite', 'wrongshape'])
def test_tampering_or_partial_streams_cannot_be_reported_as_full_diagnostics(kind):
    old, conditional = streams()
    if kind == 'Z_identity':
        conditional[0]['conditional_european_estimate'][0] += .1
    elif kind == 'X_identity':
        conditional[0]['shared_control'][0] += .1
    elif kind == 'subset':
        conditional[0]['absolute_path_indices'] = np.arange(1, 9)
    elif kind == 'nonfinite':
        old[0]['candidate3_lower'][0] = np.nan
    else:
        old[0]['european_martingale'] = np.zeros((8, 3))
    with pytest.raises(ValueError):
        naive_european_trials(old, conditional)


def test_unknown_Z_mean_does_not_enter_the_variance_projection():
    old, conditional = streams()
    first = naive_european_trials(old, conditional)
    for a, c in zip(old, conditional):
        # Shift the unknown conditional-European mean, leaving the exact X
        # identity unchanged.  No reference mean is supplied to the helper.
        a['payoff'][:, -1] += 100.
        c['conditional_european_payoff'] += 100.
        c['conditional_european_estimate'] += 100.
    second = naive_european_trials(old, conditional)
    np.testing.assert_allclose(second[['coefficient_X', 'coefficient_Z', 'diagnostic_sd']],
                               first[['coefficient_X', 'coefficient_Z', 'diagnostic_sd']],
                               rtol=0., atol=5e-13)
