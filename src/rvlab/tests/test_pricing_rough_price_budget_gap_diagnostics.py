"""Full-path gap tails and planning must not become extra acceptance data."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_price_budget_gap_diagnostics import gap_diagnostics


def inputs():
    old = [np.array([0., 0., .001, .003]), np.array([0., .001, .001, .005])]
    fresh = [np.r_[np.zeros(98), .03, .12], np.r_[np.zeros(98), .04, .15]]
    n = sum(map(len, fresh))
    gap = sum(a.sum() for a in fresh) / n
    se = np.sqrt(sum(len(a) * np.var(a, ddof=1) for a in fresh)) / n
    row = {'paths': n, 'gap': gap, 'gap_se': se, 'price_se': .0002,
           'price_allowance': .0023, 'price_error_budget': gap + 1.96 * (.0002 + se)}
    return old, fresh, [11, 12], [13, 14], row


def test_gap_difference_uses_independent_seed_standard_errors_and_all_tail_paths():
    old, fresh, a, b, row = inputs()
    tables = gap_diagnostics(old, fresh, a, b, row)
    dist = tables['gap_distribution']
    pooled = dist.query("cohort == 'fresh independent final' and scope == 'pooled'").iloc[0]
    assert pooled.paths == 200 and pooled.count_above_010 == 2
    assert pooled.count_above_003 == 3 and pooled.count_above_001 == 4
    assert pooled.top1pct_paths == 2
    assert pooled.top1pct_gap_share == pytest.approx((.12 + .15) / .34)
    assert pooled.positive_fraction == .02 and pooled.maximum == .15
    assert pooled.p999 == pytest.approx(np.quantile(np.concatenate(fresh), .999))
    change = tables['gap_shift'].iloc[0]
    oldse = np.sqrt(sum(len(v) * np.var(v, ddof=1) for v in old)) / 8
    assert change.difference_se == pytest.approx(np.hypot(oldse, row['gap_se']))
    assert change.fresh_minus_old_gap == pytest.approx(row['gap'] - np.concatenate(old).mean())
    assert not change.difference_is_noise_only_certified


def test_projection_keeps_current_gap_and_both_variances_without_extra_paths():
    *args, row = inputs()
    table = gap_diagnostics(*args, row)['precision_planning'].iloc[0]
    h = row['price_allowance'] - row['gap']
    assert table.required_lower_se_at_current_paths == pytest.approx(h / 1.96 - row['gap_se'])
    assert h > 0
    assert table.projected_paths_with_unchanged_variance == pytest.approx(
        row['paths'] * (1.96 * (row['price_se'] + row['gap_se']) / h) ** 2)
    assert table.new_paths_run == 0 and not table.new_acceptance_evidence
    assert not table.projection_is_acceptance_result and not table.tolerances_changed


def test_gap_already_above_allowance_has_no_finite_sampling_only_solution():
    *args, row = inputs()
    row['price_allowance'] = row['gap'] / 2
    plan = gap_diagnostics(*args, row)['precision_planning'].iloc[0]
    assert np.isinf(plan.projected_paths_with_unchanged_variance)
    assert plan.required_lower_se_at_current_paths < 0


@pytest.mark.parametrize('kind', ['overlap', 'missingpath', 'wronggap', 'changedgate', 'negative', 'nonfinite'])
def test_changed_data_or_gate_cannot_enter_the_diagnostics(kind):
    old, fresh, a, b, row = inputs()
    if kind == 'overlap': b[0] = a[0]
    elif kind == 'missingpath': fresh[0] = fresh[0][1:]
    elif kind == 'wronggap': row['gap'] += .1
    elif kind == 'changedgate': row['price_error_budget'] += .01
    elif kind == 'negative': fresh[0][0] = -.01
    else: fresh[0][0] = np.nan
    with pytest.raises(ValueError): gap_diagnostics(old, fresh, a, b, row)
