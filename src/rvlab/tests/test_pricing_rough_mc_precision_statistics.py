"""Fixed-budget precision diagnostics cannot select a favorable fresh pool."""
import ast
from pathlib import Path

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_mc_precision_statistics import (
    BASELINE, PRIMARY, PRIOR_SEEDS, REPLICA_DECLARATION, summarize_precision)


FRESH_SEEDS = (1110005042, 1111005042, 1112005042)


def make_stream(seed, n, level):
    j = np.arange(n)
    noise = .024 * np.where(j % 2, 1., -1.) + level
    gap = .0008 + .0001 * (j % 4)
    gap[-1] = .25
    lower = 2.86 + noise
    shared = lower - .4*noise
    return {'seed': seed, 'values': {
        BASELINE: {'lower': lower, 'upper': lower+gap},
        PRIMARY: {'lower': shared, 'upper': shared+gap}}}


def streams():
    fresh = [make_stream(s, 65536, (i-1)*.001) for i, s in enumerate(FRESH_SEEDS)]
    prior = [make_stream(s, 16384, (i-1)*.002) for i, s in enumerate(PRIOR_SEEDS)]
    return fresh, prior


def summarize(fresh, prior):
    return summarize_precision(fresh, expected_seeds=FRESH_SEEDS, prior_streams=prior)


def test_primary_SE_uses_all_paths_and_within_seed_variances_not_three_observations():
    fresh, old = streams()
    tables = summarize(fresh, old)
    result = tables['price_comparison'].set_index('estimator')
    y = [s['values'][PRIMARY]['lower'] for s in fresh]
    n = sum(map(len, y))
    manual_variance = sum(len(a)*np.dot(a-a.mean(), a-a.mean())/(len(a)-1) for a in y)/n**2
    assert result.loc[PRIMARY, 'paths'] == 196608
    assert result.loc[PRIMARY, 'price'] == pytest.approx(sum(a.sum() for a in y)/n)
    assert result.loc[PRIMARY, 'price_se'] == pytest.approx(np.sqrt(manual_variance))
    wrong_three_observations = np.std([a.mean() for a in y], ddof=1)/np.sqrt(3)
    assert abs(result.loc[PRIMARY, 'price_se']-wrong_three_observations) > .0001
    assert result.loc[PRIMARY, 'price_variance_ratio_vs_baseline'] == pytest.approx(.36)
    assert result.primary_acceptance_pool.all()
    assert not result.prior_paths_pooled_for_acceptance.any()
    assert not result.american_pricing_ready.any()


def test_same_path_gap_and_unchanged_additive_gate_are_checked_independently():
    fresh, old = streams()
    t = summarize(fresh, old)
    r = t['price_comparison'].set_index('estimator')
    p = r.loc[PRIMARY]
    assert p.price_allowance == pytest.approx(.001+1e-4*abs(p.price))
    assert p.price_error_budget == pytest.approx(p.gap+1.96*(p.price_se+p.gap_se))
    assert p.gap == pytest.approx(r.loc[BASELINE, 'gap'])
    assert p.gap_se == pytest.approx(r.loc[BASELINE, 'gap_se'])
    gap_change = t['paired_changes'].query('estimator == @PRIMARY and metric == "gap change"').iloc[0]
    assert abs(gap_change.estimate) < 2e-13 and gap_change.half_width < 2e-13
    assert not t['per_seed'].primary_acceptance_pool.any()
    assert not t['per_seed'].used_for_selection.any()


def test_four_fixed_replicates_are_disjoint_exhaustive_and_diagnostic_only():
    fresh, old = streams()
    t = summarize(fresh, old)
    replicas = t['fixed_budget_replicates']
    assert REPLICA_DECLARATION['replicates'] == 4
    assert REPLICA_DECLARATION['acceptance_or_selection_role'] is False
    assert len(replicas) == 8 and set(replicas.replicate) == set(range(4))
    for name in (BASELINE, PRIMARY):
        chosen = replicas.loc[replicas.estimator.eq(name)].sort_values('replicate')
        assert chosen.paths.eq(49152).all()
        assert chosen.index_start.tolist() == [0, 16384, 32768, 49152]
        assert chosen.index_stop.tolist() == [16384, 32768, 49152, 65536]
        primary = t['price_comparison'].query('estimator == @name').iloc[0]
        assert chosen.price.mean() == pytest.approx(primary.price)
        assert chosen.gap.mean() == pytest.approx(primary.gap)
    assert not replicas.primary_acceptance_pool.any()
    assert not replicas.stopping_or_selection_role.any()
    assert replicas.all_replicates_retained.all()


def test_old_pool_is_not_used_for_fresh_estimates_or_acceptance():
    fresh, old = streams()
    before = summarize(fresh, old)
    for s in old:
        for name in (BASELINE, PRIMARY):
            s['values'][name]['lower'] += 100.
            s['values'][name]['upper'] += 100.
    after = summarize(fresh, old)
    np.testing.assert_array_equal(before['price_comparison'].price, after['price_comparison'].price)
    np.testing.assert_array_equal(before['price_comparison'].price_error_budget,
                                  after['price_comparison'].price_error_budget)
    assert not after['planning_comparison'].historical_paths_used_for_acceptance.any()


def test_gap_shift_tails_and_seed_contrasts_are_descriptive_not_proof_of_noise():
    fresh, old = streams()
    t = summarize(fresh, old)
    shift = t['gap_shift'].iloc[0]
    assert shift.difference_se == pytest.approx(np.hypot(shift.fresh_gap_se, shift.old_gap_se))
    assert not shift.difference_certified_as_noise_only
    assert not shift.policy_and_dual_errors_identified
    tails = t['gap_distribution'].query('scope == "pooled"')
    assert set(tails.paths) == {49152, 196608}
    assert tails.count_above_010.eq(3).all()
    assert tails.maximum.eq(.25).all()
    assert len(t['seed_consistency']) == 12
    assert not t['seed_consistency'].contrasts_are_mutually_independent.any()
    assert not t['seed_consistency'].used_for_selection.any()
    assert not t['planning_comparison'].projection_is_acceptance_result.any()


@pytest.mark.parametrize('kind', ['omitted_stream', 'replaced_seed', 'reordered_seeds', 'duplicate_seed',
    'omitted_path', 'nonfinite', 'extra_endpoint', 'missing_estimator', 'changed_gap',
    'missing_prior_stream', 'replaced_prior_seed', 'fresh_prior_overlap'])
def test_omission_replacement_or_tampering_cannot_change_the_fixed_acceptance_pool(kind):
    fresh, old = streams()
    if kind == 'omitted_stream': fresh.pop()
    elif kind == 'replaced_seed': fresh[0]['seed'] = 999
    elif kind == 'reordered_seeds': fresh = fresh[::-1]
    elif kind == 'duplicate_seed': fresh[0]['seed'] = fresh[1]['seed']
    elif kind == 'omitted_path': fresh[0]['values'][PRIMARY]['lower'] = fresh[0]['values'][PRIMARY]['lower'][:-1]
    elif kind == 'nonfinite': fresh[0]['values'][PRIMARY]['lower'][0] = np.inf
    elif kind == 'extra_endpoint': fresh[0]['values'][PRIMARY]['raw'] = np.zeros(65536)
    elif kind == 'missing_estimator': del fresh[0]['values'][BASELINE]
    elif kind == 'changed_gap': fresh[0]['values'][PRIMARY]['upper'][0] += 3e-13
    elif kind == 'missing_prior_stream': old.pop()
    elif kind == 'replaced_prior_seed': old[0]['seed'] = 999
    else:
        fresh[0]['seed'] = PRIOR_SEEDS[0]
        with pytest.raises(ValueError, match='independent of the historical pool'):
            summarize_precision(fresh, expected_seeds=(PRIOR_SEEDS[0], *FRESH_SEEDS[1:]), prior_streams=old)
        return
    with pytest.raises(ValueError): summarize(fresh, old)


def test_analysis_module_imports_no_pricing_or_collector_code():
    import rvlab.experiments._pricing_rough_mc_precision_statistics as module
    tree = ast.parse(Path(module.__file__).read_text())
    names = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any('pricing' in name or 'collector' in name for name in names)
