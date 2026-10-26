"""Every prespecified delta group is shown without selecting favorable estimates."""
from copy import deepcopy

import numpy as np
import pandas as pd

from rvlab.experiments._pricing_rough_shared_delta_reporting import (
    add_shared_delta_reporting, latest_shared_delta_state,
    shared_delta_refinement_summary, shared_delta_sampling_summary)
from rvlab.experiments._pricing_rough_shared_delta_statistics import (
    RELATIVE, summarize_shared_delta)


def _result():
    streams = []; expected = []
    for seed, steps, reps in ((1, 128, ['original', 'micro']),
            (2, 128, ['original', 'micro']), (3, 256, ['micro'])):
        n = 8; x = 100*RELATIVE; z = np.tile([-1., 1.], n//2)
        raw = 2.86-.4*x[None, :]+.1*z[:, None]*(1+x[None, :])
        stable = 2.86-.4*x[None, :]+.01*z[:, None]*(1+x[None, :])
        representations = {}
        for rep in reps:
            slope = -.4+(.001 if rep == 'micro' else 0.)
            shared = 2.86+slope*x[None, :]+.001*z[:, None]*(1+x[None, :])
            representations[rep] = {'raw': raw.copy(), 'stable': stable.copy(),
                'shared': shared, 'stops': np.tile([1, 2, 64, 2, 1], (n, 1))}
        streams.append({'seed': seed, 'steps': steps, 'relative_spots': RELATIVE,
            'initial_spots': 100*(1+RELATIVE), 'representations': representations})
        expected.append({'seed': seed, 'steps': steps, 'paths': n, 'representations': reps})
    tables = summarize_shared_delta(streams, expected_streams=expected)
    tables['development_trials'] = pd.DataFrame([
        {'trial': 'Exact four-step conditioning', 'physical_steps': 128,
         'evaluated_paths': 16, 'spots': 5, 'elapsed_seconds': 10.,
         'projected_paths': 4096, 'projected_seconds': 2560., 'outcome': 'Too expensive',
         'used_for_acceptance': False, 'note': 'Development projection only'},
        {'trial': 'Retained failed projection', 'physical_steps': 128,
         'evaluated_paths': 16, 'spots': 5, 'elapsed_seconds': 1.,
         'projected_paths': np.nan, 'projected_seconds': np.nan, 'outcome': 'Not available',
         'used_for_acceptance': False, 'note': 'No substitute projection'}])
    tables['acceptance'] = pd.DataFrame([
        {'check': 'sampling', 'checks': 6, 'passed': 6, 'all_pass': True},
        {'check': 'bump', 'checks': 3, 'passed': 2, 'all_pass': False}])
    return {'tables': tables, 'figures': {}, 'manifest': {
        'primary_complete': True,
        'fixed_policy_delta_checks_pass': False,
        'american_pricing_ready': False, 'stock_delta_ready': False,
        'optimal_delta_ready': False}}


def test_plots_preserve_all_groups_comparators_and_recorded_failures():
    import matplotlib.pyplot as plt
    result = _result(); before = deepcopy(result)
    rendered = add_shared_delta_reporting(result)
    assert set(rendered['figures']) == {
        'delta_sampling_precision', 'delta_refinement_checks', 'shared_delta_trials'}
    sampling = shared_delta_sampling_summary(result['tables'])
    assert len(sampling) == 18
    assert set(sampling.estimator) == {'raw', 'stable', 'shared'}
    axis = rendered['figures']['delta_sampling_precision'].axes[1]
    points = [line for line in axis.lines if line.get_marker() == 'o']
    assert len(points) == 18
    expected = [row.half_width/row.delta_allowance for _, group in sampling.groupby(
        ['physical_steps', 'representation', 'bump_relative'], sort=False)
        for name in ('raw', 'stable', 'shared')
        for _, row in group.loc[group.estimator.eq(name)].iterrows()]
    np.testing.assert_allclose([line.get_xdata()[0] for line in points], expected)
    assert result['manifest'] == before['manifest'] and not result['figures']
    for name, table in result['tables'].items():
        pd.testing.assert_frame_equal(table, before['tables'][name])
    for fig in rendered['figures'].values():
        plt.close(fig)


def test_all_refinement_families_and_unavailable_projections_remain_visible():
    import matplotlib.pyplot as plt
    result = _result(); rendered = add_shared_delta_reporting(result)
    summary = shared_delta_refinement_summary(result['tables'])
    assert len(summary) == 9
    assert set(summary.check) == {'Bump', 'Control representation',
        'Physical time', 'Time + representation'}
    axis = rendered['figures']['delta_refinement_checks'].axes[0]
    np.testing.assert_allclose([bar.get_width() for bar in axis.patches],
        summary.change_plus_uncertainty/summary.delta_allowance)
    trials = rendered['figures']['shared_delta_trials']
    assert len(trials.axes[0].patches) == 2
    assert 'Projection unavailable' in [text.get_text() for text in trials.axes[1].texts]
    assert 'used_for_acceptance' in result['tables']['development_trials']
    for fig in rendered['figures'].values():
        plt.close(fig)


def test_latest_state_shows_every_finest_group_and_separates_readiness():
    result = _result()
    wrapped = {'manifest': {'rough_shared_delta': result['manifest']},
        'tables': {'rough_shared_delta_'+name: table for name, table in result['tables'].items()}}
    state = latest_shared_delta_state(wrapped).set_index('Question')
    assert [name for name in state.index if 'physical steps' in name] == [
        'original · 128 physical steps', 'micro · 128 physical steps', 'micro · 256 physical steps']
    assert state.loc['Fixed-policy individual checks', 'Current evidence'] == 'False'
    assert state.loc['Declared check families', 'Current evidence'] == 'sampling: 6/6; bump: 2/3'
    assert state.loc['Readiness flags', 'Current evidence'] == (
        'American price: False; stock delta: False; optimal delta: False')
    assert 'no optimal-American' in state.loc['Meaning of a passed check', 'Current evidence']


def test_missing_or_reselected_records_cannot_be_presented_as_validation():
    result = _result()
    assert latest_shared_delta_state({}).empty
    assert shared_delta_sampling_summary({}).empty
    result['tables']['delta_sampling'].loc[0, 'used_for_selection'] = True
    assert shared_delta_sampling_summary(result['tables']).empty
    result['tables']['development_trials']['used_for_acceptance'] = True
    import matplotlib.pyplot as plt
    rendered = add_shared_delta_reporting(result)
    assert 'delta_sampling_precision' not in rendered['figures']
    assert 'shared_delta_trials' not in rendered['figures']
    for fig in rendered['figures'].values():
        plt.close(fig)


def test_duplicate_missing_or_mixed_pool_records_cannot_supply_the_delta_summary():
    for change in ('missing', 'duplicate', 'mixed_paths', 'nonfinite'):
        tables = _result()['tables']; data = tables['delta_sampling']
        if change == 'missing':
            tables['delta_sampling'] = data.iloc[:-1]
        elif change == 'duplicate':
            data.loc[0] = data.iloc[1]
        elif change == 'mixed_paths':
            data.loc[0, 'paths'] = 49152
        else:
            data.loc[0, 'estimate'] = np.nan
        assert shared_delta_sampling_summary(tables).empty
