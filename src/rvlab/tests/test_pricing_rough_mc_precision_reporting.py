"""Precision figures describe complete recorded pools without selecting subsets."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_mc_precision_reporting import (
    add_mc_precision_reporting, latest_mc_precision_state, precision_price_summary)


def _result():
    rows = []
    for name, primary, se in (('unchanged control', False, .00006),
            ('shared control', True, .000035)):
        gap = .0011; gap_se = .000012; allowance = .001286
        budget = gap + 1.96*(se+gap_se)
        rows.append({'estimator': name, 'primary': primary, 'paths': 196608,
            'price': 2.86, 'price_se': se, 'gap': gap,
            'lower_sampling_term': 1.96*se, 'gap_sampling_term': 1.96*gap_se,
            'price_error_budget': budget, 'price_allowance': allowance,
            'price_budget_pass': budget <= allowance, 'primary_acceptance_pool': True,
            'prior_paths_pooled_for_acceptance': False, 'used_for_selection': False})
    planning = pd.DataFrame([{'estimator': row['estimator'],
        'projected_lower_se': row['price_se']*.9, 'actual_lower_se': row['price_se'],
        'projection_is_acceptance_result': False} for row in rows])
    replicas = pd.DataFrame([{**row, 'paths': 49152, 'replicate': k,
        'price_error_budget': .00115 + .00008*k, 'primary_acceptance_pool': False,
        'all_replicates_retained': True, 'stopping_or_selection_role': False}
        for k in range(4) for row in rows])
    tails = pd.DataFrame([{'cohort': label, 'scope': scope, 'seed': seed,
        'paths': n, 'p99': .02, 'p999': .06, 'maximum': maximum,
        'top1pct_gap_share': share, 'diagnostic_only': True}
        for label, n, maximum, share in (
            ('historical 49152-path pool', 49152, .56, .32),
            ('fresh 196608-path pool', 196608, .8, .35))
        for scope, seed in (('pooled', None), ('one seed', 1))])
    return {'tables': {'price_comparison': pd.DataFrame(rows), 'planning_comparison': planning,
        'fixed_budget_replicates': replicas, 'gap_distribution': tails}, 'figures': {},
        'manifest': {'prior_paths_used_for_acceptance': False,
            'american_pricing_ready': False, 'stock_delta_ready': False,
            'new_control_stock_delta_validated': False}}


def test_figures_read_recorded_components_without_baseline_column_or_mutation():
    import matplotlib.pyplot as plt
    result = _result(); before = deepcopy(result)
    rendered = add_mc_precision_reporting(result)
    assert set(rendered['figures']) == {'complete_pool_budget', 'precision_and_replicas', 'gap_tail_context'}
    data = result['tables']['price_comparison']
    assert 'baseline' not in data
    axis = rendered['figures']['complete_pool_budget'].axes[0]
    widths = np.array([patch.get_width() for patch in axis.patches]).reshape(3, 2).sum(axis=0)
    np.testing.assert_allclose(widths, data.price_error_budget/data.price_allowance)
    assert result['manifest'] == before['manifest'] and not result['figures']
    for name, table in result['tables'].items():
        pd.testing.assert_frame_equal(table, before['tables'][name])
    for figure in rendered['figures'].values():
        plt.close(figure)


def test_all_four_replicas_remain_visible_with_complete_pool_separate():
    import matplotlib.pyplot as plt
    rendered = add_mc_precision_reporting(_result())
    axis = rendered['figures']['precision_and_replicas'].axes[1]
    points = [line for line in axis.lines if line.get_marker() in {'o', 's'}]
    assert [len(line.get_xdata()) for line in points] == [4, 1, 4, 1]
    assert axis.get_title() == 'No passing subset can replace the complete pool'
    assert 'diagnostic only' in axis.get_xlabel()
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ['1', '2', '3', '4', 'Full pool']
    for figure in rendered['figures'].values():
        plt.close(figure)


def test_tail_plot_uses_both_pooled_cohorts_and_labels_unequal_sample_sizes():
    import matplotlib.pyplot as plt
    rendered = add_mc_precision_reporting(_result())
    figure = rendered['figures']['gap_tail_context']; axis = figure.axes[0]
    assert axis.get_yscale() == 'log'
    assert 'unequal sample sizes' in axis.get_title()
    assert len(axis.lines) == 2
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [.02, .06, .56])
    np.testing.assert_allclose(axis.lines[1].get_ydata(), [.02, .06, .8])
    assert [bar.get_height() for bar in figure.axes[1].patches] == [.32, .35]
    for fig in rendered['figures'].values():
        plt.close(fig)


def test_latest_state_keeps_recorded_sampling_flag_separate_from_readiness():
    result = _result()
    wrapped = {'manifest': {'rough_mc_precision': result['manifest']},
        'tables': {'rough_mc_precision_'+name: data for name, data in result['tables'].items()}}
    state = latest_mc_precision_state(wrapped).set_index('Question')
    assert state.loc['Frozen primary', 'Current evidence'] == 'shared control; N=196,608'
    assert 'recorded sampling check: True' in state.loc['Price budget / allowance', 'Current evidence']
    assert state.loc['Historical paths pooled for acceptance', 'Current evidence'] == 'False'
    assert state.loc['Smaller replicas', 'Current evidence'] == 'All four retained; diagnostic only'
    assert state.loc['Readiness flags', 'Current evidence'] == 'American price: False; stock delta: False'


def test_absent_or_incomplete_evidence_cannot_fabricate_a_precision_result():
    result = {'tables': {'failures': pd.DataFrame([{'error': 'unfinished fixed campaign'}])},
        'manifest': {}, 'figures': {}}
    assert not add_mc_precision_reporting(result)['figures']
    assert precision_price_summary(result['tables']).empty
    assert latest_mc_precision_state(result).empty


def test_replica_or_partial_pool_rows_cannot_supply_the_primary_summary():
    result = _result()
    result['tables']['price_comparison']['primary_acceptance_pool'] = False
    assert precision_price_summary(result['tables']).empty


@pytest.mark.parametrize('column,value', [
    ('paths', 49152), ('primary', False), ('estimator', 'different estimator'),
    ('prior_paths_pooled_for_acceptance', True), ('used_for_selection', True)])
def test_misbound_historical_or_reselected_records_cannot_look_like_complete_evidence(column, value):
    result = _result()
    result['tables']['price_comparison'].loc[1, column] = value
    assert precision_price_summary(result['tables']).empty
