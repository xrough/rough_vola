"""Changing a fixed larger pool or its role cannot make a price fit accepted."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rvlab.experiments import _pricing_rough_mc_precision_collection as c


@pytest.fixture
def design():
    return {'version': 'rough_mc_precision_v1', 'streams': deepcopy(c.STREAMS),
        'total_paths': 196608, 'parent_campaign': c.PARENT,
        'frozen_policy': 'GH16 blend1', 'frozen_dual_control': 'target 4 bins penalty0',
        'physical_steps': 128, 'exercise_intervals': 64, 'features': 193,
        'fixed_budget_replicas': deepcopy(c.REPLICA_DECLARATION),
        'no_fresh_selection_or_fitting': True, 'no_empirical_valuation_recentering': True,
        'no_regression_intercept_at_valuation': True, 'old_paths_used_for_acceptance': False,
        'all_paths_retained': True, 'both_endpoints_same_shared_shift': True,
        'gap_pointwise_unchanged': True, 'new_policy_fits': 0, 'new_dual_control_fits': 0,
        'price_atol': .001, 'price_rtol': 1e-4, 'delta_atol': .002, 'delta_rtol': .005,
        'confidence_multiplier': 1.96, 'spread_fraction': .1, 'tolerances_changed': False,
        'forbidden_prior_seeds': [1106005042, 1107005042, 1108005042],
        'source_sha256': {'module.py': 'frozen'}, 'created_at': '2026-10-20T00:00:00+00:00'}


def test_fixed_design_is_accepted_before_any_files_or_prices_are_read(design):
    c.validate_design(design)


@pytest.mark.parametrize('key,value', [
    ('total_paths', 49152), ('streams', c.STREAMS[:2]),
    ('frozen_policy', 'refitted after final paths'), ('features', 192),
    ('physical_steps', 256), ('exercise_intervals', 128),
    ('old_paths_used_for_acceptance', True), ('no_fresh_selection_or_fitting', False),
    ('no_empirical_valuation_recentering', False), ('all_paths_retained', False),
    ('price_atol', .002), ('price_rtol', 2e-4), ('confidence_multiplier', 1.),
    ('spread_fraction', .2), ('tolerances_changed', True), ('new_policy_fits', 1),
    ('fixed_budget_replicas', {'replicates': 1}), ('forbidden_prior_seeds', list(c.SEEDS)),
    ('forbidden_prior_seeds', [1, 1]), ('all_paths_retained', 1),
])
def test_changed_pool_or_interpretation_is_rejected(design, key, value):
    design[key] = value
    with pytest.raises(ValueError):
        c.validate_design(design)


@pytest.fixture
def record(design):
    return {'index': 0, **c.STREAMS[0], 'design_sha256': 'design',
        'coefficient_sha256': 'coefficient', 'source_sha256': design['source_sha256'],
        'sources_unchanged': True, 'new_fitting_or_selection': False,
        'all_paths_retained': True, 'gap_pointwise_unchanged': True,
        'empirical_valuation_recentering': False,
        'sample_sha256': {'selected': 'all-selected-paths', 'conditional': 'all-conditional-paths'},
        'started_at': '2026-10-20T00:01:00+00:00',
        'completed_at': '2026-10-20T00:02:00+00:00', 'elapsed_seconds': 60.}


def test_complete_record_preserves_the_sealed_stream_and_sources(design, record):
    c.validate_record(record, design, index=0, design_sha256='design', coefficient_sha256='coefficient')


@pytest.mark.parametrize('key,value', [
    ('index', 1), ('seed', 1106005042), ('paths', 65535),
    ('coefficient_sha256', 'reestimated'), ('source_sha256', {'module.py': 'changed'}),
    ('new_fitting_or_selection', True), ('all_paths_retained', False),
    ('sample_sha256', {'selected': 'missing conditional'}),
    ('started_at', '2026-10-19T23:59:00+00:00'),
    ('completed_at', '2026-10-20T00:00:30+00:00'), ('elapsed_seconds', 0.),
])
def test_partial_changed_or_peeked_stream_is_rejected(design, record, key, value):
    record[key] = value
    with pytest.raises(ValueError):
        c.validate_record(record, design, index=0, design_sha256='design', coefficient_sha256='coefficient')


@pytest.mark.parametrize('kwargs,error', [
    ({'seed': 43}, ValueError), ({'seed': True}, ValueError),
    ({'compute_figures': 1}, TypeError),
])
def test_invalid_study_options_fail_before_files_are_accessed(kwargs, error):
    with pytest.raises(error):
        c.mc_precision_study(SimpleNamespace(), **kwargs)
