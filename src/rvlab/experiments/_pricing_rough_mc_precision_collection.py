"""Read-only collection of the predeclared, independent Monte Carlo extension.

The old pool is audited as historical evidence. It is never added to the new
acceptance pool, and a partial new campaign cannot supply an accepted result.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_gap_time_stopping import _preserved
from ._pricing_rough_policy_validation import _entry
from ._pricing_rough_price_budget_collection import verify_sources, verify_inputs
from ._pricing_rough_mc_precision_statistics import REPLICA_DECLARATION, summarize_precision


ROOT = Path(__file__).resolve().parents[3]
SEEDS = (1119005042, 1120005042, 1121005042)
STREAMS = [{'seed': s, 'paths': 65536} for s in SEEDS]
PARENT = 'rough_price_budget_v1/shared_control_v1'
PARENT_PROFILE = 'c6c6981a6c2e915cb097403a483b4258767a870773f7ee704123ed453f8320d9'
SELECTED_FIELDS = frozenset('''actual_time_grid basis candidate3_lower candidate3_upper
    current_martingale discounts eligible european_martingale european_proxy factors
    gh16_computed_mask gh16_continuation gh16_raw gh16_stops grid_actual grid_corrections
    grid_expected intrinsic_martingale latest_martingale local_actual local_expected
    micro_boundary_martingale payoff premium_martingale prior_local_corrections
    prior_martingale shared_lower shared_upper shift stable_old_martingale stocks
    tail_corrections tail_gates tail_margin target_actual_premiums target_expected_european
    target_expected_premiums target_martingales times unchanged_lower unchanged_upper'''.split())
CONDITIONAL_FIELDS = frozenset('''absolute_path_indices conditional_actual_next
    conditional_european_estimate conditional_european_martingale conditional_european_payoff
    conditional_expected_next conditional_log_means conditional_variances physical_time_grid
    replay_physical_factors replay_physical_stocks shared_control'''.split())


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_design(design):
    expected = {
        'version': 'rough_mc_precision_v1', 'streams': STREAMS,
        'total_paths': 196608, 'parent_campaign': PARENT,
        'frozen_policy': 'GH16 blend1', 'frozen_dual_control': 'target 4 bins penalty0',
        'physical_steps': 128, 'exercise_intervals': 64, 'features': 193,
        'fixed_budget_replicas': REPLICA_DECLARATION,
        'no_fresh_selection_or_fitting': True,
        'no_empirical_valuation_recentering': True,
        'no_regression_intercept_at_valuation': True,
        'old_paths_used_for_acceptance': False, 'all_paths_retained': True,
        'both_endpoints_same_shared_shift': True, 'gap_pointwise_unchanged': True,
        'new_policy_fits': 0, 'new_dual_control_fits': 0,
        'price_atol': .001, 'price_rtol': 1e-4,
        'delta_atol': .002, 'delta_rtol': .005,
        'confidence_multiplier': 1.96, 'spread_fraction': .1,
        'tolerances_changed': False,
    }
    if any((design.get(k) is not v) if isinstance(v, bool) else design.get(k) != v
            for k, v in expected.items()):
        raise ValueError('Changed independent Monte Carlo design, pool, estimator or tolerances')
    forbidden = design.get('forbidden_prior_seeds')
    if (not isinstance(forbidden, list) or len(forbidden) != len(set(forbidden))
            or any(isinstance(s, bool) or not isinstance(s, int) for s in forbidden)
            or set(SEEDS) & set(forbidden)):
        raise ValueError('The fresh Monte Carlo seed firewall is missing or overlaps spent paths')


def validate_record(record, design, *, index, design_sha256, coefficient_sha256):
    expected = {'index': index, **STREAMS[index], 'design_sha256': design_sha256,
        'coefficient_sha256': coefficient_sha256, 'source_sha256': design['source_sha256'],
        'sources_unchanged': True, 'new_fitting_or_selection': False,
        'all_paths_retained': True, 'gap_pointwise_unchanged': True,
        'empirical_valuation_recentering': False}
    if (any((record.get(k) is not v) if isinstance(v, bool) else record.get(k) != v
            for k, v in expected.items())
            or set(record.get('sample_sha256', {})) != {'selected', 'conditional'}
            or not datetime.fromisoformat(design['created_at']) <
                datetime.fromisoformat(record['started_at']) <
                datetime.fromisoformat(record['completed_at'])
            or not np.isfinite(record.get('elapsed_seconds', np.nan))
            or record['elapsed_seconds'] <= 0):
        raise ValueError('Changed or incomplete fresh stream, source identity or chronology')


def _load_archive(path, recorded_schema):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}
    if schema != recorded_schema:
        raise ValueError('The saved independent array schema changed')
    return arrays


def _prior_streams(parent):
    result = []
    for i, seed in enumerate((1106005042, 1107005042, 1108005042)):
        with np.load(parent/f'stream_{i}_shift_samples.npz', allow_pickle=False) as saved:
            values = {label: {'lower': saved[prefix+'_lower'], 'upper': saved[prefix+'_upper']}
                for label, prefix in (('unchanged control', 'unchanged'), ('shared control', 'shared'))}
        result.append({'seed': seed, 'values': values})
    return result


def validate_preflight(folder, output, final_design):
    """Recompute spent comparisons; a saved 'passed' flag is insufficient."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    _, entries = _preserved(folder)
    design = read(folder/'design_before_evaluation.json')
    manifest = read(folder/'manifest.json')
    indices = np.r_[np.arange(251), 511, 512, 1023, 4095]
    if (design.get('version') != 'rough_mc_precision_spent_equivalence_v1'
            or design.get('seeds') != [1094005042, 1095005042]
            or design.get('full_stream_paths') != 4096 or design.get('fresh_paths') != 0
            or design.get('equivalence_atol') != 2e-13
            or not np.array_equal(design.get('indices'), indices)
            or design.get('physical_random_batch_size') != 512
            or design.get('european65_and129_kept_separate') is not True
            or manifest.get('complete') is not True or manifest.get('all_checks_passed') is not True
            or manifest.get('sources_unchanged') is not True
            or any(manifest.get(k) != v for k, v in design.items())
            or manifest.get('source_sha256') != final_design.get('source_sha256')
            or not datetime.fromisoformat(manifest['completed_at']) <
                datetime.fromisoformat(final_design['created_at'])):
        raise ValueError('The current source lacks a complete pre-fresh spent equivalence test')
    verify_sources(folder, design, set(final_design['source_sha256']))
    entries += verify_inputs(design, output)
    if sha(folder/'equivalence_checks.json') != manifest.get('checks_sha256'):
        raise ValueError('The spent implementation-check record changed')
    records = manifest.get('records', [])
    if len(records) != 2:
        raise ValueError('Both spent implementation-check streams are required')
    rows = []
    control_folder = output/'rough_bellman_validation_v1/bellman_control_v1'
    fit_folder = output/'rough_price_budget_v1/spent_shared_fit_v1'
    anchor_folder = output/'rough_price_budget_v1/spent_terminal_anchor_v1'
    references = {str((base/name).relative_to(output)) for i in range(2) for base, name in (
        (control_folder, f'final_{i}_samples.npz'), (fit_folder, f'spent_{i}_basis.npz'),
        (anchor_folder, f'stream_{i}_samples.npz'))}
    if not references.issubset(design.get('input_sha256', {})):
        raise ValueError('A mandatory original spent path, basis or anchor reference was not frozen')
    for i, seed in enumerate((1094005042, 1095005042)):
        record = records[i]
        selected_path = folder/f'stream_{i}_selected_samples.npz'
        conditional_path = folder/f'stream_{i}_conditional_samples.npz'
        elapsed = record.get('elapsed_seconds', np.nan)
        if (record.get('seed') != seed or record.get('paths') != 4096
                or record.get('evaluated') != len(indices)
                or record.get('selected_sha256') != sha(selected_path)
                or record.get('conditional_sha256') != sha(conditional_path)
                or isinstance(elapsed, bool) or not np.isfinite(elapsed) or elapsed <= 0
                or not datetime.fromisoformat(design['created_at']) <=
                    datetime.fromisoformat(record['started_at']) <
                    datetime.fromisoformat(record['completed_at']) <=
                    datetime.fromisoformat(manifest['completed_at'])):
            raise ValueError('A spent path subset, archive or seed was replaced')
        with np.load(selected_path, allow_pickle=False) as selected, np.load(
                control_folder/f'final_{i}_samples.npz', allow_pickle=False) as original:
            if set(selected.files) != SELECTED_FIELDS:
                raise ValueError('A required selected implementation-check primitive was omitted or added')
            for name in selected.files:
                value = selected[name]
                if (value.dtype.kind not in 'bifu' or np.any(np.isinf(value))
                        or (name != 'gh16_continuation' and not np.isfinite(value).all())):
                    raise ValueError('A selected implementation-check primitive has invalid type or nonfinite values')
            pairs = [(k, selected[k], original[k] if k in ('times', 'discounts', 'eligible', 'actual_time_grid')
                else original[k][indices]) for k in selected.files if k in original.files]
            with np.load(fit_folder/f'spent_{i}_basis.npz', allow_pickle=False) as fit:
                pairs.extend((k, selected[k], fit[ref][indices]) for k, ref in
                    (('basis', 'gated_anchor'), ('unchanged_lower', 'lower'), ('unchanged_upper', 'upper')))
        with np.load(conditional_path, allow_pickle=False) as conditional, np.load(
                anchor_folder/f'stream_{i}_samples.npz', allow_pickle=False) as original:
            if set(conditional.files) != CONDITIONAL_FIELDS:
                raise ValueError('A required conditional implementation-check primitive was omitted or added')
            if any(conditional[k].dtype.kind not in 'bifu' or not np.isfinite(conditional[k]).all()
                    for k in conditional.files):
                raise ValueError('A conditional implementation-check primitive has invalid type or nonfinite values')
            if not np.array_equal(conditional['absolute_path_indices'], indices):
                raise ValueError('Spent implementation checks changed absolute RNG indices')
            pairs.extend((k, conditional[k], original[k] if k == 'physical_time_grid' else original[k][indices])
                for k in conditional.files if k != 'absolute_path_indices')
        for name, actual, expected in pairs:
            if actual.shape != expected.shape:
                raise ValueError('Spent implementation-check array dimensions changed')
            if actual.dtype.kind == 'b':
                error = float(np.any(actual != expected))
            else:
                if not np.array_equal(np.isnan(actual), np.isnan(expected)):
                    raise ValueError('The equivalent stopping mask changed NaN positions')
                finite = np.isfinite(expected)
                if np.any(np.isinf(actual)) or np.any(np.isinf(expected)):
                    raise ValueError('An equivalent spent primitive is infinite')
                error = float(np.max(abs(actual[finite]-expected[finite]), initial=0.))
            if error > 2e-13:
                raise ValueError(f'The selected-only implementation changed a spent {name}')
            rows.append({'seed': seed, 'field': name, 'max_abs_error': error,
                'absolute_subset_paths': len(indices), 'original_stream_paths': 4096,
                'fresh_paths': 0, 'acceptance_role': False})
    return {'tables': {'implementation_checks': pd.DataFrame(rows),
        'implementation_runtime': pd.DataFrame([{
            'seed': r['seed'], 'original_stream_paths': r['paths'], 'evaluated_paths': r['evaluated'],
            'started_at': r['started_at'], 'completed_at': r['completed_at'],
            'elapsed_seconds': r['elapsed_seconds'], 'cpu_seconds': r.get('cpu_seconds'),
            'parent_cpu_seconds': r.get('cpu_seconds_parent_only'),
            'parallel_seconds': r.get('parallel_seconds'), 'serial_seconds': r.get('serial_seconds'),
            'role': 'spent performance projection only', 'used_for_acceptance': False}
            for r in records])}, 'input_files': entries}


def collect_precision_campaign(folder):
    from ._pricing_rough_price_budget_collection import price_budget_study
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    from ._pricing_rough_mc_precision_samples import validate_selected_samples
    from ._pricing_rough_frozen_policy import _decode_models
    from types import SimpleNamespace

    folder = Path(folder).resolve()
    output = folder.parents[1]
    inventory, entries = _preserved(folder)
    design = read(folder/'design_before_valuation.json')
    validate_design(design)
    manifest = read(folder/'manifest.json')
    dsha = sha(folder/'design_before_valuation.json')
    if (manifest.get('complete') is not True or manifest.get('completed_paths') != 196608
            or manifest.get('all_streams_retained') is not True
            or manifest.get('design_sha256') != dsha
            or any(manifest.get(k) != v for k, v in design.items())):
        raise ValueError('All three predeclared streams must complete before primary collection')
    parent = output/PARENT
    parent_design = read(parent/'design_before_valuation.json')
    expected_sources = set(parent_design['source_sha256']) | {
        'src/rvlab/experiments/_pricing_rough_mc_precision.py', 'examples/rough_mc_precision.py'}
    if len(expected_sources) != 54:
        raise ValueError('The fixed numerical dependency set changed')
    verify_sources(folder, design, expected_sources)
    entries += verify_inputs(design, output)
    preflight_folder = (output/design['preflight_folder']).resolve()
    if (not preflight_folder.is_relative_to((output/'rough_mc_precision_v1').resolve())
            or sha(preflight_folder/'manifest.json') != design['preflight_manifest_sha256']):
        raise ValueError('The current implementation-equivalence trial was omitted or replaced')
    preflight = validate_preflight(preflight_folder, output, design)
    entries += preflight['input_files']
    previous_trials = []
    for trial in sorted(folder.parent.glob('spent_equivalence_v*')):
        if trial.resolve() == preflight_folder:
            continue
        # Earlier versions are historical source snapshots, not current code.
        # They cannot replace the final source's pre-fresh equivalence check.
        _, trial_entries = _preserved(trial)
        entries += trial_entries
        recorded = read(trial/'manifest.json')
        if recorded.get('fresh_paths') != 0 or recorded.get('complete') is not True:
            raise ValueError('An earlier implementation trial changed its historical role')
        previous_trials.append({'trial': trial.name, 'fresh_paths': 0,
            'all_checks_passed': recorded.get('all_checks_passed'),
            'current_implementation_proof': False, 'used_for_acceptance': False,
            'source_snapshot_retained': True})
    if (design.get('parent_design_sha256') != sha(parent/'design_before_valuation.json')
            or not set(parent_design['input_sha256']).issubset(design['input_sha256'])
            or any(design['input_sha256'][k] != v for k, v in parent_design['input_sha256'].items())
            or set(parent_design['forbidden_prior_seeds']) | {1106005042, 1107005042, 1108005042}
                != set(design['forbidden_prior_seeds'])):
        raise ValueError('A parent estimator dependency or historical seed was omitted')
    parent_result = price_budget_study(SimpleNamespace(output_dir=output, progress=lambda _: None), compute_figures=False)
    if parent_result['manifest']['profile_fingerprint'] != PARENT_PROFILE:
        raise ValueError('Historical precision evidence changed before the new campaign')
    entries += parent_result['manifest']['input_files']
    weights_path = folder/'weights_before_final.json'
    coefficient_sha = sha(weights_path)
    seal = read(weights_path)
    if (coefficient_sha != design.get('coefficient_sha256')
            or coefficient_sha != sha(parent/'weights_before_final.json')
            or not datetime.fromisoformat(seal['sealed_at']) < datetime.fromisoformat(design['created_at'])):
        raise ValueError('New paths preceded or changed the original training-only coefficient seal')
    bundle = load_larger_policy_bundle(output/'rough_larger_policy_v1/32768_paths_two_starts')
    control_folder = output/'rough_bellman_validation_v1/bellman_control_v1'
    control_design = read(control_folder/'design_before_training.json')
    target_seal = read(control_folder/'selection_before_final.json')
    tail_seal = read(output/'rough_tail_attribution_v1/joint_switch_v1/selection_before_final.json')
    proxy_folder = output/'rough_bellman_validation_v1/spent_proxy_aligned_v2'
    proxy_seal = read(proxy_folder/'proxy_before_fresh_paths.json')
    with np.load(proxy_folder/proxy_seal['coefficient_archive'], allow_pickle=False) as saved:
        models = (None,) + _decode_models(proxy_seal['models'], saved) + (None,)
    streams, records, execution, checks, probes = [], [], [], [], []
    for i, spec in enumerate(STREAMS):
        record = read(folder/f'stream_{i}_manifest.json')
        validate_record(record, design, index=i, design_sha256=dsha, coefficient_sha256=coefficient_sha)
        archives = {}
        for kind, suffix in (('selected', 'samples'), ('conditional', 'conditional_samples')):
            path = folder/f'stream_{i}_{suffix}.npz'
            if sha(path) != record['sample_sha256'][kind]:
                raise ValueError('A complete fresh-path archive changed')
            archives[kind] = _load_archive(path, record['array_schema'][kind])
            required_fields = SELECTED_FIELDS if kind == 'selected' else CONDITIONAL_FIELDS
            if set(archives[kind]) != required_fields:
                raise ValueError('A required complete fresh-path field was omitted or added')
        arrays, conditional = archives['selected'], archives['conditional']
        selected = validate_selected_samples({**arrays, **conditional}, bundle,
            control_design['prior_control'], control_design['grid_control'], tail_seal, target_seal,
            **spec, coefficients=seal['weights'], models=models)
        streams.append({'seed': spec['seed'], 'values': selected['values']})
        checks.append({'seed': spec['seed'], 'paths': spec['paths'], **selected['checks'],
            'all_selected_primitives_verified': True, 'all_paths_retained': True})
        probes.extend({'seed': spec['seed'], **p} for p in selected['probes'])
        execution.append({**spec, 'started_at': record['started_at'], 'completed_at': record['completed_at'],
            'elapsed_seconds': record['elapsed_seconds'], 'cpu_seconds': record.get('cpu_seconds'),
            'utc_span_seconds': (datetime.fromisoformat(record['completed_at'])-
                datetime.fromisoformat(record['started_at'])).total_seconds(),
            'used_for_fitting_or_selection': False,
            'all_paths_retained': True})
        records.append(record)
        # On an8GiB machine, do not retain one full payload while loading the
        # next. Only the small endpoint arrays and audit summaries survive.
        del archives, arrays, conditional, selected
    if manifest.get('records') != records:
        raise ValueError('A declared independent stream was omitted or replaced')
    tables = summarize_precision(streams, expected_seeds=SEEDS, prior_streams=_prior_streams(parent))
    tables.update(preflight['tables'])
    tables['prior_implementation_trials'] = pd.DataFrame(previous_trials)
    tables.update(path_identity_checks=pd.DataFrame(checks), conditional_probes=pd.DataFrame(probes),
        execution=pd.DataFrame(execution))
    return {'tables': tables, 'manifest': manifest, 'input_files': entries,
        'parent_analysis_sources': parent_result['manifest']['analysis_sources']}


def mc_precision_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError('The declared independent precision research seed is 42')
    if not isinstance(compute_figures, bool):
        raise TypeError('compute_figures must be boolean')
    context.progress('Auditing the complete new 196,608-path pool with unchanged price tolerances')
    output = Path(context.output_dir).resolve()
    result = collect_precision_campaign(output/'rough_mc_precision_v1/independent_v1')
    unique = {}
    for row in result['input_files']:
        key = str(Path(row['path']).resolve())
        entry = {**row, 'path': key}
        if key in unique and unique[key] != entry:
            raise ValueError('Conflicting independent-campaign input fingerprints')
        unique[key] = entry
    entries = [unique[k] for k in sorted(unique)]
    names = ('_pricing_rough_mc_precision_collection.py', '_pricing_rough_mc_precision_samples.py',
        '_pricing_rough_mc_precision_statistics.py', '_pricing_rough_price_budget_collection.py',
        '_pricing_rough_price_budget_samples.py', '_pricing_rough_price_budget_basis.py')
    names = sorted(set(names) | set(result['parent_analysis_sources']) | {
        '_pricing_rough_policy_validation.py', '_pricing_rough_gap_time_stopping.py'})
    sources = {name: sha(Path(__file__).with_name(name)) for name in names}
    fingerprint = hashlib.sha256(json.dumps({'version': 'rough_mc_precision_v1', 'files': entries,
        'analysis_sources': sources}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    tables = result['tables']
    tables['provenance'] = pd.DataFrame(entries)
    tables['configuration'] = pd.DataFrame([
        {'item': 'Synthetic problem', 'value': 'Spot100, strike100 put;60/365years; four-factor lift; explicitly empty cash-dividend schedule'},
        {'item': 'Unchanged generating parameters', 'value': 'v0=.04, theta=.045, kappa=1.5, vol_of_vol=.3, rho=-.6, hurst=.15'},
        {'item': 'Frozen estimator', 'value': 'GH16 blend1 policy, target4bin dual,193 common-control coefficients;128physical steps/64exercise dates'},
        {'item': 'Independent acceptance pool', 'value': '196608fresh paths;1119005042/1120005042/1121005042×65536; no old paths pooled'},
        {'item': 'Secondary diagnostics', 'value': 'All3seed checks and all4disjoint49152path replicas; no selection or early stopping'},
        {'item': 'Unchanged price budget', 'value': 'gap+1.96*(lowerSE+gapSE) ≤ .001+1e-4|V|'},
        {'item': 'Implementation proof', 'value': 'Spent absolute path replay against the old executor before the new source/path freeze'},
        {'item': 'Scope', 'value': 'Finite-law price sampling gate only; kernel, time, continuous-exercise and stock-delta uncertainty remain separate'},
    ])
    primary = tables['price_comparison'].loc[tables['price_comparison'].primary].iloc[0]
    manifest = {'backend': 'rough_heston_mc_precision', 'study_version': 'rough_mc_precision_v1',
        'data_kind': 'synthetic', 'model': 'rough_heston', 'exercise_style': 'american',
        'physical_steps': 128, 'exercise_intervals': 64, 'kernel_factors': 4,
        'profile_fingerprint': fingerprint, 'input_files': entries, 'analysis_sources': sources,
        'campaign': result['manifest'], 'primary_complete': True, 'primary_paths': 196608,
        'seed_count': 3, 'parent_profile_fingerprint': PARENT_PROFILE,
        'prior_paths_used_for_acceptance': False, 'fixed_budget_replicas': REPLICA_DECLARATION,
        'historical_replay': False, 'current_engine_execution': False,
        'execution_mode': 'read-only reconstruction of a separately sealed, predeclared fresh campaign',
        'primary_policy_fits': 0, 'primary_dual_control_fits': 0, 'primary_common_control_fits': 0,
        'all_paths_retained': True, 'gap_pointwise_unchanged': True, 'tolerances_changed': False,
        'sampling_gate_pass': bool(primary.price_budget_pass),
        'price_atol': .001, 'price_rtol': 1e-4, 'delta_atol': .002, 'delta_rtol': .005,
        'confidence_multiplier': 1.96, 'spread_fraction': .1,
        'sampling_gate_only': True, 'american_pricing_ready': False, 'stock_delta_ready': False,
        'new_control_stock_delta_validated': False, 'optimal_delta_ready': False,
        'policy_and_dual_slack_separately_identified': False,
        'limitations': ['Only the complete new pool supplies this sampling gate; smaller replicas are diagnostic.',
            'One four-factor, 128-step, 64-exercise-date synthetic problem; no continuous-model accuracy certificate.',
            'The unchanged gap combines policy error and dual slack; these are not separately identified.',
            'The common price control has not been validated for stock deltas.',
            'Independent SPY spot and historically available dividend expectations remain missing.']}
    collected = {'tables': tables, 'figures': {}, 'manifest': manifest}
    if compute_figures:
        from ._pricing_rough_mc_precision_reporting import add_mc_precision_reporting
        collected = add_mc_precision_reporting(collected)
    return collected
