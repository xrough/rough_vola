"""Read-only views of the frozen lower-policy stock-delta experiment."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd


COLORS = {'raw': '#999999', 'stable': '#8a6aa6', 'shared': '#2878a5'}


def shared_delta_sampling_summary(tables):
    data = tables.get('delta_sampling', pd.DataFrame())
    required = {'physical_steps', 'representation', 'estimator', 'bump_relative',
        'paths', 'estimate', 'half_width', 'delta_allowance', 'sampling_pass',
        'same_frozen_rule', 'used_for_selection'}
    if (not required.issubset(data.columns) or not data.same_frozen_rule.eq(True).all()
            or not data.used_for_selection.eq(False).all()):
        return pd.DataFrame()
    expected = {(steps, rep, bump, est) for steps, rep in (
        (128, 'original'), (128, 'micro'), (256, 'micro'))
        for bump in (.005, .0025) for est in COLORS}
    identities = list(zip(data.physical_steps, data.representation,
        data.bump_relative, data.estimator))
    numeric = data[['paths', 'estimate', 'half_width', 'delta_allowance']].to_numpy(float)
    if (len(identities) != len(expected) or set(identities) != expected or
            not np.isfinite(numeric).all() or np.any(numeric[:, 0] < 2) or
            np.any(numeric[:, 2] < 0) or np.any(numeric[:, 3] <= 0) or
            data.groupby(['physical_steps', 'representation']).paths.nunique().gt(1).any()):
        return pd.DataFrame()
    order = data.assign(_representation_order=data.representation.map({'original': 0, 'micro': 1}))
    return order.sort_values(['physical_steps', '_representation_order', 'bump_relative', 'estimator'],
        ascending=[True, True, False, True]).drop(columns='_representation_order')


def shared_delta_refinement_summary(tables):
    rows = []
    for key, kind, flag in (
            ('bump_refinement', 'Bump', 'bump_check_pass'),
            ('representation_change', 'Control representation', 'tolerance_bias_resolved'),
            ('time_refinement', 'Physical time', 'time_check_pass'),
            ('combined_refinement', 'Time + representation', 'combined_check_pass')):
        data = tables.get(key, pd.DataFrame())
        required = {'estimator', 'change_plus_uncertainty', 'delta_allowance', flag}
        if not required.issubset(data.columns):
            continue
        for _, row in data.loc[data.estimator.eq('shared')].iterrows():
            if key == 'bump_refinement':
                detail = f'{row.representation} · {int(row.physical_steps)} steps'
            elif key == 'representation_change':
                detail = f'{int(row.physical_steps)} steps · h={100*row.bump_relative:g}%'
            elif key == 'time_refinement':
                detail = f'{row.representation} · {int(row.coarse_steps)}→{int(row.fine_steps)} · h={100*row.bump_relative:g}%'
            else:
                detail = f'{row.coarse_representation} {int(row.coarse_steps)}→{row.fine_representation} {int(row.fine_steps)} · h={100*row.bump_relative:g}%'
            rows.append({'check': kind, 'detail': detail,
                'change_plus_uncertainty': row.change_plus_uncertainty,
                'delta_allowance': row.delta_allowance, 'recorded_pass': bool(row[flag])})
    return pd.DataFrame(rows)


def _sampling_figure(tables):
    data = shared_delta_sampling_summary(tables)
    if data.empty:
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    groups = list(data.groupby(['physical_steps', 'representation', 'bump_relative'], sort=False))
    fig, axes = plt.subplots(1, 2, figsize=(13., max(5., .65*len(groups))))
    labels = []
    for i, ((steps, representation, bump), group) in enumerate(groups):
        labels.append(f'{representation} · {int(steps)} steps\nh={100*bump:g}% · N={int(group.paths.iloc[0]):,}')
        for name, offset in (('raw', -.18), ('stable', 0.), ('shared', .18)):
            chosen = group.loc[group.estimator.eq(name)]
            if len(chosen) != 1:
                continue
            row = chosen.iloc[0]
            if name != 'raw':
                axes[0].errorbar(row.estimate, i+offset, xerr=row.half_width,
                    fmt='o', color=COLORS[name], capsize=3,
                    label=name if i == 0 else None)
            axes[1].plot(row.half_width/row.delta_allowance, i+offset, 'o',
                color=COLORS[name], label=name if i == 0 else None)
    axes[0].set(yticks=np.arange(len(groups)), yticklabels=labels,
        xlabel='Frozen lower-policy stock delta · marginal 95% intervals',
        title='Both controlled estimators retain every declared group')
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))
    axes[1].set(yticks=np.arange(len(groups)), yticklabels=[],
        xlabel='95% sampling half-width / unchanged delta allowance',
        title='Raw uncertainty remains visible')
    ratios = data.half_width/data.delta_allowance
    if np.all(ratios > 0.):
        axes[1].set_xscale('log')
    axes[1].axvline(1., color='#b1533c', ls='--', label='Unchanged allowance')
    for axis in axes:
        axis.invert_yaxis(); axis.grid(alpha=.18)
        axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2)
    fig.tight_layout()
    return {'delta_sampling_precision': fig}


def _refinement_figure(tables):
    data = shared_delta_refinement_summary(tables)
    if data.empty:
        return {}
    import matplotlib.pyplot as plt
    colors = {'Bump': '#2878a5', 'Control representation': '#8a6aa6',
        'Physical time': '#79a977', 'Time + representation': '#e9b44c'}
    fig, axis = plt.subplots(figsize=(12., max(4.7, .63*len(data))))
    y = np.arange(len(data))
    axis.barh(y, data.change_plus_uncertainty/data.delta_allowance,
        color=[colors[name] for name in data.check], height=.6)
    axis.axvline(1., color='#b1533c', ls='--', label='Unchanged allowance')
    axis.set(yticks=y, yticklabels=[row.check+'\n'+row.detail for _, row in data.iterrows()],
        xlabel='Absolute change + 95% uncertainty / unchanged delta allowance',
        title='All shared-estimator refinement checks · correlated individual diagnostics')
    axis.invert_yaxis(); axis.grid(axis='x', alpha=.18)
    axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.13))
    fig.tight_layout()
    return {'delta_refinement_checks': fig}


def _trial_figure(tables):
    data = tables.get('development_trials', pd.DataFrame())
    required = {'trial', 'evaluated_paths', 'spots', 'elapsed_seconds', 'projected_paths',
        'projected_seconds', 'outcome', 'used_for_acceptance'}
    if (not required.issubset(data.columns) or data.empty or
            not data.used_for_acceptance.eq(False).all()):
        return {}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13., max(4.7, .85*len(data))))
    y = np.arange(len(data))
    labels = ['\n'.join(textwrap.wrap(str(row.trial), width=30))+
        f'\nN={int(row.evaluated_paths):,} · stocks={int(row.spots)}' for _, row in data.iterrows()]
    for axis, column, title, color in (
            (axes[0], 'elapsed_seconds', 'Measured development workload', '#999999'),
            (axes[1], 'projected_seconds', 'Projection at the stated path budget', '#e9b44c')):
        values = pd.to_numeric(data[column], errors='coerce').to_numpy()
        axis.barh(y, values, color=color, height=.6)
        if np.all(values[np.isfinite(values)] > 0.):
            axis.set_xscale('log')
        axis.set(yticks=y, yticklabels=labels if column == 'elapsed_seconds' else [],
            xlabel='Seconds · workloads differ across trials', title=title)
        for i, (_, row) in enumerate(data.iterrows()):
            projected_paths = pd.to_numeric(row.projected_paths, errors='coerce')
            annotation = str(row.outcome) if column == 'elapsed_seconds' else (
                f'Projected N={int(projected_paths):,}' if pd.notna(projected_paths) and np.isfinite(projected_paths)
                and np.isfinite(values[i]) else 'Projection unavailable')
            axis.text(.02, i, annotation, transform=axis.get_yaxis_transform(),
                va='center', fontsize=9, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        axis.invert_yaxis(); axis.grid(axis='x', alpha=.18)
    fig.suptitle('Development evidence only · no trial selects or replaces fresh validation paths')
    fig.tight_layout()
    return {'shared_delta_trials': fig}


def add_shared_delta_reporting(result):
    figures = dict(result.get('figures', {})); tables = result.get('tables', {})
    figures.update(_sampling_figure(tables)); figures.update(_refinement_figure(tables))
    figures.update(_trial_figure(tables))
    return {**result, 'figures': figures}


def latest_shared_delta_state(result):
    manifest = result.get('manifest', {}).get('rough_shared_delta')
    if not manifest or manifest.get('primary_complete') is not True:
        return pd.DataFrame()
    prefix = 'rough_shared_delta_'
    tables = {key.removeprefix(prefix): table for key, table in result.get('tables', {}).items()
        if key.startswith(prefix)}
    data = shared_delta_sampling_summary(tables)
    rows = [{'Question': 'Current target',
        'Current evidence': 'Stock finite differences of the frozen 64-date lower policy; no dual-upper derivative'},
        {'Question': 'Fixed-policy individual checks',
         'Current evidence': str(manifest.get('fixed_policy_delta_checks_pass', 'unreported'))}]
    checks = tables.get('acceptance', tables.get('delta_gate_summary', pd.DataFrame()))
    if {'check', 'checks', 'passed', 'all_pass'}.issubset(checks.columns):
        rows.append({'Question': 'Declared check families', 'Current evidence': '; '.join(
            f'{row.check}: {int(row.passed)}/{int(row.checks)}'
            for _, row in checks.iterrows())})
    if not data.empty:
        shared = data.loc[data.estimator.eq('shared')]
        if not shared.empty:
            finest = shared.loc[shared.bump_relative.eq(shared.bump_relative.min())]
            for _, row in finest.iterrows():
                rows.append({'Question': f'{row.representation} · {int(row.physical_steps)} physical steps',
                    'Current evidence': f'δ={row.estimate:.6f} ± {row.half_width:.6f} (marginal 95%); N={int(row.paths):,}'})
    rows.extend([
        {'Question': 'Meaning of a passed check',
         'Current evidence': 'Correlated diagnostics for the fixed rule; no optimal-American or continuous-model certificate'},
        {'Question': 'Readiness flags', 'Current evidence':
         f"American price: {manifest.get('american_pricing_ready', 'unreported')}; stock delta: {manifest.get('stock_delta_ready', 'unreported')}; optimal delta: {manifest.get('optimal_delta_ready', 'unreported')}"}])
    return pd.DataFrame(rows)
