"""Read-only displays for the complete independent Monte Carlo precision study."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pricing_rough_price_budget_reporting import COMPONENTS


def precision_price_summary(tables):
    data = tables.get('price_comparison', pd.DataFrame())
    required = {'estimator', 'primary', 'paths', 'gap', 'price', 'price_se',
        'lower_sampling_term', 'gap_sampling_term', 'price_error_budget',
        'price_allowance', 'price_budget_pass', 'primary_acceptance_pool',
        'prior_paths_pooled_for_acceptance', 'used_for_selection'}
    if (not required.issubset(data.columns) or
            data.estimator.tolist() != ['unchanged control', 'shared control'] or
            data.primary.tolist() != [False, True] or not data.paths.eq(196608).all() or
            not data.primary_acceptance_pool.eq(True).all() or
            not data.prior_paths_pooled_for_acceptance.eq(False).all() or
            not data.used_for_selection.eq(False).all()):
        return pd.DataFrame()
    return data.copy()


def _budget_figure(tables):
    data = precision_price_summary(tables)
    if data.empty:
        return {}
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(11.5, 4.7))
    y = np.arange(len(data)); left = np.zeros(len(data))
    for label, column, color in COMPONENTS:
        values = (data[column] / data.price_allowance).to_numpy()
        axis.barh(y, values, left=left, height=.58, label=label, color=color)
        left += values
    axis.axvline(1., color='#b1533c', ls='--', label='Unchanged price allowance')
    labels = [str(row.estimator) + ('\nFrozen primary' if row.primary else '\nSame-pool baseline')
        for _, row in data.iterrows()]
    axis.set(yticks=y, yticklabels=labels,
        xlabel='Recorded additive price budget / unchanged allowance',
        title='Only the complete new pool supplies the sampling check')
    axis.invert_yaxis(); axis.grid(axis='x', alpha=.18)
    for tick, primary in zip(axis.get_yticklabels(), data.primary):
        if primary:
            tick.set_fontweight('bold')
    axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2)
    fig.tight_layout()
    return {'complete_pool_budget': fig}


def _precision_figure(tables):
    planning = tables.get('planning_comparison', pd.DataFrame())
    replicas = tables.get('fixed_budget_replicates', pd.DataFrame())
    final = precision_price_summary(tables)
    required_plan = {'estimator', 'projected_lower_se', 'actual_lower_se'}
    required_replica = {'estimator', 'replicate', 'price_error_budget', 'price_allowance',
        'primary_acceptance_pool', 'all_replicates_retained', 'stopping_or_selection_role'}
    if (not required_plan.issubset(planning.columns) or
            not required_replica.issubset(replicas.columns) or final.empty):
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    y = np.arange(len(planning))
    for column, offset, label, color in (
            ('projected_lower_se', -.14, 'Historical variance scaled to new N · projection', '#999999'),
            ('actual_lower_se', .14, 'Complete fresh pool · measured', '#2878a5')):
        axes[0].barh(y+offset, planning[column], height=.25, label=label, color=color)
    axes[0].set(yticks=y, yticklabels=planning.estimator.tolist(),
        xlabel='Lower-price standard error ($)', title='Prediction and independent measurement')
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.6f}'))
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.19), fontsize=9)
    for label, marker, color, offset in (
            ('unchanged control', 'o', '#8a6aa6', -.06),
            ('shared control', 's', '#2878a5', .06)):
        subset = replicas.loc[replicas.estimator.eq(label)].sort_values('replicate')
        if not subset.empty:
            axes[1].plot(subset.replicate+offset,
                subset.price_error_budget/subset.price_allowance,
                marker, color=color, label=label)
        whole = final.loc[final.estimator.eq(label)]
        if len(whole) == 1:
            axes[1].plot([4.4+offset], whole.price_error_budget/whole.price_allowance,
                marker, color=color, markersize=9)
    axes[1].axhline(1., color='#b1533c', ls='--')
    axes[1].axvline(3.7, color='#cccccc', lw=1)
    axes[1].set(xticks=[0, 1, 2, 3, 4.4],
        xticklabels=['1', '2', '3', '4', 'Full pool'],
        xlabel='All four fixed replicas · diagnostic only | primary pool',
        ylabel='Price budget / unchanged allowance',
        title='No passing subset can replace the complete pool')
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2)
    for axis in axes:
        axis.grid(alpha=.18)
    fig.tight_layout()
    return {'precision_and_replicas': fig}


def _tail_figure(tables):
    data = tables.get('gap_distribution', pd.DataFrame())
    required = {'cohort', 'scope', 'paths', 'p99', 'p999', 'maximum', 'top1pct_gap_share'}
    if not required.issubset(data.columns):
        return {}
    pooled = data.loc[data.scope.eq('pooled')].copy()
    if len(pooled) != 2:
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    colors = ('#999999', '#2878a5')
    for i, (_, row) in enumerate(pooled.iterrows()):
        label = f'{row.cohort}\nN={int(row.paths):,}'
        axes[0].plot(np.arange(3), [row.p99, row.p999, row.maximum],
            'o-', color=colors[i], label=label)
        axes[1].bar(i, row.top1pct_gap_share, color=colors[i], width=.55)
    axes[0].set(xticks=[0, 1, 2], xticklabels=['p99', 'p99.9', 'Maximum'],
        ylabel='Pathwise stopping gap ($) · log scale', yscale='log',
        title='Descriptive tails · unequal sample sizes')
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.19), fontsize=9)
    axes[1].set(xticks=[0, 1], xticklabels=[f'{int(row.paths):,} paths' for _, row in pooled.iterrows()],
        ylabel='Share of total gap mass',
        title='Largest 1% of gaps · every path retained')
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.))
    axes[1].set_ylim(0., max(.45, 1.15*pooled.top1pct_gap_share.max()))
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle('Historical comparison is diagnostic; neither cohort selects a new estimator')
    fig.tight_layout()
    return {'gap_tail_context': fig}


def add_mc_precision_reporting(result):
    tables = result.get('tables', {})
    figures = dict(result.get('figures', {}))
    figures.update(_budget_figure(tables))
    figures.update(_precision_figure(tables))
    figures.update(_tail_figure(tables))
    return {**result, 'figures': figures}


def latest_mc_precision_state(result):
    manifest = result.get('manifest', {}).get('rough_mc_precision')
    if not manifest:
        return pd.DataFrame()
    prefix = 'rough_mc_precision_'
    tables = {key.removeprefix(prefix): table for key, table in result.get('tables', {}).items()
        if key.startswith(prefix)}
    data = precision_price_summary(tables)
    primary = data.loc[data.primary.eq(True)] if not data.empty else pd.DataFrame()
    rows = [{'Question': 'Current experiment',
        'Current evidence': 'Complete independent pool; frozen exercise rule, price martingale and shared control'}]
    if len(primary) == 1:
        row = primary.iloc[0]
        rows.extend([
            {'Question': 'Frozen primary', 'Current evidence': f'{row.estimator}; N={int(row.paths):,}'},
            {'Question': 'Price budget / allowance', 'Current evidence':
                f'{row.price_error_budget:.6f} / {row.price_allowance:.6f}; recorded sampling check: {bool(row.price_budget_pass)}'},
            {'Question': 'Separate components', 'Current evidence':
                f'Gap {row.gap:.6f}; lower uncertainty {row.lower_sampling_term:.6f}; gap uncertainty {row.gap_sampling_term:.6f}'}])
    else:
        rows.append({'Question': 'Frozen primary', 'Current evidence': 'No complete final-pool record'})
    rows.extend([
        {'Question': 'Historical paths pooled for acceptance', 'Current evidence': str(manifest.get('prior_paths_used_for_acceptance', 'unreported'))},
        {'Question': 'Smaller replicas', 'Current evidence': 'All four retained; diagnostic only'},
        {'Question': 'New common-control Greek validated', 'Current evidence': str(manifest.get('new_control_stock_delta_validated', 'unreported'))},
        {'Question': 'Readiness flags', 'Current evidence':
            f"American price: {manifest.get('american_pricing_ready', 'unreported')}; stock delta: {manifest.get('stock_delta_ready', 'unreported')}"}])
    return pd.DataFrame(rows)
