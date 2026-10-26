"""Read-only displays separating a stopping bracket from Monte Carlo uncertainty."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd

COMPONENTS=(
    ('Mean stopping gap','gap','#577590'),
    ('Lower-price sampling uncertainty','lower_sampling_term','#e9b44c'),
    ('Gap sampling uncertainty','gap_sampling_term','#79a977'))


def price_budget_summary(tables):
    data=tables.get('price_comparison',pd.DataFrame())
    required={'estimator','primary','baseline','paths','price','price_se','upper','upper_se',
        'gap','gap_se','lower_sampling_term','gap_sampling_term','price_error_budget',
        'price_allowance','price_budget_pass'}
    return data.copy() if required.issubset(data.columns) else pd.DataFrame()


def primary_price_changes(tables):
    data=tables.get('paired_changes',pd.DataFrame())
    required={'estimator','primary','metric','estimate','interval_low','interval_high'}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    return data.loc[data.primary.eq(True)&data.metric.isin(
        ['lower change','upper change','gap change'])].copy()


def spent_control_selection_summary(tables):
    """Describe both spent folds without selecting a specification from them."""
    data=tables.get('spent_trials',pd.DataFrame())
    required={'family','ridge','eligible_for_selection','diagnostic_sd'}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    eligible=data.loc[data.eligible_for_selection.eq(True)&data.family.isin(['gated','gated_anchor'])]
    rows=[]
    for (family,ridge),folds in eligible.groupby(['family','ridge'],sort=True):
        sd=folds.diagnostic_sd.to_numpy(dtype=float)
        if len(sd)!=2 or not np.isfinite(sd).all() or np.any(sd<0):
            return pd.DataFrame()
        rows.append({'family':family,'ridge':ridge,'diagnostic_sd_rms':float(np.sqrt(np.mean(sd*sd))),
            'fold_sd_min':float(sd.min()),'fold_sd_max':float(sd.max()),'spent_fold_count':2})
    return pd.DataFrame(rows)


def _labels(data):
    return ['\n'.join(textwrap.wrap(str(row.estimator),width=26))+
        ('\nPredeclared primary' if row.primary else '\nBaseline' if row.baseline else '')
        for _,row in data.iterrows()]


def _budget_figure(tables):
    data=price_budget_summary(tables)
    if data.empty:
        return {}
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(11.5,max(4.5,.9*len(data))))
    y=np.arange(len(data));left=np.zeros(len(data))
    for label,column,color in COMPONENTS:
        values=(data[column]/data.price_allowance).to_numpy()
        axis.barh(y,values,left=left,height=.58,color=color,label=label)
        left=left+values
    axis.axvline(1.,color='#b1533c',ls='--',label='Original allowance')
    axis.set(yticks=y,yticklabels=_labels(data),
        xlabel='Recorded additive price budget / unchanged price allowance',
        title='A common endpoint control leaves the stopping gap unchanged')
    axis.invert_yaxis();axis.grid(axis='x',alpha=.18)
    for tick,primary in zip(axis.get_yticklabels(),data.primary):
        if primary:tick.set_fontweight('bold')
    axis.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
    fig.tight_layout()
    return {'price_budget_components':fig}


def _endpoint_figure(tables):
    data=price_budget_summary(tables);changes=primary_price_changes(tables)
    if data.empty or changes.empty:
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    fig,axes=plt.subplots(1,2,figsize=(12.5,max(4.7,.9*len(data))))
    y=np.arange(len(data))
    for column,se,offset,label,color in (
            ('price','price_se',-.13,'Lower estimate','#2878a5'),
            ('upper','upper_se',.13,'Upper estimate','#8a6aa6')):
        axes[0].errorbar(data[column],y+offset,xerr=1.96*data[se],fmt='o',capsize=4,
            label=label,color=color)
    axes[0].set(yticks=y,yticklabels=_labels(data),xlabel='Price ($) · marginal 95% intervals',
        title='The exercise rule and price target remain fixed')
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
    values=changes.estimate.to_numpy()
    errors=np.vstack((values-changes.interval_low,changes.interval_high-values))
    axes[1].errorbar(values,np.arange(len(changes)),xerr=errors,fmt='o',capsize=4,color='#2878a5')
    axes[1].axvline(0.,color='#777777',lw=1)
    axes[1].set(yticks=np.arange(len(changes)),yticklabels=changes.metric.tolist(),
        xlabel='Primary minus baseline ($) · paired 95% intervals',
        title='The identical pathwise gap is an implementation identity')
    axes[1].invert_yaxis()
    for axis in axes:
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:.5f}'))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4));axis.grid(alpha=.18)
    fig.suptitle('These controls change sampling variance, not the stopping policy or dual slack')
    fig.tight_layout()
    return {'common_control_endpoints':fig}


def _spent_trials_figure(tables,manifest):
    data=spent_control_selection_summary(tables)
    selected=manifest.get('campaign',{}).get('selected_specification',{})
    if data.empty or not {'family','ridge'}.issubset(selected):
        return {}
    chosen=data.loc[data.family.eq(selected['family'])&data.ridge.eq(selected['ridge'])]
    if len(chosen)!=1:
        return {}
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(11.5,5.4))
    for family,label,color in (('gated','192 gated features','#8a6aa6'),
            ('gated_anchor','193 gated + paired-European features','#2878a5')):
        group=data.loc[data.family.eq(family)].sort_values('ridge')
        if group.empty:
            continue
        value=group.diagnostic_sd_rms.to_numpy()
        error=np.vstack((value-group.fold_sd_min,group.fold_sd_max-value))
        axis.errorbar(group.ridge,value,yerr=error,fmt='o-',capsize=4,label=label,color=color)
    naive=tables.get('naive_european_trials',pd.DataFrame())
    if {'family','diagnostic_seed','baseline_diagnostic_sd','diagnostic_sd'}.issubset(naive.columns):
        baseline=naive.groupby('diagnostic_seed').baseline_diagnostic_sd.first().to_numpy(dtype=float)
        paired=naive.loc[naive.family.eq('paired European X'),'diagnostic_sd'].to_numpy(dtype=float)
        for values,label,color,style in ((baseline,'Existing price-control baseline','#555555','--'),
                (paired,'Paired European X only · diagnostic','#999999',':')):
            if len(values)==2 and np.isfinite(values).all() and np.all(values>=0):
                axis.axhline(np.sqrt(np.mean(values*values)),color=color,ls=style,label=label)
    row=chosen.iloc[0]
    axis.plot(row.ridge,row.diagnostic_sd_rms,'*',markersize=17,color='#b1533c',
        label='Specification frozen before fresh paths',zorder=5)
    axis.set(xscale='log',xlabel='Ridge penalty · log scale',
        ylabel='Held-out lower-price standard deviation ($)\nRMS of the two spent split directions',
        title='Development evidence only: fold ranges are descriptive, not confidence intervals')
    axis.set_xticks([1e-4,1e-3,.01,.1,1.],labels=['.0001','.001','.01','.1','1'])
    axis.grid(alpha=.18)
    axis.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2)
    fig.tight_layout()
    return {'spent_control_trials':fig}


def add_price_budget_reporting(result):
    tables=result.get('tables',{});figures=dict(result.get('figures',{}))
    figures.update(_budget_figure(tables));figures.update(_endpoint_figure(tables))
    figures.update(_spent_trials_figure(tables,result.get('manifest',{})))
    return {**result,'figures':figures}


def latest_price_budget_state(result):
    manifest=result.get('manifest',{}).get('rough_price_budget')
    if not manifest:
        return pd.DataFrame()
    prefix='rough_price_budget_'
    tables={name.removeprefix(prefix):table for name,table in result.get('tables',{}).items()
        if name.startswith(prefix)}
    data=price_budget_summary(tables)
    primary=data.loc[data.primary.eq(True)] if not data.empty else pd.DataFrame()
    rows=[{'Question':'Current experiment','Current evidence':'Frozen GH16 rule; the same zero-mean control shifts both price endpoints'}]
    if len(primary)==1:
        row=primary.iloc[0]
        rows.extend([
            {'Question':'Predeclared primary','Current evidence':f'{row.estimator}; N={int(row.paths):,}'},
            {'Question':'Price budget / allowance','Current evidence':f'{row.price_error_budget:.6f} / {row.price_allowance:.6f}; recorded pass: {bool(row.price_budget_pass)}'},
            {'Question':'Separate budget components','Current evidence':f'Gap {row.gap:.6f}; lower uncertainty {row.lower_sampling_term:.6f}; gap uncertainty {row.gap_sampling_term:.6f}'},
            {'Question':'Pathwise gap changed','Current evidence':str(row.get('gap_changed_by_common_control','unreported'))}])
    else:
        rows.append({'Question':'Predeclared primary','Current evidence':'No unique completed final record'})
    rows.extend([
        {'Question':'Policy / dual slack separately identified','Current evidence':'False'},
        {'Question':'New common-control Greek validated','Current evidence':str(manifest.get('new_control_stock_delta_validated','unreported'))},
        {'Question':'Readiness flags','Current evidence':f"American price: {manifest.get('american_pricing_ready','unreported')}; stock delta: {manifest.get('stock_delta_ready','unreported')}"}])
    return pd.DataFrame(rows)
