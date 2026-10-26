"""Read-only presentation summaries for the first-stage numerical report.

Selections come from the saved training-only selection records. These helpers
perform no pricing, model fitting, new acceptance selection or data fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_CASES=tuple(f'put_{days}d_k{strike}' for days in (30,90) for strike in (90,100,110))
CONTROL_LABELS={'raw':'Raw / M=0','european':'European control','european_premium':'European + premium · primary'}
CONTROL_COLORS={'raw':'#92999c','european':'#268577','european_premium':'#2878a5'}


def generic_case_summary(tables):
    """Keep the six declared contracts, including unavailable/failed cases."""
    prices=tables.get('rough_benchmark_price_bounds',pd.DataFrame())
    status=tables.get('rough_benchmark_case_status',pd.DataFrame())
    if prices.empty and status.empty:
        return pd.DataFrame()
    if not {'trial_id','complete','declared_paths'}.issubset(status.columns):
        raise ValueError('Generic report needs explicit case-completion records')
    status=status.loc[status.trial_id.isin(BASE_CASES)].copy()
    if len(status)!=6 or set(status.trial_id)!=set(BASE_CASES):
        raise ValueError('Generic report must retain every declared base case')
    incomplete=set(status.loc[~status.complete,'trial_id'])
    if not prices.empty and prices.trial_id.isin(incomplete).any():
        raise ValueError('An incomplete case cannot supply partial control samples')
    if prices.empty:
        selected=pd.DataFrame(columns=['trial_id','primary','price','price_se','upper',
            'upper_se','gap','gap_se','price_error_budget','price_allowance','price_budget_pass'])
    else:
        selected=prices.loc[prices.control.eq('european_premium')&prices.trial_id.isin(BASE_CASES)].copy()
    if selected.trial_id.duplicated().any() or not selected.primary.eq(True).all():
        raise ValueError('Generic report cannot reselect or duplicate a primary control')
    result=status.merge(selected,on='trial_id',how='left',validate='one_to_one',suffixes=('_status',''))
    if result.loc[result.complete,'price'].isna().any():
        raise ValueError('A complete case must supply its frozen primary estimate')
    if result.loc[~result.complete,'price'].notna().any():
        raise ValueError('An incomplete case cannot supply partial acceptance prices')
    sampling=tables.get('rough_benchmark_delta_sampling',pd.DataFrame())
    if not sampling.empty:
        small=sampling.loc[sampling.trial_id.isin(BASE_CASES)&
            sampling.control.eq('european_premium')&sampling.bump_relative.eq(.0025)].copy()
        if small.trial_id.duplicated().any() or not small.primary.eq(True).all():
            raise ValueError('Generic delta report requires one predeclared primary small bump per case')
        if small.trial_id.isin(incomplete).any():
            raise ValueError('An incomplete case cannot supply partial delta samples')
        small=small[['trial_id','estimate','half_width','delta_allowance','sampling_pass']].rename(
            columns={'estimate':'delta','half_width':'delta_half_width','sampling_pass':'delta_sampling_pass'})
        result=result.merge(small,on='trial_id',how='left',validate='one_to_one')
        if result.loc[result.complete,'delta'].isna().any():
            raise ValueError('A complete case must supply its primary small-bump delta')
        bumps=tables.get('rough_benchmark_bump_refinement',pd.DataFrame())
        if not bumps.empty:
            chosen=bumps.loc[bumps.trial_id.isin(BASE_CASES)&bumps.control.eq('european_premium')]
            if chosen.trial_id.duplicated().any() or not chosen.primary.eq(True).all():
                raise ValueError('Generic delta report cannot reselect a bump comparison')
            if chosen.trial_id.isin(incomplete).any():
                raise ValueError('An incomplete case cannot supply partial bump comparisons')
            chosen=chosen[['trial_id','estimate','change_plus_uncertainty','tolerance_resolved']].rename(
                columns={'estimate':'bump_change','change_plus_uncertainty':'bump_budget',
                    'tolerance_resolved':'bump_pass'})
            result=result.merge(chosen,on='trial_id',how='left',validate='one_to_one')
            if result.loc[result.complete,'bump_pass'].isna().any():
                raise ValueError('A complete case must retain its primary bump comparison')
    result['days']=result.trial_id.map({name:int(name.split('_')[1][:-1]) for name in BASE_CASES})
    result['declared_strike']=result.trial_id.map({name:int(name.split('_')[2][1:]) for name in BASE_CASES})
    return result.set_index('trial_id').loc[list(BASE_CASES)].reset_index()


def _generic_figures(tables):
    summary=generic_case_summary(tables)
    if summary.empty:
        return {}
    import matplotlib.pyplot as plt
    labels=[f'{row.days} days · K={row.declared_strike}' for _,row in summary.iterrows()]
    y=np.arange(6);figures={}
    figure,axes=plt.subplots(1,2,figsize=(12.5,5.3))
    for i,(_,row) in enumerate(summary.iterrows()):
        if not row.complete:
            axes[0].text(.03,i,'Incomplete · no price',transform=axes[0].get_yaxis_transform(),va='center')
            axes[1].text(.03,i,'Incomplete',transform=axes[1].get_yaxis_transform(),va='center')
            continue
        axes[0].plot([row.price,row.upper],[i,i],color='#92999c',lw=3)
        for name,offset,color in (('price',-.08,'#2878a5'),('upper',.08,'#d17b32')):
            se=row.price_se if name=='price' else row.upper_se
            axes[0].errorbar(row[name],i+offset,xerr=1.96*se,fmt='o',color=color,
                capsize=3,label=('Lower policy' if name=='price' else 'Dual upper') if i==0 else None)
    left=np.zeros(6)
    for values,label,color in ((summary.gap,'Mean stopping gap','#92999c'),
            (1.96*summary.price_se,'Lower sampling term','#2878a5'),
            (1.96*summary.gap_se,'Gap sampling term','#e9b44c')):
        ratio=(values/summary.price_allowance).to_numpy(float)
        axes[1].barh(y,ratio,left=left,label=label,color=color,height=.6)
        left=left+np.nan_to_num(ratio,nan=0.)
    axes[1].axvline(1.,color='#b1533c',ls='--',label='Unchanged allowance')
    axes[0].set(title='Six matched contracts · primary EP control',xlabel='Endpoint mean ($) · marginal 95% intervals',yticks=y,yticklabels=labels)
    axes[1].set(title='Separate stopping slack from sampling uncertainty',xlabel='Price-budget components / unchanged allowance',yticks=y,yticklabels=[])
    for axis in axes:
        axis.invert_yaxis();axis.grid(alpha=.18);axis.legend(loc='upper center',bbox_to_anchor=(.5,-.15),frameon=False,ncol=2)
    figure.tight_layout();figures['rough_benchmark_generic_price_coverage']=figure
    prices=tables['rough_benchmark_price_bounds']
    if prices.empty:
        return figures
    data=prices.loc[prices.trial_id.isin(BASE_CASES)].copy()
    figure,axes=plt.subplots(1,2,figsize=(12.5,5.3))
    for control,offset in (('raw',-.18),('european',0.),('european_premium',.18)):
        chosen=data.loc[data.control.eq(control)].set_index('trial_id').reindex(BASE_CASES)
        raw=data.loc[data.control.eq('raw')].set_index('trial_id').reindex(BASE_CASES)
        ratio=np.divide(chosen.price_se,raw.price_se,out=np.full(6,np.nan),where=raw.price_se.to_numpy()>0)
        for axis,values in ((axes[0],ratio),(axes[1],chosen.price_error_budget/chosen.price_allowance)):
            axis.plot(values,y+offset,'o',color=CONTROL_COLORS[control],label=CONTROL_LABELS[control])
    for i in np.flatnonzero(raw.price_se.eq(0).to_numpy()):
        axes[0].text(.03,i,'N/A · zero raw SE',transform=axes[0].get_yaxis_transform(),va='center')
    axes[0].set(title='Same policy and paths · lower-estimate precision',xlabel='Price standard error / raw standard error',yticks=y,yticklabels=labels)
    axes[1].set(title='All control alternatives retain their price gates',xlabel='Price budget / unchanged allowance',yticks=y,yticklabels=[])
    if (data.price_error_budget/data.price_allowance).gt(0).all():axes[1].set_xscale('log')
    axes[1].axvline(1.,color='#b1533c',ls='--')
    for axis in axes:
        axis.invert_yaxis();axis.grid(alpha=.18)
    axes[0].legend(loc='upper center',bbox_to_anchor=(1.,-.15),frameon=False,ncol=3)
    figure.tight_layout();figures['rough_benchmark_generic_control_comparison']=figure
    refinement=tables.get('rough_benchmark_refinement',pd.DataFrame())
    if not refinement.empty and 'control' in refinement:
        selected=refinement.loc[refinement.control.eq('european_premium')].copy()
        if not selected.empty:
            figure,axis=plt.subplots(figsize=(11.5,max(4.7,.49*len(selected))))
            labels=[]
            for i,(_,row) in enumerate(selected.iterrows()):
                variant=str(row.fine_trial).removeprefix('put_90d_k100_')
                scope='fixed full policy' if row.same_full_policy else 'retrained construction'
                quantity='Price' if row.quantity=='price' else f'Delta · h={100*row.bump_relative:g}%'
                labels.append(f'{variant} · {scope}\n{quantity}')
                axis.barh(i,row.change_plus_uncertainty/row.allowance,
                    color='#2878a5' if row.same_full_policy else '#9a6599',height=.62)
            axis.axvline(1.,color='#b1533c',ls='--',label='Unchanged allowance')
            axis.set(yticks=np.arange(len(selected)),yticklabels=labels,
                xlabel='Comparison budget / allowance · linear 0–1, logarithmic above 1',
                title='ATM 90-day representative refinements · primary EP control')
            axis.set_xscale('symlog',linthresh=1.)
            maximum=float((selected.change_plus_uncertainty/selected.allowance).max())
            ticks=[v for v in (0,1,2,5,10,20,50,100,200,500,1000) if v<=max(1.,1.1*maximum)]
            axis.set_xticks(ticks,labels=[str(v) for v in ticks])
            axis.invert_yaxis();axis.grid(axis='x',alpha=.18);axis.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.08))
            figure.tight_layout();figures['rough_benchmark_generic_refinement_checks']=figure
    return figures


def method_matrix():
    """State the actual numerical target separately from a model's name."""
    return pd.DataFrame([
        {'component':'European controls','models':'BS / Heston / rough Heston',
         'method':'Analytic BS; shared Fourier quadrature with BS control; Riccati or fractional Riccati',
         'target':'European vanilla expectation; explicit forward and discount',
         'scope':'Synthetic references; exercise is disabled'},
        {'component':'American finite differences','models':'BS / Heston / Heston curve',
         'method':'Crank–Nicolson/policy iteration; modified Craig–Sneyd ADI/Ikonen–Toivanen splitting',
         'target':'Spot-based exercise problem with explicit discount and cash-dividend inputs',
         'scope':'Independent spatial, temporal and domain checks; theta(t) is a mean-reversion level'},
        {'component':'Rough dynamics','models':'Finite-factor rough Heston',
         'method':'Positive exponential kernel lift and audited weak splitting',
         'target':'Specified finite-factor, time-discrete inception-state law',
         'scope':'Kernel and time refinements remain distinct from Monte Carlo error'},
        {'component':'Exercise and bracket','models':'Finite-calendar rough Heston',
         'method':'Independently trained stopping policy and exactly centered discrete martingale',
         'target':'Lower policy value and upper expectation for the same discrete law/calendar',
         'scope':'A stopping gap combines removable policy error and dual slack'},
        {'component':'Variance reduction','models':'Generic EP control / focused shared control',
         'method':'Predictable controls; focused common endpoint shift has independently fixed coefficients',
         'target':'Same expected endpoints with changed sampling variance',
         'scope':'The focused 193-feature control is not a generic six-case fitted control'},
        {'component':'Stock sensitivity','models':'Frozen lower stopping policy',
         'method':'Paired central stock bumps with own continuation/stopping/control recomputation',
         'target':'Finite difference of the fixed policy expectation',
         'scope':'Sampling, bump, representation and time checks; no optimal-American Greek claim'},
    ])


def literature_sources():
    """Primary references and their limited role in the implementation report."""
    return pd.DataFrame([
        {'method':'Rough-Heston European transform','reference':'El Euch & Rosenbaum (2016)',
         'source':'https://arxiv.org/abs/1609.02108',
         'use':'Fractional Riccati formulation for the characteristic function'},
        {'method':'Markovian kernel lift','reference':'Abi Jaber & El Euch (2018/2019)',
         'source':'https://arxiv.org/abs/1801.10359',
         'use':'Finite-factor approximations of rough-volatility dynamics'},
        {'method':'Weak simulation','reference':'Bayer & Breneis (2023)',
         'source':'https://arxiv.org/abs/2310.04146',
         'use':'Low-dimensional weak schemes; our tested variant retains its own validation'},
        {'method':'American Heston finite differences','reference':'Haentjens & in ’t Hout (2013/2015)',
         'source':'https://arxiv.org/abs/1309.0110',
         'use':'ADI methods for the American complementarity problem'},
        {'method':'Regression exercise policy','reference':'Longstaff & Schwartz (2001)',
         'source':'https://doi.org/10.1093/rfs/14.1.113',
         'use':'Continuation regression as a stopping-policy construction'},
        {'method':'Martingale upper bound','reference':'Rogers (2002)',
         'source':'https://doi.org/10.1111/1467-9965.02010',
         'use':'Dual upper expectation from payoff minus a martingale'},
        {'method':'Primal–dual simulation','reference':'Andersen & Broadie (2004)',
         'source':'https://doi.org/10.1287/mnsc.1040.0258',
         'use':'Separate lower/upper estimates; no inherited accuracy certificate for our implementation'},
    ])


def selected_calibrations(tables):
    """Read the saved start/stage decision; held-out errors never select a row."""
    data=tables.get('american_joint_summary',pd.DataFrame())
    selection=tables.get('american_joint_selection',pd.DataFrame())
    if data.empty or selection.empty:
        return pd.DataFrame()
    required={'model','dataset','selected_start','continuation_executed','heldout_used_for_selection'}
    if not required.issubset(selection.columns) or not selection.heldout_used_for_selection.eq(False).all():
        raise ValueError('Calibration report requires explicit training-only selection')
    rows=[]
    for _,choice in selection.iterrows():
        stage='continuation' if choice.continuation_executed else 'exploratory'
        match=data.loc[data.model.eq(choice.model)&data.dataset.eq(choice.dataset)&
            data.start.eq(choice.selected_start)&data.stage.eq(stage)]
        if len(match)!=1:
            raise ValueError('Saved calibration selection has no unique matching attempt')
        rows.append(match.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True).copy(deep=True)


def matched_maturity_recovery(tables):
    """Compare the same held-out short contracts using training-selected fits."""
    data=tables.get('maturity_extension_common_horizon_recovery',pd.DataFrame())
    selection=tables.get('maturity_extension_selection',pd.DataFrame())
    if data.empty or selection.empty:
        return pd.DataFrame()
    if ('heldout_used_for_selection' not in selection or
            not selection.heldout_used_for_selection.eq(False).all()):
        raise ValueError('Maturity report requires explicit training-only selection')
    rows=[]
    for _,choice in selection.iterrows():
        match=data.loc[data.attempt_id.eq(choice.selected_attempt_id)&data.design.eq(choice.design)&
            data.split.eq('heldout')&data.horizon_cohort.eq('common_14_30_60')]
        if len(match)!=1:
            raise ValueError('Saved maturity selection has no unique matched held-out cohort')
        rows.append(match.iloc[0])
    result=pd.DataFrame(rows).reset_index(drop=True).copy(deep=True)
    if result.contract_count.nunique()!=1:
        raise ValueError('Maturity comparison changed the common held-out contract count')
    return result


def evidence_status(result):
    """Completion, specific empirical checks and broader readiness stay separate."""
    manifest=result.get('manifest',{});rows=[]
    benchmark=manifest.get('rough_benchmark')
    if benchmark is not None:
        rows.append({'evidence':'Six-case generic panel','origin':'New bounded benchmark',
            'recorded_state':'Complete' if benchmark.get('complete') is True else 'Incomplete',
            'interpretation':'Completion records all declared trials; it does not mean all price/delta gates pass'})
    focused=manifest.get('historical_focused_evidence',{})
    price=focused.get('rough_mc_precision')
    if price is not None:
        rows.append({'evidence':'Focused price precision','origin':'Verified preserved experiment',
            'recorded_state':f"Sampling gate: {price.get('sampling_gate_pass','unreported')}",
            'interpretation':'One 60-day ATM put; narrow sampling headroom does not certify general pricing accuracy'})
    delta=focused.get('rough_shared_delta')
    if delta is not None:
        rows.append({'evidence':'Focused shared-control stock delta','origin':'Verified preserved experiment',
            'recorded_state':f"Fixed-policy checks: {delta.get('fixed_policy_delta_checks_pass','unreported')}",
            'interpretation':'One frozen 64-date policy; raw/control bias equivalence remains a separate diagnostic'})
    if manifest.get('historical_replay') is True:
        rows.append({'evidence':'Synthetic calibration','origin':'Verified preserved experiment',
            'recorded_state':f"Optimizer calls in this report: {manifest.get('optimizer_calls_this_report','unreported')}",
            'interpretation':'Numerical outcomes retain original grids, starts, noise and source identities'})
    if 'current_source_repricing_pass' in manifest:
        rows.append({'evidence':'Current-source compatibility','origin':'New prices/deltas at preserved fitted vectors',
            'recorded_state':str(manifest['current_source_repricing_pass']),
            'interpretation':'A repricing check; optimization and the original independent refinements are not rerun'})
    if 'calibration_ready' in manifest:
        rows.append({'evidence':'Real SPY handoff','origin':'Current observed-input audit',
            'recorded_state':f"Calibration ready: {manifest['calibration_ready']}",
            'interpretation':'Independent stock and historically available cash-dividend coverage govern eligibility'})
    if 'stock_delta_ready' in manifest or 'hedging_ready' in manifest:
        ready=manifest.get('stock_delta_ready',manifest.get('hedging_ready'))
        rows.append({'evidence':'Broader market/hedging readiness','origin':'Declared scope',
            'recorded_state':str(ready),
            'interpretation':'Neither a passed local diagnostic nor a stable synthetic fit supplies a portfolio replay'})
    return pd.DataFrame(rows)


def add_first_stage_reporting(result):
    """Attach small report tables while preserving all numerical records."""
    tables=dict(result.get('tables',{}))
    tables.update(report_methods=method_matrix(),report_sources=literature_sources(),
        report_evidence_status=evidence_status(result))
    selected=selected_calibrations(tables)
    if not selected.empty:
        tables['report_selected_calibrations']=selected
    matched=matched_maturity_recovery(tables)
    if not matched.empty:
        tables['report_matched_maturity_recovery']=matched
    figures=dict(result.get('figures',{}));figures.update(_generic_figures(tables))
    generic=generic_case_summary(tables)
    if not generic.empty:
        tables['report_generic_cases']=generic
    return {**result,'tables':tables,'figures':figures}


__all__=['method_matrix','literature_sources','selected_calibrations','matched_maturity_recovery',
    'evidence_status','add_first_stage_reporting']
