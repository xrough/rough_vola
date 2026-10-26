"""Fixed-stratum paired comparisons for independent Bellman control trials."""
import numpy as np
import pandas as pd
from ._pricing_rough_policy_validation_stopping import _summary
from ._pricing_rough_bellman_control_samples import BASELINE


def row(chunks):
    estimate,se,low,high=_summary(chunks)
    return {'estimate':estimate,'standard_error':se,'interval_low':low,'interval_high':high,'half_width':1.96*se}


def summarize_control_streams(streams,selection,targets):
    if not streams or len({s['seed'] for s in streams})!=len(streams):
        raise ValueError('Independent final control streams required')
    controls=list(selection['controls']); n=sum(len(s['raw']) for s in streams)
    estimates=[]; paired=[]; seedwise=[]; tails=[]; majority=[]; target_rows=[]; centering=[]
    for name in controls:
        lowers=[s['values'][name]['lower'] for s in streams]; uppers=[s['values'][name]['upper'] for s in streams]
        gaps=[u-l for u,l in zip(uppers,lowers)]; p=row(lowers); g=row(gaps); u=row(uppers)
        allowance=.001+1e-4*abs(p['estimate']); budget=g['estimate']+1.96*(p['standard_error']+g['standard_error'])
        flags={'control':name,'selected_control':name==selection['selected_control'],
            'designated_challenger':name==selection['designated_challenger'],'paths':n}
        estimates.append({**flags,'price':p['estimate'],'price_se':p['standard_error'],'upper':u['estimate'],
            'upper_se':u['standard_error'],'gap':g['estimate'],'gap_se':g['standard_error'],
            'price_error_budget':budget,'price_allowance':allowance,'price_budget_pass':budget<=allowance,
            'optimal_value_ready':False,'new_control_delta_validated':False})
        lower_gain=[]; upper_gain=[]; total_gain=[]; half_gap=[]
        for s,lower,upper,gap in zip(streams,lowers,uppers,gaps):
            base=s['values'][BASELINE]; basegap=base['upper']-base['lower']
            dl=lower-base['lower']; du=base['upper']-upper; gain=basegap-gap
            lower_gain.append(dl); upper_gain.append(du); total_gain.append(gain); half_gap.append(du-.5*basegap)
            seedwise.append({**flags,'seed':s['seed'],'paths':len(lower),'price':float(lower.mean()),
                'gap':float(gap.mean()),'gap_gain':float(gain.mean()),'gap_gain_se':float(gain.std(ddof=1)/np.sqrt(len(gain))),
                'upper_decrease':float(du.mean()),'lower_change':float(dl.mean())})
            tails.append({**flags,'seed':s['seed'],'paths':len(gap),'gap_mean':float(gap.mean()),
                'gap_p99':float(np.quantile(gap,.99)),'gap_max':float(gap.max()),
                'path_gap_worse_fraction':float(np.mean(gap>basegap+2e-13)),
                'path_gap_better_fraction':float(np.mean(gap<basegap-2e-13)),
                'max_path_gap_increase':float(np.max(gap-basegap))})
        for metric,chunks in [('gap_gain',total_gain),('upper_decrease',upper_gain),('lower_change',lower_gain),
                ('raw_minus_controlled',[s['raw']-lower for s,lower in zip(streams,lowers)])]:
            stats=row(chunks)
            paired.append({**flags,'metric':metric,**stats,'gap_gain':stats['estimate'] if metric=='gap_gain' else np.nan,
                'gap_gain_se':stats['standard_error'] if metric=='gap_gain' else np.nan,
                'gap_gain_low':stats['interval_low'] if metric=='gap_gain' else np.nan,
                'gap_gain_high':stats['interval_high'] if metric=='gap_gain' else np.nan})
        maj=row(half_gap)
        majority.append({**flags,**maj,'criterion':'upper decrease minus half baseline gap',
            'dual_majority_demonstrated':maj['interval_low']>0,
            'unchanged_exercise_rule':True,'optimal_policy_error_identified':False})
    for index,label in enumerate(targets):
        for s in streams:
            data=s['pure_targets'][index]; base=s['values'][BASELINE]
            target_rows.append({'target':label,'seed':s['seed'],'paths':len(s['raw']),
                'gap':float(np.mean(data['upper']-data['lower'])),
                'gap_gain':float(np.mean(base['upper']-base['lower']-data['upper']+data['lower'])),
                'used_for_control_selection':False,'naive_upper_endpoint_trial':index==0})
            for j in (16,32,48,64):
                v=s['targets'][:,j,index]
                centering.append({'target':label,'seed':s['seed'],'exercise_index':j,'mean':float(v.mean()),
                    'standard_error':float(v.std(ddof=1)/np.sqrt(len(v))),
                    'empirical_mean_subtracted':False,'centering_certified_by_empirical_mean':False})
    return {'gap_comparison':pd.DataFrame(estimates),'gap_paired_changes':pd.DataFrame(paired),
        'gap_per_seed':pd.DataFrame(seedwise),'gap_tail_distribution':pd.DataFrame(tails),
        'gap_dual_majority':pd.DataFrame(majority),'gap_pure_targets':pd.DataFrame(target_rows),
        'gap_empirical_centering':pd.DataFrame(centering)}
