"""Budget displays retain sampling failure and never relabel an unchanged gap."""
from copy import deepcopy
import numpy as np
import pandas as pd
from rvlab.experiments._pricing_rough_price_budget_reporting import (
    add_price_budget_reporting,price_budget_summary,primary_price_changes,latest_price_budget_state,
    spent_control_selection_summary)


def _result():
    rows=[]
    for i,(name,se) in enumerate((('unchanged control',.00027),('shared control',.00015),('unselected comparator',.00008))):
        lower=2.86+i*.00002;gap=.001;gap_se=.00005;allowance=.001286
        budget=gap+1.96*(se+gap_se)
        rows.append({'estimator':name,'role':'independent final valuation','paths':8192,
            'primary':i==1,'baseline':i==0,'price':lower,'price_se':se,'upper':lower+gap,
            'upper_se':se*.9,'gap':gap,'gap_se':gap_se,'lower_sampling_term':1.96*se,
            'gap_sampling_term':1.96*gap_se,'price_error_budget':budget,'price_allowance':allowance,
            'price_budget_pass':budget<=allowance,'gap_changed_by_common_control':False,
            'american_pricing_ready':False,'stock_delta_ready':False})
    changes=[]
    for i,row in enumerate(rows):
        for metric in ('lower change','upper change','gap change'):
            value=0. if metric=='gap change' else i*.00002;width=0. if metric=='gap change' else .0001
            changes.append({'estimator':row['estimator'],'primary':row['primary'],'metric':metric,
                'estimate':value,'standard_error':width/1.96,'interval_low':value-width,
                'interval_high':value+width,'half_width':width,'paths':8192})
    return {'tables':{'price_comparison':pd.DataFrame(rows),'paired_changes':pd.DataFrame(changes)},
        'manifest':{'american_pricing_ready':False,'stock_delta_ready':False,
            'new_control_stock_delta_validated':False},'figures':{}}


def test_two_figures_read_recorded_budget_components_without_changing_rows_or_flags():
    import matplotlib.pyplot as plt
    result=_result();before=deepcopy(result);r=add_price_budget_reporting(result)
    assert set(r['figures'])=={'price_budget_components','common_control_endpoints'}
    axis=r['figures']['price_budget_components'].axes[0]
    assert len(axis.patches)==9
    widths=np.array([bar.get_width() for bar in axis.patches]).reshape(3,3).sum(axis=0)
    table=result['tables']['price_comparison']
    np.testing.assert_allclose(widths,table.price_error_budget/table.price_allowance)
    labels=[tick.get_text() for tick in axis.get_yticklabels()]
    assert 'Predeclared primary' in labels[1] and 'unselected' in labels[2]
    assert table.iloc[2].price_budget_pass and not table.iloc[1].price_budget_pass
    assert result['manifest']==before['manifest'] and not result['figures']
    for name,t in result['tables'].items():pd.testing.assert_frame_equal(t,before['tables'][name])
    for fig in r['figures'].values():plt.close(fig)


def test_compact_summaries_keep_every_estimator_but_only_the_sealed_primary_changes():
    result=_result();summary=price_budget_summary(result['tables'])
    pd.testing.assert_frame_equal(summary,result['tables']['price_comparison'])
    changes=primary_price_changes(result['tables'])
    assert len(changes)==3 and set(changes.estimator)=={'shared control'}
    assert list(changes.metric)==['lower change','upper change','gap change']
    gap=changes[changes.metric.eq('gap change')].iloc[0]
    assert gap.estimate==gap.interval_low==gap.interval_high==0.


def test_endpoint_plot_preserves_exact_zero_gap_change_and_uses_explicit_dollar_axes():
    import matplotlib.pyplot as plt
    result=add_price_budget_reporting(_result());fig=result['figures']['common_control_endpoints']
    assert [tick.get_text() for tick in fig.axes[1].get_yticklabels()]==['lower change','upper change','gap change']
    line=fig.axes[1].containers[0].lines[0]
    assert line.get_xdata()[-1]==0.
    assert all(axis.xaxis.get_major_formatter()(0.,0)=='0.00000' for axis in fig.axes)
    for fig in result['figures'].values():plt.close(fig)


def test_missing_or_failed_evidence_is_not_replaced_with_a_price_or_a_budget():
    result={'tables':{'failures':pd.DataFrame([{'error':'fixed-budget failure'}])},'manifest':{},'figures':{}}
    assert not add_price_budget_reporting(result)['figures']
    assert price_budget_summary(result['tables']).empty and primary_price_changes(result['tables']).empty
    assert latest_price_budget_state({'tables':{},'manifest':{}}).empty


def test_latest_panel_keeps_unchanged_gap_failed_primary_and_false_readiness():
    result=_result();wrapped={'manifest':{'rough_price_budget':result['manifest']},
        'tables':{'rough_price_budget_'+name:table for name,table in result['tables'].items()}}
    state=latest_price_budget_state(wrapped).set_index('Question')
    assert state.loc['Predeclared primary','Current evidence']=='shared control; N=8,192'
    assert state.loc['Pathwise gap changed','Current evidence']=='False'
    assert 'recorded pass: False' in state.loc['Price budget / allowance','Current evidence']
    assert state.loc['Policy / dual slack separately identified','Current evidence']=='False'
    assert state.loc['New common-control Greek validated','Current evidence']=='False'
    assert state.loc['Readiness flags','Current evidence']=='American price: False; stock delta: False'


def _development():
    result=_result();rows=[]
    for family,base in (('gated',.019),('gated_anchor',.015)):
        for j,ridge in enumerate((1e-4,.001,.01,.1,1.)):
            for fold in (0,1):
                rows.append({'family':family,'ridge':ridge,'eligible_for_selection':True,
                    'diagnostic_sd':base+j*.00005+fold*.0002,'used_for_fresh_acceptance':False})
    rows += [{'family':'terminal','ridge':1e-4,'eligible_for_selection':False,
        'diagnostic_sd':.023,'used_for_fresh_acceptance':False} for _ in range(30)]
    result['tables']['spent_trials']=pd.DataFrame(rows)
    result['tables']['naive_european_trials']=pd.DataFrame([
        {'family':family,'diagnostic_seed':seed,'baseline_diagnostic_sd':.0245,
            'diagnostic_sd':value,'Z_mean_known':False,'used_for_fresh_acceptance':False}
        for family,value in (('paired European X',.0222),('conditional European Z',.0001))
        for seed in (1094005042,1095005042)])
    result['manifest']['campaign']={'selected_specification':{'family':'gated_anchor','ridge':.01,
        'eligible_for_selection':True}}
    return result


def test_development_plot_keeps_frozen_choice_even_when_another_displayed_rms_is_lower():
    import matplotlib.pyplot as plt
    result=_development();before=deepcopy(result);rendered=add_price_budget_reporting(result)
    assert set(rendered['figures'])=={'price_budget_components','common_control_endpoints','spent_control_trials'}
    summary=spent_control_selection_summary(result['tables'])
    assert len(summary)==10 and set(summary.spent_fold_count)=={2}
    assert len(result['tables']['spent_trials'])==50
    assert summary.loc[summary.family.eq('gated_anchor')].sort_values('diagnostic_sd_rms').iloc[0].ridge==1e-4
    axis=rendered['figures']['spent_control_trials'].axes[0]
    chosen=next(line for line in axis.lines if line.get_marker()=='*')
    assert list(chosen.get_xdata())==[.01]
    assert 'descriptive, not confidence intervals' in axis.get_title()
    assert result['manifest']==before['manifest']
    for name,table in result['tables'].items():pd.testing.assert_frame_equal(table,before['tables'][name])
    for figure in rendered['figures'].values():plt.close(figure)


def test_spent_reference_lines_include_exact_X_but_never_unknown_mean_Z_projection():
    import matplotlib.pyplot as plt
    rendered=add_price_budget_reporting(_development());axis=rendered['figures']['spent_control_trials'].axes[0]
    labels=axis.get_legend_handles_labels()[1]
    assert 'Existing price-control baseline' in labels
    assert 'Paired European X only · diagnostic' in labels
    assert not any('conditional Z' in label for label in labels)
    for figure in rendered['figures'].values():plt.close(figure)


def test_incomplete_fold_evidence_cannot_fabricate_a_development_selection_plot():
    import matplotlib.pyplot as plt
    result=_development();result['tables']['spent_trials']=result['tables']['spent_trials'].iloc[1:]
    assert spent_control_selection_summary(result['tables']).empty
    rendered=add_price_budget_reporting(result)
    assert 'spent_control_trials' not in rendered['figures']
    for figure in rendered['figures'].values():plt.close(figure)
