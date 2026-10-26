"""Independent ridge, spent-selection and price-budget counterexamples."""
import numpy as np
import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_price_budget_collection import _fit,select_spent_specification
from rvlab.experiments._pricing_rough_price_budget_tables import (
    price_budget,diagnose_spent_budget,summarize_price_streams,statistics)


def test_ridge_coefficients_match_augmented_least_squares_then_raw_projection():
    a=np.tile([-1.,1.],4);b=np.repeat([-1.,1.],4)
    x=np.column_stack((a,b,np.full(8,7.)));y=2.86+6*a+.5*b;ridge=.01
    beta,center,scale,raw=_fit(x,y,ridge)
    wanted_center=x.mean(axis=0);wanted_scale=x.std(axis=0);wanted_scale[wanted_scale<1e-12]=1.
    standardized=(x-wanted_center)/wanted_scale
    design=np.vstack((standardized/len(x)**.5,np.sqrt(ridge)*np.eye(x.shape[1])))
    target=np.r_[(y-y.mean())/len(y)**.5,np.zeros(x.shape[1])]
    independent=np.linalg.lstsq(design,target,rcond=None)[0]/wanted_scale
    np.testing.assert_allclose(raw,independent,atol=3e-14,rtol=0.)
    np.testing.assert_array_equal(center,wanted_center);np.testing.assert_array_equal(scale,wanted_scale)
    np.testing.assert_allclose(beta,np.clip(independent,-2.,2.),atol=3e-14,rtol=0.)
    assert raw[0]>2 and beta[0]==2 and beta[2]==0


def selection_rows():
    rows=[]
    for k,eligible,diagnostic,training in ((0,False,(.001,.001),(.001,.001)),
            (1,True,(.01,.02),(.5,.5)),(2,True,(.03,.03),(0.,0.))):
        for direction,(seed,other) in enumerate(((1094005042,1095005042),(1095005042,1094005042))):
            rows.append({'spec_index':k,'eligible_for_selection':eligible,'fit_seed':seed,
                'diagnostic_seed':other,'diagnostic_sd':diagnostic[direction],'training_sd':training[direction]})
    return pd.DataFrame(rows)


def test_selection_uses_both_heldout_variances_not_training_fit_or_ineligible_trials():
    rows=selection_rows();selected,scores=select_spent_specification(rows)
    assert selected==1 and scores[1]==pytest.approx((.01**2+.02**2)/2)
    assert 0 not in scores and rows.loc[rows.spec_index==2,'training_sd'].eq(0).all()
    # Changing training fit diagnostics cannot change a held-out selection.
    rows.training_sd=np.arange(len(rows))*100.
    assert select_spent_specification(rows)[0]==1


@pytest.mark.parametrize('kind',['missingfold','duplicatedfold','wrongseed','negative','nonfinite','emptyeligible'])
def test_heldout_selection_rejects_omitted_or_tampered_evidence(kind):
    rows=selection_rows()
    if kind=='missingfold':rows=rows.drop(rows.index[2])
    elif kind=='duplicatedfold':rows.loc[rows.index[3],['fit_seed','diagnostic_seed']]=[1094005042,1095005042]
    elif kind=='wrongseed':rows.loc[rows.index[2],'fit_seed']=1106005042
    elif kind=='negative':rows.loc[rows.index[2],'diagnostic_sd']=-.1
    elif kind=='nonfinite':rows.loc[rows.index[2],'diagnostic_sd']=np.nan
    else:rows.eligible_for_selection=False
    with pytest.raises(ValueError):select_spent_specification(rows)


def test_fixed_seed_statistics_use_whole_path_strata_with_actual_sizes():
    chunks=[np.array([1.,2.,3.]),np.array([2.,4.,6.,8.,10.])]
    actual=statistics(chunks);n=sum(map(len,chunks))
    expected_mean=sum(a.sum() for a in chunks)/n
    # Independent computation from each seed's centered squared deviations.
    variance=sum(len(a)*np.sum((a-a.mean())**2)/(len(a)-1) for a in chunks)/n**2
    assert actual['estimate']==pytest.approx(expected_mean)
    assert actual['standard_error']==pytest.approx(np.sqrt(variance))


def test_budget_decomposition_exposes_sampling_failure_without_identifying_policy_error():
    lower=np.array([2.80,2.84,2.88,2.92]);upper=lower+.001
    row=price_budget([lower],[upper],label='unchanged',role='synthetic algebra fixture')
    assert row['gap_below_allowance'] and not row['price_budget_pass']
    assert row['price_error_budget']==pytest.approx(row['gap']+row['lower_sampling_term']+row['gap_sampling_term'])
    diagnosis=diagnose_spent_budget(row)
    assert diagnosis['sampling_causes_recorded_budget_failure']
    assert not diagnosis['optimal_policy_error_identified']
    assert diagnosis['projected_paths_with_unchanged_variance']>len(lower)


def test_common_translation_reduces_price_noise_but_cannot_reduce_any_gap():
    streams=[]
    for seed in (1,2):
        noise=np.array([-.03,-.01,.01,.03]);gap=np.array([.0005,.001,.001,.0015])
        lower=2.86+noise;upper=lower+gap;shift=.8*noise
        streams.append({'seed':seed,'values':{
            'unchanged control':{'lower':lower,'upper':upper},
            'shared control':{'lower':lower-shift,'upper':upper-shift}}})
    tables=summarize_price_streams(streams)
    rows=tables['price_comparison'].set_index('estimator')
    assert rows.loc['shared control','price_variance_ratio_vs_baseline']==pytest.approx(.04)
    assert rows.loc['shared control','gap']==pytest.approx(rows.loc['unchanged control','gap'])
    assert rows.loc['shared control','gap_se']==pytest.approx(rows.loc['unchanged control','gap_se'])
    assert not rows.gap_changed_by_common_control.any()
    changed=tables['paired_changes'].query("estimator == 'shared control' and metric == 'gap change'").iloc[0]
    assert abs(changed.estimate)<1e-15 and changed.half_width<1e-15
    streams[0]['values']['shared control']['upper'][0]+=.001
    with pytest.raises(ValueError,match='pathwise stopping gap'):summarize_price_streams(streams)
