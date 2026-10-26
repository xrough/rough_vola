"""Independent bound/gate arithmetic and complete-case protocol safeguards."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import rvlab.experiments._pricing_rough_benchmark_audit as audit


def statistics_case():
    streams=[];specs=[];dx=100.*audit.RELATIVE
    for trial,steps in [('base',128),('time',256)]:
        for index in range(2):
            n=8;spec=dict(trial_id=trial,seed=20+len(specs),training_seed=10,
                paths=n,steps=steps,exercise_steps=64,factors=4,strike=100.,maturity=90/365)
            z=np.tile([-1.,1.],4)
            controls={}
            for control,scale in zip(audit.CONTROLS,(.1,.01,.002)):
                lower=2.8+(-.4+(.0005 if steps==256 else 0.))*dx[None,:]+scale*z[:,None]*(1+dx[None,:])
                controls[control]={'lower':lower,'upper':lower+.0001}
            streams.append({**spec,'policy_fingerprint':'same policy','initial_spots':100.*(1+audit.RELATIVE),
                'stops':np.full((n,5),64,dtype=int),'controls':controls})
            specs.append(spec)
    pairs=[dict(coarse_trial='base',fine_trial='time',same_full_policy=True)]
    return streams,specs,pairs


def test_all_three_controls_and_every_predeclared_independent_stream_remain_visible():
    streams,specs,pairs=statistics_case()
    out=audit.summarize_benchmark(streams,expected_streams=specs,refinement_pairs=pairs)
    assert len(out['price_bounds'])==6 and len(out['delta_sampling'])==12
    assert len(out['bump_refinement'])==6 and len(out['control_changes'])==18
    assert len(out['refinement'])==9 and len(out['per_seed'])==36
    assert out['case_status'].complete.all() and out['failures'].empty
    assert set(out['price_bounds'].query('primary').control)=={'european_premium'}
    assert not out['price_bounds'].continuous_american_price_certified.any()
    assert not out['refinement'].optimal_american_accuracy_certified.any()


def test_price_budget_uses_gap_uncertainty_and_within_seed_sampling_variance():
    streams,specs,pairs=statistics_case()
    t=audit.summarize_benchmark(streams,expected_streams=specs)
    r=t['price_bounds'].query("trial_id=='base' and control=='european_premium'").iloc[0]
    chunks=[s['controls']['european_premium']['lower'][:,2] for s in streams[:2]]
    se=np.sqrt(sum(len(x)*np.var(x,ddof=1) for x in chunks))/16
    assert r.price_se==pytest.approx(se)
    assert r.price_error_budget==pytest.approx(r.gap+1.96*(r.price_se+r.gap_se))
    assert r.price_allowance==pytest.approx(.001+1e-4*abs(r.price))


def test_bumps_are_paired_but_time_resolutions_use_independent_standard_errors():
    streams,specs,pairs=statistics_case()
    t=audit.summarize_benchmark(streams,expected_streams=specs,refinement_pairs=pairs)
    b=t['bump_refinement'].query("trial_id=='base' and control=='european_premium'").iloc[0]
    assert abs(b.estimate)<1e-13 and b.standard_error<1e-13
    row=t['refinement'].query("quantity=='delta' and bump_relative==.0025 and control=='european_premium'").iloc[0]
    cells=t['delta_sampling'].query("control=='european_premium' and bump_relative==.0025")
    assert row.change==pytest.approx(.0005)
    assert row.standard_error==pytest.approx(np.hypot(*cells.standard_error))
    assert row.scope=='fixed lower-policy law sensitivity'


def test_statistical_zero_and_tolerance_equivalence_are_distinct():
    streams,specs,pairs=statistics_case()
    # A deterministic, tiny different control mean is detectable but within tolerance.
    for s in streams:
        for endpoint in ('lower','upper'):
            s['controls']['european_premium'][endpoint]=s['controls']['european'][endpoint]+.00001
    r=audit.summarize_benchmark(streams,expected_streams=specs)['control_changes']
    r=r.query("trial_id=='base' and comparison=='european minus european_premium' and quantity=='price'").iloc[0]
    assert r.tolerance_resolved and not r.zero_in_interval


@pytest.mark.parametrize('kind',['omitted_stream','changed_seed','duplicate_seed','missing_path',
    'missing_control','negative_gap','nonfinite','wrong_stops','wrong_spots','training_leak'])
def test_fixed_stream_pool_and_all_control_paths_cannot_be_replaced(kind):
    s,p,pairs=statistics_case()
    if kind=='omitted_stream':s.pop()
    elif kind=='changed_seed':s[0]['seed']+=100
    elif kind=='duplicate_seed':s[1]['seed']=s[0]['seed'];p[1]['seed']=p[0]['seed']
    elif kind=='missing_path':s[0]['controls']['raw']['lower']=s[0]['controls']['raw']['lower'][:-1]
    elif kind=='missing_control':del s[0]['controls']['raw']
    elif kind=='negative_gap':s[0]['controls']['raw']['upper'][0,2]-=.1
    elif kind=='nonfinite':s[0]['controls']['european']['lower'][0,1]=np.nan
    elif kind=='wrong_stops':s[0]['stops'][0,1]=65
    elif kind=='wrong_spots':s[0]['initial_spots'][0]+=.1
    else:s[0]['training_seed']=s[0]['seed'];p[0]['training_seed']=p[0]['seed']
    with pytest.raises(ValueError):audit.summarize_benchmark(s,expected_streams=p,refinement_pairs=pairs)


def test_one_failed_stream_makes_trial_incomplete_without_dropping_its_failure():
    s,p,pairs=statistics_case();s[1]['failure']={'type':'FloatingPointError','message':'retained'};s[1]['controls']={}
    t=audit.summarize_benchmark(s,expected_streams=p,refinement_pairs=pairs)
    assert len(t['case_status'])==2 and not t['case_status'].query("trial_id=='base'").complete.item()
    assert set(t['price_bounds'].trial_id)=={'time'}
    assert len(t['failures'])==1 and t['failures'].remaining_streams_not_used_for_partial_acceptance.all()
    assert not t['refinement'].complete.any() and not t['refinement'].refinement_check_pass.any()


def test_refinement_rule_change_cannot_be_relabelled_as_fixed_policy():
    s,p,pairs=statistics_case();s[2]['policy_fingerprint']='different';s[3]['policy_fingerprint']='different'
    with pytest.raises(ValueError,match='fixed-policy'):audit.summarize_benchmark(s,expected_streams=p,refinement_pairs=pairs)
    pairs[0]['same_full_policy']=False
    t=audit.summarize_benchmark(s,expected_streams=p,refinement_pairs=pairs)
    assert set(t['refinement'].scope)=={'joint model/calendar/policy construction sensitivity'}


def test_small_lower_policy_change_does_not_certify_optimal_price_when_gaps_are_large():
    s,p,pairs=statistics_case()
    for stream in s:
        for value in stream['controls'].values():
            value['lower']=np.broadcast_to(value['lower'].mean(axis=0),value['lower'].shape).copy()
            value['upper']=value['lower']+.1
    out=audit.summarize_benchmark(s,expected_streams=p,refinement_pairs=pairs)
    assert not out['price_bounds'].price_budget_pass.any()
    price=out['refinement'].query("quantity=='price'")
    assert price.refinement_check_pass.all()
    assert not price.optimal_american_accuracy_certified.any()


@pytest.mark.parametrize('kind',['cross_training_seed','missing_fingerprint','different_policy_same_trial'])
def test_cross_trial_independence_and_sealed_policy_identity(kind):
    s,p,pairs=statistics_case()
    if kind=='cross_training_seed':
        for j in (2,3):s[j]['training_seed']=s[0]['seed'];p[j]['training_seed']=s[0]['seed']
    elif kind=='missing_fingerprint':del s[0]['policy_fingerprint']
    else:s[1]['policy_fingerprint']='different policy'
    with pytest.raises(ValueError):audit.summarize_benchmark(s,expected_streams=p,refinement_pairs=pairs)


@pytest.fixture
def algebra_case(monkeypatch):
    """Deterministic endpoint fixture only; real replay probes use preserved pilots."""
    import rvlab.pricing.rough_simulation as simulation
    import rvlab.pricing.rough_stopping_fourier as fourier
    n=3;nt=5;times=np.linspace(0.,.25,nt);grid=np.linspace(0.,.25,9)
    class Model:
        def __init__(self,value):self.amount=value
        def value(self,stocks,factors,strike):return np.full(np.shape(stocks),self.amount)
    policy=SimpleNamespace(config=SimpleNamespace(training_seed=10,batch_size=512,steps=8),
        strike=100.,maturity=.25,kernel=SimpleNamespace(nodes=np.ones(4)),params=None,
        exercise_times=times,market=SimpleNamespace(spot=100.,dividends=SimpleNamespace(events=()),
            discount_curve=SimpleNamespace(discount=lambda t:np.ones(np.shape(t)))),
        is_call=False,exercise_style='american',premium_cap=.7,exercise_at_inception=False,
        continuations=tuple(Model(.2 if j<4 else 0.) for j in range(nt)))
    spec=dict(trial_id='fixture',seed=20,training_seed=10,paths=n,steps=8,
        exercise_steps=4,factors=4,strike=100.,maturity=.25)
    stocks=np.broadcast_to(.97*100.*(1+audit.RELATIVE),(n,nt,5)).copy();stocks[:,0]=100.*(1+audit.RELATIVE)
    payoff=np.maximum(100.-stocks,0.);stops=np.array([[1,2,3,4,4],[2,3,4,1,2],[4,3,2,1,4]])
    future=payoff+.1
    for path in range(n):
        for bump in range(5):future[path,stops[path,bump],bump]=payoff[path,stops[path,bump],bump]-.1
    euro=future-.2;euro[:,-1]=payoff[:,-1]
    future[:,-1]=payoff[:,-1]
    rule=(payoff>0)&(payoff>=future);rule[:,0]=False;rule[:,-1]=True
    european=np.broadcast_to(np.arange(nt)[None,:,None]*.001,(n,nt,5)).copy()
    premium=np.broadcast_to(np.arange(nt)[None,:,None]*.0001,(n,nt,5)).copy()
    a=dict(times=times,discounts=np.ones(nt),eligible=np.ones(nt,bool),actual_time_grid=grid,
        operator_time_grid=grid,relative_spots=audit.RELATIVE.copy(),initial_spots=100.*(1+audit.RELATIVE),
        stocks=stocks,factors=np.zeros((n,nt,4)),discounted_payoffs=payoff,
        european_martingale=european,premium_martingale=premium,european_proxy=euro.copy(),
        decision_european=euro,decision_continuation=future,decision_rule=rule,
        stop_indices=stops,public_stop_indices=stops.copy())
    row=np.arange(n)[:,None];bump=np.arange(5)[None]
    for name,m in [('raw',np.zeros_like(payoff)),('european',european),('european_premium',european+premium)]:
        adjusted=payoff-m;a[name+'_lower']=adjusted[row,stops,bump];a[name+'_upper']=adjusted.max(axis=1)
    monkeypatch.setattr(simulation,'_time_grid',lambda *args:grid)
    monkeypatch.setattr(fourier,'fourier_policy_fingerprint',lambda p:'fixture fingerprint')
    monkeypatch.setattr(audit,'_replay_and_probe',lambda *args:([{'bounded':True}],np.zeros(2)))
    return a,policy,spec


def test_own_stopping_and_three_genuine_dual_maxima_reconstruct(algebra_case):
    a,p,s=algebra_case;out=audit.validate_benchmark_samples(a,p,spec=s)
    np.testing.assert_array_equal(out['stream']['stops'],a['stop_indices'])
    assert out['checks']['all_payoffs_and_dual_maxima_checked']
    assert out['checks']['conditional_repricing_exhaustive'] is False


@pytest.mark.parametrize('field,index',[
    ('stocks',(0,0,0)),('factors',(0,0,0)),('discounted_payoffs',(0,2,1)),
    ('european_martingale',(0,0,0)),('premium_martingale',(1,4,2)),
    ('decision_continuation',(1,2,3)),('decision_european',(2,1,4)),
    ('stop_indices',(0,1)),('public_stop_indices',(1,2)),('raw_lower',(1,0)),
    ('european_upper',(2,3)),('european_premium_lower',(0,4))])
def test_used_saved_primitive_or_endpoint_tampering_is_rejected(algebra_case,field,index):
    a,p,s=algebra_case;a[field][index]+=.001 if a[field].dtype.kind not in 'iu' else 1
    with pytest.raises(ValueError):audit.validate_benchmark_samples(a,p,spec=s)


def test_common_posthoc_translation_cannot_be_called_a_dual_maximum(algebra_case):
    a,p,s=algebra_case;a['european_premium_lower']+=.01;a['european_premium_upper']+=.01
    with pytest.raises(ValueError,match='Lower estimator'):audit.validate_benchmark_samples(a,p,spec=s)
