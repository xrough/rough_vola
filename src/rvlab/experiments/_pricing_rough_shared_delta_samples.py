"""Independent all-bump identities and bounded centering checks for shared193 deltas."""
from numbers import Integral
import math

import numpy as np
from scipy.special import expit,ndtr,ndtri

from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_simulation import (
    _time_grid,_theta,_drift_flow_with_integral,_weak_transition_branches)
from rvlab.pricing.rough_stopping_fourier import _affine_values
from ._pricing_rough_shared_delta_statistics import RELATIVE


def close(a,b,label,atol=2e-13):
    if np.shape(a)!=np.shape(b) or not np.allclose(a,b,rtol=0.,atol=atol):
        raise ValueError(label)


def weights_for(control,features,j):
    bins=control.get('bins');w=np.asarray(control.get('weights'),float)
    if (isinstance(bins,bool) or not isinstance(bins,Integral) or bins not in (1,4)
            or w.shape!=(bins*features,) or not np.isfinite(w).all()
            or np.any(w< -1e-12) or np.any(w>1+1e-12)
            or np.any(w.reshape(bins,features).sum(axis=1)>1+1e-12)):
        raise ValueError('Frozen control dimensions or per-bin budgets changed')
    return w.reshape(bins,features)[j*bins//64]


def _rng_anchor(a,bundle,seed,paths,steps):
    proxy=bundle['proxy'];params=proxy.params;market=proxy.market;kernel=proxy.kernel
    grid=a['actual_time_grid'];nt=len(grid);nf=len(kernel.nodes)
    if (not np.array_equal(a['absolute_path_indices'],np.arange(paths))
            or not np.array_equal(a['physical_time_grid'],grid)):
        raise ValueError('Full absolute RNG paths and actual physical calendar required')
    shapes={'physical_stocks':(paths,nt,5),'physical_factors':(paths,nt,nf),
        'conditional_log_means':(paths,nt,5),'conditional_variances':(paths,nt),
        'conditional_actual_next':(paths,nt,5),'conditional_expected_next':(paths,nt,5),
        'conditional_european_payoff':(paths,5),'conditional_european_martingale':(paths,5),
        'conditional_european_estimate':(paths,5)}
    if any(k not in a or np.shape(a[k])!=v or not np.isfinite(a[k]).all() for k,v in shapes.items()):
        raise ValueError('Missing finite full-physical-state or conditional-European primitives')
    if np.any(a['conditional_variances']<0):
        raise ValueError('Conditional Gaussian clocks must be nonnegative')
    spots=a['initial_spots'];stock=np.broadcast_to(spots,(paths,5)).copy()
    factors=np.zeros((paths,nf));mu=np.broadcast_to(np.log(spots),(paths,5)).copy();clock=np.zeros(paths)
    for key,value in (('physical_stocks',stock),('physical_factors',factors),
            ('conditional_log_means',mu),('conditional_variances',clock),
            ('conditional_actual_next',np.zeros((paths,5))),('conditional_expected_next',np.zeros((paths,5)))):
        close(a[key][:,0],value,'A physical/conditional primitive has another initial state',2e-12)
    nodes=np.asarray(kernel.nodes);kw=np.asarray(kernel.weights)
    matrix=-np.diag(nodes)-params.kappa*np.ones((nf,1))*kw
    rng=np.random.default_rng(seed);cache={};probes=[];errors=np.zeros(4)
    affine=_AffineSurrogate(params,market,kernel,np.array([proxy.strike]),
        np.array([proxy.maturity]),np.array([proxy.is_call]),grid,proxy.control_config)
    dates={steps//4,steps//2,3*steps//4,steps};ids=np.arange(min(2,paths))
    for k,(left,right) in enumerate(zip(grid[:-1],grid[1:]),1):
        dt=float(right-left);theta=_theta(params,.5*(left+right));key=(dt,theta)
        if key not in cache:
            cache[key]=_drift_flow_with_integral(matrix,np.full(nf,params.kappa*(theta-params.v0)),kw,params.v0,dt/2)
        flow=cache[key];growth=math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        for begin in range(0,paths,512):
            end=min(begin+512,paths);old=factors[begin:end];u=rng.random((end-begin,3))
            law=_weak_transition_branches(old,params,nodes=nodes,weights=kw,flow=flow[:nf,:nf],
                shift=flow[:nf,-1],integrated_flow=flow,dt=dt,theta=theta,growth=growth,
                stock_integral=proxy.config.stock_integral)
            p=law['probabilities'];choice=(u[:,0]>p[:,0]).astype(int)+(u[:,0]>p[:,:2].sum(axis=1)).astype(int)
            order=(u[:,1]>=.5).astype(int);row=np.arange(end-begin)
            q=law['orthogonal'];z=ndtri(np.clip(u[:,2],np.finfo(float).tiny,1-np.finfo(float).eps))
            means=np.log(stock[begin:end])[:,None,None,:]+law['correlated'][:,:,None,None]-.5*q[...,None]
            cmean=mu[begin:end,None,None,:]+law['correlated'][:,:,None,None]-.5*q[...,None]
            cclock=clock[begin:end,None,None]+q
            nextstock=np.exp(means[row,choice,order]+np.sqrt(q[row,choice,order])[:,None]*z[:,None])
            new=law['new_factors'][row,choice]
            used=ids[(ids>=begin)&(ids<end)]
            if k in dates and len(used):
                local=used-begin
                ex=np.sum(_affine_values(affine,float(right),law['new_factors'][local],
                    cmean[local],cclock[local,...,None]).mean(axis=2)*p[local,:,None],axis=1)
                av=_affine_values(affine,float(right),new[local,None],
                    cmean[local,choice[local],order[local]][:,None,None,:],
                    cclock[local,choice[local],order[local]][:,None,None,None])[:,0,0]
                close(a['conditional_expected_next'][used,k],ex,'A conditional mean failed its bounded reference probe',2e-11)
                close(a['conditional_actual_next'][used,k],av,'A conditional actual failed its bounded reference probe',2e-11)
                probes.append({'kind':'conditional European','physical_index':k,
                    'paths':used.tolist(),'all_bumps':True,'exhaustive_conditional_repricing':False})
            stock[begin:end]=nextstock;factors[begin:end]=new
            mu[begin:end]=cmean[row,choice,order];clock[begin:end]=cclock[row,choice,order]
        for i,(key,value) in enumerate((('physical_stocks',stock),('physical_factors',factors),
                ('conditional_log_means',mu),('conditional_variances',clock))):
            errors[i]=max(errors[i],float(np.max(abs(a[key][:,k]-value))))
    if errors[0]>2e-12 or errors[1]>2e-13 or errors[2]>2e-12 or errors[3]>2e-13:
        raise ValueError('The multistock CRN/factor/log-mean/clock stream changed')
    index=np.searchsorted(grid,a['times'])
    close(a['physical_stocks'][:,index],a['stocks'],'Calendar stocks differ from full physical replay',2e-12)
    close(a['physical_factors'][:,index],a['factors'],'Calendar factors differ from full physical replay')
    root=np.sqrt(clock[:,None]);forward=np.exp(mu+.5*clock[:,None])
    d2=np.divide(mu-np.log(proxy.strike),root,out=np.zeros_like(mu),where=root>0);d1=d2+root
    qpay=(forward*ndtr(d1)-proxy.strike*ndtr(d2) if proxy.is_call
          else proxy.strike*ndtr(-d2)-forward*ndtr(-d1))
    terminal=np.maximum((1 if proxy.is_call else -1)*(forward-proxy.strike),0.)
    qpay=float(market.discount_curve.discount(proxy.maturity))*np.where(clock[:,None]==0,terminal,qpay)
    cm=np.sum(a['conditional_actual_next'][:,1:]-a['conditional_expected_next'][:,1:],axis=1)
    close(a['conditional_european_payoff'],qpay,'Conditional terminal payoff changed',2e-12)
    close(a['conditional_european_martingale'],cm,'Conditional martingale accumulation changed',2e-12)
    close(a['conditional_european_estimate'],qpay-cm,'Conditional European estimate changed',2e-12)
    return qpay-cm,probes,errors


def _spline_actual(coeff,stock,strike,knots):
    z=stock/strike;i=np.searchsorted(knots,z,side='right')
    if coeff.ndim==3:
        c=coeff[np.arange(len(stock)),i];return c[:,0]+c[:,1]*z
    c=coeff[np.arange(len(stock))[:,None],np.arange(coeff.shape[1])[None,:],i[:,None]]
    return c[...,0]+c[...,1]*z[:,None]


def _right_reference(proxy,selected,models,affine,node_proxy,j,law,left,right,factors):
    from ._pricing_rough_local_control import _correction_coefficients,_linear_coefficients,signed_linear_expectation
    from ._pricing_rough_node_refinement_fast import refined_stock_knots,snell_refinement_coefficients,signed_shared_expectations
    from ._pricing_rough_bellman_control_aligned import target_nodal_premiums
    nodes=np.r_[0.,refined_stock_knots(proxy.config.stock_knots,8)]
    new=law['new_factors'];p=law['probabilities']
    m=np.log(left)[:,None,None]+law['log_stock_shifts'];v=law['orthogonal']
    lc=_correction_coefficients(proxy,selected,j,new,3.)
    le=np.sum(signed_linear_expectation(m,v,lc[:,:,None],strike=proxy.strike,
        knots=proxy.config.stock_knots).mean(axis=2)*p,axis=1)
    gc,_=snell_refinement_coefficients(proxy,selected,j,new,fine_nodes=nodes)
    ge=np.sum(signed_shared_expectations(m,v,gc,strike=proxy.strike,knots=nodes[1:]).mean(axis=-1)*p[...,None],axis=1)
    tc=_linear_coefficients(nodes,target_nodal_premiums(proxy,selected,models,j,new,fine_nodes=nodes,node_proxy=node_proxy))
    te=np.sum(signed_shared_expectations(m,v,tc,strike=proxy.strike,knots=nodes[1:]).mean(axis=-1)*p[...,None],axis=1)
    ee=np.sum(_affine_values(affine,float(proxy.continuation_times[j]),new,m[...,None],v[...,None])[...,0].mean(axis=2)*p,axis=1)
    la=_spline_actual(_correction_coefficients(proxy,selected,j,factors,3.),right,proxy.strike,proxy.config.stock_knots)
    gc,_=snell_refinement_coefficients(proxy,selected,j,factors,fine_nodes=nodes)
    ga=_spline_actual(gc,right,proxy.strike,nodes[1:])
    tc=_linear_coefficients(nodes,target_nodal_premiums(proxy,selected,models,j,factors,fine_nodes=nodes,node_proxy=node_proxy))
    ta=_spline_actual(tc,right,proxy.strike,nodes[1:])
    ea=_affine_values(affine,float(proxy.continuation_times[j]),factors[:,None],
        np.log(right)[:,None,None,None],np.zeros((len(right),1,1,1)))[:,0,0,0]
    return np.column_stack((la,ga,ta,ea)),np.column_stack((le,ge,te,ee))


def _physical_reference(proxy,affine,law,left,right,old,new,tleft,tright):
    """Four base increments using the preserved general capped-premium integral."""
    from rvlab.pricing.rough_estimators import _conditional_payoffs
    from rvlab.pricing.rough_stopping_fourier import (
        _interpolate,_continuation_coefficients,_premium_values,_expected_premium_values)
    branch=law['new_factors'];p=law['probabilities'];n=len(left)
    m=np.log(left)[:,None,None]+law['log_stock_shifts'][...,None]
    v=law['orthogonal'][...,None]
    expected=np.sum(_affine_values(affine,float(tright),branch,m,v).mean(axis=2)*p[...,None],axis=1)
    actual=_affine_values(affine,float(tright),new[:,None],np.log(right)[:,None,None],
        np.zeros((n,1,1,5)))[:,0,0]
    ei=actual-expected
    discount=float(proxy.market.discount_curve.discount(tright))
    intrinsic=np.maximum((1 if proxy.is_call else -1)*(right-proxy.strike),0.)
    ii=discount*(intrinsic-np.sum(_conditional_payoffs(m,v,proxy.strike,proxy.is_call)
        .mean(axis=2)*p[...,None],axis=1))
    ec=_interpolate(proxy,float(tright),branch)[:,:,None,None]
    ep=np.sum(_expected_premium_values(m,v,ec,strike=proxy.strike,
        stock_knots=proxy.config.stock_knots,cap=proxy.premium_cap).mean(axis=2)*p[...,None],axis=1)
    ap=_premium_values(right,_interpolate(proxy,float(tright),new)[:,None],
        strike=proxy.strike,stock_knots=proxy.config.stock_knots,cap=proxy.premium_cap)
    pi=discount*(ap-ep)
    left_euro=_affine_values(affine,float(tleft),old[:,None],np.log(left)[:,None,None],
        np.zeros((n,1,1,5)))[:,0,0]
    q=_premium_values(left,_continuation_coefficients(proxy,float(tleft),old)[:,None],
        strike=proxy.strike,stock_knots=proxy.continuations[0].knots,cap=proxy.premium_cap)
    dl=float(proxy.market.discount_curve.discount(tleft))
    intrinsic_left=np.maximum((1 if proxy.is_call else -1)*(left-proxy.strike),0.)
    multiplier=expit((dl*(intrinsic_left-q)-left_euro)/(dl*.05))
    return np.stack((ei,pi,ii,multiplier*(ii-ei)),axis=-1)


def _panel(a,bundle,prior,grid,models,representations,q):
    from ._pricing_rough_exercise_block import exercise_block_law
    from ._pricing_rough_bellman_control_aligned import aligned_node_proxy
    from ._pricing_rough_bellman_policy import BellmanEvaluator
    proxy=bundle['proxy'];selected=bundle['selected_policy'];times=a['times'];physical=a['physical_time_grid']
    affine=_AffineSurrogate(proxy.params,proxy.market,proxy.kernel,np.array([proxy.strike]),
        np.array([proxy.maturity]),np.array([proxy.is_call]),a['operator_time_grid'],proxy.control_config)
    physical_affine=affine if np.array_equal(physical,a['operator_time_grid']) else _AffineSurrogate(
        proxy.params,proxy.market,proxy.kernel,np.array([proxy.strike]),np.array([proxy.maturity]),
        np.array([proxy.is_call]),physical,proxy.control_config)
    node_proxy=aligned_node_proxy(proxy,affine);evaluator=BellmanEvaluator(proxy,selected,orders=(16,))
    ids=np.array([i for i in (0,1,511,512) if i<len(a['stocks'])]);records=[]
    for j in (0,16,32,48,63,64):
        value=_affine_values(affine,float(times[j]),a['factors'][ids,j,None],
            np.log(a['stocks'][ids,j])[:,None,None,:],np.zeros((len(ids),1,1,5)))[:,0,0]
        close(a['calendar_european'][ids,j],value,'Calendar European function changed its frozen129 grid',2e-11)
        records.append({'kind':'frozen calendar European','calendar_index':j,'paths':ids.tolist(),
            'all_bumps':True,'operator_grid_points':129,
            'max_error':float(np.max(abs(value-a['calendar_european'][ids,j])))})
    for j in (16,32,48,63):
        left,right=np.searchsorted(physical,times[j:j+2])
        base=np.zeros((len(ids),5,4))
        for k in range(left,right):
            law=exercise_block_law(proxy,a['physical_factors'][ids,k],times=physical[k:k+2])
            base+=_physical_reference(proxy,physical_affine,law,a['physical_stocks'][ids,k],
                a['physical_stocks'][ids,k+1],a['physical_factors'][ids,k],
                a['physical_factors'][ids,k+1],float(physical[k]),float(physical[k+1]))
        for rep in representations:
            delta=a[rep+'_processes'][ids,j+1,:,4:]-a[rep+'_processes'][ids,j,:,4:]
            close(delta,base,'A physical base component failed bounded actual-law centering',2e-11)
        records.append({'kind':'physical base centering','calendar_index':j,'paths':ids.tolist(),
            'all_bumps':True,'physical_grid_points':len(physical),'components':
            ['european','premium','intrinsic','micro_boundary'],
            'general_capped_premium_reference':True,'exhaustive_conditional_repricing':False,
            'max_error':float(np.max(abs(delta-base)))})
        for rep in representations:
            ranges=[(left,right)] if rep=='original' else [(k,k+1) for k in range(left,right)]
            for bump in (0,2,4):
                total=np.zeros((len(ids),8))
                for start,end in ranges:
                    law=exercise_block_law(proxy,a['physical_factors'][ids,start],times=physical[start:end+1])
                    av,ev=_right_reference(proxy,selected,models,affine,node_proxy,j+1,law,
                        a['physical_stocks'][ids,start,bump],a['physical_stocks'][ids,end,bump],
                        a['physical_factors'][ids,end])
                    total+=av-ev
                intrinsic=np.maximum((1 if proxy.is_call else -1)*(a['stocks'][ids,j,bump]-proxy.strike),0.)
                margin=a['discounts'][j]*(intrinsic-q[ids,j,bump])-a['calendar_european'][ids,j,bump]
                gates=np.column_stack((np.ones(len(ids)),np.exp(-.5*(margin/.1)**2),np.exp(-.5*(margin/.3)**2)))
                local=(a['discounts'][j+1]*total[:,0,None]*gates)@weights_for(prior,3,j)
                refinement=(a['discounts'][j+1]*total[:,1:3,None]*gates[:,None,:]).reshape(len(ids),6)@weights_for(grid,6,j)
                stage=a[rep+'_stage_corrections']
                close(stage[ids,j+1,bump,0]-stage[ids,j,bump,0],local,'A local stage failed bounded exact centering',2e-11)
                close(stage[ids,j+1,bump,1]-stage[ids,j,bump,1],refinement,'A grid stage failed bounded exact centering',2e-11)
                target_delta=(a[rep+'_processes'][ids,j+1,bump,:4]-a[rep+'_processes'][ids,j,bump,:4]
                    +(a[rep+'_latest_martingale'][ids,j+1,bump]-a[rep+'_latest_martingale'][ids,j,bump])[:,None])
                expected=total[:,7,None]+a['discounts'][j+1]*total[:,3:7]
                close(target_delta,expected,'A target stage failed bounded exact centering',2e-11)
                active=a['gh16_computed_mask'][ids,j,bump];policy_error=0.
                if np.any(active):
                    chosen=ids[active];value=evaluator.conditional_expectations(j,a['stocks'][chosen,j,bump],a['factors'][chosen,j])['continuation'][:,0]
                    close(a['gh16_continuation'][chosen,j,bump],value,'Frozen128 GH16 definition changed',2e-11)
                    policy_error=float(np.max(abs(value-a['gh16_continuation'][chosen,j,bump])))
                records.append({'kind':'calendar centering','representation':rep,'calendar_index':j,
                    'bump_index':bump,'paths':ids.tolist(),'physical_steps_in_block':int(right-left),
                    'gh16_error':policy_error,'exhaustive_conditional_repricing':False})
    return records


def validate_shared_delta_samples(a,bundle,prior,grid,tail_seal,target_seal,*,seed,paths,steps,
                                   representations,coefficients,models):
    for value,name,minimum in ((seed,'seed',0),(paths,'paths',2)):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if steps not in (128,256) or isinstance(steps,bool):
        raise ValueError('Declared128/256 outer law required')
    expected=('original','micro') if steps==128 else ('micro',)
    if tuple(representations)!=expected:raise ValueError('Changed declared original/micro representations')
    proxy=bundle['proxy'];selected=bundle['selected_policy'];fit=bundle['weights']
    if (proxy.config.steps!=128 or proxy.config.batch_size!=512 or len(proxy.kernel.nodes)!=4
            or proxy.market.dividends.events or len(models)!=65):
        raise ValueError('Frozen128operator/four-factor/no-cash512-batch source required')
    times=np.asarray(a['times']);discount=np.asarray(a['discounts'])
    original=_time_grid(proxy.params,proxy.market,proxy.exercise_times,128)
    physical=_time_grid(proxy.params,proxy.market,proxy.exercise_times,steps)
    if (not np.array_equal(a['relative_spots'],RELATIVE)
            or not np.array_equal(a['initial_spots'],proxy.market.spot*(1+RELATIVE))
            or not np.array_equal(times,proxy.exercise_times) or not np.array_equal(times,selected.times)
            or not np.array_equal(discount,proxy.market.discount_curve.discount(times))
            or not np.array_equal(a['operator_time_grid'],original) or not np.array_equal(a['actual_time_grid'],physical)
            or np.shape(a['eligible'])!=(65,) or a['eligible'].dtype.kind!='b' or not a['eligible'].all()):
        raise ValueError('Stock bumps, curves, frozen operator or actual exercise calendar changed')
    shapes={'stocks':(paths,65,5),'factors':(paths,65,4),'payoff':(paths,65,5),
        'calendar_european':(paths,65,5),'gh16_stops':(paths,5),'gh16_computed_mask':(paths,65,5)}
    if any(k not in a or np.shape(a[k])!=s or not np.isfinite(a[k]).all() for k,s in shapes.items()):
        raise ValueError('Full finite calendar states, own stopping masks and payoffs required')
    if np.any(a['stocks']<=0) or not np.array_equal(a['stocks'][:,0],np.broadcast_to(a['initial_spots'],(paths,5))) or np.any(a['factors'][:,0]!=0):
        raise ValueError('Initial stock bumps or full factor memory changed')
    intrinsic=np.maximum((1 if proxy.is_call else -1)*(a['stocks']-proxy.strike),0.)
    if not np.array_equal(a['payoff'],intrinsic*discount[None,:,None]):raise ValueError('Raw vanilla cashflow construction changed')
    mask=a['gh16_computed_mask'];c=np.asarray(a['gh16_continuation'])
    if (mask.dtype.kind!='b' or c.shape!=mask.shape or not np.isfinite(c[mask]).all()
            or not np.isnan(c[~mask]).all() or np.any(c[mask]<0)):
        raise ValueError('Own GH16 continuations must be finite on alive-ITM mask and NaN elsewhere')
    stop=np.full((paths,5),64,dtype=int);em=np.zeros_like(mask)
    for j in range(1,64):
        active=(stop==64)&(intrinsic[:,j]>0);em[:,j]=active
        stop[active&(intrinsic[:,j]>=c[:,j])]=j
    if a['gh16_stops'].dtype.kind not in 'iu' or not np.array_equal(a['gh16_stops'],stop) or not np.array_equal(mask,em):
        raise ValueError('A stock bump reused another stopping decision or omitted an alive state')
    q=np.stack([np.clip(model.value(a['stocks'][:,j],a['factors'][:,j],proxy.strike),0.,selected.premium_cap)
                for j,model in enumerate(selected.continuations)],axis=1)
    baseline=np.full((paths,5),64,dtype=int)
    for j in range(1,64):
        act=(baseline==64)&(intrinsic[:,j]>0)&(intrinsic[:,j]>=a['calendar_european'][:,j]/discount[j]+q[:,j])
        baseline[act]=j
    conditional,probes,physical_errors=_rng_anchor(a,bundle,seed,paths,steps)
    coef=np.asarray(coefficients,float)
    if coef.shape!=(193,) or not np.isfinite(coef).all() or np.any(abs(coef)>2):raise ValueError('Frozen193 raw coefficients changed')
    values={};reference=None;row=np.arange(paths)[:,None];bump=np.arange(5)[None,:]
    tail=tail_seal['controls'][tail_seal['selected_control']]
    if target_seal['selected_control']!='target 4 bins penalty0':raise ValueError('Selected target control changed')
    target=target_seal['controls'][target_seal['selected_control']]
    for rep in representations:
        process=np.asarray(a[rep+'_processes']);stage=np.asarray(a[rep+'_stage_corrections'])
        latest=np.asarray(a[rep+'_latest_martingale']);chosen=np.asarray(a[rep+'_selected_martingale'])
        if (process.shape!=(paths,65,5,8) or stage.shape!=(paths,65,5,3) or latest.shape!=(paths,65,5)
                or chosen.shape!=latest.shape or not all(np.isfinite(x).all() for x in (process,stage,latest,chosen))
                or any(np.any(x[:,0]!=0) for x in (process,stage,latest,chosen))):
            raise ValueError('Retained process/stage/origin dimensions changed')
        stable=np.zeros((paths,65,5));addition=np.zeros_like(stable)
        basis=np.zeros((paths,5,4,8,6));margin=(a['payoff']-a['calendar_european'])/discount[None,:,None]-q
        if fit.config.predictable_features or tuple(fit.config.components)!=('european','premium','micro_boundary'):
            raise ValueError('Original stable comparator specification changed')
        tail_expected=np.zeros_like(stable)
        for j in range(64):
            idx=np.searchsorted(fit.time_edges[1:],times[j+1],side='left')
            increment=process[:,j+1]-process[:,j]
            stable[:,j+1]=stable[:,j]+np.sum(increment[...,[4,5,7]]*fit.weights[idx],axis=-1)
            addition[:,j+1]=addition[:,j]+increment[...,:4]@weights_for(target,4,j)
            current=stable[:,j]+stage[:,j,:,0]+stage[:,j,:,1]
            future=stable[:,j+1]+stage[:,j+1,:,0]+stage[:,j+1,:,1]
            di=(process[:,j+1,:,6]-future)-(process[:,j,:,6]-current)
            de=(process[:,j+1,:,4]-future)-(process[:,j,:,4]-current)
            post=(j>=baseline).astype(float);left=margin[:,j]
            gates=np.stack((post,post*expit(left/.1),post*expit(left/.3),expit(left/.03),
                expit(left/.1),expit(left/.3),(a['payoff'][:,j]==0).astype(float)),axis=-1)
            tail_expected[:,j+1]=tail_expected[:,j]+np.sum(gates*np.concatenate((np.repeat(di[...,None],6,axis=-1),de[...,None]),axis=-1)*weights_for(tail,7,j),axis=-1)
            norm=(a['factors'][:,j]-selected.continuations[j].center)/selected.continuations[j].scale
            gate=np.concatenate((np.ones((paths,5,1)),np.clip((a['stocks'][:,j,:,None]/proxy.strike-1)/.05,-3,3),
                                 np.broadcast_to(np.clip(norm,-3,3)[:,None,:],(paths,5,4))),axis=-1)
            basis[:,:,j//16]+=increment[:,:,:,None]*gate[:,:,None,:]*(stop>j)[:,:,None,None]
        close(stage[...,2],tail_expected,'Tail weighted stage changed predictable left gates or own baseline stops')
        close(latest,stable+stage.sum(axis=-1),'Latest martingale changed weighted stage composition')
        close(chosen,latest+addition,'Selected target martingale changed frozen4bin weights')
        raw=a['payoff'][row,stop,bump];stable_price=raw-stable[row,stop,bump]
        lower=raw-chosen[row,stop,bump];upper=np.max(a['payoff']-chosen,axis=1)
        anchor=a['payoff'][:,-1]-process[:,-1,:,4]-conditional
        close(a[rep+'_anchor'],anchor,'A bumped conditional European anchor changed',2e-12)
        feature=np.concatenate((basis.reshape(paths,5,192),anchor[...,None]),axis=-1)
        close(a[rep+'_basis'],feature,'193 own-bump left gates/survival/order changed',2e-12)
        shift=feature@coef;close(a[rep+'_shift'],shift,'Common control was changed or empirically recentered',2e-12)
        for suffix,expected_value in (('raw',raw),('stable',stable_price),('unchanged_lower',lower),
                ('unchanged_upper',upper),('shared_lower',lower-shift),('shared_upper',upper-shift)):
            close(a[rep+'_'+suffix],expected_value,'A bumped endpoint changed its own stopping time or common shift',2e-12)
        close(a[rep+'_shared_upper']-a[rep+'_shared_lower'],upper-lower,'A common shift changed a stopping gap')
        if reference is not None:
            close(process[...,4:],reference,'Changing control representation changed physical base martingales')
        else:reference=process[...,4:]
        values[rep]={'raw':a[rep+'_raw'],'stable':a[rep+'_stable'],'shared':a[rep+'_shared_lower'],'stops':stop}
    probes+=_panel(a,bundle,prior,grid,models,representations,q)
    return {'stream':{'seed':seed,'steps':steps,'relative_spots':a['relative_spots'],
        'initial_spots':a['initial_spots'],'representations':values},
        'checks':{'paths':paths,'seed':seed,'steps':steps,'stocks':5,'all_own_stops_checked':True,
            'all193_bases_and_shifts_checked':True,'all_factor_and_CRN_paths_checked':True,
            'max_physical_stock_error':float(physical_errors[0]),'max_physical_factor_error':float(physical_errors[1]),
            'max_log_mean_error':float(physical_errors[2]),'max_clock_error':float(physical_errors[3]),
            'all_paths_retained':True,'original_policy_operator_steps':128,
            'conditional_repricing_exhaustive':False,'empirical_valuation_mean_subtracted':False},
        'probes':probes}


__all__=['validate_shared_delta_samples']

