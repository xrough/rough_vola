"""State-only one-exercise-step Bellman challenges of a frozen rough policy.

The next Snell envelope is approximate: it uses the frozen European proxy and
continuation regression, not the conditional value of the current strategy.
Consequently this is an admissible challenge, not a policy-improvement theorem.
Only the continuation premium is capped; the Snell envelope is not capped.
"""
from numbers import Integral
import numpy as np
import pandas as pd
from ._pricing_rough_exercise_block import exercise_block_law
from rvlab.pricing.rough_estimators import _AffineSurrogate,_conditional_payoffs
from rvlab.pricing.rough_stopping_fourier import _affine_values,_expected_premium_values,fourier_policy_fingerprint
from rvlab.pricing.rough_simulation import _time_grid

ORDERS=(8,16,32)
BASELINE_NAME='frozen selected policy'

class BellmanNumericalError(FloatingPointError):
    """A conditional lookahead failed; no silent substitute is used."""

def bellman_candidate_specs():
    result={BASELINE_NAME:{'order':None,'blend':0.}}
    for order in ORDERS:
        for blend in (.5,1.):result[f'GH{order} blend{blend:g}']={'order':order,'blend':blend}
    return result

class BellmanEvaluator:
    """Exact discrete mixture plus deterministic Gaussian integration."""
    def __init__(self,proxy,selected,*,orders=ORDERS,batch_size=32):
        orders=tuple(orders)
        if (not orders or len(set(orders))!=len(orders)
                or any(isinstance(q,bool) or not isinstance(q,Integral) or q<2 for q in orders)
                or isinstance(batch_size,bool) or not isinstance(batch_size,Integral) or batch_size<1
                or proxy.exercise_style!='american' or proxy.market.dividends.events
                or fourier_policy_fingerprint(selected.base_policy)!=fourier_policy_fingerprint(proxy)
                or not np.array_equal(selected.times,proxy.exercise_times)
                or not np.array_equal(proxy.exercise_times,proxy.continuation_times)
                or selected.premium_floor!=0. or selected.premium_cap is None
                or not np.isfinite(selected.premium_cap) or selected.premium_cap<=0
                or selected.exercise_at_inception
                or max((1 if proxy.is_call else -1)*(proxy.market.spot-proxy.strike),0.)>0):
            raise ValueError('Declare positive quadrature budgets and a frozen no-dividend American policy')
        self.proxy,self.selected,self.orders,self.batch_size=proxy,selected,orders,int(batch_size)
        self.times=np.asarray(selected.times);self.grid=_time_grid(proxy.params,proxy.market,self.times,proxy.config.steps)
        self.affine=_AffineSurrogate(proxy.params,proxy.market,proxy.kernel,np.array([proxy.strike]),
            np.array([proxy.maturity]),np.array([proxy.is_call]),self.grid,proxy.control_config)
        self.discounts=proxy.market.discount_curve.discount(self.times)
        self.nodes=[];self.weights=[];self.slices=[];start=0
        for q in orders:
            x,w=np.polynomial.hermite.hermgauss(int(q));self.nodes.extend(np.sqrt(2.)*x);self.weights.append(w/np.sqrt(np.pi))
            self.slices.append(slice(start,start+q));start+=q
        self.nodes=np.asarray(self.nodes)

    def endpoint_components(self,j,stocks,factors):
        """Undiscounted next-date Euro, clipped q, payoff and true envelope."""
        if isinstance(j,bool) or not isinstance(j,Integral) or not 0<=j<len(self.times):raise ValueError('Unknown exercise index')
        stocks=np.asarray(stocks,float);factors=np.asarray(factors,float)
        if factors.ndim<1 or factors.shape[-1]!=len(self.proxy.kernel.nodes):raise ValueError('Full factor coordinates required')
        factors=np.broadcast_to(factors,stocks.shape+(factors.shape[-1],))
        if np.any(stocks<=0) or not np.isfinite(stocks).all() or not np.isfinite(factors).all():raise ValueError('Finite positive stocks and factors required')
        shape=stocks.shape;s=stocks.reshape(-1);u=factors.reshape(len(s),-1)
        payoff=np.maximum((1 if self.proxy.is_call else -1)*(s-self.proxy.strike),0.)
        if j==len(self.times)-1:eu=payoff.copy();q=np.zeros_like(s)
        else:
            eu=self.affine.values(float(self.times[j]),u[:,None,:],np.log(s)[:,None,None],np.zeros((len(s),1,1)),np.array([0]))[:,0,0,0]/self.discounts[j]
            q=np.clip(self.selected.continuations[j].value(s,u,self.proxy.strike),0.,self.selected.premium_cap)
        values={'european':eu,'premium':q,'payoff':payoff,'kink':payoff-eu-q,'envelope':np.maximum(payoff,eu+q)}
        if not all(np.isfinite(v).all() for v in values.values()):raise BellmanNumericalError('Endpoint Snell evaluation is nonfinite')
        return {k:v.reshape(shape) for k,v in values.items()}

    def endpoint_values(self,j,stocks,factors):return self.endpoint_components(j,stocks,factors)['envelope']

    def conditional_expectations(self,j,stocks,factors):
        """C_j = E[Dnext*Vhat_next|S_j,U_j]/D_j; no realized future input."""
        if isinstance(j,bool) or not isinstance(j,Integral) or not 0<=j<len(self.times)-1:raise ValueError('Unknown left exercise index')
        stocks=np.asarray(stocks,float);factors=np.asarray(factors,float)
        if (stocks.ndim!=1 or len(stocks)<1 or factors.shape!=(len(stocks),len(self.proxy.kernel.nodes))
                or not np.isfinite(stocks).all() or np.any(stocks<=0) or not np.isfinite(factors).all()):
            raise ValueError('Aligned positive stock vector and full factor matrix required')
        left,right=self.times[j:j+2];a,b=np.searchsorted(self.grid,[left,right])
        if self.grid[a]!=left or self.grid[b]!=right or not 1<=b-a<=2:raise ValueError('One/two original physical increments required')
        cv=np.empty((len(stocks),len(self.orders)));direct=np.empty_like(cv);mass=0.;minimum=np.inf
        for begin in range(0,len(stocks),self.batch_size):
            end=min(begin+self.batch_size,len(stocks));s=stocks[begin:end];u=factors[begin:end]
            law=exercise_block_law(self.proxy,u,times=self.grid[a:b+1]);new=law['new_factors'];p=law['probabilities']
            mass=max(mass,float(np.max(abs(p.sum(axis=1)-1))))
            mean=np.log(s)[:,None,None]+law['log_stock_shifts'];clock=law['orthogonal'];ratio=self.discounts[j+1]/self.discounts[j]
            weighted=lambda values:np.sum(values.mean(axis=2)*p,axis=1)
            if j+1==len(self.times)-1:
                expected=ratio*weighted(_conditional_payoffs(mean,clock,self.proxy.strike,self.proxy.is_call))
                cv[begin:end]=expected[:,None];direct[begin:end]=expected[:,None];continue
            model=self.selected.continuations[j+1];raw_coefficient=model.coefficients(new)
            coefficient=raw_coefficient if model.knots else raw_coefficient[...,None,:]
            eu_expected=_affine_values(self.affine,float(right),new,mean[...,None],clock[...,None])[...,0]/self.discounts[j+1]
            q_expected=_expected_premium_values(mean[...,None],clock[...,None],raw_coefficient[:,:,None,None],
                strike=self.proxy.strike,stock_knots=model.knots,cap=self.selected.premium_cap)[...,0]
            log_stock=mean[...,None]+np.sqrt(clock)[...,None]*self.nodes
            shape=log_stock.shape;nn,bb,oo,qq=shape
            eu=_affine_values(self.affine,float(right),new,log_stock.reshape(nn,bb,oo*qq,1),np.zeros((nn,bb,oo*qq,1)))[...,0].reshape(shape)/self.discounts[j+1]
            normalized=np.exp(log_stock)/self.proxy.strike;index=np.searchsorted(model.knots,normalized,side='right')
            c=coefficient[np.arange(nn)[:,None,None,None],np.arange(bb)[None,:,None,None],index]
            q=np.clip(c[...,0]+normalized*(c[...,1]+normalized*c[...,2]),0.,self.selected.premium_cap)
            payoff=np.maximum((1 if self.proxy.is_call else -1)*(np.exp(log_stock)-self.proxy.strike),0.)
            excess=np.maximum(payoff-eu-q,0.);envelope=np.maximum(payoff,eu+q)
            for k,(sl,w) in enumerate(zip(self.slices,self.weights)):
                extra=excess[...,sl]@w;full=envelope[...,sl]@w
                cv[begin:end,k]=ratio*weighted(eu_expected+q_expected+extra)
                direct[begin:end,k]=ratio*weighted(full)
            minimum=min(minimum,float(np.min(cv[begin:end])))
        if not np.isfinite(cv).all() or not np.isfinite(direct).all() or np.any(cv<0) or np.any(direct<0):
            raise BellmanNumericalError('Conditional Bellman quadrature is nonfinite or negative')
        return {'continuation':cv,'direct_continuation':direct,'diagnostics':{
            'orders':self.orders,'physical_increments':int(b-a),'max_probability_error':mass,
            'max_cv_direct_difference':np.max(abs(cv-direct),axis=0),'terminal_exact_payoff':j+1==len(self.times)-1,
            'only_premium_capped':True,'actual_future_state_used':False,'all_factor_branches_and_stock_orderings':True}}

    def trace_continuations(self,trace,*,forbidden_seeds=()):
        """Compute every exercise-relevant ITM state, retaining all outer paths."""
        seed=trace.get('seed');stocks=np.asarray(trace['stocks']);factors=np.asarray(trace['factors']);euro=np.asarray(trace['european_proxy'])
        if (isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0
                or any(isinstance(s,bool) or not isinstance(s,Integral) or s<0 for s in forbidden_seeds)
                or seed in set(self.selected.training_seeds)|set(forbidden_seeds)
                or trace.get('policy_fingerprint')!=fourier_policy_fingerprint(self.proxy)
                or trace.get('kernel_fingerprint')!=self.proxy.kernel.fingerprint
                or trace.get('strike')!=self.proxy.strike
                or not np.array_equal(trace.get('initial_spots'),[self.proxy.market.spot])
                or not np.array_equal(trace['times'],self.times) or not np.array_equal(trace['actual_time_grid'],self.grid)
                or not np.array_equal(trace['discounts'],self.discounts) or stocks.ndim!=3 or stocks.shape[-1]!=1
                or euro.shape!=stocks.shape or factors.shape!=(len(stocks),len(self.times),len(self.proxy.kernel.nodes))
                or np.any(stocks<=0) or not all(np.isfinite(a).all() for a in (stocks,euro,factors))):
            raise ValueError('Independent matching full-state trace required')
        n,nt=stocks.shape[:2];payoff=np.maximum((1 if self.proxy.is_call else -1)*(stocks[:,:,0]-self.proxy.strike),0.)
        mask=payoff>0;mask[:,0]=False;mask[:,-1]=False
        result=np.full((n,nt,len(self.orders)),np.nan);direct=result.copy();baseline=np.empty((n,nt));diagnostics=[]
        for j in range(nt):
            q=np.clip(self.selected.continuations[j].value(stocks[:,j,0],factors[:,j],self.proxy.strike),0.,self.selected.premium_cap)
            baseline[:,j]=euro[:,j,0]/self.discounts[j]+q
            active=mask[:,j]
            if np.any(active):
                values=self.conditional_expectations(j,stocks[active,j,0],factors[active,j])
                result[active,j]=values['continuation'];direct[active,j]=values['direct_continuation']
                diagnostics.append({'exercise_index':j,'states':int(active.sum()),**values['diagnostics']})
        return {'bellman_continuations':result,'bellman_direct_continuations':direct,'bellman_computed_mask':mask,
            'baseline_continuation':baseline,'diagnostics':diagnostics}

def evaluate_bellman_candidates(trace,lookahead,common_martingale,*,is_call=False):
    """Evaluate all fixed full/half policies using one exactly common upper."""
    c=np.asarray(lookahead['bellman_continuations']);mask=np.asarray(lookahead['bellman_computed_mask'])
    baseline=np.asarray(lookahead['baseline_continuation']);payoff=np.asarray(trace['discounted_payoffs'])[:,:,0]
    stocks=np.asarray(trace['stocks'])[:,:,0];m=np.asarray(common_martingale);strike=float(trace['strike'])
    if (c.shape!=payoff.shape+(len(ORDERS),) or mask.shape!=payoff.shape or mask.dtype.kind!='b'
            or baseline.shape!=payoff.shape or m.shape!=payoff.shape or not np.isfinite(m).all()
            or np.any(m[:,0]!=0) or not np.isfinite(c[mask]).all() or np.any(c[mask]<0)
            or not np.isfinite(baseline).all() or not np.isnan(c[~mask]).all()):
        raise ValueError('Finite aligned Bellman entries on explicit ITM mask required')
    if not isinstance(is_call,(bool,np.bool_)) or not np.isfinite(strike) or strike<=0:raise ValueError('Explicit vanilla payoff type and positive strike required')
    intrinsic=np.maximum((1 if is_call else -1)*(stocks-strike),0.)
    expected=intrinsic>0;expected[:,0]=False;expected[:,-1]=False
    if not np.array_equal(mask,expected):raise ValueError('The Bellman mask must include every ITM decision state')
    specs=bellman_candidate_specs();raw=[];stops=[];n,nt=payoff.shape
    for spec in specs.values():
        stop=np.full(n,nt-1);alive=np.ones(n,bool)
        for j in range(1,nt-1):
            if not trace['eligible'][j]:continue
            active=alive&mask[:,j]
            if not np.any(active):continue
            continuation=baseline[active,j]
            if spec['order'] is not None:
                k=ORDERS.index(spec['order']);continuation=(1-spec['blend'])*continuation+spec['blend']*c[active,j,k]
            exercise=intrinsic[active,j]>=continuation;indices=np.flatnonzero(active)[exercise]
            stop[indices]=j;alive[indices]=False
        stops.append(stop);raw.append(payoff[np.arange(n),stop])
    stops=np.column_stack(stops);raw=np.column_stack(raw);lower=raw-m[np.arange(n)[:,None],stops]
    upper=np.maximum(np.max(np.where(trace['eligible'],payoff-m,-np.inf),axis=1),payoff[:,0])
    if np.any(upper[:,None]<lower-2e-11):raise BellmanNumericalError('Common dual no longer dominates every challenge')
    return {'new_raw':raw,'new_stops':stops,'new_lower':lower,'common_upper':upper,'policy_order':list(specs)}

def select_bellman_policy(evaluated,*,selection_seed):
    values=np.asarray(evaluated['new_lower']);names=evaluated['policy_order'];rows=[];best=0.;selected=names[0];challenger=None;challenger_score=-np.inf
    if (isinstance(selection_seed,bool) or not isinstance(selection_seed,Integral) or selection_seed<0
            or values.ndim!=2 or values.shape[0]<2 or values.shape[1]!=len(names)
            or names!=list(bellman_candidate_specs()) or not np.isfinite(values).all()):
        raise ValueError('Independent identified selection stream and every finite candidate required')
    for i,name in enumerate(names):
        gain=values[:,i]-values[:,0];mean=float(gain.mean());se=float(gain.std(ddof=1)/np.sqrt(len(gain)));score=mean-1.96*se
        rows.append({'policy':name,'paired_lower_gain':mean,'paired_lower_gain_se':se,'selection_score':score,
            'exercise_change_fraction':float(np.mean(evaluated['new_stops'][:,i]!=evaluated['new_stops'][:,0])),
            'selection_seed':selection_seed,'final_paths_used_for_selection':False})
        if score>best:selected,best=name,score
        if i and score>challenger_score:challenger,challenger_score=name,score
    for row in rows:row.update(selected_before_final=row['policy']==selected,selected_challenger_before_final=row['policy']==challenger)
    return selected,challenger,pd.DataFrame(rows)

__all__=['ORDERS','BASELINE_NAME','BellmanNumericalError','BellmanEvaluator','bellman_candidate_specs',
    'evaluate_bellman_candidates','select_bellman_policy']
