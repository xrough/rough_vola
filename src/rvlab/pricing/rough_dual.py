"""Independent training of bounded exact-centered American martingale weights.

A frozen exercise policy first produces European and residual-premium
martingale traces. Deterministic coefficients on contiguous calendar bins
preserve the martingale property, including bounded multipliers measured at
the LEFT exercise state. Coefficients minimize a convex sampled
paired gap on a separate training stream. A held-out part of that stream
selects the predeclared bin count; reported valuation paths never select
coefficients or a specification. The same selected martingale controls the
primal and dual samples, retaining their pathwise ordering.

These are bounds for the declared finite kernel, integration law and exercise
calendar. They do not certify continuous American prices or stock deltas.
"""
from __future__ import annotations
from dataclasses import dataclass,replace,asdict
from numbers import Integral,Real
from types import MappingProxyType
from time import perf_counter
import math
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import expit
from .rough_simulation import RoughSimulationError


def _int(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _bounds(value,name,anchor=1.):
    try:lo,hi=value
    except (TypeError,ValueError) as exc:raise ValueError(f"{name} must contain two finite ordered bounds") from exc
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not math.isfinite(x) for x in (lo,hi)) or lo>anchor or hi<anchor or lo>hi:
        raise ValueError(f"{name} must contain finite ordered bounds including the baseline weight")
    return float(lo),float(hi)


@dataclass(frozen=True)
class RoughDualConfig:
    """Predeclared optimization budgets; no price/delta tolerance changes."""
    training_paths:int=4096
    training_seed:int=2_000_042
    candidate_bins:tuple[int,...]=(1,4,8)
    validation_fraction:float=.25
    european_bounds:tuple[float,float]=(.5,1.5)
    premium_bounds:tuple[float,float]=(0.,3.)
    objective:str='gap'
    selection_objective:str='error_budget'
    target_valuation_paths:int|None=None
    predictable_features:tuple[str,...]=()
    components:tuple[str,...]=('european','premium')
    intrinsic_bounds:tuple[float,float]=(-1.,1.)
    intrinsic_feature_bounds:tuple[float,float]=(-1.,1.)
    poststop_bounds:tuple[float,float]=(0.,1.5)
    poststop_feature_bounds:tuple[float,float]=(0.,0.)
    european_feature_bounds:tuple[float,float]=(0.,0.)
    premium_feature_bounds:tuple[float,float]=(-2.,2.)
    dispersion_penalties:tuple[float,...]=(0.,)
    max_iterations:int=100_000
    solver_tolerance:float=1e-9

    def __post_init__(self):
        _int(self.training_paths,'training_paths',4);_int(self.training_seed,'training_seed',0)
        _int(self.max_iterations,'max_iterations')
        bins=tuple(self.candidate_bins)
        if not bins or len(set(bins))!=len(bins):raise ValueError('candidate_bins must be distinct positive integers')
        for value in bins:_int(value,'candidate bin count')
        object.__setattr__(self,'candidate_bins',bins)
        for name in ('validation_fraction','solver_tolerance'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):raise ValueError(f'{name} must be finite')
        if not 0<self.validation_fraction<1:raise ValueError('validation_fraction must lie strictly between zero and one')
        if self.solver_tolerance<1e-10 or self.solver_tolerance>1e-3:raise ValueError('solver_tolerance must lie in [1e-10,1e-3]')
        object.__setattr__(self,'european_bounds',_bounds(self.european_bounds,'european_bounds'))
        object.__setattr__(self,'premium_bounds',_bounds(self.premium_bounds,'premium_bounds'))
        names=tuple(self.predictable_features)
        if len(set(names))!=len(names) or any(name not in ('moneyness','exercise_margin','itm','exercise_region',
                'soft_exercise_region','soft_exercise_region_narrow','soft_exercise_region_medium',
                'soft_exercise_region_broad','soft_exercise_region_wide','policy_stopped') for name in names):
            raise ValueError('unknown or repeated predictable feature')
        object.__setattr__(self,'predictable_features',names)
        components=tuple(self.components)
        if components not in (('european','premium'),('european','premium','intrinsic_minus_european'),
                ('european','premium','micro_boundary'),('european','premium','micro_poststop'),
                ('european','premium','micro_boundary','micro_poststop'),
                ('european','premium','intrinsic_minus_european','micro_poststop')):
            raise ValueError('unknown martingale components')
        object.__setattr__(self,'components',components)
        object.__setattr__(self,'intrinsic_bounds',_bounds(self.intrinsic_bounds,'intrinsic_bounds',0.))
        object.__setattr__(self,'intrinsic_feature_bounds',_bounds(self.intrinsic_feature_bounds,'intrinsic_feature_bounds',0.))
        object.__setattr__(self,'poststop_bounds',_bounds(self.poststop_bounds,'poststop_bounds',0.))
        object.__setattr__(self,'poststop_feature_bounds',_bounds(self.poststop_feature_bounds,'poststop_feature_bounds',0.))
        object.__setattr__(self,'european_feature_bounds',_bounds(self.european_feature_bounds,'european_feature_bounds',0.))
        object.__setattr__(self,'premium_feature_bounds',_bounds(self.premium_feature_bounds,'premium_feature_bounds',0.))
        penalties=tuple(self.dispersion_penalties)
        if (not penalties or len(set(penalties))!=len(penalties)
                or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real)
                    or not math.isfinite(x) or x<0 for x in penalties)):
            raise ValueError('dispersion_penalties must contain distinct finite nonnegative values')
        object.__setattr__(self,'dispersion_penalties',penalties)
        if self.selection_objective not in ('error_budget','gap'):
            raise ValueError('selection_objective must be error_budget or gap')
        if self.target_valuation_paths is not None:_int(self.target_valuation_paths,'target_valuation_paths',2)
        if self.objective not in ('gap','upper'):raise ValueError("objective must be 'gap' or 'upper'")


@dataclass(frozen=True)
class RoughDualWeights:
    """Frozen train-selected coefficients, with no valuation selection."""
    time_edges:np.ndarray
    weights:np.ndarray
    policy_fingerprint:str
    policy_training_seed:int
    training_seed:int
    reserved_valuation_seed:int
    config:RoughDualConfig
    diagnostics:object

    def __post_init__(self):
        for name in ('time_edges','weights'):
            value=np.asarray(getattr(self,name),float).copy();value.setflags(write=False)
            object.__setattr__(self,name,value)
        if (self.time_edges.ndim!=1 or len(self.time_edges)<2 or not np.all(np.diff(self.time_edges)>0)
                or self.weights.shape!=(len(self.time_edges)-1,len(self.config.components)*(1+len(self.config.predictable_features))) or not np.isfinite(self.weights).all()):
            raise ValueError('dual weights require ordered time edges and finite (bins,basis) coefficients')
        object.__setattr__(self,'diagnostics',MappingProxyType(dict(self.diagnostics)))


def _trace(data):
    required=('times','eligible','discounted_payoffs','european_martingale',
        'premium_martingale','stop_indices','seed','policy_training_seed','policy_fingerprint')
    if not hasattr(data,'keys') or any(key not in data for key in required):
        raise ValueError('malformed frozen-policy trace')
    times=np.asarray(data['times'])
    payoff=np.asarray(data['discounted_payoffs'])
    eligible=np.asarray(data['eligible'])
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or times[0]!=0
            or not np.all(np.diff(times)>0) or payoff.ndim!=3 or payoff.shape[0]<2
            or payoff.shape[1]!=len(times) or payoff.shape[2]<1
            or eligible.shape!=(len(times),) or eligible.dtype.kind!='b' or not eligible.any()
            or not np.isfinite(payoff).all()):raise ValueError('malformed exercise times, mask or payoff trace')
    for name in ('european_martingale','premium_martingale'):
        a=np.asarray(data[name])
        if a.shape!=payoff.shape or not np.isfinite(a).all() or np.any(a[:,0]!=0):
            raise ValueError(f'malformed {name} trace')
    stops=np.asarray(data['stop_indices'])
    if (stops.dtype.kind not in 'iu' or stops.shape!=(payoff.shape[0],payoff.shape[2])
            or np.any(stops<0) or np.any(stops>=len(times)) or not np.all(eligible[stops])):
        raise ValueError('invalid stopping indices in trace')
    for name in ('seed','policy_training_seed'):_int(data[name],name,0)
    for seed in data.get('additional_control_training_seeds',()):
        _int(seed,'additional_control_training_seed',0)
        if data['seed']==seed:raise ValueError('trace stream overlaps additional control training')
    return data


def _edges(times,bins):
    if bins>len(times)-1:raise ValueError('bin count exceeds exercise intervals')
    indices=np.rint(np.linspace(0,len(times)-1,bins+1)).astype(int)
    if np.any(np.diff(indices)<=0):raise ValueError('bin count creates empty intervals')
    return np.asarray(times,float)[indices]


def _predictable_features(data,names):
    if not names:return None
    for name in ('stocks','strike','discounts','continuation_proxy'):
        if name not in data:raise ValueError(f'predictable features require trace {name}')
    stock=np.asarray(data['stocks']);payoff=np.asarray(data['discounted_payoffs'])
    continuation=np.asarray(data['continuation_proxy']);discounts=np.asarray(data['discounts'])
    if stock.shape!=payoff.shape or continuation.shape!=payoff.shape or discounts.shape!=(payoff.shape[1],):
        raise ValueError('malformed predictable state fields')
    if not np.isfinite(stock).all() or not np.isfinite(continuation).all() or not np.all(discounts>0):
        raise ValueError('nonfinite predictable state fields')
    columns=[]
    for name in names:
        if name=='policy_stopped':
            stops=np.asarray(data['stop_indices'])
            if stops.shape!=(len(payoff),payoff.shape[2]):raise ValueError('malformed stopping indices for predictable feature')
            # A frozen policy's stopping event is already known at this LEFT
            # date. The next increment never uses its future stopping time.
            value=(np.arange(payoff.shape[1])[None,:,None]>=stops[:,None,:]).astype(float)
        elif name=='moneyness':value=np.clip((stock/float(data['strike'])-1)/.1,-2.,2.)
        elif name=='exercise_margin':value=np.clip((payoff-continuation)/(discounts[None,:,None]*.1),-2.,2.)
        elif name=='itm':value=(payoff>0).astype(float)
        elif name=='exercise_region':value=((payoff>0)&(payoff>=continuation)).astype(float)
        elif name.startswith('soft_exercise_region'):
            scale={'soft_exercise_region':.05,'soft_exercise_region_narrow':.01,
                'soft_exercise_region_medium':.03,'soft_exercise_region_broad':.1,
                'soft_exercise_region_wide':.3}[name]
            value=expit((payoff-continuation)/(discounts[None,:,None]*scale))
        columns.append(value[:,:-1])
    return np.stack(columns,axis=-1)


def _components(data,component_names):
    """Only exact-centered increments may enter a weighted martingale."""
    arrays=[data['european_martingale'],data['premium_martingale']]
    for name in component_names[2:]:
        field={'intrinsic_minus_european':'intrinsic_martingale',
            'micro_boundary':'micro_boundary_martingale',
            'micro_poststop':'micro_poststop_martingale'}[name]
        if field not in data:raise ValueError(f'{name} component requires its exact-centered microstep trace')
        value=np.asarray(data[field])
        if value.shape!=np.shape(arrays[0]) or not np.isfinite(value).all() or np.any(value[:,0]!=0):
            raise ValueError(f'malformed {name} martingale trace')
        arrays.append(value-data['european_martingale'] if name=='intrinsic_minus_european' else value)
    return arrays


def _basis(data,edges,features=(),component_names=('european','premium')):
    """Cumulative component martingales stopped at deterministic bin edges."""
    times=np.asarray(data['times']);positions=np.searchsorted(times,edges)
    if (np.any(positions>=len(times)) or not np.allclose(times[positions],edges,rtol=0,atol=2e-14)
            or edges[0]!=times[0] or edges[-1]!=times[-1]):
        raise ValueError('dual bin boundaries must be present in the valuation exercise calendar')
    arrays=_components(data,component_names)
    components=np.stack(arrays,axis=-1)
    predictable=_predictable_features(data,features)
    if predictable is None:cumulative=components
    else:
        # Each bounded multiplier is measured at the LEFT exercise date,
        # before every increment it multiplies. Hence no new centering
        # approximation is introduced by a nonlinear state feature.
        phi=np.concatenate((np.ones(predictable.shape[:-1]+(1,)),predictable),axis=-1)
        increments=np.diff(components,axis=1)[...,None,:]*phi[...,None]
        increments=increments.reshape(increments.shape[:-2]+(-1,))
        cumulative=np.concatenate((np.zeros(increments.shape[:1]+(1,)+increments.shape[2:]),
            np.cumsum(increments,axis=1)),axis=1)
    perbin=len(component_names)*(1+len(features))
    result=np.zeros(cumulative.shape[:-1]+(perbin*(len(edges)-1),))
    for b,(left,right) in enumerate(zip(positions[:-1],positions[1:])):
        indices=np.clip(np.arange(len(times)),left,right)
        result[...,perbin*b:perbin*(b+1)]=cumulative[:,indices]-cumulative[:,left:left+1]
    return result


def _summary(a):
    return float(np.mean(a)),float(np.std(a,ddof=1)/math.sqrt(len(a)))


def _endpoints(payoff,martingale,stops,eligible):
    paths=np.arange(len(payoff))[:,None];bumps=np.arange(payoff.shape[2])[None,:]
    raw=payoff[paths,stops,bumps]
    primal=raw-martingale[paths,stops,bumps]
    dual=np.max(np.where(eligible[None,:,None],payoff-martingale,-np.inf),axis=1)
    return primal,dual,raw


def _fit_lp(payoff,basis,stops,eligible,config,dispersion_penalty=0.):
    """Sparse epigraph LP, centered at the fixed stopping cashflow for gap fits."""
    n,m,d=basis.shape
    active=np.flatnonzero(eligible)
    features=basis[:,active]
    targets=payoff[:,active]
    if config.objective=='gap':
        features=features-basis[np.arange(n),stops,None,:]
        targets=targets-payoff[np.arange(n),stops,None]
    rows=n*len(active)
    features=features.reshape(rows,d);targets=targets.ravel()
    # Work around unit weights. The LP includes this feasible baseline and
    # preserves every row, including the path's exact zero stopping gap.
    perbin=len(config.components)*(1+len(config.predictable_features))
    anchor=np.tile(np.r_[1.,1.,np.zeros(perbin-2)],d//perbin)
    baseline=targets-features@anchor
    a_weights=sparse.csr_matrix(-features)
    row=np.arange(rows);column=np.repeat(np.arange(n),len(active))
    a_epigraph=sparse.csr_matrix((-np.ones(rows),(row,column)),shape=(rows,n))
    matrix=sparse.hstack((a_weights,a_epigraph),format='csr')
    bounds=[]
    for j in range(d):
        offset=j%perbin;component=offset%len(config.components);feature=offset//len(config.components)
        name=config.components[component]
        if feature==0:
            lo,hi={'european':config.european_bounds,'premium':config.premium_bounds,
                'intrinsic_minus_european':config.intrinsic_bounds,'micro_boundary':config.intrinsic_bounds,
                'micro_poststop':config.poststop_bounds}[name]
        else:
            lo,hi={'european':config.european_feature_bounds,'premium':config.premium_feature_bounds,
                'intrinsic_minus_european':config.intrinsic_feature_bounds,'micro_boundary':config.intrinsic_feature_bounds,
                'micro_poststop':config.poststop_feature_bounds}[name]
        bounds.append((lo-anchor[j],hi-anchor[j]))
    bounds+=([(0.,None)] if config.objective=='gap' else [(None,None)])*n
    objective=np.r_[np.zeros(d),np.full(n,1/n)];right=-baseline
    if dispersion_penalty:
        # Convex L1 dispersion surrogate, jointly centered on fitting paths.
        # It controls the variance inflation caused by rich martingale bases.
        # Actual held-out and valuation gates still use unchanged standard
        # errors and price/delta tolerances; this is only a fitting penalty.
        stopped_basis=basis[np.arange(n),stops]
        stopped_payoff=payoff[np.arange(n),stops]
        stopped_baseline=stopped_payoff-stopped_basis@anchor
        matrix=sparse.hstack((matrix,sparse.csr_matrix((rows,n+1))),format='csr')
        eye=sparse.eye(n,format='csr');zero=sparse.csr_matrix((n,n))
        one=sparse.csr_matrix(np.ones((n,1)))
        lower=sparse.hstack((-sparse.csr_matrix(stopped_basis),zero,-eye,-one),format='csr')
        upper=sparse.hstack((sparse.csr_matrix(stopped_basis),zero,-eye,one),format='csr')
        matrix=sparse.vstack((matrix,lower,upper),format='csr')
        right=np.r_[right,-stopped_baseline,stopped_baseline]
        objective=np.r_[objective,np.full(n,dispersion_penalty/n),0.]
        bounds+=[(0.,None)]*n+[(None,None)]
    result=linprog(objective,A_ub=matrix,b_ub=right,
        bounds=bounds,method='highs',options={'maxiter':config.max_iterations,
            'dual_feasibility_tolerance':config.solver_tolerance,
            'primal_feasibility_tolerance':config.solver_tolerance,
            'ipm_optimality_tolerance':config.solver_tolerance})
    record={'optimizer_success':bool(result.success),'optimizer_status':int(result.status),
        'optimizer_message':str(result.message),'iterations':int(result.nit),
        'constraint_count':int(matrix.shape[0]),'variable_count':int(matrix.shape[1]),
        'matrix_nonzeros':int(matrix.nnz),'dispersion_penalty':float(dispersion_penalty)}
    if result.x is None:return None,record
    weights=anchor+np.asarray(result.x[:d])
    actual=np.max(targets.reshape(n,-1)-np.einsum('nmd,d->nm',features.reshape(n,-1,d),weights),axis=1)
    epigraph=np.asarray(result.x[d:d+n])
    violation=float(np.max(actual-epigraph))
    record.update({'maximum_constraint_violation':violation,'sampled_objective':float(np.mean(actual)),
        'solver_objective':float(result.fun),'fixed_bounds':[j for j,(lo,hi) in enumerate(bounds[:d]) if lo==hi],
        'active_bounds':[j for j,(lo,hi) in enumerate(bounds[:d])
            if lo<hi and min(abs(result.x[j]-lo),abs(result.x[j]-hi))<1e-7]})
    if dispersion_penalty:
        record['fitted_stopped_cashflow_center']=float(result.x[-1])
        record['fitted_stopped_cashflow_absolute_deviation']=float(np.mean(result.x[d+n:d+2*n]))
    if (not result.success or not np.isfinite(weights).all()
            or violation>max(2e-8,20*config.solver_tolerance)):
        return None,record
    return weights,record


def _selection_record(primal,dual,nfit,target_paths):
    p=np.asarray(primal).reshape(-1);d=np.asarray(dual).reshape(-1);gap=d-p
    count=len(p)-nfit
    gap_se=float(np.std(gap[nfit:],ddof=1)/math.sqrt(target_paths))
    primal_se=float(np.std(p[nfit:],ddof=1)/math.sqrt(target_paths))
    return {'fit_gap':float(np.mean(gap[:nfit])),'heldout_gap':float(np.mean(gap[nfit:])),
        'fit_upper':float(np.mean(d[:nfit])),'heldout_upper':float(np.mean(d[nfit:])),
        'heldout_primal':float(np.mean(p[nfit:])),
        'projected_gap_standard_error':gap_se,'projected_primal_standard_error':primal_se,
        'heldout_error_budget':float(np.mean(gap[nfit:]))+1.96*(gap_se+primal_se),
        'heldout_paths':count,'target_valuation_paths':int(target_paths)}


def _dual_workspace_estimate(data,nfit,dimension,regularized=False):
    """Conservative preallocation estimate, including sparse-solver work copies.

    Sparse factorization memory is problem dependent; this guard rejects
    oversized declared attempts before constructing their dense basis or LP.
    It is a budget estimate rather than a hard allocator-level ceiling.
    """
    payoff=np.asarray(data['discounted_payoffs']);n,m,bumps=payoff.shape
    trace_bytes=sum(value.nbytes for value in data.values() if isinstance(value,np.ndarray))
    rows=nfit*int(np.sum(data['eligible']))
    basis_bytes=8*n*m*bumps*dimension
    feature_bytes=8*rows*dimension
    csr_bytes=12*rows*(dimension+1)+4*(rows+1)
    extra=6*(24*nfit*(dimension+3)+8*(nfit+1)) if regularized else 0
    return int(trace_bytes+2*basis_bytes+2*feature_bytes+6*csr_bytes+64*(nfit+dimension)+extra)


def train_rough_dual_weights(policy,*,config=None,training_trace=None):
    """Fit/select martingale coefficients using a third independent stream."""
    from .rough_stopping_fourier import (FourierStoppingPolicy,
        fourier_policy_fingerprint,value_fourier_stopping_policy)
    started=perf_counter()
    if not isinstance(policy,FourierStoppingPolicy):raise TypeError('dual training requires FourierStoppingPolicy')
    config=RoughDualConfig() if config is None else config
    if not isinstance(config,RoughDualConfig):raise TypeError('config must be RoughDualConfig')
    if config.training_seed in (policy.config.training_seed,policy.config.valuation_seed):
        raise ValueError('dual-training seed must differ from policy-training and reserved valuation seeds')
    identity=fourier_policy_fingerprint(policy)
    if training_trace is None:
        changes={"valuation_paths":config.training_paths,"valuation_seed":config.training_seed}
        if hasattr(policy.config,"dual_config"):changes["dual_config"]=None
        numerical=replace(policy.config,**changes)
        training_trace=value_fourier_stopping_policy(policy,config=numerical,trace=True)['trace']
    data=_trace(training_trace)
    if (data['seed']!=config.training_seed or data['policy_fingerprint']!=identity
            or len(data['discounted_payoffs'])!=config.training_paths):
        raise ValueError('dual-training trace does not match declared policy, seed and paths')
    if data['discounted_payoffs'].shape[2]!=1:raise ValueError('dual coefficients train at the single declared inception spot')
    n=len(data['discounted_payoffs']);nfit=int(math.floor(n*(1-config.validation_fraction)))
    if min(nfit,n-nfit)<2:raise ValueError('dual fitting and held-out training subsets need at least two paths each')
    payoff=np.asarray(data['discounted_payoffs'])[:,:,0];stops=np.asarray(data['stop_indices'])[:,0]
    eligible=np.asarray(data['eligible']);base_m=data['european_martingale'][:,:,0]+data['premium_martingale'][:,:,0]
    basep,based,_=_endpoints(payoff[:,:,None],base_m[:,:,None],stops[:,None],eligible)
    basegap=(based-basep)[:,0]
    # Unit weights remain an explicit candidate, selected only on held-out
    # dual-training paths. No reported valuation outcome enters selection.
    selected_edges=_edges(data['times'],1);selected=np.r_[1.,1.,np.zeros(len(config.components)*(1+len(config.predictable_features))-2)]
    target_paths=policy.config.valuation_paths if config.target_valuation_paths is None else config.target_valuation_paths
    score_key='heldout_error_budget' if config.selection_objective=='error_budget' else 'heldout_gap'
    baseline_record=_selection_record(basep,based,nfit,target_paths)
    selected_score=baseline_record[score_key];selected_name='unit baseline'
    records=[{'candidate':'unit baseline','bins':1,'status':'baseline',**baseline_record}]
    memory_budget_bytes=float(policy.config.memory_budget_mb)*1024**2
    for bins,penalty in ((b,p) for b in config.candidate_bins for p in config.dispersion_penalties):
        begin=perf_counter();projected_memory=None
        candidate=f'{bins} bins' if len(config.dispersion_penalties)==1 and penalty==0 else f'{bins} bins; dispersion={penalty:g}'
        try:
            edges=_edges(data['times'],bins)
            dimension=bins*len(config.components)*(1+len(config.predictable_features))
            projected_memory=_dual_workspace_estimate(data,nfit,dimension,regularized=bool(penalty))
            if projected_memory>memory_budget_bytes:
                raise MemoryError(f'dual LP candidate requires approximately {projected_memory/1024**2:.2f} MiB, exceeding memory_budget_mb={policy.config.memory_budget_mb}')
            basis=_basis(data,edges,config.predictable_features,config.components)[:,:,0]
            weights,record=_fit_lp(payoff[:nfit],basis[:nfit],stops[:nfit],eligible,config,penalty)
            record.update({'candidate':candidate,'bins':bins,'elapsed_seconds':perf_counter()-begin,
                'projected_memory_bytes':projected_memory,'memory_budget_bytes':memory_budget_bytes})
            if weights is None:
                record['status']='optimizer_failure';records.append(record);continue
            m=np.einsum('nmd,d->nm',basis,weights)
            primal,dual,_=_endpoints(payoff[:,:,None],m[:,:,None],stops[:,None],eligible)
            gap=(dual-primal)[:,0]
            record.update({'status':'completed','weights':weights.reshape(bins,-1).tolist(),
                **_selection_record(primal,dual,nfit,target_paths)})
            records.append(record)
            score=record[score_key]
            if score<selected_score:
                selected_score,selected,selected_edges,selected_name=score,weights,edges,record['candidate']
        except (ValueError,MemoryError,RoughSimulationError) as error:
            records.append({'candidate':candidate,'bins':bins,'dispersion_penalty':penalty,'status':'numerical_failure',
                'error':str(error),'elapsed_seconds':perf_counter()-begin,
                'projected_memory_bytes':projected_memory,'memory_budget_bytes':memory_budget_bytes})
    diagnostics={'policy_frozen':True,'policy_training_seed':policy.config.training_seed,
        'dual_training_seed':config.training_seed,'reserved_valuation_seed':policy.config.valuation_seed,
        'independent_policy_dual_valuation_streams':True,'fit_paths':nfit,'heldout_training_paths':n-nfit,
        'selection_source':'held-out dual-training paths only','selection_objective':config.selection_objective,
        'target_valuation_paths':target_paths,'selected_score':selected_score,
        'selected_candidate':selected_name,'candidates':records,'baseline_training_gap':_summary(basegap),
        'elapsed_seconds':perf_counter()-started,'config':asdict(config),'predictable_features':list(config.predictable_features),
        'feature_scales':{'moneyness':.1,'exercise_margin':.1,'clipping':[-2.,2.],
            'soft_exercise_region':.05,'soft_exercise_region_narrow':.01,
            'soft_exercise_region_medium':.03,'soft_exercise_region_broad':.1,'soft_exercise_region_wide':.3},
        'dispersion_penalty_scope':'LP fitting surrogate only; actual unchanged standard errors select on training holdout',
        'components':list(config.components),
        'basis_order':'each calendar bin: base components, then the same components multiplied by each LEFT-state feature',
        'basis_labels':list(config.components)+[f'{component}*{feature}' for feature in config.predictable_features for component in config.components],
        'memory_budget_bytes':memory_budget_bytes,
        'bound_scope':'finite kernel, discrete transition law and exercise calendar',
        'american_pricing_ready':False,'stock_delta_ready':False}
    return RoughDualWeights(selected_edges,selected.reshape(len(selected_edges)-1,-1),identity,
        policy.config.training_seed,config.training_seed,policy.config.valuation_seed,config,diagnostics)


def _weighted_martingale(data,fit):
    """Apply coefficients without allocating a path×time×bump×basis tensor."""
    times=np.asarray(data['times']);edges=fit.time_edges
    positions=np.searchsorted(times,edges)
    if (np.any(positions>=len(times)) or not np.allclose(times[positions],edges,rtol=0,atol=2e-14)
            or edges[0]!=times[0] or edges[-1]!=times[-1]):
        raise ValueError('dual bin boundaries must be present in the valuation exercise calendar')
    arrays=_components(data,fit.config.components)
    component=np.stack(arrays,axis=-1)
    increments=np.diff(component,axis=1)
    bins=np.searchsorted(edges[1:],times[1:],side='left')
    coefficients=fit.weights.reshape(len(edges)-1,1+len(fit.config.predictable_features),len(arrays))
    features=_predictable_features(data,fit.config.predictable_features)
    total=np.zeros(increments.shape[:-1])
    for feature in range(1+len(fit.config.predictable_features)):
        value=np.einsum('ntbc,tc->ntb',increments,coefficients[bins,feature,:])
        if feature:value*=features[...,feature-1]
        total+=value
    result=np.concatenate((np.zeros(total.shape[:1]+(1,)+total.shape[2:]),np.cumsum(total,axis=1)),axis=1)
    if not np.isfinite(result).all():raise RoughSimulationError('weighted martingale overflowed')
    return result


def apply_rough_dual_weights(data,fit):
    """Apply one frozen train-selected martingale to both endpoints on new paths."""
    if not isinstance(fit,RoughDualWeights):raise TypeError('fit must be RoughDualWeights')
    data=_trace(data)
    if data['seed'] in (fit.training_seed,fit.policy_training_seed):
        raise ValueError('valuation seed must differ from policy-training and dual-training seeds')
    if data['policy_fingerprint']!=fit.policy_fingerprint:raise ValueError('dual weights belong to another frozen policy')
    weighted=_weighted_martingale(data,fit)
    payoff=np.asarray(data['discounted_payoffs']);stops=np.asarray(data['stop_indices']);eligible=np.asarray(data['eligible'])
    primal,dual,raw=_endpoints(payoff,weighted,stops,eligible)
    baseline_m=data['european_martingale']+data['premium_martingale']
    basep,based,_=_endpoints(payoff,baseline_m,stops,eligible)
    if np.any(dual<primal-2e-11):raise RoughSimulationError('weighted martingale violated primal-dual ordering')
    for array in (primal,dual,raw,weighted):array.setflags(write=False)
    return {'primal_samples':primal,'dual_samples':dual,'raw_samples':raw,
        'martingale':weighted,'baseline_primal_samples':basep,'baseline_dual_samples':based,
        'diagnostics':{'policy_frozen':True,'dual_weights_frozen':True,'valuation_seed':int(data['seed']),
            'policy_training_seed':fit.policy_training_seed,'dual_training_seed':fit.training_seed,
            'selection_source':'held-out dual-training paths only','same_martingale_primal_dual':True,
            'selected_candidate':fit.diagnostics['selected_candidate'],'weights':fit.weights.tolist(),
            'time_edges':fit.time_edges.tolist(),'american_pricing_ready':False,'stock_delta_ready':False}}


def value_rough_dual_policy(policy,fit,*,config=None,spots=None,trace=False):
    """Independent valuation of a frozen policy and frozen train-only weights."""
    from .rough_stopping_fourier import (FourierStoppingPolicy,
        fourier_policy_fingerprint,value_fourier_stopping_policy)
    if not isinstance(policy,FourierStoppingPolicy):raise TypeError('policy must be FourierStoppingPolicy')
    if fourier_policy_fingerprint(policy)!=fit.policy_fingerprint:raise ValueError('dual weights belong to another frozen policy')
    result=value_fourier_stopping_policy(policy,config=config,spots=spots,trace=True)
    weighted=apply_rough_dual_weights(result['trace'],fit)
    weighted.update({'stop_times':result['stop_times'],'trace':result['trace'] if trace else None})
    return weighted


__all__=['RoughDualConfig','RoughDualWeights','train_rough_dual_weights',
    'apply_rough_dual_weights','value_rough_dual_policy']
