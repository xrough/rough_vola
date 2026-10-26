"""Predictable stopped controls and training-only bounded regression weights."""
import numpy as np

PROCESS_NAMES=('target0-minus-latest','target1-minus-latest','target2-minus-latest','target3-minus-latest',
    'european','premium','intrinsic','micro_boundary')


def stopped_basis(processes,stops,bins=4,gates=None):
    processes=np.asarray(processes,float);stops=np.asarray(stops)
    if (processes.ndim!=3 or processes.shape[1]!=65 or stops.shape!=(len(processes),)
            or stops.dtype.kind not in 'iu' or np.any(stops<0) or np.any(stops>64) or bins!=4
            or not np.isfinite(processes).all() or np.any(processes[:,0]!=0)):
        raise ValueError('Full zero-origin martingales and own bounded stopping dates required')
    if gates is None:gates=np.ones((len(processes),64,1))
    gates=np.asarray(gates,float)
    if gates.shape[:2]!=(len(processes),64) or not np.isfinite(gates).all():raise ValueError('Finite aligned LEFT-state features required')
    increment=np.diff(processes,axis=1); alive=np.arange(1,65)[None,:]<=stops[:,None]
    return np.concatenate([np.einsum('ntp,ntg,nt->npg',increment[:,j:j+16],gates[:,j:j+16],alive[:,j:j+16]).reshape(len(processes),-1)
        for j in range(0,64,16)],axis=1)


def predictable_basis(arrays,bundle,*,include_anchor=False,anchor=None):
    """Column order is bin, process, left gate; optional anchor is last.

    Gates use the already frozen per-date selected-policy factor center/scale;
    only declared feature values are clipped, never simulated states or rows.
    """
    selected=bundle['selected_policy'];proxy=bundle['proxy'];stock=np.asarray(arrays['stocks'])[:,:,0]
    u=np.asarray(arrays['factors']);n=len(stock)
    if (stock.shape!=(n,65) or u.shape!=(n,65,len(proxy.kernel.nodes)) or not np.isfinite(stock).all()
            or np.any(stock<=0) or not np.isfinite(u).all()):raise ValueError('Complete stock/factor states required')
    r=np.stack([(u[:,j]-selected.continuations[j].center)/selected.continuations[j].scale for j in range(64)],axis=1)
    gates=np.concatenate((np.ones((n,64,1)),np.clip((stock[:,:-1,None]/proxy.strike-1)/.05,-3,3),np.clip(r,-3,3)),axis=-1)
    processes=np.concatenate((arrays['target_martingales']-arrays['latest_martingale'][:,:,None],
        np.stack([arrays[k] for k in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale')],axis=-1)),axis=-1)
    basis=stopped_basis(processes,arrays['gh16_stops'],gates=gates)
    names=[f'bin{b}|{process}|{gate}' for b in range(4) for process in PROCESS_NAMES
        for gate in ['one','stock']+[f'factor{i}' for i in range(len(proxy.kernel.nodes))]]
    if include_anchor:
        anchor=np.asarray(anchor,float)
        if anchor.shape!=(n,) or not np.isfinite(anchor).all():raise ValueError('A complete exactly centered European anchor required')
        basis=np.column_stack((basis,anchor));names.append('paired-European-anchor')
    return basis,names


def fit_weights(basis,lower,*,ridge,bound=2.):
    """Ridge on centered TRAINING data, then fixed raw-coefficient projection.

    Center/scale statistics determine coefficients only; valuation applies
    raw features with no intercept and no empirical mean subtraction.
    """
    x=np.asarray(basis,float);y=np.asarray(lower,float)
    if (x.ndim!=2 or y.shape!=(len(x),) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(ridge) or ridge<=0 or bound!=2.):raise ValueError('Finite training samples, positive ridge and declared coefficientbound2 required')
    center=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale<1e-12,1.,scale)
    xc=(x-center)/scale
    raw=np.linalg.solve(xc.T@xc/len(x)+ridge*np.eye(x.shape[1]),xc.T@(y-y.mean())/len(x))/scale
    coefficient=np.clip(raw,-bound,bound)
    return coefficient,{'ridge':ridge,'bound':bound,'training_rows':len(x),'features':x.shape[1],
        'max_abs_unprojected_coefficient':float(np.max(abs(raw))),'active_projection_count':int(np.sum(abs(raw)>bound)),
        'training_residual_standard_deviation':float(np.std(y-x@coefficient,ddof=1)),
        'fit_center':center,'fit_scale':scale,'fit_intercept_applied_at_valuation':False}


def common_shift(lower,upper,basis,weights):
    x=np.asarray(basis,float);w=np.asarray(weights,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    if (x.ndim!=2 or w.shape!=(x.shape[1],) or lower.shape!=(len(x),) or upper.shape!=lower.shape
            or not all(np.isfinite(v).all() for v in (x,w,lower,upper)) or np.any(abs(w)>2)
            or np.any(upper<lower-2e-11)):
        raise ValueError('Finite full-path endpoints and frozen bounded coefficients required')
    shift=x@w
    return lower-shift,upper-shift,shift
