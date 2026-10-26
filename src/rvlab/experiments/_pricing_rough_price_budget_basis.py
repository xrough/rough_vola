"""Independent predictable stopped-increment basis reconstruction.

Training normalizations are the pre-existing frozen policy's date-specific
factor coordinates. No reported path is clipped, dropped or recentered; only
the declared left-state approximation gates are bounded.
"""
from numbers import Integral
import numpy as np

PROCESS_ORDER=('target0 minus latest','target1 minus latest','target2 minus latest','target3 minus latest',
    'european','premium','intrinsic','micro boundary')


def reconstruct_predictable_basis(arrays,bundle,*,bins=4,stock_width=.05,gate_clip=3.):
    """Sum g(left)*dM over right_index<=tau, then flatten bin/process/gate."""
    if (isinstance(bins,bool) or not isinstance(bins,Integral) or bins!=4
            or stock_width!=.05 or gate_clip!=3.):
        raise ValueError('The declared common-control basis uses four bins, stock width.05 and gate cap3')
    proxy=bundle['proxy'];selected=bundle['selected_policy'];stocks=np.asarray(arrays['stocks'])
    factors=np.asarray(arrays['factors']);latest=np.asarray(arrays['latest_martingale'])
    targets=np.asarray(arrays['target_martingales']);stops=np.asarray(arrays['gh16_stops'])
    count=len(proxy.kernel.nodes);times=np.asarray(arrays['times'])
    if (stocks.ndim!=3 or stocks.shape[1:]!=(65,1) or factors.shape!=(len(stocks),65,count)
            or latest.shape!=(len(stocks),65) or targets.shape!=(len(stocks),65,4)
            or stops.shape!=(len(stocks),) or stops.dtype.kind not in 'iu'
            or np.any(stops<0) or np.any(stops>64)
            or not np.array_equal(times,selected.times) or not np.array_equal(times,proxy.exercise_times)
            or np.any(stocks<=0) or not all(np.isfinite(x).all() for x in (stocks,factors,latest,targets))):
        raise ValueError('Full fixed-calendar states, four targets and own stopping times required')
    components=[targets[:,:,i]-latest for i in range(4)]
    for name in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale'):
        value=np.asarray(arrays[name])
        if value.shape!=latest.shape or not np.isfinite(value).all():
            raise ValueError('Every declared martingale must retain all path/date values')
        components.append(value)
    process=np.stack(components,axis=2)
    if np.any(process[:,0]!=0):raise ValueError('Predictable basis processes must begin at zero')
    ng=2+count;result=np.zeros((len(stocks),bins,8,ng));names=[]
    for left in range(64):
        model=selected.continuations[left];center=np.asarray(model.center);scale=np.asarray(model.scale)
        if (center.shape!=(count,) or scale.shape!=(count,) or not np.isfinite(center).all()
                or not np.isfinite(scale).all() or np.any(scale<=0)):
            raise ValueError('Use the fixed positive training normalization at each left date')
        gate=np.column_stack((np.ones(len(stocks)),
            np.clip((stocks[:,left,0]/proxy.strike-1.)/stock_width,-gate_clip,gate_clip),
            np.clip((factors[:,left]-center)/scale,-gate_clip,gate_clip)))
        # {tau >= left+1} is known at the left date. Using {tau > left+1}
        # would remove the stopping increment and generally destroy centering.
        survival=(stops>left).astype(float)
        increment=process[:,left+1]-process[:,left]
        result[:,left//16]+=increment[:,:,None]*gate[:,None,:]*survival[:,None,None]
    for b in range(bins):
        for process_name in PROCESS_ORDER:
            for gate_name in ('constant','stock',*[f'factor{i}' for i in range(count)]):
                names.append(f'bin{b}/{process_name}/{gate_name}')
    flat=result.reshape(len(stocks),-1)
    if not np.isfinite(flat).all():raise ValueError('Predictable common-control features are nonfinite')
    return flat,{'column_order':names,'process_order':list(PROCESS_ORDER),'bins':bins,
        'gates':2+count,'columns':flat.shape[1],'survival':'right_index <= own stopping index',
        'normalization':'already frozen selected continuation center/scale at each LEFT date',
        'states_deleted_or_clipped':False,'empirical_valuation_mean_subtracted':False}


__all__=['PROCESS_ORDER','reconstruct_predictable_basis']
