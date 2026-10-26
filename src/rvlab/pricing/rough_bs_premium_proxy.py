"""Structural American-BS premium proxies for finite-lift martingale controls.

The proxy uses the full factor state's expected remaining integrated variance
to interpolate deterministic BS stopping-premium tables. Its stock interpolant
is piecewise linear, so the existing conditional lognormal integration remains
exact for the proxy actually evaluated. This is a variance-reduction function,
not a rough-American price, conditional-state API or exercise-policy fit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
import numpy as np

from .black_scholes import black_scholes_price
from .rough_estimators import _AffineSurrogate
from .rough_simulation import _nonnegative_clock
from .rough_stopping_fourier import FourierStoppingPolicy
from .rough_stopping_reference import black_scholes_stopping_surface
from .types import BlackScholesParams


DEFAULT_PROXY_VOLS = (.05,.1,.15,.2,.25,.3,.4,.5,.65,.8,1.)
DEFAULT_PROXY_STOCK_KNOTS = (.6,.7,.75,.8,.85,.875,.9,.925,.95,.975,
    1.,1.025,1.05,1.1,1.2,1.5,2.)


def _readonly(values):
    array=np.asarray(values,float).copy();array.setflags(write=False);return array


@dataclass(frozen=True)
class ConditionalBSPremium:
    """Immutable exact-in-stock proxy coefficients at one calendar date."""
    stock_nodes: np.ndarray
    variance_nodes: np.ndarray
    nodal_premiums: np.ndarray
    remaining_integral_coefficient: np.ndarray
    remaining_integral_constant: float
    remaining_time: float

    def __post_init__(self):
        for name in ("stock_nodes","variance_nodes","nodal_premiums","remaining_integral_coefficient"):
            object.__setattr__(self,name,_readonly(getattr(self,name)))
        if (self.stock_nodes.ndim!=1 or len(self.stock_nodes)<2 or self.stock_nodes[0]!=0
                or not np.isfinite(self.stock_nodes).all() or np.any(np.diff(self.stock_nodes)<=0) or self.variance_nodes.ndim!=1
                or len(self.variance_nodes)<2 or np.any(np.diff(self.variance_nodes)<=0)
                or not np.isfinite(self.variance_nodes).all() or self.variance_nodes[0]<0
                or self.nodal_premiums.shape!=(len(self.variance_nodes),len(self.stock_nodes))
                or not np.isfinite(self.nodal_premiums).all() or np.any(self.nodal_premiums<0)
                or self.remaining_integral_coefficient.ndim!=1
                or not np.isfinite(self.remaining_integral_coefficient).all()
                or not np.isfinite(self.remaining_integral_constant)
                or not np.isfinite(self.remaining_time) or self.remaining_time<0):
            raise ValueError("malformed conditional-BS premium table")

    def average_variance(self,factors):
        factors=np.asarray(factors,float)
        if factors.ndim<1 or factors.shape[-1]!=len(self.remaining_integral_coefficient) or not np.isfinite(factors).all():
            raise ValueError("factor coordinates do not align with the BS premium proxy")
        if self.remaining_time==0:
            return np.zeros(factors.shape[:-1])
        return _nonnegative_clock(factors@self.remaining_integral_coefficient+
            self.remaining_integral_constant)/self.remaining_time

    def coefficients(self,factors):
        variance=self.average_variance(factors)
        if self.remaining_time==0:
            return np.zeros(variance.shape+(len(self.stock_nodes),3))
        clamped=np.clip(variance,self.variance_nodes[0],self.variance_nodes[-1])
        index=np.clip(np.searchsorted(self.variance_nodes,clamped,side="right")-1,
            0,len(self.variance_nodes)-2)
        weight=(clamped-self.variance_nodes[index])/(self.variance_nodes[index+1]-self.variance_nodes[index])
        premium=(1-weight[...,None])*self.nodal_premiums[index]+weight[...,None]*self.nodal_premiums[index+1]
        slope=np.diff(premium,axis=-1)/np.diff(self.stock_nodes)
        intercept=premium[...,:-1]-slope*self.stock_nodes[:-1]
        # A constant far-stock tail avoids unbounded linear extrapolation.
        slope=np.concatenate((slope,np.zeros(variance.shape+(1,))),axis=-1)
        intercept=np.concatenate((intercept,premium[...,-1:]),axis=-1)
        return np.stack((intercept,slope,np.zeros_like(slope)),axis=-1)

    def range_diagnostics(self,factors):
        variance=self.average_variance(factors)
        if self.remaining_time==0:
            return {"states":0,"below_variance_grid":0,"above_variance_grid":0}
        return {"states":int(variance.size),
            "below_variance_grid":int(np.sum(variance<self.variance_nodes[0])),
            "above_variance_grid":int(np.sum(variance>self.variance_nodes[-1])),
            "minimum_average_variance":float(np.min(variance)),
            "maximum_average_variance":float(np.max(variance))}


def build_bs_premium_proxy(policy, *, volatilities=DEFAULT_PROXY_VOLS,
                           stock_knots=DEFAULT_PROXY_STOCK_KNOTS,
                           spot_nodes=4097, log_half_width=2.):
    """Replace only control proxies; keep the original exercise decisions.

    The deterministic table is evaluated on the policy's finite exercise
    calendar. A table or variance-range limitation is reported, never treated
    as rough-model convergence or a change in acceptance tolerances.
    """
    if not isinstance(policy,FourierStoppingPolicy):
        raise TypeError("BS premium proxy requires FourierStoppingPolicy")
    if not np.array_equal(policy.continuation_times,policy.exercise_times):
        raise ValueError("BS premium proxy currently requires continuation dates on the exercise calendar")
    if any(e.amount>0 and e.time<=policy.maturity for e in policy.market.dividends.events):
        raise ValueError("BS premium proxy requires an explicitly empty cash schedule")
    vols=np.asarray(volatilities)
    knots=np.asarray(stock_knots)
    if (vols.dtype.kind not in "iuf" or vols.ndim!=1 or len(vols)<2
            or not np.isfinite(vols).all() or np.any(vols<=0) or np.any(np.diff(vols)<=0)):
        raise ValueError("proxy volatilities must be increasing finite positive values")
    if (knots.dtype.kind not in "iuf" or knots.ndim!=1 or len(knots)==0
            or not np.isfinite(knots).all() or np.any(knots<=0) or np.any(np.diff(knots)<=0)):
        raise ValueError("proxy stock knots must be increasing finite positive values")
    times=policy.exercise_times
    normalized=np.r_[0.,knots.astype(float)]
    stocks=policy.strike*normalized
    tables=np.empty((len(times),len(vols),len(stocks)))
    negative_count=0;minimum_premium=0.;substantial_negative_count=0
    discounts=policy.market.discount_curve.discount(times)
    for k,volatility in enumerate(vols):
        surface=black_scholes_stopping_surface(policy.market,volatility=float(volatility),
            strike=policy.strike,exercise_times=times,is_call=policy.is_call,
            spot_nodes=spot_nodes,log_half_width=log_half_width,exercise_style=policy.exercise_style)
        if stocks[1]<surface['stocks'][0] or stocks[-1]>surface['stocks'][-1]:
            raise ValueError("BS proxy stock nodes extend outside the reference domain; increase log_half_width")
        for j,time in enumerate(times):
            remaining=policy.maturity-time;discount=discounts[-1]/discounts[j]
            american=np.r_[0.,np.interp(stocks[1:],surface['stocks'],surface['values'][j])]
            if not policy.is_call:
                american[0]=policy.strike*(np.max(discounts[j:])/discounts[j]
                    if policy.exercise_style=='american' else discount)
            european=np.zeros(len(stocks))
            if not policy.is_call:european[0]=policy.strike*discount
            european[1:]=black_scholes_price(BlackScholesParams(float(volatility)),
                forward=stocks[1:]/discount,discount=discount,strike=policy.strike,
                maturity=remaining,is_call=policy.is_call)
            premiums=american-european
            negative_count+=int(np.sum(premiums<0))
            minimum_premium=min(minimum_premium,float(np.min(premiums)))
            # Record all negative interpolation/table differences explicitly.
            # Nonnegative clipping defines this arbitrary control proxy only.
            substantial_negative_count+=int(np.sum(premiums<-(.001+1e-4*np.abs(american))))
            tables[j,k]=np.maximum(premiums,0.)
    affine=_AffineSurrogate(policy.params,policy.market,policy.kernel,np.array([policy.strike]),
        np.array([policy.maturity]),np.array([policy.is_call]),times,policy.control_config)
    models=[]
    for j,time in enumerate(times):
        coefficient,constant=affine._remaining_integral(float(time),policy.maturity)
        models.append(ConditionalBSPremium(normalized,vols.astype(float)**2,tables[j],
            coefficient,constant,policy.maturity-float(time)))
    diagnostics=dict(policy.diagnostics)
    diagnostics.update(control_proxy_method='BS finite-calendar American premium at full-state conditional mean remaining variance',
        exercise_policy_unchanged=True,proxy_volatilities=vols.tolist(),
        proxy_variance_interpolation='linear in variance; range clipping defines proxy only',
        proxy_stock_interpolation='piecewise linear in S/K with a constant upper-stock tail',
        proxy_stock_knots=knots.tolist(),proxy_table_spot_nodes=int(spot_nodes),
        proxy_table_log_half_width=float(log_half_width),proxy_negative_node_count=negative_count,
        proxy_minimum_unclipped_node_premium=minimum_premium,
        proxy_negative_nodes_exceeding_unchanged_price_allowance=substantial_negative_count,
        valuation_used_in_control_fitting=False,american_pricing_ready=False,optimal_stock_delta_ready=False)
    config=replace(policy.config,stock_knots=tuple(float(x) for x in knots))
    return replace(policy,config=config,value_premiums=tuple(models),diagnostics=diagnostics)


def audit_bs_premium_proxy(policy,trace):
    """Audit variance clamping on stored exercise-date states, not all branches."""
    if not all(isinstance(model,ConditionalBSPremium) for model in policy.value_premiums):
        raise TypeError("policy does not contain the structural BS premium proxy")
    if not np.array_equal(trace['times'],policy.continuation_times):
        raise ValueError("proxy range audit requires matching exercise times")
    rows=[model.range_diagnostics(np.asarray(trace['factors'])[:,j])
        for j,model in enumerate(policy.value_premiums)]
    total=sum(row['states'] for row in rows)
    return {"scope":"recorded exercise-date factor states; not every counterfactual branch or microstep",
        "states":total,"below_variance_grid":sum(row['below_variance_grid'] for row in rows),
        "above_variance_grid":sum(row['above_variance_grid'] for row in rows),"dates":rows,
        "clipping_changes_proxy_only":True}


__all__=['ConditionalBSPremium','build_bs_premium_proxy','audit_bs_premium_proxy']
