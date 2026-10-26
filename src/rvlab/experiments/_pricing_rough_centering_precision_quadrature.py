"""Independent conditional integration for stopped-control bias experiments.

This research helper integrates the actual control functions over the six
branches and a scalar Gaussian. It never sets a conditional drift to zero.
Quadrature differences are convergence diagnostics, not rigorous error bounds.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import expit, ndtr

from ._pricing_rough_centering_audit import _integration_points
from rvlab.pricing.rough_estimators import _AffineSurrogate, _conditional_payoffs
from rvlab.pricing.rough_simulation import _drift_flow_with_integral, _theta, _weak_transition_branches
from rvlab.pricing.rough_stopping_controls import expected_stopping_value
from rvlab.pricing.rough_stopping_fourier import (
    FourierStoppingPolicy, _affine_values, _continuation_coefficients, _interpolate,
    fourier_policy_fingerprint,
)


@lru_cache(maxsize=16)
def _legendre(nodes):
    x, w = np.polynomial.legendre.leggauss(nodes)
    x.setflags(write=False); w.setflags(write=False)
    return x, w


def _direct_premium(stocks, coefficient, strike, knots, cap):
    """Direct polynomial/clipping, separate from production moment routines."""
    z = np.asarray(stocks)/strike
    c = np.asarray(coefficient)
    chosen = np.broadcast_to(c, z.shape+(3,)) if c.ndim == 1 else c[np.searchsorted(knots, z, side="right")]
    result = np.maximum(0., chosen[..., 0]+z*(chosen[..., 1]+z*chosen[..., 2]))
    return result if cap is None else np.minimum(result, cap)


def _tail_moment(power, sigma, limit):
    """E[exp(power*sigma*Z) 1{|Z|>limit}] for standard normal Z."""
    shift = power*sigma
    return math.exp(.5*shift*shift)*(ndtr(-limit-shift)+ndtr(shift-limit))


def _branch_tail_bounds(proxy, policy, time, factor, mean, variance, coefficient, limit):
    """Absolute analytic envelope for omitted Gaussian tails, by component."""
    sigma = math.sqrt(variance)
    d = float(policy.market.discount_curve.discount(policy.maturity))
    growth = math.log(policy.market.discount_curve.discount(time)/d)
    f = math.exp(mean+growth)
    k = policy.strike
    a, b = proxy.coefficients[(policy.maturity, time)]
    c, constant = proxy._remaining_integral(time, policy.maturity)
    remaining = max(0., float(factor@c+constant))
    frequency_square = proxy.frequencies**2+.25
    coefficient_sum = np.sum((np.abs(np.exp(a+factor@b.T))
        +np.exp(-.5*remaining*frequency_square))*proxy.quadrature/frequency_square)
    correction = math.sqrt(f*k)/math.pi*coefficient_sum*_tail_moment(.5, sigma, limit)
    euro = d*((f*_tail_moment(1., sigma, limit) if policy.is_call else
        k*_tail_moment(0., sigma, limit))+correction)
    if policy.premium_cap is not None:
        premium = policy.premium_cap*_tail_moment(0., sigma, limit)
    else:
        bounds = np.max(np.abs(np.atleast_2d(coefficient)), axis=0)
        premium = sum(bounds[j]*math.exp(j*(mean-math.log(k)))*_tail_moment(j, sigma, limit)
            for j in range(3))
    intrinsic = math.exp(mean)*_tail_moment(1., sigma, limit) if policy.is_call else k*_tail_moment(0., sigma, limit)
    result = np.array([euro, premium, intrinsic])
    if not np.isfinite(result).all():
        raise FloatingPointError("Gaussian-tail envelope overflowed")
    return result


class ConditionalCenteringQuadrature:
    """Enumerate actual transitions and integrate actual values independently.

    The public evaluator accepts arbitrary LEFT microstep states. All supplied
    states must share a time; repeated time calls reuse transition/Fourier
    coefficients. ``martingale_weights`` must be frozen, with calendar weights
    and no additional outer state features. The study's selected three-control
    martingale satisfies this restriction. Micro-boundary weights are recomputed
    from each LEFT state, never from Gaussian endpoints.
    """
    def __init__(self, policy, *, martingale_weights, time_grid, normal_limit=12., batch_branches=64):
        from rvlab.pricing.rough_dual import RoughDualWeights
        if not isinstance(policy, FourierStoppingPolicy) or not isinstance(martingale_weights, RoughDualWeights):
            raise TypeError("frozen Fourier policy and RoughDualWeights required")
        if martingale_weights.policy_fingerprint != fourier_policy_fingerprint(policy):
            raise ValueError("martingale does not match the frozen policy")
        if martingale_weights.config.predictable_features:
            raise ValueError("conditional precision helper requires calendar-only outer weights")
        if not np.isfinite(normal_limit) or not 8 <= normal_limit <= 14:
            raise ValueError("normal_limit must lie in [8,14]")
        if isinstance(batch_branches, bool) or not isinstance(batch_branches, int) or batch_branches < 1:
            raise ValueError("batch_branches must be a positive integer")
        grid = np.asarray(time_grid, float)
        if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0 or grid[-1] != policy.maturity or not np.all(np.diff(grid)>0):
            raise ValueError("time_grid must increase exactly from zero to maturity")
        if policy.params.vol_of_vol <= 0:
            raise ValueError("conditional precision helper requires positive vol_of_vol")
        self.policy, self.weights = policy, martingale_weights
        self.grid, self.normal_limit, self.batch_branches = grid.copy(), float(normal_limit), batch_branches
        params, kernel = policy.params, policy.kernel
        self.nodes, self.kernel_weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
        self.matrix = -np.diag(self.nodes)-params.kappa*np.ones((len(self.nodes), 1))*self.kernel_weights
        self.proxy = _AffineSurrogate(params, policy.market, kernel, np.array([policy.strike]),
            np.array([policy.maturity]), np.array([policy.is_call]), grid, policy.control_config)
        self._flows = {}

    def _law(self, left, factors):
        policy, params = self.policy, self.policy.params
        index = np.flatnonzero(self.grid == left)
        if len(index) != 1 or index[0] == len(self.grid)-1:
            raise ValueError("left_time must be an exact nonterminal integration-grid time")
        right = float(self.grid[index[0]+1]); dt = right-left
        theta = _theta(params, .5*(left+right))
        key = (dt, theta)
        if key not in self._flows:
            self._flows[key] = _drift_flow_with_integral(self.matrix,
                np.full(len(self.nodes), params.kappa*(theta-params.v0)), self.kernel_weights, params.v0, dt/2)
        flow = self._flows[key]
        growth = math.log(policy.market.discount_curve.discount(left)/policy.market.discount_curve.discount(right))
        law = _weak_transition_branches(factors, params, nodes=self.nodes, weights=self.kernel_weights,
            flow=flow[:len(self.nodes), :len(self.nodes)], shift=flow[:len(self.nodes), -1],
            integrated_flow=flow, dt=dt, theta=theta, growth=growth, stock_integral=policy.config.stock_integral)
        return right, law

    def actual_primitive_values(self, *, time, stocks, factors):
        """Evaluate actual endpoint functions without conditional averaging."""
        stocks, factors = np.asarray(stocks, float), np.asarray(factors, float)
        if (stocks.ndim != 1 or factors.shape != (len(stocks),len(self.nodes))
                or not np.isfinite(stocks).all() or np.any(stocks<=0) or not np.isfinite(factors).all()):
            raise ValueError("finite positive stocks and matching full factor states required")
        if time not in self.grid:
            raise ValueError("time must be an exact integration-grid time")
        euro=self.proxy.values(float(time),factors[:,None],np.log(stocks)[:,None,None],
            np.zeros((len(stocks),1,1)),np.array([0]))[:,0,0,0]
        coefficients=_interpolate(self.policy,float(time),factors)
        premium=np.array([_direct_premium(s,c,self.policy.strike,self.policy.config.stock_knots,
            self.policy.premium_cap).item() for s,c in zip(stocks,coefficients)])
        intrinsic=np.maximum((1 if self.policy.is_call else -1)*(stocks-self.policy.strike),0.)
        return np.column_stack((euro,premium,intrinsic))

    def evaluate(self, *, left_time, left_stocks, left_factors, original_policy_stopped=False,
                 quadrature_nodes=(32, 64, 128)):
        """Return numerical conditional drifts and independent refinement changes.

        The last quadrature level is the reported estimate. Error estimates are
        the absolute last-two-level changes plus a separately bounded truncated
        Gaussian tail; they are explicitly not mathematical quadrature bounds.
        No state is removed, trimmed, demeaned or selected by its drift.
        """
        policy = self.policy
        levels = tuple(quadrature_nodes)
        if (len(levels)<2 or any(isinstance(n, (bool,np.bool_)) or not isinstance(n, (int,np.integer)) or n<4 for n in levels)
                or not all(a<b for a,b in zip(levels[:-1], levels[1:]))):
            raise ValueError("quadrature_nodes must contain at least two strictly increasing integer levels >=4")
        stocks, factors = np.asarray(left_stocks, float), np.asarray(left_factors, float)
        if (stocks.ndim != 1 or not len(stocks) or not np.isfinite(stocks).all() or np.any(stocks<=0)
                or factors.shape!=(len(stocks),len(self.nodes)) or not np.isfinite(factors).all()):
            raise ValueError("finite positive LEFT stocks and matching full factor states required")
        stopped = np.broadcast_to(np.asarray(original_policy_stopped, bool), stocks.shape)
        left = float(left_time)
        right, law = self._law(left, factors)
        branch_factors = law["new_factors"]
        variance = np.broadcast_to(law["orthogonal"], (len(stocks),3,2))
        means = np.log(stocks)[:,None,None]+law["correlated"][:,:,None]-.5*variance
        analytic_euro = _affine_values(self.proxy, right, branch_factors, means[...,None], variance[...,None])[...,0]
        coefficients = _interpolate(policy, right, branch_factors)
        analytic_premium = expected_stopping_value(means, variance, coefficients[:,:,None],
            strike=policy.strike, is_call=False, exercise=False,
            stock_knots=policy.config.stock_knots, value_cap=policy.premium_cap)
        analytic_intrinsic = _conditional_payoffs(means, variance, policy.strike, policy.is_call)
        analytic = np.stack((analytic_euro, analytic_premium, analytic_intrinsic), axis=-1)
        probabilities = law["probabilities"][:,:,None]*np.full((1,1,2),.5)
        records=[]
        tail=np.zeros_like(analytic)
        for path in range(len(stocks)):
            for branch in range(3):
                coefficient = coefficients[path,branch]
                for order in range(2):
                    mean, var = float(means[path,branch,order]), float(variance[path,branch,order])
                    points = _integration_points(mean, var, policy.strike, coefficient,
                        policy.config.stock_knots, policy.premium_cap, self.normal_limit) if var>0 else np.array([0.,0.])
                    if var>0:
                        threshold=(math.log(policy.strike)-mean)/math.sqrt(var)
                        if -self.normal_limit<threshold<self.normal_limit:
                            points=np.unique(np.r_[points,threshold])
                        tail[path,branch,order]=_branch_tail_bounds(self.proxy,policy,right,branch_factors[path,branch],
                            mean,var,coefficient,self.normal_limit)
                    records.append((path,branch,order,mean,var,points))
        estimates=[]
        for nodes in levels:
            integrations=np.zeros_like(analytic)
            x,w=_legendre(int(nodes))
            for begin in range(0,len(records),self.batch_branches):
                batch=records[begin:begin+self.batch_branches]
                normal_nodes=[]; node_weights=[]
                for path,branch,order,mean,var,points in batch:
                    if var==0:
                        normal_nodes.append(np.array([0.]));node_weights.append(np.array([1.]))
                    else:
                        mid=(points[:-1]+points[1:])/2;half=np.diff(points)/2
                        z=(mid[:,None]+half[:,None]*x).ravel()
                        normal_nodes.append(z);node_weights.append((half[:,None]*w).ravel()*np.exp(-.5*z*z)/math.sqrt(2*math.pi))
                size=max(map(len,normal_nodes))
                logs=np.zeros((len(batch),1,size));qw=np.zeros((len(batch),size))
                bf=np.empty((len(batch),1,len(self.nodes)))
                for i,(record,z,nw) in enumerate(zip(batch,normal_nodes,node_weights)):
                    path,branch,order,mean,var,_=record
                    logs[i,0,:]=mean
                    logs[i,0,:len(z)]=mean+math.sqrt(var)*z
                    qw[i,:len(z)]=nw
                    bf[i,0]=branch_factors[path,branch]
                actual=self.proxy.values(right,bf,logs,np.zeros_like(logs),np.array([0]))[:,0,:,0]
                for i,(path,branch,order,mean,var,_) in enumerate(batch):
                    stock=np.exp(logs[i,0])
                    premium=_direct_premium(stock,coefficients[path,branch],policy.strike,policy.config.stock_knots,policy.premium_cap)
                    intrinsic=np.maximum((1 if policy.is_call else -1)*(stock-policy.strike),0.)
                    integrations[path,branch,order]=np.array([actual[i]@qw[i],premium@qw[i],intrinsic@qw[i]])
            estimates.append(np.sum(probabilities[...,None]*(integrations-analytic),axis=(1,2)))
        discount=float(policy.market.discount_curve.discount(right))
        scaled=np.array([1.,discount,discount])
        primitive=np.asarray(estimates)*scaled
        tails=np.sum(probabilities[...,None]*tail,axis=(1,2))*scaled
        left_euro=_affine_values(self.proxy,left,factors[:,None],np.log(stocks)[:,None,None,None],np.zeros((len(stocks),1,1,1)))[:,0,0,0]
        left_coeff=_continuation_coefficients(policy,left,factors)
        qleft=np.array([_direct_premium(s,c,policy.strike,policy.continuations[0].knots,policy.premium_cap).item() for s,c in zip(stocks,left_coeff)])
        ld=float(policy.market.discount_curve.discount(left))
        boundary=expit((ld*(np.maximum((1 if policy.is_call else -1)*(stocks-policy.strike),0.)-qleft)-left_euro)/(ld*.05))
        def combine(values):
            e,p,i=np.moveaxis(values,-1,0)
            return {"european":e,"premium":p,"intrinsic":i,
                "intrinsic_minus_european":i-e,"micro_boundary":boundary*(i-e),
                "micro_poststop":boundary*stopped*(i-e-p)}
        def combine_errors(values):
            e,p,i=np.moveaxis(values,-1,0)
            return {"european":e,"premium":p,"intrinsic":i,
                "intrinsic_minus_european":i+e,"micro_boundary":boundary*(i+e),
                "micro_poststop":boundary*stopped*(i+e+p)}
        bin_index=int(np.searchsorted(self.weights.time_edges[1:],right,side="left"))
        multipliers=self.weights.weights[bin_index]
        all_means=combine(primitive)
        selected=sum(w*all_means[name] for w,name in zip(multipliers,self.weights.config.components))
        error_parts=combine_errors(np.abs(primitive[-1]-primitive[-2])+tails)
        numerical=sum(abs(w)*error_parts[name] for w,name in zip(multipliers,self.weights.config.components))
        tail_parts=combine_errors(tails)
        selected_tail=sum(abs(w)*tail_parts[name] for w,name in zip(multipliers,self.weights.config.components))
        if not np.isfinite(selected).all() or not np.isfinite(numerical).all():
            raise FloatingPointError("conditional Gaussian integration returned nonfinite drifts")
        analytic_expectation=np.sum(probabilities[...,None]*analytic,axis=(1,2))
        return {"conditional_drift":selected[-1],"drift_by_level":selected,
            "analytic_primitive_expectation":analytic_expectation,
            "independent_primitive_expectation_by_level":np.asarray(estimates)+analytic_expectation,
            "left_european_proxy":left_euro,
            "primitive_drift_by_level":primitive,"refinement_and_tail_estimate":numerical,
            "absolute_tail_envelope":selected_tail,"boundary_multiplier":boundary,
            "left_time":left,"right_time":right,"quadrature_nodes":levels,
            "normal_limit":self.normal_limit,"quadrature_error_is_rigorous_bound":False,
            "component_order":("european","premium","intrinsic"),"state_count":len(stocks)}


__all__=["ConditionalCenteringQuadrature"]
