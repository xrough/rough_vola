"""Small current-source demonstrations; no legacy bundles or market downloads."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from rvlab.pricing import (BlackScholesParams,HestonParams,RoughHestonParams,
    PricingConfig,FDConfig,SpotMarket,DiscountCurve,DividendSchedule,CashDividend,
    price,price_american_calls,calibrate_american_call_surface,evaluate_vanilla)

ROOT=Path(__file__).resolve().parents[1]
TRUTHS={'Black–Scholes':BlackScholesParams(.20),
    'Heston':HestonParams(.04,.045,1.5,.30,-.60),
    'Rough Heston':RoughHestonParams(.04,.045,1.,.12,-.60,.15)}
FOURIER=PricingConfig(96,100.,256)
FINE_FOURIER=PricingConfig(192,200.,512)


def style():
    plt.rcParams.update({'figure.dpi':110,'savefig.dpi':140,'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    pd.set_option('display.max_columns',10)
    pd.set_option('display.precision',6)


def implied_vol(value,forward,discount,strike,maturity):
    intrinsic=discount*max(forward-strike,0.)
    if value<=intrinsic+1e-12:return np.nan
    return brentq(lambda vol:float(price(BlackScholesParams(vol),forward=forward,
        discount=discount,strike=strike,maturity=maturity,is_call=True))-value,1e-5,4.)


def pricing_demo():
    rows=[]
    for days in (30,90):
        maturity=days/365.;strike=np.linspace(90.,110.,9)
        discount=np.exp(-.04*maturity);forward=100./discount
        for name,params in TRUTHS.items():
            values=price(params,forward=forward,discount=discount,strike=strike,
                maturity=maturity,is_call=True,config=FOURIER)
            for k,v in zip(strike,values):
                rows.append({'model':name,'days':days,'strike':k,'price':v,
                    'implied_volatility':implied_vol(v,forward,discount,k,maturity)})
    frame=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for axis,days in zip(axes,(30,90)):
        for name in TRUTHS:
            block=frame.loc[frame.days.eq(days)&frame.model.eq(name)]
            axis.plot(block.strike,block.implied_volatility,marker='o',ms=3,label=name)
        axis.set(title=f'{days}-day synthetic calls',xlabel='Strike',ylabel='BS implied volatility')
    axes[-1].legend(frameon=False);fig.tight_layout()
    return frame,fig


def american_controls():
    no_div=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    with_div=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((CashDividend(.08,1.5),),1.))
    cfg=FDConfig(120,32,100,spot_concentration=.05)
    rows=[];refinements=[]
    for market_name,market in (('Declared zero dividends',no_div),('Cash dividend $1.50 at .08 years',with_div)):
        for name,params in list(TRUTHS.items())[:2]:
            k=np.array([90.,100.,110.,90.,100.,110.]);calls=np.array([True]*3+[False]*3)
            eu=evaluate_vanilla(params,market=market,strike=k,maturity=60/365,
                is_call=calls,exercise_style='european',config=cfg,check_refinement=False)
            am=evaluate_vanilla(params,market=market,strike=k,maturity=60/365,
                is_call=calls,exercise_style='american',config=cfg,check_refinement=False)
            for i in range(len(k)):
                rows.append({'market':market_name,'model':name,'option':'call' if calls[i] else 'put',
                    'strike':k[i],'european':eu.prices[i],'american':am.prices[i],
                    'exercise_premium':am.prices[i]-eu.prices[i],'stock_delta':am.deltas[i]})
    for name,params in list(TRUTHS.items())[:2]:
        outputs=[]
        for config in (cfg,cfg.refined('joint')):
            outputs.append(evaluate_vanilla(params,market=no_div,strike=100.,maturity=60/365,
                is_call=False,exercise_style='american',config=config,check_refinement=False))
        a,b=outputs;pv=abs(float(b.prices)-float(a.prices));dv=abs(float(b.deltas)-float(a.deltas))
        refinements.append({'model':name,'price_change':pv,'price_allowance':.001+1e-4*abs(float(b.prices)),
            'delta_change':dv,'delta_allowance':.002+.005*abs(float(b.deltas)),
            'price_check_pass':pv<=.001+1e-4*abs(float(b.prices)),
            'delta_check_pass':dv<=.002+.005*abs(float(b.deltas))})
    frame=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for axis,kind in zip(axes,('call','put')):
        for (market_name,name),group in frame.loc[frame.option.eq(kind)].groupby(['market','model'],sort=False):
            axis.plot(group.strike,group.exercise_premium,marker='o',label=name+' · '+('cash event' if '$' in market_name else 'no dividends'))
        axis.set(title=f'60-day {kind}s',xlabel='Strike',ylabel='American − European price ($/unit)')
    axes[-1].legend(frameon=False,fontsize=8);fig.tight_layout()
    return frame,pd.DataFrame(refinements),fig


def call_reduction_demo():
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    rows=[]
    for name,params in TRUTHS.items():
        american=price_american_calls(params,market=market,strike=100.,maturity=60/365,config=FOURIER)
        d=market.discount_curve.discount(60/365)
        european=price(params,forward=100./d,discount=d,strike=100.,maturity=60/365,is_call=True,config=FOURIER)
        rows.append({'model':name,'american_call':float(american),'european_call':float(european),
            'difference':float(american-european),'route':'Guarded no-early-exercise reduction'})
    return pd.DataFrame(rows)


def synthetic_calibration(seed=42):
    rng=np.random.default_rng(seed)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    maturity=np.repeat(np.array([14,30,60])/365.,6)
    strikes=np.tile([94.,97.,99.,101.,103.,106.],3)
    train=np.tile([True,False,True,True,False,True],3)
    summaries=[];parameters=[];predictions=[];failures=[];checks=[]
    for name,truth in TRUTHS.items():
        observed=price_american_calls(truth,market=market,strike=strikes,maturity=maturity,config=FINE_FOURIER)
        # One deterministic noisy panel; the small demonstration does not claim a multistart study.
        noise=rng.normal(0.,.001,len(strikes))
        target=observed+noise
        quotes=pd.DataFrame({'strike':strikes,'maturity':maturity,'is_call':True,
            'exercise_style':'american','bid':target-.025,'ask':target+.025})
        initial=(BlackScholesParams(.30) if name=='Black–Scholes' else
            HestonParams(.035,.05,1.2,.25,-.45) if name=='Heston' else replace(truth,v0=.03))
        bounds=None
        if name=='Rough Heston':
            bounds={key:(value,value) for key,value in asdict(truth).items() if key!='v0'}
            bounds['v0']=(.005,.15)
        started=time.perf_counter()
        try:
            fit=calibrate_american_call_surface(quotes.loc[train].copy(),initial,market=market,
                bounds=bounds,config=FOURIER,max_nfev=60)
            fitted=price_american_calls(fit.params,market=market,strike=strikes,maturity=maturity,config=FOURIER)
            summaries.append({'model':name,'free_parameters':len(fit.free_parameters),
                'optimizer_converged':fit.success,'numerically_stable':fit.numerical_stable,
                'numerical_ready':fit.ready,'nfev':fit.nfev,'actual_pricing_evaluations':fit.pricing_evaluations,
                'active_bounds':';'.join(fit.active_bounds),'train_price_rmse':np.sqrt(np.mean((fitted[train]-target[train])**2)),
                'heldout_clean_rmse':np.sqrt(np.mean((fitted[~train]-observed[~train])**2)),
                'max_refinement_change':fit.max_price_change,'seconds':time.perf_counter()-started})
            for key,value in asdict(truth).items():
                parameters.append({'model':name,'parameter':key,'generating':value,
                    'initial':asdict(initial)[key],'fitted':asdict(fit.params)[key],
                    'fixed':key in fit.fixed_parameters})
            for i in range(len(strikes)):
                predictions.append({'model':name,'days':round(maturity[i]*365),'strike':strikes[i],
                    'split':'train' if train[i] else 'heldout','clean_price':observed[i],
                    'observed_price':target[i],'fitted_price':fitted[i],'error_to_clean':fitted[i]-observed[i]})
            if fit.diagnostics is not None:
                table=fit.diagnostics.checks.copy();table.insert(0,'model',name);checks.append(table)
        except Exception as exc:
            failures.append({'model':name,'failure_type':type(exc).__name__,'message':str(exc)})
    pred=pd.DataFrame(predictions);fig,axis=plt.subplots(figsize=(9,3.7))
    if not pred.empty:
        for name,group in pred.groupby('model',sort=False):
            held=group.loc[group.split.eq('heldout')]
            axis.plot(np.arange(len(held)),held.error_to_clean,marker='o',label=name)
    axis.axhline(0,color='gray',lw=1);axis.set(xlabel='Held-out contract index (synthetic)',ylabel='Fit − clean generating price ($/unit)',title='Recovered prices, not unique recovered parameters')
    axis.legend(frameon=False);fig.tight_layout()
    return {'summary':pd.DataFrame(summaries),'parameters':pd.DataFrame(parameters),
        'predictions':pred,'failures':pd.DataFrame(failures),
        'refinement':pd.concat(checks,ignore_index=True) if checks else pd.DataFrame(),'figure':fig}


def aggregate(name):
    folder=ROOT/'reports/aggregate_findings'
    manifest=json.loads((folder/'manifest.json').read_text())
    path=folder/name
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest['files'].get(name)!=digest:raise ValueError('Curated aggregate fingerprint changed')
    return pd.read_csv(path)


def market_input_status():
    # Check presence only. A file's presence never establishes temporal or vendor readiness.
    configured=os.environ.get('RVLAB_SPY_DATA_ROOT')
    base=Path(configured).expanduser() if configured else ROOT/'data/processed/spy'
    files={'independent stock':'references/stock.parquet','dated discount inputs':'references/rates.parquet',
        'dated dividend expectations':'references/dividends.parquet','prepared option sessions':'options'}
    return pd.DataFrame([{'input':key,'present':(base/relative).exists(),
        'availability_verified':False,'live_calibration_executed':False}
        for key,relative in files.items()])


def pilot_figure():
    frame=aggregate('pilot_fit_attempts.csv')
    market=frame.loc[frame.dataset.str.match(r'\d{4}-\d{2}-\d{2}')]
    # Retain all full-rough starts; never select a start using held-out outcomes.
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for model,group in market.groupby('model',sort=False):
        x=group.dataset.map({'2025-08-07':0,'2025-08-08':1,'2025-08-11':2}).to_numpy(float)
        axes[0].scatter(x,group.heldout_dollar_rmse,label=model,alpha=.75)
        axes[1].scatter(x,group.heldout_spread_rmse,label=model,alpha=.75)
    for axis,label in zip(axes,('Held-out dollar RMSE ($/unit)','Held-out half-spread-normalised RMSE')):
        axis.set(xticks=[0,1,2],xticklabels=['Aug 7','Aug 8','Aug 11'],ylabel=label,xlabel='2025 valuation date',yscale='log')
    axes[-1].legend(frameon=False,fontsize=8);fig.suptitle('Archived SPY pilot aggregates · no new market fit');fig.tight_layout()
    return fig
