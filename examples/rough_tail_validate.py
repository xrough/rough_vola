"""Independent increment and decision replay of the sealed joint experiment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def increment_control(c,control):
    n,nt,_=c.shape;weights=np.asarray(control['weights']).reshape(control['bins'],-1)
    per_step=np.repeat(weights,(nt-1)//control['bins'],axis=0)
    return np.column_stack((np.zeros(n),np.cumsum(np.einsum('ntf,tf->nt',np.diff(c,axis=1),per_step),axis=1)))
def audit_sample(path,design,seal,frozen):
    policy=frozen['selected_policy'];old_weights=frozen['weights'];proxy=frozen['proxy'];strike=proxy.strike
    with np.load(path,allow_pickle=False) as z:
        t=z['times'];n=len(z['raw']);ix=np.arange(n);maximum=0.
        components=np.stack([z[k] for k in ('european_martingale','premium_martingale','micro_boundary_martingale')],axis=-1)
        bins=np.searchsorted(old_weights.time_edges,t[:-1],side='right')-1
        old=np.column_stack((np.zeros(n),np.cumsum(np.einsum('ntf,tf->nt',np.diff(components,axis=1),old_weights.weights[bins]),axis=1)))
        maximum=max(maximum,float(np.max(abs(old-z['stable_old_martingale']))))
        prior=old+increment_control(z['prior_local_corrections'],design['prior_control'])
        current=prior+increment_control(z['grid_corrections'],design['grid_control'])
        maximum=max(maximum,float(np.max(abs(prior-z['prior_martingale']))),float(np.max(abs(current-z['current_martingale']))))
        stocks=z['stocks'][:,:,0];eu=z['european_proxy'][:,:,0];discount=z['discounts']
        q=np.column_stack([np.clip(model.value(stocks[:,j],z['factors'][:,j],strike),0.,policy.premium_cap) for j,model in enumerate(policy.continuations)])
        own_margin=(z['payoff']-eu)/discount-q
        maximum=max(maximum,float(np.max(abs(own_margin-z['tail_margin']))))
        for p,(name,spec) in enumerate(design['policy_specs'].items()):
            adjusted=q if p==0 else np.clip(q*spec['multiplier']+spec['offset']*np.maximum(0.,(t[-1]-t)/t[-1]),0.,policy.premium_cap)
            stops=np.full(n,len(t)-1);alive=np.ones(n,bool)
            for j in range(1,len(t)-1):
                if not z['eligible'][j]:continue
                intrinsic=np.maximum((1 if proxy.is_call else -1)*(stocks[:,j]-strike),0.)
                stop=alive&(intrinsic>0)&(intrinsic>=eu[:,j]/discount[j]+adjusted[:,j])
                stops[stop]=j;alive&=~stop
            if not np.array_equal(stops,z['stops'][:,p]) or not np.array_equal(z['raw'][:,p],z['payoff'][ix,stops]):raise ValueError('Manual policy replay differs: '+name)
        post=np.arange(len(t)-1)[None,:]>=z['stops'][:,0,None];margin=own_margin[:,:-1]
        gates=np.stack((post,post*expit(margin/.1),post*expit(margin/.3),expit(margin/.03),expit(margin/.1),expit(margin/.3),z['payoff'][:,:-1]==0),axis=-1)
        maximum=max(maximum,float(np.max(abs(gates-z['tail_gates']))))
        im=np.diff(z['intrinsic_martingale']-current,axis=1);em=np.diff(z['european_martingale']-current,axis=1)
        increments=gates*np.concatenate((np.repeat(im[...,None],6,axis=-1),em[...,None]),axis=-1)
        corrections=np.concatenate((np.zeros((n,1,7)),np.cumsum(increments,axis=1)),axis=1)
        maximum=max(maximum,float(np.max(abs(corrections-z['tail_corrections']))))
        result={};prefix=np.max(np.where(np.arange(len(t))[None,:]<=z['stops'][:,0,None],z['payoff']-current,-np.inf),axis=1)
        for c,(name,control) in enumerate(seal['controls'].items()):
            m=current+increment_control(corrections,control)
            lower=z['raw']-m[ix[:,None],z['stops']]
            upper=np.maximum(np.max(np.where(z['eligible'],z['payoff']-m,-np.inf),axis=1),z['payoff'][:,0])
            maximum=max(maximum,float(np.max(abs(lower-z[f'control{c}_lower']))),float(np.max(abs(upper-z[f'control{c}_upper']))))
            if control['family']=='poststop':
                if not np.array_equal(z[f'control{c}_lower'][:,0],z['control0_lower'][:,0]):raise ValueError('Poststop lower identity failed')
                if np.any(upper<prefix-2e-13):raise ValueError('Poststop prefix upper bound failed')
            result[name]=(lower,upper)
    if maximum>2e-13:raise ValueError('Independent reconstruction exceeds numerical roundoff: '+str(maximum))
    return {'sample':path.name,'paths':n,'maximum_reconstruction_difference':maximum,'all_seven_policy_decisions_exact':True,
        'poststop_lower_samples_bit_identical':True,'poststop_prefix_upper_bounds_pass':True,'sample_sha256':sha(path)},result
def main():
    output=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise';base=output/'rough_tail_attribution_v1'
    folder=base/'joint_switch_v1';target=base/'joint_switch_validation_v1'
    if target.exists():raise FileExistsError(target)
    design=json.loads((folder/'design_before_training.json').read_text());seal=json.loads((folder/'selection_before_final.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    inventory=json.loads((folder/'preservation_inventory.json').read_text());canonical=json.dumps(inventory['files'],sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(canonical).hexdigest()!=inventory['inventory_sha256']:raise ValueError('Inventory digest differs')
    for entry in inventory['files']:
        p=folder/entry['path']
        if sha(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('Inventory content differs')
    changed=[name for name,digest in design['source_sha256'].items() if sha(ROOT/name)!=digest]
    if changed:raise ValueError('Frozen sources changed')
    frozen=load_larger_policy_bundle(output/'rough_larger_policy_v1/32768_paths_two_starts')
    reports=[];streams=[]
    report,_=audit_sample(folder/'training_samples.npz',design,seal,frozen);reports.append(report)
    for i in (0,1):
        report,values=audit_sample(folder/f'final_{i}_samples.npz',design,seal,frozen);reports.append(report);streams.append(values)
    selected=manifest['selected_control'];policies=manifest['policy_order'];pi=policies.index(manifest['selected_policy'])
    stats=lambda seq:(float(np.concatenate(seq).mean()),float(np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in seq))/sum(map(len,seq))))
    rows=[]
    for c,values in seal['controls'].items():
        for p,policy in enumerate(policies):
            lo=[s[c][0][:,p] for s in streams];up=[s[c][1] for s in streams];price,pse=stats(lo);gap,gse=stats([u-l for u,l in zip(up,lo)])
            row=pd.read_csv(folder/'comparison.csv').loc[lambda a:(a.control==c)&(a.policy==policy)]
            if len(row)!=1:raise ValueError('Missing cross comparison')
            for key,actual in [('price',price),('price_se',pse),('gap',gap),('gap_se',gse)]:
                if not np.isclose(row.iloc[0][key],actual,rtol=0.,atol=2e-13):raise ValueError('Cross statistics differ')
            if c==selected and p==pi:rows.append({'control':c,'policy':policy,'price':price,'price_se':pse,'gap':gap,'gap_se':gse})
    validation={'version':'rough_tail_joint_manual_validation_v1','validated_at':datetime.now(timezone.utc).isoformat(),
        'campaign_manifest_sha256':sha(folder/'manifest.json'),'campaign_inventory_file_sha256':sha(folder/'preservation_inventory.json'),
        'campaign_internal_inventory_sha256':inventory['inventory_sha256'],'inventory_files_verified':len(inventory['files']),
        'unchanged_numerical_sources':len(design['source_sha256']),'sample_replays':reports,'all_cross_statistics_reproduced':True,
        'selected_comparison':rows[0],'no_new_paths_or_fitting':True,'acceptance_tolerances_changed':False,
        'american_pricing_ready':False,'new_rule_stock_delta_validated':False}
    target.mkdir(parents=True);(target/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    source=target/'source'/Path(__file__).relative_to(ROOT);source.parent.mkdir(parents=True);source.write_bytes(Path(__file__).read_bytes())
    saved=[{'path':str(p.relative_to(target)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(target.rglob('*')) if p.is_file()]
    canonical=json.dumps(saved,sort_keys=True,separators=(',',':')).encode()
    (target/'preservation_inventory.json').write_text(json.dumps({'files':saved,'inventory_sha256':hashlib.sha256(canonical).hexdigest()},indent=2)+'\n')
    print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
