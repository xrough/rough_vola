"""Read-only whole-path audit of the sealed optimized stock-grid campaign."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rvlab.experiments._pricing_rough_gap_time_stopping import collect_grid_campaign

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def weighted_increment_sum(corrections,control):
    # A different reconstruction from the engine's concatenated masked bases:
    # choose a coefficient row at each increment, then integrate its dot product.
    weights=np.asarray(control['weights']).reshape(control['bins'],-1)
    increments=np.diff(corrections,axis=1)
    rows=np.repeat(weights,64//control['bins'],axis=0)
    value=np.cumsum(np.einsum('ntf,tf->nt',increments,rows),axis=1)
    return np.column_stack((np.zeros(len(value)),value))

def main():
    base=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise/rough_gap_time_refinement_v1'
    folder=base/'stock_grid_control_v2';target=base/'stock_grid_control_validation_v1'
    if target.exists():raise FileExistsError(target)
    tables,entries,manifest=collect_grid_campaign(folder)
    seal=json.loads((folder/'controls_before_final.json').read_text())
    prior=manifest['prior_control'];maximum=0.
    for spec in manifest['final_streams']:
        with np.load(folder/f"final_{spec['seed']}_samples.npz",allow_pickle=False) as data:
            old=data['stable_old_martingale'];baseline=old+weighted_increment_sum(data['prior_local_corrections'],prior)
            maximum=max(maximum,float(np.max(abs(baseline-data['baseline_martingale']))))
            paths=np.arange(spec['paths']);stops=data['stops'];raw=data['raw']
            for index,(name,control) in enumerate(seal['controls'].items()):
                martingale=baseline+weighted_increment_sum(data['grid_corrections'],control)
                lower=raw-martingale[paths,stops];upper=np.max(data['payoff']-martingale,axis=1)
                maximum=max(maximum,float(np.max(abs(lower-data[f'control{index}_lower']))),float(np.max(abs(upper-data[f'control{index}_upper']))))
    assert maximum<2e-13,'Independent increment-weight reconstruction differs'
    oldpilot=pd.read_csv(base/'stock_grid_control_v1/pilot_selection.csv')
    newpilot=pd.read_csv(folder/'pilot_selection.csv')
    pilot_difference=max(float(np.max(abs(oldpilot[k]-newpilot[k]))) for k in oldpilot.select_dtypes('number'))
    changed=[name for name,digest in manifest['source_sha256'].items() if sha(ROOT/name)!=digest]
    assert not changed,'A frozen numerical source changed'
    selected=tables['stopping_comparison'].loc[lambda t:t.control==manifest['selected']].iloc[0]
    baseline=tables['stopping_comparison'].loc[lambda t:t.control=='stable baseline'].iloc[0]
    changes=tables['stopping_paired_changes'].loc[lambda t:t.control==manifest['selected']].iloc[0]
    validation={'version':'rough_grid_control_validation_v1','validated_at':datetime.now(timezone.utc).isoformat(),
        'campaign_manifest_sha256':sha(folder/'manifest.json'),'campaign_inventory_file_sha256':sha(folder/'preservation_inventory.json'),
        'campaign_internal_inventory_sha256':json.loads((folder/'preservation_inventory.json').read_text())['inventory_sha256'],
        'all_collector_checks_pass':True,'independent_increment_weight_endpoint_max_difference':maximum,
        'same_spent_pilot_table_max_difference':pilot_difference,'unchanged_source_files':len(manifest['source_sha256']),
        'changed_source_files':changed,'preselected_control':manifest['selected'],'final_streams':manifest['final_streams'],
        'final_paths':8192,'all_paths_retained':True,'exercise_rule_changed':False,'no_new_paths_or_fit':True,
        'baseline_gap':float(baseline.gap),'selected_gap':float(selected.gap),
        'relative_gap_reduction':1-float(selected.gap)/float(baseline.gap),
        'paired_gap_change':float(changes.gap_change),'paired_gap_change_se':float(changes.gap_change_se),
        'selected_price_error_budget':float(selected.price_error_budget),'selected_price_allowance':float(selected.price_allowance),
        'price_budget_pass':bool(selected.price_budget_pass),'american_pricing_ready':False,'stock_delta_ready':False,
        'shared_european_node_metadata_counts_positive_nodes':True,'positive_control_nodes':[77,153],'total_control_nodes_with_zero':[78,154],
        'interpretation':'price and delta tolerances unchanged; new martingale controls do not change or certify the fixed exercise policy'}
    target.mkdir(parents=True)
    (target/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    for name in ('stopping_comparison','stopping_paired_changes','stopping_per_seed','stopping_gap_distribution'):
        tables[name].to_csv(target/(name+'.csv'),index=False)
    source=target/'source'/Path(__file__).relative_to(ROOT);source.parent.mkdir(parents=True);source.write_bytes(Path(__file__).read_bytes())
    saved=[{'path':str(p.relative_to(target)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(target.rglob('*')) if p.is_file()]
    canonical=json.dumps(saved,sort_keys=True,separators=(',',':')).encode()
    (target/'preservation_inventory.json').write_text(json.dumps({'files':saved,'inventory_sha256':hashlib.sha256(canonical).hexdigest()},indent=2)+'\n')
    print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
