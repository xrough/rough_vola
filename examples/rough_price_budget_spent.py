"""Preserve a conditional-European control probe using only spent fixed paths."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_price_budget import conditional_european_replay,shared_european_control

OUTPUT=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise'
PRIOR=OUTPUT/'rough_bellman_validation_v1/bellman_control_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(_jsonable(v),indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def main(count):
    folder=OUTPUT/'rough_price_budget_v1'/('spent_terminal_probe32_v1' if count==32 else 'spent_terminal_anchor_v1')
    if folder.exists():raise FileExistsError('Preserve the previous spent trial')
    folder.mkdir(parents=True)
    sources=[ROOT/n for n in read(PRIOR/'design_before_training.json')['source_sha256']]
    sources += [ROOT/'src/rvlab/experiments/_pricing_rough_price_budget.py',Path(__file__)]
    pins={**read(PRIOR/'design_before_training.json')['input_sha256']}
    inputs=[PRIOR/n for n in ('manifest.json','selection_before_final.json','design_before_training.json')]
    inputs += [PRIOR/f'final_{i}_{suffix}' for i in range(2) for suffix in ('manifest.json','samples.npz')]
    pins.update({str(p.relative_to(OUTPUT)):sha(p) for p in inputs})
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        target=folder/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    specs=[{'seed':1095005042,'prior_index':1,'indices':list(range(32))}] if count==32 else [
        {'seed':s,'prior_index':i,'indices':list(range(4096))} for i,s in enumerate((1094005042,1095005042))]
    design={'version':'rough_price_budget_spent_v1','created_at':now(),'source_sha256':hashes,'input_sha256':pins,
        'specifications':specs,'full_stream_paths':4096,'physical_steps':128,'exercise_intervals':64,
        'new_outer_paths':0,'new_policy_or_dual_fit':False,'used_for_fresh_acceptance':False,
        'anchor':'(realized European payoff - its centered martingale) - (conditional European payoff - its centered martingale)',
        'continuous_price_centering':False,'empirical_valuation_recentering':False,'tolerances_changed':False}
    write(folder/'design_before_replay.json',design)
    proxy=load_larger_policy_bundle(OUTPUT/'rough_larger_policy_v1/32768_paths_two_starts')['proxy']
    records=[]
    for i,spec in enumerate(specs):
        before=perf_counter();p=PRIOR/f"final_{spec['prior_index']}_samples.npz"
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in ('stocks','factors','actual_time_grid','payoff','european_martingale','candidate3_lower','candidate3_upper')}
        replay,diagnostic=conditional_european_replay(proxy,seed=spec['seed'],paths=4096,indices=np.array(spec['indices']),reference=a)
        index=replay['absolute_path_indices'];x=shared_european_control(a,replay)
        np.savez_compressed(folder/f'stream_{i}_samples.npz',**replay,shared_control=x,
            lower=a['candidate3_lower'][index],upper=a['candidate3_upper'][index],
            realized_european_payoff=a['payoff'][index,-1],realized_european_martingale=a['european_martingale'][index,-1])
        record={**spec,'elapsed_seconds':perf_counter()-before,'completed_at':now(),'diagnostics':diagnostic,
            'sample_sha256':sha(folder/f'stream_{i}_samples.npz')}
        write(folder/f'stream_{i}_manifest.json',record);records.append(record)
    if any(sha(ROOT/n)!=v for n,v in hashes.items()) or any(sha(OUTPUT/n)!=v for n,v in pins.items()):raise ValueError('Frozen spent source/input changed')
    write(folder/'manifest.json',{**design,'complete':True,'completed_at':now(),'records':records})
    entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file()]
    write(folder/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,choices=(32,4096),default=32);args=parser.parse_args();main(args.count)
