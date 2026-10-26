"""Execute fixed independent-path conditional-centering precision streams.

Use --prepare before disjoint --indices workers, then --collect. The design
is sealed before any fresh primary valuation. Prior centering paths are not
pooled with the new independent streams.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rvlab.experiments.pricing import setup_demo,_jsonable
from rvlab.experiments._pricing_rough_centering_precision import centering_precision_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--indices',type=int,nargs='+')
    action.add_argument('--collect',action='store_true')
    parser.add_argument('--label',default='worker')
    args=parser.parse_args()
    if not args.label or not all(c.isalnum() or c in '-_' for c in args.label):parser.error('simple label required')
    t=time.perf_counter();started=datetime.now(timezone.utc).isoformat();runner=Path(__file__).read_bytes()
    context=setup_demo('01_pricing_and_exercise.ipynb')
    result=centering_precision_study(context,repeat_indices=[] if args.prepare else args.indices,compute_figures=args.collect)
    dest=context.output_dir/'rough_centering_precision_v1/campaign_v1'/args.label
    if (dest/'manifest.json').exists():dest/='replay_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest.mkdir(parents=True,exist_ok=True);(dest/'executed_runner.py').write_bytes(runner)
    for name,table in result['tables'].items():table.to_csv(dest/(name+'.csv'),index=False)
    for name,figure in result['figures'].items():figure.savefig(dest/(name+'.png'),bbox_inches='tight')
    manifest=_jsonable({**result['manifest'],'mode':'prepare' if args.prepare else ('collect' if args.collect else 'worker'),
        'started_at':started,'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-t,
        'runner_sha256':hashlib.sha256(runner).hexdigest(),'tables':{name:len(table) for name,table in result['tables'].items()}})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('PROFILE',manifest['profile_fingerprint'],'SECONDS',manifest['elapsed_seconds'],flush=True)
    if not result['tables']['pooled'].empty:
        print(result['tables']['pooled'][['quantity','estimate','standard_error','confidence_low','confidence_high']].to_string(index=False),flush=True)

if __name__=='__main__':main()
