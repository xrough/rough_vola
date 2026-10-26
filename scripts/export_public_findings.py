"""Export an allowlisted aggregate derivative of private research evidence.

Run only when the private sealed campaign is available. No contract IDs,
strikes, quote observations, stock observations, vendor files or local paths
are copied. The public derivative does not replace the original audit bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export(pilot_dir, *, policy_dir=None, destination=None):
    pilot_dir = Path(pilot_dir)
    destination = Path(destination or ROOT/'reports/aggregate_findings')
    destination.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(pilot_dir/'fit_summary.csv')
    columns = ['dataset','model','start','status','optimizer_success','nfev',
        'numerical_stable','ready','market_fit_accepted','objective',
        'train_dollar_rmse','heldout_dollar_rmse','train_spread_rmse',
        'heldout_spread_rmse','train_refined_dollar_rmse','heldout_refined_dollar_rmse',
        'train_refined_spread_rmse','heldout_refined_spread_rmse']
    public = source[columns].copy()
    public['active_bound_names'] = source.active_bounds.map(
        lambda value: ';'.join(sorted(json.loads(value))))
    selection = json.loads((pilot_dir/'training_only_selection.json').read_text())
    public['training_selected_rough_start'] = source.id.isin(selection.values())
    public['evidence_role'] = 'archived_aggregate_not_recomputed'
    public.to_csv(destination/'pilot_fit_attempts.csv', index=False)
    rows = []
    extension = pilot_dir/'refinement_extension'
    extension_rows=[]
    for path in sorted(extension.glob('*/result.json')):
        item=json.loads(path.read_text())
        row={name:item.get(name) for name in ('date','model','execution_status',
            'baseline_numerical_stable','baseline_optimizer_success','baseline_ready',
            'terminal_refinement_stable','model_ranking_accepted','original_fit_ready_unchanged')}
        row['original_execution_status']=row.pop('execution_status')
        attempted=int(item.get('attempted_pricing_evaluations',0))
        completed=int(item.get('successful_pricing_evaluations',0))
        deadline=any(call.get('status')=='deadline' for call in item.get('calls',[]))
        row['display_status']=('unstarted_deadline' if attempted==0 and deadline else
            'partial_timeout' if completed<attempted and deadline else row['original_execution_status'])
        row['pricing_evaluations_attempted']=attempted
        row['pricing_evaluations_completed']=completed
        row['timed_out']=bool(attempted>completed and deadline)
        row['timeout_price_array_accepted']=False
        row.update({name:item.get('metrics',{}).get(name) for name in
            ('train_dollar_rmse','heldout_dollar_rmse','train_spread_rmse','heldout_spread_rmse')})
        row['evidence_role']='archived_aggregate_not_recomputed'
        extension_rows.append(row)
    pd.DataFrame(extension_rows).to_csv(destination/'pilot_refinement_outcomes.csv',index=False)
    for candidate in sorted(extension.glob('*/comparisons.csv')):
        try:
            frame = pd.read_csv(candidate)
        except pd.errors.EmptyDataError:
            continue  # The corresponding failed/deadline outcome is retained above.
        outcome=json.loads((candidate.parent/'result.json').read_text())
        for check, group in frame.groupby('check',sort=False):
            rows.append({'date':outcome['date'], 'model':outcome['model'],
                'check':check,'contracts_checked':len(group),
                'max_absolute_price_change':group.absolute_price_change.max(),
                'max_price_budget_fraction':(group.absolute_price_change/group.price_tolerance).max(),
                'max_spread_budget_fraction':(group.absolute_price_change/group.spread_tolerance).max(),
                'all_price_checks_pass':bool(group.price_pass.all()),
                'all_spread_checks_pass':bool(group.spread_pass.all()),
                'all_checks_pass':bool(group['pass'].all()),
                'evidence_role':'archived_aggregate_not_recomputed'})
    pd.DataFrame(rows).to_csv(destination/'pilot_refinement_summary.csv',index=False)
    hashes = {'pilot_fit_summary':sha(pilot_dir/'fit_summary.csv'),
        'pilot_training_only_selection':sha(pilot_dir/'training_only_selection.json')}
    for name in ('summary.json','campaign_inventory.json','campaign_seal.json'):
        hashes['sealed_pilot_'+name.removesuffix('.json')]=sha(pilot_dir/name)
    for candidate in sorted(extension.glob('*/comparisons.csv')):
        hashes['pilot_refinement_'+candidate.parent.name] = sha(candidate)
    for candidate in sorted(extension.glob('*/result.json')):
        hashes['pilot_extension_outcome_'+candidate.parent.name]=sha(candidate)
    if policy_dir is not None:
        policy_dir = Path(policy_dir)
        value = pd.read_csv(policy_dir/'values.csv')
        value = value.loc[value.primary, ['case_id','arm','policy','price','price_se',
            'upper','gap','price_error_budget','price_allowance','price_budget_pass']].copy()
        value['evidence_role']='archived_synthetic_not_recomputed'
        value.to_csv(destination/'synthetic_policy_values.csv',index=False)
        gains = pd.read_csv(policy_dir/'paired_policy_gains.csv')
        gains = gains.loc[gains.primary,['case_id','arm','candidate_policy','reference_policy',
            'estimate','interval_low','interval_high','positive_interval']].copy()
        gains['evidence_role']='archived_synthetic_not_recomputed'
        gains.to_csv(destination/'synthetic_policy_gains.csv',index=False)
        hashes.update({'synthetic_policy_values':sha(policy_dir/'values.csv'),
            'synthetic_policy_gains':sha(policy_dir/'paired_policy_gains.csv')})
    files={p.name:sha(p) for p in sorted(destination.glob('*.csv'))}
    manifest={'version':'public_aggregate_findings_v1',
        'scope':'Allowlisted aggregate statistics; no observed contracts or market-data records.',
        'historical_results_recomputed':False,'full_provenance_included':False,
        'full_provenance_note':'Original sealed sources, market inputs, observations and audit logs remain in the private local research archive.',
        'source_content_sha256':hashes,'files':files,
        'pilot_attempt_count':len(public),'pilot_optimizer_converged':int(public.optimizer_success.sum()),
        'pilot_market_fit_accepted':int(public.market_fit_accepted.sum()),
        'hedging_ready':False,'independent_intraday_cross_source_agreement_established':False}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir',type=Path,required=True)
    parser.add_argument('--policy-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/aggregate_findings')
    args=parser.parse_args()
    manifest=export(args.pilot_dir,policy_dir=args.policy_dir,destination=args.output_dir)
    print(f"Exported {manifest['pilot_attempt_count']} aggregate pilot attempts; no contract-level observations copied.")


if __name__=='__main__':main()
