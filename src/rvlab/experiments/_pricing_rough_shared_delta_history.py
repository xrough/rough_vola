"""Replay the sealed price-precision evidence before shared-estimator Greeks.

The original narrow sampling pass and its qualifications are historical facts.
Reading the archive neither generates paths nor validates a stock sensitivity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd


ARCHIVE = (Path(__file__).resolve().parents[3] /
    'notebooks/legacy/pricing/rough_shared_delta_2026-10-20')
MC_PRECISION_PROFILE = '93d94ff9f30cfc6155d77797c05d3996dcc8dc1588b5e811b68f152c095dcac2'


def _verified_archive(archive):
    """Verify the large snapshot with bounded memory and contained paths."""
    archive = Path(archive).resolve()
    manifest = archive/'preservation_manifest.json'
    inventory = json.loads(manifest.read_text())
    for row in inventory['files']:
        path = (archive/row['preserved']).resolve()
        if not path.is_relative_to(archive):
            raise ValueError('Historical archive paths must remain inside the archive')
        if not path.is_file():
            raise ValueError(f'Missing preserved rough-Heston evidence: {path}')
        digest = hashlib.sha256(); size = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block); size += len(block)
        if size != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise ValueError(f'Changed preserved rough-Heston evidence: {path}')
    return inventory, hashlib.sha256(manifest.read_bytes()).hexdigest()


def load_historical_mc_precision(context=None, *, archive=None, compute_figures=True):
    """Load all verified stage22 tables/pixels without a new acceptance pool."""
    if not isinstance(compute_figures, bool):
        raise TypeError('compute_figures must be boolean')
    archive = ARCHIVE if archive is None else Path(archive)
    inventory, digest = _verified_archive(archive)
    recorded = json.loads((archive/'results/manifest.json').read_text())
    stage = recorded.get('rough_mc_precision', {})
    if (recorded.get('notebook') != '01_pricing_and_exercise' or
            recorded.get('evidence_version') != 'pricing_research_v1' or
            stage.get('backend') != 'rough_heston_mc_precision' or
            stage.get('profile_fingerprint') != MC_PRECISION_PROFILE or
            stage.get('primary_complete') is not True or
            stage.get('primary_paths') != 196608 or stage.get('seed_count') != 3 or
            stage.get('sampling_gate_pass') is not True or
            any(stage.get(name) is not False for name in ('american_pricing_ready',
                'stock_delta_ready', 'new_control_stock_delta_validated', 'optimal_delta_ready')) or
            not isinstance(stage.get('input_files'), list) or
            len(stage['input_files']) != 631 or
            len({row.get('path') for row in stage['input_files']}) != 631 or
            not isinstance(stage.get('analysis_sources'), dict) or
            len(stage['analysis_sources']) != 17):
        raise ValueError('Preserved evidence is not the declared complete precision campaign')
    prefix = 'rough_mc_precision_'
    table_names = [name for name in recorded['tables'] if name.startswith(prefix)]
    figure_names = [name for name in recorded['figures'] if name.startswith(prefix)]
    if len(table_names) != 17 or len(figure_names) != 3 or len(set(figure_names)) != 3:
        raise ValueError('Preserved precision table or figure set is incomplete')
    preserved = {row['preserved'] for row in inventory['files']}
    if any('results/'+name+'.csv' not in preserved for name in table_names) or any(
            'results/'+name+'.png' not in preserved for name in figure_names):
        raise ValueError('A declared historical precision output is outside its inventory')
    tables = {}
    for name in table_names:
        try:
            table = pd.read_csv(archive/'results'/(name+'.csv'), float_precision='round_trip')
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != recorded['tables'][name]:
            raise ValueError(f'Changed historical precision table count: {name}')
        tables[name.removeprefix(prefix)] = table
    figures = ({name.removeprefix(prefix): archive/'results'/(name+'.png')
        for name in figure_names} if compute_figures else {})
    manifest = deepcopy(stage)
    manifest.update(historical_replay=True, current_engine_execution=False,
        historical_archive=str(archive.resolve()),
        historical_archive_manifest_sha256=digest,
        historical_archive_files=len(inventory['files']),
        historical_execution_created_at=recorded['created_at'])
    if context is not None:
        context.progress('Loading verified 196,608-path precision evidence; no prior paths, coefficients or sampling checks are recomputed')
    return {'tables': tables, 'figures': figures, 'manifest': manifest}
