#!/usr/bin/env python3
"""Execute the three public notebooks in fresh kernels pinned to this Python.

python scripts/run_notebooks.py --all
python scripts/run_notebooks.py --all --output-dir /tmp/executed
No private archive, sibling checkout or market download is required.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
NAMES=('01_pricing_and_exercise.ipynb','02_synthetic_calibration.ipynb','03_spy_pilot.ipynb')


def source_hashes():
    paths=[*sorted((ROOT/'src/rvlab/pricing').glob('*.py')),ROOT/'scripts/public_notebook_support.py']
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def execute(path,output=None,timeout=600):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_core.utils import ensure_async,run_sync
    path=Path(path).resolve()
    if path.parent!=ROOT/'notebooks' or path.name not in NAMES:
        raise ValueError('Only the three public notebook names are supported')
    notebook=nbformat.read(path,4); before=source_hashes();start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='rough-vola-kernel-') as temporary:
        kernel_root=Path(temporary)/'kernels';spec=kernel_root/'rough-vola';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Rough volatility (selected interpreter)','language':'python',
            'env':{'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'),
                'PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg','MPLCONFIGDIR':str(Path(temporary)/'mpl'),
                'IPYTHONDIR':str(Path(temporary)/'ipython'),'OPENBLAS_NUM_THREADS':'1',
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}}))
        manager=AsyncKernelManager(kernel_name='rough-vola',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
        client=NotebookClient(notebook,km=manager,timeout=timeout,allow_errors=False,resources={'metadata':{'path':str(ROOT)}})
        try:client.execute(cleanup_kc=True)
        finally:
            if manager.has_kernel:
                async def stop():await ensure_async(manager.shutdown_kernel(now=True))
                run_sync(stop)()
    if before!=source_hashes():raise ValueError('Pricing/demo source changed during execution')
    errors=[o for cell in notebook.cells for o in cell.get('outputs',[]) if o.output_type=='error']
    if errors:raise ValueError('Notebook error output retained')
    notebook.metadata.kernelspec={'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'}
    notebook.metadata['public_execution']={'executed_at':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':round(time.perf_counter()-start,3),'python_version':platform.python_version(),
        'kernel_pinned_to_selected_interpreter':True,'source_root_verified_in_kernel':True,
        'private_archive_required':False,'market_data_recomputed':False,'source_sha256':before}
    destination=path if output is None else Path(output)
    destination.parent.mkdir(parents=True,exist_ok=True);nbformat.validate(notebook);nbformat.write(notebook,destination)
    print(f'Completed {path.name} in {time.perf_counter()-start:.1f}s',flush=True)
    return {'notebook':path.name,'elapsed_seconds':round(time.perf_counter()-start,3),'error_outputs':0,
        'source_bindings_pass':True,'kernel_pinned_to_selected_interpreter':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true',help='Execute all three public notebooks (also the default)')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('notebooks',nargs='*',help='One or more public notebook filenames')
    args=parser.parse_args(argv)
    if args.timeout<=0:parser.error('--timeout must be positive')
    names=args.notebooks or NAMES
    if any(Path(name).name not in NAMES for name in names):parser.error('Choose one of the three public notebooks')
    if args.list:
        print('\n'.join('notebooks/'+Path(name).name for name in names));return 0
    rows=[]
    for name in names:
        basename=Path(name).name
        rows.append(execute(ROOT/'notebooks'/basename,None if args.output_dir is None else args.output_dir/basename,args.timeout))
    destination=ROOT/'reports' if args.output_dir is None else args.output_dir
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'notebook_execution.json').write_text(json.dumps({'version':'public_notebooks_v1','records':rows,
        'private_data_included':False,'historical_campaigns_recomputed':False},indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
