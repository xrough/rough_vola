"""Run/cache the bounded synthetic American maturity-identification study.

Prefill shared evidence before parallel workers::

    python examples/american_maturity_identification.py --audit-only
    python examples/american_maturity_identification.py --design augmented --start 1

Run the other two starts in separate processes, then collect the full comparison
without filters. The collector alone may perform a selected parameter restart.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from rvlab.experiments.pricing import setup_demo
from rvlab.experiments._pricing_maturity_study import maturity_extension
from rvlab.experiments._pricing_american_identification import _jsonable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=("short", "augmented"), action="append")
    parser.add_argument("--start", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    began_at = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    context = setup_demo("02_synthetic_calibration.ipynb")
    result = maturity_extension(context, seed=args.seed, designs=args.design,
        start_numbers=args.start, max_nfev=args.max_nfev, reuse=not args.no_reuse,
        compute_figures=args.figures, run_calibration=not args.audit_only)
    directory = Path(result["manifest"]["checkpoint_directory"])
    worker = ("audit" if args.audit_only else
        "_".join(args.design or ("short", "augmented"))+"_"+
        "_".join(map(str, args.start or (1, 2, 3))))
    output = directory/f"worker_{worker}"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in result["tables"].items():
        frame.to_csv(output/f"{name}.csv", index=False)
    for name, figure in result["figures"].items():
        figure.savefig(output/f"{name}.png", dpi=150, bbox_inches="tight")
    metadata = {"worker": worker, "started_at": began_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-begin,
        "thread_environment": {name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "manifest": result["manifest"]}
    destination = directory/f"worker_{worker}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(_jsonable(metadata), indent=2, allow_nan=False)+"\n")
    temporary.replace(destination)
    print(result["tables"]["summary"].to_string(index=False))
    print(f"WORKER COMPLETE {worker}: {metadata['elapsed_seconds']:.1f}s; {directory}", flush=True)


if __name__ == "__main__":
    main()
