"""Restart the three exhausted American Heston-curve fits on their exact objective.

This diagnostic preserves the original attempt files. It changes only the
starting parameters (to the saved final parameters) and supplies up to 80 new
optimizer evaluations. The market, contracts, weights, bounds, frozen fitting
grid, optimizer coordinates, derivative scheme and all acceptance thresholds
are unchanged. No held-out result selects a restart.

Run one of ``--attempt clean-1``, ``clean-2`` or ``noisy-2`` with the project
interpreter; separate processes may run concurrently. Each writes a disjoint,
fingerprinted checkpoint below the declared output directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

from rvlab.experiments import _pricing_american_identification as study
from rvlab.pricing import FDConfig, OptimizerConfig, replace_parameters


PROFILE = "american_joint_identification_v2_spot_accuracy/837b8286fe4df229"
ATTEMPTS = {"clean-1": ("clean", 1), "clean-2": ("clean", 2), "noisy-2": ("noisy", 2)}
BUDGET = 80
STAGE = "diagnostic_continuation"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restart_specification(root: Path, attempt: str):
    """Validate and fully describe a continuation without starting any pricing."""
    dataset, start = ATTEMPTS[attempt]
    active = root / "notebooks/_output/pricing/research_v1/02_synthetic_calibration"
    baseline = active / PROFILE / f"heston_curve_{dataset}_{start}_exploratory.json"
    record = json.loads(baseline.read_text())
    old, spec = record["payload"]["summary"], record["specification"]
    expected = f"heston_curve_{dataset}_{start}_exploratory"
    if old["attempt_id"] != expected or old["optimizer_status"] != 0 or old["nfev"] != 80:
        raise ValueError("diagnostic requires the declared exhausted 80-evaluation attempt")
    if spec["engine_sha256"] != study._engine_fingerprint():
        raise ValueError("current numerical engine differs from the saved baseline engine")
    config = FDConfig(**spec["fit_config"])
    if (config.spot_steps, config.variance_steps, config.time_steps,
            config.spot_concentration) != (192, 48, 180, .05):
        raise ValueError("saved fitting grid is not the declared accepted grid")
    optimizer = OptimizerConfig(**spec["optimizer"])
    expected_optimizer = OptimizerConfig(parameterization="asinh", jacobian="2-point",
        ftol=1e-8, xtol=1e-8, gtol=1e-6)
    if optimizer != expected_optimizer:
        raise ValueError("saved optimizer does not match the unchanged continuation controls")
    acceptance = {"price_atol": study.PRICE_ATOL, "price_rtol": study.PRICE_RTOL,
        "delta_atol": study.DELTA_ATOL, "delta_rtol": study.DELTA_RTOL,
        "spread_refinement_fraction": study.SPREAD_REFINEMENT_FRACTION}
    if spec["acceptance"] != acceptance or spec["spread_floor"] != study.SPREAD_FLOOR:
        raise ValueError("saved acceptance thresholds differ from the unchanged study thresholds")
    source = Path(__file__).resolve()
    specification = {"diagnostic": "american_curve_optimizer_restart_v1",
        "baseline_path": str(baseline.relative_to(root)), "baseline_sha256": _sha(baseline),
        "baseline_attempt_id": expected, "baseline_checkpoint_fingerprint": record["fingerprint"],
        "engine_sha256": study._engine_fingerprint(), "script_sha256": _sha(source),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__},
        "dataset": dataset, "start": start, "stage": STAGE,
        "initial": old["params"], "bounds": spec["bounds"],
        "config": asdict(config), "optimizer": asdict(optimizer), "max_nfev": BUDGET,
        "acceptance": acceptance, "spread_floor": study.SPREAD_FLOOR,
        "quotes": spec["quotes"], "market": spec["market"], "truth": spec["truth"],
        "heldout_used_for_selection": False,
        "interpretation": "Restart from saved final physical parameters on the same training objective; optimizer state and historical trust radius are restarted."}
    return baseline, old, spec, specification, config, optimizer


def run(root: Path, attempt: str, output_dir: Path, *, reuse=True):
    baseline, old, spec, specification, config, optimizer = restart_specification(root, attempt)
    fingerprint = study._fingerprint(specification)
    output = output_dir / f"{attempt}_{fingerprint[:16]}.json"
    if reuse and output.exists():
        record = json.loads(output.read_text())
        if record.get("fingerprint") != fingerprint:
            raise ValueError("existing diagnostic checkpoint fingerprint does not match")
        print(f"Reused {output}", flush=True)
        return record
    truths, _, _ = study.joint_specifications()
    truth = replace_parameters(truths["heston_curve"], **spec["truth"])
    initial = replace_parameters(truth, **old["params"])
    quotes = pd.DataFrame(spec["quotes"])
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    payload = study._attempt("heston_curve", spec["dataset"], spec["start"], STAGE,
        quotes, study.joint_market(), initial, spec["bounds"], truth, config, BUDGET, optimizer)
    record = {"fingerprint": fingerprint, "specification": specification,
        "execution": {"started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter()-timer,
            "thread_environment": {name: os.environ.get(name) for name in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "payload": study._jsonable(payload)}
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(study._jsonable(record), indent=2, allow_nan=False)+"\n")
    temporary.replace(output)
    summary = payload["summary"]
    print(json.dumps({name: summary.get(name) for name in
        ("attempt_id", "status", "nfev", "njev", "training_objective", "physical_optimality",
         "optimality", "active_bounds", "params", "ready", "hedge_delta_ready",
         "price_stable", "delta_stable", "pricing_evaluations", "error")}, indent=2), flush=True)
    print(f"Saved {output}", flush=True)
    return record


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=ATTEMPTS, required=True)
    parser.add_argument("--output-dir", type=Path, default=root /
        "notebooks/_output/pricing/research_v1/02_synthetic_calibration/american_curve_diagnosis_v1/optimizer")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--describe", action="store_true", help="Validate inputs and print the specification without pricing")
    args = parser.parse_args()
    if args.describe:
        _, _, _, specification, _, _ = restart_specification(root, args.attempt)
        print(json.dumps(specification, indent=2))
    else:
        run(root, args.attempt, args.output_dir, reuse=not args.no_reuse)


if __name__ == "__main__":
    main()
