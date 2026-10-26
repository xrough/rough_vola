"""Prefill or replay notebook-01 exercise-policy improvement experiments.

Use disjoint --indices workers, then --collect. This schedules one declared
numerical profile; it never reprices preserved historical stopping campaigns.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from rvlab.experiments.pricing import setup_demo, _jsonable
from rvlab.experiments._pricing_rough_policy_improvement import (
    policy_improvement_trial_specs, policy_improvement_probe_archives,
    rough_policy_improvement_study,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--indices", nargs="+", type=int)
    group.add_argument("--collect", action="store_true")
    parser.add_argument("--label", default="worker")
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    if not args.label or not all(c.isalnum() or c in "-_" for c in args.label):
        parser.error("label must be a simple alphanumeric identifier")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    runner_payload = Path(__file__).read_bytes()
    context = setup_demo("01_pricing_and_exercise.ipynb")
    result = rough_policy_improvement_study(context, seed=42,
        trial_specs=policy_improvement_trial_specs(),
        probe_archives=policy_improvement_probe_archives(),
        trial_indices=args.indices, reuse_checkpoints=not args.no_reuse,
        compute_figures=args.collect)
    destination = context.output_dir/"rough_policy_improvement_v1/campaign_v1"/args.label
    if (destination/"manifest.json").exists():
        destination /= "replay_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"executed_runner.py").write_bytes(runner_payload)
    for name, table in result["tables"].items():
        table.to_csv(destination/(name+".csv"), index=False)
    for name, figure in result["figures"].items():
        figure.savefig(destination/(name+".png"), bbox_inches="tight")
    manifest = _jsonable({**result["manifest"], "mode": "collect" if args.collect else "worker",
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-started,
        "runner_sha256": hashlib.sha256(runner_payload).hexdigest(),
        "tables": {name: len(table) for name, table in result["tables"].items()}})
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(result["tables"]["summary"].to_string(index=False), flush=True)
    print("PROFILE", manifest["profile_fingerprint"], "SECONDS", manifest["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
