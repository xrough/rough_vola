"""Predeclared, bounded policy comparison; no selection on valuation outcomes.

The pilot records timing ONLY.  Training path counts in the clock arm are
chosen from that pilot before any final path is drawn.  All fitted policies
are sealed before valuation; failures and actual allowance overruns remain
results.  Numerical source snapshots and closed inventories make the stored
experiment independently reviewable without rerunning Monte Carlo.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from . import _pricing_policy_comparison_numeric as n
from ._pricing_rough_benchmark import (
    PARAMETERS, TOLERANCES, benchmark_cases, benchmark_config,
)
from rvlab.pricing.rough_lift import approximate_fractional_kernel, ExponentialKernel
from rvlab.pricing.types import RoughHestonParams
from rvlab.pricing.vanilla_types import SpotMarket, DiscountCurve, DividendSchedule


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOLDER = ROOT/"notebooks/_output/pricing/first_stage_v2/policy_comparison_v1"
VERSION = "rough_policy_comparison_v1"
ARMS = ("matched_paths", "matched_allowance")
TRAIN_PATHS, FINAL_PATHS, REPLICAS, STREAMS = 2048, 2048, 2, 2
LADDER = (512, 1024, 2048, 4096, 8192)
PRIMARY = "european_premium"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(n._jsonable(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sources():
    historical = ROOT/"notebooks/_output/pricing/first_stage_v1/01_pricing_and_exercise/rough_generic_benchmark_v1/serialization_repair_v1/design_before_replay.json"
    pins = dict(read_json(historical)["source_sha256"])
    if len(pins) != 63:
        raise ValueError("The historical 63-source numerical closure changed")
    for name, digest in pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError("Historical numerical source changed: "+name)
    for name in ("src/rvlab/experiments/_pricing_policy_comparison_numeric.py",
            "src/rvlab/experiments/_pricing_policy_comparison_run.py",
            "src/rvlab/experiments/_pricing_policy_comparison_stats.py",
            "src/rvlab/experiments/_pricing_policy_comparison_audit.py",
            "examples/rough_policy_comparison.py"):
        pins[name] = sha(ROOT/name)
    return pins


def inventory(folder):
    folder = Path(folder)
    entries = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size,
        "sha256": sha(p)} for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "preservation_inventory.json"]
    write_json(folder/"preservation_inventory.json", {"created_at": utc(),
        "files": entries, "entry_count": len(entries),
        "entries_sha256": hashlib.sha256(n._canonical(entries).encode()).hexdigest()})


def verify_inventory(folder):
    folder = Path(folder)
    record = read_json(folder/"preservation_inventory.json")
    entries = record["files"]
    actual = sorted(str(p.relative_to(folder)) for p in folder.rglob("*")
        if p.is_file() and p.name != "preservation_inventory.json")
    if (actual != sorted(r["path"] for r in entries) or len(entries) != record["entry_count"]
            or hashlib.sha256(n._canonical(entries).encode()).hexdigest() != record["entries_sha256"]):
        raise ValueError("Policy-comparison inventory is not closed")
    for row in entries:
        p = (folder/row["path"]).resolve()
        if not p.is_relative_to(folder.resolve()) or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("Stored experiment changed: "+row["path"])
    return record


def market():
    return SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))


def config(seed, valuation_seed, *, paths=TRAIN_PATHS, values=FINAL_PATHS):
    return benchmark_config(training_seed=seed, valuation_seed=valuation_seed,
        training_paths=paths, valuation_paths=values, steps=128, exercise_steps=64)


def fail(exc):
    return {"status": "failed", "error_type": type(exc).__name__,
        "message": str(exc), "traceback": traceback.format_exc()}


def select_training_paths(timing, *, train_paths=TRAIN_PATHS, value_paths=STREAMS*FINAL_PATHS):
    """Timing-only allocation, including the full online policy evaluation.

    This is a common ALLOWANCE, not equality of actual consumed seconds.
    Fixed-grid overhead and noisy N=32 timing make projections approximate.
    No payoff, price, standard error, exercise frequency or fit score enters.
    """
    fit = timing["training_seconds"]
    replay = timing["policy_evaluation_seconds"]
    count = timing["paths"]
    fit_names = {"cashflow_lsm": "cashflow_lsm", "fourier_premium_lsm": "fourier_premium_lsm",
        "bellman_lookahead": "bellman_lookahead_total"}
    # Shared simulation/control generation is charged equally to every standalone
    # candidate. Redundant campaign fits are recorded separately, not hidden.
    shared = timing["trace_generation_seconds"]*value_paths/count
    projected = lambda name, paths: (fit[fit_names[name]]*paths/count
        +replay[name]*value_paths/count+shared)
    allowance = 1.25*max(projected(name, train_paths) for name in n.POLICY_NAMES)
    choices, details = {}, []
    for name in n.POLICY_NAMES:
        eligible = [p for p in LADDER if projected(name, p) <= allowance]
        chosen = max(eligible) if eligible else min(LADDER)
        choices[name] = chosen
        details.append({"policy": name, "training_paths": chosen,
            "allowance_seconds": allowance, "projected_total_seconds": projected(name, chosen),
            "minimum_projected_ineligible": not bool(eligible),
            "training_ladder_cap_reached": chosen == max(LADDER),
            "projected_unused_seconds": allowance-projected(name, chosen),
            "projection_is_accuracy_or_runtime_guarantee": False})
    return choices, allowance, details


def pilot(folder=DEFAULT_FOLDER):
    folder = Path(folder)
    target = folder/"timing_pilot"
    target.mkdir(parents=True, exist_ok=False)
    kernel = approximate_fractional_kernel(.15, 90/365, 4)
    records = []
    write_json(target/"design_before_timing.json", {"version": VERSION,
        "created_at": utc(), "source_sha256": sources(), "paths": 32,
        "lookahead_order": 16, "role": "timing-only implementation pilot",
        "scientific_outputs_saved_or_used_for_allocation": False,
        "acceptance_evidence": False, "case_count": 6})
    for i, case in enumerate(benchmark_cases()):
        started = time.perf_counter()
        cfg = config(1_301_000_042+1000*i, 1_302_000_042+1000*i, paths=32, values=32)
        policies, training = n.train_policy_set(RoughHestonParams(**PARAMETERS), market(), kernel,
            strike=case["strike"], maturity=case["maturity"], config=cfg, lookahead_order=16)
        trace, generation = n.generate_shared_trace(policies, cfg)
        _, replay = n.evaluate_policy_set(trace, policies)
        row = {"case_id": case["case_id"], "paths": 32,
            "training_seed": cfg.training_seed, "valuation_seed": cfg.valuation_seed,
            "training_seconds": training["training_seconds"],
            "trace_generation_seconds": generation["trace_generation_seconds"],
            "policy_evaluation_seconds": replay["policy_evaluation_seconds"],
            "elapsed_seconds": time.perf_counter()-started,
            "scientific_outcomes_recorded": False}
        write_json(target/(case["case_id"]+".json"), row)
        records.append(row)
        print("TIMING", case["case_id"], round(row["elapsed_seconds"], 2), flush=True)
    write_json(target/"manifest.json", {"version": VERSION, "complete": True,
        "records": records, "created_at": utc(), "acceptance_evidence": False})
    inventory(target)


def prepare(folder=DEFAULT_FOLDER):
    folder = Path(folder)
    verify_inventory(folder/"timing_pilot")
    timing = read_json(folder/"timing_pilot/manifest.json")
    if not timing["complete"] or len(timing["records"]) != 6:
        raise ValueError("All six timing-only pilot cases are required")
    pins = sources()
    if read_json(folder/"timing_pilot/design_before_timing.json")["source_sha256"] != pins:
        raise ValueError("Sources changed after the timing pilot")
    cases = []
    for i, case in enumerate(benchmark_cases()):
        choices, allowance, allocation = select_training_paths(timing["records"][i])
        cases.append({**case, "index": i, "clock_training_paths": choices,
            "clock_allowance_seconds": allowance, "allocation": allocation,
            "training_seeds": [1_361_000_042+100_000*i+1000*r for r in range(REPLICAS)],
            "valuation_seeds": [1_461_000_042+100_000*i+1000*s for s in range(STREAMS)]})
    refinement = {"case_id": "put_90d_k100", "training_seeds": [1_561_000_042,1_561_001_042],
        "valuation_seeds": {name: [1_661_000_042+10_000*i+1000*s for s in range(2)]
            for i, name in enumerate(("lookahead_order", "physical_steps", "kernel_factors"))},
        "paths_per_stream": 512, "streams": 2,
        "scope": "ATM diagnostics, not generic price/Greek certification",
        "variants": {"lookahead_order": [16,8], "physical_steps": [128,256], "kernel_factors": [4,8]}}
    training_seeds = [s for c in cases for s in c["training_seeds"]]+refinement["training_seeds"]
    value_seeds = [s for c in cases for s in c["valuation_seeds"]]+[
        s for seeds in refinement["valuation_seeds"].values() for s in seeds]
    pilot_seeds = [r[k] for r in timing["records"] for k in ("training_seed","valuation_seed")]
    if (len(set(training_seeds+value_seeds+pilot_seeds)) != len(training_seeds+value_seeds+pilot_seeds)):
        raise ValueError("Training, timing and final streams must be disjoint")
    design = {"version": VERSION, "created_at": utc(), "source_sha256": pins,
        "parameters": PARAMETERS, "cases": cases,
        "kernels": {str(k): asdict(approximate_fractional_kernel(.15,90/365,k)) for k in (4,8)},
        "arms": list(ARMS), "policies": list(n.POLICY_NAMES), "controls": list(n.CONTROL_NAMES),
        "primary_control": PRIMARY, "training_replicas": REPLICAS, "training_paths": TRAIN_PATHS,
        "valuation_streams": STREAMS, "valuation_paths_per_stream": FINAL_PATHS,
        "unique_main_valuation_paths": 6*STREAMS*FINAL_PATHS, "initial_spots": [100.],
        "steps": 128, "exercise_steps": 64, "lookahead_order": 16,
        "lookahead_batch_size": 16, "ridge": 1e-5, "premium_cap": .7,
        "stock_integral": "exact_drift", "control_config": {"quadrature_nodes":24,"fourier_cutoff":100.,"riccati_steps":512},
        "training_ladder": list(LADDER), "clock_allowance_multiplier":1.25,
        "clock_scope": "standalone fit + shared trace/control generation + all 4096 online stopping decisions, per training replica",
        "clock_allocation_uses": "timings only, linear N32 projection; actual overrun retained without retries",
        "common_upper_anchor": "matched_paths Fourier-premium training replica 0, fixed across both arms and every policy/replica",
        "conditional_replica_average": "average two fitted policies within each path before estimating MCSE; equal-weight randomized policy value, not training uncertainty coverage",
        "paired_contrasts": [["fourier_premium_lsm","cashflow_lsm"],["bellman_lookahead","fourier_premium_lsm"]],
        "cashflow_baseline": "existing all-training-state full-factor cashflow LSM; not literal ITM-only LSM",
        "same_seeds_with_different_N_are_nested_paths": False,
        "selection_paths": 0, "winner_or_hyperparameter_selection": False,
        "independent_training_valuation": True, "all_training_before_any_final_paths": True,
        "all_failures_and_overruns_retained": True, "maximum_workers":2,
        "new_stock_delta_validation": False, "no_valuation_recentering": True,
        "refinement": refinement, "timing_manifest_sha256":sha(folder/"timing_pilot/manifest.json"),
        "tolerances": TOLERANCES, "optimal_american_readiness_promoted": False}
    write_json(folder/"design_before_training.json", design)
    for name, digest in pins.items():
        target = folder/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        if sha(target) != digest:
            raise ValueError("Source changed while taking snapshot: "+name)
    print("SEALED", sha(folder/"design_before_training.json"), len(pins), "sources", flush=True)


def get_design(folder):
    from ._pricing_policy_comparison_audit import audit_sources
    folder = Path(folder)
    design = read_json(folder/"design_before_training.json")
    if sources() != design["source_sha256"] or sha(folder/"timing_pilot/manifest.json") != design["timing_manifest_sha256"]:
        raise ValueError("Sealed comparison source or timing input changed")
    audit_sources(folder,design)
    return design


def _training_job(folder, case_index, arm, replica, *, kernel_refinement=False):
    folder = Path(folder)
    design = get_design(folder)
    case = design["cases"][case_index]
    key = f"{case['case_id']}_{arm}_r{replica}" if not kernel_refinement else f"refinement_kernel_r{replica}"
    target = folder/"training"/key
    target.mkdir(parents=True, exist_ok=False)
    started, clock = utc(), time.perf_counter()
    seed = (design["refinement"]["training_seeds"][replica] if kernel_refinement
        else case["training_seeds"][replica])
    counts = ({name: TRAIN_PATHS for name in n.POLICY_NAMES} if arm == ARMS[0] or kernel_refinement
        else case["clock_training_paths"])
    try:
        cfg = config(seed, case["valuation_seeds"][0])
        by_name = {name:replace(cfg, training_paths=counts[name]) for name in n.POLICY_NAMES}
        policies, metrics = n.train_policy_set(RoughHestonParams(**PARAMETERS), market(),
            ExponentialKernel(**design["kernels"]["8" if kernel_refinement else "4"]),
            strike=case["strike"], maturity=case["maturity"], config=cfg,
            cashflow_config=by_name["cashflow_lsm"], fourier_config=by_name["fourier_premium_lsm"],
            bellman_config=by_name["bellman_lookahead"], lookahead_order=16)
        saved = n.save_policy_set(policies, target/"policies")
        restored = n.load_policy_set(target/"policies")
        if n.policy_set_fingerprint(restored) != n.policy_set_fingerprint(policies):
            raise ValueError("Saved policy rules changed on reload")
        record = {"status":"completed","metrics":metrics,
            "policy_set_fingerprint":saved["policy_set_fingerprint"],
            "policies_json_sha256":sha(target/"policies/policies.json")}
    except Exception as exc:
        record = fail(exc)
    record.update(case_id=case["case_id"], arm=arm, replica=replica,
        training_seed=seed, training_paths=counts, kernel_refinement=kernel_refinement,
        started_at=started, completed_at=utc(), elapsed_seconds=time.perf_counter()-clock,
        design_sha256=sha(folder/"design_before_training.json"))
    write_json(target/"manifest.json", record)
    print("TRAIN",key,record["status"],round(record["elapsed_seconds"],2),flush=True)
    return record


def train(folder=DEFAULT_FOLDER, *, workers=2):
    folder = Path(folder)
    design = get_design(folder)
    if workers not in (1,2):
        raise ValueError("At most two workers on the declared machine")
    jobs = [(str(folder),i,arm,r) for i in range(6) for arm in ARMS for r in range(REPLICAS)]
    # Finish ALL ordinary and refinement training before any final trace is drawn.
    if workers == 1:
        records = [_training_job(*j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_training_unpack, jobs))
    records += [_training_job(folder,4,ARMS[0],r,kernel_refinement=True) for r in range(2)]
    pins = {str(p.relative_to(folder)):sha(p) for p in sorted((folder/"training").rglob("*")) if p.is_file()}
    write_json(folder/"training_complete.json", {"created_at":utc(),"all_attempts_complete":True,
        "records":records,"file_sha256":pins,"design_sha256":sha(folder/"design_before_training.json"),
        "manifest_sha256":{k:v for k,v in pins.items() if k.endswith("/manifest.json")},
        "all_successful":all(r["status"]=="completed" for r in records)})


def _training_unpack(args):
    return _training_job(*args)


def verify_training(folder):
    from ._pricing_policy_comparison_audit import audit_training
    folder = Path(folder)
    design = get_design(folder)
    record = read_json(folder/"training_complete.json")
    if not record["all_attempts_complete"] or len(record["records"]) != 26 or record["design_sha256"] != sha(folder/"design_before_training.json"):
        raise ValueError("Every declared training attempt must close before final valuation")
    for name, digest in record["file_sha256"].items():
        if sha(folder/name) != digest:
            raise ValueError("A frozen fitted rule changed: "+name)
    audit_training(folder,design)
    return design, record


def load_bundle(folder, case, arm, replica):
    path = Path(folder)/"training"/f"{case['case_id']}_{arm}_r{replica}"
    record = read_json(path/"manifest.json")
    if record["status"] != "completed":
        raise ValueError("Declared training attempt failed; no replacement rule")
    return n.load_policy_set(path/"policies")


def forbidden_seeds(design):
    return [s for c in design["cases"] for s in c["training_seeds"]]+design["refinement"]["training_seeds"]


def _valuation_job(folder, case_index, stream):
    folder = Path(folder)
    design, _ = verify_training(folder)
    case = design["cases"][case_index]
    target = folder/"valuation"/f"{case['case_id']}_s{stream}"
    target.mkdir(parents=True, exist_ok=False)
    started, clock = utc(), time.perf_counter()
    try:
        bundles = {arm:[load_bundle(folder,case,arm,r) for r in range(2)] for arm in ARMS}
        anchor = bundles[ARMS[0]][0]["fourier_premium_lsm"].base_policy
        cfg = config(anchor.config.training_seed,case["valuation_seeds"][stream])
        trace, generation = n.generate_shared_trace(bundles[ARMS[0]][0],cfg,
            forbidden_seeds=forbidden_seeds(design),control_policy=anchor)
        # Keep state/controls once for a reproducible nonanticipation audit.
        trace_arrays = {k:v for k,v in trace.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(target/"trace.npz", **trace_arrays)
        write_json(target/"trace_metadata.json",{k:v for k,v in trace.items() if not isinstance(v,np.ndarray)})
        all_arrays, all_metrics = {}, {}
        common = {}
        for arm,bundle_list in bundles.items():
            replicas, timings = [], []
            for bundle in bundle_list:
                arrays, metrics = n.evaluate_policy_set(trace,bundle,
                    forbidden_seeds=forbidden_seeds(design),control_policy=anchor)
                replicas.append(arrays)
                timings.append(metrics)
                for control in n.CONTROL_NAMES:
                    upper = arrays[control+"_upper"]
                    if control in common and not np.array_equal(upper,common[control]):
                        raise ValueError("Common upper differs across candidates or arms")
                    common[control] = upper
            for key in ("stop_indices",)+tuple(c+"_lower" for c in n.CONTROL_NAMES):
                all_arrays[arm+"__"+key] = np.stack([r[key] for r in replicas],axis=1)
            all_metrics[arm] = timings
        all_arrays.update({c+"_upper":v for c,v in common.items()})
        all_arrays["times"] = trace["times"]
        all_arrays["policy_names"] = np.asarray(n.POLICY_NAMES,dtype="U32")
        np.savez_compressed(target/"samples.npz",**all_arrays)
        record = {"status":"completed","trace_generation":generation,"policy_metrics":all_metrics,
            "sample_sha256":sha(target/"samples.npz"),"trace_sha256":sha(target/"trace.npz"),
            "trace_metadata_sha256":sha(target/"trace_metadata.json"),
            "same_upper_bitwise_across_all_policies_arms_replicas":True,
            "new_training_or_selection":False,"paths":FINAL_PATHS}
    except Exception as exc:
        record = fail(exc)
    record.update(case_id=case["case_id"],stream=stream,seed=case["valuation_seeds"][stream],
        started_at=started,completed_at=utc(),elapsed_seconds=time.perf_counter()-clock,
        design_sha256=sha(folder/"design_before_training.json"),
        training_complete_sha256=sha(folder/"training_complete.json"))
    write_json(target/"manifest.json",record)
    print("VALUE",case["case_id"],stream,record["status"],round(record["elapsed_seconds"],2),flush=True)
    return record


def _valuation_unpack(args):
    return _valuation_job(*args)


def value(folder=DEFAULT_FOLDER, *, workers=2):
    folder = Path(folder)
    verify_training(folder)
    if workers not in (1,2):
        raise ValueError("At most two workers")
    jobs = [(str(folder),i,s) for i in range(6) for s in range(2)]
    if workers == 1:
        records = [_valuation_job(*j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_valuation_unpack,jobs))
    write_json(folder/"valuation_complete.json",{"created_at":utc(),"all_attempts_complete":True,
        "records":records,"all_successful":all(r["status"]=="completed" for r in records),
        "manifest_sha256":{str(p.relative_to(folder)):sha(p) for p in sorted((folder/"valuation").glob("*/manifest.json"))}})


def _refinement_job(folder, name, stream):
    folder = Path(folder)
    design,_ = verify_training(folder)
    case = design["cases"][4]
    target = folder/"refinement"/f"{name}_s{stream}"
    target.mkdir(parents=True,exist_ok=False)
    clock,started = time.perf_counter(),utc()
    try:
        base = [load_bundle(folder,case,ARMS[0],r) for r in range(2)]
        anchor = base[0]["fourier_premium_lsm"].base_policy
        cfg = config(anchor.config.training_seed,design["refinement"]["valuation_seeds"][name][stream],values=512)
        trace,generation = n.generate_shared_trace(base[0],cfg,
            forbidden_seeds=forbidden_seeds(design),control_policy=anchor)
        comparison = {}
        for label in ("base","variant"):
            candidates = base
            current_trace,current_anchor = trace,anchor
            if label == "variant" and name == "lookahead_order":
                candidates = [{**b,"bellman_lookahead":n.build_lookahead_policy(
                    b["bellman_lookahead"].base_policy,order=8)} for b in base]
            elif label == "variant" and name == "physical_steps":
                current_trace,_ = n.generate_shared_trace(base[0],cfg,
                    forbidden_seeds=forbidden_seeds(design),control_policy=anchor,physical_steps=256)
            elif label == "variant" and name == "kernel_factors":
                candidates = [n.load_policy_set(folder/"training"/f"refinement_kernel_r{r}"/"policies") for r in range(2)]
                current_anchor = candidates[0]["fourier_premium_lsm"].base_policy
                current_trace,_ = n.generate_shared_trace(candidates[0],replace(cfg,training_seed=current_anchor.config.training_seed),
                    forbidden_seeds=forbidden_seeds(design),control_policy=current_anchor)
            # Preserve physical states and BOTH actual-grid control and original
            # decision European functions. Endpoint-only evidence cannot audit
            # the time/kernel comparison. GH8 reuses the exact same trace file.
            if label=="base" or name!="lookahead_order":
                np.savez_compressed(target/(label+"_trace.npz"),**{k:v for k,v in current_trace.items() if isinstance(v,np.ndarray)})
                write_json(target/(label+"_trace_metadata.json"),{k:v for k,v in current_trace.items() if not isinstance(v,np.ndarray)})
            results = [n.evaluate_policy_set(current_trace,b,forbidden_seeds=forbidden_seeds(design),
                control_policy=current_anchor)[0] for b in candidates]
            for key in ("stop_indices",)+tuple(c+"_lower" for c in n.CONTROL_NAMES):
                comparison[label+"__"+key] = np.stack([a[key] for a in results],axis=1)
            for c in n.CONTROL_NAMES:
                comparison[label+"__"+c+"_upper"] = results[0][c+"_upper"]
        np.savez_compressed(target/"samples.npz",**comparison)
        record = {"status":"completed","paths":512,"sample_sha256":sha(target/"samples.npz"),
            "trace_sha256":{p.name:sha(p) for p in sorted(target.glob("*trace*"))},
            "scope": "same full decision functions" if name == "physical_steps" else
                ("one quadrature change; stopping decisions may change" if name == "lookahead_order" else "joint kernel and retrained policy/control construction sensitivity"),
            "same_physical_paths":name=="lookahead_order",
            "same_rng_labels_are_nested_brownian_paths":False}
    except Exception as exc:
        record = fail(exc)
    record.update(name=name,stream=stream,seed=design["refinement"]["valuation_seeds"][name][stream],
        started_at=started,completed_at=utc(),elapsed_seconds=time.perf_counter()-clock,
        training_complete_sha256=sha(folder/"training_complete.json"),
        design_sha256=sha(folder/"design_before_training.json"))
    write_json(target/"manifest.json",record)
    print("REFINE",name,stream,record["status"],round(record["elapsed_seconds"],2),flush=True)
    return record


def _refinement_unpack(args):
    return _refinement_job(*args)


def refine(folder=DEFAULT_FOLDER,*,workers=2):
    folder=Path(folder)
    design,_=verify_training(folder)
    if workers not in (1,2):
        raise ValueError("At most two workers")
    jobs=[(str(folder),name,s) for name in design["refinement"]["valuation_seeds"] for s in range(2)]
    if workers==1:
        records=[_refinement_job(*j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=list(pool.map(_refinement_unpack,jobs))
    write_json(folder/"refinement_complete.json",{"created_at":utc(),"all_attempts_complete":True,"records":records,
        "manifest_sha256":{str(p.relative_to(folder)):sha(p) for p in sorted((folder/"refinement").glob("*/manifest.json"))},
        "all_successful":all(r["status"]=="completed" for r in records)})


def endpoint_audit(trace,arrays):
    """Reconstruct every endpoint from physical trace and each OWN stop index."""
    payoff=np.asarray(trace["discounted_payoffs"])
    stops=np.asarray(arrays["stop_indices"])
    if (payoff.ndim!=3 or payoff.shape[-1]!=1 or stops.shape!=(len(payoff),2,3,1)
            or stops.dtype.kind not in "iu" or np.any(stops<0) or np.any(stops>=payoff.shape[1])):
        raise ValueError("Endpoint audit requires aligned paths/replicas/policies/own stops")
    euro,premium=np.asarray(trace["european_martingale"]),np.asarray(trace["premium_martingale"])
    if any(a.shape!=payoff.shape or not np.isfinite(a).all() for a in (payoff,euro,premium)):
        raise ValueError("Endpoint audit trace is nonfinite or misaligned")
    checks=0
    for control,martingale in (("raw",np.zeros_like(euro)),("european",euro),("european_premium",euro+premium)):
        adjusted=payoff-martingale
        if not np.array_equal(arrays[control+"_upper"],adjusted.max(axis=1)):
            raise ValueError("Saved upper differs from common pathwise dual maximum")
        checks+=1
        lower=np.asarray(arrays[control+"_lower"])
        if lower.shape!=stops.shape:
            raise ValueError("Saved policy endpoint axes differ")
        for replica in range(2):
            for policy in range(3):
                expected=adjusted[np.arange(len(payoff)),stops[:,replica,policy,0],0]
                if not np.array_equal(expected,lower[:,replica,policy,0]):
                    raise ValueError("Saved lower differs from payoff/control at own stopping date")
                checks+=1
    return checks


def decision_audit(trace,bundles):
    """Re-evaluate four predeclared path indices, without fitting or selection."""
    count=len(trace["stocks"])
    indices=np.array([0,count//3,2*count//3,count-1])
    selected={key:(value[indices] if isinstance(value,np.ndarray) and key in (
        "stocks","factors","discounted_payoffs","european_proxy","decision_european",
        "european_martingale","premium_martingale","stop_indices") else value)
        for key,value in trace.items()}
    checks=0
    for bundle,expected_stops in bundles:
        for p,policy in enumerate(bundle.values()):
            own,_,_,_=n.stopping_rule(policy,selected)
            if not np.array_equal(own,expected_stops[indices,p,:]):
                raise ValueError("Stored own stopping rule fails independent bounded replay")
            checks+=1
    return checks


def collect(folder=DEFAULT_FOLDER):
    from ._pricing_policy_comparison_stats import summarize_case, _moments
    from ._pricing_policy_comparison_audit import audit_attempts
    folder=Path(folder)
    design,training=verify_training(folder)
    valuation=read_json(folder/"valuation_complete.json")
    refinement=read_json(folder/"refinement_complete.json")
    audit_attempts(folder,design,"valuation")
    audit_attempts(folder,design,"refinement")
    if not valuation["all_attempts_complete"] or not refinement["all_attempts_complete"]:
        raise ValueError("Every planned attempt must close, including failures")
    tables={key:[] for key in ("values","paired_policy_gains","controls","training_variation","exercise")}
    failures=[]
    case_status=[]
    audit_checks={"endpoint_algebra":0,"bounded_own_policy_replay":0}
    costs=[]
    for case in design["cases"]:
        records=[read_json(folder/"valuation"/f"{case['case_id']}_s{s}"/"manifest.json") for s in range(2)]
        successful=sum(r["status"]=="completed" for r in records)
        case_status.append({"case_id":case["case_id"],"declared_streams":2,"completed_streams":successful,
            "complete":successful==2,"partial_case_excluded":successful!=2,
            "declared_paths":2*FINAL_PATHS,"accepted_comparison_paths":2*FINAL_PATHS if successful==2 else 0})
        if not all(r["status"]=="completed" for r in records):
            failures.extend(r for r in records if r["status"]!="completed")
            continue
        streams=[]
        for s,r in enumerate(records):
            path=folder/"valuation"/f"{case['case_id']}_s{s}"/"samples.npz"
            if sha(path)!=r["sample_sha256"]:
                raise ValueError("Final observations changed")
            with np.load(path,allow_pickle=False) as data:
                if tuple(data["policy_names"])!=n.POLICY_NAMES:
                    raise ValueError("Policy ordering changed")
                streams.append({k:data[k].copy() for k in data.files})
            with np.load(path.parent/"trace.npz",allow_pickle=False) as source:
                trace={k:source[k] for k in source.files}
                trace.update(read_json(path.parent/"trace_metadata.json"))
                for arm in ARMS:
                    audit_checks["endpoint_algebra"]+=endpoint_audit(trace,{**{k.removeprefix(arm+"__"):v for k,v in streams[-1].items() if k.startswith(arm+"__")},
                        **{c+"_upper":streams[-1][c+"_upper"] for c in n.CONTROL_NAMES}})
                    audit_checks["bounded_own_policy_replay"]+=decision_audit(trace,[
                        (load_bundle(folder,case,arm,r),streams[-1][arm+"__stop_indices"][:,r]) for r in range(2)])
        for arm in ARMS:
            samples=[{**{key.removeprefix(arm+"__"):value for key,value in d.items() if key.startswith(arm+"__")},
                **{c+"_upper":d[c+"_upper"] for c in n.CONTROL_NAMES},"times":d["times"]} for d in streams]
            summary=summarize_case(samples,case_id=case["case_id"],arm=arm)
            for key in tables:
                tables[key].append(summary[key])
            for replica in range(2):
                train_record=read_json(folder/"training"/f"{case['case_id']}_{arm}_r{replica}"/"manifest.json")
                fit=train_record["metrics"]["training_seconds"]
                shared=sum(r["trace_generation"]["trace_generation_seconds"] for r in records)
                for name in n.POLICY_NAMES:
                    fit_seconds=fit["bellman_lookahead_total" if name=="bellman_lookahead" else name]
                    online=sum(r["policy_metrics"][arm][replica]["policy_evaluation_seconds"][name] for r in records)
                    total=fit_seconds+shared+online
                    costs.append({"case_id":case["case_id"],"arm":arm,"replica":replica,"policy":name,
                        "training_paths":train_record["training_paths"][name],"fit_seconds":fit_seconds,
                        "online_decision_seconds":online,"shared_trace_control_seconds":shared,
                        "standalone_total_seconds":total,"allowance_seconds":case["clock_allowance_seconds"],
                        "allowance_overrun":total>case["clock_allowance_seconds"],
                        "actual_unused_seconds":case["clock_allowance_seconds"]-total,
                        "within_allowance_is_equal_consumption":False,
                        "redundant_training_campaign_seconds":train_record["metrics"]["total_elapsed_seconds"],
                        "online_decisions_charged":STREAMS*FINAL_PATHS})
    for record in training["records"]+refinement["records"]:
        if record["status"]!="completed":
            failures.append(record)
    result={key:pd.concat(frames,ignore_index=True) if frames else pd.DataFrame() for key,frames in tables.items()}
    result["costs"]=pd.DataFrame(costs)
    result["case_status"]=pd.DataFrame(case_status)
    result["failures"]=pd.DataFrame(failures,columns=None if failures else ["status","error_type","message"])
    refine_rows=[]
    for name in design["refinement"]["valuation_seeds"]:
        records=[read_json(folder/"refinement"/f"{name}_s{s}"/"manifest.json") for s in range(2)]
        if not all(r["status"]=="completed" for r in records):
            continue
        arrays=[]
        for s,r in enumerate(records):
            path=folder/"refinement"/f"{name}_s{s}"/"samples.npz"
            if sha(path)!=r["sample_sha256"]:
                raise ValueError("Refinement payload changed")
            with np.load(path,allow_pickle=False) as d:
                arrays.append({k:d[k].copy() for k in d.files})
            for label in ("base","variant"):
                trace_label="base" if label=="variant" and name=="lookahead_order" else label
                with np.load(path.parent/(trace_label+"_trace.npz"),allow_pickle=False) as source:
                    trace={k:source[k] for k in source.files}
                    audit_checks["endpoint_algebra"]+=endpoint_audit(trace,{k.removeprefix(label+"__"):v for k,v in arrays[-1].items() if k.startswith(label+"__")})
        for control in n.CONTROL_NAMES:
            for p,policy in enumerate(n.POLICY_NAMES):
                baseline=[d["base__"+control+"_lower"][:,:,p,0].mean(axis=1) for d in arrays]
                variant=[d["variant__"+control+"_lower"][:,:,p,0].mean(axis=1) for d in arrays]
                base_upper=[d["base__"+control+"_upper"][:,0] for d in arrays]
                variant_upper=[d["variant__"+control+"_upper"][:,0] for d in arrays]
                difference=_moments([v-b for v,b in zip(variant,baseline)])
                base_mean,variant_mean=_moments(baseline)["estimate"],_moments(variant)["estimate"]
                cross=_moments([l-u for l,u in zip(baseline,variant_upper)])
                other_cross=_moments([l-u for l,u in zip(variant,base_upper)])
                se=difference["standard_error"]
                allowance=.001+1e-4*abs(base_mean)
                budget=abs(difference["estimate"])+difference["half_width"]
                refine_rows.append({"case_id":"put_90d_k100","refinement":name,"policy":policy,"control":control,
                    "baseline":base_mean,"variant":variant_mean,"change":difference["estimate"],
                    "change_se":se,"ci_low":difference["interval_low"],"ci_high":difference["interval_high"],
                    "baseline_upper":_moments(base_upper)["estimate"],"variant_upper":_moments(variant_upper)["estimate"],
                    "baseline_gap":_moments([u-l for u,l in zip(base_upper,baseline)])["estimate"],
                    "variant_gap":_moments([u-l for u,l in zip(variant_upper,variant)])["estimate"],
                    "baseline_lower_minus_variant_upper":cross["estimate"],"cross_bracket_se":cross["standard_error"],
                    "cross_bracket_ci_low":cross["interval_low"],"cross_bracket_ci_high":cross["interval_high"],
                    "variant_lower_minus_baseline_upper":other_cross["estimate"],
                    "reverse_cross_bracket_ci_low":other_cross["interval_low"],"reverse_cross_bracket_ci_high":other_cross["interval_high"],
                    "change_plus_uncertainty":budget,"allowance":allowance,"tolerance_resolved":budget<=allowance,
                    "paths":difference["paths"],"scope":records[0]["scope"],"same_physical_paths":records[0]["same_physical_paths"]})
    result["refinement"]=pd.DataFrame(refine_rows)
    result["design"]=pd.DataFrame([{"item":key,"declaration":n._canonical(value)} for key,value in design.items()
        if key not in ("source_sha256","kernels","cases")])
    target=folder/"tables"
    target.mkdir(exist_ok=False)
    for key,frame in result.items():
        frame.to_csv(target/(key+".csv"),index=False)
    manifest={"version":VERSION,"created_at":utc(),"complete":True,
        "all_planned_attempts_retained":True,"all_successful":not bool(failures),
        "failure_count":len(failures),"new_main_valuation_paths":sum(FINAL_PATHS for r in valuation["records"] if r["status"]=="completed"),
        "independent_cached_audit_checks":audit_checks,
        "unique_main_paths_not_multiplied_by_policy_arms_replicas":True,
        "new_refinement_paths":sum(512*(1 if r.get("name")=="lookahead_order" else 2) for r in refinement["records"] if r["status"]=="completed"),
        "design_sha256":sha(folder/"design_before_training.json"),
        "training_complete_sha256":sha(folder/"training_complete.json"),
        "valuation_complete_sha256":sha(folder/"valuation_complete.json"),
        "refinement_complete_sha256":sha(folder/"refinement_complete.json"),
        "tables":{key:{"rows":len(frame),"sha256":sha(target/(key+".csv"))} for key,frame in result.items()},
        "american_pricing_ready":False,"optimal_delta_ready":False,"no_new_greek_validation":True,
        "tolerances_changed":False}
    write_json(folder/"manifest.json",manifest)
    inventory(folder)
    print("COLLECTED",manifest["new_main_valuation_paths"],"unique main paths",len(failures),"failures",flush=True)
    return result,manifest


def load_closed(folder=DEFAULT_FOLDER):
    folder=Path(folder)
    verify_inventory(folder)
    design=get_design(folder)
    manifest=read_json(folder/"manifest.json")
    if not manifest["complete"] or manifest["design_sha256"]!=sha(folder/"design_before_training.json"):
        raise ValueError("Comparison campaign is incomplete or changed")
    tables={}
    for key,meta in manifest["tables"].items():
        path=folder/"tables"/(key+".csv")
        if sha(path)!=meta["sha256"]:
            raise ValueError("Comparison table changed")
        try:
            frame=pd.read_csv(path,float_precision="round_trip")
        except pd.errors.EmptyDataError:
            if meta["rows"]:
                raise
            frame=pd.DataFrame()
        if len(frame)!=meta["rows"]:
            raise ValueError("Comparison table row count changed")
        tables[key]=frame
    return tables,manifest,design
