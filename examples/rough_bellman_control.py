"""Staged exact-centered controls with the accepted GH16 exercise rule frozen."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"examples")]
import numpy as np
import pandas as pd
import rough_tail_attribution as tail
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_bellman_control_aligned import TARGETS, bellman_target_martingales
from rvlab.experiments._pricing_rough_bellman_delta import bellman_policy_stops
from rvlab.experiments._pricing_rough_bellman_policy import BellmanEvaluator
from rvlab.experiments._pricing_rough_frozen_policy import _decode_models
from rvlab.experiments._pricing_rough_local_control import correction_basis, fit_correction_weights


OUTPUT = tail.OUTPUT
FOLDER = OUTPUT/"rough_bellman_validation_v1/bellman_control_v1"
PROXY = OUTPUT/"rough_bellman_validation_v1/spent_proxy_aligned_v2"
TAIL = OUTPUT/"rough_tail_attribution_v1/joint_switch_v1"
BELLMAN = OUTPUT/"rough_bellman_policy_v1/one_step_v1"
TRAIN = 1092005042; SELECT = 1093005042; FINALS = (1094005042, 1095005042)
BASELINE = "latest selected tail control"
RESERVED = tuple(tail.BASELINE_RESERVED)+(1043005042, 1054005042, 1055005042,
    1066005042, 1077005042, 1078005042, 1089005042, 1090005042, 1091005042)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False)+"\n")
def now(): return datetime.now(timezone.utc).isoformat()
def stat(chunks):
    n = sum(map(len, chunks)); value = float(np.concatenate(chunks).mean())
    se = float(np.sqrt(sum(len(v)*np.var(v, ddof=1) for v in chunks))/n)
    return value, se, value-1.96*se, value+1.96*se


def source_paths():
    return list(dict.fromkeys(tail.source_paths()+[Path(__file__), *[ROOT/"src/rvlab/experiments"/name
        for name in ("_pricing_rough_bellman_control.py", "_pricing_rough_bellman_control_aligned.py", "_pricing_rough_bellman_delta.py",
            "_pricing_rough_bellman_policy.py", "_pricing_rough_stock_delta.py")]]))


def inputs():
    frozen, prior, grid, pins = tail.inputs()
    paths = [TAIL/"selection_before_final.json", BELLMAN/"selection_before_final.json", BELLMAN/"manifest.json",
        PROXY/"proxy_before_fresh_paths.json", PROXY/"proxy_coefficients.npz", PROXY/"preservation_inventory.json"]
    pins.update({str(p.relative_to(OUTPUT)): sha(p) for p in paths})
    return frozen, prior, grid, pins


def verified():
    design = read(FOLDER/"design_before_training.json")
    if any(sha(ROOT/name) != h or sha(FOLDER/"source"/name) != h for name, h in design["source_sha256"].items()): raise ValueError("Frozen control source or snapshot changed")
    if any(sha(OUTPUT/name) != h for name, h in design["input_sha256"].items()): raise ValueError("Frozen control input changed")
    return design


def load_proxy():
    seal = read(PROXY/"proxy_before_fresh_paths.json")
    if seal["model_dates"] != list(range(1, 64)) or sha(PROXY/seal["coefficient_archive"]) != seal["coefficient_archive_sha256"]:
        raise ValueError("The frozen spent-label correction coefficients changed")
    with np.load(PROXY/seal["coefficient_archive"], allow_pickle=False) as z:
        models = _decode_models(seal["models"], z)
    return (None,)+models+(None,)


def prepare():
    if FOLDER.exists(): raise FileExistsError("Preserve the previous control attempt")
    frozen, prior, grid, pins = inputs(); accepted = read(BELLMAN/"selection_before_final.json")
    benchmark = read(PROXY/"spent_benchmark.json")
    if accepted["selected_policy"] != "GH16 blend1" or benchmark["final_budget_recommendation"] != 4096:
        raise ValueError("The predeclared frozen GH16 rule or final path budget changed")
    forbidden = set(RESERVED)|set(frozen["manifest"]["all_selection_training_seeds"])
    if {TRAIN, SELECT, *FINALS}&forbidden: raise ValueError("Fresh control streams reuse prior paths")
    FOLDER.mkdir(parents=True); hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}
    for path in source_paths():
        target = FOLDER/"source"/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    design = {"version": "rough_bellman_control_v1", "created_at": now(), "source_sha256": hashes, "input_sha256": pins,
        "training": {"seed": TRAIN, "paths": 2048, "role": "bounded control coefficient fitting"},
        "selection": {"seed": SELECT, "paths": 2048, "role": "control selection only"},
        "final_streams": [{"seed": seed, "paths": 4096} for seed in FINALS],
        "targets": list(TARGETS), "nodal_european_grid": "original129 physical grid",
        "all_target_primitives_retained": True, "bins": [1, 4], "penalties": [0., 1e-4], "coefficient_sum_per_bin_max": 1.,
        "extra_shrink_candidates": False, "controls_if_all_fits_complete": 5,
        "frozen_policy": "GH16 blend1", "operator_steps": 128, "physical_steps": 128, "exercise_intervals": 64,
        "proxy_fit_seed": 1077005042, "proxy_diagnostic_seed": 1078005042, "proxy_correction_bound": .05,
        "proxy_ridge": .001, "proxy_degree": 1, "proxy_factor_clip": 3.,
        "common_control": read(TAIL/"selection_before_final.json")["selected_control"],
        "prior_control": prior, "grid_control": grid,
        "selection_rule": "max positive paired gap-reduction95percentLCB; unchanged fallback score0",
        "designated_challenger_rule": "highest nonbaseline selectionLCB regardless of fallback acceptance",
        "pure_targets_diagnostic_only": True, "no_policy_refit_or_reselection": True,
        "all_paths_retained": True, "all_exercise_dates_retained": True, "forbidden_prior_seeds": sorted(forbidden),
        "price_allowance": ".001+1e-4*abs(V)", "delta_allowance": ".002+.005*abs(delta)",
        "confidence_multiplier": 1.96, "spread_fraction": .1, "tolerances_changed": False,
        "optimal_value_ready": False, "new_control_stock_delta_validated": False, "no_new_paths_before_design": True}
    write(FOLDER/"design_before_training.json", design)
    write(FOLDER/"execution_environment.json", {"interpreter": sys.executable, "python": sys.version,
        "numpy": np.__version__, "platform": platform.platform(),
        "environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}})
    print("PREPARED", now(), len(hashes), flush=True)


def generate(seed, paths, design):
    frozen, trace, workspace = tail.generate(seed, paths)
    old = read(TAIL/"selection_before_final.json")
    _, old_data = tail.evaluate(workspace, trace, tail.build_margin_policies(frozen["selected_policy"]), old["controls"])
    common = old["controls"][old["selected_control"]]
    latest = workspace["current_martingale"]+np.einsum("ntd,d->nt", correction_basis(workspace["tail_corrections"], trace["times"], common["bins"]), common["weights"])
    forbidden = tuple(design["forbidden_prior_seeds"])+((TRAIN,) if seed != TRAIN else ())
    forbidden += ((SELECT,) if seed in FINALS else ())
    evaluator = BellmanEvaluator(frozen["proxy"], frozen["selected_policy"], orders=(16,))
    looked = bellman_policy_stops(trace, evaluator, outer_steps=128, forbidden_seeds=forbidden)
    targets, diagnostic, components = bellman_target_martingales(frozen["proxy"], frozen["selected_policy"], load_proxy(), trace,
        forbidden_seeds=forbidden, batch_size=64, return_components=True)
    stops = looked["stops"][:, 0]; raw = workspace["payoff"][np.arange(paths), stops]
    data = {**old_data, **components, "latest_martingale": latest, "target_martingales": targets,
        "gh16_stops": stops, "gh16_raw": raw, "gh16_continuation": looked["continuation"][:, :, 0],
        "gh16_computed_mask": looked["computed_mask"][:, :, 0]}
    return trace, workspace, data, {"target_control": diagnostic, "fixed_policy": looked["diagnostics"]}


def evaluate(data, payoff, times, controls):
    result = {}; cache = {}; n = len(payoff); path = np.arange(n); stops = data["gh16_stops"]
    corrections = data["target_martingales"]-data["latest_martingale"][:, :, None]
    for index, control in enumerate(controls.values()):
        bins = control["bins"]
        if bins not in cache: cache[bins] = correction_basis(corrections, times, bins)
        m = data["latest_martingale"]+np.einsum("ntd,d->nt", cache[bins], control["weights"])
        lower = data["gh16_raw"]-m[path, stops]; upper = np.max(payoff-m, axis=1)
        if np.any(upper < lower-2e-11): raise FloatingPointError("A target control violates lower/upper ordering")
        result[f"candidate{index}_lower"] = lower; result[f"candidate{index}_upper"] = upper
    return result


def training():
    design = verified()
    if (FOLDER/"controls_before_selection.json").exists(): raise FileExistsError("Training already sealed")
    start = perf_counter(); started_at = now(); print("TRAIN", TRAIN, flush=True)
    trace, workspace, data, diagnostics = generate(TRAIN, 2048, design)
    controls = {BASELINE: {"family": "fallback", "bins": 1, "penalty": 0., "weights": np.zeros(4)}}; trials = []
    corrections = data["target_martingales"]-data["latest_martingale"][:, :, None]
    for bins in (1, 4):
        basis = correction_basis(corrections, trace["times"], bins)
        for penalty in (0., 1e-4):
            name = f"target {bins} bins penalty{penalty:g}"
            weights, record = fit_correction_weights(workspace["payoff"], data["latest_martingale"], data["gh16_stops"], basis,
                bins=bins, penalty=penalty, coefficient_budget=1.)
            trials.append({"control": name, **record})
            if weights is not None: controls[name] = {"family": "target", "bins": bins, "penalty": penalty, "weights": weights}
    data.update(evaluate(data, workspace["payoff"], trace["times"], controls))
    tail.save_samples(FOLDER/"training_samples.npz", trace, workspace, data)
    pd.DataFrame(trials).to_csv(FOLDER/"control_trials.csv", index=False)
    write(FOLDER/"controls_before_selection.json", {"sealed_at": now(), "controls": controls, "training_seed": TRAIN,
        "design_sha256": sha(FOLDER/"design_before_training.json"), "selection_paths_generated_before_seal": False})
    write(FOLDER/"training_manifest.json", {"seed": TRAIN, "paths": 2048, "started_at": started_at, "completed_at": now(), "elapsed_seconds": perf_counter()-start,
        "source_sha256": design["source_sha256"], "sources_unchanged": True, "design_sha256": sha(FOLDER/"design_before_training.json"),
        "control_sha256": sha(FOLDER/"controls_before_selection.json"), "sample_sha256": sha(FOLDER/"training_samples.npz"), "diagnostics": diagnostics})
    verified(); print("TRAIN SEALED", now(), len(controls), flush=True)


def selection():
    design = verified(); fitted = read(FOLDER/"controls_before_selection.json"); controls = fitted["controls"]
    if (FOLDER/"selection_before_final.json").exists(): raise FileExistsError("Selection already sealed")
    start = perf_counter(); started_at = now(); print("SELECT", SELECT, flush=True)
    trace, workspace, data, diagnostics = generate(SELECT, 2048, design)
    data.update(evaluate(data, workspace["payoff"], trace["times"], controls))
    base_gap = data["candidate0_upper"]-data["candidate0_lower"]; rows = []
    selected, best, challenger, best_challenger = BASELINE, 0., None, -np.inf
    for index, name in enumerate(controls):
        gain = base_gap-(data[f"candidate{index}_upper"]-data[f"candidate{index}_lower"])
        mean, se, low, high = stat([gain])
        rows.append({"control": name, "paired_gap_gain": mean, "gap_gain_se": se, "gain_low": low, "gain_high": high,
            "selection_score": low, "selection_seed": SELECT, "final_paths_used_for_selection": False})
        if low > best: selected, best = name, low
        if index and low > best_challenger: challenger, best_challenger = name, low
    for row in rows: row.update(selected_control=row["control"] == selected, designated_challenger=row["control"] == challenger)
    pd.DataFrame(rows).to_csv(FOLDER/"control_selection.csv", index=False)
    tail.save_samples(FOLDER/"selection_samples.npz", trace, workspace, data)
    write(FOLDER/"selection_before_final.json", {"sealed_at": now(), "controls": controls, "control_order": list(controls),
        "selected_control": selected, "designated_challenger": challenger, "selection_seed": SELECT,
        "design_sha256": sha(FOLDER/"design_before_training.json"), "control_sha256": sha(FOLDER/"controls_before_selection.json"),
        "final_paths_generated_before_seal": False, "frozen_policy": "GH16 blend1"})
    write(FOLDER/"selection_manifest.json", {"seed": SELECT, "paths": 2048, "started_at": started_at, "completed_at": now(), "elapsed_seconds": perf_counter()-start,
        "source_sha256": design["source_sha256"], "sources_unchanged": True, "design_sha256": sha(FOLDER/"design_before_training.json"),
        "selection_sha256": sha(FOLDER/"selection_before_final.json"), "sample_sha256": sha(FOLDER/"selection_samples.npz"), "diagnostics": diagnostics})
    verified(); print("SELECT SEALED", selected, challenger, now(), flush=True)


def final(index):
    design = verified(); seal = read(FOLDER/"selection_before_final.json"); spec = design["final_streams"][index]
    if (FOLDER/f"final_{index}_samples.npz").exists(): raise FileExistsError("Final stream already exists")
    start = perf_counter(); started_at = now(); print("FINAL", index, spec["seed"], flush=True)
    trace, workspace, data, diagnostics = generate(spec["seed"], spec["paths"], design)
    data.update(evaluate(data, workspace["payoff"], trace["times"], seal["controls"]))
    tail.save_samples(FOLDER/f"final_{index}_samples.npz", trace, workspace, data)
    write(FOLDER/f"final_{index}_manifest.json", {"index": index, **spec, "started_at": started_at, "completed_at": now(), "elapsed_seconds": perf_counter()-start,
        "source_sha256": design["source_sha256"], "sources_unchanged": True, "design_sha256": sha(FOLDER/"design_before_training.json"),
        "selection_sha256": sha(FOLDER/"selection_before_final.json"), "sample_sha256": sha(FOLDER/f"final_{index}_samples.npz"),
        "control_order": list(seal["controls"]), "diagnostics": diagnostics})
    verified(); print("FINAL COMPLETE", index, now(), flush=True)


def validate_completed_stages(design):
    """Reject broken stage chains before declaring a standalone run complete."""
    design_sha = sha(FOLDER/"design_before_training.json")
    fitted = read(FOLDER/"controls_before_selection.json"); seal = read(FOLDER/"selection_before_final.json")
    controls = fitted["controls"]
    if (fitted["design_sha256"] != design_sha or fitted["training_seed"] != TRAIN
            or fitted["selection_paths_generated_before_seal"] is not False
            or seal["design_sha256"] != design_sha or seal["control_sha256"] != sha(FOLDER/"controls_before_selection.json")
            or seal["selection_seed"] != SELECT or seal["controls"] != controls
            or seal["control_order"] != list(controls) or seal["frozen_policy"] != "GH16 blend1"
            or seal["final_paths_generated_before_seal"] is not False
            or seal["selected_control"] not in controls
            or seal["designated_challenger"] not in list(controls)[1:]):
        raise ValueError("Control training/selection seal or policy identity changed")
    records = []
    declarations = [("training", TRAIN, 2048, "control", fitted["sealed_at"]),
        ("selection", SELECT, 2048, "selection", seal["sealed_at"])]
    for stage, seed, paths, link, sealed in declarations:
        record = read(FOLDER/f"{stage}_manifest.json")
        if (record["seed"] != seed or record["paths"] != paths
                or record["source_sha256"] != design["source_sha256"] or record["sources_unchanged"] is not True
                or record["design_sha256"] != design_sha
                or record[f"{link}_sha256"] != sha(FOLDER/f"{('controls_before_selection' if link == 'control' else 'selection_before_final')}.json")
                or record["sample_sha256"] != sha(FOLDER/f"{stage}_samples.npz")
                or not design["created_at"] < record["started_at"] < sealed <= record["completed_at"]
                or record["elapsed_seconds"] <= 0):
            raise ValueError(f"Broken {stage} source, sample, chronology or seal chain")
    if not fitted["sealed_at"] < read(FOLDER/"selection_manifest.json")["started_at"]:
        raise ValueError("Selection paths preceded the coefficient seal")
    for index, spec in enumerate(design["final_streams"]):
        record = read(FOLDER/f"final_{index}_manifest.json")
        if (record["index"] != index or record["seed"] != spec["seed"] or record["paths"] != spec["paths"]
                or record["source_sha256"] != design["source_sha256"] or record["sources_unchanged"] is not True
                or record["design_sha256"] != design_sha
                or record["selection_sha256"] != sha(FOLDER/"selection_before_final.json")
                or record["sample_sha256"] != sha(FOLDER/f"final_{index}_samples.npz")
                or record["control_order"] != list(controls)
                or not seal["sealed_at"] < record["started_at"] < record["completed_at"]
                or record["elapsed_seconds"] <= 0):
            raise ValueError(f"Broken final{index} source, sample, chronology or selection chain")
        records.append(record)
    return seal, records


def collect():
    design = verified(); seal, records = validate_completed_stages(design)
    if (FOLDER/"manifest.json").exists(): raise FileExistsError("Preserve the completed campaign")
    write(FOLDER/"manifest.json", {**design, "complete": True, "completed_at": now(),
        "completed_final_paths": sum(v["paths"] for v in design["final_streams"]),
        "records": records, "selected_control": seal["selected_control"], "designated_challenger": seal["designated_challenger"],
        "control_order": list(seal["controls"]), "selection_sha256": sha(FOLDER/"selection_before_final.json"),
        "training_manifest_sha256": sha(FOLDER/"training_manifest.json"), "selection_manifest_sha256": sha(FOLDER/"selection_manifest.json")})
    entries = [{"path": str(path.relative_to(FOLDER)), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(FOLDER.rglob("*")) if path.is_file()]
    write(FOLDER/"preservation_inventory.json", {"files": entries,
        "inventory_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    print("COLLECTED", now(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "train", "select", "final", "collect"))
    parser.add_argument("--index", type=int, choices=(0, 1)); args = parser.parse_args()
    if args.stage == "final" and args.index is None: parser.error("final requires --index")
    {"prepare": prepare, "train": training, "select": selection, "final": lambda: final(args.index), "collect": collect}[args.stage]()
