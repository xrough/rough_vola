"""Independent adaptive integration on a predeclared spent-state Bellman panel."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rvlab.experiments._pricing_rough_bellman_audit import adaptive_block_expectation, clipped_premium_breakpoints
from rvlab.experiments._pricing_rough_bellman_policy import BellmanEvaluator
from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True))
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"))


def endpoint_envelope(evaluator, j, law):
    """Conservative A+B*S envelope for the actual finite Fourier proxy."""
    proxy = evaluator.proxy; k = proxy.strike
    if j == len(evaluator.times)-1:
        return (0., 1.) if proxy.is_call else (k, 0.)
    time = float(evaluator.times[j]); affine = evaluator.affine
    a, b = affine.coefficients[(proxy.maturity, time)]
    c, constant = affine._remaining_integral(time, proxy.maturity)
    factors = law["new_factors"]
    remaining = factors@c+constant
    if np.any(remaining < -2e-13): raise ValueError("A Fourier envelope has negative remaining variance")
    remaining = np.maximum(remaining, 0.)
    square = affine.frequencies**2+.25
    coefficient = np.sum((np.abs(np.exp(a+factors@b.T))+np.exp(-.5*remaining[..., None]*square))
        *affine.quadrature/square, axis=-1)
    ratio = float(proxy.market.discount_curve.discount(proxy.maturity)/evaluator.discounts[j])
    # The absolute Fourier correction is at most C*sqrt(S), then use
    # sqrt(S)<=(1+S)/2. The BS put baseline is at most K*ratio.
    bound = float(np.max(np.sqrt(ratio*k)*coefficient/np.pi))
    cap = evaluator.selected.premium_cap
    return (cap+bound/2, 1.+bound/2) if proxy.is_call else (max(k, k*ratio+cap)+bound/2, bound/2)


def panel_rows(output, bundle):
    folder = output / "rough_tail_attribution_v1/joint_switch_v1"
    proxy = bundle["proxy"]; weights = np.asarray(proxy.kernel.weights)
    records = []
    for index, seed in enumerate((1054005042, 1055005042)):
        with np.load(folder / f"final_{index}_samples.npz", allow_pickle=False) as z:
            stocks, factors, margin = z["stocks"], z["factors"], z["tail_margin"]
            for j in (16, 32, 48, 63):
                intrinsic = np.maximum((1 if proxy.is_call else -1)*(stocks[:, j, 0]-proxy.strike), 0.)
                eligible = np.flatnonzero(intrinsic > 0)
                if len(eligible) < 4: raise ValueError("The predeclared spent panel lacks four ITM states")
                chosen = [("first ITM", int(eligible[0])), ("second ITM", int(eligible[1]))]
                remaining = np.setdiff1d(eligible, [x[1] for x in chosen])
                nearest = sorted(remaining, key=lambda p: (abs(margin[p, j]), p))[0]
                chosen.append(("nearest frozen boundary", int(nearest)))
                remaining = np.setdiff1d(remaining, [nearest])
                variance = proxy.params.v0+factors[:, j]@weights
                largest = sorted(remaining, key=lambda p: (-variance[p], p))[0]
                chosen.append(("largest variance", int(largest)))
                for role, path in chosen:
                    records.append({"state_index": len(records), "seed": seed, "source_index": index,
                        "path_index": path, "exercise_index": j, "time": float(proxy.exercise_times[j]),
                        "role": role, "stock": float(stocks[path, j, 0]), "variance": float(variance[path]),
                        "frozen_margin": float(margin[path, j]), "factors": factors[path, j].tolist()})
    return records


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.limit <= 32: raise ValueError("The fixed panel has32 states")
    output = ROOT / "notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder = output / "rough_bellman_policy_v1/independent_adaptive_audit_v1"
    bundle_folder = output / "rough_larger_policy_v1/32768_paths_two_starts"
    spent = output / "rough_tail_attribution_v1/joint_switch_v1"
    inputs = [bundle_folder/"frozen_policy_before_final.json", bundle_folder/"frozen_policy_coefficients.npz",
        spent/"manifest.json", spent/"selection_before_final.json"]
    inputs += [spent/f"final_{i}_{suffix}" for i in range(2) for suffix in ("manifest.json", "samples.npz")]
    sources = list((ROOT/"src/rvlab/pricing").glob("*.py"))
    sources += [Path(__file__).resolve()]
    sources += [ROOT/"src/rvlab/experiments"/name for name in ("_pricing_rough_bellman_audit.py",
        "_pricing_rough_bellman_policy.py", "_pricing_rough_exercise_block.py", "_pricing_rough_large_policy.py",
        "_pricing_rough_frozen_policy.py")]
    input_hashes = {str(p.relative_to(output)): sha(p) for p in inputs}
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    bundle = load_larger_policy_bundle(bundle_folder)
    panel = panel_rows(output, bundle)
    design = {"version": "rough_bellman_independent_adaptive_audit_v1", "state_panel": panel,
        "spent_seeds": [1054005042, 1055005042], "dates": [16, 32, 48, 63], "states": 32,
        "selection_rule": "Per seed/date: first2ITM, smallest absolute stored frozen margin among remainingITM, largest aggregate variance among remainingITM; path-index ties.",
        "source_sha256": source_hashes, "input_sha256": input_hashes, "orders": [8, 16, 32],
        "normal_limit": 12., "absolute_tolerance": 1e-9, "relative_tolerance": 1e-10,
        "state_wall_limit_seconds": 180., "new_outer_paths": 0, "new_fits": 0,
        "allowance": ".001+1e-4*abs(C)", "tolerances_changed": False,
        "used_for_formal_policy_selection": False, "quadrature_error_is_rigorous_bound": False,
        "only_premium_capped": True, "proxy_policy_improvement_guaranteed": False,
        "normal_tail_value_envelope": "Analytic finite-Fourier absolute envelope C*sqrt(S), enlarged by sqrt(S)<=(1+S)/2; no tail renormalization."}
    path = folder/"design_before_evaluation.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != "created_at"} != design:
            raise ValueError("Preserve changed audit source, inputs or panel in a separate study")
        design = previous
    else:
        folder.mkdir(parents=True, exist_ok=True)
        design["created_at"] = datetime.now(timezone.utc).isoformat(); write(path, design)
        for source in sources:
            target = folder/"source"/source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        pd.DataFrame(panel).to_csv(folder/"panel.csv", index=False)
    evaluator = BellmanEvaluator(bundle["proxy"], bundle["selected_policy"], batch_size=8)
    design_digest = sha(path)
    def alarm(_signal, _frame): raise TimeoutError("Predeclared180-second independent integration budget exhausted")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    for state in panel[:args.limit]:
        record_path = folder/f"state_{state['state_index']:02d}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("design_sha256") != design_digest or record.get("state") != state:
                raise ValueError("Changed cached independent state audit")
            continue
        start = perf_counter(); record = {"state": state, "design_sha256": design_digest, "rows": []}
        try:
            signal.setitimer(signal.ITIMER_REAL, design["state_wall_limit_seconds"])
            j = state["exercise_index"]; factors = np.array([state["factors"]]); stocks = np.array([state["stock"]])
            a, b = np.searchsorted(evaluator.grid, evaluator.times[j:j+2])
            law = exercise_block_law(evaluator.proxy, factors, times=evaluator.grid[a:b+1])
            values = evaluator.conditional_expectations(j, stocks, factors)
            right = j+1; model = evaluator.selected.continuations[right]
            constant, linear = endpoint_envelope(evaluator, right, law)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                reference = adaptive_block_expectation(law, stocks,
                    lambda s, u: evaluator.endpoint_values(right, s, u),
                    kink_function=lambda s, u: evaluator.endpoint_components(right, s, u)["kink"],
                    stock_breakpoints=lambda u: clipped_premium_breakpoints(model, u, evaluator.proxy.strike, evaluator.selected.premium_cap),
                    normal_limit=12., absolute_tolerance=1e-9, relative_tolerance=1e-10,
                    envelope_constant=constant, envelope_stock=linear)
            ratio = evaluator.discounts[right]/evaluator.discounts[j]
            expected = float(reference["expectation"][0]*ratio)
            error = float(reference["quadrature_error_estimate"][0]*ratio)
            tail = float(reference["tail_value_bound"][0]*ratio)
            for k, order in enumerate(evaluator.orders):
                cv, direct = float(values["continuation"][0, k]), float(values["direct_continuation"][0, k])
                allowance = .001+1e-4*abs(expected)
                record["rows"].append({"state_index": state["state_index"], "seed": state["seed"],
                    "path_index": state["path_index"], "exercise_index": j, "role": state["role"], "order": order,
                    "reference_continuation": expected, "cv_continuation": cv, "direct_continuation": direct,
                    "cv_minus_reference": cv-expected, "direct_minus_reference": direct-expected,
                    "quadrature_error_estimate": error, "omitted_tail_value_bound": tail,
                    "price_allowance": allowance, "cv_within_dollar_allowance": abs(cv-expected)+error+tail <= allowance,
                    "direct_within_dollar_allowance": abs(direct-expected)+error+tail <= allowance,
                    "terminal_exact_payoff": right == len(evaluator.times)-1, "status": "completed",
                    "used_for_formal_policy_selection": False, "quadrature_error_is_rigorous_bound": False})
            record.update(status="completed", detected_snell_crossings=int(reference["detected_snell_crossings"][0]),
                function_evaluations=int(reference["function_evaluations"][0]), warnings=[str(w.message) for w in captured],
                envelope_constant=constant, envelope_stock=linear, numerical_diagnostics=values["diagnostics"])
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error_message=str(error))
            record["rows"] = [{"state_index": state["state_index"], "seed": state["seed"],
                "path_index": state["path_index"], "exercise_index": state["exercise_index"], "role": state["role"],
                "order": order, "status": "failed", "error_type": type(error).__name__, "error_message": str(error),
                "used_for_formal_policy_selection": False} for order in evaluator.orders]
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
        record["elapsed_seconds"] = perf_counter()-start
        # Convert numpy diagnostics to ordinary JSON without lossy reprs.
        record = json.loads(json.dumps(record, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()))
        write(record_path, record)
        print(f"State{state['state_index']:02d} {record['status']} {record['elapsed_seconds']:.3f}s", flush=True)
    signal.signal(signal.SIGALRM, old_alarm)
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("state_*.json"))]
    pd.DataFrame([row for record in records for row in record["rows"]]).to_csv(folder/"checks.csv", index=False)
    pd.DataFrame([{k: v for k, v in record.items() if k not in ("rows", "state", "numerical_diagnostics")}
        | {k: v for k, v in record["state"].items() if k != "factors"} for record in records]).to_csv(folder/"state_diagnostics.csv", index=False)
    if source_hashes != {str(p.relative_to(ROOT)): sha(p) for p in sources}:
        raise ValueError("Audit source changed during independent integration; retain all observations as superseded")
    if input_hashes != {str(p.relative_to(output)): sha(p) for p in inputs}: raise ValueError("Spent input changed during audit")
    if len(records) == 32:
        write(folder/"manifest.json", {**design, "design_sha256": design_digest, "complete": True,
            "successful_states": sum(r["status"] == "completed" for r in records),
            "failed_states": sum(r["status"] == "failed" for r in records), "all_predeclared_states_retained": True})
        entries = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "preservation_inventory.json"]
        digest = hashlib.sha256(canonical(entries).encode()).hexdigest()
        write(folder/"preservation_inventory.json", {"files": entries, "inventory_sha256": digest})
        print(f"Preserved {len(entries)}files: {digest}")


if __name__ == "__main__": main()
