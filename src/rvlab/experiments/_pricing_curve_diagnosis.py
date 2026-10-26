"""Focused synthetic Heston-curve identification and optimizer diagnostics.

Profile trials reoptimize six nuisance coefficients on the original frozen
training mesh. They are conditional numerical objectives, not confidence
intervals, global minima, or independently refined accepted calibrations.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import platform
import runpy
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

from rvlab.pricing import FDConfig, OptimizerConfig, PricingError, evaluate_vanilla
from rvlab.pricing import parameter_values, replace_parameters
from rvlab.pricing._vanilla_common import build_grid
from rvlab.pricing.optimization import _Coordinates, _finite_difference_jacobian
from . import _pricing_american_identification as joint

VERSION = "american_curve_diagnosis_v1"
LEVELS = {"clean": (.0025, .02, .045, .08),
          "noisy": (0., .0025, .02, .045, .08)}


def _load_baseline(directory, dataset):
    """Load and verify the exact corrected-grid baseline, never an old fit."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.glob(f"heston_curve_{dataset}_*_exploratory.json")):
        record = json.loads(path.read_text())
        if (record["fingerprint"] != joint._fingerprint(record["specification"]) or
                record["specification"]["engine_sha256"] != joint._engine_fingerprint()):
            raise ValueError(f"Incompatible source/input fingerprint: {path}")
        records.append(record)
    if len(records) != 3:
        raise FileNotFoundError(f"Expected three completed corrected-grid curve starts for {dataset} in {directory}")
    chosen = joint.select_training_attempt([record["payload"]["summary"] for record in records])
    record = next(record for record in records if record["payload"]["summary"]["start"] == chosen["start"])
    return record, records


class _ProfileProblem:
    """The shared residual/Jacobian machinery without costly acceptance probes."""

    def __init__(self, record, level):
        spec = record["specification"]
        self.quotes = pd.DataFrame(spec["quotes"])
        self.training = self.quotes.loc[self.quotes.split.eq("train")]
        self.market = joint.joint_market()
        self.truth = joint.joint_specifications()[0]["heston_curve"]
        self.config = FDConfig(**spec["fit_config"])
        self.optimizer = OptimizerConfig(**spec["optimizer"])
        self.bounds = spec["bounds"]
        self.names = tuple(name for name in parameter_values(self.truth) if name != "theta_2")
        self.lower = {name: self.bounds[name][0] for name in self.names}
        self.upper = {name: self.bounds[name][1] for name in self.names}
        self.coords = _Coordinates(self.names, self.lower, self.upper, self.optimizer)
        self.grid = build_grid(self.market, self.training.strike.to_numpy(), self.config, variance=True)
        self.grid_sha256 = hashlib.sha256(self.grid.spot.tobytes()+self.grid.variance.tobytes()).hexdigest()
        self.level = float(level)
        self.scales = np.maximum((self.training.ask-self.training.bid).to_numpy()/2, joint.SPREAD_FLOOR)
        self.targets = (self.training.ask+self.training.bid).to_numpy()/2
        self.count = 0
        self.cached_x = self.cached_y = None

    def prices(self, parameters, all_contracts=False):
        frame = self.quotes if all_contracts else self.training
        values = evaluate_vanilla(parameters, market=self.market, strike=frame.strike.to_numpy(),
            maturity=frame.maturity.to_numpy(), is_call=frame.is_call.to_numpy(),
            exercise_style="american", config=self.config, grid=self.grid)
        self.count += values.diagnostics["pricing_evaluations"]
        if values.diagnostics["grid_sha256"] != self.grid_sha256:
            raise ValueError("Profile evaluation changed the frozen training grid")
        return np.asarray(values.prices)

    def parameters(self, physical):
        return replace_parameters(self.truth, theta_2=self.level, **dict(zip(self.names, physical)))

    def physical_residuals(self, physical):
        return (self.prices(self.parameters(physical))-self.targets)/self.scales

    def residuals(self, coordinates):
        result = self.physical_residuals(self.coords.decode(coordinates))
        self.cached_x, self.cached_y = coordinates.copy(), result.copy()
        return result

    def jacobian(self, coordinates):
        base = self.cached_y if np.array_equal(coordinates, self.cached_x) else None
        result, _ = _finite_difference_jacobian(self.physical_residuals,
            self.coords.decode(coordinates), lower=self.coords.lower, upper=self.coords.upper,
            method=self.optimizer.jacobian, base=base)
        return result*self.coords.derivative(coordinates)


def _profile_attempt(record, level, initial, max_nfev):
    begin = time.perf_counter()
    problem = _ProfileProblem(record, level)
    row = {"dataset": record["specification"]["dataset"], "theta_2": float(level),
        "bound_case": "nonnegative-level sensitivity" if level < .0025 else "original bounds",
        "kind": "six-nuisance-parameter reoptimization", "optimizer_success": False,
        "status": "numerical_failure", "nfev": 0, "pricing_evaluations": 0,
        "numerically_refined": False, "accepted_calibration": False, "error": None}
    prediction = []
    try:
        start = np.asarray([initial[name] for name in problem.names])
        fit = least_squares(problem.residuals, problem.coords.encode(start),
            jac=problem.jacobian, bounds=problem.coords.bounds, method="trf", x_scale="jac",
            max_nfev=max_nfev, ftol=problem.optimizer.ftol, xtol=problem.optimizer.xtol,
            gtol=problem.optimizer.gtol)
        params = problem.parameters(problem.coords.decode(fit.x))
        prices = problem.prices(params, all_contracts=True)
        observed = problem.quotes.market_price.to_numpy()
        clean = problem.quotes.clean_price.to_numpy()
        mask = problem.quotes.split.eq("train").to_numpy()
        row.update(params=parameter_values(params), optimizer_success=bool(fit.success),
            status="converged" if fit.success else "exhausted" if fit.status == 0 else "failed",
            optimizer_status=int(fit.status), optimizer_message=str(fit.message),
            nfev=int(fit.nfev), njev=int(fit.njev), objective=float(np.mean(fit.fun**2)),
            optimality=float(fit.optimality),
            physical_optimality=problem.coords.physical_optimality(fit.x, fit.jac, fit.fun),
            active_bounds={name: "lower" if side < 0 else "upper"
                for name, side in zip(problem.names, fit.active_mask) if side},
            train_rmse=float(np.sqrt(np.mean((prices[mask]-observed[mask])**2))),
            heldout_clean_rmse=float(np.sqrt(np.mean((prices[~mask]-clean[~mask])**2))),
            grid_sha256=problem.grid_sha256)
        prediction = problem.quotes.assign(model_price=prices,
            price_residual=prices-observed, clean_price_error=prices-clean).to_dict("records")
    except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(elapsed_seconds=time.perf_counter()-begin, pricing_evaluations=problem.count)
    return {"summary": row, "predictions": prediction}


def profile_dataset(baseline_directory, output_directory, dataset, *, max_nfev=40, reuse=True):
    """One warm-start path, selected without held-out errors; failures persist."""
    if dataset not in LEVELS:
        raise ValueError("dataset must be clean or noisy")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or not 1 <= max_nfev <= 80:
        raise ValueError("profile max_nfev must be an integer in [1,80]")
    record, _ = _load_baseline(baseline_directory, dataset)
    selected = record["payload"]["summary"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    levels = sorted(set((*LEVELS[dataset], selected["params"]["theta_2"])))
    # Start next to the chosen optimum, then traverse outward. Selection here
    # controls solver initialization only; held-out observations never enter.
    levels.sort(key=lambda value: (abs(value-selected["params"]["theta_2"]), value))
    available = {selected["params"]["theta_2"]: dict(selected["params"])}
    rows, predictions = [], []
    numeric_source = "\n".join(inspect.getsource(part) for part in (_ProfileProblem, _profile_attempt, profile_dataset))
    for level in levels:
        baseline_point = level == selected["params"]["theta_2"]
        if baseline_point:
            rows.append({"dataset": dataset, "theta_2": level, "bound_case": "original bounds",
                "kind": "selected seven-parameter baseline", "optimizer_success": selected["optimizer_success"],
                "status": selected["status"], "objective": selected["training_objective"],
                "params": selected["params"], "nfev": 0, "pricing_evaluations": 0,
                "elapsed_seconds": 0., "checkpoint_reused": True,
                "numerically_refined": selected["price_stable"], "accepted_calibration": selected["ready"]})
            continue
        nearest = min(available, key=lambda value: (abs(value-level), value))
        previous = dict(available[nearest])
        specification = {"version": VERSION, "baseline_fingerprint": record["fingerprint"],
            "engine_sha256": joint._engine_fingerprint(), "theta_2": level,
            "initial": previous, "max_nfev": max_nfev,
            "source_sha256": hashlib.sha256(numeric_source.encode()).hexdigest(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "pandas": pd.__version__}}
        key = joint._fingerprint(specification)
        path = output_directory/f"{dataset}_{key[:16]}.json"
        payload, cached = joint._cached(path, specification,
            lambda: _profile_attempt(record, level, previous, max_nfev), reuse)
        row = {**payload["summary"], "checkpoint_reused": cached, "checkpoint": str(path)}
        rows.append(row)
        predictions.extend({"dataset": dataset, "theta_2": level, **item} for item in payload["predictions"])
        if row.get("params") is not None:
            available[level] = row["params"]
        print(f"Curve profile {dataset}: theta_2={level:g}, {row['status']}, "
              f"objective={row.get('objective')}, cache={cached}", flush=True)
    result = {"summary": pd.DataFrame(rows), "predictions": pd.DataFrame(predictions)}
    for name, frame in result.items():
        frame.to_csv(output_directory/f"{dataset}_{name}.csv", index=False)
    return result


def _restart_tables(root, records):
    rows, predictions, refinements = [], [], []
    for record in records:
        spec, payload = record["specification"], record["payload"]
        old = json.loads((root/spec["baseline_path"]).read_text())["payload"]
        before, after = old["summary"], payload["summary"]
        a, b = pd.DataFrame(old["predictions"]), pd.DataFrame(payload["predictions"])
        move = delta_move = np.nan
        if not b.empty:
            if a.contract_id.tolist() != b.contract_id.tolist():
                raise ValueError("Restart predictions changed contract ordering")
            move = float(np.max(np.abs(b.model_price-a.model_price)))
            delta_move = float(np.max(np.abs(b.delta-a.delta)))
        rows.append({"dataset": after["dataset"], "start": after["start"],
            "before_status": before["status"], "status": after["status"],
            "before_objective": before["training_objective"], "after_objective": after["training_objective"],
            "before_physical_optimality": before["physical_optimality"],
            "after_physical_optimality": after["physical_optimality"],
            "optimizer_status": after.get("optimizer_status"),
            "optimizer_message": after.get("optimizer_message"),
            "nfev": after["nfev"], "pricing_evaluations": after["pricing_evaluations"],
            "elapsed_seconds": after["elapsed_seconds"], "optimizer_success": after["optimizer_success"],
            "price_stable": after["price_stable"], "delta_stable": after["delta_stable"],
            "spread_stable": after["spread_stable"], "ready": after["ready"],
            "active_bounds": after["active_bounds"], "max_price_move": move,
            "max_delta_move": delta_move, "params": after["params"],
            "training_grid_matches": before.get("training_grid_sha256") == after.get("training_grid_sha256")})
        predictions.extend({"dataset": after["dataset"], "start": after["start"], **item}
                           for item in payload["predictions"])
        refinements.extend({"dataset": after["dataset"], "start": after["start"], **item}
                           for item in payload["refinement"])
    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(refinements)


def _diagnosis_figures(tables):
    import matplotlib.pyplot as plt
    figures = {}
    restarts = tables["restarts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for i, row in enumerate(restarts.itertuples()):
        label = f"{row.dataset} · start {row.start}"
        axes[0].plot([0, 1], [row.before_objective, row.after_objective], "o-", label=label)
        axes[1].bar(i, row.nfev, color=("#4c956c" if row.ready else "#c55a42"))
        axes[1].text(i, row.nfev+.25, row.status, ha="center", fontsize=8)
    axes[0].set(xticks=[0, 1], xticklabels=["Original 80-evaluation endpoint", "Parameter restart"],
                ylabel="Mean squared spread-normalized residual", title="Small objective changes, different stopping flags")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=np.arange(len(restarts)),
        xticklabels=[f"{r.dataset} · {r.start}" for r in restarts.itertuples()],
        ylabel="Additional optimizer evaluations", title="Restarted state; same pricing objective")
    axes[1].set_ylim(0, max(1., float(restarts.nfev.max()))*1.3)
    figures["restarts"] = fig

    profiles = tables["profiles"]
    slices = tables["conditional_slices"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, dataset in zip(axes, ("clean", "noisy")):
        part = profiles.loc[profiles.dataset.eq(dataset)].sort_values("theta_2")
        local = slices.loc[slices.point.eq("selected_"+dataset)].sort_values("theta_2")
        ax.plot(local.theta_2, local.objective, "o--", alpha=.6, label="Others fixed: conditional slice")
        ax.plot(part.theta_2, part.objective, "o-", label="Six nuisance coefficients reoptimized")
        exhausted = part.loc[~part.optimizer_success.fillna(False)]
        if not exhausted.empty:
            ax.scatter(exhausted.theta_2, exhausted.objective, marker="x", s=100, color="#c55a42", label="Budget exhausted / failed")
        baseline = part.loc[part.kind.eq("selected seven-parameter baseline")]
        ax.scatter(baseline.theta_2, baseline.objective, s=100, facecolors="none", edgecolors="black", zorder=5, label="Selected original fit")
        ax.axvline(.0025, color="gray", linestyle=":", lw=1, label="Original lower bound (.0025)")
        ax.axvline(.045, color="#4c956c", linestyle=":", lw=1, label="Generating level (.045)")
        ax.set(title=dataset.title(), xlabel="Final variance level theta_2",
               ylabel="Mean squared spread-normalized residual")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("Compensation changes the question: a slice is not a profile")
    figures["profiles"] = fig

    gradients = tables["gradients"]
    sensitivity = tables["sensitivities"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    labels = gradients.point.str.replace("selected_", "", regex=False)
    axes[0].bar(labels, gradients.theta2_nuisance_projected_fraction*100)
    axes[0].set(ylabel="Sensitivity norm remaining (%)", title="Other coefficients absorb most theta_2 sensitivity")
    axes[1].bar(labels, gradients.theta2_local_noise_sd, label="Local unconstrained noise SD")
    axes[1].axhline(.045, color="#4c956c", linestyle="--", label="Generating theta_2 = .045")
    axes[1].set(ylabel="Variance-level units", title="Linearized noise amplification; no confidence interval")
    axes[1].legend(fontsize=8)
    figures["identification"] = fig
    return figures


def curve_diagnosis(context, baseline_manifest, *, max_nfev=40, reuse=True):
    """Reproduce/load the focused follow-up on the preserved default study.

    The example scripts are maintained same-checkout reproducers, not sibling
    libraries. Their own input/source fingerprints gate every reused result.
    This follow-up is deliberately tied to the published corrected-grid study.
    """
    if not isinstance(reuse, (bool, np.bool_)):
        raise ValueError("curve diagnosis reuse must be boolean")
    root = Path(__file__).resolve().parents[3]
    baseline_directory = Path(baseline_manifest["checkpoint_directory"]).resolve()
    expected = root/"notebooks/_output/pricing/research_v1/02_synthetic_calibration"/joint.STUDY_VERSION/"837b8286fe4df229"
    if baseline_directory != expected.resolve() or not baseline_manifest["run_fingerprint"].startswith("837b8286fe4df229"):
        raise ValueError("Curve diagnosis requires the published corrected-grid baseline; custom study settings need a separately declared diagnosis")
    destination = Path(context.output_dir)/VERSION
    optimizer_dir, identification_dir = destination/"optimizer", destination/"identification"
    optimizer_script = root/"examples/american_curve_optimizer_diagnostic.py"
    identification_script = root/"examples/american_curve_identification_diagnostic.py"
    # run_path loads only functions: neither script's CLI main executes.
    optimizer_api = runpy.run_path(str(optimizer_script))
    identification_api = runpy.run_path(str(identification_script))
    restart_records = [optimizer_api["run"](root, attempt, optimizer_dir, reuse=reuse)
                       for attempt in ("clean-1", "clean-2", "noisy-2")]
    if not reuse:
        identification_dir = identification_dir/f"recompute_{time.time_ns()}"
    local = identification_api["run_diagnostics"](baseline_directory, identification_dir)
    profile_results = [profile_dataset(baseline_directory, destination/"profiles", dataset,
                       max_nfev=max_nfev, reuse=reuse) for dataset in ("clean", "noisy")]
    restarts, restart_predictions, refinement = _restart_tables(root, restart_records)
    points = local["points"]
    gradients = pd.DataFrame([{"point": p["point"],
        "theta2_gradient": p["weighted_gradient_half_sum_squares"]["theta_2"],
        "physical_kkt_infinity_norm": p["physical_kkt_infinity_norm"],
        "max_relative_two_three_point_difference": p["max_relative_two_three_point_difference"],
        "theta2_nuisance_projected_fraction": p["theta2_nuisance_projected_fraction"],
        "theta2_local_noise_sd": p["local_parameter_noise_sd"]["theta_2"],
        "column_normalized_condition_number": p["column_normalized_condition_number"]} for p in points])
    tables = {"restarts": restarts, "restart_predictions": restart_predictions,
        "restart_refinement": refinement, "gradients": gradients,
        "profiles": pd.concat([p["summary"] for p in profile_results], ignore_index=True),
        "profile_predictions": pd.concat([p["predictions"] for p in profile_results], ignore_index=True),
        "sensitivities": pd.read_csv(identification_dir/"local_sensitivities.csv"),
        "conditional_slices": pd.read_csv(identification_dir/"conditional_theta2_slices.csv"),
        "noise": pd.read_csv(identification_dir/"linearized_noise.csv")}
    # A derived report may itself contain evidence hashes; do not hash a
    # previous reporting manifest into its own replacement.
    evidence_paths = sorted(p for p in destination.rglob("*.json") if p.name != "report_manifest.json")
    manifest = {"study_version": VERSION, "baseline_run_fingerprint": baseline_manifest["run_fingerprint"],
        "engine_sha256": joint._engine_fingerprint(), "profile_max_nfev": max_nfev,
        "profile_levels": LEVELS, "restart_budget": 80, "unchanged_acceptance": baseline_manifest["acceptance"],
        "local_identification": local,
        "restart_executions": [record["execution"] for record in restart_records],
        "restart_pricing_evaluations": int(restarts.pricing_evaluations.sum()),
        "profile_pricing_evaluations": int(tables["profiles"].pricing_evaluations.sum()),
        "local_pricing_evaluations": int(sum(p["pricing_evaluations"] for p in points)),
        "evidence": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in evidence_paths],
        "reproducers": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in (optimizer_script, identification_script)],
        "limitations": ["Parameter restarts reset trust-region and historical Jacobian scaling; they do not resume internal optimizer state.",
            "Reoptimized points are one warm-start path, not certified global minima. Exhaustion remains visible.",
            "Profile trials have no independent refinement and cannot be called accepted calibration fits.",
            "Fixed-other slices and local nuisance projections are not reoptimized profiles.",
            "Spread weights differ from the simulated homoscedastic noise; objectives are not likelihood-ratio statistics.",
            "Linearized unconstrained noise draws can leave physical bounds; their frequencies are not nonlinear calibration boundary frequencies.",
            "Only the theta_2=0 sensitivity changes the original level bound; production bounds and numerical acceptance remain unchanged."]}
    return {"tables": tables, "figures": _diagnosis_figures(tables), "manifest": manifest}
