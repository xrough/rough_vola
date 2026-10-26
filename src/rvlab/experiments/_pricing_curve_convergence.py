"""Reproducible follow-up of the SPY variance-curve convergence experiment.

This is research orchestration, not a change to the market-data readiness
rules. Historical notebook-06 estimates below are declared starting values,
not loaded fit artifacts and not selected using held-out prices. Checkpoints
are optional accelerators: a fresh checkout can execute every trial itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from rvlab.pricing import PricingConfig, PricingError, parameter_values
from rvlab.pricing.variance_curve import (
    HestonCurveParams, MeanReversionCurve, RoughHestonCurveParams,
)
from ._pricing_data import load_spy_demo
from ._pricing_exercise import exercise_adjusted_surface
from ._pricing_fits import fit_attempt
from ._pricing_spy import _fingerprints


@dataclass(frozen=True)
class RecordedTrial:
    """The portable subset needed after a cached or freshly computed trial."""

    label: str
    params: HestonCurveParams | RoughHestonCurveParams | None
    summary: dict
    predictions: pd.DataFrame
    checkpoint_reused: bool = False


def _model_from_dict(values, model):
    if values is None:
        return None
    values = dict(values)
    curve = values.pop("curve")
    values["curve"] = MeanReversionCurve(tuple(curve["knot_times"]), tuple(curve["levels"]))
    cls = RoughHestonCurveParams if model == "rough_heston" else HestonCurveParams
    return cls(**values)


def _canonical(value):
    from .pricing import _jsonable
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _frame_fingerprint(quotes):
    """Hash all row values, order, dtypes and labels; no partial quote cache."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(quotes, index=True).to_numpy().tobytes())
    digest.update(_canonical({"columns": list(quotes.columns),
        "dtypes": [str(dtype) for dtype in quotes.dtypes]}).encode())
    return digest.hexdigest()


def _code_fingerprints():
    from .pricing import ROOT
    paths = sorted((ROOT / "src/rvlab/pricing").glob("*.py"))
    paths += [Path(__file__), *[Path(__file__).with_name(name) for name in (
        "_pricing_fits.py", "_pricing_data.py", "_pricing_exercise.py",
        "_pricing_variance_curve.py", "_pricing_curve_kappa.py", "_pricing_curve_geometry.py", "pricing.py")
        if Path(__file__).with_name(name).exists()]]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def checkpointed_attempt(label, quotes, initial_params, *, bounds, config,
                         max_nfev, checkpoint_dir=None, fit_options=None,
                         metadata=None):
    """Run a common fit, reusing only a byte-identical-input/source checkpoint.

    JSON is data only; checkpoint reuse never unpickles executable objects.
    Prices are restored against the current quote table in its original order.
    Metadata, optimizer settings, full quotes and implementation fingerprints
    must all match. A rejected cache is retained and the trial is recomputed.
    """
    from .pricing import _jsonable
    if not isinstance(label, str) or not label or any(c in label for c in "/\\"):
        raise ValueError("trial labels must be nonempty simple filename stems")
    fit_options = {} if fit_options is None else dict(fit_options)
    metadata = {} if metadata is None else dict(metadata)
    model = "rough_heston" if isinstance(initial_params, RoughHestonCurveParams) else "heston"
    identity = {"schema": 1, "label": label, "model": model,
        "environment": {"interpreter": sys.executable, "python": sys.version,
                        "versions": {name: version(name) for name in ("numpy", "scipy", "pandas")}},
        "quotes_sha256": _frame_fingerprint(quotes),
        "initial_params": asdict(initial_params), "bounds": bounds,
        "config": asdict(config), "max_nfev": int(max_nfev),
        "fit_options": fit_options, "metadata": metadata,
        "source_sha256": _code_fingerprints()}
    identity = json.loads(_canonical(identity))
    path = None if checkpoint_dir is None else Path(checkpoint_dir) / f"{label}.json"
    if path is not None and path.exists():
        try:
            stored = json.loads(path.read_text())
            if stored["identity"] == identity:
                predictions = quotes.copy(deep=True)
                for name, values in stored["prediction_columns"].items():
                    if len(values) != len(quotes):
                        raise ValueError("checkpoint row count changed")
                    predictions[name] = [np.nan if value is None else value for value in values]
                params = _model_from_dict(stored["params"], model)
                return RecordedTrial(label, params, stored["summary"], predictions, True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A stale or incomplete optional cache must not prevent a fresh run.
            pass
    attempt = fit_attempt(label, quotes, initial_params, bounds=bounds, config=config,
                          max_nfev=max_nfev, **fit_options)
    attempt.summary.update(metadata)
    attempt.summary["parameter_values"] = (parameter_values(attempt.result.params)
                                             if attempt.result is not None else None)
    params = attempt.result.params if attempt.result is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        extra_columns = [name for name in attempt.predictions if name not in quotes
                         or not attempt.predictions[name].equals(quotes[name])]
        stored = {"identity": identity, "params": None if params is None else asdict(params),
            "summary": attempt.summary,
            "prediction_columns": {name: attempt.predictions[name].tolist() for name in extra_columns}}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_jsonable(stored), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    return RecordedTrial(label, params, attempt.summary, attempt.predictions)




# Frozen observations from notebook 06's first execution. These rows are
# evidence about that historical run, not freshly rerun calibrations. The
# explicit values also make the follow-up independent of local output files.
_HISTORICAL_JOINT = {
    "raw_european_proxy": {
        "heston": dict(v0=.012553986217694357, kappa=19.999999999999996,
            vol_of_vol=1.6943186034120992, rho=-.6599295389630052,
            levels=(.04331509418545478, .01622848257282956, .0185755926158205),
            status="converged", nfev=29, optimality=.00020748926820498103,
            train_rmse=.0225539109285977, test_rmse=.020072498555358854),
        "rough_heston": dict(v0=.014926790072046437, kappa=.21119479911865344,
            vol_of_vol=.5696320554091054, rho=-.6598078594427144,
            levels=(.7921432403876234, .1341872511876212, 2.465006113229197e-35),
            hurst=.31031935, status="exhausted", nfev=80, optimality=8775.154112443728,
            train_rmse=.04287369814613971, test_rmse=.045919791976165004),
    },
    "american_bs_sensitivity": {
        "heston": dict(v0=.012251737704014484, kappa=19.999999999999996,
            vol_of_vol=1.7542523206121328, rho=-.654734461232767,
            levels=(.04383696543952553, .015352247363230596, .018056611701555113),
            status="converged", nfev=37, optimality=.0005749542264059707,
            train_rmse=.021909137407851057, test_rmse=.02136403999801943),
        "rough_heston": dict(v0=.014537458786263124, kappa=.2508313660589629,
            vol_of_vol=.5905772892782408, rho=-.6547680929662941,
            levels=(.7007049932767018, .08681983157839199, 5.2395309313429075e-33),
            hurst=.31031935, status="exhausted", nfev=80, optimality=12558.932331833099,
            train_rmse=.04031671232162874, test_rmse=.044951218869116565),
    },
}


def historical_joint_start(input_scenario, model, knot_times):
    record = _HISTORICAL_JOINT[input_scenario][model]
    scalars = {name: record[name] for name in ("v0", "kappa", "vol_of_vol", "rho")}
    scalars["curve"] = MeanReversionCurve(tuple(knot_times), tuple(record["levels"]))
    if model == "rough_heston":
        return RoughHestonCurveParams(**scalars, hurst=record["hurst"])
    return HestonCurveParams(**scalars)


def _historical_table():
    return pd.DataFrame([{**record, "input_scenario": scenario, "model": model,
        "evidence": "historical notebook 06; not recomputed in notebook 07",
        "specification": "joint_curve", "max_nfev": 80,
        "optimizer_success": record["status"] == "converged"}
        for scenario, models in _HISTORICAL_JOINT.items() for model, record in models.items()])


def _plot_profiles(profiles):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    # The public table schema is inspected explicitly rather than inferring a
    # result from an arbitrary numeric column.
    if not profiles.empty:
        for (scenario, cap), group in profiles.groupby(["input_scenario", "eta_cap"], sort=False):
            group = group.sort_values("kappa")
            axes[0].plot(group.kappa, group.training_objective, marker="o",
                         label=f"{scenario}, eta cap {cap:g}")
            if "vol_of_vol" in group:
                axes[1].plot(group.kappa, group.vol_of_vol / np.sqrt(group.kappa), marker="o",
                             label=f"{scenario}, eta cap {cap:g}")
    axes[0].set(xscale="log", xlabel=r"Fixed $\kappa$", ylabel="Training mean squared scaled residual")
    axes[1].set(xscale="log", xlabel=r"Fixed $\kappa$", ylabel=r"Fitted $\nu/\sqrt{\kappa}$")
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    return figure


def _derivative_audit(params, quotes, config, bounds):
    """Use the independently tested, bounded geometry diagnostic."""
    from ._pricing_curve_geometry import curve_geometry_diagnostics
    return curve_geometry_diagnostics(params, quotes, bounds=bounds, config=config)




__all__ = ["RecordedTrial", "checkpointed_attempt", "historical_joint_start", "curve_convergence_demo"]


def curve_convergence_demo(context, **kwargs):
    """American replacement; historical European implementation is retired."""
    from ._pricing_american import american_spy_demo
    return american_spy_demo(context, study='curve-convergence', **kwargs)
