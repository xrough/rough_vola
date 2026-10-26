"""Executable research reports, separate from strict market readiness.

Run ``python -m rvlab.experiments.pricing --execute pricing``, ``calibration``,
``spy`` or ``all`` for the three first-stage notebooks. Their explicit facade
separates current computations, sealed numerical evidence and historical replay.
No market data is downloaded and no sibling checkout is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS = (
    "01_pricing_and_exercise.ipynb",
    "02_synthetic_calibration.ipynb",
    "03_spy_calibration.ipynb",
)
RESEARCH_STEMS = frozenset(Path(name).stem for name in NOTEBOOKS)
RESEARCH_EVIDENCE_VERSION = "pricing_research_v1"


def artifact_family(stem):
    """Keep the consolidated notebooks separate from preserved evidence."""
    if stem in RESEARCH_STEMS:
        return "research_v1"
    from ._pricing_american import SPY_NOTEBOOKS
    if stem in SPY_NOTEBOOKS or stem == "08_synthetic_american_pricing":
        return "american_v1"
    return "synthetic_v2" if stem == "03_calibration_identification" else ""


def _jsonable(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (Path, date, datetime, pd.Timestamp)):
        return str(value)
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simple_stem(value):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or "\\" in value):
        raise ValueError("artifact names must be simple filename stems")
    return value


@dataclass(frozen=True)
class DemoContext:
    name: str
    output_dir: Path
    started: float

    def progress(self, message):
        print(f"[{self.name}] {message}", flush=True)


def setup_demo(notebook_name, output_dir=None):
    """Set a small plotting style and select the local artifact directory."""
    import matplotlib.pyplot as plt
    if not isinstance(notebook_name, str):
        raise ValueError("notebook_name must be a simple filename or stem")
    _simple_stem(notebook_name)
    name = Path(notebook_name).stem
    if name != notebook_name.removesuffix(".ipynb") or name in {".", ".."} or not name:
        raise ValueError("notebook_name must be a simple filename or stem")
    family = artifact_family(name)
    destination = Path(output_dir) if output_dir is not None else ROOT / "notebooks/_output/pricing" / family / name
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140,
                         "axes.grid": True, "grid.alpha": .2,
                         "axes.spines.top": False, "axes.spines.right": False})
    return DemoContext(name, destination.resolve(), time.perf_counter())


def setup_first_stage(notebook_name, output_dir=None):
    """Use the first-stage report namespace without changing legacy callers."""
    name = Path(notebook_name).stem if isinstance(notebook_name, str) else ""
    if notebook_name not in NOTEBOOKS and name not in RESEARCH_STEMS:
        raise ValueError("First-stage reports use one of the three active notebook names")
    destination = (ROOT / "notebooks/_output/pricing/first_stage_v1" / name
                   if output_dir is None else Path(output_dir))
    return setup_demo(notebook_name, output_dir=destination)


def setup_policy_comparison(notebook_name, output_dir=None):
    """Select the policy-comparison report without changing preserved callers."""
    name = Path(notebook_name).stem if isinstance(notebook_name, str) else ""
    if notebook_name not in NOTEBOOKS and name not in RESEARCH_STEMS:
        raise ValueError("Policy-comparison reports use one of the three active notebook names")
    destination = (ROOT / "notebooks/_output/pricing/first_stage_v2" / name
                   if output_dir is None else Path(output_dir))
    return setup_demo(notebook_name, output_dir=destination)


def _finish(context, result):
    destination = context.output_dir
    for name in (*result["tables"], *result["figures"]):
        _simple_stem(name)
    for name, table in result["tables"].items():
        table.to_csv(destination / f"{name}.csv", index=False)
    for name, figure in result["figures"].items():
        target = destination / f"{name}.png"
        if isinstance(figure, (str, Path)):
            # Frozen historical evidence is displayed from its verified PNG,
            # never recomputed or relabelled as the current numerical engine.
            source = Path(figure)
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        else:
            figure.savefig(target, bbox_inches="tight")
    code = list(Path(__file__).parent.glob("*.py")) + list((ROOT / "src/rvlab/pricing").glob("*.py"))
    manifest = {
        **result.get("manifest", {}), "notebook": context.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - context.started,
        "python": sys.version, "interpreter": sys.executable,
        "versions": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sorted(code)},
        "tables": {name: len(table) for name, table in result["tables"].items()},
        "figures": list(result["figures"]), "output_dir": str(destination),
    }
    if manifest.get("artifact_family") in {"first_stage_v1", "first_stage_v2"}:
        manifest["artifact_sha256"] = {
            name + suffix: _sha(destination / (name + suffix))
            for group, suffix in ((result["tables"], ".csv"), (result["figures"], ".png"))
            for name in sorted(group)}
    manifest = _jsonable(manifest)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    result["manifest"] = manifest
    context.progress(f"Saved {len(result['tables'])} tables and {len(result['figures'])} figures in {destination}")
    return result


def run_synthetic_demo(context, *, seed=42, coarse_config=None, fine_config=None, max_nfev=80):
    from ._pricing_workflows import synthetic_demo
    return _finish(context, synthetic_demo(context, seed=seed, coarse_config=coarse_config,
                                           fine_config=fine_config, max_nfev=max_nfev))


def run_pricing_research(context, **kwargs):
    """European synthetic controls and American exercise/sensitivity research."""
    from ._pricing_research import pricing_research
    return _finish(context, pricing_research(context, **kwargs))


def run_calibration_research(context, **kwargs):
    """Synthetic price recovery, numerical accuracy and identification."""
    from ._pricing_research import calibration_research
    return _finish(context, calibration_research(context, **kwargs))


def run_spy_research(context, **kwargs):
    """One American-only SPY investigation, gated by independent inputs."""
    from ._pricing_research import spy_research
    return _finish(context, spy_research(context, **kwargs))


def run_first_stage_pricing(context, **kwargs):
    """Synthetic numerical report; collect separately sealed benchmark paths."""
    from ._pricing_first_stage import first_stage_pricing
    return _finish(context, first_stage_pricing(context, **kwargs))


def run_first_stage_calibration(context, **kwargs):
    """Verified calibration research replay with original numerical provenance."""
    from ._pricing_first_stage import first_stage_calibration
    return _finish(context, first_stage_calibration(context, **kwargs))


def run_first_stage_spy(context, **kwargs):
    """Re-evaluate the American-only market-data handoff and readiness blockers."""
    from ._pricing_first_stage import first_stage_spy
    return _finish(context, first_stage_spy(context, **kwargs))


def run_policy_comparison_pricing(context, **kwargs):
    """Matched stopping-policy evidence and preserved numerical controls."""
    from ._pricing_policy_comparison_report import policy_comparison_pricing
    return _finish(context, policy_comparison_pricing(context, **kwargs))


def run_policy_comparison_calibration(context, **kwargs):
    """Preserved synthetic inference with current-source pricing compatibility."""
    from ._pricing_policy_comparison_report import policy_comparison_calibration
    return _finish(context, policy_comparison_calibration(context, **kwargs))


def run_policy_comparison_spy(context, **kwargs):
    """Current American SPY input handoff under the policy-comparison report."""
    from ._pricing_policy_comparison_report import policy_comparison_spy
    return _finish(context, policy_comparison_spy(context, **kwargs))


def run_spy_demo(context, *, seed=42, coarse_config=None, fine_config=None,
                 accurate_config=None, config=None, max_nfev=40, data_root=None,
                 dividend_coverage_through=None):
    from ._pricing_american import american_spy_demo
    if coarse_config is not None or accurate_config is not None:
        raise ValueError("European Fourier demonstration settings are retired for real SPY; supply an American FDConfig")
    if config is not None and fine_config is not None:
        raise ValueError("supply config only, not config and the former fine_config alias")
    return _finish(context, american_spy_demo(context, study="pricing", seed=seed,
        config=config if config is not None else fine_config, max_nfev=max_nfev,
        data_root=data_root, dividend_coverage_through=dividend_coverage_through))


def run_accuracy_demo(context, **kwargs):
    from ._pricing_american import american_spy_demo
    return _finish(context, american_spy_demo(context, study="accuracy", **kwargs))


def run_attribution_demo(context, **kwargs):
    from ._pricing_american import american_spy_demo
    return _finish(context, american_spy_demo(context, study="attribution", **kwargs))


def run_variance_curve_demo(context, **kwargs):
    from ._pricing_american import american_spy_demo
    return _finish(context, american_spy_demo(context, study="variance-curve", **kwargs))


def run_curve_convergence_demo(context, **kwargs):
    from ._pricing_american import american_spy_demo
    return _finish(context, american_spy_demo(context, study="curve-convergence", **kwargs))


def run_american_synthetic_demo(context, **kwargs):
    from ._pricing_american_synthetic import american_synthetic_demo
    return _finish(context, american_synthetic_demo(context, **kwargs))


def run_extended_demo(context, *, seed=42, max_nfev=80, data_root=None, config=None):
    from ._pricing_workflows import extended_demo
    return _finish(context, extended_demo(context, seed=seed, max_nfev=max_nfev,
                                          data_root=data_root, config=config))


def show_tables(results, names=None):
    """Display tables; complete versions are always exported to CSV."""
    from IPython.display import Markdown, display
    for name in results["tables"] if names is None else names:
        if name not in results["tables"]:
            continue
        table = results["tables"][name]
        display(Markdown(f"**{name.replace('_', ' ').title()}** ({len(table)} rows)"))
        with pd.option_context("display.max_columns", 16, "display.max_colwidth", 80):
            display(table.head(40))
        if len(table) > 40:
            display(Markdown("First 40 rows shown; the CSV contains the complete table."))


def show_figures(results, names=None):
    from io import BytesIO
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    for name in results["figures"] if names is None else names:
        if name in results["figures"]:
            figure = results["figures"][name]
            if isinstance(figure, (str, Path)):
                display(Image(data=Path(figure).read_bytes()))
                continue
            # Explicit PNG works with a headless Agg kernel as well as inline
            # Jupyter backends, and embeds the figure in the saved notebook.
            pixels = BytesIO()
            figure.savefig(pixels, format="png", bbox_inches="tight")
            display(Image(data=pixels.getvalue()))
            plt.close(figure)


def execute_notebook(path, output_path=None, *, timeout=1500):
    """Execute in a fresh kernel pinned to this exact Python interpreter."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_core.utils import ensure_async, run_sync
    source = Path(path).resolve()
    if source.is_relative_to((ROOT / "notebooks/legacy/pricing").resolve()):
        raise ValueError("Archived pricing notebooks are preserved evidence; execute the three active pricing notebooks instead")
    destination = source if output_path is None else Path(output_path).resolve()
    notebook = nbformat.read(source, as_version=4)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rvlab-pricing-kernel-") as temporary:
        kernel_root = Path(temporary) / "kernels"
        spec_dir = kernel_root / "rvlab-pricing"
        spec_dir.mkdir(parents=True)
        spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "RVLab pricing (pinned)", "language": "python",
            "env": {"PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(Path(temporary) / "matplotlib"),
                    "IPYTHONDIR": str(Path(temporary) / "ipython")}}
        (spec_dir / "kernel.json").write_text(json.dumps(spec))
        # Async channels are essential: blocking channel reads can prevent
        # nbclient's timeout task from running while a cell executes.
        manager = AsyncKernelManager(kernel_name="rvlab-pricing",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
        client = NotebookClient(notebook, km=manager, timeout=timeout,
            allow_errors=False, resources={"metadata": {"path": str(ROOT)}})
        print(f"Executing {source.name} with {sys.executable}", flush=True)
        try:
            client.execute(cleanup_kc=True)
        finally:
            if manager.has_kernel:
                async def shutdown():
                    await ensure_async(manager.shutdown_kernel(now=True))
                run_sync(shutdown)()
    # Persist a portable kernelspec while retaining actual execution provenance.
    notebook.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)",
                                   "language": "python", "name": "python3"}
    notebook.metadata["rvlab_execution"] = {"interpreter": sys.executable,
        "elapsed_seconds": time.perf_counter() - start,
        "executed_at": datetime.now(timezone.utc).isoformat()}
    family = notebook.metadata.get("rvlab_artifact_family")
    if family in {"first_stage_v1", "first_stage_v2"} and source.name in NOTEBOOKS:
        artifact_manifest = ROOT / "notebooks/_output/pricing" / family / source.stem / "manifest.json"
        notebook.metadata["rvlab_execution"]["artifact_manifest_sha256"] = _sha(artifact_manifest)
    nbformat.validate(notebook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    print(f"Completed {source.name} in {time.perf_counter() - start:.1f}s", flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", choices=("pricing", "calibration", "spy", "all", "quick"), required=True,
                        help="select one active notebook, all three, or quick (alias for the two synthetic notebooks)")
    parser.add_argument("--output-dir", type=Path,
                        help="Write executed copies here; default updates the requested notebooks")
    parser.add_argument("--timeout", type=int,
                        help="per-cell timeout in seconds; joint calibration cold runs may need more than the default 1500")
    args = parser.parse_args(argv)
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    names = {"pricing": NOTEBOOKS[:1], "calibration": NOTEBOOKS[1:2],
             "spy": NOTEBOOKS[2:3], "quick": NOTEBOOKS[:2], "all": NOTEBOOKS}[args.execute]
    for name in names:
        options = {} if args.timeout is None else {"timeout": args.timeout}
        execute_notebook(ROOT / "notebooks/pricing" / name,
                         None if args.output_dir is None else args.output_dir / name, **options)
    return 0


# The notebook-facing facade also exposes the small, independently testable
# experiment operations. These are separate from the core pricing API.
from ._pricing_data import (DemoSurface, implied_volatility, parity_forward,
                           select_spy_snapshot, load_spy_demo)
from ._pricing_fits import (FitAttempt, synthetic_surface, fit_attempt,
    deterministic_starts, select_attempt, refinement_table)

__all__ = ["DemoContext", "DemoSurface", "FitAttempt", "setup_demo",
    "setup_first_stage", "run_first_stage_pricing", "run_first_stage_calibration", "run_first_stage_spy",
    "setup_policy_comparison", "run_policy_comparison_pricing", "run_policy_comparison_calibration", "run_policy_comparison_spy",
    "run_pricing_research", "run_calibration_research", "run_spy_research",
    "run_synthetic_demo", "run_spy_demo", "run_extended_demo", "run_american_synthetic_demo", "run_accuracy_demo", "run_attribution_demo", "run_variance_curve_demo", "run_curve_convergence_demo", "show_tables",
    "show_figures", "implied_volatility", "parity_forward", "select_spy_snapshot",
    "load_spy_demo", "synthetic_surface", "fit_attempt", "deterministic_starts",
    "select_attempt", "refinement_table", "execute_notebook"]


if __name__ == "__main__":
    raise SystemExit(main())
