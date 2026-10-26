"""First-stage research reports with distinct current and historical evidence.

The report does not silently rerun expensive calibration or old Monte Carlo
campaigns. Historical displays are pinned to their preserved bytes and retain
the original execution/source identities. New numerical evidence is collected
only from the separately sealed, predefined generic benchmark.
"""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FAMILY = "first_stage_v1"
EVIDENCE_VERSION = "pricing_first_stage_v1"
ARCHIVE = ROOT / "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20"
ARCHIVE_SHA256 = "2239df2f215df22ce6a21e1dab603f952522b60ee97a846ab86ae8f0e15899b7"
PROJECT_TITLE = "American-option pricing and policy deltas under a finite-factor rough-Heston approximation"
FOCUSED_PREFIXES = ("rough_mc_precision_", "rough_shared_delta_", "rough_bellman_policy_")
FOCUSED_PROFILES = {
    "rough_mc_precision": "93d94ff9f30cfc6155d77797c05d3996dcc8dc1588b5e811b68f152c095dcac2",
    "rough_shared_delta": "c960a5202eba9c8493c47c889e5a8069933d0c366691331b3a694d7236f2a315",
}
FOUNDATION_PREFIXES = ("rough_precision_summary", "rough_precision_limit_summary",
                       "rough_foundation_kernel")
FOUNDATION_PROFILES = {
    "rough_foundation": "cbbd05e98703030dee4de2553d810166637f6bf8bcf7fc844de538481b8d8324",
    "rough_precision": "a6f7f9d0f9dad7ceebf517bc269bef2c4c1ca06e13bfc61f85e4eb1290fcf147",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The first-stage recorded research design uses seed 42")


def _archive_index(archive=ARCHIVE, expected_sha256=ARCHIVE_SHA256):
    archive = Path(archive).resolve()
    path = archive / "preservation_manifest.json"
    if sha(path) != expected_sha256:
        raise ValueError("The preserved first-stage research inventory changed")
    record = json.loads(path.read_text())
    index = {}
    for row in record["files"]:
        relative = row["preserved"]
        candidate = (archive / relative).resolve()
        if not candidate.is_relative_to(archive) or relative in index:
            raise ValueError("Historical inventory paths must be unique and remain inside the archive")
        index[relative] = row
    return archive, index


def _verified(archive, index, relative, entries):
    if relative not in index:
        raise ValueError(f"Historical evidence is absent from its inventory: {relative}")
    path, row = archive / relative, index[relative]
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        raise ValueError(f"Historical evidence changed: {relative}")
    entries[str(path)] = {"path": str(path), "bytes": row["bytes"], "sha256": row["sha256"]}
    return path


def load_preserved_report(stem, *, prefixes=None, archive=ARCHIVE,
                          expected_sha256=ARCHIVE_SHA256):
    """Load selected recorded tables/figures, never a synthetic fallback.

    Hashes and row counts are checked before returning a display. The complete
    old notebook and all numerical payloads remain in the preservation archive;
    replaying a display is not a new Monte Carlo or calibration validation run.
    """
    if stem not in ("01_pricing_and_exercise", "02_synthetic_calibration", "03_spy_calibration"):
        raise ValueError("Unknown preserved research notebook")
    archive, index = _archive_index(archive, expected_sha256)
    entries = {}
    relative = f"results/{stem}/manifest.json"
    path = _verified(archive, index, relative, entries)
    manifest = json.loads(path.read_text())
    if manifest.get("notebook") != stem or manifest.get("evidence_version") != "pricing_research_v1":
        raise ValueError("Preserved notebook identity or evidence version changed")
    if stem != "03_spy_calibration" and (
            manifest.get("data_kind") != "synthetic" or "spy_inputs" in manifest):
        raise ValueError("Synthetic research cannot load market observations")
    if stem == "03_spy_calibration" and (
            manifest.get("exercise_style") != "american" or
            manifest.get("backend") != "american_vanilla_fd"):
        raise ValueError("SPY evidence must use the American contract/backend")
    prefixes = tuple(prefixes) if prefixes is not None else None
    selected = lambda name: prefixes is None or name.startswith(prefixes)
    tables, figures = {}, {}
    for name, count in manifest["tables"].items():
        if not selected(name):
            continue
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid historical table stem")
        path = _verified(archive, index, f"results/{stem}/{name}.csv", entries)
        try:
            frame = pd.read_csv(path, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            if count:
                raise
            frame = pd.DataFrame()
        if len(frame) != count:
            raise ValueError(f"Historical table row count changed: {name}")
        tables[name] = frame
    for name in manifest["figures"]:
        if selected(name):
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Invalid historical figure stem")
            figures[name] = _verified(archive, index, f"results/{stem}/{name}.png", entries)
    if prefixes is not None:
        for prefix in prefixes:
            if not any(name.startswith(prefix) for name in tables):
                raise ValueError(f"Required historical evidence missing: {prefix}")
    inventory = archive / "preservation_manifest.json"
    entries[str(inventory)] = {"path": str(inventory), "bytes": inventory.stat().st_size,
                               "sha256": expected_sha256}
    return {"tables": tables, "figures": figures, "manifest": manifest,
            "input_files": list(entries.values())}


def _stamp(result, *, role):
    result["manifest"].update(evidence_version=EVIDENCE_VERSION,
        artifact_family=FAMILY, research_project=PROJECT_TITLE, report_role=role,
        report_scope="Synthetic numerical-methods research; general American rough-Heston accuracy and market hedging remain separate",
        tolerances_changed=False)
    return result


def first_stage_pricing(context, *, seed=42, fourier_config=None, fd_config=None):
    _seed(seed)
    from ._pricing_research import pricing_research
    context.progress("Building the numerical research report and verifying preserved focused evidence")
    result = pricing_research(context, seed=seed, fourier_config=fourier_config,
                             fd_config=fd_config)
    historical = load_preserved_report("01_pricing_and_exercise",
        prefixes=FOCUSED_PREFIXES + FOUNDATION_PREFIXES)
    for key, fingerprint in {**FOCUSED_PROFILES, **FOUNDATION_PROFILES}.items():
        if historical["manifest"][key]["profile_fingerprint"] != fingerprint:
            raise ValueError(f"Focused research profile changed: {key}")
    result["tables"].update(historical["tables"])
    result["figures"].update(historical["figures"])
    result["manifest"].update(
        historical_focused_evidence={key: copy.deepcopy(historical["manifest"][key])
                                     for key in (*FOCUSED_PROFILES, "rough_bellman_policy")},
        historical_foundation_evidence={key: copy.deepcopy(historical["manifest"][key])
                                        for key in FOUNDATION_PROFILES},
        historical_execution={key: historical["manifest"][key] for key in (
            "created_at", "elapsed_seconds", "interpreter", "versions", "source_sha256")},
        historical_input_files=historical["input_files"],
        historical_archive=str(ARCHIVE), historical_archive_sha256=ARCHIVE_SHA256,
        focused_evidence_recomputed=False,
        foundation_evidence_recomputed=False,
        american_pricing_ready=False, stock_delta_ready=False, optimal_delta_ready=False)
    from ._pricing_rough_benchmark import benchmark_study
    benchmark = benchmark_study(context)
    result["tables"].update({f"rough_benchmark_{name}": frame
                              for name, frame in benchmark["tables"].items()})
    result["figures"].update({f"rough_benchmark_{name}": figure
                               for name, figure in benchmark["figures"].items()})
    result["manifest"]["rough_benchmark"] = benchmark["manifest"]
    result["manifest"]["first_stage_report_complete"] = bool(benchmark["manifest"]["complete"])
    _stamp(result, role="pricing_and_policy_delta_research")
    from ._pricing_first_stage_reporting import add_first_stage_reporting
    return add_first_stage_reporting(result)


def first_stage_calibration(context, *, seed=42):
    _seed(seed)
    context.progress("Verifying the recorded synthetic calibration and identification study")
    historical = load_preserved_report("02_synthetic_calibration")
    original = copy.deepcopy(historical["manifest"])
    result = {"tables": historical["tables"], "figures": historical["figures"],
              "manifest": copy.deepcopy(original)}
    from ._pricing_first_stage_reprice import reprice_selected_calibration
    compatibility = reprice_selected_calibration(historical["tables"], original)
    result["tables"]["current_source_reprice"] = compatibility
    result["manifest"].update(historical_replay=True, current_engine_execution=True,
        execution_mode="verified_historical_calibration_with_current_source_repricing",
        historical_numerical_manifest=original, historical_input_files=historical["input_files"],
        historical_archive=str(ARCHIVE), historical_archive_sha256=ARCHIVE_SHA256,
        optimizer_calls_this_report=0, first_stage_report_complete=True,
        american_joint_executed=False, american_joint_included=True,
        curve_diagnosis_executed=False, curve_diagnosis_included=True,
        maturity_extension_executed=False, maturity_extension_included=True,
        current_calibration_recomputed=False, current_refinement_recomputed=False,
        current_source_repricing_pass=bool(compatibility.passed.all()),
        current_execution_scope="Reprice all 18 contracts for each of four recorded training-selected fits; no optimizer or historical refinement rerun")
    _stamp(result, role="synthetic_calibration_and_identification")
    from ._pricing_first_stage_reporting import add_first_stage_reporting
    return add_first_stage_reporting(result)


def first_stage_spy(context, **kwargs):
    _seed(kwargs.get("seed", 42))
    from ._pricing_research import spy_research
    result = spy_research(context, **kwargs)
    _stamp(result, role="market_data_handoff")
    result["manifest"]["first_stage_report_complete"] = True
    from ._pricing_first_stage_reporting import add_first_stage_reporting
    return add_first_stage_reporting(result)
