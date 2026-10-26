"""Validate the independent time-refinement Greek evidence before collection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _time_deltas(folder):
    """Read the declared 256-step stream, retaining every stock and path.

    Source/design fingerprints, whole-path prices, finite stock differences,
    statistical summaries and unchanged marginal gates are independently
    checked. No outcome is excluded for failing a diagnostic.
    """
    # Imported at call time: the collector uses this validator, while its
    # generic byte and statistics checks remain the single shared protocol.
    from ._pricing_rough_stopping_delta import (
        _delta_gate, _fixed_tolerances, _manifest, _sha, _stat, _table)

    folder=Path(folder)
    manifest,entries=_manifest(folder,
        required_files={"estimates.csv","comparisons.csv","samples.npz"})
    design_path=folder/"design_before_valuation.json"
    design=json.loads(design_path.read_text())
    _fixed_tolerances(design,required=True)
    if (manifest.get("seed")!=712005042 or manifest.get("paths")!=2048
            or manifest.get("steps")!=256 or manifest.get("design_sha256")!=_sha(design_path)
            or design.get("seed")!=712005042 or design.get("paths")!=2048
            or design.get("steps")!=256 or design.get("exercise_steps")!=64
            or design.get("bump_relatives")!=[.01,.005,.0025]):
        raise ValueError("Wrong or incomplete independent time-delta design")
    if (not manifest.get("source_sha256")
            or manifest["source_sha256"]!=design.get("source_sha256")):
        raise ValueError("Time-delta numerical sources differ from the sealed design")
    for name,expected in (("independent_training_valuation",True),("valuation_used_for_fitting",False),
            ("sources_unchanged",True),("fixed_parameters_factors_discount_dividends",True),
            ("decisions_reapplied_at_each_bump",True),("stopping_times_held_fixed",False),
            ("same_common_martingale_for_policies",True),("optimal_delta_ready",False),
            ("tolerances_changed",False)):
        value=manifest.get(name)
        if not isinstance(value,bool) or value!=expected:
            raise ValueError("Changed independent time-delta provenance: "+name)

    bundle=folder.parents[1]/"rough_martingale_centering_v1/frozen_joint_policy"
    bundle_manifest=bundle/"manifest.json"
    if _sha(bundle_manifest)!=design["frozen_bundle_manifest_sha256"]:
        raise ValueError("Frozen time-delta policy/control bundle changed")
    declaration=json.loads(bundle_manifest.read_text())
    coefficient_path=bundle/"coefficients.npz"
    if _sha(coefficient_path)!=declaration["coefficient_archive_sha256"]:
        raise ValueError("Frozen time-delta policy/control coefficients changed")
    spot=declaration["market"]["spot"]
    if isinstance(spot,bool) or not np.isfinite(spot) or spot!=100.:
        raise ValueError("Time-delta study requires its declared synthetic stock100")
    for path in (bundle_manifest,coefficient_path):
        entries.append({"path":str(path),"bytes":path.stat().st_size,"sha256":_sha(path)})

    with np.load(folder/"samples.npz",allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in archive.files}
    relative=np.array([-.01,-.005,-.0025,0.,.0025,.005,.01])
    if not np.array_equal(arrays.get("relative_spots"),relative):
        raise ValueError("Changed declared time-delta stock bumps")
    expected_keys={"relative_spots"}
    for policy in ("original_policy","selected_policy"):
        expected_keys|={policy+"__"+name for name in ("raw_prices","controlled_prices","stops")}
        expected_keys|={policy+"__"+f"{estimator}:{level:g}"
            for estimator in ("raw","stopped martingale") for level in (.01,.005,.0025)}
    if set(arrays)!=expected_keys:
        raise ValueError("Missing or extra declared time-delta sample arrays")
    for policy in ("original_policy","selected_policy"):
        for name in ("raw_prices","controlled_prices","stops"):
            values=arrays[policy+"__"+name]
            if (values.shape!=(2048,7) or values.dtype.kind not in "iuf"
                    or not np.isfinite(values).all()):
                raise ValueError("Time-delta samples omit declared paths or stock bumps")
        stops=arrays[policy+"__stops"]
        if stops.dtype.kind not in "iu" or np.any(stops<0) or np.any(stops>64):
            raise ValueError("Time-delta stop indices do not match the frozen exercise calendar")
        for estimator,name in (("raw","raw_prices"),("stopped martingale","controlled_prices")):
            prices=arrays[policy+"__"+name]
            for level in (.01,.005,.0025):
                values=arrays[policy+"__"+f"{estimator}:{level:g}"]
                if (values.shape!=(2048,) or values.dtype.kind not in "iuf"
                        or not np.isfinite(values).all()):
                    raise ValueError("Time-delta samples omit declared whole paths")
                lo=np.flatnonzero(relative==-level)[0];hi=np.flatnonzero(relative==level)[0]
                actual=(prices[:,hi]-prices[:,lo])/(2*spot*level)
                if not np.allclose(actual,values,rtol=0.,atol=2e-14):
                    raise ValueError("Time delta does not equal its paired stock-price differences")

    table=_table(folder/"estimates.csv",entries)
    expected={(policy,estimator,level) for policy in ("original_policy","selected_policy")
        for estimator in ("raw","stopped martingale") for level in (.01,.005,.0025)}
    if len(table)!=12 or set(zip(table.policy,table.estimator,table.bump_relative))!=expected:
        raise ValueError("Time table omits or duplicates declared policies/estimators/bumps")
    for row in table.itertuples():
        if row.seed!=712005042 or row.paths!=2048 or row.steps!=256:
            raise ValueError("Changed time-delta stream metadata")
        actual=_stat(arrays[row.policy+"__"+f"{row.estimator}:{row.bump_relative:g}"])
        if any(not np.isclose(getattr(row,key),value,rtol=1e-12,atol=2e-14)
                for key,value in actual.items()):
            raise ValueError("Changed independent time-delta statistics")
        _delta_gate(row,paths=2048)
    return table,entries,manifest
