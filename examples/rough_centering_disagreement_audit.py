"""Late local-centering mechanism check on the existing independent audit paths.

Selection uses only disagreement between the two previously frozen policies.
It is sealed before computing the new conditional residuals. No new paths,
policy/control selection, or changes to the primary eight-stream study occur.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main():
    from rvlab.experiments._pricing_rough_frozen_policy import load_frozen_policy_bundle
    from rvlab.experiments._pricing_rough_centering_audit import audit_conditional_centers
    from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint

    output = ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    profile = output/"rough_control_centering_v1/499c48aa4d689aa2"
    supplement = profile/"supplemental_disagreement_states_v1"
    if supplement.exists():
        raise FileExistsError("Refusing to replace an existing sealed supplemental audit")
    supplement.mkdir()
    source = supplement/"rough_centering_disagreement_audit.py.snapshot"
    shutil.copyfile(__file__, source)
    frozen = load_frozen_policy_bundle(output/"rough_martingale_centering_v1/frozen_joint_policy")
    primary_file = profile/"conditional_audit_499c48aa4d689aa2.json"
    primary = json.loads(primary_file.read_text())["payload"]
    trace_record = primary["sample_archives"][0]
    trace_file = Path(trace_record["path"])
    if trace_file.stat().st_size != trace_record["bytes"] or _sha(trace_file) != trace_record["sha256"]:
        raise ValueError("Changed independent conditional-state trace")
    with np.load(trace_file, allow_pickle=False) as arrays:
        trace = {key:arrays[key].copy() for key in arrays.files}
    trace.update(seed=primary["audit_manifest"]["state_trace_seed"],
        policy_training_seed=frozen["proxy"].config.training_seed,
        policy_fingerprint=fourier_policy_fingerprint(frozen["proxy"]),
        exercise_style="american", additional_control_training_seeds=())
    original = evaluate_trace_policy(trace, policy=frozen["original_policy"], martingale_weights=frozen["weights"])
    selected = evaluate_trace_policy(trace, policy=frozen["selected_policy"], martingale_weights=frozen["weights"])
    o, s = original.stop_indices[:, 0], selected.stop_indices[:, 0]
    pairs = []
    for path in range(len(o)):
        per_path = 0
        for date in range(len(trace["times"])-1):
            if trace["eligible"][date] and (o[path] > date) != (s[path] > date):
                pairs.append((path, date)); per_path += 1
                if per_path == 2 or len(pairs) == 12:
                    break
        if len(pairs) == 12:
            break
    core_hashes = primary["audit_manifest"]["source_hashes"]
    if any(_sha(path) != digest for path, digest in core_hashes.items()):
        raise ValueError("Frozen numerical/audit source changed before supplemental test")
    seal = {"study":"supplemental_disagreement_states_v1", "primary_profile":"499c48aa4d689aa2",
        "late_exploratory":True, "primary_design_input":False, "primary_pool_input":False,
        "new_paths_generated":False, "policy_or_control_selection":False,
        "source_trace":trace_record, "primary_audit_sha256":_sha(primary_file),
        "bundle_manifest_sha256":_sha(Path(frozen["folder"])/"manifest.json"),
        "seed":int(trace["seed"]), "paths":len(o),
        "selection_rule":"First eligible nonterminal (path,date) in lexicographic order where (original_stop>date)!=(selected_stop>date); at most2dates per path; at most12states total",
        "selection_uses_conditional_residuals":False, "selection_uses_frozen_decisions_only":True,
        "state_indices":[list(pair) for pair in pairs],
        "policy_stop_indices":{"original":o.tolist(), "selected":s.tolist()},
        "source_hashes":core_hashes, "script_sha256":_sha(__file__), "tolerances_changed":False}
    _write_json(supplement/"selection_before_conditional_residuals.json", seal)
    started = time.perf_counter()
    try:
        if pairs:
            result = audit_conditional_centers(frozen["proxy"], trace,
                state_indices=pairs, martingale_weights=frozen["weights"])
            for name, table in result["tables"].items():
                table.to_csv(supplement/(name+".csv"), index=False)
            result = {"status":"completed", "tables":{name:table.to_dict("records") for name, table in result["tables"].items()},
                "audit_manifest":result["manifest"]}
        else:
            result = {"status":"no_disagreement_states", "tables":{}, "audit_manifest":None}
        if any(_sha(path) != digest for path, digest in core_hashes.items()):
            raise ValueError("Frozen numerical/audit source changed during supplemental test")
    except Exception as error:
        result = {"status":"numerical_failure", "error":str(error), "traceback":traceback.format_exc()}
    result.update(late_exploratory=True, primary_design_input=False, primary_pool_input=False,
        elapsed_seconds=time.perf_counter()-started, selection_sha256=_sha(supplement/"selection_before_conditional_residuals.json"),
        new_paths_generated=False, tolerances_changed=False, american_pricing_ready=False)
    _write_json(supplement/"result.json", result)
    files = [{"path":str(path.relative_to(supplement)), "bytes":path.stat().st_size, "sha256":_sha(path)}
        for path in sorted(supplement.rglob("*")) if path.is_file()]
    inventory_hash = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write_json(supplement/"preservation_inventory.json", {"files":files, "inventory_sha256":inventory_hash})
    print(json.dumps({"folder":str(supplement), "status":result["status"], "states":len(pairs),
        "elapsed_seconds":result["elapsed_seconds"], "inventory_sha256":inventory_hash}))
    if result["status"] == "numerical_failure":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
