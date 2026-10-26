"""All scalar audit rows remain traceable, including inaccurate/failed trials."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_bellman_audit_collection as audit


def write(path, value): path.write_text(json.dumps(value))


def record_rows(state):
    rows = []
    reference = 2.+state["state_index"]*.01
    for order, change in zip(audit.ORDERS, (.0003, .0002, .0001)):
        if state["exercise_index"] == 63: change = 0.
        cv = reference+change; direct = reference+change*.9
        rows.append({**{k: state[k] for k in ("state_index", "seed", "path_index", "exercise_index", "role")},
            "order": order, "status": "completed", "reference_continuation": reference,
            "cv_continuation": cv, "direct_continuation": direct,
            "cv_minus_reference": cv-reference, "direct_minus_reference": direct-reference,
            "quadrature_error_estimate": 1e-10, "omitted_tail_value_bound": 1e-30,
            "price_allowance": .001+1e-4*abs(reference), "cv_within_dollar_allowance": True,
            "direct_within_dollar_allowance": True, "terminal_exact_payoff": state["exercise_index"] == 63,
            "quadrature_error_is_rigorous_bound": False, "used_for_formal_policy_selection": False})
    return rows


def diagnostics_frame(records):
    return pd.DataFrame([{k: v for k, v in record.items() if k not in ("rows", "state", "numerical_diagnostics")}
        | {k: v for k, v in record["state"].items() if k != "factors"} for record in records])


@pytest.fixture
def sealed_audit(tmp_path, monkeypatch):
    folder = tmp_path/"rough_bellman_policy_v1/independent_adaptive_audit_v1"; folder.mkdir(parents=True)
    panel = []
    for source_index, seed in enumerate(audit.SEEDS):
        for j in audit.DATES:
            for path, role in enumerate(("first ITM", "second ITM", "nearest frozen boundary", "largest variance")):
                panel.append({"state_index": len(panel), "seed": seed, "source_index": source_index,
                    "path_index": path, "exercise_index": j, "time": j/400., "role": role,
                    "stock": 99.-path*.1, "variance": .04, "frozen_margin": -.001,
                    "factors": [0., .001, .002, .003]})
    # Protocol/algebra fixture only: no synthetic evidence is represented as
    # output from independent adaptive integration or a market model.
    monkeypatch.setattr(audit, "_source_panel", lambda path: deepcopy(panel))
    design = {"version": "rough_bellman_independent_adaptive_audit_v1", "states": 32, "orders": [8, 16, 32],
        "dates": list(audit.DATES), "spent_seeds": list(audit.SEEDS), "absolute_tolerance": 1e-9,
        "relative_tolerance": 1e-10, "normal_limit": 12., "state_wall_limit_seconds": 180.,
        "new_outer_paths": 0, "new_fits": 0, "used_for_formal_policy_selection": False,
        "quadrature_error_is_rigorous_bound": False, "only_premium_capped": True,
        "proxy_policy_improvement_guaranteed": False, "allowance": ".001+1e-4*abs(C)", "tolerances_changed": False,
        "source_sha256": {name: "0"*64 for name in audit.required_audit_sources()},
        "input_sha256": {name: "0"*64 for name in audit.required_audit_inputs()}, "state_panel": deepcopy(panel)}
    records = []
    for state in panel:
        rows = record_rows(state)
        records.append({"state": deepcopy(state), "status": "completed", "rows": rows, "elapsed_seconds": 1.,
            "detected_snell_crossings": 2, "function_evaluations": 100, "envelope_constant": 101.,
            "envelope_stock": 5., "warnings": [], "numerical_diagnostics": {"orders": [8, 16, 32],
                "physical_increments": 2, "max_probability_error": 0., "terminal_exact_payoff": state["exercise_index"] == 63,
                "only_premium_capped": True, "actual_future_state_used": False, "all_factor_branches_and_stock_orderings": True,
                "max_cv_direct_difference": [abs(row["cv_continuation"]-row["direct_continuation"]) for row in rows]}})
    fixture = SimpleNamespace(folder=folder, design=design, panel=panel, records=records)
    def persist():
        write(folder/"design_before_evaluation.json", design)
        digest = hashlib.sha256((folder/"design_before_evaluation.json").read_bytes()).hexdigest()
        fixture.manifest = {**deepcopy(design), "complete": True, "design_sha256": digest,
            "successful_states": sum(r["status"] == "completed" for r in records),
            "failed_states": sum(r["status"] == "failed" for r in records), "all_predeclared_states_retained": True}
        for index, record in enumerate(records):
            record["design_sha256"] = digest; write(folder/f"state_{index:02d}.json", record)
        write(folder/"manifest.json", fixture.manifest)
        pd.DataFrame(panel).to_csv(folder/"panel.csv", index=False)
        pd.DataFrame([row for r in records for row in r["rows"]]).to_csv(folder/"checks.csv", index=False)
        diagnostics_frame(records).to_csv(folder/"state_diagnostics.csv", index=False)
    fixture.persist = persist; persist()
    return fixture


def validate(fixture):
    return audit.validate_adaptive_audit(fixture.folder, fixture.design, fixture.manifest,
        *[pd.read_csv(fixture.folder/name, float_precision="round_trip")
            for name in ("panel.csv", "checks.csv", "state_diagnostics.csv")])


def test_complete_scalar_protocol_reconstructs_all_states_orders_and_diagnostics(sealed_audit):
    result = validate(sealed_audit)
    assert {name: len(table) for name, table in result.items()} == {
        "spent_scalar_checks": 96, "spent_scalar_states": 32, "spent_scalar_diagnostics": 32}


@pytest.mark.parametrize("tamper", ["source_omitted", "source_added", "input_omitted", "input_added", "role",
    "quadrature_tolerance", "normal_limit", "price_allowance", "proxy_guarantee", "panel_selector",
    "completion", "count", "manifest_role", "manifest_design_pin", "retention"])
def test_scalar_design_completion_and_exact_source_input_sets_are_required(sealed_audit, tamper):
    f = sealed_audit
    if tamper == "source_omitted": f.design["source_sha256"].pop(next(iter(f.design["source_sha256"])))
    if tamper == "source_added": f.design["source_sha256"]["unexpected.py"] = "0"*64
    if tamper == "input_omitted": f.design["input_sha256"].pop(next(iter(f.design["input_sha256"])))
    if tamper == "input_added": f.design["input_sha256"]["unexpected.npz"] = "0"*64
    if tamper == "role": f.design["used_for_formal_policy_selection"] = True
    if tamper == "quadrature_tolerance": f.design["absolute_tolerance"] = 1e-4
    if tamper == "normal_limit": f.design["normal_limit"] = 8.
    if tamper == "price_allowance": f.design["allowance"] = ".01"
    if tamper == "proxy_guarantee": f.design["proxy_policy_improvement_guaranteed"] = True
    if tamper == "panel_selector": f.design["state_panel"][0]["path_index"] = 999
    f.persist()
    if tamper == "completion": f.manifest["complete"] = False
    if tamper == "count": f.manifest["successful_states"] = 31
    if tamper == "manifest_role": f.manifest["new_outer_paths"] = 32
    if tamper == "manifest_design_pin": f.manifest["design_sha256"] = "0"*64
    if tamper == "retention": f.manifest["all_predeclared_states_retained"] = False
    with pytest.raises(ValueError): validate(f)


@pytest.mark.parametrize("tamper", ["missing_state", "unexpected_state", "row_order", "row_identity", "row_role", "row_selection",
    "row_error", "row_arithmetic", "allowance", "false_pass", "reference", "terminal_flag", "nonfinite",
    "physical_steps", "future_state", "snell_cap", "mass", "cv_direct", "warnings", "runtime", "envelope", "terminal_values"])
def test_each_state_record_checks_numerical_arithmetic_scope_and_status(sealed_audit, tamper):
    f = sealed_audit; r = f.records[0]; row = r["rows"][0]
    if tamper == "row_order": row["order"] = 16
    if tamper == "row_identity": row["path_index"] = 999
    if tamper == "row_role": row["role"] = "selected hindsight tail"
    if tamper == "row_selection": row["used_for_formal_policy_selection"] = True
    if tamper == "row_error": row["quadrature_error_estimate"] = -1.
    if tamper == "row_arithmetic": row["cv_minus_reference"] += .1
    if tamper == "allowance": row["price_allowance"] = .1
    if tamper == "false_pass": row["cv_within_dollar_allowance"] = False
    if tamper == "reference": row["reference_continuation"] += .1
    if tamper == "terminal_flag": row["terminal_exact_payoff"] = True
    if tamper == "nonfinite": row["direct_continuation"] = float("inf")
    if tamper == "physical_steps": r["numerical_diagnostics"]["physical_increments"] = 1
    if tamper == "future_state": r["numerical_diagnostics"]["actual_future_state_used"] = True
    if tamper == "snell_cap": r["numerical_diagnostics"]["only_premium_capped"] = False
    if tamper == "mass": r["numerical_diagnostics"]["max_probability_error"] = .01
    if tamper == "cv_direct": r["numerical_diagnostics"]["max_cv_direct_difference"][0] += .01
    if tamper == "warnings": r["warnings"] = "hidden warning"
    if tamper == "runtime": r["elapsed_seconds"] = -1.
    if tamper == "envelope": r["envelope_constant"] = -1.
    if tamper == "terminal_values":
        r = f.records[12]; row = r["rows"][0]
        row["cv_continuation"] += .0001; row["cv_minus_reference"] += .0001
        r["numerical_diagnostics"]["max_cv_direct_difference"][0] = abs(row["cv_continuation"]-row["direct_continuation"])
    f.persist()
    if tamper == "missing_state": (f.folder/"state_00.json").unlink()
    if tamper == "unexpected_state": write(f.folder/"state_32.json", f.records[0])
    with pytest.raises(ValueError): validate(f)


@pytest.mark.parametrize("table,tamper", [("checks.csv", "duplicate"), ("checks.csv", "numerical"),
    ("state_diagnostics.csv", "missing"), ("state_diagnostics.csv", "runtime"), ("panel.csv", "state")])
def test_summary_csvs_cannot_replace_or_omit_sealed_state_results(sealed_audit, table, tamper):
    f = sealed_audit; path = f.folder/table; frame = pd.read_csv(path, float_precision="round_trip")
    if tamper == "duplicate": frame.iloc[-1] = frame.iloc[0]
    if tamper == "numerical": frame.loc[0, "cv_minus_reference"] = .01
    if tamper == "missing": frame = frame.iloc[:-1]
    if tamper == "runtime": frame.loc[0, "elapsed_seconds"] = 99.
    if tamper == "state": frame.loc[0, "stock"] += .1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): validate(f)


def test_inaccurate_but_completed_quadrature_is_retained_without_requiring_all_pass(sealed_audit):
    f = sealed_audit; r = f.records[0]; row = r["rows"][0]
    row["cv_continuation"] += .01; row["cv_minus_reference"] += .01; row["cv_within_dollar_allowance"] = False
    r["numerical_diagnostics"]["max_cv_direct_difference"][0] = abs(row["cv_continuation"]-row["direct_continuation"])
    f.persist(); result = validate(f)
    assert not result["spent_scalar_checks"].iloc[0].cv_within_dollar_allowance


def test_failed_integrations_remain_visible_in_complete_audit(sealed_audit):
    f = sealed_audit; state = f.records[0]["state"]
    f.records[0] = {"state": state, "status": "failed", "elapsed_seconds": 180.001,
        "error_type": "TimeoutError", "error_message": "Declared wall budget exhausted", "rows": [
            {**{k: state[k] for k in ("state_index", "seed", "path_index", "exercise_index", "role")},
                "order": order, "status": "failed", "error_type": "TimeoutError",
                "error_message": "Declared wall budget exhausted", "used_for_formal_policy_selection": False}
            for order in audit.ORDERS]}
    f.persist(); result = validate(f)
    assert len(result["spent_scalar_checks"]) == 96
    assert result["spent_scalar_checks"].status.eq("failed").sum() == 3
    assert f.manifest["failed_states"] == 1
    table = pd.read_csv(f.folder/"checks.csv"); table = table[table.status != "failed"]
    table.to_csv(f.folder/"checks.csv", index=False)
    with pytest.raises(ValueError): validate(f)


def test_exact_source_closure_tracks_required_modules_without_fixed_cardinality(sealed_audit, monkeypatch):
    f = sealed_audit
    required = audit.required_audit_sources() | {"src/rvlab/pricing/future_dependency.py"}
    monkeypatch.setattr(audit, "required_audit_sources", lambda: required)
    f.design["source_sha256"]["src/rvlab/pricing/future_dependency.py"] = "0"*64
    f.persist()
    assert len(validate(f)["spent_scalar_states"]) == 32
    # An equally sized replacement still omits a required dependency.
    f.design["source_sha256"].pop("src/rvlab/pricing/future_dependency.py")
    f.design["source_sha256"]["src/rvlab/pricing/unexpected_replacement.py"] = "0"*64
    f.persist()
    with pytest.raises(ValueError, match="required numerical source"):
        validate(f)
