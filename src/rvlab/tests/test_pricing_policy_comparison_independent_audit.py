"""The standalone review recalculates statistics without the experiment helper."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rvlab.experiments._pricing_policy_comparison_stats import summarize_case


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("independent_policy_audit",
    ROOT/"scripts/audit_policy_comparison_results.py")
independent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(independent)


def test_independent_moments_matches_hand_calculated_stream_variance():
    chunks = [np.array([1., 2., 3., 4.]), np.array([5., 6.])]
    result = independent.moments(chunks)
    assert result["paths"] == 6
    assert result["estimate"] == 3.5
    expected = np.sqrt(4*(5/3)+2*.5)/6
    assert result["standard_error"] == pytest.approx(expected)
    assert result["interval_low"] == pytest.approx(3.5-1.96*expected)


def fixtures():
    streams = []
    for n,offset in ((4,0.),(6,2.)):
        path = np.arange(n, dtype=float)+offset
        lower = (path[:,None,None,None]+np.array([-.3,.3])[None,:,None,None]
            +np.array([0.,.2,.5])[None,None,:,None])
        data = {"times":np.array([0.,.04,.08]),
            "stop_indices":np.ones((n,2,3,1),np.int64)}
        for j,c in enumerate(independent.CONTROLS):
            data[c+"_lower"] = lower+j*.01*path[:,None,None,None]
            data[c+"_upper"] = np.full((n,1),30.)
        streams.append(data)
    return streams


def tables(tmp_path, result):
    target = tmp_path/"tables"
    target.mkdir()
    declarations = {}
    for name,frame in result.items():
        path = target/(name+".csv")
        frame.to_csv(path,index=False)
        declarations[name] = {"rows":len(frame),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    return independent.Tables(tmp_path,{"tables":declarations})


def test_independent_recalculation_covers_every_conditional_statistical_row(tmp_path):
    streams = fixtures()
    result = summarize_case(streams,case_id="put_90d_k100",arm="matched_paths")
    reader = tables(tmp_path,result)
    independent.audit_case(reader,{"case_id":"put_90d_k100"},"matched_paths",streams)
    reader.finish(tuple(result))
    assert {name:len(rows) for name,rows in reader.seen.items()} == {
        "values":9,"paired_policy_gains":6,"controls":9,"training_variation":27,"exercise":18}
    assert reader.scalars == 726


def test_independent_recalculation_catches_coherently_rehashed_wrong_price(tmp_path):
    streams = fixtures()
    result = summarize_case(streams,case_id="put_90d_k100",arm="matched_paths")
    result["values"].loc[0,"price"] += .001
    reader = tables(tmp_path,result)
    with pytest.raises(AssertionError,match="values.price"):
        independent.audit_case(reader,{"case_id":"put_90d_k100"},"matched_paths",streams)


def test_standalone_audit_imports_no_experiment_implementation():
    import ast
    source = ast.parse((ROOT/"scripts/audit_policy_comparison_results.py").read_text())
    modules = [node.module for node in ast.walk(source) if isinstance(node,ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(source) if isinstance(node,ast.Import) for alias in node.names]
    assert not any(name and (name.startswith("rvlab") or "policy_comparison_stats" in name) for name in modules)


def endpoint_fixture():
    n = 4
    times = np.linspace(0.,.08,65)
    stocks = np.broadcast_to(100.+np.arange(n)[:,None,None]*times[None,:,None],(n,65,1)).copy()
    discount = np.exp(-.04*times)
    payoff = discount[None,:,None]*np.maximum(110.-stocks,0.)
    euro = np.broadcast_to(np.arange(n)[:,None,None]*times[None,:,None],payoff.shape).copy()
    premium = euro*.1
    stops = np.broadcast_to(np.array([0,10,64])[None,None,:,None],(n,2,3,1)).copy()
    trace = {"times":times,"stocks":stocks,"factors":np.zeros((n,65,4)),
        "actual_time_grid":np.linspace(0.,.08,129),"decision_operator_grid":np.linspace(0.,.08,129),
        "discounts":discount,"discounted_payoffs":payoff,"strike":110.,
        "european_martingale":euro,"premium_martingale":premium}
    arrays = {"stop_indices":stops}
    for control,m in (("raw",np.zeros_like(euro)),("european",euro),("european_premium",euro+premium)):
        adjusted = payoff-m
        arrays[control+"_upper"] = adjusted.max(axis=1)
        arrays[control+"_lower"] = np.stack([np.stack([adjusted[np.arange(n),stops[:,r,p,0],0]
            for p in range(3)],axis=1) for r in range(2)],axis=1)[...,None]
    return trace,arrays


def test_independent_endpoint_audit_checks_every_raw_and_controlled_own_stop():
    trace,arrays = endpoint_fixture()
    assert independent.endpoint_audit(trace,arrays,4,4,128) == 72


@pytest.mark.parametrize("defect", ["own_stop","common_upper","physical_grid","payoff"])
def test_independent_endpoint_audit_rejects_coherent_shape_but_wrong_content(defect):
    trace,arrays = endpoint_fixture()
    if defect == "own_stop": arrays["stop_indices"][0,0,0,0] = 10
    elif defect == "common_upper": arrays["european_upper"][0,0] += .01
    elif defect == "physical_grid": trace["actual_time_grid"] = trace["actual_time_grid"][:-1]
    else: trace["discounted_payoffs"][0,1,0] += .01
    with pytest.raises(AssertionError):
        independent.endpoint_audit(trace,arrays,4,4,128)
