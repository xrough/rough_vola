"""Independently audit a CLOSED policy-comparison campaign from saved arrays.

This script does not import the experiment's numerical, statistics or runner
modules. It generates no trajectories and never selects a policy. NumPy
recalculates the conditional path-paired statistics; standard-library hashing
checks provenance. Run after collection, with no campaign writers active.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT/"notebooks/_output/pricing/first_stage_v2/policy_comparison_v1"
DESIGN_SHA = "f82ddb79128de9e4173d342fdb89079b68b501799682f4c24ce47db5df43722f"
POLICIES = ("cashflow_lsm", "fourier_premium_lsm", "bellman_lookahead")
CONTROLS = ("raw", "european", "european_premium")
ARMS = ("matched_paths", "matched_allowance")
CONTRASTS = ((1, 0), (2, 1))
TOLERANCES = {"price_atol": .001, "price_rtol": 1e-4,
    "delta_atol": .002, "delta_rtol": .005, "confidence_multiplier": 1.96,
    "spread_fraction": .1, "tolerances_changed": False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(dt.utcoffset().total_seconds() == 0, "timestamp is not explicitly UTC")
    return dt


def filemap(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def inventory(folder):
    item = read(folder/"preservation_inventory.json")
    entries = item["files"]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(hashlib.sha256(canonical.encode()).hexdigest() == item["entries_sha256"], "inventory digest changed")
    require(len(entries) == item["entry_count"] == len({x["path"] for x in entries}), "inventory duplicates/count")
    expected = sorted(str(p.relative_to(folder)) for p in folder.rglob("*")
        if p.is_file() and p.name != "preservation_inventory.json")
    require(expected == sorted(x["path"] for x in entries), "inventory not closed")
    for entry in entries:
        path = (folder/entry["path"]).resolve()
        require(path.is_relative_to(folder.resolve()), "inventory escaped root")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
            "inventory bytes/hash changed: "+entry["path"])
    return len(entries)


def moments(chunks):
    """Independent rederivation of the frozen within-stream MCSE convention."""
    total = sum(len(x) for x in chunks)
    require(total >= 4 and all(np.ndim(x) == 1 and len(x) >= 2 and np.isfinite(x).all() for x in chunks),
        "bad independent-stream sample")
    mean = sum(np.sum(x, dtype=float) for x in chunks)/total
    variance = sum(len(x)*np.var(x, ddof=1) for x in chunks)/total
    se = np.sqrt(variance/total)
    return {"paths": total, "estimate": mean, "standard_error": se,
        "half_width": 1.96*se, "interval_low": mean-1.96*se,
        "interval_high": mean+1.96*se, "within_stream_variance": variance}


class Tables:
    def __init__(self, folder, manifest):
        self.frames, self.seen, self.scalars = {}, {}, 0
        for name, declaration in manifest["tables"].items():
            path = folder/"tables"/(name+".csv")
            require(sha(path) == declaration["sha256"], "CSV hash changed: "+name)
            try:
                frame = pd.read_csv(path, float_precision="round_trip")
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            require(len(frame) == declaration["rows"], "CSV row count changed: "+name)
            self.frames[name], self.seen[name] = frame, set()

    def check(self, name, keys, expected):
        frame = self.frames[name]
        mask = np.ones(len(frame), bool)
        for key, value in keys.items():
            mask &= frame[key].isna().to_numpy() if value is None else (frame[key] == value).to_numpy()
        indices = np.flatnonzero(mask)
        require(len(indices) == 1, f"nonunique/missing {name} row: {keys}")
        index = int(indices[0])
        require(index not in self.seen[name], f"duplicate audit selection {name}: {keys}")
        self.seen[name].add(index)
        row = frame.iloc[index]
        for key, value in expected.items():
            actual = row[key]
            if isinstance(value, (bool, np.bool_, str)):
                require(actual == value, f"{name}.{key}: {actual!r} != {value!r}; {keys}")
            elif value is None or np.isnan(value):
                require(pd.isna(actual), f"{name}.{key} should be NaN; {keys}")
            else:
                require(np.isclose(actual, value, rtol=2e-12, atol=2e-13),
                    f"{name}.{key}: {actual!r} != {value!r}; {keys}")
            self.scalars += 1

    def finish(self, names):
        for name in names:
            require(len(self.seen[name]) == len(self.frames[name]),
                f"unreviewed/extra {name} rows: {len(self.frames[name])-len(self.seen[name])}")


def check_sources_and_design(folder):
    design = read(folder/"design_before_training.json")
    require(sha(folder/"design_before_training.json") == DESIGN_SHA, "preapproved design SHA changed")
    require(design["tolerances"] == TOLERANCES, "acceptance tolerances changed")
    require(design["parameters"] == {"v0": .04, "theta": .045, "kappa": 1.5,
        "vol_of_vol": .3, "rho": -.6, "hurst": .15}, "model changed")
    require(design["arms"] == list(ARMS) and design["policies"] == list(POLICIES)
        and design["controls"] == list(CONTROLS), "candidate ordering changed")
    require(design["training_replicas"] == 2 and design["valuation_streams"] == 2
        and design["valuation_paths_per_stream"] == 2048 and design["training_paths"] == 2048
        and design["steps"] == 128 and design["exercise_steps"] == 64 and design["lookahead_order"] == 16
        and design["initial_spots"] == [100.] and design["unique_main_valuation_paths"] == 24576,
        "path/grid budgets changed")
    require(not design["winner_or_hyperparameter_selection"] and design["selection_paths"] == 0,
        "selection was introduced")
    pins = design["source_sha256"]
    require(len(pins) == 68, "source closure count changed")
    require(set(filemap(folder/"source")) == set(pins), "source snapshot not closed")
    for name, digest in pins.items():
        require(sha(ROOT/name) == digest == sha(folder/"source"/name), "live/snapshot source changed: "+name)
    require(inventory(folder/"timing_pilot") == 8, "timing pilot closure changed")
    pilot = read(folder/"timing_pilot/manifest.json")
    require(sha(folder/"timing_pilot/manifest.json") == design["timing_manifest_sha256"], "pilot parent changed")
    require(utc(pilot["created_at"]) < utc(design["created_at"]), "timing/design chronology changed")
    for case, timing in zip(design["cases"], pilot["records"]):
        require(case["case_id"] == timing["case_id"], "pilot case order changed")
        names = {"cashflow_lsm": "cashflow_lsm", "fourier_premium_lsm": "fourier_premium_lsm",
            "bellman_lookahead": "bellman_lookahead_total"}
        def cost(name, count):
            return timing["training_seconds"][names[name]]*count/32 + (
                timing["policy_evaluation_seconds"][name]+timing["trace_generation_seconds"])*4096/32
        allowance = 1.25*max(cost(name, 2048) for name in POLICIES)
        require(allowance == case["clock_allowance_seconds"], "timing allowance changed")
        for name in POLICIES:
            candidates = [v for v in (512,1024,2048,4096,8192) if cost(name,v) <= allowance]
            require(case["clock_training_paths"][name] == (max(candidates) if candidates else 512),
                "timing-only training allocation changed")
    seeds = ([s for c in design["cases"] for s in c["training_seeds"]]
        +design["refinement"]["training_seeds"]
        +[s for c in design["cases"] for s in c["valuation_seeds"]]
        +[s for ss in design["refinement"]["valuation_seeds"].values() for s in ss]
        +[r[k] for r in pilot["records"] for k in ("training_seed","valuation_seed")])
    require(len(seeds) == len(set(seeds)) == 44, "training/valuation/pilot seed registry changed")
    prereview = read(folder/"prereview.json")
    require(prereview["design_sha256"] == DESIGN_SHA and not prereview["scientific_outcomes_inspected"],
        "prereview identity changed")
    return design


def check_phases(folder, design):
    training = read(folder/"training_complete.json")
    require(training["all_attempts_complete"] and len(training["records"]) == 26, "training not closed")
    trainfiles = filemap(folder/"training")
    require(training["file_sha256"] == {"training/"+k:v for k,v in trainfiles.items()}, "training file closure changed")
    require(training["design_sha256"] == DESIGN_SHA, "training parent changed")
    expected = {}
    for case in design["cases"]:
        for arm in ARMS:
            for replica in range(2):
                key = f"{case['case_id']}_{arm}_r{replica}"
                expected[key] = (case["case_id"],arm,replica,case["training_seeds"][replica],
                    {p:2048 for p in POLICIES} if arm == ARMS[0] else case["clock_training_paths"],False)
    for replica in range(2):
        expected[f"refinement_kernel_r{replica}"] = ("put_90d_k100",ARMS[0],replica,
            design["refinement"]["training_seeds"][replica],{p:2048 for p in POLICIES},True)
    seen = set()
    for row in training["records"]:
        key = (f"refinement_kernel_r{row['replica']}" if row["kernel_refinement"] else
            f"{row['case_id']}_{row['arm']}_r{row['replica']}")
        require(key in expected and key not in seen, "training identity changed")
        seen.add(key)
        require((row["case_id"],row["arm"],row["replica"],row["training_seed"],row["training_paths"],
            row["kernel_refinement"]) == expected[key], "training budget/seed changed")
        require(row == read(folder/"training"/key/"manifest.json"), "training phase record differs")
        require(utc(design["created_at"]) <= utc(row["started_at"]) <= utc(row["completed_at"])
            <= utc(training["created_at"]), "training chronology changed")
    require(seen == set(expected), "missing training identity")
    phases = {}
    for phase, number in (("valuation",12),("refinement",6)):
        record = read(folder/(phase+"_complete.json"))
        require(record["all_attempts_complete"] and len(record["records"]) == number, "phase incomplete: "+phase)
        expected = ({f"{c['case_id']}_s{s}": (c["case_id"],s,c["valuation_seeds"][s])
            for c in design["cases"] for s in range(2)} if phase == "valuation" else
            {f"{name}_s{s}": (name,s,seeds[s])
                for name,seeds in design["refinement"]["valuation_seeds"].items() for s in range(2)})
        seen = set()
        for row in record["records"]:
            identity = row["case_id"] if phase == "valuation" else row["name"]
            key = f"{identity}_s{row['stream']}"
            require(key in expected and key not in seen, "phase identity changed")
            seen.add(key)
            require((identity,row["stream"],row["seed"]) == expected[key], "phase seed changed")
            require(row == read(folder/phase/key/"manifest.json"), "phase record differs")
            require(row["design_sha256"] == DESIGN_SHA
                and row["training_complete_sha256"] == sha(folder/"training_complete.json"), "phase parent changed")
            require(utc(training["created_at"]) <= utc(row["started_at"]) <= utc(row["completed_at"])
                <= utc(record["created_at"]), "final paths before training closed")
            require(row["status"] in ("completed","failed"), "attempt status hidden")
            if row["status"] == "failed":
                require(isinstance(row.get("error_type"),str) and row["error_type"]
                    and isinstance(row.get("message"),str), "failure not identifiable")
        require(seen == set(expected), "phase missing attempt")
        phases[phase] = record
    return training, phases


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k].copy() for k in source.files}


def endpoint_audit(trace, arrays, count, factor_count, physical_steps):
    times, stocks, factors = trace["times"], trace["stocks"], trace["factors"]
    require(stocks.shape == (count,65,1) and factors.shape == (count,65,factor_count), "trace dimensions changed")
    require(len(trace["actual_time_grid"]) == physical_steps+1
        and len(trace["decision_operator_grid"]) == 129, "actual/original grid changed")
    require(np.all(stocks > 0) and np.all(stocks[:,0] == 100.) and np.all(factors[:,0] == 0.), "initial states changed")
    require(np.isfinite(stocks).all() and np.isfinite(factors).all(), "nonfinite simulated state")
    require(np.allclose(trace["discounts"],np.exp(-.04*times),rtol=0,atol=3e-16), "discount convention changed")
    payoff = trace["discounted_payoffs"]
    require(np.array_equal(payoff,trace["discounts"][None,:,None]*np.maximum(trace["strike"]-stocks,0.)), "payoff changed")
    stops = arrays["stop_indices"]
    require(stops.shape == (count,2,3,1) and stops.dtype.kind in "iu"
        and np.all((stops >= 0)&(stops < 65)), "own stops changed")
    checks = 0
    for control in CONTROLS:
        m = np.zeros_like(payoff)
        if control != "raw": m += trace["european_martingale"]
        if control == "european_premium": m += trace["premium_martingale"]
        require(np.all(m[:,0] == 0.) and np.isfinite(m).all(), "martingale origin/finite values changed")
        adjusted = payoff-m
        require(np.array_equal(arrays[control+"_upper"],adjusted.max(axis=1)), "upper is not exact common dual maximum")
        for r in range(2):
            for p in range(3):
                expected = adjusted[np.arange(count),stops[:,r,p,0],0]
                require(np.array_equal(expected,arrays[control+"_lower"][:,r,p,0]), "lower changed own payoff/control")
                require(np.all(expected <= arrays[control+"_upper"][:,0]+2e-11), "pathwise bracket violated")
                checks += 1
        for a,b in CONTRASTS:
            gain = arrays[control+"_lower"][:,:,a]-arrays[control+"_lower"][:,:,b]
            gapb = arrays[control+"_upper"][:,None]-arrays[control+"_lower"][:,:,b]
            gapa = arrays[control+"_upper"][:,None]-arrays[control+"_lower"][:,:,a]
            require(np.allclose(gapb-gapa,gain,rtol=0,atol=2e-13), "common-upper gap identity changed")
    return checks*count


def audit_case(tables, case, arm, streams):
    common = {"case_id":case["case_id"],"arm":arm}
    average = {c:[s[c+"_lower"][...,0].mean(axis=1) for s in streams] for c in CONTROLS}
    scope = {"conditional_on_frozen_training":True,"training_uncertainty_in_interval":False,
        "pointwise_interval":True,"used_for_policy_selection":False,
        "optimal_continuous_price_certified":False,"training_replicas":2}
    for p,policy in enumerate(POLICIES):
        raw = [x[:,p] for x in average["raw"]]
        raw_m = moments(raw)
        for control in CONTROLS:
            key = {**common,"policy":policy,"control":control}
            lows = [x[:,p] for x in average[control]]
            ups = [s[control+"_upper"][:,0] for s in streams]
            lo,hi = moments(lows),moments(ups)
            gap = moments([u-l for u,l in zip(ups,lows)])
            budget = gap["estimate"]+1.96*(lo["standard_error"]+gap["standard_error"])
            allowance = .001+1e-4*abs(lo["estimate"])
            tables.check("values",key,{**scope,"primary":control=="european_premium",
                "paths":lo["paths"],"price":lo["estimate"],"price_se":lo["standard_error"],
                "price_interval_low":lo["interval_low"],"price_interval_high":lo["interval_high"],
                "upper":hi["estimate"],"upper_se":hi["standard_error"],
                "gap":gap["estimate"],"gap_se":gap["standard_error"],"price_error_budget":budget,
                "price_allowance":allowance,"price_budget_pass":budget<=allowance,
                "upper_is_shared_dual_maximum":True,"replicas_averaged_within_path":True})
            shift = moments([a-b for a,b in zip(lows,raw)])
            ratio = lo["within_stream_variance"]/raw_m["within_stream_variance"] if raw_m["within_stream_variance"]>0 else np.nan
            tables.check("controls",key,{"paths":lo["paths"],"price_se":lo["standard_error"],
                "raw_price_se":raw_m["standard_error"],"variance_ratio_to_raw":ratio,
                "variance_reduction_fraction":1-ratio,"paired_shift_from_raw":shift["estimate"],
                "paired_shift_se":shift["standard_error"],"paired_shift_interval_low":shift["interval_low"],
                "paired_shift_interval_high":shift["interval_high"],
                "shift_zero_in_interval":shift["interval_low"]<=0<=shift["interval_high"],
                "shift_plus_uncertainty":abs(shift["estimate"])+shift["half_width"],
                "same_frozen_policy":True,"control_unbiasedness_established_by_this_test":False})
            for r in range(2):
                stat = moments([s[control+"_lower"][:,r,p,0] for s in streams])
                tables.check("training_variation",{**key,"record_type":"individual_replica","replica_id":r},
                    {**stat,"between_training_population_uncertainty_estimated":False})
            stat = moments([s[control+"_lower"][:,1,p,0]-s[control+"_lower"][:,0,p,0] for s in streams])
            tables.check("training_variation",{**key,"record_type":"paired_replica_difference","replica_id":None},
                {**stat,"reference_replica_id":0,"comparison_replica_id":1,
                    "between_training_population_uncertainty_estimated":False})
        stops = np.concatenate([s["stop_indices"][:,:,p,0] for s in streams])
        times = streams[0]["times"]
        for replica in (0,1,None):
            chosen = stops if replica is None else stops[:,replica]
            tables.check("exercise",{**common,"policy":policy,"replica_id":replica,
                "record_type":"replica_average" if replica is None else "replica_summary"},
                {"paths":len(stops),"inception_probability":np.mean(chosen==0),
                    "early_exercise_probability":np.mean(chosen<len(times)-1),
                    "expiry_probability":np.mean(chosen==len(times)-1),"mean_stop_years":np.mean(times[chosen])})
        for j,time in enumerate(times):
            tables.check("exercise",{**common,"policy":policy,"record_type":"exercise_date_distribution","exercise_index":j},
                {"paths":len(stops),"exercise_time":time,"probability":np.mean(stops==j)})
    for a,b in CONTRASTS:
        for control in CONTROLS:
            stat = moments([x[:,a]-x[:,b] for x in average[control]])
            tables.check("paired_policy_gains",{**common,"candidate_policy":POLICIES[a],"reference_policy":POLICIES[b],"control":control},
                {**scope,**stat,"primary":control=="european_premium",
                    "zero_in_interval":stat["interval_low"]<=0<=stat["interval_high"],
                    "positive_interval":stat["interval_low"]>0,"same_paths_same_upper":True,
                    "gap_reduction_equals_controlled_gain":True,"independent_dual_improvement":False})


def audit_costs(tables, folder, case, records):
    shared = sum(r["trace_generation"]["trace_generation_seconds"] for r in records)
    for arm in ARMS:
        for r in range(2):
            fit = read(folder/"training"/f"{case['case_id']}_{arm}_r{r}"/"manifest.json")
            for policy in POLICIES:
                seconds = fit["metrics"]["training_seconds"]["bellman_lookahead_total" if policy==POLICIES[2] else policy]
                online = sum(x["policy_metrics"][arm][r]["policy_evaluation_seconds"][policy] for x in records)
                total = seconds+shared+online
                allowance = case["clock_allowance_seconds"]
                tables.check("costs",{"case_id":case["case_id"],"arm":arm,"replica":r,"policy":policy},
                    {"training_paths":fit["training_paths"][policy],"fit_seconds":seconds,
                        "online_decision_seconds":online,"shared_trace_control_seconds":shared,
                        "standalone_total_seconds":total,"allowance_seconds":allowance,
                        "allowance_overrun":total>allowance,"actual_unused_seconds":allowance-total,
                        "within_allowance_is_equal_consumption":False,
                        "redundant_training_campaign_seconds":fit["metrics"]["total_elapsed_seconds"],
                        "online_decisions_charged":4096})


def audit_refinements(tables, folder, design, phase):
    checks = 0
    for name,seeds in design["refinement"]["valuation_seeds"].items():
        records = [read(folder/"refinement"/f"{name}_s{s}"/"manifest.json") for s in range(2)]
        if not all(r["status"]=="completed" for r in records):
            continue
        streams = []
        for s,record in enumerate(records):
            base = folder/"refinement"/f"{name}_s{s}"
            require(record["sample_sha256"] == sha(base/"samples.npz"), "refinement payload changed")
            data = load_npz(base/"samples.npz")
            for label in ("base","variant"):
                trace_label = "base" if name=="lookahead_order" else label
                trace = load_npz(base/(trace_label+"_trace.npz"))
                trace.update(read(base/(trace_label+"_trace_metadata.json")))
                a = {k.removeprefix(label+"__"):v for k,v in data.items() if k.startswith(label+"__")}
                checks += endpoint_audit(trace,a,512,8 if label=="variant" and name=="kernel_factors" else 4,
                    256 if label=="variant" and name=="physical_steps" else 128)
            streams.append(data)
        for c in CONTROLS:
            if name=="lookahead_order":
                require(all(np.array_equal(s["base__"+c+"_upper"],s["variant__"+c+"_upper"]) for s in streams),
                    "GH-order change altered the common upper")
            for p,policy in enumerate(POLICIES):
                b = [s["base__"+c+"_lower"][:,:,p,0].mean(axis=1) for s in streams]
                v = [s["variant__"+c+"_lower"][:,:,p,0].mean(axis=1) for s in streams]
                bu = [s["base__"+c+"_upper"][:,0] for s in streams]
                vu = [s["variant__"+c+"_upper"][:,0] for s in streams]
                difference = moments([a-z for a,z in zip(v,b)])
                bm,vm = moments(b),moments(v)
                cross = moments([l-u for l,u in zip(b,vu)])
                reverse = moments([l-u for l,u in zip(v,bu)])
                allowance = .001+1e-4*abs(bm["estimate"])
                budget = abs(difference["estimate"])+difference["half_width"]
                tables.check("refinement",{"case_id":"put_90d_k100","refinement":name,"policy":policy,"control":c},
                    {"baseline":bm["estimate"],"variant":vm["estimate"],"change":difference["estimate"],
                        "change_se":difference["standard_error"],"ci_low":difference["interval_low"],
                        "ci_high":difference["interval_high"],"baseline_upper":moments(bu)["estimate"],
                        "variant_upper":moments(vu)["estimate"],"baseline_gap":moments([u-l for u,l in zip(bu,b)])["estimate"],
                        "variant_gap":moments([u-l for u,l in zip(vu,v)])["estimate"],
                        "baseline_lower_minus_variant_upper":cross["estimate"],"cross_bracket_se":cross["standard_error"],
                        "cross_bracket_ci_low":cross["interval_low"],"cross_bracket_ci_high":cross["interval_high"],
                        "variant_lower_minus_baseline_upper":reverse["estimate"],
                        "reverse_cross_bracket_ci_low":reverse["interval_low"],"reverse_cross_bracket_ci_high":reverse["interval_high"],
                        "change_plus_uncertainty":budget,"allowance":allowance,"tolerance_resolved":budget<=allowance,
                        "paths":1024,"scope":records[0]["scope"],"same_physical_paths":name=="lookahead_order"})
    return checks


def audit(folder):
    folder = Path(folder).resolve()
    manifest = read(folder/"manifest.json")
    require(manifest["complete"] and manifest["design_sha256"]==DESIGN_SHA, "campaign not closed")
    closure_entries = inventory(folder)
    design = check_sources_and_design(folder)
    training, phases = check_phases(folder,design)
    for phase in ("training","valuation","refinement"):
        require(manifest[phase+"_complete_sha256"] == sha(folder/(phase+"_complete.json")), "final phase parent changed")
    tables = Tables(folder,manifest)
    endpoints, complete_cases = 0,0
    for case in design["cases"]:
        records = [read(folder/"valuation"/f"{case['case_id']}_s{s}"/"manifest.json") for s in range(2)]
        successes = sum(r["status"]=="completed" for r in records)
        tables.check("case_status",{"case_id":case["case_id"]},{"declared_streams":2,"completed_streams":successes,
            "complete":successes==2,"partial_case_excluded":successes!=2,"declared_paths":4096,
            "accepted_comparison_paths":4096 if successes==2 else 0})
        if successes != 2:
            continue
        complete_cases += 1
        by_arm = {arm:[] for arm in ARMS}
        for s,record in enumerate(records):
            base = folder/"valuation"/f"{case['case_id']}_s{s}"
            require(record["sample_sha256"]==sha(base/"samples.npz") and record["trace_sha256"]==sha(base/"trace.npz"),
                "main payload hash changed")
            data,trace = load_npz(base/"samples.npz"),load_npz(base/"trace.npz")
            trace.update(read(base/"trace_metadata.json"))
            require(tuple(data["policy_names"])==POLICIES, "saved policy order changed")
            for arm in ARMS:
                arrays = {k.removeprefix(arm+"__"):v for k,v in data.items() if k.startswith(arm+"__")}
                arrays.update({c+"_upper":data[c+"_upper"] for c in CONTROLS})
                arrays["times"] = data["times"]
                endpoints += endpoint_audit(trace,arrays,2048,4,128)
                by_arm[arm].append(arrays)
        for arm in ARMS:
            audit_case(tables,case,arm,by_arm[arm])
        audit_costs(tables,folder,case,records)
    endpoints += audit_refinements(tables,folder,design,phases["refinement"])
    tables.finish(("values","paired_policy_gains","controls","training_variation","exercise","costs","refinement","case_status"))
    failed = [r for phase in (training,phases["valuation"],phases["refinement"]) for r in phase["records"] if r["status"]!="completed"]
    require(len(tables.frames["failures"])==manifest["failure_count"]==len(failed), "failure count changed")
    require(manifest["all_successful"]==(not bool(failed)), "failure hidden by success flag")
    require(manifest["new_main_valuation_paths"] == sum(2048 for r in phases["valuation"]["records"] if r["status"]=="completed"), "unique main paths counted incorrectly")
    require(manifest["new_refinement_paths"] == sum(512*(1 if r["name"]=="lookahead_order" else 2)
        for r in phases["refinement"]["records"] if r["status"]=="completed"), "refinement paths counted incorrectly")
    require(not manifest["american_pricing_ready"] and not manifest["optimal_delta_ready"]
        and not manifest["tolerances_changed"], "readiness promoted or tolerances changed")
    return {"reviewer":"independent methods/statistics agent", "completed_at":datetime.now(timezone.utc).isoformat(),
        "audit_pass":True,"script_sha256":sha(__file__),"script_imports_experiment_implementation":False,
        "generated_new_paths":False,"selected_policy_or_changed_settings":False,
        "design_sha256":DESIGN_SHA,"campaign_manifest_sha256":sha(folder/"manifest.json"),
        "campaign_inventory_sha256":sha(folder/"preservation_inventory.json"),"closed_inventory_entries":closure_entries,
        "live_and_snapshot_sources_checked":68,"training_attempts":26,"main_valuation_attempts":12,"refinement_attempts":6,
        "complete_cases":complete_cases,"failed_attempts_retained":len(failed),
        "recomputed_table_rows":{name:len(tables.seen[name]) for name in tables.seen if tables.seen[name]},
        "numeric_and_logical_field_comparisons":tables.scalars,"path_endpoint_formula_checks":endpoints,
        "upper_shared_bitwise_and_all_own_stop_endpoint_algebra":True,
        "conditional_within_path_replica_averaging_and_within_stream_MCSE":True,
        "original_price_gap_plus_sampling_gate_reproduced":True,"actual_cost_and_overrun_accounting_reproduced":True,
        "statistical_scope":"pointwise MC intervals condition on two fixed training fits; no general training-uncertainty, optimal-price, continuous-model or Greek certification",
        "python":sys.version,"numpy":np.__version__,"pandas":pd.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=DEFAULT_FOLDER)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"audit_pass":result["audit_pass"],"recomputed_table_rows":result["recomputed_table_rows"],
        "field_comparisons":result["numeric_and_logical_field_comparisons"],"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
