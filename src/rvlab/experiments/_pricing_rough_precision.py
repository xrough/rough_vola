"""Rough-Heston simulation accuracy: preserve the baseline before new trials.

This study is synthetic and inception-state.  Its recorded European controls do
not establish American exercise, stock delta or historical-state readiness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd


HISTORICAL_ARCHIVE = (Path(__file__).resolve().parents[3]/
    "notebooks/legacy/pricing/rough_simulation_precision_2026-10-18")


def _read_verified_archive(archive):
    """Validate every preserved byte before presenting historical results."""
    archive = Path(archive).resolve()
    manifest_path = archive/"preservation_manifest.json"
    inventory = json.loads(manifest_path.read_text())
    for row in inventory["files"]:
        path = (archive/row["preserved"]).resolve()
        if not path.is_relative_to(archive):
            raise ValueError("Historical archive paths must remain inside the archive")
        if not path.is_file():
            raise ValueError(f"Missing preserved rough-Heston evidence: {path}")
        payload = path.read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed preserved rough-Heston evidence: {path}")
    return inventory, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def load_historical_foundation(context=None, *, archive=None, compute_figures=True):
    """Load the verified pre-fix study without relabelling or recomputing it.

    PNGs are returned as file paths, preserving their original pixels.  The
    artifact writer and presentation layer both support such recorded figures.
    Original execution/source fingerprints and trial costs remain historical;
    loading the archive does not measure the cost of a new numerical campaign.
    """
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = HISTORICAL_ARCHIVE if archive is None else Path(archive)
    inventory, fingerprint = _read_verified_archive(archive)
    recorded = json.loads((archive/"results/manifest.json").read_text())
    if (recorded.get("evidence_version") != "pricing_research_v1" or
            recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get("rough_foundation", {}).get("backend") !=
            "rough_heston_lifted_simulation_foundation"):
        raise ValueError("Preserved results are not the rough-Heston foundation study")
    tables = {}
    prefix = "rough_foundation_"
    for name, rows in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        path = archive/"results"/f"{name}.csv"
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != rows:
            raise ValueError(f"Historical table row count changed: {path}")
        tables[name.removeprefix(prefix)] = table
    figures = {name.removeprefix(prefix): archive/"results"/f"{name}.png"
        for name in recorded["figures"] if name.startswith(prefix)} if compute_figures else {}
    manifest = deepcopy(recorded["rough_foundation"])
    manifest.update({"historical_replay": True,
        "historical_archive": str(archive.resolve()),
        "historical_archive_manifest_sha256": fingerprint,
        "historical_archive_files": len(inventory["files"]),
        "historical_execution_created_at": recorded["created_at"],
        "current_engine_execution": False,
        "american_pricing_ready": False, "stock_delta_ready": False})
    if context is not None:
        context.progress("Loading byte-verified pre-fix rough-Heston evidence; no historical trial is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}


def pooled_independent_summary(means, standard_errors, path_counts):
    """Pool all independent IID-path replications, including between-run shifts.

    This reconstructs the ordinary sample mean and sample variance of the
    concatenated independent paths from sufficient statistics.  It is not a
    randomized-QMC error estimate, a paired-grid difference, or a rule for
    choosing the most favorable seed.  Failed replications must remain in the
    trial ledger rather than silently being omitted from a claimed campaign.
    """
    import numpy as np
    means, errors = np.asarray(means, float), np.asarray(standard_errors, float)
    counts = np.asarray(path_counts)
    if (means.ndim < 1 or means.shape != errors.shape or
            counts.ndim != 1 or len(counts) != means.shape[0] or
            counts.dtype.kind not in "iu" or (counts < 2).any() or
            not np.isfinite(means).all() or not np.isfinite(errors).all() or
            (errors < 0).any()):
        raise ValueError("Independent summaries require aligned finite means/nonnegative errors and integer path counts >= 2")
    expanded = counts.reshape((-1,)+(1,)*(means.ndim-1))
    total = int(counts.sum())
    mean = np.sum(expanded*means, axis=0)/total
    within = np.sum((expanded-1)*expanded*errors**2,axis=0)
    between = np.sum(expanded*(means-mean)**2,axis=0)
    error = np.sqrt((within+between)/(total-1)/total)
    return mean, error


def precision_acceptance(estimate, standard_error, *, lift_reference,
                         fractional_reference, lift_stable, fractional_stable):
    """Require both unchanged native-price gates and independent references."""
    from ._pricing_rough_foundation import price_gate
    fixed = price_gate(estimate, lift_reference, standard_error=standard_error)
    full = price_gate(estimate, fractional_reference, standard_error=standard_error)
    return {"same_lift_absolute_error": fixed["absolute_error"],
        "full_model_absolute_error": full["absolute_error"],
        "ci_half_width": full["ci_half_width"],
        "same_lift_allowance": fixed["price_allowance"],
        "full_model_allowance": full["price_allowance"],
        "same_lift_error_allowance_ratio": fixed["conservative_error"]/fixed["price_allowance"],
        "full_model_error_allowance_ratio": full["conservative_error"]/full["price_allowance"],
        "same_lift_accepted": fixed["accepted"] & bool(lift_stable),
        "full_model_accepted": full["accepted"] & bool(fractional_stable),
        "accepted": fixed["accepted"] & full["accepted"] &
            bool(lift_stable) & bool(fractional_stable)}


def _reference_checks(params, contracts, market, kernel, checkpoint):
    """Refine cutoff, quadrature and Riccati axes independently of simulation."""
    from dataclasses import asdict, replace
    import numpy as np
    from rvlab.pricing import PricingConfig
    from ._pricing_rough_foundation import (
        _lift_prices, _reference_prices, _price_records, price_gate,
    )
    base = PricingConfig(192,160.,512)
    resolutions = [("base",base),
        ("quadrature",replace(base,quadrature_nodes=384)),
        ("cutoff_1_5x",replace(base,fourier_cutoff=240.)),
        ("cutoff_2x",replace(base,fourier_cutoff=320.)),
        ("riccati",replace(base,riccati_steps=1024)),
        ("joint",PricingConfig(384,320.,1024))]
    references, stable, rows = {}, {}, []
    for backend,calculate in (("fractional",_reference_prices),("finite lift",_lift_prices)):
        trials = {}
        for label,config in resolutions:
            specification = {"backend":backend,"axis":label,
                "params":asdict(params),"kernel":asdict(kernel),"config":asdict(config)}
            def run(fn=calculate,c=config,b=backend):
                return (fn(params,contracts,market,kernel,c) if b=="finite lift"
                    else fn(params,contracts,market,c))
            record = checkpoint(f"reference_{backend.replace(' ','_')}_{label}", specification,run)
            if record.get("status")=="completed":
                trials[label]=np.asarray(record["prices"])
        if "joint" not in trials:
            raise ValueError(f"No usable {backend} reference; inspect numerical-failure ledger")
        references[backend]=trials["joint"]
        targets = (("joint",trials["joint"]),("base",trials.get("base",trials["joint"])))
        stable[backend] = len(trials)==len(resolutions) and all(
            np.all(price_gate(value,target)["accepted"])
            for value in trials.values() for _,target in targets)
        for label,value in trials.items():
            for target_name,target in targets:
                rows += _price_records(contracts,value,target,
                    stage=f"{backend} {label} vs {target_name}",model="reference",
                    factors=len(kernel.nodes) if backend=="finite lift" else 0,
                    steps=dict(resolutions)[label].riccati_steps,
                    reference_stable=stable[backend])
    return references,stable,rows


def _estimate_trial(params, market, kernel, contracts, *, paths, steps, seed,
                    stock_integral, control_config):
    from dataclasses import asdict
    import numpy as np
    from rvlab.pricing.rough_simulation import RoughSimulationConfig
    from rvlab.pricing.rough_estimators import estimate_rough_european
    result = estimate_rough_european(params,market,kernel,
        strike=contracts.strike.to_numpy(),maturity=contracts.maturity.to_numpy(),
        is_call=contracts.is_call.to_numpy(),
        config=RoughSimulationConfig(paths=paths,steps=steps,seed=seed,
            stock_integral=stock_integral),control_config=control_config)
    arrays = {name:np.asarray(getattr(result,name)).reshape(-1).tolist()
        for name in ("prices","standard_errors","conditional_prices",
                     "conditional_standard_errors","control_means","control_standard_errors")}
    return {"status":"completed",**arrays,"settings":{"simulation":asdict(result.config),"control":asdict(result.control_config)},
        "diagnostics":dict(result.diagnostics),
        "sample_covariances":np.asarray(result.sample_covariances).tolist(),"requested_paths":paths,
        "requested_steps":steps,"seed":seed,"stock_integral":stock_integral,
        "control_config":asdict(control_config)}


def _estimate_rows(contracts, record, references, stable, *, model, stage,
                   paths, steps, seed, stock_integral, replications=1):
    import numpy as np
    rows = []
    for estimator,mean_key,se_key in (
        ("conditional Gaussian","conditional_prices","conditional_standard_errors"),
        ("discrete martingale control","prices","standard_errors")):
        values,se=np.asarray(record[mean_key]),np.asarray(record[se_key])
        gate=precision_acceptance(values,se,lift_reference=references["finite lift"],
            fractional_reference=references["fractional"],lift_stable=stable["finite lift"],
            fractional_stable=stable["fractional"])
        part=contracts.assign(model=model,stage=stage,estimator=estimator,
            paths=paths,steps=steps,seed=seed,stock_integral=stock_integral,
            replications=replications,estimate=values,standard_error=se,
            same_lift_reference=references["finite lift"],
            fractional_reference=references["fractional"],
            lift_reference_stable=bool(stable["finite lift"]),
            fractional_reference_stable=bool(stable["fractional"]))
        for name,value in gate.items():
            part[name]=value
        rows += part.to_dict("records")
    return rows


def _precision_sources():
    import inspect
    folder=Path(__file__).parents[1]/"pricing"
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.glob("*.py")) if p.name != "__init__.py"}
    functions=(pooled_independent_summary,precision_acceptance,_reference_checks,
        _estimate_trial,_estimate_rows,_precision_limits)
    sources["study_numerics"]=hashlib.sha256("\n".join(
        inspect.getsource(fn) for fn in functions).encode()).hexdigest()
    return sources


def _preserve_numeric_sources(folder, sources):
    study_source=Path(__file__)
    study_payload=study_source.read_bytes()
    study_sha=hashlib.sha256(study_payload).hexdigest()
    study_snapshot=Path(folder)/"raw/study_sources"/study_sha/study_source.name
    study_snapshot.parent.mkdir(parents=True,exist_ok=True)
    study_snapshot.write_bytes(study_payload)
    source_root=Path(__file__).parents[1]/"pricing"
    for name,digest in sources.items():
        source=source_root/name
        if not source.is_file():
            continue
        payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:
            raise ValueError(f"Numerical source changed while starting the precision study: {source}")
        destination=Path(folder)/"raw/pricing_sources"/digest/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and destination.read_bytes()!=payload:
            raise ValueError(f"Preserved numerical source has changed: {destination}")
        destination.write_bytes(payload)
    return {"sha256":study_sha,"snapshot":str(study_snapshot)}


def _precision_limits(checkpoint, contracts, market, kernel, *, seed):
    """Small current-engine deterministic and cash-event controls, never old replay."""
    from dataclasses import asdict,replace
    import numpy as np
    from rvlab.pricing import BlackScholesParams,PricingConfig
    from rvlab.pricing.rough_simulation import (
        RoughSimulationConfig,simulate_rough_heston,european_payoff_samples,
    )
    from ._pricing_rough_foundation import _market,_truths,_reference_prices,sample_summary,price_gate
    rows,gains=[],[]
    truth=_truths()["flat"]
    def simulate(params,mkt):
        result=simulate_rough_heston(params,mkt,kernel,
            observation_times=(14/365,60/365,1.),
            config=RoughSimulationConfig(paths=2048,steps=1024,seed=seed,
                stock_integral="exact_drift"))
        prices,errors=[],[]
        if not mkt.dividends.events:
            for row in contracts.itertuples():
                samples=european_payoff_samples(result,strike=row.strike,
                    maturity=row.maturity,is_call=row.is_call,conditional=True)
                mean,error=sample_summary(samples)
                prices.append(float(mean));errors.append(float(error))
        mean,error=sample_summary(result.discounted_gains)
        return {"status":"completed","prices":prices,"standard_errors":errors,
            "gains_mean":mean.tolist(),"gains_standard_error":error.tolist(),
            "diagnostics":dict(result.diagnostics)}
    for label,params,reference_params in (
        ("Constant variance",replace(truth,theta=truth.v0,vol_of_vol=0.),BlackScholesParams(.2)),
        ("Deterministic fractional variance",replace(truth,vol_of_vol=0.),replace(truth,vol_of_vol=0.))):
        record=checkpoint("limit_"+label.replace(" ","_"),
            {"params":asdict(params),"market":asdict(market),"kernel":asdict(kernel),
             "paths":2048,"steps":1024,"seed":seed},lambda p=params:simulate(p,market))
        reference=checkpoint("limit_reference_"+label.replace(" ","_"),
            {"params":asdict(reference_params),"market":asdict(market)},
            lambda p=reference_params:_reference_prices(p,contracts,market,PricingConfig(384,320.,1024)))
        if record.get("status") != "completed" or reference.get("status") != "completed":
            continue
        gate=price_gate(record["prices"],reference["prices"],standard_error=record["standard_errors"])
        table=contracts.assign(control=label,estimate=record["prices"],reference=reference["prices"],
            standard_error=record["standard_errors"])
        for name,value in gate.items():table[name]=value
        rows+=table.to_dict("records")
    cash_market=_market(cash=True)
    record=checkpoint("capped_cash_gains",{"params":asdict(truth),"market":asdict(cash_market),
        "kernel":asdict(kernel),"paths":2048,"steps":1024,"seed":seed},
        lambda:simulate(truth,cash_market))
    if record.get("status")=="completed":
        for i,t in enumerate((14/365,60/365,1.)):
            estimate,error=record["gains_mean"][i],record["gains_standard_error"][i]
            gains.append({"days":int(round(365*t)),"estimate":estimate,"reference":100.,
                "standard_error":error,"z_score":(estimate-100.)/error if error else 0.,
                "statistical_control_only":True,"american_ready":False})
    return rows,gains


def rough_precision_study(context, *, seed=42, paths=2048, steps=(256,1024,2048),
                          final_steps=2048, replications=3, reuse_checkpoints=True,
                          compute_figures=True):
    """Run a predeclared precision campaign; no stopping or fitted control weights.

    Every independent seed contributes to the pooled final estimates.  Integer
    seeds on different grids are separate trials, never paired Brownian paths.
    The four-factor kernel is fixed by the preceding deterministic European
    approximation checks, before inspecting any new Monte Carlo outcome.
    """
    from dataclasses import asdict,replace
    from numbers import Integral
    import numpy as np
    import platform
    import scipy
    from rvlab.pricing import approximate_fractional_kernel
    from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
    from ._pricing_rough_foundation import (
        _checkpoint,_fingerprint,_jsonable,_market,_truths,foundation_contracts,
    )
    for name,value,minimum in (("seed",seed,0),("paths",paths,2),
            ("final_steps",final_steps,1),("replications",replications,1)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(reuse_checkpoints,bool) or not isinstance(compute_figures,bool):
        raise TypeError("reuse_checkpoints and compute_figures must be boolean")
    if (not isinstance(steps,(tuple,list)) or not steps or any(
            isinstance(s,(bool,np.bool_)) or not isinstance(s,Integral) or s<1 for s in steps)
            or tuple(steps)!=tuple(sorted(set(steps)))):
        raise ValueError("steps must be distinct increasing positive integers")
    contracts,truths,market=foundation_contracts(),_truths(),_market()
    truths["Hurst .5"]=replace(truths["flat"],hurst=.5)
    kernel=approximate_fractional_kernel(.15,1.,4,min_time_ratio=.003)
    heston_kernel=approximate_fractional_kernel(.5,1.,1)
    control_config=RoughEuropeanControlConfig(24,100.,512)
    sources=_precision_sources()
    common={"study_version":"rough_precision_v1","seed":int(seed),"paths":int(paths),
        "steps":list(steps),"final_steps":int(final_steps),"replications":int(replications),
        "contracts":contracts.to_dict("records"),"truths":{k:asdict(p) for k,p in truths.items()},
        "market":asdict(market),"kernel":asdict(kernel),"heston_kernel":asdict(heston_kernel),
        "control_config":asdict(control_config),"sources":sources,
        "acceptance":{"price_atol":.001,"price_rtol":1e-4,"ci_multiplier":1.96,
            "delta_atol":.002,"delta_rtol":.005,"effective_half_spread_fraction":.1},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,
            "scipy":scipy.__version__,"pandas":pd.__version__}}
    fingerprint=_fingerprint(common)
    folder=Path(context.output_dir)/"rough_precision_v1"/fingerprint[:16]
    shared_guard=_ensure_shared_helper_guard(folder)
    study_source=_preserve_numeric_sources(Path(context.output_dir)/"rough_precision_v1",sources)
    provenance,failures,prices,trial_rows,reference_rows,control_rows=[],[],[],[],[],[]
    def checkpoint(label,specification,calculate):
        def guarded_calculate():
            try:
                return calculate()
            except MemoryError as exc:
                return {"status":"numerical_failure","error":str(exc)}
        value,origin=_checkpoint(folder,label,{**common,**specification},guarded_calculate,
            reuse=reuse_checkpoints)
        provenance.append(origin)
        if value.get("status")=="numerical_failure":
            failures.append({"trial":label,"error":value["error"],
                "elapsed_seconds":origin["original_elapsed_seconds"]})
        return value
    references,stabilities={},{}
    context.progress("Rough-Heston precision: independent fractional and same-lift European references")
    for model,params in truths.items():
        k=heston_kernel if model=="Hurst .5" else kernel
        references[model],stabilities[model],rows=_reference_checks(params,contracts,market,k,checkpoint)
        reference_rows += [dict(row,model=model) for row in rows]
    design=[("flat","trapezoidal",steps[0],seed,"earlier stock integral")]
    design += [("flat","exact_drift",s,seed,"time refinement") for s in sorted(set((*steps,final_steps)))]
    coarser=[s for s in steps if s<final_steps]
    if coarser:
        design += [(model,"exact_drift",max(coarser),seed,"time refinement")
            for model in truths if model!="flat"]
    design += [(model,"exact_drift",final_steps,seed+i,"independent final")
        for model in truths for i in range(replications)
        if not (model=="flat" and i==0)]
    completed={}
    context.progress("Rough-Heston precision: exact-drift stock integration and zero-mean discrete martingale controls")
    for model,scheme,count,trial_seed,stage in design:
        k=heston_kernel if model=="Hurst .5" else kernel
        label=f"estimate_{model.replace(' ','_')}_{scheme}_{count}_{trial_seed}"
        specification={"model":model,"params":asdict(truths[model]),"kernel":asdict(k),
            "stock_integral":scheme,"steps":int(count),"trial_seed":int(trial_seed)}
        record=checkpoint(label,specification,lambda p=truths[model],kernel_=k,s=count,q=trial_seed,g=scheme:
            _estimate_trial(p,market,kernel_,contracts,paths=paths,steps=s,seed=q,
                stock_integral=g,control_config=control_config))
        origin=provenance[-1]
        trial_rows.append({"model":model,"stock_integral":scheme,"steps":count,"seed":trial_seed,
            "paths":paths,"stage":stage,"status":record.get("status"),
            "elapsed_seconds":origin["original_elapsed_seconds"],"reused":origin["reused"],
            "checkpoint":origin["checkpoint"]})
        if record.get("status")!="completed":continue
        completed[model,scheme,count,trial_seed]=record
        for contract,mean,error,raw_se,cv_se in zip(contracts.to_dict("records"),
                record["control_means"],record["control_standard_errors"],
                record["conditional_standard_errors"],record["standard_errors"]):
            control_rows.append({**contract,"model":model,"stock_integral":scheme,
                "steps":count,"seed":trial_seed,"control_mean":mean,"control_standard_error":error,
                "control_z_score":mean/error if error else 0.,
                "conditional_standard_error":raw_se,"controlled_standard_error":cv_se,
                "variance_reduction_factor":(raw_se/cv_se)**2 if cv_se else None,
                "continuous_reference_used_to_center":False})
        prices+=_estimate_rows(contracts,record,references[model],stabilities[model],
            model=model,stage=stage,paths=paths,steps=count,seed=trial_seed,stock_integral=scheme)
    pooled_rows=[]
    for model in truths:
        records=[completed.get((model,"exact_drift",final_steps,seed+i)) for i in range(replications)]
        if any(record is None for record in records):
            failures.append({"trial":f"pooled_final_{model}",
                "error":"A predeclared replication failed; no incomplete campaign is pooled", "elapsed_seconds":0.})
            continue
        pooled={}
        for mean_key,se_key in (("prices","standard_errors"),
                ("conditional_prices","conditional_standard_errors"),
                ("control_means","control_standard_errors")):
            mean,error=pooled_independent_summary([r[mean_key] for r in records],
                [r[se_key] for r in records],[paths]*replications)
            pooled[mean_key],pooled[se_key]=mean,error
        rows=_estimate_rows(contracts,pooled,references[model],stabilities[model],
            model=model,stage="pooled final",paths=paths*replications,steps=final_steps,
            seed=None,stock_integral="exact_drift",replications=replications)
        prices+=rows;pooled_rows+=rows
    limits,gains=_precision_limits(checkpoint,contracts,market,kernel,seed=seed)
    price_table=pd.DataFrame(prices)
    summaries=[]
    if not price_table.empty:
        for keys,part in price_table.groupby(["stage","model","stock_integral","steps","paths","estimator"],sort=False,dropna=False):
            summaries.append(dict(zip(("stage","model","stock_integral","steps","paths","estimator"),keys))|{
                "contracts":len(part[["maturity","strike","is_call"]].drop_duplicates()),
                "contract_seed_rows":len(part),
                "seed_count":int(part.replications.max()) if part.stage.eq("pooled final").all() else int(part.seed.nunique()),
                "max_price_error":part.full_model_absolute_error.max(),
                "max_ci_half_width":part.ci_half_width.max(),
                "max_same_lift_ratio":part.same_lift_error_allowance_ratio.max(),
                "max_full_model_ratio":part.full_model_error_allowance_ratio.max(),
                "accepted":bool(part.accepted.all())})
    limit_table=pd.DataFrame(limits)
    limit_summary=[]
    if not limit_table.empty:
        for label,part in limit_table.groupby("control",sort=False):
            limit_summary.append({"control":label,"contracts":len(part),
                "max_absolute_error":part.absolute_error.max(),
                "max_ci_half_width":part.ci_half_width.max(),
                "max_error_allowance_ratio":(part.conservative_error/part.price_allowance).max(),
                "accepted":bool(part.accepted.all())})
    tables={"prices":price_table,"summary":pd.DataFrame(summaries),
        "trials":pd.DataFrame(trial_rows),"reference_refinement":pd.DataFrame(reference_rows),
        "limits":limit_table,"limit_summary":pd.DataFrame(limit_summary),"cash_gains":pd.DataFrame(gains),
        "controls":pd.DataFrame(control_rows),"failures":pd.DataFrame(failures),
        "provenance":pd.DataFrame(provenance)}
    prototype_table,prototype_provenance=_prototype_trials(
        Path(context.output_dir)/"rough_precision_v1/prototypes")
    tables["prototypes"]=prototype_table
    final=price_table[price_table.stage.eq("pooled final") & price_table.estimator.eq("discrete martingale control")] if not price_table.empty else pd.DataFrame()
    final_accepted=len(final)==len(contracts)*len(truths) and bool(final.accepted.all())
    manifest={**_jsonable(common),"backend":"rough_heston_simulation_precision",
        "profile_fingerprint":fingerprint,"checkpoint_dir":str(folder),
        "study_source_sha256":study_source["sha256"],"study_source_snapshot":study_source["snapshot"],
        "shared_numeric_helper_guard":shared_guard,
        "design":[{"model":m,"stock_integral":g,"steps":s,"seed":q,"stage":stage}
            for m,g,s,q,stage in design],
        "kernel_selection_basis":"Four factors and .003 first cell fixed by preceding deterministic European gates, stiffness and state dimension; no new simulation outcome selects it",
        "source_elapsed_seconds":sum(p["original_elapsed_seconds"] for p in provenance),
        "timing_interpretation":"Sum of recorded per-checkpoint wall runtimes; concurrent prefills mean this is neither campaign wall time nor CPU time. Checkpoint replay is separate; the separately audited interrupted partial overlap had no checkpoint and an unmeasured runtime, excluded here.",
        "checkpoint_replay":all(p["reused"] for p in provenance),"provenance":provenance,
        "prototype_provenance":prototype_provenance,
        "execution_evidence":_execution_evidence(Path(context.output_dir)/"rough_precision_v1"),
        "prototype_elapsed_seconds":float(prototype_table.elapsed_seconds.sum()) if not prototype_table.empty else 0.,
        "tested_european_prices_accepted":bool(final_accepted),
        "american_pricing_ready":False,"stock_delta_ready":False,
        "limitations":["Acceptance concerns this finite inception-state European sample; no global parameter-range certificate follows.",
            "Discrete martingale controls reduce sampling variance and preserve the discrete model expectation; they do not correct time bias.",
            "A common integer seed on different grids is not a paired-path time-refinement experiment.",
            "Normal 95% pathwise sampling intervals are not simultaneous confidence bands and omit kernel/time/reference error.",
            "No American exercise, stock delta, historical state, calibration or hedge replay is implemented."]}
    return {"tables":tables,"figures":_precision_figures(tables) if compute_figures else {},"manifest":manifest}


def _precision_figures(tables):
    import matplotlib.pyplot as plt
    import numpy as np
    values=tables["prices"]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),constrained_layout=True)
    tenor_colors={14:"#2878A5",60:"#D17B32",365:"#268577"}
    for scheme,marker in (("trapezoidal","x"),("exact_drift","o")):
        part=values[values.model.eq("flat") & values.stock_integral.eq(scheme) &
            values.estimator.eq("discrete martingale control") &
            ~values.stage.eq("pooled final")]
        if not part.empty:
            part=part[part.seed.eq(part.seed.min()) & part.strike.eq(100.)]
            for days,p in part.groupby("days"):
                p=p.sort_values("steps")
                axes[0].errorbar(p.steps,p.estimate-p.same_lift_reference,
                    yerr=p.ci_half_width,marker=marker,capsize=3,
                    linestyle="-" if scheme=="exact_drift" else "none",
                    color=tenor_colors.get(days),label=f"{days}d · {scheme.replace('_',' ')}")
    axes[0].axhline(0.,color="black",lw=.8)
    axes[0].set(xlabel="Nominal time intervals",ylabel="ATM estimate − same-lift Fourier ($)",
        title="Corrected stock steps expose time convergence")
    axes[0].legend(fontsize=8)
    final=values[values.stage.eq("pooled final") & values.model.eq("flat")]
    for estimator,part in final.groupby("estimator",sort=False):
        axes[1].semilogy(np.arange(len(part)),part.standard_error,marker="o",label=estimator)
    axes[1].set(xlabel="Contract: 14 / 60 / 365 days × 95 / 100 / 105",ylabel="Sampling standard error ($, log scale)",
        title="Fixed discrete controls reduce Monte Carlo uncertainty")
    axes[1].legend(fontsize=8)
    contract_order=values[["days","strike","is_call"]].drop_duplicates()
    contract_labels=[f"{row.days}d\n{row.strike:g}{'C' if row.is_call else 'P'}"
        for row in contract_order.itertuples()]
    axes[1].set_xticks(np.arange(len(contract_labels)),contract_labels,fontsize=8)
    axes[1].set_xlabel("Maturity and strike/payoff")
    second,axes=plt.subplots(1,2,figsize=(11.5,4),constrained_layout=True)
    final=values[values.stage.eq("pooled final") & values.estimator.eq("discrete martingale control")]
    for model,part in final.groupby("model",sort=False):
        axes[0].plot(np.arange(len(part)),part.same_lift_error_allowance_ratio,
            marker="o",label=f"{model}: same lift")
        axes[1].plot(np.arange(len(part)),part.full_model_error_allowance_ratio,
            marker="o",label=model)
    for ax in axes:
        ax.axhline(1.,color="#D17B32",linestyle="--",label="Unchanged acceptance threshold")
        ax.set(xlabel="Contract in maturity/strike order",ylabel="(Absolute error + 1.96 SE) / price allowance")
        ax.legend(fontsize=8)
        ax.set_xticks(np.arange(len(contract_labels)),contract_labels,fontsize=8)
        ax.set_xlabel("Maturity and strike/payoff")
    axes[0].set_title("Final same-lift European checks")
    axes[1].set_title("Final checks against the full fractional model")
    return {"bias_and_uncertainty":fig,"final_acceptance":second}


def _local_evidence_path(folder, reference, *, subdirectories=()):
    """Resolve preserved references inside the current local evidence tree.

    Absolute historical locations are metadata, never runtime dependencies.
    Anchored suffixes and the recorded basename locate the copied local file;
    the caller must still verify its recorded content hash.
    """
    folder=Path(folder).resolve()
    reference=Path(reference)
    if not reference.is_absolute() and ".." in reference.parts:
        raise ValueError("Evidence paths cannot traverse outside the local tree")
    relative=[]
    if not reference.is_absolute():relative.append(reference)
    for index,part in enumerate(reference.parts):
        if part==folder.name and index+1<len(reference.parts):
            relative.append(Path(*reference.parts[index+1:]))
    relative.extend(Path(name)/reference.name for name in subdirectories)
    relative.append(Path(reference.name))
    for suffix in relative:
        candidate=(folder/suffix).resolve()
        if candidate.is_relative_to(folder) and candidate.is_file():return candidate
    raise ValueError(f"Missing local evidence for historical reference: {reference}")


def _prototype_trials(folder):
    """Read hash-verified exploratory trials without promoting them to new fits."""
    import numpy as np
    from ._pricing_rough_foundation import price_gate
    folder=Path(folder).resolve()
    rows,provenance=[],[]
    def verify(path,digest,size=None):
        path=_local_evidence_path(folder,path)
        if not path.is_relative_to(folder):
            raise ValueError("Prototype evidence paths must remain inside the trial directory")
        if not path.is_file():raise ValueError(f"Missing prototype evidence: {path}")
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest or (size is not None and len(payload)!=size):
            raise ValueError(f"Changed prototype evidence: {path}")
        return path
    manifest_path=folder/"variance_reduction_manifest.json"
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            verify(item["path"],item["sha256"],item["bytes"])
        provenance.append({"path":str(manifest_path),"sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "verified_files":len(manifest["files"]),"historical_prototype":True,
            "source_note":manifest.get("note")})
        for name,label,scheme in (("flat_256_4096.json","BS surrogate","trapezoidal"),
                ("flat_256_4096_fourier.json","Fourier surrogate; earlier stock rule","trapezoidal"),
                ("exact_flat_1024_4096.json","Exact-drift production pilot","exact_drift")):
            path=folder/name
            record=json.loads(path.read_text())
            means=record.get("controlled_mean",record.get("prices"))
            errors=record.get("controlled_se",record.get("standard_errors"))
            gate=price_gate(means,record["reference"],standard_error=errors)
            diag=record.get("diagnostics",{})
            rows.append({"trial":label,"stock_integral":scheme,"estimator":"discrete martingale control",
                "paths":record.get("n",diag.get("paths")),
                "steps":record.get("steps",1024),"seed":record.get("seed",42),
                "max_absolute_error":float(np.max(gate["absolute_error"])),
                "max_ci_half_width":float(np.max(gate["ci_half_width"])),
                "max_error_allowance_ratio":float(np.max(gate["conservative_error"]/gate["price_allowance"])),
                "accepted":bool(np.all(gate["accepted"])),
                "elapsed_seconds":record.get("seconds",diag.get("elapsed_seconds")),
                "reference_scope":"same finite lift only","historical_prototype":True})
    path=folder/"naive_exact_drift_trials.json"
    if path.is_file():
        record=json.loads(path.read_text())
        for item in (*record["source_records"],record["runner"]):
            verify(item["path"],item["sha256"])
        provenance.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified_files":len(record["source_records"])+1,"historical_prototype":True})
        for trial in record["trials"]:
            part=trial["rows"]
            gate=price_gate([r["estimate"] for r in part],[r["reference"] for r in part],
                standard_error=[r["SE"] for r in part])
            args=trial["args"]
            rows.append({"trial":"Exact drift without martingale variance reduction",
                "stock_integral":"exact_drift","estimator":"conditional Gaussian",
                "paths":args["paths"],"steps":args["steps"],"seed":args["seed"],
                "max_absolute_error":float(np.max(gate["absolute_error"])),
                "max_ci_half_width":float(np.max(gate["ci_half_width"])),
                "max_error_allowance_ratio":float(np.max(gate["conservative_error"]/gate["price_allowance"])),
                "accepted":bool(np.all(gate["accepted"])),"elapsed_seconds":trial["elapsed"],
                "reference_scope":"same finite lift only","historical_prototype":True})
    return pd.DataFrame(rows),provenance


_SHARED_HELPER_FUNCTIONS=("price_gate","sample_summary","_price_args",
    "_reference_prices","_lift_prices","_price_records")
_SHARED_HELPER_CONSTANTS=("PRICE_ATOL","PRICE_RTOL","CONFIDENCE_MULTIPLIER")


def _shared_helper_ast(path):
    """Fingerprint the selected numerical dependencies without executing them."""
    import ast
    nodes=ast.parse(Path(path).read_text()).body
    selected={}
    for node in nodes:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in _SHARED_HELPER_FUNCTIONS:
            selected[node.name]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in _SHARED_HELPER_CONSTANTS:
                    selected[target.id]=hashlib.sha256(ast.dump(node.value,include_attributes=False).encode()).hexdigest()
    if set(selected)!=set(_SHARED_HELPER_FUNCTIONS+_SHARED_HELPER_CONSTANTS):
        raise ValueError(f"Shared numerical dependencies are missing from {path}")
    return selected


def _guard_fingerprint(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _ensure_shared_helper_guard(folder, *, current_source=None, archive=None):
    """Audit missing shared-source metadata without relabelling valid trials.

    The first audit proves selected function/constant ASTs equal the byte-
    preserved pre-campaign source.  It explicitly records its later audit time.
    A subsequent change refuses these checkpoints.  This sidecar does not
    change any numerical value, tolerance, trial specification or profile.
    """
    from datetime import datetime,timezone
    folder=Path(folder)
    current_source=(Path(__file__).with_name("_pricing_rough_foundation.py")
        if current_source is None else Path(current_source))
    archive=HISTORICAL_ARCHIVE if archive is None else Path(archive)
    current_ast=_shared_helper_ast(current_source)
    guard_path=folder/"shared_helper_guard.json"
    if guard_path.is_file():
        guard=json.loads(guard_path.read_text())
        expected=guard.pop("fingerprint",None)
        if expected!=_guard_fingerprint(guard):
            raise ValueError(f"Changed shared-helper guard: {guard_path}")
        # Keep the original absolute path as provenance, but resolve the
        # saved bytes within this local study tree after a checkout is copied.
        snapshot=folder.parent/"raw/study_sources"/guard["source_sha256"]/Path(guard["source_snapshot"]).name
        if (not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()
                !=guard["source_sha256"] or _shared_helper_ast(snapshot)!=guard["selected_ast_sha256"]):
            raise ValueError(f"Changed shared-helper source snapshot: {snapshot}")
        if current_ast!=guard["selected_ast_sha256"]:
            raise ValueError("Shared numerical helper changed; existing precision checkpoints cannot be reused. Create a new study version with a new shared-source baseline.")
        return {"path":str(guard_path),"sha256":hashlib.sha256(guard_path.read_bytes()).hexdigest(),
            "selected_ast_sha256":current_ast,"created_at":guard["created_at"],
            "audit_basis":guard["audit_basis"],"existing_guard":True,
            "resolved_source_snapshot":str(snapshot.resolve())}
    original=archive/"source/src/rvlab/experiments/_pricing_rough_foundation.py"
    inventory_path=archive/"preservation_manifest.json"
    inventory=json.loads(inventory_path.read_text())
    relative=str(original.relative_to(archive))
    item=next((item for item in inventory["files"] if item["preserved"]==relative),None)
    payload=original.read_bytes()
    if (item is None or len(payload)!=item["bytes"] or
            hashlib.sha256(payload).hexdigest()!=item["sha256"]):
        raise ValueError(f"Changed pre-campaign shared-helper evidence: {original}")
    if current_ast!=_shared_helper_ast(original):
        raise ValueError("Shared numerical helper ASTs do not match the archived pre-campaign baseline")
    current_payload=current_source.read_bytes()
    current_sha=hashlib.sha256(current_payload).hexdigest()
    snapshot=folder.parent/"raw/study_sources"/current_sha/current_source.name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes()!=current_payload:
        raise ValueError(f"Changed shared-helper source snapshot: {snapshot}")
    snapshot.write_bytes(current_payload)
    guard={"created_at":datetime.now(timezone.utc).isoformat(),
        "audit_basis":"Late provenance audit after the initial numerical campaign: selected function and constant ASTs equal the verified pre-campaign 321-file archive. The unused historical _simulate_trial change is outside the selected numerical dependencies.",
        "source_path":str(current_source.resolve()),"source_sha256":current_sha,
        "source_snapshot":str(snapshot.resolve()),"selected_ast_sha256":current_ast,
        "pre_campaign_source":str(original.resolve()),"pre_campaign_source_sha256":item["sha256"],
        "pre_campaign_inventory_sha256":hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "numerical_results_recomputed":False,"acceptance_tolerances_changed":False}
    guard["fingerprint"]=_guard_fingerprint(guard)
    folder.mkdir(parents=True,exist_ok=True)
    temporary=guard_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(guard,indent=2)+"\n");temporary.replace(guard_path)
    return {"path":str(guard_path),"sha256":hashlib.sha256(guard_path.read_bytes()).hexdigest(),
        "selected_ast_sha256":current_ast,"created_at":guard["created_at"],
        "audit_basis":guard["audit_basis"],"existing_guard":False}


def _execution_evidence(folder):
    """Link scheduling/prefill records separately from numerical failures."""
    folder=Path(folder).resolve()
    entries=[]
    for relative in ("raw/campaign_scheduling_audit.json","raw/prefill_curve_2048_44.json",
            "prototypes/hurst_checkpoint_prefill_completion.json"):
        path=folder/relative
        if not path.is_file():continue
        record=json.loads(path.read_text())
        runner_name=record.get("runner_path",record.get("runner"))
        expected=record.get("runner_sha256")
        if runner_name is None and "records" in record:
            runner=folder/"prototypes/hurst_checkpoint_prefill.py"
            hashes={item["runner_sha256"] for item in record["records"]}
            if len(hashes)!=1:raise ValueError("Prefill runner source hashes disagree")
            expected=next(iter(hashes))
        else:
            runner=_local_evidence_path(folder,runner_name,subdirectories=("raw","prototypes"))
        if (not runner.resolve().is_relative_to(folder) or not runner.is_file() or
                hashlib.sha256(runner.read_bytes()).hexdigest()!=expected):
            raise ValueError(f"Changed precision execution runner: {runner}")
        entries.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "runner_path":str(runner),"runner_sha256":expected,
            "historical_runner_path":runner_name,
            "scope":"Scheduling or independent prefill evidence; not an additional numerical failure or extra accepted sample"})
    return entries
