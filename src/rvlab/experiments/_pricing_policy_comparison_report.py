"""Readable report assembly; sealed comparison evidence, no hidden refitting."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from . import _pricing_first_stage as old
from . import _pricing_policy_comparison_run as run


FAMILY = "first_stage_v2"
EVIDENCE_VERSION = "pricing_first_stage_v2"
BASELINE = old.ROOT/"notebooks/legacy/pricing/first_stage_policy_baseline_2026-10-21"
BASELINE_SHA = "f9f27d1c6ec471054e64cc2ce3d546392564b656742a7e2b940d3335ec7c83a4"
LABELS = {"cashflow_lsm":"Cashflow LSM", "fourier_premium_lsm":"Fourier-premium LSM",
    "bellman_lookahead":"Bellman lookahead"}
COLORS = {"cashflow_lsm":"#70818a", "fourier_premium_lsm":"#207e89", "bellman_lookahead":"#be713e"}
CASE_IDS = tuple(c["case_id"] for c in run.benchmark_cases())


def load_baseline(stem, *, prefixes=None):
    """Use the original v1 execution identities; a replay is not fresh evidence."""
    archive,index = old._archive_index(BASELINE,BASELINE_SHA)
    entries = {}
    prefix = f"results/first_stage_v1/{stem}"
    path = old._verified(archive,index,prefix+"/manifest.json",entries)
    manifest = run.read_json(path)
    if manifest.get("notebook")!=stem or manifest.get("evidence_version")!="pricing_first_stage_v1":
        raise ValueError("Preserved first-stage baseline identity changed")
    if stem!="03_spy_calibration" and (manifest.get("data_kind")!="synthetic" or "spy_inputs" in manifest):
        raise ValueError("Synthetic policy/calibration reports cannot load SPY observations")
    selected = lambda key: prefixes is None or key.startswith(tuple(prefixes))
    tables,figures = {},{}
    for key,count in manifest["tables"].items():
        if not selected(key):
            continue
        if Path(key).name!=key or key in (".",".."):
            raise ValueError("Invalid preserved table name")
        path=old._verified(archive,index,f"{prefix}/{key}.csv",entries)
        try:
            frame=pd.read_csv(path,float_precision="round_trip")
        except pd.errors.EmptyDataError:
            if count:
                raise
            frame=pd.DataFrame()
        if len(frame)!=count:
            raise ValueError("Preserved baseline table row count changed")
        tables[key]=frame
    for key in manifest["figures"]:
        if selected(key):
            if Path(key).name!=key or key in (".",".."):
                raise ValueError("Invalid preserved figure name")
            figures[key]=old._verified(archive,index,f"{prefix}/{key}.png",entries)
    inventory=archive/"preservation_manifest.json"
    entries[str(inventory)]={"path":str(inventory),"bytes":inventory.stat().st_size,"sha256":BASELINE_SHA}
    return {"tables":tables,"figures":figures,"manifest":manifest,"input_files":list(entries.values())}


def reported_estimate_costs(costs):
    """Cost the TWO rules averaged in the plotted estimate; share trace once.

    The sealed clock allowance applies to each standalone training replica.
    This derived ledger instead matches the plotted conditional two-fit value
    and budget. Artifact I/O and redundant campaign fitting remain separate.
    """
    rows=[]
    for (case,arm,policy),group in costs.groupby(["case_id","arm","policy"],sort=False):
        if len(group)!=2 or set(group.replica)!={0,1}:
            raise ValueError("Reported two-fit cost requires both distinct training replicas")
        shared=group.shared_trace_control_seconds.to_numpy(float)
        if not np.isfinite(shared).all() or not np.all(shared==shared[0]):
            raise ValueError("The two reported fits must share exactly one trace/control cost")
        fit=float(group.fit_seconds.sum())
        online=float(group.online_decision_seconds.sum())
        rows.append({"case_id":case,"arm":arm,"policy":policy,"training_replicas":2,
            "combined_fit_seconds":fit,"combined_online_seconds":online,
            "shared_trace_control_seconds_once":shared[0],
            "reported_estimate_seconds":fit+online+shared[0],
            "per_replica_allowance_unchanged":True,"artifact_io_and_redundant_campaign_fits_included":False})
    return pd.DataFrame(rows)


def inception_value_diagnostics(values,cases):
    """Compare unmodified policy estimates with the KNOWN available payoff.

    This is an audit, not a post-valuation policy choice or a price floor.
    A suboptimal admissible policy can be worth less than immediate exercise.
    """
    definitions={case["case_id"]:case for case in cases}
    rows=[]
    for _,row in values.loc[values.primary].iterrows():
        case=definitions[row.case_id]
        intrinsic=max((case["spot"]-case["strike"]) if case.get("is_call",False)
            else case["strike"]-case["spot"],0.)
        rows.append({"case_id":row.case_id,"arm":row.arm,"policy":row.policy,
            "known_immediate_payoff":intrinsic,"unmodified_policy_value":row.price,
            "policy_value_minus_immediate":row.price-intrinsic,
            "conditional_interval_high":row.price_interval_high,
            "conditional_interval_below_immediate":row.price_interval_high<intrinsic,
            "new_policy_selected_or_value_floored":False,
            "cause_is_identified_as_implementation_bug":False})
    return pd.DataFrame(rows)


def comparison_figures(tables):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    figures={}
    fig,ax=plt.subplots(figsize=(12,3.2))
    ax.axis("off")
    boxes=[(.01,.30,.19,.4,"Independent training\n2 replicas per contract"),
        (.26,.16,.23,.68,"Three frozen policies\n\nCashflow LSM\nFourier-premium LSM\nBellman lookahead"),
        (.55,.30,.18,.4,"Same independent\nvaluation paths"),
        (.79,.16,.20,.68,"Own stopping payoffs\nPaired policy gains\nCommon dual upper\n\nSeparate cost ledger")]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012",facecolor="#edf4f4",edgecolor="#88a8ad"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)
    for left,right in ((.21,.25),(.50,.54),(.74,.78)):
        ax.annotate("",xy=(right,.5),xytext=(left,.5),arrowprops={"arrowstyle":"->","color":"#51767e","lw":1.8})
    ax.set_title("Change the exercise rule; keep the evaluation problem fixed",loc="left",pad=15)
    fig.tight_layout();figures["policy_comparison_design_flow"]=fig
    values=tables["values"]
    if values.empty:
        # A closed failed campaign is still a report, never a synthetic fallback.
        for key in ("policy_values","policy_gains","control_precision","cost_accuracy","exercise","refinement"):
            fig,axis=plt.subplots(figsize=(10,3))
            axis.axis("off")
            axis.text(.5,.5,"No complete independent-path comparison available.\nSee the retained attempt and failure tables.",
                ha="center",va="center",transform=axis.transAxes)
            figures["policy_comparison_"+key]=fig
        return figures
    primary=values.loc[values.primary]
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    for axis,case in zip(axes.flat,CASE_IDS):
        for arm,offset,marker in ((run.ARMS[0],-.12,"o"),(run.ARMS[1],.12,"s")):
            block=primary.loc[primary.case_id.eq(case)&primary.arm.eq(arm)].set_index("policy")
            for i,policy in enumerate(run.n.POLICY_NAMES):
                if policy not in block.index:
                    continue
                row=block.loc[policy]
                axis.errorbar(row.price,i+offset,xerr=1.96*row.price_se,fmt=marker,
                    color=COLORS[policy],capsize=3,label=arm.replace("matched_","") if i==0 else None)
            if not block.empty:
                axis.axvline(block.upper.iloc[0],ls="--",lw=1,color="#8b6262")
                if arm==run.ARMS[0] and "upper_se" in block:
                    mean,se=block.upper.iloc[0],block.upper_se.iloc[0]
                    axis.axvspan(mean-1.96*se,mean+1.96*se,color="#8b6262",alpha=.12)
        definition=next(c for c in run.benchmark_cases() if c["case_id"]==case)
        intrinsic=max(definition["strike"]-definition["spot"],0.)
        if intrinsic>0:
            axis.axvline(intrinsic,ls=":",lw=1.3,color="#444")
            axis.text(intrinsic,1.01,"Immediate exercise $10",transform=axis.get_xaxis_transform(),
                ha="right",fontsize=8,color="#444")
        axis.set(title=case.replace("put_","").replace("_k"," · K=").replace("d"," days"),
            yticks=range(3),yticklabels=[LABELS[p] for p in run.n.POLICY_NAMES],xlabel="Policy value ($) · conditional 95% interval")
        axis.invert_yaxis();axis.grid(axis="x",alpha=.2)
    axes.flat[0].legend(title="Budget arm",frameon=False,fontsize=9)
    fig.suptitle("Independent-path policy values · dashed line is the same EP upper",fontsize=14)
    fig.tight_layout();figures["policy_comparison_policy_values"]=fig
    gains=tables["paired_policy_gains"]
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for row,arm in enumerate(run.ARMS):
        for column,candidate in enumerate(("fourier_premium_lsm","bellman_lookahead")):
            axis=axes[row,column]
            block=gains.loc[gains.arm.eq(arm)&gains.primary&gains.candidate_policy.eq(candidate)].set_index("case_id").reindex(CASE_IDS)
            label="Fourier − cashflow" if candidate=="fourier_premium_lsm" else "Bellman − Fourier"
            axis.errorbar(block.estimate,np.arange(6),xerr=block.half_width,fmt="o",capsize=3,
                color=COLORS[candidate],label="Fourier − cashflow" if candidate=="fourier_premium_lsm" else "Bellman − Fourier")
            axis.axvline(0,color="#888",lw=1);axis.grid(axis="x",alpha=.2)
            axis.set(title=arm.removeprefix("matched_").title()+" · "+label,
                xlabel="Paired payoff gain ($ per unit) · EP control",
                yticks=range(6),yticklabels=[c.replace("put_","").replace("_k"," · K=") for c in CASE_IDS])
            axis.ticklabel_format(axis="x",style="sci",scilimits=(-3,3),useMathText=True)
            axis.legend(frameon=False,loc="upper right",fontsize=9)
    axes[0,0].invert_yaxis()
    fig.suptitle("A changed rule must earn its gain on independent paths",fontsize=14)
    fig.tight_layout();figures["policy_comparison_policy_gains"]=fig
    controls=tables["controls"]
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),sharey=True,sharex=True)
    for row,arm in enumerate(run.ARMS):
        for column,policy in enumerate(run.n.POLICY_NAMES):
            axis=axes[row,column]
            block=controls.loc[controls.arm.eq(arm)&controls.policy.eq(policy)]
            for control,offset,color,label in (("european",-.12,"#268577","European control"),
                    ("european_premium",.12,"#2878a5","European + premium")):
                chosen=block.loc[block.control.eq(control)].set_index("case_id").reindex(CASE_IDS)
                ratios=np.sqrt(chosen.variance_ratio_to_raw)
                axis.plot(ratios,np.arange(6)+offset,"o",color=color,label=label)
            zero=block.loc[block.control.eq("raw")&block.raw_price_se.eq(0)].case_id
            for case in zero:
                axis.text(.025,CASE_IDS.index(case),"N/A · raw SE = 0",transform=axis.get_yaxis_transform(),va="center",fontsize=8)
            axis.axvline(1,color="#999",ls="--",lw=1)
            axis.set(title=LABELS[policy]+" · "+arm.removeprefix("matched_").replace("_"," "),
                xscale="log",xlabel="Controlled SE / raw SE · smaller is more precise",
                yticks=range(6),yticklabels=[c.replace("put_","").replace("_k"," · K=") for c in CASE_IDS])
            axis.grid(axis="x",alpha=.2)
    axes[0,0].invert_yaxis()
    axes[1,1].legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.24),ncol=2)
    fig.suptitle("Change the estimator, holding each exercise rule fixed",fontsize=14)
    fig.tight_layout();figures["policy_comparison_control_precision"]=fig
    costs=reported_estimate_costs(tables["costs"])
    joined=primary.merge(costs,on=["case_id","arm","policy"],validate="one_to_one")
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.8),sharey=True)
    markers=dict(zip(CASE_IDS,("o","s","^","D","P","X")))
    for axis,arm in zip(axes,run.ARMS):
        for policy in run.n.POLICY_NAMES:
            block=joined.loc[joined.arm.eq(arm)&joined.policy.eq(policy)]
            for _,row in block.iterrows():
                axis.scatter(row.reported_estimate_seconds,row.price_error_budget/row.price_allowance,
                    marker=markers[row.case_id],color=COLORS[policy],
                    label=LABELS[policy] if row.case_id==CASE_IDS[0] else None,s=64,edgecolor="white")
        axis.axhline(1,color="#b95844",ls="--",label="Unchanged price allowance")
        axis.set(xscale="log",yscale="log",xlabel="Cost of the reported two-fit estimate (seconds)",
            title=arm.replace("_"," "))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Price error budget / unchanged allowance")
    axes[1].legend(frameon=False,fontsize=9)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker=markers[c],color="#60737c",ls="",label=c.replace("put_","").replace("_k"," · K=")) for c in CASE_IDS]
    fig.legend(handles=handles,loc="lower center",ncol=6,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));figures["policy_comparison_cost_accuracy"]=fig
    exercise=tables["exercise"]
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),sharey=True)
    for axis,strike in zip(axes,(90,100,110)):
        for policy in run.n.POLICY_NAMES:
            block=exercise.loc[exercise.case_id.eq(f"put_90d_k{strike}")&exercise.arm.eq(run.ARMS[0])
                &exercise.policy.eq(policy)&exercise.record_type.eq("exercise_date_distribution")].sort_values("exercise_time")
            axis.step(block.exercise_time*365,block.probability.cumsum(),where="post",color=COLORS[policy],label=LABELS[policy])
        axis.set(title=f"90-day put · K={strike}",xlabel="Exercise date (days)",ylim=(-.02,1.02));axis.grid(alpha=.2)
    axes[0].set_ylabel("Cumulative fraction stopped · replica average")
    axes[-1].legend(frameon=False,fontsize=9)
    fig.tight_layout();figures["policy_comparison_exercise"]=fig
    refinement=tables["refinement"]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    for axis,name in zip(axes,("lookahead_order","physical_steps","kernel_factors")):
        if refinement.empty or not refinement.refinement.eq(name).any():
            axis.text(.5,.5,"Unavailable · retained failed attempt",ha="center",va="center",transform=axis.transAxes)
            axis.set_title(name.replace("_"," "))
            continue
        block=refinement.loc[refinement.refinement.eq(name)&refinement.control.eq(run.PRIMARY)].set_index("policy").reindex(run.n.POLICY_NAMES)
        axis.errorbar(block.change,range(3),xerr=1.96*block.change_se,fmt="o",capsize=4,color="#207e89")
        axis.axvline(0,color="#888",lw=1)
        allowance=float(block.allowance.max())
        axis.axvspan(-allowance,allowance,color="#79af99",alpha=.2)
        axis.set(title={"lookahead_order":"GH16 → GH8 decisions","physical_steps":"128 → 256 physical steps","kernel_factors":"4 → 8 factors + retraining"}[name],
            xlabel="Paired policy value change ($)",yticks=range(3),yticklabels=[LABELS[p] for p in run.n.POLICY_NAMES])
        axis.grid(axis="x",alpha=.2);axis.invert_yaxis()
    fig.suptitle("ATM 90-day diagnostics · sampling intervals do not cover approximation error",fontsize=13)
    fig.tight_layout();figures["policy_comparison_refinement"]=fig
    return figures


def _stamp(result,role):
    result["manifest"].update(evidence_version=EVIDENCE_VERSION,artifact_family=FAMILY,
        report_role=role,tolerances_changed=False,research_project=old.PROJECT_TITLE)
    return result


def policy_comparison_pricing(context,*,seed=42,fourier_config=None,fd_config=None,campaign=None):
    old._seed(seed)
    from ._pricing_research import pricing_research
    from ._pricing_first_stage_reporting import add_first_stage_reporting
    context.progress("Running BS/Heston references and verifying the sealed policy comparison")
    result=pricing_research(context,seed=seed,fourier_config=fourier_config,fd_config=fd_config)
    historical=load_baseline("01_pricing_and_exercise",prefixes=("rough_",))
    result["tables"].update(historical["tables"])
    result["figures"].update(historical["figures"])
    tables,manifest,design=run.load_closed(run.DEFAULT_FOLDER if campaign is None else campaign)
    result["tables"].update({"policy_comparison_"+key:frame for key,frame in tables.items()})
    if not tables["costs"].empty:
        result["tables"]["policy_comparison_reported_estimate_costs"]=reported_estimate_costs(tables["costs"])
    if not tables["values"].empty:
        result["tables"]["policy_comparison_inception_diagnostics"]=inception_value_diagnostics(tables["values"],design["cases"])
    result["figures"].update(comparison_figures(tables))
    folder=Path(run.DEFAULT_FOLDER if campaign is None else campaign).resolve()
    inputs=[folder/"design_before_training.json",folder/"manifest.json",folder/"preservation_inventory.json"]
    result["manifest"].update(policy_comparison={**copy.deepcopy(manifest),"folder":str(folder),
        "design":design,"input_files":[{"path":str(p),"bytes":p.stat().st_size,"sha256":run.sha(p)} for p in inputs]},
        rough_benchmark={**historical["manifest"]["rough_benchmark"],"historical_preserved_evidence":True},
        historical_focused_evidence=copy.deepcopy(historical["manifest"]["historical_focused_evidence"]),
        historical_input_files=historical["input_files"],historical_archive=str(BASELINE),historical_archive_sha256=BASELINE_SHA,
        historical_execution={key:historical["manifest"].get(key) for key in ("created_at","source_sha256","interpreter","versions")},
        focused_evidence_recomputed=False,generic_v1_recomputed=False,first_stage_report_complete=manifest["complete"],
        american_pricing_ready=False,stock_delta_ready=False,optimal_delta_ready=False,
        new_comparison_greek_validation=False)
    return add_first_stage_reporting(_stamp(result,"matched_exercise_policy_comparison"))


def policy_comparison_calibration(context,**kwargs):
    return _stamp(old.first_stage_calibration(context,**kwargs),"synthetic_calibration_and_identification")


def policy_comparison_spy(context,**kwargs):
    return _stamp(old.first_stage_spy(context,**kwargs),"market_data_handoff")
