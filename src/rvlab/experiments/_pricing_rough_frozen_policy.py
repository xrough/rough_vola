"""Reconstruct the sealed joint-grid policy without new model selection.

The first reconstruction repeats only the original deterministic base fit and
the already-selected controlled regression on its saved training trace. Every
decision coefficient and the old endpoint replay must match exactly. The
resulting explicit JSON/NPZ bundle avoids pickle and subsequent retraining.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np


PROFILE = "16a0ade1e6eda5dc"
CHECKPOINT = "case_05_472e71ccbae710e8.json"
BUNDLE_SUBDIR = "rough_martingale_centering_v1/frozen_joint_policy"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _config(payload):
    from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    values = dict(payload)
    values["stock_knots"] = tuple(values["stock_knots"])
    if values.get("dual_proxy_knots") is not None:
        values["dual_proxy_knots"] = tuple(values["dual_proxy_knots"])
    values["premium_control"] = RoughEuropeanControlConfig(**values["premium_control"])
    if values.get("dual_config") is not None:
        raise ValueError("Frozen source unexpectedly contains an automatic dual fit")
    return RoughAmericanConfig(**values)


def _market(payload):
    from rvlab.pricing.vanilla_types import SpotMarket, DiscountCurve, DividendSchedule, CashDividend
    return SpotMarket(payload["spot"], DiscountCurve(**payload["discount_curve"]),
        DividendSchedule(tuple(CashDividend(**event) for event in payload["dividends"]["events"]),
            payload["dividends"]["coverage"]))


def _weights(payload, proxy):
    from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights
    options = dict(payload["config"])
    for key in ("candidate_bins", "european_bounds", "premium_bounds", "predictable_features",
            "components", "intrinsic_bounds", "intrinsic_feature_bounds", "poststop_bounds",
            "poststop_feature_bounds", "european_feature_bounds", "premium_feature_bounds",
            "dispersion_penalties"):
        if key in options:
            options[key] = tuple(options[key])
    return RoughDualWeights(np.asarray(payload["time_edges"]), np.asarray(payload["weights"]),
        payload["policy_fingerprint"], proxy.config.training_seed,
        payload["training_seed"], payload["reserved_valuation_seed"],
        RoughDualConfig(**options), payload["diagnostics"])


def _load_trace(record, name):
    from ._pricing_rough_stopping_attribution import _verify_sample_archives
    _verify_sample_archives(record)
    matches = [a for a in record["sample_archives"] if Path(a["path"]).name == name+"_trace.npz"]
    if len(matches) != 1:
        raise ValueError("Frozen source requires one sealed trace: "+name)
    with np.load(matches[0]["path"], allow_pickle=False) as arrays:
        trace = {key: arrays[key].copy() for key in arrays.files}
    trace.update(record["trace_metadata"][name])
    return trace


def _assert_identities(proxy, original, selected, declaration):
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    from ._pricing_rough_policy_improvement import _policy_model_identity
    if fourier_policy_fingerprint(proxy) != declaration["proxy_fingerprint"]:
        raise ValueError("Reconstructed frozen proxy fingerprint differs from the sealed source")
    for name, policy in (("original", original), (declaration["selected_policy"], selected)):
        if _canonical(_policy_model_identity(policy)) != _canonical(declaration["policy_model_identities"][name]):
            raise ValueError("Reconstructed frozen decision coefficients differ: "+name)


def _replay(record, proxy, original, selected, weights):
    from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
    trace = _load_trace(record, "final_valuation")
    source = next(a for a in record["sample_archives"] if Path(a["path"]).name == "paired_policy_endpoints.npz")
    with np.load(source["path"], allow_pickle=False) as arrays:
        expected = {key: arrays[key].copy() for key in arrays.files}
    comparisons = []
    for name, policy in (("original", original), (record["selected_policy"], selected)):
        evaluated = evaluate_trace_policy(trace, policy=policy, martingale_weights=weights)
        index = record["policy_names"].index(name)
        pairs = {"raw_payoffs": (evaluated.raw_samples[:, 0], expected["raw_payoffs"][:, index]),
            "controlled_primals": (evaluated.controlled_samples[:, 0], expected["controlled_primals"][:, index]),
            "stop_indices": (evaluated.stop_indices[:, 0], expected["stop_indices"][:, index]),
            "dual_samples": (evaluated.dual_samples[:, 0], expected["dual_samples"])}
        for quantity, (actual, target) in pairs.items():
            exact = bool(np.array_equal(actual, target))
            comparisons.append({"policy": name, "quantity": quantity, "bit_identical": exact,
                "max_absolute_change": float(np.max(np.abs(actual-target)))})
            if not exact:
                raise ValueError("Frozen endpoint replay differs: "+name+" / "+quantity)
    return {"paths": len(expected["dual_samples"]), "old_paths_are_spent": True,
        "new_acceptance_evidence": False, "all_bit_identical": True, "comparisons": comparisons}


def _encode_models(models, prefix, arrays):
    from rvlab.pricing.rough_stopping import _Continuation
    from rvlab.pricing.rough_tensor_premium import TensorPremium
    declaration = []
    for index, model in enumerate(models):
        if type(model) not in (_Continuation, TensorPremium):
            raise ValueError("Frozen bundle only supports explicit continuation and tensor models")
        content = {}
        for field in fields(model):
            value = getattr(model, field.name)
            if isinstance(value, np.ndarray):
                key = f"{prefix}_{index}_{field.name}"
                arrays[key] = value
                content[field.name] = {"array": key}
            else:
                content[field.name] = value
        declaration.append({"class": type(model).__name__, "fields": content})
    return declaration


def _decode_models(declaration, arrays):
    from rvlab.pricing.rough_stopping import _Continuation
    from rvlab.pricing.rough_tensor_premium import TensorPremium
    classes = {"_Continuation": _Continuation, "TensorPremium": TensorPremium}
    result = []
    for model in declaration:
        if set(model) != {"class", "fields"} or model["class"] not in classes:
            raise ValueError("Unknown class in explicit frozen model bundle")
        cls = classes[model["class"]]
        if set(model["fields"]) != {field.name for field in fields(cls)}:
            raise ValueError("Malformed frozen continuation fields")
        values = {}
        for key, value in model["fields"].items():
            if isinstance(value, dict):
                if set(value) != {"array"} or value["array"] not in arrays:
                    raise ValueError("Malformed frozen coefficient array")
                value = arrays[value["array"]]
            if key == "knots":
                value = tuple(value)
            values[key] = value
        result.append(cls(**values))
    return tuple(result)


def load_frozen_policy_bundle(folder):
    """Load sealed explicit coefficients; no paths, regression or dual fitting."""
    from rvlab.pricing.types import RoughHestonParams
    from rvlab.pricing.rough_lift import ExponentialKernel
    from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
    folder = Path(folder)
    manifest_path, array_path = folder/"manifest.json", folder/"coefficients.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "frozen_joint_policy_v1" or _sha(array_path) != manifest["coefficient_archive_sha256"]:
        raise ValueError("Changed frozen coefficient bundle")
    source = manifest["source_checkpoint"]
    if _sha(source["path"]) != source["sha256"]:
        raise ValueError("Changed frozen policy source checkpoint")
    with np.load(array_path, allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files}
    continuations = _decode_models(manifest["original_models"], arrays)
    selected_models = _decode_models(manifest["selected_models"], arrays)
    config = _config(manifest["base_config"])
    contract = manifest["contract"]
    base = FourierStoppingPolicy(RoughHestonParams(**manifest["params"]), _market(manifest["market"]),
        ExponentialKernel(**manifest["kernel"]), contract["strike"], contract["maturity"],
        contract["is_call"], "american", arrays["exercise_times"], arrays["continuation_times"],
        continuations, continuations, config, config.premium_control,
        manifest["declaration"]["policy_model_identities"]["original"]["exercise_at_inception"],
        manifest["original_diagnostics"], config.premium_cap)
    proxy = build_rough_obstacle_proxy(base, stock_knots=manifest["obstacle_stock_knots"])
    declaration = manifest["declaration"]
    original_id = declaration["policy_model_identities"]["original"]
    selected_id = declaration["policy_model_identities"][declaration["selected_policy"]]
    original = ControlledExercisePolicy(proxy, arrays["policy_times"], continuations,
        original_id["exercise_at_inception"], tuple(original_id["training_seeds"]),
        original_id["floor"], original_id["cap"], manifest["original_diagnostics"])
    selected = ControlledExercisePolicy(proxy, arrays["policy_times"], selected_models,
        selected_id["exercise_at_inception"], tuple(selected_id["training_seeds"]),
        selected_id["floor"], selected_id["cap"], manifest["selected_diagnostics"])
    weights = _weights(manifest["dual"], proxy)
    _assert_identities(proxy, original, selected, declaration)
    return {"proxy": proxy, "original_policy": original, "selected_policy": selected,
        "weights": weights, "manifest": manifest, "folder": str(folder.resolve())}


def reconstruct_joint_frozen_policy(output_dir):
    """Verify exact sealed fits once and persist a reusable no-pickle bundle."""
    from rvlab.pricing.types import RoughHestonParams
    from rvlab.pricing.rough_lift import ExponentialKernel
    from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy, fit_controlled_policy_iteration
    from rvlab.pricing.rough_tensor_premium import tensor_premium_fitter
    from rvlab.pricing.rough_dual import _edges
    from ._pricing_rough_stopping_attribution import _freeze_json
    root = Path(output_dir)
    folder = root/BUNDLE_SUBDIR
    if (folder/"manifest.json").exists():
        return load_frozen_policy_bundle(folder)
    checkpoint = root/"rough_policy_improvement_v1"/PROFILE/CHECKPOINT
    record = json.loads(checkpoint.read_text())
    payload, specification = record["payload"], record["specification"]
    frozen = payload["frozen_decisions"]
    if _sha(frozen["path"]) != frozen["sha256"]:
        raise ValueError("Changed policy decisions sealed before original valuation")
    declaration = json.loads(Path(frozen["path"]).read_text())
    if (specification["case_index"] != 5 or payload["selected_policy"] != "dense tensor quadratic"
            or payload["selected_control"] != "obstacle / micro boundary"):
        raise ValueError("Unexpected frozen policy target")
    case = specification["design"][5]
    config, contract = _config(payload["config"]), case["contract"]
    params = RoughHestonParams(**specification["models"][contract["model"]])
    market = _market(specification["market"])
    kernel_payload = specification["kernels"][str(case["factors"])]
    kernel = ExponentialKernel(**kernel_payload)
    base = train_fourier_stopping_policy(params, market, kernel,
        strike=contract["strike"], maturity=contract["maturity"], is_call=contract["is_call"],
        config=config, control_config=config.premium_control, premium_cap=config.premium_cap)
    proxy = build_rough_obstacle_proxy(base, stock_knots=case["obstacle_proxy"]["stock_knots"])
    dual_diag = payload["dual_fit_diagnostics"][payload["selected_control"]]
    dual = {"time_edges": _edges(proxy.exercise_times, len(declaration["dual_weights"])).tolist(),
        "weights": declaration["dual_weights"], "policy_fingerprint": declaration["proxy_fingerprint"],
        "training_seed": dual_diag["dual_training_seed"],
        "reserved_valuation_seed": dual_diag["reserved_valuation_seed"],
        "config": dual_diag["config"], "diagnostics": dual_diag}
    weights = _weights(dual, proxy)
    trace = _load_trace(payload, "policy_improvement_training")
    selected = fit_controlled_policy_iteration(proxy, trace, martingale_weights=weights,
        premium_floor=0., premium_cap=.7, fit_continuation=tensor_premium_fitter(degree=2, ridge=1e-4),
        stock_knots=case["obstacle_proxy"]["stock_knots"], ridge=1e-4)
    selected_id = declaration["policy_model_identities"][payload["selected_policy"]]
    selected = replace(selected, training_seeds=tuple(selected_id["training_seeds"]),
        diagnostics=declaration["policy_diagnostics"][payload["selected_policy"]])
    original_id = declaration["policy_model_identities"]["original"]
    original = ControlledExercisePolicy(proxy, trace["times"], proxy.continuations,
        original_id["exercise_at_inception"], tuple(original_id["training_seeds"]),
        original_id["floor"], original_id["cap"], declaration["policy_diagnostics"]["original"])
    _assert_identities(proxy, original, selected, declaration)
    replay = _replay(payload, proxy, original, selected, weights)
    arrays = {"exercise_times": proxy.exercise_times, "continuation_times": proxy.continuation_times,
        "policy_times": trace["times"]}
    original_models = _encode_models(original.continuations, "original", arrays)
    selected_models = _encode_models(selected.continuations, "selected", arrays)
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder/"coefficients.npz"
    if array_path.exists():
        raise ValueError("Refusing to replace an incomplete frozen coefficient bundle")
    np.savez_compressed(array_path, **arrays)
    manifest = {"version": "frozen_joint_policy_v1", "profile": PROFILE, "case_index": 5,
        "source_checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)},
        "sealed_decisions": frozen, "declaration": declaration,
        "coefficient_archive_sha256": _sha(array_path),
        "params": asdict(params), "market": asdict(market), "kernel": asdict(kernel),
        "base_config": asdict(config), "contract": contract,
        "obstacle_stock_knots": case["obstacle_proxy"]["stock_knots"],
        "original_models": original_models, "selected_models": selected_models,
        "original_diagnostics": dict(original.diagnostics), "selected_diagnostics": dict(selected.diagnostics),
        "dual": dual, "replay": replay, "new_dual_fit": False, "new_selection": False,
        "pickle_used": False, "old_valuation_used_only_for_exact_replay": True,
        "source_sha256": _sha(__file__)}
    _freeze_json(folder/"manifest.json", manifest)
    return load_frozen_policy_bundle(folder)
