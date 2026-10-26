"""Reproduce the local weighted-Jacobian review without calibrating a surface.

Run with the project's interpreter. The default 80/24/60 probe takes roughly
12 seconds; --include-alternate also evaluates the preliminary 100/32/100 grid.
This diagnostic does not choose starts or change any calibration settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

from rvlab.pricing import FDConfig, evaluate_vanilla, parameter_values, replace_parameters
from rvlab.pricing._vanilla_common import build_grid
from rvlab.pricing.optimization import _finite_difference_jacobian
from rvlab.experiments import _pricing_american_identification as study


def run_probe(config, role):
    contracts = study.joint_contracts()
    training = contracts.loc[contracts.split.eq("train")]
    market = study.joint_market()
    truths, _, bounds = study.joint_specifications()
    grid = build_grid(market, training.strike.to_numpy(), config, variance=True)
    scales = np.maximum(training.half_spread.to_numpy(), study.SPREAD_FLOOR)
    records = []
    for model, truth in truths.items():
        begin = time.perf_counter()
        physical = parameter_values(truth)
        names, values = list(physical), np.asarray(list(physical.values()), float)
        lower = np.asarray([bounds[model][name][0] for name in names])
        upper = np.asarray([bounds[model][name][1] for name in names])

        def weighted_prices(candidate):
            params = replace_parameters(truth, **dict(zip(names, candidate)))
            result = evaluate_vanilla(params, market=market,
                strike=training.strike.to_numpy(), maturity=training.maturity.to_numpy(),
                is_call=training.is_call.to_numpy(), exercise_style="american",
                config=config, grid=grid)
            return np.asarray(result.prices) / scales

        two, details_two = _finite_difference_jacobian(weighted_prices, values,
            lower=lower, upper=upper, method="2-point")
        three, details_three = _finite_difference_jacobian(weighted_prices, values,
            lower=lower, upper=upper, method="3-point")
        norms = np.linalg.norm(three, axis=0)
        singular = np.linalg.svd(three, compute_uv=False)
        normalized = np.linalg.svd(three / norms, compute_uv=False)
        record = {"model": model, "role": role, "config": asdict(config),
            "physical_parameters": physical, "physical_bounds": bounds[model],
            "training_contracts": training[["contract_id", "strike", "maturity",
                "is_call", "half_spread"]].to_dict("records"),
            "effective_half_spreads": scales.tolist(),
            "two_point_details": dict(zip(names, details_two)),
            "three_point_details": dict(zip(names, details_three)),
            "column_norms": dict(zip(names, norms.tolist())),
            "relative_2_vs_3_point_differences": dict(zip(names,
                (np.linalg.norm(two-three, axis=0)/norms).tolist())),
            "weighted_singular_values": singular.tolist(),
            "weighted_condition_number": float(singular[0]/singular[-1]),
            "column_normalized_singular_values": normalized.tolist(),
            "column_normalized_condition_number": float(normalized[0]/normalized[-1]),
            "elapsed_seconds": time.perf_counter()-begin}
        records.append(record)
        print(f"{model}: column-normalized condition {normalized[0]/normalized[-1]:.3f}; "
              f"max two/three-point relative column difference "
              f"{max(record['relative_2_vs_3_point_differences'].values()):.3g}", flush=True)
    return records


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-alternate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=root /
        "notebooks/_output/pricing/research_v1/02_synthetic_calibration/american_joint_identification")
    args = parser.parse_args()
    probes = run_probe(FDConfig(80, 24, 60), "actual exploratory-study grid")
    if args.include_alternate:
        probes += run_probe(FDConfig(100, 32, 100),
            "independent alternate-grid probe; not the executed exploratory grid")
    source = Path(__file__).resolve()
    metadata = {"study": "Independent weighted-Jacobian consistency review",
        "method": "Differentiate price / effective half-spread in physical parameter coordinates on one frozen training mesh. Targets are constants, so omitted. Compare two-point and three-point derivatives; SVD uses the three-point Jacobian. Column-normalized SVD divides each column by its Euclidean norm.",
        "limitations": ["A local check at the generating parameters, not a global identifiability proof.",
            "Raw condition numbers depend on physical units and residual weights.",
            "No held-out quote, calibration selection, bound or acceptance tolerance is changed."],
        "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
        "script": str(source.relative_to(root)),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "engine_sha256": study._engine_fingerprint(),
        "market": asdict(study.joint_market()), "probes": probes}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "jacobian_review.json").write_text(json.dumps(metadata, indent=2)+"\n")
    pd.DataFrame([{"model": row["model"],
        "grid": "/".join(str(row["config"][name]) for name in
                         ("spot_steps", "variance_steps", "time_steps")),
        "role": row["role"],
        "column_normalized_condition_number": row["column_normalized_condition_number"],
        "raw_weighted_condition_number": row["weighted_condition_number"],
        "max_relative_2_vs_3_point_column_difference": max(row["relative_2_vs_3_point_differences"].values())}
        for row in probes]).to_csv(args.output_dir / "jacobian_review.csv", index=False)
    (args.output_dir / "jacobian_review_method.md").write_text(
        "Independent local weighted-Jacobian check\n\n"
        "Reproduce from the MVP directory with `.venv/bin/python examples/american_joint_jacobian_review.py`. "
        "Add `--include-alternate` to evaluate the separate 100/32/100 probe as well.\n\n"
        "The default probe uses the executed exploratory grid (80 spot intervals, 24 variance intervals, "
        "60 time steps), the declared synthetic dividend/rate inputs, the 12 training contracts and "
        "their effective half-spread weights. The mesh is fixed throughout differentiation. All five "
        "Heston or seven Heston-curve coefficients remain free.\n\n"
        "Physical two-point and three-point derivatives use nominal steps machine-epsilon**(1/2 or 1/3) "
        "times max(abs(parameter), 1). The JSON records exact representable steps and boundary schemes. "
        "SVD uses the three-point Jacobian; normalized SVD divides each column by its Euclidean norm. "
        "Raw condition numbers depend on parameter units. Normalized values describe local column alignment.\n\n"
        "These local checks neither prove global non-identification nor select starts. They use no held-out "
        "contracts and change no calibration tolerance. Script and engine hashes are saved in JSON.\n")


if __name__ == "__main__":
    main()
