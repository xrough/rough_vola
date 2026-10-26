"""Explicit cold reproduction of the preserved American calibration design.

This is intentionally separate from the quick first-stage report replay. It
may take hours, uses the original budgets/tolerances, and writes only a new
first_stage_v1 subdirectory. It never modifies the preserved research_v1 tree.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rvlab.pricing import FDConfig,PricingConfig
from rvlab.experiments.pricing import setup_first_stage,run_calibration_research


def run_cold(output_dir):
    destination=Path(output_dir).resolve()
    allowed=(ROOT/'notebooks/_output/pricing/first_stage_v1').resolve()
    if not destination.is_relative_to(allowed) or destination==allowed:
        raise ValueError('Cold reproduction must use a new first_stage_v1 subdirectory')
    if destination.name in {'01_pricing_and_exercise','02_synthetic_calibration','03_spy_calibration','presentation'}:
        raise ValueError('Cold reproduction must not overwrite an active report directory')
    if destination.exists():
        raise FileExistsError('Preserve the existing cold trial; choose a new output directory')
    context=setup_first_stage('02_synthetic_calibration.ipynb',output_dir=destination)
    fit=FDConfig(192,48,180,spot_concentration=.05)
    generation=FDConfig(640,128,600,spot_concentration=.05)
    return run_calibration_research(context,seed=42,
        coarse_config=PricingConfig(96,60.,256),fine_config=PricingConfig(128,75.,512),max_nfev=80,
        run_american_joint=True,joint_fit_config=fit,joint_generation_config=generation,
        joint_continuation_config=fit,joint_max_nfev=80,joint_continuation_max_nfev=160,
        joint_run_continuation=True,joint_reuse_checkpoints=False,
        run_curve_diagnosis=True,curve_diagnosis_max_nfev=40,curve_diagnosis_reuse=False,
        run_maturity_extension=True,maturity_fit_config=FDConfig(192,64,180,spot_concentration=.05),
        maturity_generation_config=generation,maturity_max_nfev=80,maturity_reuse=False,
        run_controlled_american=False,run_extended=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,
        default=ROOT/'notebooks/_output/pricing/first_stage_v1/cold_calibration_v1')
    args=parser.parse_args(argv)
    run_cold(args.output_dir)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
