"""Execute the sealed policy comparison; see notebooks/pricing/README.md."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rvlab.experiments import _pricing_policy_comparison_run as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=study.DEFAULT_FOLDER)
    parser.add_argument("--phase",choices=("pilot","prepare","train","value","refine","collect","verify","all"),default="verify")
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    folder=args.root.resolve()
    phases=("pilot","prepare","train","value","refine","collect") if args.phase=="all" else (args.phase,)
    for phase in phases:
        if phase=="verify":
            tables,manifest,_=study.load_closed(folder)
            print("VERIFIED",manifest["new_main_valuation_paths"],{k:len(v) for k,v in tables.items()})
        elif phase in ("train","value","refine"):
            getattr(study,phase)(folder,workers=args.workers)
        else:
            getattr(study,phase)(folder)


if __name__=="__main__":
    main()
