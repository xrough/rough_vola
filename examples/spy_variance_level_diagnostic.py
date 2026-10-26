"""Run the consolidated American SPY study, including Heston variance curves.

The workflow emits an explicit blocked-input audit until independent stock and
historically available cash-dividend expectations exist. No old fit is reused.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/spy")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-nfev", type=int, default=40)
    args = parser.parse_args(argv)
    from rvlab.experiments.pricing import setup_demo, run_spy_research
    context = setup_demo("03_spy_calibration", output_dir=args.output_dir)
    results = run_spy_research(context, data_root=args.data_root, max_nfev=args.max_nfev)
    print(results["tables"]["input_audit"].to_string(index=False))
    if results["manifest"]["status"] == "blocked_inputs":
        print("No calibration performed: observed-stock and dividend-input blockers remain.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
