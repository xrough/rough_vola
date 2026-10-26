#!/usr/bin/env python3
"""Run the SPY data CLI from a source checkout without an editable install."""

from pathlib import Path
import sys


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rvlab.market_data.__main__ import main as market_data_main

    return market_data_main()


if __name__ == "__main__":
    raise SystemExit(main())
