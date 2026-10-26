#!/usr/bin/env python3
"""Prompt locally for credentials and preserve a separate raw SIP repeat."""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvlab.market_data.spy_call_pilot import acquire_repeat_stock, REPEAT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / REPEAT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Destination already exists; existing acquisition evidence is never overwritten")
    if not sys.stdin.isatty():
        parser.error("Run this script in your local Terminal to enter credentials at hidden prompts")
    key = secret = None
    try:
        key = getpass.getpass("Alpaca API key (hidden): ")
        secret = getpass.getpass("Alpaca secret key (hidden): ")
        out = acquire_repeat_stock(args.output, api_key=key, api_secret=secret)
    except Exception as exc:
        # Do not print arbitrary HTTP exception payloads or traceback locals.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
            "message": "Acquisition failed; inspect the sanitized manifest if created."}), file=sys.stderr)
        return 1
    finally:
        key = secret = None
    print(json.dumps({"status": "complete", "stock_rows": len(out),
        "path": str(args.output / "stock_prices.parquet"), "manifest": str(args.output / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
