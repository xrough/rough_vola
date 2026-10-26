"""Prepare, audit or acquire SPY inputs: python -m rvlab.market_data --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _progress(item):
    print(json.dumps(item, default=str), flush=True)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-options", "prepare", "audit"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--output", type=Path, default=root / "data/processed/spy")
        if name != "audit":
            cmd.add_argument("--archive", type=Path,
                             default=root / "demo/data/OPRA-20260208-3T68RYYKF9.zip")
            cmd.add_argument("--max-days", type=int)
            cmd.add_argument("--chunk-size", type=int, default=250_000)
            cmd.add_argument("--workers", type=int, default=2)
        if name != "prepare-options":
            cmd.add_argument("--raw-references", type=Path, default=root / "data/raw/spy_reference")
            cmd.add_argument("--stock", type=Path)
            cmd.add_argument("--rates", type=Path)
            cmd.add_argument("--dividends", type=Path)
            cmd.add_argument("--stock-timestamp-convention", choices=("observation", "minute_bar_start", "minute_bar_end"))
            cmd.add_argument("--stock-source")
            cmd.add_argument("--stock-price-basis", choices=("raw",))
            cmd.add_argument("--allow-dividend-proxy", action="store_true",
                help="Retained for old invocations; American readiness rejects trailing-yield proxies")
            cmd.add_argument("--dividend-coverage-through",
                help="Assert that the supplied dated dividend forecast schedule covers this date")
            cmd.add_argument("--require-ready", action="store_true")
            cmd.add_argument("--report", type=Path,
                             default=root / "lab/reports/spy-data-readiness.md")
    public = commands.add_parser("fetch-public")
    public.add_argument("--start", default="2024-01-01")
    public.add_argument("--end", default="2026-04-30")
    public.add_argument("--output", type=Path, default=root / "data/raw/spy_reference")
    alpaca = commands.add_parser("fetch-alpaca")
    alpaca.add_argument("--start", default="2025-08-07")
    alpaca.add_argument("--end", default="2026-02-06")
    alpaca.add_argument("--output", type=Path, default=root / "data/raw/spy_reference/stock_prices.parquet")
    args = parser.parse_args(argv)
    if args.command == "fetch-public":
        from .references import fetch_public_dividends, fetch_public_rates
        args.output.mkdir(parents=True, exist_ok=True)
        rates = fetch_public_rates(args.start, args.end, args.output / "rates.parquet")
        dividends = fetch_public_dividends(args.start, args.end, args.output / "dividends.parquet")
        _progress({"rates_rows": len(rates), "dividend_rows": len(dividends)})
        return 0
    if args.command == "fetch-alpaca":
        from .references import acquire_alpaca_stock
        stock = acquire_alpaca_stock(args.start, args.end, args.output)
        _progress({"stock_rows": len(stock), "path": args.output})
        return 0
    if args.command == "prepare-options":
        from .ingest import prepare_archive
        manifest = prepare_archive(args.archive, args.output, max_days=args.max_days,
                                    chunk_size=args.chunk_size, workers=args.workers,
                                    on_progress=_progress)
        _progress({name: manifest[name] for name in (
            "completed_day_count", "is_full_archive_complete", "raw_rows", "output_bytes")})
        return 0
    from .pipeline import audit_prepared_dataset, prepare_spy_dataset
    options = dict(raw_references=args.raw_references, stock_path=args.stock,
        rates_path=args.rates, dividends_path=args.dividends,
        stock_timestamp_convention=args.stock_timestamp_convention,
        stock_source=args.stock_source, stock_price_basis=args.stock_price_basis,
        allow_dividend_proxy=args.allow_dividend_proxy,
        dividend_coverage_through=args.dividend_coverage_through, report_path=args.report,
        on_progress=_progress)
    if args.command == "prepare":
        report = prepare_spy_dataset(args.archive, args.output,
            max_days=args.max_days, chunk_size=args.chunk_size, workers=args.workers, **options)
    else:
        report = audit_prepared_dataset(args.output, **options)
    _progress({name: report[name] for name in (
        "options_complete", "calibration_ready", "hedging_ready", "missing_inputs")})
    return 2 if args.require_ready and not report["calibration_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
