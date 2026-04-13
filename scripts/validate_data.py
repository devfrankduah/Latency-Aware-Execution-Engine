#!/usr/bin/env python3
"""
Validate and process downloaded Binance data.

Run this AFTER download_data.py to:
  1. Load all raw CSVs per symbol
  2. Run quality validation checks
  3. Save cleaned, combined data as processed files
  4. Print summary statistics useful for the project writeup

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --symbols BTCUSDT ETHUSDT
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data.loader import load_klines_directory, save_processed
from src.data.validator import validate_klines
from src.data.schemas import KlineSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_symbol(
    symbol: str,
    raw_dir: Path,
    processed_dir: Path,
) -> bool:
    """Load, validate, and save processed data for one symbol.

    Returns True if validation passed.
    """
    kline_dir = raw_dir / "klines" / symbol

    if not kline_dir.exists():
        logger.warning(f"No data directory found for {symbol} at {kline_dir}")
        return False

    # --- Load ---
    logger.info(f"\n{'='*60}")
    logger.info(f"  Processing {symbol}")
    logger.info(f"{'='*60}")

    df = load_klines_directory(kline_dir, symbol=symbol)

    # --- Validate ---
    report = validate_klines(df, expected_freq_minutes=1)
    print(report)

    # --- Summary stats (useful for project writeup) ---
    print(f"  SUMMARY STATISTICS - {symbol}")
    print(f"  {'─'*50}")
    print(f"  Total bars:        {len(df):>15,}")
    print(f"  Total days:        {df[KlineSchema.TIMESTAMP].dt.date.nunique():>15,}")
    print(f"  Avg price:         ${df[KlineSchema.CLOSE].mean():>14,.2f}")
    print(f"  Min price:         ${df[KlineSchema.CLOSE].min():>14,.2f}")
    print(f"  Max price:         ${df[KlineSchema.CLOSE].max():>14,.2f}")
    print(f"  Avg volume/bar:    {df[KlineSchema.VOLUME].mean():>15,.4f}")
    print(f"  Total volume:      {df[KlineSchema.VOLUME].sum():>15,.2f}")

    if KlineSchema.TRADES in df.columns:
        print(f"  Avg trades/bar:    {df[KlineSchema.TRADES].mean():>15,.1f}")

    # Spread proxy (High - Low as % of Close)
    spread_proxy = (df[KlineSchema.HIGH] - df[KlineSchema.LOW]) / df[KlineSchema.CLOSE]
    print(f"  Avg spread proxy:  {spread_proxy.mean():>15.4%}")
    print(f"  Med spread proxy:  {spread_proxy.median():>15.4%}")

    # Volatility (annualized from 1-min returns)
    log_returns = np.log(df[KlineSchema.CLOSE] / df[KlineSchema.CLOSE].shift(1)).dropna()
    annual_vol = log_returns.std() * np.sqrt(365.25 * 24 * 60)  # crypto = 24/7
    print(f"  Annualized vol:    {annual_vol:>15.2%}")
    print()

    # --- Save processed ---
    output_path = processed_dir / f"{symbol}_klines_1m"
    save_processed(df, output_path)

    return report.is_valid


def main():
    parser = argparse.ArgumentParser(description="Validate and process downloaded data")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to process",
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for symbol in args.symbols:
        passed = process_symbol(symbol, args.raw_dir, args.processed_dir)
        results[symbol] = passed

    # Final summary
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    for symbol, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {symbol:12s} {status}")
    print(f"{'='*60}\n")

    # Check processed files
    print("  Processed files:")
    for f in sorted(args.processed_dir.glob("*")):
        if f.name.startswith("."):
            continue
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:40s} {size_mb:>8.1f} MB")
    print()


if __name__ == "__main__":
    main()
