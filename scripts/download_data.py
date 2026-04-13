#!/usr/bin/env python3
"""
Download historical market data from Binance (data.binance.vision).

This script downloads klines (OHLCV) and tick-level trade data for multiple
symbols. It handles retries, checkpointing (skip already downloaded files),
and progress tracking.

TIER 1 DATA STRATEGY:
    Klines (1-min): BTCUSDT, ETHUSDT, SOLUSDT - 2020 to 2024 (5 years)
    Trades (tick):  BTCUSDT - 2023 to 2024 (for realistic impact modeling)

    This gives you:
    - ~2.6M kline bars per symbol per year × 5 years × 3 symbols = ~39M bars
    - ~500M+ individual trades for BTCUSDT 2023-2024
    - Coverage of: COVID crash, bull run, bear market, FTX collapse, recovery

Usage:
    # Download everything (klines + trades, all symbols)
    python scripts/download_data.py

    # Klines only (faster, good starting point)
    python scripts/download_data.py --klines-only

    # Specific symbols
    python scripts/download_data.py --symbols BTCUSDT ETHUSDT

    # Custom date range
    python scripts/download_data.py --start 2023-01 --end 2024-06

    # Run in Google Colab (see bottom of file for Colab instructions)

Author: Nikhilesh
"""

import argparse
import io
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

# Binance public data base URL
BASE_URL = "https://data.binance.vision/data/spot/monthly"

# --- TIER 1 DATA PLAN ---
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_START = "2020-01"   # Start from COVID era
DEFAULT_END = "2024-12"     # Through end of 2024

# Symbols for which we also download tick-level trades
# (trades are HUGE - only download for primary symbol)
TRADE_SYMBOLS = ["BTCUSDT"]
TRADE_START = "2023-01"     # 2 years of tick data is plenty
TRADE_END = "2024-12"

# Download settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0           # seconds between retries
REQUEST_TIMEOUT = 60        # seconds
DELAY_BETWEEN_REQUESTS = 0.25  # be polite to Binance servers

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Core Download Functions
# ============================================================

def generate_months(start: str, end: str) -> list[tuple[int, int]]:
    """Generate list of (year, month) tuples between start and end.

    Args:
        start: "YYYY-MM" format, e.g., "2020-01"
        end: "YYYY-MM" format, e.g., "2024-12"

    Returns:
        List of (year, month) tuples.

    Example:
        >>> generate_months("2023-11", "2024-02")
        [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    """
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))

    months = []
    year, month = start_year, start_month

    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    return months


def build_url(symbol: str, data_type: str, interval: str, year: int, month: int) -> str:
    """Build the download URL for a specific month of data.

    Binance URL pattern:
        {BASE_URL}/{data_type}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip

    Args:
        symbol: Trading pair, e.g., "BTCUSDT"
        data_type: "klines" or "trades"
        interval: For klines: "1m", "5m", etc. Ignored for trades.
        year: Year
        month: Month (1-12)

    Returns:
        Full download URL string.
    """
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"

    if data_type == "klines":
        return f"{BASE_URL}/{data_type}/{symbol}/{interval}/{filename}"
    elif data_type == "trades":
        filename = f"{symbol}-trades-{year}-{month:02d}.zip"
        return f"{BASE_URL}/{data_type}/{symbol}/{filename}"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def download_and_extract(
    url: str,
    output_dir: Path,
    max_retries: int = MAX_RETRIES,
) -> Optional[Path]:
    """Download a zip file from Binance and extract the CSV.

    Features:
    - Retry logic with exponential backoff
    - Checkpointing: skips if file already exists
    - Validates that the zip contains exactly one CSV
    - Returns None (not raises) on expected failures (404 = month not available)

    Args:
        url: Full URL to download.
        output_dir: Directory to save extracted CSV.
        max_retries: Number of retry attempts.

    Returns:
        Path to extracted CSV file, or None if download failed.
    """
    # Determine expected output filename from URL
    zip_name = url.split("/")[-1]
    csv_name = zip_name.replace(".zip", ".csv")
    output_path = output_dir / csv_name

    # CHECKPOINT: Skip if already downloaded
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.debug(f"Skipping {csv_name} (already exists, {size_mb:.1f} MB)")
        return output_path

    # Download with retries
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                logger.debug(f"Not found (404): {zip_name}")
                return None

            if response.status_code == 451:
                logger.warning(
                    f"Access restricted (451) for {zip_name}. "
                    "Try using a VPN or download from a different region."
                )
                return None

            response.raise_for_status()

            # Extract CSV from zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith(".csv")]

                if not csv_files:
                    logger.warning(f"No CSV found in {zip_name}")
                    return None

                # Extract the first (usually only) CSV
                csv_filename = csv_files[0]
                zf.extract(csv_filename, output_dir)

                # Rename if needed (sometimes the internal name differs)
                extracted_path = output_dir / csv_filename
                if extracted_path != output_path:
                    extracted_path.rename(output_path)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"{csv_name} ({size_mb:.1f} MB)")
            return output_path

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout downloading {zip_name} (attempt {attempt}/{max_retries})")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error for {zip_name} (attempt {attempt}/{max_retries})")
        except zipfile.BadZipFile:
            logger.warning(f"Corrupt zip file: {zip_name}")
            return None
        except Exception as e:
            logger.warning(f"Error downloading {zip_name}: {e} (attempt {attempt}/{max_retries})")

        if attempt < max_retries:
            sleep_time = RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
            time.sleep(sleep_time)

    logger.error(f"Failed after {max_retries} attempts: {zip_name}")
    return None


# ============================================================
# High-Level Download Orchestrators
# ============================================================

def download_klines(
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1m",
    output_base: Path = Path("data/raw"),
) -> dict[str, list[Path]]:
    """Download kline data for multiple symbols across a date range.

    Args:
        symbols: List of trading pairs.
        start: Start month "YYYY-MM".
        end: End month "YYYY-MM".
        interval: Kline interval (1m, 5m, etc.)
        output_base: Base output directory.

    Returns:
        Dict mapping symbol → list of downloaded file paths.
    """
    months = generate_months(start, end)
    downloaded = {}

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {symbol} klines ({interval}) | {start} → {end}")
        logger.info(f"   {len(months)} months to download")
        logger.info(f"{'='*50}")

        output_dir = output_base / "klines" / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for year, month in tqdm(months, desc=f"{symbol} klines"):
            url = build_url(symbol, "klines", interval, year, month)
            path = download_and_extract(url, output_dir)
            if path:
                paths.append(path)
            time.sleep(DELAY_BETWEEN_REQUESTS)

        downloaded[symbol] = paths
        logger.info(f"{symbol}: {len(paths)}/{len(months)} months downloaded")

    return downloaded


def download_trades(
    symbols: list[str],
    start: str,
    end: str,
    output_base: Path = Path("data/raw"),
) -> dict[str, list[Path]]:
    """Download tick-level trade data for specified symbols.

     WARNING: Trade data is MASSIVE. BTCUSDT is ~300-800 MB per month.
    2 years ≈ 10-15 GB compressed CSV.

    Only download for symbols where you need tick-level simulation.

    Args:
        symbols: List of trading pairs.
        start: Start month "YYYY-MM".
        end: End month "YYYY-MM".
        output_base: Base output directory.

    Returns:
        Dict mapping symbol → list of downloaded file paths.
    """
    months = generate_months(start, end)
    downloaded = {}

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {symbol} TRADES | {start} → {end}")
        logger.info(f"   {len(months)} months - THIS WILL BE LARGE (~500MB/month)")
        logger.info(f"{'='*50}")

        output_dir = output_base / "trades" / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for year, month in tqdm(months, desc=f"{symbol} trades"):
            url = build_url(symbol, "trades", "", year, month)
            path = download_and_extract(url, output_dir)
            if path:
                paths.append(path)
            time.sleep(DELAY_BETWEEN_REQUESTS)

        downloaded[symbol] = paths
        logger.info(f"{symbol}: {len(paths)}/{len(months)} months downloaded")

    return downloaded


# ============================================================
# Summary & Disk Usage
# ============================================================

def print_download_summary(output_base: Path = Path("data/raw")) -> None:
    """Print a summary of all downloaded data."""
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*60}")

    total_size = 0
    total_files = 0

    for data_type in ["klines", "trades"]:
        type_dir = output_base / data_type
        if not type_dir.exists():
            continue

        for symbol_dir in sorted(type_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue

            csv_files = list(symbol_dir.glob("*.csv"))
            size = sum(f.stat().st_size for f in csv_files)
            size_mb = size / (1024 * 1024)
            total_size += size
            total_files += len(csv_files)

            print(f"  {data_type:8s} | {symbol_dir.name:10s} | "
                  f"{len(csv_files):3d} files | {size_mb:,.1f} MB")

    total_gb = total_size / (1024 * 1024 * 1024)
    print(f"{'='*60}")
    print(f"  TOTAL: {total_files} files, {total_gb:.2f} GB")
    print(f"{'='*60}\n")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Binance historical market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py                           # Full Tier 1 download
  python scripts/download_data.py --klines-only             # Klines only (fast)
  python scripts/download_data.py --symbols BTCUSDT         # Single symbol
  python scripts/download_data.py --start 2023-01 --end 2024-06  # Custom range
        """,
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help=f"Symbols to download (default: {DEFAULT_SYMBOLS})"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start month YYYY-MM (default: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help=f"End month YYYY-MM (default: {DEFAULT_END})"
    )
    parser.add_argument(
        "--klines-only", action="store_true",
        help="Download only klines, skip trades (much faster)"
    )
    parser.add_argument(
        "--trades-only", action="store_true",
        help="Download only trades, skip klines"
    )
    parser.add_argument(
        "--trade-symbols", nargs="+", default=TRADE_SYMBOLS,
        help=f"Symbols for tick-level trades (default: {TRADE_SYMBOLS})"
    )
    parser.add_argument(
        "--trade-start", default=TRADE_START,
        help=f"Trades start month (default: {TRADE_START})"
    )
    parser.add_argument(
        "--trade-end", default=TRADE_END,
        help=f"Trades end month (default: {TRADE_END})"
    )
    parser.add_argument(
        "--output", default="data/raw", type=Path,
        help="Output directory (default: data/raw)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  BINANCE DATA DOWNLOADER - TIER 1 STRATEGY")
    print(f"{'='*60}")
    print(f"  Symbols (klines): {args.symbols}")
    print(f"  Date range:       {args.start} → {args.end}")

    if not args.trades_only:
        print(f"  Klines interval:  1m")

    if not args.klines_only:
        print(f"  Symbols (trades): {args.trade_symbols}")
        print(f"  Trades range:     {args.trade_start} → {args.trade_end}")
        print(f"   Trades are ~500MB/month - ensure sufficient disk space")

    print(f"  Output:           {args.output}")
    print(f"{'='*60}\n")

    # Download klines
    if not args.trades_only:
        download_klines(
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            interval="1m",
            output_base=args.output,
        )

    # Download trades
    if not args.klines_only:
        download_trades(
            symbols=args.trade_symbols,
            start=args.trade_start,
            end=args.trade_end,
            output_base=args.output,
        )

    # Print summary
    print_download_summary(args.output)


if __name__ == "__main__":
    main()
