"""
Data loader for Binance historical market data.

Downloads klines (OHLCV) and trades from data.binance.vision,
the official Binance public data repository.

PRODUCTION PATTERNS demonstrated:
  1. Type hints everywhere - your IDE catches bugs before you run code
  2. Docstrings on every public function - your team thanks you later
  3. Logging instead of print() - configurable, timestamped, leveled
  4. Defensive coding - validate inputs, handle errors gracefully
  5. Separation of concerns - download ≠ parse ≠ validate ≠ store
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schemas import KlineSchema, TradeSchema

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Binance data.binance.vision CSV column mappings
# --------------------------------------------------------------------------
# Binance klines CSVs have NO header. Columns are positional.
BINANCE_KLINE_COLUMNS = [
    "open_time",        # 0: Kline open time (ms timestamp)
    "open",             # 1: Open price
    "high",             # 2: High price
    "low",              # 3: Low price
    "close",            # 4: Close price
    "volume",           # 5: Volume (base asset)
    "close_time",       # 6: Kline close time (ms timestamp)
    "quote_volume",     # 7: Quote asset volume
    "num_trades",       # 8: Number of trades
    "taker_buy_vol",    # 9: Taker buy base asset volume
    "taker_buy_quote",  # 10: Taker buy quote asset volume
    "ignore",           # 11: Ignore field
]

BINANCE_TRADE_COLUMNS = [
    "trade_id",         # 0: Trade ID
    "price",            # 1: Price
    "quantity",         # 2: Quantity
    "quote_quantity",   # 3: Quote quantity
    "timestamp",        # 4: Trade time (ms timestamp)
    "is_buyer_maker",   # 5: Is buyer maker?
    "is_best_match",    # 6: Is best match?
]


def load_klines_from_csv(
    filepath: str | Path,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Load and parse a Binance klines CSV file into a clean DataFrame.

    Binance kline CSVs from data.binance.vision have no header row.
    Timestamps are Unix milliseconds. This function handles all parsing.

    Args:
        filepath: Path to the CSV file (can be .csv or .csv.gz).
        symbol: Trading pair symbol to tag the data with.

    Returns:
        DataFrame with columns matching KlineSchema.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If the CSV has unexpected format.

    Example:
        >>> df = load_klines_from_csv("data/raw/BTCUSDT-1m-2023-01.csv")
        >>> df.columns.tolist()
        ['timestamp', 'open', 'high', 'low', 'close', 'volume', ...]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Kline file not found: {filepath}")

    logger.info(f"Loading klines from {filepath.name}")

    # Read CSV with no header - assign column names ourselves
    df = pd.read_csv(
        filepath,
        header=None,
        names=BINANCE_KLINE_COLUMNS,
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "volume": np.float64,
            "quote_volume": np.float64,
            "num_trades": np.int64,
        },
    )

    # Validate shape
    if df.shape[1] != len(BINANCE_KLINE_COLUMNS):
        raise ValueError(
            f"Expected {len(BINANCE_KLINE_COLUMNS)} columns, got {df.shape[1]}. "
            "Is this a Binance klines file?"
        )

    # Convert millisecond timestamps to proper datetime
    df[KlineSchema.TIMESTAMP] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # Select and rename to our standard schema
    result = df[[
        KlineSchema.TIMESTAMP,
        "open", "high", "low", "close", "volume", "quote_volume", "num_trades",
    ]].copy()

    # Rename to match our schema exactly
    result.columns = [
        KlineSchema.TIMESTAMP,
        KlineSchema.OPEN,
        KlineSchema.HIGH,
        KlineSchema.LOW,
        KlineSchema.CLOSE,
        KlineSchema.VOLUME,
        KlineSchema.QUOTE_VOLUME,
        KlineSchema.TRADES,
    ]

    # Add symbol column
    result[KlineSchema.SYMBOL] = symbol

    # Sort by time (defensive - files should already be sorted)
    result = result.sort_values(KlineSchema.TIMESTAMP).reset_index(drop=True)

    logger.info(
        f"Loaded {len(result):,} klines for {symbol} "
        f"from {result[KlineSchema.TIMESTAMP].iloc[0]} "
        f"to {result[KlineSchema.TIMESTAMP].iloc[-1]}"
    )

    return result


def load_trades_from_csv(
    filepath: str | Path,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Load and parse a Binance trades CSV file.

    Args:
        filepath: Path to the CSV file.
        symbol: Trading pair symbol.

    Returns:
        DataFrame with columns matching TradeSchema.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trades file not found: {filepath}")

    logger.info(f"Loading trades from {filepath.name}")

    df = pd.read_csv(
        filepath,
        header=None,
        names=BINANCE_TRADE_COLUMNS,
        dtype={
            "trade_id": np.int64,
            "price": np.float64,
            "quantity": np.float64,
            "quote_quantity": np.float64,
            "is_buyer_maker": bool,
        },
    )

    # Convert timestamp
    df[TradeSchema.TIMESTAMP] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Select columns matching our schema
    result = df[[
        TradeSchema.TRADE_ID,
        TradeSchema.TIMESTAMP,
        TradeSchema.PRICE,
        TradeSchema.QUANTITY,
        "quote_quantity",
        TradeSchema.IS_BUYER_MAKER,
    ]].copy()

    result.columns = [
        TradeSchema.TRADE_ID,
        TradeSchema.TIMESTAMP,
        TradeSchema.PRICE,
        TradeSchema.QUANTITY,
        TradeSchema.QUOTE_QTY,
        TradeSchema.IS_BUYER_MAKER,
    ]

    result[TradeSchema.SYMBOL] = symbol
    result = result.sort_values(TradeSchema.TIMESTAMP).reset_index(drop=True)

    logger.info(f"Loaded {len(result):,} trades for {symbol}")

    return result


def load_klines_directory(
    directory: str | Path,
    symbol: str = "BTCUSDT",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load and concatenate all kline CSV files from a directory.

    This handles the common case of having monthly files like:
        BTCUSDT-1m-2023-01.csv
        BTCUSDT-1m-2023-02.csv
        ...

    Args:
        directory: Path to directory containing CSV files.
        symbol: Trading pair symbol.
        start_date: Optional filter start (ISO format, e.g., "2023-01-01").
        end_date: Optional filter end (ISO format, e.g., "2023-12-31").

    Returns:
        Combined DataFrame, sorted by timestamp, duplicates removed.
    """
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv")) + sorted(directory.glob("*.csv.gz"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    logger.info(f"Found {len(csv_files)} files in {directory}")

    # Load and concatenate all files
    frames = []
    for f in csv_files:
        try:
            df = load_klines_from_csv(f, symbol=symbol)
            frames.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")

    if not frames:
        raise ValueError("No files could be loaded successfully")

    combined = pd.concat(frames, ignore_index=True)

    # Remove duplicates (overlap between monthly files)
    combined = combined.drop_duplicates(
        subset=[KlineSchema.TIMESTAMP], keep="first"
    )

    # Apply date filters
    if start_date:
        start = pd.Timestamp(start_date, tz="UTC")
        combined = combined[combined[KlineSchema.TIMESTAMP] >= start]

    if end_date:
        end = pd.Timestamp(end_date, tz="UTC")
        combined = combined[combined[KlineSchema.TIMESTAMP] <= end]

    combined = combined.sort_values(KlineSchema.TIMESTAMP).reset_index(drop=True)

    logger.info(
        f"Combined dataset: {len(combined):,} bars, "
        f"{combined[KlineSchema.TIMESTAMP].iloc[0]} → "
        f"{combined[KlineSchema.TIMESTAMP].iloc[-1]}"
    )

    return combined


def save_processed(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save a processed DataFrame to Parquet (preferred) or CSV (fallback).

    PRODUCTION PATTERN: Always prefer Parquet over CSV.
    Why?
      - 10x smaller file size (columnar compression)
      - 10x faster read/write
      - Preserves dtypes (CSV loses datetime types, int vs float, etc.)
      - Supports partitioning for large datasets

    Falls back to CSV if pyarrow is not installed.

    Args:
        df: DataFrame to save.
        filepath: Output path (.parquet or .csv).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow  # noqa: F401
        parquet_path = filepath.with_suffix(".parquet")
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df):,} rows to {parquet_path} ({size_mb:.1f} MB)")
    except ImportError:
        csv_path = filepath.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Saved {len(df):,} rows to {csv_path} ({size_mb:.1f} MB) "
            "[CSV fallback - install pyarrow for Parquet]"
        )


def load_processed(filepath: str | Path) -> pd.DataFrame:
    """Load a processed Parquet or CSV file.

    Args:
        filepath: Path to parquet or csv file.

    Returns:
        DataFrame.
    """
    filepath = Path(filepath)

    # Try parquet first, fall back to csv
    for ext in [".parquet", ".csv"]:
        candidate = filepath.with_suffix(ext)
        if candidate.exists():
            if ext == ".parquet":
                df = pd.read_parquet(candidate, engine="pyarrow")
            else:
                df = pd.read_csv(candidate, parse_dates=[KlineSchema.TIMESTAMP])
            logger.info(f"Loaded {len(df):,} rows from {candidate}")
            return df

    raise FileNotFoundError(f"No processed file found at {filepath} (.parquet or .csv)")
