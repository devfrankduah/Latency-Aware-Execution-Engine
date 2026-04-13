"""
Tests for data loading and validation.

PRODUCTION PATTERN: Test with synthetic data, not real data.
Why?
  1. Tests must run anywhere (CI/CD) without downloading 100MB files
  2. You control the test data - so you know the expected answer
  3. Tests run in milliseconds, not minutes
  4. You can create edge cases that rarely appear in real data

Testing philosophy:
  - Test the happy path (normal data works)
  - Test edge cases (empty data, single row, duplicates)
  - Test error cases (missing columns, wrong types)
  - Test contracts (output schema matches what's promised)
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import csv
from pathlib import Path

from src.data.schemas import KlineSchema, TradeSchema
from src.data.validator import validate_klines
from src.data.loader import (
    load_klines_from_csv,
    load_trades_from_csv,
    load_processed,
    save_processed,
    BINANCE_KLINE_COLUMNS,
    BINANCE_TRADE_COLUMNS,
)


def make_synthetic_klines(
    n_bars: int = 1000,
    start_price: float = 40000.0,
    volatility: float = 0.001,
    start_time: str = "2023-01-01",
    freq_minutes: int = 1,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Generate realistic synthetic kline data for testing.

    Uses a geometric Brownian motion model to produce realistic OHLCV bars.
    This is a utility used across many tests.

    Args:
        n_bars: Number of bars to generate.
        start_price: Starting price.
        volatility: Per-bar volatility (std of log returns).
        start_time: Start timestamp.
        freq_minutes: Bar size in minutes.
        symbol: Symbol name.

    Returns:
        DataFrame matching KlineSchema.
    """
    rng = np.random.default_rng(seed=42)  # Reproducible!

    # Generate price path via geometric Brownian motion
    log_returns = rng.normal(0, volatility, n_bars)
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    # Generate OHLC from close (realistic: H > C > L, with noise)
    noise = rng.uniform(0.0001, 0.001, n_bars)
    highs = close_prices * (1 + noise)
    lows = close_prices * (1 - noise)
    opens = np.roll(close_prices, 1)
    opens[0] = start_price

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, close_prices))
    lows = np.minimum(lows, np.minimum(opens, close_prices))

    # Volume with realistic intraday pattern (U-shape)
    base_volume = rng.exponential(10.0, n_bars)
    intraday_pattern = 1 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, min(n_bars, 1440)))
    if n_bars > 1440:
        intraday_pattern = np.tile(intraday_pattern, n_bars // 1440 + 1)[:n_bars]
    volume = base_volume * intraday_pattern[:n_bars]

    timestamps = pd.date_range(
        start=start_time,
        periods=n_bars,
        freq=f"{freq_minutes}min",
        tz="UTC",
    )

    df = pd.DataFrame({
        KlineSchema.TIMESTAMP: timestamps,
        KlineSchema.OPEN: opens,
        KlineSchema.HIGH: highs,
        KlineSchema.LOW: lows,
        KlineSchema.CLOSE: close_prices,
        KlineSchema.VOLUME: volume,
        KlineSchema.QUOTE_VOLUME: volume * close_prices,
        KlineSchema.TRADES: rng.integers(50, 500, n_bars),
        KlineSchema.SYMBOL: symbol,
    })

    return df


# =====================================================================
# TESTS
# =====================================================================

class TestSyntheticDataGeneration:
    """Test that our test data generator works correctly."""

    def test_basic_generation(self):
        df = make_synthetic_klines(n_bars=100)
        assert len(df) == 100
        assert set(KlineSchema.required_columns()).issubset(df.columns)

    def test_reproducibility(self):
        """Same seed → same data. Critical for debugging."""
        df1 = make_synthetic_klines(n_bars=50)
        df2 = make_synthetic_klines(n_bars=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_ohlc_consistency(self):
        """High >= max(Open, Close) and Low <= min(Open, Close)."""
        df = make_synthetic_klines(n_bars=5000)
        assert (df[KlineSchema.HIGH] >= df[KlineSchema.OPEN]).all()
        assert (df[KlineSchema.HIGH] >= df[KlineSchema.CLOSE]).all()
        assert (df[KlineSchema.LOW] <= df[KlineSchema.OPEN]).all()
        assert (df[KlineSchema.LOW] <= df[KlineSchema.CLOSE]).all()

    def test_positive_prices(self):
        df = make_synthetic_klines(n_bars=1000)
        for col in [KlineSchema.OPEN, KlineSchema.HIGH, KlineSchema.LOW, KlineSchema.CLOSE]:
            assert (df[col] > 0).all()

    def test_positive_volume(self):
        df = make_synthetic_klines(n_bars=1000)
        assert (df[KlineSchema.VOLUME] > 0).all()


class TestValidation:
    """Test the data validation pipeline."""

    def test_clean_data_passes(self):
        """Valid data should pass validation."""
        df = make_synthetic_klines(n_bars=1000)
        report = validate_klines(df)
        print(report)  # pytest -s to see this
        assert report.is_valid is True
        assert report.total_rows == 1000
        assert report.negative_prices == 0
        assert report.ohlc_violations == 0

    def test_missing_columns_fails(self):
        """Data missing required columns should fail."""
        df = make_synthetic_klines(n_bars=100)
        df = df.drop(columns=[KlineSchema.CLOSE])
        report = validate_klines(df)
        assert report.is_valid is False
        assert any("Missing required columns" in issue for issue in report.issues)

    def test_duplicate_timestamps_detected(self):
        """Duplicate timestamps should be flagged."""
        df = make_synthetic_klines(n_bars=100)
        # Duplicate the last row
        df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
        report = validate_klines(df)
        assert report.duplicate_timestamps == 1
        assert report.is_valid is False

    def test_negative_prices_detected(self):
        """Negative prices should fail validation."""
        df = make_synthetic_klines(n_bars=100)
        df.loc[5, KlineSchema.CLOSE] = -100.0
        report = validate_klines(df)
        assert report.negative_prices > 0
        assert report.is_valid is False

    def test_ohlc_violation_detected(self):
        """High < Low should be flagged."""
        df = make_synthetic_klines(n_bars=100)
        df.loc[10, KlineSchema.HIGH] = df.loc[10, KlineSchema.LOW] - 1.0
        report = validate_klines(df)
        assert report.ohlc_violations > 0
        assert report.is_valid is False

    def test_missing_bars_calculated(self):
        """Gaps in timestamps should be detected."""
        df = make_synthetic_klines(n_bars=100)
        # Remove bars 50-60 (creating a 10-minute gap)
        df = df.drop(index=range(50, 60)).reset_index(drop=True)
        report = validate_klines(df, expected_freq_minutes=1)
        assert report.missing_bars >= 9  # At least 9 missing
        assert report.missing_pct > 0

    def test_empty_dataframe(self):
        """Edge case: empty DataFrame shouldn't crash."""
        df = pd.DataFrame(columns=KlineSchema.all_columns())
        # This tests defensive coding - empty data shouldn't throw
        # It should fail validation gracefully
        report = validate_klines(df)
        assert report.total_rows == 0


# =====================================================================
# KlineSchema and TradeSchema
# =====================================================================

class TestKlineSchema:
    def test_required_columns_is_list(self):
        cols = KlineSchema.required_columns()
        assert isinstance(cols, list) and len(cols) > 0

    def test_all_columns_is_list(self):
        cols = KlineSchema.all_columns()
        assert isinstance(cols, list) and len(cols) > 0

    def test_required_columns_subset_of_all(self):
        assert set(KlineSchema.required_columns()).issubset(set(KlineSchema.all_columns()))

    def test_no_duplicate_required_columns(self):
        cols = KlineSchema.required_columns()
        assert len(cols) == len(set(cols))

    def test_no_duplicate_all_columns(self):
        cols = KlineSchema.all_columns()
        assert len(cols) == len(set(cols))

    def test_timestamp_in_required(self):
        assert KlineSchema.TIMESTAMP in KlineSchema.required_columns()

    def test_close_in_required(self):
        assert KlineSchema.CLOSE in KlineSchema.required_columns()

    def test_constants_are_strings(self):
        for attr in [KlineSchema.TIMESTAMP, KlineSchema.OPEN, KlineSchema.HIGH,
                     KlineSchema.LOW, KlineSchema.CLOSE, KlineSchema.VOLUME,
                     KlineSchema.QUOTE_VOLUME, KlineSchema.TRADES, KlineSchema.SYMBOL]:
            assert isinstance(attr, str)


class TestTradeSchema:
    def test_required_columns_is_list(self):
        cols = TradeSchema.required_columns()
        assert isinstance(cols, list) and len(cols) > 0

    def test_no_duplicate_required_columns(self):
        cols = TradeSchema.required_columns()
        assert len(cols) == len(set(cols))

    def test_trade_id_in_required(self):
        assert TradeSchema.TRADE_ID in TradeSchema.required_columns()

    def test_price_in_required(self):
        assert TradeSchema.PRICE in TradeSchema.required_columns()

    def test_constants_are_strings(self):
        for attr in [TradeSchema.TRADE_ID, TradeSchema.TIMESTAMP, TradeSchema.PRICE,
                     TradeSchema.QUANTITY, TradeSchema.IS_BUYER_MAKER, TradeSchema.SYMBOL]:
            assert isinstance(attr, str)


# =====================================================================
# CSV Loader
# =====================================================================

import csv
import tempfile


def _make_binance_klines_csv(n: int = 100, tmp_dir: str = None) -> Path:
    """Write a minimal Binance-style klines CSV (no header, positional cols)."""
    rng = np.random.default_rng(0)
    base_ms = 1_672_531_200_000  # 2023-01-01 00:00:00 UTC in ms
    rows = []
    price = 16_000.0
    for i in range(n):
        open_ms = base_ms + i * 60_000
        close_ms = open_ms + 59_999
        close = price * (1 + rng.normal(0, 0.001))
        high = close * (1 + abs(rng.normal(0, 0.0005)))
        low = close * (1 - abs(rng.normal(0, 0.0005)))
        vol = abs(rng.normal(10, 2))
        rows.append([
            open_ms, round(price, 2), round(high, 2), round(low, 2),
            round(close, 2), round(vol, 4), close_ms,
            round(vol * close, 2), rng.integers(100, 1000),
            round(vol * 0.5, 4), round(vol * 0.5 * close, 2), 0,
        ])
        price = close
    path = Path(tmp_dir) / "BTCUSDT-1m-2023-01.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


def _make_binance_trades_csv(n: int = 100, tmp_dir: str = None) -> Path:
    """Write a minimal Binance-style trades CSV."""
    rng = np.random.default_rng(1)
    base_ms = 1_672_531_200_000
    rows = []
    for i in range(n):
        rows.append([
            i,
            round(16_000 + rng.normal(0, 10), 2),
            round(abs(rng.normal(0.01, 0.005)), 6),
            round(abs(rng.normal(160, 10)), 2),
            base_ms + i * 100,
            bool(rng.integers(0, 2)),
            True,
        ])
    path = Path(tmp_dir) / "BTCUSDT-trades-2023-01.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


class TestLoadKlinesFromCsv:
    def test_loads_correct_row_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_klines_csv(n=50, tmp_dir=td)
            df = load_klines_from_csv(path)
        assert len(df) == 50

    def test_output_has_required_columns(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_klines_csv(n=30, tmp_dir=td)
            df = load_klines_from_csv(path)
        for col in KlineSchema.required_columns():
            assert col in df.columns

    def test_timestamps_are_utc_datetimes(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_klines_csv(n=30, tmp_dir=td)
            df = load_klines_from_csv(path)
        assert pd.api.types.is_datetime64_any_dtype(df[KlineSchema.TIMESTAMP])
        assert str(df[KlineSchema.TIMESTAMP].dt.tz) == "UTC"

    def test_symbol_column_set(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_klines_csv(n=10, tmp_dir=td)
            df = load_klines_from_csv(path, symbol="BTCUSDT")
        assert (df[KlineSchema.SYMBOL] == "BTCUSDT").all()

    def test_sorted_by_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_klines_csv(n=50, tmp_dir=td)
            df = load_klines_from_csv(path)
        assert (df[KlineSchema.TIMESTAMP].diff().dropna() >= pd.Timedelta(0)).all()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_klines_from_csv("/non/existent/file.csv")

    def test_prices_are_positive(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_klines_csv(n=50, tmp_dir=td)
            df = load_klines_from_csv(path)
        for col in [KlineSchema.OPEN, KlineSchema.HIGH, KlineSchema.LOW, KlineSchema.CLOSE]:
            assert (df[col] > 0).all()


class TestLoadTradesFromCsv:
    def test_loads_correct_row_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_trades_csv(n=50, tmp_dir=td)
            df = load_trades_from_csv(path)
        assert len(df) == 50

    def test_output_has_required_columns(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_trades_csv(n=30, tmp_dir=td)
            df = load_trades_from_csv(path)
        for col in TradeSchema.required_columns():
            assert col in df.columns

    def test_timestamps_are_utc(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_trades_csv(n=30, tmp_dir=td)
            df = load_trades_from_csv(path)
        assert pd.api.types.is_datetime64_any_dtype(df[TradeSchema.TIMESTAMP])
        assert str(df[TradeSchema.TIMESTAMP].dt.tz) == "UTC"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_trades_from_csv("/non/existent/trades.csv")

    def test_symbol_column_set(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_binance_trades_csv(n=10, tmp_dir=td)
            df = load_trades_from_csv(path, symbol="ETHUSDT")
        assert (df[TradeSchema.SYMBOL] == "ETHUSDT").all()


class TestSaveAndLoadProcessed:
    def test_parquet_round_trip(self):
        df = make_synthetic_klines(n_bars=100)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data"
            save_processed(df, path)
            loaded = load_processed(path)
        assert len(loaded) == len(df)
        assert set(df.columns) == set(loaded.columns)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_processed("/non/existent/data")

    def test_save_creates_parent_dirs(self):
        df = make_synthetic_klines(n_bars=10)
        with tempfile.TemporaryDirectory() as td:
            deep_path = Path(td) / "a" / "b" / "c" / "data"
            save_processed(df, deep_path)
            parquet = deep_path.with_suffix(".parquet")
            csv_file = deep_path.with_suffix(".csv")
            assert parquet.exists() or csv_file.exists()
