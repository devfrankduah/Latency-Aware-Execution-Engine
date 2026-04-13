"""
Data validation for market data quality assurance.

PRODUCTION PATTERN: Validate early, validate often.
Why? Because:
  1. Bad data → bad features → bad model → bad decisions → lost money
  2. Silent data corruption is the #1 cause of ML system failures
  3. Catching issues at ingestion is 100x cheaper than debugging a model
  4. Validation reports build trust with stakeholders

Real-world examples of data issues this catches:
  - Exchange outages creating gaps (e.g., Binance maintenance windows)
  - Split/dividend adjustments creating price jumps
  - Volume spikes from wash trading or flash crashes
  - Timezone misalignment between data sources
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schemas import KlineSchema

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary of data quality checks.

    PRODUCTION PATTERN: Return structured results, not just True/False.
    This lets you log, visualize, and set thresholds on data quality.
    """

    total_rows: int
    date_range: tuple[str, str]
    missing_bars: int
    missing_pct: float
    zero_volume_bars: int
    duplicate_timestamps: int
    price_outliers: int
    negative_prices: int
    ohlc_violations: int  # Cases where High < Low, etc.
    is_valid: bool
    issues: list[str]

    def __str__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        report = [
            f"\n{'='*60}",
            f"  DATA VALIDATION REPORT - {status}",
            f"{'='*60}",
            f"  Rows:              {self.total_rows:,}",
            f"  Date range:        {self.date_range[0]} → {self.date_range[1]}",
            f"  Missing bars:      {self.missing_bars:,} ({self.missing_pct:.2%})",
            f"  Zero-volume bars:  {self.zero_volume_bars:,}",
            f"  Duplicate times:   {self.duplicate_timestamps:,}",
            f"  Price outliers:    {self.price_outliers:,}",
            f"  Negative prices:   {self.negative_prices:,}",
            f"  OHLC violations:   {self.ohlc_violations:,}",
        ]

        if self.issues:
            report.append(f"\n  Issues found:")
            for issue in self.issues:
                report.append(f"     {issue}")

        report.append(f"{'='*60}\n")
        return "\n".join(report)


def validate_klines(
    df: pd.DataFrame,
    expected_freq_minutes: int = 1,
    max_missing_pct: float = 0.05,
    outlier_zscore: float = 5.0,
) -> ValidationReport:
    """Run comprehensive validation on kline data.

    Args:
        df: DataFrame with KlineSchema columns.
        expected_freq_minutes: Expected bar frequency in minutes.
        max_missing_pct: Maximum allowed missing bar percentage.
        outlier_zscore: Z-score threshold for price outlier detection.

    Returns:
        ValidationReport with all quality metrics.
    """
    issues = []
    ts = KlineSchema.TIMESTAMP

    # --- Check required columns ---
    missing_cols = set(KlineSchema.required_columns()) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return ValidationReport(
            total_rows=len(df),
            date_range=("unknown", "unknown"),
            missing_bars=0, missing_pct=0, zero_volume_bars=0,
            duplicate_timestamps=0, price_outliers=0,
            negative_prices=0, ohlc_violations=0,
            is_valid=False, issues=issues,
        )

    # --- Early exit for empty DataFrame ---
    if len(df) == 0:
        return ValidationReport(
            total_rows=0,
            date_range=("unknown", "unknown"),
            missing_bars=0, missing_pct=0.0, zero_volume_bars=0,
            duplicate_timestamps=0, price_outliers=0,
            negative_prices=0, ohlc_violations=0,
            is_valid=False, issues=["Empty DataFrame"],
        )

    # --- Date range ---
    date_range = (
        str(df[ts].min()),
        str(df[ts].max()),
    )

    # --- Duplicates ---
    duplicate_timestamps = df[ts].duplicated().sum()
    if duplicate_timestamps > 0:
        issues.append(f"{duplicate_timestamps} duplicate timestamps found")

    # --- Missing bars ---
    # Calculate expected number of bars between first and last timestamp
    time_range = df[ts].max() - df[ts].min()
    expected_bars = int(time_range.total_seconds() / (expected_freq_minutes * 60)) + 1
    missing_bars = max(0, expected_bars - len(df))
    missing_pct = missing_bars / expected_bars if expected_bars > 0 else 0

    if missing_pct > max_missing_pct:
        issues.append(
            f"Missing {missing_pct:.1%} of expected bars "
            f"(threshold: {max_missing_pct:.1%})"
        )

    # --- Zero volume ---
    zero_volume_bars = (df[KlineSchema.VOLUME] == 0).sum()
    if zero_volume_bars > len(df) * 0.01:
        issues.append(f"{zero_volume_bars} zero-volume bars (>{1}% of data)")

    # --- Negative prices ---
    price_cols = [KlineSchema.OPEN, KlineSchema.HIGH, KlineSchema.LOW, KlineSchema.CLOSE]
    negative_prices = 0
    for col in price_cols:
        negs = (df[col] < 0).sum()
        negative_prices += negs
    if negative_prices > 0:
        issues.append(f"{negative_prices} negative price values")

    # --- OHLC consistency: High >= Low, High >= Open/Close, Low <= Open/Close ---
    ohlc_violations = (
        (df[KlineSchema.HIGH] < df[KlineSchema.LOW]).sum()
        + (df[KlineSchema.HIGH] < df[KlineSchema.OPEN]).sum()
        + (df[KlineSchema.HIGH] < df[KlineSchema.CLOSE]).sum()
        + (df[KlineSchema.LOW] > df[KlineSchema.OPEN]).sum()
        + (df[KlineSchema.LOW] > df[KlineSchema.CLOSE]).sum()
    )
    if ohlc_violations > 0:
        issues.append(f"{ohlc_violations} OHLC consistency violations")

    # --- Price outliers (z-score on log returns) ---
    log_returns = np.log(df[KlineSchema.CLOSE] / df[KlineSchema.CLOSE].shift(1)).dropna()
    if len(log_returns) > 0:
        z_scores = np.abs((log_returns - log_returns.mean()) / log_returns.std())
        price_outliers = int((z_scores > outlier_zscore).sum())
    else:
        price_outliers = 0

    if price_outliers > 10:
        issues.append(f"{price_outliers} price outliers (z > {outlier_zscore})")

    # --- Determine overall validity ---
    is_valid = bool(
        len(missing_cols) == 0
        and missing_pct <= max_missing_pct
        and negative_prices == 0
        and ohlc_violations == 0
        and duplicate_timestamps == 0
    )

    report = ValidationReport(
        total_rows=len(df),
        date_range=date_range,
        missing_bars=missing_bars,
        missing_pct=missing_pct,
        zero_volume_bars=zero_volume_bars,
        duplicate_timestamps=duplicate_timestamps,
        price_outliers=price_outliers,
        negative_prices=negative_prices,
        ohlc_violations=ohlc_violations,
        is_valid=is_valid,
        issues=issues,
    )

    logger.info(f"Validation {'passed' if is_valid else 'FAILED'} for {len(df):,} rows")
    return report
