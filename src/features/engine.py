"""
Feature engineering for the execution simulator.

These features capture the MARKET STATE that an execution policy needs
to make decisions. The key insight: good execution isn't about predicting
price direction - it's about understanding current liquidity, volatility,
and urgency conditions.

FEATURES WE COMPUTE:
  1. Rolling volatility     → "How risky is it to wait?"
  2. Rolling volume profile → "How much liquidity is available?"
  3. Volume imbalance       → "Is volume abnormally high/low right now?"
  4. Spread proxy           → "How much does crossing the spread cost?"
  5. Return momentum        → "Is price moving against us?"
  6. Time-of-day features   → "What part of the 24h cycle are we in?"

PRODUCTION PATTERN: Feature functions are pure (no side effects).
  Input: DataFrame → Output: DataFrame with new columns.
  This makes them testable, composable, and cacheable.
"""

import logging

import numpy as np
import pandas as pd

from src.data.schemas import KlineSchema

logger = logging.getLogger(__name__)


# ============================================================
# Feature column names (single source of truth)
# ============================================================

class FeatureCols:
    """Column names for engineered features."""

    # Volatility
    ROLLING_VOL = "rolling_volatility"          # Annualized rolling vol
    LOG_RETURN = "log_return"                    # Per-bar log return

    # Volume
    ROLLING_VOL_MA = "rolling_volume_ma"        # Moving average of volume
    VOLUME_IMBALANCE = "volume_imbalance"       # Current vol / rolling avg
    VOLUME_ZSCORE = "volume_zscore"             # Z-score of volume

    # Spread / Liquidity
    SPREAD_PROXY = "spread_proxy"               # (High - Low) / Close
    SPREAD_PROXY_MA = "spread_proxy_ma"         # Rolling avg of spread proxy

    # Momentum
    RETURN_5 = "return_5bar"                    # 5-bar return
    RETURN_20 = "return_20bar"                  # 20-bar return

    # Time features (crypto is 24/7 but has intraday patterns)
    HOUR_OF_DAY = "hour_of_day"
    MINUTE_OF_HOUR = "minute_of_hour"
    HOUR_SIN = "hour_sin"                       # Cyclical encoding
    HOUR_COS = "hour_cos"


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from close prices.

    Log returns are additive across time (unlike simple returns),
    which makes them the standard in quantitative finance.

    Math: r_t = ln(P_t / P_{t-1})
    """
    df = df.copy()
    close = df[KlineSchema.CLOSE]

    df[FeatureCols.LOG_RETURN] = np.log(close / close.shift(1))
    df[FeatureCols.RETURN_5] = np.log(close / close.shift(5))
    df[FeatureCols.RETURN_20] = np.log(close / close.shift(20))

    return df


def compute_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compute rolling realized volatility.

    This is the most important feature for execution:
    HIGH volatility → more risk from waiting → execute faster
    LOW volatility  → less risk → can be patient for better fills

    We annualize using crypto convention (365.25 days × 24 hours × 60 minutes)
    since crypto trades 24/7.

    Args:
        df: DataFrame with log returns.
        window: Rolling window size in bars.
        annualize: Whether to annualize (default True).

    Returns:
        DataFrame with rolling_volatility column added.
    """
    df = df.copy()

    if FeatureCols.LOG_RETURN not in df.columns:
        df = compute_returns(df)

    rolling_std = df[FeatureCols.LOG_RETURN].rolling(window=window, min_periods=5).std()

    if annualize:
        # Crypto: 365.25 * 24 * 60 = 525,960 minutes per year
        annualization_factor = np.sqrt(525_960)
        df[FeatureCols.ROLLING_VOL] = rolling_std * annualization_factor
    else:
        df[FeatureCols.ROLLING_VOL] = rolling_std

    return df


def compute_volume_features(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute volume-based features.

    Volume tells us about LIQUIDITY - how much can we trade without
    moving the price? Key features:

    - Volume MA: baseline "normal" volume level
    - Volume imbalance: current bar vs. average (>1 = more liquid than usual)
    - Volume Z-score: how unusual is current volume? (for anomaly detection)

    For execution: TRADE MORE when volume is high (lower market impact).

    Args:
        df: DataFrame with volume column.
        window: Rolling window size.
    """
    df = df.copy()
    vol = df[KlineSchema.VOLUME]

    # Rolling average
    df[FeatureCols.ROLLING_VOL_MA] = vol.rolling(window=window, min_periods=5).mean()

    # Imbalance: current / average (1.0 = normal, 2.0 = double normal volume)
    df[FeatureCols.VOLUME_IMBALANCE] = vol / df[FeatureCols.ROLLING_VOL_MA]
    # Clip extreme values (e.g., during flash crashes)
    df[FeatureCols.VOLUME_IMBALANCE] = df[FeatureCols.VOLUME_IMBALANCE].clip(0, 10)

    # Z-score
    rolling_mean = vol.rolling(window=window, min_periods=5).mean()
    rolling_std = vol.rolling(window=window, min_periods=5).std()
    df[FeatureCols.VOLUME_ZSCORE] = (vol - rolling_mean) / rolling_std.replace(0, np.nan)
    df[FeatureCols.VOLUME_ZSCORE] = df[FeatureCols.VOLUME_ZSCORE].clip(-5, 5)

    return df


def compute_spread_features(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute spread proxy features.

    Without order book data, we approximate the spread using:
        spread_proxy = (High - Low) / Close

    This correlates with actual bid-ask spread because:
    - In high-spread periods, price bounces between bid/ask → wider H-L range
    - This is a well-known approximation in market microstructure literature
      (Corwin & Schultz 2012, Abdi & Ranaldo 2017)

    For execution: HIGH spread → higher cost per trade → execute smaller slices.
    """
    df = df.copy()

    df[FeatureCols.SPREAD_PROXY] = (
        (df[KlineSchema.HIGH] - df[KlineSchema.LOW]) / df[KlineSchema.CLOSE]
    )

    df[FeatureCols.SPREAD_PROXY_MA] = (
        df[FeatureCols.SPREAD_PROXY].rolling(window=window, min_periods=5).mean()
    )

    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-of-day features.

    Even though crypto trades 24/7, volume has strong intraday patterns:
    - Peaks during US market hours (14:00-21:00 UTC)
    - Peaks during Asian market hours (00:00-08:00 UTC)
    - Trough during European evening (18:00-22:00 UTC)

    We use CYCLICAL encoding (sin/cos) so the model understands
    that hour 23 is close to hour 0 (not 23 units away).
    """
    df = df.copy()
    ts = df[KlineSchema.TIMESTAMP]

    df[FeatureCols.HOUR_OF_DAY] = ts.dt.hour
    df[FeatureCols.MINUTE_OF_HOUR] = ts.dt.minute

    # Cyclical encoding: maps 0-23 hours to a smooth circle
    df[FeatureCols.HOUR_SIN] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df[FeatureCols.HOUR_COS] = np.cos(2 * np.pi * ts.dt.hour / 24)

    return df


def compute_all_features(
    df: pd.DataFrame,
    vol_window: int = 20,
    volume_window: int = 20,
    spread_window: int = 20,
) -> pd.DataFrame:
    """Compute all features in the correct order.

    This is the main entry point for feature engineering.
    Call this once on your processed kline data, and all features
    are added as new columns.

    Args:
        df: DataFrame with KlineSchema columns.
        vol_window: Lookback for volatility calculation.
        volume_window: Lookback for volume features.
        spread_window: Lookback for spread features.

    Returns:
        DataFrame with all feature columns added.
        Note: First `max(windows)` rows will have NaN features.
    """
    logger.info(f"Computing features (windows: vol={vol_window}, "
                f"volume={volume_window}, spread={spread_window})")

    df = compute_returns(df)
    df = compute_volatility(df, window=vol_window)
    df = compute_volume_features(df, window=volume_window)
    df = compute_spread_features(df, window=spread_window)
    df = compute_time_features(df)

    # Count NaNs introduced by rolling windows
    n_nan = df[FeatureCols.ROLLING_VOL].isna().sum()
    logger.info(f"Features computed. {n_nan} warmup rows have NaN values.")

    return df
