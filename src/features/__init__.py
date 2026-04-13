"""
Feature engineering for execution signals.

Computes rolling volatility, volume imbalance, spread proxy,
momentum, and time-of-day features from raw kline data.

Usage:
    from src.features.engine import compute_all_features, FeatureCols

    df = compute_all_features(raw_df)
    print(df[FeatureCols.ROLLING_VOL].describe())
"""

from src.features.engine import compute_all_features, FeatureCols
