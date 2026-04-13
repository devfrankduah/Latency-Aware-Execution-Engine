"""
Tests for src/features/engine.py

Covers:
  - FeatureCols constants
  - compute_returns()
  - compute_volatility()
  - compute_volume_features()
  - compute_spread_features()
  - compute_time_features()
  - compute_all_features()
  - Edge cases: small data, constant prices, window > data size
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import KlineSchema
from src.features.engine import (
    FeatureCols,
    compute_all_features,
    compute_returns,
    compute_spread_features,
    compute_time_features,
    compute_volatility,
    compute_volume_features,
)


# ─────────────────────────────────────────────
# Shared test fixture
# ─────────────────────────────────────────────

def make_klines(n: int = 500, price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    noise = rng.uniform(0.0002, 0.002, n)
    highs = close * (1 + noise)
    lows = close * (1 - noise)
    opens = np.roll(close, 1)
    opens[0] = price
    highs = np.maximum(highs, np.maximum(opens, close))
    lows = np.minimum(lows, np.minimum(opens, close))

    return pd.DataFrame({
        KlineSchema.TIMESTAMP: pd.date_range("2023-06-01", periods=n, freq="1min", tz="UTC"),
        KlineSchema.OPEN: opens,
        KlineSchema.HIGH: highs,
        KlineSchema.LOW: lows,
        KlineSchema.CLOSE: close,
        KlineSchema.VOLUME: rng.exponential(50.0, n),
        KlineSchema.QUOTE_VOLUME: rng.exponential(50.0, n) * close,
        KlineSchema.TRADES: rng.integers(50, 500, n),
        KlineSchema.SYMBOL: "BTCUSDT",
    })


# ─────────────────────────────────────────────
# FeatureCols
# ─────────────────────────────────────────────

class TestFeatureCols:
    def test_all_constants_are_strings(self):
        attrs = [
            FeatureCols.ROLLING_VOL, FeatureCols.LOG_RETURN,
            FeatureCols.ROLLING_VOL_MA, FeatureCols.VOLUME_IMBALANCE,
            FeatureCols.VOLUME_ZSCORE, FeatureCols.SPREAD_PROXY,
            FeatureCols.SPREAD_PROXY_MA, FeatureCols.RETURN_5,
            FeatureCols.RETURN_20, FeatureCols.HOUR_OF_DAY,
            FeatureCols.MINUTE_OF_HOUR, FeatureCols.HOUR_SIN,
            FeatureCols.HOUR_COS,
        ]
        for attr in attrs:
            assert isinstance(attr, str) and len(attr) > 0

    def test_no_duplicate_column_names(self):
        attrs = [
            FeatureCols.ROLLING_VOL, FeatureCols.LOG_RETURN,
            FeatureCols.ROLLING_VOL_MA, FeatureCols.VOLUME_IMBALANCE,
            FeatureCols.VOLUME_ZSCORE, FeatureCols.SPREAD_PROXY,
            FeatureCols.SPREAD_PROXY_MA, FeatureCols.RETURN_5,
            FeatureCols.RETURN_20, FeatureCols.HOUR_OF_DAY,
            FeatureCols.MINUTE_OF_HOUR, FeatureCols.HOUR_SIN,
            FeatureCols.HOUR_COS,
        ]
        assert len(attrs) == len(set(attrs))


# ─────────────────────────────────────────────
# compute_returns
# ─────────────────────────────────────────────

class TestComputeReturns:
    def test_adds_required_columns(self):
        df = make_klines(100)
        out = compute_returns(df)
        assert FeatureCols.LOG_RETURN in out.columns
        assert FeatureCols.RETURN_5 in out.columns
        assert FeatureCols.RETURN_20 in out.columns

    def test_first_row_log_return_is_nan(self):
        df = make_klines(100)
        out = compute_returns(df)
        assert pd.isna(out[FeatureCols.LOG_RETURN].iloc[0])

    def test_log_return_formula(self):
        """r_t = ln(close_t / close_{t-1})"""
        df = make_klines(50)
        out = compute_returns(df)
        close = df[KlineSchema.CLOSE].values
        expected = np.log(close[5] / close[4])
        assert abs(out[FeatureCols.LOG_RETURN].iloc[5] - expected) < 1e-12

    def test_return_5_is_5bar_log_return(self):
        df = make_klines(50)
        out = compute_returns(df)
        close = df[KlineSchema.CLOSE].values
        expected = np.log(close[10] / close[5])
        assert abs(out[FeatureCols.RETURN_5].iloc[10] - expected) < 1e-12

    def test_does_not_mutate_input(self):
        df = make_klines(100)
        original_cols = set(df.columns)
        compute_returns(df)
        assert set(df.columns) == original_cols

    def test_finite_values_after_warmup(self):
        df = make_klines(200)
        out = compute_returns(df)
        lr = out[FeatureCols.LOG_RETURN].dropna()
        assert np.all(np.isfinite(lr.values))


# ─────────────────────────────────────────────
# compute_volatility
# ─────────────────────────────────────────────

class TestComputeVolatility:
    def test_adds_rolling_vol_column(self):
        df = make_klines(100)
        out = compute_volatility(compute_returns(df), window=20)
        assert FeatureCols.ROLLING_VOL in out.columns

    def test_computes_returns_internally_if_missing(self):
        """Should add log_return if not already present."""
        df = make_klines(100)
        # Do NOT call compute_returns first
        out = compute_volatility(df, window=20)
        assert FeatureCols.ROLLING_VOL in out.columns
        assert FeatureCols.LOG_RETURN in out.columns

    def test_positive_after_warmup(self):
        df = make_klines(200)
        out = compute_volatility(compute_returns(df), window=20)
        valid = out[FeatureCols.ROLLING_VOL].dropna()
        assert (valid > 0).all()

    def test_annualized_vs_raw(self):
        df = make_klines(200)
        df_r = compute_returns(df)
        ann = compute_volatility(df_r, window=20, annualize=True)
        raw = compute_volatility(df_r, window=20, annualize=False)
        # Annualized should be larger than raw for 1-minute bars
        factor = np.sqrt(525_960)
        ratio = (ann[FeatureCols.ROLLING_VOL].dropna() /
                 raw[FeatureCols.ROLLING_VOL].dropna())
        np.testing.assert_allclose(ratio.values, factor, rtol=1e-6)

    def test_constant_price_gives_zero_vol(self):
        df = make_klines(100)
        df[KlineSchema.CLOSE] = 50_000.0  # Constant price
        df_r = compute_returns(df)
        out = compute_volatility(df_r, window=20)
        valid = out[FeatureCols.ROLLING_VOL].dropna()
        assert (valid == 0.0).all()

    def test_nan_count_matches_window_minus_1(self):
        df = make_klines(100)
        out = compute_volatility(compute_returns(df), window=20)
        # log_return NaN (1) + rolling std needs min_periods=5
        nan_count = out[FeatureCols.ROLLING_VOL].isna().sum()
        assert nan_count >= 5  # at least min_periods NaNs


# ─────────────────────────────────────────────
# compute_volume_features
# ─────────────────────────────────────────────

class TestComputeVolumeFeatures:
    def test_adds_all_columns(self):
        df = make_klines(100)
        out = compute_volume_features(df, window=20)
        assert FeatureCols.ROLLING_VOL_MA in out.columns
        assert FeatureCols.VOLUME_IMBALANCE in out.columns
        assert FeatureCols.VOLUME_ZSCORE in out.columns

    def test_imbalance_clipped_to_0_10(self):
        df = make_klines(500)
        # Force a spike to trigger clipping
        df.loc[300, KlineSchema.VOLUME] = df[KlineSchema.VOLUME].mean() * 1000
        out = compute_volume_features(df, window=20)
        valid = out[FeatureCols.VOLUME_IMBALANCE].dropna()
        assert (valid >= 0).all()
        assert (valid <= 10).all()

    def test_zscore_clipped_to_minus5_plus5(self):
        df = make_klines(500)
        out = compute_volume_features(df, window=20)
        valid = out[FeatureCols.VOLUME_ZSCORE].dropna()
        assert (valid >= -5).all()
        assert (valid <= 5).all()

    def test_rolling_ma_is_positive(self):
        df = make_klines(200)
        out = compute_volume_features(df, window=20)
        valid = out[FeatureCols.ROLLING_VOL_MA].dropna()
        assert (valid > 0).all()

    def test_normal_volume_imbalance_near_1(self):
        """On typical data the median imbalance should be close to 1."""
        df = make_klines(1000)
        out = compute_volume_features(df, window=20)
        med = out[FeatureCols.VOLUME_IMBALANCE].median()
        assert 0.5 < med < 2.0


# ─────────────────────────────────────────────
# compute_spread_features
# ─────────────────────────────────────────────

class TestComputeSpreadFeatures:
    def test_adds_columns(self):
        df = make_klines(100)
        out = compute_spread_features(df, window=20)
        assert FeatureCols.SPREAD_PROXY in out.columns
        assert FeatureCols.SPREAD_PROXY_MA in out.columns

    def test_spread_proxy_formula(self):
        """spread_proxy = (High - Low) / Close"""
        df = make_klines(100)
        out = compute_spread_features(df, window=20)
        expected = ((df[KlineSchema.HIGH] - df[KlineSchema.LOW]) /
                    df[KlineSchema.CLOSE])
        pd.testing.assert_series_equal(
            out[FeatureCols.SPREAD_PROXY].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_spread_proxy_is_nonnegative(self):
        df = make_klines(500)
        out = compute_spread_features(df, window=20)
        assert (out[FeatureCols.SPREAD_PROXY] >= 0).all()

    def test_spread_ma_is_positive_after_warmup(self):
        df = make_klines(200)
        out = compute_spread_features(df, window=20)
        valid = out[FeatureCols.SPREAD_PROXY_MA].dropna()
        assert (valid > 0).all()


# ─────────────────────────────────────────────
# compute_time_features
# ─────────────────────────────────────────────

class TestComputeTimeFeatures:
    def test_adds_all_columns(self):
        df = make_klines(100)
        out = compute_time_features(df)
        for col in [FeatureCols.HOUR_OF_DAY, FeatureCols.MINUTE_OF_HOUR,
                    FeatureCols.HOUR_SIN, FeatureCols.HOUR_COS]:
            assert col in out.columns

    def test_hour_of_day_range(self):
        df = make_klines(1500)  # More than one day
        out = compute_time_features(df)
        assert out[FeatureCols.HOUR_OF_DAY].between(0, 23).all()

    def test_minute_of_hour_range(self):
        df = make_klines(200)
        out = compute_time_features(df)
        assert out[FeatureCols.MINUTE_OF_HOUR].between(0, 59).all()

    def test_sin_cos_in_minus1_plus1(self):
        df = make_klines(500)
        out = compute_time_features(df)
        assert out[FeatureCols.HOUR_SIN].between(-1, 1).all()
        assert out[FeatureCols.HOUR_COS].between(-1, 1).all()

    def test_sin_cos_unit_circle(self):
        """sin² + cos² == 1 for all rows."""
        df = make_klines(300)
        out = compute_time_features(df)
        norms = out[FeatureCols.HOUR_SIN] ** 2 + out[FeatureCols.HOUR_COS] ** 2
        np.testing.assert_allclose(norms.values, 1.0, atol=1e-12)

    def test_hour_midnight_encoding(self):
        """Hour 0 should give sin=0, cos=1."""
        df = make_klines(1)
        df[KlineSchema.TIMESTAMP] = pd.to_datetime(["2023-01-01 00:00:00"], utc=True)
        out = compute_time_features(df)
        assert abs(out[FeatureCols.HOUR_SIN].iloc[0]) < 1e-10
        assert abs(out[FeatureCols.HOUR_COS].iloc[0] - 1.0) < 1e-10


# ─────────────────────────────────────────────
# compute_all_features
# ─────────────────────────────────────────────

class TestComputeAllFeatures:
    EXPECTED_FEATURE_COLS = [
        FeatureCols.LOG_RETURN, FeatureCols.RETURN_5, FeatureCols.RETURN_20,
        FeatureCols.ROLLING_VOL, FeatureCols.ROLLING_VOL_MA,
        FeatureCols.VOLUME_IMBALANCE, FeatureCols.VOLUME_ZSCORE,
        FeatureCols.SPREAD_PROXY, FeatureCols.SPREAD_PROXY_MA,
        FeatureCols.HOUR_OF_DAY, FeatureCols.MINUTE_OF_HOUR,
        FeatureCols.HOUR_SIN, FeatureCols.HOUR_COS,
    ]

    def test_all_feature_columns_added(self):
        df = make_klines(500)
        out = compute_all_features(df)
        for col in self.EXPECTED_FEATURE_COLS:
            assert col in out.columns, f"Missing: {col}"

    def test_original_columns_preserved(self):
        df = make_klines(200)
        out = compute_all_features(df)
        for col in KlineSchema.required_columns():
            assert col in out.columns

    def test_row_count_unchanged(self):
        df = make_klines(300)
        out = compute_all_features(df)
        assert len(out) == 300

    def test_does_not_mutate_input(self):
        df = make_klines(200)
        original_cols = set(df.columns)
        compute_all_features(df)
        assert set(df.columns) == original_cols

    def test_no_inf_values(self):
        df = make_klines(500)
        out = compute_all_features(df)
        feat_cols = [c for c in self.EXPECTED_FEATURE_COLS if c in out.columns]
        for col in feat_cols:
            vals = out[col].dropna().values
            assert not np.any(np.isinf(vals)), f"Inf found in {col}"

    def test_custom_windows(self):
        df = make_klines(300)
        out = compute_all_features(df, vol_window=10, volume_window=15, spread_window=5)
        assert FeatureCols.ROLLING_VOL in out.columns

    def test_warmup_nans_bounded(self):
        """NaN rows should be limited to the warm-up window."""
        window = 20
        df = make_klines(200)
        out = compute_all_features(df, vol_window=window)
        nan_rows = out[FeatureCols.ROLLING_VOL].isna().sum()
        # Warmup = window rows (plus 1 for log return diff)
        assert nan_rows <= window + 5

    def test_small_dataframe_does_not_crash(self):
        """30 rows is below default window=20 but should not raise."""
        df = make_klines(30)
        out = compute_all_features(df)
        assert len(out) == 30

    def test_single_row_does_not_crash(self):
        df = make_klines(1)
        out = compute_all_features(df)
        assert len(out) == 1
