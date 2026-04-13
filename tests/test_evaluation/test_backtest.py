"""
Tests for src/evaluation/backtest.py

Covers:
  - compute_strategy_stats()
  - run_backtest()
  - classify_regimes()
  - generate_results_dataframe()
  - print_comparison_table()
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import KlineSchema
from src.evaluation.backtest import (
    StrategyStats,
    classify_regimes,
    compute_strategy_stats,
    generate_results_dataframe,
    print_comparison_table,
    run_backtest,
)
from src.features.engine import compute_all_features
from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy
from src.simulator.engine import ChildOrder, ExecutionResult, Order
from src.simulator.impact import ImpactParams


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_klines(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 50_000.0
    close = price * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    noise = rng.uniform(0.0002, 0.002, n)
    highs = close * (1 + noise)
    lows = close * (1 - noise)
    opens = np.roll(close, 1)
    opens[0] = price
    highs = np.maximum(highs, np.maximum(opens, close))
    lows = np.minimum(lows, np.minimum(opens, close))

    df = pd.DataFrame({
        KlineSchema.TIMESTAMP: pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC"),
        KlineSchema.OPEN: opens,
        KlineSchema.HIGH: highs,
        KlineSchema.LOW: lows,
        KlineSchema.CLOSE: close,
        KlineSchema.VOLUME: rng.exponential(50.0, n),
        KlineSchema.QUOTE_VOLUME: rng.exponential(50.0, n) * close,
        KlineSchema.TRADES: rng.integers(50, 500, n),
        KlineSchema.SYMBOL: "BTCUSDT",
    })
    return compute_all_features(df)


def _make_execution_result(
    is_bps: float = 2.0,
    cost_usd: float = 50.0,
    vwap_bps: float = 1.5,
    n_orders: int = 60,
    fill_ratio: float = 1.0,
) -> ExecutionResult:
    """Build a minimal fake ExecutionResult for unit testing stats functions."""
    order = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=60)
    child = ChildOrder(
        bar_index=0,
        timestamp=pd.Timestamp("2023-01-01", tz="UTC"),
        quantity=1.0,
        mid_price=50_000.0,
        exec_price=50_001.0,
        spread_cost=0.5,
        temporary_cost=0.5,
        permanent_cost=0.1,
        participation_rate=0.05,
        remaining_quantity=0.0,
    )
    return ExecutionResult(
        order=order,
        child_orders=[child] * n_orders,
        arrival_price=50_000.0,
        final_price=50_050.0,
        total_executed=order.total_quantity * fill_ratio,
        avg_exec_price=50_000.0 * (1 + is_bps / 1e4),
        implementation_shortfall_bps=is_bps,
        vwap_slippage_bps=vwap_bps,
        total_cost_usd=cost_usd,
        cost_breakdown={"spread": 25.0, "temporary": 20.0, "permanent": 5.0},
        n_child_orders=n_orders,
        execution_time_bars=n_orders,
    )


# ─────────────────────────────────────────────
# compute_strategy_stats
# ─────────────────────────────────────────────

class TestComputeStrategyStats:
    def test_empty_results_returns_zero_stats(self):
        stats = compute_strategy_stats("TWAP", [])
        assert stats.n_simulations == 0
        assert stats.is_mean == 0
        assert stats.fill_rate == 0

    def test_single_result(self):
        result = _make_execution_result(is_bps=3.0, cost_usd=75.0)
        stats = compute_strategy_stats("TWAP", [result])
        assert stats.n_simulations == 1
        assert abs(stats.is_mean - 3.0) < 1e-9
        assert abs(stats.cost_mean - 75.0) < 1e-9
        assert stats.is_std == 0.0  # Only one sample

    def test_mean_is_correct(self):
        results = [_make_execution_result(is_bps=v) for v in [1.0, 3.0, 5.0]]
        stats = compute_strategy_stats("test", results)
        assert abs(stats.is_mean - 3.0) < 1e-9

    def test_percentiles_ordered(self):
        results = [_make_execution_result(is_bps=float(v)) for v in range(1, 101)]
        stats = compute_strategy_stats("test", results)
        assert stats.is_p5 <= stats.is_median <= stats.is_p95

    def test_fill_rate_all_complete(self):
        results = [_make_execution_result(fill_ratio=1.0) for _ in range(10)]
        stats = compute_strategy_stats("test", results)
        assert abs(stats.fill_rate - 1.0) < 1e-9

    def test_fill_rate_partial(self):
        full = [_make_execution_result(fill_ratio=1.0) for _ in range(7)]
        partial = [_make_execution_result(fill_ratio=0.5) for _ in range(3)]
        stats = compute_strategy_stats("test", full + partial)
        assert abs(stats.fill_rate - 0.7) < 1e-9

    def test_avg_participation_rate(self):
        results = [_make_execution_result(n_orders=5) for _ in range(10)]
        stats = compute_strategy_stats("test", results)
        # Each child order has participation_rate=0.05
        assert abs(stats.avg_participation_rate - 0.05) < 1e-9

    def test_str_does_not_crash(self):
        stats = compute_strategy_stats("TWAP", [_make_execution_result()])
        s = str(stats)
        assert "TWAP" in s

    def test_name_is_preserved(self):
        stats = compute_strategy_stats("MyPolicy", [_make_execution_result()])
        assert stats.name == "MyPolicy"


# ─────────────────────────────────────────────
# run_backtest
# ─────────────────────────────────────────────

class TestRunBacktest:
    def setup_method(self):
        self.df = _make_klines(n=2000)
        self.order = Order(
            symbol="BTCUSDT", side="buy",
            total_quantity=1.0, time_horizon_bars=60,
        )
        self.policies = {
            "TWAP": TWAPPolicy(),
            "Immediate": ImmediatePolicy(),
        }

    def test_returns_stats_for_each_policy(self):
        stats = run_backtest(self.df, self.policies, self.order, n_simulations=10)
        assert set(stats.keys()) == {"TWAP", "Immediate"}

    def test_stats_type(self):
        stats = run_backtest(self.df, self.policies, self.order, n_simulations=10)
        for v in stats.values():
            assert isinstance(v, StrategyStats)

    def test_n_simulations_matches(self):
        stats = run_backtest(self.df, self.policies, self.order, n_simulations=20)
        for v in stats.values():
            assert v.n_simulations == 20

    def test_reproducible_with_same_seed(self):
        s1 = run_backtest(self.df, self.policies, self.order, n_simulations=15, seed=99)
        s2 = run_backtest(self.df, self.policies, self.order, n_simulations=15, seed=99)
        assert abs(s1["TWAP"].is_mean - s2["TWAP"].is_mean) < 1e-9

    def test_different_seeds_give_different_results(self):
        s1 = run_backtest(self.df, self.policies, self.order, n_simulations=50, seed=1)
        s2 = run_backtest(self.df, self.policies, self.order, n_simulations=50, seed=2)
        # Very unlikely they'd be identical with different seeds
        assert s1["TWAP"].is_mean != s2["TWAP"].is_mean

    def test_immediate_higher_cost_than_twap(self):
        """Core theoretical result: immediate execution costs more."""
        stats = run_backtest(self.df, self.policies, self.order, n_simulations=50, seed=0)
        assert stats["Immediate"].is_mean > stats["TWAP"].is_mean

    def test_insufficient_data_raises(self):
        tiny_df = _make_klines(n=100)
        order = Order("BTCUSDT", "buy", 1.0, time_horizon_bars=200)
        with pytest.raises(ValueError, match="Not enough data"):
            run_backtest(tiny_df, self.policies, order, n_simulations=5)

    def test_factory_callable_policies(self):
        """Policies can be passed as factories (callables)."""
        policies = {"TWAP": TWAPPolicy}
        stats = run_backtest(self.df, policies, self.order, n_simulations=10)
        assert "TWAP" in stats

    def test_sell_order_works(self):
        order = Order("BTCUSDT", "sell", 1.0, time_horizon_bars=60)
        stats = run_backtest(self.df, {"TWAP": TWAPPolicy()}, order, n_simulations=10)
        assert stats["TWAP"].n_simulations == 10


# ─────────────────────────────────────────────
# classify_regimes
# ─────────────────────────────────────────────

class TestClassifyRegimes:
    def test_returns_series_same_length(self):
        df = _make_klines(2000)
        regimes = classify_regimes(df)
        assert len(regimes) == len(df)

    def test_only_three_labels(self):
        df = _make_klines(2000)
        regimes = classify_regimes(df)
        assert set(regimes.dropna().unique()).issubset({"high_vol", "normal", "low_vol"})

    def test_all_three_regimes_present(self):
        df = _make_klines(2000)
        regimes = classify_regimes(df)
        labels = set(regimes.dropna().unique())
        assert "high_vol" in labels
        assert "low_vol" in labels
        assert "normal" in labels

    def test_raises_if_features_missing(self):
        df = _make_klines(2000)
        # Strip the feature column
        df = df.drop(columns=["rolling_volatility"])
        with pytest.raises(ValueError, match="Features not computed"):
            classify_regimes(df)


# ─────────────────────────────────────────────
# generate_results_dataframe
# ─────────────────────────────────────────────

class TestGenerateResultsDataframe:
    def test_correct_columns(self):
        df = _make_klines(2000)
        order = Order("BTCUSDT", "buy", 1.0, 60)
        stats = run_backtest(df, {"TWAP": TWAPPolicy()}, order, n_simulations=10)
        result_df = generate_results_dataframe(stats)
        expected_cols = {
            "strategy", "n_simulations", "is_mean_bps", "is_median_bps",
            "is_std_bps", "is_p5_bps", "is_p95_bps", "cost_mean_usd",
            "cost_std_usd", "vwap_mean_bps", "avg_child_orders", "fill_rate",
        }
        assert expected_cols.issubset(set(result_df.columns))

    def test_one_row_per_strategy(self):
        df = _make_klines(2000)
        order = Order("BTCUSDT", "buy", 1.0, 60)
        policies = {"TWAP": TWAPPolicy(), "Immediate": ImmediatePolicy()}
        stats = run_backtest(df, policies, order, n_simulations=10)
        result_df = generate_results_dataframe(stats)
        assert len(result_df) == 2

    def test_empty_stats_gives_empty_dataframe(self):
        result_df = generate_results_dataframe({})
        assert len(result_df) == 0


# ─────────────────────────────────────────────
# print_comparison_table (smoke test)
# ─────────────────────────────────────────────

class TestPrintComparisonTable:
    def test_does_not_crash(self, capsys):
        results = {
            "TWAP": _make_execution_result(is_bps=2.0),
            "Immediate": _make_execution_result(is_bps=8.0),
        }
        stats = {k: compute_strategy_stats(k, [v]) for k, v in results.items()}
        print_comparison_table(stats)
        captured = capsys.readouterr()
        assert "TWAP" in captured.out
        assert "Immediate" in captured.out

    def test_empty_stats_does_not_crash(self, capsys):
        print_comparison_table({})
        # Just shouldn't raise
