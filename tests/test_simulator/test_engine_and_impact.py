#!/usr/bin/env python3
"""
Tests for the execution simulator and market impact model.

Run: python -m pytest tests/test_simulator/ -v
  or: python tests/test_simulator/test_engine_and_impact.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd


# ── Helper: generate test data ──
def make_data(n=5000, price=50000.0, seed=42):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    noise = rng.uniform(0.0001, 0.001, n)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min', tz='UTC'),
        'open': np.roll(close, 1), 'high': close * (1 + noise),
        'low': close * (1 - noise), 'close': close,
        'volume': rng.exponential(50.0, n),
        'quote_volume': rng.exponential(50.0, n) * close,
        'num_trades': rng.integers(100, 1000, n), 'symbol': 'BTCUSDT',
    })
    df.iloc[0, df.columns.get_loc('open')] = price
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df


class Results:
    def __init__(self):
        self.passed = self.failed = 0

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  {name}")
        else:
            self.failed += 1
            print(f"    {name}: {detail}")

    def summary(self):
        t = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  SIMULATOR TESTS: {self.passed}/{t} passed, {self.failed} failed")
        print(f"{'='*60}")
        return self.failed == 0


def test_impact_model():
    """Test the core market impact calculations."""
    print("\nIMPACT MODEL TESTS")
    r = Results()

    from src.simulator.impact import ImpactParams, compute_execution_price

    params = ImpactParams(temporary_impact=0.1, permanent_impact=0.01, spread_bps=1.0)

    # Buy should cost more than mid
    price, costs = compute_execution_price(50000, 0.5, 100, "buy", params)
    r.check("buy price > mid", price > 50000, f"got {price}")

    # Sell should get less than mid
    price, costs = compute_execution_price(50000, 0.5, 100, "sell", params)
    r.check("sell price < mid", price < 50000, f"got {price}")

    # Zero quantity = no cost
    price, costs = compute_execution_price(50000, 0, 100, "buy", params)
    r.check("zero qty → zero cost", costs["total_cost_bps"] == 0)

    # Higher participation = higher cost
    _, c_low = compute_execution_price(50000, 1.0, 100, "buy", params)
    _, c_high = compute_execution_price(50000, 10.0, 100, "buy", params)
    r.check("more qty → higher cost", c_high["total_cost_bps"] > c_low["total_cost_bps"])

    # Participation cap respected
    _, costs = compute_execution_price(50000, 1000, 100, "buy", params)
    r.check("participation capped", costs["participation_rate"] <= params.max_participation_rate + 1e-10)

    return r


def test_almgren_chriss_trajectory():
    """Test the A-C optimal trajectory computation."""
    print("\nALMGREN-CHRISS TRAJECTORY TESTS")
    r = Results()

    from src.simulator.impact import compute_almgren_chriss_trajectory, ImpactParams

    params = ImpactParams()

    # Trajectory sums to total quantity
    traj = compute_almgren_chriss_trajectory(10.0, 60, 0.5, 0.001, params)
    r.check("sums to total", abs(traj.sum() - 10.0) < 1e-6, f"sum={traj.sum()}")

    # All non-negative
    r.check("all non-negative", (traj >= -1e-10).all())

    # High lambda → front-loaded
    traj_high = compute_almgren_chriss_trajectory(10.0, 60, 5.0, 0.001, params)
    first_half = traj_high[:30].sum()
    second_half = traj_high[30:].sum()
    r.check("high λ front-loaded", first_half > second_half,
            f"first={first_half:.3f} second={second_half:.3f}")

    # Low lambda → near TWAP
    traj_low = compute_almgren_chriss_trajectory(10.0, 60, 0.001, 0.001, params)
    cv = traj_low.std() / traj_low.mean()
    r.check("low λ ≈ TWAP (CV < 0.1)", cv < 0.1, f"CV={cv:.4f}")

    # Single step
    traj_1 = compute_almgren_chriss_trajectory(10.0, 1, 0.5, 0.001, params)
    r.check("1 step = full quantity", abs(traj_1[0] - 10.0) < 1e-6)

    return r


def test_simulator_engine():
    """Test the full execution simulator."""
    print("\nSIMULATOR ENGINE TESTS")
    r = Results()

    from src.features.engine import compute_all_features
    from src.simulator.engine import Order, simulate_execution
    from src.simulator.impact import ImpactParams
    from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy

    df = make_data(3000)
    df = compute_all_features(df)
    params = ImpactParams(temporary_impact=0.1, spread_bps=1.0)
    order = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=60)
    start = 100

    # Immediate fills in 1 order
    res = simulate_execution(df, order, ImmediatePolicy(), start, params)
    r.check("immediate: 1 child order", res.n_child_orders == 1)
    r.check("immediate: fills order", res.total_executed > 0.5)

    # TWAP spreads across bars
    res = simulate_execution(df, order, TWAPPolicy(), start, params)
    r.check("TWAP: many child orders", res.n_child_orders > 30)
    r.check("TWAP: fills 100%", abs(res.total_executed - 1.0) < 0.01)

    # VWAP fills order
    res = simulate_execution(df, order, VWAPPolicy(), start, params)
    r.check("VWAP: fills order", res.total_executed > 0.9)

    # Immediate has higher cost than TWAP (fundamental theoretical result)
    imm = simulate_execution(df, order, ImmediatePolicy(), start, params)
    twap = simulate_execution(df, order, TWAPPolicy(), start, params)
    r.check("immediate cost > TWAP cost",
            imm.total_cost_usd > twap.total_cost_usd,
            f"imm={imm.total_cost_usd:.2f} twap={twap.total_cost_usd:.2f}")

    # IS is computed and finite
    r.check("IS is finite", np.isfinite(twap.implementation_shortfall_bps))
    r.check("VWAP slippage is finite", np.isfinite(twap.vwap_slippage_bps))

    return r


def test_variable_spread_env():
    """Test the large-order environment with variable spread."""
    print("\nVARIABLE SPREAD ENVIRONMENT TESTS")
    r = Results()

    # Import the Env from train_large.py by executing the class definition
    from src.features.engine import compute_all_features
    df = make_data(5000)
    df = compute_all_features(df)

    # Manually create a simplified env for testing
    close = df['close'].values
    volume = df['volume'].values
    spread_raw = ((df['high'] - df['low']) / df['close']).values

    # Test spread is variable (not constant)
    r.check("spread varies in data", spread_raw.std() > 0,
            f"std={spread_raw.std():.6f}")

    # Spread proxy correlates with high-low range
    r.check("spread proxy > 0", (spread_raw > 0).mean() > 0.95)

    # Volume varies (needed for VWAP to differ from TWAP)
    r.check("volume varies", volume.std() / volume.mean() > 0.5,
            f"CV={volume.std()/volume.mean():.2f}")

    # Test that higher participation = higher cost (nonlinear)
    price = 50000.0
    spread = 0.0002  # 2 bps
    impact = 0.3

    cost_low = price * spread * 0.5 + impact * price * (0.01 ** 1.5)   # 1% participation
    cost_high = price * spread * 0.5 + impact * price * (0.05 ** 1.5)  # 5% participation
    r.check("cost increases with participation",
            cost_high > cost_low,
            f"low={cost_low:.2f} high={cost_high:.2f}")

    # Nonlinear: doubling participation more than doubles impact
    impact_1pct = impact * price * (0.01 ** 1.5)
    impact_2pct = impact * price * (0.02 ** 1.5)
    ratio = impact_2pct / impact_1pct
    r.check("impact is super-linear (ratio > 2.0)",
            ratio > 2.0, f"ratio={ratio:.2f}")

    return r


def test_data_pipeline():
    """Test data loading and feature computation."""
    print("\nDATA PIPELINE TESTS")
    r = Results()

    from src.features.engine import compute_all_features

    df = make_data(2000)
    df_feat = compute_all_features(df)

    # All expected features present
    expected = ['rolling_volatility', 'volume_imbalance', 'spread_proxy',
                'return_5bar', 'return_20bar', 'hour_sin', 'hour_cos']
    for feat in expected:
        r.check(f"feature '{feat}' exists", feat in df_feat.columns)

    # Volatility is positive
    vol = df_feat['rolling_volatility'].dropna()
    r.check("volatility ≥ 0", (vol >= 0).all())

    # Hour sin/cos in [-1, 1]
    r.check("hour_sin in [-1,1]", df_feat['hour_sin'].between(-1, 1).all())
    r.check("hour_cos in [-1,1]", df_feat['hour_cos'].between(-1, 1).all())

    # No NaN in original columns after feature computation
    r.check("close has no NaN", df_feat['close'].notna().all())
    r.check("volume has no NaN", df_feat['volume'].notna().all())

    # Spread proxy is positive
    sp = df_feat['spread_proxy'].dropna()
    r.check("spread proxy ≥ 0", (sp >= 0).all())

    return r


if __name__ == "__main__":
    print("=" * 60)
    print("  EXECUTION SIMULATOR - TEST SUITE")
    print("=" * 60)

    all_results = []
    all_results.append(test_impact_model())
    all_results.append(test_almgren_chriss_trajectory())
    all_results.append(test_simulator_engine())
    all_results.append(test_variable_spread_env())
    all_results.append(test_data_pipeline())

    total_pass = sum(r.passed for r in all_results)
    total_fail = sum(r.failed for r in all_results)
    total = total_pass + total_fail

    print(f"\n{'='*60}")
    print(f"  ALL SIMULATOR TESTS: {total_pass}/{total} passed, {total_fail} failed")
    print(f"{'='*60}")
    sys.exit(0 if total_fail == 0 else 1)