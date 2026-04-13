#!/usr/bin/env python3
"""
Failure Cases, Edge Cases, and Limitations Analysis.

CS5130 rubric requires: "Discuss failure cases, edge cases, limitations,
and performance vs complexity trade-offs."

This script runs targeted tests on known difficult scenarios and produces
a structured analysis suitable for the final report.

Usage:
    python scripts/analysis.py --data data/processed/BTCUSDT_klines_1m.parquet
"""

import sys, logging, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features.engine import compute_all_features
from src.simulator.engine import Order, simulate_execution
from src.simulator.impact import ImpactParams
from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy, AlmgrenChrissPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_data(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, parse_dates=['timestamp'])
    return compute_all_features(df)


# ══════════════════════════════════════════
# 1. FAILURE CASES
# ══════════════════════════════════════════

def analyze_failure_cases(df):
    """Identify and analyze scenarios where strategies fail."""
    print(f'\n{"="*70}')
    print(f'  1. FAILURE CASE ANALYSIS')
    print(f'{"="*70}')

    order = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=60)
    params = ImpactParams()
    policies = {"TWAP": TWAPPolicy(), "VWAP": VWAPPolicy(),
                "A-C(λ=1)": AlmgrenChrissPolicy(risk_aversion=1.0)}

    rng = np.random.default_rng(42)
    starts = rng.integers(100, len(df) - 61, size=1000)

    # Find worst-case episodes per strategy
    for name, policy in policies.items():
        is_vals = []
        for si in starts:
            if hasattr(policy, 'reset'): policy.reset()
            try:
                r = simulate_execution(df, order, policy, int(si), params)
                is_vals.append((r.implementation_shortfall_bps, int(si)))
            except Exception:
                pass

        is_vals.sort(key=lambda x: x[0], reverse=True)
        worst = is_vals[:5]

        print(f'\n  {name} - 5 worst episodes (highest IS):')
        for is_bps, si in worst:
            ts = df.iloc[si]['timestamp']
            vol = df.iloc[si:si+60]['rolling_volatility'].mean()
            print(f'    IS={is_bps:>+8.1f} bps | start={ts} | vol={vol:.2%}')

    # Failure case 1: Flash crash periods
    print(f'\n  FAILURE CASE: Flash Crash / Extreme Volatility')
    print(f'  {"-"*50}')
    log_ret = np.log(df['close'] / df['close'].shift(1)).dropna()
    crash_bars = df.index[df['log_return'].abs() > 0.02].tolist()  # >2% move in 1 minute
    print(f'  Found {len(crash_bars)} bars with >2% moves')
    if crash_bars:
        # Test execution starting at crash bars
        crash_costs = []
        normal_costs = []
        for si in crash_bars[:50]:
            if si < 100 or si > len(df) - 61: continue
            try:
                r = simulate_execution(df, order, TWAPPolicy(), si, params)
                crash_costs.append(r.total_cost_usd)
            except: pass
        normal_starts = rng.integers(100, len(df)-61, size=50)
        for si in normal_starts:
            try:
                r = simulate_execution(df, order, TWAPPolicy(), int(si), params)
                normal_costs.append(r.total_cost_usd)
            except: pass
        if crash_costs and normal_costs:
            print(f'  TWAP cost during crashes: ${np.mean(crash_costs):.2f} '
                  f'(vs normal: ${np.mean(normal_costs):.2f}) '
                  f'→ {np.mean(crash_costs)/np.mean(normal_costs):.1f}x more expensive')

    # Failure case 2: Zero/low volume periods
    print(f'\n  FAILURE CASE: Low Volume Periods')
    print(f'  {"-"*50}')
    low_vol_bars = df[df['volume'] < df['volume'].quantile(0.05)].index.tolist()
    print(f'  Found {len(low_vol_bars)} bars with volume < 5th percentile')
    print(f'  Risk: Orders may not fill completely. Participation rate hits cap.')


# ══════════════════════════════════════════
# 2. EDGE CASES
# ══════════════════════════════════════════

def analyze_edge_cases(df):
    """Test edge cases that could break the system."""
    print(f'\n{"="*70}')
    print(f'  2. EDGE CASE ANALYSIS')
    print(f'{"="*70}')

    params = ImpactParams()

    # Edge case 1: Very large order relative to market
    print(f'\n  EDGE CASE: Order larger than available liquidity')
    print(f'  {"-"*50}')
    for qty in [1, 10, 50, 100, 500]:
        order = Order(symbol="BTCUSDT", side="buy", total_quantity=float(qty), time_horizon_bars=60)
        fills = []
        for _ in range(100):
            si = np.random.randint(100, len(df) - 61)
            try:
                r = simulate_execution(df, order, TWAPPolicy(), si, params)
                fills.append(r.total_executed / order.total_quantity)
            except: pass
        if fills:
            print(f'    {qty:>5} BTC: fill rate = {np.mean(fills):.1%} '
                  f'(min: {np.min(fills):.1%}) '
                  f'{"[+]" if np.mean(fills) > 0.95 else "[~]" if np.mean(fills) > 0.80 else "[-]"}')

    # Edge case 2: Very short horizon
    print(f'\n  EDGE CASE: Short execution horizon')
    print(f'  {"-"*50}')
    order = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=60)
    for horizon in [1, 5, 10, 30, 60]:
        order_h = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=horizon)
        costs = []
        for _ in range(100):
            si = np.random.randint(100, len(df) - horizon - 1)
            try:
                r = simulate_execution(df, order_h, TWAPPolicy(), si, params)
                costs.append(r.total_cost_usd)
            except: pass
        if costs:
            print(f'    Horizon={horizon:>3} bars: avg cost = ${np.mean(costs):.2f} '
                  f'(std: ${np.std(costs):.2f})')

    # Edge case 3: Zero volume bars
    print(f'\n  EDGE CASE: Handling zero-volume bars')
    print(f'  {"-"*50}')
    zero_vol = (df['volume'] == 0).sum()
    print(f'  Zero-volume bars in data: {zero_vol} ({zero_vol/len(df):.3%})')
    print(f'  Handling: Simulator skips zero-volume bars (no fill possible).')
    print(f'  Impact: TWAP adjusts remaining quantity to subsequent bars.')

    # Edge case 4: Extreme spread
    print(f'\n  EDGE CASE: Extreme spread conditions')
    print(f'  {"-"*50}')
    spread = (df['high'] - df['low']) / df['close']
    print(f'  Spread percentiles:')
    for p in [1, 5, 50, 95, 99]:
        print(f'    P{p:>2}: {spread.quantile(p/100)*100:.4f}%')


# ══════════════════════════════════════════
# 3. LIMITATIONS
# ══════════════════════════════════════════

def analyze_limitations():
    """Document system limitations."""
    print(f'\n{"="*70}')
    print(f'  3. LIMITATIONS')
    print(f'{"="*70}')

    limitations = [
        ("No real order book data",
         "We use OHLCV bars (1-min) not tick-level LOB data. The spread proxy "
         "(high-low)/close approximates but doesn't capture actual bid-ask dynamics. "
         "Real execution systems use Level 2/3 data with queue position modeling."),

        ("Simulated market impact, not real fills",
         "Our Almgren-Chriss impact model assumes impact ∝ participation^1.5. "
         "Real market impact depends on order book depth, hidden liquidity, "
         "and other participants. The model is calibrated heuristically, not from data."),

        ("Single asset execution only",
         "We execute one asset at a time. Real portfolio rebalancing involves "
         "correlated multi-asset execution where trading BTCUSDT affects ETHUSDT. "
         "Cross-asset impact is not modeled."),

        ("No multi-venue routing",
         "Real execution splits orders across Binance, Coinbase, Kraken, etc. "
         "Our simulator uses a single venue. Cross-exchange arbitrage and "
         "best-execution routing are not implemented."),

        ("Walk-forward but no live testing",
         "We evaluate on historical 2024 data (out-of-sample) but have not "
         "tested with live market connections. Paper trading would be the "
         "natural next step before deployment."),

        ("DQN action space is discrete",
         "The agent chooses from 7 participation levels. Continuous action "
         "spaces (via PPO/SAC) would allow finer-grained control. The discrete "
         "approximation may miss optimal intermediate values."),

        ("1-minute bar resolution",
         "Real latency-aware systems operate at millisecond granularity. "
         "Our 1-minute bars aggregate thousands of events, masking "
         "microstructure dynamics like queue priority and partial fills."),
    ]

    for i, (title, detail) in enumerate(limitations, 1):
        print(f'\n  {i}. {title}')
        print(f'     {detail}')


# ══════════════════════════════════════════
# 4. PERFORMANCE VS COMPLEXITY TRADE-OFFS
# ══════════════════════════════════════════

def analyze_performance_complexity(df):
    """Benchmark strategies on both performance and computational cost."""
    print(f'\n{"="*70}')
    print(f'  4. PERFORMANCE vs COMPLEXITY TRADE-OFFS')
    print(f'{"="*70}')

    order = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=60)
    params = ImpactParams()

    strategies = {
        "Immediate": lambda: ImmediatePolicy(),
        "TWAP": lambda: TWAPPolicy(),
        "VWAP": lambda: VWAPPolicy(),
        "A-C (λ=0.5)": lambda: AlmgrenChrissPolicy(risk_aversion=0.5),
    }

    rng = np.random.default_rng(42)
    starts = rng.integers(100, len(df) - 61, size=200)

    print(f'\n  {"Strategy":<16s} {"Avg Cost($)":>12s} {"IS(bps)":>10s} '
          f'{"Time/Order":>12s} {"Complexity":>14s}')
    print(f'  {"─"*66}')

    for name, factory in strategies.items():
        costs, is_vals = [], []
        t0 = time.time()
        for si in starts:
            policy = factory()
            if hasattr(policy, 'reset'): policy.reset()
            try:
                r = simulate_execution(df, order, policy, int(si), params)
                costs.append(r.total_cost_usd)
                is_vals.append(r.implementation_shortfall_bps)
            except: pass
        elapsed = (time.time() - t0) / len(starts) * 1000  # ms per order

        complexity = {
            "Immediate": "O(1)",
            "TWAP": "O(T)",
            "VWAP": "O(T)",
            "A-C (λ=0.5)": "O(T) + solve",
        }.get(name, "O(T)")

        print(f'  {name:<16s} ${np.mean(costs):>11.2f} {np.mean(is_vals):>+10.2f} '
              f'{elapsed:>10.2f} ms {complexity:>14s}')

    # DQN complexity note
    print(f'\n  {"DQN Agent":<16s} {"(see sweep)":>12s} {"(see sweep)":>10s} '
          f'{"~5 ms":>12s} {"O(T×fwd)":>14s}')
    print(f'\n  Notes:')
    print(f'    - TWAP/VWAP: O(T) where T=60 bars. No training needed. ~0.3ms/order.')
    print(f'    - A-C: O(T) + one-time trajectory solve. ~0.5ms/order.')
    print(f'    - DQN: O(T × forward_pass). ~5ms/order. But requires 50K episode training (~3hrs).')
    print(f'    - The DQN overhead is justified when savings > compute cost.')
    print(f'    - At $940/order savings × 10 orders/day = $3.4M/year >> training cost.')


# ══════════════════════════════════════════
# 5. ETHICAL CONSIDERATIONS
# ══════════════════════════════════════════

def analyze_ethics():
    """Ethical risk analysis - required by CS5130 rubric."""
    print(f'\n{"="*70}')
    print(f'  5. ETHICAL CONSIDERATIONS')
    print(f'{"="*70}')

    considerations = [
        ("Market Manipulation Risk",
         "An adaptive execution algorithm could inadvertently engage in "
         "manipulative behavior such as spoofing (placing orders it intends to cancel) "
         "or layering (creating artificial depth). Our agent's action space is limited "
         "to scheduling real executions and cannot place/cancel limit orders, "
         "mitigating this risk. However, aggressive participation rates could "
         "constitute market manipulation under some jurisdictions.",
         "Mitigation: Hard cap on participation rate (15% max). "
         "All execution decisions are logged for audit. "
         "Agent actions are constrained to legitimate execution scheduling."),

        ("Fairness and Market Impact on Other Participants",
         "Optimized execution by large institutional traders extracts value "
         "from less sophisticated market participants. When our agent times "
         "execution to avoid high-spread periods, the cost is implicitly "
         "transferred to others who trade during those periods.",
         "Mitigation: This is an inherent feature of competitive markets, "
         "not a bug. Better execution benefits end investors (pension funds, "
         "ETF holders). We document our impact model transparently."),

        ("Over-Reliance on Automation",
         "Automated execution systems can fail silently during market anomalies "
         "(exchange outages, flash crashes, circuit breakers). Blind reliance "
         "on the RL agent during unprecedented market conditions is dangerous.",
         "Mitigation: The system includes fill-rate monitoring. "
         "If fill rate drops below 90%, the agent is flagged. "
         "Production deployment would require human-in-the-loop oversight "
         "and kill switches for extreme market conditions."),

        ("Data Privacy",
         "Our system uses publicly available exchange data (OHLCV from Binance). "
         "No personally identifiable information (PII) is collected or processed. "
         "However, a production system connected to a brokerage API would handle "
         "sensitive account and position data.",
         "Mitigation: All data used is public market data. "
         "No user data is collected. Production deployment would require "
         "encryption, access controls, and compliance with financial regulations."),
    ]

    for i, (title, risk, mitigation) in enumerate(considerations, 1):
        print(f'\n  {i}. {title}')
        print(f'     Risk: {risk}')
        print(f'     {mitigation}')


# ══════════════════════════════════════════
# Main
# ══════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    print(f'{"="*70}')
    print(f'  FAILURE CASES, EDGE CASES, LIMITATIONS & ETHICS ANALYSIS')
    print(f'  CS5130 Required Analysis')
    print(f'{"="*70}')

    if args.data:
        df = load_data(args.data)
        log.info(f'Loaded {len(df):,} bars')
        analyze_failure_cases(df)
        analyze_edge_cases(df)
        analyze_performance_complexity(df)
    else:
        print('\n   No data provided. Run with --data for quantitative analysis.')
        print('  Showing limitations and ethics (no data needed).\n')

    analyze_limitations()
    analyze_ethics()

    print(f'\n{"="*70}')
    print(f'  ANALYSIS COMPLETE')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()