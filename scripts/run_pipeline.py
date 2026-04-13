#!/usr/bin/env python3
"""
End-to-end pipeline - runs on new unseen data without manual intervention.

CS5130 requirement: "The system should be able to run on new, unseen data
without manual intervention."

This script takes raw kline CSV/Parquet → features → baselines → evaluation
in a single command. No configuration needed.

Usage:
    # Run on any processed data file
    python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet

    # Run on raw CSV (handles feature computation automatically)
    python scripts/run_pipeline.py --data data/raw/klines/BTCUSDT/BTCUSDT-1m-2024-01.csv

    # Run with custom order size
    python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet --qty 50

    # Full analysis including failure cases and ethics
    python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet --full
"""

import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Latency-Aware Execution Engine - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet
  python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet --qty 50 --full
        """)
    parser.add_argument('--data', required=True, help='Path to kline data (parquet or csv)')
    parser.add_argument('--qty', type=float, default=1.0, help='Order quantity in BTC (default: 1.0)')
    parser.add_argument('--horizon', type=int, default=60, help='Execution horizon in bars (default: 60)')
    parser.add_argument('--n-sims', type=int, default=500, help='Monte Carlo simulations (default: 500)')
    parser.add_argument('--full', action='store_true', help='Include failure/edge/ethics analysis')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    args = parser.parse_args()

    t_start = time.time()

    print(f'\n{"="*70}')
    print(f'  LATENCY-AWARE EXECUTION ENGINE - FULL PIPELINE')
    print(f'{"="*70}')

    # ── Step 1: Load Data ──
    print(f'\n  STEP 1: Loading data...')
    path = Path(args.data)
    if not path.exists():
        print(f'  File not found: {path}')
        sys.exit(1)

    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        # Try loading as Binance CSV (no header)
        try:
            cols = ['open_time','open','high','low','close','volume',
                    'close_time','quote_volume','num_trades',
                    'taker_buy_vol','taker_buy_quote','ignore']
            df = pd.read_csv(path, header=None, names=cols)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['symbol'] = path.stem.split('-')[0]
        except Exception:
            df = pd.read_csv(path, parse_dates=['timestamp'])

    print(f'  Loaded {len(df):,} bars')
    print(f'  Date range: {df["timestamp"].min()} → {df["timestamp"].max()}')
    print(f'  Price: ${df["close"].min():,.2f} → ${df["close"].max():,.2f}')

    # ── Step 2: Validate Data ──
    print(f'\n  STEP 2: Validating data...')
    from src.data.validator import validate_klines
    report = validate_klines(df)
    print(report)
    if not report.is_valid:
        print('   Validation issues found (continuing anyway)')

    # ── Step 3: Compute Features ──
    print(f'  STEP 3: Computing features...')
    from src.features.engine import compute_all_features
    df = compute_all_features(df)
    print(f'  Features computed: {len(df.columns)} columns')

    # ── Step 4: Run Baseline Strategies ──
    print(f'\n  STEP 4: Running baseline strategies ({args.n_sims} simulations)...')
    from src.simulator.engine import Order, simulate_execution
    from src.simulator.impact import ImpactParams
    from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy, AlmgrenChrissPolicy

    order = Order(symbol=df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTCUSDT',
                  side='buy', total_quantity=args.qty, time_horizon_bars=args.horizon)
    params = ImpactParams()

    policies = {
        'Immediate': lambda: ImmediatePolicy(),
        'TWAP': lambda: TWAPPolicy(),
        'VWAP': lambda: VWAPPolicy(),
        'A-C (λ=0.1)': lambda: AlmgrenChrissPolicy(risk_aversion=0.1),
        'A-C (λ=1.0)': lambda: AlmgrenChrissPolicy(risk_aversion=1.0),
        'A-C (λ=5.0)': lambda: AlmgrenChrissPolicy(risk_aversion=5.0),
    }

    rng = np.random.default_rng(42)
    starts = rng.integers(100, len(df) - args.horizon - 1, size=args.n_sims)

    results = {}
    for name, factory in policies.items():
        is_vals, costs, fills = [], [], []
        for si in starts:
            policy = factory()
            if hasattr(policy, 'reset'): policy.reset()
            try:
                r = simulate_execution(df, order, policy, int(si), params)
                is_vals.append(r.implementation_shortfall_bps)
                costs.append(r.total_cost_usd)
                fills.append(r.total_executed / order.total_quantity)
            except Exception:
                pass
        results[name] = {
            'is_mean': np.mean(is_vals), 'is_std': np.std(is_vals),
            'cost_mean': np.mean(costs), 'cost_std': np.std(costs),
            'fill': np.mean(fills), 'n': len(is_vals),
        }

    # ── Step 5: Display Results ──
    print(f'\n{"="*80}')
    print(f'  RESULTS - {args.qty} BTC over {args.horizon} min | {args.n_sims} simulations')
    print(f'{"="*80}')
    print(f'  {"Strategy":<16s} │ {"IS Mean":>9s} {"IS Std":>9s} │ {"Cost":>10s} │ {"Fill":>6s} │ {"vs TWAP":>10s}')
    print(f'  {"─"*16}─┼{"─"*20}─┼{"─"*11}─┼{"─"*7}─┼{"─"*11}')

    twap_cost = results.get('TWAP', {}).get('cost_mean', 1)
    for name, r in results.items():
        vs_twap = (1 - r['cost_mean'] / twap_cost) * 100 if twap_cost > 0 else 0
        print(f'  {name:<16s} │ {r["is_mean"]:>+9.2f} {r["is_std"]:>9.2f} │ '
              f'${r["cost_mean"]:>9.2f} │ {r["fill"]:>5.1%} │ {vs_twap:>+9.1f}%')

    best = min(results.items(), key=lambda x: x[1]['cost_mean'])
    worst = max(results.items(), key=lambda x: x[1]['cost_mean'])

    print(f'  {"─"*80}')
    print(f'  Best:  {best[0]} (${best[1]["cost_mean"]:.2f}/order)')
    print(f'  Worst: {worst[0]} (${worst[1]["cost_mean"]:.2f}/order)')
    print(f'  Cost reduction (best vs worst): {(1-best[1]["cost_mean"]/worst[1]["cost_mean"])*100:.1f}%')
    print(f'{"="*80}')

    # ── Step 6: Save Results ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{'strategy': k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(out_dir / 'pipeline_results.csv', index=False)
    print(f'\n  Results saved to {out_dir}/pipeline_results.csv')

    # ── Step 7 (optional): Full analysis ──
    if args.full:
        print(f'\n  Running full analysis (failure cases, edge cases, ethics)...\n')
        from scripts.analysis import (
            analyze_failure_cases, analyze_edge_cases,
            analyze_limitations, analyze_ethics, analyze_performance_complexity
        )
        analyze_failure_cases(df)
        analyze_edge_cases(df)
        analyze_performance_complexity(df)
        analyze_limitations()
        analyze_ethics()

    elapsed = time.time() - t_start
    print(f'\n  Pipeline complete in {elapsed:.1f}s')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()