#!/usr/bin/env python3
"""
Run full backtest on real or synthetic data.

This is THE script that produces publishable results.
It runs all strategies across hundreds of random start times,
computes TCA metrics, and generates comparison tables.

Usage:
    # On real data (after downloading):
    python scripts/run_backtest.py --data data/processed/BTCUSDT_klines_1m.parquet

    # On synthetic data (no download needed):
    python scripts/run_backtest.py --synthetic

    # With regime analysis:
    python scripts/run_backtest.py --data data/processed/BTCUSDT_klines_1m.parquet --regimes

    # Quick test (fewer simulations):
    python scripts/run_backtest.py --synthetic --n-sims 50

Author: Nikhilesh
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.schemas import KlineSchema
from src.features.engine import compute_all_features
from src.simulator.engine import Order
from src.simulator.impact import ImpactParams
from src.policies.baselines import (
    ImmediatePolicy, TWAPPolicy, VWAPPolicy, AlmgrenChrissPolicy,
)
from src.evaluation.backtest import (
    run_backtest, run_regime_backtest,
    print_comparison_table, print_regime_table,
    generate_results_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_synthetic_data(n_bars: int = 100_000) -> pd.DataFrame:
    """Generate synthetic data for testing without real data."""
    logger.info(f"Generating {n_bars:,} bars of synthetic BTCUSDT data...")
    rng = np.random.default_rng(42)

    log_ret = rng.normal(0, 0.0008, n_bars)
    close = 40000.0 * np.exp(np.cumsum(log_ret))
    noise = rng.uniform(0.0001, 0.001, n_bars)

    df = pd.DataFrame({
        KlineSchema.TIMESTAMP: pd.date_range(
            "2023-01-01", periods=n_bars, freq="1min", tz="UTC"
        ),
        KlineSchema.OPEN: np.roll(close, 1),
        KlineSchema.HIGH: close * (1 + noise),
        KlineSchema.LOW: close * (1 - noise),
        KlineSchema.CLOSE: close,
        KlineSchema.VOLUME: rng.exponential(50.0, n_bars),
        KlineSchema.QUOTE_VOLUME: rng.exponential(50.0, n_bars) * close,
        KlineSchema.TRADES: rng.integers(100, 2000, n_bars),
        KlineSchema.SYMBOL: "BTCUSDT",
    })
    df.iloc[0, df.columns.get_loc(KlineSchema.OPEN)] = 40000.0
    df[KlineSchema.HIGH] = df[[KlineSchema.OPEN, KlineSchema.HIGH, KlineSchema.CLOSE]].max(axis=1)
    df[KlineSchema.LOW] = df[[KlineSchema.OPEN, KlineSchema.LOW, KlineSchema.CLOSE]].min(axis=1)

    return df


def load_real_data(path: str) -> pd.DataFrame:
    """Load real processed data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run 'python scripts/download_data.py' first, then "
            f"'python scripts/validate_data.py' to process."
        )

    logger.info(f"Loading data from {path}...")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[KlineSchema.TIMESTAMP])

    logger.info(f"Loaded {len(df):,} bars")
    return df


def build_policies() -> dict:
    """Build all policies to compare.

    Returns dict of {name: factory_or_instance}.
    Factory functions are used for policies that need resetting between runs.
    """
    return {
        "Immediate": ImmediatePolicy(),
        "TWAP": TWAPPolicy(),
        "VWAP": VWAPPolicy(volume_sensitivity=1.0),
        "VWAP (aggressive)": VWAPPolicy(volume_sensitivity=2.0),
        "A-C (λ=0.01)": lambda: AlmgrenChrissPolicy(risk_aversion=0.01),
        "A-C (λ=0.1)": lambda: AlmgrenChrissPolicy(risk_aversion=0.1),
        "A-C (λ=1.0)": lambda: AlmgrenChrissPolicy(risk_aversion=1.0),
        "A-C (λ=5.0)": lambda: AlmgrenChrissPolicy(risk_aversion=5.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run execution strategy backtest")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to processed data file (.parquet or .csv)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no download needed)")
    parser.add_argument("--n-sims", type=int, default=500,
                        help="Number of simulations per strategy (default: 500)")
    parser.add_argument("--horizon", type=int, default=60,
                        help="Execution horizon in bars/minutes (default: 60)")
    parser.add_argument("--quantity", type=float, default=1.0,
                        help="Order quantity in BTC (default: 1.0)")
    parser.add_argument("--regimes", action="store_true",
                        help="Run regime-specific analysis")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results CSV to this path")
    args = parser.parse_args()

    # ─── Load data ───
    if args.synthetic or args.data is None:
        df = make_synthetic_data()
        data_source = "synthetic"
    else:
        df = load_real_data(args.data)
        data_source = Path(args.data).stem

    # ─── Compute features ───
    logger.info("Computing features...")
    t0 = time.time()
    df = compute_all_features(df)
    logger.info(f"Features computed in {time.time() - t0:.1f}s")

    # ─── Configure ───
    order = Order(
        symbol="BTCUSDT",
        side="buy",
        total_quantity=args.quantity,
        time_horizon_bars=args.horizon,
    )

    impact_params = ImpactParams(
        temporary_impact=0.1,
        permanent_impact=0.01,
        spread_bps=1.0,
        max_participation_rate=0.10,
    )

    policies = build_policies()

    # ─── Print header ───
    print(f"\n{'='*70}")
    print(f"  EXECUTION STRATEGY BACKTEST")
    print(f"{'='*70}")
    print(f"  Data:        {data_source} ({len(df):,} bars)")
    print(f"  Order:       {order.side.upper()} {order.total_quantity} BTC "
          f"over {order.time_horizon_bars} minutes")
    print(f"  Strategies:  {len(policies)}")
    print(f"  Simulations: {args.n_sims} per strategy")
    print(f"  Impact:      η={impact_params.temporary_impact}, "
          f"γ={impact_params.permanent_impact}, "
          f"spread={impact_params.spread_bps} bps")
    print(f"{'='*70}")

    # ─── Run main backtest ───
    logger.info("\nRunning main backtest...")
    t0 = time.time()

    all_stats = run_backtest(
        df=df,
        policies=policies,
        order_template=order,
        n_simulations=args.n_sims,
        impact_params=impact_params,
        seed=42,
    )

    elapsed = time.time() - t0
    total_runs = args.n_sims * len(policies)
    logger.info(f"Completed {total_runs:,} simulations in {elapsed:.1f}s "
                f"({total_runs/elapsed:.0f} sims/sec)")

    print_comparison_table(all_stats)

    # ─── Regime analysis ───
    regime_results = None
    if args.regimes:
        logger.info("\nRunning regime analysis...")
        t0 = time.time()

        regime_results = run_regime_backtest(
            df=df,
            policies=policies,
            order_template=order,
            n_per_regime=min(200, args.n_sims),
            impact_params=impact_params,
            seed=123,
        )

        print(f"\n{'='*70}")
        print(f"  REGIME ANALYSIS")
        print(f"{'='*70}")
        print_regime_table(regime_results)
        print()

        logger.info(f"Regime analysis done in {time.time() - t0:.1f}s")

    # ─── Build results DataFrame (needed for figures and CSV export) ───
    results_df = generate_results_dataframe(all_stats)

    # ─── Generate figures ───
    try:
        from src.evaluation.visualizations import (
            plot_strategy_comparison, plot_efficient_frontier,
            plot_regime_comparison,
        )

        fig_dir = f"reports/figures/{data_source}"
        logger.info(f"\nGenerating figures to {fig_dir}/...")

        # Strategy comparison bar chart
        plot_strategy_comparison(
            results_df, metric="is_mean_bps",
            title=f"Implementation Shortfall - {data_source}",
            output_dir=fig_dir,
        )

        # Cost comparison (save with different filename)
        from src.evaluation.visualizations import save_fig, _get_plt
        plt = _get_plt()

        fig, ax = plt.subplots(figsize=(12, 6))
        strategies = results_df["strategy"].tolist()
        costs = results_df["cost_mean_usd"].tolist()
        colors = ["#2ecc71" if v == min(costs) else
                  "#e74c3c" if v == max(costs) else "#3498db" for v in costs]
        bars = ax.barh(strategies, costs, color=colors, edgecolor="white")
        for bar, val in zip(bars, costs):
            ax.text(val + max(costs) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"${val:.2f}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("Average Execution Cost ($)", fontsize=12)
        ax.set_title(f"Average Execution Cost - {data_source}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "cost_comparison.png", fig_dir)

        # Efficient frontier from A-C lambda sweep
        ac_stats = {k: v for k, v in all_stats.items() if k.startswith("A-C")}
        if ac_stats:
            # Parse lambda from name "A-C (λ=X.X)"
            frontier_pts = []
            for name, stats in ac_stats.items():
                try:
                    lam = float(name.split("=")[1].rstrip(")"))
                    frontier_pts.append((lam, stats.is_mean, stats.is_std))
                except (IndexError, ValueError):
                    pass

            # Add TWAP (λ→0) and Immediate (λ→∞) as endpoints
            if "TWAP" in all_stats:
                s = all_stats["TWAP"]
                frontier_pts.insert(0, (0.001, s.is_mean, s.is_std))
            if "Immediate" in all_stats:
                s = all_stats["Immediate"]
                frontier_pts.append((100.0, s.is_mean, s.is_std))

            frontier_pts.sort(key=lambda x: x[0])
            plot_efficient_frontier(frontier_pts, output_dir=fig_dir)

        # Regime comparison
        if args.regimes and regime_results:
            plot_regime_comparison(regime_results, metric="is_mean",
                                   output_dir=fig_dir)

        logger.info("Figures generated successfully!")

    except ImportError as e:
        logger.warning(f"Could not generate figures (missing dependency): {e}")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    # ─── Save results CSV ───

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    else:
        # Default: save to reports/
        default_path = Path("reports") / f"backtest_{data_source}.csv"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(default_path, index=False)
        logger.info(f"Results saved to {default_path}")

    # ─── Key findings ───
    print(f"\n{'='*70}")
    print(f"  KEY FINDINGS")
    print(f"{'='*70}")

    # Find best and worst strategy
    best = min(all_stats.values(), key=lambda s: s.cost_mean)
    worst = max(all_stats.values(), key=lambda s: s.cost_mean)
    twap = all_stats.get("TWAP")

    print(f"  Best strategy:    {best.name} (avg cost ${best.cost_mean:.2f})")
    print(f"  Worst strategy:   {worst.name} (avg cost ${worst.cost_mean:.2f})")
    print(f"  Cost reduction:   {(1 - best.cost_mean/worst.cost_mean):.1%} "
          f"(best vs worst)")

    if twap:
        improvement = (1 - best.cost_mean / twap.cost_mean) * 100
        print(f"  vs TWAP baseline: {improvement:+.1f}% "
              f"({'better' if improvement > 0 else 'worse'})")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
