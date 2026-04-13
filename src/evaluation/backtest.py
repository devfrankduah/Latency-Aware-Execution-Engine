"""
Transaction Cost Analysis (TCA) evaluation module.

This is the REPORTING ENGINE that turns raw simulation results into
the numbers and plots you'd present to a quant interviewer or include
in a research paper.

KEY METRICS (what every execution desk tracks):
  1. Implementation Shortfall (IS) - primary metric, cost vs arrival price
  2. VWAP Slippage - cost vs market VWAP
  3. Execution Cost Variance - timing risk (spread of IS across runs)
  4. Participation Rate - how much of market volume we consumed

METHODOLOGY:
  We don't just run one simulation. We run THOUSANDS of simulations
  across random start times to get a distribution of costs per strategy.
  This is called Monte Carlo backtesting.

  Why? Because a single simulation could be lucky/unlucky. Running 1000+
  gives us confidence intervals and statistical significance tests.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.schemas import KlineSchema
from src.features.engine import compute_all_features
from src.simulator.engine import Order, ExecutionResult, simulate_execution
from src.simulator.impact import ImpactParams

logger = logging.getLogger(__name__)


# ============================================================
# Aggregated results across many simulations
# ============================================================

@dataclass
class StrategyStats:
    """Aggregated statistics for one strategy across many simulations."""

    name: str
    n_simulations: int

    # Implementation Shortfall (bps)
    is_mean: float
    is_median: float
    is_std: float
    is_p5: float      # 5th percentile (best case)
    is_p95: float      # 95th percentile (worst case)

    # Cost ($)
    cost_mean: float
    cost_std: float
    cost_total: float

    # VWAP Slippage (bps)
    vwap_mean: float
    vwap_std: float

    # Participation
    avg_child_orders: float
    avg_participation_rate: float

    # Fill rate
    fill_rate: float   # Fraction of runs that fully completed

    def __str__(self) -> str:
        return (
            f"  {self.name:<20s} │ "
            f"IS: {self.is_mean:+7.2f} ± {self.is_std:5.2f} bps │ "
            f"Cost: ${self.cost_mean:8.2f} │ "
            f"VWAP: {self.vwap_mean:+7.2f} bps │ "
            f"Orders: {self.avg_child_orders:5.1f}"
        )


def compute_strategy_stats(
    name: str,
    results: list[ExecutionResult],
) -> StrategyStats:
    """Aggregate metrics from multiple simulation runs into summary stats."""
    if not results:
        return StrategyStats(
            name=name, n_simulations=0,
            is_mean=0, is_median=0, is_std=0, is_p5=0, is_p95=0,
            cost_mean=0, cost_std=0, cost_total=0,
            vwap_mean=0, vwap_std=0,
            avg_child_orders=0, avg_participation_rate=0, fill_rate=0,
        )

    is_vals = np.array([r.implementation_shortfall_bps for r in results])
    cost_vals = np.array([r.total_cost_usd for r in results])
    vwap_vals = np.array([r.vwap_slippage_bps for r in results])
    n_orders = np.array([r.n_child_orders for r in results])

    # Average participation rate across all child orders
    all_participation = []
    for r in results:
        for c in r.child_orders:
            all_participation.append(c.participation_rate)
    avg_part = np.mean(all_participation) if all_participation else 0.0

    # Fill rate: how often we executed >= 99% of the order
    fill_rate = np.mean([
        r.total_executed / r.order.total_quantity >= 0.99 for r in results
    ])

    return StrategyStats(
        name=name,
        n_simulations=len(results),
        is_mean=float(np.mean(is_vals)),
        is_median=float(np.median(is_vals)),
        is_std=float(np.std(is_vals)),
        is_p5=float(np.percentile(is_vals, 5)),
        is_p95=float(np.percentile(is_vals, 95)),
        cost_mean=float(np.mean(cost_vals)),
        cost_std=float(np.std(cost_vals)),
        cost_total=float(np.sum(cost_vals)),
        vwap_mean=float(np.mean(vwap_vals)),
        vwap_std=float(np.std(vwap_vals)),
        avg_child_orders=float(np.mean(n_orders)),
        avg_participation_rate=avg_part,
        fill_rate=fill_rate,
    )


# ============================================================
# Monte Carlo Backtester
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    policies: dict,
    order_template: Order,
    n_simulations: int = 500,
    impact_params: ImpactParams | None = None,
    seed: int = 42,
    warmup_bars: int = 50,
) -> dict[str, StrategyStats]:
    """Run Monte Carlo backtest across random start times for all policies.

    This is the MAIN EVALUATION FUNCTION. It:
    1. Picks N random start times from the data
    2. Runs each policy at each start time
    3. Aggregates results into summary statistics

    The random starts ensure we test across different market conditions:
    - High volatility periods
    - Low volatility periods
    - High volume (liquid) periods
    - Low volume (illiquid) periods
    - Trending up / down / sideways

    Args:
        df: DataFrame with klines AND features already computed.
        policies: Dict of {name: policy_instance}.
        order_template: Base order (start times will vary).
        n_simulations: Number of random start times.
        impact_params: Market impact parameters.
        seed: Random seed for reproducibility.
        warmup_bars: Skip first N bars (features have NaN from rolling windows).

    Returns:
        Dict of {strategy_name: StrategyStats}.
    """
    if impact_params is None:
        impact_params = ImpactParams()

    rng = np.random.default_rng(seed)

    # Valid start range: after warmup, before end minus horizon
    max_start = len(df) - order_template.time_horizon_bars - 1
    min_start = warmup_bars

    if max_start <= min_start:
        raise ValueError(
            f"Not enough data. Need at least {warmup_bars + order_template.time_horizon_bars} "
            f"bars, have {len(df)}."
        )

    # Generate random start indices
    start_indices = rng.integers(min_start, max_start, size=n_simulations)

    logger.info(
        f"Running backtest: {n_simulations} simulations × {len(policies)} policies "
        f"= {n_simulations * len(policies)} total runs"
    )

    # Run simulations for each policy
    all_stats = {}
    for policy_name, policy_factory in policies.items():
        results = []

        for i, start_idx in enumerate(start_indices):
            # Create a fresh policy instance for each simulation
            if callable(policy_factory) and not hasattr(policy_factory, 'get_action'):
                policy = policy_factory()
            else:
                policy = policy_factory
                # Reset A-C policies between runs
                if hasattr(policy, 'reset'):
                    policy.reset()

            try:
                result = simulate_execution(
                    df=df,
                    order=order_template,
                    policy=policy,
                    start_idx=int(start_idx),
                    impact_params=impact_params,
                )
                results.append(result)
            except Exception as e:
                logger.debug(f"Simulation {i} failed for {policy_name}: {e}")

        stats = compute_strategy_stats(policy_name, results)
        all_stats[policy_name] = stats

        logger.info(
            f"  {policy_name}: IS = {stats.is_mean:+.2f} ± {stats.is_std:.2f} bps "
            f"({stats.n_simulations} runs)"
        )

    return all_stats


# ============================================================
# Regime-Aware Backtesting
# ============================================================

def classify_regimes(
    df: pd.DataFrame,
    vol_window: int = 1440,     # 1 day in minutes
    vol_threshold_high: float = 0.75,  # percentile
    vol_threshold_low: float = 0.25,
) -> pd.Series:
    """Classify each bar into a volatility regime.

    Returns a Series with values: "high_vol", "low_vol", "normal".
    This lets us test strategies across market conditions.
    """
    from src.features.engine import FeatureCols

    if FeatureCols.ROLLING_VOL not in df.columns:
        raise ValueError("Features not computed. Call compute_all_features first.")

    vol = df[FeatureCols.ROLLING_VOL].dropna()
    high_thresh = vol.quantile(vol_threshold_high)
    low_thresh = vol.quantile(vol_threshold_low)

    regimes = pd.Series("normal", index=df.index)
    regimes[df[FeatureCols.ROLLING_VOL] >= high_thresh] = "high_vol"
    regimes[df[FeatureCols.ROLLING_VOL] <= low_thresh] = "low_vol"

    return regimes


def run_regime_backtest(
    df: pd.DataFrame,
    policies: dict,
    order_template: Order,
    n_per_regime: int = 200,
    impact_params: ImpactParams | None = None,
    seed: int = 42,
    warmup_bars: int = 50,
) -> dict[str, dict[str, StrategyStats]]:
    """Run backtests separately for each volatility regime.

    Returns:
        Dict of {regime: {strategy: StrategyStats}}.
        E.g., results["high_vol"]["TWAP"] gives TWAP stats during volatile periods.
    """
    if impact_params is None:
        impact_params = ImpactParams()

    regimes = classify_regimes(df)
    rng = np.random.default_rng(seed)

    horizon = order_template.time_horizon_bars
    regime_results = {}

    for regime_name in ["low_vol", "normal", "high_vol"]:
        # Find valid start indices for this regime
        regime_mask = regimes == regime_name
        valid_indices = regime_mask[regime_mask].index.tolist()

        # Filter to valid range
        valid_indices = [
            i for i in valid_indices
            if warmup_bars <= i <= len(df) - horizon - 1
        ]

        if len(valid_indices) < 20:
            logger.warning(f"Too few valid starts for regime '{regime_name}': {len(valid_indices)}")
            continue

        # Sample start indices
        n_sims = min(n_per_regime, len(valid_indices))
        chosen = rng.choice(valid_indices, size=n_sims, replace=False)

        logger.info(f"\nRegime: {regime_name} ({n_sims} simulations)")

        policy_stats = {}
        for policy_name, policy_factory in policies.items():
            results = []
            for start_idx in chosen:
                if callable(policy_factory) and not hasattr(policy_factory, 'get_action'):
                    policy = policy_factory()
                else:
                    policy = policy_factory
                    if hasattr(policy, 'reset'):
                        policy.reset()

                try:
                    result = simulate_execution(
                        df=df, order=order_template, policy=policy,
                        start_idx=int(start_idx), impact_params=impact_params,
                    )
                    results.append(result)
                except Exception:
                    pass

            stats = compute_strategy_stats(policy_name, results)
            policy_stats[policy_name] = stats

        regime_results[regime_name] = policy_stats

    return regime_results


# ============================================================
# Report Formatting
# ============================================================

def print_comparison_table(all_stats: dict[str, StrategyStats]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  {'Strategy':<20s} │ {'IS Mean':>8s} {'IS Std':>8s} │ "
          f"{'Cost Mean':>10s} │ {'VWAP':>8s} │ {'Orders':>7s} │ {'Fill%':>6s}")
    print(f"  {'─'*20}─┼{'─'*18}─┼{'─'*11}─┼{'─'*9}─┼{'─'*8}─┼{'─'*7}")

    for name, stats in all_stats.items():
        print(
            f"  {name:<20s} │ {stats.is_mean:>+8.2f} {stats.is_std:>8.2f} │ "
            f"${stats.cost_mean:>9.2f} │ {stats.vwap_mean:>+8.2f} │ "
            f"{stats.avg_child_orders:>7.1f} │ {stats.fill_rate:>5.1%}"
        )

    print(f"{'='*90}\n")


def print_regime_table(regime_results: dict[str, dict[str, StrategyStats]]) -> None:
    """Print regime-specific comparison."""
    for regime, stats_dict in regime_results.items():
        emoji = ""
        print(f"\n  {emoji} REGIME: {regime.upper()}")
        print(f"  {'─'*70}")
        print(f"  {'Strategy':<20s} │ {'IS Mean':>8s} {'IS Std':>8s} │ "
              f"{'Cost':>10s} │ {'N':>5s}")
        print(f"  {'─'*20}─┼{'─'*18}─┼{'─'*11}─┼{'─'*6}")

        for name, stats in stats_dict.items():
            print(
                f"  {name:<20s} │ {stats.is_mean:>+8.2f} {stats.is_std:>8.2f} │ "
                f"${stats.cost_mean:>9.2f} │ {stats.n_simulations:>5d}"
            )


def generate_results_dataframe(
    all_stats: dict[str, StrategyStats],
) -> pd.DataFrame:
    """Convert results to a DataFrame for further analysis or export."""
    rows = []
    for name, stats in all_stats.items():
        rows.append({
            "strategy": name,
            "n_simulations": stats.n_simulations,
            "is_mean_bps": stats.is_mean,
            "is_median_bps": stats.is_median,
            "is_std_bps": stats.is_std,
            "is_p5_bps": stats.is_p5,
            "is_p95_bps": stats.is_p95,
            "cost_mean_usd": stats.cost_mean,
            "cost_std_usd": stats.cost_std,
            "vwap_mean_bps": stats.vwap_mean,
            "avg_child_orders": stats.avg_child_orders,
            "fill_rate": stats.fill_rate,
        })
    return pd.DataFrame(rows)
