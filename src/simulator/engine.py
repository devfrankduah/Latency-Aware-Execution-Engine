"""
Core execution simulator.

This is the ENGINE of the entire project. It takes:
  1. Historical market data (klines with features)
  2. An order to execute (symbol, side, quantity, time horizon)
  3. A policy (strategy that decides HOW to slice the order)

And produces:
  - A detailed execution log (every child order with its fill price)
  - Performance metrics (implementation shortfall, slippage, etc.)

ARCHITECTURE:
  The simulator is POLICY-AGNOSTIC. It doesn't know or care what
  strategy is being used. The policy just returns "trade X quantity
  this bar" and the simulator handles the market impact math.

  This clean separation means:
  - Adding a new strategy = just implement one function
  - Backtesting is reproducible (same data + same policy = same result)
  - You can A/B test strategies trivially
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.schemas import KlineSchema
from src.features.engine import FeatureCols
from src.simulator.impact import ImpactParams, compute_execution_price

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================

@dataclass
class Order:
    """An order to execute.

    This represents the PARENT order - the total quantity we need
    to execute over a time horizon. The policy will slice this into
    CHILD orders executed at each bar.
    """
    symbol: str             # e.g., "BTCUSDT"
    side: str               # "buy" or "sell"
    total_quantity: float   # Total quantity in base asset (e.g., 1.0 BTC)
    time_horizon_bars: int  # Number of bars to execute over (e.g., 60 = 1 hour)

    def __post_init__(self):
        assert self.side in ("buy", "sell"), f"Side must be 'buy' or 'sell', got {self.side}"
        assert self.total_quantity > 0, "Quantity must be positive"
        assert self.time_horizon_bars > 0, "Time horizon must be positive"


@dataclass
class ChildOrder:
    """A single child order (one slice of the parent order)."""
    bar_index: int          # Index into the kline DataFrame
    timestamp: pd.Timestamp
    quantity: float         # Quantity actually executed
    mid_price: float        # Market mid price at execution
    exec_price: float       # Realized execution price (after impact)
    spread_cost: float
    temporary_cost: float
    permanent_cost: float
    participation_rate: float
    remaining_quantity: float  # How much is left after this fill


@dataclass
class ExecutionResult:
    """Complete result of simulating an order execution."""
    order: Order
    child_orders: list[ChildOrder]
    arrival_price: float        # Price when the order was placed (decision price)
    final_price: float          # Price at the end of execution
    total_executed: float       # Total quantity actually filled
    avg_exec_price: float       # Volume-weighted average execution price
    implementation_shortfall_bps: float  # Primary metric
    vwap_slippage_bps: float
    total_cost_usd: float       # Total $ cost of execution
    cost_breakdown: dict        # Spread vs. impact vs. timing
    n_child_orders: int
    execution_time_bars: int    # How many bars it actually took

    def summary(self) -> str:
        """Human-readable summary of execution results."""
        return (
            f"\n{'─'*55}\n"
            f"  EXECUTION RESULT - {self.order.symbol} {self.order.side.upper()}\n"
            f"{'─'*55}\n"
            f"  Order:            {self.order.total_quantity:.4f} over "
            f"{self.order.time_horizon_bars} bars\n"
            f"  Executed:         {self.total_executed:.4f} "
            f"({self.total_executed/self.order.total_quantity:.1%})\n"
            f"  Arrival price:    ${self.arrival_price:,.2f}\n"
            f"  Avg exec price:   ${self.avg_exec_price:,.2f}\n"
            f"  Final price:      ${self.final_price:,.2f}\n"
            f"  ────────────────────────────────\n"
            f"  Impl. Shortfall:  {self.implementation_shortfall_bps:+.2f} bps\n"
            f"  VWAP Slippage:    {self.vwap_slippage_bps:+.2f} bps\n"
            f"  Total Cost:       ${self.total_cost_usd:,.2f}\n"
            f"  Child Orders:     {self.n_child_orders}\n"
            f"  ────────────────────────────────\n"
            f"  Cost Breakdown:\n"
            f"    Spread:         ${self.cost_breakdown['spread']:,.2f}\n"
            f"    Temp Impact:    ${self.cost_breakdown['temporary']:,.2f}\n"
            f"    Perm Impact:    ${self.cost_breakdown['permanent']:,.2f}\n"
            f"{'─'*55}\n"
        )


# ============================================================
# Policy Interface (Protocol)
# ============================================================

class ExecutionPolicy(Protocol):
    """Interface that all execution policies must implement.

    PRODUCTION PATTERN: Protocol (structural typing).
    Any class with a `get_action` method matching this signature works.
    No need to inherit from a base class.
    """

    def get_action(
        self,
        state: dict,
    ) -> float:
        """Decide what fraction of remaining quantity to execute this bar.

        Args:
            state: Dictionary containing:
                - remaining_quantity: How much is left to execute
                - remaining_bars: How many bars left in the horizon
                - current_bar: Current bar data (price, volume, features)
                - bar_index: Index within execution window (0-based)
                - total_quantity: Original total order quantity
                - time_horizon: Total time horizon in bars

        Returns:
            Fraction of REMAINING quantity to execute this bar.
            Must be in [0.0, 1.0].
            0.0 = don't trade this bar
            1.0 = execute everything remaining
        """
        ...


# ============================================================
# The Simulator
# ============================================================

def simulate_execution(
    df: pd.DataFrame,
    order: Order,
    policy: ExecutionPolicy,
    start_idx: int,
    impact_params: ImpactParams | None = None,
) -> ExecutionResult:
    """Simulate executing an order through historical data using a given policy.

    This is the MAIN FUNCTION of the simulator. It:
    1. Steps through the data bar-by-bar
    2. Asks the policy how much to trade each bar
    3. Applies the market impact model to get fill prices
    4. Records everything in a detailed execution log
    5. Computes performance metrics

    Args:
        df: DataFrame with kline data AND features (call compute_all_features first).
        order: The parent order to execute.
        policy: The execution policy to use.
        start_idx: Index in df where execution begins.
        impact_params: Market impact parameters (uses defaults if None).

    Returns:
        ExecutionResult with full execution details.
    """
    if impact_params is None:
        impact_params = ImpactParams()

    # Validate we have enough data
    end_idx = start_idx + order.time_horizon_bars
    if end_idx > len(df):
        end_idx = len(df)
        logger.warning(
            f"Not enough data for full horizon. "
            f"Requested {order.time_horizon_bars} bars from idx {start_idx}, "
            f"but only {len(df) - start_idx} available."
        )

    # Arrival price = price at the moment the order is placed
    arrival_price = df.iloc[start_idx][KlineSchema.CLOSE]

    # Compute market VWAP over the execution window (benchmark)
    window = df.iloc[start_idx:end_idx]
    typical_price = (window[KlineSchema.HIGH] + window[KlineSchema.LOW] +
                     window[KlineSchema.CLOSE]) / 3
    window_volume = window[KlineSchema.VOLUME]

    if window_volume.sum() > 0:
        market_vwap = (typical_price * window_volume).sum() / window_volume.sum()
    else:
        market_vwap = arrival_price

    # Execute bar by bar
    child_orders = []
    remaining = order.total_quantity
    total_spread_cost = 0.0
    total_temp_cost = 0.0
    total_perm_cost = 0.0

    actual_horizon = end_idx - start_idx

    for step in range(actual_horizon):
        if remaining <= 1e-10:
            break  # Order fully filled

        bar_idx = start_idx + step
        bar = df.iloc[bar_idx]

        # Build state for the policy
        state = {
            "remaining_quantity": remaining,
            "remaining_bars": actual_horizon - step,
            "bar_index": step,
            "total_quantity": order.total_quantity,
            "time_horizon": actual_horizon,
            "mid_price": bar[KlineSchema.CLOSE],
            "bar_volume": bar[KlineSchema.VOLUME],
            "bar_high": bar[KlineSchema.HIGH],
            "bar_low": bar[KlineSchema.LOW],
        }

        # Add feature columns if they exist
        for feat in [FeatureCols.ROLLING_VOL, FeatureCols.VOLUME_IMBALANCE,
                     FeatureCols.SPREAD_PROXY, FeatureCols.LOG_RETURN,
                     FeatureCols.RETURN_5, FeatureCols.VOLUME_ZSCORE]:
            if feat in df.columns:
                val = bar[feat]
                state[feat] = val if not pd.isna(val) else 0.0

        # Ask policy: what fraction of remaining should we trade?
        action = policy.get_action(state)
        action = float(np.clip(action, 0.0, 1.0))

        # Convert fraction to quantity
        quantity_to_trade = remaining * action

        if quantity_to_trade < 1e-10:
            continue  # Policy decided not to trade this bar

        # Apply market impact model
        exec_price, costs = compute_execution_price(
            mid_price=bar[KlineSchema.CLOSE],
            quantity=quantity_to_trade,
            bar_volume=bar[KlineSchema.VOLUME],
            side=order.side,
            params=impact_params,
        )

        actual_qty = costs["actual_quantity"]
        remaining -= actual_qty

        # Record child order
        child = ChildOrder(
            bar_index=bar_idx,
            timestamp=bar[KlineSchema.TIMESTAMP],
            quantity=actual_qty,
            mid_price=bar[KlineSchema.CLOSE],
            exec_price=exec_price,
            spread_cost=costs["spread_cost"],
            temporary_cost=costs["temporary_cost"],
            permanent_cost=costs["permanent_cost"],
            participation_rate=costs["participation_rate"],
            remaining_quantity=remaining,
        )
        child_orders.append(child)

        total_spread_cost += costs["spread_cost"]
        total_temp_cost += costs["temporary_cost"]
        total_perm_cost += costs["permanent_cost"]

    # Compute results
    total_executed = order.total_quantity - remaining
    final_price = df.iloc[min(end_idx - 1, len(df) - 1)][KlineSchema.CLOSE]

    # Volume-weighted average execution price
    if child_orders:
        total_value = sum(c.exec_price * c.quantity for c in child_orders)
        avg_exec_price = total_value / total_executed if total_executed > 0 else arrival_price
    else:
        avg_exec_price = arrival_price

    # Implementation Shortfall (in basis points)
    # IS = (avg_exec_price - arrival_price) / arrival_price × 10000
    # For buys: positive IS means you paid MORE than arrival (bad)
    # For sells: negative IS means you sold LOWER than arrival (bad)
    direction = 1.0 if order.side == "buy" else -1.0
    is_bps = direction * (avg_exec_price - arrival_price) / arrival_price * 10_000

    # VWAP Slippage (in basis points)
    vwap_slippage_bps = direction * (avg_exec_price - market_vwap) / market_vwap * 10_000

    total_cost = total_spread_cost + total_temp_cost + total_perm_cost

    return ExecutionResult(
        order=order,
        child_orders=child_orders,
        arrival_price=arrival_price,
        final_price=final_price,
        total_executed=total_executed,
        avg_exec_price=avg_exec_price,
        implementation_shortfall_bps=is_bps,
        vwap_slippage_bps=vwap_slippage_bps,
        total_cost_usd=total_cost,
        cost_breakdown={
            "spread": total_spread_cost,
            "temporary": total_temp_cost,
            "permanent": total_perm_cost,
        },
        n_child_orders=len(child_orders),
        execution_time_bars=len(child_orders),
    )
