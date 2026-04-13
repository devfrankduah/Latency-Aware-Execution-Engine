"""
Market impact models for execution simulation.

THE THEORY (Almgren & Chriss, 2000):
=====================================
When you trade, you move the price against yourself. This has two components:

1. PERMANENT IMPACT (γ): Your trade permanently shifts the fair price.
   - Caused by information content of your trade (others infer you know something)
   - Proportional to total quantity traded
   - Formula: ΔS_permanent = γ · n_t
     where n_t = shares traded at time t

2. TEMPORARY IMPACT (η): Your trade temporarily pushes the execution price
   away from the fair price (you pay a premium for immediacy).
   - Caused by consuming liquidity from the order book
   - Proportional to trading RATE (how fast you're trading)
   - Formula: ΔS_temporary = η · (n_t / V_t)
     where V_t = market volume at time t

3. SPREAD COST: You pay half the bid-ask spread on every trade.
   - Formula: cost_spread = 0.5 · spread · n_t

Total execution cost per bar:
    cost_t = n_t · (spread/2 + η · n_t/V_t) + γ · n_t · S_t

WHERE:
    n_t = quantity executed at bar t
    V_t = market volume at bar t
    S_t = current price
    η   = temporary impact coefficient
    γ   = permanent impact coefficient

The key trade-off:
    Execute FAST → high temporary impact (you're consuming the order book)
    Execute SLOW → low temporary impact, but more TIMING RISK (price may drift)

This trade-off is exactly what our policies will optimize.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImpactParams:
    """Parameters for the Almgren-Chriss market impact model.

    These can be calibrated from real data (tick trades + order book)
    or set to reasonable defaults for the asset class.

    For BTCUSDT on Binance:
    - Temporary impact is small for retail-size orders (<1 BTC)
    - Becomes significant for institutional-size (>10 BTC)
    - Permanent impact is negligible for most non-informed traders
    """

    # Temporary impact: cost increases with participation rate
    # η in the A-C model. Units: price impact per unit participation rate
    temporary_impact: float = 0.1

    # Permanent impact: shifts the fair price
    # γ in the A-C model. Units: price impact per unit traded
    permanent_impact: float = 0.01

    # Half-spread in basis points (1 bps = 0.01%)
    # For BTCUSDT: typically 0.5-2 bps during normal hours
    spread_bps: float = 1.0

    # Maximum participation rate (fraction of bar volume we can consume)
    # Trading >10% of volume in a bar is extremely aggressive
    max_participation_rate: float = 0.10

    def __post_init__(self):
        assert self.temporary_impact >= 0, "Temporary impact must be non-negative"
        assert self.permanent_impact >= 0, "Permanent impact must be non-negative"
        assert self.spread_bps >= 0, "Spread must be non-negative"
        assert 0 < self.max_participation_rate <= 1.0


def compute_execution_price(
    mid_price: float,
    quantity: float,
    bar_volume: float,
    side: str,
    params: ImpactParams,
) -> tuple[float, dict]:
    """Compute the realized execution price for a child order.

    This is the core function of the simulator. Given market conditions
    and an order, it returns what price we'd actually get (after impact).

    Args:
        mid_price: Current mid/close price of the asset.
        quantity: Quantity to execute in this bar (in base asset, e.g., BTC).
        bar_volume: Total market volume in this bar.
        side: "buy" or "sell".
        params: Impact model parameters.

    Returns:
        Tuple of (execution_price, cost_breakdown_dict).

    Example:
        >>> price, costs = compute_execution_price(
        ...     mid_price=50000.0, quantity=0.5, bar_volume=100.0,
        ...     side="buy", params=ImpactParams()
        ... )
        >>> price > 50000.0  # Buying pushes price up
        True
    """
    if quantity <= 0:
        return mid_price, {
            "spread_cost": 0.0, "temporary_cost": 0.0,
            "permanent_cost": 0.0, "participation_rate": 0.0,
            "total_cost_bps": 0.0,
        }

    # Direction: +1 for buy (price goes up), -1 for sell (price goes down)
    direction = 1.0 if side == "buy" else -1.0

    # Participation rate: what fraction of this bar's volume are we?
    if bar_volume > 0:
        participation_rate = quantity / bar_volume
    else:
        # Zero volume bar - massive impact (we ARE the market)
        participation_rate = 1.0

    # Cap participation rate (enforces realistic constraint)
    actual_quantity = quantity
    if participation_rate > params.max_participation_rate:
        actual_quantity = params.max_participation_rate * bar_volume
        participation_rate = params.max_participation_rate

    # 1. Spread cost: always pay half the spread
    spread_decimal = params.spread_bps / 10_000  # Convert bps to decimal
    spread_cost_per_unit = mid_price * spread_decimal * 0.5

    # 2. Temporary impact: increases with participation rate
    # Higher participation → more aggressive → more impact
    temp_impact_per_unit = params.temporary_impact * mid_price * participation_rate

    # 3. Permanent impact: proportional to quantity
    perm_impact = params.permanent_impact * mid_price * (actual_quantity / max(bar_volume, 1e-10))

    # Execution price = mid + direction * (spread + temporary impact)
    execution_price = mid_price + direction * (spread_cost_per_unit + temp_impact_per_unit)

    # Cost breakdown (in USDT per unit)
    total_cost_per_unit = spread_cost_per_unit + temp_impact_per_unit
    total_cost_bps = (total_cost_per_unit / mid_price) * 10_000

    cost_breakdown = {
        "spread_cost": spread_cost_per_unit * actual_quantity,
        "temporary_cost": temp_impact_per_unit * actual_quantity,
        "permanent_cost": perm_impact * actual_quantity,
        "participation_rate": participation_rate,
        "total_cost_bps": total_cost_bps,
        "actual_quantity": actual_quantity,
    }

    return execution_price, cost_breakdown


def compute_almgren_chriss_trajectory(
    total_quantity: float,
    n_steps: int,
    risk_aversion: float,
    volatility: float,
    params: ImpactParams,
) -> np.ndarray:
    """Compute the optimal execution trajectory using Almgren-Chriss.

    THE KEY RESULT of the A-C paper:
    Given a risk-aversion parameter λ, the optimal strategy is to
    execute according to a specific schedule that balances:
    - Expected cost (minimized by trading slowly, like TWAP)
    - Cost variance / timing risk (minimized by trading quickly)

    The solution is:
        n_j = (2 * sinh(κ/2) / sinh(κ*T)) * cosh(κ * (T - t_j - 0.5))

    where κ = sqrt(λ * σ² / η) controls the shape:
        κ → 0 (low risk aversion): TWAP (uniform)
        κ → ∞ (high risk aversion): immediate execution (front-loaded)

    Args:
        total_quantity: Total amount to execute.
        n_steps: Number of time steps.
        risk_aversion: λ parameter (higher = more front-loaded).
        volatility: Per-step price volatility (σ).
        params: Impact parameters.

    Returns:
        Array of shape (n_steps,) with quantity to execute at each step.
        Sums to total_quantity.
    """
    if n_steps <= 0:
        return np.array([])

    if n_steps == 1:
        return np.array([total_quantity])

    eta = params.temporary_impact
    if eta <= 0 or volatility <= 0 or risk_aversion <= 0:
        # Degenerate case → TWAP
        return np.full(n_steps, total_quantity / n_steps)

    # κ (kappa) controls the shape of the trajectory
    # Higher κ → more front-loaded (trade faster at the start)
    kappa_sq = risk_aversion * (volatility ** 2) / eta
    kappa = np.sqrt(max(kappa_sq, 1e-10))

    T = n_steps  # Total time steps

    # Optimal trade list: n_j for j = 0, 1, ..., T-1
    # n_j = (2 * sinh(κ/2)) / sinh(κ*T) * cosh(κ * (T - j - 0.5))
    numerator = 2.0 * np.sinh(kappa / 2.0)
    denominator = np.sinh(kappa * T)

    if abs(denominator) < 1e-15:
        # Numerical stability: if kappa*T is huge, everything front-loaded
        trajectory = np.zeros(n_steps)
        trajectory[0] = total_quantity
        return trajectory

    j = np.arange(n_steps, dtype=np.float64)
    trajectory = (numerator / denominator) * np.cosh(kappa * (T - j - 0.5))

    # Normalize to sum to total_quantity
    trajectory = trajectory / trajectory.sum() * total_quantity

    # Ensure non-negative (numerical precision)
    trajectory = np.maximum(trajectory, 0)

    return trajectory
