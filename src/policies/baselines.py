"""
Execution policies (strategies).

Each policy implements the same interface: given the current state
(remaining qty, time left, market conditions), return a fraction [0,1]
of remaining quantity to execute this bar.

STRATEGIES IMPLEMENTED:
  1. ImmediatePolicy    - Execute everything immediately (worst case baseline)
  2. TWAPPolicy         - Equal slices over time (standard baseline)
  3. VWAPPolicy         - Slices proportional to expected volume (smarter baseline)
  4. AlmgrenChrissPolicy - Optimal trajectory with risk-aversion parameter

A quant interviewer will expect you to explain:
  - WHY each policy makes sense in different market conditions
  - The mathematical derivation of Almgren-Chriss
  - When TWAP beats A-C and vice versa
"""

import numpy as np


class ImmediatePolicy:
    """Execute the entire order immediately in bar 0.

    This is the WORST CASE baseline - maximum market impact.
    Equivalent to Almgren-Chriss with λ → ∞ (infinite risk aversion).

    Use case: When you MUST execute NOW (margin call, forced liquidation).
    """

    def get_action(self, state: dict) -> float:
        if state["bar_index"] == 0:
            return 1.0  # Execute 100% of remaining in first bar
        return 0.0


class TWAPPolicy:
    """Time-Weighted Average Price - equal slices over time.

    The most common baseline in execution research.
    Optimal when: price follows a random walk with no volume patterns.
    Equivalent to Almgren-Chriss with λ = 0 (zero risk aversion).

    The schedule is simple:
        Each bar, execute (1 / remaining_bars) of remaining quantity.

    This naturally handles partial fills: if we couldn't execute
    in a previous bar, the remaining bars absorb the difference.
    """

    def get_action(self, state: dict) -> float:
        remaining_bars = state["remaining_bars"]
        if remaining_bars <= 0:
            return 1.0
        return 1.0 / remaining_bars


class VWAPPolicy:
    """Volume-Weighted Average Price - trade proportional to volume.

    Smarter than TWAP: trade MORE when volume is high (less impact),
    trade LESS when volume is low (avoiding moving the market).

    Uses the volume_imbalance feature:
        imbalance > 1.0 → volume above average → trade more
        imbalance < 1.0 → volume below average → trade less

    Still aims to complete by the end of the horizon (increases
    aggressiveness as time runs out).
    """

    def __init__(self, volume_sensitivity: float = 1.0):
        """
        Args:
            volume_sensitivity: How much to adjust for volume.
                0.0 = ignore volume (becomes TWAP)
                1.0 = full adjustment
                2.0 = very aggressive volume-following
        """
        self.volume_sensitivity = volume_sensitivity

    def get_action(self, state: dict) -> float:
        remaining_bars = state["remaining_bars"]
        if remaining_bars <= 0:
            return 1.0

        # Base rate = TWAP rate
        base_rate = 1.0 / remaining_bars

        # Adjust for volume conditions
        vol_imbalance = state.get("volume_imbalance", 1.0)
        if vol_imbalance == 0 or np.isnan(vol_imbalance):
            vol_imbalance = 1.0

        # Volume adjustment: trade more when volume is above average
        # Power function gives smooth, bounded adjustment
        vol_multiplier = vol_imbalance ** self.volume_sensitivity

        # Scale so we still approximately complete on time
        adjusted_rate = base_rate * vol_multiplier

        # Urgency: as we approach deadline, increase rate
        time_remaining_pct = remaining_bars / max(state["time_horizon"], 1)
        if time_remaining_pct < 0.2:
            # Last 20% of time: ensure completion
            urgency_boost = 1.0 + (1.0 - time_remaining_pct / 0.2)
            adjusted_rate *= urgency_boost

        return float(np.clip(adjusted_rate, 0.0, 1.0))


class AlmgrenChrissPolicy:
    """Optimal execution following the Almgren-Chriss (2000) trajectory.

    This precomputes the optimal schedule and follows it.

    The optimal trajectory depends on:
    - λ (risk_aversion): Higher → trade faster (front-loaded)
    - σ (volatility): Higher → more timing risk → trade faster
    - η (temporary impact): Higher → slower is cheaper

    The math:
        κ = sqrt(λ · σ² / η)
        n_j ∝ cosh(κ · (T - j - 0.5))  (front-loaded hyperbolic cosine)
        
    Special cases:
        λ → 0:  κ → 0 → TWAP (uniform schedule)
        λ → ∞:  κ → ∞ → Immediate execution

    This is the policy that a quant interviewer expects you to implement.
    """

    def __init__(
        self,
        risk_aversion: float = 0.1,
        volatility: float | None = None,
        temporary_impact: float = 0.1,
    ):
        """
        Args:
            risk_aversion: λ parameter. Try [0.01, 0.1, 0.5, 1.0, 5.0].
            volatility: Per-bar σ. If None, uses rolling vol from features.
            temporary_impact: η parameter from impact model.
        """
        self.risk_aversion = risk_aversion
        self.fixed_volatility = volatility
        self.temporary_impact = temporary_impact
        self._schedule = None
        self._schedule_computed = False

    def _compute_schedule(self, n_steps: int, volatility: float) -> np.ndarray:
        """Compute the A-C optimal schedule."""
        from src.simulator.impact import compute_almgren_chriss_trajectory, ImpactParams

        params = ImpactParams(temporary_impact=self.temporary_impact)
        schedule = compute_almgren_chriss_trajectory(
            total_quantity=1.0,  # Normalized - will scale by remaining qty
            n_steps=n_steps,
            risk_aversion=self.risk_aversion,
            volatility=volatility,
            params=params,
        )
        # Convert from absolute quantities to fraction-of-remaining
        fractions = np.zeros(n_steps)
        cumulative_remaining = 1.0
        for i in range(n_steps):
            if cumulative_remaining > 1e-10:
                fractions[i] = schedule[i] / cumulative_remaining
            else:
                fractions[i] = 0.0
            cumulative_remaining -= schedule[i]

        return np.clip(fractions, 0.0, 1.0)

    def get_action(self, state: dict) -> float:
        bar_index = state["bar_index"]
        time_horizon = state["time_horizon"]

        # Compute schedule on first call (lazy initialization)
        if not self._schedule_computed:
            vol = self.fixed_volatility
            if vol is None:
                # Use rolling volatility from features (annualized → per-bar)
                annual_vol = state.get("rolling_volatility", 0.5)
                if annual_vol == 0 or np.isnan(annual_vol):
                    annual_vol = 0.5
                # De-annualize: per-bar vol = annual / sqrt(minutes_per_year)
                vol = annual_vol / np.sqrt(525_960)

            self._schedule = self._compute_schedule(time_horizon, vol)
            self._schedule_computed = True

        if bar_index < len(self._schedule):
            return float(self._schedule[bar_index])

        # Past the schedule - execute remaining
        return 1.0

    def reset(self):
        """Reset for a new execution (recomputes schedule)."""
        self._schedule = None
        self._schedule_computed = False
