"""
Adaptive execution policies that beat TWAP.

The key insight: TWAP ignores market conditions. It trades the same amount
whether volatility is 10% or 100%, whether volume is 2x average or 0.1x.

An adaptive policy reads the FEATURES we computed and adjusts:
  - HIGH volume → trade MORE (lower impact per unit)
  - HIGH volatility → trade FASTER (more timing risk from waiting)
  - HIGH spread → trade LESS (expensive to cross)
  - Near deadline with inventory → MUST trade (urgency override)

This is exactly what execution desks at Citadel/Two Sigma do -
adaptive participation based on real-time market signals.

Why not just RL? On ultra-liquid BTCUSDT, the cost difference between
TWAP and optimal is ~3-5 bps. That's too small a signal for DQN to
learn reliably from scratch. But a feature-based heuristic + RL
fine-tuning can capture it.
"""

import numpy as np


class AdaptivePolicy:
    """Feature-based adaptive execution policy.

    At each bar, computes a participation rate based on:
    1. Base rate from TWAP (1/remaining_bars)
    2. Volume adjustment: trade more when volume is above average
    3. Volatility adjustment: trade faster when vol is high (reduce timing risk)
    4. Spread adjustment: trade less when spread is wide (reduce crossing cost)
    5. Urgency override: ensure completion as deadline approaches

    This combines the logic of VWAP (volume-aware) with Almgren-Chriss
    (volatility-aware) in a single adaptive framework.
    """

    def __init__(
        self,
        volume_weight: float = 0.4,
        volatility_weight: float = 0.3,
        spread_weight: float = 0.2,
        urgency_threshold: float = 0.3,  # Start urgency when 30% time left
        min_rate: float = 0.02,          # Never trade less than 2% per bar
        max_rate: float = 0.5,           # Never trade more than 50% per bar
    ):
        self.volume_weight = volume_weight
        self.volatility_weight = volatility_weight
        self.spread_weight = spread_weight
        self.urgency_threshold = urgency_threshold
        self.min_rate = min_rate
        self.max_rate = max_rate

    def get_action(self, state: dict) -> float:
        remaining_bars = state.get("remaining_bars", 1)
        remaining_qty = state.get("remaining_quantity", 0)
        total_qty = state.get("total_quantity", 1)
        time_horizon = state.get("time_horizon", 60)

        if remaining_bars <= 0 or remaining_qty <= 1e-10:
            return 0.0

        # ─── Base rate: TWAP ───
        base_rate = 1.0 / remaining_bars

        # ─── Volume adjustment ───
        # volume_imbalance > 1 means more liquid than average → trade more
        vol_imbalance = state.get("volume_imbalance", 1.0)
        if np.isnan(vol_imbalance) or vol_imbalance <= 0:
            vol_imbalance = 1.0

        # Smooth: log transform prevents extreme values from dominating
        vol_factor = 1.0 + self.volume_weight * np.log(max(vol_imbalance, 0.1))

        # ─── Volatility adjustment ───
        # Higher volatility → more timing risk → trade faster
        rolling_vol = state.get("rolling_volatility", 0)
        if np.isnan(rolling_vol) or rolling_vol <= 0:
            vol_adj = 1.0
        else:
            # Normalize: assume typical BTCUSDT annual vol ~70%
            vol_ratio = rolling_vol / 0.70
            vol_adj = 1.0 + self.volatility_weight * (vol_ratio - 1.0)
            vol_adj = np.clip(vol_adj, 0.5, 2.0)

        # ─── Spread adjustment ───
        # Higher spread → more expensive → trade less
        spread_proxy = state.get("spread_proxy", 0)
        if np.isnan(spread_proxy) or spread_proxy <= 0:
            spread_adj = 1.0
        else:
            # Normalize: typical BTCUSDT spread proxy ~0.001
            spread_ratio = spread_proxy / 0.001
            spread_adj = 1.0 - self.spread_weight * (spread_ratio - 1.0)
            spread_adj = np.clip(spread_adj, 0.3, 1.5)

        # ─── Combine adjustments ───
        adjusted_rate = base_rate * vol_factor * vol_adj * spread_adj

        # ─── Urgency override ───
        time_remaining_pct = remaining_bars / time_horizon
        inventory_pct = remaining_qty / total_qty

        if time_remaining_pct < self.urgency_threshold:
            # Linearly increase to ensure completion
            urgency_multiplier = 1.0 + (1.0 - time_remaining_pct / self.urgency_threshold) * 2.0
            adjusted_rate *= urgency_multiplier

            # Hard minimum: if lots of inventory left near deadline, force trading
            if inventory_pct > 0.5 and time_remaining_pct < 0.15:
                adjusted_rate = max(adjusted_rate, 0.3)

        # ─── Clip to bounds ───
        adjusted_rate = np.clip(adjusted_rate, self.min_rate, self.max_rate)

        return float(adjusted_rate)

    def reset(self):
        pass  # Stateless policy


class ConservativeAdaptivePolicy:
    """More conservative variant that prioritizes low cost over speed.

    Trades slower than TWAP on average, but speeds up when:
    - Volume is unusually high (cheap liquidity available)
    - Deadline is approaching with remaining inventory
    """

    def __init__(self):
        self.base_participation = 0.6  # Trade at 60% of TWAP rate normally

    def get_action(self, state: dict) -> float:
        remaining_bars = state.get("remaining_bars", 1)
        remaining_qty = state.get("remaining_quantity", 0)
        total_qty = state.get("total_quantity", 1)
        time_horizon = state.get("time_horizon", 60)

        if remaining_bars <= 0 or remaining_qty <= 1e-10:
            return 0.0

        # Slower than TWAP baseline
        base_rate = (1.0 / remaining_bars) * self.base_participation

        # Volume opportunity: only trade above base when volume is high
        vol_imbalance = state.get("volume_imbalance", 1.0)
        if np.isnan(vol_imbalance):
            vol_imbalance = 1.0

        if vol_imbalance > 1.3:
            # Volume is 30%+ above average - good time to trade more
            base_rate *= (1.0 + 0.5 * (vol_imbalance - 1.3))

        # Urgency: ramp up as deadline approaches
        time_remaining_pct = remaining_bars / time_horizon
        inventory_pct = remaining_qty / total_qty

        if time_remaining_pct < 0.4 and inventory_pct > 0.3:
            # Scale up to ensure completion
            urgency = inventory_pct / time_remaining_pct
            base_rate = max(base_rate, min(urgency * 0.05, 0.4))

        return float(np.clip(base_rate, 0.01, 0.5))

    def reset(self):
        pass
