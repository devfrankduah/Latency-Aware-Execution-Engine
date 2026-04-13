"""
Gym-style environment for trade execution.

Wraps our execution simulator into a standard RL interface that any
RL algorithm (DQN, PPO, etc.) can train on.

STATE SPACE (what the agent sees at each step):
  - remaining_inventory_pct: How much is left to execute [0, 1]
  - time_remaining_pct: How much time is left [0, 1]
  - rolling_volatility: Current market volatility (normalized)
  - volume_imbalance: Current volume vs average [0, 10]
  - spread_proxy: Current spread proxy (normalized)
  - recent_return_5: 5-bar return (momentum signal)
  - recent_return_20: 20-bar return (trend signal)
  - hour_sin: Cyclical hour encoding [-1, 1]
  - hour_cos: Cyclical hour encoding [-1, 1]

ACTION SPACE (what the agent decides):
  Discrete: {0, 1, 2, 3, 4, 5} → fraction of remaining to execute
  [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]

REWARD:
  -implementation_shortfall per step (negative = good, we want to minimize cost)

WHY DISCRETE ACTIONS?
  DQN requires discrete actions. For a semester project, 6 action levels
  provide enough granularity. Production systems use continuous actions
  with PPO/SAC, but DQN + discrete is simpler and easier to debug.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Action mapping: index → fraction of remaining to execute
ACTION_MAP = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
N_ACTIONS = len(ACTION_MAP)

# State feature names (must match what we extract from data)
STATE_FEATURES = [
    "remaining_inventory_pct",
    "time_remaining_pct",
    "rolling_volatility_norm",
    "volume_imbalance",
    "spread_proxy_norm",
    "recent_return_5",
    "recent_return_20",
    "hour_sin",
    "hour_cos",
]
STATE_DIM = len(STATE_FEATURES)


@dataclass
class EnvConfig:
    """Configuration for the execution environment."""
    order_quantity: float = 1.0          # BTC
    time_horizon_bars: int = 60          # 1 hour
    side: str = "buy"

    # Impact model parameters
    temporary_impact: float = 0.1
    permanent_impact: float = 0.01
    spread_bps: float = 1.0
    max_participation_rate: float = 0.10

    # Reward shaping
    reward_scale: float = 100.0          # Scale reward for better gradient signal
    completion_bonus: float = 0.5        # Bonus for completing the order on time
    non_completion_penalty: float = -2.0 # Penalty for unfinished order at deadline


class ExecutionEnv:
    """Gym-style environment for trade execution RL.

    Each episode:
    1. Picks a random start point in the data
    2. Agent executes an order over `time_horizon_bars` steps
    3. Each step: agent chooses action → environment returns (next_state, reward, done)
    4. Reward = negative implementation shortfall for that step

    Usage:
        env = ExecutionEnv(df, config)
        state = env.reset()
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: EnvConfig | None = None,
        start_indices: np.ndarray | None = None,
    ):
        """
        Args:
            df: DataFrame with klines AND features already computed.
            config: Environment configuration.
            start_indices: Pre-computed valid start indices (for reproducibility).
                          If None, computes them from the data.
        """
        self.df = df
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng()

        # Pre-extract numpy arrays for speed (avoid pandas overhead in hot loop)
        self._close = df["close"].values.astype(np.float64)
        self._volume = df["volume"].values.astype(np.float64)
        self._high = df["high"].values.astype(np.float64)
        self._low = df["low"].values.astype(np.float64)

        # Feature arrays (with NaN → 0 filling)
        def _safe_col(name: str) -> np.ndarray:
            if name in df.columns:
                return np.nan_to_num(df[name].values.astype(np.float64), nan=0.0)
            return np.zeros(len(df), dtype=np.float64)

        self._rolling_vol = _safe_col("rolling_volatility")
        self._vol_imbalance = _safe_col("volume_imbalance")
        self._spread_proxy = _safe_col("spread_proxy")
        self._return_5 = _safe_col("return_5bar")
        self._return_20 = _safe_col("return_20bar")
        self._hour_sin = _safe_col("hour_sin")
        self._hour_cos = _safe_col("hour_cos")

        # Normalization stats (computed once, used for every state)
        valid_vol = self._rolling_vol[self._rolling_vol > 0]
        self._vol_mean = valid_vol.mean() if len(valid_vol) > 0 else 1.0
        self._vol_std = valid_vol.std() if len(valid_vol) > 0 else 1.0

        valid_spread = self._spread_proxy[self._spread_proxy > 0]
        self._spread_mean = valid_spread.mean() if len(valid_spread) > 0 else 1.0
        self._spread_std = valid_spread.std() if len(valid_spread) > 0 else 1.0

        # Valid start indices
        warmup = 50
        max_start = len(df) - self.config.time_horizon_bars - 1

        if start_indices is not None:
            self._start_indices = start_indices
        else:
            self._start_indices = np.arange(warmup, max_start)

        # Episode state
        self._current_idx = 0
        self._start_idx = 0
        self._remaining = 0.0
        self._arrival_price = 0.0
        self._total_executed = 0.0
        self._total_cost = 0.0
        self._step_count = 0
        self._done = False
        self._exec_prices = []
        self._exec_quantities = []

    def reset(self, start_idx: int | None = None) -> np.ndarray:
        """Reset environment for a new episode.

        Args:
            start_idx: Specific start index. If None, picks randomly.

        Returns:
            Initial state vector.
        """
        if start_idx is not None:
            self._start_idx = start_idx
        else:
            self._start_idx = self.rng.choice(self._start_indices)

        self._current_idx = self._start_idx
        self._remaining = self.config.order_quantity
        self._arrival_price = self._close[self._start_idx]
        self._total_executed = 0.0
        self._total_cost = 0.0
        self._step_count = 0
        self._done = False
        self._exec_prices = []
        self._exec_quantities = []

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Integer action index (0 to N_ACTIONS-1).

        Returns:
            (next_state, reward, done, info)
        """
        if self._done:
            return self._get_state(), 0.0, True, {}

        assert 0 <= action < N_ACTIONS, f"Invalid action: {action}"

        idx = self._current_idx
        fraction = ACTION_MAP[action]

        # Compute quantity to trade
        quantity = self._remaining * fraction

        # Market conditions
        mid_price = self._close[idx]
        bar_volume = self._volume[idx]
        direction = 1.0 if self.config.side == "buy" else -1.0

        # Apply impact model
        step_cost = 0.0
        exec_price = mid_price

        if quantity > 1e-10 and bar_volume > 0:
            participation_rate = min(
                quantity / bar_volume,
                self.config.max_participation_rate
            )
            actual_quantity = participation_rate * bar_volume
            actual_quantity = min(actual_quantity, self._remaining)

            # Spread cost
            spread_decimal = self.config.spread_bps / 10_000
            spread_cost = mid_price * spread_decimal * 0.5

            # Temporary impact
            temp_impact = self.config.temporary_impact * mid_price * participation_rate

            exec_price = mid_price + direction * (spread_cost + temp_impact)
            step_cost = (spread_cost + temp_impact) * actual_quantity

            self._remaining -= actual_quantity
            self._total_executed += actual_quantity
            self._total_cost += step_cost
            self._exec_prices.append(exec_price)
            self._exec_quantities.append(actual_quantity)

        # ═══════════════════════════════════════════════════════════
        # REWARD DESIGN v3: Sparse terminal reward
        #
        # The lesson from v1 and v2:
        #   v1: Per-step IS reward → agent learned to not trade (avoids cost)
        #   v2: Per-step progress reward → agent learned to dump immediately
        #
        # Solution: ONLY give reward at episode end, based on TOTAL cost.
        # Per-step: just a tiny shaping signal to guide exploration.
        # Terminal: compare our total cost against what TWAP would have paid.
        #
        # This is the approach used in Ning et al. (2021) and JP Morgan's LOXM.
        # ═══════════════════════════════════════════════════════════

        reward = 0.0

        if quantity > 1e-10 and bar_volume > 0:
            # Tiny per-step signal: prefer lower participation rate
            # This just guides exploration, doesn't dominate the terminal reward
            reward = -0.001 * (participation_rate / self.config.max_participation_rate)
        else:
            # Tiny nudge to keep trading (not enough to cause rushing)
            time_pct = self._step_count / self.config.time_horizon_bars
            if time_pct > 0.5 and self._remaining > 0.5 * self.config.order_quantity:
                reward = -0.01  # Very mild: "you should have traded more by now"

        # Check if done
        self._step_count += 1
        time_up = self._step_count >= self.config.time_horizon_bars
        order_complete = self._remaining < 1e-8

        if time_up or order_complete:
            self._done = True

            fill_pct = self._total_executed / self.config.order_quantity

            if fill_pct < 0.95:
                # Must complete the order - non-negotiable
                reward = -10.0 * (1.0 - fill_pct)
            else:
                # TERMINAL REWARD: penalize total cost directly
                # Normalize cost by (price × quantity) to get cost in bps
                notional = self._arrival_price * self.config.order_quantity
                cost_bps = (self._total_cost / max(notional, 1.0)) * 10_000

                # Reward: higher cost → more negative reward
                # TWAP on real BTCUSDT costs ~3 bps, immediate costs ~45 bps
                # Map to roughly [-5, +1] range
                reward = 2.0 - cost_bps * 0.15

                # Floor: completing is worth at least -3 (better than not completing)
                reward = max(reward, -3.0)

        # Advance to next bar
        self._current_idx += 1

        info = {
            "step_cost": step_cost,
            "remaining": self._remaining,
            "fill_pct": self._total_executed / self.config.order_quantity,
            "participation_rate": quantity / bar_volume if bar_volume > 0 else 0,
        }

        return self._get_state(), reward, self._done, info

    def _get_state(self) -> np.ndarray:
        """Extract normalized state vector from current market conditions."""
        idx = min(self._current_idx, len(self.df) - 1)
        horizon = self.config.time_horizon_bars

        state = np.array([
            # Inventory state
            self._remaining / self.config.order_quantity,       # [0, 1]
            max(0, (horizon - self._step_count)) / horizon,     # [0, 1]

            # Market state (normalized)
            (self._rolling_vol[idx] - self._vol_mean) / max(self._vol_std, 1e-8),
            np.clip(self._vol_imbalance[idx], 0, 5) / 5.0,     # [0, 1]
            (self._spread_proxy[idx] - self._spread_mean) / max(self._spread_std, 1e-8),

            # Momentum
            np.clip(self._return_5[idx] * 100, -5, 5),         # Scaled returns
            np.clip(self._return_20[idx] * 100, -5, 5),

            # Time features
            self._hour_sin[idx],
            self._hour_cos[idx],
        ], dtype=np.float32)

        return state

    def get_episode_metrics(self) -> dict:
        """Get metrics for the completed episode."""
        if not self._exec_prices:
            return {"is_bps": 0, "cost_usd": 0, "fill_pct": 0, "n_trades": 0}

        prices = np.array(self._exec_prices)
        qtys = np.array(self._exec_quantities)
        avg_price = np.average(prices, weights=qtys)

        direction = 1.0 if self.config.side == "buy" else -1.0
        is_bps = direction * (avg_price - self._arrival_price) / self._arrival_price * 10_000

        return {
            "is_bps": is_bps,
            "cost_usd": self._total_cost,
            "fill_pct": self._total_executed / self.config.order_quantity,
            "n_trades": len(self._exec_prices),
            "avg_price": avg_price,
            "arrival_price": self._arrival_price,
        }
