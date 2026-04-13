#!/usr/bin/env python3
"""
Tests for execution policies: baselines + RL agent + environment.

Run: python -m pytest tests/test_policies/ -v
  or: python tests/test_policies/test_all_policies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd


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

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print(f"  {name}")
        else:
            self.failed += 1
            print(f"    {name}: {detail}")

    def summary(self):
        t = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  POLICY TESTS: {self.passed}/{t} passed, {self.failed} failed")
        print(f"{'='*60}")
        return self.failed == 0


def test_baseline_policies():
    """Test TWAP, VWAP, Immediate, and Almgren-Chriss policies."""
    print("\nBASELINE POLICY TESTS")
    r = Results()

    from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy, AlmgrenChrissPolicy

    # ── ImmediatePolicy ──
    policy = ImmediatePolicy()
    state0 = {"bar_index": 0, "remaining_quantity": 1.0, "remaining_bars": 60,
              "total_quantity": 1.0, "time_horizon": 60}
    action0 = policy.get_action(state0)
    r.check("immediate: bar 0 → execute all", action0 == 1.0)

    state1 = {**state0, "bar_index": 1}
    action1 = policy.get_action(state1)
    r.check("immediate: bar 1 → do nothing", action1 == 0.0)

    # ── TWAPPolicy ──
    policy = TWAPPolicy()
    # Bar 0 of 60: should execute 1/60
    action = policy.get_action({"remaining_bars": 60, "bar_index": 0})
    r.check("TWAP: bar 0 → 1/60", abs(action - 1/60) < 1e-6, f"got {action}")

    # Last bar: should execute everything
    action = policy.get_action({"remaining_bars": 1, "bar_index": 59})
    r.check("TWAP: last bar → 1.0", abs(action - 1.0) < 1e-6)

    # ── VWAPPolicy ──
    policy = VWAPPolicy(volume_sensitivity=1.0)
    # Normal volume → similar to TWAP
    action = policy.get_action({"remaining_bars": 60, "bar_index": 0,
                                 "time_horizon": 60, "volume_imbalance": 1.0})
    r.check("VWAP: normal volume ≈ TWAP", 0.01 < action < 0.05)

    # High volume → trade more
    action_high = policy.get_action({"remaining_bars": 60, "bar_index": 0,
                                      "time_horizon": 60, "volume_imbalance": 3.0})
    action_low = policy.get_action({"remaining_bars": 60, "bar_index": 0,
                                     "time_horizon": 60, "volume_imbalance": 0.3})
    r.check("VWAP: high vol → trade more", action_high > action_low,
            f"high={action_high:.4f} low={action_low:.4f}")

    # ── AlmgrenChrissPolicy ──
    policy = AlmgrenChrissPolicy(risk_aversion=0.5)
    action = policy.get_action({"bar_index": 0, "remaining_quantity": 1.0,
                                 "remaining_bars": 60, "total_quantity": 1.0,
                                 "time_horizon": 60, "rolling_volatility": 0.5})
    r.check("A-C: returns valid fraction", 0 <= action <= 1.0, f"got {action}")

    # Different lambdas produce different schedules
    low_lam = AlmgrenChrissPolicy(risk_aversion=0.01)
    high_lam = AlmgrenChrissPolicy(risk_aversion=5.0)
    state = {"bar_index": 0, "remaining_quantity": 1.0, "remaining_bars": 60,
             "total_quantity": 1.0, "time_horizon": 60, "rolling_volatility": 0.5}
    a_low = low_lam.get_action(state)
    a_high = high_lam.get_action(state)
    r.check("A-C: high λ more aggressive at start", a_high >= a_low,
            f"low={a_low:.4f} high={a_high:.4f}")

    return r


def test_rl_environment():
    """Test the Gym-style execution environment."""
    print("\nRL ENVIRONMENT TESTS")
    r = Results()

    from src.features.engine import compute_all_features
    from src.policies.rl_env import ExecutionEnv, EnvConfig, STATE_DIM, N_ACTIONS, ACTION_MAP

    df = make_data(5000)
    df = compute_all_features(df)
    env = ExecutionEnv(df, EnvConfig(time_horizon_bars=60))

    # Reset returns correct state shape
    state = env.reset()
    r.check("reset: state shape", state.shape == (STATE_DIM,), f"got {state.shape}")
    r.check("reset: no NaN", not np.any(np.isnan(state)))

    # Step with valid action
    ns, reward, done, info = env.step(2)
    r.check("step: returns state", ns.shape == (STATE_DIM,))
    r.check("step: reward is float", isinstance(reward, float))
    r.check("step: done is bool", isinstance(done, bool))
    r.check("step: info has fill_pct", "fill_pct" in info)

    # Full episode completes
    state = env.reset()
    steps = 0
    done = False
    while not done:
        state, reward, done, info = env.step(2)  # 10% of remaining
        steps += 1
    r.check("episode completes", done)
    r.check(f"episode ≤ 60 steps", steps <= 60)

    metrics = env.get_episode_metrics()
    r.check("metrics: IS is finite", np.isfinite(metrics['is_bps']))
    r.check("metrics: fill > 0", metrics['fill_pct'] > 0)
    r.check("metrics: cost ≥ 0", metrics['cost_usd'] >= 0)

    # Action space valid
    r.check(f"N_ACTIONS = {N_ACTIONS}", N_ACTIONS == 6)
    r.check("all actions in [0, 1]", all(0 <= a <= 1 for a in ACTION_MAP))

    # Action 0 = don't trade
    state = env.reset()
    _, _, _, info = env.step(0)
    r.check("action 0 → no fill", info['fill_pct'] == 0)

    # Action 5 = execute everything
    state = env.reset()
    _, _, _, info = env.step(5)
    r.check("action 5 → significant fill", info['fill_pct'] > 0.05)

    return r


def test_dqn_agent():
    """Test the DQN agent: creation, action selection, learning, save/load."""
    print("\nDQN AGENT TESTS")
    r = Results()

    try:
        import torch
    except ImportError:
        r.check("PyTorch available", False, "pip install torch")
        return r

    r.check("PyTorch available", True)

    from src.policies.dqn_agent import DQNAgent, DQNConfig
    from src.policies.rl_env import STATE_DIM, N_ACTIONS

    config = DQNConfig(hidden_dims=(32, 32), buffer_size=1000,
                       batch_size=16, min_buffer_size=50, device="cpu")
    agent = DQNAgent(config)

    r.check("agent created", agent is not None)
    r.check(f"device: {agent.device}", str(agent.device) == "cpu")

    # Action selection
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action = agent.select_action(state)
    r.check("action is valid int", 0 <= action < N_ACTIONS)

    # Eval mode gives deterministic action
    a1 = agent.select_action(state, eval_mode=True)
    a2 = agent.select_action(state, eval_mode=True)
    r.check("eval mode deterministic", a1 == a2)

    # Fill buffer
    for _ in range(100):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        ns = np.random.randn(STATE_DIM).astype(np.float32)
        agent.buffer.push(s, np.random.randint(N_ACTIONS), np.random.randn(), ns, 0.0)
    r.check(f"buffer: {len(agent.buffer)} transitions", len(agent.buffer) == 100)

    # Training step
    loss = agent.update()
    r.check(f"training step loss={loss:.4f}", loss > 0)

    # Save/Load
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.pt"
        agent.save(path)
        r.check("model saved", path.exists())

        agent2 = DQNAgent(config)
        agent2.load(path)
        r.check("model loaded", agent2.total_steps == agent.total_steps)

        # Loaded model gives same action
        a_orig = agent.select_action(state, eval_mode=True)
        a_loaded = agent2.select_action(state, eval_mode=True)
        r.check("loaded model same action", a_orig == a_loaded)

    # Policy interface
    state_dict = {
        "remaining_quantity": 0.5, "remaining_bars": 30, "bar_index": 15,
        "total_quantity": 1.0, "time_horizon": 60,
        "rolling_volatility": 0.5, "volume_imbalance": 1.2, "spread_proxy": 0.001,
        "return_5bar": 0.001, "return_20bar": -0.002,
    }
    fraction = agent.get_action(state_dict)
    r.check(f"policy interface: fraction={fraction:.3f} in [0,1]", 0 <= fraction <= 1.0)

    return r


def test_mini_training_loop():
    """Test that the full training loop runs without errors."""
    print("\nMINI TRAINING LOOP TEST")
    r = Results()

    try:
        import torch
    except ImportError:
        r.check("skip (no PyTorch)", True)
        return r

    from src.features.engine import compute_all_features
    from src.policies.rl_env import ExecutionEnv, EnvConfig
    from src.policies.dqn_agent import DQNAgent, DQNConfig

    df = make_data(3000)
    df = compute_all_features(df)
    env = ExecutionEnv(df, EnvConfig(time_horizon_bars=30))

    config = DQNConfig(hidden_dims=(32, 32), learning_rate=1e-3,
                       epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=500,
                       buffer_size=5000, batch_size=32, min_buffer_size=200,
                       target_update_freq=100, device="cpu")
    agent = DQNAgent(config)

    import time
    t0 = time.time()
    rewards = []

    for ep in range(100):
        agent.epsilon = max(0.1, 1.0 - ep / 80)
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.select_action(state)
            ns, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, ns, float(done))
            agent.update()
            state = ns
            ep_reward += reward
        rewards.append(ep_reward)

    elapsed = time.time() - t0

    r.check(f"100 episodes in {elapsed:.1f}s", elapsed < 60)
    r.check(f"epsilon decayed: {agent.epsilon:.3f}", agent.epsilon < config.epsilon_start)
    r.check(f"buffer filled: {len(agent.buffer)}", len(agent.buffer) > 500)

    # Rewards should be finite
    r.check("all rewards finite", all(np.isfinite(r_) for r_ in rewards))

    return r


def test_policy_comparison():
    """Test that different policies produce different costs (sanity check)."""
    print("\nPOLICY COMPARISON TEST")
    r = Results()

    from src.features.engine import compute_all_features
    from src.simulator.engine import Order, simulate_execution
    from src.simulator.impact import ImpactParams
    from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy

    df = make_data(5000)
    df = compute_all_features(df)
    order = Order(symbol="BTCUSDT", side="buy", total_quantity=1.0, time_horizon_bars=60)
    params = ImpactParams()
    start = 100

    # Run all policies
    imm = simulate_execution(df, order, ImmediatePolicy(), start, params)
    twap = simulate_execution(df, order, TWAPPolicy(), start, params)
    vwap = simulate_execution(df, order, VWAPPolicy(), start, params)

    # Core theoretical results
    r.check("immediate cost > TWAP", imm.total_cost_usd > twap.total_cost_usd)
    r.check("TWAP fills 100%", abs(twap.total_executed - 1.0) < 0.01)
    r.check("VWAP fills > 90%", vwap.total_executed > 0.9)

    # All produce valid IS
    r.check("immediate IS finite", np.isfinite(imm.implementation_shortfall_bps))
    r.check("TWAP IS finite", np.isfinite(twap.implementation_shortfall_bps))
    r.check("VWAP IS finite", np.isfinite(vwap.implementation_shortfall_bps))

    # Immediate uses fewer child orders
    r.check("immediate: 1 order", imm.n_child_orders == 1)
    r.check("TWAP: 60 orders", twap.n_child_orders == 60)

    # Cost breakdown makes sense
    r.check("spread cost ≥ 0", twap.cost_breakdown['spread'] >= 0)
    r.check("impact cost ≥ 0", twap.cost_breakdown['temporary'] >= 0)

    print(f"\n  Cost comparison at start_idx={start}:")
    print(f"    Immediate: ${imm.total_cost_usd:.2f} (IS: {imm.implementation_shortfall_bps:+.2f} bps)")
    print(f"    TWAP:      ${twap.total_cost_usd:.2f} (IS: {twap.implementation_shortfall_bps:+.2f} bps)")
    print(f"    VWAP:      ${vwap.total_cost_usd:.2f} (IS: {vwap.implementation_shortfall_bps:+.2f} bps)")

    return r


# ─────────────────────────────────────────────────────────────
# AdaptivePolicy and ConservativeAdaptivePolicy tests
# ─────────────────────────────────────────────────────────────

class TestAdaptivePolicy:
    """Tests for AdaptivePolicy (feature-based heuristic)."""

    def _make_state(self, **overrides):
        base = {
            "remaining_bars": 30,
            "remaining_quantity": 0.5,
            "total_quantity": 1.0,
            "time_horizon": 60,
            "bar_index": 30,
            "volume_imbalance": 1.0,
            "rolling_volatility": 0.70,
            "spread_proxy": 0.001,
        }
        base.update(overrides)
        return base

    def test_action_in_valid_range(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        action = policy.get_action(self._make_state())
        assert 0.0 <= action <= 1.0

    def test_action_respects_min_rate(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy(min_rate=0.05)
        # Deliberately narrow spread and low volume to suppress rate
        action = policy.get_action(self._make_state(volume_imbalance=0.01, spread_proxy=1.0))
        assert action >= 0.05

    def test_action_respects_max_rate(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy(max_rate=0.3)
        # High volume + high volatility to push rate up
        action = policy.get_action(self._make_state(volume_imbalance=10.0, rolling_volatility=5.0))
        assert action <= 0.3

    def test_high_volume_trades_more_than_low_volume(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        high_vol = policy.get_action(self._make_state(volume_imbalance=3.0))
        low_vol = policy.get_action(self._make_state(volume_imbalance=0.3))
        assert high_vol > low_vol

    def test_high_spread_trades_less_than_low_spread(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        tight_spread = policy.get_action(self._make_state(spread_proxy=0.0001))
        wide_spread = policy.get_action(self._make_state(spread_proxy=0.01))
        assert tight_spread > wide_spread

    def test_urgency_kicks_in_near_deadline(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy(urgency_threshold=0.3)
        # 5 bars left out of 60 = ~8% time remaining → well within urgency zone
        normal = policy.get_action(self._make_state(remaining_bars=30, time_horizon=60))
        urgent = policy.get_action(self._make_state(remaining_bars=3, time_horizon=60))
        assert urgent >= normal

    def test_zero_remaining_quantity_returns_zero(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        action = policy.get_action(self._make_state(remaining_quantity=0.0))
        assert action == 0.0

    def test_zero_remaining_bars_returns_zero(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        action = policy.get_action(self._make_state(remaining_bars=0))
        assert action == 0.0

    def test_nan_volume_imbalance_handled(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        action = policy.get_action(self._make_state(volume_imbalance=float("nan")))
        assert 0.0 <= action <= 1.0

    def test_nan_rolling_vol_handled(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        action = policy.get_action(self._make_state(rolling_volatility=float("nan")))
        assert 0.0 <= action <= 1.0

    def test_reset_is_idempotent(self):
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy()
        policy.reset()  # Stateless - should not raise
        action = policy.get_action(self._make_state())
        assert 0.0 <= action <= 1.0

    def test_hard_deadline_minimum_rate(self):
        """With 10% time left and 50%+ inventory, must trade at least 0.3."""
        from src.policies.adaptive import AdaptivePolicy
        policy = AdaptivePolicy(urgency_threshold=0.3, min_rate=0.02)
        # 6 bars left of 60 = 10% remaining, 60% inventory left
        action = policy.get_action(self._make_state(
            remaining_bars=6, time_horizon=60, remaining_quantity=0.6, total_quantity=1.0
        ))
        assert action >= 0.3


class TestConservativeAdaptivePolicy:
    """Tests for ConservativeAdaptivePolicy."""

    def _make_state(self, **overrides):
        base = {
            "remaining_bars": 30,
            "remaining_quantity": 0.5,
            "total_quantity": 1.0,
            "time_horizon": 60,
            "bar_index": 30,
            "volume_imbalance": 1.0,
        }
        base.update(overrides)
        return base

    def test_action_in_valid_range(self):
        from src.policies.adaptive import ConservativeAdaptivePolicy
        policy = ConservativeAdaptivePolicy()
        action = policy.get_action(self._make_state())
        assert 0.0 <= action <= 1.0

    def test_normal_volume_slower_than_twap(self):
        """With normal volume, should trade below TWAP rate (60% base)."""
        from src.policies.adaptive import ConservativeAdaptivePolicy
        from src.policies.baselines import TWAPPolicy
        policy = ConservativeAdaptivePolicy()
        twap = TWAPPolicy()
        state = self._make_state(volume_imbalance=1.0)
        conservative_action = policy.get_action(state)
        twap_action = twap.get_action(state)
        assert conservative_action <= twap_action + 1e-6  # Allow floating-point tolerance

    def test_high_volume_increases_rate(self):
        """Volume > 1.3x average should boost participation."""
        from src.policies.adaptive import ConservativeAdaptivePolicy
        policy = ConservativeAdaptivePolicy()
        normal = policy.get_action(self._make_state(volume_imbalance=1.0))
        high = policy.get_action(self._make_state(volume_imbalance=2.0))
        assert high > normal

    def test_urgency_with_large_inventory_late(self):
        """Near deadline with significant inventory should ramp up."""
        from src.policies.adaptive import ConservativeAdaptivePolicy
        policy = ConservativeAdaptivePolicy()
        early = policy.get_action(self._make_state(remaining_bars=30))
        # 15 bars left (25% time) with 40% inventory
        late = policy.get_action(self._make_state(
            remaining_bars=15, remaining_quantity=0.4, total_quantity=1.0
        ))
        assert late >= early

    def test_zero_remaining_quantity_returns_zero(self):
        from src.policies.adaptive import ConservativeAdaptivePolicy
        policy = ConservativeAdaptivePolicy()
        assert policy.get_action(self._make_state(remaining_quantity=0.0)) == 0.0

    def test_zero_remaining_bars_returns_zero(self):
        from src.policies.adaptive import ConservativeAdaptivePolicy
        policy = ConservativeAdaptivePolicy()
        assert policy.get_action(self._make_state(remaining_bars=0)) == 0.0

    def test_reset_does_not_raise(self):
        from src.policies.adaptive import ConservativeAdaptivePolicy
        policy = ConservativeAdaptivePolicy()
        policy.reset()


if __name__ == "__main__":
    print("=" * 60)
    print("  EXECUTION POLICIES - TEST SUITE")
    print("=" * 60)

    all_r = []
    all_r.append(test_baseline_policies())
    all_r.append(test_rl_environment())
    all_r.append(test_dqn_agent())
    all_r.append(test_mini_training_loop())
    all_r.append(test_policy_comparison())

    tp = sum(x.passed for x in all_r)
    tf = sum(x.failed for x in all_r)

    print(f"\n{'='*60}")
    print(f"  ALL POLICY TESTS: {tp}/{tp+tf} passed, {tf} failed")
    print(f"{'='*60}")
    sys.exit(0 if tf == 0 else 1)
