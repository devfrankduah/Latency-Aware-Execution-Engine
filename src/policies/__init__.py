"""
Execution policies (strategies).

Baseline strategies:
    - ImmediatePolicy: execute everything at once
    - TWAPPolicy: equal slices over time
    - VWAPPolicy: slices proportional to volume
    - AlmgrenChrissPolicy: optimal trajectory with risk aversion λ

ML-based strategies:
    - DQNAgent: Double DQN with dueling architecture + PER
    - ExecutionEnv: Gym-style RL environment for training

Usage:
    from src.policies.baselines import TWAPPolicy, VWAPPolicy, AlmgrenChrissPolicy
    from src.policies.dqn_agent import DQNAgent, DQNConfig
    from src.policies.rl_env import ExecutionEnv, EnvConfig
"""

from src.policies.baselines import (
    ImmediatePolicy,
    TWAPPolicy,
    VWAPPolicy,
    AlmgrenChrissPolicy,
)

