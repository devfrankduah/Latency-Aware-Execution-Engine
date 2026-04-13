"""
Double DQN Agent for optimal trade execution.

ARCHITECTURE:
  - Double DQN (van Hasselt et al., 2016): Uses separate online/target
    networks to reduce Q-value overestimation.
  - Experience replay buffer with uniform sampling.
  - Epsilon-greedy exploration with decay.
  - Target network updated via soft update (Polyak averaging).

WHY DQN for execution?
  1. Discrete action space maps naturally to participation rate levels
  2. Off-policy learning → can learn from historical data efficiently
  3. Well-understood, debuggable - important for a semester project
  4. Ning et al. (2021) showed Double DQN recovers near-optimal A-C solutions

NETWORK:
  State (9 features) → FC(128) → ReLU → FC(128) → ReLU → FC(64) → ReLU → Q-values (6 actions)

This is intentionally simple. For production, you'd use:
  - LSTM/Transformer for temporal patterns
  - PPO/SAC for continuous actions
  - Distributional RL (IQN) for risk-sensitive execution
  But for CS5130 + interviews, DQN is the right choice.
"""

import logging
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try importing torch - graceful fallback if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. RL agent requires: pip install torch")

from src.policies.rl_env import STATE_DIM, N_ACTIONS, ACTION_MAP


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class DQNConfig:
    """Hyperparameters for DQN training."""

    # Network
    hidden_dims: tuple = (128, 128, 64)
    learning_rate: float = 1e-4

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 50_000

    # Replay buffer
    buffer_size: int = 100_000
    batch_size: int = 64
    min_buffer_size: int = 1_000  # Start training after this many transitions

    # Target network
    target_update_freq: int = 1_000  # Hard update every N steps
    tau: float = 0.005               # Soft update rate (if using Polyak)

    # Training
    gamma: float = 0.99              # Discount factor
    max_grad_norm: float = 10.0      # Gradient clipping
    n_episodes: int = 10_000
    eval_every: int = 500            # Evaluate every N episodes
    save_every: int = 2_000          # Save checkpoint every N episodes

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    def get_device(self) -> str:
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device


# ============================================================
# Q-Network
# ============================================================

if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """Q-value network: state → Q-values for each action."""

        def __init__(self, state_dim: int, n_actions: int, hidden_dims: tuple = (128, 128, 64)):
            super().__init__()

            layers = []
            prev_dim = state_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, n_actions))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy learning.

    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    """Double DQN Agent for optimal execution.

    Training loop:
    1. Collect experience using epsilon-greedy policy
    2. Store transitions in replay buffer
    3. Sample mini-batch, compute TD targets using target network
    4. Update online network with gradient descent
    5. Periodically sync target network

    After training, the agent can be used as an ExecutionPolicy
    by calling get_action(state_dict).
    """

    def __init__(self, config: DQNConfig | None = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install: pip install torch")

        self.config = config or DQNConfig()
        self.device = torch.device(self.config.get_device())
        logger.info(f"DQN Agent using device: {self.device}")

        # Networks
        self.online_net = QNetwork(
            STATE_DIM, N_ACTIONS, self.config.hidden_dims
        ).to(self.device)

        self.target_net = QNetwork(
            STATE_DIM, N_ACTIONS, self.config.hidden_dims
        ).to(self.device)

        # Initialize target = online
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(self.config.buffer_size)

        # Training state
        self.total_steps = 0
        self.epsilon = self.config.epsilon_start

        # Logging
        self.training_history = {
            "episode": [],
            "reward": [],
            "is_bps": [],
            "fill_pct": [],
            "epsilon": [],
            "loss": [],
        }

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: State vector from environment.
            eval_mode: If True, always pick best action (no exploration).

        Returns:
            Action index (0 to N_ACTIONS-1).
        """
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def update(self) -> float:
        """Perform one gradient update step.

        Returns:
            Loss value.
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return 0.0

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions we took
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Online network selects the best action for next state
        # 2. Target network evaluates Q-value of that action
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        # Huber loss (more robust than MSE to outliers)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Update target network
        self.total_steps += 1
        if self.total_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # NOTE: epsilon decay removed from here.
        # Epsilon is now controlled by the training loop: agent.epsilon = get_epsilon(episode)
        # This prevents update() from overwriting the training loop's schedule.

        return loss.item()

    def get_action(self, state: dict) -> float:
        """ExecutionPolicy interface: returns fraction of remaining to execute.

        This allows the trained DQN agent to be used directly with
        our existing simulate_execution() function.
        """
        # Convert state dict to state vector
        state_vec = np.array([
            state.get("remaining_quantity", 1.0) / state.get("total_quantity", 1.0),
            max(0, state.get("remaining_bars", 60)) / state.get("time_horizon", 60),
            state.get("rolling_volatility", 0.0) / max(self._vol_mean, 1e-8) - 1.0
                if hasattr(self, '_vol_mean') else 0.0,
            np.clip(state.get("volume_imbalance", 1.0), 0, 5) / 5.0,
            state.get("spread_proxy", 0.0) / max(self._spread_mean, 1e-8) - 1.0
                if hasattr(self, '_spread_mean') else 0.0,
            np.clip(state.get("return_5bar", 0.0) * 100, -5, 5),
            np.clip(state.get("return_20bar", 0.0) * 100, -5, 5),
            state.get("hour_sin", 0.0) if "hour_sin" in state else 0.0,
            state.get("hour_cos", 0.0) if "hour_cos" in state else 0.0,
        ], dtype=np.float32)

        action_idx = self.select_action(state_vec, eval_mode=True)
        return ACTION_MAP[action_idx]

    def save(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "training_history": self.training_history,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]
        self.training_history = checkpoint.get("training_history", {})
        logger.info(f"Model loaded from {path} (step {self.total_steps})")