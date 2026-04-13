#!/usr/bin/env python3
"""
QR-DQN: Quantile Regression DQN for Risk-Sensitive Optimal Execution.

Instead of learning E[Q(s,a)], QR-DQN learns N quantile estimates of the
return distribution for each action. This enables:
  - Risk-neutral execution (mean of quantiles = standard DQN)
  - Conservative execution (CVaR at 10th percentile = minimize tail risk)
  - Aggressive execution (CVaR at 90th percentile = maximize upside)

The efficient execution frontier (cost vs risk at different CVaR levels)
is the key deliverable - it shows the agent learned meaningful
risk-return tradeoffs, not just cost minimization.

Architecture:
  State(14) → FC(256)+LN+ReLU → FC(256)+LN+ReLU → N_quantiles × N_actions

References:
  - Dabney et al. (2018) "Distributional RL with Quantile Regression"
  - Weston (2020) "Distributional RL for Optimal Execution" (Imperial College)
  - Hêche et al. (2025) "Risk-averse policies via distributional RL"

Usage:
    # Train QR-DQN
    python scripts/train_qrdqn.py --train --episodes 50000 --qty 50

    # Train on synthetic (quick test)
    python scripts/train_qrdqn.py --train --synthetic --episodes 3000 --qty 50

    # Evaluate with risk frontier
    python scripts/train_qrdqn.py --eval --model models/qrdqn/best.pt

    # Compare DQN vs QR-DQN
    python scripts/train_qrdqn.py --compare --dqn-model models/multi/best.pt --model models/qrdqn/best.pt
"""

import argparse, logging, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ──
ACTIONS = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
N_ACT = len(ACTIONS)
VWAP_IDX = 3
STATE_DIM = 14
N_QUANTILES = 51  # Standard from Dabney et al.


# ══════════════════════════════════════════
# Environment (same as train_large.py)
# ══════════════════════════════════════════
class Env:
    def __init__(self, df, qty=50.0, horizon=60, impact=0.3, max_part=0.15, name=''):
        self.qty, self.horizon, self.impact, self.max_part, self.name = qty, horizon, impact, max_part, name
        self.rng = np.random.default_rng()
        self._c = df['close'].values.astype(np.float64)
        self._v = df['volume'].values.astype(np.float64)
        raw = ((df['high'] - df['low']) / df['close']).values.astype(np.float64)
        med = np.nanmedian(raw[raw > 0])
        self._spread = np.clip(np.nan_to_num(raw / med * 0.0002, nan=0.0002), 0.00005, 0.002)
        def _s(n):
            return np.nan_to_num(df[n].values.astype(np.float64), nan=0.0) if n in df.columns else np.zeros(len(df))
        self._rvol = _s('rolling_volatility'); self._vimb = _s('volume_imbalance')
        self._sprd_raw = _s('spread_proxy'); self._r5 = _s('return_5bar'); self._r20 = _s('return_20bar')
        self._hsin = _s('hour_sin'); self._hcos = _s('hour_cos')
        vm = pd.Series(self._v).rolling(60, min_periods=10).mean().values
        self._vr = np.clip(np.nan_to_num(self._v / np.where(vm > 0, vm, 1.0), nan=1.0), 0, 5)
        sm = pd.Series(self._spread).rolling(60, min_periods=10).mean().values
        ss = pd.Series(self._spread).rolling(60, min_periods=10).std().values
        self._sprd_z = np.clip(np.nan_to_num((self._spread - sm) / np.where(ss > 0, ss, 1e-6), nan=0.0), -3, 3)
        v = self._rvol[self._rvol > 0]
        self._vm, self._vs = (float(v.mean()), float(v.std())) if len(v) > 0 else (1.0, 1.0)
        s = self._sprd_raw[self._sprd_raw > 0]
        self._sm, self._ss = (float(s.mean()), float(s.std())) if len(s) > 0 else (1e-4, 1e-4)
        self._starts = np.arange(70, len(df) - horizon - 1)
        self._reset()
        mv = np.median(self._v[self._v > 0]); mp = np.median(self._c[self._c > 0])
        log.info(f'  {name}: {len(df):,} bars, ${mp:,.0f}, vol={mv:.1f}, part={qty/horizon/max(mv,1):.2%}')

    def _reset(self):
        self.idx = self.start = 0; self.rem = self.arr = self.cost = self.exe = 0.0
        self.step_n = 0; self.done = False; self._mc = self._vc = 0.0

    def reset(self, si=None):
        self._reset()
        self.start = si if si is not None else self.rng.choice(self._starts)
        self.idx = self.start; self.rem = self.qty; self.arr = self._c[self.start]; return self._obs()

    def _fill(self, q, i):
        v, p = self._v[i], self._c[i]
        if q < 1e-10 or v < 1e-10: return 0.0, 0.0
        pr = min(q / v, self.max_part); act = min(pr * v, self.rem)
        return act, (p * self._spread[i] * 0.5 + self.impact * p * (pr ** 1.5)) * act

    def _vwap_qty(self, rem, bl):
        if bl <= 0: return rem
        return (rem / bl) * self._vr[min(self.idx, len(self._vr) - 1)]

    def step(self, action):
        if self.done: return self._obs(), 0.0, True, {}
        bl = self.horizon - self.step_n; vt = self._vwap_qty(self.rem, bl)
        if bl <= 3 and self.rem > 0.01 * self.qty: tgt = self.rem / max(bl, 1)
        else: tgt = vt * ACTIONS[action]
        act, c = self._fill(tgt, self.idx); _, vc = self._fill(vt, self.idx)
        self.rem -= act; self.exe += act; self.cost += c; self._mc += c; self._vc += vc
        self.step_n += 1; self.done = self.step_n >= self.horizon or self.rem < 1e-8
        reward = 0.0
        if self.done:
            if self.exe / self.qty < 0.90: reward = -20.0
            else:
                vw = self._run_vwap(); n = self.arr * self.qty
                reward = (vw - self.cost) / max(n, 1) * 10_000
        self.idx += 1; return self._obs(), float(reward), self.done, {}

    def _run_vwap(self):
        rem, tot = self.qty, 0.0
        for s in range(self.horizon):
            i = self.start + s
            if i >= len(self._c): break
            q = (rem / max(self.horizon - s, 1)) * self._vr[min(i, len(self._vr) - 1)]
            v, p = self._v[i], self._c[i]
            if q < 1e-10 or v < 1e-10: continue
            pr = min(q / v, self.max_part); act = min(pr * v, rem)
            tot += (p * self._spread[i] * 0.5 + self.impact * p * (pr ** 1.5)) * act; rem -= act
        return tot

    def _run_twap(self):
        rem, tot = self.qty, 0.0
        for s in range(self.horizon):
            i = self.start + s
            if i >= len(self._c): break
            q = rem / max(self.horizon - s, 1); v, p = self._v[i], self._c[i]
            if q < 1e-10 or v < 1e-10: continue
            pr = min(q / v, self.max_part); act = min(pr * v, rem)
            tot += (p * self._spread[i] * 0.5 + self.impact * p * (pr ** 1.5)) * act; rem -= act
        return tot

    def _obs(self):
        i = min(self.idx, len(self._c) - 1); h = self.horizon
        fp, tp = self.exe / self.qty, self.step_n / max(h, 1)
        ca = np.clip((self._vc - self._mc) / max(self._vc, 1e-10), -1, 1) if self._vc > 1e-10 else 0.0
        pm = np.clip((self._c[i] - self.arr) / max(self.arr, 1e-10) * 100, -5, 5) if self.arr > 0 else 0.0
        return np.array([self.rem / self.qty, max(0, h - self.step_n) / h,
            (self._rvol[i] - self._vm) / max(self._vs, 1e-8), np.clip(self._vimb[i], 0, 5) / 5.0,
            (self._sprd_raw[i] - self._sm) / max(self._ss, 1e-8),
            np.clip(self._r5[i] * 100, -5, 5), np.clip(self._r20[i] * 100, -5, 5),
            self._hsin[i], self._hcos[i], float(ca), float(fp - tp), float(pm),
            np.clip(self._vr[i], 0, 5) / 5.0, np.clip(self._sprd_z[i] / 3.0, -1, 1)], dtype=np.float32)

    def metrics(self):
        vw = self._run_vwap(); tw = self._run_twap(); n = self.arr * self.qty
        return {'cost': self.cost, 'fill': self.exe / self.qty, 'vwap_cost': vw, 'twap_cost': tw,
                'vs_vwap': (vw - self.cost) / n * 10_000 if n > 0 else 0,
                'vs_twap': (tw - self.cost) / n * 10_000 if n > 0 else 0}


# ══════════════════════════════════════════
# QR-DQN Network
# ══════════════════════════════════════════
class QRNet(nn.Module):
    """Quantile Regression DQN network.

    Instead of outputting Q(s,a) as a single scalar per action,
    outputs N_QUANTILES values per action - the quantile estimates
    of the return distribution.

    Output shape: (batch, N_ACT, N_QUANTILES)
    """
    def __init__(self, n_quantiles=N_QUANTILES):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        # Output: N_ACT * N_QUANTILES
        self.quantile_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N_ACT * n_quantiles),
        )

    def forward(self, x):
        """Returns quantile estimates: (batch, N_ACT, N_QUANTILES)"""
        h = self.shared(x)
        q = self.quantile_head(h)
        return q.view(-1, N_ACT, self.n_quantiles)

    def q_values(self, x, cvar_alpha=None):
        """Compute Q-values from quantile estimates.

        Args:
            x: state tensor
            cvar_alpha: if None, use mean (risk-neutral).
                        if float in (0,1), use CVaR at that level.
                        e.g. 0.1 = pessimistic (10th percentile region)
                             0.9 = optimistic (top 90% region)
        """
        quantiles = self.forward(x)  # (batch, N_ACT, N_QUANTILES)

        if cvar_alpha is None:
            # Risk-neutral: mean of all quantiles
            return quantiles.mean(dim=2)
        elif cvar_alpha <= 0.5:
            # Risk-averse: mean of bottom alpha fraction of quantiles
            k = max(1, int(self.n_quantiles * cvar_alpha))
            sorted_q, _ = quantiles.sort(dim=2)
            return sorted_q[:, :, :k].mean(dim=2)
        else:
            # Risk-seeking: mean of top (1-alpha) fraction
            k = max(1, int(self.n_quantiles * (1 - cvar_alpha)))
            sorted_q, _ = quantiles.sort(dim=2, descending=True)
            return sorted_q[:, :, :k].mean(dim=2)


# ══════════════════════════════════════════
# QR-DQN Agent
# ══════════════════════════════════════════
class QRDQNAgent:
    def __init__(self, lr=3e-4, n_quantiles=N_QUANTILES):
        self.n_quantiles = n_quantiles
        self.online = QRNet(n_quantiles)
        self.target = QRNet(n_quantiles)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = optim.AdamW(self.online.parameters(), lr=lr, weight_decay=1e-5)
        self.base_lr = lr

        # Precompute quantile midpoints: τ_i = (i + 0.5) / N
        self.tau = torch.FloatTensor(
            [(i + 0.5) / n_quantiles for i in range(n_quantiles)]
        )  # (N_QUANTILES,)

        # Replay buffer
        cap = 500_000
        self.buf_s = np.zeros((cap, STATE_DIM), np.float32)
        self.buf_a = np.zeros(cap, np.int64)
        self.buf_r = np.zeros(cap, np.float32)
        self.buf_ns = np.zeros((cap, STATE_DIM), np.float32)
        self.buf_d = np.zeros(cap, np.float32)
        self.buf_p = np.ones(cap, np.float64)
        self.buf_pos = self.buf_size = 0
        self.buf_cap = cap

        self.eps = 1.0
        self.steps = 0
        self.gamma = 0.97
        self.tau_soft = 0.005
        self.bs = 128
        self.cvar_alpha = None  # Risk-neutral during training

    def push(self, s, a, r, ns, d):
        i = self.buf_pos
        self.buf_s[i] = s; self.buf_a[i] = a; self.buf_r[i] = r
        self.buf_ns[i] = ns; self.buf_d[i] = d
        self.buf_p[i] = self.buf_p[:self.buf_size].max() if self.buf_size > 0 else 1.0
        self.buf_pos = (i + 1) % self.buf_cap
        self.buf_size = min(self.buf_size + 1, self.buf_cap)

    def set_lr(self, ep, total):
        lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * ep / total))
        for pg in self.opt.param_groups: pg['lr'] = max(lr, 1e-5)

    def act(self, s, greedy=False, cvar_alpha=None):
        if not greedy and np.random.random() < self.eps:
            return np.random.randint(N_ACT)
        with torch.no_grad():
            st = torch.FloatTensor(s).unsqueeze(0)
            alpha = cvar_alpha if cvar_alpha is not None else self.cvar_alpha
            q = self.online.q_values(st, cvar_alpha=alpha)
            return q.argmax(1).item()

    def learn(self):
        if self.buf_size < 5000:
            return 0.0

        # PER sampling
        beta = min(1.0, 0.4 + self.steps * 0.6 / 150_000)
        p = self.buf_p[:self.buf_size] ** 0.6
        p /= p.sum()
        idx = np.random.choice(self.buf_size, self.bs, p=p, replace=False)
        w = (self.buf_size * p[idx]) ** (-beta)
        w /= w.max()

        s = torch.FloatTensor(self.buf_s[idx])
        a = torch.LongTensor(self.buf_a[idx])
        r = torch.FloatTensor(self.buf_r[idx])
        ns = torch.FloatTensor(self.buf_ns[idx])
        d = torch.FloatTensor(self.buf_d[idx])
        wt = torch.FloatTensor(w)

        # Current quantile estimates for chosen actions
        # online(s): (bs, N_ACT, N_Q) → gather action → (bs, N_Q)
        curr_quantiles = self.online(s)  # (bs, N_ACT, N_Q)
        curr_q = curr_quantiles[torch.arange(self.bs), a]  # (bs, N_Q)

        # Target quantiles (Double DQN style)
        with torch.no_grad():
            # Select actions with online network
            next_q_vals = self.online.q_values(ns)  # (bs, N_ACT)
            next_actions = next_q_vals.argmax(1)  # (bs,)

            # Get quantiles from target network
            target_quantiles = self.target(ns)  # (bs, N_ACT, N_Q)
            next_q = target_quantiles[torch.arange(self.bs), next_actions]  # (bs, N_Q)

            # Target: r + γ * next_quantiles * (1 - done)
            target = r.unsqueeze(1) + self.gamma * next_q * (1 - d.unsqueeze(1))  # (bs, N_Q)

        # ── Quantile Huber Loss ──
        # For each pair (i,j): compute td = target_j - current_i
        # Shape: (bs, N_Q, N_Q) via broadcasting
        td = target.unsqueeze(1) - curr_q.unsqueeze(2)  # (bs, N_Q_curr, N_Q_target)

        # Huber loss (kappa=1.0)
        huber = torch.where(td.abs() <= 1.0, 0.5 * td ** 2, td.abs() - 0.5)

        # Quantile weights: |τ_i - I(td < 0)|
        tau_i = self.tau.view(1, -1, 1)  # (1, N_Q, 1)
        quantile_weight = torch.abs(tau_i - (td < 0).float())

        # Weighted Huber loss, averaged over quantile pairs
        loss_per_sample = (quantile_weight * huber).sum(dim=2).mean(dim=1)  # (bs,)

        # PER weighted loss
        loss = (wt * loss_per_sample).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 0.5)
        self.opt.step()

        # Update priorities
        td_errors = loss_per_sample.detach().numpy()
        for ii, t in zip(idx, td_errors):
            self.buf_p[ii] = abs(t) + 1e-6

        # Soft target update
        with torch.no_grad():
            for po, pt in zip(self.online.parameters(), self.target.parameters()):
                pt.data.copy_(self.tau_soft * po.data + (1 - self.tau_soft) * pt.data)

        self.steps += 1
        return loss.item()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'on': self.online.state_dict(), 'tg': self.target.state_dict(),
            'opt': self.opt.state_dict(), 'steps': self.steps, 'eps': self.eps,
            'n_quantiles': self.n_quantiles,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.online.load_state_dict(ck['on'])
        self.target.load_state_dict(ck['tg'])
        self.opt.load_state_dict(ck['opt'])
        self.steps = ck['steps']
        self.eps = ck['eps']


# ══════════════════════════════════════════
# Evaluation with risk levels
# ══════════════════════════════════════════

def evaluate(agent, env, n=500, cvar_alpha=None):
    """Evaluate agent at a specific risk level."""
    old_eps = agent.eps
    agent.eps = 0.0
    vv, vt, costs, fills = [], [], [], []

    for _ in range(n):
        s = env.reset()
        while not env.done:
            s, _, _, _ = env.step(agent.act(s, greedy=True, cvar_alpha=cvar_alpha))
        m = env.metrics()
        vv.append(m['vs_vwap']); vt.append(m['vs_twap'])
        costs.append(m['cost']); fills.append(m['fill'])

    agent.eps = old_eps
    return {
        'vs_vwap': np.mean(vv), 'vs_vwap_std': np.std(vv),
        'beat_vwap': np.mean([v > 0 for v in vv]) * 100,
        'vs_twap': np.mean(vt), 'vs_twap_std': np.std(vt),
        'beat_twap': np.mean([v > 0 for v in vt]) * 100,
        'cost_mean': np.mean(costs), 'cost_std': np.std(costs),
        'fill': np.mean(fills),
        'costs': costs, 'vs_vwap_all': vv, 'vs_twap_all': vt,
    }


def evaluate_risk_frontier(agent, env, n=500):
    """Evaluate at multiple CVaR levels to trace the efficient frontier."""
    alphas = [0.05, 0.10, 0.20, 0.30, 0.40, None, 0.60, 0.70, 0.80, 0.90, 0.95]
    labels = ['CVaR 5%', 'CVaR 10%', 'CVaR 20%', 'CVaR 30%', 'CVaR 40%',
              'Risk-Neutral', 'CVaR 60%', 'CVaR 70%', 'CVaR 80%', 'CVaR 90%', 'CVaR 95%']

    frontier = []
    for alpha, label in zip(alphas, labels):
        r = evaluate(agent, env, n=n, cvar_alpha=alpha)
        frontier.append({
            'alpha': alpha if alpha is not None else 0.5,
            'label': label,
            'cost_mean': r['cost_mean'],
            'cost_std': r['cost_std'],
            'vs_vwap': r['vs_vwap'],
            'vs_twap': r['vs_twap'],
            'beat_twap': r['beat_twap'],
        })
        log.info(f'  {label:<15s}: cost=${r["cost_mean"]:.2f} ±${r["cost_std"]:.2f} | '
                 f'vsTWAP={r["vs_twap"]:+.2f} beatTWAP={r["beat_twap"]:.0f}%')

    return frontier


# ══════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════

def load_asset(data_dir, symbol):
    from src.features.engine import compute_all_features
    path = Path(data_dir) / f'{symbol}_klines_1m.parquet'
    if not path.exists():
        return None, None, None
    df = pd.read_parquet(path)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    train = df[df['year'] <= 2023].copy().reset_index(drop=True)
    val = df[(df['year'] == 2024) & (df['month'] <= 6)].copy().reset_index(drop=True)
    test = df[(df['year'] == 2024) & (df['month'] > 6)].copy().reset_index(drop=True)
    for d in [train, val, test]:
        d.drop(columns=['year', 'month'], inplace=True, errors='ignore')
    return compute_all_features(train), compute_all_features(val), compute_all_features(test)


# ══════════════════════════════════════════
# Main
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--frontier', action='store_true')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--qty', type=float, default=50.0)
    parser.add_argument('--save-dir', default='models/qrdqn')
    parser.add_argument('--model', default='models/qrdqn/best.pt')
    parser.add_argument('--dqn-model', default='models/multi/best.pt')
    args = parser.parse_args()

    from src.features.engine import compute_all_features

    # ── Load data ──
    if args.synthetic:
        log.info('Generating synthetic data...')
        rng = np.random.default_rng(42)
        n = 200_000; bp = 40000; c = np.zeros(n); c[0] = bp
        for i in range(1, n): c[i] = c[i-1] * np.exp(rng.normal(0, 0.0008))
        noise = rng.uniform(0.0001, 0.002, n)
        bvol = rng.exponential(30, n)
        for i in range(n):
            h = (i % 1440) // 60
            bvol[i] *= [0.3,0.2,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0,2.5,2.0,
                        1.8,2.2,2.5,2.8,3.0,2.5,2.0,1.5,1.0,0.7,0.5,0.4][h]
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min', tz='UTC'),
            'open': np.roll(c,1), 'high': c*(1+noise), 'low': c*(1-noise), 'close': c,
            'volume': bvol, 'quote_volume': bvol*c,
            'num_trades': rng.integers(100,2000,n), 'symbol': 'BTCUSDT'})
        df.iloc[0, df.columns.get_loc('open')] = bp
        df['high'] = df[['open','high','close']].max(axis=1)
        df['low'] = df[['open','low','close']].min(axis=1)
        nt = len(df)
        train_df = compute_all_features(df.iloc[:int(nt*0.7)].copy().reset_index(drop=True))
        val_df = compute_all_features(df.iloc[int(nt*0.7):int(nt*0.85)].copy().reset_index(drop=True))
        test_df = compute_all_features(df.iloc[int(nt*0.85):].copy().reset_index(drop=True))
    else:
        log.info('Loading BTCUSDT...')
        train_df, val_df, test_df = load_asset(args.data_dir, 'BTCUSDT')
        if train_df is None:
            log.error('No data found'); return

    train_env = Env(train_df, qty=args.qty, name='BTC-train')
    val_env = Env(val_df, qty=args.qty, name='BTC-val')
    test_env = Env(test_df, qty=args.qty, name='BTC-test')

    # ── Train ──
    if args.train:
        agent = QRDQNAgent(lr=3e-4)
        sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
        n_ep = args.episodes
        explore_until = int(n_ep * 0.40)
        eps_floor = 0.06
        eval_every = max(500, n_ep // 25)

        # Warmstart
        log.info('Warmstarting (3000 episodes)...')
        for _ in range(3000):
            s = train_env.reset()
            while not train_env.done:
                r = np.random.random()
                if r < 0.45: a = VWAP_IDX
                elif r < 0.60: a = np.random.choice([4, 5, 6])
                elif r < 0.75: a = np.random.choice([0, 1, 2])
                else: a = np.random.randint(N_ACT)
                ns, rw, _, _ = train_env.step(a)
                agent.push(s, a, rw, ns, float(train_env.done)); s = ns
        log.info(f'Buffer: {agent.buf_size:,}')

        bl = evaluate(agent, val_env, n=300)
        log.info(f'Baseline: vsVWAP={bl["vs_vwap"]:+.3f} vsTWAP={bl["vs_twap"]:+.3f}')

        log.info(f'\n{"="*70}')
        log.info(f'  QR-DQN TRAINING: {n_ep:,} ep | {args.qty} BTC | {N_QUANTILES} quantiles')
        log.info(f'{"="*70}\n')

        best_vwap = -999; t0 = time.time()
        history = []

        for ep in range(1, n_ep + 1):
            agent.eps = max(eps_floor, 1.0 - (1.0 - eps_floor) * ep / explore_until)
            agent.set_lr(ep, n_ep)

            s = train_env.reset()
            while not train_env.done:
                a = agent.act(s)
                ns, r, _, _ = train_env.step(a)
                agent.push(s, a, r, ns, float(train_env.done))
                agent.learn()
                s = ns

            if ep % eval_every == 0:
                v = evaluate(agent, val_env, n=500)
                el = time.time() - t0; mk = ''
                if v['vs_vwap'] > best_vwap:
                    best_vwap = v['vs_vwap']; agent.save(sd / 'best.pt'); mk = ' *'
                eta = (n_ep - ep) / max(ep / el, 0.1) / 3600

                history.append({'ep': ep, 'vs_vwap': v['vs_vwap'], 'vs_twap': v['vs_twap'],
                                'beat_vwap': v['beat_vwap'], 'beat_twap': v['beat_twap'],
                                'cost_mean': v['cost_mean'], 'cost_std': v['cost_std']})

                log.info(f'  Ep {ep:>6,}/{n_ep:,} │ ε={agent.eps:.3f} │ '
                         f'vsVWAP: {v["vs_vwap"]:>+7.3f} beat:{v["beat_vwap"]:>3.0f}% │ '
                         f'vsTWAP: {v["vs_twap"]:>+7.3f} beat:{v["beat_twap"]:>3.0f}% │ '
                         f'cost: ${v["cost_mean"]:.0f}±{v["cost_std"]:.0f} │ '
                         f'{ep/el:.1f}/s ETA:{eta:.1f}h{mk}')

            if ep % 5000 == 0:
                agent.save(sd / f'checkpoint_{ep}.pt')

        agent.save(sd / 'final.pt')
        pd.DataFrame(history).to_csv(sd / 'training_history.csv', index=False)
        log.info(f'\nQR-DQN training done: {n_ep:,} ep in {(time.time()-t0)/3600:.1f}h')

        # Final test
        bp = sd / 'best.pt'
        if bp.exists(): agent.load(bp)
        agent.eps = 0.0
        r = evaluate(agent, test_env, n=1000)
        log.info(f'Test: vsVWAP={r["vs_vwap"]:+.3f} beat={r["beat_vwap"]:.0f}% | '
                 f'vsTWAP={r["vs_twap"]:+.3f} beat={r["beat_twap"]:.0f}%')

    # ── Risk Frontier ──
    if args.eval or args.frontier:
        agent = QRDQNAgent()
        agent.load(args.model)
        agent.eps = 0.0

        print(f'\n{"="*70}')
        print(f'  EFFICIENT EXECUTION FRONTIER - QR-DQN')
        print(f'  {N_QUANTILES} quantiles, {args.qty} BTC')
        print(f'{"="*70}')

        frontier = evaluate_risk_frontier(agent, test_env, n=500)

        print(f'\n  {"Risk Level":<16s} {"Cost($)":>12s} {"Cost Std":>12s} {"vs TWAP":>10s} {"Beat%":>7s}')
        print(f'  {"─"*60}')
        for f in frontier:
            print(f'  {f["label"]:<16s} ${f["cost_mean"]:>11.2f} ${f["cost_std"]:>11.2f} '
                  f'{f["vs_twap"]:>+10.2f} {f["beat_twap"]:>6.0f}%')

        print(f'\n  Key insight: as CVaR level decreases (more conservative),')
        print(f'  cost increases but variance decreases - the cost-risk tradeoff.')
        print(f'{"="*70}')

        # Save frontier
        sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(frontier).to_csv(sd / 'risk_frontier.csv', index=False)
        log.info(f'Frontier saved to {sd}/risk_frontier.csv')

        # Plot if matplotlib available
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                            facecolor='#0d1117')
            for ax in [ax1, ax2]:
                ax.set_facecolor('#161b22')

            costs = [f['cost_mean'] for f in frontier]
            stds = [f['cost_std'] for f in frontier]
            alphas_plot = [f['alpha'] for f in frontier]
            vs_twap = [f['vs_twap'] for f in frontier]

            # Left: Efficient execution frontier
            sc = ax1.scatter(stds, costs, c=alphas_plot, cmap='RdYlGn',
                            s=100, zorder=5, edgecolors='white', linewidth=1)
            ax1.plot(stds, costs, color='white', alpha=0.3, linewidth=1, zorder=3)
            plt.colorbar(sc, ax=ax1, label='CVaR α (0=conservative, 1=aggressive)')

            # Annotate extremes
            ax1.annotate('Conservative\n(minimize tail risk)',
                        xy=(stds[0], costs[0]), fontsize=9, color='#e74c3c',
                        xytext=(stds[0] + 20, costs[0] + 10),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c'))
            ax1.annotate('Aggressive\n(maximize upside)',
                        xy=(stds[-1], costs[-1]), fontsize=9, color='#2ecc71',
                        xytext=(stds[-1] - 50, costs[-1] - 10),
                        arrowprops=dict(arrowstyle='->', color='#2ecc71'))

            ax1.set_xlabel('Execution Cost Std ($)', fontsize=12)
            ax1.set_ylabel('Expected Execution Cost ($)', fontsize=12)
            ax1.set_title('Efficient Execution Frontier', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.2)

            # Right: vs TWAP at each risk level
            colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(frontier)))
            labels = [f['label'] for f in frontier]
            ax2.barh(labels, vs_twap, color=colors, edgecolor='white', linewidth=0.5)
            ax2.axvline(0, color='white', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Savings vs TWAP (bps)', fontsize=12)
            ax2.set_title('Risk-Adjusted Performance vs TWAP', fontsize=14, fontweight='bold')
            ax2.grid(True, axis='x', alpha=0.2)

            plt.suptitle('QR-DQN Risk-Sensitive Execution - Cost vs Risk Tradeoff',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()

            fig_path = sd / 'risk_frontier.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
            plt.close()
            log.info(f'Frontier plot saved to {fig_path}')

        except ImportError:
            log.info('matplotlib not available - skip plot')

    # ── Compare DQN vs QR-DQN ──
    if args.compare:
        print(f'\n{"="*70}')
        print(f'  DQN vs QR-DQN COMPARISON')
        print(f'{"="*70}')

        # Load QR-DQN
        qr_agent = QRDQNAgent()
        qr_agent.load(args.model)
        qr_agent.eps = 0.0

        # Load standard DQN
        from scripts.train_large import Agent as DQNAgent
        dqn_agent = DQNAgent()
        dqn_agent.load(args.dqn_model)
        dqn_agent.eps = 0.0

        print(f'\n  {"Agent":<20s} {"vs VWAP":>10s} {"vs TWAP":>10s} {"Beat TWAP":>10s} '
              f'{"Cost($)":>12s} {"Cost Std":>10s}')
        print(f'  {"─"*74}')

        # DQN
        # DQN agent doesn't support cvar_alpha, so evaluate manually
        dqn_agent.eps = 0.0
        vv, vt, costs_list = [], [], []
        for _ in range(1000):
            s = test_env.reset()
            while not test_env.done:
                s, _, _, _ = test_env.step(dqn_agent.act(s, greedy=True))
            m = test_env.metrics()
            vv.append(m['vs_vwap']); vt.append(m['vs_twap']); costs_list.append(m['cost'])
        r = {'vs_vwap': np.mean(vv), 'vs_twap': np.mean(vt), 'beat_twap': np.mean([v>0 for v in vt])*100,
             'cost_mean': np.mean(costs_list), 'cost_std': np.std(costs_list)}
        print(f'  {"DQN (standard)":<20s} {r["vs_vwap"]:>+10.3f} {r["vs_twap"]:>+10.3f} '
              f'{r["beat_twap"]:>9.0f}% ${r["cost_mean"]:>11.2f} ${r["cost_std"]:>9.2f}')

        # QR-DQN at different risk levels
        for alpha, label in [(0.10, 'QR-DQN (CVaR 10%)'),
                              (None, 'QR-DQN (neutral)'),
                              (0.90, 'QR-DQN (CVaR 90%)')]:
            r = evaluate(qr_agent, test_env, n=1000, cvar_alpha=alpha)
            print(f'  {label:<20s} {r["vs_vwap"]:>+10.3f} {r["vs_twap"]:>+10.3f} '
                  f'{r["beat_twap"]:>9.0f}% ${r["cost_mean"]:>11.2f} ${r["cost_std"]:>9.2f}')

        print(f'  {"─"*74}')
        print(f'  QR-DQN advantage: risk-aware execution with adjustable conservatism')
        print(f'{"="*70}')


if __name__ == '__main__':
    main()