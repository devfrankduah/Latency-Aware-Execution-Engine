#!/usr/bin/env python3
"""
Large Order Execution - 10-100 BTC with Variable Spread.

THIS is the actual project: optimizing execution of institutional-size orders
where market impact is material and strategy choice saves real money.

At 50 BTC on BTCUSDT:
  - TWAP participation: ~2.4% of bar volume
  - Impact cost: ~$200-500 per order (non-trivial)
  - Spread cost: variable, $100-300 per order
  - Total savings opportunity: $50-200 per order (5-15%)

The agent can exploit:
  1. Volume timing (trade when volume is high → lower impact)
  2. Spread timing (trade when spread is narrow → lower spread cost)
  3. Volatility timing (trade less during volatile bars → avoid adverse moves)
  4. Momentum (front-load when price is moving against us)

Usage:
    # Quick test
    python scripts/train_large.py --synthetic --episodes 3000 --qty 50

    # Real data, overnight
    nohup python scripts/train_large.py \
        --data data/processed/BTCUSDT_klines_1m.parquet \
        --qty 50 --episodes 50000 > large_50btc.txt 2>&1 &

    # Sweep order sizes for the paper
    python scripts/train_large.py --data data/processed/BTCUSDT_klines_1m.parquet --sweep
"""

import argparse, logging, math, sys, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ACTIONS = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
N_ACT = len(ACTIONS)
VWAP_IDX = 3
STATE_DIM = 14


class Env:
    def __init__(self, df, qty=50.0, horizon=60, impact=0.3, max_part=0.15):
        self.qty, self.horizon, self.impact, self.max_part = qty, horizon, impact, max_part
        self.rng = np.random.default_rng()
        self._c = df['close'].values.astype(np.float64)
        self._v = df['volume'].values.astype(np.float64)

        # Variable spread from actual data
        raw = ((df['high'] - df['low']) / df['close']).values.astype(np.float64)
        med = np.nanmedian(raw[raw > 0])
        # Scale so median ≈ 2 bps (realistic for large orders that cross the book)
        self._spread = np.nan_to_num(raw / med * 0.0002, nan=0.0002)
        self._spread = np.clip(self._spread, 0.00005, 0.002)

        def _s(n):
            return np.nan_to_num(df[n].values.astype(np.float64), nan=0.0) if n in df.columns else np.zeros(len(df))

        self._rvol = _s('rolling_volatility')
        self._vimb = _s('volume_imbalance')
        self._sprd_raw = _s('spread_proxy')
        self._r5, self._r20 = _s('return_5bar'), _s('return_20bar')
        self._hsin, self._hcos = _s('hour_sin'), _s('hour_cos')

        vol_ma = pd.Series(self._v).rolling(60, min_periods=10).mean().values
        self._vr = np.clip(np.nan_to_num(self._v / np.where(vol_ma > 0, vol_ma, 1.0), nan=1.0), 0, 5)

        sprd_ma = pd.Series(self._spread).rolling(60, min_periods=10).mean().values
        sprd_std = pd.Series(self._spread).rolling(60, min_periods=10).std().values
        self._sprd_z = np.clip(np.nan_to_num(
            (self._spread - sprd_ma) / np.where(sprd_std > 0, sprd_std, 1e-6), nan=0.0), -3, 3)

        v = self._rvol[self._rvol > 0]
        self._vm, self._vs = (float(v.mean()), float(v.std())) if len(v) > 0 else (1.0, 1.0)
        s = self._sprd_raw[self._sprd_raw > 0]
        self._sm, self._ss = (float(s.mean()), float(s.std())) if len(s) > 0 else (1e-4, 1e-4)
        self._starts = np.arange(70, len(df) - horizon - 1)
        self._reset()

        med_vol = np.median(self._v[self._v > 0])
        twap_part = qty / horizon / med_vol
        log.info(f'  Env: {qty} BTC, med_vol={med_vol:.1f}, TWAP part={twap_part:.2%}, '
                 f'impact={impact}, spread med={np.median(self._spread)*10000:.1f}bps '
                 f'[{np.percentile(self._spread,10)*10000:.1f}-{np.percentile(self._spread,90)*10000:.1f}]')

    def _reset(self):
        self.idx = self.start = 0
        self.rem = self.arr = self.cost = self.exe = 0.0
        self.step_n = 0; self.done = False
        self._mc = self._vc = 0.0

    def reset(self, si=None):
        self._reset()
        self.start = si if si is not None else self.rng.choice(self._starts)
        self.idx = self.start; self.rem = self.qty; self.arr = self._c[self.start]
        return self._obs()

    def _fill(self, q, i):
        v, p = self._v[i], self._c[i]
        if q < 1e-10 or v < 1e-10: return 0.0, 0.0
        pr = min(q / v, self.max_part)
        act = min(pr * v, self.rem)
        spread_cost = p * self._spread[i] * 0.5
        impact_cost = self.impact * p * (pr ** 1.5)
        return act, (spread_cost + impact_cost) * act

    def _vwap_qty(self, rem, bl):
        if bl <= 0: return rem
        return (rem / bl) * self._vr[min(self.idx, len(self._vr) - 1)]

    def step(self, action):
        if self.done: return self._obs(), 0.0, True, {}
        bl = self.horizon - self.step_n
        vwap_target = self._vwap_qty(self.rem, bl)
        if bl <= 3 and self.rem > 0.01 * self.qty:
            tgt = self.rem / max(bl, 1)
        else:
            tgt = vwap_target * ACTIONS[action]
        act, c = self._fill(tgt, self.idx)
        _, vc = self._fill(vwap_target, self.idx)
        self.rem -= act; self.exe += act; self.cost += c
        self._mc += c; self._vc += vc
        self.step_n += 1
        self.done = self.step_n >= self.horizon or self.rem < 1e-8
        reward = 0.0
        if self.done:
            if self.exe / self.qty < 0.90: reward = -20.0
            else:
                vwap_total = self._run_policy('vwap')
                n = self.arr * self.qty
                reward = (vwap_total - self.cost) / max(n, 1) * 10_000
        self.idx += 1
        return self._obs(), float(reward), self.done, {}

    def _run_policy(self, policy='vwap'):
        rem, total = self.qty, 0.0
        for s in range(self.horizon):
            i = self.start + s
            if i >= len(self._c): break
            bl = self.horizon - s
            if policy == 'vwap':
                q = (rem / max(bl, 1)) * self._vr[min(i, len(self._vr) - 1)]
            else:
                q = rem / max(bl, 1)
            v, p = self._v[i], self._c[i]
            if q < 1e-10 or v < 1e-10: continue
            pr = min(q / v, self.max_part)
            act = min(pr * v, rem)
            total += (p * self._spread[i] * 0.5 + self.impact * p * (pr ** 1.5)) * act
            rem -= act
        return total

    def _obs(self):
        i = min(self.idx, len(self._c) - 1); h = self.horizon
        fp, tp = self.exe / self.qty, self.step_n / max(h, 1)
        ca = np.clip((self._vc - self._mc) / max(self._vc, 1e-10), -1, 1) if self._vc > 1e-10 else 0.0
        pm = np.clip((self._c[i] - self.arr) / max(self.arr, 1e-10) * 100, -5, 5) if self.arr > 0 else 0.0
        return np.array([
            self.rem / self.qty, max(0, h - self.step_n) / h,
            (self._rvol[i] - self._vm) / max(self._vs, 1e-8),
            np.clip(self._vimb[i], 0, 5) / 5.0,
            (self._sprd_raw[i] - self._sm) / max(self._ss, 1e-8),
            np.clip(self._r5[i] * 100, -5, 5), np.clip(self._r20[i] * 100, -5, 5),
            self._hsin[i], self._hcos[i],
            float(ca), float(fp - tp), float(pm),
            np.clip(self._vr[i], 0, 5) / 5.0,
            np.clip(self._sprd_z[i] / 3.0, -1, 1),
        ], dtype=np.float32)

    def metrics(self):
        vw = self._run_policy('vwap'); tw = self._run_policy('twap')
        n = self.arr * self.qty
        return {'cost': self.cost, 'fill': self.exe / self.qty,
                'vwap_cost': vw, 'twap_cost': tw,
                'vs_vwap': (vw - self.cost) / n * 10_000 if n > 0 else 0,
                'vs_twap': (tw - self.cost) / n * 10_000 if n > 0 else 0}


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU())
        self.val = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, N_ACT))
    def forward(self, x):
        s = self.shared(x); v, a = self.val(s), self.adv(s)
        return v + a - a.mean(-1, keepdim=True)

class Agent:
    def __init__(self, lr=3e-4):
        self.online = QNet(); self.target = QNet()
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = optim.AdamW(self.online.parameters(), lr=lr, weight_decay=1e-5)
        self.base_lr = lr
        cap = 300_000
        self.buf_s = np.zeros((cap, STATE_DIM), np.float32)
        self.buf_a = np.zeros(cap, np.int64); self.buf_r = np.zeros(cap, np.float32)
        self.buf_ns = np.zeros((cap, STATE_DIM), np.float32); self.buf_d = np.zeros(cap, np.float32)
        self.buf_p = np.ones(cap, np.float64)
        self.buf_pos = self.buf_size = 0; self.buf_cap = cap
        self.eps = 1.0; self.steps = 0; self.gamma = 0.97; self.tau = 0.005; self.bs = 128

    def push(self, s, a, r, ns, d):
        i = self.buf_pos
        self.buf_s[i]=s; self.buf_a[i]=a; self.buf_r[i]=r; self.buf_ns[i]=ns; self.buf_d[i]=d
        self.buf_p[i] = self.buf_p[:self.buf_size].max() if self.buf_size > 0 else 1.0
        self.buf_pos = (i+1) % self.buf_cap; self.buf_size = min(self.buf_size+1, self.buf_cap)

    def set_lr(self, ep, total):
        lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * ep / total))
        for pg in self.opt.param_groups: pg['lr'] = max(lr, 1e-5)

    def act(self, s, greedy=False):
        if not greedy and np.random.random() < self.eps: return np.random.randint(N_ACT)
        with torch.no_grad(): return self.online(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()

    def learn(self):
        if self.buf_size < 3000: return 0.0
        beta = min(1.0, 0.4 + self.steps * 0.6 / 100_000)
        p = self.buf_p[:self.buf_size] ** 0.6; p /= p.sum()
        idx = np.random.choice(self.buf_size, self.bs, p=p, replace=False)
        w = (self.buf_size * p[idx]) ** (-beta); w /= w.max()
        s=torch.FloatTensor(self.buf_s[idx]); a=torch.LongTensor(self.buf_a[idx])
        r=torch.FloatTensor(self.buf_r[idx]); ns=torch.FloatTensor(self.buf_ns[idx])
        d=torch.FloatTensor(self.buf_d[idx]); wt=torch.FloatTensor(w)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na = self.online(ns).argmax(1)
            nq = self.target(ns).gather(1, na.unsqueeze(1)).squeeze(1)
            tgt = r + self.gamma * nq * (1 - d)
        td = (q - tgt).detach().numpy()
        loss = (wt * nn.SmoothL1Loss(reduction='none')(q, tgt)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 0.5); self.opt.step()
        for ii, t in zip(idx, td): self.buf_p[ii] = abs(t) + 1e-6
        with torch.no_grad():
            for po, pt in zip(self.online.parameters(), self.target.parameters()):
                pt.data.copy_(self.tau * po.data + (1 - self.tau) * pt.data)
        self.steps += 1; return loss.item()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'on': self.online.state_dict(), 'tg': self.target.state_dict(),
                     'opt': self.opt.state_dict(), 'steps': self.steps, 'eps': self.eps}, path)
    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.online.load_state_dict(ck['on']); self.target.load_state_dict(ck['tg'])
        self.opt.load_state_dict(ck['opt']); self.steps = ck['steps']; self.eps = ck['eps']


def evaluate(agent, env, n=500):
    old = agent.eps; agent.eps = 0.0
    vv, vt, costs, fills = [], [], [], []
    for _ in range(n):
        s = env.reset()
        while not env.done: s, _, _, _ = env.step(agent.act(s, True))
        m = env.metrics()
        vv.append(m['vs_vwap']); vt.append(m['vs_twap'])
        costs.append(m['cost']); fills.append(m['fill'])
    agent.eps = old
    return {'vs_vwap': np.mean(vv), 'vs_vwap_std': np.std(vv),
            'beat_vwap': np.mean([s > 0 for s in vv]) * 100,
            'vs_twap': np.mean(vt), 'beat_twap': np.mean([s > 0 for s in vt]) * 100,
            'cost': np.mean(costs), 'fill': np.mean(fills)}


def train_one(train_df, val_df, qty, n_ep, save_dir):
    """Train one configuration. Returns agent and best savings."""
    train_env = Env(train_df, qty=qty)
    val_env = Env(val_df, qty=qty)
    agent = Agent(lr=3e-4)
    sd = Path(save_dir); sd.mkdir(parents=True, exist_ok=True)
    explore_until = int(n_ep * 0.40); eps_floor = 0.06
    eval_every = max(500, n_ep // 30)

    log.info(f'Warmstarting...')
    for _ in range(2500):
        s = train_env.reset()
        while not train_env.done:
            r = np.random.random()
            if r < 0.45: a = VWAP_IDX
            elif r < 0.60: a = np.random.choice([4, 5, 6])
            elif r < 0.75: a = np.random.choice([0, 1, 2])
            else: a = np.random.randint(N_ACT)
            a = max(0, min(a, N_ACT - 1))
            ns, rw, _, _ = train_env.step(a)
            agent.push(s, a, rw, ns, float(train_env.done)); s = ns
    log.info(f'Buffer: {agent.buf_size:,}')

    best_vwap = -999; t0 = time.time()
    for ep in range(1, n_ep + 1):
        agent.eps = max(eps_floor, 1.0 - (1.0 - eps_floor) * ep / explore_until)
        agent.set_lr(ep, n_ep)
        s = train_env.reset()
        while not train_env.done:
            a = agent.act(s); ns, r, _, _ = train_env.step(a)
            agent.push(s, a, r, ns, float(train_env.done)); agent.learn(); s = ns
        if ep % eval_every == 0:
            v = evaluate(agent, val_env, n=500)
            el = time.time() - t0; mk = ''
            if v['vs_vwap'] > best_vwap:
                best_vwap = v['vs_vwap']; agent.save(sd / 'best.pt'); mk = ' *'
            eta = (n_ep - ep) / max(ep / el, 0.1) / 3600
            log.info(f'  Ep {ep:>6,}/{n_ep:,} │ ε={agent.eps:.3f} │ '
                     f'vsVWAP: {v["vs_vwap"]:>+7.3f}±{v["vs_vwap_std"]:.1f} '
                     f'beat:{v["beat_vwap"]:>3.0f}% │ '
                     f'vsTWAP: {v["vs_twap"]:>+7.3f} beat:{v["beat_twap"]:>3.0f}% │ '
                     f'fill:{v["fill"]:.1%} │ {ep/el:.1f}/s ETA:{eta:.1f}h{mk}')
        if ep % 5000 == 0: agent.save(sd / f'checkpoint_{ep}.pt')
    agent.save(sd / 'final.pt')
    return agent, best_vwap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--qty', type=float, default=50.0)
    parser.add_argument('--save-dir', type=str, default='models/large')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.synthetic or args.data is None:
        log.info('Generating synthetic data with realistic volume patterns...')
        rng = np.random.default_rng(42)
        n = 300_000; c = np.zeros(n); c[0] = 40000.0
        for i in range(1, n):
            vol = [0.0003, 0.0008, 0.0015, 0.0005][(i // 5000) % 4]
            c[i] = c[i-1] * np.exp(rng.normal(0, vol))
        noise = rng.uniform(0.0001, 0.003, n)
        bv = rng.exponential(30.0, n)
        for i in range(n):
            h = (i % 1440) // 60
            bv[i] *= [0.3,0.2,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0,2.5,2.0,
                       1.8,2.2,2.5,2.8,3.0,2.5,2.0,1.5,1.0,0.7,0.5,0.4][h]
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min', tz='UTC'),
            'open': np.roll(c,1), 'high': c*(1+noise), 'low': c*(1-noise), 'close': c,
            'volume': bv, 'quote_volume': bv*c,
            'num_trades': rng.integers(100,2000,n), 'symbol': 'BTCUSDT'})
        df.iloc[0, df.columns.get_loc('open')] = 40000.0
        df['high'] = df[['open','high','close']].max(axis=1)
        df['low'] = df[['open','low','close']].min(axis=1)
    else:
        df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data, parse_dates=['timestamp'])
        log.info(f'Loaded {len(df):,} bars')

    from src.features.engine import compute_all_features
    nt = len(df)
    train_df = compute_all_features(df.iloc[:int(nt*0.6)].copy().reset_index(drop=True))
    val_df = compute_all_features(df.iloc[int(nt*0.6):int(nt*0.8)].copy().reset_index(drop=True))
    test_df = compute_all_features(df.iloc[int(nt*0.8):].copy().reset_index(drop=True))
    log.info(f'Split: Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}')

    if args.sweep:
        sizes = [10, 25, 50, 100]
        ep_map = {10: 30000, 25: 40000, 50: 50000, 100: 50000}
        results = []

        for qty in sizes:
            n_ep = ep_map.get(qty, 50000)
            log.info(f'\n{"="*60}')
            log.info(f'  SWEEP: {qty} BTC, {n_ep:,} episodes')
            log.info(f'{"="*60}')
            agent, _ = train_one(train_df, val_df, qty, n_ep, f'models/sweep/{qty}btc')
            bp = Path(f'models/sweep/{qty}btc/best.pt')
            if bp.exists(): agent.load(bp)
            agent.eps = 0.0
            test_env = Env(test_df, qty=qty)
            r = evaluate(agent, test_env, n=1000)
            results.append({'qty': qty, **r})
            log.info(f'  {qty} BTC: vsVWAP={r["vs_vwap"]:+.3f} beat={r["beat_vwap"]:.0f}% | '
                     f'vsTWAP={r["vs_twap"]:+.3f} beat={r["beat_twap"]:.0f}%')

        print(f'\n{"="*80}')
        print(f'  LARGE ORDER SWEEP - Variable Spread + Impact')
        print(f'{"="*80}')
        print(f'  {"Order":>8s} {"vs VWAP":>12s} {"Beat%":>7s} {"vs TWAP":>12s} {"Beat%":>7s} '
              f'{"Cost($)":>12s} {"Fill":>6s}')
        print(f'  {"─"*68}')
        for r in results:
            vmark = '[+]' if r['vs_vwap'] > 0 else '[~]' if abs(r['vs_vwap']) < 1.0 else '[-]'
            print(f'  {r["qty"]:>7.0f}  {r["vs_vwap"]:>+12.3f} {r["beat_vwap"]:>6.0f}%  '
                  f'{r["vs_twap"]:>+12.3f} {r["beat_twap"]:>6.0f}%  '
                  f'${r["cost"]:>11.2f}  {r["fill"]:>5.1%}  {vmark}')
        print(f'{"="*80}')
        return

    # Single run
    log.info(f'\n{"="*65}')
    log.info(f'  LARGE ORDER TRAINING: {args.qty} BTC | {args.episodes:,} episodes')
    log.info(f'  Variable spread + impact | VWAP-relative actions')
    log.info(f'{"="*65}\n')

    agent, best = train_one(train_df, val_df, args.qty, args.episodes, args.save_dir)
    bp = Path(args.save_dir) / 'best.pt'
    if bp.exists(): agent.load(bp)
    agent.eps = 0.0

    test_env = Env(test_df, qty=args.qty)
    dc, vc, tc = [], [], []
    vv, vt = [], []
    for _ in range(1000):
        s = test_env.reset()
        while not test_env.done: s, _, _, _ = test_env.step(agent.act(s, True))
        m = test_env.metrics()
        dc.append(m['cost']); vc.append(m['vwap_cost']); tc.append(m['twap_cost'])
        vv.append(m['vs_vwap']); vt.append(m['vs_twap'])

    print(f'\n{"="*80}')
    print(f'  OUT-OF-SAMPLE - {args.qty} BTC × 1000 episodes (Variable Spread + Impact)')
    print(f'{"="*80}')
    print(f'  {"Strategy":<20s} {"Cost ($)":>14s} {"vs TWAP":>12s} {"vs VWAP":>12s}')
    print(f'  {"─"*60}')
    print(f'  {"TWAP":<20s} ${np.mean(tc):>13.2f} {"baseline":>12s} {"-":>12s}')
    print(f'  {"VWAP":<20s} ${np.mean(vc):>13.2f} '
          f'{(1-np.mean(vc)/np.mean(tc))*100:>+11.1f}% {"baseline":>12s}')
    print(f'  {"ML Agent (ours)":<20s} ${np.mean(dc):>13.2f} '
          f'{(1-np.mean(dc)/np.mean(tc))*100:>+11.1f}% '
          f'{(1-np.mean(dc)/np.mean(vc))*100:>+11.1f}%')
    print(f'  {"─"*60}')
    print(f'  vs VWAP: {np.mean(vv):>+.3f} ± {np.std(vv):.2f} bps | Win: {np.mean([s>0 for s in vv])*100:.0f}%')
    print(f'  vs TWAP: {np.mean(vt):>+.3f} ± {np.std(vt):.2f} bps | Win: {np.mean([s>0 for s in vt])*100:.0f}%')
    if np.mean(vv) > 0:
        saved = np.mean(vc) - np.mean(dc)
        print(f'\n  ML AGENT BEATS BOTH TWAP AND VWAP')
        print(f'     Saves ${saved:.2f} per {args.qty} BTC order vs VWAP')
        print(f'     Over 1000 orders: ${saved*1000:,.0f} saved')
    elif np.mean(vt) > 0:
        print(f'\n  Beats TWAP |  Within {abs(np.mean(vv)):.3f} bps of VWAP')
    print(f'{"="*80}')
    pd.DataFrame({'qty': [args.qty], 'vs_vwap': [np.mean(vv)], 'vs_twap': [np.mean(vt)],
                   'dqn_cost': [np.mean(dc)], 'vwap_cost': [np.mean(vc)], 'twap_cost': [np.mean(tc)]
                   }).to_csv(Path(args.save_dir) / 'results.csv', index=False)


if __name__ == '__main__':
    main()