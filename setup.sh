#!/bin/bash
# ══════════════════════════════════════════════════════════
# Latency-Aware Execution Engine - Automated Setup
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# This script:
#   1. Creates a virtual environment
#   2. Installs all dependencies
#   3. Runs the test suite
#   4. Runs the full pipeline on synthetic data (no download needed)
#   5. Reports results
# ══════════════════════════════════════════════════════════

set -e  # Exit on any error

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  LATENCY-AWARE EXECUTION ENGINE - SETUP"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Virtual environment ──
echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   Created venv/"
else
    echo "   venv/ already exists"
fi

source venv/bin/activate
echo "   Activated: $(python3 --version)"

# ── Step 2: Install dependencies ──
echo ""
echo "Step 2: Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "   All dependencies installed"

# ── Step 3: Verify imports ──
echo ""
echo "Step 3: Verifying package imports..."
python3 -c "
import src
print(f'   src v{src.__version__}')
from src.data.schemas import KlineSchema
print('   src.data')
from src.features.engine import compute_all_features
print('   src.features')
from src.simulator.engine import Order, simulate_execution
from src.simulator.impact import ImpactParams
print('   src.simulator')
from src.policies.baselines import TWAPPolicy, VWAPPolicy
print('   src.policies')
import torch
print(f'   PyTorch {torch.__version__}')
print('   All imports OK')
"

# ── Step 4: Run tests ──
echo ""
echo "Step 4: Running test suite..."
python3 tests/test_simulator/test_engine_and_impact.py
echo ""
python3 tests/test_policies/test_all_policies.py

# ── Step 5: Run pipeline on synthetic data ──
echo ""
echo "Step 5: Running pipeline (synthetic data - no download needed)..."
python3 -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')

from src.features.engine import compute_all_features
from src.simulator.engine import Order, simulate_execution
from src.simulator.impact import ImpactParams
from src.policies.baselines import ImmediatePolicy, TWAPPolicy, VWAPPolicy

# Generate synthetic data
rng = np.random.default_rng(42)
n = 50_000
close = 50000.0 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
noise = rng.uniform(0.0001, 0.001, n)
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min', tz='UTC'),
    'open': np.roll(close, 1), 'high': close * (1 + noise),
    'low': close * (1 - noise), 'close': close,
    'volume': rng.exponential(50.0, n),
    'quote_volume': rng.exponential(50.0, n) * close,
    'num_trades': rng.integers(100, 1000, n), 'symbol': 'BTCUSDT',
})
df.iloc[0, df.columns.get_loc('open')] = 50000.0
df['high'] = df[['open', 'high', 'close']].max(axis=1)
df['low'] = df[['open', 'low', 'close']].min(axis=1)
df = compute_all_features(df)

order = Order(symbol='BTCUSDT', side='buy', total_quantity=1.0, time_horizon_bars=60)
params = ImpactParams()

policies = {'Immediate': ImmediatePolicy(), 'TWAP': TWAPPolicy(), 'VWAP': VWAPPolicy()}
starts = rng.integers(100, len(df) - 61, size=100)

print()
print('  Synthetic Pipeline Results (100 simulations):')
print('  ' + '─' * 50)
for name, policy in policies.items():
    costs = []
    for si in starts:
        if hasattr(policy, 'reset'): policy.reset()
        r = simulate_execution(df, order, policy, int(si), params)
        costs.append(r.total_cost_usd)
    print(f'  {name:<14s} avg cost: \${np.mean(costs):.2f}')
print('  ' + '─' * 50)
print('  Pipeline runs successfully on synthetic data')
print()
"

# ── Done ──
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo ""
echo "  Next steps:"
echo "    # Run on real data (if downloaded):"
echo "    python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet"
echo ""
echo "    # Run with full analysis:"
echo "    python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet --full"
echo ""
echo "    # Train RL agent:"
echo "    python scripts/train_large.py --data data/processed/BTCUSDT_klines_1m.parquet --qty 50"
echo "══════════════════════════════════════════════════════════"