"""
Execution simulator with Almgren-Chriss market impact model.

Core components:
    - ImpactParams: market impact model configuration
    - Order: parent order specification
    - simulate_execution: run a policy through historical data
    - compute_almgren_chriss_trajectory: optimal A-C schedule

Usage:
    from src.simulator.engine import Order, simulate_execution
    from src.simulator.impact import ImpactParams, compute_almgren_chriss_trajectory

    order = Order(symbol="BTCUSDT", side="buy", total_quantity=10.0, time_horizon_bars=60)
    result = simulate_execution(df, order, policy, start_idx=100, impact_params=ImpactParams())
    print(result.summary())
"""

from src.simulator.impact import ImpactParams, compute_execution_price, compute_almgren_chriss_trajectory
from src.simulator.engine import Order, ExecutionResult, simulate_execution

