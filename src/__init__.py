"""
Latency-Aware Execution Engine for Portfolio Rebalancing.

A production-quality trade execution simulator that optimizes order
execution using adaptive, latency- and liquidity-aware policies.
Built on real crypto market data with ML-driven execution.

Modules:
    src.data        - Data loading, validation, schemas
    src.features    - Feature engineering pipeline
    src.simulator   - Execution simulator and market impact models
    src.policies    - Execution strategies (TWAP, VWAP, A-C, DQN)
    src.evaluation  - Backtesting, TCA metrics, visualization
    src.utils       - Configuration, logging, helpers
"""

__version__ = "1.0.0"
__author__ = "Nikhilesh"
