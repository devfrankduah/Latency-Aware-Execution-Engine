"""
Error handling utilities.

Provides decorators and helpers for robust error handling
throughout the pipeline. Ensures graceful degradation rather
than silent failures or cryptic stack traces.

Usage:
    from src.utils.errors import safe_execute, validate_dataframe, PipelineError

    @safe_execute("Loading data")
    def load_data(path):
        ...

    validate_dataframe(df, required_cols=['close', 'volume'])
"""

import logging
import functools
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline failures with context."""
    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")


def safe_execute(stage_name: str, default=None):
    """Decorator that wraps a function with error handling.

    Args:
        stage_name: Human-readable name for error messages.
        default: Value to return on failure (None = re-raise).

    Usage:
        @safe_execute("Feature computation")
        def compute_features(df):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PipelineError:
                raise  # Don't wrap our own errors
            except FileNotFoundError as e:
                msg = f"File not found: {e}"
                logger.error(f"[{stage_name}] {msg}")
                if default is not None:
                    return default
                raise PipelineError(stage_name, msg, recoverable=False)
            except pd.errors.EmptyDataError as e:
                msg = f"Empty data: {e}"
                logger.error(f"[{stage_name}] {msg}")
                if default is not None:
                    return default
                raise PipelineError(stage_name, msg, recoverable=False)
            except MemoryError:
                msg = "Out of memory. Try reducing data size or batch size."
                logger.error(f"[{stage_name}] {msg}")
                raise PipelineError(stage_name, msg, recoverable=False)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                logger.error(f"[{stage_name}] {msg}")
                if default is not None:
                    logger.warning(f"[{stage_name}] Returning default value")
                    return default
                raise PipelineError(stage_name, msg, recoverable=True)
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, required_cols: list[str],
                       stage: str = "Validation", min_rows: int = 100):
    """Validate a DataFrame has required columns and sufficient data.

    Raises PipelineError with helpful messages if validation fails.
    """
    if df is None:
        raise PipelineError(stage, "DataFrame is None")

    if not isinstance(df, pd.DataFrame):
        raise PipelineError(stage, f"Expected DataFrame, got {type(df).__name__}")

    if len(df) < min_rows:
        raise PipelineError(stage,
            f"Only {len(df)} rows (need at least {min_rows}). "
            f"Check data loading or date range.",
            recoverable=True)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise PipelineError(stage,
            f"Missing columns: {missing}. "
            f"Available: {sorted(df.columns.tolist())}",
            recoverable=False)

    # Check for all-NaN columns
    nan_cols = [c for c in required_cols if df[c].isna().all()]
    if nan_cols:
        raise PipelineError(stage,
            f"All-NaN columns: {nan_cols}. Check data source.",
            recoverable=True)

    # Check for negative prices
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in required_cols]
    for col in price_cols:
        if (df[col] < 0).any():
            raise PipelineError(stage,
                f"Negative values in '{col}' ({(df[col] < 0).sum()} rows)",
                recoverable=True)

    return True


def validate_order(qty: float, horizon: int, stage: str = "Order validation"):
    """Validate order parameters."""
    if qty <= 0:
        raise PipelineError(stage, f"Order quantity must be positive, got {qty}")
    if horizon <= 0:
        raise PipelineError(stage, f"Horizon must be positive, got {horizon}")
    if qty > 10000:
        logger.warning(f"[{stage}] Very large order: {qty} BTC. "
                       f"Fill rate may be low.")
    if horizon < 5:
        logger.warning(f"[{stage}] Very short horizon: {horizon} bars. "
                       f"Execution will be aggressive.")
    return True


def validate_model_path(path: str, stage: str = "Model loading"):
    """Validate model checkpoint exists and is readable."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise PipelineError(stage,
            f"Model not found: {path}. Train first with train_large.py",
            recoverable=False)
    if p.stat().st_size < 1000:
        raise PipelineError(stage,
            f"Model file too small ({p.stat().st_size} bytes). May be corrupted.",
            recoverable=False)
    return True


def check_environment():
    """Check that all required packages are available."""
    issues = []

    try:
        import numpy
    except ImportError:
        issues.append("numpy not installed: pip install numpy")

    try:
        import pandas
    except ImportError:
        issues.append("pandas not installed: pip install pandas")

    try:
        import torch
    except ImportError:
        issues.append("PyTorch not installed: pip install torch")

    try:
        import yaml
    except ImportError:
        issues.append("PyYAML not installed: pip install pyyaml")

    if issues:
        for i in issues:
            logger.error(f"    {i}")
        raise PipelineError("Environment check",
            f"{len(issues)} missing dependencies. Run: pip install -r requirements.txt")

    logger.info("  All dependencies available")
    return True