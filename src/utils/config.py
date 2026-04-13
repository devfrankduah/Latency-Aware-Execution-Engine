"""
Configuration loader for the execution engine.

PRODUCTION PATTERN: Centralized configuration management.
Why? Because:
  1. No magic numbers scattered across files
  2. Easy to reproduce experiments (just save the config)
  3. Easy to sweep hyperparameters
  4. Team members can override without changing code
"""

from pathlib import Path
from typing import Any

import yaml


# Project root = 2 levels up from this file (src/utils/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. Uses default.yaml if None.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Example:
        get_nested(config, "simulator.spread_bps")  → 1.0
        get_nested(config, "data.symbols")           → ["BTCUSDT"]

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value.
        default: Default value if key doesn't exist.

    Returns:
        The config value, or default if not found.
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current
