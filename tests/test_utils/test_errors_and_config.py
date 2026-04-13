"""
Tests for src/utils/errors.py and src/utils/config.py

Covers:
  - PipelineError
  - safe_execute decorator
  - validate_dataframe()
  - validate_order()
  - load_config()
  - get_nested()
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.config import get_nested, load_config
from src.utils.errors import (
    PipelineError,
    safe_execute,
    validate_dataframe,
    validate_order,
)


# ─────────────────────────────────────────────
# PipelineError
# ─────────────────────────────────────────────

class TestPipelineError:
    def test_is_exception(self):
        err = PipelineError("loading", "something went wrong")
        assert isinstance(err, Exception)

    def test_stage_attribute(self):
        err = PipelineError("feature_eng", "bad column")
        assert err.stage == "feature_eng"

    def test_message_contains_stage(self):
        err = PipelineError("MyStage", "oops")
        assert "MyStage" in str(err)

    def test_message_contains_detail(self):
        err = PipelineError("MyStage", "oops detail")
        assert "oops detail" in str(err)

    def test_recoverable_default_false(self):
        err = PipelineError("stage", "msg")
        assert err.recoverable is False

    def test_recoverable_can_be_true(self):
        err = PipelineError("stage", "msg", recoverable=True)
        assert err.recoverable is True


# ─────────────────────────────────────────────
# safe_execute
# ─────────────────────────────────────────────

class TestSafeExecute:
    def test_passes_through_return_value(self):
        @safe_execute("test")
        def fn():
            return 42

        assert fn() == 42

    def test_reraises_pipeline_error(self):
        @safe_execute("test")
        def fn():
            raise PipelineError("inner", "already wrapped")

        with pytest.raises(PipelineError, match="inner"):
            fn()

    def test_wraps_generic_exception_and_reraises_as_pipeline_error(self):
        @safe_execute("test_stage")
        def fn():
            raise RuntimeError("something broke")

        with pytest.raises(PipelineError) as exc_info:
            fn()
        assert "test_stage" in str(exc_info.value)

    def test_returns_default_instead_of_raising(self):
        @safe_execute("test", default=-1)
        def fn():
            raise ValueError("bad input")

        result = fn()
        assert result == -1

    def test_file_not_found_wraps_to_pipeline_error(self):
        @safe_execute("loader")
        def fn():
            raise FileNotFoundError("no such file")

        with pytest.raises(PipelineError):
            fn()

    def test_file_not_found_with_default(self):
        """default=None cannot be distinguished from 'no default' - still raises."""
        @safe_execute("loader", default=None)
        def fn():
            raise FileNotFoundError("no such file")

        with pytest.raises(PipelineError):
            fn()

    def test_passes_args_and_kwargs(self):
        @safe_execute("test")
        def fn(a, b, c=10):
            return a + b + c

        assert fn(1, 2, c=3) == 6

    def test_functools_wraps_preserves_name(self):
        @safe_execute("test")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


# ─────────────────────────────────────────────
# validate_dataframe
# ─────────────────────────────────────────────

def _make_valid_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 50_000 + rng.normal(0, 100, n)
    return pd.DataFrame({
        "close": close,
        "volume": rng.exponential(10, n),
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
    })


class TestValidateDataframe:
    def test_valid_df_returns_true(self):
        df = _make_valid_df(200)
        assert validate_dataframe(df, required_cols=["close", "volume"]) is True

    def test_none_raises_pipeline_error(self):
        with pytest.raises(PipelineError, match="None"):
            validate_dataframe(None, required_cols=["close"])

    def test_non_dataframe_raises(self):
        with pytest.raises(PipelineError):
            validate_dataframe([1, 2, 3], required_cols=["close"])

    def test_missing_columns_raises(self):
        df = _make_valid_df(200).drop(columns=["close"])
        with pytest.raises(PipelineError, match="Missing columns"):
            validate_dataframe(df, required_cols=["close", "volume"])

    def test_too_few_rows_raises(self):
        df = _make_valid_df(50)
        with pytest.raises(PipelineError, match="rows"):
            validate_dataframe(df, required_cols=["close"], min_rows=100)

    def test_negative_price_raises(self):
        df = _make_valid_df(200)
        df.loc[10, "close"] = -500.0
        with pytest.raises(PipelineError, match="Negative"):
            validate_dataframe(df, required_cols=["close", "open", "high", "low"])

    def test_all_nan_column_raises(self):
        df = _make_valid_df(200)
        df["close"] = np.nan
        with pytest.raises(PipelineError, match="NaN"):
            validate_dataframe(df, required_cols=["close", "volume"])

    def test_custom_min_rows_threshold(self):
        df = _make_valid_df(10)
        assert validate_dataframe(df, required_cols=["close"], min_rows=5) is True


# ─────────────────────────────────────────────
# validate_order
# ─────────────────────────────────────────────

class TestValidateOrder:
    def test_valid_order_returns_true(self):
        assert validate_order(qty=1.0, horizon=60) is True

    def test_zero_qty_raises(self):
        with pytest.raises(PipelineError):
            validate_order(qty=0.0, horizon=60)

    def test_negative_qty_raises(self):
        with pytest.raises(PipelineError):
            validate_order(qty=-1.0, horizon=60)

    def test_zero_horizon_raises(self):
        with pytest.raises(PipelineError):
            validate_order(qty=1.0, horizon=0)

    def test_negative_horizon_raises(self):
        with pytest.raises(PipelineError):
            validate_order(qty=1.0, horizon=-5)

    def test_large_order_warns_but_passes(self):
        # Should not raise, just warn
        assert validate_order(qty=9999.0, horizon=60) is True

    def test_short_horizon_warns_but_passes(self):
        assert validate_order(qty=1.0, horizon=3) is True


# ─────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────

class TestLoadConfig:
    def test_default_config_loads(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_default_config_has_expected_keys(self):
        cfg = load_config()
        assert "data" in cfg
        assert "simulator" in cfg
        assert "policies" in cfg
        assert "evaluation" in cfg

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_custom_config_loads(self):
        data = {"foo": {"bar": 42}, "baz": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        cfg = load_config(tmp_path)
        assert cfg["foo"]["bar"] == 42
        assert cfg["baz"] == [1, 2, 3]

    def test_malformed_yaml_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("key: [unclosed bracket\n")
            tmp_path = f.name

        with pytest.raises(yaml.YAMLError):
            load_config(tmp_path)


# ─────────────────────────────────────────────
# get_nested
# ─────────────────────────────────────────────

class TestGetNested:
    def setup_method(self):
        self.cfg = {
            "data": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "frequency": "1m",
                "nested": {"deep": {"value": 99}},
            },
            "simulator": {"spread_bps": 1.0},
        }

    def test_top_level_key(self):
        assert get_nested(self.cfg, "simulator") == {"spread_bps": 1.0}

    def test_two_level_key(self):
        assert get_nested(self.cfg, "simulator.spread_bps") == 1.0

    def test_three_level_key(self):
        assert get_nested(self.cfg, "data.nested.deep.value") == 99

    def test_list_value(self):
        result = get_nested(self.cfg, "data.symbols")
        assert result == ["BTCUSDT", "ETHUSDT"]

    def test_missing_top_key_returns_default(self):
        assert get_nested(self.cfg, "nonexistent", default="fallback") == "fallback"

    def test_missing_nested_key_returns_default(self):
        assert get_nested(self.cfg, "data.missing_key", default=0) == 0

    def test_default_is_none_when_not_specified(self):
        assert get_nested(self.cfg, "does.not.exist") is None

    def test_empty_config(self):
        assert get_nested({}, "any.key", default="x") == "x"
