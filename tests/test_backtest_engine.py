"""Regression tests for the backtest engine."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from bot.backtest.engine import BacktestEngine, BacktestMetrics

vectorbt_available = importlib.util.find_spec("vectorbt") is not None

pytestmark = pytest.mark.skipif(not vectorbt_available, reason="vectorbt required for this test")


def test_backtest_engine_run_vectorbt() -> None:
    """Ensure the engine can run end-to-end when vectorbt is available."""

    if vectorbt_available:
        __import__("vectorbt")

    periods = 200
    index = pd.date_range("2021-01-01", periods=periods, freq="h")
    close = np.linspace(100, 120, periods) + np.sin(np.linspace(0, 3 * np.pi, periods))
    data = pd.DataFrame(
        {
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=index,
    )

    engine = BacktestEngine()
    metrics = engine.run(data)

    assert isinstance(metrics, BacktestMetrics)
