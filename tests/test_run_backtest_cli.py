"""Tests for the run_backtest CLI helper."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from bot.backtest.engine import BacktestMetrics
from bot.scripts import run_backtest_cli as cli


def test_run_backtest_uses_real_loader(monkeypatch) -> None:
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = start + timedelta(minutes=5)
    index = pd.date_range(start=start, periods=5, freq="1min")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 10, 10, 10, 10],
        },
        index=index,
    )

    fetch_called: dict[str, object] = {}

    def fake_fetch(request, settings=None):  # type: ignore[override]
        fetch_called["request"] = request
        return df

    monkeypatch.setattr(cli, "fetch_binance_ohlcv", fake_fetch)

    engine = MagicMock()
    metrics = BacktestMetrics(roi=0.1, sharpe=1.5, max_drawdown=0.05, profit_factor=1.4, win_rate=0.55)
    engine.run.return_value = metrics
    engine.last_training_report = None
    engine.meets_go_live.return_value = True
    monkeypatch.setattr(cli, "BacktestEngine", lambda: engine)

    fake_settings = SimpleNamespace(default_data_source="binance", data_source_csv_path=None)

    payload = cli.run_backtest(
        "BTCUSDT",
        "1m",
        start,
        end,
        settings=fake_settings,  # type: ignore[arg-type]
    )

    assert "request" in fetch_called
    call_args = engine.run.call_args[0]
    passed_df = call_args[0]
    assert isinstance(passed_df, pd.DataFrame)
    assert passed_df.index.is_monotonic_increasing
    assert passed_df.index.equals(df.index)
    assert call_args[1] == "1m"
    assert payload["metrics"]["roi"] == metrics.roi
    engine.meets_go_live.assert_called_once_with(metrics)
