from datetime import datetime, timedelta, timezone

import pytest

from bot.ml.live_optimizer import AutoTuneResult, tune_intraday_settings
from bot.scripts import auto_tune_cli


def _ensure_vectorbt_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)


def test_tune_intraday_settings_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 1, 1)
    end = start + timedelta(hours=6)

    result = tune_intraday_settings(
        "BTCUSDT",
        "1m",
        start,
        end,
        data_source="synthetic",
        n_trials=2,
    )

    assert isinstance(result, AutoTuneResult)
    assert result.best_params
    assert "momentum_fast" in result.best_params
    assert result.metrics
    assert "roi" in result.metrics
    assert result.baseline_metrics
    assert result.best_score != pytest.approx(0.0) or result.metrics["roi"] == pytest.approx(0.0)


def test_auto_tune_cli_prints_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 3, 1)
    end = start + timedelta(hours=4)

    auto_tune_cli.main(
        [
            "BTCUSDT",
            "1m",
            start.isoformat(),
            end.isoformat(),
            "--data-source",
            "synthetic",
            "--trials",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert "Auto-tune summary" in captured.out
    assert "Best parameters" in captured.out
    assert "Optimized metrics" in captured.out


def test_auto_tune_cli_accepts_z_datetime(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 4, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)

    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    auto_tune_cli.main(
        [
            "BTCUSDT",
            "1m",
            start_str,
            end_str,
            "--data-source",
            "synthetic",
            "--trials",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert "Auto-tune summary" in captured.out
    assert "Best parameters" in captured.out
    assert "Optimized metrics" in captured.out


def test_tune_intraday_settings_accepts_z_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 5, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    result = tune_intraday_settings(
        "ETHUSDT",
        "5m",
        start_str,
        end_str,
        data_source="synthetic",
        n_trials=1,
    )

    assert isinstance(result, AutoTuneResult)
    assert result.metrics
    assert "roi" in result.metrics
    assert result.baseline_metrics


def test_tune_intraday_settings_accepts_aware_datetimes(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 6, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=3)

    result = tune_intraday_settings(
        "BTCUSDT",
        "1m",
        start,
        end,
        data_source="synthetic",
        n_trials=1,
    )

    assert isinstance(result, AutoTuneResult)
    assert result.metrics
    assert result.baseline_metrics
