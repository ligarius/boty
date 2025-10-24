from datetime import datetime, timedelta, timezone

import optuna
import pandas as pd
import pytest

from bot.backtest.engine import BacktestEngine, BacktestMetrics
from bot.ml.live_optimizer import AutoTuneResult, tune_intraday_settings
from bot.ml.tuner import CandidateResult
from bot.scripts import auto_tune_cli


def _ensure_vectorbt_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)


def _stub_result() -> AutoTuneResult:
    return AutoTuneResult(
        study_name="stub",
        trials=1,
        best_score=1.2345,
        baseline_score=0.9876,
        best_params={
            "momentum_fast": 11,
            "momentum_slow": 37,
            "mean_window": 25,
            "selector_window": 64,
            "selector_threshold": 0.1234,
        },
        metrics={"roi": 0.05, "sharpe": 0.5},
        training={"accuracy": 0.7},
        go_live_ready=True,
        passes_thresholds=True,
        baseline_metrics={"roi": 0.01},
        baseline_training=None,
    )


def test_tune_intraday_settings_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 1, 1)
    end = start + timedelta(hours=6)

    dummy_data = pd.DataFrame(
        {"close": [100.0 + i for i in range(120)]},
        index=pd.date_range("2023-01-01", periods=120, freq="T"),
    )

    monkeypatch.setattr("bot.ml.live_optimizer._load_data", lambda *args, **kwargs: dummy_data)

    def fake_run(self: BacktestEngine, data, timeframe=None, **kwargs):  # type: ignore[override]
        self.last_training_report = None
        self.last_resolved_selector_window = 64
        self.last_resolved_selector_threshold = 0.15
        return BacktestMetrics(
            roi=0.12,
            sharpe=0.3,
            max_drawdown=0.04,
            profit_factor=1.7,
            win_rate=0.6,
            training_accuracy=0.8,
            training_f1=0.75,
        )

    monkeypatch.setattr(BacktestEngine, "run", fake_run)

    def fake_optimize(self, objective, n_trials: int = 1):  # type: ignore[override]
        params = {
            "momentum_fast": 12,
            "momentum_slow": 40,
            "mean_window": 20,
            "mean_z_threshold": 1.2,
            "selector_threshold": 0.15,
            "selector_window": 64,
            "selector_horizon": 5,
        }
        metrics = {
            "value": 1.5,
            "metrics": {
                "roi": 0.12,
                "sharpe": 0.3,
                "max_drawdown": 0.04,
                "profit_factor": 1.7,
                "win_rate": 0.6,
                "training_accuracy": 0.8,
                "training_f1": 0.75,
            },
            "training": {"training_accuracy": 0.8, "training_f1": 0.75},
            "go_live_ready": True,
            "baseline": 0.9,
        }
        return CandidateResult(params=params, metrics=metrics, is_promoted=True, passes_thresholds=True)

    monkeypatch.setattr(
        "bot.ml.live_optimizer.EvolutionaryTuner.optimize",
        fake_optimize,
    )

    result = tune_intraday_settings(
        "BTCUSDT",
        "1m",
        start,
        end,
        data_source="synthetic",
        n_trials=1,
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

    result = _stub_result()

    monkeypatch.setattr(
        auto_tune_cli,
        "tune_intraday_settings",
        lambda *args, **kwargs: result,
    )

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
    assert "Applied parameters (resolved)" in captured.out
    assert "Optimized metrics" in captured.out
    assert "Passes thresholds" in captured.out

    for key, value in sorted(result.best_params.items()):
        if isinstance(value, (int, float)):
            expected = f"  - {key}: {float(value):.4f}"
        else:
            expected = f"  - {key}: {value}"
        assert expected in captured.out


def test_auto_tune_cli_accepts_z_datetime(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 4, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)

    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    result = _stub_result()

    def fake_tune(symbol: str, timeframe: str, start_arg: str, end_arg: str, **_: object) -> AutoTuneResult:
        assert start_arg.endswith("Z")
        assert end_arg.endswith("Z")
        return result

    monkeypatch.setattr(auto_tune_cli, "tune_intraday_settings", fake_tune)

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
    assert "Applied parameters (resolved)" in captured.out
    assert "Optimized metrics" in captured.out
    assert "Passes thresholds" in captured.out


def test_tune_intraday_settings_accepts_z_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 5, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    dummy_data = pd.DataFrame(
        {"close": [200.0 + i for i in range(60)]},
        index=pd.date_range("2023-05-01", periods=60, freq="T"),
    )

    captured: dict[str, datetime] = {}

    def fake_generate(request):
        captured["start"] = request.start
        captured["end"] = request.end
        return dummy_data

    monkeypatch.setattr("bot.ml.live_optimizer.generate_synthetic_data", fake_generate)

    def fake_run(self: BacktestEngine, data, timeframe=None, **kwargs):  # type: ignore[override]
        self.last_training_report = None
        self.last_resolved_selector_window = 50
        self.last_resolved_selector_threshold = 0.2
        return BacktestMetrics(
            roi=0.05,
            sharpe=0.1,
            max_drawdown=0.03,
            profit_factor=1.4,
            win_rate=0.55,
            training_accuracy=0.7,
            training_f1=0.65,
        )

    monkeypatch.setattr(BacktestEngine, "run", fake_run)

    def fake_optimize(self, objective, n_trials: int = 1):  # type: ignore[override]
        params = {
            "momentum_fast": 15,
            "momentum_slow": 45,
            "mean_window": 18,
            "mean_z_threshold": 1.1,
            "selector_threshold": 0.2,
            "selector_window": 50,
            "selector_horizon": 3,
        }
        metrics = {
            "value": 0.9,
            "metrics": {
                "roi": 0.05,
                "sharpe": 0.1,
                "max_drawdown": 0.03,
                "profit_factor": 1.4,
                "win_rate": 0.55,
                "training_accuracy": 0.7,
                "training_f1": 0.65,
            },
            "training": {"training_accuracy": 0.7, "training_f1": 0.65},
            "go_live_ready": False,
            "baseline": 0.5,
        }
        return CandidateResult(params=params, metrics=metrics, is_promoted=False, passes_thresholds=False)

    monkeypatch.setattr(
        "bot.ml.live_optimizer.EvolutionaryTuner.optimize",
        fake_optimize,
    )

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
    assert captured["start"].tzinfo is None
    assert captured["end"].tzinfo is None


def test_tune_intraday_settings_records_resolved_selector_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    recorded_window = 42
    recorded_threshold = 0.3141

    dummy_data = pd.DataFrame(
        {"close": [100.0 + i for i in range(30)]},
        index=pd.date_range("2023-01-01", periods=30, freq="T"),
    )

    monkeypatch.setattr("bot.ml.live_optimizer._load_data", lambda *args, **kwargs: dummy_data)

    def fake_run(self: BacktestEngine, data, timeframe=None, **kwargs):  # type: ignore[override]
        self.last_training_report = None
        self.last_resolved_selector_window = recorded_window
        self.last_resolved_selector_threshold = recorded_threshold
        return BacktestMetrics(
            roi=0.1,
            sharpe=0.2,
            max_drawdown=0.05,
            profit_factor=1.5,
            win_rate=0.55,
            training_accuracy=None,
            training_f1=None,
        )

    monkeypatch.setattr(BacktestEngine, "run", fake_run)

    def fake_optimize(self, objective, n_trials: int = 1):  # type: ignore[override]
        trial = optuna.trial.FixedTrial(
            {
                "momentum_fast": 14,
                "momentum_slow": 38,
                "momentum_adx_period": 12,
                "mean_window": 19,
                "mean_z_threshold": 1.25,
                "selector_threshold": 0.2,
                "selector_window": 30,
                "selector_horizon": 4,
            }
        )
        value = objective(trial)
        metrics = {"value": value}
        metrics.update(trial.user_attrs)
        assert trial.user_attrs["resolved_selector_window"] == recorded_window
        assert trial.user_attrs["resolved_selector_threshold"] == pytest.approx(
            recorded_threshold
        )
        resolved = trial.user_attrs["resolved_params"]
        return CandidateResult(params=resolved, metrics=metrics, is_promoted=True, passes_thresholds=True)

    monkeypatch.setattr(
        "bot.ml.live_optimizer.EvolutionaryTuner.optimize",
        fake_optimize,
    )

    start = datetime(2023, 7, 1)
    end = start + timedelta(hours=2)

    result = tune_intraday_settings(
        "BTCUSDT",
        "1m",
        start,
        end,
        data_source="synthetic",
        n_trials=1,
    )

    assert result.best_params["selector_window"] == recorded_window
    assert result.best_params["selector_threshold"] == pytest.approx(recorded_threshold)


def test_tune_intraday_settings_accepts_aware_datetimes(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vectorbt_fallback(monkeypatch)

    start = datetime(2023, 6, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=3)

    dummy_data = pd.DataFrame(
        {"close": [150.0 + i for i in range(90)]},
        index=pd.date_range("2023-06-01", periods=90, freq="T"),
    )

    captured: dict[str, datetime] = {}

    def fake_generate(request):
        captured["start"] = request.start
        captured["end"] = request.end
        return dummy_data

    monkeypatch.setattr("bot.ml.live_optimizer.generate_synthetic_data", fake_generate)

    def fake_run(self: BacktestEngine, data, timeframe=None, **kwargs):  # type: ignore[override]
        self.last_training_report = None
        self.last_resolved_selector_window = 70
        self.last_resolved_selector_threshold = 0.25
        return BacktestMetrics(
            roi=0.07,
            sharpe=0.12,
            max_drawdown=0.02,
            profit_factor=1.3,
            win_rate=0.5,
            training_accuracy=0.65,
            training_f1=0.6,
        )

    monkeypatch.setattr(BacktestEngine, "run", fake_run)

    def fake_optimize(self, objective, n_trials: int = 1):  # type: ignore[override]
        params = {
            "momentum_fast": 9,
            "momentum_slow": 35,
            "mean_window": 22,
            "mean_z_threshold": 1.3,
            "selector_threshold": 0.25,
            "selector_window": 70,
            "selector_horizon": 4,
        }
        metrics = {
            "value": 1.1,
            "metrics": {
                "roi": 0.07,
                "sharpe": 0.12,
                "max_drawdown": 0.02,
                "profit_factor": 1.3,
                "win_rate": 0.5,
                "training_accuracy": 0.65,
                "training_f1": 0.6,
            },
            "training": {"training_accuracy": 0.65, "training_f1": 0.6},
            "go_live_ready": True,
            "baseline": 0.4,
        }
        return CandidateResult(params=params, metrics=metrics, is_promoted=True, passes_thresholds=True)

    monkeypatch.setattr(
        "bot.ml.live_optimizer.EvolutionaryTuner.optimize",
        fake_optimize,
    )

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
    assert captured["start"].tzinfo is None
    assert captured["end"].tzinfo is None

