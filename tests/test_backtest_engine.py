"""Regression tests for the backtest engine."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from bot.backtest.engine import BacktestEngine, BacktestMetrics
from bot.core.config import Settings
from bot.ml.selector import SelectorReport, SignalSelector
from bot.strategies import mean_reversion, momentum

vectorbt_available = importlib.util.find_spec("vectorbt") is not None
requires_vectorbt = pytest.mark.skipif(not vectorbt_available, reason="vectorbt required for this test")


@requires_vectorbt
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
    metrics = engine.run(data, timeframe="1h")

    assert isinstance(metrics, BacktestMetrics)


def test_backtest_engine_reads_selector_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure BacktestEngine picks selector configuration from Settings."""

    custom_settings = Settings(
        selector_threshold=0.8,
        selector_horizon=50,
        selector_window=2000,
    )

    monkeypatch.setattr("bot.backtest.engine.get_settings", lambda: custom_settings)

    engine = BacktestEngine()

    assert engine.selector_threshold == pytest.approx(custom_settings.selector_threshold)
    assert engine.selector_horizon == custom_settings.selector_horizon
    assert engine.selector_window == custom_settings.selector_window


@requires_vectorbt
def test_backtest_engine_masks_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test ensuring warm-up rows are trimmed before vectorbt execution."""

    if not vectorbt_available:
        pytest.skip("vectorbt required for this test")

    import vectorbt as vbt

    periods = 160
    index = pd.date_range("2022-01-01", periods=periods, freq="h")
    base = np.linspace(50, 75, periods)
    noise = np.sin(np.linspace(0, 12 * np.pi, periods))
    close = base + noise
    data = pd.DataFrame(
        {
            "open": close + 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(periods, 500.0),
        },
        index=index,
    )

    momentum_df = momentum.momentum_signals(data).reindex(data.index)
    mean_df = mean_reversion.mean_reversion_signals(data).reindex(data.index)
    momentum_ready = momentum_df.drop(columns=["signal", "score"], errors="ignore").notna().all(axis=1)
    mean_ready = mean_df.drop(columns=["signal", "score"], errors="ignore").notna().all(axis=1)
    valid_mask = (momentum_ready & mean_ready).reindex(data.index, fill_value=False)
    if valid_mask.any():
        expected_index = data.loc[valid_mask].index
        assert expected_index[0] != data.index[0]
    else:
        expected_index = data.index

    original_from_signals = vbt.Portfolio.from_signals

    def checking(price, entries, exits, **kwargs):  # type: ignore[no-untyped-def]
        assert len(price) == len(entries) == len(exits)
        assert not isinstance(entries, np.ndarray)  # ensure we keep pandas alignment
        assert not isinstance(exits, np.ndarray)
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert isinstance(price, pd.Series)
        assert entries.index.equals(price.index)
        assert exits.index.equals(price.index)
        assert price.index.equals(expected_index)
        assert entries.dtype == bool
        assert exits.dtype == bool
        assert not entries.isna().any()
        assert not exits.isna().any()
        return original_from_signals(price, entries=entries, exits=exits, **kwargs)

    monkeypatch.setattr(vbt.Portfolio, "from_signals", checking)

    engine = BacktestEngine()
    metrics = engine.run(data, timeframe="1h")

    assert isinstance(metrics, BacktestMetrics)


def test_backtest_engine_trains_and_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the selector is trained and probabilities reweight the signals."""

    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)

    fit_calls = 0

    def fake_fit_ordered(
        self: SignalSelector, features: pd.DataFrame, labels: pd.Series, *, n_splits: int = 5
    ) -> SelectorReport:
        nonlocal fit_calls
        fit_calls += 1
        self.fitted = True
        assert "trend" in features.columns, "Momentum features should expose the trend signal"
        trend_values = features["trend"].astype(float)
        assert np.any(np.abs(trend_values.to_numpy()) > 0), "Trend feature should carry directional information"
        feature_weights = {feature: float(idx + 1) for idx, feature in enumerate(features.columns)}
        self.decision_threshold = 0.6
        return SelectorReport(
            accuracy=0.75,
            f1=0.6,
            precision=0.55,
            recall=0.5,
            threshold=0.6,
            feature_importances=feature_weights,
        )

    def fake_predict_proba(self: SignalSelector, features: pd.DataFrame) -> np.ndarray:
        assert fit_calls > 0
        count = len(features)
        if count == 0:
            return np.array([])
        return np.full(count, 0.7)

    monkeypatch.setattr(SignalSelector, "fit_ordered", fake_fit_ordered, raising=False)
    monkeypatch.setattr(SignalSelector, "predict_proba", fake_predict_proba, raising=False)

    periods = 240
    index = pd.date_range("2023-01-01", periods=periods, freq="h")
    base = np.linspace(100, 140, periods)
    noise = np.sin(np.linspace(0, 8 * np.pi, periods)) * 2
    close = base + noise
    data = pd.DataFrame(
        {
            "open": close + 0.3,
            "high": close + 0.9,
            "low": close - 0.9,
            "close": close,
            "volume": np.full(periods, 1500.0),
        },
        index=index,
    )

    engine = BacktestEngine()
    metrics = engine.run(data, timeframe="1h")

    assert fit_calls > 0, "SignalSelector.fit_ordered should be invoked"
    assert engine.last_training_report is not None
    assert metrics.training_accuracy == pytest.approx(0.75)
    assert metrics.training_f1 == pytest.approx(0.6)
    assert metrics.training_precision == pytest.approx(0.55)
    assert metrics.training_recall == pytest.approx(0.5)
    assert metrics.selector_threshold == pytest.approx(0.6)
    assert metrics.training_accuracy > 0.5

    probabilities = engine.last_signal_probabilities
    assert probabilities is not None
    assert not probabilities.empty
    assert set(probabilities["probability"].unique()).issubset({1.0, 0.7})
    assert (probabilities["probability"] == 0.7).any()
    np.testing.assert_allclose(
        probabilities["weighted_score"].to_numpy(),
        probabilities["score"].to_numpy() * probabilities["probability"].to_numpy(),
    )

    weighted_scores = engine.last_weighted_scores
    assert weighted_scores is not None
    assert weighted_scores.index.equals(data.index)
    raw_scores = engine.last_weighted_scores_raw
    assert raw_scores is not None
    assert raw_scores.index.equals(data.index)
    max_abs = raw_scores.abs().max()
    if max_abs > 0:
        np.testing.assert_allclose(weighted_scores.to_numpy(), raw_scores.to_numpy() / max_abs)
        assert pytest.approx(1.0) == weighted_scores.abs().max()
    else:
        np.testing.assert_allclose(weighted_scores.to_numpy(), raw_scores.to_numpy())


def test_backtest_engine_accepts_strategy_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure custom strategy parameters are forwarded to the strategy functions."""

    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)

    periods = 120
    index = pd.date_range("2023-02-01", periods=periods, freq="h")
    base = np.linspace(100, 120, periods)
    data = pd.DataFrame(
        {
            "open": base + 0.1,
            "high": base + 0.3,
            "low": base - 0.3,
            "close": base,
            "volume": np.full(periods, 1000.0),
        },
        index=index,
    )

    momentum_calls: dict[str, object] = {}
    mean_calls: dict[str, object] = {}

    original_momentum = momentum.momentum_signals
    original_mean = mean_reversion.mean_reversion_signals

    def wrapped_momentum(df, **kwargs):  # type: ignore[no-untyped-def]
        momentum_calls.update(kwargs)
        return original_momentum(df, **kwargs)

    def wrapped_mean(df, **kwargs):  # type: ignore[no-untyped-def]
        mean_calls.update(kwargs)
        return original_mean(df, **kwargs)

    monkeypatch.setattr(momentum, "momentum_signals", wrapped_momentum)
    monkeypatch.setattr(mean_reversion, "mean_reversion_signals", wrapped_mean)

    engine = BacktestEngine()
    engine.selector_threshold = 0.05

    metrics = engine.run(
        data,
        timeframe="1h",
        momentum_params={"fast": 7, "slow": 55, "adx_period": 16},
        mean_reversion_params={"window": 22, "z_threshold": 1.9},
    )

    assert isinstance(metrics, BacktestMetrics)
    assert momentum_calls == {"fast": 7, "slow": 55, "adx_period": 16}
    assert mean_calls == {"window": 22, "z_threshold": 1.9}


def test_backtest_engine_normalizes_and_generates_trades(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic scenario where normalized scores trigger trades and metrics are non-zero."""

    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)

    custom_settings = Settings(selector_threshold=0.2, selector_horizon=1, selector_window=10)
    monkeypatch.setattr("bot.backtest.engine.get_settings", lambda: custom_settings)

    def fake_momentum_signals(data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "feature": 0.5,
                "atr": 1000.0,
                "signal": 0,
                "score": 0.0,
            },
            index=data.index,
        )
        long_idx = data.index[10]
        short_idx = data.index[80]
        df.loc[long_idx, "signal"] = 1
        df.loc[long_idx, "score"] = 1e-4
        df.loc[short_idx, "signal"] = -1
        df.loc[short_idx, "score"] = -1e-4
        return df[["signal", "score", "atr", "feature"]]

    def fake_mean_reversion_signals(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal": np.zeros(len(data), dtype=int),
                "score": np.zeros(len(data), dtype=float),
                "atr": np.full(len(data), 1000.0),
            },
            index=data.index,
        )

    monkeypatch.setattr(momentum, "momentum_signals", fake_momentum_signals, raising=False)
    monkeypatch.setattr(mean_reversion, "mean_reversion_signals", fake_mean_reversion_signals, raising=False)

    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    first_leg = np.linspace(100, 130, periods // 2, endpoint=False)
    second_leg = np.linspace(130, 110, periods // 2)
    close = np.concatenate([first_leg, second_leg])
    close = close + 0.2 * np.sin(np.linspace(0, 4 * np.pi, periods))
    data = pd.DataFrame(
        {
            "open": close + 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(periods, 1200.0),
        },
        index=index,
    )

    engine = BacktestEngine()
    metrics = engine.run(data, timeframe="1h")

    normalized_scores = engine.last_weighted_scores
    raw_scores = engine.last_weighted_scores_raw
    assert normalized_scores is not None
    assert raw_scores is not None
    assert normalized_scores.index.equals(data.index)
    assert raw_scores.index.equals(data.index)

    max_abs = raw_scores.abs().max()
    assert max_abs > 0
    np.testing.assert_allclose(normalized_scores.to_numpy(), raw_scores.to_numpy() / max_abs)
    assert normalized_scores.abs().max() == pytest.approx(1.0)
    assert max_abs < engine.selector_threshold
    assert (normalized_scores > engine.selector_threshold).any()
    assert (normalized_scores < -engine.selector_threshold).any()

    assert metrics.roi != 0.0
    assert metrics.sharpe != 0.0
    assert metrics.profit_factor != 0.0
    assert metrics.win_rate != 0.0


def test_backtest_engine_last_signals_preserve_timeframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Signals produced by the engine should retain the timeframe provided to run()."""

    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)

    def fake_momentum_signals(data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "signal": np.zeros(len(data), dtype=int),
                "score": np.zeros(len(data), dtype=float),
                "atr": np.full(len(data), 500.0),
                "feature_a": np.linspace(0.1, 1.0, len(data)),
            },
            index=data.index,
        )
        trigger_idx = data.index[6]
        df.loc[trigger_idx, "signal"] = 1
        df.loc[trigger_idx, "score"] = 0.5
        return df

    def fake_mean_reversion_signals(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal": np.zeros(len(data), dtype=int),
                "score": np.zeros(len(data), dtype=float),
                "atr": np.full(len(data), 400.0),
                "feature_b": np.linspace(1.0, 0.1, len(data)),
            },
            index=data.index,
        )

    monkeypatch.setattr(momentum, "momentum_signals", fake_momentum_signals, raising=False)
    monkeypatch.setattr(mean_reversion, "mean_reversion_signals", fake_mean_reversion_signals, raising=False)

    periods = 32
    index = pd.date_range("2024-06-01", periods=periods, freq="h")
    close = np.linspace(100, 120, periods)
    data = pd.DataFrame(
        {
            "open": close + 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(periods, 900.0),
        },
        index=index,
    )

    engine = BacktestEngine()
    assert engine.settings.timeframes, "Expected default timeframes"
    non_default_timeframe = "45m"
    assert non_default_timeframe not in {engine.settings.timeframes[0]}, "Test requires a non-default timeframe"

    engine.run(data, timeframe=non_default_timeframe)

    assert engine.last_signals, "Expected at least one generated signal"
    assert all(signal.timeframe == non_default_timeframe for signal in engine.last_signals)
