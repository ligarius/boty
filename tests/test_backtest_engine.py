"""Regression tests for the backtest engine."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from bot.backtest.engine import BacktestEngine, BacktestMetrics
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
    metrics = engine.run(data)

    assert isinstance(metrics, BacktestMetrics)


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
    metrics = engine.run(data)

    assert isinstance(metrics, BacktestMetrics)


def test_backtest_engine_trains_and_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the selector is trained and probabilities reweight the signals."""

    monkeypatch.setattr("bot.backtest.engine.vbt", None, raising=False)

    fit_called = False
    recorded_features: pd.DataFrame | None = None

    def fake_fit(self: SignalSelector, features: pd.DataFrame, labels: pd.Series) -> SelectorReport:
        nonlocal fit_called, recorded_features
        fit_called = True
        recorded_features = features.copy()
        self.fitted = True
        feature_weights = {feature: float(idx + 1) for idx, feature in enumerate(features.columns)}
        return SelectorReport(accuracy=0.75, f1=0.6, feature_importances=feature_weights)

    def fake_predict_proba(self: SignalSelector, features: pd.DataFrame) -> np.ndarray:
        assert fit_called
        if recorded_features is not None:
            assert list(features.columns) == list(recorded_features.columns)
        count = len(features)
        if count == 0:
            return np.array([])
        return np.linspace(0.2, 0.8, num=count)

    monkeypatch.setattr(SignalSelector, "fit", fake_fit, raising=False)
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
    metrics = engine.run(data)

    assert fit_called, "SignalSelector.fit should be invoked"
    assert engine.last_training_report is not None
    assert metrics.training_accuracy == pytest.approx(0.75)
    assert metrics.training_f1 == pytest.approx(0.6)

    probabilities = engine.last_signal_probabilities
    assert probabilities is not None
    assert not probabilities.empty
    expected = np.linspace(0.2, 0.8, num=len(probabilities))
    np.testing.assert_allclose(probabilities["probability"].to_numpy(), expected)
    np.testing.assert_allclose(
        probabilities["weighted_score"].to_numpy(),
        probabilities["score"].to_numpy() * probabilities["probability"].to_numpy(),
    )

    weighted_scores = engine.last_weighted_scores
    assert weighted_scores is not None
    assert weighted_scores.index.equals(data.index)
