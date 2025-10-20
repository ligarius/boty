import numpy as np
import pandas as pd

import bot.backtest.engine as engine_module
from bot.backtest.engine import BacktestEngine


def _synthetic_price_series(periods: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2022-01-01", periods=periods, freq="h")
    trend = np.linspace(0, 1.5, periods)
    oscillation = np.sin(np.linspace(0, 8 * np.pi, periods))
    noise = rng.normal(scale=0.25, size=periods)
    close = 100 + trend + 1.5 * oscillation + noise
    open_price = close + rng.normal(scale=0.05, size=periods)
    width = 0.8 + rng.normal(scale=0.05, size=periods)
    high = open_price + width
    low = open_price - width
    volume = 1000 + 5 * np.cos(np.linspace(0, 4 * np.pi, periods))
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_walk_forward_probabilities_stable_without_future_data():
    engine_module.vbt = None
    data_full = _synthetic_price_series(240)
    cutoff_timestamp = data_full.index[200]
    data_truncated = data_full.loc[:cutoff_timestamp]

    full_engine = BacktestEngine()
    full_engine.selector_window = 40
    full_engine.run(data_full)
    probabilities_full = full_engine.last_signal_probabilities
    assert probabilities_full is not None
    assert not probabilities_full.empty
    probabilities_full = probabilities_full.copy()

    truncated_engine = BacktestEngine()
    truncated_engine.selector_window = 40
    truncated_engine.run(data_truncated)
    probabilities_truncated = truncated_engine.last_signal_probabilities
    assert probabilities_truncated is not None
    assert not probabilities_truncated.empty
    probabilities_truncated = probabilities_truncated.copy()

    merged = probabilities_truncated.merge(
        probabilities_full,
        on=["timestamp", "source", "signal"],
        suffixes=("_truncated", "_full"),
    )

    assert not merged.empty, "Expected overlapping probability records"
    np.testing.assert_allclose(
        merged["probability_truncated"].to_numpy(),
        merged["probability_full"].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
        err_msg="Probabilities changed when future data was removed",
    )
    assert merged["timestamp"].max() == probabilities_truncated["timestamp"].max()
