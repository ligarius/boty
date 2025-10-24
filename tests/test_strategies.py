from datetime import datetime, timedelta

import pandas as pd

from bot.data.loader import OHLCVRequest, generate_synthetic_data
from bot.strategies.momentum import momentum_signals
from bot.strategies.mean_reversion import mean_reversion_signals
from bot.strategies.volatility_breakout import volatility_breakout_signals
from bot.strategies.market_making import market_making_signals


def test_momentum_generates_scores():
    end = datetime.utcnow()
    start = end - timedelta(hours=10)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    signals = momentum_signals(data)
    assert not signals.empty
    assert (signals["score"].abs() >= 0).all()
    assert {"ma_fast", "ma_slow"}.issubset(signals.columns)


def test_mean_reversion_generates_signals():
    end = datetime.utcnow()
    start = end - timedelta(hours=10)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    signals = mean_reversion_signals(data)
    assert not signals.empty
    assert set(signals.columns) >= {"signal", "score", "atr"}


def test_volatility_breakout_adds_columns():
    end = datetime.utcnow()
    start = end - timedelta(hours=10)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    signals = volatility_breakout_signals(data)
    assert {"breakout_high", "breakout_low"}.issubset(signals.columns)
    assert signals["signal"].isin([-1, 0, 1]).all()


def test_market_making_signals_include_liquidity():
    end = datetime.utcnow()
    start = end - timedelta(hours=10)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    signals = market_making_signals(data)
    assert "liquidity_score" in signals.columns
    assert signals["signal"].isin([-1, 0, 1]).all()
