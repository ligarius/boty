from datetime import datetime, timedelta

import pandas as pd

from bot.data.loader import OHLCVRequest, generate_synthetic_data
from bot.strategies.momentum import momentum_signals
from bot.strategies.mean_reversion import mean_reversion_signals


def test_momentum_generates_scores():
    end = datetime.utcnow()
    start = end - timedelta(hours=10)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    signals = momentum_signals(data)
    assert not signals.empty
    assert (signals["score"].abs() >= 0).all()


def test_mean_reversion_generates_signals():
    end = datetime.utcnow()
    start = end - timedelta(hours=10)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    signals = mean_reversion_signals(data)
    assert not signals.empty
    assert set(signals.columns) >= {"signal", "score", "atr"}
