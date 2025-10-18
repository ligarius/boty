import pandas as pd

from bot.strategies.ensemble import EnsembleSelector, Signal


def test_ensemble_selector_scores_signals():
    selector = EnsembleSelector()
    signals = [
        Signal(symbol="BTCUSDT", timeframe="1m", signal=1, score=1.0, atr=100.0, features={"atr": 100.0}),
        Signal(symbol="ETHUSDT", timeframe="1m", signal=-1, score=0.5, atr=80.0, features={"atr": 80.0}),
    ]
    ranked = selector.select_top_n(signals, 2)
    assert len(ranked) == 2
    assert ranked[0].score >= ranked[1].score
