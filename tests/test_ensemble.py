import pandas as pd

from bot.strategies.ensemble import EnsembleSelector, Signal


def test_ensemble_selector_scores_signals():
    selector = EnsembleSelector()
    signals = [
        Signal(
            symbol="BTCUSDT",
            timeframe="1m",
            signal=1,
            score=1.0,
            atr=100.0,
            features={"atr": 100.0},
            strategy="momentum",
        ),
        Signal(
            symbol="ETHUSDT",
            timeframe="1m",
            signal=-1,
            score=0.5,
            atr=80.0,
            features={"atr": 80.0},
            strategy="mean_reversion",
        ),
    ]
    ranked = selector.select_top_n(signals, 2)
    assert len(ranked) == 2
    assert ranked[0].score >= ranked[1].score


def test_performance_weighting_prioritizes_strong_strategy():
    selector = EnsembleSelector(decay=0.5)
    selector.is_trained = True

    good_signal = Signal(
        symbol="BTCUSDT",
        timeframe="1m",
        signal=1,
        score=0.8,
        atr=50.0,
        features={"feature": 1.0},
        strategy="momentum",
    )
    weak_signal = Signal(
        symbol="ETHUSDT",
        timeframe="1m",
        signal=-1,
        score=0.8,
        atr=60.0,
        features={"feature": 1.0},
        strategy="mean_reversion",
    )

    selector.update_performance("momentum", {"roi": 0.1, "sharpe": 1.5, "max_drawdown": 0.05})
    selector.update_performance("mean_reversion", {"roi": -0.05, "sharpe": -0.5, "max_drawdown": 0.2})

    # simulate fitted model probability output by monkeypatching model
    selector.model = type("Dummy", (), {"predict_proba": lambda self, X: [[0.4, 0.6]] * len(X)})()

    ranked = selector.score_signals([good_signal, weak_signal])
    assert ranked[0].strategy == "momentum"
