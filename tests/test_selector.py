import pandas as pd
import pytest

from bot.ml.selector import SignalSelector
from bot.strategies.ensemble import Signal


def test_signal_selector_scores_using_pipeline(tmp_path):
    model_path = tmp_path / "selector.joblib"

    features = pd.DataFrame(
        [
            {"atr": 1.0, "rsi": 30.0},
            {"atr": 2.0, "rsi": 50.0},
            {"atr": 1.5, "rsi": 70.0},
            {"atr": 0.5, "rsi": 40.0},
        ]
    )
    labels = pd.Series([1, 0, 1, 0])

    selector = SignalSelector(str(model_path))
    selector.fit(features, labels)

    reloaded_selector = SignalSelector(str(model_path))
    reloaded_selector.load()

    signals = [
        Signal(
            symbol=f"asset-{idx}",
            timeframe="1m",
            signal=1,
            score=float(idx + 1),
            atr=1.0,
            features=row.to_dict(),
        )
        for idx, row in features.iterrows()
    ]

    expected_probabilities = reloaded_selector.pipeline.predict_proba(features)[:, 1]
    scored_signals = reloaded_selector.score_signals(signals)

    scored_by_symbol = {sig.symbol: sig for sig in scored_signals}

    for idx, (signal, expected_proba) in enumerate(zip(signals, expected_probabilities)):
        scored_signal = scored_by_symbol[signal.symbol]
        assert scored_signal.features["meta_probability"] == pytest.approx(float(expected_proba))
        assert scored_signal.score == pytest.approx(signal.score * float(expected_proba))
