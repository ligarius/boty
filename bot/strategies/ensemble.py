"""Meta strategy combining base signals using ML selector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class Signal:
    symbol: str
    timeframe: str
    signal: int
    score: float
    atr: float
    features: Dict[str, float]


class EnsembleSelector:
    """Select top signals using a simple logistic regression meta-model."""

    def __init__(self) -> None:
        self.model = LogisticRegression()
        self.is_trained = False

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if features.empty:
            raise ValueError("No features provided")
        self.model.fit(features, labels)
        self.is_trained = True

    def score_signals(self, signals: Iterable[Signal]) -> List[Signal]:
        signals_list = list(signals)
        if not signals_list:
            return []
        if not self.is_trained:
            return sorted(signals_list, key=lambda sig: sig.score, reverse=True)
        features = pd.DataFrame([sig.features for sig in signals_list])
        probabilities = self.model.predict_proba(features)[:, 1]
        enhanced: List[Signal] = []
        for sig, proba in zip(signals_list, probabilities):
            enhanced.append(
                Signal(
                    symbol=sig.symbol,
                    timeframe=sig.timeframe,
                    signal=sig.signal,
                    score=float(sig.score * proba),
                    atr=sig.atr,
                    features=dict(sig.features) | {"meta_probability": float(proba)},
                )
            )
        return sorted(enhanced, key=lambda sig: sig.score, reverse=True)

    def select_top_n(self, signals: Iterable[Signal], top_n: int) -> List[Signal]:
        scored = self.score_signals(signals)
        return scored[:top_n]


def signal_from_row(symbol: str, timeframe: str, row: pd.Series | Dict[str, float]) -> Signal:
    if isinstance(row, pd.Series):
        iterable = row.to_dict()
    else:
        iterable = row
    features = {col: float(value) for col, value in iterable.items() if col not in {"signal", "score"}}
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        signal=int(iterable["signal"]),
        score=float(iterable["score"]),
        atr=float(iterable.get("atr", np.nan)),
        features=features,
    )
