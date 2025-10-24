"""Meta strategy combining base signals using ML selector."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

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
    strategy: Optional[str] = None


@dataclass
class StrategyPerformance:
    roi: float = 0.0
    sharpe: float = 0.0
    drawdown: float = 0.0
    samples: int = 0


class StrategyPerformanceTracker:
    """Keep exponential moving averages of per-strategy performance."""

    def __init__(self, decay: float = 0.9):
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self._stats: Dict[str, StrategyPerformance] = {}

    def update(self, strategy: str, roi: float, sharpe: float, drawdown: float) -> None:
        stats = self._stats.setdefault(strategy, StrategyPerformance())
        stats.samples += 1
        alpha = 1.0 - self.decay
        stats.roi = stats.roi * self.decay + roi * alpha
        stats.sharpe = stats.sharpe * self.decay + sharpe * alpha
        stats.drawdown = stats.drawdown * self.decay + drawdown * alpha

    def weight(self, strategy: Optional[str]) -> float:
        if not strategy or strategy not in self._stats:
            return 1.0
        stats = self._stats[strategy]
        base = 1.0 + stats.roi + stats.sharpe * 0.5 - stats.drawdown * 2.0
        return float(max(0.1, base))


class EnsembleSelector:
    """Select top signals using a simple logistic regression meta-model."""

    def __init__(self, decay: float = 0.9) -> None:
        self.model = LogisticRegression()
        self.is_trained = False
        self.performance = StrategyPerformanceTracker(decay=decay)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if features.empty:
            raise ValueError("No features provided")
        self.model.fit(features, labels)
        self.is_trained = True

    def update_performance(self, strategy: str, metrics: Dict[str, float]) -> None:
        roi = float(metrics.get("roi", 0.0))
        sharpe = float(metrics.get("sharpe", 0.0))
        drawdown = float(metrics.get("max_drawdown", 0.0))
        self.performance.update(strategy, roi, sharpe, drawdown)

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
            weight = self.performance.weight(sig.strategy)
            enhanced.append(
                Signal(
                    symbol=sig.symbol,
                    timeframe=sig.timeframe,
                    signal=sig.signal,
                    score=float(sig.score * proba * weight),
                    atr=sig.atr,
                    features=dict(sig.features)
                    | {"meta_probability": float(proba), "performance_weight": weight},
                    strategy=sig.strategy,
                )
            )
        return sorted(enhanced, key=lambda sig: sig.score, reverse=True)

    def select_top_n(self, signals: Iterable[Signal], top_n: int) -> List[Signal]:
        scored = self.score_signals(signals)
        return scored[:top_n]


def signal_from_row(
    symbol: str,
    timeframe: str,
    row: pd.Series | Dict[str, float],
    *,
    strategy: Optional[str] = None,
) -> Signal:
    if isinstance(row, pd.Series):
        iterable = row.to_dict()
    else:
        iterable = row
    excluded = {"signal", "score"}
    features = {col: float(value) for col, value in iterable.items() if col not in excluded}
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        signal=int(iterable["signal"]),
        score=float(iterable["score"]),
        atr=float(iterable.get("atr", np.nan)),
        features=features,
        strategy=strategy,
    )
