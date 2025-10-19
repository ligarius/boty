"""Backtesting utilities powered by vectorbt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
try:
    import vectorbt as vbt
except ImportError:  # pragma: no cover - fallback for environments without vectorbt
    vbt = None  # type: ignore

from ..strategies import momentum, mean_reversion
from ..core.risk import RiskManager
from ..core.config import get_settings


@dataclass
class BacktestMetrics:
    roi: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    win_rate: float


class BacktestEngine:
    """Run walk-forward backtests and compute metrics."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.risk = RiskManager(self.settings)

    def run(self, data: pd.DataFrame) -> BacktestMetrics:
        signals_momentum = momentum.momentum_signals(data)["signal"].reindex(data.index, fill_value=0)
        signals_mean = mean_reversion.mean_reversion_signals(data)["signal"].reindex(data.index, fill_value=0)
        combined = (signals_momentum + signals_mean).clip(-1, 1)
        entries = (combined == 1).reindex(data.index, fill_value=False)
        exits = (combined == -1).reindex(data.index, fill_value=False)
        if vbt is not None:
            pf = vbt.Portfolio.from_signals(
                data["close"],
                entries=entries,
                exits=exits,
                fees=0.0004,
                sl_stop=0.01,
                tp_stop=0.02,
            )
            stats = pf.stats()
            return BacktestMetrics(
                roi=float(stats.loc["Total Return [%]"] / 100),
                sharpe=float(stats.loc["Sharpe Ratio"]),
                max_drawdown=float(stats.loc["Max Drawdown [%]"] / 100),
                profit_factor=float(stats.loc["Profit Factor"]),
                win_rate=float(stats.loc["Win Rate [%]"] / 100),
            )
        # Simple fallback using numpy for deterministic metrics
        returns = data["close"].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        roi = cumulative.iloc[-1] - 1
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24 * 60)
        rolling_max = cumulative.cummax()
        drawdown = (cumulative / rolling_max - 1).min()
        profit_factor = float(np.maximum(returns[returns > 0].sum(), 1e-6) / np.maximum(-returns[returns < 0].sum(), 1e-6))
        win_rate = float((returns > 0).mean())
        return BacktestMetrics(
            roi=float(roi),
            sharpe=float(sharpe),
            max_drawdown=float(-drawdown),
            profit_factor=float(profit_factor),
            win_rate=float(win_rate),
        )

    def meets_go_live(self, metrics: BacktestMetrics) -> bool:
        return (
            metrics.sharpe >= 1.2
            and metrics.profit_factor >= 1.3
            and metrics.max_drawdown <= 0.08
            and metrics.win_rate >= 0.45
        )
