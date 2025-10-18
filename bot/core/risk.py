"""Risk management primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Settings


@dataclass
class PositionSizingResult:
    symbol: str
    quantity: float
    notional: float
    risk_amount: float
    atr: float
    leverage: float


class RiskManager:
    """Compute position sizes and enforce drawdown limits."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.daily_start_equity: Optional[float] = None
        self.weekly_start_equity: Optional[float] = None

    def reset_daily(self, equity: float) -> None:
        self.daily_start_equity = equity

    def reset_weekly(self, equity: float) -> None:
        self.weekly_start_equity = equity

    def check_drawdown(self, equity: float) -> bool:
        """Return True if trading should continue."""

        if self.daily_start_equity is None:
            self.daily_start_equity = equity
        if self.weekly_start_equity is None:
            self.weekly_start_equity = equity

        daily_dd = 1 - equity / max(self.daily_start_equity, 1e-8)
        weekly_dd = 1 - equity / max(self.weekly_start_equity, 1e-8)
        return daily_dd <= self.settings.max_dd_daily and weekly_dd <= self.settings.max_dd_weekly

    def compute_position_size(
        self,
        symbol: str,
        equity: float,
        atr: float,
        entry_price: float,
        risk_multiplier: float = 1.0,
    ) -> PositionSizingResult:
        """Size a position using fixed-fractional risk with ATR-based stop distance."""

        if atr <= 0:
            raise ValueError("ATR must be positive")
        risk_pct = self.settings.risk_pct * risk_multiplier
        risk_amount = equity * risk_pct
        stop_distance = atr * 1.5  # ATR multiple stop
        quantity = risk_amount / stop_distance
        notional = quantity * entry_price / max(self.settings.leverage, 1e-6)
        min_qty = self._min_contract_notional(entry_price)
        if notional < min_qty:
            quantity = min_qty * self.settings.leverage / entry_price
            notional = min_qty
        return PositionSizingResult(
            symbol=symbol,
            quantity=float(np.round(quantity, 6)),
            notional=float(np.round(notional, 2)),
            risk_amount=float(np.round(risk_amount, 2)),
            atr=float(np.round(atr, 4)),
            leverage=self.settings.leverage,
        )

    def _min_contract_notional(self, price: float) -> float:
        """Approximate Binance minimum notional using conservative default."""

        return max(5.0, 5.0 * price / max(price, 1e-8))
