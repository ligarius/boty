from datetime import datetime, timedelta

import pandas as pd
import pytest

from bot.core.config import Settings
from bot.core.risk import RiskManager


def test_position_size_respects_min_notional():
    settings = Settings(leverage=20)
    risk = RiskManager(settings)

    equity = 10000
    atr = 100
    stop_distance = atr * 1.5
    risk_amount = equity * settings.risk_pct
    expected_quantity = risk_amount / stop_distance

    high_price = 10
    result = risk.compute_position_size("BTCUSDT", equity=equity, atr=atr, entry_price=high_price)
    assert result.quantity == pytest.approx(expected_quantity, rel=1e-4)
    assert result.notional == pytest.approx(expected_quantity * high_price, rel=1e-3)
    assert result.notional > 5.0  # true notional clears the minimum without leverage adjustments
    expected_margin = round((expected_quantity * high_price) / settings.leverage, 2)
    assert result.margin == expected_margin

    low_price = 0.5
    min_notional = 5.0
    result_min = risk.compute_position_size("BTCUSDT", equity=equity, atr=atr, entry_price=low_price)
    assert result_min.notional == pytest.approx(min_notional, rel=1e-3)
    assert result_min.quantity == pytest.approx(min_notional / low_price, rel=1e-4)
    assert result_min.margin == round(min_notional / settings.leverage, 2)


def test_drawdown_limits():
    settings = Settings()
    risk = RiskManager(settings)
    risk.reset_daily(10000)
    risk.reset_weekly(10000)
    assert risk.check_drawdown(9800)
    assert not risk.check_drawdown(9000)
