from datetime import datetime, timedelta

import pandas as pd

from bot.core.config import Settings
from bot.core.risk import RiskManager


def test_position_size_respects_min_notional():
    settings = Settings()
    risk = RiskManager(settings)
    result = risk.compute_position_size("BTCUSDT", equity=10000, atr=100, entry_price=20000)
    assert result.notional >= 5.0
    assert result.quantity > 0


def test_drawdown_limits():
    settings = Settings()
    risk = RiskManager(settings)
    risk.reset_daily(10000)
    risk.reset_weekly(10000)
    assert risk.check_drawdown(9800)
    assert not risk.check_drawdown(9000)
