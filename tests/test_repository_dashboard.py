from datetime import datetime, timedelta

import pytest

from bot.core.config import Settings
from bot.persistence.repository import Repository, TradeRecord


def build_repo() -> Repository:
    settings = Settings(db_url="sqlite+pysqlite:///:memory:")
    return Repository(settings)


def test_trade_summary_with_and_without_trades():
    repo = build_repo()

    empty_summary = repo.get_trade_summary()
    assert empty_summary["closed_trades"] == 0
    assert empty_summary["open_trades"] == 0
    assert empty_summary["win_rate"] == 0

    now = datetime.utcnow()
    repo.record_trade(
        TradeRecord(
            symbol="BTCUSDT",
            entry_price=100.0,
            exit_price=110.0,
            quantity=1.0,
            pnl=10.0,
            opened_at=now - timedelta(hours=2),
            closed_at=now - timedelta(hours=1),
        )
    )
    repo.record_trade(
        TradeRecord(
            symbol="ETHUSDT",
            entry_price=50.0,
            exit_price=45.0,
            quantity=2.0,
            pnl=-10.0,
            opened_at=now - timedelta(days=1, hours=2),
            closed_at=now - timedelta(days=1, hours=1),
        )
    )
    # Open trade should not be included in summary metrics
    repo.record_trade(
        TradeRecord(
            symbol="BNBUSDT",
            entry_price=70.0,
            exit_price=None,
            quantity=3.0,
            pnl=None,
            opened_at=now,
            closed_at=None,
        )
    )

    summary = repo.get_trade_summary()
    assert summary["closed_trades"] == 2
    assert summary["open_trades"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["total_pnl"] == pytest.approx(0.0)
    assert summary["avg_win"] == pytest.approx(10.0)
    assert summary["avg_loss"] == pytest.approx(-10.0)
    assert summary["win_rate"] == pytest.approx(0.5)


def test_recent_trades_and_daily_pnl_ordering():
    repo = build_repo()
    now = datetime.utcnow()
    repo.record_trade(
        TradeRecord(
            symbol="BTCUSDT",
            entry_price=100.0,
            exit_price=110.0,
            quantity=1.0,
            pnl=10.0,
            opened_at=now - timedelta(hours=2),
            closed_at=now - timedelta(hours=1),
        )
    )
    repo.record_trade(
        TradeRecord(
            symbol="ETHUSDT",
            entry_price=50.0,
            exit_price=55.0,
            quantity=1.5,
            pnl=7.5,
            opened_at=now - timedelta(days=1, hours=2),
            closed_at=now - timedelta(days=1, hours=1),
        )
    )

    trades = repo.get_recent_trades(limit=5)
    assert [trade["symbol"] for trade in trades] == ["BTCUSDT", "ETHUSDT"]

    daily = repo.get_daily_pnl(limit=10)
    assert len(daily) == 2
    assert daily[0]["pnl"] == pytest.approx(10.0)
    assert daily[1]["pnl"] == pytest.approx(7.5)
