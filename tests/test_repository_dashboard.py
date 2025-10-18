from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from bot.core.config import Settings
from bot.persistence.repository import Repository, TradeRecord
from bot.api import main as api_main
from bot.exec import celery_app as celery_module


@pytest.fixture
def api_repository(tmp_path, monkeypatch):
    """Configure an in-memory-like SQLite repository for API calls."""

    original_db_url = api_main.settings.db_url
    db_path = tmp_path / "api.db"
    db_url = f"sqlite+pysqlite:///{db_path}"
    api_main.settings.db_url = db_url  # type: ignore[misc]
    repo = Repository(api_main.settings)
    monkeypatch.setattr(api_main, "_repository", repo)
    celery_module.set_repository(repo)
    celery_module._last_evaluation_at = None
    celery_module._last_evaluation_error = None

    try:
        yield repo
    finally:
        api_main.settings.db_url = original_db_url  # type: ignore[misc]
        monkeypatch.setattr(api_main, "_repository", None)
        celery_module.set_repository(None)
        celery_module._last_evaluation_at = None
        celery_module._last_evaluation_error = None


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


def test_status_and_dashboard_reflect_repository_data(api_repository):
    client = TestClient(api_main.app)
    repo = api_repository

    now = datetime.utcnow()
    repo.record_trade(
        TradeRecord(
            symbol="BTCUSDT",
            entry_price=100.0,
            exit_price=95.0,
            quantity=1.0,
            pnl=-5.0,
            opened_at=now - timedelta(minutes=10),
            closed_at=now - timedelta(minutes=5),
        )
    )
    repo.record_trade(
        TradeRecord(
            symbol="ETHUSDT",
            entry_price=50.0,
            exit_price=60.0,
            quantity=1.0,
            pnl=10.0,
            opened_at=now - timedelta(minutes=4),
            closed_at=now - timedelta(minutes=2),
        )
    )
    repo.record_trade(
        TradeRecord(
            symbol="BNBUSDT",
            entry_price=30.0,
            exit_price=None,
            quantity=2.0,
            pnl=None,
            opened_at=now - timedelta(minutes=1),
            closed_at=None,
        )
    )

    status_response = client.get("/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()

    assert status_payload["equity"] == pytest.approx(10005.0)
    assert status_payload["positions"] == 1
    assert status_payload["daily_dd"] == pytest.approx(0.0005)

    dashboard_response = client.get("/dashboard/data")
    assert dashboard_response.status_code == 200
    dashboard_payload = dashboard_response.json()

    assert dashboard_payload["status"] == status_payload
    assert dashboard_payload["trade_summary"]["open_trades"] == 1
    assert dashboard_payload["trade_summary"]["closed_trades"] == 2
    assert dashboard_payload["activity"]["repository_ready"] is True
    assert dashboard_payload["activity"]["worker"]["active"] is False
