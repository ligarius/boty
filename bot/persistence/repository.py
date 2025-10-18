"""Persistence layer for Postgres using SQLAlchemy."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine

from ..core.config import Settings

metadata = MetaData()

signals_table = Table(
    "signals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False),
    Column("timeframe", String(10), nullable=False),
    Column("direction", Integer, nullable=False),
    Column("score", Float, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)

trades_table = Table(
    "trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("exit_price", Float, nullable=True),
    Column("quantity", Float, nullable=False),
    Column("pnl", Float, nullable=True),
    Column("opened_at", DateTime, default=datetime.utcnow, nullable=False),
    Column("closed_at", DateTime, nullable=True),
)


@dataclass
class TradeRecord:
    symbol: str
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl: float | None
    opened_at: datetime
    closed_at: datetime | None


class Repository:
    """Simple repository handling inserts for signals and trades."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine: Engine = create_engine(settings.db_url, echo=False, future=True)
        metadata.create_all(self.engine)

    @contextmanager
    def _connection(self):
        with self.engine.begin() as conn:
            yield conn

    def record_signals(self, symbol: str, timeframe: str, signals: Iterable[Dict[str, float]]) -> None:
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": sig["signal"],
                "score": sig["score"],
                "created_at": datetime.utcnow(),
            }
            for sig in signals
        ]
        if not rows:
            return
        with self._connection() as conn:
            conn.execute(insert(signals_table), rows)

    def record_trade(self, trade: TradeRecord) -> None:
        with self._connection() as conn:
            conn.execute(
                insert(trades_table),
                [
                    {
                        "symbol": trade.symbol,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "quantity": trade.quantity,
                        "pnl": trade.pnl,
                        "opened_at": trade.opened_at,
                        "closed_at": trade.closed_at,
                    }
                ],
            )
