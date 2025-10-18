"""Persistence layer for Postgres using SQLAlchemy."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    case,
    create_engine,
    func,
    insert,
    select,
)
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

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def get_trade_summary(self) -> Dict[str, float]:
        """Return aggregate metrics for closed trades."""

        pnl_col = trades_table.c.pnl
        closed_stmt = (
            select(
                func.count(pnl_col).label("closed_trades"),
                func.sum(case((pnl_col > 0, 1), else_=0)).label("wins"),
                func.sum(case((pnl_col < 0, 1), else_=0)).label("losses"),
                func.sum(case((pnl_col > 0, pnl_col), else_=0.0)).label("gross_wins"),
                func.sum(case((pnl_col < 0, pnl_col), else_=0.0)).label("gross_losses"),
                func.sum(pnl_col).label("total_pnl"),
                func.max(pnl_col).label("best_trade"),
                func.min(pnl_col).label("worst_trade"),
            )
            .where(pnl_col.is_not(None))
        )

        open_stmt = select(func.count()).where(trades_table.c.closed_at.is_(None))

        with self._connection() as conn:
            summary_row = conn.execute(closed_stmt).one()
            open_trades = conn.execute(open_stmt).scalar_one()

        closed_trades = int(summary_row.closed_trades or 0)
        wins = int(summary_row.wins or 0)
        losses = int(summary_row.losses or 0)
        gross_wins = float(summary_row.gross_wins or 0.0)
        gross_losses = float(summary_row.gross_losses or 0.0)
        total_pnl = float(summary_row.total_pnl or 0.0)
        win_rate = (wins / closed_trades) if closed_trades else 0.0
        avg_win = (gross_wins / wins) if wins else 0.0
        avg_loss = (gross_losses / losses) if losses else 0.0

        return {
            "closed_trades": closed_trades,
            "open_trades": int(open_trades or 0),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "loss_rate": (losses / closed_trades) if closed_trades else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "gross_wins": gross_wins,
            "gross_losses": gross_losses,
            "best_trade": float(summary_row.best_trade or 0.0),
            "worst_trade": float(summary_row.worst_trade or 0.0),
        }

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, object]]:
        """Return the most recent trades ordered by close time."""

        stmt = (
            select(
                trades_table.c.symbol,
                trades_table.c.entry_price,
                trades_table.c.exit_price,
                trades_table.c.quantity,
                trades_table.c.pnl,
                trades_table.c.opened_at,
                trades_table.c.closed_at,
            )
            .order_by(trades_table.c.closed_at.desc().nullslast(), trades_table.c.opened_at.desc())
            .limit(limit)
        )

        with self._connection() as conn:
            rows = conn.execute(stmt).fetchall()

        def _serialize(row: Sequence[object]) -> Dict[str, object]:
            return {
                "symbol": row.symbol,
                "entry_price": float(row.entry_price),
                "exit_price": float(row.exit_price) if row.exit_price is not None else None,
                "quantity": float(row.quantity),
                "pnl": float(row.pnl) if row.pnl is not None else None,
                "opened_at": row.opened_at.isoformat() if row.opened_at else None,
                "closed_at": row.closed_at.isoformat() if row.closed_at else None,
            }

        return [_serialize(row) for row in rows]

    def get_daily_pnl(self, limit: int = 30) -> List[Dict[str, object]]:
        """Return aggregated daily PnL for closed trades."""

        stmt = (
            select(trades_table.c.closed_at, trades_table.c.pnl)
            .where(trades_table.c.closed_at.is_not(None))
            .order_by(trades_table.c.closed_at.desc())
        )

        with self._connection() as conn:
            rows = conn.execute(stmt).fetchall()

        buckets: Dict[str, Dict[str, object]] = {}
        for row in rows:
            if row.closed_at is None:
                continue
            day_key = row.closed_at.date().isoformat()
            bucket = buckets.setdefault(
                day_key,
                {"day": f"{day_key}T00:00:00", "pnl": 0.0, "trades": 0},
            )
            if row.pnl is not None:
                bucket["pnl"] = float(bucket["pnl"]) + float(row.pnl)
            bucket["trades"] = int(bucket["trades"]) + 1

        ordered_days = sorted(buckets.values(), key=lambda item: item["day"], reverse=True)
        return ordered_days[:limit]
