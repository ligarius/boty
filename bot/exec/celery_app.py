"""Celery configuration and trading tasks."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from celery import Celery
from celery.schedules import crontab

from ..core.config import get_settings
from ..core.risk import RiskManager
from ..data.loader import OHLCVRequest, generate_synthetic_data
from ..strategies import momentum, mean_reversion
from ..strategies.ensemble import signal_from_row, EnsembleSelector
from ..persistence.repository import Repository, TradeRecord

logger = logging.getLogger(__name__)

settings = get_settings()
celery_app = Celery("bot", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.timezone = "UTC"

try:
    repository: Optional[Repository] = Repository(settings)
except Exception as exc:  # pragma: no cover - defensive fallback
    logger.warning("Unable to initialise repository in Celery app: %s", exc)
    repository = None

_last_evaluation_at: datetime | None = None
_last_evaluation_error: str | None = None


def set_repository(instance: Optional[Repository]) -> None:
    """Allow tests to override the repository instance."""

    global repository
    repository = instance


def get_worker_state() -> Dict[str, object]:
    """Return diagnostic information about the evaluation worker."""

    return {
        "active": _last_evaluation_error is None and _last_evaluation_at is not None,
        "last_run": _last_evaluation_at.isoformat() if _last_evaluation_at else None,
        "last_error": _last_evaluation_error,
    }


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender: Celery, **_: Dict) -> None:
    sender.add_periodic_task(
        60.0,
        evaluate_strategies.s(),
        name="evaluate-strategies-1m",
    )
    sender.add_periodic_task(
        crontab(minute=0, hour="0"),
        daily_reset.s(),
        name="daily-risk-reset",
    )


@celery_app.task
def daily_reset() -> None:
    risk = RiskManager(settings)
    risk.reset_daily(equity=10000.0)
    logger.info("Daily risk reset executed")


@celery_app.task
def evaluate_strategies() -> Dict[str, List[Dict[str, float]]]:
    global _last_evaluation_at, _last_evaluation_error
    end = datetime.utcnow()
    start = end - timedelta(hours=12)
    selector = EnsembleSelector()
    results: Dict[str, List[Dict[str, float]]] = {}
    try:
        for symbol in settings.universe:
            request = OHLCVRequest(symbol=symbol, timeframe="1m", start=start, end=end)
            data = generate_synthetic_data(request)
            momentum_df = momentum.momentum_signals(data)
            mean_df = mean_reversion.mean_reversion_signals(data)
            signals = [
                signal_from_row(symbol, "1m", row._asdict())
                for row in momentum_df.tail(5).itertuples(index=False)
            ] + [
                signal_from_row(symbol, "1m", row._asdict())
                for row in mean_df.tail(5).itertuples(index=False)
            ]
            top_signals = selector.select_top_n(signals, settings.top_n_signals)
            payload = [
                {
                    "signal": sig.signal,
                    "score": sig.score,
                    "atr": sig.atr,
                    "probability": sig.features.get("meta_probability", 0.0),
                }
                for sig in top_signals
            ]
            results[symbol] = payload

            if repository is not None and payload:
                repository.record_signals(symbol, "1m", payload)

                now = datetime.utcnow()
                reference_price = float(data["close"].iloc[-1])
                for index, sig in enumerate(top_signals):
                    if sig.signal == 0:
                        continue
                    atr = float(sig.atr) if not math.isnan(float(sig.atr)) else 0.0
                    price_offset = atr if atr else 10.0
                    exit_price = reference_price + sig.signal * price_offset * 0.1
                    pnl = (exit_price - reference_price) * sig.signal
                    trade = TradeRecord(
                        symbol=symbol,
                        entry_price=reference_price,
                        exit_price=exit_price,
                        quantity=1.0,
                        pnl=pnl,
                        opened_at=now - timedelta(minutes=index + 1),
                        closed_at=now,
                    )
                    repository.record_trade(trade)

        _last_evaluation_at = datetime.utcnow()
        _last_evaluation_error = None
        logger.info("Evaluated strategies", extra={"results": json.dumps(results)})
        return results
    except Exception as exc:
        _last_evaluation_at = datetime.utcnow()
        _last_evaluation_error = str(exc)
        logger.exception("Strategy evaluation failed")
        raise
