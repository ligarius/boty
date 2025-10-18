"""Celery configuration and trading tasks."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

from celery import Celery
from celery.schedules import crontab

from ..core.config import get_settings
from ..core.risk import RiskManager
from ..data.loader import OHLCVRequest, generate_synthetic_data
from ..strategies import momentum, mean_reversion
from ..strategies.ensemble import signal_from_row, EnsembleSelector

logger = logging.getLogger(__name__)

settings = get_settings()
celery_app = Celery("bot", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.timezone = "UTC"


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
    end = datetime.utcnow()
    start = end - timedelta(hours=12)
    selector = EnsembleSelector()
    results: Dict[str, List[Dict[str, float]]] = {}
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
        results[symbol] = [
            {
                "signal": sig.signal,
                "score": sig.score,
                "atr": sig.atr,
                "probability": sig.features.get("meta_probability", 0.0),
            }
            for sig in top_signals
        ]
    logger.info("Evaluated strategies", extra={"results": json.dumps(results)})
    return results
