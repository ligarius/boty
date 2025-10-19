"""Validate readiness before enabling live trading."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict
import sys

from ..backtest.engine import BacktestEngine
from ..data.loader import OHLCVRequest, generate_synthetic_data
from ..core.config import get_settings


def evaluate_guard(window_days: int = 30) -> Dict[str, object]:
    """Evaluate readiness metrics for live trading."""

    settings = get_settings()
    if settings.mode != "live":
        raise ValueError("Live mode not enabled")

    engine = BacktestEngine()
    end = datetime.utcnow()
    start = end - timedelta(days=window_days)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    metrics = engine.run(data)
    ready = engine.meets_go_live(metrics)
    return {
        "mode": settings.mode,
        "timeframe": "1m",
        "symbol": "BTCUSDT",
        "window_days": window_days,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "metrics": metrics.to_dict(),
        "go_live_ready": ready,
    }


def main() -> None:
    try:
        payload = evaluate_guard()
    except ValueError:
        print("Live mode not enabled. Set MODE=live once validation is complete.")
        sys.exit(1)

    if not payload["go_live_ready"]:
        print("Metrics below threshold. Aborting live trading.")
        sys.exit(2)

    print("Live trading unlocked.")


if __name__ == "__main__":
    main()
