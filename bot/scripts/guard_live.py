"""Validate readiness before enabling live trading."""
from __future__ import annotations

from datetime import datetime, timedelta
import sys

from ..backtest.engine import BacktestEngine
from ..data.loader import OHLCVRequest, generate_synthetic_data
from ..core.config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.mode != "live":
        print("Live mode not enabled. Set MODE=live once validation is complete.")
        sys.exit(1)
    engine = BacktestEngine()
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    metrics = engine.run(data)
    if not engine.meets_go_live(metrics):
        print("Metrics below threshold. Aborting live trading.")
        sys.exit(2)
    print("Live trading unlocked.")


if __name__ == "__main__":
    main()
