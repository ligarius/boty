"""Generate daily performance report."""
from __future__ import annotations

from datetime import datetime, timedelta

from ..backtest.engine import BacktestEngine
from ..data.loader import OHLCVRequest, generate_synthetic_data


def main() -> None:
    engine = BacktestEngine()
    end = datetime.utcnow()
    start = end - timedelta(days=1)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    metrics = engine.run(data)
    print("Daily Report")
    print(metrics)


if __name__ == "__main__":
    main()
