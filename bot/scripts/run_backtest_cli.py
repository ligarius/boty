"""CLI entrypoint for running backtests."""
from __future__ import annotations

from datetime import datetime
import sys

from ..backtest.engine import BacktestEngine
from ..data.loader import OHLCVRequest, generate_synthetic_data


def main(symbol: str, timeframe: str, start: str, end: str) -> None:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    engine = BacktestEngine()
    data = generate_synthetic_data(OHLCVRequest(symbol, timeframe, start_dt, end_dt))
    metrics = engine.run(data)
    print("Backtest Metrics")
    print(metrics)
    print("Go-live ready:", engine.meets_go_live(metrics))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
