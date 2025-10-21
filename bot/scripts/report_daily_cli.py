"""Generate daily performance report."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

from ..backtest.engine import BacktestEngine
from ..data.loader import OHLCVRequest, generate_synthetic_data


def generate_daily_report() -> Dict[str, object]:
    """Create a synthetic daily report payload."""

    engine = BacktestEngine()
    end = datetime.utcnow()
    start = end - timedelta(days=1)
    timeframe = "1m"
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", timeframe, start, end))
    metrics = engine.run(data, timeframe)
    chart_csv = data[["close"]].reset_index().rename(columns={"index": "timestamp"}).to_csv(index=False)
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "metrics": metrics.to_dict(),
        "chart_csv": chart_csv,
    }


def main() -> None:
    payload = generate_daily_report()
    print("Daily Report")
    for key, value in payload["metrics"].items():
        print(f"  - {key}: {value:.4f}")


if __name__ == "__main__":
    main()
