"""CLI entrypoint and helpers for running backtests."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, MutableMapping, Optional
import sys

from ..backtest.engine import BacktestEngine, BacktestMetrics
from ..core.config import Settings, get_settings
from ..data.loader import OHLCVRequest, fetch_binance_ohlcv, generate_synthetic_data, load_local_csv


def _ensure_datetime(value: datetime | str, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field} datetime") from exc


def run_backtest(
    symbol: str,
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    *,
    data_source: Optional[str] = None,
    settings: Optional[Settings] = None,
    csv_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Execute the backtest using the requested data source and return metrics."""

    start_dt = _ensure_datetime(start, "start")
    end_dt = _ensure_datetime(end, "end")
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    settings = settings or get_settings()
    source = (data_source or settings.default_data_source).lower()

    request = OHLCVRequest(symbol, timeframe, start_dt, end_dt)
    if source == "synthetic":
        data = generate_synthetic_data(request)
    elif source == "csv":
        resolved_candidate = csv_path or settings.data_source_csv_path
        if not resolved_candidate:
            raise ValueError("CSV data source requires a path")
        resolved_path = Path(resolved_candidate).expanduser()
        data = load_local_csv(resolved_path)
    elif source == "binance":
        data = fetch_binance_ohlcv(request, settings=settings)
    else:
        raise ValueError(f"Unsupported data source '{source}'")

    data = data.loc[(data.index >= start_dt) & (data.index < end_dt)]

    if data.empty:
        raise ValueError("No OHLCV data returned for the requested period")

    engine = BacktestEngine()
    metrics = engine.run(data)
    training_report = (
        engine.last_training_report.to_dict() if engine.last_training_report is not None else None
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "metrics": metrics.to_dict(),
        "training": training_report,
        "go_live_ready": engine.meets_go_live(metrics),
    }


def _pretty_metrics(metrics: MutableMapping[str, float]) -> str:
    formatted = []
    for key, value in metrics.items():
        formatted.append(f"  - {key}: {value:.4f}")
    return "\n".join(formatted)


def _pretty_training(report: MutableMapping[str, object]) -> str:
    formatted = []
    accuracy = report.get("accuracy")
    if isinstance(accuracy, (float, int)):
        formatted.append(f"  - accuracy: {accuracy:.4f}")
    f1 = report.get("f1")
    if isinstance(f1, (float, int)):
        formatted.append(f"  - f1: {f1:.4f}")
    feature_importances = report.get("feature_importances")
    if isinstance(feature_importances, MutableMapping) and feature_importances:
        formatted.append("  - feature_importances:")
        for feature, weight in feature_importances.items():
            if isinstance(weight, (float, int)):
                formatted.append(f"      * {feature}: {weight:.4f}")
    return "\n".join(formatted)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a backtest for a given symbol/timeframe")
    parser.add_argument("symbol", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("timeframe", help="Candlestick interval, e.g. 1m")
    parser.add_argument("start", help="Start datetime in ISO format")
    parser.add_argument("end", help="End datetime in ISO format")
    parser.add_argument(
        "--data-source",
        choices=["binance", "synthetic", "csv"],
        default=None,
        help="Data source to use (default: configuration value)",
    )
    parser.add_argument(
        "--csv-path",
        help="Path to a local CSV file when using the csv data source",
    )
    args = parser.parse_args(argv)

    payload = run_backtest(
        args.symbol,
        args.timeframe,
        args.start,
        args.end,
        data_source=args.data_source,
        csv_path=args.csv_path,
    )
    metrics = BacktestMetrics(**payload["metrics"])  # type: ignore[arg-type]
    print("Backtest Metrics")
    print(_pretty_metrics(metrics.to_dict()))
    training_report = payload.get("training")
    if training_report:
        print("Training Metrics")
        print(_pretty_training(training_report))  # type: ignore[arg-type]
    else:
        print("Training Metrics: unavailable")
    print("Go-live ready:", payload["go_live_ready"])


if __name__ == "__main__":
    main(sys.argv[1:])
