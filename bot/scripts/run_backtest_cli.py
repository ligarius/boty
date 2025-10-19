"""CLI entrypoint and helpers for running backtests."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, MutableMapping
import sys

from ..backtest.engine import BacktestEngine, BacktestMetrics
from ..data.loader import OHLCVRequest, generate_synthetic_data


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
) -> Dict[str, object]:
    """Execute the synthetic backtest and return metrics."""

    start_dt = _ensure_datetime(start, "start")
    end_dt = _ensure_datetime(end, "end")
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    engine = BacktestEngine()
    data = generate_synthetic_data(OHLCVRequest(symbol, timeframe, start_dt, end_dt))
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


def main(symbol: str, timeframe: str, start: str, end: str) -> None:
    payload = run_backtest(symbol, timeframe, start, end)
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
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
