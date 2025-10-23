"""CLI entrypoint to run automated intraday hyperparameter search."""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

from ..ml.live_optimizer import AutoTuneResult, tune_intraday_settings


def _pretty_dict(items: Iterable[tuple[str, object]]) -> str:
    lines = []
    for key, value in items:
        if isinstance(value, (int, float)):
            lines.append(f"  - {key}: {value:.4f}")
        else:
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def _pretty_metrics(metrics: dict[str, float]) -> str:
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted.append(f"  - {key}: {value:.4f}")
    return "\n".join(formatted)


def _display(result: AutoTuneResult) -> None:
    print("Auto-tune summary")
    print(f"  Study: {result.study_name} (trials={result.trials})")
    print(f"  Composite score: {result.best_score:.4f} (baseline {result.baseline_score:.4f})")
    print(f"  Go-live ready: {result.go_live_ready}")

    print("Applied parameters")
    sorted_params = sorted(result.best_params.items())
    print(_pretty_dict(sorted_params))

    print("Optimized metrics")
    print(_pretty_metrics(result.metrics))

    if result.training:
        print("Training report")
        print(_pretty_metrics(result.training))
    else:
        print("Training report: unavailable")

    print("Baseline metrics")
    print(_pretty_metrics(result.baseline_metrics))

    if result.baseline_training:
        print("Baseline training")
        print(_pretty_metrics(result.baseline_training))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automate intraday ML tuning to validate live readiness",
    )
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
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of Optuna trials to execute (default: 25)",
    )
    parser.add_argument(
        "--study-name",
        default="auto_live_intraday",
        help="Name of the Optuna study (default: auto_live_intraday)",
    )
    parser.add_argument(
        "--storage",
        help="Optional Optuna storage URI to persist study results",
    )

    args = parser.parse_args(argv)

    result = tune_intraday_settings(
        args.symbol,
        args.timeframe,
        args.start,
        args.end,
        data_source=args.data_source,
        csv_path=args.csv_path,
        study_name=args.study_name,
        storage=args.storage,
        n_trials=args.trials,
    )

    _display(result)


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    main()

