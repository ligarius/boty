"""Automated intraday hyperparameter search leveraging ML selectors."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import optuna

from ..backtest.engine import BacktestEngine, BacktestMetrics
from ..core.config import Settings, get_settings
from ..data.loader import (
    OHLCVRequest,
    fetch_binance_ohlcv,
    generate_synthetic_data,
    load_local_csv,
)
from .tuner import EvolutionaryTuner, CandidateResult


@dataclass
class AutoTuneResult:
    """Structured information summarizing an optimization run."""

    study_name: str
    trials: int
    best_score: float
    baseline_score: float
    best_params: Dict[str, float]
    metrics: Dict[str, float]
    training: Optional[Dict[str, float]]
    go_live_ready: bool
    passes_thresholds: bool
    baseline_metrics: Dict[str, float]
    baseline_training: Optional[Dict[str, float]]


def _ensure_datetime(value: datetime | str, field: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        candidate = value
        if isinstance(value, str) and value.endswith("Z"):
            candidate = value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid {field} datetime") from exc

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _composite_score(metrics: BacktestMetrics) -> float:
    """Combine multiple metrics into a single score for optimization."""

    roi_component = metrics.roi
    sharpe_component = metrics.sharpe * 2.0
    profit_component = (metrics.profit_factor - 1.0) * 1.5
    win_rate_component = metrics.win_rate * 2.0
    drawdown_penalty = metrics.max_drawdown * 3.0

    score = roi_component + sharpe_component + profit_component + win_rate_component - drawdown_penalty

    # Penalize regimes that fall below minimum acceptable thresholds.
    if metrics.win_rate < 0.45:
        score -= (0.45 - metrics.win_rate) * 10
    if metrics.sharpe < 0:
        score += metrics.sharpe  # negative sharpe reduces score even further
    if metrics.profit_factor < 1.0:
        score -= (1.0 - metrics.profit_factor) * 5

    return score


def _load_data(
    symbol: str,
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    *,
    data_source: Optional[str] = None,
    csv_path: Optional[str | Path] = None,
    settings: Optional[Settings] = None,
):
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
    else:  # pragma: no cover - validated by argparse choices
        raise ValueError(f"Unsupported data source '{source}'")

    trimmed = data.loc[(data.index >= start_dt) & (data.index < end_dt)]
    if trimmed.empty:
        raise ValueError("No OHLCV data returned for the requested period")
    return trimmed


def tune_intraday_settings(
    symbol: str,
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    *,
    data_source: Optional[str] = None,
    csv_path: Optional[str | Path] = None,
    settings: Optional[Settings] = None,
    study_name: str = "auto_live_intraday",
    storage: str | None = None,
    n_trials: int = 25,
) -> AutoTuneResult:
    """Search for the most profitable intraday configuration using ML selectors."""

    resolved_settings = settings or get_settings()
    ohlcv = _load_data(
        symbol,
        timeframe,
        start,
        end,
        data_source=data_source,
        csv_path=csv_path,
        settings=resolved_settings,
    )

    baseline_engine = BacktestEngine()
    baseline_metrics = baseline_engine.run(ohlcv, timeframe=timeframe)
    baseline_training = (
        baseline_engine.last_training_report.to_dict()
        if baseline_engine.last_training_report is not None
        else None
    )
    baseline_score = _composite_score(baseline_metrics)

    tuner = EvolutionaryTuner(
        study_name=study_name,
        storage=storage,
        min_roi=resolved_settings.auto_tune_min_roi,
        min_profit_factor=resolved_settings.auto_tune_min_profit_factor,
        max_drawdown=resolved_settings.auto_tune_max_drawdown,
    )

    def objective(trial: optuna.Trial) -> float:
        momentum_fast = trial.suggest_int("momentum_fast", 5, 30)
        momentum_slow = trial.suggest_int("momentum_slow", 31, 120)
        momentum_adx = trial.suggest_int("momentum_adx_period", 7, 30)
        momentum_adx_threshold = trial.suggest_float("momentum_adx_threshold", 15.0, 40.0)
        momentum_atr_multiplier = trial.suggest_float("momentum_atr_multiplier", 0.5, 3.0)

        mean_window = trial.suggest_int("mean_window", 10, 80)
        mean_z = trial.suggest_float("mean_z_threshold", 0.8, 3.0)

        breakout_lookback = trial.suggest_int("breakout_lookback", 10, 60)
        breakout_multiplier = trial.suggest_float("breakout_multiplier", 1.0, 3.0)
        breakout_cooldown = trial.suggest_int("breakout_cooldown", 1, 10)

        market_spread = trial.suggest_float("market_spread", 0.0005, 0.003)
        market_liquidity_window = trial.suggest_int("market_liquidity_window", 20, 120)
        market_inventory_alpha = trial.suggest_float("market_inventory_alpha", 0.1, 0.9)

        selector_threshold = trial.suggest_float("selector_threshold", 0.0, 0.4)
        selector_window = trial.suggest_int("selector_window", 20, 500)
        selector_horizon = trial.suggest_int("selector_horizon", 1, 20)

        engine = BacktestEngine()
        engine.selector_threshold = selector_threshold
        engine.selector_window = selector_window
        engine.selector_horizon = selector_horizon

        metrics = engine.run(
            ohlcv,
            timeframe=timeframe,
            momentum_params={
                "fast": momentum_fast,
                "slow": momentum_slow,
                "adx_period": momentum_adx,
                "adx_threshold": momentum_adx_threshold,
                "atr_multiplier": momentum_atr_multiplier,
            },
            mean_reversion_params={
                "window": mean_window,
                "z_threshold": mean_z,
            },
            volatility_params={
                "lookback": breakout_lookback,
                "breakout_multiplier": breakout_multiplier,
                "cooldown": breakout_cooldown,
            },
            market_making_params={
                "spread": market_spread,
                "liquidity_window": market_liquidity_window,
                "inventory_alpha": market_inventory_alpha,
            },
        )

        score = _composite_score(metrics)

        training_report = (
            engine.last_training_report.to_dict() if engine.last_training_report is not None else None
        )

        passes_thresholds = (
            metrics.roi >= resolved_settings.auto_tune_min_roi
            and metrics.profit_factor >= resolved_settings.auto_tune_min_profit_factor
            and metrics.max_drawdown <= resolved_settings.auto_tune_max_drawdown
        )

        resolved_selector_window = engine.last_resolved_selector_window
        if resolved_selector_window is None:
            resolved_selector_window = int(engine.selector_window)

        resolved_selector_threshold = engine.last_resolved_selector_threshold
        if resolved_selector_threshold is None:
            resolved_selector_threshold = float(engine.selector_threshold)

        trial.set_user_attr(
            "metrics",
            {
                **metrics.to_dict(),
                "training_accuracy": metrics.training_accuracy,
                "training_f1": metrics.training_f1,
            },
        )
        trial.set_user_attr("training", training_report)
        trial.set_user_attr("baseline", baseline_score)
        trial.set_user_attr("go_live_ready", engine.meets_go_live(metrics))
        trial.set_user_attr("passes_thresholds", passes_thresholds)
        trial.set_user_attr("resolved_selector_window", int(resolved_selector_window))
        trial.set_user_attr(
            "resolved_selector_threshold", float(resolved_selector_threshold)
        )

        resolved_params = dict(trial.params)
        resolved_params["selector_window"] = int(resolved_selector_window)
        resolved_params["selector_threshold"] = float(resolved_selector_threshold)
        trial.set_user_attr("resolved_params", resolved_params)

        return score

    candidate: CandidateResult = tuner.optimize(objective, n_trials=n_trials)

    metrics_payload = candidate.metrics.get("metrics")
    training_payload = candidate.metrics.get("training")
    go_live_ready = bool(candidate.metrics.get("go_live_ready", False)) and candidate.passes_thresholds
    value = float(candidate.metrics.get("value", 0.0))

    if metrics_payload is None:
        metrics_payload = {}

    return AutoTuneResult(
        study_name=study_name,
        trials=n_trials,
        best_score=value,
        baseline_score=baseline_score,
        best_params=candidate.params,
        metrics=metrics_payload,
        training=training_payload,
        go_live_ready=go_live_ready,
        passes_thresholds=candidate.passes_thresholds,
        baseline_metrics=baseline_metrics.to_dict(),
        baseline_training=baseline_training,
    )

