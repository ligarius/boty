"""Prometheus metrics exporter."""
from __future__ import annotations

from typing import Dict

from prometheus_client import Counter, Gauge, Histogram

pnl_gauge = Gauge("bot_pnl", "Cumulative PnL in USDT")
drawdown_gauge = Gauge("bot_drawdown", "Current drawdown")
latency_hist = Histogram("bot_latency_seconds", "Exchange latency")
retry_counter = Counter("bot_retries", "Number of retries executed")
rate_limit_counter = Counter("bot_rate_limit_hits", "Binance rate limit hits")
roi_gauge = Gauge("bot_roi", "Rolling return on investment")
profit_factor_gauge = Gauge("bot_profit_factor", "Rolling profit factor")
win_rate_gauge = Gauge("bot_win_rate", "Rolling win rate")
pipeline_counter = Counter(
    "bot_pipeline_runs",
    "Number of automated pipeline executions",
    labelnames=("status",),
)


def record_metrics(pnl: float, drawdown: float, latency: float, retries: int, rate_limit_hits: int) -> None:
    pnl_gauge.set(pnl)
    drawdown_gauge.set(drawdown)
    latency_hist.observe(latency)
    retry_counter.inc(retries)
    rate_limit_counter.inc(rate_limit_hits)


def record_backtest_metrics(metrics: Dict[str, float]) -> None:
    """Record core performance metrics from backtests or live trading."""

    roi = float(metrics.get("roi", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    roi_gauge.set(roi)
    profit_factor_gauge.set(profit_factor)
    win_rate_gauge.set(win_rate)


def record_pipeline_status(status: str) -> None:
    """Track automated pipeline runs by status (success/failure)."""

    pipeline_counter.labels(status=status).inc()
