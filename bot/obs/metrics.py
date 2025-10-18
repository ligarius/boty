"""Prometheus metrics exporter."""
from __future__ import annotations

from typing import Dict

from prometheus_client import Counter, Gauge, Histogram

pnl_gauge = Gauge("bot_pnl", "Cumulative PnL in USDT")
drawdown_gauge = Gauge("bot_drawdown", "Current drawdown")
latency_hist = Histogram("bot_latency_seconds", "Exchange latency")
retry_counter = Counter("bot_retries", "Number of retries executed")
rate_limit_counter = Counter("bot_rate_limit_hits", "Binance rate limit hits")


def record_metrics(pnl: float, drawdown: float, latency: float, retries: int, rate_limit_hits: int) -> None:
    pnl_gauge.set(pnl)
    drawdown_gauge.set(drawdown)
    latency_hist.observe(latency)
    retry_counter.inc(retries)
    rate_limit_counter.inc(rate_limit_hits)
