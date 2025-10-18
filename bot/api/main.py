"""FastAPI application exposing control and monitoring endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..backtest.engine import BacktestEngine
from ..core.config import get_settings
from ..core.risk import RiskManager
from ..data.loader import OHLCVRequest, generate_synthetic_data
from ..exec.celery_app import evaluate_strategies
from ..obs.logging import configure_logging
from ..obs.metrics import pnl_gauge, drawdown_gauge

configure_logging()
app = FastAPI(title="Prompt Maestro", version="1.0")
settings = get_settings()
risk = RiskManager(settings)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/status")
def status() -> Dict[str, float]:
    equity = 10000.0
    pnl = pnl_gauge._value.get()  # type: ignore[attr-defined]
    drawdown = drawdown_gauge._value.get()  # type: ignore[attr-defined]
    return {
        "equity": equity + pnl,
        "daily_dd": drawdown,
        "positions": 0,
    }


@app.post("/mode")
def change_mode(payload: Dict[str, str]) -> Dict[str, str]:
    mode = payload.get("mode")
    if mode not in {"backtest", "paper", "live"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    if mode == "live" and settings.mode != "live":
        raise HTTPException(status_code=403, detail="Live mode locked until validation")
    settings.mode = mode  # type: ignore[misc]
    return {"mode": settings.mode}


@app.post("/pause")
def pause_trading() -> Dict[str, str]:
    settings.top_n_signals = 0  # type: ignore[misc]
    return {"status": "paused"}


@app.post("/resume")
def resume_trading() -> Dict[str, str]:
    settings.top_n_signals = 3  # type: ignore[misc]
    return {"status": "resumed"}


@app.get("/report/daily")
def report_daily() -> Dict[str, str]:
    end = datetime.utcnow()
    start = end - timedelta(days=1)
    data = generate_synthetic_data(OHLCVRequest("BTCUSDT", "1m", start, end))
    engine = BacktestEngine()
    metrics = engine.run(data)
    df = pd.DataFrame(
        {
            "timestamp": data.index,
            "close": data["close"],
        }
    )
    chart = df.to_csv(index=False)
    return {
        "roi": f"{metrics.roi:.2%}",
        "sharpe": f"{metrics.sharpe:.2f}",
        "profit_factor": f"{metrics.profit_factor:.2f}",
        "chart_csv": chart,
    }


@app.post("/universe")
def update_universe(payload: Dict[str, str]) -> Dict[str, str]:
    tokens = payload.get("symbols")
    if not tokens:
        raise HTTPException(status_code=400, detail="symbols required")
    settings.universe = [token.strip() for token in tokens.split(",")]
    return {"status": "updated"}


@app.post("/risk")
def update_risk(payload: Dict[str, float]) -> Dict[str, float]:
    risk_pct = payload.get("risk_pct")
    if risk_pct is None or risk_pct <= 0 or risk_pct > 0.05:
        raise HTTPException(status_code=400, detail="invalid risk_pct")
    settings.risk_pct = risk_pct  # type: ignore[misc]
    return {"risk_pct": settings.risk_pct}


@app.get("/metrics")
def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
