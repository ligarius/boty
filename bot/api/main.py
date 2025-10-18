"""FastAPI application exposing control and monitoring endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..backtest.engine import BacktestEngine
from ..core.config import get_settings
from ..core.risk import RiskManager
from ..data.loader import OHLCVRequest, generate_synthetic_data
from ..exec.celery_app import evaluate_strategies
from ..obs.logging import configure_logging
from ..obs.metrics import pnl_gauge, drawdown_gauge
from ..persistence.repository import Repository

configure_logging()
app = FastAPI(title="Prompt Maestro", version="1.0")
settings = get_settings()
risk = RiskManager(settings)
logger = getLogger(__name__)

static_dir = Path(__file__).with_name("static")
templates_dir = Path(__file__).with_name("templates")

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=str(templates_dir))

_repository: Optional[Repository] = None


def get_repository() -> Optional[Repository]:
    """Return a cached repository instance, initialising lazily."""

    global _repository
    if _repository is None:
        try:
            _repository = Repository(settings)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Unable to initialise repository: %s", exc)
            _repository = None
    return _repository


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
        "mode": settings.mode,
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


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/dashboard/data")
def dashboard_data() -> Dict[str, object]:
    repo = get_repository()
    trade_summary: Dict[str, float] | None = None
    recent_trades: list[Dict[str, object]] = []
    daily_pnl: list[Dict[str, object]] = []
    report_payload: Dict[str, str] | None = None

    if repo is not None:
        trade_summary = repo.get_trade_summary()
        recent_trades = repo.get_recent_trades(limit=25)
        daily_pnl = repo.get_daily_pnl(limit=30)

    status_payload = status()
    try:
        report_payload = report_daily()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Unable to compute synthetic daily report: %s", exc)
        report_payload = None

    return {
        "status": status_payload,
        "report": report_payload,
        "trade_summary": trade_summary,
        "recent_trades": recent_trades,
        "daily_pnl": daily_pnl,
    }
