"""FastAPI application exposing control and monitoring endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..core.config import get_settings
from ..core.risk import RiskManager
from ..exec.celery_app import evaluate_strategies, get_worker_state
from ..obs.logging import configure_logging
from ..obs.metrics import pnl_gauge, drawdown_gauge
from ..persistence.repository import Repository
from ..scripts.guard_live import evaluate_guard
from ..scripts.report_daily_cli import generate_daily_report
from ..scripts.run_backtest_cli import run_backtest

configure_logging()
app = FastAPI(title="PulseForge", version="1.0")
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
def status() -> Dict[str, object]:
    base_equity = 10000.0
    repo = get_repository()

    if repo is not None:
        portfolio = repo.get_portfolio_metrics(base_equity)
        equity = portfolio["equity"]
        drawdown = portfolio["drawdown"]
        positions = portfolio["open_positions"]
    else:
        pnl = pnl_gauge._value.get()  # type: ignore[attr-defined]
        equity = base_equity + pnl
        drawdown = drawdown_gauge._value.get()  # type: ignore[attr-defined]
        positions = 0

    return {
        "equity": equity,
        "daily_dd": drawdown,
        "positions": positions,
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


@app.post("/backtest")
def run_backtest_endpoint(payload: Dict[str, str]) -> Dict[str, object]:
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe")
    start = payload.get("start")
    end = payload.get("end")

    if not symbol or not timeframe or not start or not end:
        raise HTTPException(status_code=400, detail="symbol, timeframe, start and end are required")

    try:
        result = run_backtest(symbol, timeframe, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.post("/live/guard")
def guard_live_endpoint() -> Dict[str, object]:
    try:
        return evaluate_guard()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/report/daily")
def report_daily_endpoint() -> Dict[str, object]:
    return generate_daily_report()


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
    worker_state = get_worker_state()
    try:
        report_payload = report_daily_endpoint()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Unable to compute synthetic daily report: %s", exc)
        report_payload = None

    return {
        "status": status_payload,
        "report": report_payload,
        "trade_summary": trade_summary,
        "recent_trades": recent_trades,
        "daily_pnl": daily_pnl,
        "activity": {
            "worker": worker_state,
            "repository_ready": repo is not None,
        },
    }
