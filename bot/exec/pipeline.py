"""Automated orchestration pipeline for the trading bot."""
from __future__ import annotations

from typing import Optional

from ..core.config import Settings, get_settings
from ..core.risk import RiskManager
from ..data.quality import DataQualityError, validate_ohlcv
from ..ml.live_optimizer import AutoTuneResult, tune_intraday_settings, _load_data, _ensure_datetime
from ..obs.metrics import record_backtest_metrics, record_pipeline_status
from ..persistence.repository import Repository


class AutoTradingPipeline:
    """Coordinate data ingestion, tuning, validation and persistence."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        repository: Optional[Repository] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or Repository(self.settings)
        self.risk = RiskManager(self.settings)

    def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str,
        end: datetime | str,
        *,
        data_source: Optional[str] = None,
        csv_path: Optional[str] = None,
        trials: int = 25,
    ) -> AutoTuneResult:
        record_pipeline_status("started")
        try:
            start_dt = _ensure_datetime(start, "start")
            end_dt = _ensure_datetime(end, "end")

            result = tune_intraday_settings(
                symbol,
                timeframe,
                start_dt,
                end_dt,
                data_source=data_source,
                csv_path=csv_path,
                settings=self.settings,
                n_trials=trials,
            )
            dataset = _load_data(
                symbol,
                timeframe,
                start_dt,
                end_dt,
                data_source=data_source,
                csv_path=csv_path,
                settings=self.settings,
            )
            validate_ohlcv(dataset)
            record_backtest_metrics(result.metrics or {})
            self.repository.record_tuning_run(
                study_name=result.study_name,
                symbol=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                best_score=result.best_score,
                go_live_ready=result.go_live_ready,
                passes_thresholds=result.passes_thresholds,
                metrics=result.metrics,
                params=result.best_params,
            )
            status = "success" if result.passes_thresholds else "deferred"
            record_pipeline_status(status)
            return result
        except DataQualityError:
            record_pipeline_status("data_quality_error")
            raise
        except Exception:
            record_pipeline_status("failed")
            raise
