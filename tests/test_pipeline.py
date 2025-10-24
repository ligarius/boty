from datetime import datetime

import pandas as pd

from bot.exec.pipeline import AutoTradingPipeline
from bot.ml.live_optimizer import AutoTuneResult


class DummyRepository:
    def __init__(self) -> None:
        self.calls = []

    def record_tuning_run(self, **payload):  # type: ignore[override]
        self.calls.append(payload)


def test_pipeline_records_success(monkeypatch):
    repo = DummyRepository()
    pipeline = AutoTradingPipeline(repository=repo)

    result = AutoTuneResult(
        study_name="test",
        trials=1,
        best_score=1.0,
        baseline_score=0.5,
        best_params={"momentum_fast": 10},
        metrics={"roi": 0.1, "profit_factor": 1.5, "max_drawdown": 0.1, "win_rate": 0.55},
        training=None,
        go_live_ready=True,
        passes_thresholds=True,
        baseline_metrics={"roi": 0.0},
        baseline_training=None,
    )

    monkeypatch.setattr("bot.exec.pipeline.tune_intraday_settings", lambda *args, **kwargs: result)
    dummy_df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 11, 12],
        },
        index=pd.date_range("2023-01-01", periods=3, freq="T"),
    )
    monkeypatch.setattr("bot.exec.pipeline._load_data", lambda *args, **kwargs: dummy_df)

    output = pipeline.run(
        "BTCUSDT",
        "1m",
        datetime(2023, 1, 1),
        datetime(2023, 1, 1, 1),
        data_source="synthetic",
        trials=1,
    )

    assert output is result
    assert repo.calls
    recorded = repo.calls[0]
    assert recorded["passes_thresholds"] == 1
