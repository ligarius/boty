"""Data loading utilities for OHLCV data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import Settings


@dataclass
class OHLCVRequest:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    limit: int = 1500


def load_local_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def generate_synthetic_data(request: OHLCVRequest, seed: int = 7) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for testing/backtests."""

    np.random.seed(seed)
    periods = int((request.end - request.start) / _timeframe_to_timedelta(request.timeframe))
    index = pd.date_range(request.start, periods=periods, freq=_to_pandas_freq(request.timeframe))
    price = 20000 + np.cumsum(np.random.normal(0, 50, size=periods))
    high = price + np.random.uniform(0, 25, size=periods)
    low = price - np.random.uniform(0, 25, size=periods)
    open_ = price + np.random.uniform(-10, 10, size=periods)
    volume = np.random.uniform(10, 100, size=periods)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    raise ValueError(f"Unsupported timeframe {timeframe}")


def _to_pandas_freq(timeframe: str) -> str:
    unit = timeframe[-1]
    value = timeframe[:-1]
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}H"
    raise ValueError(f"Unsupported timeframe {timeframe}")
