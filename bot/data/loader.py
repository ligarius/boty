"""Data loading utilities for OHLCV data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import httpx

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


def fetch_binance_ohlcv(
    request: OHLCVRequest,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance Futures API and return a timestamp-indexed DataFrame."""

    settings = settings or Settings()
    base_url = settings.binance_base_url.rstrip("/")
    headers = {}
    if settings.binance_api_key:
        headers["X-MBX-APIKEY"] = settings.binance_api_key

    start_ms = int(request.start.timestamp() * 1000)
    end_ms = int(request.end.timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("end must be after start for OHLCV fetch")

    interval_ms = int(_timeframe_to_timedelta(request.timeframe).total_seconds() * 1000)
    batch_limit = min(max(request.limit, 1), 1500)

    records: list[dict[str, object]] = []
    next_start = start_ms
    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        while next_start < end_ms:
            params = {
                "symbol": request.symbol,
                "interval": request.timeframe,
                "startTime": next_start,
                "endTime": end_ms - 1,
                "limit": batch_limit,
            }
            response = client.get("/fapi/v1/klines", params=params)
            response.raise_for_status()
            candles = response.json()
            if not candles:
                break

            for candle in candles:
                open_time = int(candle[0])
                if open_time < start_ms or open_time >= end_ms:
                    continue
                records.append(
                    {
                        "timestamp": pd.to_datetime(open_time, unit="ms", utc=True).tz_localize(None),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                )

            last_open_time = int(candles[-1][0])
            proposed_next = last_open_time + interval_ms
            if proposed_next <= next_start:
                break
            next_start = proposed_next

    if not records:
        empty_index = pd.DatetimeIndex([], name="timestamp")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=empty_index)

    df = pd.DataFrame.from_records(records)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.set_index("timestamp", inplace=True)
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
