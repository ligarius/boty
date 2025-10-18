"""Mean reversion strategy using Bollinger bands and RSI."""
from __future__ import annotations

import pandas as pd
from pandas import DataFrame


def mean_reversion_signals(data: DataFrame, window: int = 20, z_threshold: float = 1.5) -> DataFrame:
    """Generate mean reversion signals from OHLCV data."""

    df = data.copy()
    df["rolling_mean"] = df["close"].rolling(window=window).mean()
    df["rolling_std"] = df["close"].rolling(window=window).std(ddof=0)
    df["zscore"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    df["rsi"] = _rsi(df["close"], period=window)
    df["atr"] = _atr(df)
    df["signal"] = 0
    df.loc[df["zscore"] > z_threshold, "signal"] = -1
    df.loc[df["zscore"] < -z_threshold, "signal"] = 1
    df.loc[df["rsi"] > 70, "signal"] = -1
    df.loc[df["rsi"] < 30, "signal"] = 1
    df["score"] = df["signal"] * (1 / (df["atr"].replace(0, pd.NA)))
    df["score"].fillna(0.0, inplace=True)
    return df[["signal", "score", "atr", "rsi", "zscore"]].dropna()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = down.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(df: DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
