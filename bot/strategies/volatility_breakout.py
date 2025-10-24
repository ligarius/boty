"""Volatility breakout strategy that reacts to range expansions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame


def volatility_breakout_signals(
    data: DataFrame,
    lookback: int = 20,
    atr_period: int = 14,
    breakout_multiplier: float = 1.5,
    cooldown: int = 3,
) -> DataFrame:
    """Generate breakout signals based on rolling highs/lows and ATR."""

    df = data.copy()
    df["atr"] = _atr(df, period=atr_period)
    df["range_high"] = df["high"].rolling(window=lookback).max()
    df["range_low"] = df["low"].rolling(window=lookback).min()
    df["breakout_high"] = df["range_high"] + df["atr"] * breakout_multiplier
    df["breakout_low"] = df["range_low"] - df["atr"] * breakout_multiplier

    signal = np.zeros(len(df), dtype=int)
    active_cooldown = 0
    for idx in range(len(df)):
        if active_cooldown > 0:
            active_cooldown -= 1
            continue
        price = df.iloc[idx]["close"]
        breakout_high = df.iloc[idx]["breakout_high"]
        breakout_low = df.iloc[idx]["breakout_low"]
        if np.isnan(price) or np.isnan(breakout_high) or np.isnan(breakout_low):
            continue
        if price > breakout_high:
            signal[idx] = 1
            active_cooldown = cooldown
        elif price < breakout_low:
            signal[idx] = -1
            active_cooldown = cooldown

    df["signal"] = signal
    df["score"] = df["signal"] * (1 / (df["atr"].replace(0, np.nan)))
    df["score"] = df["score"].fillna(0.0)
    return df[["signal", "score", "atr", "breakout_high", "breakout_low"]].reindex(data.index)


def _atr(df: DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
