"""Momentum strategy using moving average crossover with ADX/ATR filters."""
from __future__ import annotations

import pandas as pd
from pandas import DataFrame


def momentum_signals(data: DataFrame, fast: int = 9, slow: int = 21, adx_period: int = 14) -> DataFrame:
    """Generate momentum trading signals."""

    df = data.copy()
    df["ma_fast"] = df["close"].rolling(window=fast).mean()
    df["ma_slow"] = df["close"].rolling(window=slow).mean()
    df["atr"] = _atr(df, period=14)
    df["adx"] = _adx(df, period=adx_period)
    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int) - (df["ma_fast"] < df["ma_slow"]).astype(int)
    df["signal"] = df.apply(lambda row: row["trend"] if row["adx"] > 20 else 0, axis=1)
    df["score"] = df["signal"] * (1 / (df["atr"].replace(0, pd.NA)))
    df["score"].fillna(0.0, inplace=True)
    return df[["signal", "score", "atr", "adx"]].dropna()


def _atr(df: DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _adx(df: DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = _atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / tr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = 100 * dx.ewm(alpha=1 / period).mean()
    return adx.fillna(0.0)
