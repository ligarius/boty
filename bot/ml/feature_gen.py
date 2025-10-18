"""Feature engineering utilities for ML models."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a set of technical indicators for ML models."""

    features = pd.DataFrame(index=df.index)
    features["returns"] = df["close"].pct_change().fillna(0.0)
    features["volatility"] = df["returns"].rolling(window=20).std().fillna(0.0)
    features["rsi"] = _rsi(df["close"], period=14)
    features["atr"] = _atr(df)
    features["ma_fast"] = df["close"].rolling(window=9).mean().fillna(method="bfill")
    features["ma_slow"] = df["close"].rolling(window=26).mean().fillna(method="bfill")
    features["ma_ratio"] = (features["ma_fast"] / features["ma_slow"] - 1).fillna(0.0)
    features["volume_z"] = (df["volume"] - df["volume"].rolling(window=20).mean()) / df["volume"].rolling(window=20).std()
    features["volume_z"].fillna(0.0, inplace=True)
    features["price_z"] = (df["close"] - df["close"].rolling(window=20).mean()) / df["close"].rolling(window=20).std()
    features["price_z"].fillna(0.0, inplace=True)
    return features.fillna(0.0)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().fillna(method="bfill")
