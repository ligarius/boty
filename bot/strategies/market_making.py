"""Simple inventory-aware market making heuristic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame


def market_making_signals(
    data: DataFrame,
    spread: float = 0.0015,
    liquidity_window: int = 30,
    inventory_alpha: float = 0.4,
) -> DataFrame:
    """Generate soft market making signals based on price deviations and liquidity."""

    df = data.copy()
    mid_price = (df["high"] + df["low"]) / 2
    rolling_mid = mid_price.rolling(window=liquidity_window).mean()
    rolling_vol = df["volume"].rolling(window=liquidity_window).mean()
    atr = _atr(df, period=max(14, liquidity_window // 2))

    deviation = (mid_price - rolling_mid) / rolling_mid.replace(0, np.nan)
    liquidity_score = (rolling_vol / rolling_vol.shift()).fillna(1.0)
    inventory_bias = -np.tanh(deviation / max(spread, 1e-6))
    signal_strength = inventory_alpha * inventory_bias + (1 - inventory_alpha) * np.sign(-deviation)

    signal = np.zeros(len(df), dtype=int)
    signal[signal_strength > spread] = 1
    signal[signal_strength < -spread] = -1

    score = signal_strength * (1 / (atr.replace(0, np.nan)))
    result = pd.DataFrame(
        {
            "signal": signal,
            "score": score.fillna(0.0),
            "atr": atr,
            "liquidity_score": liquidity_score.fillna(1.0),
            "deviation": deviation.fillna(0.0),
        },
        index=data.index,
    )
    result["signal"] = result["signal"].astype(int)
    result["score"] = result["score"].astype(float)
    return result


def _atr(df: DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
