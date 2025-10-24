"""Utilities for validating and cleaning market data feeds."""
from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


class DataQualityError(ValueError):
    """Raised when the input data fails validation checks."""


def validate_ohlcv(df: pd.DataFrame, required_columns: Iterable[str] | None = None) -> None:
    """Ensure OHLCV data satisfies minimum quality thresholds."""

    if df is None or df.empty:
        raise DataQualityError("OHLCV dataset is empty")

    columns = required_columns or ("open", "high", "low", "close", "volume")
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise DataQualityError(f"Missing columns: {', '.join(missing)}")

    if df.index.has_duplicates:
        raise DataQualityError("Duplicate timestamps detected in OHLCV data")

    if df.isna().any().any():
        raise DataQualityError("OHLCV data contains NaN values")

    if not df.index.is_monotonic_increasing:
        raise DataQualityError("OHLCV index must be sorted in ascending order")


def summarize_data(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """Return a tuple with the number of rows and basic descriptive stats."""

    validate_ohlcv(df)
    return len(df), df[["close", "volume"]].describe().loc["mean"]
