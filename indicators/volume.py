"""
Volume indicators including Relative Volume (RVOL) and Volume Classification.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def calculate_rvol(
    completed_volumes: pd.Series,
    lookback: int = 20,
) -> Tuple[float, float, str]:
    """
    Calculate RVOL for the most recent completed candle relative to the previous N completed candles.

    RVOL = current completed 1H volume / median volume of previous 20 completed 1H candles.

    Args:
        completed_volumes: Series of volume for COMPLETED candles only
        lookback: number of previous completed candles to compute median (default 20)

    Returns:
        Tuple of (rvol_value, median_volume, classification_label)
        where classification_label is 'HIGH' (>= 1.5x), 'LOW' (<= 0.5x), or 'AVERAGE'.
    """
    if len(completed_volumes) < lookback + 1:
        # Not enough data for confirmed RVOL
        return 0.0, 0.0, "UNKNOWN"

    # Current completed volume is the last item
    current_vol = float(completed_volumes.iloc[-1])
    # Previous completed candles (excluding the current completed one)
    prev_vols = completed_volumes.iloc[-(lookback + 1): -1]
    median_vol = float(prev_vols.median())

    if median_vol <= 0:
        return 0.0, 0.0, "UNKNOWN"

    rvol = current_vol / median_vol

    if rvol >= 1.5:
        classification = "HIGH"
    elif rvol <= 0.5:
        classification = "LOW"
    else:
        classification = "AVERAGE"

    return round(rvol, 2), median_vol, classification


def add_volume_profile(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Adds rolling median volume and rolling rvol series to a DataFrame.
    """
    df = df.copy()
    rolling_median = df["volume"].shift(1).rolling(window=lookback).median()
    df["vol_median"] = rolling_median
    df["rvol"] = df["volume"] / rolling_median.replace(0, np.nan)
    return df
