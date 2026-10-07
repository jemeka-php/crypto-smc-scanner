"""
ATR (Average True Range) calculation using Wilder's RMA (Running Moving Average).
Period = 14 by default.
"""

import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder's Smoothing (RMA).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: RMA period, default 14

    Returns:
        pd.Series containing Wilder's ATR values
    """
    if len(df) < 2:
        return pd.Series(np.nan, index=df.index)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.full(n, np.nan)
    if n >= period:
        # Initial ATR is simple moving average of first `period` TR values
        atr[period - 1] = np.mean(tr[:period])
        # Wilder's RMA recursion: atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) * alpha

    return pd.Series(atr, index=df.index, name="atr")
