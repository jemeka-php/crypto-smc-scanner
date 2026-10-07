"""
Fair Value Gap (FVG) detection and state tracking (OPEN, PARTIAL, FILLED).
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class FVG:
    index: int                  # Index of the middle candle (candle 2)
    timestamp: int              # Timestamp of candle 2
    fvg_type: str               # "BULLISH" or "BEARISH"
    top: float                  # Upper price boundary
    bottom: float               # Lower price boundary
    gap_size: float             # Gap size in price
    atr: float                  # ATR value
    state: str                  # "OPEN", "PARTIAL", "FILLED"
    timeframe: str = "1H"


def find_fvgs(
    df: pd.DataFrame,
    atr_series: pd.Series,
    min_gap_atr_mult: float = 0.25,
    timeframe: str = "1H",
    completed_candles_only: bool = False,
) -> List[FVG]:
    """
    Detect standard 3-candle Fair Value Gaps and track their fill states.

    Bullish FVG:
        candle 1 high < candle 3 low
        gap = low[3] - high[1] >= min_gap_atr_mult * ATR
        Top = low[3], Bottom = high[1]

    Bearish FVG:
        candle 1 low > candle 3 high
        gap = low[1] - high[3] >= min_gap_atr_mult * ATR
        Top = low[1], Bottom = high[3]

    States:
        - OPEN: zone untouched
        - PARTIAL: price touched into the zone but did not fully close through
        - FILLED: price traded completely through the zone
    """
    n = len(df)
    if n < 3:
        return []

    highs = df["high"].values
    lows = df["low"].values
    atrs = atr_series.values
    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else df.index.astype(int).values
    )

    fvgs: List[FVG] = []

    # Standard 3-candle FVG: candle 1 is i-1, candle 2 is i, candle 3 is i+1
    # When completed_candles_only is True, candle 3 must be a closed candle (< n - 1),
    # preventing the unclosed forming bar at n - 1 from serving as candle 3.
    max_idx = (n - 2) if completed_candles_only else (n - 1)
    for i in range(1, max_idx):
        atr_val = atrs[i]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        min_required_gap = min_gap_atr_mult * atr_val

        # Bullish FVG
        if lows[i + 1] > highs[i - 1]:
            gap = lows[i + 1] - highs[i - 1]
            if gap >= min_required_gap:
                top = float(lows[i + 1])
                bottom = float(highs[i - 1])
                # Track state from i + 2 to n - 1
                state = "OPEN"
                for k in range(i + 2, n):
                    curr_low = lows[k]
                    if curr_low <= bottom:
                        state = "FILLED"
                        break
                    elif curr_low < top:
                        state = "PARTIAL"

                fvgs.append(
                    FVG(
                        index=i,
                        timestamp=int(timestamps[i]),
                        fvg_type="BULLISH",
                        top=top,
                        bottom=bottom,
                        gap_size=round(float(gap), 4),
                        atr=float(atr_val),
                        state=state,
                        timeframe=timeframe,
                    )
                )

        # Bearish FVG
        elif highs[i + 1] < lows[i - 1]:
            gap = lows[i - 1] - highs[i + 1]
            if gap >= min_required_gap:
                top = float(lows[i - 1])
                bottom = float(highs[i + 1])
                # Track state from i + 2 to n - 1
                state = "OPEN"
                for k in range(i + 2, n):
                    curr_high = highs[k]
                    if curr_high >= top:
                        state = "FILLED"
                        break
                    elif curr_high > bottom:
                        state = "PARTIAL"

                fvgs.append(
                    FVG(
                        index=i,
                        timestamp=int(timestamps[i]),
                        fvg_type="BEARISH",
                        top=top,
                        bottom=bottom,
                        gap_size=round(float(gap), 4),
                        atr=float(atr_val),
                        state=state,
                        timeframe=timeframe,
                    )
                )

    return fvgs
