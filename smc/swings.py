"""
Confirmed swing highs and swing lows detection with zero lookahead bias.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class Swing:
    index: int                  # Bar index where swing occurred
    timestamp: int              # Timestamp of the swing bar
    price: float                # Swing high or low price
    swing_type: str             # "HIGH" or "LOW"
    confirmed_at_index: int     # Bar index where swing was confirmed (index + lookback)
    confirmed_at_timestamp: int # Timestamp of confirmation bar


def find_swings(df: pd.DataFrame, lookback: int = 5) -> List[Swing]:
    """
    Find confirmed swing highs and lows with strictly chronological confirmation.

    A swing high at bar `i` is confirmed at bar `i + lookback` if:
        high[i] > high[i - k] for all k in 1..lookback
        high[i] >= high[i + k] for all k in 1..lookback (tie-breaking: first occurrence wins)

    A swing low at bar `i` is confirmed at bar `i + lookback` if:
        low[i] < low[i - k] for all k in 1..lookback
        low[i] <= low[i + k] for all k in 1..lookback (tie-breaking: first occurrence wins)

    Args:
        df: DataFrame with 'high', 'low', 'timestamp' (or index)
        lookback: number of bars on left and right (default 5)

    Returns:
        List of confirmed Swing objects sorted by confirmed_at_index ascending.
    """
    n = len(df)
    if n < (lookback * 2 + 1):
        return []

    highs = df["high"].values
    lows = df["low"].values
    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else df.index.astype(int).values
    )

    swings: List[Swing] = []

    # A swing at `i` can only be evaluated up to `n - 1 - lookback`
    for i in range(lookback, n - lookback):
        # Check swing high
        is_swing_high = True
        curr_high = highs[i]

        for k in range(1, lookback + 1):
            if highs[i - k] >= curr_high:
                is_swing_high = False
                break
            if highs[i + k] > curr_high:
                is_swing_high = False
                break

        if is_swing_high:
            swings.append(
                Swing(
                    index=i,
                    timestamp=int(timestamps[i]),
                    price=float(curr_high),
                    swing_type="HIGH",
                    confirmed_at_index=i + lookback,
                    confirmed_at_timestamp=int(timestamps[i + lookback]),
                )
            )

        # Check swing low
        is_swing_low = True
        curr_low = lows[i]

        for k in range(1, lookback + 1):
            if lows[i - k] <= curr_low:
                is_swing_low = False
                break
            if lows[i + k] < curr_low:
                is_swing_low = False
                break

        if is_swing_low:
            swings.append(
                Swing(
                    index=i,
                    timestamp=int(timestamps[i]),
                    price=float(curr_low),
                    swing_type="LOW",
                    confirmed_at_index=i + lookback,
                    confirmed_at_timestamp=int(timestamps[i + lookback]),
                )
            )

    # Sort deterministically by confirmed_at_index then index
    swings.sort(key=lambda s: (s.confirmed_at_index, s.index))
    return swings


def get_latest_swings(
    swings: List[Swing],
) -> tuple[Optional[Swing], Optional[Swing]]:
    """
    Get the most recent confirmed swing high and confirmed swing low.
    """
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None

    for s in reversed(swings):
        if s.swing_type == "HIGH" and last_high is None:
            last_high = s
        elif s.swing_type == "LOW" and last_low is None:
            last_low = s
        if last_high and last_low:
            break

    return last_high, last_low
