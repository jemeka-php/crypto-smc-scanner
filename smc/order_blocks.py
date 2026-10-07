"""
Order Block detection and state tracking.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from smc.displacement import Displacement


@dataclass
class OrderBlock:
    index: int                  # Bar index of the order block candle
    timestamp: int              # Timestamp of the candle
    ob_type: str                # "BULLISH" or "BEARISH"
    high: float                 # OB upper boundary
    low: float                  # OB lower boundary
    open: float
    close: float
    state: str                  # "ACTIVE", "MITIGATED", "INVALIDATED"
    displacement_ratio: float   # Ratio of the triggering displacement
    timeframe: str = "1H"


def find_order_blocks(
    df: pd.DataFrame,
    atr_series: pd.Series,
    displacements: List[Displacement],
    doji_ratio: float = 0.1,
    timeframe: str = "1H",
) -> List[OrderBlock]:
    """
    Find order blocks preceding qualifying displacement moves and evaluate their current state.

    Rules:
    - Bullish OB: last bearish candle before bullish displacement.
    - Bearish OB: last bullish candle before bearish displacement.
    - Doji fallback: if candidate body < doji_ratio * ATR, inspect the preceding candle.
    - State tracking:
      * ACTIVE: unvisited.
      * MITIGATED: price dipped into zone without closing beyond invalidation level.
      * INVALIDATED:
        - Bullish: close < OB.low
        - Bearish: close > OB.high
    """
    n = len(df)
    if n < 2 or not displacements:
        return []

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = atr_series.values
    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else df.index.astype(int).values
    )

    raw_obs: List[OrderBlock] = []

    for disp in displacements:
        start_idx = disp.start_index
        if start_idx <= 0:
            continue

        atr_val = atrs[start_idx] if not pd.isna(atrs[start_idx]) else 1.0

        if disp.direction == "BULLISH":
            # Search backwards from start_idx - 1 for the last bearish candle
            candidate_idx: Optional[int] = None
            for idx in range(start_idx - 1, max(-1, start_idx - 5), -1):
                is_bearish = closes[idx] < opens[idx]
                body_size = abs(closes[idx] - opens[idx])
                is_doji = body_size < (doji_ratio * atr_val)

                if is_bearish and not is_doji:
                    candidate_idx = idx
                    break
                elif is_bearish and is_doji:
                    # Doji fallback: keep going back or use this if no better
                    candidate_idx = idx

            if candidate_idx is None:
                candidate_idx = start_idx - 1

            raw_obs.append(
                OrderBlock(
                    index=candidate_idx,
                    timestamp=int(timestamps[candidate_idx]),
                    ob_type="BULLISH",
                    high=float(highs[candidate_idx]),
                    low=float(lows[candidate_idx]),
                    open=float(opens[candidate_idx]),
                    close=float(closes[candidate_idx]),
                    state="ACTIVE",
                    displacement_ratio=disp.displacement_ratio,
                    timeframe=timeframe,
                )
            )

        elif disp.direction == "BEARISH":
            # Search backwards for the last bullish candle
            candidate_idx: Optional[int] = None
            for idx in range(start_idx - 1, max(-1, start_idx - 5), -1):
                is_bullish = closes[idx] > opens[idx]
                body_size = abs(closes[idx] - opens[idx])
                is_doji = body_size < (doji_ratio * atr_val)

                if is_bullish and not is_doji:
                    candidate_idx = idx
                    break
                elif is_bullish and is_doji:
                    candidate_idx = idx

            if candidate_idx is None:
                candidate_idx = start_idx - 1

            raw_obs.append(
                OrderBlock(
                    index=candidate_idx,
                    timestamp=int(timestamps[candidate_idx]),
                    ob_type="BEARISH",
                    high=float(highs[candidate_idx]),
                    low=float(lows[candidate_idx]),
                    open=float(opens[candidate_idx]),
                    close=float(closes[candidate_idx]),
                    state="ACTIVE",
                    displacement_ratio=disp.displacement_ratio,
                    timeframe=timeframe,
                )
            )

    # Now evaluate states chronologically from creation up to the latest candle
    tracked_obs: List[OrderBlock] = []
    seen_indices = set()

    for ob in raw_obs:
        if (ob.index, ob.ob_type) in seen_indices:
            continue
        seen_indices.add((ob.index, ob.ob_type))

        current_state = "ACTIVE"
        for bar in range(ob.index + 1, n):
            c_high = highs[bar]
            c_low = lows[bar]
            c_close = closes[bar]

            if ob.ob_type == "BULLISH":
                # Invalidation: close below OB.low
                if c_close < ob.low:
                    current_state = "INVALIDATED"
                    break
                # Mitigation: price enters zone
                elif c_low <= ob.high and current_state == "ACTIVE":
                    current_state = "MITIGATED"

            elif ob.ob_type == "BEARISH":
                # Invalidation: close above OB.high
                if c_close > ob.high:
                    current_state = "INVALIDATED"
                    break
                # Mitigation: price enters zone
                elif c_high >= ob.low and current_state == "ACTIVE":
                    current_state = "MITIGATED"

        ob.state = current_state
        tracked_obs.append(ob)

    return tracked_obs
