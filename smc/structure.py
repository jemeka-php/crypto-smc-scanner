"""
Market Structure: Break of Structure (BOS), Market Structure Shift (MSS), and Trend Bias.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from smc.swings import Swing, find_swings
from smc.displacement import Displacement, detect_displacements


@dataclass
class StructureBreak:
    index: int                  # Bar where break occurred
    timestamp: int              # Timestamp of the break bar
    break_type: str             # "BOS" or "MSS"
    direction: str              # "BULLISH" or "BEARISH"
    broken_level: float         # The swing price that was broken
    broken_swing: Swing         # The swing broken
    has_displacement: bool      # Whether displacement accompanied the break
    displacement: Optional[Displacement] = None


@dataclass
class MarketStructure:
    timeframe: str              # e.g., "4H", "1H", "15M"
    bias: str                   # "BULLISH", "BEARISH", "NEUTRAL"
    swings: List[Swing]
    latest_swing_high: Optional[Swing]
    latest_swing_low: Optional[Swing]
    recent_breaks: List[StructureBreak]
    latest_bos: Optional[StructureBreak]
    latest_mss: Optional[StructureBreak]


def analyze_market_structure(
    df: pd.DataFrame,
    atr_series: pd.Series,
    timeframe: str = "1H",
    swing_lookback: int = 5,
    mss_requires_displacement: bool = True,
    mss_lookback_bars: int = 3,
    displacement_atr_mult: float = 1.5,
) -> MarketStructure:
    """
    Evaluate swings, BOS, MSS, and directional bias for a timeframe.

    Rules:
    - Bullish BOS: Price closes above a confirmed swing high.
    - Bearish BOS: Price closes below a confirmed swing low.
    - MSS: An opposite-direction break of a confirmed structure level within an existing
      directional context, with qualifying displacement.
    """
    n = len(df)
    if n < (swing_lookback * 2 + 1):
        return MarketStructure(
            timeframe=timeframe,
            bias="NEUTRAL",
            swings=[],
            latest_swing_high=None,
            latest_swing_low=None,
            recent_breaks=[],
            latest_bos=None,
            latest_mss=None,
        )

    swings = find_swings(df, lookback=swing_lookback)
    displacements = detect_displacements(
        df,
        atr_series,
        min_atr_mult=displacement_atr_mult,
        max_bars=mss_lookback_bars,
    )

    closes = df["close"].values
    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else df.index.astype(int).values
    )

    active_highs: List[Swing] = []
    active_lows: List[Swing] = []
    breaks: List[StructureBreak] = []

    current_bias = "NEUTRAL"

    # Track swing activations as of each candle
    swing_idx = 0
    total_swings = len(swings)

    for i in range(n):
        # Add swings that have been confirmed at or before bar i
        while swing_idx < total_swings and swings[swing_idx].confirmed_at_index <= i:
            s = swings[swing_idx]
            if s.swing_type == "HIGH":
                active_highs.append(s)
            else:
                active_lows.append(s)
            swing_idx += 1

        curr_close = closes[i]

        # Check Bullish break: close > swing high
        broken_h = [h for h in active_highs if curr_close > h.price]
        for bh in broken_h:
            # Check displacement completed at or before this bar (within mss_lookback_bars)
            matching_disp = next(
                (
                    d for d in displacements
                    if d.direction == "BULLISH"
                    and d.end_index <= i
                    and (i - d.end_index) <= mss_lookback_bars
                ),
                None,
            )
            has_disp = matching_disp is not None

            # Determine whether this is BOS (trend continuation) or MSS (trend reversal)
            is_mss = (current_bias == "BEARISH") and (not mss_requires_displacement or has_disp)
            break_type = "MSS" if is_mss else "BOS"

            st_break = StructureBreak(
                index=i,
                timestamp=int(timestamps[i]),
                break_type=break_type,
                direction="BULLISH",
                broken_level=bh.price,
                broken_swing=bh,
                has_displacement=has_disp,
                displacement=matching_disp,
            )
            breaks.append(st_break)
            current_bias = "BULLISH"
            active_highs.remove(bh)

        # Check Bearish break: close < swing low
        broken_l = [l for l in active_lows if curr_close < l.price]
        for bl in broken_l:
            matching_disp = next(
                (
                    d for d in displacements
                    if d.direction == "BEARISH"
                    and d.end_index <= i
                    and (i - d.end_index) <= mss_lookback_bars
                ),
                None,
            )
            has_disp = matching_disp is not None

            is_mss = (current_bias == "BULLISH") and (not mss_requires_displacement or has_disp)
            break_type = "MSS" if is_mss else "BOS"

            st_break = StructureBreak(
                index=i,
                timestamp=int(timestamps[i]),
                break_type=break_type,
                direction="BEARISH",
                broken_level=bl.price,
                broken_swing=bl,
                has_displacement=has_disp,
                displacement=matching_disp,
            )
            breaks.append(st_break)
            current_bias = "BEARISH"
            active_lows.remove(bl)

    # Latest swings
    last_h = next((s for s in reversed(swings) if s.swing_type == "HIGH"), None)
    last_l = next((s for s in reversed(swings) if s.swing_type == "LOW"), None)

    latest_bos = next((b for b in reversed(breaks) if b.break_type == "BOS"), None)
    latest_mss = next((b for b in reversed(breaks) if b.break_type == "MSS"), None)

    return MarketStructure(
        timeframe=timeframe,
        bias=current_bias,
        swings=swings,
        latest_swing_high=last_h,
        latest_swing_low=last_l,
        recent_breaks=breaks[-10:],
        latest_bos=latest_bos,
        latest_mss=latest_mss,
    )
