"""
Displacement detection based on consecutive same-direction candle bodies and ATR multiples.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class Displacement:
    start_index: int
    end_index: int
    start_timestamp: int
    end_timestamp: int
    direction: str              # "BULLISH" or "BEARISH"
    total_displacement: float   # Net body displacement in price
    atr: float                  # ATR value at displacement trigger
    displacement_ratio: float   # total_displacement / atr


def detect_displacements(
    df: pd.DataFrame,
    atr_series: pd.Series,
    min_atr_mult: float = 1.5,
    max_bars: int = 3,
) -> List[Displacement]:
    """
    Detect displacement events where 1 to max_bars consecutive same-direction candle bodies
    achieve a total body displacement >= min_atr_mult * ATR.

    Bullish displacement:
        All candles in sequence have close >= open.
        Net body movement = close[end] - open[start] >= min_atr_mult * ATR[end].

    Bearish displacement:
        All candles in sequence have close <= open.
        Net body movement = open[start] - close[end] >= min_atr_mult * ATR[end].

    Args:
        df: DataFrame with 'open', 'close', 'timestamp'
        atr_series: pd.Series of ATR
        min_atr_mult: Minimum ratio of displacement to ATR (default 1.5)
        max_bars: Window of consecutive candles (default 3)

    Returns:
        List of Displacement objects.
    """
    n = len(df)
    if n < 2:
        return []

    opens = df["open"].values
    closes = df["close"].values
    atrs = atr_series.values
    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else df.index.astype(int).values
    )

    displacements: List[Displacement] = []

    for i in range(n):
        current_atr = atrs[i]
        if pd.isna(current_atr) or current_atr <= 0:
            continue

        # Check sequences ending at index i of length 1..max_bars
        for k in range(1, min(max_bars + 1, i + 2)):
            start_idx = i - k + 1
            seq_opens = opens[start_idx : i + 1]
            seq_closes = closes[start_idx : i + 1]

            # Bullish check: all candles green/neutral and net body move >= threshold
            all_bullish = all(c >= o for o, c in zip(seq_opens, seq_closes))
            if all_bullish:
                net_disp = closes[i] - opens[start_idx]
                ratio = net_disp / current_atr
                if ratio >= min_atr_mult:
                    displacements.append(
                        Displacement(
                            start_index=start_idx,
                            end_index=i,
                            start_timestamp=int(timestamps[start_idx]),
                            end_timestamp=int(timestamps[i]),
                            direction="BULLISH",
                            total_displacement=float(net_disp),
                            atr=float(current_atr),
                            displacement_ratio=round(float(ratio), 2),
                        )
                    )
                    break  # Found displacement ending at i

            # Bearish check: all candles red/neutral and net body move >= threshold
            all_bearish = all(c <= o for o, c in zip(seq_opens, seq_closes))
            if all_bearish:
                net_disp = opens[start_idx] - closes[i]
                ratio = net_disp / current_atr
                if ratio >= min_atr_mult:
                    displacements.append(
                        Displacement(
                            start_index=start_idx,
                            end_index=i,
                            start_timestamp=int(timestamps[start_idx]),
                            end_timestamp=int(timestamps[i]),
                            direction="BEARISH",
                            total_displacement=float(net_disp),
                            atr=float(current_atr),
                            displacement_ratio=round(float(ratio), 2),
                        )
                    )
                    break

    return displacements
