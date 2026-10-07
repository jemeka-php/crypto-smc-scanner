"""
Liquidity pools, sweeps (Sell-side / Buy-side), and HTF Location detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

from smc.swings import Swing, find_swings
from smc.order_blocks import OrderBlock
from smc.fvg import FVG


@dataclass
class LiquidityLevel:
    price: float
    timeframe: str              # "4H", "1H", "15M", "D"
    source: str                 # "SWING_HIGH", "SWING_LOW", "EQH", "EQL", "PDH", "PDL"
    created_at: int             # Timestamp or index
    status: str                 # "ACTIVE", "SWEPT", "VIOLATED"
    side: str                   # "BUY_SIDE" or "SELL_SIDE"


@dataclass
class LiquiditySweep:
    level: LiquidityLevel
    sweep_type: str             # "BULLISH_SWEEP" (swept sell-side) or "BEARISH_SWEEP" (swept buy-side)
    extreme_price: float        # Low of the sweep wick or high of the sweep wick
    close_price: float          # Close that validated the sweep
    bar_index: int
    timestamp: int


def detect_liquidity_levels(
    df: pd.DataFrame,
    swings: List[Swing],
    timeframe: str = "1H",
    eq_tolerance_pct: float = 0.1,
    pdh_pdl: Optional[Tuple[float, float]] = None,
) -> List[LiquidityLevel]:
    """
    Generate liquidity levels from confirmed swings, equal highs/lows, and PDH/PDL.
    """
    levels: List[LiquidityLevel] = []

    # 1. Swings
    high_swings = [s for s in swings if s.swing_type == "HIGH"]
    low_swings = [s for s in swings if s.swing_type == "LOW"]

    for s in high_swings:
        levels.append(
            LiquidityLevel(
                price=s.price,
                timeframe=timeframe,
                source="SWING_HIGH",
                created_at=s.timestamp,
                status="ACTIVE",
                side="BUY_SIDE",
            )
        )

    for s in low_swings:
        levels.append(
            LiquidityLevel(
                price=s.price,
                timeframe=timeframe,
                source="SWING_LOW",
                created_at=s.timestamp,
                status="ACTIVE",
                side="SELL_SIDE",
            )
        )

    # 2. Equal Highs (EQH) and Equal Lows (EQL)
    # Check pairs of recent swings within tolerance
    for i in range(len(high_swings)):
        for j in range(i + 1, min(i + 4, len(high_swings))):
            p1, p2 = high_swings[i].price, high_swings[j].price
            diff_pct = abs(p1 - p2) / p1 * 100.0
            if diff_pct <= eq_tolerance_pct:
                levels.append(
                    LiquidityLevel(
                        price=round((p1 + p2) / 2.0, 4),
                        timeframe=timeframe,
                        source="EQH",
                        created_at=high_swings[j].timestamp,
                        status="ACTIVE",
                        side="BUY_SIDE",
                    )
                )

    for i in range(len(low_swings)):
        for j in range(i + 1, min(i + 4, len(low_swings))):
            p1, p2 = low_swings[i].price, low_swings[j].price
            diff_pct = abs(p1 - p2) / p1 * 100.0
            if diff_pct <= eq_tolerance_pct:
                levels.append(
                    LiquidityLevel(
                        price=round((p1 + p2) / 2.0, 4),
                        timeframe=timeframe,
                        source="EQL",
                        created_at=low_swings[j].timestamp,
                        status="ACTIVE",
                        side="SELL_SIDE",
                    )
                )

    # 3. Previous Day High / Low
    if pdh_pdl:
        pdh, pdl = pdh_pdl
        levels.append(
            LiquidityLevel(
                price=pdh,
                timeframe="1D",
                source="PDH",
                created_at=int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else 0,
                status="ACTIVE",
                side="BUY_SIDE",
            )
        )
        levels.append(
            LiquidityLevel(
                price=pdl,
                timeframe="1D",
                source="PDL",
                created_at=int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else 0,
                status="ACTIVE",
                side="SELL_SIDE",
            )
        )

    return levels


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    levels: List[LiquidityLevel],
    lookback_bars: int = 5,
) -> List[LiquiditySweep]:
    """
    Detect liquidity sweeps across recent bars.

    Bullish sweep:
        Price trades below identified sell-side liquidity (low < level.price)
        AND subsequently closes back above the level (close > level.price).

    Bearish sweep:
        Price trades above identified buy-side liquidity (high > level.price)
        AND subsequently closes back below the level (close < level.price).
    """
    n = len(df)
    if n < 2 or not levels:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else df.index.astype(int).values
    )

    sweeps: List[LiquiditySweep] = []
    start_bar = max(0, n - lookback_bars)
    swept_level_ids = set()

    for i in range(start_bar, n):
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]

        for lvl_idx, lvl in enumerate(levels):
            if lvl_idx in swept_level_ids:
                continue

            if lvl.side == "SELL_SIDE":
                # Bullish sweep check: trades below sell-side liquidity then closes back above
                if curr_low < lvl.price and curr_close > lvl.price:
                    sweeps.append(
                        LiquiditySweep(
                            level=lvl,
                            sweep_type="BULLISH_SWEEP",
                            extreme_price=float(curr_low),
                            close_price=float(curr_close),
                            bar_index=i,
                            timestamp=int(timestamps[i]),
                        )
                    )
                    swept_level_ids.add(lvl_idx)
            elif lvl.side == "BUY_SIDE":
                # Bearish sweep check: trades above buy-side liquidity then closes back below
                if curr_high > lvl.price and curr_close < lvl.price:
                    sweeps.append(
                        LiquiditySweep(
                            level=lvl,
                            sweep_type="BEARISH_SWEEP",
                            extreme_price=float(curr_high),
                            close_price=float(curr_close),
                            bar_index=i,
                            timestamp=int(timestamps[i]),
                        )
                    )
                    swept_level_ids.add(lvl_idx)

    return sweeps


def determine_htf_location(
    current_price: float,
    htf_obs: List[OrderBlock],
    htf_fvgs: List[FVG],
    htf_levels: List[LiquidityLevel],
    htf_swings: List[Swing],
    proximity_pct: float = 0.8,
    timeframe: str = "HTF",
) -> Tuple[str, str]:
    """
    Determine whether current price is near a meaningful higher-timeframe area.

    Display options:
    - NEAR DEMAND
    - NEAR SUPPLY
    - NEAR LIQUIDITY
    - MID-RANGE
    - NO SIGNIFICANT LOCATION

    Returns:
        Tuple of (location_code, description)
    """
    if current_price <= 0:
        return "NO SIGNIFICANT LOCATION", "Invalid current price."

    # 1. Check near Demand (Active Bullish OB or Bullish FVG)
    # Directional guard: If current_price > ob.high, price is cleared above demand zone -> does NOT qualify
    for ob in htf_obs:
        if ob.ob_type == "BULLISH" and ob.state in ("ACTIVE", "MITIGATED"):
            if ob.low * (1.0 - proximity_pct / 100.0) <= current_price <= ob.high:
                return "NEAR DEMAND", f"Within {timeframe} Bullish OB zone [{ob.low:.2f} - {ob.high:.2f}]"

    for fvg in htf_fvgs:
        if fvg.fvg_type == "BULLISH" and fvg.state in ("OPEN", "PARTIAL"):
            if fvg.bottom * (1.0 - proximity_pct / 100.0) <= current_price <= fvg.top:
                return "NEAR DEMAND", f"Within {timeframe} Bullish FVG [{fvg.bottom:.2f} - {fvg.top:.2f}]"

    # 2. Check near Supply (Active Bearish OB or Bearish FVG)
    # Directional guard: If current_price < ob.low, price is cleared below supply zone -> does NOT qualify
    for ob in htf_obs:
        if ob.ob_type == "BEARISH" and ob.state in ("ACTIVE", "MITIGATED"):
            if ob.low <= current_price <= ob.high * (1.0 + proximity_pct / 100.0):
                return "NEAR SUPPLY", f"Within {timeframe} Bearish OB zone [{ob.low:.2f} - {ob.high:.2f}]"

    for fvg in htf_fvgs:
        if fvg.fvg_type == "BEARISH" and fvg.state in ("OPEN", "PARTIAL"):
            if fvg.bottom <= current_price <= fvg.top * (1.0 + proximity_pct / 100.0):
                return "NEAR SUPPLY", f"Within {timeframe} Bearish FVG [{fvg.bottom:.2f} - {fvg.top:.2f}]"

    # 3. Check near Liquidity pool (Within proximity of key levels)
    for lvl in htf_levels:
        diff_pct = abs(current_price - lvl.price) / current_price * 100.0
        if diff_pct <= proximity_pct:
            return "NEAR LIQUIDITY", f"Close to {lvl.source} ({lvl.side}) at {lvl.price:.2f}"

    # 4. Check Range position between latest swing high and swing low
    high_swing = next((s for s in reversed(htf_swings) if s.swing_type == "HIGH"), None)
    low_swing = next((s for s in reversed(htf_swings) if s.swing_type == "LOW"), None)

    if high_swing and low_swing and high_swing.price > low_swing.price:
        range_span = high_swing.price - low_swing.price
        rel_pos = (current_price - low_swing.price) / range_span
        if 0.40 <= rel_pos <= 0.60:
            return "MID-RANGE", f"In {timeframe} equilibrium mid-range (~{rel_pos*100:.0f}% of swing span)"

    return "NO SIGNIFICANT LOCATION", f"Price is between key {timeframe} reference points."
