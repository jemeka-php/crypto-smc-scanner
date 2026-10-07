import pandas as pd
from smc.swings import Swing
from smc.liquidity import (
    LiquidityLevel,
    detect_liquidity_levels,
    detect_liquidity_sweeps,
    determine_htf_location,
)
from smc.order_blocks import OrderBlock


def test_liquidity_sweep_detection():
    # Level at 100.0 (Sell-side)
    # Candle dips to 95.0 but closes at 102.0 -> Bullish sweep
    level = LiquidityLevel(
        price=100.0,
        timeframe="1H",
        source="SWING_LOW",
        created_at=100,
        status="ACTIVE",
        side="SELL_SIDE",
    )

    df = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "open": [105.0, 101.0, 99.0],
        "high": [106.0, 103.0, 104.0],
        "low": [102.0, 95.0, 98.0],
        "close": [103.0, 102.0, 103.0],  # Bar 1 dips to 95 and closes at 102
    })

    sweeps = detect_liquidity_sweeps(df, [level], lookback_bars=3)

    assert len(sweeps) == 1
    assert sweeps[0].sweep_type == "BULLISH_SWEEP"
    assert sweeps[0].extreme_price == 95.0
    assert sweeps[0].close_price == 102.0


def test_determine_htf_location():
    # 4H Bullish OB between 100 and 105
    ob = OrderBlock(
        index=0,
        timestamp=100,
        ob_type="BULLISH",
        high=105.0,
        low=100.0,
        open=104.0,
        close=101.0,
        state="ACTIVE",
        displacement_ratio=2.0,
        timeframe="4H",
    )

    # Price at 102.0 (inside demand)
    loc, desc = determine_htf_location(
        current_price=102.0,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
    )
    assert loc == "NEAR DEMAND"
