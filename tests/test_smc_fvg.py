import pandas as pd
from smc.fvg import find_fvgs


def test_bullish_fvg_detection_and_states():
    # 3-candle FVG:
    # Candle 0 (1): High = 100
    # Candle 1 (2): Large green candle
    # Candle 2 (3): Low = 110
    # Gap = 110 - 100 = 10.0 (ATR = 10.0, min gap mult 0.25 -> 2.5) -> Valid Bullish FVG
    # Candle 3: Low = 105 (dips into gap -> PARTIAL)
    # Candle 4: Low = 98 (trades through bottom 100 -> FILLED)
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "open": [95.0, 101.0, 112.0, 114.0, 104.0],
        "high": [100.0, 115.0, 120.0, 116.0, 106.0],
        "low": [90.0, 100.5, 110.0, 105.0, 98.0],
        "close": [98.0, 114.0, 118.0, 106.0, 100.0],
    })
    atr_series = pd.Series([10.0] * 5)

    fvgs = find_fvgs(df, atr_series, min_gap_atr_mult=0.25)
    bullish_fvgs = [f for f in fvgs if f.fvg_type == "BULLISH"]

    assert len(bullish_fvgs) == 1
    fvg = bullish_fvgs[0]
    assert fvg.fvg_type == "BULLISH"
    assert fvg.top == 110.0
    assert fvg.bottom == 100.0
    assert fvg.gap_size == 10.0
    assert fvg.state == "FILLED"
