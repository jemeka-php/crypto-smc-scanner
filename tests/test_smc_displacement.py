import pandas as pd
from smc.displacement import detect_displacements


def test_detect_displacements():
    # Construct 3 green candles with total body move > 1.5 * ATR
    # ATR is constant 10.0
    # Candle 1: Open 100, Close 110 (move 10)
    # Candle 2: Open 110, Close 120 (move 10)
    # Candle 3: Open 120, Close 125 (move 5)
    # Total displacement = 125 - 100 = 25 >= 1.5 * 10 = 15.0
    df = pd.DataFrame({
        "timestamp": [1, 2, 3, 4],
        "open": [90.0, 100.0, 110.0, 120.0],
        "high": [95.0, 112.0, 122.0, 128.0],
        "low": [88.0, 99.0, 109.0, 119.0],
        "close": [92.0, 110.0, 120.0, 125.0],
    })
    atr_series = pd.Series([10.0, 10.0, 10.0, 10.0])

    displacements = detect_displacements(df, atr_series, min_atr_mult=1.5, max_bars=3)

    assert len(displacements) >= 1
    bullish = [d for d in displacements if d.direction == "BULLISH"]
    assert len(bullish) >= 1
    assert bullish[-1].displacement_ratio >= 1.5
