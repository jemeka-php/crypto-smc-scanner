import pandas as pd
from smc.structure import analyze_market_structure


def test_bos_requires_closing_break():
    # Setup dataframe where candle reaches swing high but:
    # Case A: Wick exceeds swing high but close is below -> NO BOS
    # Case B: Close exceeds swing high -> BOS
    n = 20
    opens = [100.0] * n
    highs = [105.0] * n
    lows = [95.0] * n
    closes = [100.0] * n

    # Swing high at index 4 with lookback 2:
    highs[4] = 120.0
    closes[4] = 115.0
    # confirmed at 4 + 2 = 6

    # Bar 10: wick to 125, but close at 118 (< 120) -> Wick only, NOT BOS
    highs[10] = 125.0
    closes[10] = 118.0

    # Bar 12: close at 122 (> 120) -> Qualifying close -> BOS
    highs[12] = 123.0
    closes[12] = 122.0

    df = pd.DataFrame({
        "timestamp": list(range(n)),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })
    atr_series = pd.Series([5.0] * n)

    structure = analyze_market_structure(df, atr_series, swing_lookback=2)

    # Must detect the BOS on bar 12
    bos_breaks = [b for b in structure.recent_breaks if b.broken_level == 120.0]
    assert len(bos_breaks) == 1
    assert bos_breaks[0].index == 12
    assert bos_breaks[0].direction == "BULLISH"
