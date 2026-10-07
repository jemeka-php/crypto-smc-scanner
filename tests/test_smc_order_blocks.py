import pandas as pd
from smc.displacement import detect_displacements
from smc.order_blocks import find_order_blocks


def test_order_block_lifecycle():
    # Candle 0: Bearish candle (OB candidate) -> Open 105, Close 98, High 106, Low 97
    # Candle 1 & 2: Bullish displacement -> Open 98, Close 120 (move of 22 vs ATR 5)
    # Candle 3: Retest into zone (Mitigated) -> Low 102, Close 104
    # Candle 4: Still above OB low -> MITIGATED state
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "open": [105.0, 98.0, 110.0, 115.0, 108.0],
        "high": [106.0, 112.0, 120.0, 118.0, 112.0],
        "low": [97.0, 97.5, 108.0, 102.0, 106.0],  # Bar 3 dips to 102 (between OB low 97 and high 106)
        "close": [98.0, 110.0, 120.0, 108.0, 110.0],
    })
    atr_series = pd.Series([5.0] * 5)

    displacements = detect_displacements(df, atr_series, min_atr_mult=1.5)
    obs = find_order_blocks(df, atr_series, displacements)

    assert len(obs) >= 1
    bullish_ob = obs[0]
    assert bullish_ob.ob_type == "BULLISH"
    assert bullish_ob.high == 106.0
    assert bullish_ob.low == 97.0
    # Price traded into the zone on candle 3, so state must be MITIGATED
    assert bullish_ob.state == "MITIGATED"
