import pandas as pd
from smc.swings import find_swings


def test_find_swings_chronological_confirmation():
    # Peak at index 5 with lookback=3
    # Left: indices 2, 3, 4 (highs: 10, 11, 12)
    # Peak: index 5 (high: 20)
    # Right: indices 6, 7, 8 (highs: 15, 14, 13)
    # Confirmation must occur at index 5 + 3 = 8
    n = 15
    highs = [10.0] * n
    lows = [5.0] * n

    highs[2] = 11.0
    highs[3] = 12.0
    highs[4] = 13.0
    highs[5] = 20.0  # Peak
    highs[6] = 15.0
    highs[7] = 14.0
    highs[8] = 13.0

    df = pd.DataFrame({
        "timestamp": list(range(1000, 1000 + n)),
        "high": highs,
        "low": lows,
        "close": [7.0] * n,
        "open": [7.0] * n,
    })

    swings = find_swings(df, lookback=3)
    high_swings = [s for s in swings if s.swing_type == "HIGH"]

    assert len(high_swings) == 1
    assert high_swings[0].index == 5
    assert high_swings[0].price == 20.0
    assert high_swings[0].confirmed_at_index == 8  # 5 + lookback
