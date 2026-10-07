import numpy as np
import pandas as pd
from indicators.atr import calculate_atr


def test_calculate_atr_wilder_rma():
    # Construct a deterministic dataframe
    n = 30
    highs = [100.0 + i + (2.0 if i % 2 == 0 else 1.0) for i in range(n)]
    lows = [100.0 + i - (2.0 if i % 2 == 0 else 1.0) for i in range(n)]
    closes = [100.0 + i for i in range(n)]

    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    atr = calculate_atr(df, period=14)

    assert len(atr) == n
    assert pd.isna(atr.iloc[0])
    assert pd.isna(atr.iloc[12])
    # Bar index 13 (the 14th bar) must have initial SMA value
    assert not pd.isna(atr.iloc[13])
    # Subsequent values must exist and be strictly positive
    assert atr.iloc[-1] > 0.0
