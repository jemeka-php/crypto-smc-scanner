import pandas as pd
from indicators.volume import calculate_rvol


def test_rvol_calculation_and_classification():
    # 20 previous completed candles with constant volume of 1000
    prev_vols = [1000.0] * 20
    # Current completed candle volume of 2200 -> RVOL = 2.2x -> HIGH
    vols = pd.Series(prev_vols + [2200.0])

    rvol, median_vol, classification = calculate_rvol(vols, lookback=20)

    assert rvol == 2.2
    assert median_vol == 1000.0
    assert classification == "HIGH"


def test_rvol_low_volume():
    prev_vols = [1000.0] * 20
    vols = pd.Series(prev_vols + [400.0])

    rvol, median_vol, classification = calculate_rvol(vols, lookback=20)
    assert rvol == 0.4
    assert classification == "LOW"
