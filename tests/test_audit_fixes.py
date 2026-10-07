"""
Targeted regression tests for the surgical audit fixes:
- Fix 1: Timeframe-aware location descriptions
- Fix 2: Directional guards preventing cleared demand/supply from qualifying as near
- Fix 3: LTF chronological event ordering (stale 1M MSS cannot confirm newer 5M sweep)
- Fix 4: LTF directional coherence (opposite displacement rejected)
- Fix 5: Displacement must be completed before or at break bar to qualify MSS
- Fix 6: Scoring weights validation (detecting totals != 100)
- Fix 7: OI observational wording
- Fix 8: Demo OI neutral fallback for unlisted symbols
- Fix 10: FVG forming-candle protection in Stage 1
"""

import pandas as pd
import pytest

from config import AppConfig, ScoringWeights
from data.exchange import ExchangeClient
from indicators.oi import classify_price_oi, compute_oi_metrics
from smc.fvg import FVG, find_fvgs
from smc.order_blocks import OrderBlock
from smc.liquidity import LiquidityLevel, LiquiditySweep, determine_htf_location
from smc.displacement import Displacement
from smc.structure import Swing, analyze_market_structure
from scanner.candidates import Candidate, evaluate_ltf_confirmation


# =====================================================================
# Fix 1 & 2: Location timeframe descriptions & directional guards
# =====================================================================
def test_location_timeframe_aware_description():
    """Verify determine_htf_location uses the provided timeframe in descriptions."""
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
        timeframe="1H",
    )
    # Test 1H call
    loc_1h, desc_1h = determine_htf_location(
        current_price=102.0,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
        timeframe="1H",
    )
    assert loc_1h == "NEAR DEMAND"
    assert "Within 1H Bullish OB zone" in desc_1h
    assert "4H" not in desc_1h

    # Test 15M call
    loc_15m, desc_15m = determine_htf_location(
        current_price=102.0,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
        timeframe="15M",
    )
    assert loc_15m == "NEAR DEMAND"
    assert "Within 15M Bullish OB zone" in desc_15m


def test_location_directional_guards_demand():
    """
    Verify directional guard for demand:
    - Price inside demand -> qualifies
    - Price slightly above demand (cleared) -> does NOT qualify
    """
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
        timeframe="1H",
    )
    # Inside demand [100, 105] -> qualifies
    loc, _ = determine_htf_location(
        current_price=103.0,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
        timeframe="1H",
    )
    assert loc == "NEAR DEMAND"

    # Price slightly above demand at 105.5 (0.47% above high 105.0, within old 0.8% band)
    # Directional guard MUST reject this because current_price > ob.high
    loc_above, _ = determine_htf_location(
        current_price=105.5,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
        timeframe="1H",
    )
    assert loc_above == "NO SIGNIFICANT LOCATION"


def test_location_directional_guards_supply():
    """
    Verify directional guard for supply:
    - Price inside supply -> qualifies
    - Price slightly below supply (cleared) -> does NOT qualify
    """
    ob = OrderBlock(
        index=0,
        timestamp=100,
        ob_type="BEARISH",
        high=105.0,
        low=100.0,
        open=101.0,
        close=104.0,
        state="ACTIVE",
        displacement_ratio=2.0,
        timeframe="1H",
    )
    # Inside supply [100, 105] -> qualifies
    loc, _ = determine_htf_location(
        current_price=103.0,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
        timeframe="1H",
    )
    assert loc == "NEAR SUPPLY"

    # Price slightly below supply at 99.5 (0.5% below low 100.0, within old 0.8% band)
    # Directional guard MUST reject this because current_price < ob.low
    loc_below, _ = determine_htf_location(
        current_price=99.5,
        htf_obs=[ob],
        htf_fvgs=[],
        htf_levels=[],
        htf_swings=[],
        timeframe="1H",
    )
    assert loc_below == "NO SIGNIFICANT LOCATION"


# =====================================================================
# Fix 5: MSS displacement completion
# =====================================================================
def test_incomplete_future_displacement_cannot_qualify_mss():
    """
    Verify that an MSS occurring at bar 11 cannot qualify using a 3-bar displacement
    whose final bar occurs after bar 11 (ending at bar 12).
    """
    n = 30
    opens = [100.0] * n
    highs = [101.0] * n
    lows = [98.0] * n
    closes = [100.0] * n
    timestamps = [i * 60000 for i in range(n)]

    # 1. Swing high at bar 3 with lookback=2 (price 110.0, confirmed at bar 5)
    highs[1], highs[2] = 101.0, 101.0
    highs[3], closes[3] = 110.0, 105.0
    highs[4], highs[5] = 101.0, 101.0

    # 2. Swing low at bar 6 with lookback=2 (price 90.0, confirmed at bar 8)
    lows[4], lows[5] = 95.0, 95.0
    lows[6], closes[6] = 90.0, 92.0
    lows[7], lows[8] = 95.0, 95.0

    # 3. Break below 90.0 at bar 9 to establish BEARISH bias
    lows[9], closes[9] = 88.0, 88.0

    # 4. 3-bar displacement spanning bars 10 -> 12:
    # ATR = 10.0, displacement mult = 1.5 -> requires net movement >= 15.0
    # bar 10: open=92, close=97 (body = 5.0 < 15.0, NOT displacement alone)
    # bar 11: open=97, close=112 (body = 15.0? No, let's keep it under 15: open=97, close=108 -> body 11 < 15, 2-bar body = 16?
    # To ensure displacement ONLY completes at bar 12:
    # bar 10: open=92, close=96 (body = 4.0)
    # bar 11: open=96, close=106 (body = 10.0, 2-bar body = 14.0 < 15.0, NOT displacement yet!)
    # Note: bar 11 closes at 111.0 > 110.0 (breaks swing high at bar 11!)
    # So: open=100, close=111 -> body = 11.0 < 15.0
    # bar 10: open=98, close=100 (body = 2.0)
    # bar 11: open=100, close=111 (body = 11.0, 2-bar body = 13.0 < 15.0, NOT displacement yet!)
    # bar 12: open=111, close=116 (3-bar body = 116 - 98 = 18.0 >= 15.0 -> displacement ends at bar 12!)
    opens[10], closes[10], highs[10] = 98.0, 100.0, 101.0
    opens[11], closes[11], highs[11] = 100.0, 111.0, 112.0
    opens[12], closes[12], highs[12] = 111.0, 116.0, 117.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })
    atr_series = pd.Series([10.0] * n)

    struct = analyze_market_structure(
        df,
        atr_series,
        timeframe="15M",
        swing_lookback=2,
        mss_requires_displacement=True,
        mss_lookback_bars=3,
        displacement_atr_mult=1.5,
    )

    # Break at bar 11 (close 111 > swing high 110)
    breaks_at_11 = [b for b in struct.recent_breaks if b.index == 11]
    assert len(breaks_at_11) == 1
    # At bar 11, the displacement ending at bar 12 is INCOMPLETE
    # The break must NOT receive credit from that future displacement!
    assert breaks_at_11[0].has_displacement is False
    assert breaks_at_11[0].displacement is None
    # Because mss_requires_displacement=True and displacement is incomplete, it cannot be an MSS
    assert breaks_at_11[0].break_type != "MSS"


def test_completed_displacement_qualifies_mss():
    """
    Verify that an MSS occurring at or after the end of displacement (bar 12) qualifies.
    """
    n = 30
    opens = [100.0] * n
    highs = [101.0] * n
    lows = [98.0] * n
    closes = [100.0] * n
    timestamps = [i * 60000 for i in range(n)]

    # 1. Swing high at bar 3 with lookback=2 (price 110.0, confirmed at bar 5)
    highs[1], highs[2] = 101.0, 101.0
    highs[3], closes[3] = 110.0, 105.0
    highs[4], highs[5] = 101.0, 101.0

    # 2. Swing low at bar 6 with lookback=2 (price 90.0, confirmed at bar 8)
    lows[4], lows[5] = 95.0, 95.0
    lows[6], closes[6] = 90.0, 92.0
    lows[7], lows[8] = 95.0, 95.0

    # 3. Break below 90.0 at bar 9 to establish BEARISH bias
    lows[9], closes[9] = 88.0, 88.0

    # 4. 3-bar displacement spanning bars 10 -> 12:
    # bar 10: open=98, close=100 (body = 2.0)
    # bar 11: open=100, close=108 (body = 8.0, under swing high 110!)
    # bar 12: open=108, close=116 (3-bar body = 116 - 98 = 18.0 >= 15.0 -> displacement ends at bar 12!)
    # Note: bar 12 closes at 116 > 110 (breaks swing high at bar 12, exactly when displacement completes!)
    opens[10], closes[10], highs[10] = 98.0, 100.0, 101.0
    opens[11], closes[11], highs[11] = 100.0, 108.0, 109.0
    opens[12], closes[12], highs[12] = 108.0, 116.0, 117.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })
    atr_series = pd.Series([10.0] * n)

    struct = analyze_market_structure(
        df,
        atr_series,
        timeframe="15M",
        swing_lookback=2,
        mss_requires_displacement=True,
        mss_lookback_bars=3,
        displacement_atr_mult=1.5,
    )

    # Break at bar 12 (close 116 > swing high 110) occurs when displacement is complete
    breaks_at_12 = [b for b in struct.recent_breaks if b.index == 12]
    assert len(breaks_at_12) == 1
    assert breaks_at_12[0].has_displacement is True
    assert breaks_at_12[0].displacement is not None
    assert breaks_at_12[0].displacement.end_index == 12
    assert breaks_at_12[0].break_type == "MSS"


# =====================================================================
# Fix 3 & 4: LTF Chronological event ordering & directional coherence
# =====================================================================
def _create_micro_df(bars: int, start_ts: int, step_s: int, base: float = 100.0) -> pd.DataFrame:
    records = []
    for i in range(bars):
        ts = start_ts + i * step_s * 1000
        records.append({
            "timestamp": ts,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": 10000.0,
        })
    return pd.DataFrame(records)


def test_stale_1m_mss_before_5m_sweep_not_confirmed():
    """
    Verify that an old 1M MSS preceding a newer 5M sweep does NOT produce CONFIRMED.
    """
    config = AppConfig(demo_mode=True)
    base_ts = 1_700_000_000_000

    # 5M data: 35 bars (step 300s). Extra bar at end to satisfy drop-forming-candle.
    df_5m = _create_micro_df(35, base_ts, 300)
    # Create swing low at bar 20 (price 96)
    df_5m.loc[20, "low"] = 96.0
    # Create sell-side sweep at bar 29 (timestamp: base_ts + 29*300*1000)
    df_5m.loc[29, "low"] = 94.0
    df_5m.loc[29, "close"] = 97.0

    df_3m = _create_micro_df(40, base_ts, 180)

    # 1M data: Stale MSS occurred early at bar 8 (timestamp: base_ts + 8*60*1000)
    # Sweep at 29*300 = 8700s is much newer than bar 8 at 480s.
    df_1m = _create_micro_df(50, base_ts, 60)
    df_1m.loc[3, "high"] = 105.0
    df_1m.loc[3, "close"] = 104.0
    df_1m.loc[6, "low"] = 92.0
    df_1m.loc[6, "close"] = 92.0
    df_1m.loc[8, "high"] = 107.0
    df_1m.loc[8, "close"] = 106.0

    cand = Candidate(
        symbol="TESTUSDT",
        last_price=97.0,
        quote_volume_24h=50_000_000.0,
        price_change_24h_pct=1.0,
        rvol=1.5,
        oi_metrics=compute_oi_metrics(100.0, 95.0, 90.0),
        structure_4h=None,
        structure_1h=None,
        structure_15m=None,
        htf_location="NEAR DEMAND",
        htf_location_desc="1H Demand",
        active_obs_1h=[],
        active_fvgs_1h=[],
        sweeps_1h=[],
        nearby_liquidity=[],
        displacements_1h=[],
        stage="STAGE_1_QUALIFIED",
        direction="BULLISH",
        score_breakdown=None,
        total_score=80.0,
        why_interesting="",
        waiting_for="",
        trade_plan=None,
        filter_result=None,
    )

    conf = evaluate_ltf_confirmation(cand, df_5m, df_3m, df_1m, config)

    # Stale MSS prior to 5M sweep MUST NOT result in CONFIRMED
    assert conf.status != "CONFIRMED"
    assert conf.status == "PARTIAL_CONFIRMATION"
    assert "prior MSS preceded sweep" in conf.summary


def test_correctly_ordered_sweep_then_mss_confirms():
    """
    Verify that 5M sweep followed by 1M MSS in valid temporal sequence produces CONFIRMED.
    """
    config = AppConfig(demo_mode=True)
    base_ts = 1_700_000_000_000

    # 5M data: 30 bars (step 300s)
    df_5m = _create_micro_df(30, base_ts, 300)
    df_5m.loc[15, "low"] = 96.0
    df_5m.loc[24, "low"] = 94.0
    df_5m.loc[24, "close"] = 97.0

    df_3m = _create_micro_df(35, base_ts, 180)

    # 1M data covering the period of the sweep (from 7000s onwards)
    sweep_start_ts = base_ts + 24 * 300 * 1000
    df_1m = _create_micro_df(40, sweep_start_ts - 600 * 1000, 60)
    df_1m.loc[12, "high"] = 105.0
    df_1m.loc[12, "close"] = 104.0
    df_1m.loc[15, "low"] = 92.0
    df_1m.loc[15, "close"] = 92.0
    df_1m.loc[20, "high"] = 107.0
    df_1m.loc[20, "close"] = 106.0

    cand = Candidate(
        symbol="TESTUSDT",
        last_price=97.0,
        quote_volume_24h=50_000_000.0,
        price_change_24h_pct=1.0,
        rvol=1.5,
        oi_metrics=compute_oi_metrics(100.0, 95.0, 90.0),
        structure_4h=None,
        structure_1h=None,
        structure_15m=None,
        htf_location="NEAR DEMAND",
        htf_location_desc="1H Demand",
        active_obs_1h=[],
        active_fvgs_1h=[],
        sweeps_1h=[],
        nearby_liquidity=[],
        displacements_1h=[],
        stage="STAGE_1_QUALIFIED",
        direction="BULLISH",
        score_breakdown=None,
        total_score=80.0,
        why_interesting="",
        waiting_for="",
        trade_plan=None,
        filter_result=None,
    )

    conf = evaluate_ltf_confirmation(cand, df_5m, df_3m, df_1m, config)
    assert conf.status == "CONFIRMED"


def test_contradictory_displacement_direction_rejected():
    """
    Verify that an opposite-direction (bearish) displacement does NOT satisfy
    the displacement requirement for a bullish setup.
    """
    config = AppConfig(demo_mode=True)
    base_ts = 1_700_000_000_000

    df_5m = _create_micro_df(30, base_ts, 300)
    df_3m = _create_micro_df(30, base_ts, 180)
    # Consecutive red candles expanding downwards (bearish displacement)
    df_3m.loc[25, "open"], df_3m.loc[25, "close"] = 105.0, 100.0
    df_3m.loc[26, "open"], df_3m.loc[26, "close"] = 100.0, 92.0
    df_1m = _create_micro_df(30, base_ts, 60)

    cand = Candidate(
        symbol="TESTUSDT",
        last_price=92.0,
        quote_volume_24h=50_000_000.0,
        price_change_24h_pct=1.0,
        rvol=1.5,
        oi_metrics=compute_oi_metrics(100.0, 95.0, 90.0),
        structure_4h=None,
        structure_1h=None,
        structure_15m=None,
        htf_location="NEAR DEMAND",
        htf_location_desc="1H Demand",
        active_obs_1h=[],
        active_fvgs_1h=[],
        sweeps_1h=[],
        nearby_liquidity=[],
        displacements_1h=[],
        stage="STAGE_1_QUALIFIED",
        direction="BULLISH",
        score_breakdown=None,
        total_score=80.0,
        why_interesting="",
        waiting_for="",
        trade_plan=None,
        filter_result=None,
    )

    conf = evaluate_ltf_confirmation(cand, df_5m, df_3m, df_1m, config)
    # Bearish displacement must NOT be accepted as bullish displacement
    assert conf.m3_displacement is False


# =====================================================================
# Fix 6: Scoring Weights Validation
# =====================================================================
def test_scoring_weights_validation():
    """Verify that scoring weights totaling != 100 are detected by is_valid."""
    w_valid = ScoringWeights()
    assert w_valid.total() == 100.0
    assert w_valid.is_valid() is True

    # Custom weights totaling 150
    w_invalid = ScoringWeights(liquidity=25.0, rvol=25.0, oi=25.0, volatility=25.0, htf_location=25.0, liquidity_sweep=25.0)
    assert w_invalid.total() != 100.0
    assert w_invalid.is_valid() is False


# =====================================================================
# Fix 7 & 8: Open Interest Wording & Fallback
# =====================================================================
def test_oi_observational_wording():
    """Verify observational wording is returned for position contraction quadrants."""
    _, interp_weak = classify_price_oi(price_change_pct=-1.5, oi_change_pct=-3.0)
    assert "position contraction during weakness" in interp_weak
    assert "direction indeterminate from price and OI alone" in interp_weak
    assert "long liquidation or short covering" not in interp_weak

    _, interp_str = classify_price_oi(price_change_pct=1.5, oi_change_pct=-3.0)
    assert "position contraction during strength" in interp_str
    assert "direction indeterminate from price and OI alone" in interp_str
    assert "short covering or long reduction" not in interp_str


def test_demo_unlisted_symbol_neutral_oi():
    """Verify that an unlisted symbol in demo mode returns (0.0, None, None) and 0% change."""
    client = ExchangeClient(demo_mode=True)
    curr_oi, oi_1h, oi_4h = client.fetch_oi_history("UNLISTED_DEMO_COIN_USDT")
    assert curr_oi == 0.0
    assert oi_1h is None
    assert oi_4h is None

    metrics = compute_oi_metrics(curr_oi, oi_1h, oi_4h)
    assert metrics.change_1h_pct == 0.0
    assert metrics.change_4h_pct == 0.0


# =====================================================================
# Fix 10: FVG Forming Candle Exclusion
# =====================================================================
def test_fvg_forming_candle_excluded_when_completed_only():
    """
    Verify that an unclosed forming candle at len(df)-1 cannot serve as candle 3
    when completed_candles_only=True.
    """
    # 4 candles:
    # 0: high = 100
    # 1: green candle
    # 2: low = 101 (gap low[2] - high[0] = 1.0 < min_required 2.5) -> No FVG on completed bars 0,1,2
    # 3: low = 115 (unclosed forming bar! gap low[3] - high[1] = 15.0 >= 2.5) -> FVG on bars 1,2,3
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3],
        "open": [95.0, 101.0, 102.0, 116.0],
        "high": [100.0, 108.0, 108.0, 120.0],
        "low": [94.0, 100.5, 101.0, 115.0],
        "close": [98.0, 106.0, 105.0, 118.0],
    })
    atr_series = pd.Series([10.0] * 4)

    # When completed_candles_only=True, bar 3 cannot be candle 3
    fvgs_completed = find_fvgs(df, atr_series, min_gap_atr_mult=0.25, completed_candles_only=True)
    assert len(fvgs_completed) == 0

    # When completed_candles_only=False, bar 3 is included as candle 3
    fvgs_all = find_fvgs(df, atr_series, min_gap_atr_mult=0.25, completed_candles_only=False)
    assert len(fvgs_all) == 1
    assert fvgs_all[0].index == 2
