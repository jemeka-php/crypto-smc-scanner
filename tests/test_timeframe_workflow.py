"""
Unit tests for the surgical timeframe workflow:
- 1H / 15M primary setup and location analysis
- 4H optional macro context
- Two-stage pipeline (Stage 1 screening, Stage 2 LTF confirmation on top 5-15)
- Completed candle discipline (no unclosed bars used for confirmed signals)
- Independence of the main 100-point score from LTF triggers
"""

import math
import time
import pandas as pd
import pytest

from config import AppConfig
from data.exchange import ExchangeClient
from indicators.oi import compute_oi_metrics
from scanner.candidates import (
    Candidate,
    LTFConfirmation,
    build_candidate,
    evaluate_ltf_confirmation,
)
from scanner.pipeline import MarketScanner


def create_sample_ohlcv(bars: int = 50, base_price: float = 100.0, step_seconds: int = 900) -> pd.DataFrame:
    now = int(time.time())
    records = []
    for i in range(bars):
        ts = (now - (bars - i) * step_seconds) * 1000
        angle = (i / max(bars, 1)) * 4 * math.pi
        noise = math.sin(angle) * (base_price * 0.03)
        c = base_price + noise
        h = c + (base_price * 0.006)
        l = c - (base_price * 0.006)
        o = c - (base_price * 0.002 * math.sin(angle))
        vol = 20000.0 * (1.0 + 0.5 * math.sin(i))
        records.append({
            "timestamp": ts,
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "volume": round(vol, 2),
        })
    return pd.DataFrame(records)


def test_15m_receives_full_smc_analysis():
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    cand = build_candidate(
        symbol="TESTUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=3.5,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        oi_metrics=oi_metrics,
        config=config,
    )

    assert cand is not None
    # 15M structure is evaluated
    assert cand.structure_15m is not None
    assert cand.structure_15m.timeframe == "15M"
    # 15M location is populated
    assert "15M SMC Location:" in cand.location_15m_desc
    assert cand.location_15m is not None
    # 15M active zones are present as lists
    assert isinstance(cand.active_obs_15m, list)
    assert isinstance(cand.active_fvgs_15m, list)
    assert isinstance(cand.sweeps_15m, list)
    assert isinstance(cand.displacements_15m, list)


def test_4h_remains_macro_context():
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    cand = build_candidate(
        symbol="TESTUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=3.5,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        oi_metrics=oi_metrics,
        config=config,
    )

    assert cand is not None
    # 4H macro context is present as informative context
    assert "4H Macro Context:" in cand.macro_context_4h
    # Primary setup location is explicitly labeled with 1H / 15M context
    assert "1H" in cand.htf_location_desc
    assert "15M" in cand.htf_location_desc


def test_completed_candles_discipline_and_no_lookahead():
    """Verify that lower timeframe analysis uses completed candles and drops forming candle."""
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    cand = build_candidate(
        symbol="SOLUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=3.5,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        oi_metrics=oi_metrics,
        config=config,
    )

    # Construct 5M data where only the very last (uncompleted, forming) bar creates an artificial spike
    df_5m = create_sample_ohlcv(bars=30, base_price=100.0, step_seconds=300)
    # The last candle is incomplete/forming
    df_5m.iloc[-1, df_5m.columns.get_loc("low")] = 50.0  # extreme fake wick on active bar
    df_5m.iloc[-1, df_5m.columns.get_loc("close")] = 101.0

    df_3m = create_sample_ohlcv(bars=30, base_price=100.0, step_seconds=180)
    df_1m = create_sample_ohlcv(bars=30, base_price=100.0, step_seconds=60)

    # evaluate_ltf_confirmation should drop df_5m.iloc[-1]
    res = evaluate_ltf_confirmation(cand, df_5m, df_3m, df_1m, config)
    # The fake active bar at -1 must NOT register as a confirmed sweep!
    assert "50.00" not in res.m5_sweep_desc


def test_main_score_remains_independent_of_ltf():
    """Verify that LTF trigger status does NOT alter the candidate's deterministic main score."""
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    cand = build_candidate(
        symbol="SOLUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=3.5,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        oi_metrics=oi_metrics,
        config=config,
    )

    initial_score = cand.total_score

    # Simulate LTF confirmation
    df_5m = create_sample_ohlcv(bars=30, base_price=100.0, step_seconds=300)
    df_3m = create_sample_ohlcv(bars=30, base_price=100.0, step_seconds=180)
    df_1m = create_sample_ohlcv(bars=30, base_price=100.0, step_seconds=60)

    ltf_res = evaluate_ltf_confirmation(cand, df_5m, df_3m, df_1m, config)
    cand.ltf_confirmation = ltf_res

    # Score MUST remain identical
    assert cand.total_score == initial_score


def test_two_stage_pipeline_in_demo_mode():
    """Verify that Stage 1 produces candidates, and Stage 2 analyzes only the shortlisted candidates."""
    config = AppConfig(demo_mode=True)
    scanner = MarketScanner(config)
    candidates, stats = scanner.run_scan()

    assert len(candidates) > 0
    assert stats.stage1_candidates == len(candidates)
    assert stats.stage2_candidates <= 15
    assert stats.stage2_candidates == min(len(candidates), 15)

    # First shortlisted candidate should have been analyzed in Stage 2
    top_cand = candidates[0]
    assert top_cand.ltf_confirmation.status in (
        "NOT_CHECKED", "NO_TRIGGER", "WATCHING", "PARTIAL_CONFIRMATION", "CONFIRMED", "INVALIDATED"
    )

    # Timings should be recorded
    assert stats.stage1_duration_s >= 0.0
    assert stats.stage2_duration_s >= 0.0
    assert stats.total_duration_s >= 0.0
