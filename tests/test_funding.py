"""
Targeted regression tests for the display-only Funding Rate integration:
1. Valid positive funding
2. Valid negative funding
3. Neutral funding
4. Missing funding data (null/None preservation, never 0.0)
5. Funding API failure does not remove candidate
6. Funding does not change candidate score (0 points contributed)
7. Funding does not change candidate ranking
8. Configurable funding thresholds
"""

import math
import time
import pandas as pd
import pytest

from config import AppConfig, FundingConfig
from indicators.oi import compute_oi_metrics
from scanner.candidates import (
    Candidate,
    FundingData,
    build_candidate,
    classify_funding_rate,
)
from scanner.scoring import calculate_candidate_score


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


def test_valid_positive_funding():
    """Verify rate > +0.0001 is classified as POSITIVE and percentage is calculated."""
    rate = 0.00012
    label = classify_funding_rate(rate)
    assert label == "POSITIVE"

    funding = FundingData(
        rate=rate,
        rate_pct=round(rate * 100, 6),
        label=label,
        timestamp=1791388800000,
        next_funding_time=1791392400000,
    )
    assert funding.rate == 0.00012
    assert funding.rate_pct == 0.012
    assert funding.label == "POSITIVE"
    assert funding.timestamp == 1791388800000
    assert funding.next_funding_time == 1791392400000
    d = funding.to_dict()
    assert d["label"] == "POSITIVE"
    assert d["rate_pct"] == 0.012


def test_valid_negative_funding():
    """Verify rate < -0.0001 is classified as NEGATIVE and percentage is calculated."""
    rate = -0.00015
    label = classify_funding_rate(rate)
    assert label == "NEGATIVE"

    funding = FundingData(
        rate=rate,
        rate_pct=round(rate * 100, 6),
        label=label,
        timestamp=1791388800000,
        next_funding_time=1791392400000,
    )
    assert funding.rate == -0.00015
    assert funding.rate_pct == -0.015
    assert funding.label == "NEGATIVE"


def test_neutral_funding():
    """Verify rate within [-0.0001, +0.0001] is classified as NEUTRAL."""
    assert classify_funding_rate(0.00005) == "NEUTRAL"
    assert classify_funding_rate(-0.00005) == "NEUTRAL"
    assert classify_funding_rate(0.0) == "NEUTRAL"
    assert classify_funding_rate(0.0001) == "NEUTRAL"
    assert classify_funding_rate(-0.0001) == "NEUTRAL"


def test_missing_funding_data_null_fields():
    """Missing funding rate must produce None/null fields, never 0.0."""
    label = classify_funding_rate(None)
    assert label is None

    empty_funding = FundingData()
    assert empty_funding.rate is None
    assert empty_funding.rate_pct is None
    assert empty_funding.label is None
    assert empty_funding.timestamp is None
    assert empty_funding.next_funding_time is None

    # Never substitute 0.0
    d = empty_funding.to_dict()
    assert d["rate"] is None
    assert d["rate_pct"] is None
    assert d["label"] is None
    assert d["rate"] != 0.0


def test_funding_api_failure_does_not_remove_candidate():
    """When funding data is missing/failed, candidate is built completely normally."""
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    # Candidate with empty/failed funding
    cand_failed_funding = build_candidate(
        symbol="FAILUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=2.5,
        oi_metrics=oi_metrics,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        config=config,
        funding=FundingData(),  # failed funding
    )

    assert cand_failed_funding is not None
    assert cand_failed_funding.symbol == "FAILUSDT"
    assert cand_failed_funding.funding.rate is None
    assert cand_failed_funding.funding.label is None
    assert cand_failed_funding.total_score > 0.0


def test_funding_does_not_change_candidate_score():
    """Funding rate must contribute 0 points: candidates with different funding have identical scores."""
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    cand_pos = build_candidate(
        symbol="TESTUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=2.5,
        oi_metrics=oi_metrics,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        config=config,
        funding=FundingData(rate=0.0005, rate_pct=0.05, label="POSITIVE", timestamp=100, next_funding_time=200),
    )

    cand_neg = build_candidate(
        symbol="TESTUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=2.5,
        oi_metrics=oi_metrics,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        config=config,
        funding=FundingData(rate=-0.0005, rate_pct=-0.05, label="NEGATIVE", timestamp=100, next_funding_time=200),
    )

    cand_none = build_candidate(
        symbol="TESTUSDT",
        last_price=100.0,
        quote_volume_24h=80_000_000.0,
        price_change_24h_pct=2.5,
        oi_metrics=oi_metrics,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        config=config,
        funding=FundingData(),
    )

    # All three candidates have exactly the same score
    assert cand_pos.total_score == cand_neg.total_score
    assert cand_pos.total_score == cand_none.total_score
    assert cand_pos.score_breakdown == cand_neg.score_breakdown
    assert cand_pos.score_breakdown == cand_none.score_breakdown


def test_funding_does_not_change_candidate_ranking():
    """Ranking order by total_score is completely independent of funding rate."""
    config = AppConfig(demo_mode=True)
    df_1h = create_sample_ohlcv(bars=60, base_price=100.0, step_seconds=3600)
    df_4h = create_sample_ohlcv(bars=40, base_price=100.0, step_seconds=14400)
    df_15m = create_sample_ohlcv(bars=50, base_price=100.0, step_seconds=900)
    oi_metrics = compute_oi_metrics(105.0, 100.0, 95.0, price_1h_change_pct=1.5)

    cand_high_vol = build_candidate(
        symbol="HIGHUSDT",
        last_price=100.0,
        quote_volume_24h=120_000_000.0,
        price_change_24h_pct=5.0,
        oi_metrics=oi_metrics,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        config=config,
        funding=FundingData(rate=-0.0008, rate_pct=-0.08, label="NEGATIVE"),  # negative funding on higher score
    )

    cand_low_vol = build_candidate(
        symbol="LOWUSDT",
        last_price=100.0,
        quote_volume_24h=60_000_000.0,
        price_change_24h_pct=1.0,
        oi_metrics=oi_metrics,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        config=config,
        funding=FundingData(rate=0.0008, rate_pct=0.08, label="POSITIVE"),  # positive funding on lower score
    )

    candidates = [cand_low_vol, cand_high_vol]
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    assert candidates[0].symbol == "HIGHUSDT"
    assert candidates[1].symbol == "LOWUSDT"
    assert candidates[0].total_score >= candidates[1].total_score


def test_configurable_thresholds():
    """Thresholds can be customized via FundingConfig."""
    cfg = FundingConfig(positive_threshold=0.0005, negative_threshold=-0.0005)
    # 0.0002 is neutral under custom 0.0005 threshold
    assert classify_funding_rate(0.0002, cfg.positive_threshold, cfg.negative_threshold) == "NEUTRAL"
    # 0.0006 is positive
    assert classify_funding_rate(0.0006, cfg.positive_threshold, cfg.negative_threshold) == "POSITIVE"
    # -0.0006 is negative
    assert classify_funding_rate(-0.0006, cfg.positive_threshold, cfg.negative_threshold) == "NEGATIVE"
