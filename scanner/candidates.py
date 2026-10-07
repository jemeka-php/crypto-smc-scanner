"""
Candidate representation, stage progression (WATCH, SETUP_FORMING, CONFIRMED, NO_TRADE),
and reference trade plan generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from config import AppConfig
from indicators.oi import OIMetrics
from indicators.atr import calculate_atr
from indicators.volume import calculate_rvol
from smc.swings import Swing, find_swings
from smc.structure import MarketStructure, analyze_market_structure
from smc.displacement import Displacement, detect_displacements
from smc.order_blocks import OrderBlock, find_order_blocks
from smc.fvg import FVG, find_fvgs
from smc.liquidity import (
    LiquidityLevel,
    LiquiditySweep,
    detect_liquidity_levels,
    detect_liquidity_sweeps,
    determine_htf_location,
)
from scanner.filters import evaluate_candidate_filters, FilterResult
from scanner.scoring import calculate_candidate_score, ScoreBreakdown


@dataclass
class ReferenceTradePlan:
    direction: str              # "LONG" or "SHORT"
    entry: float
    stop: float
    risk_amount: float
    target_1r: float
    target_2r: float
    target_3r: float
    risk_reward: str


@dataclass
class LTFConfirmation:
    status: str = "NOT_CHECKED"  # "NOT_CHECKED", "NO_TRIGGER", "WATCHING", "PARTIAL_CONFIRMATION", "CONFIRMED", "INVALIDATED"
    direction: str = "NEUTRAL"
    m5_sweep: bool = False
    m5_sweep_desc: str = ""
    m3_displacement: bool = False
    m3_displacement_desc: str = ""
    m1_mss: bool = False
    m1_mss_desc: str = ""
    summary: str = "Not checked"
    details: List[str] = field(default_factory=list)


@dataclass
class FundingData:
    rate: Optional[float] = None
    rate_pct: Optional[float] = None
    label: Optional[str] = None
    timestamp: Optional[int] = None
    next_funding_time: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[object]]:
        return {
            "rate": self.rate,
            "rate_pct": self.rate_pct,
            "label": self.label,
            "timestamp": self.timestamp,
            "next_funding_time": self.next_funding_time,
        }


def classify_funding_rate(
    rate: Optional[float],
    pos_threshold: float = 0.0001,
    neg_threshold: float = -0.0001,
) -> Optional[str]:
    """
    Classify raw funding rate into POSITIVE, NEUTRAL, or NEGATIVE.
    Returns None if rate is None.
    Thresholds:
        POSITIVE: rate > pos_threshold (default +0.0001 / +0.01%)
        NEUTRAL:  neg_threshold <= rate <= pos_threshold
        NEGATIVE: rate < neg_threshold (default -0.0001 / -0.01%)
    """
    if rate is None:
        return None
    if rate > pos_threshold:
        return "POSITIVE"
    elif rate < neg_threshold:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


@dataclass
class Candidate:
    symbol: str
    last_price: float
    quote_volume_24h: float
    price_change_24h_pct: float
    rvol: float
    oi_metrics: OIMetrics
    
    # Structure
    structure_4h: MarketStructure
    structure_1h: MarketStructure
    structure_15m: Optional[MarketStructure]
    
    # SMC elements
    htf_location: str
    htf_location_desc: str
    active_obs_1h: List[OrderBlock]
    active_fvgs_1h: List[FVG]
    sweeps_1h: List[LiquiditySweep]
    nearby_liquidity: List[LiquidityLevel]
    displacements_1h: List[Displacement]
    
    # Evaluation
    stage: str                  # "WATCH", "SETUP_FORMING", "CONFIRMED", "NO_TRADE"
    direction: str              # "BULLISH", "BEARISH", "NEUTRAL"
    score_breakdown: ScoreBreakdown
    total_score: float
    why_interesting: str
    waiting_for: str
    
    # Optional Plan & Filters
    trade_plan: Optional[ReferenceTradePlan] = None
    filter_result: Optional[FilterResult] = None

    # 1H & 15M Setup Locations (Objective SMC Context)
    location_1h: str = "NO SIGNIFICANT LOCATION"
    location_1h_desc: str = ""
    location_15m: str = "NO SIGNIFICANT LOCATION"
    location_15m_desc: str = ""
    active_obs_15m: List[OrderBlock] = field(default_factory=list)
    active_fvgs_15m: List[FVG] = field(default_factory=list)
    sweeps_15m: List[LiquiditySweep] = field(default_factory=list)
    displacements_15m: List[Displacement] = field(default_factory=list)

    # 4H Macro Context
    macro_context_4h: str = ""

    # Stage 2 Lower Timeframe (LTF) Entry Confirmation
    ltf_confirmation: LTFConfirmation = field(default_factory=LTFConfirmation)

    # Display-only Funding Rate Context
    funding: FundingData = field(default_factory=FundingData)


def build_candidate(
    symbol: str,
    last_price: float,
    quote_volume_24h: float,
    price_change_24h_pct: float,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_15m: Optional[pd.DataFrame],
    oi_metrics: OIMetrics,
    config: AppConfig,
    funding: Optional[FundingData] = None,
) -> Optional[Candidate]:
    """
    Run full multi-timeframe SMC pipeline and score the market.
    """
    if df_1h.empty or len(df_1h) < 25:
        return None

    # Completed candles for RVOL (drop currently forming candle at the end)
    completed_1h_vol = df_1h["volume"].iloc[:-1]
    rvol, median_vol, vol_label = calculate_rvol(
        completed_1h_vol,
        lookback=config.smc.volume_lookback,
    )

    # ATRs
    atr_1h = calculate_atr(df_1h, period=config.smc.atr_period)
    atr_4h = calculate_atr(df_4h, period=config.smc.atr_period) if not df_4h.empty else pd.Series()
    atr_15m = calculate_atr(df_15m, period=config.smc.atr_period) if df_15m is not None and not df_15m.empty else None

    # Filter check
    filter_res = evaluate_candidate_filters(
        quote_volume_24h=quote_volume_24h,
        price_change_24h_pct=price_change_24h_pct,
        rvol=rvol,
        oi_1h_change_pct=oi_metrics.change_1h_pct,
        config=config.filters,
    )

    # 4H Market Structure & HTF elements
    # 4H Macro Context (Optional Context Only - not sole location anchor)
    structure_4h = (
        analyze_market_structure(
            df_4h,
            atr_4h,
            timeframe="4H",
            swing_lookback=config.smc.swing_lookback,
            mss_requires_displacement=config.smc.mss_requires_displacement,
            mss_lookback_bars=config.smc.mss_lookback_bars,
            displacement_atr_mult=config.smc.displacement_atr_mult,
        )
        if not df_4h.empty
        else MarketStructure("4H", "NEUTRAL", [], None, None, [], None, None)
    )
    disp_4h = detect_displacements(df_4h, atr_4h, min_atr_mult=config.smc.displacement_atr_mult) if not df_4h.empty else []
    macro_context_4h = f"4H Macro Context: {structure_4h.bias} bias" + (f" (Displacement {disp_4h[-1].displacement_ratio:.1f}x ATR)" if disp_4h else "")

    # 1H Primary Setup Timeframe (Full SMC Analysis)
    structure_1h = analyze_market_structure(
        df_1h,
        atr_1h,
        timeframe="1H",
        swing_lookback=config.smc.swing_lookback,
        mss_requires_displacement=config.smc.mss_requires_displacement,
        mss_lookback_bars=config.smc.mss_lookback_bars,
        displacement_atr_mult=config.smc.displacement_atr_mult,
    )
    disp_1h = detect_displacements(df_1h, atr_1h, min_atr_mult=config.smc.displacement_atr_mult)
    obs_1h = find_order_blocks(df_1h, atr_1h, disp_1h, timeframe="1H")
    fvgs_1h = find_fvgs(df_1h, atr_1h, timeframe="1H", completed_candles_only=True)
    liq_1h = detect_liquidity_levels(df_1h, structure_1h.swings, timeframe="1H")
    sweeps_1h = detect_liquidity_sweeps(df_1h, liq_1h, lookback_bars=5)

    # 1H SMC Location
    location_1h, desc_1h = determine_htf_location(
        current_price=last_price,
        htf_obs=obs_1h,
        htf_fvgs=fvgs_1h,
        htf_levels=liq_1h,
        htf_swings=structure_1h.swings,
        proximity_pct=config.smc.htf_proximity_threshold_pct,
        timeframe="1H",
    )
    location_1h_desc = f"1H SMC Location: {desc_1h}"

    # 15M Primary Setup Refinement Timeframe (Full SMC Analysis)
    structure_15m = None
    disp_15m = []
    obs_15m = []
    fvgs_15m = []
    liq_15m = []
    sweeps_15m = []
    location_15m = "NO SIGNIFICANT LOCATION"
    location_15m_desc = "15M SMC Location: No data"

    if df_15m is not None and not df_15m.empty and len(df_15m) >= 15:
        if atr_15m is None:
            atr_15m = calculate_atr(df_15m, period=config.smc.atr_period)
        structure_15m = analyze_market_structure(
            df_15m,
            atr_15m,
            timeframe="15M",
            swing_lookback=config.smc.swing_lookback,
            mss_requires_displacement=config.smc.mss_requires_displacement,
            mss_lookback_bars=config.smc.mss_lookback_bars,
            displacement_atr_mult=config.smc.displacement_atr_mult,
        )
        disp_15m = detect_displacements(df_15m, atr_15m, min_atr_mult=config.smc.displacement_atr_mult)
        obs_15m = find_order_blocks(df_15m, atr_15m, disp_15m, timeframe="15M")
        fvgs_15m = find_fvgs(df_15m, atr_15m, timeframe="15M", completed_candles_only=True)
        liq_15m = detect_liquidity_levels(df_15m, structure_15m.swings, timeframe="15M")
        sweeps_15m = detect_liquidity_sweeps(df_15m, liq_15m, lookback_bars=6)

        loc_15m_raw, desc_15m_raw = determine_htf_location(
            current_price=last_price,
            htf_obs=obs_15m,
            htf_fvgs=fvgs_15m,
            htf_levels=liq_15m,
            htf_swings=structure_15m.swings,
            proximity_pct=config.smc.htf_proximity_threshold_pct,
            timeframe="15M",
        )
        location_15m = loc_15m_raw
        location_15m_desc = f"15M SMC Location: {desc_15m_raw}"

    # Primary Setup Location (Synthesized from 1H and 15M)
    if location_1h == "NEAR DEMAND" or location_15m == "NEAR DEMAND":
        htf_location = "NEAR DEMAND"
    elif location_1h == "NEAR SUPPLY" or location_15m == "NEAR SUPPLY":
        htf_location = "NEAR SUPPLY"
    elif location_1h == "NEAR LIQUIDITY" or location_15m == "NEAR LIQUIDITY":
        htf_location = "NEAR LIQUIDITY"
    elif location_1h != "NO SIGNIFICANT LOCATION":
        htf_location = location_1h
    else:
        htf_location = location_15m

    htf_loc_desc = f"{location_1h_desc} | {location_15m_desc}"

    # Active OBs and FVGs across 1H and 15M
    active_obs_1h = [ob for ob in obs_1h if ob.state in ("ACTIVE", "MITIGATED")]
    active_fvgs_1h = [fvg for fvg in fvgs_1h if fvg.state in ("OPEN", "PARTIAL")]
    active_obs_15m = [ob for ob in obs_15m if ob.state in ("ACTIVE", "MITIGATED")]
    active_fvgs_15m = [fvg for fvg in fvgs_15m if fvg.state in ("OPEN", "PARTIAL")]

    # Combined active zones for confluence scoring
    all_active_obs = active_obs_1h + active_obs_15m
    all_active_fvgs = active_fvgs_1h + active_fvgs_15m
    all_sweeps = sweeps_1h + sweeps_15m

    # Nearby liquidity levels
    nearby_liq = [
        lvl for lvl in (liq_1h + liq_15m)
        if abs(last_price - lvl.price) / last_price * 100.0 <= 1.5
    ]

    # Calculate 100-pt Score (Independent from transient LTF micro triggers)
    score_breakdown = calculate_candidate_score(
        quote_volume_24h=quote_volume_24h,
        min_volume=config.filters.min_24h_quote_volume,
        rvol=rvol,
        min_rvol=config.filters.min_rvol,
        oi_metrics=oi_metrics,
        min_oi_pct=config.filters.min_1h_oi_change_pct,
        price_change_24h_pct=price_change_24h_pct,
        min_volatility_pct=config.filters.min_abs_price_change_pct,
        htf_location=htf_location,
        sweeps=all_sweeps,
        nearby_liquidity=nearby_liq,
        displacements=disp_1h,
        structure_1h=structure_1h,
        active_obs=all_active_obs,
        active_fvgs=all_active_fvgs,
        weights=config.weights,
    )

    # Determine Direction based on primary 1H / 15M setup context
    if structure_1h.bias == "BULLISH" or (structure_15m and structure_15m.bias == "BULLISH"):
        direction = "BULLISH"
    elif structure_1h.bias == "BEARISH" or (structure_15m and structure_15m.bias == "BEARISH"):
        direction = "BEARISH"
    else:
        direction = structure_4h.bias if structure_4h.bias != "NEUTRAL" else "NEUTRAL"

    # Stage logic
    has_recent_sweep = len(sweeps_1h) > 0 or len(sweeps_15m) > 0
    has_mss = structure_1h.latest_mss is not None or (structure_15m and structure_15m.latest_mss is not None)
    has_displacement = len(disp_1h) > 0 or len(disp_15m) > 0
    at_htf = htf_location in ("NEAR DEMAND", "NEAR SUPPLY")

    if has_recent_sweep and has_mss and (at_htf or has_displacement):
        stage = "CONFIRMED"
    elif at_htf or has_recent_sweep or has_displacement or (rvol >= config.filters.min_rvol and oi_metrics.change_1h_pct >= config.filters.min_1h_oi_change_pct):
        stage = "SETUP_FORMING"
    elif score_breakdown.total_score >= 45.0:
        stage = "WATCH"
    else:
        stage = "NO_TRADE"

    # Why interesting
    why_items = []
    if rvol >= config.filters.min_rvol:
        why_items.append(f"Elevated 1H RVOL ({rvol:.1f}x)")
    if oi_metrics.change_1h_pct >= config.filters.min_1h_oi_change_pct:
        why_items.append(f"OI Expansion ({oi_metrics.change_1h_pct:+.1f}%) with {oi_metrics.classification}")
    if at_htf:
        why_items.append(f"Located at {htf_location}")
    if has_recent_sweep:
        swp = sweeps_1h[-1] if sweeps_1h else sweeps_15m[-1]
        why_items.append(f"Liquidity swept ({swp.sweep_type})")
    if has_displacement:
        d = disp_1h[-1] if disp_1h else disp_15m[-1]
        why_items.append(f"Displacement move confirmed ({d.displacement_ratio:.1f}x ATR)")
    if structure_1h.bias != "NEUTRAL":
        why_items.append(f"1H {structure_1h.bias} market structure")
    if structure_4h.bias != "NEUTRAL":
        why_items.append(f"4H {structure_4h.bias} macro context")

    why_interesting = " | ".join(why_items) if why_items else "Volume and volatility qualify for active watchlist."

    # Waiting for
    waiting_items = []
    if not has_recent_sweep:
        waiting_items.append("Liquidity sweep" if not nearby_liq else f"Sweep of {nearby_liq[0].source} ({nearby_liq[0].side})")
    if not has_mss:
        waiting_items.append("MSS on 1H/15M with displacement")
    if not at_htf and htf_location != "NEAR LIQUIDITY":
        waiting_items.append("Retest into key 1H/15M Demand/Supply zone")
    if stage == "CONFIRMED":
        waiting_items = ["Manual chart inspection on TradingView / ATAS orderflow confirmation"]

    waiting_for = " + ".join(waiting_items) if waiting_items else "Setup confirmed; awaiting orderflow confluence."

    # Reference Trade Plan (only if enough SMC structure exists)
    trade_plan = None
    curr_atr = atr_1h.iloc[-1] if not atr_1h.empty and not pd.isna(atr_1h.iloc[-1]) else (last_price * 0.015)

    if stage in ("CONFIRMED", "SETUP_FORMING"):
        if direction == "BULLISH" and all_active_obs:
            matching_ob = next((ob for ob in all_active_obs if ob.ob_type == "BULLISH"), all_active_obs[0])
            entry = matching_ob.high if last_price >= matching_ob.high else last_price
            stop = round(matching_ob.low - (1.0 * curr_atr), 4)
            risk = entry - stop
            if risk > 0:
                t1 = round(entry + 1.0 * risk, 4)
                t2 = round(entry + 2.0 * risk, 4)
                t3 = round(entry + 3.0 * risk, 4)
                trade_plan = ReferenceTradePlan(
                    direction="LONG",
                    entry=entry,
                    stop=stop,
                    risk_amount=round(risk, 4),
                    target_1r=t1,
                    target_2r=t2,
                    target_3r=t3,
                    risk_reward="1:3.0",
                )
        elif direction == "BEARISH" and all_active_obs:
            matching_ob = next((ob for ob in all_active_obs if ob.ob_type == "BEARISH"), all_active_obs[0])
            entry = matching_ob.low if last_price <= matching_ob.low else last_price
            stop = round(matching_ob.high + (1.0 * curr_atr), 4)
            risk = stop - entry
            if risk > 0:
                t1 = round(entry - 1.0 * risk, 4)
                t2 = round(entry - 2.0 * risk, 4)
                t3 = round(entry - 3.0 * risk, 4)
                trade_plan = ReferenceTradePlan(
                    direction="SHORT",
                    entry=entry,
                    stop=stop,
                    risk_amount=round(risk, 4),
                    target_1r=t1,
                    target_2r=t2,
                    target_3r=t3,
                    risk_reward="1:3.0",
                )

    return Candidate(
        symbol=symbol,
        last_price=last_price,
        quote_volume_24h=quote_volume_24h,
        price_change_24h_pct=price_change_24h_pct,
        rvol=rvol,
        oi_metrics=oi_metrics,
        structure_4h=structure_4h,
        structure_1h=structure_1h,
        structure_15m=structure_15m,
        htf_location=htf_location,
        htf_location_desc=htf_loc_desc,
        active_obs_1h=active_obs_1h,
        active_fvgs_1h=active_fvgs_1h,
        sweeps_1h=sweeps_1h,
        nearby_liquidity=nearby_liq,
        displacements_1h=disp_1h,
        stage=stage,
        direction=direction,
        score_breakdown=score_breakdown,
        total_score=score_breakdown.total_score,
        why_interesting=why_interesting,
        waiting_for=waiting_for,
        trade_plan=trade_plan,
        filter_result=filter_res,
        location_1h=location_1h,
        location_1h_desc=location_1h_desc,
        location_15m=location_15m,
        location_15m_desc=location_15m_desc,
        active_obs_15m=active_obs_15m,
        active_fvgs_15m=active_fvgs_15m,
        sweeps_15m=sweeps_15m,
        displacements_15m=disp_15m,
        macro_context_4h=macro_context_4h,
        ltf_confirmation=LTFConfirmation(
            status="NOT_CHECKED",
            direction=direction,
            summary="Awaiting Stage 2 analysis",
        ),
        funding=funding if funding is not None else FundingData(),
    )


def evaluate_ltf_confirmation(
    candidate: Candidate,
    df_5m: pd.DataFrame,
    df_3m: pd.DataFrame,
    df_1m: pd.DataFrame,
    config: AppConfig,
) -> LTFConfirmation:
    """
    Evaluate lower-timeframe entry confirmation (5M, 3M, 1M) for a shortlisted candidate.
    Strictly uses completed candles (df.iloc[:-1]) to prevent lookahead and incomplete bar bias.
    """
    if df_5m.empty or len(df_5m) < 15:
        return LTFConfirmation(
            status="NOT_CHECKED",
            direction=candidate.direction,
            summary="Insufficient 5M data for micro confirmation",
        )

    # 1. Strict candle discipline: Completed candles only (drop the currently forming unclosed bar)
    c_5m = df_5m.iloc[:-1].copy()
    c_3m = df_3m.iloc[:-1].copy() if not df_3m.empty and len(df_3m) >= 15 else pd.DataFrame()
    c_1m = df_1m.iloc[:-1].copy() if not df_1m.empty and len(df_1m) >= 20 else pd.DataFrame()

    target_dir = candidate.direction if candidate.direction in ("BULLISH", "BEARISH") else (
        "BULLISH" if candidate.structure_1h.bias == "BULLISH" else "BEARISH"
    )

    # 5M Analysis: Swings and Liquidity Sweeps
    atr_5m = calculate_atr(c_5m, period=config.smc.atr_period)
    swings_5m = find_swings(c_5m, lookback=config.smc.ltf_swing_lookback)
    liq_5m = detect_liquidity_levels(c_5m, swings_5m, timeframe="5M")
    sweeps_5m = detect_liquidity_sweeps(c_5m, liq_5m, lookback_bars=config.smc.ltf_sweep_lookback_bars)

    m5_sweep = False
    m5_sweep_desc = ""
    latest_sweep_5m = None
    for sw in reversed(sweeps_5m):
        if target_dir == "BULLISH" and sw.sweep_type == "BULLISH_SWEEP":
            m5_sweep = True
            m5_sweep_desc = f"5M sell-side sweep detected (${sw.extreme_price:.2f})"
            latest_sweep_5m = sw
            break
        elif target_dir == "BEARISH" and sw.sweep_type == "BEARISH_SWEEP":
            m5_sweep = True
            m5_sweep_desc = f"5M buy-side sweep detected (${sw.extreme_price:.2f})"
            latest_sweep_5m = sw
            break

    # 3M Analysis: Displacements and 3M Structure
    m3_disp = False
    m3_disp_desc = ""
    latest_disp_3m = None
    disps_3m = []
    if not c_3m.empty:
        atr_3m = calculate_atr(c_3m, period=config.smc.atr_period)
        disps_3m = detect_displacements(c_3m, atr_3m, min_atr_mult=config.smc.ltf_displacement_atr_mult)
        recent_disps_3m = [d for d in disps_3m if d.end_index >= len(c_3m) - 6]
        for d in reversed(recent_disps_3m):
            if d.direction == target_dir:
                m3_disp = True
                m3_disp_desc = f"3M {target_dir.lower()} displacement detected ({d.displacement_ratio:.1f}x ATR)"
                latest_disp_3m = d
                break

    # 1M Analysis: Microstructure Shift (MSS) or BOS
    m1_mss = False
    m1_mss_desc = ""
    latest_break_1m = None
    if not c_1m.empty:
        atr_1m = calculate_atr(c_1m, period=config.smc.atr_period)
        struct_1m = analyze_market_structure(
            c_1m,
            atr_1m,
            timeframe="1M",
            swing_lookback=config.smc.ltf_swing_lookback,
            mss_requires_displacement=False,
            mss_lookback_bars=config.smc.ltf_mss_lookback_bars,
        )
        # Find matching 1M breaks in target direction (newest first)
        matching_breaks = [b for b in reversed(struct_1m.recent_breaks) if b.direction == target_dir]
        
        # If 5M sweep exists, prefer a break that occurred at or after the sweep
        target_break = None
        if latest_sweep_5m is not None:
            target_break = next(
                (b for b in matching_breaks if b.timestamp >= latest_sweep_5m.timestamp),
                None
            )
        
        # If no post-sweep break, take newest break (will be evaluated for temporal ordering below)
        if target_break is None and matching_breaks:
            target_break = matching_breaks[0]

        if target_break is not None:
            m1_mss = True
            latest_break_1m = target_break
            m1_mss_desc = f"1M {target_dir.lower()} {target_break.break_type} confirmed at ${target_break.broken_level:.2f}"

    # Chronology validation:
    # 5M sweep timestamp <= 1M MSS timestamp for sweep -> MSS confirmation
    sweep_mss_valid = False
    if m5_sweep and m1_mss and latest_sweep_5m is not None and latest_break_1m is not None:
        sweep_mss_valid = (latest_sweep_5m.timestamp <= latest_break_1m.timestamp)

    sweep_disp_valid = False
    if m5_sweep and m3_disp and latest_sweep_5m is not None and latest_disp_3m is not None:
        sweep_disp_valid = (latest_sweep_5m.timestamp <= latest_disp_3m.end_timestamp)

    # Invalidation Check: Strong displacement blowing through stop
    is_invalidated = False
    if not c_3m.empty and target_dir == "BULLISH":
        opp_disps = [d for d in disps_3m if d.direction == "BEARISH" and d.displacement_ratio >= 2.2 and d.end_index >= len(c_3m) - 4]
        if opp_disps and candidate.trade_plan and candidate.last_price < candidate.trade_plan.stop:
            is_invalidated = True
    elif not c_3m.empty and target_dir == "BEARISH":
        opp_disps = [d for d in disps_3m if d.direction == "BULLISH" and d.displacement_ratio >= 2.2 and d.end_index >= len(c_3m) - 4]
        if opp_disps and candidate.trade_plan and candidate.last_price > candidate.trade_plan.stop:
            is_invalidated = True

    # Assemble details and summary
    details = []
    if m5_sweep:
        details.append(m5_sweep_desc)
    if m3_disp:
        details.append(m3_disp_desc)
    if m1_mss:
        details.append(m1_mss_desc)

    if is_invalidated:
        status = "INVALIDATED"
        summary = "Invalidated: Lower timeframe aggressive expansion against setup direction."
    elif m5_sweep and m3_disp and m1_mss and sweep_mss_valid and sweep_disp_valid:
        status = "CONFIRMED"
        summary = f"{m5_sweep_desc}; {m3_disp_desc}; {m1_mss_desc}."
    elif m5_sweep and m1_mss and sweep_mss_valid:
        status = "CONFIRMED"
        summary = f"{m5_sweep_desc}; {m3_disp_desc or '3M displacement pending'}; {m1_mss_desc}."
    elif (m5_sweep and m1_mss) or (m5_sweep and m3_disp) or (m3_disp and m1_mss):
        status = "PARTIAL_CONFIRMATION"
        if m5_sweep and m1_mss and not sweep_mss_valid:
            summary = f"{m5_sweep_desc}; 1M MSS pending post-sweep (prior MSS preceded sweep); {m3_disp_desc or '3M displacement pending'}."
        else:
            mss_status = m1_mss_desc if m1_mss else "1M MSS pending"
            summary = f"{m5_sweep_desc or '5M sweep pending'}; {m3_disp_desc or '3M displacement pending'}; {mss_status}."
    elif m5_sweep or m3_disp or m1_mss:
        status = "PARTIAL_CONFIRMATION"
        summary = f"{'; '.join(details)}; waiting for remaining micro alignment."
    elif candidate.stage in ("SETUP_FORMING", "CONFIRMED"):
        status = "WATCHING"
        summary = "Watching 5M/3M/1M: In 1H/15M setup location; awaiting micro sweep or MSS."
    else:
        status = "NO_TRIGGER"
        summary = "No lower-timeframe entry trigger detected."

    return LTFConfirmation(
        status=status,
        direction=target_dir,
        m5_sweep=m5_sweep,
        m5_sweep_desc=m5_sweep_desc,
        m3_displacement=m3_disp,
        m3_displacement_desc=m3_disp_desc,
        m1_mss=m1_mss,
        m1_mss_desc=m1_mss_desc,
        summary=summary,
        details=details,
    )
