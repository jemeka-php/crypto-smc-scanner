"""
Deterministic 100-point candidate scoring engine.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from config import ScoringWeights
from indicators.oi import OIMetrics
from smc.structure import MarketStructure
from smc.displacement import Displacement
from smc.liquidity import LiquiditySweep, LiquidityLevel
from smc.order_blocks import OrderBlock
from smc.fvg import FVG


@dataclass
class ScoreBreakdown:
    liquidity_score: float
    rvol_score: float
    oi_score: float
    volatility_score: float
    htf_location_score: float
    liquidity_sweep_score: float
    displacement_score: float
    bos_mss_score: float
    ob_fvg_confluence_score: float
    total_score: float
    details: Dict[str, str]


def calculate_candidate_score(
    quote_volume_24h: float,
    min_volume: float,
    rvol: float,
    min_rvol: float,
    oi_metrics: OIMetrics,
    min_oi_pct: float,
    price_change_24h_pct: float,
    min_volatility_pct: float,
    htf_location: str,
    sweeps: List[LiquiditySweep],
    nearby_liquidity: List[LiquidityLevel],
    displacements: List[Displacement],
    structure_1h: MarketStructure,
    active_obs: List[OrderBlock],
    active_fvgs: List[FVG],
    weights: ScoringWeights,
) -> ScoreBreakdown:
    """
    Compute deterministic score out of 100 points based on configurable weights.
    """
    details: Dict[str, str] = {}

    # 1. Liquidity (Max: weights.liquidity = 10)
    vol_ratio = quote_volume_24h / max(min_volume, 1.0)
    if vol_ratio >= 2.0:
        liq_score = weights.liquidity
    elif vol_ratio >= 1.0:
        liq_score = weights.liquidity * (0.7 + 0.3 * (vol_ratio - 1.0))
    else:
        liq_score = weights.liquidity * (0.7 * vol_ratio)
    liq_score = round(min(liq_score, weights.liquidity), 1)
    details["liquidity"] = f"${quote_volume_24h:,.0f} (Ratio: {vol_ratio:.1f}x threshold)"

    # 2. RVOL (Max: weights.rvol = 15)
    if rvol >= 2.5:
        rvol_score = weights.rvol
    elif rvol >= 2.0:
        rvol_score = weights.rvol * 0.85
    elif rvol >= min_rvol:
        rvol_score = weights.rvol * 0.70
    elif rvol > 1.0:
        rvol_score = weights.rvol * (0.5 * (rvol / min_rvol))
    else:
        rvol_score = 0.0
    rvol_score = round(min(rvol_score, weights.rvol), 1)
    details["rvol"] = f"{rvol:.2f}x (Threshold: {min_rvol:.1f}x)"

    # 3. Open Interest (Max: weights.oi = 15)
    oi_chg = oi_metrics.change_1h_pct
    if oi_chg >= 10.0:
        oi_score = weights.oi
    elif oi_chg >= 7.0:
        oi_score = weights.oi * 0.85
    elif oi_chg >= min_oi_pct:
        oi_score = weights.oi * 0.70
    elif oi_chg > 0:
        oi_score = weights.oi * (0.4 * (oi_chg / min_oi_pct))
    else:
        oi_score = 0.0
    oi_score = round(min(oi_score, weights.oi), 1)
    details["oi"] = f"{oi_chg:+.2f}% 1H ({oi_metrics.classification})"

    # 4. Volatility (Max: weights.volatility = 10)
    abs_chg = abs(price_change_24h_pct)
    if abs_chg >= 5.0:
        vol_score = weights.volatility
    elif abs_chg >= min_volatility_pct:
        vol_score = weights.volatility * 0.75
    else:
        vol_score = weights.volatility * (abs_chg / min_volatility_pct) * 0.5
    vol_score = round(min(vol_score, weights.volatility), 1)
    details["volatility"] = f"{price_change_24h_pct:+.2f}% 24H (Req: >={min_volatility_pct:.1f}%)"

    # 5. HTF Location (Max: weights.htf_location = 15)
    if htf_location in ("NEAR DEMAND", "NEAR SUPPLY"):
        htf_score = weights.htf_location
    elif htf_location == "NEAR LIQUIDITY":
        htf_score = weights.htf_location * 0.80
    elif htf_location == "MID-RANGE":
        htf_score = weights.htf_location * 0.40
    else:
        htf_score = 0.0
    htf_score = round(min(htf_score, weights.htf_location), 1)
    details["htf_location"] = htf_location

    # 6. Liquidity Sweep (Max: weights.liquidity_sweep = 10)
    if sweeps:
        sweep_score = weights.liquidity_sweep
        details["liquidity_sweep"] = f"Confirmed {sweeps[-1].sweep_type}"
    elif nearby_liquidity:
        sweep_score = weights.liquidity_sweep * 0.60
        details["liquidity_sweep"] = f"Nearby {nearby_liquidity[0].source} ({nearby_liquidity[0].side})"
    else:
        sweep_score = 0.0
        details["liquidity_sweep"] = "No recent sweep or immediate pool"
    sweep_score = round(min(sweep_score, weights.liquidity_sweep), 1)

    # 7. Displacement (Max: weights.displacement = 10)
    if displacements:
        last_disp = displacements[-1]
        disp_score = weights.displacement if last_disp.displacement_ratio >= 2.0 else weights.displacement * 0.80
        details["displacement"] = f"{last_disp.direction} ({last_disp.displacement_ratio:.1f}x ATR)"
    else:
        disp_score = 0.0
        details["displacement"] = "No qualifying displacement"
    disp_score = round(min(disp_score, weights.displacement), 1)

    # 8. BOS / MSS (Max: weights.bos_mss = 10)
    if structure_1h.latest_mss and structure_1h.latest_mss.index >= (len(structure_1h.swings) - 4):
        bos_mss_score = weights.bos_mss
        details["bos_mss"] = f"Recent MSS ({structure_1h.latest_mss.direction})"
    elif structure_1h.latest_bos:
        bos_mss_score = weights.bos_mss * 0.70
        details["bos_mss"] = f"Recent BOS ({structure_1h.latest_bos.direction})"
    else:
        bos_mss_score = weights.bos_mss * 0.30 if structure_1h.bias != "NEUTRAL" else 0.0
        details["bos_mss"] = f"Bias: {structure_1h.bias}"
    bos_mss_score = round(min(bos_mss_score, weights.bos_mss), 1)

    # 9. OB / FVG Confluence (Max: weights.ob_fvg_confluence = 5)
    has_ob = len(active_obs) > 0
    has_fvg = len(active_fvgs) > 0
    if has_ob and has_fvg:
        confluence_score = weights.ob_fvg_confluence
        details["confluence"] = f"{len(active_obs)} Active OB(s) + {len(active_fvgs)} FVG(s)"
    elif has_ob or has_fvg:
        confluence_score = weights.ob_fvg_confluence * 0.60
        details["confluence"] = f"{'OB present' if has_ob else 'FVG present'}"
    else:
        confluence_score = 0.0
        details["confluence"] = "No active OB or FVG confluence"
    confluence_score = round(min(confluence_score, weights.ob_fvg_confluence), 1)

    total = (
        liq_score
        + rvol_score
        + oi_score
        + vol_score
        + htf_score
        + sweep_score
        + disp_score
        + bos_mss_score
        + confluence_score
    )

    return ScoreBreakdown(
        liquidity_score=liq_score,
        rvol_score=rvol_score,
        oi_score=oi_score,
        volatility_score=vol_score,
        htf_location_score=htf_score,
        liquidity_sweep_score=sweep_score,
        displacement_score=disp_score,
        bos_mss_score=bos_mss_score,
        ob_fvg_confluence_score=confluence_score,
        total_score=round(total, 1),
        details=details,
    )
