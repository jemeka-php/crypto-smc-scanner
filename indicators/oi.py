"""
Open Interest (OI) metrics and Price + OI Classification.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OIMetrics:
    current_oi: float
    oi_1h_ago: Optional[float]
    oi_4h_ago: Optional[float]
    change_1h_pct: float
    change_4h_pct: float
    classification: str
    interpretation: str


def classify_price_oi(price_change_pct: float, oi_change_pct: float) -> tuple[str, str]:
    """
    Classify Price/OI relationship strictly as candidate discovery signals.

    Rules:
    - Price falling + OI increasing:
        Display: 'OI BUILD + PRICE WEAKNESS'
        Interpretation: Potentially interesting for a bullish reversal if the market
        is reaching a relevant SMC location and subsequently confirms bullish structure.
    - Price rising + OI increasing:
        Display: 'OI BUILD + PRICE STRENGTH'
        Interpretation: Potentially interesting for a bearish reversal only if SMC
        structure subsequently confirms it.
    - Price falling + OI falling:
        Display: 'OI CONTRACTION + PRICE WEAKNESS'
        Interpretation: Positions closing as price declines.
    - Price rising + OI falling:
        Display: 'OI CONTRACTION + PRICE STRENGTH'
        Interpretation: Positions closing as price rallies.
    """
    price_rising = price_change_pct >= 0
    oi_increasing = oi_change_pct >= 0

    if not price_rising and oi_increasing:
        classification = "OI BUILD + PRICE WEAKNESS"
        interpretation = (
            "Potentially interesting for a bullish reversal IF market reaches a relevant "
            "SMC location and subsequently confirms bullish structure."
        )
    elif price_rising and oi_increasing:
        classification = "OI BUILD + PRICE STRENGTH"
        interpretation = (
            "Potentially interesting for a bearish reversal ONLY IF SMC structure "
            "subsequently confirms it."
        )
    elif not price_rising and not oi_increasing:
        classification = "OI CONTRACTION + PRICE WEAKNESS"
        interpretation = "Open interest declining during price drop (position contraction during weakness; direction indeterminate from price and OI alone)."
    else:
        classification = "OI CONTRACTION + PRICE STRENGTH"
        interpretation = "Open interest declining during price rise (position contraction during strength; direction indeterminate from price and OI alone)."

    return classification, interpretation


def compute_oi_metrics(
    current_oi: float,
    oi_1h_ago: Optional[float],
    oi_4h_ago: Optional[float],
    price_1h_change_pct: float = 0.0,
) -> OIMetrics:
    """
    Compute 1H and 4H OI changes and generate discovery classification.
    """
    change_1h = 0.0
    if oi_1h_ago and oi_1h_ago > 0:
        change_1h = ((current_oi - oi_1h_ago) / oi_1h_ago) * 100.0

    change_4h = 0.0
    if oi_4h_ago and oi_4h_ago > 0:
        change_4h = ((current_oi - oi_4h_ago) / oi_4h_ago) * 100.0

    classification, interpretation = classify_price_oi(price_1h_change_pct, change_1h)

    return OIMetrics(
        current_oi=current_oi,
        oi_1h_ago=oi_1h_ago,
        oi_4h_ago=oi_4h_ago,
        change_1h_pct=round(change_1h, 2),
        change_4h_pct=round(change_4h, 2),
        classification=classification,
        interpretation=interpretation,
    )
