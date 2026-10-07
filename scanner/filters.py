"""
Quantitative screening filters: Liquidity, Volatility, RVOL, and Open Interest.
"""

from dataclasses import dataclass
from typing import List, Tuple
from config import FilterConfig
from data.exchange import MarketTicker


@dataclass
class FilterResult:
    passed_liquidity: bool
    passed_volatility: bool
    passed_rvol: bool
    passed_oi: bool
    all_passed: bool
    reasons: List[str]


def filter_tickers_initial(
    tickers: List[MarketTicker],
    config: FilterConfig,
) -> Tuple[List[MarketTicker], int, int]:
    """
    Step 1: Apply fast 24H Liquidity and Volatility filters across the entire universe.

    Returns:
        (filtered_tickers, passed_liquidity_count, passed_volatility_count)
    """
    passed_liq = []
    passed_vol_count = 0

    for t in tickers:
        if t.quote_volume_24h >= config.min_24h_quote_volume:
            passed_liq.append(t)
            if abs(t.price_change_24h_pct) >= config.min_abs_price_change_pct:
                passed_vol_count += 1

    # Return those that passed liquidity (volatility is checked in pipeline)
    return passed_liq, len(passed_liq), passed_vol_count


def evaluate_candidate_filters(
    quote_volume_24h: float,
    price_change_24h_pct: float,
    rvol: float,
    oi_1h_change_pct: float,
    config: FilterConfig,
) -> FilterResult:
    """
    Evaluate all 4 quantitative filters for an individual market.
    """
    reasons = []

    passed_liq = quote_volume_24h >= config.min_24h_quote_volume
    if not passed_liq:
        reasons.append(f"24H Vol ${quote_volume_24h:,.0f} < ${config.min_24h_quote_volume:,.0f}")

    passed_vol = abs(price_change_24h_pct) >= config.min_abs_price_change_pct
    if not passed_vol:
        reasons.append(f"|24H Change| {abs(price_change_24h_pct):.2f}% < {config.min_abs_price_change_pct:.1f}%")

    passed_rvol = rvol >= config.min_rvol
    if not passed_rvol:
        reasons.append(f"1H RVOL {rvol:.2f}x < {config.min_rvol:.2f}x")

    passed_oi = oi_1h_change_pct >= config.min_1h_oi_change_pct
    if not passed_oi:
        reasons.append(f"1H OI {oi_1h_change_pct:+.2f}% < +{config.min_1h_oi_change_pct:.1f}%")

    all_passed = passed_liq and passed_vol and passed_rvol and passed_oi

    return FilterResult(
        passed_liquidity=passed_liq,
        passed_volatility=passed_vol,
        passed_rvol=passed_rvol,
        passed_oi=passed_oi,
        all_passed=all_passed,
        reasons=reasons,
    )
