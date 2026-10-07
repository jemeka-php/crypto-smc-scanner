"""
Volatility calculation and filter helpers.
"""

from typing import Tuple


def check_24h_volatility(
    price_change_24h_pct: float,
    threshold_pct: float = 3.0,
) -> Tuple[bool, float]:
    """
    Check if the absolute 24H price change meets the volatility threshold.

    Args:
        price_change_24h_pct: 24h price change as percentage (e.g. +4.5 or -5.2)
        threshold_pct: Minimum absolute percentage required (default 3.0%)

    Returns:
        Tuple of (qualifies_bool, absolute_change_pct)
    """
    abs_change = abs(price_change_24h_pct)
    qualifies = abs_change >= threshold_pct
    return qualifies, round(abs_change, 2)
