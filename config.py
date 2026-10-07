"""
Configuration module for Crypto SMC Candidate Scanner.
Provides strategy parameters, filter thresholds, scoring weights, and environment settings.
"""

from dataclasses import dataclass, field
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


@dataclass
class FilterConfig:
    min_24h_quote_volume: float = 50_000_000.0  # $50M quote volume
    min_rvol: float = 1.5                        # 1.5x median 20 completed 1H volume
    min_abs_price_change_pct: float = 3.0       # 3% absolute 24H price change
    min_1h_oi_change_pct: float = 5.0           # +5% 1H OI change


@dataclass
class SMCConfig:
    # ATR
    atr_period: int = 14                        # Wilder RMA period
    
    # Volume
    volume_lookback: int = 20                   # Rolling median lookback
    volume_high_mult: float = 1.5               # Volume >= 1.5x median
    volume_low_mult: float = 0.5                # Volume <= 0.5x median
    
    # Swings
    swing_lookback: int = 5                     # Bars on left and right for swing confirmation
    
    # Displacement
    displacement_atr_mult: float = 1.5          # Total body displacement >= 1.5 * ATR
    displacement_max_bars: int = 3              # Consecutive candles window
    
    # MSS
    mss_requires_displacement: bool = True
    mss_lookback_bars: int = 3
    
    # Order Blocks
    ob_doji_body_ratio: float = 0.1             # Body < 0.1 * ATR considered doji fallback
    
    # FVG
    fvg_min_gap_atr_mult: float = 0.25          # Min gap >= 0.25 * ATR
    
    # Liquidity
    equal_high_low_threshold_pct: float = 0.1   # Tolerance for equal highs/lows in % (e.g. 0.1%)
    htf_proximity_threshold_pct: float = 0.8    # Within 0.8% of HTF level is considered NEAR

    # Lower Timeframe (LTF) Entry Confirmation (5M / 3M / 1M)
    ltf_swing_lookback: int = 3
    ltf_sweep_lookback_bars: int = 6
    ltf_displacement_atr_mult: float = 1.3
    ltf_mss_lookback_bars: int = 4


@dataclass
class ScoringWeights:
    liquidity: float = 10.0
    rvol: float = 15.0
    oi: float = 15.0
    volatility: float = 10.0
    htf_location: float = 15.0
    liquidity_sweep: float = 10.0
    displacement: float = 10.0
    bos_mss: float = 10.0
    ob_fvg_confluence: float = 5.0

    def total(self) -> float:
        return (
            self.liquidity
            + self.rvol
            + self.oi
            + self.volatility
            + self.htf_location
            + self.liquidity_sweep
            + self.displacement
            + self.bos_mss
            + self.ob_fvg_confluence
        )

    def is_valid(self) -> bool:
        return abs(self.total() - 100.0) < 1e-6


@dataclass
class FundingConfig:
    positive_threshold: float = 0.0001   # > +0.0001 (+0.01%) is POSITIVE
    negative_threshold: float = -0.0001  # < -0.0001 (-0.01%) is NEGATIVE


@dataclass
class AppConfig:
    exchange_id: str = "binance"
    settle_currency: str = "USDT"
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() == "true"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    refresh_interval_seconds: int = int(os.getenv("REFRESH_INTERVAL_SECONDS", "60"))
    
    filters: FilterConfig = field(default_factory=FilterConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    funding: FundingConfig = field(default_factory=FundingConfig)


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
