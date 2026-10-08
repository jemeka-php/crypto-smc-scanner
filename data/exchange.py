"""
Exchange client for Binance USDT-M Perpetual Futures data collection with caching and demo mode support.
"""

from dataclasses import dataclass
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests

from data.cache import GLOBAL_CACHE

logger = logging.getLogger(__name__)


@dataclass
class MarketTicker:
    symbol: str                 # e.g. "BTCUSDT"
    last_price: float
    price_change_24h_pct: float
    quote_volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: int


class ExchangeClient:
    """
    Binance USDT-M Perpetual Futures data client.
    Uses Binance public REST endpoints for batch scanning, with rate limit protection and caching.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(
        self,
        demo_mode: bool = False,
        session: Optional[requests.Session] = None,
        proxy_url: Optional[str] = None,
    ):
        self.demo_mode = demo_mode
        self.session = session or requests.Session()
        self.proxy_url = proxy_url or os.getenv("PROXY_URL") or os.getenv("HTTPS_PROXY")
        if self.proxy_url:
            self.session.proxies.update({
                "http": self.proxy_url,
                "https": self.proxy_url,
            })
        self.session.headers.update({
            "User-Agent": "CryptoSMCScanner/1.0",
            "Accept": "application/json",
        })
        self.is_stale = False
        self.last_error_message: Optional[str] = None
        self.last_fetch_time = 0.0
        self.request_count = 0


    def reset_request_count(self) -> None:
        """Reset live API request counter."""
        self.request_count = 0

    def fetch_universe_tickers(self) -> List[MarketTicker]:
        """
        Fetch all active USDT-settled perpetual contracts and 24h ticker metrics in one call.
        Filters out inactive, quarterly/delivery futures, and non-USDT pairs.
        """
        if self.demo_mode:
            return self._get_demo_tickers()

        cache_key = "universe_tickers_24h"
        cached = GLOBAL_CACHE.get(cache_key, max_age_seconds=30)
        if cached:
            return cached

        url = f"{self.BASE_URL}/fapi/v1/ticker/24hr"
        try:
            self.request_count += 1
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                if resp.status_code in (403, 451):
                    self.last_error_message = (
                        f"Binance Geo-Restriction (HTTP {resp.status_code}): US cloud servers "
                        "(e.g. Streamlit Community Cloud) are restricted by Binance Futures. "
                        "Enable 'Demo Mode' in the sidebar to run the full SMC scanner, or configure a non-US proxy in secrets."
                    )
                else:
                    self.last_error_message = f"Exchange returned HTTP {resp.status_code}"
                logger.error(f"Failed to fetch tickers: {self.last_error_message}")
                self.is_stale = True
                return []

            data = resp.json()
            tickers: List[MarketTicker] = []
            now_ms = int(time.time() * 1000)

            for item in data:
                sym = item.get("symbol", "")
                # Must be USDT perpetual (e.g. BTCUSDT, exclude quarterly like BTCUSDT_241227)
                if not sym.endswith("USDT") or "_" in sym:
                    continue

                try:
                    last_price = float(item.get("lastPrice", 0.0))
                    change_pct = float(item.get("priceChangePercent", 0.0))
                    quote_vol = float(item.get("quoteVolume", 0.0))
                    high_price = float(item.get("highPrice", last_price))
                    low_price = float(item.get("lowPrice", last_price))
                    close_time = int(item.get("closeTime", now_ms))

                    if last_price <= 0 or quote_vol <= 0:
                        continue

                    tickers.append(
                        MarketTicker(
                            symbol=sym,
                            last_price=last_price,
                            price_change_24h_pct=change_pct,
                            quote_volume_24h=quote_vol,
                            high_24h=high_price,
                            low_24h=low_price,
                            timestamp=close_time,
                        )
                    )
                except (ValueError, TypeError):
                    continue

            self.is_stale = False
            self.last_error_message = None
            self.last_fetch_time = time.time()
            GLOBAL_CACHE.set(cache_key, tickers)
            return tickers

        except Exception as e:
            self.last_error_message = f"Connection error: {str(e)}"
            logger.error(f"Error fetching 24hr tickers: {e}")
            self.is_stale = True
            return []

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol and interval ('15m', '1h', '4h').
        Returns DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        if self.demo_mode:
            return self._get_demo_ohlcv(symbol, interval, limit)

        cache_key = f"ohlcv_{symbol}_{interval}_{limit}"
        cached = GLOBAL_CACHE.get(cache_key, max_age_seconds=45)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        try:
            self.request_count += 1
            resp = self.session.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                logger.warning(f"Error fetching klines for {symbol} {interval}: HTTP {resp.status_code}")
                return pd.DataFrame()

            raw_klines = resp.json()
            if not raw_klines or not isinstance(raw_klines, list):
                return pd.DataFrame()

            # Format: [open_time, open, high, low, close, volume, close_time, quote_vol, trades, ...]
            records = []
            for k in raw_klines:
                records.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

            GLOBAL_CACHE.set(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Exception fetching OHLCV for {symbol} {interval}: {e}")
            return pd.DataFrame()

    def fetch_oi_history(self, symbol: str) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Fetch Open Interest history for symbol.
        Returns:
            Tuple of (current_oi, oi_1h_ago, oi_4h_ago) in quote asset value (USDT).
        """
        if self.demo_mode:
            return self._get_demo_oi(symbol)

        cache_key = f"oi_hist_{symbol}"
        cached = GLOBAL_CACHE.get(cache_key, max_age_seconds=60)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": "1h",
            "limit": 6,
        }

        try:
            self.request_count += 1
            resp = self.session.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                logger.warning(f"Error fetching OI history for {symbol}: HTTP {resp.status_code}")
                return 0.0, None, None

            data = resp.json()
            if not data or not isinstance(data, list) or len(data) < 2:
                return 0.0, None, None

            # Items are chronological. Last item is current/most recent.
            current_oi = float(data[-1].get("sumOpenInterestValue", 0.0))
            oi_1h_ago = float(data[-2].get("sumOpenInterestValue", 0.0))
            oi_4h_ago = float(data[-5].get("sumOpenInterestValue", 0.0)) if len(data) >= 5 else None

            res = (current_oi, oi_1h_ago, oi_4h_ago)
            GLOBAL_CACHE.set(cache_key, res)
            return res

        except Exception as e:
            logger.error(f"Exception fetching OI for {symbol}: {e}")
            return 0.0, None, None

    def fetch_funding_info(self, symbol: str) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """
        Fetch current funding rate info for symbol from Binance /fapi/v1/premiumIndex.
        Returns:
            Tuple of (last_funding_rate, time_ms, next_funding_time_ms)
            Returns (None, None, None) on failure or when unavailable.
        """
        if self.demo_mode:
            return self._get_demo_funding(symbol)

        cache_key = f"funding_{symbol}"
        cached = GLOBAL_CACHE.get(cache_key, max_age_seconds=60)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/fapi/v1/premiumIndex"
        params = {"symbol": symbol}

        try:
            self.request_count += 1
            resp = self.session.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                logger.warning(f"Error fetching funding for {symbol}: HTTP {resp.status_code}")
                return None, None, None

            data = resp.json()
            if not data or not isinstance(data, dict):
                return None, None, None

            raw_rate = data.get("lastFundingRate")
            rate = float(raw_rate) if raw_rate is not None else None
            time_ms = int(data.get("time")) if data.get("time") is not None else None
            next_time_ms = int(data.get("nextFundingTime")) if data.get("nextFundingTime") is not None else None

            res = (rate, time_ms, next_time_ms)
            GLOBAL_CACHE.set(cache_key, res)
            return res

        except Exception as e:
            logger.warning(f"Exception fetching funding for {symbol}: {e}")
            return None, None, None

    # ---------------- Demo Mode Mock Data ----------------

    def _get_demo_tickers(self) -> List[MarketTicker]:
        """Realistic simulated tickers for demo mode or offline development."""
        now_ms = int(time.time() * 1000)
        demo_data = [
            ("SOLUSDT", 142.50, -4.10, 480_000_000.0, 150.2, 140.1),
            ("LINKUSDT", 18.25, -3.70, 115_000_000.0, 19.4, 18.0),
            ("ETHUSDT", 2620.0, -3.20, 1_450_000_000.0, 2715.0, 2605.0),
            ("BTCUSDT", 63400.0, -2.10, 3_800_000_000.0, 64900.0, 62800.0),
            ("AVAXUSDT", 28.40, +5.20, 160_000_000.0, 29.1, 26.8),
            ("SUIUSDT", 1.95, +7.80, 290_000_000.0, 2.05, 1.78),
            ("NEARUSDT", 4.80, +3.60, 95_000_000.0, 5.02, 4.60),
            ("DOGEUSDT", 0.125, +4.20, 240_000_000.0, 0.131, 0.119),
            ("XRPUSDT", 0.585, +1.10, 180_000_000.0, 0.595, 0.578),
            ("ADAUSDT", 0.355, -1.40, 42_000_000.0, 0.362, 0.350),  # Low vol
            ("BNBUSDT", 580.0, +0.80, 210_000_000.0, 588.0, 574.0),
            ("PEPEUSDT", 0.0000098, -5.50, 320_000_000.0, 0.0000105, 0.0000095),
        ]
        return [
            MarketTicker(
                symbol=d[0],
                last_price=d[1],
                price_change_24h_pct=d[2],
                quote_volume_24h=d[3],
                high_24h=d[4],
                low_24h=d[5],
                timestamp=now_ms,
            )
            for d in demo_data
        ]

    def _get_demo_oi(self, symbol: str) -> Tuple[float, Optional[float], Optional[float]]:
        """Simulated OI for demo mode."""
        oi_map = {
            "SOLUSDT": (540_000_000.0, 499_000_000.0, 480_000_000.0),   # +8.2% 1H
            "LINKUSDT": (95_000_000.0, 88_700_000.0, 85_000_000.0),    # +7.1% 1H
            "ETHUSDT": (2_100_000_000.0, 1_985_000_000.0, 1_900_000_000.0), # +5.8% 1H
            "AVAXUSDT": (145_000_000.0, 137_000_000.0, 130_000_000.0), # +5.8% 1H
            "SUIUSDT": (180_000_000.0, 168_000_000.0, 155_000_000.0),  # +7.1% 1H
            "BTCUSDT": (8_100_000_000.0, 8_000_000_000.0, 7_900_000_000.0), # +1.2% 1H
            "PEPEUSDT": (120_000_000.0, 114_000_000.0, 110_000_000.0), # +5.2% 1H
        }
        return oi_map.get(symbol, (0.0, None, None))

    def _get_demo_funding(self, symbol: str) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Simulated funding rate for demo mode."""
        now_ms = int(time.time() * 1000)
        next_funding = now_ms + (4 * 3600 * 1000)
        demo_map = {
            "SOLUSDT": (0.00015, now_ms, next_funding),    # +0.015% POSITIVE
            "LINKUSDT": (-0.00012, now_ms, next_funding),   # -0.012% NEGATIVE
            "ETHUSDT": (0.00008, now_ms, next_funding),    # +0.008% NEUTRAL
            "BTCUSDT": (0.00010, now_ms, next_funding),    # +0.010% NEUTRAL
            "PEPEUSDT": (0.00025, now_ms, next_funding),   # +0.025% POSITIVE
            "AVAXUSDT": (-0.00005, now_ms, next_funding),  # -0.005% NEUTRAL
        }
        return demo_map.get(symbol, (0.00005, now_ms, next_funding))

    def _get_demo_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Generate realistic synthetic OHLCV with clear SMC patterns for 4h, 1h, 15m, 5m, 3m, 1m."""
        now = int(time.time())
        interval_seconds = {
            "4h": 14400,
            "1h": 3600,
            "15m": 900,
            "5m": 300,
            "3m": 180,
            "1m": 60,
        }
        step_seconds = interval_seconds.get(interval.lower(), 900)
        base_price = 142.5 if "SOL" in symbol else (2620.0 if "ETH" in symbol else (18.25 if "LINK" in symbol else 100.0))

        records = []
        is_micro = interval.lower() in ("5m", "3m", "1m")

        for i in range(limit):
            ts = (now - (limit - i) * step_seconds) * 1000

            if is_micro:
                # Lower timeframe micro simulation: simulated sweep and turn near end
                # e.g. bar limit-5 is low, bar limit-4 dips below and closes back inside (sweep),
                # bar limit-3 strong displacement, bar limit-2 breaks swing high
                noise = math.sin((i / max(limit, 1)) * 3 * math.pi) * (base_price * 0.015)
                current_close = base_price + noise

                if i == limit - 4 and "SOL" in symbol:
                    # Bullish sweep candle: sharp wick down then close up
                    low = current_close - (base_price * 0.02)
                    high = current_close + (base_price * 0.003)
                    open_p = current_close - (base_price * 0.005)
                    close_p = current_close
                    volume = 75000.0
                elif i == limit - 3 and "SOL" in symbol:
                    # Strong displacement candle
                    open_p = current_close - (base_price * 0.01)
                    close_p = current_close + (base_price * 0.015)
                    high = close_p + (base_price * 0.002)
                    low = open_p - (base_price * 0.001)
                    current_close = close_p
                    volume = 120000.0
                elif i == limit - 2 and "SOL" in symbol:
                    # MSS break candle
                    open_p = current_close
                    close_p = current_close + (base_price * 0.008)
                    high = close_p + (base_price * 0.003)
                    low = open_p - (base_price * 0.001)
                    current_close = close_p
                    volume = 90000.0
                else:
                    high = current_close + abs(current_close * 0.004)
                    low = current_close - abs(current_close * 0.004)
                    open_p = current_close - (current_close * 0.002 * math.cos(i))
                    close_p = current_close
                    volume = 30000.0 * (1.0 + 0.3 * math.sin(i))
            else:
                # Wave cycle with high volume at recent turn
                angle = (i / max(limit, 1)) * 4 * math.pi
                noise = math.sin(angle) * (base_price * 0.04)
                current_close = base_price + noise

                # For recent candles, simulate high RVOL
                vol_mult = 2.4 if i == limit - 2 else (0.8 + 0.4 * math.sin(i))
                volume = 50000.0 * vol_mult

                high = current_close + abs(current_close * 0.008)
                low = current_close - abs(current_close * 0.008)
                open_p = current_close - (current_close * 0.003 * math.sin(angle * 2))
                close_p = current_close

            records.append({
                "timestamp": ts,
                "open": round(open_p, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close_p, 4),
                "volume": round(volume, 2),
            })

        df = pd.DataFrame(records)
        return df
