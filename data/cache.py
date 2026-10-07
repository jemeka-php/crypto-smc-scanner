"""
Thread-safe TTL Cache for market data, tickers, OHLCV, and OI observations.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional
import threading


@dataclass
class CacheEntry:
    value: Any
    timestamp: float


class DataCache:
    """
    Lightweight in-memory cache with Time-To-Live (TTL) expiration.
    """

    def __init__(self, default_ttl_seconds: int = 60):
        self.default_ttl = default_ttl_seconds
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str, max_age_seconds: Optional[int] = None) -> Optional[Any]:
        ttl = max_age_seconds if max_age_seconds is not None else self.default_ttl
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if time.time() - entry.timestamp > ttl:
                # Expired
                return None
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = CacheEntry(value=value, timestamp=time.time())

    def get_timestamp(self, key: str) -> Optional[float]:
        with self._lock:
            entry = self._store.get(key)
            return entry.timestamp if entry else None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Global cache instance
GLOBAL_CACHE = DataCache()
