"""
Caching system for The Earnings Hunter.

Provides file-based caching for analysis results to avoid redundant API calls.
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisCache:
    """
    File-based cache for analysis results.

    Caches analysis results with configurable expiry time.
    Each cache entry is stored as a JSON file with timestamp metadata.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        expiry_hours: int = 24
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            expiry_hours: Hours until cache entries expire
        """
        self.cache_dir = Path(cache_dir)
        self.expiry_hours = expiry_hours
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, symbol: str, date: Optional[str] = None) -> str:
        """
        Generate cache key from symbol and date.

        Args:
            symbol: Stock ticker symbol
            date: Optional date string (defaults to today)

        Returns:
            Cache key string (e.g., "NVDA_2026-01-28")
        """
        symbol = symbol.upper().strip()
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return f"{symbol}_{date}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, symbol: str, date: Optional[str] = None) -> Optional[dict]:
        """
        Get cached analysis if it exists and is not expired.

        Args:
            symbol: Stock ticker symbol
            date: Optional date string

        Returns:
            Cached data dict or None if not found/expired
        """
        cache_key = self._get_cache_key(symbol, date)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss for {cache_key}")
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            # Check expiry
            cached_time = datetime.fromisoformat(cached["cached_at"])
            if datetime.now() - cached_time > timedelta(hours=self.expiry_hours):
                logger.debug(f"Cache expired for {cache_key}")
                cache_path.unlink()  # Delete expired cache
                return None

            logger.info(f"Cache hit for {cache_key}")
            return cached["data"]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache file for {cache_key}: {e}")
            cache_path.unlink()  # Delete invalid cache
            return None

    def set(self, symbol: str, data: dict, date: Optional[str] = None) -> None:
        """
        Save analysis to cache.

        Args:
            symbol: Stock ticker symbol
            data: Analysis data to cache
            date: Optional date string
        """
        cache_key = self._get_cache_key(symbol, date)
        cache_path = self._get_cache_path(cache_key)

        cached = {
            "cached_at": datetime.now().isoformat(),
            "symbol": symbol.upper(),
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "data": data
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2, default=str)
            logger.info(f"Cached analysis for {cache_key}")
        except Exception as e:
            logger.error(f"Failed to cache {cache_key}: {e}")

    def clear(self, symbol: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            symbol: Optional symbol to clear (clears all if None)

        Returns:
            Number of entries cleared
        """
        count = 0

        if symbol:
            # Clear specific symbol
            pattern = f"{symbol.upper()}_*.json"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cache entries for {symbol}")
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared all {count} cache entries")

        return count

    def clear_expired(self) -> int:
        """
        Clear only expired cache entries.

        Returns:
            Number of expired entries cleared
        """
        count = 0
        cutoff = datetime.now() - timedelta(hours=self.expiry_hours)

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached["cached_at"])
                if cached_time < cutoff:
                    cache_file.unlink()
                    count += 1
            except Exception:
                # Delete invalid cache files too
                cache_file.unlink()
                count += 1

        logger.info(f"Cleared {count} expired cache entries")
        return count

    def list_entries(self) -> list[dict]:
        """
        List all cache entries with metadata.

        Returns:
            List of cache entry metadata dicts
        """
        entries = []

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                entries.append({
                    "key": cache_file.stem,
                    "symbol": cached.get("symbol"),
                    "date": cached.get("date"),
                    "cached_at": cached.get("cached_at"),
                    "file_size": cache_file.stat().st_size
                })
            except Exception:
                pass

        return sorted(entries, key=lambda x: x.get("cached_at", ""), reverse=True)

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (count, size, oldest, newest)
        """
        entries = self.list_entries()

        if not entries:
            return {
                "count": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None
            }

        total_size = sum(e["file_size"] for e in entries)

        return {
            "count": len(entries),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest": entries[-1]["cached_at"] if entries else None,
            "newest": entries[0]["cached_at"] if entries else None
        }
