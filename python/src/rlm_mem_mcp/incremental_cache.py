"""
Incremental Cache for RLM Processing

Implements file hash-based caching to skip re-analyzing unchanged files.
This dramatically speeds up repeated analyses on the same codebase.

Features:
- Content hash tracking (SHA-256)
- Modification time tracking (mtime)
- Analysis result caching
- LRU eviction for memory management
- Persistence to disk (optional)
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from collections import OrderedDict


@dataclass
class FileFingerprint:
    """Fingerprint of a file for change detection."""
    path: str
    content_hash: str  # SHA-256 of content
    mtime: float  # Last modification time
    size: int  # File size in bytes

    def matches(self, other: 'FileFingerprint') -> bool:
        """Check if this fingerprint matches another."""
        # Content hash is the authoritative check
        return self.content_hash == other.content_hash


@dataclass
class CachedAnalysis:
    """Cached analysis result for a file."""
    file_path: str
    fingerprint: FileFingerprint
    query: str  # The query that was run
    result: str  # The analysis result
    timestamp: float  # When the analysis was cached
    confidence: float = 1.0  # Confidence in the result


@dataclass
class IncrementalCacheStats:
    """Statistics for the incremental cache."""
    total_files: int = 0
    cached_files: int = 0  # Files with valid cache entries
    changed_files: int = 0  # Files that changed since last cache
    new_files: int = 0  # Files not in cache
    cache_hits: int = 0  # Reused from cache
    cache_misses: int = 0  # Had to re-analyze
    time_saved_estimate: float = 0.0  # Estimated seconds saved

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "cached_files": self.cached_files,
            "changed_files": self.changed_files,
            "new_files": self.new_files,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "time_saved_estimate": f"{self.time_saved_estimate:.1f}s",
        }


class IncrementalCache:
    """
    File hash-based incremental cache for RLM analysis.

    Tracks file fingerprints and caches analysis results to avoid
    re-analyzing unchanged files on repeated queries.
    """

    # Average time per file analysis (for time saved estimates)
    AVG_ANALYSIS_TIME_SECONDS = 0.5

    def __init__(
        self,
        max_entries: int = 5000,
        cache_file: str | None = None,
        auto_persist: bool = True
    ):
        """
        Initialize the incremental cache.

        Args:
            max_entries: Maximum number of file entries to cache
            cache_file: Path to persist cache to disk (optional)
            auto_persist: Whether to auto-save on changes
        """
        self.max_entries = max_entries
        self.cache_file = cache_file
        self.auto_persist = auto_persist

        # OrderedDict for LRU eviction
        self._fingerprints: OrderedDict[str, FileFingerprint] = OrderedDict()

        # Query-specific analysis cache: (file_path, query_hash) -> CachedAnalysis
        self._analysis_cache: OrderedDict[tuple[str, str], CachedAnalysis] = OrderedDict()

        # Stats
        self._stats = IncrementalCacheStats()

        # Load from disk if available
        if cache_file and os.path.exists(cache_file):
            self._load_from_disk()

    def compute_file_fingerprint(self, path: str, content: str) -> FileFingerprint:
        """
        Compute a fingerprint for a file.

        Args:
            path: The file path
            content: The file content

        Returns:
            FileFingerprint for the file
        """
        # Use SHA-256 for content hash
        content_hash = hashlib.sha256(
            content.encode('utf-8', errors='replace')
        ).hexdigest()

        # Get mtime if file exists on disk
        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            # Virtual file or not accessible
            mtime = time.time()
            size = len(content.encode('utf-8', errors='replace'))

        return FileFingerprint(
            path=path,
            content_hash=content_hash,
            mtime=mtime,
            size=size
        )

    def _query_hash(self, query: str) -> str:
        """Create a hash for a query string."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def has_changed(self, path: str, content: str) -> bool:
        """
        Check if a file has changed since last analysis.

        Args:
            path: The file path
            content: Current file content

        Returns:
            True if file is new or has changed
        """
        new_fingerprint = self.compute_file_fingerprint(path, content)

        if path not in self._fingerprints:
            return True

        old_fingerprint = self._fingerprints[path]
        return not old_fingerprint.matches(new_fingerprint)

    def get_cached_analysis(
        self,
        path: str,
        content: str,
        query: str
    ) -> CachedAnalysis | None:
        """
        Get cached analysis for a file if still valid.

        Args:
            path: The file path
            content: Current file content
            query: The analysis query

        Returns:
            CachedAnalysis if cache hit, None if miss
        """
        cache_key = (path, self._query_hash(query))

        if cache_key not in self._analysis_cache:
            self._stats.cache_misses += 1
            return None

        cached = self._analysis_cache[cache_key]

        # Check if file has changed
        new_fingerprint = self.compute_file_fingerprint(path, content)
        if not cached.fingerprint.matches(new_fingerprint):
            # File changed, invalidate cache
            del self._analysis_cache[cache_key]
            self._stats.cache_misses += 1
            self._stats.changed_files += 1
            return None

        # Cache hit - move to end for LRU
        self._analysis_cache.move_to_end(cache_key)
        self._stats.cache_hits += 1
        self._stats.time_saved_estimate += self.AVG_ANALYSIS_TIME_SECONDS

        return cached

    def cache_analysis(
        self,
        path: str,
        content: str,
        query: str,
        result: str,
        confidence: float = 1.0
    ) -> None:
        """
        Cache an analysis result.

        Args:
            path: The file path
            content: The file content
            query: The analysis query
            result: The analysis result
            confidence: Confidence in the result (0.0-1.0)
        """
        fingerprint = self.compute_file_fingerprint(path, content)
        cache_key = (path, self._query_hash(query))

        cached = CachedAnalysis(
            file_path=path,
            fingerprint=fingerprint,
            query=query,
            result=result,
            timestamp=time.time(),
            confidence=confidence
        )

        self._analysis_cache[cache_key] = cached
        self._fingerprints[path] = fingerprint

        # LRU eviction if needed
        while len(self._analysis_cache) > self.max_entries:
            self._analysis_cache.popitem(last=False)

        while len(self._fingerprints) > self.max_entries:
            self._fingerprints.popitem(last=False)

        # Auto-persist
        if self.auto_persist and self.cache_file:
            self._save_to_disk()

    def update_fingerprint(self, path: str, content: str) -> None:
        """
        Update the fingerprint for a file without caching analysis.

        Useful for tracking file changes without running analysis.
        """
        fingerprint = self.compute_file_fingerprint(path, content)
        self._fingerprints[path] = fingerprint

        # Move to end for LRU
        self._fingerprints.move_to_end(path)

    def get_changed_files(
        self,
        files: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """
        Partition files into changed and unchanged.

        Args:
            files: List of (path, content) tuples

        Returns:
            (changed_files, unchanged_files) where each is list of (path, content)
        """
        changed = []
        unchanged = []

        self._stats.total_files = len(files)

        for path, content in files:
            if self.has_changed(path, content):
                changed.append((path, content))
            else:
                unchanged.append((path, content))

        self._stats.changed_files = len(changed)
        self._stats.cached_files = len(unchanged)

        return changed, unchanged

    def invalidate(self, path: str | None = None) -> int:
        """
        Invalidate cache entries.

        Args:
            path: Specific path to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        if path is None:
            count = len(self._analysis_cache)
            self._analysis_cache.clear()
            self._fingerprints.clear()
            return count

        count = 0

        # Remove fingerprint
        if path in self._fingerprints:
            del self._fingerprints[path]
            count += 1

        # Remove all analysis entries for this path
        keys_to_remove = [k for k in self._analysis_cache.keys() if k[0] == path]
        for key in keys_to_remove:
            del self._analysis_cache[key]
            count += 1

        return count

    def get_stats(self) -> IncrementalCacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = IncrementalCacheStats()

    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.cache_file:
            return

        try:
            data = {
                "fingerprints": {
                    path: asdict(fp)
                    for path, fp in self._fingerprints.items()
                },
                "analysis_cache": {
                    f"{k[0]}|{k[1]}": {
                        "file_path": v.file_path,
                        "fingerprint": asdict(v.fingerprint),
                        "query": v.query,
                        "result": v.result,
                        "timestamp": v.timestamp,
                        "confidence": v.confidence,
                    }
                    for k, v in self._analysis_cache.items()
                },
                "version": 1,
            }

            # Write atomically
            tmp_file = f"{self.cache_file}.tmp"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp_file, self.cache_file)

        except Exception as e:
            # Log but don't fail on cache errors
            import sys
            print(f"Warning: Failed to save incremental cache: {e}", file=sys.stderr)

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.cache_file:
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get("version") != 1:
                return  # Incompatible version

            # Load fingerprints
            for path, fp_data in data.get("fingerprints", {}).items():
                self._fingerprints[path] = FileFingerprint(**fp_data)

            # Load analysis cache
            for key_str, cached_data in data.get("analysis_cache", {}).items():
                parts = key_str.split("|", 1)
                if len(parts) == 2:
                    path, query_hash = parts
                    fingerprint = FileFingerprint(**cached_data["fingerprint"])
                    cached = CachedAnalysis(
                        file_path=cached_data["file_path"],
                        fingerprint=fingerprint,
                        query=cached_data["query"],
                        result=cached_data["result"],
                        timestamp=cached_data["timestamp"],
                        confidence=cached_data.get("confidence", 1.0),
                    )
                    self._analysis_cache[(path, query_hash)] = cached

        except Exception as e:
            # Log but don't fail on cache errors
            import sys
            print(f"Warning: Failed to load incremental cache: {e}", file=sys.stderr)

    def get_summary(self) -> str:
        """Get a human-readable cache summary."""
        stats = self._stats
        return (
            f"Incremental Cache: {len(self._fingerprints)} files tracked, "
            f"{len(self._analysis_cache)} analyses cached\n"
            f"Hit rate: {stats.hit_rate:.1%} "
            f"({stats.cache_hits} hits, {stats.cache_misses} misses)\n"
            f"Estimated time saved: {stats.time_saved_estimate:.1f}s"
        )


# Global cache instance (singleton pattern for MCP server)
_global_cache: IncrementalCache | None = None


def get_incremental_cache(
    cache_file: str | None = None,
    reset: bool = False
) -> IncrementalCache:
    """
    Get the global incremental cache instance.

    Args:
        cache_file: Path to persist cache (only used on first call)
        reset: If True, create a new cache instance

    Returns:
        The global IncrementalCache instance
    """
    global _global_cache

    if _global_cache is None or reset:
        _global_cache = IncrementalCache(cache_file=cache_file)

    return _global_cache
