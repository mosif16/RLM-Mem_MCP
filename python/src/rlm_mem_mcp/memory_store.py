"""
Persistent Memory Store for RLM-Mem MCP Server

Implements SQLite-backed persistent storage for memory entries with:
- Async operations via aiosqlite
- Write-ahead logging (WAL) for durability
- Inverted tag index for O(1) lookups
- Memory usage monitoring
- Automatic schema migration
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

import aiosqlite


# Default database path
DEFAULT_DB_PATH = Path.home() / ".rlm-mem" / "memory.db"

# Memory limits
MAX_MEMORY_ENTRIES = 10000
MAX_ENTRY_SIZE_BYTES = 1_000_000  # 1MB per entry
MAX_TOTAL_SIZE_BYTES = 100_000_000  # 100MB total


@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    value: str
    tags: list[str]
    token_count: int
    stored_at: float
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "tags": self.tags,
            "token_count": self.token_count,
            "stored_at": self.stored_at,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            tags=data.get("tags", []),
            token_count=data.get("token_count", 0),
            stored_at=data.get("stored_at", time.time()),
            size_bytes=data.get("size_bytes", len(data["value"].encode("utf-8"))),
        )


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    entry_count: int = 0
    total_size_bytes: int = 0
    total_tokens: int = 0
    tag_count: int = 0
    oldest_entry_at: float = 0
    newest_entry_at: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "total_size_bytes": self.total_size_bytes,
            "total_tokens": self.total_tokens,
            "tag_count": self.tag_count,
            "oldest_entry_at": self.oldest_entry_at,
            "newest_entry_at": self.newest_entry_at,
            "size_limit_bytes": MAX_TOTAL_SIZE_BYTES,
            "entry_limit": MAX_MEMORY_ENTRIES,
        }


class MemoryStore:
    """
    Persistent memory store with SQLite backend.

    Features:
    - Async operations with aiosqlite
    - WAL mode for durability and performance
    - Inverted tag index for fast lookups
    - Memory usage monitoring and limits
    - Graceful degradation on errors
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._initialized = False

        # In-memory tag index for O(1) lookups
        self._tag_index: dict[str, set[str]] = defaultdict(set)

        # Stats cache
        self._stats_cache: MemoryStats | None = None
        self._stats_cache_time: float = 0
        self._stats_cache_ttl: float = 5.0  # 5 seconds

    async def initialize(self) -> None:
        """Initialize the database connection and schema."""
        if self._initialized:
            return

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._db = await aiosqlite.connect(str(self.db_path))

        # Enable WAL mode for durability and concurrent reads
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=10000")
        await self._db.execute("PRAGMA temp_store=MEMORY")

        # Create schema
        await self._create_schema()

        # Load tag index into memory
        await self._load_tag_index()

        self._initialized = True

    async def _create_schema(self) -> None:
        """Create database schema if not exists."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                token_count INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                stored_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_stored_at ON memories(stored_at);

            CREATE TABLE IF NOT EXISTS tags (
                tag TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                PRIMARY KEY (tag, memory_key),
                FOREIGN KEY (memory_key) REFERENCES memories(key) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
        """)
        await self._db.commit()

        # Set schema version
        async with self._db.execute("SELECT version FROM schema_version") as cursor:
            row = await cursor.fetchone()
            if row is None:
                await self._db.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,)
                )
                await self._db.commit()

    async def _load_tag_index(self) -> None:
        """Load tag index into memory for O(1) lookups."""
        self._tag_index.clear()

        async with self._db.execute("SELECT tag, memory_key FROM tags") as cursor:
            async for row in cursor:
                tag, key = row
                self._tag_index[tag].add(key)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def store(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        token_count: int = 0
    ) -> MemoryEntry:
        """
        Store a memory entry.

        Args:
            key: Unique identifier
            value: Content to store
            tags: Optional list of tags
            token_count: Pre-computed token count

        Returns:
            The stored MemoryEntry

        Raises:
            ValueError: If entry exceeds size limits
        """
        await self.initialize()

        tags = tags or []
        size_bytes = len(value.encode("utf-8"))
        stored_at = time.time()

        # Check size limits
        if size_bytes > MAX_ENTRY_SIZE_BYTES:
            raise ValueError(
                f"Entry size {size_bytes:,} bytes exceeds limit of {MAX_ENTRY_SIZE_BYTES:,} bytes"
            )

        # Check if we need to evict old entries
        stats = await self.get_stats()
        if stats.entry_count >= MAX_MEMORY_ENTRIES:
            await self._evict_oldest(count=100)  # Evict 100 oldest

        if stats.total_size_bytes + size_bytes > MAX_TOTAL_SIZE_BYTES:
            await self._evict_by_size(target_bytes=size_bytes * 2)

        # Remove old tags if key exists
        await self._remove_tags_for_key(key)

        # Store the entry
        await self._db.execute(
            """
            INSERT OR REPLACE INTO memories (key, value, tags, token_count, size_bytes, stored_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, value, json.dumps(tags), token_count, size_bytes, stored_at)
        )

        # Store tags
        for tag in tags:
            await self._db.execute(
                "INSERT OR IGNORE INTO tags (tag, memory_key) VALUES (?, ?)",
                (tag, key)
            )
            self._tag_index[tag].add(key)

        await self._db.commit()

        # Invalidate stats cache
        self._stats_cache = None

        return MemoryEntry(
            key=key,
            value=value,
            tags=tags,
            token_count=token_count,
            stored_at=stored_at,
            size_bytes=size_bytes,
        )

    async def _remove_tags_for_key(self, key: str) -> None:
        """Remove all tags for a key from index and database."""
        # Remove from in-memory index
        for tag, keys in self._tag_index.items():
            keys.discard(key)

        # Remove from database
        await self._db.execute("DELETE FROM tags WHERE memory_key = ?", (key,))

    async def _evict_oldest(self, count: int) -> int:
        """Evict the oldest entries."""
        async with self._db.execute(
            "SELECT key FROM memories ORDER BY stored_at ASC LIMIT ?",
            (count,)
        ) as cursor:
            keys = [row[0] async for row in cursor]

        for key in keys:
            await self.delete(key)

        return len(keys)

    async def _evict_by_size(self, target_bytes: int) -> int:
        """Evict oldest entries until target_bytes are freed."""
        freed = 0
        evicted = 0

        async with self._db.execute(
            "SELECT key, size_bytes FROM memories ORDER BY stored_at ASC"
        ) as cursor:
            async for row in cursor:
                if freed >= target_bytes:
                    break
                key, size = row
                await self.delete(key)
                freed += size
                evicted += 1

        return evicted

    async def get(self, key: str) -> MemoryEntry | None:
        """Get a memory entry by key."""
        await self.initialize()

        async with self._db.execute(
            "SELECT key, value, tags, token_count, size_bytes, stored_at FROM memories WHERE key = ?",
            (key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None

            return MemoryEntry(
                key=row[0],
                value=row[1],
                tags=json.loads(row[2]),
                token_count=row[3],
                size_bytes=row[4],
                stored_at=row[5],
            )

    async def search_by_tags(self, tags: list[str]) -> list[MemoryEntry]:
        """
        Search for entries by tags using inverted index.

        Args:
            tags: List of tags to search for (OR logic)

        Returns:
            List of matching MemoryEntry objects
        """
        await self.initialize()

        # Use in-memory index for O(1) lookup
        matching_keys: set[str] = set()
        for tag in tags:
            if tag in self._tag_index:
                matching_keys.update(self._tag_index[tag])

        if not matching_keys:
            return []

        # Fetch full entries
        placeholders = ",".join("?" * len(matching_keys))
        entries = []

        async with self._db.execute(
            f"SELECT key, value, tags, token_count, size_bytes, stored_at FROM memories WHERE key IN ({placeholders})",
            tuple(matching_keys)
        ) as cursor:
            async for row in cursor:
                entries.append(MemoryEntry(
                    key=row[0],
                    value=row[1],
                    tags=json.loads(row[2]),
                    token_count=row[3],
                    size_bytes=row[4],
                    stored_at=row[5],
                ))

        return entries

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[MemoryEntry]:
        """List all memory entries with pagination."""
        await self.initialize()

        entries = []
        async with self._db.execute(
            "SELECT key, value, tags, token_count, size_bytes, stored_at FROM memories ORDER BY stored_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ) as cursor:
            async for row in cursor:
                entries.append(MemoryEntry(
                    key=row[0],
                    value=row[1],
                    tags=json.loads(row[2]),
                    token_count=row[3],
                    size_bytes=row[4],
                    stored_at=row[5],
                ))

        return entries

    async def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        await self.initialize()

        # Remove from tag index
        await self._remove_tags_for_key(key)

        # Delete from database
        cursor = await self._db.execute("DELETE FROM memories WHERE key = ?", (key,))
        await self._db.commit()

        # Invalidate stats cache
        self._stats_cache = None

        return cursor.rowcount > 0

    async def clear(self) -> int:
        """Clear all memory entries."""
        await self.initialize()

        cursor = await self._db.execute("DELETE FROM memories")
        await self._db.execute("DELETE FROM tags")
        await self._db.commit()

        self._tag_index.clear()
        self._stats_cache = None

        return cursor.rowcount

    async def get_stats(self) -> MemoryStats:
        """Get memory usage statistics (cached)."""
        await self.initialize()

        # Return cached stats if fresh
        if self._stats_cache and (time.time() - self._stats_cache_time) < self._stats_cache_ttl:
            return self._stats_cache

        stats = MemoryStats()

        # Get counts and totals
        async with self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), COALESCE(SUM(token_count), 0), MIN(stored_at), MAX(stored_at) FROM memories"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                stats.entry_count = row[0]
                stats.total_size_bytes = row[1]
                stats.total_tokens = row[2]
                stats.oldest_entry_at = row[3] or 0
                stats.newest_entry_at = row[4] or 0

        # Get unique tag count
        async with self._db.execute("SELECT COUNT(DISTINCT tag) FROM tags") as cursor:
            row = await cursor.fetchone()
            if row:
                stats.tag_count = row[0]

        # Cache the stats
        self._stats_cache = stats
        self._stats_cache_time = time.time()

        return stats

    async def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        await self.initialize()
        await self._db.execute("VACUUM")

    async def checkpoint(self) -> None:
        """Force a WAL checkpoint."""
        await self.initialize()
        await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


# Global instance for server use
_global_store: MemoryStore | None = None


async def get_memory_store(db_path: Path | str | None = None) -> MemoryStore:
    """Get or create the global memory store instance."""
    global _global_store

    if _global_store is None:
        _global_store = MemoryStore(db_path)
        await _global_store.initialize()

    return _global_store


async def close_memory_store() -> None:
    """Close the global memory store."""
    global _global_store

    if _global_store:
        await _global_store.close()
        _global_store = None
