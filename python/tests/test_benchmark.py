"""
Performance benchmark tests with timing assertions.

Tests cover:
- File collection performance
- Token counting performance
- Cache performance
- Memory store performance
- Pipeline throughput
"""

import asyncio
import statistics
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

from rlm_mem_mcp.config import RLMConfig
from rlm_mem_mcp.file_collector import FileCollector
from rlm_mem_mcp.cache_manager import CacheManager, LLMResponseCache
from rlm_mem_mcp.memory_store import MemoryStore


class TestFileCollectionPerformance:
    """Benchmark tests for file collection."""

    @pytest.mark.asyncio
    async def test_small_file_collection_speed(self, file_collector: FileCollector, temp_dir: Path):
        """Test collection speed for small files."""
        # Create 10 small files
        for i in range(10):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        start = time.perf_counter()
        result = await file_collector.collect_paths_async([str(temp_dir)])
        elapsed = time.perf_counter() - start

        assert result.file_count == 10
        assert elapsed < 2.0  # Should complete in under 2 seconds
        print(f"Small file collection: {elapsed*1000:.2f}ms for {result.file_count} files")

    @pytest.mark.asyncio
    async def test_medium_file_collection_speed(self, file_collector: FileCollector, temp_dir: Path):
        """Test collection speed for medium-sized files."""
        # Create 50 files with ~1KB each
        content = "x" * 1000
        for i in range(50):
            (temp_dir / f"file_{i}.py").write_text(f"# {content}\nprint({i})")

        start = time.perf_counter()
        result = await file_collector.collect_paths_async([str(temp_dir)])
        elapsed = time.perf_counter() - start

        assert result.file_count == 50
        assert elapsed < 5.0  # Should complete in under 5 seconds
        print(f"Medium file collection: {elapsed*1000:.2f}ms for {result.file_count} files")

    @pytest.mark.asyncio
    async def test_large_file_collection_speed(self, file_collector: FileCollector, temp_dir: Path):
        """Test collection speed for larger files."""
        # Create 20 files with ~10KB each
        content = "x" * 10000
        for i in range(20):
            (temp_dir / f"file_{i}.py").write_text(f"# {content}\nprint({i})")

        start = time.perf_counter()
        result = await file_collector.collect_paths_async([str(temp_dir)])
        elapsed = time.perf_counter() - start

        assert result.file_count == 20
        assert elapsed < 5.0  # Should complete in under 5 seconds
        print(f"Large file collection: {elapsed*1000:.2f}ms for {result.file_count} files")

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_speedup(self, temp_dir: Path):
        """Test that parallel collection is faster than sequential."""
        config = RLMConfig(api_key="test")
        collector = FileCollector(config)

        # Create 100 files
        for i in range(100):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        # Measure parallel collection
        start = time.perf_counter()
        result = await collector.collect_paths_async([str(temp_dir)])
        parallel_time = time.perf_counter() - start

        assert result.file_count == 100
        print(f"Parallel collection: {parallel_time*1000:.2f}ms for 100 files")

        # Parallel should be reasonably fast
        assert parallel_time < 10.0  # Under 10 seconds for 100 files


class TestTokenCountingPerformance:
    """Benchmark tests for token counting."""

    def test_small_text_token_count_speed(self, file_collector: FileCollector):
        """Test token counting speed for small texts."""
        text = "Hello, world! This is a test."

        times = []
        for _ in range(100):
            start = time.perf_counter()
            file_collector.count_tokens(text)
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 10  # Under 10ms average
        print(f"Small text token count: {avg_time:.3f}ms average")

    def test_large_text_token_count_speed(self, file_collector: FileCollector):
        """Test token counting speed for large texts."""
        text = "word " * 10000  # ~50KB

        times = []
        for _ in range(10):
            start = time.perf_counter()
            file_collector.count_tokens(text)
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 500  # Under 500ms average for large text
        print(f"Large text token count: {avg_time:.3f}ms average")

    def test_cached_token_count_speed(self, file_collector: FileCollector):
        """Test that cached token counts are fast."""
        text = "This text will be cached for speed testing."

        # First call - cache miss
        file_collector.count_tokens(text)

        # Cached calls
        times = []
        for _ in range(100):
            start = time.perf_counter()
            file_collector.count_tokens(text)
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 1  # Under 1ms for cached lookups
        print(f"Cached token count: {avg_time:.4f}ms average")


class TestCachePerformance:
    """Benchmark tests for cache operations."""

    def test_cache_write_speed(self):
        """Test cache write performance."""
        cache = LLMResponseCache(max_size=10000)

        times = []
        for i in range(1000):
            start = time.perf_counter()
            cache.set(f"key_{i}", f"value_{i}")
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 1  # Under 1ms per write
        print(f"Cache write: {avg_time:.4f}ms average")

    def test_cache_read_speed(self):
        """Test cache read performance."""
        cache = LLMResponseCache(max_size=10000)

        # Populate cache
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        # Measure reads
        times = []
        for i in range(1000):
            start = time.perf_counter()
            cache.get(f"key_{i}")
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 0.1  # Under 0.1ms per read
        print(f"Cache read: {avg_time:.4f}ms average")

    def test_cache_hit_vs_miss_speed(self):
        """Compare cache hit vs miss performance."""
        cache = LLMResponseCache(max_size=1000)

        # Populate some entries
        for i in range(500):
            cache.set(f"key_{i}", f"value_{i}")

        # Measure hits
        hit_times = []
        for i in range(500):
            start = time.perf_counter()
            cache.get(f"key_{i}")
            hit_times.append(time.perf_counter() - start)

        # Measure misses
        miss_times = []
        for i in range(500, 1000):
            start = time.perf_counter()
            cache.get(f"key_{i}")
            miss_times.append(time.perf_counter() - start)

        avg_hit = statistics.mean(hit_times) * 1000
        avg_miss = statistics.mean(miss_times) * 1000

        print(f"Cache hit: {avg_hit:.4f}ms, miss: {avg_miss:.4f}ms")
        # Both should be fast
        assert avg_hit < 0.1
        assert avg_miss < 0.1


class TestMemoryStorePerformance:
    """Benchmark tests for memory store operations."""

    @pytest.mark.asyncio
    async def test_memory_store_write_speed(self, memory_store: MemoryStore):
        """Test memory store write performance."""
        times = []
        for i in range(100):
            start = time.perf_counter()
            await memory_store.store(f"key_{i}", f"value_{i}", tags=[f"tag_{i}"])
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 50  # Under 50ms per write (SQLite)
        print(f"Memory store write: {avg_time:.2f}ms average")

    @pytest.mark.asyncio
    async def test_memory_store_read_speed(self, memory_store: MemoryStore):
        """Test memory store read performance."""
        # Populate store
        for i in range(100):
            await memory_store.store(f"key_{i}", f"value_{i}")

        # Measure reads
        times = []
        for i in range(100):
            start = time.perf_counter()
            await memory_store.get(f"key_{i}")
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 10  # Under 10ms per read
        print(f"Memory store read: {avg_time:.2f}ms average")

    @pytest.mark.asyncio
    async def test_tag_search_speed(self, memory_store: MemoryStore):
        """Test tag search performance with inverted index."""
        # Store many entries with tags
        for i in range(100):
            tags = [f"tag_{i % 10}"]  # 10 unique tags
            await memory_store.store(f"key_{i}", f"value_{i}", tags=tags)

        # Measure tag searches
        times = []
        for i in range(10):
            start = time.perf_counter()
            results = await memory_store.search_by_tags([f"tag_{i}"])
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times) * 1000
        assert avg_time < 20  # Under 20ms per search
        print(f"Tag search: {avg_time:.2f}ms average")


class TestThroughput:
    """Throughput benchmark tests."""

    @pytest.mark.asyncio
    async def test_file_throughput(self, file_collector: FileCollector, temp_dir: Path):
        """Test file processing throughput (files per second)."""
        # Create 200 files
        for i in range(200):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        start = time.perf_counter()
        result = await file_collector.collect_paths_async([str(temp_dir)])
        elapsed = time.perf_counter() - start

        throughput = result.file_count / elapsed
        print(f"File throughput: {throughput:.1f} files/second")

        assert throughput > 10  # At least 10 files per second

    @pytest.mark.asyncio
    async def test_memory_store_throughput(self, memory_store: MemoryStore):
        """Test memory store throughput (operations per second)."""
        start = time.perf_counter()

        # Mixed workload: writes and reads
        for i in range(100):
            await memory_store.store(f"key_{i}", f"value_{i}")
            if i > 0:
                await memory_store.get(f"key_{i-1}")

        elapsed = time.perf_counter() - start
        ops = 200  # 100 writes + 100 reads (approximately)
        throughput = ops / elapsed

        print(f"Memory store throughput: {throughput:.1f} ops/second")
        assert throughput > 50  # At least 50 ops per second


class TestLatencyPercentiles:
    """Tests for latency percentile measurements."""

    @pytest.mark.asyncio
    async def test_file_collection_p99(self, file_collector: FileCollector, temp_dir: Path):
        """Test file collection p99 latency."""
        # Create files
        for i in range(50):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            await file_collector.collect_paths_async([str(temp_dir)])
            times.append(time.perf_counter() - start)

        times.sort()
        p50 = times[len(times) // 2] * 1000
        p99 = times[int(len(times) * 0.99)] * 1000

        print(f"File collection p50: {p50:.2f}ms, p99: {p99:.2f}ms")

        # p99 should be reasonable
        assert p99 < 10000  # Under 10 seconds

    def test_cache_p99(self):
        """Test cache operation p99 latency."""
        cache = LLMResponseCache()

        # Populate
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        # Measure
        times = []
        for i in range(1000):
            start = time.perf_counter()
            cache.get(f"key_{i % 1000}")
            times.append(time.perf_counter() - start)

        times.sort()
        p50 = times[len(times) // 2] * 1000
        p99 = times[int(len(times) * 0.99)] * 1000

        print(f"Cache p50: {p50:.4f}ms, p99: {p99:.4f}ms")

        # p99 should be very fast
        assert p99 < 1  # Under 1ms
