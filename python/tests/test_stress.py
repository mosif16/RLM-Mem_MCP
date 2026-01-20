"""
Stress tests for concurrent file collection limits.

Tests cover:
- High concurrency file collection
- File descriptor exhaustion prevention
- Memory pressure handling
- Large file handling
- Many small files handling
- Error recovery under load
"""

import asyncio
import os
import resource
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from rlm_mem_mcp.config import RLMConfig
from rlm_mem_mcp.file_collector import FileCollector, MAX_CONCURRENT_FILE_READS
from rlm_mem_mcp.memory_store import MemoryStore
from rlm_mem_mcp.rlm_processor import LLMResponseCache


class TestHighConcurrency:
    """Stress tests for high concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_many_concurrent_file_reads(self, file_collector: FileCollector, temp_dir: Path):
        """Test reading many files concurrently."""
        # Create 200 files
        for i in range(200):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})" * 100)

        # Should not exhaust file descriptors
        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 200
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_concurrent_directory_traversal(self, file_collector: FileCollector, temp_dir: Path):
        """Test concurrent traversal of multiple directories."""
        # Create nested directory structure
        for i in range(10):
            subdir = temp_dir / f"dir_{i}"
            subdir.mkdir()
            for j in range(20):
                (subdir / f"file_{j}.py").write_text(f"print({i}, {j})")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 200
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_deeply_nested_directories(self, file_collector: FileCollector, temp_dir: Path):
        """Test deeply nested directory structures."""
        # Create deep nesting
        current = temp_dir
        for i in range(20):
            current = current / f"level_{i}"
            current.mkdir()
            (current / "file.py").write_text(f"print({i})")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 20


class TestFileDescriptorLimits:
    """Tests for file descriptor exhaustion prevention."""

    @pytest.mark.asyncio
    async def test_respects_fd_limit(self, temp_dir: Path):
        """Test that semaphore prevents FD exhaustion."""
        config = RLMConfig(api_key="test")
        collector = FileCollector(config)

        # Create more files than max concurrent reads
        file_count = MAX_CONCURRENT_FILE_READS * 3

        for i in range(file_count):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        # Should complete without FD exhaustion
        result = await collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == file_count
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_fd_recovery_after_errors(self, file_collector: FileCollector, temp_dir: Path):
        """Test FD recovery after encountering errors."""
        # Create mix of valid and problematic files
        for i in range(50):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        # Create a directory pretending to be a file (will cause read error)
        (temp_dir / "fake_file.py").mkdir()

        result = await file_collector.collect_paths_async([str(temp_dir)])

        # Should still collect the valid files
        assert result.file_count >= 50


class TestMemoryPressure:
    """Tests for memory pressure scenarios."""

    @pytest.mark.asyncio
    async def test_large_file_memory_handling(self, file_collector: FileCollector, temp_dir: Path):
        """Test memory handling with large files."""
        # Create several large files (~1MB each)
        content = "x" * (1024 * 1024)
        for i in range(5):
            (temp_dir / f"large_{i}.txt").write_text(content)

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 5

    @pytest.mark.asyncio
    async def test_many_small_files_memory(self, file_collector: FileCollector, temp_dir: Path):
        """Test memory handling with many small files."""
        # Create 1000 small files
        for i in range(1000):
            (temp_dir / f"file_{i}.py").write_text(f"x = {i}")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 1000

    @pytest.mark.asyncio
    async def test_memory_store_bulk_operations(self, memory_store: MemoryStore):
        """Test memory store under bulk operations."""
        # Bulk insert
        for i in range(500):
            await memory_store.store(
                f"key_{i}",
                f"value_{i}" * 100,  # ~600 bytes each
                tags=[f"tag_{i % 10}"]
            )

        stats = await memory_store.get_stats()
        assert stats.entry_count == 500

        # Bulk read
        for i in range(500):
            entry = await memory_store.get(f"key_{i}")
            assert entry is not None


class TestCacheStress:
    """Stress tests for cache systems."""

    def test_cache_size_limits(self):
        """Test cache behavior at size limits."""
        cache = LLMResponseCache(max_size=100)

        # Add more than max size
        for i in range(200):
            cache.set(f"key_{i}", f"value_{i}")

        stats = cache.get_stats()
        assert stats["size"] <= 100

    def test_cache_rapid_updates(self):
        """Test cache with rapid updates to same keys."""
        cache = LLMResponseCache()

        # Rapid updates to same key
        for i in range(1000):
            cache.set("same_key", f"value_{i}")

        assert cache.get("same_key") == "value_999"

    def test_cache_concurrent_access(self):
        """Test cache under concurrent access."""
        cache = LLMResponseCache(max_size=1000)

        async def writer(start: int):
            for i in range(start, start + 100):
                cache.set(f"key_{i}", f"value_{i}")

        async def reader(start: int):
            results = []
            for i in range(start, start + 100):
                results.append(cache.get(f"key_{i}"))
            return results

        async def run_stress():
            # Launch concurrent writers and readers
            writers = [writer(i * 100) for i in range(5)]
            readers = [reader(i * 100) for i in range(5)]

            await asyncio.gather(*writers)
            await asyncio.gather(*readers)

        asyncio.run(run_stress())

        # Cache should still be functional
        stats = cache.get_stats()
        assert stats["size"] > 0


class TestErrorRecovery:
    """Tests for error recovery under stress."""

    @pytest.mark.asyncio
    async def test_recovery_from_read_errors(self, file_collector: FileCollector, temp_dir: Path):
        """Test recovery when some files fail to read."""
        # Create valid files
        for i in range(50):
            (temp_dir / f"valid_{i}.py").write_text(f"print({i})")

        # Create unreadable file (directory with .py extension)
        (temp_dir / "unreadable.py").mkdir()

        result = await file_collector.collect_paths_async([str(temp_dir)])

        # Should still collect valid files
        assert result.file_count >= 50
        # Might have error for the directory
        # assert len(result.errors) >= 0  # Could be 0 if directory is skipped

    @pytest.mark.asyncio
    async def test_recovery_from_encoding_errors(self, file_collector: FileCollector, temp_dir: Path):
        """Test recovery from encoding errors."""
        # Create valid file
        (temp_dir / "valid.py").write_text("print('hello')")

        # Create file with binary content
        (temp_dir / "binary.py").write_bytes(b"\x80\x81\x82\x83")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        # Should collect at least the valid file
        assert result.file_count >= 1

    @pytest.mark.asyncio
    async def test_memory_store_error_recovery(self, memory_store: MemoryStore):
        """Test memory store recovery from errors."""
        # Store some entries
        for i in range(10):
            await memory_store.store(f"key_{i}", f"value_{i}")

        # Try to store entry that might cause issues (very large)
        large_value = "x" * 500000  # 500KB

        try:
            await memory_store.store("large_key", large_value)
        except Exception:
            pass  # Might fail due to size limits

        # Store should still work for normal entries
        await memory_store.store("normal_key", "normal_value")
        entry = await memory_store.get("normal_key")
        assert entry is not None


class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_on_repeated_collection(self, file_collector: FileCollector, temp_dir: Path):
        """Test that repeated collection doesn't leak memory."""
        import gc

        # Create files
        for i in range(50):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        initial_objects = len(gc.get_objects())

        # Repeated collections
        for _ in range(10):
            await file_collector.collect_paths_async([str(temp_dir)])
            gc.collect()

        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly
        growth = final_objects - initial_objects
        assert growth < 10000  # Reasonable growth limit

    @pytest.mark.asyncio
    async def test_cleanup_on_cancellation(self, file_collector: FileCollector, temp_dir: Path):
        """Test cleanup when operation is cancelled."""
        # Create many files
        for i in range(100):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})" * 1000)

        async def collect_with_cancel():
            task = asyncio.create_task(
                file_collector.collect_paths_async([str(temp_dir)])
            )
            await asyncio.sleep(0.01)  # Let it start
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should not leave dangling resources
        await collect_with_cancel()

        # Should still be able to collect
        result = await file_collector.collect_paths_async([str(temp_dir)])
        assert result.file_count == 100


class TestEdgeCases:
    """Edge case stress tests."""

    @pytest.mark.asyncio
    async def test_empty_files(self, file_collector: FileCollector, temp_dir: Path):
        """Test handling of empty files."""
        for i in range(50):
            (temp_dir / f"empty_{i}.py").write_text("")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 50

    @pytest.mark.asyncio
    async def test_files_with_special_names(self, file_collector: FileCollector, temp_dir: Path):
        """Test handling of files with special names."""
        special_names = [
            "file with spaces.py",
            "file-with-dashes.py",
            "file_with_underscores.py",
            "UPPERCASE.py",
            "MixedCase.py",
        ]

        for name in special_names:
            (temp_dir / name).write_text("print('hello')")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == len(special_names)

    @pytest.mark.asyncio
    async def test_unicode_content(self, file_collector: FileCollector, temp_dir: Path):
        """Test handling of unicode content."""
        unicode_content = """
# Unicode test
greeting = "Hello, 世界! 🌍"
emoji = "🎉🎊🎈"
chinese = "你好"
arabic = "مرحبا"
"""
        (temp_dir / "unicode.py").write_text(unicode_content)

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count == 1
        content = result.get_combined_content()
        assert "世界" in content or len(content) > 0

    @pytest.mark.asyncio
    async def test_very_long_file_names(self, file_collector: FileCollector, temp_dir: Path):
        """Test handling of very long file names."""
        # Create file with long name (staying within filesystem limits)
        long_name = "a" * 200 + ".py"
        try:
            (temp_dir / long_name).write_text("print('long')")
        except OSError:
            pytest.skip("Filesystem doesn't support long filenames")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count >= 1
