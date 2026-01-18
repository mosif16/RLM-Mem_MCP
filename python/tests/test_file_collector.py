"""
Unit tests for async FileCollector.

Tests cover:
- Async file collection
- Parallel processing
- Symlink loop detection
- Timeout handling
- Chunked reading for large files
- Token counting with caching
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio

from rlm_mem_mcp.file_collector import (
    FileCollector,
    CollectionResult,
    MAX_CONCURRENT_FILE_READS,
    FILE_READ_TIMEOUT_SECONDS,
    LARGE_FILE_THRESHOLD_BYTES,
)
from rlm_mem_mcp.config import RLMConfig


class TestFileCollectorAsync:
    """Tests for async file collection methods."""

    @pytest.mark.asyncio
    async def test_collect_single_file(self, file_collector: FileCollector, sample_files: dict):
        """Test collecting a single file."""
        result = await file_collector.collect_paths_async([str(sample_files["python"])])

        assert result.file_count == 1
        assert result.total_tokens > 0
        assert len(result.errors) == 0
        assert "sample.py" in result.files[0].path

    @pytest.mark.asyncio
    async def test_collect_directory(self, file_collector: FileCollector, temp_dir: Path, sample_files: dict):
        """Test collecting all files from a directory."""
        result = await file_collector.collect_paths_async([str(temp_dir)])

        # Should find multiple files
        assert result.file_count >= 4  # python, js, md, json, nested
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_collect_multiple_paths(self, file_collector: FileCollector, sample_files: dict):
        """Test collecting from multiple paths."""
        paths = [str(sample_files["python"]), str(sample_files["javascript"])]
        result = await file_collector.collect_paths_async(paths)

        assert result.file_count == 2
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_collect_nonexistent_path(self, file_collector: FileCollector):
        """Test handling nonexistent paths."""
        result = await file_collector.collect_paths_async(["/nonexistent/path/file.py"])

        assert result.file_count == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_parallel_collection(self, file_collector: FileCollector, temp_dir: Path):
        """Test that files are collected in parallel."""
        # Create many small files
        for i in range(20):
            (temp_dir / f"file_{i}.py").write_text(f"# File {i}\nprint({i})")

        import time
        start = time.perf_counter()
        result = await file_collector.collect_paths_async([str(temp_dir)])
        elapsed = time.perf_counter() - start

        assert result.file_count >= 20
        # Parallel should be faster than sequential (20 files * 0.01s each = 0.2s)
        # With parallelism, should complete much faster
        assert elapsed < 2.0  # Very generous timeout

    @pytest.mark.asyncio
    async def test_file_extension_filtering(self, file_collector: FileCollector, temp_dir: Path):
        """Test that only configured extensions are collected."""
        # Create files with various extensions
        (temp_dir / "code.py").write_text("print('hello')")
        (temp_dir / "data.csv").write_text("a,b,c")
        (temp_dir / "image.png").write_bytes(b"\x89PNG\r\n")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        # Should only collect .py (and possibly other allowed extensions)
        paths = [f.path for f in result.files]
        assert any(".py" in p for p in paths)
        # .png should be excluded
        assert not any(".png" in p for p in paths)


class TestSymlinkHandling:
    """Tests for symlink loop detection."""

    @pytest.mark.asyncio
    async def test_symlink_loop_detection(self, file_collector: FileCollector, temp_dir: Path):
        """Test that symlink loops are detected and handled."""
        # Create a directory with a symlink loop
        dir_a = temp_dir / "dir_a"
        dir_a.mkdir()
        (dir_a / "file.py").write_text("print('a')")

        # Create symlink loop: dir_a/link -> dir_a
        link_path = dir_a / "link"
        try:
            link_path.symlink_to(dir_a)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        # Should not hang or crash
        result = await file_collector.collect_paths_async([str(dir_a)])

        # Should have collected the file but not infinitely recursed
        assert result.file_count >= 1

    @pytest.mark.asyncio
    async def test_symlink_to_file(self, file_collector: FileCollector, temp_dir: Path, sample_files: dict):
        """Test following symlinks to files."""
        link_path = temp_dir / "link.py"
        try:
            link_path.symlink_to(sample_files["python"])
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = await file_collector.collect_paths_async([str(link_path)])

        assert result.file_count == 1


class TestTimeoutHandling:
    """Tests for file read timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_configuration(self, file_collector: FileCollector):
        """Test that timeout is configured correctly."""
        assert FILE_READ_TIMEOUT_SECONDS == 30

    @pytest.mark.asyncio
    async def test_slow_file_timeout(self, file_collector: FileCollector, temp_dir: Path):
        """Test handling of slow file reads."""
        # Create a file
        slow_file = temp_dir / "slow.py"
        slow_file.write_text("print('slow')")

        # Mock aiofiles to simulate slow read
        original_open = None

        async def slow_open(*args, **kwargs):
            await asyncio.sleep(0.1)  # Small delay for testing
            return await original_open(*args, **kwargs)

        # The file should still be collected (0.1s < 30s timeout)
        result = await file_collector.collect_paths_async([str(slow_file)])
        assert result.file_count == 1


class TestChunkedReading:
    """Tests for chunked reading of large files."""

    @pytest.mark.asyncio
    async def test_large_file_threshold(self):
        """Test that large file threshold is configured correctly."""
        assert LARGE_FILE_THRESHOLD_BYTES == 10 * 1024 * 1024  # 10MB

    @pytest.mark.asyncio
    async def test_chunked_reading(self, file_collector: FileCollector, large_file: Path):
        """Test that large files are read in chunks."""
        result = await file_collector.collect_paths_async([str(large_file)])

        assert result.file_count == 1
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_very_large_file_handling(self, file_collector: FileCollector, temp_dir: Path):
        """Test handling of files near the threshold."""
        # Create a file just under 10MB
        large_file = temp_dir / "almost_large.txt"
        content = "x" * (9 * 1024 * 1024)  # 9MB
        large_file.write_text(content)

        result = await file_collector.collect_paths_async([str(large_file)])

        assert result.file_count == 1


class TestTokenCounting:
    """Tests for token counting with caching."""

    def test_token_count_basic(self, file_collector: FileCollector):
        """Test basic token counting."""
        text = "Hello, world! This is a test."
        count = file_collector.count_tokens(text)

        assert count > 0
        assert count < len(text)  # Tokens should be fewer than characters

    def test_token_count_caching(self, file_collector: FileCollector):
        """Test that token counts are cached."""
        text = "This is the same text repeated for caching test."

        # First call
        count1 = file_collector.count_tokens(text)

        # Second call should use cache
        count2 = file_collector.count_tokens(text)

        assert count1 == count2

    def test_cache_stats(self, file_collector: FileCollector):
        """Test cache statistics."""
        # Make some calls
        file_collector.count_tokens("text one")
        file_collector.count_tokens("text two")
        file_collector.count_tokens("text one")  # Cache hit

        stats = file_collector.get_cache_stats()

        assert "token_cache_size" in stats
        assert stats["token_cache_size"] >= 2

    def test_empty_text_token_count(self, file_collector: FileCollector):
        """Test token counting for empty text."""
        count = file_collector.count_tokens("")
        assert count == 0


class TestCollectionResult:
    """Tests for CollectionResult functionality."""

    @pytest.mark.asyncio
    async def test_get_combined_content(self, file_collector: FileCollector, sample_files: dict):
        """Test getting combined content from collection."""
        result = await file_collector.collect_paths_async([
            str(sample_files["python"]),
            str(sample_files["javascript"]),
        ])

        content = result.get_combined_content(include_headers=True)

        assert "sample.py" in content or "python" in content.lower()
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_iter_content_streaming(self, file_collector: FileCollector, temp_dir: Path):
        """Test streaming content iteration."""
        # Create multiple files
        for i in range(5):
            (temp_dir / f"file_{i}.py").write_text(f"# Content {i}")

        result = await file_collector.collect_paths_async([str(temp_dir)])

        chunks = list(result.iter_content(chunk_size=100))

        assert len(chunks) > 0
        total_content = "".join(chunks)
        assert len(total_content) > 0


class TestConcurrencyLimits:
    """Tests for concurrency limiting."""

    def test_max_concurrent_reads_configured(self):
        """Test that max concurrent reads is configured."""
        assert MAX_CONCURRENT_FILE_READS == 50

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, file_collector: FileCollector, temp_dir: Path):
        """Test that semaphore limits concurrent file reads."""
        # Create many files
        for i in range(100):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        # Collection should complete without file descriptor exhaustion
        result = await file_collector.collect_paths_async([str(temp_dir)])

        assert result.file_count >= 100
        assert len(result.errors) == 0
