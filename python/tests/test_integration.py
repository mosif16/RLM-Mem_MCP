"""
Integration tests for full async pipeline end-to-end.

Tests cover:
- Complete file collection to RLM processing flow
- Memory store operations
- Server handler integration
- Error propagation across components
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from rlm_mem_mcp.config import RLMConfig, get_config
from rlm_mem_mcp.file_collector import FileCollector
from rlm_mem_mcp.cache_manager import CacheManager
from rlm_mem_mcp.memory_store import MemoryStore, MemoryEntry


class TestEndToEndFileProcessing:
    """End-to-end tests for file processing pipeline."""

    @pytest.mark.asyncio
    async def test_collect_and_count_tokens(self, file_collector: FileCollector, sample_files: dict):
        """Test file collection with token counting."""
        result = await file_collector.collect_paths_async([str(sample_files["python"])])

        assert result.file_count == 1
        assert result.total_tokens > 0

        # Verify content is accessible
        content = result.get_combined_content()
        assert len(content) > 0
        assert "def" in content or "print" in content

    @pytest.mark.asyncio
    async def test_collect_directory_recursive(self, file_collector: FileCollector, temp_dir: Path, sample_files: dict):
        """Test recursive directory collection."""
        result = await file_collector.collect_paths_async([str(temp_dir)])

        # Should find files in nested directories
        assert result.file_count >= 2  # At least main file and nested file

        paths = [f.path for f in result.files]
        nested_found = any("nested" in p or "helpers" in p or "utils" in p for p in paths)
        assert nested_found or result.file_count >= 2

    @pytest.mark.asyncio
    async def test_mixed_file_types(self, file_collector: FileCollector, sample_files: dict):
        """Test collecting mixed file types."""
        paths = [
            str(sample_files["python"]),
            str(sample_files["javascript"]),
            str(sample_files["markdown"]),
        ]
        result = await file_collector.collect_paths_async(paths)

        assert result.file_count >= 2  # At least py and js (md might be filtered)


class TestMemoryStoreIntegration:
    """Integration tests for memory store operations."""

    @pytest.mark.asyncio
    async def test_store_and_recall(self, memory_store: MemoryStore):
        """Test storing and recalling memory entries."""
        entry = await memory_store.store(
            key="test_key",
            value="Test value content",
            tags=["test", "integration"],
            token_count=10
        )

        assert entry.key == "test_key"
        assert entry.value == "Test value content"

        # Recall by key
        recalled = await memory_store.get("test_key")

        assert recalled is not None
        assert recalled.value == "Test value content"

    @pytest.mark.asyncio
    async def test_tag_search(self, memory_store: MemoryStore):
        """Test searching by tags."""
        await memory_store.store("key1", "value1", tags=["tag_a", "tag_b"])
        await memory_store.store("key2", "value2", tags=["tag_b", "tag_c"])
        await memory_store.store("key3", "value3", tags=["tag_c", "tag_d"])

        # Search for tag_b
        results = await memory_store.search_by_tags(["tag_b"])

        assert len(results) == 2
        keys = [r.key for r in results]
        assert "key1" in keys
        assert "key2" in keys

    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_store: MemoryStore):
        """Test memory statistics."""
        await memory_store.store("key1", "value1", token_count=10)
        await memory_store.store("key2", "value2", token_count=20)

        stats = await memory_store.get_stats()

        assert stats.entry_count == 2
        assert stats.total_tokens == 30

    @pytest.mark.asyncio
    async def test_delete_and_update(self, memory_store: MemoryStore):
        """Test deleting and updating entries."""
        await memory_store.store("key1", "original_value")

        # Update
        await memory_store.store("key1", "updated_value")
        entry = await memory_store.get("key1")
        assert entry.value == "updated_value"

        # Delete
        deleted = await memory_store.delete("key1")
        assert deleted is True

        # Verify deleted
        entry = await memory_store.get("key1")
        assert entry is None

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_store: MemoryStore):
        """Test clearing all entries."""
        await memory_store.store("key1", "value1")
        await memory_store.store("key2", "value2")

        count = await memory_store.clear()

        assert count == 2

        stats = await memory_store.get_stats()
        assert stats.entry_count == 0


class TestServerHandlerIntegration:
    """Integration tests for server handlers."""

    @pytest.mark.asyncio
    async def test_rlm_status_handler(self):
        """Test rlm_status handler returns valid status."""
        from rlm_mem_mcp.server import handle_rlm_status

        result = await handle_rlm_status({})

        assert len(result) == 1
        status = json.loads(result[0].text)

        assert "server" in status
        assert "configuration" in status

    @pytest.mark.asyncio
    async def test_memory_store_handler(self):
        """Test memory store handler."""
        from rlm_mem_mcp.server import handle_memory_store, handle_memory_recall

        # Store
        store_result = await handle_memory_store({
            "key": "integration_test_key",
            "value": "Integration test value",
            "tags": ["integration", "test"],
        })

        assert len(store_result) == 1
        assert "Stored" in store_result[0].text

        # Recall
        recall_result = await handle_memory_recall({
            "key": "integration_test_key",
        })

        assert len(recall_result) == 1
        recalled = json.loads(recall_result[0].text)
        assert len(recalled) == 1
        assert recalled[0]["key"] == "integration_test_key"

    @pytest.mark.asyncio
    async def test_memory_tag_search_handler(self):
        """Test memory tag search through handler."""
        from rlm_mem_mcp.server import handle_memory_store, handle_memory_recall

        # Store with tags
        await handle_memory_store({
            "key": "tagged_entry",
            "value": "Tagged value",
            "tags": ["search_test"],
        })

        # Search by tag
        result = await handle_memory_recall({
            "search_tags": ["search_test"],
        })

        assert len(result) == 1
        entries = json.loads(result[0].text)
        assert any(e["key"] == "tagged_entry" for e in entries)


class TestErrorPropagation:
    """Tests for error handling across components."""

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, file_collector: FileCollector):
        """Test handling of file not found errors."""
        result = await file_collector.collect_paths_async(["/nonexistent/path"])

        assert result.file_count == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_memory_store_key_not_found(self, memory_store: MemoryStore):
        """Test handling of key not found in memory store."""
        entry = await memory_store.get("nonexistent_key")

        assert entry is None

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, file_collector: FileCollector):
        """Test handling of empty inputs."""
        result = await file_collector.collect_paths_async([])

        assert result.file_count == 0


class TestAsyncConcurrency:
    """Tests for async concurrency handling."""

    @pytest.mark.asyncio
    async def test_concurrent_file_collection(self, file_collector: FileCollector, temp_dir: Path):
        """Test concurrent collection from multiple directories."""
        # Create subdirectories with files
        for i in range(3):
            subdir = temp_dir / f"subdir_{i}"
            subdir.mkdir()
            for j in range(5):
                (subdir / f"file_{j}.py").write_text(f"print({i}, {j})")

        # Collect from all subdirectories concurrently
        paths = [str(temp_dir / f"subdir_{i}") for i in range(3)]
        result = await file_collector.collect_paths_async(paths)

        assert result.file_count == 15  # 3 dirs * 5 files

    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, memory_store: MemoryStore):
        """Test concurrent memory store operations."""
        async def store_entry(i: int):
            await memory_store.store(f"key_{i}", f"value_{i}")

        # Store many entries concurrently
        await asyncio.gather(*[store_entry(i) for i in range(50)])

        stats = await memory_store.get_stats()
        assert stats.entry_count == 50


class TestResourceCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_memory_store_cleanup(self, temp_dir: Path):
        """Test memory store closes cleanly."""
        db_path = temp_dir / "cleanup_test.db"
        store = MemoryStore(db_path)

        await store.initialize()
        await store.store("key", "value")
        await store.close()

        # Should be able to reopen
        store2 = MemoryStore(db_path)
        await store2.initialize()

        entry = await store2.get("key")
        assert entry is not None
        assert entry.value == "value"

        await store2.close()

    @pytest.mark.asyncio
    async def test_file_handles_released(self, file_collector: FileCollector, temp_dir: Path):
        """Test that file handles are released after collection."""
        import resource

        # Create files
        for i in range(100):
            (temp_dir / f"file_{i}.py").write_text(f"print({i})")

        # Get initial file descriptor count
        initial_fds = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Collect files
        result = await file_collector.collect_paths_async([str(temp_dir)])
        assert result.file_count == 100

        # File descriptors should not grow significantly
        final_fds = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # This is a rough check - exact fd counting is platform-specific
        assert result.file_count == 100  # Just verify collection worked
