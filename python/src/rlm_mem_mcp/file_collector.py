"""
File Collector for RLM-Mem MCP Server

Collects and processes files from directories for RLM analysis.
Implements intelligent filtering and content aggregation.

Performance Optimizations:
- Async file I/O with aiofiles (non-blocking)
- Parallel file collection with asyncio.gather
- Semaphore-based concurrency control (prevent fd exhaustion)
- Symlink loop detection
- Timeout for file reads (network mount safety)
- Chunked reading for large files (memory efficiency)
- Incremental token counting
"""

import asyncio
import fnmatch
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Callable

import aiofiles
import tiktoken

from .config import RLMConfig


# Constants for performance tuning
MAX_CONCURRENT_FILE_READS = 50  # Prevent fd exhaustion
FILE_READ_TIMEOUT_SECONDS = 30  # Timeout for slow/network files
LARGE_FILE_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10MB - use chunked reading
CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB chunks for large files


@dataclass
class CollectedFile:
    """Represents a collected file with metadata."""

    path: str
    relative_path: str
    content: str
    token_count: int
    size_bytes: int
    extension: str


@dataclass
class FileMetadata:
    """Lightweight file metadata for RAG-style file list."""

    path: str
    relative_path: str
    size_bytes: int
    extension: str
    first_line: str = ""  # First non-empty line for context

    def to_summary(self) -> str:
        """Get a one-line summary of the file."""
        preview = self.first_line[:60] + "..." if len(self.first_line) > 60 else self.first_line
        return f"{self.relative_path} ({self.size_bytes:,} bytes) - {preview}"


@dataclass
class CollectionResult:
    """Result of file collection."""

    files: list[CollectedFile] = field(default_factory=list)
    total_tokens: int = 0
    total_bytes: int = 0
    skipped_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Lazy loading support
    _file_metadata: list[FileMetadata] = field(default_factory=list)
    _content_loaded: bool = True  # False when using lazy loading

    @property
    def file_count(self) -> int:
        return len(self.files) if self._content_loaded else len(self._file_metadata)

    def get_file_list(self) -> list[str]:
        """Get list of file paths (no content loading)."""
        if self._content_loaded:
            return [f.relative_path for f in self.files]
        return [m.relative_path for m in self._file_metadata]

    def get_file_summaries(self) -> str:
        """Get formatted file list with summaries (for RAG-style prompts)."""
        lines = []
        if self._content_loaded:
            for f in self.files:
                first_line = f.content.split('\n')[0][:60] if f.content else ""
                lines.append(f"- {f.relative_path} ({f.size_bytes:,} bytes, ~{f.token_count} tokens)")
        else:
            for m in self._file_metadata:
                lines.append(f"- {m.to_summary()}")
        return "\n".join(lines)

    def get_combined_content(self, include_headers: bool = True) -> str:
        """Get all file contents combined into a single string."""
        parts = []
        for f in self.files:
            if include_headers:
                parts.append(f"### File: {f.relative_path}\n```{f.extension.lstrip('.')}\n{f.content}\n```\n")
            else:
                parts.append(f.content)
        return "\n".join(parts)

    def iter_content(self, include_headers: bool = True, chunk_size: int = 0) -> Iterator[str]:
        """Iterate over file contents (memory-efficient for large collections)."""
        for f in self.files:
            if include_headers:
                yield f"### File: {f.relative_path}\n```{f.extension.lstrip('.')}\n{f.content}\n```\n"
            else:
                yield f.content

    def get_file_content(self, relative_path: str) -> str | None:
        """Get content of a specific file by path."""
        for f in self.files:
            if f.relative_path == relative_path or f.path == relative_path:
                return f.content
        return None


class FileCollector:
    """
    Collects files from directories for RLM processing.

    Features:
    - Async file I/O for non-blocking operations
    - Parallel file collection with configurable concurrency
    - Filters by extension
    - Skips common non-code directories
    - Respects file size limits
    - Tracks token counts for context management
    - Symlink loop detection
    - Timeout protection for slow reads
    """

    def __init__(self, config: RLMConfig | None = None):
        self.config = config or RLMConfig()
        self._encoder: tiktoken.Encoding | None = None
        self._semaphore: asyncio.Semaphore | None = None

        # Token count cache with LRU-style cleanup
        self._token_cache: dict[str, int] = {}  # sha256_hash -> token_count
        self._token_cache_max_size = 10000
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            self._encoder = tiktoken.encoding_for_model("gpt-4")
        return self._encoder

    async def get_encoder_async(self) -> tiktoken.Encoding:
        """Get encoder without blocking event loop (first load can be slow)."""
        if self._encoder is None:
            self._encoder = await asyncio.to_thread(
                tiktoken.encoding_for_model, "gpt-4"
            )
        return self._encoder

    def _compute_text_hash(self, text: str) -> str:
        """Compute a collision-resistant hash for text caching."""
        # Use SHA-256 truncated to 16 bytes for balance of collision resistance and memory
        return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()[:32]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with caching."""
        if not text:
            return 0

        # Use SHA-256 for collision-resistant cache key
        text_hash = self._compute_text_hash(text)

        if text_hash in self._token_cache:
            self._cache_hits += 1
            return self._token_cache[text_hash]

        self._cache_misses += 1
        token_count = len(self.encoder.encode(text))

        # LRU-style cleanup when cache gets too large
        if len(self._token_cache) >= self._token_cache_max_size:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._token_cache.keys())[:self._token_cache_max_size // 5]
            for key in keys_to_remove:
                del self._token_cache[key]

        self._token_cache[text_hash] = token_count
        return token_count

    async def count_tokens_async(self, text: str) -> int:
        """Count tokens without blocking event loop."""
        if not text:
            return 0

        # Use SHA-256 for collision-resistant cache key
        text_hash = self._compute_text_hash(text)
        if text_hash in self._token_cache:
            self._cache_hits += 1
            return self._token_cache[text_hash]

        self._cache_misses += 1
        encoder = await self.get_encoder_async()
        token_count = await asyncio.to_thread(lambda: len(encoder.encode(text)))

        if len(self._token_cache) >= self._token_cache_max_size:
            keys_to_remove = list(self._token_cache.keys())[:self._token_cache_max_size // 5]
            for key in keys_to_remove:
                del self._token_cache[key]

        self._token_cache[text_hash] = token_count
        return token_count

    def get_cache_stats(self) -> dict[str, int]:
        """Get token cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "size": len(self._token_cache),
        }

    def should_skip_directory(self, dir_name: str) -> bool:
        """Check if a directory should be skipped."""
        for pattern in self.config.skipped_directories:
            if fnmatch.fnmatch(dir_name, pattern):
                return True
        return False

    def should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on extension."""
        return file_path.suffix.lower() in self.config.included_extensions

    # -------------------------------------------------------------------------
    # Synchronous API (backwards compatible)
    # -------------------------------------------------------------------------

    def collect_paths(self, paths: list[str]) -> CollectionResult:
        """
        Collect files from multiple paths (sync wrapper for async).

        Args:
            paths: List of file or directory paths to collect

        Returns:
            CollectionResult with collected files and metadata
        """
        return asyncio.run(self.collect_paths_async(paths))

    def collect_text(self, text: str, name: str = "input") -> CollectionResult:
        """
        Create a CollectionResult from raw text input.

        Args:
            text: The text content to process
            name: Name to use for the "file"

        Returns:
            CollectionResult containing the text
        """
        result = CollectionResult()

        token_count = self.count_tokens(text)

        collected_file = CollectedFile(
            path=name,
            relative_path=name,
            content=text,
            token_count=token_count,
            size_bytes=len(text.encode("utf-8")),
            extension=".txt",
        )

        result.files.append(collected_file)
        result.total_tokens = token_count
        result.total_bytes = collected_file.size_bytes

        return result

    # -------------------------------------------------------------------------
    # Async API (new, high-performance)
    # -------------------------------------------------------------------------

    async def collect_paths_async(
        self,
        paths: list[str],
        progress_callback: Callable[[str], None] | None = None
    ) -> CollectionResult:
        """
        Collect files from multiple paths asynchronously.

        Features:
        - Parallel file reading with semaphore-controlled concurrency
        - Symlink loop detection
        - Timeout protection for slow reads
        - Early termination on token limit

        Args:
            paths: List of file or directory paths to collect
            progress_callback: Optional callback for progress updates

        Returns:
            CollectionResult with collected files and metadata
        """
        result = CollectionResult()

        # Initialize semaphore for this collection run
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILE_READS)

        # Track visited directories to detect symlink loops
        visited_dirs: set[str] = set()

        # Collect all file paths first (fast, sync operation)
        all_file_paths: list[tuple[Path, Path]] = []  # (file_path, base_dir)

        for path_str in paths:
            path = Path(path_str).resolve()

            if not path.exists():
                result.errors.append(f"Path not found: {path_str}")
                continue

            if path.is_file():
                all_file_paths.append((path, path.parent))
            elif path.is_dir():
                # Collect file paths from directory
                try:
                    for file_path in self._walk_directory_sync(path, visited_dirs):
                        all_file_paths.append((file_path, path))
                except Exception as e:
                    result.errors.append(f"Error walking {path}: {e}")
            else:
                result.errors.append(f"Invalid path type: {path_str}")

        if progress_callback:
            progress_callback(f"Found {len(all_file_paths)} files to process")

        # Process files in parallel with controlled concurrency
        if all_file_paths:
            tasks = [
                self._collect_file_async(file_path, base_dir, result)
                for file_path, base_dir in all_file_paths
            ]

            # Use gather with return_exceptions to handle individual failures
            await asyncio.gather(*tasks, return_exceptions=True)

        if progress_callback:
            progress_callback(f"Collected {result.file_count} files, {result.total_tokens:,} tokens")

        return result

    def _walk_directory_sync(
        self,
        directory: Path,
        visited: set[str]
    ) -> Iterator[Path]:
        """
        Walk directory tree synchronously, with symlink loop detection.

        This is kept sync because directory traversal is fast and
        the overhead of async isn't worth it for stat() calls.
        """
        try:
            # Resolve to detect symlink loops
            resolved = str(directory.resolve())
            if resolved in visited:
                return  # Symlink loop detected
            visited.add(resolved)

            for item in directory.iterdir():
                try:
                    if item.is_dir():
                        if not self.should_skip_directory(item.name):
                            yield from self._walk_directory_sync(item, visited)
                    elif item.is_file():
                        if self.should_include_file(item):
                            yield item
                except PermissionError:
                    pass  # Skip files/dirs we can't access
                except OSError:
                    pass  # Handle other OS errors (e.g., too long paths on Windows)
        except PermissionError:
            pass  # Skip directories we can't access

    async def _collect_file_async(
        self,
        file_path: Path,
        base_dir: Path,
        result: CollectionResult
    ) -> None:
        """
        Collect a single file asynchronously with timeout and concurrency control.
        """
        async with self._semaphore:
            try:
                # Apply timeout for the entire file operation
                await asyncio.wait_for(
                    self._read_and_process_file(file_path, base_dir, result),
                    timeout=FILE_READ_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                result.skipped_files.append(f"{file_path} (read timeout)")
            except Exception as e:
                result.errors.append(f"Error reading {file_path}: {e}")

    async def _read_and_process_file(
        self,
        file_path: Path,
        base_dir: Path,
        result: CollectionResult
    ) -> None:
        """Read and process a single file."""
        try:
            # Check file size (sync is fine for stat)
            size = file_path.stat().st_size

            if size > self.config.max_file_size_bytes:
                result.skipped_files.append(
                    f"{file_path} (too large: {size:,} bytes)"
                )
                return

            if size == 0:
                result.skipped_files.append(f"{file_path} (empty file)")
                return

            # Read content asynchronously
            content = await self._read_file_content(file_path, size)
            if content is None:
                result.skipped_files.append(f"{file_path} (encoding error)")
                return

            # Count tokens (use async for large files)
            if size > 100_000:  # 100KB threshold for async tokenization
                token_count = await self.count_tokens_async(content)
            else:
                token_count = self.count_tokens(content)

            # Check if adding this file would exceed limits
            if result.total_tokens + token_count > self.config.max_total_tokens:
                result.skipped_files.append(
                    f"{file_path} (would exceed token limit)"
                )
                return

            # Calculate relative path
            try:
                relative_path = str(file_path.relative_to(base_dir))
            except ValueError:
                relative_path = str(file_path)

            # Add to results (use sys.intern for path strings to reduce memory)
            collected_file = CollectedFile(
                path=sys.intern(str(file_path)),
                relative_path=sys.intern(relative_path),
                content=content,
                token_count=token_count,
                size_bytes=size,
                extension=file_path.suffix.lower(),
            )

            result.files.append(collected_file)
            result.total_tokens += token_count
            result.total_bytes += size

        except FileNotFoundError:
            result.skipped_files.append(f"{file_path} (file not found)")
        except PermissionError:
            result.skipped_files.append(f"{file_path} (permission denied)")

    async def _read_file_content(self, file_path: Path, size: int) -> str | None:
        """
        Read file content with proper encoding handling.
        Uses chunked reading for large files.
        """
        encodings = ["utf-8", "latin-1", "cp1252"]

        last_error: Exception | None = None
        for encoding in encodings:
            try:
                if size > LARGE_FILE_THRESHOLD_BYTES:
                    # Chunked reading for large files
                    return await self._read_file_chunked(file_path, encoding)
                else:
                    # Standard async read for normal files
                    async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
                        return await f.read()
            except UnicodeDecodeError:
                continue  # Expected - try next encoding
            except Exception as e:
                last_error = e
                continue

        # Log if we failed for reasons other than encoding
        if last_error is not None and not isinstance(last_error, UnicodeDecodeError):
            print(f"Warning: Failed to read {file_path}: {type(last_error).__name__}: {last_error}", file=sys.stderr)

        return None

    async def _read_file_chunked(self, file_path: Path, encoding: str) -> str:
        """Read large file in chunks to reduce memory pressure."""
        chunks: list[str] = []

        async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
            while True:
                chunk = await f.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)

        return "".join(chunks)

    async def collect_text_async(self, text: str, name: str = "input") -> CollectionResult:
        """
        Create a CollectionResult from raw text input (async version).

        Args:
            text: The text content to process
            name: Name to use for the "file"

        Returns:
            CollectionResult containing the text
        """
        result = CollectionResult()

        token_count = await self.count_tokens_async(text)

        collected_file = CollectedFile(
            path=name,
            relative_path=name,
            content=text,
            token_count=token_count,
            size_bytes=len(text.encode("utf-8")),
            extension=".txt",
        )

        result.files.append(collected_file)
        result.total_tokens = token_count
        result.total_bytes = collected_file.size_bytes

        return result
