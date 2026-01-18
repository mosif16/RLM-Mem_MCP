"""
File Collector for RLM-Mem MCP Server

Collects and processes files from directories for RLM analysis.
Implements intelligent filtering and content aggregation.
"""

import os
import fnmatch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

import tiktoken

from .config import RLMConfig


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
class CollectionResult:
    """Result of file collection."""

    files: list[CollectedFile] = field(default_factory=list)
    total_tokens: int = 0
    total_bytes: int = 0
    skipped_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def get_combined_content(self, include_headers: bool = True) -> str:
        """Get all file contents combined into a single string."""
        parts = []
        for f in self.files:
            if include_headers:
                parts.append(f"### File: {f.relative_path}\n```{f.extension.lstrip('.')}\n{f.content}\n```\n")
            else:
                parts.append(f.content)
        return "\n".join(parts)


class FileCollector:
    """
    Collects files from directories for RLM processing.

    Features:
    - Filters by extension
    - Skips common non-code directories
    - Respects file size limits
    - Tracks token counts for context management
    """

    def __init__(self, config: RLMConfig | None = None):
        self.config = config or RLMConfig()
        self._encoder: tiktoken.Encoding | None = None

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            self._encoder = tiktoken.encoding_for_model("gpt-4")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def should_skip_directory(self, dir_name: str) -> bool:
        """Check if a directory should be skipped."""
        for pattern in self.config.skipped_directories:
            if fnmatch.fnmatch(dir_name, pattern):
                return True
        return False

    def should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on extension."""
        return file_path.suffix.lower() in self.config.included_extensions

    def collect_paths(self, paths: list[str]) -> CollectionResult:
        """
        Collect files from multiple paths.

        Args:
            paths: List of file or directory paths to collect

        Returns:
            CollectionResult with collected files and metadata
        """
        result = CollectionResult()

        for path_str in paths:
            path = Path(path_str).resolve()

            if not path.exists():
                result.errors.append(f"Path not found: {path_str}")
                continue

            if path.is_file():
                self._collect_file(path, path.parent, result)
            elif path.is_dir():
                self._collect_directory(path, result)
            else:
                result.errors.append(f"Invalid path type: {path_str}")

            # Check if we've exceeded token limit
            if result.total_tokens >= self.config.max_total_tokens:
                result.errors.append(
                    f"Token limit reached ({self.config.max_total_tokens}). "
                    "Some files may not be included."
                )
                break

        return result

    def _collect_directory(self, directory: Path, result: CollectionResult) -> None:
        """Recursively collect files from a directory."""
        try:
            for entry in self._walk_directory(directory):
                if result.total_tokens >= self.config.max_total_tokens:
                    break
                self._collect_file(entry, directory, result)
        except PermissionError as e:
            result.errors.append(f"Permission denied: {directory} - {e}")

    def _walk_directory(self, directory: Path) -> Iterator[Path]:
        """Walk directory tree, skipping excluded directories."""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    if not self.should_skip_directory(item.name):
                        yield from self._walk_directory(item)
                elif item.is_file():
                    if self.should_include_file(item):
                        yield item
        except PermissionError:
            pass  # Skip directories we can't access

    def _collect_file(
        self,
        file_path: Path,
        base_dir: Path,
        result: CollectionResult
    ) -> None:
        """Collect a single file."""
        try:
            # Check file size
            size = file_path.stat().st_size
            if size > self.config.max_file_size_bytes:
                result.skipped_files.append(
                    f"{file_path} (too large: {size} bytes)"
                )
                return

            if size == 0:
                result.skipped_files.append(f"{file_path} (empty file)")
                return

            # Read content
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try with latin-1 as fallback
                try:
                    content = file_path.read_text(encoding="latin-1")
                except Exception:
                    result.skipped_files.append(
                        f"{file_path} (encoding error)"
                    )
                    return

            # Count tokens
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

            # Add to results
            collected_file = CollectedFile(
                path=str(file_path),
                relative_path=relative_path,
                content=content,
                token_count=token_count,
                size_bytes=size,
                extension=file_path.suffix.lower(),
            )

            result.files.append(collected_file)
            result.total_tokens += token_count
            result.total_bytes += size

        except Exception as e:
            result.errors.append(f"Error reading {file_path}: {e}")

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
