"""
Ripgrep integration for fast search (v2.5 - 10-100x faster).

Provides wrappers around ripgrep for efficient pattern searching.
Falls back to Python regex if ripgrep is not installed.
"""

import re
import subprocess
import shutil
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Check if ripgrep is available (cached at import time for performance)
_RG_AVAILABLE_CACHED: Optional[bool] = None

# Legacy constant for backwards compatibility
RG_AVAILABLE = shutil.which("rg") is not None


def _check_rg_available() -> bool:
    """v2.7: Dynamic ripgrep availability check (checked at call time)."""
    global _RG_AVAILABLE_CACHED
    if _RG_AVAILABLE_CACHED is None:
        _RG_AVAILABLE_CACHED = shutil.which("rg") is not None
    return _RG_AVAILABLE_CACHED


@dataclass
class RgMatch:
    """A single match from ripgrep."""
    file: str
    line: int
    content: str
    match_text: str
    column_start: int = 0
    column_end: int = 0
    # v2.7: Context lines support
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Backwards compatibility property for match_text."""
        return self.match_text

    def to_dict(self) -> dict:
        result = {
            "file": self.file,
            "line": self.line,
            "content": self.content,
            "match": self.match_text,
        }
        if self.context_before:
            result["context_before"] = self.context_before
        if self.context_after:
            result["context_after"] = self.context_after
        return result


@dataclass
class RgSearchResult:
    """v2.7: Result from rg_search with error indication."""
    matches: list[RgMatch]
    timed_out: bool = False
    error: Optional[str] = None

    def __iter__(self):
        """Allow iteration over matches for backwards compatibility."""
        return iter(self.matches)

    def __len__(self):
        return len(self.matches)


def rg_search(
    pattern: str,
    paths: Optional[list[str] | str] = None,
    case_insensitive: bool = False,
    word_boundary: bool = False,
    context_lines: int = 0,
    file_type: Optional[str] = None,
    glob: Optional[str] = None,
    max_count: Optional[int] = None,
    fixed_strings: bool = False,
) -> list[RgMatch]:
    """
    Fast search using ripgrep (rg).

    10-100x faster than Python regex for large codebases.

    Args:
        pattern: Regex pattern to search for (or literal if fixed_strings=True)
        paths: File/directory paths to search (default: current directory)
        case_insensitive: Ignore case (-i flag)
        word_boundary: Match whole words only (-w flag)
        context_lines: Lines of context before/after (-C flag)
        file_type: Limit to file type (e.g., "py", "swift", "js")
        glob: Glob pattern to filter files (e.g., "*.swift")
        max_count: Maximum matches per file (-m flag)
        fixed_strings: Treat pattern as literal string, not regex (-F flag)

    Returns:
        List of RgMatch objects with file, line, content, match

    Example:
        # Find all TODO comments
        matches = rg_search("TODO|FIXME", file_type="py")

        # Find literal string (fast)
        matches = rg_search("API_KEY", fixed_strings=True)

        # Case-insensitive search
        matches = rg_search("password", case_insensitive=True)
    """
    # v2.7: Dynamic availability check
    if not _check_rg_available():
        # Fallback to Python search if rg not installed
        return _python_fallback_search(pattern, paths, case_insensitive)

    # Build command
    cmd = ["rg", "--json"]

    if case_insensitive:
        cmd.append("-i")
    if word_boundary:
        cmd.append("-w")
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if file_type:
        cmd.extend(["--type", file_type])
    if glob:
        cmd.extend(["--glob", glob])
    if max_count:
        cmd.extend(["-m", str(max_count)])
    if fixed_strings:
        cmd.append("-F")

    cmd.append(pattern)

    # Add paths
    if paths:
        if isinstance(paths, str):
            cmd.append(paths)
        else:
            cmd.extend(paths)
    else:
        cmd.append(".")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )

        return _parse_rg_json(result.stdout, context_lines > 0)

    except subprocess.TimeoutExpired:
        # v2.7: Return error indicator instead of silent empty list
        logging.warning("RLM: ripgrep search timed out after 60 seconds")
        # Return empty list for backwards compatibility
        return []
    except Exception as e:
        logging.warning(f"RLM: ripgrep search failed: {e}")
        return []


def _parse_rg_json(output: str, has_context: bool = False) -> list[RgMatch]:
    """Parse ripgrep JSON output into RgMatch objects.

    Args:
        output: Raw JSON output from ripgrep
        has_context: If True, preserve context lines from -C flag
    """
    matches = []
    empty_path_count = 0

    # v2.7: Track context lines when -C flag is used
    context_buffer: dict[str, list[tuple[int, str]]] = {}  # path -> [(line_num, content)]
    pending_matches: list[tuple[RgMatch, str]] = []  # (match, path) for context assignment

    for line in output.strip().split("\n"):
        if not line:
            continue

        try:
            data = json.loads(line)
            msg_type = data.get("type")

            # ripgrep outputs different message types
            if msg_type == "match":
                match_data = data.get("data", {})
                path = match_data.get("path", {}).get("text", "")
                line_num = match_data.get("line_number", 0)
                line_text = match_data.get("lines", {}).get("text", "").rstrip("\n")

                # Validate path - warn on empty/invalid paths
                if not path:
                    empty_path_count += 1
                    logging.warning(f"RLM: Empty path in ripgrep match at line {line_num}")
                    path = "unknown"  # Fallback to prevent crashes
                else:
                    # Normalize path to absolute for consistent deduplication
                    try:
                        path = str(Path(path).resolve())
                    except (OSError, ValueError):
                        pass  # Keep original path if resolution fails

                # Extract submatch info
                submatches = match_data.get("submatches", [])
                match_text = ""
                col_start = 0
                col_end = 0

                if submatches:
                    first_match = submatches[0]
                    match_text = first_match.get("match", {}).get("text", "")
                    col_start = first_match.get("start", 0)
                    col_end = first_match.get("end", 0)

                rg_match = RgMatch(
                    file=path,
                    line=line_num,
                    content=line_text,
                    match_text=match_text,
                    column_start=col_start,
                    column_end=col_end,
                )

                if has_context:
                    # Collect context lines from buffer
                    if path in context_buffer:
                        for ctx_line_num, ctx_content in context_buffer[path]:
                            if ctx_line_num < line_num:
                                rg_match.context_before.append(ctx_content)
                        context_buffer[path] = []  # Clear used context
                    pending_matches.append((rg_match, path))
                else:
                    matches.append(rg_match)

            elif msg_type == "context" and has_context:
                # v2.7: Preserve context lines
                ctx_data = data.get("data", {})
                path = ctx_data.get("path", {}).get("text", "")
                line_num = ctx_data.get("line_number", 0)
                line_text = ctx_data.get("lines", {}).get("text", "").rstrip("\n")

                if path:
                    try:
                        path = str(Path(path).resolve())
                    except (OSError, ValueError):
                        pass

                    if path not in context_buffer:
                        context_buffer[path] = []
                    context_buffer[path].append((line_num, line_text))

                    # Check if this is context_after for any pending match
                    for rg_match, match_path in pending_matches:
                        if match_path == path and line_num > rg_match.line:
                            rg_match.context_after.append(line_text)

        except json.JSONDecodeError:
            continue

    # Add pending matches (with context)
    if has_context:
        matches.extend([m for m, _ in pending_matches])

    if empty_path_count > 0:
        logging.warning(f"RLM: {empty_path_count} ripgrep match(es) had empty paths")

    return matches


def _python_fallback_search(
    pattern: str,
    paths: Optional[list[str] | str],
    case_insensitive: bool = False,
) -> list[RgMatch]:
    """
    Fallback Python search when ripgrep is not available.

    This is slower but ensures functionality without rg installed.
    """
    matches = []
    flags = re.IGNORECASE if case_insensitive else 0

    try:
        regex = re.compile(pattern, flags)
    except re.error:
        return []

    # Determine paths to search
    search_paths = []
    if paths is None:
        search_paths = ["."]
    elif isinstance(paths, str):
        search_paths = [paths]
    else:
        search_paths = list(paths)

    for search_path in search_paths:
        if os.path.isfile(search_path):
            _search_file(search_path, regex, matches)
        elif os.path.isdir(search_path):
            for root, _, files in os.walk(search_path):
                # Skip common non-code directories
                if any(skip in root for skip in [".git", "node_modules", "__pycache__", "venv"]):
                    continue
                for fname in files:
                    filepath = os.path.join(root, fname)
                    _search_file(filepath, regex, matches)

    return matches


def _search_file(filepath: str, regex: re.Pattern, matches: list[RgMatch]) -> None:
    """Search a single file with regex."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                match = regex.search(line)
                if match:
                    matches.append(RgMatch(
                        file=filepath,
                        line=line_num,
                        content=line.rstrip("\n"),
                        match_text=match.group(0),
                        column_start=match.start(),
                        column_end=match.end(),
                    ))
    except (IOError, OSError):
        pass


def rg_literal(text: str, paths: Optional[list[str] | str] = None, case_insensitive: bool = False) -> list[RgMatch]:
    """
    Fast literal string search (not regex).

    This is the fastest search mode - use when you don't need regex.

    Args:
        text: Exact string to find
        paths: Where to search
        case_insensitive: Ignore case

    Example:
        matches = rg_literal("def main(")
        matches = rg_literal("API_KEY", case_insensitive=True)
    """
    return rg_search(text, paths, case_insensitive=case_insensitive, fixed_strings=True)


def rg_files(pattern: str, paths: Optional[list[str] | str] = None, **kwargs) -> list[str]:
    """
    Return only file paths that match pattern (no line details).

    Useful for finding which files contain a pattern without full details.

    Args:
        pattern: Regex pattern
        paths: Where to search
        **kwargs: Additional args passed to rg_search

    Returns:
        List of unique file paths

    Example:
        files = rg_files("class.*Service")
        files = rg_files("import React", file_type="tsx")
    """
    matches = rg_search(pattern, paths, **kwargs)
    return list(set(m.file for m in matches))


def rg_count(pattern: str, paths: Optional[list[str] | str] = None, **kwargs) -> dict[str, int]:
    """
    Count matches per file.

    Args:
        pattern: Regex pattern
        paths: Where to search
        **kwargs: Additional args passed to rg_search

    Returns:
        Dict of {filepath: match_count}

    Example:
        counts = rg_count("TODO|FIXME")
        # {'src/main.py': 5, 'src/utils.py': 2}
    """
    matches = rg_search(pattern, paths, **kwargs)
    counts: dict[str, int] = {}
    for m in matches:
        counts[m.file] = counts.get(m.file, 0) + 1
    return counts
