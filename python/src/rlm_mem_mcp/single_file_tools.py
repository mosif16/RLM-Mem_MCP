"""
Single-file tools for fast file operations (v2.6).

These tools replace Claude's native Read, Grep, and Glob tools.
They provide fast, direct access without full RLM overhead.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ripgrep_tools import rg_search, RgMatch


@dataclass
class FileContent:
    """Result from read_file()."""
    path: str
    content: str
    lines: list[str]
    total_lines: int
    offset: int
    limit: Optional[int]
    exists: bool
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"Error reading {self.path}: {self.error}"
        if not self.exists:
            return f"File not found: {self.path}"

        # Format like Claude's Read tool with line numbers
        output_lines = []
        for i, line in enumerate(self.lines):
            line_num = self.offset + i + 1
            output_lines.append(f"{line_num:6}→{line}")
        return "\n".join(output_lines)


@dataclass
class GrepResult:
    """Result from grep_pattern()."""
    pattern: str
    matches: list[RgMatch]
    files_matched: int
    total_matches: int
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if not self.matches:
            return f"No matches found for pattern: {self.pattern}"

        # Group by file
        by_file: dict[str, list[RgMatch]] = {}
        for m in self.matches:
            if m.file not in by_file:
                by_file[m.file] = []
            by_file[m.file].append(m)

        output = [f"## {self.total_matches} matches in {self.files_matched} files\n"]
        for file, file_matches in by_file.items():
            output.append(f"\n**{file}**")
            for m in file_matches[:20]:  # Limit per file
                output.append(f"  {m.line}: {m.text[:150]}")
            if len(file_matches) > 20:
                output.append(f"  ... and {len(file_matches) - 20} more")

        return "\n".join(output)


@dataclass
class GlobResult:
    """Result from glob_files()."""
    pattern: str
    files: list[str]
    total_count: int
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if not self.files:
            return f"No files found matching: {self.pattern}"

        output = [f"## {self.total_count} files matching '{self.pattern}'\n"]
        for f in self.files[:50]:
            output.append(f"  {f}")
        if self.total_count > 50:
            output.append(f"\n  ... and {self.total_count - 50} more")

        return "\n".join(output)


def read_file(
    path: str,
    offset: int = 0,
    limit: Optional[int] = None,
    encoding: str = "utf-8"
) -> FileContent:
    """
    Read a single file with optional offset and limit.

    Replaces Claude's native Read tool. Fast, no LLM overhead.

    Args:
        path: Absolute or relative path to the file
        offset: Line number to start from (0-indexed, default 0)
        limit: Maximum number of lines to read (default: all)
        encoding: File encoding (default: utf-8)

    Returns:
        FileContent with file contents and metadata

    Example:
        # Read entire file
        content = read_file("/path/to/file.py")
        print(content)

        # Read lines 100-200
        content = read_file("/path/to/file.py", offset=100, limit=100)

        # Access raw content
        text = content.content
        lines = content.lines
    """
    try:
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            return FileContent(
                path=str(file_path),
                content="",
                lines=[],
                total_lines=0,
                offset=offset,
                limit=limit,
                exists=False
            )

        if not file_path.is_file():
            return FileContent(
                path=str(file_path),
                content="",
                lines=[],
                total_lines=0,
                offset=offset,
                limit=limit,
                exists=False,
                error=f"Path is not a file: {file_path}"
            )

        # Read file
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        # Apply offset and limit
        if offset > 0:
            all_lines = all_lines[offset:]
        if limit is not None:
            all_lines = all_lines[:limit]

        # Strip trailing newlines for cleaner display
        lines = [line.rstrip('\n\r') for line in all_lines]
        content = "\n".join(lines)

        return FileContent(
            path=str(file_path),
            content=content,
            lines=lines,
            total_lines=total_lines,
            offset=offset,
            limit=limit,
            exists=True
        )

    except Exception as e:
        return FileContent(
            path=path,
            content="",
            lines=[],
            total_lines=0,
            offset=offset,
            limit=limit,
            exists=False,
            error=str(e)
        )


def grep_pattern(
    pattern: str,
    paths: Optional[list[str] | str] = None,
    case_insensitive: bool = False,
    fixed_strings: bool = False,
    context_lines: int = 0,
    file_type: Optional[str] = None,
    glob: Optional[str] = None,
    max_matches: int = 1000
) -> GrepResult:
    """
    Search for a pattern in files using ripgrep.

    Replaces Claude's native Grep tool. Uses ripgrep for speed.
    Falls back to Python regex if ripgrep not installed.

    Args:
        pattern: Regex pattern (or literal if fixed_strings=True)
        paths: File/directory paths to search (default: current dir)
        case_insensitive: Ignore case (-i)
        fixed_strings: Treat pattern as literal string (-F)
        context_lines: Lines of context around matches (-C)
        file_type: Limit to file type (e.g., "py", "js", "swift")
        glob: Glob pattern filter (e.g., "*.tsx", "**/*.swift")
        max_matches: Maximum matches to return (default: 1000)

    Returns:
        GrepResult with matches and metadata

    Example:
        # Find all TODO comments in Python files
        result = grep_pattern("TODO|FIXME", file_type="py")
        print(result)

        # Case-insensitive literal search
        result = grep_pattern("api_key", fixed_strings=True, case_insensitive=True)

        # Search with context
        result = grep_pattern("def main", context_lines=3)
    """
    try:
        matches = rg_search(
            pattern=pattern,
            paths=paths,
            case_insensitive=case_insensitive,
            fixed_strings=fixed_strings,
            context_lines=context_lines,
            file_type=file_type,
            glob=glob
        )

        # Limit results
        if len(matches) > max_matches:
            matches = matches[:max_matches]

        # Count unique files
        files_matched = len(set(m.file for m in matches))

        return GrepResult(
            pattern=pattern,
            matches=matches,
            files_matched=files_matched,
            total_matches=len(matches)
        )

    except Exception as e:
        return GrepResult(
            pattern=pattern,
            matches=[],
            files_matched=0,
            total_matches=0,
            error=str(e)
        )


def glob_files(
    pattern: str,
    path: Optional[str] = None,
    include_hidden: bool = False,
    max_results: int = 1000
) -> GlobResult:
    """
    Find files matching a glob pattern.

    Replaces Claude's native Glob tool. Fast filesystem traversal.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.swift", "*.json")
        path: Base directory to search from (default: current dir)
        include_hidden: Include hidden files/directories (default: False)
        max_results: Maximum files to return (default: 1000)

    Returns:
        GlobResult with matching file paths

    Example:
        # Find all Python files
        result = glob_files("**/*.py")
        print(result)

        # Find Swift files in src directory
        result = glob_files("**/*.swift", path="src")

        # Get list of files
        files = result.files
    """
    try:
        if path is None:
            path = "."

        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            return GlobResult(
                pattern=pattern,
                files=[],
                total_count=0,
                error=f"Path does not exist: {path}"
            )

        # Use pathlib.glob for pattern matching
        all_files = []
        for file_path in base_path.glob(pattern):
            if file_path.is_file():
                # Skip hidden files unless requested
                if not include_hidden and any(part.startswith('.') for part in file_path.parts):
                    continue
                all_files.append(str(file_path))

        # Limit results
        files_to_return = all_files[:max_results]
        total_count = len(all_files)

        return GlobResult(
            pattern=pattern,
            files=files_to_return,
            total_count=total_count
        )

    except Exception as e:
        return GlobResult(
            pattern=pattern,
            files=[],
            total_count=0,
            error=str(e)
        )
