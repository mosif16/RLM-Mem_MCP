"""
File Tool Handlers for RLM-Mem MCP Server.

Provides handlers for single-file operations:
- rlm_read: Read file with line numbers
- rlm_grep: Search patterns using ripgrep
- rlm_glob: Find files by pattern
"""

from typing import Any

from mcp.types import TextContent

from ..structured_tools import read_file, grep_pattern, glob_files


# Input validation constants
MAX_PATH_LENGTH = 4096


def validate_path(path: str) -> tuple[bool, str]:
    """
    Validate a file path for safety.

    Returns:
        (is_valid, error_message) tuple
    """
    if not path:
        return False, "Empty path"

    if len(path) > MAX_PATH_LENGTH:
        return False, f"Path too long ({len(path)} > {MAX_PATH_LENGTH})"

    # Check for null bytes (path injection)
    if '\x00' in path:
        return False, "Path contains null bytes"

    # Block obvious traversal attempts (defense in depth)
    suspicious_patterns = ['../', '..\\', '/etc/', '/proc/', '/sys/', '/dev/']
    path_lower = path.lower()
    for pattern in suspicious_patterns:
        if pattern in path_lower:
            return False, f"Suspicious path pattern: {pattern}"

    return True, ""


async def handle_rlm_read(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle rlm_read tool call.

    Replaces Claude's native Read tool with identical functionality.
    Fast, no LLM overhead - just reads the file directly.
    """
    path = arguments.get("path", "")
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit", None)

    # Input validation
    if not path:
        return [TextContent(type="text", text="Error: path is required")]

    is_valid, error = validate_path(path)
    if not is_valid:
        return [TextContent(type="text", text=f"Error: {error}")]

    # Read the file
    result = read_file(path, offset=offset, limit=limit)

    if result.error:
        return [TextContent(type="text", text=f"Error reading file: {result.error}")]

    if not result.exists:
        return [TextContent(type="text", text=f"File not found: {path}")]

    # Format output like Claude's Read tool
    output = str(result)

    # Add metadata footer
    if result.limit:
        output += f"\n\n*Showing lines {result.offset + 1}-{result.offset + len(result.lines)} of {result.total_lines} total*"
    else:
        output += f"\n\n*{result.total_lines} lines total*"

    return [TextContent(type="text", text=output)]


async def handle_rlm_grep(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle rlm_grep tool call.

    Replaces Claude's native Grep tool with identical functionality.
    Uses ripgrep for fast searching with fallback to Python regex.
    """
    pattern = arguments.get("pattern", "")
    path = arguments.get("path", None)
    case_insensitive = arguments.get("case_insensitive", False)
    fixed_strings = arguments.get("fixed_strings", False)
    context_lines = arguments.get("context_lines", 0)
    file_type = arguments.get("file_type", None)
    glob = arguments.get("glob", None)
    output_mode = arguments.get("output_mode", "content")

    # Input validation
    if not pattern:
        return [TextContent(type="text", text="Error: pattern is required")]

    # Convert path to list for grep_pattern
    paths = [path] if path else None

    # Perform the search
    result = grep_pattern(
        pattern=pattern,
        paths=paths,
        case_insensitive=case_insensitive,
        fixed_strings=fixed_strings,
        context_lines=context_lines,
        file_type=file_type,
        glob=glob
    )

    if result.error:
        return [TextContent(type="text", text=f"Error: {result.error}")]

    # Format output based on output_mode
    if output_mode == "files_with_matches":
        if not result.matches:
            return [TextContent(type="text", text=f"No files match pattern: {pattern}")]
        files = sorted(set(m.file for m in result.matches))
        output = "\n".join(files)
        output += f"\n\n*{len(files)} files match*"

    elif output_mode == "count":
        if not result.matches:
            return [TextContent(type="text", text=f"0 matches for pattern: {pattern}")]
        # Count per file
        counts: dict[str, int] = {}
        for m in result.matches:
            counts[m.file] = counts.get(m.file, 0) + 1
        output = "\n".join(f"{f}: {c}" for f, c in sorted(counts.items()))
        output += f"\n\n*{result.total_matches} total matches in {result.files_matched} files*"

    else:  # content mode (default)
        output = str(result)

    return [TextContent(type="text", text=output)]


async def handle_rlm_glob(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle rlm_glob tool call.

    Replaces Claude's native Glob tool with identical functionality.
    Fast filesystem pattern matching.
    """
    pattern = arguments.get("pattern", "")
    path = arguments.get("path", None)
    include_hidden = arguments.get("include_hidden", False)

    # Input validation
    if not pattern:
        return [TextContent(type="text", text="Error: pattern is required")]

    # Perform the glob
    result = glob_files(
        pattern=pattern,
        path=path,
        include_hidden=include_hidden
    )

    if result.error:
        return [TextContent(type="text", text=f"Error: {result.error}")]

    if not result.files:
        return [TextContent(type="text", text=f"No files match pattern: {pattern}")]

    # Format output - one file per line
    output = "\n".join(result.files)
    output += f"\n\n*{result.total_count} files match '{pattern}'*"

    return [TextContent(type="text", text=output)]
