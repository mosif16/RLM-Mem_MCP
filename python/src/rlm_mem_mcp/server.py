#!/usr/bin/env python3
"""
RLM-Mem MCP Server

An MCP server implementing the TRUE RLM technique from arXiv:2512.24601.

KEY INSIGHT: Content is stored as a VARIABLE, not in LLM context.
- LLM writes CODE to examine portions of the content
- Sub-LLM responses stored in VARIABLES (not summarized!)
- Full data PRESERVED - LLM can access any part anytime

This is NOT summarization - data is kept intact and accessible.

Performance Optimizations:
- Async file collection with parallel I/O
- Inverted tag index for O(1) memory lookups
- Connection pooling for LLM calls
- Graceful shutdown with resource cleanup
- Structured logging with timing

Tools provided:
- rlm_analyze: Analyze files/directories using RLM
- rlm_query_text: Process large text using RLM
- rlm_status: Check server health and configuration
- rlm_memory_store: Store findings for later recall
- rlm_memory_recall: Recall stored information
"""

import asyncio
import atexit
import json
import signal
import sys
import time
from typing import Any
from collections import defaultdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

from .config import get_config, RLMConfig, ServerConfig
from .file_collector import FileCollector
from .rlm_processor import RLMProcessor, SemanticCache, ProgressEvent
from .cache_manager import CacheManager
from .utils import get_memory_monitor, get_metrics_collector
from .result_verifier import ResultVerifier, BatchVerifier
from .project_analyzer import ProjectAnalyzer
from .structured_tools import (
    StructuredTools, ToolResult,
    # v2.6: Single-file tools
    read_file, grep_pattern, glob_files,
    FileContent, GrepResult, GlobResult,
)

# Import handlers from refactored modules
from .handlers import (
    handle_rlm_read,
    handle_rlm_grep,
    handle_rlm_glob,
    handle_memory_store,
    handle_memory_recall,
    get_memory_store,
    get_tag_index,
    get_memory_count,
)
from .handlers.query import (
    handle_rlm_analyze,
    handle_rlm_query_text,
    handle_rlm_status,
)


# Input validation constants
MAX_QUERY_LENGTH = 50_000  # 50K chars max for query
MAX_PATHS_COUNT = 100  # Max number of paths in a single request
MAX_PATH_LENGTH = 4096  # Max length of a single path
MAX_TEXT_LENGTH = 10_000_000  # 10MB max for text input
MAX_MEMORY_KEY_LENGTH = 256  # Max length for memory keys
MAX_MEMORY_VALUE_LENGTH = 1_000_000  # 1MB max for memory values
MAX_MEMORY_ENTRIES = 10_000  # Max number of memory entries
MAX_TAGS_PER_ENTRY = 50  # Max tags per memory entry


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


def _extract_signatures_from_file(file_path: str) -> list[str]:
    """
    Extract function/class/struct signatures from a file without loading full content.

    Uses regex to find declarations, reading only the first portion of the file.

    Args:
        file_path: Path to the file

    Returns:
        List of signature strings (e.g., "func fetchData()", "class UserManager")
    """
    import re
    signatures = []
    try:
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            return []

        ext = path.suffix.lower()

        # Read first 50KB to extract signatures (enough for most files' declarations)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(50000)

        # Swift patterns
        if ext == '.swift':
            # Classes, structs, enums, protocols
            for match in re.findall(r'^\s*((?:public|private|internal|open|final)\s+)?(class|struct|enum|protocol|actor)\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"{match[1]} {match[2]}")
            # Functions
            for match in re.findall(r'^\s*((?:public|private|internal|open|override)\s+)?func\s+(\w+)\s*\([^)]*\)', content, re.MULTILINE):
                signatures.append(f"func {match[1]}()")

        # Python patterns
        elif ext == '.py':
            for match in re.findall(r'^(class|def|async def)\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"{match[0]} {match[1]}")

        # JavaScript/TypeScript patterns
        elif ext in ('.js', '.ts', '.tsx', '.jsx'):
            # Classes
            for match in re.findall(r'^(?:export\s+)?class\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"class {match}")
            # Functions
            for match in re.findall(r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"function {match}()")
            # Arrow functions assigned to const
            for match in re.findall(r'^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', content, re.MULTILINE):
                signatures.append(f"const {match} = () =>")

        # Go patterns
        elif ext == '.go':
            for match in re.findall(r'^func\s+(?:\([^)]+\)\s+)?(\w+)', content, re.MULTILINE):
                signatures.append(f"func {match}()")
            for match in re.findall(r'^type\s+(\w+)\s+(struct|interface)', content, re.MULTILINE):
                signatures.append(f"type {match[0]} {match[1]}")

        # Rust patterns
        elif ext == '.rs':
            for match in re.findall(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"fn {match}()")
            for match in re.findall(r'^(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"struct/enum {match}")

    except Exception:
        pass

    return signatures[:20]  # Limit to 20 signatures


def _format_skipped_files_summary(
    skipped_files: list[str],
    max_display: int = 10,
    include_signatures: bool = False
) -> str:
    """
    Format a summary of skipped files for user visibility.

    Args:
        skipped_files: List of skipped file paths with reasons
        max_display: Maximum number of skipped files to show
        include_signatures: Whether to extract and show function/class signatures

    Returns:
        Formatted markdown string with skipped files summary
    """
    if not skipped_files:
        return ""

    # Group by reason
    by_reason: dict[str, list[str]] = {}
    for entry in skipped_files:
        # Parse "path (reason)" format
        if " (" in entry and entry.endswith(")"):
            path, reason = entry.rsplit(" (", 1)
            reason = reason.rstrip(")")
        else:
            path = entry
            reason = "unknown"

        if reason not in by_reason:
            by_reason[reason] = []
        by_reason[reason].append(path)

    lines = [f"\n\n## Skipped Files ({len(skipped_files)} total)"]

    for reason, paths in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        count = len(paths)
        lines.append(f"\n**{reason}** ({count} files)")

        for path in paths[:max_display]:
            # Show file EXISTS confirmation
            from pathlib import Path
            file_exists = Path(path).exists() if not path.startswith("(") else False
            existence = "EXISTS" if file_exists else "NOT FOUND"
            lines.append(f"  - `{path}` [{existence}]")

            # Extract signatures if requested
            if include_signatures and file_exists:
                signatures = _extract_signatures_from_file(path)
                if signatures:
                    lines.append(f"    Signatures: {', '.join(signatures[:5])}")
                    if len(signatures) > 5:
                        lines.append(f"    ... and {len(signatures) - 5} more")

        if count > max_display:
            lines.append(f"  - ... and {count - max_display} more")

    lines.append("\n*Tip: Use `include_skipped_signatures: true` to see function/class names in skipped files.*")

    return "\n".join(lines)


# Global instances
_rlm_config: RLMConfig | None = None
_server_config: ServerConfig | None = None
_file_collector: FileCollector | None = None
_rlm_processor: RLMProcessor | None = None
_cache_manager: CacheManager | None = None
_semantic_cache: SemanticCache | None = None
_result_verifier: ResultVerifier | None = None
_project_analyzer: ProjectAnalyzer | None = None

# Memory store with inverted tag index for O(1) lookups
_memory_store: dict[str, Any] = {}
_tag_index: dict[str, set[str]] = defaultdict(set)  # tag -> set of keys

# Shutdown flag for graceful termination
_shutdown_event: asyncio.Event | None = None

# Memory monitor for resource management
_memory_monitor = get_memory_monitor(max_bytes=2_000_000_000)  # 2GB limit


def get_instances() -> tuple[RLMConfig, ServerConfig, FileCollector, RLMProcessor, CacheManager, SemanticCache, ResultVerifier, ProjectAnalyzer]:
    """Get or create singleton instances."""
    global _rlm_config, _server_config, _file_collector, _rlm_processor, _cache_manager
    global _semantic_cache, _result_verifier, _project_analyzer

    if _rlm_config is None:
        _rlm_config, _server_config = get_config()
        _cache_manager = CacheManager(_rlm_config)
        _file_collector = FileCollector(_rlm_config)
        _rlm_processor = RLMProcessor(_rlm_config, _cache_manager)
        _semantic_cache = SemanticCache(similarity_threshold=0.85, max_size=100)
        _result_verifier = ResultVerifier(strict_mode=False)
        _project_analyzer = ProjectAnalyzer()

    return _rlm_config, _server_config, _file_collector, _rlm_processor, _cache_manager, _semantic_cache, _result_verifier, _project_analyzer


async def cleanup_resources() -> None:
    """Cleanup resources on shutdown."""
    global _rlm_processor

    if _rlm_processor is not None:
        try:
            await _rlm_processor.close()
        except Exception as e:
            print(f"Error closing RLM processor: {e}", file=sys.stderr)


def _log_timing(operation: str, start_time: float, **extra: Any) -> None:
    """Log operation timing (structured logging)."""
    elapsed_ms = int((time.time() - start_time) * 1000)
    log_data = {"operation": operation, "elapsed_ms": elapsed_ms, **extra}
    # Only log in debug mode to avoid noise
    # print(json.dumps(log_data), file=sys.stderr)


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("rlm-recursive-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="rlm_analyze",
                description=(
                    "Analyze files/directories using TRUE RLM technique (arXiv:2512.24601). "
                    "Content stored as variable, LLM writes code to examine portions. "
                    "Use for: large codebases (50+ files), security audits, architecture reviews. "
                    "Query types auto-detected: security, ios, python, javascript, api, database, quality, architecture, testing."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "CRITICAL: Be specific about WHAT to find and HOW to report it. "
                                "Your query drives the Python code generated to search.\n\n"
                                "QUERY PATTERNS BY TYPE:\n"
                                "• SECURITY: 'Find (1) SQL injection via string concat (2) hardcoded secrets matching sk-, api_key, password (3) eval/exec with user input. Report: file:line, code, severity.'\n"
                                "• iOS/SWIFT: 'Find (1) force unwraps (!) excluding != (2) closures missing [weak self] (3) @ObservedObject with default value. Report: file:line, code, fix.'\n"
                                "• PYTHON: 'Find (1) pickle.loads with untrusted data (2) bare except clauses (3) mutable default args. Report: file:line, code.'\n"
                                "• JAVASCRIPT: 'Find (1) innerHTML XSS (2) missing await (3) useEffect missing deps. Report: file:line, code.'\n"
                                "• API: 'Find (1) endpoints missing auth (2) SQL injection in params (3) missing rate limiting. Report: file:line, code.'\n"
                                "• ARCHITECTURE: 'Map all modules with purpose, entry points, dependencies, data flow.'\n\n"
                                "BAD: 'find problems' or 'check security' (too vague)\n"
                                "GOOD: Numbered list of specific patterns + output format"
                            ),
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File or directory paths to analyze. Use ['.'] for current directory.",
                        },
                        "query_mode": {
                            "type": "string",
                            "enum": ["auto", "semantic", "scanner", "literal", "custom"],
                            "description": (
                                "How to interpret and execute the query. Options:\n"
                                "• 'auto' (default): Auto-detect best mode based on query complexity\n"
                                "• 'semantic': LLM interprets query and writes custom search code (TRUE RLM)\n"
                                "• 'scanner': Use pre-built scanners only (ios, security, quality)\n"
                                "• 'literal': Fast grep-style literal search for quoted strings (no LLM)\n"
                                "• 'custom': Query-driven semantic analysis WITHOUT pre-built scanners"
                            ),
                        },
                        "scan_mode": {
                            "type": "string",
                            "enum": ["auto", "ios", "ios-strict", "security", "quality", "web", "rust", "node", "frontend", "backend", "all", "custom"],
                            "description": (
                                "Pre-configured scan mode (used when query_mode='scanner' or 'auto'). Options:\n"
                                "• 'auto' (default): Auto-detect based on query and file types\n"
                                "• 'ios': Run iOS/Swift scanners (security + crash issues)\n"
                                "• 'ios-strict': iOS scan with HIGH confidence only (minimal noise)\n"
                                "• 'security': Run security scanners (secrets, injection, XSS)\n"
                                "• 'quality': Run code quality scanners (long functions, TODOs)\n"
                                "• 'web': Run web/frontend scanners (React, Vue, Angular, DOM, a11y, CSS)\n"
                                "• 'rust': Run Rust scanners (unsafe, unwrap, concurrency, clippy)\n"
                                "• 'node': Run Node.js scanners (callbacks, promises, security, async)\n"
                                "• 'frontend': Combined web + node scanners\n"
                                "• 'backend': Combined node + security scanners\n"
                                "• 'all': Run all scanners including quality checks\n"
                                "• 'custom': Skip all pre-built scanners, use query-driven analysis only"
                            ),
                        },
                        "min_confidence": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                            "description": (
                                "Minimum confidence level for findings. Options:\n"
                                "• 'LOW': Include all findings (comprehensive but noisy)\n"
                                "• 'MEDIUM' (default): Filter out low-confidence noise\n"
                                "• 'HIGH': Only high-confidence, verified findings"
                            ),
                        },
                        "include_quality": {
                            "type": "boolean",
                            "description": (
                                "Include code quality checks (long functions, TODOs). "
                                "Default: false. Quality checks are excluded by default to focus on bugs/security."
                            ),
                        },
                        "include_skipped_signatures": {
                            "type": "boolean",
                            "description": (
                                "Extract function/class signatures from files that were skipped due to size limits. "
                                "This provides visibility into large files without loading full content. "
                                "Default: false."
                            ),
                        },
                    },
                    "required": ["query", "paths"],
                },
            ),
            Tool(
                name="rlm_query_text",
                description=(
                    "Process large text using TRUE RLM technique. "
                    "Text stored as `prompt` variable, LLM writes code to examine it. "
                    "Use for: logs, transcripts, documents, configs, any large text input."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "CRITICAL: Be specific about WHAT to extract and HOW to format output.\n\n"
                                "QUERY PATTERNS BY TEXT TYPE:\n"
                                "• LOGS: 'Extract (1) ERROR/WARN entries with timestamps (2) stack traces with root cause (3) error frequency by type. Format: timestamp | level | message | count'\n"
                                "• CONFIG: 'Extract (1) all environment variables (2) connection strings (3) feature flags. Format: key = value with file location'\n"
                                "• TRANSCRIPT: 'Extract (1) key decisions made (2) action items with owners (3) unresolved questions. Format: bullet points with timestamps'\n"
                                "• JSON/DATA: 'Extract (1) all unique field names (2) data types per field (3) nested structure depth. Format: field: type (count)'\n\n"
                                "BAD: 'summarize this' (too vague)\n"
                                "GOOD: Numbered extraction criteria + output format"
                            ),
                        },
                        "text": {
                            "type": "string",
                            "description": "Large text content to process",
                        },
                    },
                    "required": ["query", "text"],
                },
            ),
            Tool(
                name="rlm_status",
                description=(
                    "Check RLM server health, configuration, and cache statistics."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="rlm_memory_store",
                description=(
                    "Store important information for later recall across conversations. "
                    "Use this to persist key findings, summaries, or context that should "
                    "be remembered."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Unique identifier for this memory",
                        },
                        "value": {
                            "type": "string",
                            "description": "Content to store",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                        },
                    },
                    "required": ["key", "value"],
                },
            ),
            Tool(
                name="rlm_memory_recall",
                description=(
                    "Recall stored information by key or search by tags."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Exact key to recall",
                        },
                        "search_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to search for",
                        },
                    },
                },
            ),
            # ===== v2.6: SINGLE-FILE TOOLS =====
            # These replace Claude's native Read, Grep, and Glob tools
            Tool(
                name="rlm_read",
                description=(
                    "Read a single file with optional line offset and limit. "
                    "Fast, no LLM overhead. Replaces Claude's native Read tool. "
                    "Returns file content with line numbers in format: '  123→content'"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file to read",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start from (0-indexed). Default: 0",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read. Default: all lines",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="rlm_grep",
                description=(
                    "Search for a pattern in files using ripgrep. "
                    "Fast regex or literal search. Replaces Claude's native Grep tool. "
                    "Returns matches with file paths, line numbers, and content."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for (or literal string if fixed_strings=true)",
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory to search in. Default: current directory",
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Ignore case when searching. Default: false",
                        },
                        "fixed_strings": {
                            "type": "boolean",
                            "description": "Treat pattern as literal string, not regex. Default: false",
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context before and after match. Default: 0",
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Limit to file type (e.g., 'py', 'js', 'swift', 'rs')",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern filter (e.g., '*.tsx', '**/*.swift')",
                        },
                        "output_mode": {
                            "type": "string",
                            "enum": ["content", "files_with_matches", "count"],
                            "description": (
                                "Output mode: 'content' shows matching lines, "
                                "'files_with_matches' shows only file paths, "
                                "'count' shows match counts per file. Default: 'content'"
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
            ),
            Tool(
                name="rlm_glob",
                description=(
                    "Find files matching a glob pattern. "
                    "Fast filesystem search. Replaces Claude's native Glob tool. "
                    "Returns list of matching file paths."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.swift', '*.json')",
                        },
                        "path": {
                            "type": "string",
                            "description": "Base directory to search from. Default: current directory",
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "Include hidden files/directories. Default: false",
                        },
                    },
                    "required": ["pattern"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        start_time = time.time()
        try:
            if name == "rlm_analyze":
                result = await handle_rlm_analyze(arguments, get_instances)
            elif name == "rlm_query_text":
                result = await handle_rlm_query_text(arguments, get_instances)
            elif name == "rlm_status":
                result = await handle_rlm_status(
                    arguments, get_instances, _memory_monitor, get_metrics_collector()
                )
            elif name == "rlm_memory_store":
                # Get file_collector for token counting
                _, _, file_collector, _, _, _, _, _ = get_instances()
                result = await handle_memory_store(arguments, file_collector.count_tokens)
            elif name == "rlm_memory_recall":
                result = await handle_memory_recall(arguments)
            # v2.6: Single-file tools
            elif name == "rlm_read":
                result = await handle_rlm_read(arguments)
            elif name == "rlm_grep":
                result = await handle_rlm_grep(arguments)
            elif name == "rlm_glob":
                result = await handle_rlm_glob(arguments)
            else:
                result = [TextContent(type="text", text=f"Unknown tool: {name}")]

            _log_timing(f"tool:{name}", start_time, success=True)
            return result

        except Exception as e:
            _log_timing(f"tool:{name}", start_time, success=False, error=str(e))
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


# =============================================================================
# NOTE: The following have been moved to refactored modules:
# - handlers/query.py: handle_rlm_analyze, handle_rlm_query_text, handle_rlm_status
# - handlers/memory.py: handle_memory_store, handle_memory_recall, write_progress
# - handlers/files.py: handle_rlm_read, handle_rlm_grep, handle_rlm_glob
# - mcp_tools.py: Literal search, LLM routing, tool execution
# =============================================================================


async def run_server():
    """Run the MCP server with graceful shutdown."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    server = create_server()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def handle_shutdown(sig):
        print(f"\nReceived {sig.name}, shutting down gracefully...", file=sys.stderr)
        _shutdown_event.set()

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    try:
        async with stdio_server() as (read_stream, write_stream):
            # Run server until shutdown signal
            server_task = asyncio.create_task(
                server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
            )

            # Wait for either server completion or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, asyncio.create_task(_shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    finally:
        # Cleanup resources
        await cleanup_resources()


def main():
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
