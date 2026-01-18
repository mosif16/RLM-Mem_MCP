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
from .rlm_processor import RLMProcessor
from .cache_manager import CacheManager
from .utils import get_memory_monitor, get_metrics_collector


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


# Global instances
_rlm_config: RLMConfig | None = None
_server_config: ServerConfig | None = None
_file_collector: FileCollector | None = None
_rlm_processor: RLMProcessor | None = None
_cache_manager: CacheManager | None = None

# Memory store with inverted tag index for O(1) lookups
_memory_store: dict[str, Any] = {}
_tag_index: dict[str, set[str]] = defaultdict(set)  # tag -> set of keys

# Shutdown flag for graceful termination
_shutdown_event: asyncio.Event | None = None

# Memory monitor for resource management
_memory_monitor = get_memory_monitor(max_bytes=2_000_000_000)  # 2GB limit


def get_instances() -> tuple[RLMConfig, ServerConfig, FileCollector, RLMProcessor, CacheManager]:
    """Get or create singleton instances."""
    global _rlm_config, _server_config, _file_collector, _rlm_processor, _cache_manager

    if _rlm_config is None:
        _rlm_config, _server_config = get_config()
        _cache_manager = CacheManager(_rlm_config)
        _file_collector = FileCollector(_rlm_config)
        _rlm_processor = RLMProcessor(_rlm_config, _cache_manager)

    return _rlm_config, _server_config, _file_collector, _rlm_processor, _cache_manager


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
                    "Sub-LLM responses stored in full (NOT summarized). "
                    "Use for: large codebases (50+ files), security audits, architecture reviews."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "CRITICAL: Be exhaustively specific. Your query drives the code written to search. "
                                "BAD: 'find security issues' "
                                "GOOD: 'Find security vulnerabilities: (1) INJECTION - SQL via string concat, "
                                "command via subprocess/os.system, code via eval/exec (2) SECRETS - hardcoded API keys, "
                                "passwords, tokens (3) PATH TRAVERSAL - user input in file paths (4) DESERIALIZATION - "
                                "pickle.loads, unsafe yaml.load. For each: file, line, code snippet, severity.'"
                            ),
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File or directory paths to analyze",
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
                    "Use for: logs, transcripts, documents, any large text input."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "CRITICAL: Be exhaustively specific. Your query drives the code written to search. "
                                "BAD: 'summarize this' "
                                "GOOD: 'Extract from these logs: (1) all ERROR and WARN entries with timestamps "
                                "(2) stack traces with root cause (3) frequency of each error type (4) time periods "
                                "with highest error rates. Format as: timestamp | level | message | count'"
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
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        start_time = time.time()
        try:
            if name == "rlm_analyze":
                result = await handle_rlm_analyze(arguments)
            elif name == "rlm_query_text":
                result = await handle_rlm_query_text(arguments)
            elif name == "rlm_status":
                result = await handle_rlm_status(arguments)
            elif name == "rlm_memory_store":
                result = await handle_memory_store(arguments)
            elif name == "rlm_memory_recall":
                result = await handle_memory_recall(arguments)
            else:
                result = [TextContent(type="text", text=f"Unknown tool: {name}")]

            _log_timing(f"tool:{name}", start_time, success=True)
            return result

        except Exception as e:
            _log_timing(f"tool:{name}", start_time, success=False, error=str(e))
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def handle_rlm_analyze(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_analyze tool call."""
    start_time = time.time()

    query = arguments.get("query", "")
    paths = arguments.get("paths", [])

    # Input validation
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    if len(query) > MAX_QUERY_LENGTH:
        return [TextContent(type="text", text=f"Error: query too long ({len(query)} > {MAX_QUERY_LENGTH} chars)")]

    if not paths:
        return [TextContent(type="text", text="Error: paths is required")]

    if not isinstance(paths, list):
        return [TextContent(type="text", text="Error: paths must be a list")]

    if len(paths) > MAX_PATHS_COUNT:
        return [TextContent(type="text", text=f"Error: too many paths ({len(paths)} > {MAX_PATHS_COUNT})")]

    # Validate each path
    for path in paths:
        if not isinstance(path, str):
            return [TextContent(type="text", text=f"Error: path must be string, got {type(path).__name__}")]
        is_valid, error = validate_path(path)
        if not is_valid:
            return [TextContent(type="text", text=f"Error: invalid path '{path}': {error}")]

    rlm_config, _, file_collector, rlm_processor, _ = get_instances()

    # Validate config
    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    # Collect files using async method
    _log_timing("file_collection:start", start_time, path_count=len(paths))
    collection = await file_collector.collect_paths_async(paths)
    _log_timing("file_collection:complete", start_time, file_count=collection.file_count)

    if collection.file_count == 0:
        error_msg = "No matching files found"
        if collection.errors:
            error_msg += f": {'; '.join(collection.errors)}"
        return [TextContent(type="text", text=error_msg)]

    # Analyze query and use decomposition for broad queries
    query_analysis = rlm_processor.analyze_query_quality(query)
    _log_timing("query_analysis", start_time, is_broad=query_analysis["is_broad"], query_type=query_analysis["query_type"])

    # Process with RLM (auto-decompose if broad)
    _log_timing("rlm_process:start", start_time)
    result = await rlm_processor.process_with_decomposition(query, collection)
    _log_timing("rlm_process:complete", start_time, chunks=len(result.chunk_results))

    # Format output
    output = rlm_processor.format_result(result)

    return [TextContent(type="text", text=output)]


async def handle_rlm_query_text(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_query_text tool call."""
    start_time = time.time()

    query = arguments.get("query", "")
    text = arguments.get("text", "")

    # Input validation
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    if len(query) > MAX_QUERY_LENGTH:
        return [TextContent(type="text", text=f"Error: query too long ({len(query)} > {MAX_QUERY_LENGTH} chars)")]

    if not text:
        return [TextContent(type="text", text="Error: text is required")]

    if len(text) > MAX_TEXT_LENGTH:
        return [TextContent(type="text", text=f"Error: text too long ({len(text)} > {MAX_TEXT_LENGTH} chars)")]

    rlm_config, _, file_collector, rlm_processor, _ = get_instances()

    # Validate config
    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    # Create collection from text (use async version)
    collection = await file_collector.collect_text_async(text, "input_text")

    # Process with RLM (auto-decompose if broad)
    _log_timing("rlm_process:start", start_time)
    result = await rlm_processor.process_with_decomposition(query, collection)
    _log_timing("rlm_process:complete", start_time)

    # Format output
    output = rlm_processor.format_result(result)

    return [TextContent(type="text", text=output)]


async def handle_rlm_status(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_status tool call."""
    rlm_config, server_config, file_collector, rlm_processor, _ = get_instances()

    # Get memory and metrics stats
    memory_stats = _memory_monitor.get_stats()
    metrics_stats = get_metrics_collector().get_stats()

    status = {
        "server": {
            "name": server_config.name,
            "version": server_config.version,
        },
        "configuration": {
            "model": rlm_config.model,
            "aggregator_model": rlm_config.aggregator_model,
            "api_base_url": rlm_config.api_base_url,
            "api_key_set": bool(rlm_config.api_key),
            "max_result_tokens": rlm_config.max_result_tokens,
            "max_chunk_tokens": rlm_config.max_chunk_tokens,
        },
        "memory_entries": len(_memory_store),
        "performance": {
            "token_cache": file_collector.get_cache_stats(),
            "processor": rlm_processor.get_stats(),
            "metrics": metrics_stats,
        },
        "resources": {
            "memory": memory_stats,
        },
    }

    # Add warnings if memory is high
    if memory_stats.get("usage_percent", 0) > 0.8:
        status["warnings"] = ["Memory usage above 80% threshold"]

    # Validate configuration
    errors = rlm_config.validate()
    if errors:
        status["errors"] = errors

    output = json.dumps(status, indent=2)
    return [TextContent(type="text", text=output)]


async def handle_memory_store(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_memory_store tool call with inverted tag index."""
    global _memory_store, _tag_index

    key = arguments.get("key", "")
    value = arguments.get("value", "")
    tags = arguments.get("tags", [])

    # Input validation
    if not key:
        return [TextContent(type="text", text="Error: key is required")]

    if len(key) > MAX_MEMORY_KEY_LENGTH:
        return [TextContent(type="text", text=f"Error: key too long ({len(key)} > {MAX_MEMORY_KEY_LENGTH} chars)")]

    if not value:
        return [TextContent(type="text", text="Error: value is required")]

    if len(value) > MAX_MEMORY_VALUE_LENGTH:
        return [TextContent(type="text", text=f"Error: value too long ({len(value)} > {MAX_MEMORY_VALUE_LENGTH} chars)")]

    # Validate tags
    if not isinstance(tags, list):
        return [TextContent(type="text", text="Error: tags must be a list")]

    if len(tags) > MAX_TAGS_PER_ENTRY:
        return [TextContent(type="text", text=f"Error: too many tags ({len(tags)} > {MAX_TAGS_PER_ENTRY})")]

    for tag in tags:
        if not isinstance(tag, str) or len(tag) > 100:
            return [TextContent(type="text", text="Error: tags must be strings with max 100 chars")]

    # Check memory store size limit (only for new keys)
    if key not in _memory_store and len(_memory_store) >= MAX_MEMORY_ENTRIES:
        return [TextContent(type="text", text=f"Error: memory store full ({MAX_MEMORY_ENTRIES} entries max)")]

    _, _, file_collector, _, _ = get_instances()

    # Remove old tags from index if key exists
    if key in _memory_store:
        old_tags = _memory_store[key].get("tags", [])
        for tag in old_tags:
            _tag_index[tag].discard(key)

    # Store in memory
    _memory_store[key] = {
        "value": value,
        "tags": tags,
        "token_count": file_collector.count_tokens(value),
        "stored_at": time.time(),
    }

    # Update inverted tag index for O(1) lookup
    for tag in tags:
        _tag_index[tag].add(key)

    return [TextContent(
        type="text",
        text=f"Stored memory with key '{key}' ({len(value)} chars, {len(tags)} tags)"
    )]


async def handle_memory_recall(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_memory_recall tool call with O(1) tag lookup."""
    key = arguments.get("key")
    search_tags = arguments.get("search_tags", [])

    results = []

    # Exact key lookup - O(1)
    if key:
        if key in _memory_store:
            entry = _memory_store[key]
            results.append({
                "key": key,
                "value": entry["value"],
                "tags": entry["tags"],
            })
        else:
            return [TextContent(type="text", text=f"No memory found with key '{key}'")]

    # Tag search using inverted index - O(tags * avg_keys_per_tag) instead of O(n)
    elif search_tags:
        matching_keys: set[str] = set()

        for tag in search_tags:
            if tag in _tag_index:
                matching_keys.update(_tag_index[tag])

        for k in matching_keys:
            if k in _memory_store:
                entry = _memory_store[k]
                results.append({
                    "key": k,
                    "value": entry["value"],
                    "tags": entry["tags"],
                })

        if not results:
            return [TextContent(
                type="text",
                text=f"No memories found with tags: {search_tags}"
            )]

    # List all if no filters
    else:
        for k, entry in _memory_store.items():
            results.append({
                "key": k,
                "value": entry["value"][:200] + "..." if len(entry["value"]) > 200 else entry["value"],
                "tags": entry["tags"],
            })

        if not results:
            return [TextContent(type="text", text="No memories stored")]

    output = json.dumps(results, indent=2)
    return [TextContent(type="text", text=output)]


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
