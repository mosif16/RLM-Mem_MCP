#!/usr/bin/env python3
"""
RLM-Mem MCP Server

An MCP server implementing the TRUE RLM technique from arXiv:2512.24601.

KEY INSIGHT: Content is stored as a VARIABLE, not in LLM context.
- LLM writes CODE to examine portions of the content
- Sub-LLM responses stored in VARIABLES (not summarized!)
- Full data PRESERVED - LLM can access any part anytime

This is NOT summarization - data is kept intact and accessible.

Tools provided:
- rlm_analyze: Analyze files/directories using RLM
- rlm_query_text: Process large text using RLM
- rlm_status: Check server health and configuration
- rlm_memory_store: Store findings for later recall
- rlm_memory_recall: Recall stored information
"""

import asyncio
import json
import sys
from typing import Any

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


# Global instances
_rlm_config: RLMConfig | None = None
_server_config: ServerConfig | None = None
_file_collector: FileCollector | None = None
_rlm_processor: RLMProcessor | None = None
_cache_manager: CacheManager | None = None
_memory_store: dict[str, Any] = {}


def get_instances() -> tuple[RLMConfig, ServerConfig, FileCollector, RLMProcessor, CacheManager]:
    """Get or create singleton instances."""
    global _rlm_config, _server_config, _file_collector, _rlm_processor, _cache_manager

    if _rlm_config is None:
        _rlm_config, _server_config = get_config()
        _cache_manager = CacheManager(_rlm_config)
        _file_collector = FileCollector(_rlm_config)
        _rlm_processor = RLMProcessor(_rlm_config, _cache_manager)

    return _rlm_config, _server_config, _file_collector, _rlm_processor, _cache_manager


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
                                "What to find/analyze. Be specific. "
                                "Examples: 'find SQL injection vulnerabilities', "
                                "'list all API endpoints with their HTTP methods', "
                                "'find functions that handle user authentication'"
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
                            "description": "What to find/extract from the text",
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
        try:
            if name == "rlm_analyze":
                return await handle_rlm_analyze(arguments)
            elif name == "rlm_query_text":
                return await handle_rlm_query_text(arguments)
            elif name == "rlm_status":
                return await handle_rlm_status(arguments)
            elif name == "rlm_memory_store":
                return await handle_memory_store(arguments)
            elif name == "rlm_memory_recall":
                return await handle_memory_recall(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def handle_rlm_analyze(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_analyze tool call."""
    query = arguments.get("query", "")
    paths = arguments.get("paths", [])

    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    if not paths:
        return [TextContent(type="text", text="Error: paths is required")]

    rlm_config, _, file_collector, rlm_processor, _ = get_instances()

    # Validate config
    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    # Collect files
    collection = file_collector.collect_paths(paths)

    if collection.file_count == 0:
        error_msg = "No matching files found"
        if collection.errors:
            error_msg += f": {'; '.join(collection.errors)}"
        return [TextContent(type="text", text=error_msg)]

    # Process with RLM
    result = await rlm_processor.process(query, collection)

    # Format output
    output = rlm_processor.format_result(result)

    return [TextContent(type="text", text=output)]


async def handle_rlm_query_text(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_query_text tool call."""
    query = arguments.get("query", "")
    text = arguments.get("text", "")

    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    if not text:
        return [TextContent(type="text", text="Error: text is required")]

    rlm_config, _, file_collector, rlm_processor, _ = get_instances()

    # Validate config
    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    # Create collection from text
    collection = file_collector.collect_text(text, "input_text")

    # Process with RLM
    result = await rlm_processor.process(query, collection)

    # Format output
    output = rlm_processor.format_result(result)

    return [TextContent(type="text", text=output)]


async def handle_rlm_status(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_status tool call."""
    rlm_config, server_config, _, _, cache_manager = get_instances()

    # Check if RLM library is available
    rlm_installed = False
    try:
        import rlm
        rlm_installed = True
    except ImportError:
        pass

    # Check if Claude Agent SDK is available
    agent_sdk_installed = False
    agent_sdk_version = None
    try:
        import claude_agent_sdk
        agent_sdk_installed = True
        agent_sdk_version = getattr(claude_agent_sdk, "__version__", "unknown")
    except ImportError:
        pass

    status = {
        "server": {
            "name": server_config.name,
            "version": server_config.version,
        },
        "configuration": {
            "model": rlm_config.model,
            "aggregator_model": rlm_config.aggregator_model,
            "backend": rlm_config.backend,
            "api_key_set": bool(rlm_config.api_key),
            "max_result_tokens": rlm_config.max_result_tokens,
            "max_chunk_tokens": rlm_config.max_chunk_tokens,
            "use_agent_sdk": rlm_config.use_agent_sdk,
        },
        "cache": {
            "enabled": rlm_config.use_cache,
            "ttl": rlm_config.cache_ttl,
            **cache_manager.get_stats(),
        },
        "agent_sdk": {
            "installed": agent_sdk_installed,
            "version": agent_sdk_version,
            "enabled": rlm_config.use_agent_sdk and agent_sdk_installed,
        },
        "rlm_library_installed": rlm_installed,
        "memory_entries": len(_memory_store),
        "claude_max_optimized": True,  # Using Haiku 4.5 by default
    }

    # Validate configuration
    errors = rlm_config.validate()
    if errors:
        status["errors"] = errors

    output = json.dumps(status, indent=2)
    return [TextContent(type="text", text=output)]


async def handle_memory_store(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_memory_store tool call."""
    global _memory_store

    key = arguments.get("key", "")
    value = arguments.get("value", "")
    tags = arguments.get("tags", [])

    if not key:
        return [TextContent(type="text", text="Error: key is required")]

    if not value:
        return [TextContent(type="text", text="Error: value is required")]

    _, _, file_collector, _, _ = get_instances()

    # Store in memory
    _memory_store[key] = {
        "value": value,
        "tags": tags,
        "token_count": file_collector.count_tokens(value),
    }

    return [TextContent(
        type="text",
        text=f"Stored memory with key '{key}' ({len(value)} chars, {len(tags)} tags)"
    )]


async def handle_memory_recall(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_memory_recall tool call."""
    key = arguments.get("key")
    search_tags = arguments.get("search_tags", [])

    results = []

    # Exact key lookup
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

    # Tag search
    elif search_tags:
        for k, entry in _memory_store.items():
            if any(tag in entry.get("tags", []) for tag in search_tags):
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
    """Run the MCP server."""
    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


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
