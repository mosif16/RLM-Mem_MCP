"""
Memory Handlers for RLM-Mem MCP Server.

Provides handlers for memory store/recall operations:
- rlm_memory_store: Store key-value pairs with tags
- rlm_memory_recall: Recall by key or search by tags
"""

import json
import time
from typing import Any
from collections import defaultdict

from mcp.types import TextContent


# Input validation constants
MAX_MEMORY_KEY_LENGTH = 256
MAX_MEMORY_VALUE_LENGTH = 1_000_000  # 1MB
MAX_MEMORY_ENTRIES = 10_000
MAX_TAGS_PER_ENTRY = 50


# Memory store with inverted tag index for O(1) lookups
_memory_store: dict[str, Any] = {}
_tag_index: dict[str, set[str]] = defaultdict(set)


def get_memory_store() -> dict[str, Any]:
    """Get the global memory store."""
    return _memory_store


def get_tag_index() -> dict[str, set[str]]:
    """Get the global tag index."""
    return _tag_index


def get_memory_count() -> int:
    """Get the number of entries in memory store."""
    return len(_memory_store)


async def handle_memory_store(
    arguments: dict[str, Any],
    token_counter: Any = None
) -> list[TextContent]:
    """
    Handle rlm_memory_store tool call with inverted tag index.

    Args:
        arguments: Tool arguments (key, value, tags)
        token_counter: Optional callable to count tokens
    """
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

    # Remove old tags from index if key exists
    if key in _memory_store:
        old_tags = _memory_store[key].get("tags", [])
        for tag in old_tags:
            _tag_index[tag].discard(key)

    # Store in memory
    entry = {
        "value": value,
        "tags": tags,
        "stored_at": time.time(),
    }

    # Add token count if counter provided
    if token_counter:
        entry["token_count"] = token_counter(value)

    _memory_store[key] = entry

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


async def write_progress(
    session_id: str,
    event_type: str,
    message: str,
    progress: float,
    details: dict = None
) -> None:
    """
    Write progress event to memory store for client polling.

    This is used by handlers to report progress during long operations.
    """
    global _memory_store, _tag_index

    if not session_id:
        return

    progress_data = {
        "event_type": event_type,
        "message": message,
        "progress_percent": progress,
        "timestamp": time.time(),
        "details": details or {}
    }

    key = f"progress:{session_id}"
    _memory_store[key] = {
        "value": json.dumps(progress_data),
        "tags": ["progress", "active"],
        "timestamp": time.time()
    }
    # Update tag index
    for tag in ["progress", "active"]:
        _tag_index[tag].add(key)
