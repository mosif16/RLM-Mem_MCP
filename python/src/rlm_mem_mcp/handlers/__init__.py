"""
Request Handlers for RLM-Mem MCP Server.

This package contains the tool handlers extracted from server.py:
- query: Main analysis handlers (rlm_analyze, rlm_query_text, rlm_status)
- memory: Memory store/recall handlers
- files: Single-file tool handlers (rlm_read, rlm_grep, rlm_glob)
"""

from .files import (
    handle_rlm_read,
    handle_rlm_grep,
    handle_rlm_glob,
)
from .memory import (
    handle_memory_store,
    handle_memory_recall,
    write_progress,
    get_memory_store,
    get_tag_index,
    get_memory_count,
)
from .query import (
    handle_rlm_analyze,
    handle_rlm_query_text,
    handle_rlm_status,
)

__all__ = [
    # Query handlers
    "handle_rlm_analyze",
    "handle_rlm_query_text",
    "handle_rlm_status",
    # File handlers
    "handle_rlm_read",
    "handle_rlm_grep",
    "handle_rlm_glob",
    # Memory handlers
    "handle_memory_store",
    "handle_memory_recall",
    "write_progress",
    "get_memory_store",
    "get_tag_index",
    "get_memory_count",
]
