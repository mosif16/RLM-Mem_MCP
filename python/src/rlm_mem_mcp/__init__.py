"""
RLM-Mem MCP Server

An MCP server implementing the Recursive Language Model (RLM) technique
for ultimate context management with Claude Code.

Based on:
- arXiv:2512.24601 (Recursive Language Models)
- Anthropic MCP Best Practices
- Anthropic Prompt Caching Documentation
"""

__version__ = "1.0.0"

from .server import main, create_server
from .rlm_processor import RLMProcessor
from .file_collector import FileCollector
from .cache_manager import CacheManager

__all__ = [
    "main",
    "create_server",
    "RLMProcessor",
    "FileCollector",
    "CacheManager",
]
