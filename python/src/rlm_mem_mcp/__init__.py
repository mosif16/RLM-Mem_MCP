"""
RLM-Mem MCP Server

An MCP server implementing the Recursive Language Model (RLM) technique
for ultimate context management with Claude Code.

Based on:
- arXiv:2512.24601 (Recursive Language Models)
- Anthropic MCP Best Practices
- Anthropic Prompt Caching Documentation
- Claude Agent SDK for subagent orchestration

Models:
- Claude Haiku 4.5: Default for chunk processing (fast, included in Claude Max)
- Claude Sonnet: Used for complex aggregation tasks
"""

__version__ = "1.1.0"

from .server import main, create_server
from .rlm_processor import RLMProcessor
from .file_collector import FileCollector
from .cache_manager import CacheManager
from .agent_pipeline import RLMAgentPipeline, run_rlm_pipeline

__all__ = [
    "main",
    "create_server",
    "RLMProcessor",
    "RLMAgentPipeline",
    "run_rlm_pipeline",
    "FileCollector",
    "CacheManager",
]
