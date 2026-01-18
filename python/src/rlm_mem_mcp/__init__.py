"""
RLM-Mem MCP Server

An MCP server implementing the Recursive Language Model (RLM) technique
for ultimate context management with Claude Code.

Based on:
- arXiv:2512.24601 (Recursive Language Models)
- Anthropic MCP Best Practices
- Anthropic Prompt Caching Documentation
- Claude Agent SDK for subagent orchestration

THE KEY INSIGHT (from the paper):
- Content is stored as a VARIABLE in a Python REPL (NOT in LLM context)
- The LLM writes CODE to examine portions of the content
- Sub-LLM responses are stored in VARIABLES (NOT summarized)
- Full data is PRESERVED - the LLM can access any part at any time

This is NOT summarization! Data is kept intact and accessible.

Models:
- Claude Haiku 4.5: Sub-LLM queries (fast, included in Claude Max)
- Claude Sonnet: Orchestration - writing code to examine content
"""

__version__ = "1.2.0"

from .server import main, create_server
from .rlm_processor import RLMProcessor
from .file_collector import FileCollector
from .cache_manager import CacheManager
from .agent_pipeline import RLMAgentPipeline, run_rlm_pipeline
from .repl_environment import RLMReplEnvironment

__all__ = [
    "main",
    "create_server",
    "RLMProcessor",
    "RLMReplEnvironment",
    "RLMAgentPipeline",
    "run_rlm_pipeline",
    "FileCollector",
    "CacheManager",
]
