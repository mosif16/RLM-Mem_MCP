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
from .structured_tools import StructuredTools, ToolResult


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


def _format_skipped_files_summary(skipped_files: list[str], max_display: int = 10) -> str:
    """
    Format a summary of skipped files for user visibility.

    Args:
        skipped_files: List of skipped file paths with reasons
        max_display: Maximum number of skipped files to show

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
            lines.append(f"  - {path}")
        if count > max_display:
            lines.append(f"  - ... and {count - max_display} more")

    lines.append("\n*Tip: Modify config.skipped_directories or config.included_extensions to change what gets collected.*")

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
                        "scan_mode": {
                            "type": "string",
                            "enum": ["auto", "ios", "ios-strict", "security", "quality", "all"],
                            "description": (
                                "Pre-configured scan mode. Options:\n"
                                "• 'auto' (default): Auto-detect based on query and file types\n"
                                "• 'ios': Run iOS/Swift scanners (security + crash issues)\n"
                                "• 'ios-strict': iOS scan with HIGH confidence only (minimal noise)\n"
                                "• 'security': Run security scanners (secrets, injection, XSS)\n"
                                "• 'quality': Run code quality scanners (long functions, TODOs)\n"
                                "• 'all': Run all scanners including quality checks"
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


# =============================================================================
# LITERAL SEARCH DETECTION AND FAST-PATH
# =============================================================================
# These functions detect when a query is looking for literal strings (grep-like)
# and bypass the LLM routing for faster, more accurate results.


import re as _re


def _extract_quoted_strings(query: str) -> list[str]:
    """
    Extract quoted strings from a query.

    Handles both single and double quotes:
    - "tasks-main" -> tasks-main
    - 'task tracking' -> task tracking

    Returns:
        List of strings found in quotes
    """
    # Match both single and double quoted strings
    pattern = r'["\']([^"\']+)["\']'
    matches = _re.findall(pattern, query)
    return matches


def _is_literal_search_query(query: str) -> tuple[bool, list[str]]:
    """
    Detect if a query is asking for literal string search.

    Indicators of literal search:
    1. Contains quoted strings ("foo", 'bar')
    2. Uses search keywords: "find files", "search for", "grep", "containing", "mentioning"
    3. NOT asking for analysis/audit/review/security

    Returns:
        (is_literal_search, list_of_search_terms)
    """
    query_lower = query.lower()

    # Extract quoted strings
    quoted_strings = _extract_quoted_strings(query)

    if not quoted_strings:
        return False, []

    # Keywords that indicate literal search intent
    search_keywords = [
        "find files", "find all files", "search for", "grep", "containing",
        "mentioning", "with the string", "that contain", "that mention",
        "where is", "which files", "what files", "list files",
        "files with", "files containing", "files mentioning",
        "related to", "references to", "uses of"
    ]

    # Keywords that indicate analysis intent (NOT literal search)
    analysis_keywords = [
        "security", "audit", "vulnerab", "injection", "xss",
        "memory leak", "retain cycle", "force unwrap",
        "code quality", "refactor", "bug", "issue", "problem"
    ]

    # Check for analysis keywords - if present, it's not a pure literal search
    has_analysis_intent = any(kw in query_lower for kw in analysis_keywords)

    # Check for search keywords
    has_search_intent = any(kw in query_lower for kw in search_keywords)

    # If we have quoted strings AND search keywords AND NOT analysis keywords
    # -> This is a literal search
    if quoted_strings and has_search_intent and not has_analysis_intent:
        return True, quoted_strings

    # If we have quoted strings but no analysis keywords and query is short
    # -> Likely a literal search
    if quoted_strings and not has_analysis_intent and len(query.split()) < 15:
        return True, quoted_strings

    return False, []


def _perform_literal_search(content: str, search_terms: list[str]) -> str:
    """
    Perform grep-like literal search on content.

    This is the fast-path that bypasses LLM routing entirely.

    Args:
        content: The combined file content (with ### File: headers)
        search_terms: List of literal strings to search for

    Returns:
        Formatted search results
    """
    results = []
    current_file = "unknown"
    line_number = 0

    # Track findings per search term
    findings: dict[str, list[dict]] = {term: [] for term in search_terms}

    for line in content.split('\n'):
        line_number += 1

        # Track current file from headers
        if line.startswith("### File:"):
            current_file = line.replace("### File:", "").strip()
            line_number = 0  # Reset for new file
            continue

        # Check each search term (case-insensitive)
        line_lower = line.lower()
        for term in search_terms:
            if term.lower() in line_lower:
                findings[term].append({
                    "file": current_file,
                    "line": line_number,
                    "content": line.strip()[:200]  # Truncate long lines
                })

    # Format results
    output_parts = ["## Literal Search Results\n"]
    total_matches = 0

    for term, matches in findings.items():
        total_matches += len(matches)
        output_parts.append(f"\n### \"{term}\" ({len(matches)} matches)\n")

        if not matches:
            output_parts.append("No matches found.\n")
        else:
            # Group by file
            by_file: dict[str, list[dict]] = {}
            for match in matches:
                file = match["file"]
                if file not in by_file:
                    by_file[file] = []
                by_file[file].append(match)

            for file, file_matches in by_file.items():
                output_parts.append(f"\n**{file}**\n")
                for m in file_matches[:10]:  # Limit per file
                    output_parts.append(f"  Line {m['line']}: `{m['content']}`\n")
                if len(file_matches) > 10:
                    output_parts.append(f"  ... and {len(file_matches) - 10} more matches\n")

    output_parts.append(f"\n---\n*Total: {total_matches} matches across {len(search_terms)} search term(s)*\n")
    output_parts.append("*Tip: Use native Grep tool for better performance on literal searches.*\n")

    return "".join(output_parts)


# Tool routing prompt for LLM-based auto-routing
TOOL_ROUTER_PROMPT = """You are a code analysis tool router. Given a user query, decide which tools to run.

PRIORITY ORDER (when query is ambiguous, prefer this order):
1. Security issues (bugs that can be exploited)
2. iOS/Swift specific issues (crashes, memory leaks)
3. Code quality (style, maintainability)

AVAILABLE TOOLS:
Scan Tools (comprehensive):
- ios_scan: Full iOS audit (force unwraps, weak self, retain cycles, @MainActor, Task cancellation, CloudKit, deprecated APIs, SwiftData, StateObject issues, insecure storage, jailbreak detection) - 13 scanners
- security_scan: Security vulnerabilities (secrets, SQL injection, command injection, XSS, Python security, insecure storage, input sanitization) - 7 scanners
- quality_scan: Code quality (long functions, TODOs/FIXMEs) - 2 scanners

Individual iOS Tools:
- find_force_unwraps: Swift force unwraps (!) - excludes string literals
- find_weak_self_issues: Missing [weak self] in closures
- find_retain_cycles: Strong reference cycles
- find_mainactor_issues: Missing @MainActor on ObservableObject/ViewModel
- find_task_cancellation_issues: Task {} without cancellation handling
- find_stateobject_issues: @ObservedObject used where @StateObject needed
- find_main_thread_violations: UI updates off main thread
- find_cloudkit_issues: CloudKit error handling issues
- find_swiftdata_issues: SwiftData threading issues
- find_deprecated_apis: Deprecated API usage (params: min_severity=LOW|MEDIUM|HIGH)

Individual Security Tools:
- find_secrets: Hardcoded secrets/API keys
- find_sql_injection: SQL injection vulnerabilities
- find_command_injection: Command injection (os.system, subprocess)
- find_xss_vulnerabilities: XSS in JavaScript
- find_insecure_storage: Sensitive data in UserDefaults instead of Keychain
- find_input_sanitization_issues: User input not sanitized
- find_missing_jailbreak_detection: Check for jailbreak detection (finance apps)
- find_python_security: Python-specific security (pickle, yaml, eval)

Architecture Tools:
- map_architecture: Map codebase structure and dependencies

RULES:
1. SECURITY queries should run security_scan FIRST, not quality_scan
2. iOS/Swift queries should run ios_scan which includes security checks
3. For "audit" or "review" queries, run comprehensive scans not individual tools
4. Only use quality_scan when explicitly asked about code quality/style
5. Return empty array for non-analysis queries ("explain this", "how does X work")

Respond with ONLY a JSON object:
{"tools": ["tool_name", ...], "params": {"tool_name": {"param": "value"}}}

Examples:
Query: "audit this iOS app for security issues"
{"tools": ["ios_scan", "security_scan"], "params": {}}

Query: "find security vulnerabilities"
{"tools": ["security_scan"], "params": {}}

Query: "check this Swift code for issues"
{"tools": ["ios_scan"], "params": {}}

Query: "find memory leaks in Swift code"
{"tools": ["find_weak_self_issues", "find_retain_cycles", "find_mainactor_issues"], "params": {}}

Query: "check for hardcoded API keys"
{"tools": ["find_secrets"], "params": {}}

Query: "find @MainActor issues"
{"tools": ["find_mainactor_issues", "find_main_thread_violations"], "params": {}}

Query: "check Task cancellation handling"
{"tools": ["find_task_cancellation_issues"], "params": {}}

Query: "find code quality issues"
{"tools": ["quality_scan"], "params": {}}

Query: "find deprecated APIs, only high severity"
{"tools": ["find_deprecated_apis"], "params": {"find_deprecated_apis": {"min_severity": "HIGH"}}}

Query: "explain how this function works"
{"tools": [], "params": {}}
"""


async def _llm_route_tools(query: str, config: "RLMConfig") -> dict[str, Any]:
    """
    Use LLM to intelligently decide which tools to run.

    This is Layer 1 of the double-layer system:
    - Layer 1: LLM decides WHAT to run (intelligent routing)
    - Layer 2: Structured tools execute deterministically (reliable execution)

    Returns:
        {"tools": ["tool_name", ...], "params": {"tool_name": {"param": "value"}}}
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{config.api_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "HTTP-Referer": "https://github.com/mosif16/RLM-Mem_MCP",
                    "X-Title": "RLM-Mem MCP Server"
                },
                json={
                    "model": config.model,  # Use fast model for routing
                    "messages": [
                        {"role": "system", "content": TOOL_ROUTER_PROMPT},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    "max_tokens": 200,
                    "temperature": 0  # Deterministic routing
                }
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            import json as json_mod
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()

            # Handle empty or non-JSON responses
            if not content or content.lower() in ["none", "null", "n/a"]:
                return {"tools": [], "params": {}}

            result = json_mod.loads(content)

            # Validate structure
            if not isinstance(result, dict):
                return {"tools": [], "params": {}}
            if "tools" not in result:
                result["tools"] = []
            if "params" not in result:
                result["params"] = {}

            return result

    except Exception as e:
        # Fall back to empty (will use keyword matching)
        print(f"[RLM Router] LLM routing failed: {e}, falling back to keywords", file=sys.stderr)
        return {"tools": [], "params": {}}


def _execute_routed_tools(
    tools_to_run: list[str],
    params: dict[str, dict],
    content: str,
    min_confidence: str = "LOW"
) -> tuple[str, list[ToolResult]]:
    """
    Execute the tools selected by the LLM router.

    This is Layer 2 of the double-layer system:
    - Deterministic execution of structured tools
    - No LLM involved in actual scanning

    Args:
        tools_to_run: List of tool names to execute
        params: Tool-specific parameters
        content: File content to analyze
        min_confidence: Global minimum confidence filter ("LOW", "MEDIUM", "HIGH")
    """
    tools = StructuredTools(content)
    all_results: list[ToolResult] = []

    for tool_name in tools_to_run:
        tool_params = params.get(tool_name, {})

        try:
            if tool_name == "ios_scan":
                all_results.extend(tools.run_ios_scan(min_confidence=min_confidence))
            elif tool_name == "security_scan":
                all_results.extend(tools.run_security_scan(min_confidence=min_confidence))
            elif tool_name == "quality_scan":
                all_results.extend(tools.run_quality_scan(min_confidence=min_confidence))
            elif tool_name == "find_secrets":
                all_results.append(tools.find_secrets())
            elif tool_name == "find_force_unwraps":
                all_results.append(tools.find_force_unwraps())
            elif tool_name == "find_weak_self_issues":
                all_results.append(tools.find_weak_self_issues())
            elif tool_name == "find_sql_injection":
                all_results.append(tools.find_sql_injection())
            elif tool_name == "find_deprecated_apis":
                min_sev = tool_params.get("min_severity", "LOW")
                all_results.append(tools.find_deprecated_apis(min_severity=min_sev))
            elif tool_name == "map_architecture":
                all_results.append(tools.map_architecture())
            elif tool_name == "find_retain_cycles":
                all_results.append(tools.find_retain_cycles())
            elif tool_name == "find_main_thread_violations":
                all_results.append(tools.find_main_thread_violations())
            elif tool_name == "find_cloudkit_issues":
                all_results.append(tools.find_cloudkit_issues())
            elif tool_name == "find_swiftdata_issues":
                all_results.append(tools.find_swiftdata_issues())
            elif tool_name == "find_command_injection":
                all_results.append(tools.find_command_injection())
            elif tool_name == "find_xss_vulnerabilities":
                all_results.append(tools.find_xss_vulnerabilities())
            elif tool_name == "find_python_security":
                all_results.append(tools.find_python_security())
            elif tool_name == "find_long_functions":
                all_results.append(tools.find_long_functions())
            elif tool_name == "find_todos":
                all_results.append(tools.find_todos())
            elif tool_name == "find_insecure_storage":
                all_results.append(tools.find_insecure_storage())
            elif tool_name == "find_input_sanitization_issues":
                all_results.append(tools.find_input_sanitization_issues())
            elif tool_name == "find_missing_jailbreak_detection":
                all_results.append(tools.find_missing_jailbreak_detection())
            # New iOS tools
            elif tool_name == "find_mainactor_issues":
                all_results.append(tools.find_mainactor_issues())
            elif tool_name == "find_task_cancellation_issues":
                all_results.append(tools.find_task_cancellation_issues())
            elif tool_name == "find_stateobject_issues":
                all_results.append(tools.find_stateobject_issues())
        except Exception as e:
            print(f"[RLM Router] Tool {tool_name} failed: {e}", file=sys.stderr)

    # Apply global confidence filter to individual tool results
    if min_confidence.upper() != "LOW":
        all_results = tools._filter_by_confidence(all_results, min_confidence)

    return _format_tool_results(all_results, tools_to_run)


def _format_tool_results(
    all_results: list[ToolResult],
    tools_run: list[str]
) -> tuple[str, list[ToolResult]]:
    """Format tool results into markdown output with severity grouping."""
    if not all_results:
        return "", []

    # Count findings by severity
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    total_findings = 0

    for result in all_results:
        for finding in result.findings:
            total_findings += 1
            sev = finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity)
            conf = finding.confidence.value if hasattr(finding.confidence, 'value') else str(finding.confidence)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    if total_findings == 0:
        return f"## Scan Results\n\n**Tools Run**: {', '.join(tools_run)}\n**Findings**: None - code looks clean!\n", all_results

    # Build severity summary
    output_parts = [f"## Scan Results\n"]

    # Severity breakdown (actionable summary)
    summary_lines = []
    if severity_counts["CRITICAL"] > 0:
        summary_lines.append(f"🔴 **Critical**: {severity_counts['CRITICAL']} (fix immediately)")
    if severity_counts["HIGH"] > 0:
        summary_lines.append(f"🟠 **High**: {severity_counts['HIGH']} (fix soon)")
    if severity_counts["MEDIUM"] > 0:
        summary_lines.append(f"🟡 **Medium**: {severity_counts['MEDIUM']} (review when possible)")
    if severity_counts["LOW"] > 0:
        summary_lines.append(f"🟢 **Low**: {severity_counts['LOW']} (minor issues)")
    if severity_counts["INFO"] > 0:
        summary_lines.append(f"ℹ️ **Info**: {severity_counts['INFO']} (informational)")

    output_parts.append("\n".join(summary_lines))
    output_parts.append(f"\n\n**Total**: {total_findings} findings | **High confidence**: {confidence_counts['HIGH']}\n")
    output_parts.append(f"**Tools**: {', '.join(tools_run)}\n")
    output_parts.append("\n---\n")

    # Group findings by severity (Critical/High first)
    critical_high = []
    medium_low = []

    for result in all_results:
        for finding in result.findings:
            sev = finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity)
            if sev in ("CRITICAL", "HIGH"):
                critical_high.append((result.tool_name, finding))
            else:
                medium_low.append((result.tool_name, finding))

    # Output critical/high findings first
    if critical_high:
        output_parts.append("\n### Critical & High Priority\n")
        for tool_name, finding in critical_high:
            conf = finding.confidence.value if hasattr(finding.confidence, 'value') else str(finding.confidence)
            sev = finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity)
            output_parts.append(f"**{finding.file}:{finding.line}** [{sev}] [{conf} confidence]\n")
            output_parts.append(f"  {finding.issue}\n")
            if finding.code:
                output_parts.append(f"  ```\n  {finding.code[:150]}\n  ```\n")
            if finding.fix:
                output_parts.append(f"  💡 Fix: {finding.fix}\n")
            output_parts.append("\n")

    # Output medium/low findings (collapsed if many)
    if medium_low:
        if len(medium_low) > 10:
            output_parts.append(f"\n### Medium & Low Priority ({len(medium_low)} findings)\n")
            output_parts.append("<details><summary>Click to expand</summary>\n\n")
            for tool_name, finding in medium_low[:20]:  # Limit to 20
                output_parts.append(f"- **{finding.file}:{finding.line}**: {finding.issue}\n")
            if len(medium_low) > 20:
                output_parts.append(f"\n*... and {len(medium_low) - 20} more*\n")
            output_parts.append("\n</details>\n")
        else:
            output_parts.append(f"\n### Medium & Low Priority\n")
            for tool_name, finding in medium_low:
                conf = finding.confidence.value if hasattr(finding.confidence, 'value') else str(finding.confidence)
                output_parts.append(f"- **{finding.file}:{finding.line}** [{conf}]: {finding.issue}\n")

    return "\n".join(output_parts), all_results


def _run_structured_tools_for_query_type(
    query_type: str,
    content: str,
    query_lower: str,
    min_confidence: str = "LOW"
) -> tuple[str, list[ToolResult]]:
    """
    Fallback: Auto-route to structured tools based on detected query type.
    Used when LLM routing fails or returns empty.

    Args:
        query_type: Detected query type (ios, security, quality, etc.)
        content: File content to analyze
        query_lower: Lowercase query for keyword matching
        min_confidence: Minimum confidence filter ("LOW", "MEDIUM", "HIGH")

    Returns:
        (formatted_output, list_of_results)
    """
    tools = StructuredTools(content)
    all_results: list[ToolResult] = []
    tools_run: list[str] = []

    # Determine which scans to run based on query type
    if query_type == "ios":
        all_results = tools.run_ios_scan(min_confidence=min_confidence)
        tools_run = ["ios_scan"]
    elif query_type == "security":
        all_results = tools.run_security_scan(min_confidence=min_confidence)
        tools_run = ["security_scan"]
    elif query_type == "quality":
        all_results = tools.run_quality_scan(min_confidence=min_confidence)
        tools_run = ["quality_scan"]
    else:
        # For mixed/general queries, check keywords and run matching scans
        # PRIORITY: Security first, then iOS, then quality
        if any(kw in query_lower for kw in ["security", "secret", "inject", "vulnerab", "xss", "attack"]):
            all_results.extend(tools.run_security_scan(min_confidence=min_confidence))
            tools_run.append("security_scan")
        if any(kw in query_lower for kw in ["swift", "ios", "unwrap", "swiftui", "xcode", "mainactor"]):
            all_results.extend(tools.run_ios_scan(min_confidence=min_confidence))
            tools_run.append("ios_scan")
        if any(kw in query_lower for kw in ["quality", "long function", "todo", "fixme", "style"]):
            all_results.extend(tools.run_quality_scan(min_confidence=min_confidence))
            tools_run.append("quality_scan")

    if not all_results:
        return "", []

    return _format_tool_results(all_results, tools_run)


async def _write_progress(session_id: str, event_type: str, message: str, progress: float, details: dict = None):
    """L4: Write progress event to memory store for client polling."""
    global _memory_store, _tag_index
    import uuid

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


async def handle_rlm_analyze(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_analyze tool call with semantic caching and result verification."""
    import uuid
    start_time = time.time()

    # L4: Generate session ID for progress tracking
    session_id = arguments.get("session_id") or str(uuid.uuid4())[:8]

    query = arguments.get("query", "")
    paths = arguments.get("paths", [])
    scan_mode = arguments.get("scan_mode", "auto")  # auto, ios, ios-strict, security, quality, all
    min_confidence = arguments.get("min_confidence", "MEDIUM")  # LOW, MEDIUM, HIGH (default MEDIUM to reduce noise)
    include_quality = arguments.get("include_quality", False)  # Include code style checks

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

    rlm_config, _, file_collector, rlm_processor, _, semantic_cache, result_verifier, project_analyzer = get_instances()

    # Validate config
    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    # L4: Start progress tracking
    await _write_progress(session_id, "start", "Starting analysis...", 0, {"query": query[:100], "paths": paths})

    # Collect files using async method
    _log_timing("file_collection:start", start_time, path_count=len(paths))
    collection = await file_collector.collect_paths_async(paths)
    _log_timing("file_collection:complete", start_time, file_count=collection.file_count)

    # L4: Progress after file collection
    await _write_progress(session_id, "files_collected", f"Collected {collection.file_count} files", 20, {"file_count": collection.file_count})

    if collection.file_count == 0:
        error_msg = "## No Matching Files Found\n\n"
        if collection.errors:
            error_msg += f"**Errors:** {'; '.join(collection.errors)}\n\n"

        # Show skipped files to help user understand why
        if collection.skipped_files:
            error_msg += _format_skipped_files_summary(collection.skipped_files, max_display=20)
        else:
            error_msg += "No files matched the configured extensions.\n\n"
            error_msg += "**Included extensions:** " + ", ".join(sorted(rlm_config.included_extensions)[:20]) + "...\n"
            error_msg += "**Skipped directories:** " + ", ".join(sorted(rlm_config.skipped_directories)[:10]) + "..."

        return [TextContent(type="text", text=error_msg)]

    # Check semantic cache for similar query
    context_summary = ", ".join(f.relative_path for f in collection.files[:50])
    cached_response, similarity = semantic_cache.get_similar(query, context_summary)

    if cached_response:
        _log_timing("semantic_cache:hit", start_time, similarity=similarity)
        output = f"## Cached Result (similarity: {similarity:.2f})\n\n{cached_response}"
        return [TextContent(type="text", text=output)]

    # Analyze project for context
    project_info = project_analyzer.analyze(collection)
    _log_timing("project_analysis", start_time, project_type=project_info.project_type)

    # Analyze query and use decomposition for broad queries
    query_analysis = rlm_processor.analyze_query_quality(query)
    _log_timing("query_analysis", start_time, is_broad=query_analysis["is_broad"], query_type=query_analysis["query_type"])

    # L3: Auto-select scan_mode based on detected query type when mode is "auto"
    detected_type = query_analysis["query_type"]
    if scan_mode == "auto" and detected_type != "general":
        # Map query types to scan modes
        type_to_scan_mode = {
            "ios": "ios",
            "security": "security",
            "quality": "quality",
        }
        if detected_type in type_to_scan_mode:
            scan_mode = type_to_scan_mode[detected_type]
            print(f"[RLM] Auto-selected scan_mode='{scan_mode}' based on query type '{detected_type}'", file=sys.stderr)

    # Get combined content early for fast-path checks
    content = collection.get_combined_content(include_headers=True)

    # ===== FAST-PATH: LITERAL SEARCH =====
    # Check if this is a literal string search (grep-like query with quoted strings)
    # If so, bypass all LLM routing and perform direct search
    is_literal, search_terms = _is_literal_search_query(query)
    if is_literal and search_terms:
        _log_timing("literal_search:detected", start_time, terms=search_terms)
        print(f"[RLM] Literal search detected: {search_terms}", file=sys.stderr)

        # Perform fast grep-like search
        literal_result = _perform_literal_search(content, search_terms)
        processing_time = int((time.time() - start_time) * 1000)

        literal_result += f"\n*Scanned {collection.file_count} files in {processing_time}ms (literal search fast-path)*"

        # Add skipped files info if any
        if collection.skipped_files:
            literal_result += _format_skipped_files_summary(collection.skipped_files)

        return [TextContent(type="text", text=literal_result)]

    # ===== DOUBLE-LAYER AUTO-ROUTING =====
    # Layer 1: LLM decides which tools to run (intelligent routing)
    # Layer 2: Structured tools execute deterministically (reliable execution)

    # Handle explicit scan_mode
    if scan_mode != "auto":
        _log_timing("explicit_scan_mode", start_time, mode=scan_mode)
        tools = StructuredTools(content)

        if scan_mode == "ios":
            structured_results = tools.run_ios_scan(
                min_confidence=min_confidence,
                include_quality=include_quality
            )
            tools_to_run = ["ios_scan"]
        elif scan_mode == "ios-strict":
            # iOS-strict: only security/crash issues, no quality, HIGH confidence
            structured_results = tools.run_ios_scan(
                min_confidence="HIGH",
                include_quality=False
            )
            tools_to_run = ["ios_scan (strict)"]
        elif scan_mode == "security":
            structured_results = tools.run_security_scan(
                min_confidence=min_confidence,
                include_quality=include_quality
            )
            tools_to_run = ["security_scan"]
        elif scan_mode == "quality":
            structured_results = tools.run_quality_scan(min_confidence=min_confidence)
            tools_to_run = ["quality_scan"]
        elif scan_mode == "all":
            structured_results = (
                tools.run_ios_scan(min_confidence=min_confidence, include_quality=True) +
                tools.run_security_scan(min_confidence=min_confidence, include_quality=False)
            )
            tools_to_run = ["ios_scan", "security_scan", "quality_scan"]
        else:
            structured_results = []
            tools_to_run = []

        structured_output, _ = _format_tool_results(structured_results, tools_to_run)
    else:
        # Auto mode: Try LLM-based routing first
        routing_decision = await _llm_route_tools(query, rlm_config)
        tools_to_run = routing_decision.get("tools", [])
        tool_params = routing_decision.get("params", {})

        if tools_to_run:
            # LLM routing succeeded - execute selected tools
            _log_timing("llm_routing", start_time, tools=tools_to_run)
            structured_output, structured_results = _execute_routed_tools(
                tools_to_run, tool_params, content, min_confidence
            )
        else:
            # Fall back to keyword-based routing
            _log_timing("keyword_routing", start_time, query_type=query_analysis["query_type"])
            structured_output, structured_results = _run_structured_tools_for_query_type(
                query_analysis["query_type"],
                content,
                query.lower(),
                min_confidence
            )

    _log_timing("structured_tools", start_time,
                tools_run=len(structured_results),
                findings=sum(r.count for r in structured_results))

    # If structured tools found significant results, return them directly
    # without additional LLM processing (faster, more reliable)
    total_findings = sum(r.count for r in structured_results)
    high_conf_findings = sum(len(r.high_confidence) for r in structured_results)

    if total_findings > 0:
        # Add file scan stats
        structured_output += f"\n\n*Scanned {collection.file_count} files in {int((time.time() - start_time) * 1000)}ms*"

        # Add skipped files info if any
        if collection.skipped_files:
            structured_output += _format_skipped_files_summary(collection.skipped_files)

        # For comprehensive results, skip LLM processing entirely
        if high_conf_findings >= 3 or total_findings >= 10:
            return [TextContent(type="text", text=structured_output)]

    # L4: Progress before RLM processing
    await _write_progress(session_id, "analyzing", "Running RLM analysis...", 50, {"scan_mode": scan_mode})

    # Progress callback that logs to stderr and memory store
    async def progress_log_async(msg: str) -> None:
        import sys
        print(f"[RLM Progress] {msg}", file=sys.stderr)

    def progress_log(msg: str) -> None:
        import sys
        print(f"[RLM Progress] {msg}", file=sys.stderr)

    # Process with RLM (auto-decompose if broad) - for additional context or if no structured tools matched
    _log_timing("rlm_process:start", start_time)
    result = await rlm_processor.process_with_decomposition(query, collection, progress_callback=progress_log)
    _log_timing("rlm_process:complete", start_time, chunks=len(result.chunk_results))

    # L4: Progress after RLM processing
    await _write_progress(session_id, "verifying", "Verifying findings...", 80, {"chunks_processed": len(result.chunk_results)})

    # Format output
    output = rlm_processor.format_result(result)

    # Verify findings and add verification summary
    verified_output, verification_stats = result_verifier.verify_findings(output, collection)
    _log_timing("verification", start_time,
                verified=verification_stats.verified_files,
                invalid=verification_stats.invalid_lines)

    # Cache the result for future similar queries
    semantic_cache.set(query, context_summary, verified_output)

    # Add project context if relevant
    if project_info.key_files:
        project_context = f"\n\n## Project Context\n- Type: {project_info.project_type}\n- Key files from docs: {', '.join(project_info.key_files[:10])}"
        verified_output += project_context

    # Combine structured tool output with LLM output
    if structured_output:
        final_output = structured_output + "\n\n---\n\n## Additional LLM Analysis\n\n" + verified_output
    else:
        final_output = verified_output

    # L4: Progress complete
    await _write_progress(session_id, "complete", "Analysis complete", 100, {
        "duration_ms": int((time.time() - start_time) * 1000),
        "session_id": session_id
    })

    return [TextContent(type="text", text=final_output)]


async def handle_rlm_query_text(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_query_text tool call with result verification."""
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

    rlm_config, _, file_collector, rlm_processor, _, semantic_cache, result_verifier, _ = get_instances()

    # Validate config
    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    # Check semantic cache
    context_summary = f"text:{len(text)}chars"
    cached_response, similarity = semantic_cache.get_similar(query, context_summary)

    if cached_response:
        _log_timing("semantic_cache:hit", start_time, similarity=similarity)
        output = f"## Cached Result (similarity: {similarity:.2f})\n\n{cached_response}"
        return [TextContent(type="text", text=output)]

    # Create collection from text (use async version)
    collection = await file_collector.collect_text_async(text, "input_text")

    # Process with RLM (auto-decompose if broad)
    _log_timing("rlm_process:start", start_time)
    result = await rlm_processor.process_with_decomposition(query, collection)
    _log_timing("rlm_process:complete", start_time)

    # Format output
    output = rlm_processor.format_result(result)

    # Cache result
    semantic_cache.set(query, context_summary, output)

    return [TextContent(type="text", text=output)]


async def handle_rlm_status(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_status tool call with semantic cache stats."""
    from .rlm_processor import get_trajectory_logger, get_model_selector

    rlm_config, server_config, file_collector, rlm_processor, _, semantic_cache, _, _ = get_instances()

    # Get memory and metrics stats
    memory_stats = _memory_monitor.get_stats()
    metrics_stats = get_metrics_collector().get_stats()

    # L12: Get trajectory logger stats
    trajectory_logger = get_trajectory_logger()
    model_selector = get_model_selector()

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
            # Dynamic timeout config
            "base_timeout_seconds": rlm_config.base_timeout_seconds,
            "max_timeout_seconds": rlm_config.max_timeout_seconds,
            "max_iterations": rlm_config.max_iterations,
            # OpenRouter prompt caching config
            "prompt_cache_enabled": rlm_config.use_prompt_cache,
            "prompt_cache_ttl": rlm_config.prompt_cache_ttl,
            "cache_usage_tracking": rlm_config.track_cache_usage,
        },
        "memory_entries": len(_memory_store),
        "performance": {
            "token_cache": file_collector.get_cache_stats(),
            "processor": rlm_processor.get_stats(),
            "semantic_cache": semantic_cache.get_stats(),
            "metrics": metrics_stats,
        },
        "resources": {
            "memory": memory_stats,
        },
        # L12: Trajectory logging for debugging
        "recent_trajectories": trajectory_logger.get_recent(5),
        # L10: Model selector stats
        "model_selector": model_selector.get_stats(),
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

    rlm_config, server_config, file_collector, rlm_processor, cache_manager, semantic_cache, result_verifier, project_analyzer = get_instances()

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
