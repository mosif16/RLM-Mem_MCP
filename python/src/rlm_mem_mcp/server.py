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


# Tool routing prompt for LLM-based auto-routing
TOOL_ROUTER_PROMPT = """You are a code analysis tool router. Given a user query, decide which tools to run.

AVAILABLE TOOLS:
- ios_scan: Full iOS audit (force unwraps, weak self, retain cycles, main thread, CloudKit, deprecated APIs, SwiftData, insecure storage, input sanitization, jailbreak detection)
- security_scan: Security vulnerabilities (secrets, SQL injection, command injection, XSS, Python security, insecure storage, input sanitization)
- quality_scan: Code quality (long functions, TODOs/FIXMEs)

Individual tools (for targeted searches):
- find_secrets: Hardcoded secrets/API keys
- find_force_unwraps: Swift force unwraps (!)
- find_weak_self_issues: Missing [weak self] in closures
- find_sql_injection: SQL injection vulnerabilities
- find_insecure_storage: Sensitive data in UserDefaults instead of Keychain
- find_input_sanitization_issues: User input not sanitized
- find_deprecated_apis: Deprecated API usage (params: min_severity=LOW|MEDIUM|HIGH)
- find_missing_jailbreak_detection: Check for jailbreak detection (finance apps)
- map_architecture: Map codebase structure and dependencies

RULES:
1. For broad audits, prefer scan tools (ios_scan, security_scan) over individual tools
2. For specific issues, use targeted tools
3. Can run multiple tools if query spans categories
4. Return empty array if query doesn't match any tools (e.g., "explain this code")

Respond with ONLY a JSON object:
{"tools": ["tool_name", ...], "params": {"tool_name": {"param": "value"}}}

Examples:
Query: "audit this iOS app for issues"
{"tools": ["ios_scan", "security_scan"], "params": {}}

Query: "find memory leaks in Swift code"
{"tools": ["find_weak_self_issues", "find_force_unwraps"], "params": {}}

Query: "check for hardcoded API keys"
{"tools": ["find_secrets"], "params": {}}

Query: "is sensitive data stored securely"
{"tools": ["find_insecure_storage", "find_secrets"], "params": {}}

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
    content: str
) -> tuple[str, list[ToolResult]]:
    """
    Execute the tools selected by the LLM router.

    This is Layer 2 of the double-layer system:
    - Deterministic execution of structured tools
    - No LLM involved in actual scanning
    """
    tools = StructuredTools(content)
    all_results: list[ToolResult] = []

    for tool_name in tools_to_run:
        tool_params = params.get(tool_name, {})

        try:
            if tool_name == "ios_scan":
                all_results.extend(tools.run_ios_scan())
            elif tool_name == "security_scan":
                all_results.extend(tools.run_security_scan())
            elif tool_name == "quality_scan":
                all_results.extend(tools.run_quality_scan())
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
        except Exception as e:
            print(f"[RLM Router] Tool {tool_name} failed: {e}", file=sys.stderr)

    return _format_tool_results(all_results, tools_to_run)


def _format_tool_results(
    all_results: list[ToolResult],
    tools_run: list[str]
) -> tuple[str, list[ToolResult]]:
    """Format tool results into markdown output."""
    if not all_results:
        return "", []

    # Format results
    output_parts = [f"## Structured Tool Results (LLM-Routed: {', '.join(tools_run)})\n"]
    total_findings = 0
    high_confidence_findings = 0

    for result in all_results:
        if result.findings:
            output_parts.append(result.to_markdown())
            output_parts.append("\n---\n")
            total_findings += result.count
            high_confidence_findings += len(result.high_confidence)

    if total_findings == 0:
        return f"## Structured Tool Results\n\n**Tools Run**: {', '.join(tools_run)}\n**Findings**: None\n", all_results

    # Add summary at top
    summary = f"**Summary**: {total_findings} findings ({high_confidence_findings} high confidence) from {len(tools_run)} tools\n\n"
    output_parts.insert(1, summary)

    return "\n".join(output_parts), all_results


def _run_structured_tools_for_query_type(
    query_type: str,
    content: str,
    query_lower: str
) -> tuple[str, list[ToolResult]]:
    """
    Fallback: Auto-route to structured tools based on detected query type.
    Used when LLM routing fails or returns empty.

    Returns:
        (formatted_output, list_of_results)
    """
    tools = StructuredTools(content)
    all_results: list[ToolResult] = []
    tools_run: list[str] = []

    # Determine which scans to run based on query type
    if query_type == "ios":
        all_results = tools.run_ios_scan()
        tools_run = ["ios_scan"]
    elif query_type == "security":
        all_results = tools.run_security_scan()
        tools_run = ["security_scan"]
    elif query_type == "quality":
        all_results = tools.run_quality_scan()
        tools_run = ["quality_scan"]
    else:
        # For mixed/general queries, check keywords and run matching scans
        if any(kw in query_lower for kw in ["swift", "ios", "unwrap", "swiftui", "xcode"]):
            all_results.extend(tools.run_ios_scan())
            tools_run.append("ios_scan")
        if any(kw in query_lower for kw in ["security", "secret", "inject", "vulnerab", "xss"]):
            all_results.extend(tools.run_security_scan())
            tools_run.append("security_scan")
        if any(kw in query_lower for kw in ["quality", "long function", "todo", "fixme"]):
            all_results.extend(tools.run_quality_scan())
            tools_run.append("quality_scan")

    if not all_results:
        return "", []

    return _format_tool_results(all_results, tools_run)


async def handle_rlm_analyze(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle rlm_analyze tool call with semantic caching and result verification."""
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

    rlm_config, _, file_collector, rlm_processor, _, semantic_cache, result_verifier, project_analyzer = get_instances()

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

    # ===== DOUBLE-LAYER AUTO-ROUTING =====
    # Layer 1: LLM decides which tools to run (intelligent routing)
    # Layer 2: Structured tools execute deterministically (reliable execution)
    content = collection.get_combined_content(include_headers=True)

    # Try LLM-based routing first
    routing_decision = await _llm_route_tools(query, rlm_config)
    tools_to_run = routing_decision.get("tools", [])
    tool_params = routing_decision.get("params", {})

    if tools_to_run:
        # LLM routing succeeded - execute selected tools
        _log_timing("llm_routing", start_time, tools=tools_to_run)
        structured_output, structured_results = _execute_routed_tools(
            tools_to_run, tool_params, content
        )
    else:
        # Fall back to keyword-based routing
        _log_timing("keyword_routing", start_time, query_type=query_analysis["query_type"])
        structured_output, structured_results = _run_structured_tools_for_query_type(
            query_analysis["query_type"],
            content,
            query.lower()
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

        # For comprehensive results, skip LLM processing entirely
        if high_conf_findings >= 3 or total_findings >= 10:
            return [TextContent(type="text", text=structured_output)]

    # Progress callback that logs to stderr for visibility
    def progress_log(msg: str) -> None:
        import sys
        print(f"[RLM Progress] {msg}", file=sys.stderr)

    # Process with RLM (auto-decompose if broad) - for additional context or if no structured tools matched
    _log_timing("rlm_process:start", start_time)
    result = await rlm_processor.process_with_decomposition(query, collection, progress_callback=progress_log)
    _log_timing("rlm_process:complete", start_time, chunks=len(result.chunk_results))

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
    rlm_config, server_config, file_collector, rlm_processor, _, semantic_cache, _, _ = get_instances()

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
            "semantic_cache": semantic_cache.get_stats(),
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
