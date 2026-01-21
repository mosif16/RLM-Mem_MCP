"""
Query Handlers for RLM-Mem MCP Server.

Provides the main analysis handlers:
- handle_rlm_analyze: Analyze files/directories using RLM
- handle_rlm_query_text: Process large text using RLM
- handle_rlm_status: Check server health and configuration
"""

import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from mcp.types import TextContent

from ..structured_tools import StructuredTools, ToolResult
from ..mcp_tools import (
    is_literal_search_query,
    perform_literal_search,
    llm_route_tools,
    execute_routed_tools,
    format_tool_results,
    run_structured_tools_for_query_type,
)
from .memory import write_progress


# Input validation constants
MAX_QUERY_LENGTH = 50_000
MAX_PATHS_COUNT = 100
MAX_PATH_LENGTH = 4096
MAX_TEXT_LENGTH = 10_000_000


def validate_path(path: str) -> tuple[bool, str]:
    """Validate a file path for safety."""
    if not path:
        return False, "Empty path"

    if len(path) > MAX_PATH_LENGTH:
        return False, f"Path too long ({len(path)} > {MAX_PATH_LENGTH})"

    if '\x00' in path:
        return False, "Path contains null bytes"

    suspicious_patterns = ['../', '..\\', '/etc/', '/proc/', '/sys/', '/dev/']
    path_lower = path.lower()
    for pattern in suspicious_patterns:
        if pattern in path_lower:
            return False, f"Suspicious path pattern: {pattern}"

    return True, ""


def _extract_signatures_from_file(file_path: str) -> list[str]:
    """Extract function/class/struct signatures from a file."""
    signatures = []
    try:
        path = Path(file_path)
        if not path.exists():
            return []

        ext = path.suffix.lower()

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(50000)

        if ext == '.swift':
            for match in re.findall(r'^\s*((?:public|private|internal|open|final)\s+)?(class|struct|enum|protocol|actor)\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"{match[1]} {match[2]}")
            for match in re.findall(r'^\s*((?:public|private|internal|open|override)\s+)?func\s+(\w+)\s*\([^)]*\)', content, re.MULTILINE):
                signatures.append(f"func {match[1]}()")

        elif ext == '.py':
            for match in re.findall(r'^(class|def|async def)\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"{match[0]} {match[1]}")

        elif ext in ('.js', '.ts', '.tsx', '.jsx'):
            for match in re.findall(r'^(?:export\s+)?class\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"class {match}")
            for match in re.findall(r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"function {match}()")
            for match in re.findall(r'^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', content, re.MULTILINE):
                signatures.append(f"const {match} = () =>")

        elif ext == '.go':
            for match in re.findall(r'^func\s+(?:\([^)]+\)\s+)?(\w+)', content, re.MULTILINE):
                signatures.append(f"func {match}()")
            for match in re.findall(r'^type\s+(\w+)\s+(struct|interface)', content, re.MULTILINE):
                signatures.append(f"type {match[0]} {match[1]}")

        elif ext == '.rs':
            for match in re.findall(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"fn {match}()")
            for match in re.findall(r'^(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)', content, re.MULTILINE):
                signatures.append(f"struct/enum {match}")

    except Exception:
        pass

    return signatures[:20]


def _format_skipped_files_summary(
    skipped_files: list[str],
    max_display: int = 10,
    include_signatures: bool = False
) -> str:
    """Format a summary of skipped files for user visibility."""
    if not skipped_files:
        return ""

    by_reason: dict[str, list[str]] = {}
    for entry in skipped_files:
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
            file_exists = Path(path).exists() if not path.startswith("(") else False
            existence = "EXISTS" if file_exists else "NOT FOUND"
            lines.append(f"  - `{path}` [{existence}]")

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


def _log_timing(operation: str, start_time: float, **extra: Any) -> None:
    """Log operation timing (structured logging)."""
    elapsed_ms = int((time.time() - start_time) * 1000)
    # Only log in debug mode to avoid noise
    pass


async def handle_rlm_analyze(
    arguments: dict[str, Any],
    get_instances: Callable,
) -> list[TextContent]:
    """Handle rlm_analyze tool call with semantic caching and result verification."""
    start_time = time.time()

    session_id = arguments.get("session_id") or str(uuid.uuid4())[:8]

    query = arguments.get("query", "")
    paths = arguments.get("paths", [])
    query_mode = arguments.get("query_mode", "auto")
    scan_mode = arguments.get("scan_mode", "auto")
    min_confidence = arguments.get("min_confidence", "MEDIUM")
    include_quality = arguments.get("include_quality", False)
    include_skipped_signatures = arguments.get("include_skipped_signatures", False)

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

    for path in paths:
        if not isinstance(path, str):
            return [TextContent(type="text", text=f"Error: path must be string, got {type(path).__name__}")]
        is_valid, error = validate_path(path)
        if not is_valid:
            return [TextContent(type="text", text=f"Error: invalid path '{path}': {error}")]

    rlm_config, _, file_collector, rlm_processor, _, semantic_cache, result_verifier, project_analyzer = get_instances()

    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    await write_progress(session_id, "start", "Starting analysis...", 0, {"query": query[:100], "paths": paths})

    collection = await file_collector.collect_paths_async(paths)

    await write_progress(session_id, "files_collected", f"Collected {collection.file_count} files", 20, {"file_count": collection.file_count})

    if collection.file_count == 0:
        error_msg = "## No Matching Files Found\n\n"
        if collection.errors:
            error_msg += f"**Errors:** {'; '.join(collection.errors)}\n\n"

        if collection.skipped_files:
            error_msg += _format_skipped_files_summary(collection.skipped_files, max_display=20, include_signatures=include_skipped_signatures)
        else:
            error_msg += "No files matched the configured extensions.\n\n"
            error_msg += "**Included extensions:** " + ", ".join(sorted(rlm_config.included_extensions)[:20]) + "...\n"
            error_msg += "**Skipped directories:** " + ", ".join(sorted(rlm_config.skipped_directories)[:10]) + "..."

        return [TextContent(type="text", text=error_msg)]

    context_summary = ", ".join(f.relative_path for f in collection.files[:50])
    cached_response, similarity = semantic_cache.get_similar(query, context_summary)

    if cached_response:
        output = f"## Cached Result (similarity: {similarity:.2f})\n\n{cached_response}"
        return [TextContent(type="text", text=output)]

    project_info = project_analyzer.analyze(collection)
    query_analysis = rlm_processor.analyze_query_quality(query)

    content = collection.get_combined_content(include_headers=True)

    # Determine effective query_mode
    effective_query_mode = query_mode
    if query_mode == "auto":
        is_literal, search_terms = is_literal_search_query(query)
        if is_literal and search_terms:
            effective_query_mode = "literal"
        elif query_analysis["is_broad"] or len(query.split()) <= 5:
            effective_query_mode = "scanner"
        elif len(query.split()) > 15 or "(" in query or query.count(",") >= 2:
            effective_query_mode = "semantic"
        else:
            effective_query_mode = "scanner"

    print(f"[RLM] Using query_mode='{effective_query_mode}', scan_mode='{scan_mode}'", file=sys.stderr)

    # MODE: LITERAL
    if effective_query_mode == "literal":
        is_literal, search_terms = is_literal_search_query(query)
        if not search_terms:
            search_terms = [w for w in query.split() if len(w) > 3][:5]

        if search_terms:
            literal_result = perform_literal_search(content, search_terms)
            processing_time = int((time.time() - start_time) * 1000)
            literal_result += f"\n*Scanned {collection.file_count} files in {processing_time}ms (literal mode)*"

            if collection.skipped_files:
                literal_result += _format_skipped_files_summary(collection.skipped_files, include_signatures=include_skipped_signatures)

            return [TextContent(type="text", text=literal_result)]

    # MODE: SEMANTIC or CUSTOM
    if effective_query_mode in ("semantic", "custom"):
        await write_progress(session_id, "analyzing", f"Running {effective_query_mode} analysis...", 50, {"query_mode": effective_query_mode})

        def progress_log(msg: str) -> None:
            print(f"[RLM Progress] {msg}", file=sys.stderr)

        result = await rlm_processor.process_with_decomposition(query, collection, progress_callback=progress_log)

        output = rlm_processor.format_result(result)
        verified_output, verification_stats = result_verifier.verify_findings(output, collection)

        semantic_cache.set(query, context_summary, verified_output)

        if project_info.key_files:
            verified_output += f"\n\n## Project Context\n- Type: {project_info.project_type}\n- Key files: {', '.join(project_info.key_files[:10])}"

        verified_output += f"\n\n*Scanned {collection.file_count} files in {int((time.time() - start_time) * 1000)}ms ({effective_query_mode} mode)*"

        if collection.skipped_files:
            verified_output += _format_skipped_files_summary(collection.skipped_files, include_signatures=include_skipped_signatures)

        await write_progress(session_id, "complete", "Analysis complete", 100, {"query_mode": effective_query_mode})
        return [TextContent(type="text", text=verified_output)]

    # MODE: SCANNER
    detected_type = query_analysis["query_type"]
    if scan_mode == "auto" and detected_type != "general":
        type_to_scan_mode = {
            "ios": "ios", "security": "security", "quality": "quality",
            "web": "web", "react": "web", "vue": "web", "angular": "web",
            "frontend": "frontend", "rust": "rust", "node": "node",
            "nodejs": "node", "backend": "backend",
        }
        if detected_type in type_to_scan_mode:
            scan_mode = type_to_scan_mode[detected_type]

    if scan_mode == "custom":
        arguments_copy = dict(arguments)
        arguments_copy["query_mode"] = "semantic"
        arguments_copy["scan_mode"] = "auto"
        return await handle_rlm_analyze(arguments_copy, get_instances)

    structured_results: list[ToolResult] = []
    tools_to_run: list[str] = []

    if scan_mode != "auto":
        tools = StructuredTools(content)

        if scan_mode == "ios":
            structured_results = tools.run_ios_scan(min_confidence=min_confidence, include_quality=include_quality)
            tools_to_run = ["ios_scan"]
        elif scan_mode == "ios-strict":
            structured_results = tools.run_ios_scan(min_confidence="HIGH", include_quality=False)
            tools_to_run = ["ios_scan (strict)"]
        elif scan_mode == "security":
            structured_results = tools.run_security_scan(min_confidence=min_confidence, include_quality=include_quality)
            tools_to_run = ["security_scan"]
        elif scan_mode == "quality":
            structured_results = tools.run_quality_scan(min_confidence=min_confidence)
            tools_to_run = ["quality_scan"]
        elif scan_mode == "web":
            structured_results = tools.run_web_scan(min_confidence=min_confidence)
            tools_to_run = ["web_scan"]
        elif scan_mode == "rust":
            structured_results = tools.run_rust_scan(min_confidence=min_confidence)
            tools_to_run = ["rust_scan"]
        elif scan_mode == "node":
            structured_results = tools.run_node_scan(min_confidence=min_confidence)
            tools_to_run = ["node_scan"]
        elif scan_mode == "frontend":
            structured_results = tools.run_frontend_scan(min_confidence=min_confidence)
            tools_to_run = ["frontend_scan"]
        elif scan_mode == "backend":
            structured_results = tools.run_backend_scan(min_confidence=min_confidence)
            tools_to_run = ["backend_scan"]
        elif scan_mode == "all":
            structured_results = (
                tools.run_ios_scan(min_confidence=min_confidence, include_quality=True) +
                tools.run_security_scan(min_confidence=min_confidence, include_quality=False) +
                tools.run_web_scan(min_confidence=min_confidence) +
                tools.run_rust_scan(min_confidence=min_confidence) +
                tools.run_node_scan(min_confidence=min_confidence)
            )
            tools_to_run = ["ios_scan", "security_scan", "quality_scan", "web_scan", "rust_scan", "node_scan"]

        structured_output, _ = format_tool_results(structured_results, tools_to_run)
    else:
        routing_decision = await llm_route_tools(query, rlm_config)
        tools_to_run = routing_decision.get("tools", [])
        tool_params = routing_decision.get("params", {})

        if tools_to_run:
            structured_output, structured_results = execute_routed_tools(
                tools_to_run, tool_params, content, min_confidence
            )
        else:
            structured_output, structured_results = run_structured_tools_for_query_type(
                query_analysis["query_type"], content, query.lower(), min_confidence
            )

    total_findings = sum(r.count for r in structured_results) if structured_results else 0
    high_conf_findings = sum(len(r.high_confidence) for r in structured_results) if structured_results else 0

    if total_findings > 0:
        structured_output += f"\n\n*Scanned {collection.file_count} files in {int((time.time() - start_time) * 1000)}ms (scanner mode)*"

        if collection.skipped_files:
            structured_output += _format_skipped_files_summary(collection.skipped_files, include_signatures=include_skipped_signatures)

        if high_conf_findings >= 3 or total_findings >= 10:
            return [TextContent(type="text", text=structured_output)]

    await write_progress(session_id, "analyzing", "Running additional RLM analysis...", 50, {"scan_mode": scan_mode})

    def progress_log(msg: str) -> None:
        print(f"[RLM Progress] {msg}", file=sys.stderr)

    result = await rlm_processor.process_with_decomposition(query, collection, progress_callback=progress_log)

    await write_progress(session_id, "verifying", "Verifying findings...", 80, {"chunks_processed": len(result.chunk_results)})

    output = rlm_processor.format_result(result)
    verified_output, verification_stats = result_verifier.verify_findings(output, collection)

    semantic_cache.set(query, context_summary, verified_output)

    if project_info.key_files:
        project_context = f"\n\n## Project Context\n- Type: {project_info.project_type}\n- Key files from docs: {', '.join(project_info.key_files[:10])}"
        verified_output += project_context

    if structured_output:
        final_output = structured_output + "\n\n---\n\n## Additional LLM Analysis\n\n" + verified_output
    else:
        final_output = verified_output

    await write_progress(session_id, "complete", "Analysis complete", 100, {
        "duration_ms": int((time.time() - start_time) * 1000),
        "session_id": session_id
    })

    return [TextContent(type="text", text=final_output)]


async def handle_rlm_query_text(
    arguments: dict[str, Any],
    get_instances: Callable,
) -> list[TextContent]:
    """Handle rlm_query_text tool call with result verification."""
    start_time = time.time()

    query = arguments.get("query", "")
    text = arguments.get("text", "")

    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    if len(query) > MAX_QUERY_LENGTH:
        return [TextContent(type="text", text=f"Error: query too long ({len(query)} > {MAX_QUERY_LENGTH} chars)")]

    if not text:
        return [TextContent(type="text", text="Error: text is required")]

    if len(text) > MAX_TEXT_LENGTH:
        return [TextContent(type="text", text=f"Error: text too long ({len(text)} > {MAX_TEXT_LENGTH} chars)")]

    rlm_config, _, file_collector, rlm_processor, _, semantic_cache, result_verifier, _ = get_instances()

    errors = rlm_config.validate()
    if errors:
        return [TextContent(type="text", text=f"Configuration error: {'; '.join(errors)}")]

    context_summary = f"text:{len(text)}chars"
    cached_response, similarity = semantic_cache.get_similar(query, context_summary)

    if cached_response:
        output = f"## Cached Result (similarity: {similarity:.2f})\n\n{cached_response}"
        return [TextContent(type="text", text=output)]

    collection = await file_collector.collect_text_async(text, "input_text")

    result = await rlm_processor.process_with_decomposition(query, collection)

    output = rlm_processor.format_result(result)

    semantic_cache.set(query, context_summary, output)

    return [TextContent(type="text", text=output)]


async def handle_rlm_status(
    arguments: dict[str, Any],
    get_instances: Callable,
    memory_monitor: Any,
    metrics_collector: Any,
) -> list[TextContent]:
    """Handle rlm_status tool call with semantic cache stats."""
    from ..rlm_processor import get_trajectory_logger, get_model_selector
    from .memory import get_memory_count

    rlm_config, server_config, file_collector, rlm_processor, _, semantic_cache, _, _ = get_instances()

    memory_stats = memory_monitor.get_stats()
    metrics_stats = metrics_collector.get_stats()

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
            "base_timeout_seconds": rlm_config.base_timeout_seconds,
            "max_timeout_seconds": rlm_config.max_timeout_seconds,
            "max_iterations": rlm_config.max_iterations,
            "prompt_cache_enabled": rlm_config.use_prompt_cache,
            "prompt_cache_ttl": rlm_config.prompt_cache_ttl,
            "cache_usage_tracking": rlm_config.track_cache_usage,
        },
        "memory_entries": get_memory_count(),
        "performance": {
            "token_cache": file_collector.get_cache_stats(),
            "processor": rlm_processor.get_stats(),
            "semantic_cache": semantic_cache.get_stats(),
            "metrics": metrics_stats,
        },
        "resources": {
            "memory": memory_stats,
        },
        "recent_trajectories": trajectory_logger.get_recent(5),
        "model_selector": model_selector.get_stats(),
    }

    if memory_stats.get("usage_percent", 0) > 0.8:
        status["warnings"] = ["Memory usage above 80% threshold"]

    errors = rlm_config.validate()
    if errors:
        status["errors"] = errors

    output = json.dumps(status, indent=2)
    return [TextContent(type="text", text=output)]
