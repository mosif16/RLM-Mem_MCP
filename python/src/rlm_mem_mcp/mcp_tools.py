"""
MCP Tool Routing and Utilities.

Provides tool routing logic for the RLM-Mem MCP server:
- Literal search detection and fast-path
- LLM-based tool routing
- Tool execution and result formatting
"""

import re
import sys
from typing import Any

from .structured_tools import StructuredTools, ToolResult


# =============================================================================
# LITERAL SEARCH DETECTION AND FAST-PATH
# =============================================================================


def extract_quoted_strings(query: str) -> list[str]:
    """
    Extract quoted strings from a query.

    Handles both single and double quotes:
    - "tasks-main" -> tasks-main
    - 'task tracking' -> task tracking

    Returns:
        List of strings found in quotes
    """
    pattern = r'["\']([^"\']+)["\']'
    matches = re.findall(pattern, query)
    return matches


def is_literal_search_query(query: str) -> tuple[bool, list[str]]:
    """
    Detect if a query is asking for literal string search.

    Indicators of literal search:
    1. Contains quoted strings ("foo", 'bar')
    2. Uses search keywords: "find files", "search for", "grep", "containing"
    3. NOT asking for analysis/audit/review/security

    Returns:
        (is_literal_search, list_of_search_terms)
    """
    query_lower = query.lower()

    # Extract quoted strings
    quoted_strings = extract_quoted_strings(query)

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

    has_analysis_intent = any(kw in query_lower for kw in analysis_keywords)
    has_search_intent = any(kw in query_lower for kw in search_keywords)

    # If we have quoted strings AND search keywords AND NOT analysis keywords
    if quoted_strings and has_search_intent and not has_analysis_intent:
        return True, quoted_strings

    # If we have quoted strings but no analysis keywords and query is short
    if quoted_strings and not has_analysis_intent and len(query.split()) < 15:
        return True, quoted_strings

    return False, []


def perform_literal_search(content: str, search_terms: list[str]) -> str:
    """
    Perform grep-like literal search on content.

    This is the fast-path that bypasses LLM routing entirely.

    Args:
        content: The combined file content (with ### File: headers)
        search_terms: List of literal strings to search for

    Returns:
        Formatted search results
    """
    current_file = "unknown"
    line_number = 0
    in_fence = False
    files_seen = 0
    unknown_matches = 0

    findings: dict[str, list[dict]] = {term: [] for term in search_terms}

    for line in content.split('\n'):
        if line.startswith("### File:"):
            current_file = line.replace("### File:", "").strip()
            line_number = 0
            in_fence = False
            files_seen += 1
            continue

        stripped = line.strip()
        if stripped.startswith('```'):
            in_fence = not in_fence
            continue

        if in_fence:
            line_number += 1
            line_lower = line.lower()
            for term in search_terms:
                if term.lower() in line_lower:
                    if current_file == "unknown":
                        unknown_matches += 1
                    findings[term].append({
                        "file": current_file,
                        "line": line_number,
                        "content": line.strip()[:200]
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
            by_file: dict[str, list[dict]] = {}
            for match in matches:
                file = match["file"]
                if file not in by_file:
                    by_file[file] = []
                by_file[file].append(match)

            for file, file_matches in by_file.items():
                output_parts.append(f"\n**{file}**\n")
                for m in file_matches[:10]:
                    output_parts.append(f"  Line {m['line']}: `{m['content']}`\n")
                if len(file_matches) > 10:
                    output_parts.append(f"  ... and {len(file_matches) - 10} more matches\n")

    output_parts.append(f"\n---\n*Total: {total_matches} matches across {len(search_terms)} search term(s)*\n")

    if files_seen == 0 and total_matches > 0:
        output_parts.append("\n⚠️ **Warning:** No `### File:` headers found in content.\n")
    elif unknown_matches > 0:
        output_parts.append(f"\n⚠️ **Warning:** {unknown_matches} match(es) have 'unknown' file location.\n")

    output_parts.append("*Tip: Use native Grep tool for better performance on literal searches.*\n")

    return "".join(output_parts)


# =============================================================================
# TOOL ROUTER PROMPT
# =============================================================================

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
"""


# =============================================================================
# LLM TOOL ROUTING
# =============================================================================


async def llm_route_tools(query: str, config: Any) -> dict[str, Any]:
    """
    Use LLM to intelligently decide which tools to run.

    This is Layer 1 of the double-layer system:
    - Layer 1: LLM decides WHAT to run (intelligent routing)
    - Layer 2: Structured tools execute deterministically

    Returns:
        {"tools": ["tool_name", ...], "params": {"tool_name": {"param": "value"}}}
    """
    import httpx
    import json as json_mod

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
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": TOOL_ROUTER_PROMPT},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    "max_tokens": 200,
                    "temperature": 0
                }
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()

            if not content or content.lower() in ["none", "null", "n/a"]:
                return {"tools": [], "params": {}}

            result = json_mod.loads(content)

            if not isinstance(result, dict):
                return {"tools": [], "params": {}}
            if "tools" not in result:
                result["tools"] = []
            if "params" not in result:
                result["params"] = {}

            return result

    except Exception as e:
        print(f"[RLM Router] LLM routing failed: {e}, falling back to keywords", file=sys.stderr)
        return {"tools": [], "params": {}}


# =============================================================================
# TOOL EXECUTION
# =============================================================================


def execute_routed_tools(
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
            elif tool_name == "find_mainactor_issues":
                all_results.append(tools.find_mainactor_issues())
            elif tool_name == "find_task_cancellation_issues":
                all_results.append(tools.find_task_cancellation_issues())
            elif tool_name == "find_stateobject_issues":
                all_results.append(tools.find_stateobject_issues())
            elif tool_name == "find_async_await_issues":
                all_results.append(tools.find_async_await_issues())
            elif tool_name == "find_sendable_issues":
                all_results.append(tools.find_sendable_issues())
            elif tool_name == "find_memory_management_issues":
                all_results.append(tools.find_memory_management_issues())
            elif tool_name == "find_error_handling_issues":
                all_results.append(tools.find_error_handling_issues())
            elif tool_name == "find_swiftui_performance_issues":
                all_results.append(tools.find_swiftui_performance_issues())
            elif tool_name == "find_accessibility_issues":
                all_results.append(tools.find_accessibility_issues())
            elif tool_name == "find_localization_issues":
                all_results.append(tools.find_localization_issues())
        except Exception as e:
            print(f"[RLM Router] Tool {tool_name} failed: {e}", file=sys.stderr)

    # Apply global confidence filter
    if min_confidence.upper() != "LOW":
        all_results = tools._filter_by_confidence(all_results, min_confidence)

    return format_tool_results(all_results, tools_to_run)


def format_tool_results(
    all_results: list[ToolResult],
    tools_run: list[str]
) -> tuple[str, list[ToolResult]]:
    """Format tool results into markdown output with severity grouping."""
    if not all_results:
        return "", []

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
        files_scanned = sum(r.files_scanned for r in all_results) if all_results else 0
        return (
            f"## Scan Results\n\n"
            f"**Tools Run**: {', '.join(tools_run)}\n"
            f"**Files Scanned**: {files_scanned}\n"
            f"**Findings**: None - code looks clean!\n\n"
            f"*Tip: If you expected findings, try using scan_mode='all' or min_confidence='LOW'.*\n"
        ), all_results

    output_parts = [f"## Scan Results\n"]

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

    critical_high = []
    medium_low = []

    for result in all_results:
        for finding in result.findings:
            sev = finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity)
            if sev in ("CRITICAL", "HIGH"):
                critical_high.append((result.tool_name, finding))
            else:
                medium_low.append((result.tool_name, finding))

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

    if medium_low:
        if len(medium_low) > 10:
            output_parts.append(f"\n### Medium & Low Priority ({len(medium_low)} findings)\n")
            output_parts.append("<details><summary>Click to expand</summary>\n\n")
            for tool_name, finding in medium_low[:20]:
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


def run_structured_tools_for_query_type(
    query_type: str,
    content: str,
    query_lower: str,
    min_confidence: str = "LOW"
) -> tuple[str, list[ToolResult]]:
    """
    Fallback: Auto-route to structured tools based on detected query type.
    Used when LLM routing fails or returns empty.
    """
    tools = StructuredTools(content)
    all_results: list[ToolResult] = []
    tools_run: list[str] = []

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

    return format_tool_results(all_results, tools_run)
