"""
Parallel execution utilities for RLM tools (v2.5 - 2-4x faster).

Provides parallelism for batch tool operations and multi-pattern searching.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .common_types import ToolResult, get_optimal_workers
from .ripgrep_tools import RgMatch, rg_search


def parallel_scan(
    tools_instance: "StructuredTools",  # noqa: F821
    scan_functions: list[Callable[[], ToolResult]],
    max_workers: Optional[int] = None,
) -> list[ToolResult]:
    """
    Run multiple scan functions in parallel.

    v2.8: Dynamic worker sizing for better CPU utilization (2-4x faster than sequential).

    Args:
        tools_instance: StructuredTools instance (for context)
        scan_functions: List of bound methods to call (e.g., [tools.find_secrets, tools.find_xss])
        max_workers: Maximum parallel threads (default: auto-detect from CPU count)

    Returns:
        List of ToolResult from all functions

    Example:
        tools = StructuredTools(content)
        results = parallel_scan(tools, [
            tools.find_secrets,
            tools.find_sql_injection,
            tools.find_xss_vulnerabilities,
        ])
    """
    results = []

    # v2.8: Use dynamic worker sizing if not specified
    if max_workers is None:
        max_workers = get_optimal_workers()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_func = {
            executor.submit(func): func.__name__
            for func in scan_functions
        }

        # Collect results as they complete
        for future in as_completed(future_to_func):
            func_name = future_to_func[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Create error result for failed function
                results.append(ToolResult(
                    tool_name=func_name,
                    findings=[],
                    summary=f"Error: {str(e)}",
                    errors=[str(e)],
                ))

    return results


def parallel_rg_search(
    patterns: list[str],
    paths: Optional[list[str] | str] = None,
    max_workers: Optional[int] = None,
    deduplicate: bool = True,
    **kwargs,
) -> list[RgMatch]:
    """
    Search multiple patterns in parallel.

    v2.8: Dynamic worker sizing - useful for scanning with many patterns simultaneously.

    Args:
        patterns: List of regex patterns to search
        paths: Where to search
        max_workers: Maximum parallel threads (default: auto-detect from CPU count)
        deduplicate: Remove duplicate matches (same file:line)
        **kwargs: Additional args passed to rg_search

    Returns:
        Combined list of RgMatch from all patterns

    Example:
        # Search for multiple secret patterns at once
        matches = parallel_rg_search([
            r"API_KEY\\s*=",
            r"SECRET\\s*=",
            r"PASSWORD\\s*=",
        ])
    """
    all_matches = []

    # v2.8: Use dynamic worker sizing if not specified
    if max_workers is None:
        max_workers = get_optimal_workers()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(rg_search, pattern, paths, **kwargs)
            for pattern in patterns
        ]

        for future in as_completed(futures):
            try:
                matches = future.result()
                all_matches.extend(matches)
            except Exception:
                pass

    if deduplicate:
        # Remove duplicates based on file:line
        seen = set()
        unique_matches = []
        for m in all_matches:
            key = (m.file, m.line)
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)
        return unique_matches

    return all_matches
