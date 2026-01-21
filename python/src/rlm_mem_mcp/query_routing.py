"""
Query Routing for RLM Processing.

Provides query analysis and routing utilities:
- Session context preservation
- Iterative query refinement
- Query type detection
"""

import time


# =============================================================================
# L8: ITERATIVE REFINEMENT - SESSION CONTEXT PRESERVATION
# =============================================================================

# Global session context store
_session_contexts: dict[str, dict] = {}


def get_session_context(session_id: str) -> dict | None:
    """L8: Retrieve prior session context for iterative refinement."""
    return _session_contexts.get(session_id)


def save_session_context(session_id: str, context: dict):
    """L8: Save session context for follow-up queries."""
    # Limit stored sessions
    if len(_session_contexts) >= 100:
        # Remove oldest
        oldest = min(_session_contexts.keys(), key=lambda k: _session_contexts[k].get('timestamp', 0))
        del _session_contexts[oldest]

    _session_contexts[session_id] = {
        **context,
        'timestamp': time.time()
    }


def build_iterative_query(query: str, session_id: str) -> str:
    """
    L8: Build query that incorporates prior session context.

    This enables iterative refinement without re-analyzing everything.
    """
    prior = get_session_context(session_id)
    if not prior:
        return query

    findings_summary = prior.get('findings_summary', '')
    files_analyzed = prior.get('files_analyzed', [])

    if findings_summary:
        return f"""Prior analysis found:
{findings_summary}

Files already analyzed: {', '.join(files_analyzed[:10])}

Follow-up query: {query}

Focus on:
1. Areas not yet covered
2. Deeper analysis of flagged issues
3. New patterns not in prior findings"""

    return query


def clear_session_context(session_id: str):
    """L8: Clear session context."""
    if session_id in _session_contexts:
        del _session_contexts[session_id]


def detect_query_type(query: str) -> tuple[str, str]:
    """
    Detect query type for routing and timeout calculation.

    Returns:
        (query_type, complexity) - e.g., ("pattern", "normal") or ("semantic", "deep")
    """
    query_lower = query.lower()

    # Pattern queries - fast, regex-based
    pattern_indicators = [
        "find", "search", "grep", "locate", "list", "count",
        "force unwrap", "secrets", "todo", "fixme", "import"
    ]

    # Semantic queries - need LLM analysis
    semantic_indicators = [
        "analyze", "explain", "understand", "review", "assess",
        "architecture", "design", "refactor", "improve", "why",
        "how does", "what is the purpose", "security audit"
    ]

    # Complexity indicators
    deep_indicators = [
        "thorough", "comprehensive", "all", "every", "complete",
        "deep", "detailed", "full analysis"
    ]

    is_semantic = any(ind in query_lower for ind in semantic_indicators)
    is_deep = any(ind in query_lower for ind in deep_indicators)

    query_type = "semantic" if is_semantic else "pattern"
    complexity = "deep" if is_deep else ("normal" if is_semantic else "simple")

    return query_type, complexity


def is_security_query(query: str) -> bool:
    """Check if query is security-related."""
    security_keywords = [
        'security', 'vulnerability', 'exploit', 'injection', 'xss',
        'csrf', 'secret', 'password', 'credential', 'auth', 'token',
        'api key', 'hardcoded', 'sensitive', 'leak'
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in security_keywords)


def is_ios_query(query: str) -> bool:
    """Check if query is iOS/Swift-related."""
    ios_keywords = [
        'swift', 'ios', 'xcode', 'swiftui', 'uikit', 'cocoa',
        'force unwrap', 'optional', 'weak self', 'retain cycle',
        '@mainactor', 'sendable', '@observable', '@stateobject'
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in ios_keywords)


def is_quality_query(query: str) -> bool:
    """Check if query is code quality-related."""
    quality_keywords = [
        'quality', 'refactor', 'complexity', 'long function',
        'todo', 'fixme', 'hack', 'debt', 'smell', 'clean'
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in quality_keywords)


def suggest_scan_mode(query: str) -> str:
    """Suggest appropriate scan mode based on query."""
    if is_security_query(query):
        return "security"
    elif is_ios_query(query):
        return "ios"
    elif is_quality_query(query):
        return "quality"
    else:
        return "auto"
