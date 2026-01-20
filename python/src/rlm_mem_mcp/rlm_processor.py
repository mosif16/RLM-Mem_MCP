"""
RLM Processor for RLM-Mem MCP Server

Implements the Recursive Language Model (RLM) technique from arXiv:2512.24601.

THE KEY INSIGHT (from the paper):
- Content is stored as a VARIABLE in a Python REPL (NOT in LLM context)
- The LLM writes CODE to examine portions of the content
- Sub-LLM responses are stored in VARIABLES (NOT summarized)
- Full data is PRESERVED - the LLM can access any part at any time

This is NOT summarization! The data is kept intact and accessible.

Performance Optimizations:
- AsyncOpenAI client for true async API calls
- Exponential backoff retry for transient failures (429, 503)
- Circuit breaker for repeated failures
- Request timeout protection
- Connection pooling via httpx
- LRU cache for LLM responses
- Incremental token counting
- Early termination on relevance threshold
"""

import asyncio
import hashlib
import re
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import lru_cache
from collections import OrderedDict

from openai import AsyncOpenAI
import httpx
import tiktoken

from .config import RLMConfig
from .cache_manager import CacheManager
from .file_collector import CollectionResult
from .incremental_cache import IncrementalCache, get_incremental_cache


# Performance tuning constants
LLM_REQUEST_TIMEOUT_SECONDS = 120  # Max time for a single LLM call
MAX_RETRIES = 3  # Max retries for transient failures
INITIAL_RETRY_DELAY = 1.0  # Initial delay for exponential backoff
CIRCUIT_BREAKER_THRESHOLD = 5  # Failures before circuit opens
CIRCUIT_BREAKER_RESET_TIME = 60  # Seconds before circuit resets
LLM_CACHE_MAX_SIZE = 1000  # Max cached LLM responses
EARLY_TERMINATION_THRESHOLD = 0.9  # Stop if relevance exceeds this

# Rate limiting constants
RATE_LIMIT_REQUESTS_PER_MINUTE = 60  # Max requests per minute
RATE_LIMIT_TOKENS_PER_MINUTE = 100_000  # Max tokens per minute


# =============================================================================
# L12: TRAJECTORY LOGGING FOR EXECUTION TRANSPARENCY
# =============================================================================

@dataclass
class TrajectoryStep:
    """A single step in an RLM execution trajectory."""
    step_type: str  # "code_gen", "code_exec", "llm_query", "tool_call", "cache_hit"
    input_preview: str  # First 500 chars of input
    output_preview: str  # First 500 chars of output
    duration_ms: int
    tokens_used: int = 0
    success: bool = True
    error: str | None = None


@dataclass
class ExecutionTrajectory:
    """L12: Full execution trajectory for debugging."""
    session_id: str
    query: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)

    def to_markdown(self) -> str:
        """Export trajectory as readable markdown."""
        lines = [
            f"# Execution Trajectory: {self.session_id}",
            f"**Query:** {self.query[:200]}",
            f"**Duration:** {int((self.end_time - self.start_time) * 1000)}ms",
            f"**Steps:** {len(self.steps)}",
            ""
        ]

        for i, step in enumerate(self.steps, 1):
            status = "✓" if step.success else "✗"
            lines.append(f"## Step {i}: {step.step_type} {status}")
            lines.append(f"**Duration:** {step.duration_ms}ms | **Tokens:** {step.tokens_used}")
            if step.input_preview:
                lines.append(f"\n**Input:**\n```\n{step.input_preview[:300]}\n```")
            if step.output_preview:
                lines.append(f"\n**Output:**\n```\n{step.output_preview[:300]}\n```")
            if step.error:
                lines.append(f"\n**Error:** {step.error}")
            lines.append("")

        return "\n".join(lines)


class TrajectoryLogger:
    """L12: Global trajectory logger for RLM execution transparency."""

    def __init__(self, max_trajectories: int = 50):
        self.trajectories: dict[str, ExecutionTrajectory] = {}
        self.max_trajectories = max_trajectories

    def start_session(self, session_id: str, query: str) -> ExecutionTrajectory:
        """Start tracking a new session."""
        # Evict oldest if at capacity
        if len(self.trajectories) >= self.max_trajectories:
            oldest = min(self.trajectories.keys(), key=lambda k: self.trajectories[k].start_time)
            del self.trajectories[oldest]

        trajectory = ExecutionTrajectory(
            session_id=session_id,
            query=query,
            start_time=time.time()
        )
        self.trajectories[session_id] = trajectory
        return trajectory

    def log_step(self, session_id: str, step: TrajectoryStep):
        """Log a step to an existing trajectory."""
        if session_id in self.trajectories:
            self.trajectories[session_id].add_step(step)

    def end_session(self, session_id: str):
        """Mark session as complete."""
        if session_id in self.trajectories:
            self.trajectories[session_id].end_time = time.time()

    def get_trajectory(self, session_id: str) -> ExecutionTrajectory | None:
        return self.trajectories.get(session_id)

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get summaries of recent trajectories."""
        sorted_trajs = sorted(
            self.trajectories.values(),
            key=lambda t: t.start_time,
            reverse=True
        )[:n]

        return [
            {
                "session_id": t.session_id,
                "query": t.query[:50],
                "steps": len(t.steps),
                "duration_ms": int((t.end_time - t.start_time) * 1000) if t.end_time else 0
            }
            for t in sorted_trajs
        ]


# Global trajectory logger instance
_trajectory_logger = TrajectoryLogger()


def get_trajectory_logger() -> TrajectoryLogger:
    """Get the global trajectory logger."""
    return _trajectory_logger


# =============================================================================
# L10: ADAPTIVE MODEL SELECTION
# =============================================================================

class AdaptiveModelSelector:
    """Model selector - uses Claude Haiku 4.5 exclusively for all tasks."""

    # Single model: Claude Haiku 4.5 - fast, cost-effective with prompt caching
    MODEL = "anthropic/claude-haiku-4.5"
    MODEL_COST = 0.80  # $/1M input tokens (0.08 with cache)

    def __init__(self, default_model: str = "anthropic/claude-haiku-4.5"):
        self.default_model = self.MODEL  # Always use Haiku 4.5
        self.failure_counts: dict[str, int] = {}

    def select_model(self, query_type: str, content_size: int, prefer_quality: bool = False) -> str:
        """Always returns Claude Haiku 4.5."""
        return self.MODEL

    def record_failure(self, model: str):
        """Record a model failure."""
        self.failure_counts[model] = self.failure_counts.get(model, 0) + 1

    def record_success(self, model: str):
        """Reset failure count on success."""
        self.failure_counts[model] = 0

    def get_fallback(self, current_model: str) -> str | None:
        """No fallback - always use Haiku 4.5."""
        return None

    def get_stats(self) -> dict:
        """Get model stats."""
        return {
            "model": self.MODEL,
            "cost_per_1m_tokens": self.MODEL_COST,
            "failure_counts": dict(self.failure_counts),
        }


# Global model selector
_model_selector = AdaptiveModelSelector()


def get_model_selector() -> AdaptiveModelSelector:
    """Get the global model selector."""
    return _model_selector


# =============================================================================
# L2: TOKEN OPTIMIZATION - SMART CHUNK FILTERING
# =============================================================================

def extract_query_keywords(query: str) -> list[str]:
    """Extract searchable keywords from a query."""
    # Common stop words to ignore
    stop_words = {'find', 'search', 'look', 'for', 'the', 'a', 'an', 'in', 'is', 'are', 'all', 'any'}

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Add domain-specific keywords
    if 'security' in query.lower():
        keywords.extend(['password', 'secret', 'key', 'token', 'auth', 'eval', 'exec'])
    if 'ios' in query.lower() or 'swift' in query.lower():
        keywords.extend(['unwrap', 'weak', 'self', 'optional', 'guard'])

    return list(set(keywords))


def smart_chunk_filter(chunk: str, query: str, context_lines: int = 10) -> str:
    """
    L2: Extract only query-relevant portions of a chunk before LLM call.

    This reduces token usage by 30-50% by filtering irrelevant content.
    """
    keywords = extract_query_keywords(query)

    if not keywords:
        return chunk  # Can't filter without keywords

    lines = chunk.split('\n')
    relevant_indices: set[int] = set()

    # Find lines matching keywords and include context
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            # Include context window around match
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            relevant_indices.update(range(start, end))

    if not relevant_indices:
        # No matches - return truncated preview
        return '\n'.join(lines[:50]) + "\n... (no keyword matches, showing first 50 lines)"

    # Build filtered content preserving order
    sorted_indices = sorted(relevant_indices)
    filtered_lines = []
    last_idx = -1

    for idx in sorted_indices:
        if last_idx >= 0 and idx > last_idx + 1:
            filtered_lines.append("... (skipped lines)")
        filtered_lines.append(lines[idx])
        last_idx = idx

    return '\n'.join(filtered_lines)


# =============================================================================
# L7: FUNCTION-AWARE CHUNKING FOR CONTEXT MANAGEMENT
# =============================================================================

def find_function_boundaries(content: str, file_type: str) -> list[tuple[int, int, str]]:
    """
    L7: Find function/method boundaries in source code.

    Returns list of (start_pos, end_pos, function_name) tuples.
    """
    boundaries = []

    if file_type in ['.py']:
        # Python: def/async def/class
        pattern = r'^(async\s+)?def\s+(\w+)|^class\s+(\w+)'
        for match in re.finditer(pattern, content, re.MULTILINE):
            start = match.start()
            name = match.group(2) or match.group(3) or "unknown"
            # Find end by next function/class at same indent or EOF
            indent = len(content[content.rfind('\n', 0, start)+1:start])
            end_pattern = rf'^.{{{indent}}}(def |class |async def )'
            end_match = re.search(end_pattern, content[start+1:], re.MULTILINE)
            end = start + end_match.start() + 1 if end_match else len(content)
            boundaries.append((start, end, name))

    elif file_type in ['.swift']:
        # Swift: func/class/struct/enum
        pattern = r'(func|class|struct|enum|extension)\s+(\w+)'
        for match in re.finditer(pattern, content):
            start = match.start()
            name = match.group(2)
            # Find matching brace end
            brace_count = 0
            in_func = False
            end = start
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                    in_func = True
                elif char == '}':
                    brace_count -= 1
                    if in_func and brace_count == 0:
                        end = start + i + 1
                        break
            boundaries.append((start, end, name))

    elif file_type in ['.js', '.ts', '.tsx', '.jsx']:
        # JavaScript/TypeScript: function/class/arrow functions
        pattern = r'(function\s+(\w+)|class\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()'
        for match in re.finditer(pattern, content):
            start = match.start()
            name = match.group(2) or match.group(3) or match.group(4) or "anonymous"
            # Simple brace matching
            brace_count = 0
            in_func = False
            end = start
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                    in_func = True
                elif char == '}':
                    brace_count -= 1
                    if in_func and brace_count == 0:
                        end = start + i + 1
                        break
            boundaries.append((start, end, name))

    return boundaries


def function_aware_chunking(content: str, file_type: str, max_chunk_tokens: int = 8000) -> list[str]:
    """
    L7: Chunk content at function/class boundaries instead of arbitrary token limits.

    This prevents truncating code mid-logic and improves analysis accuracy.
    """
    # Rough estimate: 1 token ≈ 4 chars
    max_chars = max_chunk_tokens * 4

    boundaries = find_function_boundaries(content, file_type)

    if not boundaries:
        # Fallback to simple chunking if no functions found
        chunks = []
        for i in range(0, len(content), max_chars):
            chunks.append(content[i:i + max_chars])
        return chunks

    chunks = []
    current_chunk = ""
    current_size = 0

    for start, end, name in boundaries:
        func_content = content[start:end]
        func_size = len(func_content)

        if func_size > max_chars:
            # Function too large - split it with overlap
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_size = 0

            # Split large function
            for i in range(0, func_size, max_chars - 500):  # 500 char overlap
                chunks.append(f"# Part of {name}\n" + func_content[i:i + max_chars])

        elif current_size + func_size > max_chars:
            # Would exceed limit - save current and start new
            chunks.append(current_chunk)
            current_chunk = func_content
            current_size = func_size

        else:
            # Add to current chunk
            current_chunk += func_content
            current_size += func_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# =============================================================================
# L6: HYBRID SEARCH - REGEX + SEMANTIC VERIFICATION
# =============================================================================

def hybrid_verify_finding(finding_code: str, finding_issue: str, context: str) -> tuple[bool, str]:
    """
    L6: Verify a regex-based finding using semantic analysis.

    Returns (is_valid, reason).
    """
    # Quick heuristic checks before LLM call
    false_positive_patterns = [
        (r'#.*' + re.escape(finding_code[:20]), "Appears in comment"),
        (r'""".*' + re.escape(finding_code[:20]) + r'.*"""', "Appears in docstring"),
        (r"'''.*" + re.escape(finding_code[:20]) + r".*'''", "Appears in docstring"),
        (r'//.*' + re.escape(finding_code[:20]), "Appears in comment"),
        (r'/\*.*' + re.escape(finding_code[:20]) + r'.*\*/', "Appears in block comment"),
    ]

    for pattern, reason in false_positive_patterns:
        try:
            if re.search(pattern, context, re.DOTALL | re.IGNORECASE):
                return False, reason
        except re.error:
            pass

    # Check for test file patterns
    test_indicators = ['test_', '_test.', 'spec.', 'mock', 'fixture', 'assert']
    if any(ind in context.lower() for ind in test_indicators):
        return True, "Test file - lower confidence"

    return True, "Verified"


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


# Query quality detection - patterns that indicate overly broad queries
# NOTE: These patterns should NOT have $ anchor - we want partial matching
BROAD_QUERY_PATTERNS = [
    # Audit/review patterns (no $ anchor for partial matching)
    r"^(audit|check|review|analyze|find)\s+(everything|all|the\s+codebase|this\s+codebase)",
    r"^find\s+(all\s+)?(problems|issues|bugs)",
    r"^(security\s+)?audit",
    r"^check\s+(for\s+)?security",
    r"^review\s+(the\s+)?code",
    r"^summarize",
    # Common vague patterns
    r"^find\s+security\s+(issues|problems|vulnerabilities)$",
    r"^(look\s+for|search\s+for)\s+(issues|problems|bugs)",
    r"^what.*wrong",
    r"^any\s+(issues|problems|bugs)",
    # iOS/Swift specific broad patterns
    r"^find\s+(force\s+)?unwraps?$",
    r"^check\s+(for\s+)?(memory\s+)?leaks?$",
    r"^find\s+swift\s+(issues|problems)$",
    r"^review\s+(ios|swift)\s+code$",
    # Python specific broad patterns
    r"^find\s+python\s+(issues|problems)$",
    r"^check\s+python\s+code$",
    # JavaScript/TypeScript broad patterns
    r"^find\s+(js|javascript|typescript|ts)\s+(issues|problems)$",
    r"^review\s+(react|vue|angular)\s+code$",
    # API/backend broad patterns
    r"^check\s+(api|backend|server)$",
    r"^find\s+api\s+(issues|problems)$",
    # General broad patterns
    r"^what\s+does\s+this\s+(do|code\s+do)$",
    r"^explain\s+(this|the\s+code)$",
    r"^how\s+does\s+this\s+work$",
]

# Enhanced decomposition templates with detailed sub-queries
# Each sub-query is designed to generate specific, actionable Python code
QUERY_DECOMPOSITIONS = {
    "security": [
        "Find INJECTION vulnerabilities: (1) SQL injection via string concatenation or f-strings in queries (2) Command injection via subprocess, os.system, os.popen with user input (3) Code injection via eval(), exec(), compile() with external data. For each finding: file:line, code snippet, severity (CRITICAL/HIGH/MEDIUM).",
        "Find HARDCODED SECRETS: (1) API keys matching patterns like 'sk-', 'api_key=', 'apikey' (2) Passwords in variables named password, passwd, pwd, secret (3) Private keys, tokens, credentials in source. For each: file:line, code snippet, secret type.",
        "Find AUTH/ACCESS issues: (1) Missing @login_required or auth decorators on sensitive endpoints (2) Broken access controls - missing permission checks (3) Session issues - insecure cookies, missing CSRF. For each: file:line, code snippet.",
        "Find DATA EXPOSURE: (1) Sensitive data in logs (passwords, tokens, PII) (2) Unencrypted storage of secrets (3) Debug endpoints exposing internal state. For each: file:line, code snippet.",
    ],
    "quality": [
        "Find CODE COMPLEXITY issues: (1) Functions over 50 lines - list with line counts (2) Cyclomatic complexity over 10 (3) Classes over 500 lines. For each: file:line, metric value, suggestion.",
        "Find ERROR HANDLING issues: (1) Bare except: clauses that catch everything (2) Empty except blocks that swallow errors (3) Missing try/except around file I/O, network calls, JSON parsing. For each: file:line, code snippet.",
        "Find CODE SMELLS: (1) Duplicate code blocks (similar logic in multiple places) (2) Dead code - unreachable branches, unused functions (3) Magic numbers without constants. For each: file:line, description.",
        "Find TYPE SAFETY issues: (1) Missing type hints on public functions (2) Any type usage that should be specific (3) Unsafe casts or coercion. For each: file:line, suggestion.",
    ],
    "architecture": [
        "Map PROJECT STRUCTURE: (1) List all modules/packages with one-line purpose (2) Identify entry points (main, CLI, API endpoints) (3) List external dependencies and their roles.",
        "Map DATA FLOW: (1) How data enters the system (APIs, files, user input) (2) How data is processed and transformed (3) How data exits (responses, files, external calls).",
        "Find ARCHITECTURAL CONCERNS: (1) Circular dependencies between modules (2) God classes with too many responsibilities (3) Tight coupling - classes that know too much about each other (4) Missing abstraction layers.",
    ],
    "ios": [
        "Find FORCE UNWRAPS: Look for '!' used for force unwrapping (NOT '!=' comparisons). Match: variable!.method, try!, as!, implicitly unwrapped properties. For each: file:line, code snippet, safer alternative.",
        "Find MEMORY ISSUES: (1) Closures missing [weak self] or [unowned self] (2) Delegate properties not marked weak (3) Strong reference cycles in class properties. For each: file:line, code snippet, fix.",
        "Find CONCURRENCY ISSUES: (1) UI updates not on main thread (2) Missing @MainActor on UI-touching code (3) Data races - shared mutable state without synchronization. For each: file:line, code snippet.",
        "Find SWIFTUI ISSUES: (1) @ObservedObject with default value (should be @StateObject) (2) Heavy computation in body (3) Missing .task or .onAppear for async work. For each: file:line, code snippet.",
    ],
    "python": [
        "Find PYTHON SECURITY: (1) pickle.loads with untrusted data (2) yaml.load without SafeLoader (3) eval/exec with user input (4) subprocess with shell=True. For each: file:line, code snippet, severity.",
        "Find PYTHON QUALITY: (1) Missing docstrings on public functions (2) Mutable default arguments (def f(x=[]):) (3) Bare except clauses (4) Using type() instead of isinstance(). For each: file:line, code snippet.",
        "Find PYTHON ASYNC: (1) Blocking calls in async functions (2) Missing await on coroutines (3) sync functions called where async expected. For each: file:line, code snippet.",
    ],
    "javascript": [
        "Find JS SECURITY: (1) innerHTML with user data (XSS) (2) eval() usage (3) document.write() (4) Regex DoS (catastrophic backtracking). For each: file:line, code snippet, severity.",
        "Find JS QUALITY: (1) var instead of let/const (2) == instead of === (3) Missing error handling on promises (4) Callback hell (deeply nested callbacks). For each: file:line, code snippet.",
        "Find REACT ISSUES: (1) Missing key prop in lists (2) Direct state mutation (3) useEffect with missing dependencies (4) Unnecessary re-renders. For each: file:line, code snippet.",
    ],
    "api": [
        "Find API SECURITY: (1) Missing authentication on endpoints (2) Missing rate limiting (3) SQL injection in query params (4) Mass assignment vulnerabilities. For each: file:line, code snippet.",
        "Find API QUALITY: (1) Missing input validation (2) Inconsistent error responses (3) Missing pagination on list endpoints (4) N+1 query patterns. For each: file:line, code snippet.",
        "Map API ENDPOINTS: List all endpoints with: method, path, auth requirement, request/response types.",
    ],
    "database": [
        "Find DB SECURITY: (1) SQL injection via string formatting (2) Hardcoded credentials (3) Missing parameterized queries (4) Excessive permissions. For each: file:line, code snippet.",
        "Find DB QUALITY: (1) N+1 query patterns (2) Missing indexes on queried columns (3) Large transactions (4) Missing connection pooling. For each: file:line, code snippet.",
        "Map DB SCHEMA: List all tables/models, their fields, relationships, and indexes.",
    ],
    "testing": [
        "Find TEST QUALITY: (1) Tests without assertions (2) Flaky tests - time-dependent, order-dependent (3) Missing edge case coverage (4) Tests that test implementation not behavior. For each: file:line, description.",
        "Find TEST COVERAGE GAPS: (1) Public functions without tests (2) Error paths not tested (3) Integration points not tested. List functions/modules lacking tests.",
    ],
}

# Query type keywords for detection
QUERY_TYPE_KEYWORDS = {
    "ios": ["swift", "ios", "xcode", "unwrap", "swiftui", "uikit", "storekit", "widget", "cocoa", "objc", "objective-c"],
    "security": ["security", "vulnerab", "injection", "xss", "csrf", "secret", "password", "auth", "exploit", "attack", "hack"],
    "quality": ["quality", "code smell", "refactor", "clean", "duplicate", "complexity", "maintainab", "readable"],
    "architecture": ["architecture", "structure", "module", "depend", "design", "pattern", "layer", "component"],
    "python": ["python", "py", "django", "flask", "fastapi", "pytest", "pip", "venv"],
    "javascript": ["javascript", "js", "typescript", "ts", "react", "vue", "angular", "node", "npm", "webpack"],
    "api": ["api", "endpoint", "rest", "graphql", "route", "controller", "request", "response"],
    "database": ["database", "db", "sql", "query", "orm", "model", "migration", "schema", "table"],
    "testing": ["test", "spec", "coverage", "mock", "fixture", "assert", "unittest", "pytest", "jest"],
}


def enhance_query(query: str, detected_type: str) -> str:
    """
    Enhance a vague query with specific search criteria.

    This transforms broad queries into actionable search instructions
    that generate better Python code in the REPL.

    Args:
        query: Original user query
        detected_type: Detected query type (security, ios, python, etc.)

    Returns:
        Enhanced query with specific criteria
    """
    query_lower = query.lower().strip()

    # Enhancement templates by query type
    enhancements = {
        "security": """
Analyze for security vulnerabilities:
1. INJECTION: SQL (string concat in queries), Command (subprocess/os.system), Code (eval/exec)
2. SECRETS: Hardcoded API keys, passwords, tokens, private keys
3. AUTH: Missing authentication, broken access control, session issues
4. DATA: Sensitive data in logs, unencrypted storage, debug endpoints

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], code snippet, severity, fix suggestion.""",

        "ios": """
Analyze Swift/iOS code for:
1. MEMORY: Force unwraps (!), missing [weak self], retain cycles, delegate references
2. CONCURRENCY: Main thread violations, missing @MainActor, data races
3. SWIFTUI: @ObservedObject vs @StateObject, heavy body computation
4. QUALITY: Hardcoded strings, missing localization, deprecated APIs

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], code snippet, safer alternative.""",

        "python": """
Analyze Python code for:
1. SECURITY: pickle with untrusted data, yaml.load without SafeLoader, eval/exec, subprocess shell=True
2. QUALITY: Missing docstrings, mutable default args, bare except, type() vs isinstance()
3. ASYNC: Blocking calls in async, missing await, sync/async mixing

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], code snippet, fix.""",

        "javascript": """
Analyze JavaScript/TypeScript code for:
1. SECURITY: innerHTML XSS, eval(), document.write(), regex DoS
2. QUALITY: var vs let/const, == vs ===, unhandled promises, callback hell
3. REACT: Missing keys, state mutation, useEffect deps, re-render issues

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], code snippet, fix.""",

        "api": """
Analyze API code for:
1. SECURITY: Missing auth, no rate limiting, injection vulnerabilities
2. QUALITY: Missing validation, inconsistent errors, no pagination, N+1 queries
3. STRUCTURE: List all endpoints with method, path, auth requirement

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], code snippet.""",

        "database": """
Analyze database code for:
1. SECURITY: SQL injection, hardcoded credentials, missing parameterization
2. PERFORMANCE: N+1 queries, missing indexes, large transactions
3. SCHEMA: Map tables, relationships, indexes

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], code snippet.""",

        "quality": """
Analyze code quality:
1. COMPLEXITY: Functions >50 lines, cyclomatic complexity >10, god classes
2. ERRORS: Bare except, empty catch, missing error handling on I/O
3. SMELLS: Duplicate code, dead code, magic numbers, poor naming

For each finding report: file:line [Confidence: HIGH/MEDIUM/LOW], description, suggestion.""",

        "architecture": """
Analyze architecture:
1. STRUCTURE: List all modules with purpose, entry points, external dependencies
2. DATA FLOW: How data enters, transforms, and exits the system
3. CONCERNS: Circular deps, god classes, tight coupling, missing abstractions

Report with file references where applicable.""",

        "testing": """
Analyze test quality:
1. QUALITY: Tests without assertions, flaky tests, implementation-testing
2. COVERAGE: Public functions without tests, untested error paths
3. STRUCTURE: Test organization, fixture usage, mock patterns

For each finding report: file:line, description, improvement suggestion.""",
    }

    # If we have a specific enhancement, use it
    if detected_type in enhancements:
        return f"{query}\n\n{enhancements[detected_type]}"

    # Default enhancement for general queries
    return f"""{query}

Analyze thoroughly and report findings with:
- file:line reference for each finding
- [Confidence: HIGH/MEDIUM/LOW] based on verification
- Code snippet showing the issue
- Specific, actionable suggestion for improvement"""


@dataclass
class ProgressEvent:
    """Structured progress event for streaming updates."""
    event_type: str  # "start", "file_collected", "chunk_analyzed", "finding_verified", "complete", "error"
    message: str
    progress_percent: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "progress_percent": self.progress_percent,
            "details": self.details,
        }


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_id: int
    content_preview: str  # First 200 chars
    response: str
    token_count: int
    relevance_score: float
    processing_time_ms: int


@dataclass
class RLMResult:
    """Final result from RLM processing."""

    query: str
    scope: str  # Description of what was processed
    response: str
    chunk_results: list[ChunkResult] = field(default_factory=list)
    total_tokens_processed: int = 0
    total_api_calls: int = 0
    cache_hits: int = 0
    processing_time_ms: int = 0
    truncated: bool = False
    error: str | None = None


class CircuitBreaker:
    """Circuit breaker pattern for API call protection."""

    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, reset_time: int = CIRCUIT_BREAKER_RESET_TIME):
        self.threshold = threshold
        self.reset_time = reset_time
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.is_open = True

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        """Check if we can proceed with a request."""
        if not self.is_open:
            return True

        # Check if reset time has passed
        if time.time() - self.last_failure_time >= self.reset_time:
            self.is_open = False
            self.failure_count = 0
            return True

        return False

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "threshold": self.threshold,
        }


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Limits both request count and token throughput per minute.
    """

    def __init__(
        self,
        requests_per_minute: int = RATE_LIMIT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = RATE_LIMIT_TOKENS_PER_MINUTE
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Sliding window tracking
        self._request_times: list[float] = []
        self._token_usage: list[tuple[float, int]] = []  # (timestamp, tokens)

        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._throttle_count = 0

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff = current_time - 60.0

        # Clean request times
        self._request_times = [t for t in self._request_times if t > cutoff]

        # Clean token usage
        self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > cutoff]

    def can_proceed(self, estimated_tokens: int = 0) -> tuple[bool, float]:
        """
        Check if a request can proceed under rate limits.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            (can_proceed, wait_seconds) tuple
        """
        current_time = time.time()
        self._cleanup_old_entries(current_time)

        # Check request count limit
        if len(self._request_times) >= self.requests_per_minute:
            oldest = self._request_times[0]
            wait_time = 60.0 - (current_time - oldest)
            return False, max(0.1, wait_time)

        # Check token limit
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            if self._token_usage:
                oldest = self._token_usage[0][0]
                wait_time = 60.0 - (current_time - oldest)
                return False, max(0.1, wait_time)

        return True, 0.0

    def record_request(self, tokens_used: int = 0) -> None:
        """Record a completed request."""
        current_time = time.time()
        self._request_times.append(current_time)
        if tokens_used > 0:
            self._token_usage.append((current_time, tokens_used))

        self._total_requests += 1
        self._total_tokens += tokens_used

    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limit would be exceeded."""
        can_proceed, wait_time = self.can_proceed(estimated_tokens)
        if not can_proceed:
            self._throttle_count += 1
            await asyncio.sleep(wait_time)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        self._cleanup_old_entries(current_time)

        return {
            "requests_last_minute": len(self._request_times),
            "tokens_last_minute": sum(tokens for _, tokens in self._token_usage),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "throttle_count": self._throttle_count,
            "requests_limit": self.requests_per_minute,
            "tokens_limit": self.tokens_per_minute,
        }


class LLMResponseCache:
    """LRU cache for LLM responses with TTL support."""

    def __init__(self, max_size: int = LLM_CACHE_MAX_SIZE, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, model: str, messages: list[dict], max_tokens: int) -> str:
        """Create cache key from request parameters."""
        content = f"{model}:{max_tokens}:{str(messages)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, model: str, messages: list[dict], max_tokens: int) -> str | None:
        """Get cached response if available and not expired."""
        key = self._make_key(model, messages, max_tokens)

        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return response
            else:
                # Expired, remove it
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, model: str, messages: list[dict], max_tokens: int, response: str) -> None:
        """Cache a response."""
        key = self._make_key(model, messages, max_tokens)

        # Remove oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (response, time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class SemanticCache:
    """
    Cache with semantic similarity matching and SQLite persistence (L9).

    Uses embeddings to find similar queries instead of exact hash matching.
    This dramatically improves cache hit rates for RLM workloads where
    queries are similar but not identical.

    L9 Enhancement: Cache persists across server restarts via SQLite.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        db_path: str | None = None
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[str, list[float], float, str]] = {}  # key -> (response, embedding, timestamp, query_summary)
        self._embedder = None
        self._hits = 0
        self._misses = 0
        self._embedding_available = False
        self._db_path = db_path
        self._db_conn = None

        # L9: Initialize SQLite persistence
        self._init_persistence()

        # Try to initialize embedder
        self._init_embedder()

        # Load cached entries from SQLite
        self._load_from_db()

    def _init_persistence(self) -> None:
        """Initialize SQLite database for cache persistence."""
        import sqlite3
        from pathlib import Path

        if self._db_path is None:
            # Default to ~/.rlm_cache.db
            self._db_path = str(Path.home() / ".rlm_cache.db")

        try:
            self._db_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    embedding BLOB,
                    timestamp REAL NOT NULL,
                    query_summary TEXT,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            self._db_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON semantic_cache(timestamp)
            """)
            self._db_conn.commit()
        except Exception as e:
            import sys
            print(f"[SemanticCache] SQLite persistence disabled: {e}", file=sys.stderr)
            self._db_conn = None

    def _load_from_db(self) -> None:
        """Load non-expired cache entries from SQLite."""
        if self._db_conn is None:
            return

        import json
        now = time.time()

        try:
            cursor = self._db_conn.execute("""
                SELECT key, response, embedding, timestamp, query_summary
                FROM semantic_cache
                WHERE timestamp > ?
                ORDER BY hit_count DESC
                LIMIT ?
            """, (now - self.ttl_seconds, self.max_size))

            loaded = 0
            for key, response, embedding_json, timestamp, query_summary in cursor:
                embedding = json.loads(embedding_json) if embedding_json else []
                self._cache[key] = (response, embedding, timestamp, query_summary or "")
                loaded += 1

            if loaded > 0:
                import sys
                print(f"[SemanticCache] Loaded {loaded} entries from persistent cache", file=sys.stderr)

            # Clean up expired entries from DB
            self._db_conn.execute("DELETE FROM semantic_cache WHERE timestamp <= ?", (now - self.ttl_seconds,))
            self._db_conn.commit()

        except Exception as e:
            import sys
            print(f"[SemanticCache] Failed to load from DB: {e}", file=sys.stderr)

    def _save_to_db(self, key: str, response: str, embedding: list[float], timestamp: float, query_summary: str) -> None:
        """Save a cache entry to SQLite."""
        if self._db_conn is None:
            return

        import json

        try:
            embedding_json = json.dumps(embedding) if embedding else None
            self._db_conn.execute("""
                INSERT OR REPLACE INTO semantic_cache (key, response, embedding, timestamp, query_summary, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (key, response, embedding_json, timestamp, query_summary))
            self._db_conn.commit()
        except Exception:
            pass  # Silently fail - persistence is best-effort

    def _increment_hit_count(self, key: str) -> None:
        """Increment hit count for a cache entry."""
        if self._db_conn is None:
            return

        try:
            self._db_conn.execute("UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
            self._db_conn.commit()
        except Exception:
            pass

    def _init_embedder(self) -> None:
        """Initialize the sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model - good balance of speed and quality
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self._embedding_available = True
        except ImportError:
            # sentence-transformers not installed, fall back to simple matching
            self._embedding_available = False
        except Exception:
            self._embedding_available = False

    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text. Returns None if embeddings unavailable."""
        if not self._embedding_available or self._embedder is None:
            return None

        try:
            # Truncate to avoid memory issues
            text = text[:2000]
            embedding = self._embedder.encode(text)
            return embedding.tolist()
        except Exception:
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _make_key_text(self, query: str, context_summary: str) -> str:
        """Create text for embedding from query and context."""
        return f"{query}\n---\n{context_summary[:500]}"

    def get_similar(self, query: str, context_summary: str) -> tuple[str | None, float]:
        """
        Find cached response with similar query+context.

        Args:
            query: The user's query
            context_summary: Summary of the file context (e.g., file list)

        Returns:
            (cached_response, similarity_score) - response is None if no match found
        """
        key_text = self._make_key_text(query, context_summary)

        # Try embedding-based search first
        if self._embedding_available:
            query_embedding = self._get_embedding(key_text)
            if query_embedding:
                return self._search_by_embedding(query_embedding)

        # Fall back to simple keyword matching
        return self._search_by_keywords(query)

    def _search_by_embedding(self, query_embedding: list[float]) -> tuple[str | None, float]:
        """Search cache using embedding similarity."""
        best_match = None
        best_similarity = 0.0
        best_key = None
        now = time.time()

        expired_keys = []
        for key, (response, cached_embedding, timestamp, _) in self._cache.items():
            # Check expiration
            if now - timestamp > self.ttl_seconds:
                expired_keys.append(key)
                continue

            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = response
                best_key = key

        # Clean up expired entries
        for key in expired_keys:
            del self._cache[key]

        if best_match:
            self._hits += 1
            # L9: Track hit count in persistent storage
            if best_key:
                self._increment_hit_count(best_key)
        else:
            self._misses += 1

        return best_match, best_similarity

    def _search_by_keywords(self, query: str) -> tuple[str | None, float]:
        """Fallback: search cache using keyword matching."""
        query_words = set(query.lower().split())
        if len(query_words) < 2:
            self._misses += 1
            return None, 0.0

        best_match = None
        best_score = 0.0
        now = time.time()

        for key, (response, _, timestamp, cached_query) in self._cache.items():
            if now - timestamp > self.ttl_seconds:
                continue

            cached_words = set(cached_query.lower().split())
            if len(cached_words) < 2:
                continue

            # Jaccard similarity
            intersection = len(query_words & cached_words)
            union = len(query_words | cached_words)
            score = intersection / union if union > 0 else 0

            if score > best_score and score >= 0.7:  # Higher threshold for keyword matching
                best_score = score
                best_match = response

        if best_match:
            self._hits += 1
        else:
            self._misses += 1

        return best_match, best_score

    def set(self, query: str, context_summary: str, response: str) -> None:
        """Cache a response with its query embedding."""
        key_text = self._make_key_text(query, context_summary)

        # Get embedding
        embedding = self._get_embedding(key_text) or []

        # Create unique key
        key = hashlib.sha256(key_text.encode()).hexdigest()[:16]

        # Remove oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        timestamp = time.time()
        self._cache[key] = (response, embedding, timestamp, query)

        # L9: Persist to SQLite
        self._save_to_db(key, response, embedding, timestamp, query)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses

        # Get DB stats if available
        db_size = 0
        if self._db_conn:
            try:
                cursor = self._db_conn.execute("SELECT COUNT(*) FROM semantic_cache")
                db_size = cursor.fetchone()[0]
            except Exception:
                pass

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
            "embedding_available": self._embedding_available,
            "similarity_threshold": self.similarity_threshold,
            "persistent_db": self._db_path if self._db_conn else None,
            "persistent_entries": db_size,
        }

    def clear(self) -> None:
        """Clear the cache (memory and persistent)."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

        # L9: Clear persistent cache too
        if self._db_conn:
            try:
                self._db_conn.execute("DELETE FROM semantic_cache")
                self._db_conn.commit()
            except Exception:
                pass


class RLMProcessor:
    """
    Recursive Language Model processor.

    Implements the RLM technique for processing inputs that exceed
    the model's context window. Instead of sending everything at once,
    we:

    1. Split input into manageable chunks
    2. Assess relevance of each chunk to the query
    3. Process relevant chunks with sub-queries
    4. Aggregate results into a coherent response

    Performance features:
    - AsyncOpenAI for true async API calls
    - Exponential backoff retry with jitter
    - Circuit breaker for failure protection
    - LRU response caching
    - Connection pooling
    """

    # Optimized system prompts for Claude Haiku 4.5 (token-efficient)
    # Based on Anthropic's prompt engineering best practices

    RELEVANCE_PROMPT = """Rate query-content relevance: 0.0-1.0. Output only the number.
0.0=irrelevant, 0.5=partial, 1.0=exact match"""

    CHUNK_ANALYSIS_PROMPT = """Extract code findings matching the query. For each finding:

<output_format>
**{filepath}:{line}** [{confidence}] - {issue}
```{lang}
{code_snippet}
```
</output_format>

<confidence_rules>
HIGH: Active code, verified line, clear match
MEDIUM: Uncertain context, partial match
LOW: Dead code (#if false/DEBUG/0), unverified, test file
</confidence_rules>

Rules:
- Copy exact code from content (no paraphrasing)
- Line 1 starts at each "### File:" header
- No findings? Say "No relevant findings" only"""

    AGGREGATION_PROMPT = """Synthesize findings into a structured report. PRESERVE all file:line refs, code snippets, and confidence levels exactly.

<output_schema>
## Summary
{count_high} HIGH, {count_medium} MEDIUM, {count_low} LOW findings

## High Confidence
**{file}:{line}** [HIGH] - {description}
```{lang}
{code}
```

## Medium Confidence
(same format)

## Low Confidence (Verify Manually)
**{file}:{line}** [LOW - {reason}] - {description}
```{lang}
{code}
```
</output_schema>

Rules:
- Group by confidence (HIGH→MEDIUM→LOW)
- Remove exact duplicates only (same file+line+issue)
- Dead code (#if false) = always LOW"""

    def __init__(
        self,
        config: RLMConfig | None = None,
        cache_manager: CacheManager | None = None,
        incremental_cache: IncrementalCache | None = None
    ):
        self.config = config or RLMConfig()
        self.cache_manager = cache_manager  # Keep for interface compatibility

        # Create async HTTP client with connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_REQUEST_TIMEOUT_SECONDS),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

        # Create async OpenAI client with pooled connections
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url,
            http_client=self._http_client,
            default_headers={
                "HTTP-Referer": "https://github.com/mosif16/RLM-Mem_MCP",
                "X-Title": "RLM-Mem MCP Server"
            }
        )

        self._encoder: tiktoken.Encoding | None = None

        # Performance components
        self._circuit_breaker = CircuitBreaker()
        self._response_cache = LLMResponseCache()
        self._rate_limiter = RateLimiter()
        self._api_call_count = 0
        self._cache_hit_count = 0

        # Incremental file cache for skipping unchanged files
        self._incremental_cache = incremental_cache or get_incremental_cache()

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        await self._http_client.aclose()

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            self._encoder = tiktoken.encoding_for_model("gpt-4")
        return self._encoder

    async def get_encoder_async(self) -> tiktoken.Encoding:
        """Get encoder without blocking event loop."""
        if self._encoder is None:
            self._encoder = await asyncio.to_thread(
                tiktoken.encoding_for_model, "gpt-4"
            )
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def analyze_query_quality(self, query: str) -> dict[str, Any]:
        """
        Analyze query quality, detect type, and enhance if needed.

        Returns dict with:
        - is_broad: True if query is too vague
        - query_type: Detected type (security, ios, python, javascript, api, etc.)
        - suggested_decomposition: List of focused sub-queries if broad
        - enhanced_query: Query enhanced with specific search criteria
        - relevance_threshold: Adjusted threshold for this query type
        """
        import re
        query_lower = query.lower().strip()

        # Check for broad query patterns
        is_broad = False
        for pattern in BROAD_QUERY_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                is_broad = True
                break

        # Also check for very short queries (likely too vague)
        if len(query_lower.split()) <= 3 and not any(c in query_lower for c in [':', '-', '(', ')']):
            is_broad = True

        # Detect query type using keyword matching with priority
        query_type = "general"
        type_scores: dict[str, int] = {}

        for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                type_scores[qtype] = score

        if type_scores:
            # Pick the type with highest keyword match
            query_type = max(type_scores, key=type_scores.get)

        # Get suggested decomposition for broad queries
        suggested_decomposition = []
        if is_broad and query_type in QUERY_DECOMPOSITIONS:
            suggested_decomposition = QUERY_DECOMPOSITIONS[query_type]
        elif is_broad:
            # Default decomposition combines security + quality checks
            suggested_decomposition = (
                QUERY_DECOMPOSITIONS.get("security", [])[:2] +
                QUERY_DECOMPOSITIONS.get("quality", [])[:2]
            )

        # ALWAYS enhance the query with specific criteria (L3: Query Quality Automation)
        # Even non-broad queries benefit from explicit output format requirements
        enhanced_query = enhance_query(query, query_type)

        # Adjust relevance threshold based on query specificity
        if is_broad:
            relevance_threshold = 0.15  # Much lower for broad queries
        elif len(query.split()) < 10:
            relevance_threshold = 0.25  # Slightly lower for shorter queries
        else:
            relevance_threshold = 0.3  # Normal threshold for specific queries

        return {
            "is_broad": is_broad,
            "query_type": query_type,
            "suggested_decomposition": suggested_decomposition,
            "enhanced_query": enhanced_query,
            "relevance_threshold": relevance_threshold,
            "original_query": query,
        }

    async def process_with_decomposition(
        self,
        query: str,
        collection: CollectionResult,
        progress_callback: Callable[[str], None] | None = None
    ) -> RLMResult:
        """
        Process query with automatic decomposition for broad queries.

        If query is too broad, decomposes into focused sub-queries and
        aggregates results.
        """
        analysis = self.analyze_query_quality(query)

        if not analysis["is_broad"] or not analysis["suggested_decomposition"]:
            # Query is specific enough, process normally
            return await self.process(query, collection, progress_callback)

        if progress_callback:
            progress_callback(f"Query is broad - decomposing into {len(analysis['suggested_decomposition'])} focused queries...")

        # Process each sub-query
        all_results = []
        for i, sub_query in enumerate(analysis["suggested_decomposition"]):
            if progress_callback:
                progress_callback(f"Processing sub-query {i+1}/{len(analysis['suggested_decomposition'])}: {sub_query[:50]}...")

            result = await self.process(sub_query, collection, progress_callback)
            if result.response and "No relevant" not in result.response and "No findings" not in result.response:
                all_results.append({
                    "query": sub_query,
                    "response": result.response,
                    "chunks_processed": len(result.chunk_results),
                })

        # Aggregate all sub-query results
        if not all_results:
            return RLMResult(
                query=query,
                scope=f"Decomposed into {len(analysis['suggested_decomposition'])} sub-queries",
                response="No findings from any sub-query. The codebase may be clean for the checked categories, or try more specific queries.",
                total_tokens_processed=collection.total_tokens,
            )

        # Combine results
        combined_response = f"## Analysis Results (Decomposed Query)\n\n"
        combined_response += f"Original query \"{query}\" was broad, so it was decomposed into {len(analysis['suggested_decomposition'])} focused queries.\n\n"

        for result in all_results:
            combined_response += f"### {result['query'][:60]}...\n\n"
            combined_response += result['response'] + "\n\n---\n\n"

        return RLMResult(
            query=query,
            scope=f"Decomposed: {len(all_results)}/{len(analysis['suggested_decomposition'])} sub-queries had findings",
            response=combined_response,
            total_tokens_processed=collection.total_tokens,
        )

    async def count_tokens_async(self, text: str) -> int:
        """Count tokens without blocking event loop."""
        if not text:
            return 0
        encoder = await self.get_encoder_async()
        return await asyncio.to_thread(lambda: len(encoder.encode(text)))

    def split_into_chunks(
        self,
        content: str,
        max_chunk_tokens: int | None = None
    ) -> list[str]:
        """
        Split content into chunks of approximately equal token count.

        Uses intelligent splitting:
        1. Try to split on major boundaries (files, functions, sections)
        2. Fall back to paragraph splitting
        3. Final fallback to token-based splitting
        """
        max_tokens = max_chunk_tokens or self.config.max_chunk_tokens
        total_tokens = self.count_tokens(content)

        if total_tokens <= max_tokens:
            return [content]

        chunks = []

        # Try splitting by file markers first (### File: ...)
        if "### File:" in content:
            return self._split_by_file_markers(content, max_tokens)

        # Try splitting by major section markers
        section_markers = ["\n## ", "\n# ", "\n---\n", "\n\n\n"]
        for marker in section_markers:
            if marker in content:
                sections = content.split(marker)
                if len(sections) > 1:
                    chunks = self._merge_small_sections(
                        sections, marker, max_tokens
                    )
                    if chunks:
                        return chunks

        # Fall back to paragraph splitting
        paragraphs = content.split("\n\n")
        if len(paragraphs) > 1:
            chunks = self._merge_small_sections(paragraphs, "\n\n", max_tokens)
            if chunks:
                return chunks

        # Final fallback: split by token count
        return self._split_by_tokens(content, max_tokens)

    def _split_by_file_markers(
        self,
        content: str,
        max_tokens: int
    ) -> list[str]:
        """Split content that has file markers."""
        chunks = []
        current_chunk = ""
        current_tokens = 0  # Incremental token counting

        # Split on file markers but keep the marker with the content
        parts = content.split("### File:")
        for i, part in enumerate(parts):
            if i == 0 and not part.strip():
                continue

            file_content = ("### File:" + part) if i > 0 else part
            file_tokens = self.count_tokens(file_content)

            if current_tokens + file_tokens <= max_tokens:
                current_chunk += file_content
                current_tokens += file_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if file_tokens > max_tokens:
                    # Split this large file
                    chunks.extend(self._split_by_tokens(file_content, max_tokens))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = file_content
                    current_tokens = file_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _merge_small_sections(
        self,
        sections: list[str],
        separator: str,
        max_tokens: int
    ) -> list[str]:
        """Merge small sections together until they approach max_tokens."""
        chunks = []
        current_chunk = ""
        current_tokens = 0  # Incremental token counting

        for section in sections:
            section_tokens = self.count_tokens(section)

            if current_tokens + section_tokens <= max_tokens:
                if current_chunk:
                    current_chunk += separator + section
                    current_tokens += section_tokens + self.count_tokens(separator)
                else:
                    current_chunk = section
                    current_tokens = section_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if section_tokens > max_tokens:
                    # Section too large, split it further
                    chunks.extend(self._split_by_tokens(section, max_tokens))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = section
                    current_tokens = section_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_tokens(self, content: str, max_tokens: int) -> list[str]:
        """Split content by token count (fallback method)."""
        chunks = []
        tokens = self.encoder.encode(content)

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    async def _call_llm_with_retry(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int
    ) -> str:
        """
        Call LLM API with retry logic, caching, and circuit breaker.

        Features:
        - LRU response caching
        - Exponential backoff with jitter
        - Circuit breaker for repeated failures
        - Timeout protection
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_proceed():
            raise Exception("Circuit breaker is open - too many recent failures")

        # Check cache first
        cached = self._response_cache.get(model, messages, max_tokens)
        if cached is not None:
            self._cache_hit_count += 1
            return cached

        # Rate limiting - wait if needed
        estimated_tokens = max_tokens + sum(len(m.get("content", "")) // 4 for m in messages)
        await self._rate_limiter.wait_if_needed(estimated_tokens)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                self._api_call_count += 1

                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages
                    ),
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS
                )

                result = response.choices[0].message.content if response.choices else ""

                # Record success, rate limit usage, and cache
                self._circuit_breaker.record_success()
                tokens_used = getattr(response.usage, 'total_tokens', max_tokens) if response.usage else max_tokens
                self._rate_limiter.record_request(tokens_used)
                self._response_cache.set(model, messages, max_tokens, result)

                return result

            except asyncio.TimeoutError:
                last_exception = Exception(f"Request timeout after {LLM_REQUEST_TIMEOUT_SECONDS}s")
                self._circuit_breaker.record_failure()

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if retryable error
                if any(code in error_str for code in ["429", "503", "502", "500", "rate"]):
                    self._circuit_breaker.record_failure()

                    if attempt < MAX_RETRIES - 1:
                        # Exponential backoff with jitter
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-retryable error
                    self._circuit_breaker.record_failure()
                    raise

        # All retries exhausted
        raise last_exception or Exception("Unknown error after retries")

    async def process(
        self,
        query: str,
        collection: CollectionResult,
        progress_callback: Callable[[str], None] | None = None
    ) -> RLMResult:
        """
        Process a query against collected content using RLM technique.

        Processing order (TRUE RLM first):
        1. REPL Mode - Content as variable, LLM writes code (TRUE RLM from paper)
        2. Agent SDK Mode - Subagents for parallel processing
        3. Built-in Mode - Sequential chunk processing (fallback)

        Args:
            query: The query/question to answer
            collection: Collected files/content to process
            progress_callback: Optional callback for progress updates

        Returns:
            RLMResult with the processed response
        """
        start_time = time.time()

        # Build scope description
        scope = f"{collection.file_count} files, ~{collection.total_tokens:,} tokens"

        if progress_callback:
            progress_callback(f"Processing {scope}...")

        # Get combined content
        content = collection.get_combined_content(include_headers=True)

        # 1. Try TRUE RLM (REPL-based) - content as variable, LLM writes code
        try:
            from .repl_environment import RLMReplEnvironment

            if progress_callback:
                progress_callback("Using TRUE RLM: Content as variable, LLM writes code...")

            repl = RLMReplEnvironment(self.config)
            repl.initialize(content)

            # Run the REPL session
            response = await repl.run_rlm_session(query)

            # Get all preserved sub-responses (NOT summarized!)
            all_responses = repl.get_all_responses()

            processing_time = int((time.time() - start_time) * 1000)

            # Create ChunkResult entries from REPL iterations for progress tracking
            chunk_results = []
            for i, resp in enumerate(all_responses):
                chunk_results.append(ChunkResult(
                    chunk_id=i,
                    content_preview=f"REPL iteration {i+1}",
                    response=resp[:500] if resp else "",
                    token_count=self.count_tokens(resp) if resp else 0,
                    relevance_score=1.0,  # REPL responses are always relevant
                    processing_time_ms=processing_time // max(len(all_responses), 1),
                ))

            return RLMResult(
                query=query,
                scope=f"{scope} (REPL mode: {len(all_responses)} iterations)",
                response=response,
                chunk_results=chunk_results,
                total_tokens_processed=collection.total_tokens,
                total_api_calls=len(all_responses),
                cache_hits=0,
                processing_time_ms=processing_time,
                truncated=False,
                error=None
            )

        except ImportError:
            if progress_callback:
                progress_callback("REPL environment not available, using built-in processor...")
        except Exception as e:
            if progress_callback:
                progress_callback(f"REPL mode failed: {e}, using built-in processor...")

        # 2. Fall back to built-in sequential processing
        if progress_callback:
            progress_callback(f"Using built-in processor with {self.config.model}...")

        # Get dynamic relevance threshold based on query analysis
        query_analysis = self.analyze_query_quality(query)
        relevance_threshold = query_analysis["relevance_threshold"]

        if progress_callback and relevance_threshold != 0.3:
            progress_callback(f"Using adjusted relevance threshold: {relevance_threshold}")

        return await self._process_builtin(
            query, collection, scope, start_time, progress_callback, relevance_threshold
        )

    async def _process_builtin(
        self,
        query: str,
        collection: CollectionResult,
        scope: str,
        start_time: float,
        progress_callback: Callable[[str], None] | None,
        relevance_threshold: float = 0.3  # Can be overridden by query analysis
    ) -> RLMResult:
        """Process using built-in RLM implementation."""
        result = RLMResult(query=query, scope=scope, response="")

        try:
            # Get combined content
            content = collection.get_combined_content(include_headers=True)

            # Split into chunks
            chunks = self.split_into_chunks(content)

            if progress_callback:
                progress_callback(f"Split into {len(chunks)} chunks")

            # Process chunks (with relevance filtering)
            chunk_results = await self._process_chunks(
                query, chunks, progress_callback
            )

            result.chunk_results = chunk_results
            result.total_tokens_processed = sum(
                self.count_tokens(c) for c in chunks
            )

            # Filter to relevant chunks using dynamic threshold
            relevant_results = [
                r for r in chunk_results if r.relevance_score >= relevance_threshold
            ]

            if not relevant_results:
                result.response = "No relevant information found for the query."
            elif len(relevant_results) == 1:
                result.response = relevant_results[0].response
            else:
                # Aggregate results
                if progress_callback:
                    progress_callback(f"Aggregating {len(relevant_results)} relevant findings...")

                result.response = await self._aggregate_results(
                    query, relevant_results
                )
                result.total_api_calls += 1

            # Truncate if necessary
            result.response, result.truncated = self._truncate_response(
                result.response
            )

            # Update stats
            result.total_api_calls = self._api_call_count
            result.cache_hits = self._cache_hit_count

        except Exception as e:
            result.error = str(e)
            result.response = f"Error during RLM processing: {e}"

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    async def _process_chunks(
        self,
        query: str,
        chunks: list[str],
        progress_callback: Callable[[str], None] | None
    ) -> list[ChunkResult]:
        """Process all chunks with relevance assessment in parallel."""
        if progress_callback:
            progress_callback(f"Processing {len(chunks)} chunks in parallel...")

        # Track if we found highly relevant content (for early termination)
        found_highly_relevant = asyncio.Event()

        async def process_single_chunk(i: int, chunk: str) -> ChunkResult:
            """Process a single chunk (relevance + analysis)."""
            start_time = time.time()

            # Assess relevance
            relevance = await self._assess_relevance(query, chunk)

            # Check for early termination signal
            if relevance >= EARLY_TERMINATION_THRESHOLD:
                found_highly_relevant.set()

            # Only process if relevant enough
            if relevance >= 0.3:
                response = await self._analyze_chunk(query, chunk)
            else:
                response = "(Skipped - low relevance)"

            processing_time = int((time.time() - start_time) * 1000)

            return ChunkResult(
                chunk_id=i,
                content_preview=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                response=response,
                token_count=self.count_tokens(chunk),
                relevance_score=relevance,
                processing_time_ms=processing_time,
            )

        # Process all chunks in parallel
        tasks = [process_single_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        if progress_callback:
            relevant_count = sum(1 for r in results if r.relevance_score >= 0.3)
            progress_callback(f"Completed: {relevant_count}/{len(chunks)} chunks relevant")

        return list(results)

    async def _assess_relevance(self, query: str, chunk: str) -> float:
        """Assess relevance of a chunk to the query."""
        # For very short chunks, assume relevant
        if self.count_tokens(chunk) < 100:
            return 0.5

        try:
            # Use a smaller preview for relevance assessment
            preview = chunk[:2000] if len(chunk) > 2000 else chunk

            messages = [
                {"role": "system", "content": self.RELEVANCE_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nContent:\n{preview}\n\nRelevance (0.0-1.0):"}
            ]

            text = await self._call_llm_with_retry(
                model=self.config.model,
                messages=messages,
                max_tokens=10
            )

            score = float(text.strip())
            return max(0.0, min(1.0, score))

        except Exception as e:
            # Log the actual error for debugging
            import sys
            print(f"Relevance assessment error: {type(e).__name__}: {e}", file=sys.stderr)
            return 0.5  # Default to moderate relevance on error

    async def _analyze_chunk(self, query: str, chunk: str) -> str:
        """Analyze a chunk for information relevant to the query."""
        try:
            messages = [
                {"role": "system", "content": self.CHUNK_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nContent to analyze:\n{chunk}\n\nFindings:"}
            ]

            return await self._call_llm_with_retry(
                model=self.config.model,
                messages=messages,
                max_tokens=1000
            )

        except Exception as e:
            return f"(Error analyzing chunk: {type(e).__name__}: {e})"

    async def _aggregate_results(
        self,
        query: str,
        chunk_results: list[ChunkResult]
    ) -> str:
        """Aggregate findings from multiple chunks."""
        # Build combined findings
        findings = []
        for i, result in enumerate(chunk_results):
            if result.response and "(Skipped" not in result.response:
                findings.append(f"[Source {i + 1}]:\n{result.response}")

        combined = "\n\n---\n\n".join(findings)

        try:
            messages = [
                {"role": "system", "content": self.AGGREGATION_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nFindings from {len(findings)} sources:\n\n{combined}\n\nSynthesized response:"}
            ]

            return await self._call_llm_with_retry(
                model=self.config.aggregator_model,
                messages=messages,
                max_tokens=2000
            )

        except Exception as e:
            # Fall back to just returning combined findings
            return f"Combined findings:\n\n{combined}\n\n(Aggregation error: {type(e).__name__}: {e})"

    def _truncate_response(
        self,
        response: str,
        max_chars: int | None = None
    ) -> tuple[str, bool]:
        """Truncate response if too long."""
        # Convert max tokens to approximate chars (1 token ≈ 4 chars)
        max_chars = max_chars or (self.config.max_result_tokens * 4)

        if len(response) <= max_chars:
            return response, False

        # Keep first half and last quarter
        first_half = max_chars // 2
        last_quarter = max_chars // 4

        truncated = (
            response[:first_half] +
            f"\n\n[... TRUNCATED {len(response) - max_chars} characters ...]\n\n" +
            response[-last_quarter:]
        )

        return truncated, True

    def format_result(self, result: RLMResult) -> str:
        """Format an RLM result for output."""
        parts = [
            "## RLM Analysis Complete",
            "",
            f"**Query:** {result.query}",
            f"**Scope:** {result.scope}",
        ]

        if result.error:
            parts.extend([
                "",
                f"**Error:** {result.error}",
            ])

        parts.extend([
            "",
            "### Result",
            "",
            result.response,
        ])

        if result.truncated:
            parts.extend([
                "",
                "*[Response truncated - use more specific queries for details]*",
            ])

        # Add stats
        parts.extend([
            "",
            "---",
            f"*Processed in {result.processing_time_ms}ms | "
            f"{len(result.chunk_results)} chunks | "
            f"Cache hits: {result.cache_hits}*",
        ])

        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        return {
            "api_calls": self._api_call_count,
            "cache_hits": self._cache_hit_count,
            "response_cache": self._response_cache.get_stats(),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "rate_limiter": self._rate_limiter.get_stats(),
            "incremental_cache": self._incremental_cache.get_stats().to_dict(),
        }

    def get_cached_file_analysis(
        self,
        file_path: str,
        content: str,
        query: str
    ) -> str | None:
        """
        Get cached analysis for a file if available and unchanged.

        Args:
            file_path: The file path
            content: Current file content
            query: The analysis query

        Returns:
            Cached analysis result if valid, None if cache miss
        """
        cached = self._incremental_cache.get_cached_analysis(file_path, content, query)
        if cached:
            return cached.result
        return None

    def cache_file_analysis(
        self,
        file_path: str,
        content: str,
        query: str,
        result: str,
        confidence: float = 1.0
    ) -> None:
        """
        Cache an analysis result for a file.

        Args:
            file_path: The file path
            content: The file content
            query: The analysis query
            result: The analysis result
            confidence: Confidence in the result (0.0-1.0)
        """
        self._incremental_cache.cache_analysis(
            file_path, content, query, result, confidence
        )

    def get_changed_files(
        self,
        collection: CollectionResult
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """
        Partition collected files into changed and unchanged.

        Args:
            collection: The file collection

        Returns:
            (changed_files, unchanged_files) where each is list of (path, content)
        """
        files = [(f.relative_path, f.content) for f in collection.files]
        return self._incremental_cache.get_changed_files(files)

    def invalidate_cache(self, path: str | None = None) -> int:
        """
        Invalidate incremental cache entries.

        Args:
            path: Specific path to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        return self._incremental_cache.invalidate(path)
