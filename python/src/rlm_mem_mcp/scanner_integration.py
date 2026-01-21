"""
Scanner Integration for RLM Processing.

Provides query analysis and enhancement for scanner integration:
- Broad query detection patterns
- Query decomposition templates for different analysis types
- Query type detection keywords
- Query enhancement for specific scanner types
"""

import re


# =============================================================================
# BROAD QUERY DETECTION PATTERNS
# =============================================================================

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


# =============================================================================
# QUERY DECOMPOSITION TEMPLATES
# =============================================================================

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


# =============================================================================
# QUERY TYPE KEYWORDS
# =============================================================================

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


# =============================================================================
# QUERY ENHANCEMENT
# =============================================================================

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


def is_broad_query(query: str) -> bool:
    """
    Check if a query is too broad/vague.

    Args:
        query: The user's query

    Returns:
        True if query matches broad patterns
    """
    query_lower = query.lower().strip()

    for pattern in BROAD_QUERY_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return True

    # Also check for very short queries (likely too vague)
    if len(query_lower.split()) <= 3 and not any(c in query_lower for c in [':', '-', '(', ')']):
        return True

    return False


def detect_query_type_from_keywords(query: str) -> str:
    """
    Detect query type using keyword matching.

    Args:
        query: The user's query

    Returns:
        Detected type (security, ios, python, etc.) or "general"
    """
    query_lower = query.lower()
    type_scores: dict[str, int] = {}

    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            type_scores[qtype] = score

    if type_scores:
        return max(type_scores, key=lambda k: type_scores[k])

    return "general"


def get_decomposition_queries(query_type: str) -> list[str]:
    """
    Get decomposition queries for a given query type.

    Args:
        query_type: The detected query type

    Returns:
        List of focused sub-queries for decomposition
    """
    if query_type in QUERY_DECOMPOSITIONS:
        return QUERY_DECOMPOSITIONS[query_type]

    # Default: combine security + quality
    return (
        QUERY_DECOMPOSITIONS.get("security", [])[:2] +
        QUERY_DECOMPOSITIONS.get("quality", [])[:2]
    )
