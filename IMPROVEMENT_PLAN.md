# RLM Tool Improvement Plan

Based on user feedback from iOS codebase analysis.

## Current State

**Strengths:**
- Fast scanning (1.7s for 189 files)
- Pattern matching works well
- Structured output is parseable

**Weaknesses to Address:**
1. No semantic understanding (can't tell if force unwrap is guarded)
2. No data flow analysis (can't trace nil-checks)
3. Noisy code quality metrics (72 "long function" findings)
4. Missing deeper iOS-specific checks

---

## Phase 1: Reduce Noise (Quick Wins)

### 1.1 Smart Filtering for Code Quality
**Problem:** 72 "long function" findings buries real issues
**Solution:**
- Default code quality scans to OFF unless explicitly requested
- Add `include_quality: bool` parameter to scans
- Prioritize security/crash issues over style

```python
# Change default behavior
def run_ios_scan(self, min_confidence="LOW", include_quality=False):
    results = [security_scans...]
    if include_quality:
        results.extend([find_long_functions(), find_todos()])
```

**Effort:** 1 hour

### 1.2 Confidence-Based Default Filtering
**Problem:** Too many LOW confidence findings
**Solution:**
- Default `min_confidence` to "MEDIUM" instead of "LOW"
- Add prominent summary showing filtered count

```
## Summary
Found 15 issues (8 HIGH, 5 MEDIUM, 2 LOW)
Showing: 13 issues (filtered 2 LOW confidence)
```

**Effort:** 30 minutes

---

## Phase 2: Context-Aware Analysis (Medium Effort)

### 2.1 Guard Detection for Force Unwraps
**Problem:** Can't tell if `value!` is guarded by `if let` or `guard let`
**Solution:** Look back N lines for guard patterns before flagging

```python
def find_force_unwraps(self) -> ToolResult:
    # For each match, check preceding lines for guards
    guard_patterns = [
        r'if\s+let\s+{var_name}\s*=',      # if let x = ...
        r'guard\s+let\s+{var_name}\s*=',   # guard let x = ...
        r'if\s+{var_name}\s*!=\s*nil',     # if x != nil
        r'guard\s+{var_name}\s*!=\s*nil',  # guard x != nil
    ]

    # Look back 10 lines for matching guard
    for i in range(max(0, line_num - 10), line_num):
        if any(re.search(p.format(var_name=var), lines[i]) for p in guard_patterns):
            confidence = Confidence.LOW  # Guarded, likely safe
            break
```

**Effort:** 4 hours

### 2.2 Scope-Aware Nil Checking
**Problem:** Can't trace if value was nil-checked earlier in scope
**Solution:** Track variable state within function scope

```python
class ScopeAnalyzer:
    def __init__(self, function_lines: list[str]):
        self.nil_checked_vars: set[str] = set()
        self._analyze_scope(function_lines)

    def _analyze_scope(self, lines):
        for line in lines:
            # Track nil checks
            if match := re.search(r'guard\s+let\s+(\w+)', line):
                self.nil_checked_vars.add(match.group(1))
            if match := re.search(r'if\s+let\s+(\w+)', line):
                self.nil_checked_vars.add(match.group(1))

    def is_safe_unwrap(self, var_name: str) -> bool:
        return var_name in self.nil_checked_vars
```

**Effort:** 8 hours

### 2.3 Function Boundary Detection
**Problem:** Analysis doesn't understand function boundaries
**Solution:** Parse function declarations and track scope

```python
def _find_function_boundaries(self, lines: list[str]) -> list[tuple[int, int, str]]:
    """Return list of (start_line, end_line, func_name) tuples."""
    functions = []
    brace_depth = 0
    func_start = None
    func_name = None

    for i, line in enumerate(lines):
        if match := re.search(r'func\s+(\w+)', line):
            func_start = i
            func_name = match.group(1)

        brace_depth += line.count('{') - line.count('}')

        if func_start and brace_depth == 0:
            functions.append((func_start, i, func_name))
            func_start = None

    return functions
```

**Effort:** 6 hours

---

## Phase 3: Deep iOS Analysis (Significant Effort)

### 3.1 Keychain Security Scanner
**Problem:** Missing Keychain usage checks
**Solution:** New scanner for secure storage patterns

```python
def find_keychain_issues(self) -> ToolResult:
    """
    Check for:
    1. Sensitive data not using Keychain
    2. Keychain access without proper error handling
    3. Missing Keychain accessibility settings
    4. Hardcoded Keychain keys
    """
    patterns = [
        # Missing accessibility
        (r'SecItemAdd\([^)]*(?!kSecAttrAccessible)',
         "SecItemAdd without accessibility setting", Severity.HIGH),

        # No error handling
        (r'SecItemCopyMatching\([^)]*\)\s*(?!\s*==|\s*!=)',
         "SecItemCopyMatching result not checked", Severity.HIGH),

        # Biometric without fallback
        (r'LAContext\(\)\.canEvaluatePolicy[^}]*(?!fallback|password)',
         "Biometric auth without fallback", Severity.MEDIUM),
    ]
```

**Effort:** 8 hours

### 3.2 CloudKit Sync Pattern Scanner
**Problem:** Missing proper CloudKit sync pattern checks
**Solution:** Detect common CloudKit anti-patterns

```python
def find_cloudkit_sync_issues(self) -> ToolResult:
    """
    Check for:
    1. CKModifyRecordsOperation without conflict handling
    2. Missing CKServerChangeToken persistence
    3. Subscription setup without error recovery
    4. Batch operations exceeding limits
    """
    patterns = [
        # No conflict resolution
        (r'CKModifyRecordsOperation\([^}]*(?!perRecordSaveBlock|modifyRecordsResultBlock)',
         "CKModifyRecordsOperation missing result handlers", Severity.HIGH),

        # Change token not persisted
        (r'fetchDatabaseChanges[^}]*serverChangeToken[^}]*(?!UserDefaults|save|persist)',
         "Server change token may not be persisted", Severity.MEDIUM),

        # No retry logic
        (r'CKOperation[^}]*(?!qualityOfService|retry)',
         "CKOperation without QoS or retry configuration", Severity.LOW),
    ]
```

**Effort:** 12 hours

### 3.3 Concurrency Safety Scanner
**Problem:** Can't detect data races or actor isolation issues
**Solution:** Track Sendable conformance and actor boundaries

```python
def find_concurrency_issues(self) -> ToolResult:
    """
    Check for:
    1. Non-Sendable types crossing actor boundaries
    2. @MainActor methods calling non-isolated async
    3. Shared mutable state without synchronization
    4. Missing Sendable conformance on passed types
    """
    # Track actor-isolated types
    isolated_types = self._find_actor_isolated_types()

    # Track Sendable conformance
    sendable_types = self._find_sendable_types()

    # Find violations
    for call in self._find_cross_actor_calls():
        param_type = call.parameter_type
        if param_type not in sendable_types:
            findings.append(...)
```

**Effort:** 16 hours

---

## Phase 4: Data Flow Analysis (Major Effort)

### 4.1 Basic Data Flow Tracking
**Problem:** No understanding of how values flow through code
**Solution:** Build simplified control flow graph

```python
class DataFlowAnalyzer:
    """
    Track variable state through basic blocks.

    Limitations:
    - Intra-procedural only (single function)
    - No pointer/reference tracking
    - Simplified control flow (if/else, loops)
    """

    def __init__(self, function_lines: list[str]):
        self.blocks = self._build_basic_blocks(function_lines)
        self.var_states: dict[str, VarState] = {}

    def trace_variable(self, var_name: str) -> list[VarState]:
        """Trace all states a variable passes through."""
        states = []
        for block in self.blocks:
            for stmt in block.statements:
                if var_name in stmt:
                    states.append(self._analyze_stmt(stmt, var_name))
        return states

    def is_nil_at_line(self, var_name: str, line_num: int) -> bool | None:
        """Return True if definitely nil, False if definitely not, None if unknown."""
        # Backward analysis from line_num
        ...
```

**Effort:** 40 hours

### 4.2 Integration with Tree-sitter
**Problem:** Regex-based parsing is limited
**Solution:** Use tree-sitter for proper AST parsing

```python
# pip install tree-sitter tree-sitter-swift

import tree_sitter_swift as ts_swift
from tree_sitter import Language, Parser

class ASTAnalyzer:
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(Language(ts_swift.language(), "swift"))

    def parse(self, code: str) -> tree_sitter.Tree:
        return self.parser.parse(bytes(code, "utf8"))

    def find_force_unwraps(self, tree: tree_sitter.Tree) -> list[Finding]:
        # Query for force_unwrap_expression nodes
        query = self.parser.language.query("""
            (force_unwrap_expression) @unwrap
        """)

        for match in query.captures(tree.root_node):
            # AST gives us proper context
            parent = match.node.parent
            if parent.type == "guard_statement":
                continue  # Skip guarded unwraps
            ...
```

**Effort:** 60 hours (includes learning curve)

---

## Implementation Priority

| Phase | Task | Impact | Effort | Priority |
|-------|------|--------|--------|----------|
| 1 | Default quality scans to OFF | High | 1h | P0 |
| 1 | Default min_confidence to MEDIUM | High | 30m | P0 |
| 2 | Guard detection for unwraps | High | 4h | P1 |
| 2 | Function boundary detection | Medium | 6h | P1 |
| 3 | Keychain security scanner | High | 8h | P1 |
| 3 | CloudKit sync scanner | Medium | 12h | P2 |
| 2 | Scope-aware nil checking | High | 8h | P2 |
| 3 | Concurrency safety scanner | High | 16h | P2 |
| 4 | Basic data flow | Very High | 40h | P3 |
| 4 | Tree-sitter integration | Very High | 60h | P3 |

---

## Quick Wins (Can Do Now)

### Immediate Changes (< 2 hours total)

1. **Default quality scans OFF**
```python
# In run_ios_scan and run_security_scan
def run_ios_scan(self, min_confidence="MEDIUM", include_quality=False):
```

2. **Separate scan modes**
```python
# Add to rlm_analyze scan_mode options
"scan_mode": {
    "enum": ["auto", "ios", "ios-strict", "security", "quality", "all"],
    # ios-strict = iOS scans only, no quality metrics
}
```

3. **Better summary output**
```markdown
## Scan Results

**Critical:** 2 findings (act immediately)
**High:** 5 findings (fix soon)
**Medium:** 8 findings (review when time permits)
**Low/Style:** 72 findings (hidden, use `include_quality=true` to show)
```

---

## Metrics for Success

After improvements:

| Metric | Current | Target |
|--------|---------|--------|
| False positive rate | ~40% | <15% |
| Time to find real bugs | Buried in noise | Top 10 results |
| iOS-specific coverage | Basic | Comprehensive |
| Semantic understanding | None | Guard/nil-check aware |

---

*Created: 2026-01-18*
*Last updated: 2026-01-18*
