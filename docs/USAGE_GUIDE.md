# RLM-Mem Usage Guide

Practical guide for using RLM-Mem tools in Claude Code with real-world examples.

## Quick Start

### 1. Installation

```bash
# Clone and install
git clone https://github.com/mosif16/RLM-Mem_MCP.git
cd RLM-Mem_MCP/python
pip install -e .

# Set API key (OpenRouter recommended)
export OPENROUTER_API_KEY=sk-or-v1-...

# Add to Claude Code
claude mcp add --transport stdio rlm -- python -m rlm_mem_mcp.server
```

### 2. Verify Installation

In Claude Code, ask:
```
Can you check the RLM server status?
```

Expected response: Server health, cache stats, configuration details.

---

## Common Use Cases

## 1. Security Audit

### Task: Find all security vulnerabilities in a codebase

**In Claude Code:**
```
Please audit this project for security vulnerabilities. Look for:
- SQL injection (string concatenation, dynamic queries)
- XSS vulnerabilities (innerHTML, unsanitized DOM)
- Hardcoded secrets (API keys, passwords)
- Unsafe deserialization
- Path traversal issues
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Find (1) SQL injection via string concat (2) XSS via innerHTML
            (3) hardcoded secrets matching sk-, api_key, password
            (4) eval/exec with user input (5) path traversal.
            Report: file:line, code, severity, fix.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "HIGH"
})
```

**What it finds:**
- SQL injection in user_service.py:42
- Hardcoded API key in config.py:15
- XSS in template rendering (app.js:128)
- Unsafe pickle.loads() in serializer.py:56

**Next steps:**
- Review each finding
- Apply suggested fixes
- Re-run audit to verify fixes
- Store findings in memory for reference

---

## 2. Architecture Review

### Task: Understand how the system is structured

**In Claude Code:**
```
Explain the architecture of this project. What are the main components,
how do they interact, what's the data flow, and what patterns are used?
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Map architecture: (1) main components and modules
            (2) entry points (3) data flow (4) key dependencies
            (5) design patterns. Format as tree structure with
            descriptions and examples.",
  "paths": ["."],
  "min_confidence": "MEDIUM"
})
```

**What it finds:**
- Project structure (API, Services, Data layer)
- Entry points (main.py, cli.py, etc.)
- Component interactions (request flow)
- Design patterns (Factory, Strategy, etc.)
- Data flow (user input → processing → output)

**Understanding the result:**
```
Architecture Overview:
├── API Layer (Flask, FastAPI)
│   ├── /users endpoint
│   ├── /posts endpoint
│   └── /auth endpoint
├── Service Layer
│   ├── UserService
│   ├── PostService
│   └── AuthService
├── Data Layer
│   ├── Repository pattern
│   ├── ORM (SQLAlchemy)
│   └── Database schema
└── Utilities
    ├── Logging
    ├── Error handling
    └── Validation
```

---

## 3. Performance Analysis

### Task: Find performance bottlenecks

**In Claude Code:**
```
Analyze this codebase for performance issues. Look for:
- Inefficient database queries (N+1 problems)
- Missing indexes
- Slow algorithms
- Memory leaks
- Blocking operations
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Find (1) nested loops with database queries (N+1)
            (2) unindexed database queries (3) O(n²) algorithms
            (4) blocking I/O in async code (5) large objects in loops.
            Report: file:line, issue, fix.",
  "paths": ["."],
  "min_confidence": "MEDIUM",
  "include_quality": True
})
```

**What it finds:**
- N+1 queries in user list endpoint
- Inefficient sorting algorithms
- Blocking database calls in async code
- Memory growth in long-running processes

---

## 4. Code Quality Assessment

### Task: Evaluate code quality and standards

**In Claude Code:**
```
Check the code quality. Find:
- Functions that are too long
- High complexity functions
- Dead code
- Missing error handling
- Inconsistent style
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Find (1) functions >100 lines (2) cyclomatic complexity >10
            (3) dead code (4) unused variables (5) missing error handling.
            Report: file:line, metric, suggestion.",
  "paths": ["."],
  "include_quality": True
})
```

---

## 5. Dependency Analysis

### Task: Understand dependencies and identify issues

**In Claude Code:**
```
Analyze the project dependencies. Find:
- Unused dependencies
- Conflicting versions
- Security vulnerabilities in dependencies
- Heavy/bloated dependencies
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Find (1) imports that aren't used (2) outdated dependencies
            (3) conflicting version requirements (4) dependencies with
            known vulnerabilities. Report: file, dependency, issue.",
  "paths": ["requirements.txt", "package.json", "setup.py"],
  "min_confidence": "HIGH"
})
```

---

## 6. iOS/Swift Security

### Task: Audit iOS app for Swift-specific issues

**In Claude Code:**
```
Security audit for this iOS app. Check for:
- Force unwraps
- Missing weak references in closures
- Unsafe memory patterns
- Missing error handling
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Find (1) force unwraps (!) excluding != (2) closures missing
            [weak self] (3) @ObservedObject with default initialization
            (4) unhandled optionals. Report: file:line, code, fix.",
  "paths": ["ios/", "swift/"],
  "scan_mode": "ios",
  "min_confidence": "HIGH"
})
```

---

## 7. Log Analysis

### Task: Parse and analyze large log files

**In Claude Code:**
```
Here's a 500MB production log file. Find all errors, warnings,
and suspicious patterns.
```

**Claude uses:**
```python
mcp__rlm__rlm_query_text({
  "query": "Extract (1) ERROR and CRITICAL entries with timestamps
            (2) stack traces (3) error frequency by type (4) services
            affected (5) time periods with highest errors.
            Format: timestamp | level | service | message | count",
  "text": "[2024-01-19 10:23:45.123] ERROR [auth] Invalid token... \n..."
})
```

**Understanding logs:**
```
Error Summary:
- Total Errors: 1,247
- Timeframe: 2024-01-19 08:00 - 2024-01-19 14:00
- Most common error: "Connection timeout" (45%)
- Affected services: auth (60%), api (30%), db (10%)
- Peak time: 10:30-11:00 (250 errors in 30 min)
```

---

## 8. Configuration Review

### Task: Extract and review all configuration

**In Claude Code:**
```
Extract all configuration from the project. Show:
- Environment variables
- Configuration files
- Feature flags
- Database connection details
```

**Claude uses:**
```python
mcp__rlm__rlm_query_text({
  "query": "Extract (1) all environment variable assignments
            (2) connection strings (3) feature flags (4) API endpoints
            (5) timeouts and limits. Format: key = value | location | usage",
  "text": "entire config file content..."
})
```

---

## 9. Merge Conflict Resolution

### Task: Understand complex merge conflicts

**In Claude Code:**
```
Here's a merge conflict. Help me understand what changed and find
a good resolution.
```

**Claude uses:**
```python
mcp__rlm__rlm_query_text({
  "query": "Analyze merge conflict: (1) what was changed in each branch
            (2) why might there be conflicts (3) what is the intended
            behavior (4) suggest safe resolution. Show original, branch1,
            branch2, and recommended merge.",
  "text": "<<<<<<< HEAD\n...\n=======\n...\n>>>>>>> feature"
})
```

---

## 10. Documentation Generation

### Task: Generate documentation from code

**In Claude Code:**
```
Generate documentation for this module. Include:
- What it does
- Main components
- Key functions
- Data structures
- Example usage
```

**Claude uses:**
```python
mcp__rlm__rlm_analyze({
  "query": "Document this module: (1) purpose and responsibilities
            (2) main classes/functions (3) key algorithms (4) data
            structures (5) dependencies. Format: markdown with examples",
  "paths": ["module/"],
  "min_confidence": "MEDIUM"
})
```

---

## Advanced Patterns

### Pattern 1: Iterative Analysis

**Workflow:**
```
Initial audit → Find issues → Store in memory → Fix issues
    ↓
Verify fixes → Run targeted analysis → Update memory → Repeat
```

**In Claude Code:**
```
# First run - find all security issues
[Audit finds 15 vulnerabilities]

# Store findings
Store in memory with tag "security-audit-2024-01"

# Fix issues
[Developer fixes 5 issues]

# Run targeted re-check
Analyze only modified files for the same issues

# Track progress
Recall memory: "security-audit-2024-01"
Compare: Found 10 issues (5 fixed, 5 to go)
```

### Pattern 2: Comparative Analysis

**Workflow:**
```
Analyze version 1 → Analyze version 2 → Compare → Document changes
```

**In Claude Code:**
```
# Analyze old version
What security issues are in version 1?

# Analyze new version
What security issues are in version 2?

# Store both
Memory store both analyses with tags "version-1" and "version-2"

# Compare
What vulnerabilities were fixed? What new ones appeared?
```

### Pattern 3: Compliance Checking

**Workflow:**
```
Define requirements → Audit against requirements → Report gaps → Verify fixes
```

**In Claude Code:**
```
# Check OWASP compliance
Audit for: - Input validation - Authentication - Authorization
           - Encryption - Error handling - Logging

# Check PCI-DSS compliance
Audit for: - Secure coding - Data protection - Network security

# Generate compliance report
Store findings with tag "compliance-2024"
```

---

## Tips & Tricks

### Tip 1: Be Specific in Queries

**Bad query:**
```
"Find problems in the code"
```

**Good query:**
```
"Find SQL injection by (1) string concatenation in queries
 (2) unparameterized user input to database (3) dynamic
 SQL construction. Report: file:line, code, severity"
```

**Why?** Specific queries generate more targeted code analysis, reducing false positives.

### Tip 2: Use Min Confidence

```
# Find only high-confidence issues
min_confidence: "HIGH"

# Good for: Critical security audits, production issues

# Find all potential issues (more false positives)
min_confidence: "LOW"

# Good for: Initial exploration, catching edge cases
```

### Tip 3: Leverage Scan Modes

```
# Security analysis
scan_mode: "security"

# iOS-specific checks
scan_mode: "ios"

# Code quality
scan_mode: "quality"

# Everything
scan_mode: "all"
```

### Tip 4: Use Memory for Long Projects

```
# After each analysis session
Store findings: rlm_memory_store({
  "key": "project_audit_phase_1",
  "value": "Summary of findings",
  "tags": ["audit", "phase-1", "critical"]
})

# Later, recall progress
Findings = rlm_memory_recall({
  "search_tags": ["audit", "critical"]
})
```

### Tip 5: Iterate on Complex Analysis

```
# Don't try to analyze everything at once
# Break into phases:

Phase 1: Security vulnerabilities
Phase 2: Performance issues
Phase 3: Code quality
Phase 4: Architecture review
Phase 5: Dependency analysis

# Each phase is faster, cleaner results
```

---

## Troubleshooting

### Issue: Getting too many false positives

**Solution:**
```
# Increase confidence threshold
min_confidence: "HIGH"

# Be more specific in query
# Add semantic context
# Exclude test files
```

### Issue: Analysis takes too long

**Solution:**
```
# Reduce files analyzed
paths: ["src/", "api/"]  # Instead of ["."]

# Exclude large directories
export RLM_EXTRA_SKIP_DIRS=node_modules,dist,build

# Use faster model
export RLM_MODEL=google/gemini-2-flash
```

### Issue: Results are too brief

**Solution:**
```
# Increase result tokens
export RLM_MAX_RESULT_TOKENS=8000

# Use more capable model
export RLM_MODEL=anthropic/claude-opus-4.5

# Ask for more detail in query
"Provide detailed analysis with specific examples"
```

### Issue: Results are not specific enough

**Solution:**
```
# Improve query specificity
"Find exact lines and code snippets, not general patterns"

# Add multiple criteria
"Find A, B, and C - report where all are present"

# Use structured output
"Format as: file | line | code | issue | fix"
```

---

## Cost Optimization

### Strategy 1: Cache Frequently

```
# Enable caching
export RLM_USE_CACHE=true
export RLM_CACHE_TTL=1h

# Re-run analysis frequently
# 90% cost reduction on cache hits
```

### Strategy 2: Use Fast Models

```
# Gemini Flash: ~$0.15/1M tokens (recommended)
export RLM_MODEL=google/gemini-2.5-flash-lite

# Claude 3.5: ~$3/1M tokens (10x more expensive)
export RLM_MODEL=anthropic/claude-3-5-sonnet

# Gemini Flash 2: ~$0.10/1M tokens
export RLM_MODEL=google/gemini-2-flash
```

### Strategy 3: Optimize Queries

```
# Small, specific queries = fewer tokens
"Find SQL injection" → costs $0.10
"Find all possible vulnerabilities" → costs $2.00

# Narrow paths = fewer files
paths: ["src/api/"] → faster, cheaper
paths: ["."] → comprehensive, expensive
```

---

## See Also

- [API Reference](./API.md) - Complete tool parameters
- [Configuration Guide](./CONFIGURATION.md) - Environment variables
- [Architecture](./ARCHITECTURE.md) - How it works internally
- [README](../README.md) - Project overview

For questions or issues:
- Check [GitHub Issues](https://github.com/mosif16/RLM-Mem_MCP/issues)
- Read the [paper](https://arxiv.org/abs/2512.24601) - Recursive Language Models
