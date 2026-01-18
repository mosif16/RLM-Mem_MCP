# CLAUDE.md - Guidance for Claude Code

This repository contains an MCP server implementing the Recursive Language Model (RLM) technique for handling large context with Claude.

## Project Overview

**RLM-Mem MCP** provides tools for Claude Code to process arbitrarily large inputs (codebases, logs, documents) that would otherwise exceed the context window.

## How RLM Works

The RLM technique stores content in a **variable** (not in LLM context):
1. Files are collected and combined into a `prompt` variable
2. An LLM writes **Python code** to examine portions of the content
3. Sub-LLM calls analyze chunks, results stored in **full** (not summarized)
4. Final answer compiled from all findings

This means: **Your query drives the code the LLM writes to search the content.**

## Large Codebase Protocol

### When to use RLM tools:

- **Analyzing 50+ files** - Use `rlm_analyze` instead of reading each file
- **Searching entire codebase** - For queries like "find all X in the repo"
- **Tasks with "all", "every", or "entire" scope**
- **Security audits** - Need to check all files for vulnerabilities
- **Architecture reviews** - Understanding overall structure
- **Log analysis** - Processing large log files

### When NOT to use RLM tools:

- Working with 1-5 specific files (just read them directly)
- Making targeted edits to known files
- Quick lookups in known locations
- Simple file operations

## ⚠️ RLM Behavior Notes

**RLM now auto-handles broad queries** by decomposing them into focused sub-queries.

| Feature | Behavior |
|---------|----------|
| Broad queries ("audit everything") | Auto-decomposed into 3-4 focused queries |
| Truncated responses | Auto-detected and continued |
| Code execution failures | Retried up to 3 times with recovery |
| Relevance threshold | Dynamically adjusted based on query specificity |

### Still Requires Manual Verification

Even with these improvements, ~40-60% of findings benefit from Grep/Read verification:
- **False positives** can still occur (test code flagged as vulnerable, etc.)
- **Context matters** - RLM can't understand business logic
- **Dead code detection** - Now includes confidence levels (HIGH/MEDIUM/LOW)

## Query Construction Rules (CRITICAL)

### Anti-Patterns to Avoid

These query patterns consistently fail or produce unreliable results:

| ❌ AVOID | Why It Fails | ✅ INSTEAD |
|----------|--------------|------------|
| "audit this codebase" | Too broad - LLM doesn't know what to look for | Run 3-4 focused queries (security, architecture, quality) |
| "find all problems" | Undefined scope - returns nothing or false positives | "Find X, Y, Z issues with file:line output" |
| "check everything for security" | Overloaded - code generation gets truncated | One query per vulnerability class |
| "summarize and also find bugs and review architecture" | Multiple unrelated goals - partial results | Separate queries for each goal |

### Rule 1: Be Exhaustively Specific

The LLM uses your query to decide what code to write. Vague queries = vague searches.

| ❌ BAD | ✅ GOOD |
|--------|---------|
| "find problems" | "find: SQL injection, XSS, CSRF, command injection, path traversal, hardcoded secrets, insecure deserialization, auth bypasses" |
| "summarize" | "list: all modules, their purpose, entry points, data flow between components, external dependencies" |
| "check security" | "find security issues: (1) injection flaws - SQL, command, LDAP, XPath (2) broken auth - weak sessions, credential exposure (3) sensitive data exposure - hardcoded keys, API tokens in code (4) XXE, SSRF, insecure deserialization" |

### Rule 2: Use Structured Query Format

```
TASK: [what you want to find/analyze]
LOOK FOR:
- [specific item 1]
- [specific item 2]
- [specific item 3]
OUTPUT: [how you want results formatted]
```

### Rule 3: One Focused Query Per Call

Instead of one massive query, make multiple focused calls:

```
# Call 1: Security
query: "Find injection vulnerabilities: SQL injection via string concatenation, command injection via os.system/subprocess, path traversal via user-controlled file paths"

# Call 2: Architecture
query: "Map the architecture: list all classes/modules, their responsibilities, how they connect, entry points, data flow"

# Call 3: Error Handling
query: "Find error handling issues: bare except clauses, swallowed exceptions, missing try/except around I/O, resource leaks"
```

### Example Queries by Task Type

**Security Audit:**
```
Find security vulnerabilities in this Python codebase:
1. INJECTION: SQL via string concat, command via subprocess/os.system, code via eval/exec
2. SECRETS: hardcoded API keys, passwords, tokens in source code
3. PATH TRAVERSAL: user input used in file paths without sanitization
4. DESERIALIZATION: pickle.loads, yaml.load without SafeLoader
5. AUTH: weak session handling, credential exposure, missing access controls
For each finding: file path, line number, code snippet, severity (critical/high/medium/low)
```

**Architecture Review:**
```
Analyze the codebase architecture:
1. List all modules/packages and their single-line purpose
2. Identify entry points (main functions, API endpoints, CLI commands)
3. Map dependencies between modules (who imports whom)
4. Identify external dependencies and what they're used for
5. Note any circular dependencies or architectural concerns
```

**Code Quality:**
```
Find code quality issues:
1. Functions over 50 lines (list them with line counts)
2. Deeply nested code (3+ levels of indentation)
3. Duplicate code patterns (similar logic in multiple places)
4. Missing type hints on public functions
5. TODO/FIXME/HACK comments (list with context)
```

## Verification Workflow (REQUIRED)

**Every RLM finding must be verified.** RLM surfaces candidates; you confirm them.

### Standard Verification Steps

```
1. RLM_ANALYZE → Get list of potential issues with file:line references
2. GREP → Search for the specific pattern RLM flagged
3. READ → Examine the actual code in context
4. CONFIRM/REJECT → Mark as real issue or false positive
```

### Example Workflow

```bash
# Step 1: RLM finds potential SQL injection
rlm_analyze("Find SQL injection: string concatenation in queries", ["./src"])
# Result: "Potential SQLi in src/db/users.py:45 - query built with f-string"

# Step 2: Verify with Grep
Grep("f\"SELECT.*{", path="./src/db")
# Result: Confirms pattern exists in users.py:45

# Step 3: Read the actual code
Read("./src/db/users.py", offset=40, limit=15)
# Result: See full context - is it actually user input? Parameterized elsewhere?

# Step 4: Verdict
# Real issue: Report with evidence
# False positive: Note why (e.g., "input is sanitized at line 38")
```

### Verification Priority

| RLM Confidence | Verification Effort |
|----------------|---------------------|
| "Definitely found X at file:line" | Quick Grep to confirm |
| "Possibly X in file" | Grep + Read full function |
| "May have X patterns" | Read + trace data flow |
| "No findings" | Run narrower queries, don't trust "clean bill" |

### Common False Positives

RLM often flags these incorrectly:
- **Already-implemented features** marked as "missing"
- **Test code** flagged as "vulnerable" (intentional test cases)
- **Commented-out code** reported as active
- **Type-safe patterns** misidentified as injection risks

## Available Tools

### `rlm_analyze`
```json
{
  "query": "What to find/analyze",
  "paths": ["./src", "./lib"]
}
```

### `rlm_query_text`
```json
{
  "query": "What to extract",
  "text": "<large text content>"
}
```

### `rlm_status`
Check server health and cache statistics.

### `rlm_memory_store` / `rlm_memory_recall`
Store and retrieve important findings for cross-conversation persistence.

## Development Commands

```bash
# Install dependencies
cd python && pip install -e .

# Run the server manually
python -m rlm_mem_mcp.server

# Run tests
pytest
```

## File Structure

- `python/src/rlm_mem_mcp/` - Main Python MCP server
- `src/` - Optional TypeScript implementation
- `.mcp.json` - MCP configuration for this project

## Key Files

- `python/src/rlm_mem_mcp/server.py` - MCP server entry point
- `python/src/rlm_mem_mcp/rlm_processor.py` - RLM algorithm implementation
- `python/src/rlm_mem_mcp/cache_manager.py` - Prompt caching logic
- `python/src/rlm_mem_mcp/file_collector.py` - File collection utilities
