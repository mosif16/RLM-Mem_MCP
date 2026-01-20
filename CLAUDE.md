# RLM-Mem MCP Server - Claude Code Guidelines

This is the main documentation for the RLM-Mem MCP (Model Context Protocol) server. For working with Claude Code on this project, use the patterns and tools described here.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Usage Guide](#usage-guide)
7. [Development](#development)

---

## Project Overview

**RLM-Mem MCP Server** implements the TRUE Recursive Language Model (RLM) technique from [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) for ultimate context management with Claude Code.

### The Problem

Claude Code has a context window of ~200k tokens. When analyzing large codebases (500k+ tokens), Claude either:
- Fails to process everything
- Experiences "context rot" (degraded performance)
- Runs out of space for reasoning

### The Solution: TRUE RLM

**Key Insight**: Content is stored as a **VARIABLE** in a Python REPL, NOT in LLM context.

```
Traditional Summarization (NOT what we do):
    Large content → LLM summarizes → Information LOST

TRUE RLM Technique:
    Large content → Stored as `prompt` variable
    LLM writes Python CODE to examine portions
    Sub-LLM responses stored as VARIABLES (NOT summarized)
    Full data PRESERVED - accessible at any time
```

The LLM acts as a programmer, writing code to search and analyze the content rather than trying to hold it all in context.

### Features

- **TRUE RLM Processing**: Content stored as variables, LLM writes code to examine it
- **Prompt Caching**: Leverages caching for cost reduction on repeated content
- **Intelligent Chunking**: Respects file/function/section boundaries when splitting
- **Memory Store**: Persist important findings across conversations (SQLite-backed)
- **Robust Architecture**: Circuit breakers, rate limiters, exponential backoff
- **Async Pipeline**: Fully async with connection pooling and concurrent operations
- **Multi-Model Support**: Works with OpenRouter (Gemini, Claude, etc.) or Anthropic direct

---

## Core Concepts

### TRUE RLM Technique

Unlike simple summarization, TRUE RLM:

1. **Content as Variable**: Files stored in `prompt` variable, NOT in LLM context
2. **LLM Writes Code**: The LLM generates Python to examine `prompt`
3. **Sub-LLM Queries**: `llm_query()` calls analyze specific portions
4. **Results as Variables**: Sub-LLM responses stored in full, NOT summarized
5. **Full Preservation**: Original data always accessible for re-examination

### Processing Steps

1. **File Collection**: Async walk directories, filter by extension, respect limits
2. **Variable Storage**: Content stored in REPL environment as `prompt` variable
3. **Code Generation**: LLM writes Python code to search/analyze content
4. **Sandboxed Execution**: Code runs in restricted environment with `llm_query()`
5. **Result Aggregation**: Findings combined into coherent response
6. **Truncation**: Ensure result fits in context (max 4000 tokens)

### Cost Comparison

| Method | 500k token input | Context Used | Cost |
|--------|------------------|--------------|------|
| Direct (if possible) | Fails or degrades | 200k+ (full) | N/A |
| Premium 1M context | Works | 500k | ~$15 |
| **RLM via MCP** | Works | ~4k summary | **~$0.10-0.50** |

RLM with Claude Haiku 4.5 and prompt caching is **extremely cost-effective**:
- Base: $0.80/1M input tokens
- With cache hits: $0.08/1M (90% savings!)
- Output: $4/1M tokens

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code (User)                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │ MCP Protocol (JSON-RPC over stdio)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RLM-Mem MCP Server                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ MCP Resource Handler (server.py)                         │  │
│  │ - Tool definitions and request routing                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ RLM Processor (rlm_processor.py)                         │  │
│  │ - File collection, chunking, orchestration              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ REPL Environment (repl_environment.py)                  │  │
│  │ - Python execution sandbox with llm_query()            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Support Layer                                            │  │
│  │ - Cache, Files, Memory, Verification, Tools            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┬─────────┐
        ▼         ▼         ▼         ▼
    ┌─────┐  ┌────────┐  ┌──────┐  ┌──────┐
    │Cache│  │OpenRouter │Anthropic│SQLite│
    │    │  │   API     │  API    │(Mem) │
    └─────┘  └────────┘  └──────┘  └──────┘
```

### Core Components

#### 1. MCP Server (server.py)
- Handle MCP protocol and resource management
- Route tool requests to appropriate handler
- Error handling and response formatting

**Tools Exposed**:
- `rlm_analyze` - File/directory analysis
- `rlm_query_text` - Text block processing
- `rlm_memory_store` - Persist findings
- `rlm_memory_recall` - Retrieve findings
- `rlm_status` - Server health check

#### 2. RLM Processor (rlm_processor.py)
Orchestrates the TRUE RLM analysis pipeline:
- File collection and chunking
- Code generation for analysis
- Result aggregation and verification

**Process**:
```
Input → Collection → Chunking → Storage → Code Generation
  → Execution → Aggregation → Output
```

#### 3. REPL Environment (repl_environment.py)
Provides execution sandbox with custom functions:
```python
prompt        # str - File/text content
context       # dict - Metadata about content
results       # list - Accumulate findings
llm_query()   # func - Query LLM for specific portions
```

#### 4. Structured Tools (structured_tools.py)
Pre-built analysis functions available in REPL:
- `find_secrets()` - Detect hardcoded credentials
- `find_sql_injection()` - SQL injection patterns
- `find_xss()` - XSS vulnerabilities
- `find_force_unwraps()` - iOS unsafe unwrapping
- `map_architecture()` - System architecture
- `analyze_performance()` - Performance issues

#### 5. Result Verifier (result_verifier.py)
Ensures result quality and accuracy:
- Pattern verification
- Semantic verification
- False positive filtering
- Confidence scoring (L11 algorithm)
- Deduplication

**Confidence Scoring**:
```
Start Score: 100

Deductions:
- In dead code: -50
- In test file: -30
- Pattern match only: -10
- In comment: -40
- Can't verify line: -20

Boosts:
- Semantic verification: +20
- Multiple indicators: +15

Final:
- ≥80: HIGH
- 50-79: MEDIUM
- 20-49: LOW
- <20: FILTERED (false positive)
```

#### 6. Cache Manager (cache_manager.py)
Implements Anthropic-style prompt caching:
- System prompts cached with 5m/1h TTL
- Reduces cost by ~90% on cache hits
- LRU response cache for redundant calls

#### 7. File Collector (file_collector.py)
Efficiently traverse directories:
- Async I/O for non-blocking access
- Filtering by extension and path pattern
- File limits and size constraints

#### 8. Memory Store (memory_store.py)
Persistent finding storage:
- SQLite backend (survives server restarts)
- Fast indexed lookup
- Tag-based searching

---

## API Reference

### rlm_analyze

Analyze files or directories using custom search queries.

**Parameters**:
```javascript
{
  "query": "string (required)",        // Specific analysis query
  "paths": ["string"] (required),      // Files/directories to analyze
  "scan_mode": "string",               // 'auto', 'security', 'ios', 'quality'
  "min_confidence": "string",          // 'HIGH', 'MEDIUM', 'LOW'
  "include_quality": "boolean"         // Include code quality checks
}
```

**Query Patterns**:

**Security Analysis:**
```
"Find (1) SQL injection via string concat (2) hardcoded secrets
 matching sk-, api_key, password (3) eval/exec with user input.
 Report: file:line, code, severity."
```

**iOS/Swift Specific:**
```
"Find (1) force unwraps (!) excluding != (2) closures missing
 [weak self] (3) @ObservedObject with default value.
 Report: file:line, code, fix."
```

**Python Vulnerabilities:**
```
"Find (1) pickle.loads with untrusted data (2) bare except clauses
 (3) mutable default args. Report: file:line, code."
```

**JavaScript Issues:**
```
"Find (1) innerHTML XSS (2) missing await (3) useEffect missing deps.
 Report: file:line, code."
```

**Architecture Review:**
```
"Map all modules with purpose, entry points, dependencies,
 data flow."
```

**Response Structure**:
```javascript
{
  "findings": [
    {
      "file": "path/to/file.py",
      "line": 42,
      "code": "source code snippet",
      "issue": "description of issue",
      "confidence": "HIGH|MEDIUM|LOW|FILTERED",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "fix": "suggested fix (optional)",
      "category": "category tag"
    }
  ],
  "summary": "Brief summary of findings",
  "files_scanned": 123,
  "errors": []
}
```

### rlm_query_text

Process large text blocks (logs, transcripts, documents) directly.

**Parameters**:
```javascript
{
  "query": "string (required)",        // What to extract/analyze
  "text": "string (required)"           // The text content to process
}
```

**Query Patterns**:

**Log Analysis:**
```
"Extract (1) ERROR/WARN entries with timestamps (2) stack traces
 with root cause (3) error frequency by type.
 Format: timestamp | level | message | count"
```

**Configuration Extraction:**
```
"Extract (1) all environment variables (2) connection strings
 (3) feature flags. Format: key = value with file location"
```

**Transcript Analysis:**
```
"Extract (1) key decisions made (2) action items with owners
 (3) unresolved questions. Format: bullet points with timestamps"
```

**JSON/Data Analysis:**
```
"Extract (1) all unique field names (2) data types per field
 (3) nested structure depth. Format: field: type (count)"
```

### rlm_memory_store

Persist important findings for later recall.

**Parameters**:
```javascript
{
  "key": "string (required)",           // Unique identifier
  "value": "string (required)",         // Content to store
  "tags": ["string"] (optional)         // Categorization tags
}
```

### rlm_memory_recall

Retrieve stored findings.

**Parameters (Option 1: By Key)**:
```javascript
{
  "key": "security_audit_2024-01-19"
}
```

**Parameters (Option 2: By Tags)**:
```javascript
{
  "search_tags": ["security", "high-priority"]
}
```

### rlm_status

Check server health, cache stats, and configuration.

**Response Structure**:
```javascript
{
  "status": "healthy|degraded|unhealthy",
  "uptime_seconds": 3600,
  "config": { ... },
  "cache_stats": { ... },
  "memory": { ... },
  "rate_limit": { ... }
}
```

### Confidence Levels

The system uses a 4-level confidence scale:

- **HIGH** (80+): Multiple indicators verified, semantic confirmation
- **MEDIUM** (50-79): Pattern matched, some uncertainty
- **LOW** (20-49): Weak pattern match, likely false positive
- **FILTERED**: Ruled out (dead code, test file, etc.)

---

## Configuration

### Installation

```bash
# Clone the repository
git clone https://github.com/mosif16/RLM-Mem_MCP.git
cd RLM-Mem_MCP

# Install Python dependencies
cd python
pip install -e .

# Set your API key (OpenRouter recommended for flexibility)
export OPENROUTER_API_KEY=sk-or-...

# Or use Anthropic directly
export ANTHROPIC_API_KEY=sk-ant-...
```

### Configure Claude Code

Add the MCP server to Claude Code:

```bash
# Using the CLI
claude mcp add --transport stdio rlm -- python -m rlm_mem_mcp.server

# Or add to ~/.claude/mcp_servers.json manually
```

**Manual configuration** (`~/.claude/mcp_servers.json`):

```json
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["-m", "rlm_mem_mcp.server"],
      "env": {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `RLM_MODEL` | `anthropic/claude-haiku-4.5` | Model for RLM processing |
| `RLM_AGGREGATOR_MODEL` | `anthropic/claude-haiku-4.5` | Model for final aggregation |
| `RLM_USE_CACHE` | `true` | Enable prompt caching |
| `RLM_CACHE_TTL` | `1h` | Cache TTL (`5m` or `1h`) |
| `RLM_USE_PREFILLED` | `true` | Enable prefilled responses for token efficiency |
| `RLM_MAX_RESULT_TOKENS` | `4000` | Max tokens in result |
| `RLM_MAX_CHUNK_TOKENS` | `8000` | Max tokens per chunk |
| `RLM_OVERLAP_TOKENS` | `200` | Overlap tokens between chunks |

### File Filtering

**Included extensions**:
- Code: `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, etc.
- Config: `.json`, `.yaml`, `.toml`, `.ini`
- Docs: `.md`, `.txt`, `.rst`

**Skipped directories**:
- `.git`, `node_modules`, `__pycache__`, `venv`, `dist`, `build`, etc.

---

## Usage Guide

### Common Use Cases

#### 1. Security Audit

Find all security vulnerabilities in a codebase.

**In Claude Code:**
```
Please audit this project for security vulnerabilities. Look for:
- SQL injection (string concatenation, dynamic queries)
- XSS vulnerabilities (innerHTML, unsanitized DOM)
- Hardcoded secrets (API keys, passwords)
- Unsafe deserialization
- Path traversal issues
```

**What it finds:**
- SQL injection in user_service.py:42
- Hardcoded API key in config.py:15
- XSS in template rendering (app.js:128)
- Unsafe pickle.loads() in serializer.py:56

#### 2. Architecture Review

Understand how the system is structured.

**In Claude Code:**
```
Explain the architecture of this project. What are the main components,
how do they interact, what's the data flow, and what patterns are used?
```

**What it finds:**
- Project structure (API, Services, Data layer)
- Entry points (main.py, cli.py, etc.)
- Component interactions (request flow)
- Design patterns (Factory, Strategy, etc.)
- Data flow (user input → processing → output)

#### 3. Performance Analysis

Find performance bottlenecks.

**In Claude Code:**
```
Analyze this codebase for performance issues. Look for:
- Inefficient database queries (N+1 problems)
- Missing indexes
- Slow algorithms
- Memory leaks
- Blocking operations
```

**What it finds:**
- N+1 queries in user list endpoint
- Inefficient sorting algorithms
- Blocking database calls in async code
- Memory growth in long-running processes

#### 4. Code Quality Assessment

Evaluate code quality and standards.

**In Claude Code:**
```
Check the code quality. Find:
- Functions that are too long
- High complexity functions
- Dead code
- Missing error handling
- Inconsistent style
```

#### 5. Log Analysis

Parse and analyze large log files.

**In Claude Code:**
```
Here's a 500MB production log file. Find all errors, warnings,
and suspicious patterns.
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

#### 6. iOS/Swift Security

Audit iOS app for Swift-specific issues.

**In Claude Code:**
```
Security audit for this iOS app. Check for:
- Force unwraps
- Missing weak references in closures
- Unsafe memory patterns
- Missing error handling
```

#### 7. Dependency Analysis

Understand dependencies and identify issues.

**In Claude Code:**
```
Analyze the project dependencies. Find:
- Unused dependencies
- Conflicting versions
- Security vulnerabilities in dependencies
- Heavy/bloated dependencies
```

### Advanced Patterns

#### Pattern 1: Iterative Analysis

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

#### Pattern 2: Comparative Analysis

**Workflow:**
```
Analyze version 1 → Analyze version 2 → Compare → Document changes
```

#### Pattern 3: Compliance Checking

**Workflow:**
```
Define requirements → Audit against requirements → Report gaps → Verify fixes
```

### Tips & Tricks

#### Tip 1: Be Specific in Queries

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

#### Tip 2: Use Min Confidence

```
# Find only high-confidence issues
min_confidence: "HIGH"

# Good for: Critical security audits, production issues

# Find all potential issues (more false positives)
min_confidence: "LOW"

# Good for: Initial exploration, catching edge cases
```

#### Tip 3: Leverage Scan Modes

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

#### Tip 4: Use Memory for Long Projects

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

#### Tip 5: Iterate on Complex Analysis

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

### Troubleshooting

#### Issue: Getting too many false positives

**Solution:**
```
# Increase confidence threshold
min_confidence: "HIGH"

# Be more specific in query
# Add semantic context
# Exclude test files
```

#### Issue: Analysis takes too long

**Solution:**
```
# Reduce files analyzed
paths: ["src/", "api/"]  # Instead of ["."]

# Exclude large directories
export RLM_EXTRA_SKIP_DIRS=node_modules,dist,build

# Use faster model
export RLM_MODEL=google/gemini-2-flash
```

#### Issue: Results are too brief

**Solution:**
```
# Increase result tokens
export RLM_MAX_RESULT_TOKENS=8000

# Use more capable model
export RLM_MODEL=anthropic/claude-opus-4.5

# Ask for more detail in query
"Provide detailed analysis with specific examples"
```

#### Issue: Results are not specific enough

**Solution:**
```
# Improve query specificity
"Find exact lines and code snippets, not general patterns"

# Add multiple criteria
"Find A, B, and C - report where all are present"

# Use structured output
"Format as: file | line | code | issue | fix"
```

### Cost Optimization

#### Strategy 1: Cache Frequently

```
# Enable caching
export RLM_USE_CACHE=true
export RLM_CACHE_TTL=1h

# Re-run analysis frequently
# 90% cost reduction on cache hits
```

#### Strategy 2: Model Selection

```
# Uses Claude Haiku 4.5 exclusively
# ~$0.80/1M tokens input, $0.08 with cache (90% savings)
export RLM_MODEL=anthropic/claude-haiku-4.5
```

#### Strategy 3: Optimize Queries

```
# Small, specific queries = fewer tokens
"Find SQL injection" → costs $0.10
"Find all possible vulnerabilities" → costs $2.00

# Narrow paths = fewer files
paths: ["src/api/"] → faster, cheaper
paths: ["."] → comprehensive, expensive
```

---

## Development

### Project Structure

```
RLM-Mem_MCP/
├── python/
│   ├── src/
│   │   └── rlm_mem_mcp/
│   │       ├── __init__.py              # Package exports
│   │       ├── server.py                # MCP server entry point
│   │       ├── rlm_processor.py         # Core RLM implementation
│   │       ├── repl_environment.py      # TRUE RLM REPL with llm_query()
│   │       ├── file_collector.py        # Async file collection
│   │       ├── cache_manager.py         # Prompt caching
│   │       ├── memory_store.py          # SQLite-backed persistent memory
│   │       ├── structured_tools.py      # Pre-built analysis tools
│   │       ├── result_verifier.py       # Confidence scoring and verification
│   │       ├── config.py                # Environment configuration
│   │       ├── utils.py                 # Performance monitoring
│   │       └── ...other modules
│   ├── tests/
│   │   ├── test_integration.py          # End-to-end tests
│   │   ├── test_benchmark.py            # Performance benchmarks
│   │   ├── test_stress.py               # Stress tests
│   │   └── conftest.py                  # Test fixtures
│   ├── requirements.txt
│   └── pyproject.toml
├── .claude/                             # Claude Code session storage
│   └── ... (auto-managed)
├── docs/                                # Documentation (archived)
├── .mcp.json                            # Project MCP config
├── CLAUDE.md                            # This file
└── README.md                            # Quick start
```

### Running Tests

```bash
cd python
pip install -e ".[dev]"
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Key Design Patterns

#### Circuit Breaker
```
State: CLOSED (normal operation)
    ↓
[3 consecutive failures]
    ↓
State: OPEN (stop requests)
    ↓
[Wait 30 seconds]
    ↓
State: HALF_OPEN (test request)
    ↓
Success? → CLOSED
Failure? → OPEN (reset timer)
```

#### Exponential Backoff
```
Attempt 1: Wait 1s
Attempt 2: Wait 2s
Attempt 3: Wait 4s
Attempt 4: Wait 8s
Attempt 5: Wait 16s
Max: 32s, Max attempts: 5
```

#### Rate Limiting
```
Token bucket algorithm:
- Bucket size: 1,000,000 tokens/min
- Refill rate: Configured
- On request: Deduct tokens
- If insufficient: Reject (429)
```

### Performance Characteristics

#### Time Complexity
- File collection: O(n) where n = file count
- Chunking: O(m) where m = total tokens
- Analysis: O(k) where k = chunk count (parallelizable)
- Aggregation: O(r) where r = result count

#### Real-World Performance
| Operation | Typical Time | Range |
|-----------|--------------|-------|
| Collect 100 files | 50ms | 10-200ms |
| Chunk 100k tokens | 100ms | 50-300ms |
| Analyze 10 chunks | 5-10s | 3-30s |
| Aggregate results | 100ms | 50-500ms |
| **Total** | **5-12s** | **3-35s** |

### Security Considerations

#### Sandbox Security
- REPL executes in restricted Python environment
- No file system access (except `prompt` variable)
- No network access (except controlled `llm_query()`)
- Timeout protection (30s per chunk)
- Memory limits enforced

#### API Security
- API keys stored in environment variables
- No keys in logs or error messages
- Credentials not returned to Claude Code
- Rate limiting prevents abuse

#### Data Privacy
- File content never stored persistently (except REPL variable)
- Memory store stores only findings, not source code
- Cache expires automatically
- User can disable memory store

---

## References

### Papers
- [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601) - Zhang, Kraska, Khattab - MIT, 2025

### Anthropic Documentation
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)

### Related Projects
- [Official RLM Library](https://github.com/alexzhang13/rlm)
- [MCP SDK](https://github.com/anthropics/mcp)

---

## Quick Reference

### When to use rlm_analyze
- Analyzing 50+ files
- Searching entire codebase
- Tasks with "all", "every", or "entire" scope
- Security audits
- Architecture reviews

### When NOT to use rlm_analyze
- Working with 1-5 specific files
- Making targeted edits
- Quick lookups in known locations

### Best Practices
1. Be specific in your queries - avoid vague searches
2. Use appropriate confidence levels (HIGH for critical, MEDIUM for general)
3. Leverage scan modes for targeted analysis
4. Store important findings in memory for reference
5. Break complex analysis into phases
6. Cache results when possible for cost savings

---

## License

MIT License - see LICENSE for details.

---

**Last Updated**: January 2026
**Version**: 2.0
**Status**: Production Ready
