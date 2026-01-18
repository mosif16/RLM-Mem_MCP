# RLM-Mem MCP - Codebase State & Agent Notes

> This file tracks codebase state, recent changes, and agent observations.
> For usage documentation, see [README.md](README.md).

---

## Codebase Overview

**Purpose**: MCP server implementing TRUE RLM technique (arXiv:2512.24601) for processing large codebases that exceed Claude's context window.

**Tech Stack**: Python 3.10+, OpenRouter API (Gemini Flash), MCP Protocol

**Entry Point**: `python -m rlm_mem_mcp.server`

---

## Key Architecture

```
User Query → Claude Code → MCP Tool Call
                              ↓
                    ┌─────────────────────┐
                    │   RLM MCP Server    │
                    │   (server.py)       │
                    └──────────┬──────────┘
                               ↓
              ┌────────────────┴────────────────┐
              ↓                                 ↓
    ┌─────────────────┐              ┌─────────────────┐
    │ File Collector  │              │ RLM Processor   │
    │ (async I/O)     │              │ (rlm_processor) │
    └─────────────────┘              └────────┬────────┘
                                              ↓
                                   ┌─────────────────┐
                                   │ REPL Environment│
                                   │ (repl_env.py)   │
                                   │                 │
                                   │ prompt = content│
                                   │ llm_query(...)  │
                                   │ verify_line()   │
                                   └─────────────────┘
```

---

## Critical Files

| File | Purpose | Last Modified |
|------|---------|---------------|
| `server.py` | MCP server, tool definitions, request routing | 2025-01 |
| `rlm_processor.py` | Query analysis, decomposition, chunk processing | 2025-01 |
| `repl_environment.py` | TRUE RLM: content as variable, LLM writes code | 2025-01 |
| `structured_tools.py` | **NEW**: Preconfigured search tools with typed output | 2025-01 |
| `config.py` | Environment config (OPENROUTER_API_KEY, models) | 2025-01 |
| `fallback_analyzer.py` | Pattern-based fallback when REPL fails | 2025-01 |
| `content_analyzer.py` | Dead code detection, line verification | 2025-01 |

---

## Structured Tools System

The REPL now exposes preconfigured tools instead of requiring raw Python:

```python
# OLD: Write Python code (error-prone)
matches = []
for line in prompt.split('\n'):
    if re.search(r'api_key', line):
        matches.append(line)

# NEW: Call structured tool (reliable)
result = find_secrets()
FINAL_ANSWER = result.to_markdown()
```

### Tool Output Format

All tools return `ToolResult`:
```python
result.findings      # List[Finding] - structured findings
result.count         # int - total findings
result.high_confidence  # List[Finding] - only HIGH confidence
result.to_markdown() # str - formatted for FINAL_ANSWER
```

Each `Finding` contains:
```python
finding.file        # str - file path
finding.line        # int - line number
finding.code        # str - code snippet
finding.issue       # str - issue description
finding.confidence  # HIGH/MEDIUM/LOW
finding.severity    # CRITICAL/HIGH/MEDIUM/LOW/INFO
finding.fix         # str - suggested fix
```

---

## Recent Changes

### 2025-01-18: Structured Tools System (v2 - Improved)

**Added** `structured_tools.py`:
- Preconfigured search tools that agents call directly instead of writing raw Python
- Each tool returns `ToolResult` with structured `Finding` objects
- Includes confidence levels, severity, and fix suggestions

**Security Tools** (with false positive filtering):
- `find_secrets()` - API keys, passwords, tokens (skips .md, test files = LOW confidence)
- `find_sql_injection()` - SQL injection patterns (code files only)
- `find_command_injection()` - os.system, subprocess issues
- `find_xss()` - innerHTML, document.write XSS
- `find_python_security()` - pickle, yaml.load, bare except

**iOS/Swift Tools** (expanded):
- `find_force_unwraps()` - ! unwraps (filters safe patterns: NSRegularExpression, static let)
- `find_retain_cycles()` - Missing [weak self] on delegates
- `find_main_thread_violations()` - UI off main thread
- `find_weak_self_issues()` - **NEW**: NotificationCenter, Timer, Combine sinks without [weak self]
- `find_cloudkit_issues()` - **NEW**: CKError handling, missing completion handlers
- `find_deprecated_apis()` - **NEW**: UIWebView, keyWindow, foregroundColor, etc.
- `find_swiftdata_issues()` - **NEW**: ModelContext threading, @MainActor issues

**Quality Tools**:
- `find_long_functions(max_lines)` - Functions over N lines
- `find_todos()` - TODO/FIXME/HACK comments

**Architecture Tools**:
- `map_architecture()` - Categorize files
- `find_imports(module)` - Find module imports

**Batch Scans**:
- `run_security_scan()` - All security tools
- `run_ios_scan()` - All 7 iOS tools (expanded)
- `run_quality_scan()` - All quality tools

**Improvements based on feedback**:
1. Non-code files (markdown, docs) filtered from security scans
2. Test files get LOW confidence automatically
3. Safe Swift patterns filtered (try! NSRegularExpression, static let)
4. Added missing iOS checks: CloudKit, deprecated APIs, SwiftData, weak self

### 2025-01-18: Query Enhancement System

**Changed**:
- `rlm_processor.py`: Added 10 query types with auto-detection
- `rlm_processor.py`: Added `enhance_query()` for automatic enhancement
- `server.py`: Enhanced tool descriptions with query patterns

**Why**: Vague queries produced poor results. Now auto-detected and enhanced.

### 2025-01 (Earlier): Core RLM Implementation

- Implemented TRUE RLM from arXiv:2512.24601
- Content stored as `prompt` variable, not in LLM context
- Sub-LLM responses preserved in full (not summarized)
- Added confidence levels (HIGH/MEDIUM/LOW) to findings
- Added dead code detection (#if false, #if DEBUG blocks)
- Added line verification before reporting findings

---

## Known Issues

### Authentication (401 Errors)
- **Symptom**: `AuthenticationError: 401 - User not found`
- **Cause**: OpenRouter API key invalid/expired
- **Fix**: Update `OPENROUTER_API_KEY` in `~/.mcp.json`

### Circuit Breaker Trips
- **Symptom**: `Circuit breaker is open - too many recent failures`
- **Cause**: 5+ consecutive API failures
- **Fix**: Wait 60s for auto-reset, or restart server

### REPL Execution Failures
- **Symptom**: `NameError: 'findings' not defined`
- **Cause**: LLM generates code without initializing variables
- **Mitigation**: Pre-initialized variables in sandbox (findings, results, output, issues, files)
- **Fallback**: `fallback_analyzer.py` provides pattern-based analysis

---

## Agent Notes

### Query Quality Matters Most
The entire system's effectiveness depends on query specificity. The query is used to generate Python code that searches the content. Vague query → vague code → poor results.

### Confidence Levels
- **HIGH**: Line verified, code in active block, function has implementation
- **MEDIUM**: Context unclear, cannot verify reachability
- **LOW**: Dead code (#if false), unverified line, signature-only

### Verification Still Required
~40-60% of RLM findings need Grep/Read verification:
- Test code flagged as vulnerable
- Commented-out code reported as active
- Type-safe patterns misidentified

### Model Selection
Using `google/gemini-2.5-flash-lite` via OpenRouter:
- Fast (~1-3s per chunk)
- Cost-effective (~$0.15/1M input tokens)
- Good at generating Python code for search

---

## Development Commands

```bash
# Install
cd python && pip install -e .

# Run server
python -m rlm_mem_mcp.server

# Test
pytest

# Check status via Claude Code
# Use: rlm_status tool
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENROUTER_API_KEY` | Yes | - | API authentication |
| `RLM_MODEL` | No | `google/gemini-2.5-flash-lite` | Processing model |
| `RLM_AGGREGATOR_MODEL` | No | `google/gemini-2.5-flash-lite` | Aggregation model |
| `RLM_MAX_CHUNK_TOKENS` | No | `8000` | Max tokens per chunk |
| `RLM_MAX_RESULT_TOKENS` | No | `4000` | Max tokens in result |

---

## Performance Notes

- **File Collection**: Async with parallel I/O
- **Chunk Processing**: Parallel relevance assessment
- **Caching**: LRU response cache (1000 entries), semantic cache for similar queries
- **Rate Limiting**: 60 req/min, 100k tokens/min
- **Circuit Breaker**: Opens after 5 failures, resets after 60s

---

*Last updated: 2025-01-18*
