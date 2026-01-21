# RLM-Mem MCP Server

**Version**: 2.6 | **Status**: Production Ready | [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)

## Overview

Implements TRUE Recursive Language Model (RLM) technique for Claude Code context management.

**Problem**: Claude's ~200k token context fails on large codebases (500k+ tokens).

**Solution**: Content stored as `prompt` variable in Python REPL. LLM writes code to examine it, preserving full data while using minimal context (~4k summary).

**Cost**: ~$0.10-0.50 for 500k tokens (vs $15+ for premium 1M context).

---

## MCP Tools

### rlm_analyze
Analyze files/directories (50+ files, security audits, architecture reviews).

```javascript
{
  "query": "string (required)",           // Specific analysis query
  "paths": ["string"] (required),         // Files/directories
  "query_mode": "auto|semantic|scanner|literal|custom",
  "scan_mode": "auto|security|ios|quality|web|rust|node|all|custom",
  "min_confidence": "HIGH|MEDIUM|LOW",
  "include_quality": false,
  "include_skipped_signatures": false
}
```

**Query Modes**:
- `auto`: Auto-detect best mode
- `semantic`: LLM writes custom search code (TRUE RLM)
- `scanner`: Pre-built scanners only (fastest)
- `literal`: Fast grep-style search (~40ms)
- `custom`: Semantic analysis without scanners

**Query Examples**:
```
Security: "Find (1) SQL injection (2) hardcoded secrets (3) eval/exec. Report: file:line, code, severity"
iOS: "Find (1) force unwraps (2) missing [weak self] (3) @ObservedObject issues"
Architecture: "Map modules with purpose, entry points, dependencies, data flow"
```

### rlm_query_text
Process large text (logs, transcripts, documents).

```javascript
{ "query": "string", "text": "string" }
```

### rlm_read
Read single file (replaces native Read). Fast, no LLM overhead.

```javascript
{ "path": "string", "offset": 0, "limit": null }
```

### rlm_grep
Pattern search with ripgrep (replaces native Grep).

```javascript
{
  "pattern": "string",
  "path": ".",
  "case_insensitive": false,
  "fixed_strings": false,
  "context_lines": 0,
  "file_type": "py|js|swift|rs",
  "glob": "*.tsx",
  "output_mode": "content|files_with_matches|count"
}
```

### rlm_glob
Find files by pattern (replaces native Glob).

```javascript
{ "pattern": "**/*.py", "path": ".", "include_hidden": false }
```

### rlm_memory_store / rlm_memory_recall
Persist and retrieve findings (SQLite-backed).

```javascript
// Store
{ "key": "audit_2024", "value": "findings...", "tags": ["security"] }

// Recall by key or tags
{ "key": "audit_2024" }
{ "search_tags": ["security"] }
```

### rlm_status
Server health, cache stats, configuration.

---

## Tool Selection Guide

| Task | Best Tool |
|------|-----------|
| Read 1 file | `rlm_read` |
| Search patterns | `rlm_grep` |
| Find files | `rlm_glob` |
| Analyze 50+ files | `rlm_analyze` |
| Security audit | `rlm_analyze` + `scan_mode="security"` |
| iOS audit | `rlm_analyze` + `scan_mode="ios"` |

---

## Available Analysis Tools

### Security
`find_secrets()`, `find_sql_injection()`, `find_xss()`, `find_command_injection()`, `find_python_security()`

### iOS/Swift
`find_force_unwraps()`, `find_retain_cycles()`, `find_weak_self_issues()`, `find_async_await_issues()`, `find_sendable_issues()`, `find_mainactor_issues()`, `find_swiftui_performance_issues()`, `find_memory_management_issues()`, `find_error_handling_issues()`, `find_accessibility_issues()`, `find_localization_issues()`

### Web/Frontend
`find_react_issues()`, `find_vue_issues()`, `find_angular_issues()`, `find_dom_security()`, `find_a11y_issues()`, `find_css_issues()`

### Rust
`find_unsafe_blocks()`, `find_unwrap_usage()`, `find_rust_concurrency_issues()`, `find_rust_error_handling()`, `find_rust_clippy_patterns()`

### Node.js
`find_callback_hell()`, `find_promise_issues()`, `find_node_security()`, `find_require_issues()`, `find_node_async_issues()`

### Quality
`find_long_functions()`, `find_complex_functions()`, `find_code_smells()`, `find_dead_code()`, `find_todos()`

### Architecture
`map_architecture()`, `find_imports()`, `analyze_typescript_imports()`, `build_call_graph()`

### Batch Scans (parallel by default)
`run_security_scan()`, `run_ios_scan()`, `run_quality_scan()`, `run_web_scan()`, `run_rust_scan()`, `run_node_scan()`, `run_frontend_scan()`, `run_backend_scan()`

---

## REPL Environment

Code executes in sandboxed Python with these globals:

```python
prompt        # str - File/text content
context       # dict - Metadata
results       # list - Accumulate findings
llm_query()   # func - Query LLM for analysis

# Ripgrep functions (10-100x faster)
rg_search(pattern, **flags)      # Full regex search
rg_literal(text)                 # Literal string search
rg_files(pattern)                # Return file paths only
rg_count(pattern)                # Count matches per file
parallel_rg_search(patterns)     # Multiple patterns concurrently
RG_AVAILABLE                     # bool - ripgrep installed?

# File functions
read_file(path)                  # Read with line numbers
grep_pattern(pattern, **flags)   # Search patterns
glob_files(pattern)              # Find files
```

---

## Confidence Levels

| Level | Score | Meaning |
|-------|-------|---------|
| HIGH | 80+ | Multiple indicators verified |
| MEDIUM | 50-79 | Pattern matched, some uncertainty |
| LOW | 20-49 | Weak match, likely false positive |
| FILTERED | <20 | Ruled out (dead code, test, comment) |

**Scoring**: Start 100, deduct for dead code (-50), test file (-30), comment (-40), boost for semantic verification (+20).

---

## Configuration

### Installation

```bash
git clone https://github.com/mosif16/RLM-Mem_MCP.git
cd RLM-Mem_MCP/python && pip install -e .
export OPENROUTER_API_KEY=sk-or-...
```

### Claude Code Setup

```bash
claude mcp add --transport stdio rlm -- python -m rlm_mem_mcp.server
```

Or in `~/.claude/mcp_servers.json`:
```json
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["-m", "rlm_mem_mcp.server"],
      "env": { "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}" }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | required | API key |
| `RLM_MODEL` | `x-ai/grok-code-fast-1` | Processing model |
| `RLM_USE_CACHE` | `true` | Enable caching (90% cost savings) |
| `RLM_CACHE_TTL` | `1h` | Cache TTL |
| `RLM_MAX_RESULT_TOKENS` | `4000` | Max result tokens |
| `RLM_MAX_CHUNK_TOKENS` | `8000` | Max chunk tokens |

---

## Project Structure

```
RLM-Mem_MCP/
├── python/src/rlm_mem_mcp/
│   ├── server.py              # MCP entry point
│   ├── rlm_processor.py       # Core RLM implementation
│   ├── repl_environment.py    # REPL sandbox with llm_query()
│   ├── structured_tools.py    # Pre-built analysis tools
│   ├── result_verifier.py     # Confidence scoring
│   ├── cache_manager.py       # Prompt caching
│   ├── memory_store.py        # SQLite persistence
│   └── file_collector.py      # Async file collection
├── CLAUDE.md                  # This file
└── README.md
```

---

## Architecture

```
Claude Code → MCP Protocol → RLM-Mem Server
                              ├── MCP Handler (server.py)
                              ├── RLM Processor (orchestration)
                              ├── REPL Environment (sandboxed execution)
                              └── Support (cache, memory, verification)
                                    ↓
                              OpenRouter/Anthropic API + SQLite
```

**Flow**: Input → File Collection → Chunking → Store as `prompt` → LLM writes code → Execute → Aggregate → Output

---

## File Support

**Code**: `.py`, `.js`, `.ts`, `.tsx`, `.swift`, `.rs`, `.go`, `.java`, `.c`, `.cpp`, `.rb`, `.php`

**Web**: `.html`, `.css`, `.scss`, `.vue`, `.svelte`, `.astro`

**Config**: `.json`, `.yaml`, `.toml`, `.plist`, `.xcconfig`

**iOS**: `.swift`, `.storyboard`, `.xib`, `.strings`, `.entitlements`

**Skipped**: `.git`, `node_modules`, `__pycache__`, `venv`, `dist`, `build`, `DerivedData`, `Pods`, `target/`

---

## Best Practices

1. **Be specific** - "Find SQL injection via string concat" not "find problems"
2. **Use scan modes** - `ios`, `security`, `web`, `rust`, `node` for targeted analysis
3. **Set confidence** - `HIGH` for critical audits, `MEDIUM` for general
4. **Store findings** - Use `rlm_memory_store` to persist across sessions
5. **Break into phases** - Security → Performance → Quality → Architecture
6. **Use ripgrep** - Install for 10-100x faster searches

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many false positives | Use `min_confidence: "HIGH"` |
| Analysis too slow | Narrow `paths`, use `scan_mode`, install ripgrep |
| Results too brief | Increase `RLM_MAX_RESULT_TOKENS` |
| Not specific enough | Add numbered criteria + output format to query |

---

## Version History

**v2.6** (Current): Single-file tools (`rlm_read`, `rlm_grep`, `rlm_glob`)

**v2.5**: Ripgrep integration (10-100x speedup), parallel batch scans (2-4x speedup)

**v2.4**: Web/React, Rust, Node.js support with 25+ new analysis tools

**v2.3**: Query mode system (`auto`, `semantic`, `scanner`, `literal`, `custom`)

**v2.2**: Full Swift/iOS support with concurrency, SwiftUI, accessibility tools

---

## References

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601) - Zhang, Kraska, Khattab - MIT
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP SDK](https://github.com/anthropics/mcp)

---

**License**: MIT
