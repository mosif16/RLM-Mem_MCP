# CLAUDE.md - Guidance for Claude Code

This repository contains an MCP server implementing the Recursive Language Model (RLM) technique for handling large context with Claude.

## Project Overview

**RLM-Mem MCP** provides tools for Claude Code to process arbitrarily large inputs (codebases, logs, documents) that would otherwise exceed the context window.

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

## Query Tips

Be specific in RLM queries for better results:

```
BAD:  "find problems"
GOOD: "find SQL injection, XSS, CSRF, hardcoded secrets"

BAD:  "summarize"
GOOD: "summarize architecture, main components, data flow, entry points"

BAD:  "check security"
GOOD: "find security vulnerabilities: injection attacks, auth bypasses,
       insecure deserialization, path traversal, hardcoded credentials"
```

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
