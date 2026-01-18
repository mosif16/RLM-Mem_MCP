# RLM-Mem MCP Server

An MCP (Model Context Protocol) server implementing the **Recursive Language Model (RLM)** technique for ultimate context management with Claude Code.

Based on:
- **[arXiv:2512.24601](https://arxiv.org/abs/2512.24601)** - Recursive Language Models (Zhang, Kraska, Khattab - MIT, 2025)
- **[Anthropic MCP Documentation](https://modelcontextprotocol.io/)**
- **[Anthropic Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)**

## The Problem

Claude Code has a context window of ~200k tokens. When analyzing large codebases (500k+ tokens), Claude either:
- Fails to process everything
- Experiences "context rot" (degraded performance)
- Runs out of space for reasoning

## The Solution

RLM stores large content **outside** the model's context and processes it recursively:

```
Without RLM:
    Claude tries to read all files → exceeds context → fails
    OR reads partial files → misses information

With RLM:
    Claude calls RLM tool → RLM processes externally → returns summary
    Claude receives 4k token summary → has 196k tokens left for reasoning
```

## Features

- **Recursive Processing**: Handles arbitrarily large inputs by chunking and aggregating
- **Prompt Caching**: Leverages Anthropic's caching for 90% cost reduction on repeated content
- **Intelligent Chunking**: Respects file/function/section boundaries when splitting
- **Memory Store**: Persist important findings across conversations
- **Token Optimization**: Minimizes context usage with efficient tool schemas

## Installation

### Prerequisites

- Python 3.10+
- Claude Code CLI
- Anthropic API key

### Setup

```bash
# Clone the repository
git clone https://github.com/mosif16/RLM-Mem_MCP.git
cd RLM-Mem_MCP

# Install Python dependencies
cd python
pip install -e .

# Set your API key
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
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

## Usage

### Tools Available

#### `rlm_analyze`
Analyze files or directories recursively.

```
Query: "Find all security vulnerabilities"
Paths: ["./src", "./api"]
```

#### `rlm_query_text`
Process large text blocks directly.

```
Query: "Extract all error messages with timestamps"
Text: <massive log file content>
```

#### `rlm_status`
Check server health and configuration.

#### `rlm_memory_store` / `rlm_memory_recall`
Persist and retrieve important findings.

### Example Workflows

**Security Audit:**
```
User: "Check this repo for security vulnerabilities"

Claude uses rlm_analyze({
  "query": "security vulnerabilities: SQL injection, XSS, CSRF,
            hardcoded secrets, insecure deserialization, path traversal",
  "paths": ["./src", "./api"]
})
```

**Architecture Review:**
```
User: "Explain the architecture of this project"

Claude uses rlm_analyze({
  "query": "describe architecture: main components, data flow,
            dependencies, entry points, design patterns used",
  "paths": ["."]
})
```

**Log Analysis:**
```
User: "Here's a 50MB log file. Find all errors."

Claude uses rlm_query_text({
  "query": "extract all ERROR and EXCEPTION entries with timestamps",
  "text": "<log content>"
})
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `RLM_MODEL` | `claude-sonnet-4-5-20241022` | Model for RLM processing |
| `RLM_USE_CACHE` | `true` | Enable prompt caching |
| `RLM_CACHE_TTL` | `5m` | Cache TTL (`5m` or `1h`) |
| `RLM_MAX_RESULT_TOKENS` | `4000` | Max tokens in result |
| `RLM_MAX_CHUNK_TOKENS` | `8000` | Max tokens per chunk |

### File Filtering

**Included extensions:**
- Code: `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, etc.
- Config: `.json`, `.yaml`, `.toml`, `.ini`
- Docs: `.md`, `.txt`, `.rst`

**Skipped directories:**
- `.git`, `node_modules`, `__pycache__`, `venv`, `dist`, `build`, etc.

## How It Works

### RLM Architecture

```
+------------------+
|   Claude Code    |
|                  |
| - Sees RLM tools |
| - Decides to use |
| - Calls tool     |
+--------+---------+
         |
         | MCP Protocol (JSON-RPC over stdio)
         v
+------------------+
|  RLM MCP Server  |
|                  |
| - Collects files |
| - Chunks content |
| - Recursive LLM  |
| - Aggregates     |
+--------+---------+
         |
         | API calls (with caching)
         v
+------------------+
|  Anthropic API   |
|  (Claude Model)  |
+------------------+
```

### Processing Steps

1. **File Collection**: Walk directories, filter by extension, skip ignored dirs
2. **Chunking**: Split content at semantic boundaries (files, functions, sections)
3. **Relevance Assessment**: Score each chunk's relevance to the query
4. **Processing**: Query relevant chunks with sub-LLM calls
5. **Aggregation**: Combine findings into coherent response
6. **Truncation**: Ensure result fits in context (max 4000 tokens)

### Prompt Caching Strategy

The server uses Anthropic's prompt caching to optimize costs:

- **System prompts** are cached (90% cost reduction on hits)
- **5-minute TTL** by default, refreshes on each use
- **1-hour TTL** available for less frequent access
- **Cache statistics** tracked and reported via `rlm_status`

```python
# Cache control is applied automatically to system prompts
system = [
    {
        "type": "text",
        "text": "You are a precise information extractor...",
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }
]
```

## Cost Comparison

| Method | 500k token input | Context Used | Cost |
|--------|------------------|--------------|------|
| Direct (if possible) | Fails or degrades | 200k+ (full) | N/A |
| Premium 1M context | Works | 500k | ~$15 |
| **RLM via MCP** | Works | ~4k summary | **~$3** |

RLM is often **cheaper** and leaves context for reasoning.

## Adding to CLAUDE.md

Add guidance to your project's `CLAUDE.md`:

```markdown
## Large Codebase Protocol

When to use `rlm_analyze`:
- Analyzing 50+ files
- Searching entire codebase
- Tasks with "all", "every", or "entire" scope
- Security audits
- Architecture reviews

When NOT to use:
- Working with 1-5 specific files
- Making targeted edits
- Quick lookups in known locations

## Query Tips

Be specific in RLM queries:

BAD:  "find problems"
GOOD: "find SQL injection, XSS, hardcoded secrets"

BAD:  "summarize"
GOOD: "summarize architecture, main components, data flow"
```

## Development

### Project Structure

```
RLM-Mem_MCP/
├── python/
│   ├── src/
│   │   └── rlm_mem_mcp/
│   │       ├── __init__.py
│   │       ├── server.py          # MCP server entry point
│   │       ├── rlm_processor.py   # RLM implementation
│   │       ├── file_collector.py  # File collection
│   │       ├── cache_manager.py   # Prompt caching
│   │       └── config.py          # Configuration
│   ├── requirements.txt
│   └── pyproject.toml
├── src/                           # TypeScript implementation (optional)
├── .mcp.json                      # Project MCP config
└── README.md
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

## License

MIT License - see [LICENSE](LICENSE) for details.
