# RLM-Mem MCP Server

An MCP (Model Context Protocol) server implementing the **TRUE Recursive Language Model (RLM)** technique for ultimate context management with Claude Code.

## ⚖️ Licensing & Pricing

### Free Use (MIT License)
- ✅ Personal projects
- ✅ Academic research
- ✅ Non-commercial open source
- ✅ Non-profit organizations
- ✅ Annual revenue < $50K

### Commercial Use (Revenue Sharing Required)
- 💰 **10% Revenue Share** for services/products using this technology
- 🏢 Commercial licenses required for companies/enterprises
- 📧 Contact: `msayf@recordandlearn.info`
- 🌐 Website: recordandlearn.info
- 📄 [Commercial License Details](./COMMERCIAL_LICENSE.md)

**Important**: Commercial use without proper licensing voids all warranties and may result in legal action.

**v2.9 Status**: 🚀 Optimization Initiative Underway
- ✅ Code organization complete (5 new modules, 1,524 LOC)
- 📊 Performance roadmap planned (60-100% cumulative gain)
- 🎯 3 optimization phases documented
- 📋 16 implementation tasks ready

See [.claude/claude.md](.claude/claude.md) for detailed optimization roadmap.

Based on:
- **[arXiv:2512.24601](https://arxiv.org/abs/2512.24601)** - Recursive Language Models (Zhang, Kraska, Khattab - MIT, 2025)
- **[Anthropic MCP Documentation](https://modelcontextprotocol.io/)**
- **[Anthropic Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)**

## The Problem

Claude Code has a context window of ~200k tokens. When analyzing large codebases (500k+ tokens), Claude either:
- Fails to process everything
- Experiences "context rot" (degraded performance)
- Runs out of space for reasoning

## The TRUE RLM Solution

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

## Features

- **TRUE RLM Processing**: Content stored as variables, LLM writes code to examine it
- **Prompt Caching**: Leverages caching for cost reduction on repeated content
- **Intelligent Chunking**: Respects file/function/section boundaries when splitting
- **Memory Store**: Persist important findings across conversations (SQLite-backed)
- **Robust Architecture**: Circuit breakers, rate limiters, exponential backoff
- **Async Pipeline**: Fully async with connection pooling and concurrent operations
- **Claude Haiku 4.5**: Default model with 90% cost savings via prompt caching

## Installation

### Prerequisites

- Python 3.10+
- Claude Code CLI
- OpenRouter API key (or Anthropic API key for direct access)

### Setup

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
# export ANTHROPIC_API_KEY=sk-ant-...
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

## Documentation

Complete documentation is available in the [`docs/`](./docs/) directory:

- **[Usage Guide](./docs/USAGE_GUIDE.md)** - Practical examples and workflows (start here!)
- **[API Reference](./docs/API.md)** - Complete tool specifications
- **[Configuration Guide](./docs/CONFIGURATION.md)** - Environment variables and setup
- **[Architecture](./docs/ARCHITECTURE.md)** - Technical deep-dive

## Quick Usage

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

#### Core Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `RLM_MODEL` | `x-ai/grok-code-fast-1` | Model for RLM processing |
| `RLM_AGGREGATOR_MODEL` | `x-ai/grok-code-fast-1` | Model for final aggregation |
| `RLM_USE_CACHE` | `true` | Enable prompt caching |
| `RLM_CACHE_TTL` | `5m` | Cache TTL (`5m` or `1h`) |
| `RLM_MAX_RESULT_TOKENS` | `4000` | Max tokens in result |
| `RLM_MAX_CHUNK_TOKENS` | `8000` | Max tokens per chunk |
| `RLM_OVERLAP_TOKENS` | `200` | Overlap tokens between chunks |

#### Commercial Licensing (Optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_COMMERCIAL_LICENSE_KEY` | (empty) | Commercial license key for revenue sharing |
| `RLM_ORGANIZATION_NAME` | (empty) | Organization name for commercial licensing |
| `RLM_ENABLE_TELEMETRY` | `false` | Enable usage telemetry for license compliance |
| `RLM_LICENSE_SERVER_URL` | `https://recordandlearn.info/license` | License validation server URL |

### File Filtering

**Included extensions:**
- Code: `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, etc.
- Config: `.json`, `.yaml`, `.toml`, `.ini`
- Docs: `.md`, `.txt`, `.rst`

**Skipped directories:**
- `.git`, `node_modules`, `__pycache__`, `venv`, `dist`, `build`, etc.

## How It Works

### TRUE RLM Architecture

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
+------------------+     +------------------+
|  RLM MCP Server  |     |   REPL Environ   |
|                  |     |                  |
| - Collects files |---->| prompt = content |
| - Stores as var  |     | results = []     |
| - LLM writes code|     | llm_query(...)   |
| - Executes code  |     |                  |
+--------+---------+     +------------------+
         |
         | API calls (with caching, rate limiting, circuit breaker)
         v
+------------------+
|   OpenRouter /   |
|   Anthropic API  |
+------------------+
```

### The TRUE RLM Technique (arXiv:2512.24601)

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
| **RLM via MCP** | Works | ~4k summary | **~$0.50-3** |

RLM is often **cheaper** and leaves context for reasoning. Using OpenRouter with Gemini Flash makes it even more cost-effective.

## Robust Architecture

The server includes production-ready features:

- **Circuit Breaker**: Stops requests after consecutive failures, auto-recovers
- **Rate Limiter**: Respects API rate limits (requests/min, tokens/min)
- **Exponential Backoff**: Retries with increasing delays on 429/503 errors
- **Connection Pooling**: Reuses HTTP connections via `httpx.AsyncClient`
- **LRU Response Cache**: Caches LLM responses to avoid redundant calls
- **Async Everything**: Non-blocking I/O for file collection and API calls
- **Graceful Shutdown**: Proper resource cleanup on server stop

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
│   │       ├── __init__.py          # Package exports
│   │       ├── server.py            # MCP server entry point
│   │       ├── rlm_processor.py     # Core RLM implementation
│   │       ├── repl_environment.py  # TRUE RLM REPL with llm_query()
│   │       ├── file_collector.py    # Async file collection
│   │       ├── cache_manager.py     # Prompt caching (Anthropic-style)
│   │       ├── memory_store.py      # SQLite-backed persistent memory
│   │       ├── agent_pipeline.py    # Claude Agent SDK integration
│   │       ├── config.py            # Environment configuration
│   │       └── utils.py             # Performance monitoring
│   ├── tests/
│   │   ├── test_integration.py      # End-to-end tests
│   │   ├── test_benchmark.py        # Performance benchmarks
│   │   ├── test_stress.py           # Stress tests
│   │   └── conftest.py              # Test fixtures
│   ├── requirements.txt
│   └── pyproject.toml
├── src/                             # TypeScript implementation (optional)
│   ├── index.ts                     # MCP server (Node.js)
│   ├── core/
│   │   └── rlm-context-manager.ts   # RLM tree-based context
│   ├── utils/
│   │   ├── tokenizer.ts             # Token counting
│   │   └── text-splitter.ts         # Document chunking
│   └── types/
│       └── index.ts                 # TypeScript interfaces
├── .mcp.json                        # Project MCP config
├── CLAUDE.md                        # Claude Code guidance
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

Business Source License 1.1 with commercial licensing for monetization - see [LICENSE](LICENSE) for details.

- **Free Use**: MIT license for non-commercial personal/academic use
- **Commercial Use**: Revenue sharing required - see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)
- **Terms of Service**: [TERMS_OF_SERVICE.md](TERMS_OF_SERVICE.md)
