# Documentation

Complete documentation for RLM-Mem MCP server.

## Quick Navigation

| Document | Purpose |
|----------|---------|
| **[Usage Guide](./USAGE_GUIDE.md)** | Practical examples and workflows (start here!) |
| **[API Reference](./API.md)** | Complete tool specifications and parameters |
| **[Configuration Guide](./CONFIGURATION.md)** | Environment variables and setup |
| **[Architecture](./ARCHITECTURE.md)** | Technical deep-dive into system design |

---

## Document Descriptions

### [Usage Guide](./USAGE_GUIDE.md)

**For:** Developers using RLM-Mem tools in Claude Code

**Contains:**
- Quick start (installation, verification)
- 10+ real-world use cases:
  - Security audits
  - Architecture review
  - Performance analysis
  - iOS security
  - Log parsing
  - Configuration extraction
  - And more...
- Advanced patterns (iterative analysis, comparative analysis)
- Tips and tricks for effective queries
- Troubleshooting common issues
- Cost optimization strategies

**Best for:** Learning by example, solving specific problems

---

### [API Reference](./API.md)

**For:** Developers integrating RLM-Mem into applications

**Contains:**
- Tool overview table
- Detailed specs for each tool:
  - `rlm_analyze` - File/directory analysis
  - `rlm_query_text` - Text block processing
  - `rlm_memory_store` - Persist findings
  - `rlm_memory_recall` - Retrieve findings
  - `rlm_status` - Server health
- Query patterns for different domains:
  - Security (SQL injection, XSS, secrets)
  - iOS/Swift (force unwraps, weak references)
  - Python (pickle, bare except, mutable defaults)
  - JavaScript (innerHTML, missing await)
  - Architecture (components, data flow)
- Response structure specifications
- Confidence levels and scoring
- Error handling and retry logic
- Rate limits and authentication
- Performance tips

**Best for:** Understanding parameters, integrating tools, advanced usage

---

### [Configuration Guide](./CONFIGURATION.md)

**For:** Operators configuring the RLM server

**Contains:**
- All environment variables with descriptions:
  - API keys (OpenRouter, Anthropic)
  - Model selection
  - Caching options
  - Token limits
  - File filtering
  - Feature flags
  - Advanced options
- Configuration file (`.env`) format
- Claude Code MCP integration setup
- Recommended configurations for different scenarios:
  - Development (fast, cheap)
  - Production (balanced)
  - Security audit (thorough)
  - Large codebase (resource conscious)
- Troubleshooting common issues
- Cost estimation
- Best practices

**Best for:** Setup and tuning, optimizing performance/cost

---

### [Architecture](./ARCHITECTURE.md)

**For:** Contributors, maintainers, and architects

**Contains:**
- System overview diagram
- Core components:
  - MCP Server
  - RLM Processor
  - REPL Environment
  - Structured Tools
  - Result Verifier
  - Cache Manager
  - File Collector
  - Memory Store
- Data flow examples for:
  - Security audits
  - Architecture mapping
  - Log analysis
- Component interaction sequences
- Resilience patterns (circuit breaker, backoff, rate limiting)
- Performance characteristics
- Security considerations
- Extension points for adding tools/models

**Best for:** Understanding system design, contributing code, debugging issues

---

## Learning Path

### Beginner (Just starting)
1. Read [Quick Start](./USAGE_GUIDE.md#quick-start) in Usage Guide
2. Try one use case from [Common Use Cases](./USAGE_GUIDE.md#common-use-cases)
3. Reference [API Reference](./API.md) when needed

### Intermediate (Regular user)
1. Read [Usage Guide](./USAGE_GUIDE.md) completely
2. Bookmark [API Reference](./API.md) for lookups
3. Use [Tips & Tricks](./USAGE_GUIDE.md#tips--tricks) to optimize queries
4. Learn [Cost Optimization](./USAGE_GUIDE.md#cost-optimization)

### Advanced (Integrating/extending)
1. Study [Architecture](./ARCHITECTURE.md) thoroughly
2. Review [Configuration Guide](./CONFIGURATION.md) for tuning
3. Check [Extension Points](./ARCHITECTURE.md#extension-points) for adding features
4. Examine source code with architectural context

---

## Common Questions

**Q: How do I get started?**
A: Start with [Quick Start](./USAGE_GUIDE.md#quick-start) in the Usage Guide.

**Q: What tools are available?**
A: See [Tool Overview](./API.md#tool-overview) in the API Reference.

**Q: How do I reduce costs?**
A: See [Cost Optimization](./USAGE_GUIDE.md#cost-optimization) in the Usage Guide.

**Q: How do I set up the server?**
A: See [Configuration Guide](./CONFIGURATION.md) for environment variables and setup.

**Q: How does it work internally?**
A: See [Architecture](./ARCHITECTURE.md) for technical details.

**Q: How do I write better queries?**
A: See [Tips & Tricks](./USAGE_GUIDE.md#tips--tricks) in the Usage Guide.

**Q: What are query patterns?**
A: See [Query Patterns](./API.md#query-patterns) in the API Reference.

**Q: How do I debug issues?**
A: Check [Troubleshooting](./USAGE_GUIDE.md#troubleshooting) in the Usage Guide, or [Configuration Guide](./CONFIGURATION.md#troubleshooting) for setup issues.

---

## Documentation Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| Usage Guide | ✅ Complete | 2024-01-19 |
| API Reference | ✅ Complete | 2024-01-19 |
| Configuration Guide | ✅ Complete | 2024-01-19 |
| Architecture | ✅ Complete | 2024-01-19 |

All documentation is current and includes:
- Verification guardrail (L11 confidence scoring)
- Persistence/memory tools
- Structured analysis tools
- Updated configuration options

---

## Contributing to Docs

Found an issue or want to improve documentation?

1. **Report issues**: GitHub Issues
2. **Suggest improvements**: Pull Requests
3. **Update examples**: Add use cases or patterns
4. **Fix typos**: Submit quick fixes

---

## Related Resources

### Project
- **[README](../README.md)** - Project overview
- **[GitHub](https://github.com/mosif16/RLM-Mem_MCP)** - Source code

### External
- **[arXiv:2512.24601](https://arxiv.org/abs/2512.24601)** - Recursive Language Models paper
- **[MCP Documentation](https://modelcontextprotocol.io/)** - Model Context Protocol
- **[Claude API Docs](https://platform.claude.com/docs)** - Anthropic Claude
- **[OpenRouter](https://openrouter.ai/)** - Model marketplace

---

## Version Information

- **RLM-Mem Version**: 1.0+
- **Python**: 3.10+
- **MCP Version**: Latest
- **Documentation Version**: 1.0 (2024-01-19)

---

**Last updated**: 2024-01-19
**Maintained by**: RLM-Mem Team
**License**: MIT
