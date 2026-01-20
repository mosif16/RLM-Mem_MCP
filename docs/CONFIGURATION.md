# RLM-Mem Configuration Guide

Complete guide to configuring the RLM-Mem MCP server for your environment.

## Environment Variables

### API Configuration

#### OPENROUTER_API_KEY (Recommended)
- **Type**: String
- **Required**: Yes (for OpenRouter)
- **Example**: `sk-or-v1-1234567890abcdef`
- **Where to get**: https://openrouter.ai/keys

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

#### ANTHROPIC_API_KEY (Alternative)
- **Type**: String
- **Required**: Yes (for Anthropic direct)
- **Example**: `sk-ant-v1-1234567890abcdef`
- **Where to get**: https://console.anthropic.com/account/keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Note**: OpenRouter is recommended for cost efficiency and model flexibility.

---

### Model Configuration

#### RLM_MODEL
- **Type**: String
- **Default**: `google/gemini-2.5-flash-lite`
- **Used for**: Main RLM processing (file/text analysis)
- **Examples**:
  - `google/gemini-2.5-flash-lite` - Fast, cheap, recommended
  - `anthropic/claude-opus-4.5` - More capable but expensive
  - `google/gemini-2-flash` - Very fast, lower cost

```bash
export RLM_MODEL=google/gemini-2.5-flash-lite
```

**Model Selection Guide:**
| Model | Speed | Cost | Quality | Use Case |
|-------|-------|------|---------|----------|
| Gemini 2.5 Flash Lite | ⚡⚡⚡ | $ | Good | Default choice |
| Gemini 2 Flash | ⚡⚡⚡ | $ | Good | Budget sensitive |
| Claude 3.5 Sonnet | ⚡⚡ | $$ | Excellent | Complex analysis |
| Claude Opus | ⚡ | $$$ | Best | Critical security |

#### RLM_AGGREGATOR_MODEL
- **Type**: String
- **Default**: `google/gemini-2.5-flash-lite`
- **Used for**: Final result aggregation and formatting
- **Recommendation**: Same as `RLM_MODEL`

```bash
export RLM_AGGREGATOR_MODEL=google/gemini-2.5-flash-lite
```

---

### Caching Configuration

#### RLM_USE_CACHE
- **Type**: Boolean
- **Default**: `true`
- **Purpose**: Enable/disable prompt caching to reduce costs
- **Impact**: ~90% cost reduction on cache hits

```bash
export RLM_USE_CACHE=true
```

#### RLM_CACHE_TTL
- **Type**: String
- **Default**: `5m`
- **Allowed values**: `5m`, `1h`
- **Purpose**: How long cache entries persist

```bash
# 5-minute cache (recommended for active sessions)
export RLM_CACHE_TTL=5m

# 1-hour cache (for infrequent access)
export RLM_CACHE_TTL=1h
```

---

### Token Limits

#### RLM_MAX_RESULT_TOKENS
- **Type**: Integer
- **Default**: `4000`
- **Purpose**: Maximum tokens in final result
- **Impact**: Controls result length/detail

```bash
export RLM_MAX_RESULT_TOKENS=4000  # Detailed results
export RLM_MAX_RESULT_TOKENS=2000  # Brief results
```

#### RLM_MAX_CHUNK_TOKENS
- **Type**: Integer
- **Default**: `8000`
- **Purpose**: Maximum tokens per analysis chunk
- **Note**: Internal use, rarely needs adjustment

```bash
export RLM_MAX_CHUNK_TOKENS=8000
```

#### RLM_OVERLAP_TOKENS
- **Type**: Integer
- **Default**: `200`
- **Purpose**: Token overlap between chunks (prevents missing findings at boundaries)
- **Note**: Internal use, rarely needs adjustment

```bash
export RLM_OVERLAP_TOKENS=200
```

---

### File Filtering

#### RLM_EXTRA_EXTENSIONS
- **Type**: CSV string
- **Default**: (empty)
- **Purpose**: Additional file extensions to include
- **Example**: Add `.xml` and `.xsl` files

```bash
export RLM_EXTRA_EXTENSIONS=.xml,.xsl,.custom
```

#### RLM_SKIP_EXTENSIONS
- **Type**: CSV string
- **Default**: (empty)
- **Purpose**: File extensions to exclude
- **Example**: Exclude markdown from analysis

```bash
export RLM_SKIP_EXTENSIONS=.md,.txt
```

#### RLM_EXTRA_SKIP_DIRS
- **Type**: CSV string
- **Default**: (empty)
- **Purpose**: Additional directories to skip
- **Example**: Skip vendor and logs directories

```bash
export RLM_EXTRA_SKIP_DIRS=vendor,logs,tmp,.artifacts
```

#### RLM_INCLUDE_DIRS
- **Type**: CSV string
- **Default**: (empty)
- **Purpose**: Force include directories (remove from skip list)
- **Example**: Include `.vscode` for settings analysis

```bash
export RLM_INCLUDE_DIRS=.vscode,node_modules
```

---

### Feature Flags

#### RLM_ENABLE_MEMORY
- **Type**: Boolean
- **Default**: `true`
- **Purpose**: Enable persistent memory store

```bash
export RLM_ENABLE_MEMORY=true
```

#### RLM_ENABLE_VERIFICATION
- **Type**: Boolean
- **Default**: `true`
- **Purpose**: Enable result verification guardrail

```bash
export RLM_ENABLE_VERIFICATION=true
```

#### RLM_ENABLE_QUALITY_CHECKS
- **Type**: Boolean
- **Default**: `false`
- **Purpose**: Include code quality checks in analysis

```bash
export RLM_ENABLE_QUALITY_CHECKS=true
```

---

### Advanced Configuration

#### RLM_CONFIDENCE_THRESHOLD
- **Type**: String
- **Default**: `MEDIUM`
- **Allowed values**: `HIGH`, `MEDIUM`, `LOW`
- **Purpose**: Filter findings by confidence level

```bash
# Only high-confidence findings
export RLM_CONFIDENCE_THRESHOLD=HIGH

# All findings
export RLM_CONFIDENCE_THRESHOLD=LOW
```

#### RLM_CIRCUIT_BREAKER_THRESHOLD
- **Type**: Integer
- **Default**: `3`
- **Purpose**: Number of failures before circuit breaker trips

```bash
export RLM_CIRCUIT_BREAKER_THRESHOLD=3
```

#### RLM_RATE_LIMIT_REQUESTS
- **Type**: Integer
- **Default**: `100`
- **Purpose**: Max requests per minute

```bash
export RLM_RATE_LIMIT_REQUESTS=100
```

#### RLM_RATE_LIMIT_TOKENS
- **Type**: Integer
- **Default**: `1000000`
- **Purpose**: Max tokens per minute

```bash
export RLM_RATE_LIMIT_TOKENS=1000000
```

---

## Configuration File (.env)

Create a `.env` file in the project root instead of exporting variables:

```bash
# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Selection
RLM_MODEL=google/gemini-2.5-flash-lite
RLM_AGGREGATOR_MODEL=google/gemini-2.5-flash-lite

# Caching
RLM_USE_CACHE=true
RLM_CACHE_TTL=5m

# Token Limits
RLM_MAX_RESULT_TOKENS=4000
RLM_MAX_CHUNK_TOKENS=8000
RLM_OVERLAP_TOKENS=200

# File Filtering
RLM_EXTRA_EXTENSIONS=.xml,.custom
RLM_SKIP_EXTENSIONS=
RLM_EXTRA_SKIP_DIRS=vendor,logs
RLM_INCLUDE_DIRS=

# Features
RLM_ENABLE_MEMORY=true
RLM_ENABLE_VERIFICATION=true
RLM_ENABLE_QUALITY_CHECKS=false

# Advanced
RLM_CONFIDENCE_THRESHOLD=MEDIUM
RLM_CIRCUIT_BREAKER_THRESHOLD=3
```

The server automatically loads `.env` on startup.

---

## Claude Code Integration

### MCP Configuration

Add to `~/.claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["-m", "rlm_mem_mcp.server"],
      "env": {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}",
        "RLM_MODEL": "google/gemini-2.5-flash-lite",
        "RLM_USE_CACHE": "true"
      }
    }
  }
}
```

### Alternative: Using CLI

```bash
claude mcp add --transport stdio rlm -- python -m rlm_mem_mcp.server
```

---

## Recommended Configurations

### Development (Fast & Cheap)
```bash
RLM_MODEL=google/gemini-2.5-flash-lite
RLM_USE_CACHE=false        # Faster iteration without caching
RLM_MAX_RESULT_TOKENS=2000 # Brief results
```

### Production (Balanced)
```bash
RLM_MODEL=google/gemini-2.5-flash-lite
RLM_USE_CACHE=true
RLM_CACHE_TTL=5m
RLM_CONFIDENCE_THRESHOLD=MEDIUM
RLM_MAX_RESULT_TOKENS=4000
```

### Security Audit (Thorough)
```bash
RLM_MODEL=anthropic/claude-opus-4.5
RLM_USE_CACHE=true
RLM_CONFIDENCE_THRESHOLD=HIGH
RLM_ENABLE_VERIFICATION=true
RLM_MAX_RESULT_TOKENS=8000
```

### Large Codebase (Resource Conscious)
```bash
RLM_MODEL=google/gemini-2-flash
RLM_MAX_CHUNK_TOKENS=6000
RLM_MAX_RESULT_TOKENS=2000
RLM_SKIP_EXTENSIONS=.test.js,.test.ts
RLM_EXTRA_SKIP_DIRS=docs,examples
```

---

## Troubleshooting

### Rate Limiting (429 errors)

```bash
# Increase rate limits
export RLM_RATE_LIMIT_REQUESTS=200
export RLM_RATE_LIMIT_TOKENS=2000000
```

### OutOfMemory Errors

```bash
# Reduce chunk size
export RLM_MAX_CHUNK_TOKENS=4000

# Skip more directories
export RLM_EXTRA_SKIP_DIRS=node_modules,dist,build,.cache
```

### Slow Responses

```bash
# Use faster model
export RLM_MODEL=google/gemini-2-flash

# Reduce result size
export RLM_MAX_RESULT_TOKENS=2000

# Disable verification for speed
export RLM_ENABLE_VERIFICATION=false
```

### Poor Results Quality

```bash
# Use more capable model
export RLM_MODEL=anthropic/claude-opus-4.5

# Enable verification
export RLM_ENABLE_VERIFICATION=true

# Lower confidence threshold to see more options
export RLM_CONFIDENCE_THRESHOLD=LOW
```

---

## Cost Estimation

### Factors Affecting Cost

1. **Model Choice**: ~2-10x cost difference
2. **Cache Hits**: ~90% reduction with caching
3. **Content Size**: Larger files = higher cost
4. **Query Complexity**: More complex queries = higher cost

### Example Costs

**100 file security audit with Gemini Flash:**
- First run: ~$0.50 (1M tokens input)
- Cached runs: ~$0.05 each (90% reduction)

**Same audit with Claude Opus:**
- First run: ~$3.00 (10x more expensive)
- Cached runs: ~$0.30 each

---

## Next Steps

1. **Set API Key**: `export OPENROUTER_API_KEY=...`
2. **Choose Model**: Start with Gemini Flash Lite
3. **Configure Claude Code**: Add MCP server to `mcp_servers.json`
4. **Test**: Run `rlm_status` to verify connection
5. **Tune**: Adjust configuration based on results

See [API Reference](./API.md) for tool usage and [Architecture](./ARCHITECTURE.md) for system details.
